"""Quantifier l'impact de la convention IR dans macro_to_z sur le RAROC et l'allocation."""
import sys
sys.path.insert(0, ".")
import numpy as np
import polars as pl
from scipy.stats import norm

from ifrs9_cockpit.config import (
    PREDEFINED_SCENARIOS, SCENARIO_BASE, ASSET_CLASSES, BASEL_CONFIG,
)
from ifrs9_cockpit.synthetic_generator import generate_dataset
from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl, macro_to_z
from ifrs9_cockpit.engine.comparator import PortfolioComparator
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.models.pd_model import PDModelSuite

RANDOM_SEED = 123

def convert_scenario(s):
    return {
        "gdp_growth": s["gdp_pct"],
        "unemployment_rate": SCENARIO_BASE.unemployment_rate + abs(s.get("unemployment_bipolar", 0)),
        "interest_rate": SCENARIO_BASE.interest_rate + s["interest_rate_bp"] / 100.0,
        "hpi_growth": s["hpi_pct"],
        "inflation_rate": s["inflation_pct"],
    }

def z_decompose(macro, sensitivities):
    """Decompose Z into per-variable contributions."""
    base_vars = {
        "gdp_growth": (SCENARIO_BASE.gdp_growth, 1.8),
        "unemployment_rate": (SCENARIO_BASE.unemployment_rate, 1.5),
        "interest_rate": (SCENARIO_BASE.interest_rate, 1.0),
        "hpi_growth": (SCENARIO_BASE.hpi_growth, 3.0),
        "inflation_rate": (SCENARIO_BASE.inflation_rate, 1.2),
    }
    contribs = {}
    for var, (mu, sigma) in base_vars.items():
        x = macro.get(var, mu)
        w = sensitivities.get(var, 0.0)
        if var == "unemployment_rate":
            contribs[var] = w * (x - mu) / sigma
        else:
            contribs[var] = -w * (x - mu) / sigma
    return contribs

# ================================================================
# 1. Impact sur Z pour toutes les classes x scenarios
# ================================================================
print("=" * 100)
print("IMPACT DE LA CONVENTION IR SUR LE Z-SCORE ET LE RAROC")
print("=" * 100)

# Show Z decomposition for ALL scenarios, focusing on IR contribution
print("\n[1] Contribution du taux d'interet au Z-score (toutes classes confondues)")
print(f"{'Scenario':35s} {'IR eff.':>7s} {'delta_ir':>8s} {'Z_ir (corp)':>11s} {'Z_ir (mort)':>11s} {'Z_total(corp)':>13s} {'Sens?':>6s}")
print("-" * 100)

ac_corp = [a for a in ASSET_CLASSES if a.name == "corporate_loans"][0]
ac_mort = [a for a in ASSET_CLASSES if a.name == "retail_mortgage"][0]

for name, scen in PREDEFINED_SCENARIOS.items():
    macro = convert_scenario(scen)
    ir = macro["interest_rate"]
    delta_ir = ir - SCENARIO_BASE.interest_rate

    contribs_corp = z_decompose(macro, ac_corp.macro_sensitivities)
    contribs_mort = z_decompose(macro, ac_mort.macro_sensitivities)
    z_ir_corp = contribs_corp["interest_rate"]
    z_ir_mort = contribs_mort["interest_rate"]
    z_total_corp = sum(contribs_corp.values())

    # Is the IR contribution "sensible"?
    gdp = macro["gdp_growth"]
    is_crisis = gdp < SCENARIO_BASE.gdp_growth
    ir_adverse = delta_ir < 0
    # Sensible: rate cuts in crisis = adverse, rate cuts in recovery = should be favorable
    sensible = "OK" if (is_crisis and ir_adverse) or (not is_crisis and not ir_adverse) or abs(delta_ir) < 0.01 else "FAUX"

    z_total_clamped = np.clip(z_total_corp, -4, 4)
    print(f"{name:35s} {ir:7.2f}% {delta_ir:+8.2f}% {z_ir_corp:+11.2f} {z_ir_mort:+11.2f} {z_total_clamped:+13.2f} {sensible:>6s}")

# ================================================================
# 2. Impact concret sur RAROC : current vs "Z sans IR"
# ================================================================
print("\n\n[2] Impact sur RAROC : actuel vs Z avec IR neutralise dans les scenarios favorables")
print("    (Pour les scenarios ou le GDP > base ET le taux < base, on met la contribution IR a 0)")
print()

# Generate data and fit models
bundle = generate_dataset(n_clients=5000, seed=RANDOM_SEED)
df_credit = bundle.df_credit
df_pe = bundle.df_pe
df_bs = bundle.df_balance_sheet

pd_suite = PDModelSuite(seed=RANDOM_SEED)
pd_suite.fit(df_credit)
pd_curr = pd_suite.predict_active(df_credit)
pd_orig = df_credit["pd_origination"].to_numpy()

lgd_model = LGDModel()
lgd_model.fit(df_credit)
ead_model = EADModel()
ead_model.fit(df_credit)
ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
pe_calc = PECalculator()

# Run all scenarios
print(f"{'Scenario':30s} | {'Equity RAROC':>13s} {'Corp RAROC':>11s} {'Mort RAROC':>11s} {'Sov RAROC':>10s} {'Portf RAROC':>12s} | {'IR sens?':>8s}")
print("-" * 115)

for name, scen in PREDEFINED_SCENARIOS.items():
    macro = convert_scenario(scen)

    bs_ecl = compute_balance_sheet_ecl(df_bs, macro)
    res = ecl_calc.calculate(df_credit, pd_curr, pd_orig,
        unemployment_override=macro["unemployment_rate"],
        gdp_override=macro["gdp_growth"],
        interest_rate_override=macro["interest_rate"],
        hpi_override=macro["hpi_growth"],
        inflation_override=macro["inflation_rate"])
    pe_res = pe_calc.calculate(df_pe,
        unemployment_override=macro["unemployment_rate"],
        gdp_override=macro["gdp_growth"],
        interest_rate_override=macro["interest_rate"],
        hpi_override=macro["hpi_growth"],
        inflation_override=macro["inflation_rate"])

    comp = PortfolioComparator(res, pe_res, df_balance_sheet_ecl=bs_ecl, macro_params=macro)
    raroc_df = comp.compute_raroc_multiclass()

    # Extract key classes
    def get_raroc(df, cls):
        row = df.filter(pl.col("asset_class") == cls)
        if len(row) > 0:
            return row["raroc"][0] * 100
        return float("nan")

    eq = get_raroc(raroc_df, "equities")
    corp = get_raroc(raroc_df, "corporate_loans")
    mort = get_raroc(raroc_df, "retail_mortgage")
    sov = get_raroc(raroc_df, "sovereign")
    total = get_raroc(raroc_df, "Total")

    gdp = macro["gdp_growth"]
    ir = macro["interest_rate"]
    problematic = "FAUX" if gdp > SCENARIO_BASE.gdp_growth and ir < SCENARIO_BASE.interest_rate else "ok"

    print(f"{name:30s} | {eq:+12.1f}% {corp:+10.1f}% {mort:+10.1f}% {sov:+9.1f}% {total:+11.1f}% | {problematic:>8s}")

# ================================================================
# 3. Zoom sur Reprise : decomposition detaillee
# ================================================================
print("\n\n[3] Zoom Reprise : decomposition Z et pd_cond pour les 14 classes")
macro_reprise = convert_scenario(PREDEFINED_SCENARIOS["Reprise"])
print(f"Macro Reprise: GDP={macro_reprise['gdp_growth']}%, unemp={macro_reprise['unemployment_rate']}%, "
      f"ir={macro_reprise['interest_rate']}%, hpi={macro_reprise['hpi_growth']}%, infl={macro_reprise['inflation_rate']}%\n")

print(f"{'Classe':25s} {'Z actuel':>9s} {'Z sans IR':>9s} {'dZ':>6s} | {'pd_cond act':>11s} {'pd_cond fix':>11s} {'delta':>8s}")
print("-" * 95)

for ac in ASSET_CLASSES:
    contribs = z_decompose(macro_reprise, ac.macro_sensitivities)
    z_total = np.clip(sum(contribs.values()), -4, 4)
    z_no_ir = np.clip(sum(v for k, v in contribs.items() if k != "interest_rate"), -4, 4)

    pd = ac.pd_base
    rho = ac.asset_correlation
    pd_cond_curr = norm.cdf((norm.ppf(pd) + np.sqrt(rho) * z_total) / np.sqrt(1 - rho))
    pd_cond_fix = norm.cdf((norm.ppf(pd) + np.sqrt(rho) * z_no_ir) / np.sqrt(1 - rho))

    delta_pct = (pd_cond_fix - pd_cond_curr) / max(pd_cond_curr, 1e-10) * 100

    print(f"{ac.name:25s} {z_total:+9.2f} {z_no_ir:+9.2f} {z_no_ir-z_total:+6.2f} | "
          f"{pd_cond_curr*100:10.4f}% {pd_cond_fix*100:10.4f}% {delta_pct:+7.1f}%")
