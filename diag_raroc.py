"""Diagnostic RAROC Credit — Multi-scenario analysis."""
import numpy as np
import pandas as pd
from ifrs9_cockpit.config import (
    PREDEFINED_SCENARIOS, SCENARIO_BASE, SECTORS, LGD_CONFIG, BASEL_CONFIG, RANDOM_SEED
)
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.engine.comparator import PortfolioComparator as Comparator

# Generate + train once
df_credit, df_pe, _ = generate_dataset()
pd_model = PDModelSuite(seed=RANDOM_SEED)
pd_model.fit(df_credit)
lgd_model = LGDModel(seed=RANDOM_SEED)
lgd_model.fit(df_credit)
ead_model = EADModel(seed=RANDOM_SEED)
ead_model.fit(df_credit)
ecl_calc = ECLCalculator(lgd_model, ead_model)
pe_calc = PECalculator()

# PD is scenario-independent (trained once, macro stress applied via ECL overrides)
pd_current = pd_model.predict(df_credit)["LR_WoE"]

print(f"{'Scenario':<25} {'RAROC':>7} {'ECL/EAD':>8} {'EL(Md)':>8} {'NII(Md)':>9} {'Profit(Md)':>11}")
print("=" * 72)

results_all = {}
for name, params in sorted(PREDEFINED_SCENARIOS.items()):
    un_rate = SCENARIO_BASE.unemployment_rate + params["unemployment_bipolar"]
    ir = SCENARIO_BASE.interest_rate + params["interest_rate_bp"] / 100.0
    result = ecl_calc.calculate(
        df_credit, pd_current, df_credit["pd_origination"].values,
        unemployment_override=un_rate,
        gdp_override=params["gdp_pct"],
        interest_rate_override=ir,
        hpi_override=params["hpi_pct"],
        inflation_override=params["inflation_pct"],
    )
    res_pe = pe_calc.calculate(
        df_pe, unemployment_override=un_rate,
        gdp_override=params["gdp_pct"],
        interest_rate_override=ir,
        hpi_override=params["hpi_pct"],
        inflation_override=params["inflation_pct"],
    )
    comp = Comparator(result, res_pe)
    raroc_df = comp.compute_raroc_eva()

    credit_total = raroc_df[(raroc_df["canal"] == "Credit") & (raroc_df["sector"] == "Total")]
    if len(credit_total) > 0:
        row = credit_total.iloc[0]
        raroc = row["raroc"]
        nii = row["revenue"]
        el = row["loss"]
        capital = row["capital"]
        prof = (nii * (1 - BASEL_CONFIG.cir) - el) * (1 - BASEL_CONFIG.tax_rate)

        ecl_total = result["ecl_weighted"].sum()
        ead_total = result["ead"].sum()
        ecl_ead = ecl_total / max(ead_total, 1)

        marker = " <--" if name == "Reprise" else ""
        print(f"{name:<25} {raroc:>6.1%} {ecl_ead:>7.2%} {el/1e9:>7.1f}B {nii/1e9:>8.1f}B {prof/1e9:>10.1f}B{marker}")
        results_all[name] = {"raroc": raroc, "ecl_ead": ecl_ead, "el": el, "nii": nii}

print()
print(f"Hurdle (k_e) = {BASEL_CONFIG.cost_of_equity:.0%} | CIR = {BASEL_CONFIG.cir:.0%} | Tax = {BASEL_CONFIG.tax_rate:.0%} | CET1 = {BASEL_CONFIG.cet1_target:.0%}")

# Detail par secteur pour Reprise
print("\n--- Detail Reprise par secteur ---")
params = PREDEFINED_SCENARIOS["Reprise"]
un_rate = SCENARIO_BASE.unemployment_rate + params["unemployment_bipolar"]
ir = SCENARIO_BASE.interest_rate + params["interest_rate_bp"] / 100.0
result = ecl_calc.calculate(
    df_credit, pd_current, df_credit["pd_origination"].values,
    unemployment_override=un_rate,
    gdp_override=params["gdp_pct"],
    interest_rate_override=ir,
    hpi_override=params["hpi_pct"],
    inflation_override=params["inflation_pct"],
)
res_pe = pe_calc.calculate(
    df_pe, unemployment_override=un_rate,
    gdp_override=params["gdp_pct"],
    interest_rate_override=ir,
    hpi_override=params["hpi_pct"],
    inflation_override=params["inflation_pct"],
)
comp = Comparator(result, res_pe)
raroc_df = comp.compute_raroc_eva()

credit_rows = raroc_df[(raroc_df["canal"] == "Credit") & (raroc_df["sector"] != "Total")]
print(f"{'Sector':<15} {'RAROC':>7} {'Spread':>7} {'EL/EAD':>7} {'EVA(M)':>8}")
for _, r in credit_rows.iterrows():
    el_ead = r["loss"] / max(r["exposure"], 1)
    nii_ead = r["revenue"] / max(r["exposure"], 1)
    print(f"{r['sector']:<15} {r['raroc']:>6.1%} {nii_ead:>6.2%} {el_ead:>6.2%} {r['eva']/1e6:>7.0f}M")

# Decomposition NII
print("\n--- Decomposition du spread (NII/EAD) ---")
for sector in SECTORS:
    cs = float(np.clip(
        -np.log(max(1 - sector.base_default_rate * LGD_CONFIG.lgd_ttc_mean, 1e-10))
        + (BASEL_CONFIG.liquidity_premium_bps + BASEL_CONFIG.commercial_margin_bps) / 10000,
        0.005, 0.2,
    ))
    merton = -np.log(max(1 - sector.base_default_rate * LGD_CONFIG.lgd_ttc_mean, 1e-10))
    liq = BASEL_CONFIG.liquidity_premium_bps / 10000
    margin = BASEL_CONFIG.commercial_margin_bps / 10000
    print(f"  {sector.name:<15} DR={sector.base_default_rate:.1%}  Merton={merton:.2%}  +Liq={liq:.2%}  +Margin={margin:.2%}  => cs={cs:.2%}")

# RAROC decomposition
print("\n--- RAROC Formula Decomposition (Reprise) ---")
ct = raroc_df[(raroc_df["canal"] == "Credit") & (raroc_df["sector"] == "Total")].iloc[0]
nii, el, rwa, cap = ct["revenue"], ct["loss"], ct["rwa"], ct["capital"]
cir, tax = BASEL_CONFIG.cir, BASEL_CONFIG.tax_rate
prof = (nii * (1 - cir) - el) * (1 - tax)
print(f"  NII (EAD x spread)     = {nii/1e9:.2f} Md EUR")
print(f"  NII x (1-CIR)          = {nii*(1-cir)/1e9:.2f} Md EUR  (CIR={cir:.0%})")
print(f"  EL (PD x LGD x EAD)    = {el/1e9:.2f} Md EUR")
print(f"  Profit avant impot     = {(nii*(1-cir)-el)/1e9:.2f} Md EUR")
print(f"  Profit apres impot     = {prof/1e9:.2f} Md EUR  (Tax={tax:.0%})")
print(f"  RWA                    = {rwa/1e9:.2f} Md EUR")
print(f"  Capital (RWA x CET1)   = {cap/1e9:.2f} Md EUR  (CET1={BASEL_CONFIG.cet1_target:.0%})")
print(f"  RAROC = Profit/Capital = {prof/max(cap,1):.2%}")
print(f"  Hurdle (k_e)           = {BASEL_CONFIG.cost_of_equity:.0%}")
print(f"  EVA = (RAROC-k_e)*Cap  = {(prof/max(cap,1)-BASEL_CONFIG.cost_of_equity)*cap/1e6:.0f} MEUR")
