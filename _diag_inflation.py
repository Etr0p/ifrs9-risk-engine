"""Quantifier l'impact de la convention inflation dans macro_to_z."""
import sys
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import norm
from ifrs9_cockpit.config import PREDEFINED_SCENARIOS, SCENARIO_BASE, ASSET_CLASSES

_dm = {
    "gdp_growth": (SCENARIO_BASE.gdp_growth, 1.8),
    "unemployment_rate": (SCENARIO_BASE.unemployment_rate, 1.5),
    "interest_rate": (SCENARIO_BASE.interest_rate, 1.0),
    "hpi_growth": (SCENARIO_BASE.hpi_growth, 3.0),
    "inflation_rate": (SCENARIO_BASE.inflation_rate, 1.2),
}

def convert_scenario(s):
    bp = s.get("unemployment_bipolar", 0)
    return {
        "gdp_growth": s["gdp_pct"],
        "unemployment_rate": SCENARIO_BASE.unemployment_rate + abs(bp),
        "interest_rate": SCENARIO_BASE.interest_rate + s["interest_rate_bp"] / 100.0,
        "hpi_growth": s["hpi_pct"],
        "inflation_rate": s["inflation_pct"],
    }

def z_contribs(macro, sens):
    contribs = {}
    for var, (mu, sigma) in _dm.items():
        x = macro.get(var, mu)
        w = sens.get(var, 0.0)
        if var in ("unemployment_rate", "interest_rate"):
            contribs[var] = w * (x - mu) / sigma
        else:
            contribs[var] = -w * (x - mu) / sigma
    return contribs

# ═════════════════════════════════════════════════════════════════
# 1. Contribution de l'inflation au Z pour tous les scenarios
# ═════════════════════════════════════════════════════════════════
print("=" * 120)
print("CONTRIBUTION DE L'INFLATION AU Z-SCORE")
print("Convention actuelle: inflation haute = favorable (z -= w * delta/sigma)")
print("=" * 120)

ac_corp = [a for a in ASSET_CLASSES if a.name == "corporate_loans"][0]
ac_conso = [a for a in ASSET_CLASSES if a.name == "consumer_credit"][0]

header = f"{'Scenario':30s} | infl% | Z_infl(corp) Z_infl(conso) | Z_tot(corp) Z_tot(conso)"
print(header)
print("-" * len(header))

for name, scen in PREDEFINED_SCENARIOS.items():
    macro = convert_scenario(scen)
    infl = macro["inflation_rate"]

    c_corp = z_contribs(macro, ac_corp.macro_sensitivities)
    c_conso = z_contribs(macro, ac_conso.macro_sensitivities)

    z_infl_corp = c_corp["inflation_rate"]
    z_infl_conso = c_conso["inflation_rate"]
    z_tot_corp = np.clip(sum(c_corp.values()), -4, 4)
    z_tot_conso = np.clip(sum(c_conso.values()), -4, 4)

    print(f"{name:30s} | {infl:4.1f}% | {z_infl_corp:+11.2f}  {z_infl_conso:+12.2f}  | "
          f"{z_tot_corp:+10.2f}  {z_tot_conso:+11.2f}")

# ═════════════════════════════════════════════════════════════════
# 2. Impact si inflation traitee comme adverse
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 120)
print("IMPACT SI INFLATION TRAITEE COMME ADVERSE (z += w * delta/sigma)")
print("Focus: scenarios ou inflation > base (2.5%)")
print("=" * 120)

# All scenarios with non-zero inflation deviation
for sc_name, scen in PREDEFINED_SCENARIOS.items():
    macro = convert_scenario(scen)
    infl = macro["inflation_rate"]
    if abs(infl - SCENARIO_BASE.inflation_rate) < 0.1:
        continue  # skip scenarios near base inflation

    print(f"\n--- {sc_name} (inflation={infl}%, delta={infl - SCENARIO_BASE.inflation_rate:+.1f}pp) ---")
    print(f"  {'Classe':25s} {'Z_actuel':>9s} {'Z_infl_adv':>11s} {'dZ':>6s} | "
          f"{'pd_curr':>11s} {'pd_fix':>11s} {'delta':>8s}")
    print("  " + "-" * 95)

    for ac in ASSET_CLASSES:
        sens = ac.macro_sensitivities
        w_infl = sens.get("inflation_rate", 0)
        if w_infl < 0.01:
            continue  # skip classes with zero inflation sensitivity

        # Current: inflation favorable
        z_curr = 0.0
        for var, (mu, sigma) in _dm.items():
            x = macro.get(var, mu)
            w = sens.get(var, 0.0)
            if var in ("unemployment_rate", "interest_rate"):
                z_curr += w * (x - mu) / sigma
            else:
                z_curr -= w * (x - mu) / sigma
        z_curr_c = np.clip(z_curr, -4, 4)

        # Fixed: inflation adverse (same sign as unemployment, interest_rate)
        z_fix = 0.0
        for var, (mu, sigma) in _dm.items():
            x = macro.get(var, mu)
            w = sens.get(var, 0.0)
            if var in ("unemployment_rate", "interest_rate", "inflation_rate"):
                z_fix += w * (x - mu) / sigma
            else:
                z_fix -= w * (x - mu) / sigma
        z_fix_c = np.clip(z_fix, -4, 4)

        pd_b = ac.pd_base
        rho = ac.asset_correlation
        pd_curr = norm.cdf((norm.ppf(pd_b) + np.sqrt(rho) * z_curr_c) / np.sqrt(1 - rho))
        pd_fix = norm.cdf((norm.ppf(pd_b) + np.sqrt(rho) * z_fix_c) / np.sqrt(1 - rho))
        delta = (pd_fix - pd_curr) / max(pd_curr, 1e-10) * 100

        print(f"  {ac.name:25s} {z_curr_c:+9.2f} {z_fix_c:+11.2f} {z_fix_c - z_curr_c:+6.2f} | "
              f"{pd_curr * 100:10.4f}% {pd_fix * 100:10.4f}% {delta:+7.1f}%")

# ═════════════════════════════════════════════════════════════════
# 3. Analyse economique: quelle convention est correcte?
# ═════════════════════════════════════════════════════════════════
print("\n" + "=" * 120)
print("ANALYSE: L'INFLATION EST-ELLE ADVERSE OU FAVORABLE?")
print("=" * 120)
print("""
CANAL FAVORABLE (convention actuelle):
  - Erosion de la dette reelle: inflation reduit le poids reel de la dette
  - Correle au GDP: inflation moderee = economie en surchauffe = moins de defauts
  - Financial repression: taux reel negatif quand inflation > taux nominal

CANAL ADVERSE:
  - Couts d'intrants: inflation des couts → marges compressees → defauts
  - Pouvoir d'achat: inflation erode les revenus reels des menages → defauts conso
  - Reaction monetaire: inflation → hausse de taux (DEJA CAPTURE par IR convention)

POINT CLE: Le canal adverse via la reaction monetaire est DEJA capture par
la convention IR (taux haut = adverse). Ajouter l'inflation comme adverse
risquerait un double-comptage.

CONCLUSION: La convention actuelle (inflation favorable) est DEFENDABLE pour
le modele actuel. L'effet adverse de l'inflation passe principalement par
les taux d'interet, qui sont desormais correctement traites comme adverses.
""")
