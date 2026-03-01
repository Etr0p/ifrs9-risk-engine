#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit mathematique exhaustif -- proprietes, monotonie, cas limites.

Methode : pour chaque formule critique, on teste :
    1. MONOTONIE   -- f(x+) vs f(x-) dans la bonne direction
    2. SIGNES      -- le resultat a le bon signe dans chaque quadrant
    3. CAS LIMITES -- valeurs extremes, zeros, negatifs, NaN/inf
    4. COHERENCE   -- A > B implique f(A) > f(B) (ranking preservation)

Chaque test est une PROPRIETE (invariant mathematique), pas une valeur numerique.
"""

import sys
import os
import numpy as np
import pandas as pd

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Setup path
sys.path.insert(0, ".")
from ifrs9_cockpit.config import (
    SECTORS, SCENARIO_BASE, SCENARIOS, BASEL_CONFIG, PREDEFINED_SCENARIOS,
    SECTOR_NAMES, IFRS9_CONFIG, LGD_CONFIG,
)
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.comparator import PortfolioComparator
from ifrs9_cockpit.engine.pe_calculator import PECalculator

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {name}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {name}  {detail}")
    return condition


print("=" * 70)
print("AUDIT FORMULES -- Proprietes Mathematiques")
print("=" * 70)

# -- SETUP ---------------------------------------------------------
print("\n[SETUP] Generation des donnees...")
df_credit, df_pe, df_history = generate_dataset()
pd_suite = PDModelSuite()
pd_suite.fit(df_credit)
pd_current = pd_suite.predict(df_credit)["LR_WoE"]
lgd_model = LGDModel()
lgd_model.fit(df_credit)
ead_model = EADModel()
ead_model.fit(df_credit)
ecl_calc = ECLCalculator(lgd_model, ead_model)
pe_calc = PECalculator()

# Cache for run_pipeline to avoid redundant computations
_pipeline_cache = {}


def run_pipeline(scenario_name):
    """Execute pipeline complet pour un scenario predefini."""
    if scenario_name in _pipeline_cache:
        return _pipeline_cache[scenario_name]
    p = PREDEFINED_SCENARIOS[scenario_name]
    un_rate = SCENARIO_BASE.unemployment_rate + abs(p["unemployment_bipolar"])
    ir_rate = SCENARIO_BASE.interest_rate + p["interest_rate_bp"] / 100.0
    gdp = p["gdp_pct"]
    hpi = p["hpi_pct"]
    infl = p["inflation_pct"]
    crisis = p["unemployment_bipolar"] < 0

    res_stress = ecl_calc.calculate(
        df_credit, pd_current, df_credit["pd_origination"].values,
        unemployment_override=un_rate, gdp_override=gdp,
        interest_rate_override=ir_rate, hpi_override=hpi, inflation_override=infl,
    )
    res_pe = pe_calc.calculate(
        df_pe, unemployment_override=un_rate, gdp_override=gdp,
        interest_rate_override=ir_rate, hpi_override=hpi, inflation_override=infl,
        unemployment_crisis=crisis,
    )
    comp = PortfolioComparator(res_stress, res_pe)
    raroc_df = comp.compute_raroc_eva()
    opt = comp.optimize_allocation()
    result = {
        "ecl": float(res_stress["ecl_weighted"].sum()),
        "raroc_credit": float(raroc_df.loc[(raroc_df["canal"] == "Credit") & (raroc_df["sector"] == "Total"), "raroc"].values[0]),
        "raroc_pe": float(raroc_df.loc[(raroc_df["canal"] == "PE") & (raroc_df["sector"] == "Total"), "raroc"].values[0]),
        "nav": float(res_pe["nav"].sum()),
        "drawdown": float(max(0, 1 - res_pe["nav"].sum() / max(res_pe["capital_invested"].sum(), 1))),
        "opt": opt,
        "raroc_df": raroc_df,
        "res_stress": res_stress,
        "res_pe": res_pe,
    }
    _pipeline_cache[scenario_name] = result
    return result


# ===================================================================
# 1. MARKOWITZ TANGENCY -- CAS LIMITES
# ===================================================================
print("\n" + "=" * 70)
print("1. MARKOWITZ _markowitz_tangency() -- Cas Limites")
print("=" * 70)

tangency = PortfolioComparator._markowitz_tangency

# Q1: Les deux positifs, PE meilleur -> PE > 0
w = tangency(0.04, 0.08, 0.05, 0.15, 0.5)
check("Q1: mu_c=4%, mu_p=8% -> w_pe > 0", w > 0, f"w_pe={w:.2f}")

# Q2: Les deux positifs, Credit meilleur -> PE < 50%
w = tangency(0.08, 0.04, 0.05, 0.15, 0.5)
check("Q2: mu_c=8%, mu_p=4% -> w_pe < 50%", w < 0.50, f"w_pe={w:.2f}")

# Q3: Les deux negatifs -> 100% dans le moins mauvais
w = tangency(-0.02, -0.12, 0.05, 0.15, 0.5)
check("Q3: mu_c=-2%, mu_p=-12% -> w_pe=0 (Credit moins mauvais)",
      w == 0.0, f"w_pe={w:.2f}")

w = tangency(-0.12, -0.02, 0.05, 0.15, 0.5)
check("Q4: mu_c=-12%, mu_p=-2% -> w_pe=1 (PE moins mauvais)",
      w == 1.0, f"w_pe={w:.2f}")

# Q5: Un positif, un negatif
w = tangency(0.04, -0.10, 0.05, 0.15, 0.5)
check("Q5: mu_c=4%, mu_p=-10% -> w_pe=0 (seul Credit positif)",
      w == 0.0, f"w_pe={w:.2f}")

w = tangency(-0.10, 0.04, 0.05, 0.15, 0.5)
check("Q6: mu_c=-10%, mu_p=4% -> w_pe=1 (seul PE positif)",
      w == 1.0, f"w_pe={w:.2f}")

# Q7: Cas zero
w = tangency(0.0, 0.0, 0.05, 0.15, 0.5)
check("Q7: mu_c=0, mu_p=0 -> w_pe=0 (les deux zero = negatif-like)",
      w == 0.0, f"w_pe={w:.2f}")

# Q8: Volatilites egales, rendements egaux -> 50% (diversification)
w = tangency(0.05, 0.05, 0.10, 0.10, 0.0)
check("Q8: mu=sigma identiques, rho=0 -> ~50%", abs(w - 0.50) < 0.01, f"w_pe={w:.2f}")

# Q9: Correlation parfaite (rho=0.99) -> pas de benefice de diversification
w_lo = tangency(0.04, 0.08, 0.05, 0.15, 0.0)
w_hi = tangency(0.04, 0.08, 0.05, 0.15, 0.99)
check("Q9: rho=0 vs rho=0.99 -> w_pe different", abs(w_lo - w_hi) > 0.01,
      f"w(rho=0)={w_lo:.2f}, w(rho=0.99)={w_hi:.2f}")

# Q10: w_pe toujours dans [0, 1]
for mu_c, mu_p, sc, sp, rho in [
    (0.20, 0.01, 0.01, 0.50, -0.5),
    (0.01, 0.20, 0.50, 0.01, -0.5),
    (0.001, 0.001, 0.001, 0.001, 0.0),
]:
    w = tangency(mu_c, mu_p, sc, sp, rho)
    check(f"Q10: w_pe in [0,1] (mu_c={mu_c}, mu_p={mu_p})", 0.0 <= w <= 1.0, f"w_pe={w:.4f}")


# ===================================================================
# 2. RAROC PE -- COHERENCE AVEC NAV
# ===================================================================
print("\n" + "=" * 70)
print("2. RAROC PE -- Coherence avec NAV (IRR)")
print("=" * 70)

# Scenario favorable: NAV > capital -> MOIC > 1 -> IRR > 0 -> RAROC > 0
r_fav = run_pipeline("Reprise")
cap_inv_fav = float(r_fav["res_pe"]["capital_invested"].sum())
check("Reprise: NAV > capital_invested (MOIC > 1)",
      r_fav["nav"] > cap_inv_fav,
      f"NAV={r_fav['nav']:.0f}, Cap={cap_inv_fav:.0f}")
check("Reprise: RAROC PE > 0 (quand IRR > 0)", r_fav["raroc_pe"] > 0,
      f"RAROC_PE={r_fav['raroc_pe']:.4f}")

# Scenario adverse: NAV stresse < NAV baseline et RAROC PE < 0
r_adv = run_pipeline("Crise financiere (GFC)")
nav_adv = r_adv["nav"]
nav_fav = r_fav["nav"]
check("GFC: NAV < NAV Reprise (stress reduit la NAV)",
      nav_adv < nav_fav,
      f"NAV_GFC={nav_adv:.0f}, NAV_Reprise={nav_fav:.0f}")
check("GFC: RAROC PE < 0 (perte en capital)", r_adv["raroc_pe"] < 0,
      f"RAROC_PE={r_adv['raroc_pe']:.4f}")

# Monotonie: RAROC PE favorable > base > adverse
r_base = run_pipeline("Central")
check("Monotonie RAROC PE: Reprise > Central",
      r_fav["raroc_pe"] > r_base["raroc_pe"],
      f"Reprise={r_fav['raroc_pe']:.4f}, Central={r_base['raroc_pe']:.4f}")
check("Monotonie RAROC PE: Central > GFC",
      r_base["raroc_pe"] > r_adv["raroc_pe"],
      f"Central={r_base['raroc_pe']:.4f}, GFC={r_adv['raroc_pe']:.4f}")

# Equity risk premium: RAROC PE > RAROC Credit en scenario favorable
check("Equity risk premium: RAROC PE > RAROC Credit en Reprise",
      r_fav["raroc_pe"] > r_fav["raroc_credit"],
      f"RAROC_PE={r_fav['raroc_pe']:.4f}, RAROC_C={r_fav['raroc_credit']:.4f}")

# First-loss: RAROC PE < RAROC Credit en scenario adverse
check("First-loss: RAROC PE < RAROC Credit en GFC",
      r_adv["raroc_pe"] < r_adv["raroc_credit"],
      f"RAROC_PE={r_adv['raroc_pe']:.4f}, RAROC_C={r_adv['raroc_credit']:.4f}")


# ===================================================================
# 3. MARKOWITZ x SCENARIOS -- Coherence de l'allocation ideale
# ===================================================================
print("\n" + "=" * 70)
print("3. MARKOWITZ ALLOCATION -- Coherence par scenario")
print("=" * 70)

scenarios_to_test = [
    "Central", "Reprise", "Crise financiere (GFC)",
    "Stagflation", "Choc pandemique (COVID)",
]

for sc_name in scenarios_to_test:
    r = run_pipeline(sc_name)
    rc = r["raroc_credit"]
    rp = r["raroc_pe"]
    ideal_pe = r["opt"]["ideal_pe_alloc"]
    optimal_pe = r["opt"]["optimal_pe_alloc"]

    # Propriete fondamentale: si les deux RAROC < 0, ideal doit etre 0% PE (Credit moins mauvais)
    # ou 100% PE (PE moins mauvais)
    if rc < 0 and rp < 0:
        if rc >= rp:
            check(f"{sc_name}: RAROC<0 & Credit meilleur -> ideal_pe=0%",
                  ideal_pe == 0.0,
                  f"RAROC_C={rc:.2%}, RAROC_PE={rp:.2%}, ideal_pe={ideal_pe:.0%}")
        else:
            check(f"{sc_name}: RAROC<0 & PE meilleur -> ideal_pe=100%",
                  ideal_pe == 1.0,
                  f"RAROC_C={rc:.2%}, RAROC_PE={rp:.2%}, ideal_pe={ideal_pe:.0%}")
    elif rc >= 0 and rp < 0:
        check(f"{sc_name}: Credit>0 & PE<0 -> ideal_pe=0%",
              ideal_pe == 0.0,
              f"RAROC_C={rc:.2%}, RAROC_PE={rp:.2%}, ideal_pe={ideal_pe:.0%}")
    elif rc < 0 and rp >= 0:
        check(f"{sc_name}: Credit<0 & PE>0 -> ideal_pe=100%",
              ideal_pe == 1.0,
              f"RAROC_C={rc:.2%}, RAROC_PE={rp:.2%}, ideal_pe={ideal_pe:.0%}")
    else:
        # Les deux positifs: l'allocation ideale doit etre dans [0, 1]
        check(f"{sc_name}: RAROC>0 -> ideal_pe in [0,1]",
              0.0 <= ideal_pe <= 1.0,
              f"RAROC_C={rc:.2%}, RAROC_PE={rp:.2%}, ideal_pe={ideal_pe:.0%}")

    # L'allocation optimale (apres CET1) doit toujours etre <= ideal
    check(f"{sc_name}: optimal_pe <= ideal_pe (CET1 ne fait que reduire)",
          optimal_pe <= ideal_pe + 0.01,
          f"optimal={optimal_pe:.0%}, ideal={ideal_pe:.0%}")

    # L'allocation optimale doit etre faisable
    check(f"{sc_name}: allocation optimale faisable",
          r["opt"]["feasible"],
          f"CET1={r['opt']['cet1_ratio']:.2%}")


# ===================================================================
# 4. ECL -- MONOTONIE ET COHERENCE
# ===================================================================
print("\n" + "=" * 70)
print("4. ECL -- Monotonie et Coherence")
print("=" * 70)

# ECL: pire scenario -> ECL plus eleve
r_stag = run_pipeline("Stagflation")
r_reprise = run_pipeline("Reprise")

check("ECL Stagflation > ECL Central",
      r_stag["ecl"] > r_base["ecl"],
      f"Stag={r_stag['ecl']:.0f}, Central={r_base['ecl']:.0f}")
check("ECL Central > ECL Reprise",
      r_base["ecl"] > r_reprise["ecl"],
      f"Central={r_base['ecl']:.0f}, Reprise={r_reprise['ecl']:.0f}")

# RAROC Credit monotonie
check("RAROC Credit: Reprise > Central",
      r_reprise["raroc_credit"] > r_base["raroc_credit"],
      f"Reprise={r_reprise['raroc_credit']:.4f}, Central={r_base['raroc_credit']:.4f}")
check("RAROC Credit: Central > Stagflation",
      r_base["raroc_credit"] > r_stag["raroc_credit"],
      f"Central={r_base['raroc_credit']:.4f}, Stag={r_stag['raroc_credit']:.4f}")


# ===================================================================
# 5. RAROC CREDIT FORMULA -- Revenue fixe a l'origination
# ===================================================================
print("\n" + "=" * 70)
print("5. RAROC Credit -- Revenue fixe a l'origination")
print("=" * 70)

# Le NII (revenu) doit etre le MEME quel que soit le scenario
# car il est calcule sur base_default_rate (fixe a l'origination)
r1 = r_base["raroc_df"]
r2 = r_stag["raroc_df"]
rev_base = float(r1.loc[(r1["canal"] == "Credit") & (r1["sector"] == "Total"), "revenue"].values[0])
rev_stag = float(r2.loc[(r2["canal"] == "Credit") & (r2["sector"] == "Total"), "revenue"].values[0])
# NII vient de ead x spread_origination. EAD varie un peu (CCF adverse) mais pas beaucoup.
# On verifie que la variation est < 5%
delta_rev = abs(rev_base - rev_stag) / max(rev_base, 1)
check("NII Credit quasi-stable entre scenarios (<5% variation)",
      delta_rev < 0.05,
      f"Rev Base={rev_base:.0f}, Rev Stag={rev_stag:.0f}, delta={delta_rev:.1%}")

# Loss doit augmenter avec le stress
loss_base = float(r1.loc[(r1["canal"] == "Credit") & (r1["sector"] == "Total"), "loss"].values[0])
loss_stag = float(r2.loc[(r2["canal"] == "Credit") & (r2["sector"] == "Total"), "loss"].values[0])
check("EL Credit: Stagflation > Central",
      loss_stag > loss_base,
      f"EL Base={loss_base:.0f}, EL Stag={loss_stag:.0f}")


# ===================================================================
# 6. RAROC PE FORMULA -- rev_pe = NAV x max(0, IRR)
# ===================================================================
print("\n" + "=" * 70)
print("6. RAROC PE -- Formule rev_pe = NAV x max(0, IRR)")
print("=" * 70)

# Quand IRR > 0, rev_pe > 0
rev_pe_fav = float(r_fav["raroc_df"].loc[
    (r_fav["raroc_df"]["canal"] == "PE") & (r_fav["raroc_df"]["sector"] == "Total"), "revenue"
].values[0])
check("Reprise: rev_pe > 0 (IRR > 0)",
      rev_pe_fav > 0,
      f"rev_pe={rev_pe_fav:.0f}")

# Revenue PE GFC < Reprise (monotonie)
r_gfc = r_adv
rev_pe_gfc = float(r_gfc["raroc_df"].loc[
    (r_gfc["raroc_df"]["canal"] == "PE") & (r_gfc["raroc_df"]["sector"] == "Total"), "revenue"
].values[0])
check("GFC: rev_pe < rev_pe Reprise (stress reduit le revenu)",
      rev_pe_gfc < rev_pe_fav,
      f"rev_pe_GFC={rev_pe_gfc:.0f}, rev_pe_Reprise={rev_pe_fav:.0f}")

# Revenue PE doit etre sensible au scenario (pas constant)
rev_pe_central = float(r_base["raroc_df"].loc[
    (r_base["raroc_df"]["canal"] == "PE") & (r_base["raroc_df"]["sector"] == "Total"), "revenue"
].values[0])
check("Rev PE: Reprise > Central",
      rev_pe_fav > rev_pe_central,
      f"Reprise={rev_pe_fav:.0f}, Central={rev_pe_central:.0f}")


# ===================================================================
# 7. SOFTMAX WEIGHTS -- Bornes et somme
# ===================================================================
print("\n" + "=" * 70)
print("7. SOFTMAX WEIGHTS -- Proprietes")
print("=" * 70)

for sc_name in ["Central", "Stagflation", "Reprise"]:
    r = run_pipeline(sc_name)
    for canal, key in [("Credit", "sector_weights_credit"), ("PE", "sector_weights_pe")]:
        weights = r["opt"][key]
        w_vals = list(weights.values())
        w_sum = sum(w_vals)
        check(f"{sc_name} {canal}: poids somment a 1.0",
              abs(w_sum - 1.0) < 0.01,
              f"sum={w_sum:.4f}")
        check(f"{sc_name} {canal}: chaque poids in [5%, 70%]",
              all(0.049 <= w <= 0.701 for w in w_vals),
              f"poids={[f'{w:.1%}' for w in w_vals]}")


# ===================================================================
# 8. CET1 COHERENCE
# ===================================================================
print("\n" + "=" * 70)
print("8. CET1 -- Coherence reglementaire")
print("=" * 70)

for sc_name in scenarios_to_test:
    r = run_pipeline(sc_name)
    cet1 = r["opt"]["cet1_ratio"]
    feasible = r["opt"]["feasible"]
    if feasible:
        check(f"{sc_name}: CET1 >= 13% si faisable",
              cet1 >= BASEL_CONFIG.cet1_target - 0.001,
              f"CET1={cet1:.2%}")
    else:
        check(f"{sc_name}: CET1 < 13% si non faisable",
              cet1 < BASEL_CONFIG.cet1_target,
              f"CET1={cet1:.2%}")


# ===================================================================
# 9. PE NAV -- Bornes et clipping
# ===================================================================
print("\n" + "=" * 70)
print("9. PE Model -- NAV bornes et monotonie")
print("=" * 70)

from ifrs9_cockpit.models.pe_model import PEModel

pe_model = PEModel()
nav_base_pe, _ = pe_model.calculate_nav(df_pe)
check("NAV PE toutes positives", (nav_base_pe > 0).all(),
      f"min={nav_base_pe.min():.6f}")

# Rate up -> NAV down (taux penalisent les multiples et le leverage)
pe_model2 = PEModel()
nav_rate_up, _ = pe_model2.calculate_nav(df_pe, interest_rate_override=6.0)
pe_model3 = PEModel()
nav_rate_down, _ = pe_model3.calculate_nav(df_pe, interest_rate_override=1.0)
check("NAV: taux=1% > taux=6%",
      nav_rate_down.sum() > nav_rate_up.sum(),
      f"NAV(1%)={nav_rate_down.sum():.0f}, NAV(6%)={nav_rate_up.sum():.0f}")

# GDP up -> NAV up
pe_model4 = PEModel()
nav_gdp_up, _ = pe_model4.calculate_nav(df_pe, gdp_override=5.0)
pe_model5 = PEModel()
nav_gdp_down, _ = pe_model5.calculate_nav(df_pe, gdp_override=-3.0)
check("NAV: GDP=+5% > GDP=-3%",
      nav_gdp_up.sum() > nav_gdp_down.sum(),
      f"NAV(+5%)={nav_gdp_up.sum():.0f}, NAV(-3%)={nav_gdp_down.sum():.0f}")


# ===================================================================
# 10. VOLATILITE ESTIMATION -- Planchers respectes
# ===================================================================
print("\n" + "=" * 70)
print("10. Volatilite -- Planchers et estimation")
print("=" * 70)

for sc_name in ["Central", "Stagflation"]:
    r = run_pipeline(sc_name)
    sc = r["opt"]["sigma_credit"]
    sp = r["opt"]["sigma_pe"]
    rho = r["opt"]["rho_credit_pe"]
    check(f"{sc_name}: sigma_credit >= 5%", sc >= 0.05, f"sigma_c={sc:.2%}")
    check(f"{sc_name}: sigma_pe >= 15%", sp >= 0.15, f"sigma_p={sp:.2%}")
    check(f"{sc_name}: rho in [-0.90, 0.95]",
          -0.90 <= rho <= 0.95, f"rho={rho:.2f}")

# En scenario Central, PE doit etre plus volatil que Credit
r_central = run_pipeline("Central")
check("Central: sigma_pe > sigma_credit (PE plus risque en base)",
      r_central["opt"]["sigma_pe"] > r_central["opt"]["sigma_credit"],
      f"sigma_c={r_central['opt']['sigma_credit']:.2%}, sigma_p={r_central['opt']['sigma_pe']:.2%}")


# ===================================================================
# BILAN
# ===================================================================
print("\n" + "=" * 70)
total = PASS_COUNT + FAIL_COUNT
print(f"BILAN : {PASS_COUNT}/{total} PASS | {FAIL_COUNT} FAIL")
if FAIL_COUNT == 0:
    print("Toutes les proprietes mathematiques sont respectees.")
else:
    print(f"ATTENTION : {FAIL_COUNT} proprietes violees -- a corriger.")
print("=" * 70)
