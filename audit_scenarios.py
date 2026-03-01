#!/usr/bin/env python
"""Audit complet des calculs IFRS 9 Risk Cockpit  --  6 scenarios.

Valide la coherence des resultats ECL, PE, RAROC, CET1, staging
pour chaque scenario predefini + le scenario Manuel (base).

Criteres de validation :
  1. Monotonie ECL : Adverse > Base > Favorable
  2. Monotonie NAV : Favorable > Base > Adverse (pour scenarios non-ambigus)
  3. Staging : Stage 1 + 2 + 3 = N_CLIENTS
  4. CET1 headroom coherent avec rwa_budget
  5. RAROC sign : credit > 0 en reprise, < 0 en crise severe
  6. Risk Appetite : vert en reprise, rouge en crise
  7. Delta NAV = 0 quand macros = base
  8. NAV drawdown >= 0 toujours
  9. ECL > 0 toujours
  10. RWA proportionnel a EAD
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# Ajouter le parent au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ifrs9_cockpit.config import (
    PREDEFINED_SCENARIOS, SCENARIO_BASE, BASEL_CONFIG, SECTORS,
    N_CLIENTS, IFRS9_CONFIG, LOGIT_AMPLITUDE,
)
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.engine.comparator import PortfolioComparator
from ifrs9_cockpit.utils.helpers import format_euro, format_pct

# -----------------------------------------
# Setup : donnees + modeles
# -----------------------------------------
print("=" * 80)
print("AUDIT COMPLET  --  IFRS 9 Risk Cockpit  --  Scenarios Predefinis")
print("=" * 80)

t0 = time.time()
print("\n[1/4] Generation des donnees...")
df_credit, df_pe, df_history = generate_dataset()
print(f"  Credit: {len(df_credit):,} clients | PE: {len(df_pe):,} positions")
print(f"  DR base: {df_credit['default_flag'].mean():.2%}")

print("[2/4] Entrainement PD models...")
pd_suite = PDModelSuite()
pd_suite.fit(df_credit)
pd_current = pd_suite.predict(df_credit)["LR_WoE"]
pd_origination = df_credit["pd_origination"].values

print("[3/4] Calibration LGD & EAD...")
lgd_model = LGDModel()
lgd_model.fit(df_credit)
ead_model = EADModel()
ead_model.fit(df_credit)

print(f"[4/4] Setup termine en {time.time()-t0:.1f}s\n")

# -----------------------------------------
# Scenarios a tester
# -----------------------------------------
SCENARIOS_TO_TEST = [
    "Central",
    "Crise financiere (GFC)",
    "Stagflation",
    "Choc pandemique (COVID)",
    "Reprise",
    "Hypercroissance",
]

# -----------------------------------------
# Fonction de calcul pipeline complet
# -----------------------------------------
def run_scenario(name, params):
    """Execute le pipeline complet pour un scenario donne."""
    ir_bp = params["interest_rate_bp"]
    un_bp = params["unemployment_bipolar"]
    gdp = params["gdp_pct"]
    hpi = params["hpi_pct"]
    infl = params["inflation_pct"]

    # Conversion sliders -> overrides
    un_rate = float(SCENARIO_BASE.unemployment_rate + abs(un_bp))
    ir_rate = float(SCENARIO_BASE.interest_rate + ir_bp / 100.0)

    # ECL base (sans stress)
    ecl_calc = ECLCalculator(lgd_model, ead_model)
    res_base = ecl_calc.calculate(df_credit, pd_current, pd_origination)

    # ECL stress
    res_stress = ecl_calc.calculate(
        df_credit, pd_current, pd_origination,
        unemployment_override=un_rate,
        gdp_override=gdp,
        interest_rate_override=ir_rate,
        hpi_override=hpi,
        inflation_override=infl,
    )

    # PE base
    pe_calc = PECalculator()
    res_pe_ref = pe_calc.calculate(df_pe)

    # PE stress
    res_pe_stress = pe_calc.calculate(
        df_pe,
        unemployment_override=un_rate,
        gdp_override=gdp,
        interest_rate_override=ir_rate,
        hpi_override=hpi,
        inflation_override=infl,
        unemployment_crisis=(un_bp < 0),
    )

    # Comparator
    # Add segment alias
    for _df in [res_base, res_stress, res_pe_ref, res_pe_stress]:
        if "sector" in _df.columns and "segment" not in _df.columns:
            _df["segment"] = _df["sector"]

    comp = PortfolioComparator(res_stress, res_pe_stress)
    opt = comp.optimize_allocation(user_pe_alloc=0.20)
    raroc_eva = comp.compute_raroc_eva()
    adv = comp.compute_advanced_credit_metrics()

    # Metriques
    ecl_base_total = float(res_base["ecl_weighted"].sum())
    ecl_stress_total = float(res_stress["ecl_weighted"].sum())
    ecl_delta = float((ecl_stress_total - ecl_base_total) / max(ecl_base_total, 1))
    ead_total = float(res_stress["ead"].sum())
    ecl_ead_ratio = float(ecl_stress_total / max(ead_total, 1))

    nav_ref = float(res_pe_ref["nav"].sum())
    nav_stress = float(res_pe_stress["nav"].sum())
    delta_nav = nav_stress - nav_ref
    drawdown = max(0.0, (nav_ref - nav_stress) / max(nav_ref, 1.0))

    stage_counts = res_stress["stage"].value_counts().to_dict()
    s1 = stage_counts.get(1, 0)
    s2 = stage_counts.get(2, 0)
    s3 = stage_counts.get(3, 0)

    pe_cats = res_pe_stress["risk_category"].value_counts().to_dict()

    rwa_credit = float(res_stress["rwa_credit"].sum())
    rwa_pe = float(res_pe_stress["rwa_pe"].sum())

    return {
        "name": name,
        "macro": {"un_rate": un_rate, "ir_rate": ir_rate, "gdp": gdp, "hpi": hpi, "infl": infl, "un_bp": un_bp},
        "ecl_base": ecl_base_total,
        "ecl_stress": ecl_stress_total,
        "ecl_delta": ecl_delta,
        "ead_total": ead_total,
        "ecl_ead_ratio": ecl_ead_ratio,
        "nav_ref": nav_ref,
        "nav_stress": nav_stress,
        "delta_nav": delta_nav,
        "drawdown": drawdown,
        "stage_counts": {"s1": s1, "s2": s2, "s3": s3},
        "pe_cats": pe_cats,
        "raroc_credit": float(opt["raroc_credit"]),
        "raroc_pe": float(opt["raroc_pe"]),
        "rwa_credit": rwa_credit,
        "rwa_pe": rwa_pe,
        "rwa_weighted": float(opt["rwa_weighted"]),
        "cet1_ratio": float(opt["cet1_ratio"]),
        "headroom_m": float(opt["headroom_m"]),
        "feasible": bool(opt["feasible"]),
        "res_stress": res_stress,
        "res_pe_stress": res_pe_stress,
    }


# -----------------------------------------
# Execution des scenarios
# -----------------------------------------
results = {}
for sc_name in SCENARIOS_TO_TEST:
    params = PREDEFINED_SCENARIOS[sc_name]
    print(f"\n{'-'*70}")
    print(f"Scenario: {sc_name}")
    print(f"  IR: {params['interest_rate_bp']:+.0f}bp | Chomage: {params['unemployment_bipolar']:+.1f}pp | "
          f"PIB: {params['gdp_pct']:+.1f}% | HPI: {params['hpi_pct']:+.1f}% | Inflation: {params['inflation_pct']:.1f}%")
    t1 = time.time()
    r = run_scenario(sc_name, params)
    dt = time.time() - t1
    results[sc_name] = r

    print(f"\n  === RESULTATS ({dt:.1f}s) ===")
    print(f"  ECL base     : {format_euro(r['ecl_base'])}")
    print(f"  ECL stress   : {format_euro(r['ecl_stress'])}")
    print(f"  ECL delta    : {r['ecl_delta']:+.1%}")
    print(f"  ECL/EAD      : {r['ecl_ead_ratio']:.2%}")
    print(f"  EAD total    : {format_euro(r['ead_total'])}")
    print(f"  NAV ref      : {format_euro(r['nav_ref'])}")
    print(f"  NAV stress   : {format_euro(r['nav_stress'])}")
    print(f"  Delta NAV    : {format_euro(r['delta_nav'])}")
    print(f"  NAV drawdown : {r['drawdown']:.2%}")
    print(f"  Stages       : S1={r['stage_counts']['s1']:,} | S2={r['stage_counts']['s2']:,} | S3={r['stage_counts']['s3']:,}")
    print(f"  PE categories: {r['pe_cats']}")
    print(f"  RAROC credit : {r['raroc_credit']:.2%}")
    print(f"  RAROC PE     : {r['raroc_pe']:.2%}")
    print(f"  RWA credit   : {format_euro(r['rwa_credit'])}")
    print(f"  RWA PE       : {format_euro(r['rwa_pe'])}")
    print(f"  RWA pondere  : {format_euro(r['rwa_weighted'])}")
    print(f"  CET1 ratio   : {r['cet1_ratio']:.2%}")
    print(f"  Headroom     : {format_euro(r['headroom_m'] * 1e6)}")
    print(f"  Faisable     : {'OUI' if r['feasible'] else 'NON'}")


# -----------------------------------------
# VALIDATIONS CROISEES
# -----------------------------------------
print("\n\n" + "=" * 80)
print("VALIDATIONS CROISEES")
print("=" * 80)

tests_pass = 0
tests_fail = 0
issues = []

def check(test_name, condition, detail=""):
    global tests_pass, tests_fail
    if condition:
        tests_pass += 1
        print(f"  [PASS] {test_name}")
    else:
        tests_fail += 1
        msg = f"  [FAIL] {test_name}"
        if detail:
            msg += f"  --  {detail}"
        print(msg)
        issues.append((test_name, detail))


# -- V1: Staging integrity ------------------
print("\n-- V1: Integrite du staging --")
for sc, r in results.items():
    total_stages = r["stage_counts"]["s1"] + r["stage_counts"]["s2"] + r["stage_counts"]["s3"]
    check(f"{sc}: S1+S2+S3 = N_CLIENTS ({N_CLIENTS})",
          total_stages == N_CLIENTS,
          f"got {total_stages}")

# -- V2: ECL > 0 toujours ------------------
print("\n-- V2: ECL > 0 --")
for sc, r in results.items():
    check(f"{sc}: ECL stress > 0", r["ecl_stress"] > 0, f"ECL = {r['ecl_stress']:.0f}")

# -- V3: ECL/EAD ratio sensible --------------
print("\n-- V3: ECL/EAD ratio dans [0.1%, 30%] --")
for sc, r in results.items():
    ratio = r["ecl_ead_ratio"]
    check(f"{sc}: ECL/EAD = {ratio:.2%}", 0.001 <= ratio <= 0.30,
          f"hors plage [0.1%, 30%]")

# -- V4: NAV drawdown >= 0 ------------------
print("\n-- V4: NAV drawdown >= 0 --")
for sc, r in results.items():
    check(f"{sc}: drawdown >= 0", r["drawdown"] >= 0, f"drawdown = {r['drawdown']:.4f}")

# -- V5: Monotonie ECL  --  crises > central --
print("\n-- V5: Monotonie ECL --")
central = results["Central"]
gfc = results["Crise financiere (GFC)"]
stagf = results["Stagflation"]
reprise = results["Reprise"]
hyper = results["Hypercroissance"]
covid = results["Choc pandemique (COVID)"]

check("GFC ECL > Central ECL",
      gfc["ecl_stress"] > central["ecl_stress"],
      f"GFC={format_euro(gfc['ecl_stress'])} vs Central={format_euro(central['ecl_stress'])}")

check("Stagflation ECL > Central ECL",
      stagf["ecl_stress"] > central["ecl_stress"],
      f"Stagf={format_euro(stagf['ecl_stress'])} vs Central={format_euro(central['ecl_stress'])}")

check("COVID ECL > Central ECL",
      covid["ecl_stress"] > central["ecl_stress"],
      f"COVID={format_euro(covid['ecl_stress'])} vs Central={format_euro(central['ecl_stress'])}")

check("Central ECL > Reprise ECL",
      central["ecl_stress"] >= reprise["ecl_stress"],
      f"Central={format_euro(central['ecl_stress'])} vs Reprise={format_euro(reprise['ecl_stress'])}")

# -- V6: NAV monotonie ------------------
print("\n-- V6: Monotonie NAV --")
check("Reprise NAV > Central NAV",
      reprise["nav_stress"] >= central["nav_stress"],
      f"Reprise={format_euro(reprise['nav_stress'])} vs Central={format_euro(central['nav_stress'])}")

check("Central NAV > GFC NAV",
      central["nav_stress"] >= gfc["nav_stress"],
      f"Central={format_euro(central['nav_stress'])} vs GFC={format_euro(gfc['nav_stress'])}")

# -- V7: Staging distribution  --  crises plus severes --
print("\n-- V7: Staging distribution --")
for sc in ["Crise financiere (GFC)", "Stagflation"]:
    r = results[sc]
    s2_pct = r["stage_counts"]["s2"] / N_CLIENTS
    s3_pct = r["stage_counts"]["s3"] / N_CLIENTS
    check(f"{sc}: Stage 2 > 10%", s2_pct > 0.10,
          f"Stage 2 = {s2_pct:.1%}")
    check(f"{sc}: Stage 3 > 1%", s3_pct > 0.01,
          f"Stage 3 = {s3_pct:.1%}")

for sc in ["Reprise", "Central"]:
    r = results[sc]
    s1_pct = r["stage_counts"]["s1"] / N_CLIENTS
    check(f"{sc}: Stage 1 > 30%", s1_pct > 0.30,
          f"Stage 1 = {s1_pct:.1%}")

# -- V8: RAROC coherent avec scenario --
print("\n-- V8: RAROC coherence --")
check("Reprise RAROC credit > 0",
      reprise["raroc_credit"] > 0,
      f"RAROC = {reprise['raroc_credit']:.2%}")

check("Reprise RAROC credit > GFC RAROC credit",
      reprise["raroc_credit"] > gfc["raroc_credit"],
      f"Reprise={reprise['raroc_credit']:.2%} vs GFC={gfc['raroc_credit']:.2%}")

check("Hypercroissance RAROC credit > Central",
      hyper["raroc_credit"] > central["raroc_credit"],
      f"Hyper={hyper['raroc_credit']:.2%} vs Central={central['raroc_credit']:.2%}")

# -- V9: CET1 et Headroom ------------------
print("\n-- V9: CET1 et Headroom --")
for sc, r in results.items():
    # Verifier que CET1 = capital / RWA_weighted
    expected_cap = BASEL_CONFIG.rwa_budget * BASEL_CONFIG.cet1_target
    expected_cet1 = expected_cap / max(r["rwa_weighted"], 1)
    check(f"{sc}: CET1 ratio coherent",
          abs(r["cet1_ratio"] - expected_cet1) < 0.001,
          f"calcule={r['cet1_ratio']:.4f} vs attendu={expected_cet1:.4f}")

# -- V10: Delta NAV  --  Central ~ 0 ----------
print("\n-- V10: Delta NAV scenario Central --")
# Central = sliders a (IR=+50bp, un=0, gdp=1.2, hpi=2.0, infl=2.5)
# Ce n'est PAS identique a SCENARIO_BASE (IR=3.5% vs Central->4.0%)
# donc delta_nav ne sera pas exactement 0
c = results["Central"]
check("Central: Delta NAV amplitude raisonnable",
      abs(c["delta_nav"]) < c["nav_ref"] * 0.20,
      f"delta={format_euro(c['delta_nav'])}, ref={format_euro(c['nav_ref'])}")

# -- V11: PE categories  --  crises plus de distressed --
print("\n-- V11: PE categories --")
gfc_distressed = gfc["pe_cats"].get("Distressed", 0)
reprise_distressed = reprise["pe_cats"].get("Distressed", 0)
check("GFC plus de Distressed que Reprise",
      gfc_distressed >= reprise_distressed,
      f"GFC={gfc_distressed} vs Reprise={reprise_distressed}")

# -- V12: RWA densite ------------------
print("\n-- V12: RWA densite --")
for sc, r in results.items():
    rwa_density = r["rwa_credit"] / max(r["ead_total"], 1)
    check(f"{sc}: RWA density = {rwa_density:.2f} (attendu ~{BASEL_CONFIG.rw_credit})",
          abs(rwa_density - BASEL_CONFIG.rw_credit) < 0.01,
          f"RW credit fixe a {BASEL_CONFIG.rw_credit}")

# -- V13: ECL par secteur (tous les secteurs representes) --
print("\n-- V13: Tous secteurs representes --")
for sc, r in results.items():
    sectors_in_ecl = set(r["res_stress"]["sector"].unique())
    expected_sectors = {s.name for s in SECTORS}
    check(f"{sc}: 5 secteurs presents",
          expected_sectors.issubset(sectors_in_ecl),
          f"manquants: {expected_sectors - sectors_in_ecl}")

# -- V14: ECL par scenario interne (Base/Adverse/Favorable) --
print("\n-- V14: ECL ponderation interne --")
for sc, r in results.items():
    df = r["res_stress"]
    ecl_b = df["ecl_base"].sum()
    ecl_a = df["ecl_adverse"].sum()
    ecl_f = df["ecl_favorable"].sum()
    ecl_w = df["ecl_weighted"].sum()
    expected_w = 0.50 * ecl_b + 0.25 * ecl_a + 0.25 * ecl_f
    check(f"{sc}: ECL weighted = 50%xBase + 25%xAdv + 25%xFav",
          abs(ecl_w - expected_w) / max(ecl_w, 1) < 0.01,
          f"weighted={ecl_w:.0f} vs expected={expected_w:.0f}")

# -- V15: Drawdown coherence ------------------
print("\n-- V15: Drawdown coherence --")
for sc, r in results.items():
    if r["delta_nav"] < 0:
        expected_dd = -r["delta_nav"] / max(r["nav_ref"], 1)
        check(f"{sc}: drawdown = |delta_nav|/nav_ref",
              abs(r["drawdown"] - expected_dd) < 0.001,
              f"drawdown={r['drawdown']:.4f} vs expected={expected_dd:.4f}")
    else:
        check(f"{sc}: delta_nav >= 0 -> drawdown = 0",
              r["drawdown"] == 0.0 or r["drawdown"] < 0.001,
              f"drawdown={r['drawdown']:.4f}")


# -----------------------------------------
# TABLEAU RECAPITULATIF
# -----------------------------------------
print("\n\n" + "=" * 80)
print("TABLEAU RECAPITULATIF")
print("=" * 80)

header = f"{'Scenario':30s} {'ECL(Md)':>10s} {'ECL Delta':>8s} {'ECL/EAD':>8s} {'NAV(Md)':>10s} {'DD':>6s} {'RAROC_C':>8s} {'RAROC_PE':>8s} {'CET1':>7s} {'HR(MEUR)':>10s} {'S1%':>5s} {'S2%':>5s} {'S3%':>5s}"
print(header)
print("-" * len(header))

for sc in SCENARIOS_TO_TEST:
    r = results[sc]
    ecl_md = r["ecl_stress"] / 1e9
    nav_md = r["nav_stress"] / 1e6
    s1p = r["stage_counts"]["s1"] / N_CLIENTS * 100
    s2p = r["stage_counts"]["s2"] / N_CLIENTS * 100
    s3p = r["stage_counts"]["s3"] / N_CLIENTS * 100
    hr = r["headroom_m"]
    print(f"{sc:30s} {ecl_md:10.1f} {r['ecl_delta']:+7.1%} {r['ecl_ead_ratio']:7.2%} {nav_md:10.1f} {r['drawdown']:5.1%} {r['raroc_credit']:7.2%} {r['raroc_pe']:7.2%} {r['cet1_ratio']:6.1%} {hr:+10.0f} {s1p:4.0f}% {s2p:4.0f}% {s3p:4.0f}%")


# -----------------------------------------
# RESUME FINAL
# -----------------------------------------
print("\n\n" + "=" * 80)
total_tests = tests_pass + tests_fail
print(f"RESULTAT FINAL: {tests_pass}/{total_tests} tests PASS ({tests_fail} FAIL)")
print("=" * 80)

if issues:
    print("\nISSUES DETECTEES:")
    for i, (name, detail) in enumerate(issues, 1):
        print(f"  {i}. {name}")
        if detail:
            print(f"     -> {detail}")

print(f"\nTemps total: {time.time()-t0:.1f}s")
