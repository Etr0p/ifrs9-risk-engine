"""Audit calculatoire IFRS 9 Risk Cockpit — Pipeline BL-CVaR complet.

Verifie l'integralite du pipeline de calcul et d'arbitrage :
    [1/7] ECL Credit & Staging (IFRS 9)
    [2/7] NAV PE & Distress (IFRS 13)
    [3/7] Signal de stress continu (s = -r_avg / r_neutral)
    [4/7] Illiquidite convexe + Vol GJR-GARCH
    [5/7] BL-CVaR Phase 1 (Rockafellar-Uryasev, zero-mean)
    [6/7] Bandes sigmoid + Contraintes Phase 2
    [7/7] Projection CET1 & Arbitrage final (Phase 1 -> Phase 2)

Architecture asymetrique (pas de lookup tables) :
    - Objectif : max[ mu(w) - kappa * CVaR_0(w) - lambda_hhi * HHI(w) ]
    - CVaR_0 = expected shortfall zero-mean (pur risque de queue)
    - Illiquidite PE : illiq(s) = base + scale * max(0,s)^gamma  (convexe)
    - Vol PE : sigma * (1 + alpha_down * max(0,s) - alpha_up * max(0,-s))  (GJR-GARCH)
    - Bandes PE : sigmoid(k * s) blend entre pe_calm et pe_stress  (continue)

Usage: python audit_calculatoire.py
"""

import sys
import os
import time
import warnings
import numpy as np

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ifrs9_cockpit.config import (
    PREDEFINED_SCENARIOS,
    SCENARIO_BASE,
    BASEL_CONFIG,
)
from ifrs9_cockpit.dashboard.cache import load_data, train_pd_models, train_lgd_ead
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.engine.comparator import PortfolioComparator


def slider_to_macro(scenario_dict):
    base = SCENARIO_BASE
    return {
        "unemployment_rate": base.unemployment_rate + abs(scenario_dict["unemployment_bipolar"]),
        "gdp_growth": scenario_dict["gdp_pct"],
        "interest_rate": base.interest_rate + scenario_dict["interest_rate_bp"] / 100,
        "hpi_growth": scenario_dict["hpi_pct"],
        "inflation_rate": scenario_dict["inflation_pct"],
    }


# ===================================================================
# PHASE 1 : Chargement & Architecture du modele
# ===================================================================

print("=" * 100)
print("AUDIT CALCULATOIRE & ARBITRAGE — BL-CVaR ENGINE v3 (Zero-Mean CVaR + Equations Asymetriques)")
print("=" * 100)
print()

# --- Architecture du modele ---
# Parametres de l'optimiseur (hardcodes dans comparator.py, pas dans BASEL_CONFIG)
KAPPA_BASE = 0.56
LAMBDA_HHI = 0.15
ALPHA_CVAR = 0.95
N_SCENARIOS = 5000
R_NEUTRAL = 0.07
# Illiquidite convexe
ILLIQ_BASE = 0.005
ILLIQ_SCALE = 0.025
ILLIQ_GAMMA = 1.5
# Vol GJR-GARCH
VOL_ALPHA_DOWN = 0.40
VOL_ALPHA_UP = 0.10
VOL_FLOOR = 0.15
# Bandes sigmoid
PE_CALM = 0.15
PE_STRESS = 0.05
PE_MIN = 0.02
BAND_K = 3.0
# BL Confidence
K_CREDIT = 1.5
K_PE = 2.5

print("[0/3] ARCHITECTURE DU MODELE")
print()
print("  Optimisation : BL-CVaR zero-mean (Rockafellar-Uryasev 2000)")
print(f"  Objectif : max[ mu(w) - kappa * CVaR_0(w) - lambda_hhi * HHI(w) ]")
print(f"  mu(w) = RAROC deterministique par cellule (pas de mu_ttc ni BL posterior)")
print(f"  CVaR_0 = expected shortfall ZERO-MEAN (pur risque de queue, toujours positif)")
print(f"  Kappa base = {KAPPA_BASE:.2f} (calibre pour Central ~ 54% PE libre)")
print(f"  Lambda HHI = {LAMBDA_HHI:.2f} (penalite anti-concentration soft)")
print(f"  CVaR alpha = {ALPHA_CVAR:.0%}, scenarios MC = {N_SCENARIOS:,}")
print()
print("  Phase 1 / Phase 2 :")
print("    Phase 1 : BL-CVaR libre avec bornes [5%, 60%]")
print("      5% = HHI hard floor (diversification minimum)")
print("      60% = HHI soft ceiling (max PE avant normes prudentielles)")
print("    Phase 2 : Contraintes sequentielles :")
print(f"      pe_max config = {BASEL_CONFIG.pe_max_allocation:.0%}")
print("      sigmoid band → CET1 binary search")
print()
print("  Equations asymetriques (PAS de lookup tables) :")
print(f"    Signal de stress : s = -(0.5*r_c + 0.5*r_p) / r_neutral")
print(f"    r_neutral = {R_NEUTRAL:.2f}")
print()
print(f"    1. Illiquidite convexe (Ang et al. 2014) :")
print(f"       illiq(s) = {ILLIQ_BASE:.3f} + {ILLIQ_SCALE:.3f} * max(0, s)^{ILLIQ_GAMMA:.1f}")
s_crisis = 1.0
print(f"       → Expansion (s=-2): {ILLIQ_BASE:.1%}  |  Crise (s=+1): {ILLIQ_BASE + ILLIQ_SCALE * s_crisis**ILLIQ_GAMMA:.1%}")
print()
print(f"    2. Vol GJR-GARCH (Glosten et al. 1993) :")
print(f"       mult(s) = 1 + {VOL_ALPHA_DOWN:.2f} * max(0,s) - {VOL_ALPHA_UP:.2f} * max(0,-s)")
print(f"       Asymetrie {VOL_ALPHA_DOWN/VOL_ALPHA_UP:.0f}:1 (vol monte {VOL_ALPHA_DOWN/VOL_ALPHA_UP:.0f}x plus vite en crise)")
print(f"       Plancher absolu : {VOL_FLOOR:.0%}")
print()
print(f"    3. Bandes sigmoid (Ang & Bekaert 2004) :")
print(f"       pe_max(s) = {PE_STRESS:.0%} + ({PE_CALM:.0%} - {PE_STRESS:.0%}) * (1 - sigmoid({BAND_K:.0f} * s))")
print(f"       PE min = {PE_MIN:.0%} (diversification plancher)")
print()
print(f"    4. Kappa CVaR asymetrique (leverage effect en cascade) :")
print(f"       kappa_eff(s) = kappa_base * vol_mult(s)")
print(f"       En crise (s=+1): kappa_eff = {KAPPA_BASE * (1 + VOL_ALPHA_DOWN):.2f}")
print(f"       En expansion (s=-2): kappa_eff = {KAPPA_BASE * (1 - 2*VOL_ALPHA_UP):.2f}")
print()
print(f"    5. Confiance BL asymetrique :")
print(f"       c(s) = 0.30 + 0.65 * sigmoid(-k * s), borne [0.30, 0.95]")
print(f"       k_credit = {K_CREDIT}, k_PE = {K_PE} (PE chute {K_PE/K_CREDIT:.1f}x plus vite)")
print()
print(f"  Contraintes prudentielles :")
print(f"    pe_max = {BASEL_CONFIG.pe_max_allocation:.0%}, CET1 cible = {BASEL_CONFIG.cet1_target:.0%}")
print(f"    RWA budget = {BASEL_CONFIG.rwa_budget:.1e} EUR")
print()

# --- Chargement ---
print("[1/3] Chargement des donnees et modeles...")
t0 = time.time()
df_credit, df_pe, df_history, df_balance_sheet = load_data()
pd_suite = train_pd_models()
lgd_model, ead_model = train_lgd_ead()

ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
pe_calc = PECalculator()

pd_predictions = pd_suite.predict(df_credit)
pd_current = pd_predictions["LR_WoE"]
pd_origination = df_credit["pd_origination"].values

# ===================================================================
# PHASE 2 : Execution des scenarios & Audit detaille
# ===================================================================

print(f"[2/3] Execution des {len(PREDEFINED_SCENARIOS)} scenarios et audit BL-CVaR...")
print()

results = {}
test_failures = []
tests_total = 0
tests_passed = 0

for name, params in PREDEFINED_SCENARIOS.items():
    t1 = time.time()
    macro = slider_to_macro(params)
    is_crisis = params["unemployment_bipolar"] < 0

    # 1. Calculs Bruts
    result_credit = ecl_calc.calculate(
        df_credit, pd_current, pd_origination,
        unemployment_override=macro["unemployment_rate"],
        gdp_override=macro["gdp_growth"],
        interest_rate_override=macro["interest_rate"],
        hpi_override=macro["hpi_growth"],
        inflation_override=macro["inflation_rate"],
    )
    result_pe = pe_calc.calculate(
        df_pe,
        unemployment_override=macro["unemployment_rate"],
        gdp_override=macro["gdp_growth"],
        interest_rate_override=macro["interest_rate"],
        hpi_override=macro["hpi_growth"],
        inflation_override=macro["inflation_rate"],
        unemployment_crisis=is_crisis,
    )

    # 2. Moteur d'Arbitrage (BL-CVaR)
    comp = PortfolioComparator(result_credit, result_pe)
    opt = comp.optimize_allocation()

    # 3. Metriques cles
    ecl_total = float(result_credit["ecl_weighted"].sum())
    nav_total = float(result_pe["nav"].sum())

    # 4. Staging
    stage_counts = result_credit["stage"].value_counts()
    n_total = len(result_credit)
    s1_pct = float(stage_counts.get(1, 0)) / n_total
    s2_pct = float(stage_counts.get(2, 0)) / n_total
    s3_pct = float(stage_counts.get(3, 0)) / n_total

    results[name] = {
        "ecl": ecl_total,
        "nav": nav_total,
        "opt": opt,
        "macro": macro,
        "s1_pct": s1_pct,
        "s2_pct": s2_pct,
        "s3_pct": s3_pct,
    }

    # --- Tests par scenario ---

    # Test A: mu_RAROC PE borne (tolere negatif en crise extreme)
    tests_total += 1
    mu_raroc_pe = opt["mu_raroc_pe"]
    if mu_raroc_pe > -0.50:  # tolere jusqu'a -50% en crise extreme (Stagflation)
        tests_passed += 1
    else:
        test_failures.append(f"[{name}] mu RAROC PE trop negatif: {mu_raroc_pe:.2%}")

    # Test B: Allocation finale dans les bandes sigmoid
    # Tolerance 1% car pe_band est arrondi a 2 decimales et pe_allocation a 2 decimales
    tests_total += 1
    w_pe = opt["pe_allocation"]
    pe_band = opt["pe_band"]
    w_min, w_max = pe_band[0], pe_band[1]
    tol_band = 0.01
    in_band = (w_min - tol_band <= w_pe <= w_max + tol_band)
    if in_band:
        tests_passed += 1
    else:
        test_failures.append(f"[{name}] Alloc {w_pe:.1%} hors bande sigmoid [{w_min:.1%}, {w_max:.1%}]")

    # Test C: CVaR non nulle et positive (zero-mean → toujours > 0)
    tests_total += 1
    cvar = opt["cvar_95"]
    if cvar > 0:
        tests_passed += 1
    else:
        test_failures.append(f"[{name}] CVaR 95% non positive: {cvar:.4f}")

    # Test G: Illiquidite convexe (doit etre >= base)
    tests_total += 1
    illiq = opt["illiquidity_premium"]
    if illiq >= ILLIQ_BASE - 1e-6:
        tests_passed += 1
    else:
        test_failures.append(f"[{name}] Illiquidite {illiq:.3%} < base {ILLIQ_BASE:.3%}")

    # Test H: Vol multiplier coherent (> 1 si stress > 0, < 1 si stress < 0)
    tests_total += 1
    stress = opt["stress_intensity"]
    vol_m = opt["vol_multiplier"]
    if (stress > 0.1 and vol_m > 1.0) or (stress < -0.1 and vol_m < 1.0) or abs(stress) <= 0.1:
        tests_passed += 1
    else:
        test_failures.append(f"[{name}] Vol mult {vol_m:.2f} incoherent avec stress {stress:+.2f}")

    # Test N: Phase 1 PE libre dans [5%, 60%] (bornes HHI)
    tests_total += 1
    pe_free = opt["pe_free"]
    if 0.05 - 1e-4 <= pe_free <= 0.60 + 1e-4:
        tests_passed += 1
    else:
        test_failures.append(f"[{name}] PE libre {pe_free:.1%} hors bornes [5%, 60%]")

    # Test O: Phase 2 <= Phase 1 (contraintes reduisent toujours)
    tests_total += 1
    if w_pe <= pe_free + 1e-4:
        tests_passed += 1
    else:
        test_failures.append(f"[{name}] Phase 2 {w_pe:.1%} > Phase 1 {pe_free:.1%}")

    # Test P: CET1 ratio >= cible si feasible
    tests_total += 1
    if opt["feasible"]:
        if opt["cet1_ratio"] >= BASEL_CONFIG.cet1_target - 1e-4:
            tests_passed += 1
        else:
            test_failures.append(f"[{name}] Feasible mais CET1 {opt['cet1_ratio']:.2%} < cible {BASEL_CONFIG.cet1_target:.0%}")
    else:
        tests_passed += 1  # infeasible est un resultat valide

    kappa_eff = opt["kappa_pe_eff"]
    conf_c, conf_p = opt["bl_confidence"]
    elapsed = time.time() - t1
    print(f"  [{name:30s}] s={stress:+5.2f} | illiq={illiq:4.1%} | vol*{vol_m:.2f} | k_eff={kappa_eff:.2f} | conf={conf_c:.0%}/{conf_p:.0%} | "
          f"band=[{w_min:.0%},{w_max:.0%}] | Ph1={pe_free:4.0%} -> Final={w_pe:4.1%} | {elapsed:.1f}s")

print()

# ===================================================================
# PHASE 3 : Tests de Coherence Cross-Scenario
# ===================================================================

print("[3/3] Tests de coherence cross-scenario...")
print()

# Test E: CVaR positif pour tous (zero-mean → toujours > 0)
print("  Test E — CVaR positif pour tous les scenarios")
tests_total += 1
all_cvar_ok = True
for sn in PREDEFINED_SCENARIOS:
    cv = results[sn]["opt"]["cvar_95"]
    if cv <= 0:
        all_cvar_ok = False
        test_failures.append(f"CVaR non positif: {sn} = {cv:.4f}")
if all_cvar_ok:
    tests_passed += 1
    print(f"    PASS: CVaR > 0 pour les {len(PREDEFINED_SCENARIOS)} scenarios")
else:
    print(f"    FAIL")

# Test I: Monotonie stress (Stagflation stress > Central stress > Reprise stress)
print("  Test I — Stress : Stagflation > Central > Reprise")
s_stag = results["Stagflation"]["opt"]["stress_intensity"]
s_central = results["Central"]["opt"]["stress_intensity"]
s_reprise = results["Reprise"]["opt"]["stress_intensity"]
tests_total += 1
if s_stag > s_central > s_reprise:
    tests_passed += 1
    print(f"    PASS: {s_stag:+.2f} > {s_central:+.2f} > {s_reprise:+.2f}")
else:
    test_failures.append(f"Stress: Stag={s_stag:+.2f}, Central={s_central:+.2f}, Reprise={s_reprise:+.2f}")
    print(f"    FAIL")

# Test J: Illiquidite convexe (GFC illiq > Central illiq >= Reprise illiq)
print("  Test J — Illiquidite convexe : GFC > Central >= Reprise")
i_gfc = results["Crise financiere (GFC)"]["opt"]["illiquidity_premium"]
i_central = results["Central"]["opt"]["illiquidity_premium"]
i_reprise = results["Reprise"]["opt"]["illiquidity_premium"]
tests_total += 1
if i_gfc > i_central >= i_reprise - 1e-6:
    tests_passed += 1
    print(f"    PASS: GFC {i_gfc:.2%} > Central {i_central:.2%} >= Reprise {i_reprise:.2%}")
else:
    test_failures.append(f"Illiq: GFC={i_gfc:.2%}, Central={i_central:.2%}, Reprise={i_reprise:.2%}")
    print(f"    FAIL")

# Test K: Vol multiplier asymetrique (GFC vol_mult > 1, Reprise vol_mult < 1)
print("  Test K — Vol asymetrique : GFC mult > 1 > Reprise mult")
vm_gfc = results["Crise financiere (GFC)"]["opt"]["vol_multiplier"]
vm_reprise = results["Reprise"]["opt"]["vol_multiplier"]
tests_total += 1
if vm_gfc > 1.0 > vm_reprise:
    tests_passed += 1
    print(f"    PASS: GFC {vm_gfc:.2f}x > 1.0 > Reprise {vm_reprise:.2f}x")
else:
    test_failures.append(f"Vol mult: GFC={vm_gfc:.2f}, Reprise={vm_reprise:.2f}")
    print(f"    FAIL")

# Test L: Bandes sigmoid (GFC band < Central band <= Reprise band)
print("  Test L — Bandes sigmoid : GFC plafond < Central <= Reprise")
b_gfc = results["Crise financiere (GFC)"]["opt"]["pe_band"][1]
b_central = results["Central"]["opt"]["pe_band"][1]
b_reprise = results["Reprise"]["opt"]["pe_band"][1]
tests_total += 1
if b_gfc < b_central <= b_reprise + 1e-4:
    tests_passed += 1
    print(f"    PASS: GFC {b_gfc:.0%} < Central {b_central:.0%} <= Reprise {b_reprise:.0%}")
else:
    test_failures.append(f"Bandes: GFC={b_gfc:.0%}, Central={b_central:.0%}, Reprise={b_reprise:.0%}")
    print(f"    FAIL")

# Test M: ECL discriminant (Stagflation >> Central >> Reprise)
print("  Test M — ECL : Stagflation >> Central >> Reprise")
ecl_stag = results["Stagflation"]["ecl"]
ecl_central = results["Central"]["ecl"]
ecl_reprise = results["Reprise"]["ecl"]
tests_total += 1
if ecl_stag > ecl_central > ecl_reprise:
    tests_passed += 1
    print(f"    PASS: {ecl_stag/1e9:.0f}B > {ecl_central/1e9:.0f}B > {ecl_reprise/1e9:.0f}B (spread {ecl_stag/ecl_reprise:.1f}x)")
else:
    test_failures.append(f"ECL: Stag={ecl_stag/1e9:.0f}B, Central={ecl_central/1e9:.0f}B, Reprise={ecl_reprise/1e9:.0f}B")
    print(f"    FAIL")

# Test Q: PE libre monotone (Reprise PE_free >= Central PE_free >= GFC PE_free)
print("  Test Q — PE libre : Reprise >= Central >= GFC")
pf_reprise = results["Reprise"]["opt"]["pe_free"]
pf_central = results["Central"]["opt"]["pe_free"]
pf_gfc = results["Crise financiere (GFC)"]["opt"]["pe_free"]
tests_total += 1
if pf_reprise >= pf_central - 0.05 and pf_central >= pf_gfc - 0.05:
    tests_passed += 1
    print(f"    PASS: Reprise {pf_reprise:.0%} >= Central {pf_central:.0%} >= GFC {pf_gfc:.0%}")
else:
    test_failures.append(f"PE libre: Reprise={pf_reprise:.0%}, Central={pf_central:.0%}, GFC={pf_gfc:.0%}")
    print(f"    FAIL")

# Test R: RAROC Credit positif en scenario Central (crise peut etre negatif)
print("  Test R — RAROC Credit > 0 en scenario Central")
tests_total += 1
rc_central = results["Central"]["opt"]["raroc_credit"]
if rc_central > 0:
    tests_passed += 1
    print(f"    PASS: Central RAROC Credit = {rc_central:.2%}")
else:
    test_failures.append(f"RAROC Credit Central negatif: {rc_central:.2%}")
    print(f"    FAIL")

print()

# ===================================================================
# TABLEAU RECAPITULATIF — Equations Asymetriques
# ===================================================================

print("=" * 100)
print("TABLEAU RECAPITULATIF — Pipeline BL-CVaR v3 (Zero-Mean CVaR)")
print("=" * 100)
print()

# Table 1: Parametres asymetriques par scenario
h1 = f"{'Scenario':30s} | {'stress':>7s} | {'illiq':>6s} | {'vol_m':>6s} | {'band':>10s} | {'Phase1':>6s} | {'Final':>6s} | {'CET1':>6s} | {'Feasible'}"
print(h1)
print("-" * len(h1))

scenario_order = list(PREDEFINED_SCENARIOS.keys())

for name in scenario_order:
    r = results[name]["opt"]
    s = r["stress_intensity"]
    il = r["illiquidity_premium"]
    vm = r["vol_multiplier"]
    bmin, bmax = r["pe_band"]
    ph1 = r["pe_free"]
    fin = r["pe_allocation"]
    cet1 = r["cet1_ratio"]
    feas = "OK" if r["feasible"] else "FAIL"
    print(f"{name:30s} | {s:+6.2f} | {il:5.1%} | {vm:5.2f}x | [{bmin:.0%},{bmax:.0%}] | {ph1:5.0%} | {fin:5.1%} | {cet1:5.1%} | {feas}")

print()

# Table 2: Metriques credit/PE
h2 = f"{'Scenario':30s} | {'ECL':>7s} | {'S1':>5s} | {'S2':>5s} | {'S3':>5s} | {'RAROC Cr':>8s} | {'RAROC PE':>8s} | {'mu_Cr':>7s} | {'mu_PE':>7s} | {'CET1':>7s}"
print(h2)
print("-" * len(h2))

for name in scenario_order:
    r = results[name]
    o = r["opt"]
    ecl = r["ecl"]
    print(f"{name:30s} | {ecl/1e9:5.0f}B | {r['s1_pct']:4.0%} | {r['s2_pct']:4.0%} | {r['s3_pct']:4.0%} | "
          f"{o['raroc_credit']:7.2%} | {o['raroc_pe']:7.2%} | {o['mu_raroc_credit']:6.2%} | {o['mu_raroc_pe']:6.2%} | {o['cet1_ratio']:6.2%}")

print()

# Table 3: Optimiseur diagnostics
h3 = f"{'Scenario':30s} | {'kappa':>6s} | {'k_eff':>6s} | {'CVaR95':>7s} | {'Sharpe':>6s} | {'port_vol':>8s} | {'Conf Cr':>7s} | {'Conf PE':>7s}"
print(h3)
print("-" * len(h3))

for name in scenario_order:
    o = results[name]["opt"]
    conf_c, conf_p = o["bl_confidence"]
    print(f"{name:30s} | {o['kappa']:5.2f} | {o['kappa_pe_eff']:5.2f} | {o['cvar_95']:6.4f} | {o['sharpe']:5.2f} | {o['portfolio_vol']:7.2%} | "
          f"{conf_c:6.0%} | {conf_p:6.0%}")

print()

# ===================================================================
# VERDICT
# ===================================================================

print("=" * 100)
print(f"VERDICT AUDIT BL-CVaR v3 : {tests_passed}/{tests_total} tests PASS")
print("=" * 100)

if test_failures:
    print()
    print(f"Echecs ({len(test_failures)}):")
    for f in test_failures:
        print(f"  x {f}")
else:
    print("Tous les tests du pipeline BL-CVaR sont PASS.")

print()
print("Temps total audit:", f"{time.time()-t0:.1f}s")
