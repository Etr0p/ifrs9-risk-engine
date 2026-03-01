#!/usr/bin/env python3
"""
================================================================
AUDIT DES SCENARIOS PREDEFINED — REGARD D'ECONOMISTE / CRO
================================================================

Analyse critique des 8 scenarios predefinis :
  1. Pertinence historique (ancrage factuel)
  2. Equilibre positif/negatif (biais adverse?)
  3. Redondance entre scenarios (distance Mahalanobis, correlation)
  4. Couverture de l'espace macro (PCA, enveloppe convexe)
  5. Coherence interne de chaque scenario
  6. Impact differentiel sur le portefeuille (ECL, NAV PE, RAROC)
  7. Discussion : scenario base = "bonne sante" ?
"""

import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import combinations

from ifrs9_cockpit.config import (
    PREDEFINED_SCENARIOS, SCENARIO_BASE, SCENARIOS, SECTORS,
    MACRO_COVARIANCE, MACRO_VARIABLES_ORDER,
)
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.models.pe_model import PEModel
from ifrs9_cockpit.config import RANDOM_SEED

print("=" * 80)
print("  AUDIT DES SCENARIOS PREDEFINED")
print("  9 scenarios historiquement calibres — regard d'economiste CRO")
print("=" * 80)

# ────────────────────────────────────────────
# 1. INVENTAIRE & CLASSIFICATION
# ────────────────────────────────────────────
print("\n" + "-" * 80)
print("1. INVENTAIRE & CLASSIFICATION DES 9 SCENARIOS")
print("-" * 80)

base = SCENARIO_BASE
slider_keys = ["interest_rate_bp", "unemployment_bipolar", "gdp_pct", "hpi_pct", "inflation_pct"]

def slider_to_macro(s: dict) -> dict:
    """Convertit les valeurs slider en niveaux macro absolus."""
    ir_abs = base.interest_rate + s["interest_rate_bp"] / 100
    unemp_abs = base.unemployment_rate + abs(s["unemployment_bipolar"])
    gdp_abs = s["gdp_pct"]
    hpi_abs = s["hpi_pct"]
    infl_abs = s["inflation_pct"]
    return {
        "interest_rate": ir_abs,
        "unemployment_rate": unemp_abs,
        "gdp_growth": gdp_abs,
        "hpi_growth": hpi_abs,
        "inflation_rate": infl_abs,
    }

scenarios_macro = {}
for name, sliders in PREDEFINED_SCENARIOS.items():
    scenarios_macro[name] = slider_to_macro(sliders)

def classify_scenario(macro: dict) -> str:
    """Classifie le scenario en positif/negatif/neutre."""
    score = 0
    if macro["gdp_growth"] > 2.0: score += 1
    elif macro["gdp_growth"] < 0: score -= 1
    if macro["unemployment_rate"] < 7.5: score += 1
    elif macro["unemployment_rate"] > 9.0: score -= 1
    if macro["inflation_rate"] < 3.0 and macro["inflation_rate"] > 0.5: score += 1
    elif macro["inflation_rate"] > 5.0 or macro["inflation_rate"] < 0: score -= 1
    if macro["hpi_growth"] > 3.0: score += 1
    elif macro["hpi_growth"] < 0: score -= 1
    if score >= 2: return "POSITIF"
    elif score <= -2: return "NEGATIF"
    else: return "NEUTRE"

print(f"\n  {'Scenario':<28s} {'PIB%':>6s} {'Chom%':>6s} {'Taux%':>6s} {'HPI%':>6s} {'Infl%':>6s}  {'Nature':>8s}")
print("  " + "-" * 76)

n_pos = 0
n_neg = 0
n_neu = 0
for name, macro in scenarios_macro.items():
    nature = classify_scenario(macro)
    if nature == "POSITIF": n_pos += 1
    elif nature == "NEGATIF": n_neg += 1
    else: n_neu += 1
    print(f"  {name:<28s} {macro['gdp_growth']:>6.1f} {macro['unemployment_rate']:>6.1f}"
          f" {macro['interest_rate']:>6.1f} {macro['hpi_growth']:>6.1f}"
          f" {macro['inflation_rate']:>6.1f}  {nature:>8s}")

print(f"\n  Balance : {n_pos} positifs, {n_neg} negatifs, {n_neu} neutres")
ratio = n_neg / max(n_pos, 1)
print(f"  Ratio negatif/positif : {ratio:.1f}x")

if n_neg > n_pos + 2:
    print(f"  >> BIAIS ADVERSE : {n_neg} scenarios de crise vs {n_pos} positifs")

# ────────────────────────────────────────────
# 2. REDONDANCE — DISTANCE MAHALANOBIS
# ────────────────────────────────────────────
print("\n" + "-" * 80)
print("2. REDONDANCE ENTRE SCENARIOS (Distance Mahalanobis)")
print("-" * 80)

var_order = ["unemployment_rate", "gdp_growth", "interest_rate", "hpi_growth", "inflation_rate"]
X = np.array([[scenarios_macro[name][v] for v in var_order] for name in PREDEFINED_SCENARIOS])
names = list(PREDEFINED_SCENARIOS.keys())

Sigma = np.array(MACRO_COVARIANCE)
Sigma_inv = np.linalg.inv(Sigma)

n_sc = len(names)
D_maha = np.zeros((n_sc, n_sc))
for i in range(n_sc):
    for j in range(n_sc):
        diff = X[i] - X[j]
        D_maha[i, j] = np.sqrt(diff @ Sigma_inv @ diff)

print(f"\n  Paires les plus proches (distance Mahalanobis) :")
all_pairs = []
for i, j in combinations(range(n_sc), 2):
    d = D_maha[i, j]
    all_pairs.append((names[i], names[j], d))

all_pairs.sort(key=lambda x: x[2])
for name_i, name_j, d in all_pairs[:8]:
    tag = " << REDONDANT" if d < 2.0 else " < PROCHE" if d < 3.5 else ""
    print(f"    {name_i:<24s} <-> {name_j:<24s} d={d:.2f}{tag}")

redundant = [p for p in all_pairs if p[2] < 2.0]
if redundant:
    print(f"\n  >> {len(redundant)} paire(s) potentiellement redondante(s) (d < 2.0)")
else:
    print(f"\n  >> Aucune paire redondante (d < 2.0) -- scenarios bien differencies")

# ────────────────────────────────────────────
# 3. COUVERTURE DE L'ESPACE MACRO (PCA)
# ────────────────────────────────────────────
print("\n" + "-" * 80)
print("3. COUVERTURE DE L'ESPACE MACRO (PCA)")
print("-" * 80)

sigma_diag = np.sqrt(np.diag(Sigma))
X_norm = (X - X.mean(axis=0)) / sigma_diag

U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)
var_explained = (S ** 2) / np.sum(S ** 2) * 100

print(f"\n  Variance expliquee par composante principale :")
for i, ve in enumerate(var_explained):
    bar = "#" * int(ve / 2)
    print(f"    PC{i+1}: {ve:>5.1f}%  {bar}")
print(f"    PC1+PC2 = {var_explained[0]+var_explained[1]:.1f}%")

# Loadings de PC1 et PC2
print(f"\n  Interpretation des composantes :")
for pc_idx in range(2):
    loading = Vt[pc_idx]
    print(f"    PC{pc_idx+1} loadings:")
    for v, l in zip(var_order, loading):
        bar = "+" * max(0, int(l * 10)) + "-" * max(0, int(-l * 10))
        print(f"      {v:<20s} {l:>+.3f}  {bar}")

proj = X_norm @ Vt.T[:, :2]
print(f"\n  Projection des scenarios sur PC1-PC2 :")
print(f"  {'Scenario':<28s} {'PC1':>8s} {'PC2':>8s}")
for i, name in enumerate(names):
    print(f"  {name:<28s} {proj[i, 0]:>8.2f} {proj[i, 1]:>8.2f}")

q1 = sum(1 for i in range(n_sc) if proj[i, 0] > 0 and proj[i, 1] > 0)
q2 = sum(1 for i in range(n_sc) if proj[i, 0] < 0 and proj[i, 1] > 0)
q3 = sum(1 for i in range(n_sc) if proj[i, 0] < 0 and proj[i, 1] < 0)
q4 = sum(1 for i in range(n_sc) if proj[i, 0] > 0 and proj[i, 1] < 0)
print(f"\n  Couverture des quadrants PC1-PC2 : Q1={q1} Q2={q2} Q3={q3} Q4={q4}")
if min(q1, q2, q3, q4) == 0:
    print(f"    >> Un quadrant est vide -- couverture incomplete de l'espace")
else:
    print(f"    >> Tous les quadrants couverts")

# ────────────────────────────────────────────
# 4. COHERENCE INTERNE DE CHAQUE SCENARIO
# ────────────────────────────────────────────
print("\n" + "-" * 80)
print("4. COHERENCE INTERNE DE CHAQUE SCENARIO")
print("-" * 80)

def check_coherence(name: str, m: dict) -> list:
    issues = []
    if m["gdp_growth"] > 3.0 and m["unemployment_rate"] > 9.0:
        issues.append(f"PIB={m['gdp_growth']:.1f}% avec chomage={m['unemployment_rate']:.1f}% (Okun)")
    if m["inflation_rate"] < 0 and m["interest_rate"] > 4.0:
        issues.append(f"Deflation {m['inflation_rate']:.1f}% avec taux={m['interest_rate']:.1f}% (Taylor)")
    if m["gdp_growth"] > 2.5 and m["hpi_growth"] < -2.0:
        issues.append(f"PIB={m['gdp_growth']:.1f}% avec HPI={m['hpi_growth']:.1f}% (rare)")
    if m["gdp_growth"] < -2.0 and m["inflation_rate"] < 1.0 and m["interest_rate"] > 5.0:
        issues.append(f"Recession + deflation + taux eleves")
    if m["unemployment_rate"] < 6.0 and m["gdp_growth"] < -1.0:
        issues.append(f"Chomage={m['unemployment_rate']:.1f}% avec PIB={m['gdp_growth']:.1f}%")
    return issues

for name, macro in scenarios_macro.items():
    issues = check_coherence(name, macro)
    if issues:
        print(f"  !! {name}: {'; '.join(issues)}")
    else:
        print(f"  OK {name}: coherent")

# ────────────────────────────────────────────
# 5. IMPACT DIFFERENTIEL SUR LE PORTEFEUILLE
# ────────────────────────────────────────────
print("\n" + "-" * 80)
print("5. IMPACT DIFFERENTIEL SUR LE PORTEFEUILLE")
print("-" * 80)

print("\n  Calcul des metriques PE sous chaque scenario predefined...")
df_credit, df_pe, _ = generate_dataset()
pe_calc = PECalculator(pe_model=PEModel(seed=RANDOM_SEED))

result_central = pe_calc.calculate(df_pe)
nav_central = result_central["nav"].sum()
el_central = result_central["expected_loss_pe"].sum()
irr_central = result_central["irr"].mean()
n_dist_central = (result_central["risk_category"] == "Distressed").sum()

results_impact = {}
for name, sliders in PREDEFINED_SCENARIOS.items():
    macro = scenarios_macro[name]
    is_crisis = sliders.get("unemployment_bipolar", 0) < 0
    result = pe_calc.calculate(
        df_pe,
        unemployment_override=macro["unemployment_rate"],
        gdp_override=macro["gdp_growth"],
        interest_rate_override=macro["interest_rate"],
        hpi_override=macro["hpi_growth"],
        inflation_override=macro["inflation_rate"],
        unemployment_crisis=is_crisis,
    )
    nav = result["nav"].sum()
    el = result["expected_loss_pe"].sum()
    irr = result["irr"].mean()
    n_dist = (result["risk_category"] == "Distressed").sum()
    dd = (nav_central - nav) / nav_central * 100
    results_impact[name] = {
        "nav": nav, "el": el, "irr": irr,
        "n_distressed": n_dist, "drawdown_pct": dd,
    }

print(f"\n  {'Scenario':<28s} {'NAV (Md)':>10s} {'DD%':>7s} {'IRR%':>7s} {'EL (Md)':>10s} {'Dist.':>6s}")
print("  " + "-" * 70)
for name, imp in results_impact.items():
    nav_md = imp["nav"] / 1e6
    el_md = imp["el"] / 1e6
    print(f"  {name:<28s} {nav_md:>10,.0f} {imp['drawdown_pct']:>+7.1f} {imp['irr']*100:>7.1f}"
          f" {el_md:>10,.0f} {imp['n_distressed']:>6,}")

sorted_by_impact = sorted(results_impact.items(), key=lambda x: x[1]["drawdown_pct"], reverse=True)
print(f"\n  Classement par drawdown (pire -> meilleur) :")
for i, (name, imp) in enumerate(sorted_by_impact, 1):
    print(f"    {i}. {name:<28s} DD={imp['drawdown_pct']:>+.1f}%")

# Verifier que chaque scenario produit un impact UNIQUE
print(f"\n  Verification de la differentiation :")
dds = [imp["drawdown_pct"] for imp in results_impact.values()]
if len(set(round(d, 1) for d in dds)) == len(dds):
    print(f"    >> Chaque scenario produit un drawdown unique (pas de doublons)")
else:
    print(f"    >> Attention : certains scenarios produisent des drawdowns similaires")

# ────────────────────────────────────────────
# 6. SEVERITE RELATIVE & DISTANCE AU BASE
# ────────────────────────────────────────────
print("\n" + "-" * 80)
print("6. SEVERITE RELATIVE (Distance Mahalanobis au Central)")
print("-" * 80)

base_vec = np.array([base.unemployment_rate, base.gdp_growth, base.interest_rate,
                      base.hpi_growth, base.inflation_rate])

print(f"\n  {'Scenario':<28s} {'Maha.':>7s} {'Severite':>10s}")
for i, name in enumerate(names):
    diff = X[i] - base_vec
    d = np.sqrt(diff @ Sigma_inv @ diff)
    severity = "EXTREME" if d > 5 else "FORT" if d > 3 else "MODERE" if d > 1.5 else "FAIBLE"
    print(f"  {name:<28s} {d:>7.2f} {severity:>10s}")

# ────────────────────────────────────────────
# 7. DISCUSSION : FAUT-IL UN SCENARIO BASE "BONNE SANTE" ?
# ────────────────────────────────────────────
print("\n" + "-" * 80)
print("7. FAUT-IL CHANGER LE SCENARIO DE BASE EN 'BONNE SANTE' ?")
print("-" * 80)

print("""
  SCENARIO DE BASE ACTUEL (Central / BCE 2024) :
    PIB=+1.2%, Chomage=7.5%, Taux=3.5%, HPI=+2%, Inflation=2.5%

  C'est la CONJONCTURE COURANTE zone euro -- ni bon ni mauvais.

  ARGUMENTS POUR un base "bonne sante" :
  -------------------------------------------------------
  1. Mieux voir la degradation relative sous stress
  2. Les stress tests EBA partent d'un baseline favorable
  3. MOIC/classification plus intuitifs (Performing ~90%)

  ARGUMENTS CONTRE (position du CRO) :
  -------------------------------------------------------
  1. IFRS 9 B5.5.17 : le base doit etre l'ESTIMATION CENTRALE,
     pas un ideal. Un base "bonne sante" = ECL sous-estime.
  2. Le base sert de PIVOT pour les sliders. Si le pivot est
     "bonne sante", la conjoncture 2024 serait en zone "stress".
  3. La BCE utilise "projections centrales", pas "best case".
     S'ecarter de cette convention rend le cockpit incomparable.
  4. MACRO_STRUCTURAL_EQUILIBRIUM existe deja pour le long-terme
     (r*=2.5%, NAIRU=6.5%) -- c'est un concept different.

  VERDICT : MAINTENIR le baseline conjoncture courante.
  Mais AJOUTER un scenario predefined "Soft Landing" pour
  combler le manque de scenarios positifs moderes.
""")

# ────────────────────────────────────────────
# 8. PROPOSITION CONCRETE
# ────────────────────────────────────────────
print("-" * 80)
print("8. VERIFICATION : 9EME SCENARIO 'BOOM IMMOBILIER' AJOUTE")
print("-" * 80)

print(f"\n  Le scenario 'Boom immobilier' est maintenant dans PREDEFINED_SCENARIOS.")
print(f"  9 scenarios au total : {len(PREDEFINED_SCENARIOS)}")

# Verification: recalculer le ratio avec le 9eme scenario
all_macro = {}
for name, sliders in PREDEFINED_SCENARIOS.items():
    all_macro[name] = slider_to_macro(sliders)

n_pos_final, n_neg_final, n_neu_final = 0, 0, 0
for name, macro in all_macro.items():
    nature = classify_scenario(macro)
    if nature == "POSITIF": n_pos_final += 1
    elif nature == "NEGATIF": n_neg_final += 1
    else: n_neu_final += 1

print(f"  Balance finale : {n_pos_final} positifs, {n_neg_final} negatifs, {n_neu_final} neutres")
print(f"  Ratio negatif/positif : {n_neg_final}/{n_pos_final} = {n_neg_final/max(n_pos_final,1):.1f}x")

# Redondance check avec 9 scenarios
X_full = np.array([[all_macro[name][v] for v in var_order] for name in PREDEFINED_SCENARIOS])
names_full = list(PREDEFINED_SCENARIOS.keys())
n_full = len(names_full)
print(f"\n  Verification de redondance (9 scenarios) :")
redundant_final = 0
for i, j in combinations(range(n_full), 2):
    diff = X_full[i] - X_full[j]
    d = np.sqrt(diff @ Sigma_inv @ diff)
    if d < 2.0:
        redundant_final += 1
        print(f"    {names_full[i]:<28s} <-> {names_full[j]:<28s} d={d:.2f} << REDONDANT")

if redundant_final == 0:
    print(f"    >> Aucune paire redondante (d >= 2.0 pour toutes les paires)")
else:
    print(f"    >> {redundant_final} paire(s) redondante(s)")

print("\n" + "=" * 80)
print("  FIN DE L'AUDIT DES SCENARIOS")
print("=" * 80)
