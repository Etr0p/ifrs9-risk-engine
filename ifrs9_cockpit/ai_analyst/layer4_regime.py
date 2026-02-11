"""Couche 4 — Classification de regime macro (FR29).

Classifie le regime macro actuel avec probabilites continues sur 5 regimes
canoniques via softmax sur les signatures macro.
"""

from __future__ import annotations

import numpy as np
from typing import Dict

from ifrs9_cockpit.config import SCENARIO_BASE
from ifrs9_cockpit.ai_analyst.types import RegimeClassification


# Signatures des 5 regimes canoniques.
# CONVENTION : les signatures sont dans le meme espace que le stress_vector
# defini dans classify_regime(). Le stress_vector inverse GDP et HPI :
#   stress[gdp] = base.gdp - current.gdp  (baisse GDP -> positif)
#   stress[hpi] = base.hpi - current.hpi  (baisse HPI -> positif)
# Donc une signature avec gdp_growth = +3.0 signifie "baisse du GDP de 3pp".
#
# Signatures calibrees sur les regimes historiques observes :
# - Crise financiere : GFC 2008-09 (unemp +3, gdp baisse 3, rate -1, hpi baisse 5)
# - Stagflation : annees 1970 (unemp +2, gdp baisse 2, rate +2, inflation +3)
# - Resserrement : 2022-23 (rate +3, hpi baisse 3, inflation +1.5)
# - Reprise : 2021 (unemp -2, gdp hausse 2, hpi hausse 2)
# - Rupture techno : hypothetique (unemp +1, gdp baisse 1.5)
# Source : BCE Statistical Data Warehouse, calibration pedagogique.
_REGIME_SIGNATURES: Dict[str, Dict[str, float]] = {
    "Crise financiere": {
        "unemployment_rate": 3.0,
        "gdp_growth": 3.0,        # baisse GDP = positif dans l'espace stress
        "interest_rate": -1.0,
        "hpi_growth": 5.0,        # baisse HPI = positif dans l'espace stress
        "inflation_rate": 0.0,
    },
    "Stagflation": {
        "unemployment_rate": 2.0,
        "gdp_growth": 2.0,        # baisse GDP = positif
        "interest_rate": 2.0,
        "hpi_growth": 1.0,        # baisse HPI legere = positif
        "inflation_rate": 3.0,
    },
    "Rupture techno": {
        "unemployment_rate": 1.0,
        "gdp_growth": 1.5,        # baisse GDP = positif
        "interest_rate": 0.5,
        "hpi_growth": 0.0,
        "inflation_rate": 0.5,
    },
    "Resserrement": {
        "unemployment_rate": 0.5,
        "gdp_growth": 0.5,        # baisse GDP legere = positif
        "interest_rate": 3.0,
        "hpi_growth": 3.0,        # baisse HPI = positif
        "inflation_rate": 1.5,
    },
    "Reprise": {
        "unemployment_rate": -2.0,
        "gdp_growth": -2.0,       # hausse GDP = negatif dans l'espace stress
        "interest_rate": 0.0,
        "hpi_growth": -2.0,       # hausse HPI = negatif
        "inflation_rate": -0.5,
    },
}

_SOFTMAX_SCALE: float = 2.0


def classify_regime(
    macro_params: Dict[str, float],
) -> RegimeClassification:
    """Classifie le regime macro actuel (FR29).

    Calcule la similarite cosinus entre le vecteur de stress actuel
    et chaque signature de regime, puis applique softmax.

    Args:
        macro_params: Variables macro actuelles (5 cles).

    Returns:
        RegimeClassification avec probabilites et regime dominant.
    """
    base = SCENARIO_BASE

    # Vecteur de stress actuel (deltas par rapport au baseline)
    stress_vector = np.array([
        macro_params.get("unemployment_rate", base.unemployment_rate) - base.unemployment_rate,
        base.gdp_growth - macro_params.get("gdp_growth", base.gdp_growth),  # Inverse : baisse = positif
        macro_params.get("interest_rate", base.interest_rate) - base.interest_rate,
        base.hpi_growth - macro_params.get("hpi_growth", base.hpi_growth),  # Inverse
        macro_params.get("inflation_rate", base.inflation_rate) - base.inflation_rate,
    ])

    # Calcul des scores par regime (produit scalaire avec signature)
    regimes = list(_REGIME_SIGNATURES.keys())
    scores = np.zeros(len(regimes))

    for i, regime in enumerate(regimes):
        sig = _REGIME_SIGNATURES[regime]
        sig_vector = np.array([
            sig["unemployment_rate"],
            sig["gdp_growth"],
            sig["interest_rate"],
            sig["hpi_growth"],
            sig["inflation_rate"],
        ])
        # Produit scalaire (similarite)
        scores[i] = np.dot(stress_vector, sig_vector)

    # Softmax avec temperature
    scores_scaled = scores * _SOFTMAX_SCALE
    exp_scores = np.exp(scores_scaled - np.max(scores_scaled))  # Stabilite numerique
    probabilities = exp_scores / exp_scores.sum()

    prob_dict = {regime: round(float(p), 4) for regime, p in zip(regimes, probabilities)}
    detected = regimes[int(np.argmax(probabilities))]

    return RegimeClassification(
        probabilities=prob_dict,
        detected_regime=detected,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("IFRS 9 COCKPIT — AI Analyst : Couche 4 (Regime)")
    print("=" * 70)

    # Baseline (pas de stress)
    macro_base = {
        "unemployment_rate": SCENARIO_BASE.unemployment_rate,
        "gdp_growth": SCENARIO_BASE.gdp_growth,
        "interest_rate": SCENARIO_BASE.interest_rate,
        "hpi_growth": SCENARIO_BASE.hpi_growth,
        "inflation_rate": SCENARIO_BASE.inflation_rate,
    }

    print("\n[1/3] Regime baseline (pas de stress)...")
    regime_base = classify_regime(macro_base)
    print(f"  Regime detecte : {regime_base.detected_regime}")
    for name, prob in regime_base.probabilities.items():
        print(f"    {name:20s} : {prob:.4f}")

    # Scenario adverse (Stagflation-like)
    print("\n[2/3] Regime sous stress Stagflation...")
    macro_stagflation = macro_base.copy()
    macro_stagflation["unemployment_rate"] = 10.0
    macro_stagflation["gdp_growth"] = -1.0
    macro_stagflation["inflation_rate"] = 5.0
    macro_stagflation["interest_rate"] = 5.5

    regime_stag = classify_regime(macro_stagflation)
    print(f"  Regime detecte : {regime_stag.detected_regime}")
    for name, prob in regime_stag.probabilities.items():
        print(f"    {name:20s} : {prob:.4f}")

    # Scenario Crise financiere
    print("\n[3/3] Regime sous stress Crise financiere...")
    macro_crise = macro_base.copy()
    macro_crise["unemployment_rate"] = 12.0
    macro_crise["gdp_growth"] = -3.0
    macro_crise["hpi_growth"] = -5.0
    macro_crise["interest_rate"] = 1.0

    regime_crise = classify_regime(macro_crise)
    print(f"  Regime detecte : {regime_crise.detected_regime}")
    for name, prob in regime_crise.probabilities.items():
        print(f"    {name:20s} : {prob:.4f}")

    # Validations
    print("\n--- Validations ---")
    all_ok = True

    # V1: 5 regimes, somme = 1
    ok = abs(sum(regime_base.probabilities.values()) - 1.0) < 0.01
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Probabilites somment a 1 ({sum(regime_base.probabilities.values()):.4f})")
    all_ok &= ok

    # V2: Stagflation detectee
    ok = regime_stag.detected_regime == "Stagflation"
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Stagflation detectee ({regime_stag.detected_regime})")
    all_ok &= ok

    # V3: Crise detectee
    ok = regime_crise.detected_regime == "Crise financiere"
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Crise financiere detectee ({regime_crise.detected_regime})")
    all_ok &= ok

    print(f"\n{'=' * 70}")
    if all_ok:
        print("Couche 4 (Regime) validee.")
    else:
        print("ATTENTION : certaines validations ont echoue.")
    print(f"{'=' * 70}")
