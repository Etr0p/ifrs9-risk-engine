"""Couche 3 — Seuils de basculement & Reverse Stress Test (FR28).

Identifie les seuils de basculement univaries par balayage parametrique
et execute un reverse stress test joint pour trouver le scenario de rupture
minimal (distance de Mahalanobis minimale au baseline produisant un breach CET1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from ifrs9_cockpit.config import (
    SCENARIO_BASE,
    SECTORS,
    BASEL_CONFIG,
    RISK_APPETITE_CONFIG,
    MACRO_COVARIANCE,
    MACRO_VARIABLES_ORDER,
)
from ifrs9_cockpit.ai_analyst.types import RegimeClassification


# Variables macro avec leur fourchette de balayage et pas.
# Les valeurs de base sont tirees de SCENARIO_BASE (pas hardcodees).
_MACRO_SWEEP = {
    "unemployment_rate": {
        "base": SCENARIO_BASE.unemployment_rate,
        "min": 4.0, "max": 15.0, "step": 0.5, "adverse": 1,
    },
    "gdp_growth": {
        "base": SCENARIO_BASE.gdp_growth,
        "min": -5.0, "max": 5.0, "step": 0.5, "adverse": -1,
    },
    "interest_rate": {
        "base": SCENARIO_BASE.interest_rate,
        "min": 0.0, "max": 10.0, "step": 0.5, "adverse": 1,
    },
    "hpi_growth": {
        "base": SCENARIO_BASE.hpi_growth,
        "min": -10.0, "max": 10.0, "step": 0.5, "adverse": -1,
    },
    "inflation_rate": {
        "base": SCENARIO_BASE.inflation_rate,
        "min": 0.0, "max": 8.0, "step": 0.5, "adverse": 1,
    },
}


def find_tipping_points(
    ecl_total_fn,
    macro_params: Dict[str, float],
    regime: Optional[RegimeClassification] = None,
    ecl_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Identifie les seuils de basculement univaries (FR28).

    Balaye chaque variable macro dans sa fourchette adverse et mesure
    le point ou la sortie de ecl_total_fn depasse le seuil.

    Args:
        ecl_total_fn: Callable(macro_dict) -> float (ECL sous stress).
        macro_params: Variables macro actuelles.
        regime: Classification de regime (optionnel).
        ecl_threshold: Seuil de basculement. Si None, utilise ecl_ead_amber (ratio).

    Returns:
        DataFrame avec variable, base, tipping_value, distance_pp.
    """
    if ecl_threshold is None:
        ecl_threshold = RISK_APPETITE_CONFIG.ecl_ead_amber
    records = []

    for var, sweep in _MACRO_SWEEP.items():
        base_val = macro_params.get(var, sweep["base"])
        tipping_val = None

        # Direction adverse
        if sweep["adverse"] > 0:
            vals = np.arange(base_val, sweep["max"] + sweep["step"], sweep["step"])
        else:
            vals = np.arange(base_val, sweep["min"] - sweep["step"], -sweep["step"])

        for v in vals:
            test_params = macro_params.copy()
            test_params[var] = float(v)
            ecl = ecl_total_fn(test_params)
            if ecl > ecl_threshold:
                tipping_val = float(v)
                break

        distance = abs(tipping_val - base_val) if tipping_val is not None else float("inf")

        records.append({
            "variable": var,
            "base_value": round(base_val, 2),
            "tipping_value": round(tipping_val, 2) if tipping_val is not None else None,
            "distance_pp": round(distance, 2),
            "breached": tipping_val is not None,
        })

    return pd.DataFrame(records)


def _build_stress_params(factor: float) -> Dict[str, float]:
    """Construit les parametres macro stresses pour un facteur donne.

    Args:
        factor: Facteur d'echelle (1.0 = baseline, 2.0 = double stress).

    Returns:
        Dict des 5 variables macro stressees.
    """
    stress_params = {}
    for var, sweep in _MACRO_SWEEP.items():
        base_val = sweep["base"]
        delta = (sweep["max"] - base_val) * sweep["adverse"] * (factor - 1) / 5
        stress_params[var] = base_val + delta
    return stress_params


def _compute_mahalanobis(
    stress_params: Dict[str, float],
    base: object,
) -> float:
    """Calcule la distance de Mahalanobis entre le scenario stress et le baseline.

    Utilise np.linalg.solve au lieu de np.linalg.inv pour une meilleure
    stabilite numerique (evite l'inversion explicite de la matrice).

    Args:
        stress_params: Parametres macro du scenario stress.
        base: Scenario de reference (SCENARIO_BASE).

    Returns:
        Distance de Mahalanobis (float).
    """
    x = np.array([
        stress_params.get(var, getattr(base, var)) - getattr(base, var)
        for var in MACRO_VARIABLES_ORDER
    ])

    Sigma = np.array(MACRO_COVARIANCE)

    try:
        # Utilise solve(Sigma, x) au lieu de inv(Sigma) @ x pour la stabilite.
        # solve(A, b) resout Ax = b, soit Sigma_inv @ x sans inversion explicite.
        Sigma_inv_x = np.linalg.solve(Sigma, x)
        return float(np.sqrt(x @ Sigma_inv_x))
    except np.linalg.LinAlgError:
        # Fallback : diagonale (= euclidienne normalisee)
        diag = np.diag(Sigma)
        return float(np.sqrt(np.sum(x**2 / np.maximum(diag, 1e-10))))


def reverse_stress_test(
    ecl_total_fn,
    macro_params: Dict[str, float],
    regime: Optional[RegimeClassification] = None,
    target_ecl: Optional[float] = None,
    capital_base: Optional[float] = None,
) -> Dict[str, object]:
    """Reverse Stress Test joint — scenario de rupture minimal (FR28, FR54).

    Cherche le facteur d'echelle minimal sur toutes les variables macro
    simultanement qui produit un breach CET1 (ECL_weighted > seuil).

    Le seuil de rupture est le max entre :
        - target_ecl (personnalise) ou 10% du capital CET1 (defaut)
        - 1.5 × ECL baseline (garantit que le seuil est au-dessus du baseline)

    Distance : Mahalanobis (H6) — mesure la distance statistique entre
    le scenario de rupture et le baseline en tenant compte de la structure
    de correlation des variables macro.

    Args:
        ecl_total_fn: Callable(macro_dict) -> float (ECL total sous stress).
        macro_params: Variables macro actuelles.
        regime: Classification de regime (optionnel).
        target_ecl: Seuil ECL personnalise (EUR). Si None, utilise 10% du capital CET1.
        capital_base: Capital CET1 reel en EUR. Si None, utilise rwa_budget × cet1_target.

    Returns:
        Dict avec rst_scenario, rst_ecl, rst_distance, breach.
    """
    base = SCENARIO_BASE
    capital = capital_base if capital_base is not None else BASEL_CONFIG.rwa_budget * BASEL_CONFIG.cet1_target

    # ECL seuil de rupture (FR54 : personnalisable)
    if target_ecl is not None:
        # Cible personnalisee par l'utilisateur : on la respecte telle quelle.
        ecl_breach = target_ecl
    else:
        # Seuil par defaut : 10% du capital CET1, mais garanti au-dessus du baseline.
        # Sans le max, si l'ECL baseline > seuil, le RST retourne distance=0
        # (breach immediat au facteur 1.0, aucun stress necessaire).
        ecl_baseline = ecl_total_fn(macro_params)
        _RST_MULTIPLIER = 1.5  # Le stress doit causer +50% d'ECL au minimum
        ecl_breach = max(capital * 0.10, ecl_baseline * _RST_MULTIPLIER)

    # Balayage du facteur de stress avec pas fin (0.1) pour une detection
    # plus precise du point de breach.
    best_factor = None
    best_ecl = 0
    best_stress_params = {}

    for factor in np.arange(1.0, 10.05, 0.1):
        stress_params = _build_stress_params(factor)
        ecl = ecl_total_fn(stress_params)
        if ecl > ecl_breach:
            best_factor = factor
            best_ecl = ecl
            best_stress_params = stress_params
            break

    # H6. Distance de Mahalanobis (remplace l'approximation 1 factor ~ 1 sigma)
    rst_distance = _compute_mahalanobis(best_stress_params, base) if best_factor is not None else float("inf")

    # Scenario de rupture (reutilise _build_stress_params pour eviter la duplication)
    rst_scenario = {}
    if best_factor is not None:
        for var, val in _build_stress_params(best_factor).items():
            rst_scenario[var] = round(val, 2)

    return {
        "rst_scenario": rst_scenario,
        "rst_ecl": round(best_ecl, 0),
        "rst_distance_sigma": round(rst_distance, 1),
        "breach": best_factor is not None,
        "capital_base": round(capital, 0),
        "ecl_breach_threshold": round(ecl_breach, 0),
    }


if __name__ == "__main__":
    from ifrs9_cockpit.config import SCENARIO_BASE, LOGIT_AMPLITUDE
    from ifrs9_cockpit.utils.helpers import logit, expit

    print("=" * 70)
    print("IFRS 9 COCKPIT — AI Analyst : Couche 3 (Seuils & RST)")
    print("=" * 70)

    # Variables inversees (baisse = adverse)
    _INVERTED_VARS = {"gdp_growth", "hpi_growth"}

    # Map variable -> attribut SectorConfig
    _VAR_TO_SENS = {
        "unemployment_rate": "unemployment_sensitivity_credit",
        "gdp_growth": "gdp_sensitivity_credit",
        "interest_rate": "interest_rate_sensitivity_credit",
        "hpi_growth": "hpi_sensitivity_credit",
        "inflation_rate": "inflation_sensitivity_credit",
    }

    # ECL proxy logit (coherent avec ECLCalculator)
    # Reference EAD pour convertir ratio -> montant absolu (EUR)
    _REF_EAD = 2_000_000_000_000.0  # ~2T EUR (portefeuille 30K entreprises)
    _ECL_BASE_RATIO = 0.040  # ~4% ECL/EAD baseline (coherent avec le DGP v4.5)
    _LOGIT_BASE = float(logit(np.array(_ECL_BASE_RATIO)))

    def ecl_proxy(params: Dict[str, float]) -> float:
        """ECL proxy logit : retourne le montant ECL absolu (EUR)."""
        stress_sum = 0.0
        for var, sens_attr in _VAR_TO_SENS.items():
            base_val = getattr(SCENARIO_BASE, var)
            delta = params.get(var, base_val) - base_val
            if var in _INVERTED_VARS:
                delta = -delta
            avg_sens = sum(
                getattr(s, sens_attr) * s.proportion for s in SECTORS
            )
            stress_sum += (delta / 100) * avg_sens
        ecl_ratio = float(expit(np.array(_LOGIT_BASE + stress_sum * LOGIT_AMPLITUDE)))
        return ecl_ratio * _REF_EAD  # Montant absolu EUR

    macro_params = {
        "unemployment_rate": SCENARIO_BASE.unemployment_rate,
        "gdp_growth": SCENARIO_BASE.gdp_growth,
        "interest_rate": SCENARIO_BASE.interest_rate,
        "hpi_growth": SCENARIO_BASE.hpi_growth,
        "inflation_rate": SCENARIO_BASE.inflation_rate,
    }

    # Tipping points (seuil en montant absolu, coherent avec le proxy)
    # Seuil = ECL baseline × 1.25 (cherche +25% d'augmentation par variable)
    print("\n[1/2] Seuils de basculement univaries...")
    _ecl_baseline = ecl_proxy(macro_params)
    tipping_threshold_abs = _ecl_baseline * 1.25
    tipping = find_tipping_points(ecl_proxy, macro_params, ecl_threshold=tipping_threshold_abs)
    print(tipping.to_string(index=False))

    # Reverse stress test
    print("\n[2/2] Reverse Stress Test joint (Mahalanobis)...")
    rst = reverse_stress_test(ecl_proxy, macro_params)
    print(f"  Distance RST   : {rst['rst_distance_sigma']:.1f} sigma (Mahalanobis)")
    print(f"  ECL rupture    : {rst['rst_ecl']:,.0f}")
    print(f"  Capital base   : {rst['capital_base']:,.0f}")
    print(f"  Breach         : {rst['breach']}")
    if rst["rst_scenario"]:
        print(f"  Scenario RST   : {rst['rst_scenario']}")

    # Validations
    print("\n--- Validations ---")
    all_ok = True

    ok = len(tipping) == 5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Tipping points : 5 variables ({len(tipping)})")
    all_ok &= ok

    ok = rst["rst_distance_sigma"] > 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] RST distance > 0 ({rst['rst_distance_sigma']})")
    all_ok &= ok

    print(f"\n{'=' * 70}")
    if all_ok:
        print("Couche 3 (Seuils & RST, Mahalanobis) validee.")
    else:
        print("ATTENTION : certaines validations ont echoue.")
    print(f"{'=' * 70}")
