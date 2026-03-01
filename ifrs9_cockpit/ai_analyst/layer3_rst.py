"""Couche 3 — Seuils de basculement & Reverse Stress Test (FR28).

Identifie les seuils de basculement univaries par balayage parametrique
et execute un reverse stress test joint pour trouver le scenario de rupture
minimal (distance de Mahalanobis minimale au baseline produisant un breach CET1).

Inclut un moteur adversarial par Evolution Differentielle (DE) dans l'espace
Cholesky (Z-space) pour contourner les cliff effects SICR S1/S2/S3 et generer
un front de Pareto plausibilite × severite.

References :
    - Traccucci et al. (2019), "A Triptych Approach for Reverse Stress Testing"
    - Hurlin, Lajaunie & Pull (2026), "Reverse Stress Testing Geopolitical Risk"
    - scipy.optimize.differential_evolution (gradient-free, handles discontinuities)
"""

from __future__ import annotations

import logging
import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple

from scipy.optimize import differential_evolution
from scipy.stats.qmc import Sobol

from ifrs9_cockpit.config import (
    SCENARIO_BASE,
    SECTORS,
    BASEL_CONFIG,
    RISK_APPETITE_CONFIG,
    MACRO_COVARIANCE,
    MACRO_VARIABLES_ORDER,
    PREDEFINED_SCENARIOS,
)
from ifrs9_cockpit.ai_analyst.types import RegimeClassification

logger = logging.getLogger(__name__)


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
) -> pl.DataFrame:
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

    return pl.DataFrame(records)


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


# ============================================================
# Adversarial Reverse Stress Test (DE dans l'espace Cholesky)
# ============================================================

# Decomposition de Cholesky de la matrice de covariance macro.
# Σ = L·Lᵀ  →  X = μ + L·Z  →  ‖Z‖₂ = distance de Mahalanobis
# L'optimisateur travaille dans l'espace Z (orthogonal, spherique),
# pas dans l'espace macro correle (X).
_SIGMA = np.array(MACRO_COVARIANCE)
_CHOLESKY_L = np.linalg.cholesky(_SIGMA)

# Vecteur baseline μ dans l'ordre MACRO_VARIABLES_ORDER
_MU = np.array([getattr(SCENARIO_BASE, var) for var in MACRO_VARIABLES_ORDER])

# Bornes physiques des variables macro (empechent les scenarios impossibles)
_MACRO_BOUNDS = {
    "unemployment_rate": (2.0, 25.0),
    "gdp_growth": (-15.0, 10.0),
    "interest_rate": (-1.0, 15.0),
    "hpi_growth": (-20.0, 15.0),
    "inflation_rate": (-2.0, 15.0),
}
_BOUNDS_LO = np.array([_MACRO_BOUNDS[v][0] for v in MACRO_VARIABLES_ORDER])
_BOUNDS_HI = np.array([_MACRO_BOUNDS[v][1] for v in MACRO_VARIABLES_ORDER])


def _z_to_macro(z: np.ndarray) -> Dict[str, float]:
    """Transforme un vecteur Z (espace Cholesky) en parametres macro.

    X = μ + L·Z, puis clip aux bornes physiques.
    """
    x = _MU + _CHOLESKY_L @ z
    x = np.clip(x, _BOUNDS_LO, _BOUNDS_HI)
    return {var: float(x[i]) for i, var in enumerate(MACRO_VARIABLES_ORDER)}


def _macro_to_z(macro_params: Dict[str, float]) -> np.ndarray:
    """Transforme des parametres macro en vecteur Z (espace Cholesky).

    Z = L⁻¹·(X - μ)
    """
    x = np.array([macro_params.get(var, getattr(SCENARIO_BASE, var))
                   for var in MACRO_VARIABLES_ORDER])
    return np.linalg.solve(_CHOLESKY_L, x - _MU)


def _predefined_scenarios_as_z() -> List[np.ndarray]:
    """Convertit les scenarios predefinis du dashboard en vecteurs Z.

    Les scenarios predefinis utilisent des valeurs slider (bp, bipolar, pct).
    On les convertit d'abord en valeurs macro absolues, puis en Z-space.
    """
    z_vectors = []
    for name, slider_vals in PREDEFINED_SCENARIOS.items():
        macro = {
            "unemployment_rate": SCENARIO_BASE.unemployment_rate + abs(slider_vals.get("unemployment_bipolar", 0.0)),
            "gdp_growth": slider_vals.get("gdp_pct", SCENARIO_BASE.gdp_growth),
            "interest_rate": SCENARIO_BASE.interest_rate + slider_vals.get("interest_rate_bp", 0.0) / 100.0,
            "hpi_growth": slider_vals.get("hpi_pct", SCENARIO_BASE.hpi_growth),
            "inflation_rate": slider_vals.get("inflation_pct", SCENARIO_BASE.inflation_rate),
        }
        z_vectors.append(_macro_to_z(macro))
    return z_vectors


def _build_elite_population(
    n_total: int,
    n_dims: int = 5,
    z_bound: float = 6.0,
    seed: int = 42,
) -> np.ndarray:
    """Construit la population initiale pour DE : elite historique + Sobol.

    Les 8 scenarios predefinis (crises historiques) peuplent les premiers
    individus. Le reste est rempli par une sequence de Sobol quasi-aleatoire
    dans la boite [-z_bound, +z_bound]^5.

    Args:
        n_total: Taille totale de la population.
        n_dims: Nombre de dimensions (5 variables macro).
        z_bound: Borne de l'hypercube Z (6σ = couvre 99.99997%).
        seed: Graine aleatoire.

    Returns:
        Array (n_total, n_dims) dans l'espace Z, normalise dans [0, 1]
        pour scipy DE (qui attend init dans [0, 1]^d).
    """
    # Scenarios historiques convertis en Z-space
    z_hist = _predefined_scenarios_as_z()

    # Sequence Sobol pour remplir le reste (n = puissance de 2 pour Sobol)
    n_sobol = max(0, n_total - len(z_hist))
    if n_sobol > 0:
        # Arrondir a la puissance de 2 superieure (Sobol balance)
        n_sobol_pow2 = 2 ** int(np.ceil(np.log2(max(n_sobol, 2))))
        sampler = Sobol(d=n_dims, scramble=True, seed=seed)
        sobol_unit = sampler.random(n_sobol_pow2)[:n_sobol]
        z_sobol = sobol_unit * 2 * z_bound - z_bound
    else:
        z_sobol = np.empty((0, n_dims))

    # Combine : historique + sobol
    z_all = np.vstack(z_hist[:n_total] + [z_sobol]) if len(z_hist) > 0 else z_sobol
    z_all = z_all[:n_total]

    # Normaliser dans [0, 1] pour scipy DE (init= attend des valeurs dans [0,1]^d)
    pop_unit = (z_all + z_bound) / (2 * z_bound)
    pop_unit = np.clip(pop_unit, 0.0, 1.0)
    return pop_unit


def adversarial_reverse_stress_test(
    ecl_total_fn,
    macro_params: Dict[str, float],
    regime: Optional[RegimeClassification] = None,
    target_ecl: Optional[float] = None,
    capital_base: Optional[float] = None,
    sigma_budgets: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    n_restarts: int = 3,
    seed: int = 42,
) -> Dict[str, object]:
    """Adversarial Reverse Stress Test par Evolution Differentielle en espace Cholesky.

    Optimise dans l'espace Z orthogonal (Σ = L·Lᵀ, X = μ + L·Z) ou la
    contrainte de plausibilite Mahalanobis se reduit a ‖Z‖₂ ≤ σ_budget.

    Architecture :
        1. Soft penalty funnel (pas de death penalty) pour guider DE vers la
           zone plausible meme depuis des points hors-limites.
        2. Population elite : 8 crises historiques + Sobol quasi-aleatoire.
        3. Multi-start (3 seeds) pour echapper aux optima locaux.
        4. ε-constraint Pareto : boucle sur sigma_budgets pour tracer le front
           plausibilite × severite.

    References :
        - Hurlin, Lajaunie & Pull (2026), arXiv:2601.03983
        - Traccucci et al. (2019), "A Triptych Approach", Risk.net

    Args:
        ecl_total_fn: Callable(macro_dict) -> float (ECL en EUR).
        macro_params: Variables macro actuelles.
        regime: Classification de regime (optionnel).
        target_ecl: Seuil ECL personnalise (EUR).
        capital_base: Capital CET1 en EUR.
        sigma_budgets: Seuils Mahalanobis pour le front de Pareto.
        n_restarts: Nombre de runs DE independants par budget sigma.
        seed: Graine de base pour reproductibilite.

    Returns:
        Dict compatible avec reverse_stress_test() + champs additionnels :
            - rst_scenario, rst_ecl, rst_distance_sigma, breach,
              capital_base, ecl_breach_threshold (backward-compat)
            - pareto_front: List[Dict] avec sigma_budget, ecl, distance, scenario
            - method: "adversarial_de_cholesky"
    """
    capital = capital_base if capital_base is not None else (
        BASEL_CONFIG.rwa_budget * BASEL_CONFIG.cet1_target
    )

    # Seuil de breach (meme logique que reverse_stress_test)
    if target_ecl is not None:
        ecl_breach = target_ecl
    else:
        ecl_baseline = ecl_total_fn(macro_params)
        ecl_breach = max(capital * 0.10, ecl_baseline * 1.5)

    # --- Parametres DE ---
    z_bound = 6.0  # Borne de recherche dans Z-space (6σ)
    bounds_z = [(-z_bound, z_bound)] * 5
    popsize = 12  # Total = popsize × 5 = 60 individus

    # Population elite (meme pour tous les budgets sigma)
    init_pop = _build_elite_population(popsize * 5, seed=seed)

    # ECL baseline (calcule une seule fois)
    ecl_baseline_val = ecl_total_fn(macro_params)
    # Coefficient de penalite proportionnel a l'ECL baseline
    penalty_scale = max(ecl_baseline_val, 1.0)

    def _run_de_for_budget(sigma_max: float, run_seed: int) -> Dict:
        """Un run DE pour un budget sigma donne.

        Objectif : maximiser ECL (= minimiser -ECL) sous ‖Z‖ ≤ sigma_max.
        Soft penalty funnel : au-dela de sigma_max, la penalite quadratique
        ramene le solver vers la zone plausible.
        """
        def objective(z):
            z_norm = float(np.linalg.norm(z))
            params = _z_to_macro(z)
            ecl_val = ecl_total_fn(params)

            # Soft penalty funnel : penalite quadratique hors-budget
            if z_norm > sigma_max:
                penalty = penalty_scale * (z_norm - sigma_max) ** 2
            else:
                penalty = 0.0

            # Minimiser le negatif de l'ECL (= maximiser l'ECL)
            return -(ecl_val - penalty)

        result = differential_evolution(
            func=objective,
            bounds=bounds_z,
            init=init_pop,
            seed=run_seed,
            maxiter=80,
            popsize=popsize,
            mutation=(0.5, 1.5),  # Dithering pour les cliff effects SICR
            recombination=0.5,    # Lower CR pour fonctions non-lisses
            tol=1e-5,
            polish=False,  # Pas de L-BFGS-B (non-differentiable)
        )

        z_opt = result.x
        z_norm = float(np.linalg.norm(z_opt))
        macro_opt = _z_to_macro(z_opt)
        ecl_opt = ecl_total_fn(macro_opt)

        return {
            "z": z_opt,
            "z_norm": z_norm,
            "macro": macro_opt,
            "ecl": ecl_opt,
            "converged": result.success,
        }

    def _find_design_point(run_seed: int) -> Dict:
        """Trouve le design point : scenario de breach a distance minimale.

        Objectif : minimiser ‖Z‖ sous ECL(Z) > ecl_breach.
        C'est le dual du Pareto : au lieu de maximiser ECL sous contrainte
        de distance, on minimise la distance sous contrainte de breach.
        Ref : Hurlin et al. (2026), FORM design point.
        """
        def objective(z):
            z_norm = float(np.linalg.norm(z))
            params = _z_to_macro(z)
            ecl_val = ecl_total_fn(params)

            # Si pas de breach : forte penalite proportionnelle au deficit
            if ecl_val < ecl_breach:
                deficit = (ecl_breach - ecl_val) / max(ecl_breach, 1.0)
                return z_norm + 100.0 * deficit
            # Si breach : minimiser la distance (= trouver le scenario le plus proche)
            return z_norm

        result = differential_evolution(
            func=objective,
            bounds=bounds_z,
            init=init_pop,
            seed=run_seed,
            maxiter=80,
            popsize=popsize,
            mutation=(0.5, 1.5),
            recombination=0.5,
            tol=1e-5,
            polish=False,
        )

        z_opt = result.x
        z_norm = float(np.linalg.norm(z_opt))
        macro_opt = _z_to_macro(z_opt)
        ecl_opt = ecl_total_fn(macro_opt)

        return {
            "z": z_opt,
            "z_norm": z_norm,
            "macro": macro_opt,
            "ecl": ecl_opt,
            "converged": result.success,
        }

    # --- Design point : scenario de breach a distance minimale ---
    design_point = None
    for restart in range(n_restarts):
        dp = _find_design_point(seed + restart * 100 + 777)
        if dp["ecl"] > ecl_breach:
            if design_point is None or dp["z_norm"] < design_point["z_norm"]:
                design_point = dp

    # --- Front de Pareto par ε-constraint ---
    pareto_front = []

    for sigma_budget in sigma_budgets:
        best_run = None
        for restart in range(n_restarts):
            run_seed = seed + restart * 100 + int(sigma_budget * 10)
            result = _run_de_for_budget(sigma_budget, run_seed)

            if best_run is None or result["ecl"] > best_run["ecl"]:
                best_run = result

        pareto_point = {
            "sigma_budget": sigma_budget,
            "ecl": round(best_run["ecl"], 0),
            "distance_sigma": round(best_run["z_norm"], 2),
            "scenario": {var: round(val, 2) for var, val in best_run["macro"].items()},
            "breach": bool(best_run["ecl"] > ecl_breach),
        }
        pareto_front.append(pareto_point)

    # Le resultat principal = design point (si breach) ou meilleur Pareto breach
    if design_point is not None:
        best_overall = design_point
    else:
        # Fallback : run DE sans contrainte sigma pour trouver le max ECL
        best_fallback = None
        for restart in range(n_restarts):
            result = _run_de_for_budget(z_bound, seed + restart * 100 + 999)
            if best_fallback is None or result["ecl"] > best_fallback["ecl"]:
                best_fallback = result
        best_overall = best_fallback

    # Construire le resultat (backward-compatible avec reverse_stress_test)
    rst_scenario = {var: round(val, 2) for var, val in best_overall["macro"].items()}
    rst_ecl = round(best_overall["ecl"], 0)
    rst_distance = round(best_overall["z_norm"], 1)
    # bool() natif Python pour compatibilite avec `is True` / `is False`
    breach = bool(best_overall["ecl"] > ecl_breach)

    return {
        # Backward-compatible keys
        "rst_scenario": rst_scenario,
        "rst_ecl": rst_ecl,
        "rst_distance_sigma": rst_distance,
        "breach": breach,
        "capital_base": round(capital, 0),
        "ecl_breach_threshold": round(ecl_breach, 0),
        # Nouveaux champs adversarial
        "pareto_front": pareto_front,
        "method": "adversarial_de_cholesky",
    }


if __name__ == "__main__":
    import pandas as pd
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
    print(tipping.to_pandas().to_string(index=False))

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
