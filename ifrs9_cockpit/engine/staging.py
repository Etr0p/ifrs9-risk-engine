"""Module de staging IFRS 9 (classification 3-stage).

Implémente la logique d'affectation des expositions aux 3 stages :
    - Stage 1 : Performing (PD 12 mois, ECL 12 mois)
    - Stage 2 : SICR détecté (PD lifetime, ECL lifetime)
    - Stage 3 : Défaut avéré (PD = 100%, ECL lifetime)

Le SICR (Significant Increase in Credit Risk) est déterminé par un
score multi-facteurs (IFRS 9 §B5.5.17) combinant :
    - Ratio PD relatif (PD_current / PD_origination - 1)
    - Delta PD absolu (PD_current - PD_origination)
    - DPD normalisé (DPD / 30)
    - Z-score macro (forward-looking)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from ifrs9_cockpit.config import IFRS9_CONFIG, SICR_CONFIG, SCENARIO_BASE


def compute_macro_z(macro_params: Dict[str, float]) -> float:
    """Calcule le Z-score macro composite normalise pour le SICR.

    Le Z-score mesure la deviation des conditions macro courantes
    par rapport au scenario de base. Positif = conditions adverses.

    Chaque delta macro est normalise par la plage typique de la variable
    (ecart adverse - base) pour eviter que les variables a grande echelle
    (HPI, chomage) ne dominent le score.

    Args:
        macro_params: Dict avec cles unemployment_rate, gdp_growth,
            interest_rate, hpi_growth, inflation_rate.

    Returns:
        Z-score macro composite (scalaire), normalise, ~[-1, +1] en conditions normales.
    """
    base = SCENARIO_BASE
    # Plages typiques adverse-base pour normalisation (depuis SCENARIO_ADVERSE)
    # Evite que HPI (range ~30pp) domine vs taux (range ~2pp)
    _NORM = {
        "unemployment_rate": 3.0,   # 10.5 - 7.5
        "gdp_growth": 2.7,          # 1.2 - (-1.5)
        "interest_rate": 1.5,        # 5.0 - 3.5
        "hpi_growth": 10.0,          # 2.0 - (-8.0)
        "inflation_rate": 3.0,       # 5.5 - 2.5
    }
    z = 0.0
    # Chomage : hausse = adverse
    z += (macro_params.get("unemployment_rate", base.unemployment_rate)
          - base.unemployment_rate) / _NORM["unemployment_rate"]
    # PIB : baisse = adverse
    z += (base.gdp_growth
          - macro_params.get("gdp_growth", base.gdp_growth)) / _NORM["gdp_growth"]
    # Taux : hausse = adverse
    z += (macro_params.get("interest_rate", base.interest_rate)
          - base.interest_rate) / _NORM["interest_rate"]
    # HPI : baisse = adverse
    z += (base.hpi_growth
          - macro_params.get("hpi_growth", base.hpi_growth)) / _NORM["hpi_growth"]
    # Inflation : hausse = adverse
    z += (macro_params.get("inflation_rate", base.inflation_rate)
          - base.inflation_rate) / _NORM["inflation_rate"]
    # Moyenne des 5 composantes pour un Z-score unitaire
    return z / 5.0


def compute_sicr_score(
    pd_current: np.ndarray,
    pd_origination: np.ndarray,
    dpd: np.ndarray,
    macro_params: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Score SICR multi-facteurs (IFRS 9 §B5.5.17).

    Combine 4 facteurs ponderes par SICR_CONFIG :
        score = w_pd_ratio × (PD_current / PD_origination - 1)
              + w_pd_delta × max(0, PD_current - PD_origination)
              + w_dpd × (DPD / 30)
              + w_macro × macro_z

    Args:
        pd_current: PD courante (Point-In-Time).
        pd_origination: PD a l'origination.
        dpd: Jours de retard courants.
        macro_params: Parametres macro courants (optionnel).

    Returns:
        Array de scores SICR.
    """
    # Ratio PD relatif (capped a 5.0 pour eviter une sensibilite perverse
    # sur les credits a PD origination tres basse, e.g. 0.1%→0.5% = ratio 4)
    pd_ratio = np.clip(
        pd_current / np.maximum(pd_origination, 1e-6) - 1,
        0.0, 5.0,
    )
    # Delta PD absolu
    pd_delta = np.maximum(0, pd_current - pd_origination)
    # DPD normalise
    dpd_norm = dpd / 30.0
    # Z-score macro (0 si pas de parametres macro)
    if macro_params is not None:
        macro_z = compute_macro_z(macro_params)
    else:
        macro_z = 0.0

    score = (
        SICR_CONFIG.w_pd_ratio * pd_ratio
        + SICR_CONFIG.w_pd_delta * pd_delta
        + SICR_CONFIG.w_dpd * dpd_norm
        + SICR_CONFIG.w_macro * macro_z
    )
    return score


class StagingEngine:
    """Moteur d'affectation aux stages IFRS 9.

    Utilise un score SICR multi-facteurs pour le declenchement Stage 2
    et les criteres quantitatifs/qualitatifs classiques pour Stage 3.

    Attributes:
        stage3_dpd: Jours de retard seuil pour Stage 3.
        stage3_pd: PD seuil pour Stage 3.
        sicr_threshold: Seuil SICR pour declenchement Stage 2.
    """

    def __init__(
        self,
        stage3_dpd: int = IFRS9_CONFIG.stage3_dpd_threshold,
        stage3_pd: float = IFRS9_CONFIG.stage3_pd_threshold,
        sicr_threshold: float = SICR_CONFIG.threshold,
    ) -> None:
        """Initialise le moteur de staging.

        Args:
            stage3_dpd: Jours de retard minimum pour Stage 3.
            stage3_pd: PD minimum pour Stage 3.
            sicr_threshold: Seuil de score SICR pour Stage 2.
        """
        self.stage3_dpd = stage3_dpd
        self.stage3_pd = stage3_pd
        self.sicr_threshold = sicr_threshold

    def assign_stages(
        self,
        pd_current: np.ndarray,
        pd_origination: np.ndarray,
        dpd: np.ndarray,
        default_flag: np.ndarray,
        macro_params: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Assigne un stage IFRS 9 à chaque exposition.

        Logique de priorité :
            1. Stage 3 si default_flag = 1 OU DPD >= 90 OU PD >= 30%
            2. Stage 2 si SICR_score > threshold (multi-facteurs)
            3. Stage 1 sinon (performing)

        Args:
            pd_current: PD courante (modèle PIT).
            pd_origination: PD à l'origination du prêt.
            dpd: Jours de retard courants.
            default_flag: Flag de défaut observé (0/1).
            macro_params: Parametres macro courants (optionnel, pour SICR).

        Returns:
            Array d'entiers (1, 2 ou 3).
        """
        n = len(pd_current)
        stages = np.ones(n, dtype=int)  # Default : Stage 1

        # Stage 2 : SICR multi-facteurs
        sicr_score = compute_sicr_score(
            pd_current, pd_origination, dpd, macro_params,
        )
        stages[sicr_score > self.sicr_threshold] = 2

        # Stage 3 : Défaut avéré (priorité sur Stage 2)
        stage3_mask = (
            (default_flag == 1)
            | (dpd >= self.stage3_dpd)
            | (pd_current >= self.stage3_pd)
        )
        stages[stage3_mask] = 3

        return stages

    def compute_transition_matrix(
        self,
        stages_t0: np.ndarray,
        stages_t1: np.ndarray,
    ) -> pd.DataFrame:
        """Calcule la matrice de transition entre deux dates.

        Args:
            stages_t0: Stages à la date initiale.
            stages_t1: Stages à la date finale.

        Returns:
            DataFrame 3×3 avec les probabilités de transition.
        """
        matrix = np.zeros((3, 3))

        for from_stage in [1, 2, 3]:
            mask = stages_t0 == from_stage
            total = mask.sum()
            if total == 0:
                continue
            for to_stage in [1, 2, 3]:
                count = ((stages_t0 == from_stage) & (stages_t1 == to_stage)).sum()
                matrix[from_stage - 1, to_stage - 1] = count / total

        labels = ["Stage 1", "Stage 2", "Stage 3"]
        return pd.DataFrame(matrix, index=labels, columns=labels).round(4)

    def get_stage_summary(
        self,
        stages: np.ndarray,
        ead: np.ndarray,
    ) -> pd.DataFrame:
        """Résumé de la distribution des stages.

        Args:
            stages: Array de stages (1, 2, 3).
            ead: Array d'EAD correspondantes.

        Returns:
            DataFrame avec count, %, EAD totale et EAD % par stage.
        """
        total_count = len(stages)
        total_ead = ead.sum()

        records = []
        for stage in [1, 2, 3]:
            mask = stages == stage
            count = mask.sum()
            stage_ead = ead[mask].sum()
            records.append({
                "stage": f"Stage {stage}",
                "count": int(count),
                "pct_count": round(count / total_count, 4) if total_count > 0 else 0,
                "total_ead": round(stage_ead, 2),
                "pct_ead": round(stage_ead / total_ead, 4) if total_ead > 0 else 0,
            })

        return pd.DataFrame(records)
