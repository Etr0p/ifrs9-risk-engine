"""Module de staging IFRS 9 (classification 3-stage).

Implémente la logique d'affectation des expositions aux 3 stages :
    - Stage 1 : Performing (PD 12 mois, ECL 12 mois)
    - Stage 2 : SICR détecté (PD lifetime, ECL lifetime)
    - Stage 3 : Défaut avéré (PD = 100%, ECL lifetime)

Le SICR (Significant Increase in Credit Risk) est déclenché
lorsque la PD courante dépasse un multiple de la PD à l'origination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from ifrs9_cockpit.config import IFRS9_CONFIG


class StagingEngine:
    """Moteur d'affectation aux stages IFRS 9.

    Applique les critères quantitatifs (PD relative + PD absolue)
    et qualitatifs (DPD) pour classifier chaque exposition.

    Attributes:
        sicr_multiplier: Multiplicateur PD pour déclenchement SICR.
        stage3_dpd: Jours de retard seuil pour Stage 3.
        stage3_pd: PD seuil pour Stage 3.
    """

    def __init__(
        self,
        sicr_multiplier: float = IFRS9_CONFIG.sicr_threshold_multiplier,
        stage3_dpd: int = IFRS9_CONFIG.stage3_dpd_threshold,
        stage3_pd: float = IFRS9_CONFIG.stage3_pd_threshold,
    ) -> None:
        """Initialise le moteur de staging.

        Args:
            sicr_multiplier: Ratio PD courante / PD origination pour SICR.
            stage3_dpd: Jours de retard minimum pour Stage 3.
            stage3_pd: PD minimum pour Stage 3.
        """
        self.sicr_multiplier = sicr_multiplier
        self.stage3_dpd = stage3_dpd
        self.stage3_pd = stage3_pd

    def assign_stages(
        self,
        pd_current: np.ndarray,
        pd_origination: np.ndarray,
        dpd: np.ndarray,
        default_flag: np.ndarray,
    ) -> np.ndarray:
        """Assigne un stage IFRS 9 à chaque exposition.

        Logique de priorité :
            1. Stage 3 si default_flag = 1 OU DPD >= 90 OU PD >= 30%
            2. Stage 2 si PD_current >= SICR_multiplier × PD_origination
            3. Stage 1 sinon (performing)

        Args:
            pd_current: PD courante (modèle PIT).
            pd_origination: PD à l'origination du prêt.
            dpd: Jours de retard courants.
            default_flag: Flag de défaut observé (0/1).

        Returns:
            Array d'entiers (1, 2 ou 3).
        """
        n = len(pd_current)
        stages = np.ones(n, dtype=int)  # Default : Stage 1

        # Stage 2 : SICR détecté (critère relatif)
        sicr_mask = pd_current >= self.sicr_multiplier * pd_origination
        stages[sicr_mask] = 2

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
