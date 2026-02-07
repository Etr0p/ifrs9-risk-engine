"""Modèle Exposure At Default (EAD) pour le Cockpit IFRS 9.

L'EAD représente l'exposition attendue au moment du défaut.
Pour les lignes revolving, l'EAD inclut un tirage additionnel
estimé via le Credit Conversion Factor (CCF) :

    EAD = Drawn + CCF × Undrawn

Pour les prêts à terme, EAD ≈ encours courant (CCF = 1.0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

from ifrs9_cockpit.config import EAD_CONFIG, RANDOM_SEED, SEGMENTS


class EADModel:
    """Modèle EAD avec CCF pour lignes revolving.

    Calcule l'exposition au défaut en distinguant les produits
    revolving (carte de crédit, découvert) des prêts à terme.
    Applique un stress sur le tirage en scénario adverse.

    Attributes:
        seed: Graine aléatoire.
        rng: Générateur numpy.
        avg_ccf_by_type_: CCF moyens calibrés par type de prêt.
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        """Initialise le modèle EAD.

        Args:
            seed: Graine pour reproductibilité.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.avg_ccf_by_type_: Dict[str, float] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "EADModel":
        """Calibre les CCF par type de prêt.

        Dans un vrai contexte, les CCF seraient calibrés sur les
        défauts historiques. Ici on utilise les valeurs réglementaires
        avec un ajustement par segment.

        Args:
            df: DataFrame clients avec loan_type, loan_amount,
                utilization_rate.

        Returns:
            Self (pattern fluent).
        """
        # CCF de base par type de produit
        self.avg_ccf_by_type_ = {
            "Revolving": EAD_CONFIG.ccf_revolving,
            "Term": EAD_CONFIG.ccf_term_loan,
        }
        self._fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        stressed: bool = False,
    ) -> np.ndarray:
        """Calcule l'EAD pour chaque client.

        Pour les revolving :
            - Drawn = loan_amount × utilization_rate
            - Undrawn = loan_amount × (1 - utilization_rate)
            - EAD = Drawn + CCF × Undrawn

        Pour les term loans :
            - EAD = loan_amount (encours total)

        Args:
            df: DataFrame avec loan_type, loan_amount, utilization_rate.
            stressed: Si True, applique un stress sur le tirage.

        Returns:
            Array d'EAD en euros.
        """
        n = len(df)
        loan_amount = df["loan_amount"].values.astype(float)
        utilization = df["utilization_rate"].values.astype(float)
        is_revolving = df["loan_type"].values == "Revolving"

        ead = np.zeros(n)

        # Term loans : EAD = encours courant
        term_mask = ~is_revolving
        ead[term_mask] = loan_amount[term_mask]

        # Revolving : EAD = drawn + CCF × undrawn
        drawn = loan_amount[is_revolving] * utilization[is_revolving]
        undrawn = loan_amount[is_revolving] * (1 - utilization[is_revolving])
        ccf = self.avg_ccf_by_type_.get("Revolving", EAD_CONFIG.ccf_revolving)

        if stressed:
            # En stress, les clients tirent plus sur leurs lignes
            ccf = min(ccf + EAD_CONFIG.utilization_draw_stress, 1.0)

        ead[is_revolving] = drawn + ccf * undrawn

        # Ajustement segments fragiles en stress
        if stressed:
            for seg in SEGMENTS:
                if seg.unemployment_sensitivity > 1.5:
                    seg_mask = df["segment"].values == seg.name
                    ead[seg_mask] *= 1.05  # +5% de tirage additionnel

        # Dispersion réaliste (+/- 5%)
        noise = self.rng.normal(1.0, 0.02, n)
        ead *= noise

        return np.maximum(ead, 0)

    def get_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Résumé des EAD par segment et type de prêt.

        Args:
            df: DataFrame clients.

        Returns:
            DataFrame récapitulatif avec EAD base et stressée.
        """
        ead_base = self.predict(df, stressed=False)
        ead_stress = self.predict(df, stressed=True)

        summary_df = df[["segment", "loan_type"]].copy()
        summary_df["ead_base"] = ead_base
        summary_df["ead_stressed"] = ead_stress
        summary_df["loan_amount"] = df["loan_amount"].values

        return (
            summary_df.groupby(["segment", "loan_type"])
            .agg(
                count=("ead_base", "size"),
                avg_loan=("loan_amount", "mean"),
                avg_ead_base=("ead_base", "mean"),
                avg_ead_stressed=("ead_stressed", "mean"),
                total_ead_base=("ead_base", "sum"),
            )
            .round(2)
            .reset_index()
        )

    def get_ccf_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyse des CCF implicites par segment.

        Calcule le CCF effectif = (EAD - Drawn) / Undrawn.

        Args:
            df: DataFrame clients.

        Returns:
            DataFrame avec CCF implicites par segment/type.
        """
        revolving = df[df["loan_type"] == "Revolving"].copy()
        if len(revolving) == 0:
            return pd.DataFrame()

        ead = self.predict(revolving, stressed=False)
        drawn = revolving["loan_amount"].values * revolving["utilization_rate"].values
        undrawn = revolving["loan_amount"].values * (1 - revolving["utilization_rate"].values)

        # CCF implicite
        ccf_implicit = np.where(
            undrawn > 0,
            (ead - drawn) / undrawn,
            1.0,
        )

        revolving = revolving.copy()
        revolving["ccf_implicit"] = ccf_implicit
        revolving["drawn"] = drawn
        revolving["undrawn"] = undrawn
        revolving["ead"] = ead

        return (
            revolving.groupby("segment")
            .agg(
                count=("ccf_implicit", "size"),
                avg_ccf=("ccf_implicit", "mean"),
                avg_utilization=("utilization_rate", "mean"),
                total_drawn=("drawn", "sum"),
                total_undrawn=("undrawn", "sum"),
                total_ead=("ead", "sum"),
            )
            .round(4)
            .reset_index()
        )
