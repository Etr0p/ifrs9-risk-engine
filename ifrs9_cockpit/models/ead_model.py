"""Modele Exposure At Default (EAD) pour le Cockpit IFRS 9.

L'EAD represente l'exposition attendue au moment du defaut.
Pour les lignes revolving, l'EAD inclut un tirage additionnel
estime via le Credit Conversion Factor (CCF) :

    EAD = Drawn + CCF x Undrawn

Pour les prets a terme, EAD = encours courant (CCF = 1.0).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Dict

from ifrs9_cockpit.config import EAD_CONFIG, RANDOM_SEED, SECTORS
from ifrs9_cockpit.utils.frame_compat import to_pandas, to_polars, ensure_numpy


class EADModel:
    """Modele EAD avec CCF pour lignes revolving.

    Calcule l'exposition au defaut en distinguant les produits
    revolving des prets a terme. Applique un stress sur le tirage
    en scenario adverse.

    Attributes:
        seed: Graine aleatoire.
        rng: Generateur numpy.
        avg_ccf_by_type_: CCF moyens calibres par type de pret.
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        """Initialise le modele EAD.

        Args:
            seed: Graine pour reproductibilite.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.avg_ccf_by_type_: Dict[str, float] = {}
        self._fitted = False

    def fit(self, df) -> EADModel:
        """Calibre les CCF par type de pret.

        Args:
            df: DataFrame credit avec loan_type, loan_amount,
                utilization_rate.

        Returns:
            Self (pattern fluent).
        """
        self.avg_ccf_by_type_ = {
            "Revolving": EAD_CONFIG.ccf_revolving,
            "Term": EAD_CONFIG.ccf_term_loan,
        }
        self._fitted = True
        return self

    def predict(
        self,
        df,
        stressed: bool = False,
    ) -> np.ndarray:
        """Calcule l'EAD pour chaque entreprise.

        Pour les revolving :
            EAD = Drawn + CCF x Undrawn

        Pour les term loans :
            EAD = loan_amount

        Args:
            df: DataFrame avec loan_type, loan_amount, utilization_rate.
            stressed: Si True, applique un stress sur le tirage.

        Returns:
            Array d'EAD en euros.
        """
        df = to_pandas(df)
        n = len(df)
        loan_amount = df["loan_amount"].values.astype(float)
        utilization = df["utilization_rate"].values.astype(float)
        # Imputer les utilization_rate manquantes par la mediane observee
        nan_mask = np.isnan(utilization)
        if nan_mask.any():
            median_util = np.nanmedian(utilization)
            utilization[nan_mask] = median_util
        is_revolving = df["loan_type"].values == "Revolving"

        ead = np.zeros(n)

        # Term loans : EAD = encours courant
        term_mask = ~is_revolving
        ead[term_mask] = loan_amount[term_mask]

        # Revolving : EAD = drawn + CCF x undrawn
        drawn = loan_amount[is_revolving] * utilization[is_revolving]
        undrawn = loan_amount[is_revolving] * (1 - utilization[is_revolving])
        ccf = self.avg_ccf_by_type_.get("Revolving", EAD_CONFIG.ccf_revolving)

        if stressed:
            ccf = min(ccf + EAD_CONFIG.utilization_draw_stress, 1.0)

        ead[is_revolving] = drawn + ccf * undrawn

        # Ajustement secteurs fragiles en stress
        if stressed:
            for sector in SECTORS:
                if sector.unemployment_sensitivity_credit > 1.5:
                    sector_mask = df["sector"].values == sector.name
                    ead[sector_mask] *= 1.05

        # Dispersion realiste (+/- 2%)
        noise = self.rng.normal(1.0, 0.02, n)
        ead *= noise

        return np.maximum(ead, 0)

    def get_summary(self, df) -> pl.DataFrame:
        """Resume des EAD par secteur et type de pret.

        Args:
            df: DataFrame credit.

        Returns:
            DataFrame recapitulatif avec EAD base et stressee.
        """
        df = to_pandas(df)
        ead_base = self.predict(df, stressed=False)
        ead_stress = self.predict(df, stressed=True)

        import pandas as pd
        summary_df = df[["sector", "loan_type"]].copy()
        summary_df["ead_base"] = ead_base
        summary_df["ead_stressed"] = ead_stress
        summary_df["loan_amount"] = df["loan_amount"].values

        result_pd = (
            summary_df.groupby(["sector", "loan_type"])
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
        return pl.from_pandas(result_pd)

    def get_ccf_analysis(self, df) -> pl.DataFrame:
        """Analyse des CCF implicites par secteur.

        Args:
            df: DataFrame credit.

        Returns:
            DataFrame avec CCF implicites par secteur.
        """
        df = to_pandas(df)
        revolving = df[df["loan_type"] == "Revolving"].copy()
        if len(revolving) == 0:
            return pl.DataFrame()

        ead = self.predict(revolving, stressed=False)
        drawn = revolving["loan_amount"].values * revolving["utilization_rate"].values
        undrawn = revolving["loan_amount"].values * (1 - revolving["utilization_rate"].values)

        ccf_implicit = np.ones_like(ead, dtype=float)
        valid = undrawn > 0
        np.divide(ead - drawn, undrawn, out=ccf_implicit, where=valid)

        revolving = revolving.copy()
        revolving["ccf_implicit"] = ccf_implicit
        revolving["drawn"] = drawn
        revolving["undrawn"] = undrawn
        revolving["ead"] = ead

        result_pd = (
            revolving.groupby("sector")
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
        return pl.from_pandas(result_pd)
