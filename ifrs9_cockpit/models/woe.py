"""Weight of Evidence (WoE) binning pour la transformation des features.

Le WoE binning est une technique standard en credit scoring qui :
    1. Discrétise les variables continues en bins optimaux
    2. Calcule le pouvoir prédictif de chaque bin via le WoE
    3. Produit l'Information Value (IV) pour le feature selection
    4. Transforme les features en valeurs WoE pour la régression logistique
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.config import PD_CONFIG, TARGET


class WoEBinner:
    """Transformateur WoE (Weight of Evidence) pour credit scoring.

    Effectue un binning par quantiles sur les variables numériques,
    calcule le WoE par bin et l'IV globale par variable. Suit le
    pattern fit/transform de scikit-learn.

    Attributes:
        n_bins: Nombre de bins cible.
        min_bin_pct: Part minimale de la population par bin.
        bins_: Dictionnaire {feature: breakpoints} après fit.
        woe_maps_: Dictionnaire {feature: {bin_label: woe_value}} après fit.
        iv_: Dictionnaire {feature: iv_value} après fit.
    """

    def __init__(
        self,
        n_bins: int = PD_CONFIG.n_woe_bins,
        min_bin_pct: float = 0.05,
    ) -> None:
        """Initialise le binner WoE.

        Args:
            n_bins: Nombre de bins pour la discrétisation.
            min_bin_pct: Proportion minimale par bin (protection mono-bin).
        """
        self.n_bins = n_bins
        self.min_bin_pct = min_bin_pct
        self.bins_: Dict[str, np.ndarray] = {}
        self.woe_maps_: Dict[str, Dict[str, float]] = {}
        self.iv_: Dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str = TARGET,
    ) -> "WoEBinner":
        """Calcule les bins et les WoE sur le jeu d'entraînement.

        Pour chaque feature :
            1. Crée des bins par quantiles (gestion des doublons)
            2. Calcule le WoE = ln(% non-défauts / % défauts) par bin
            3. Calcule l'IV = Σ (% non-défauts - % défauts) × WoE

        Args:
            df: DataFrame d'entraînement.
            features: Liste des variables numériques à binner.
            target: Nom de la colonne cible binaire.

        Returns:
            Self (pattern fluent).
        """
        y = df[target].values
        total_events = max(y.sum(), 1)
        total_non_events = max(len(y) - total_events, 1)

        for feat in features:
            x = df[feat].values.copy()

            # Remplacer NaN par la médiane pour le binning
            mask_nan = np.isnan(x) if np.issubdtype(x.dtype, np.floating) else np.zeros(len(x), dtype=bool)
            if mask_nan.any():
                median_val = np.nanmedian(x)
                x[mask_nan] = median_val

            # Créer les breakpoints par quantiles
            breakpoints = self._compute_breakpoints(x)
            self.bins_[feat] = breakpoints

            # Assigner les bins
            bin_indices = np.digitize(x, breakpoints[1:-1])

            # Calculer WoE par bin
            woe_map: Dict[str, float] = {}
            iv_total = 0.0
            n_actual_bins = len(breakpoints) - 1

            for b in range(n_actual_bins):
                mask = bin_indices == b
                n_events = y[mask].sum()
                n_non_events = mask.sum() - n_events

                # Lissage Laplace pour éviter log(0)
                pct_events = (n_events + 0.5) / (total_events + 1)
                pct_non_events = (n_non_events + 0.5) / (total_non_events + 1)

                woe = np.log(pct_non_events / pct_events)
                iv_contrib = (pct_non_events - pct_events) * woe

                bin_label = f"bin_{b}"
                woe_map[bin_label] = float(woe)
                iv_total += iv_contrib

            self.woe_maps_[feat] = woe_map
            self.iv_[feat] = float(iv_total)

        return self

    def transform(
        self,
        df: pd.DataFrame,
        features: List[str],
    ) -> pd.DataFrame:
        """Transforme les features en valeurs WoE.

        Chaque valeur numérique est remplacée par le WoE de son bin.
        Les NaN sont assignés au WoE du bin médian.

        Args:
            df: DataFrame à transformer.
            features: Variables à transformer (doivent avoir été fit).

        Returns:
            DataFrame avec colonnes '{feature}_woe' ajoutées.
        """
        df_out = df.copy()

        for feat in features:
            if feat not in self.bins_:
                continue

            x = df[feat].values.copy()
            breakpoints = self.bins_[feat]
            woe_map = self.woe_maps_[feat]
            n_bins = len(breakpoints) - 1

            # Remplacer NaN par la médiane
            mask_nan = np.isnan(x) if np.issubdtype(x.dtype, np.floating) else np.zeros(len(x), dtype=bool)
            if mask_nan.any():
                # Assigner au bin médian
                median_bin = n_bins // 2
                x[mask_nan] = (breakpoints[median_bin] + breakpoints[min(median_bin + 1, n_bins)]) / 2

            bin_indices = np.digitize(x, breakpoints[1:-1])
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            # Mapper les indices vers les WoE
            woe_values = np.array([
                woe_map.get(f"bin_{b}", 0.0) for b in bin_indices
            ])

            df_out[f"{feat}_woe"] = woe_values

        return df_out

    def fit_transform(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str = TARGET,
    ) -> pd.DataFrame:
        """Fit puis transform en une seule opération.

        Args:
            df: DataFrame d'entraînement.
            features: Variables numériques à traiter.
            target: Colonne cible.

        Returns:
            DataFrame transformé avec colonnes WoE.
        """
        self.fit(df, features, target)
        return self.transform(df, features)

    def get_iv_table(self) -> pd.DataFrame:
        """Retourne un tableau récapitulatif de l'Information Value par feature.

        L'IV mesure le pouvoir prédictif global d'une variable :
            - IV < 0.02 : non prédictif
            - 0.02-0.10 : faible
            - 0.10-0.30 : moyen
            - 0.30-0.50 : fort
            - > 0.50 : suspect (possible overfitting)

        Returns:
            DataFrame trié par IV décroissante avec colonne 'strength'.
        """
        records = []
        for feat, iv in self.iv_.items():
            if iv < 0.02:
                strength = "Non predictif"
            elif iv < 0.10:
                strength = "Faible"
            elif iv < 0.30:
                strength = "Moyen"
            elif iv < 0.50:
                strength = "Fort"
            else:
                strength = "Suspect"
            records.append({
                "feature": feat,
                "iv": round(iv, 4),
                "strength": strength,
                "n_bins": len(self.woe_maps_.get(feat, {})),
            })
        return (
            pd.DataFrame(records)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

    def get_woe_detail(self, feature: str) -> pd.DataFrame:
        """Retourne le détail WoE bin par bin pour une variable donnée.

        Args:
            feature: Nom de la variable.

        Returns:
            DataFrame avec colonnes : bin, lower, upper, woe.
        """
        if feature not in self.bins_:
            raise ValueError(f"Feature '{feature}' non trouvée. Fit d'abord.")

        breakpoints = self.bins_[feature]
        woe_map = self.woe_maps_[feature]
        records = []

        for i in range(len(breakpoints) - 1):
            bin_label = f"bin_{i}"
            records.append({
                "bin": bin_label,
                "lower": breakpoints[i],
                "upper": breakpoints[i + 1],
                "woe": woe_map.get(bin_label, 0.0),
            })
        return pd.DataFrame(records)

    def _compute_breakpoints(self, x: np.ndarray) -> np.ndarray:
        """Calcule les breakpoints par quantiles avec gestion des doublons.

        Si les quantiles produisent des doublons (distribution concentrée),
        on réduit le nombre de bins automatiquement.

        Args:
            x: Valeurs numériques (sans NaN).

        Returns:
            Array de breakpoints incluant -inf et +inf.
        """
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        breakpoints = np.quantile(x, quantiles)

        # Dédupliquer en gardant les breakpoints uniques
        breakpoints = np.unique(breakpoints)

        # Forcer les bornes infinies
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Garantir au minimum 2 bins
        if len(breakpoints) < 3:
            median = np.median(x)
            breakpoints = np.array([-np.inf, median, np.inf])

        return breakpoints
