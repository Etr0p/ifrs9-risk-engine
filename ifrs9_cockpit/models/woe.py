"""Weight of Evidence (WoE) binning pour la transformation des features.

Le WoE binning est une technique standard en credit scoring qui :
    1. Discrétise les variables continues en bins optimaux
    2. Calcule le pouvoir prédictif de chaque bin via le WoE
    3. Produit l'Information Value (IV) pour le feature selection
    4. Transforme les features en valeurs WoE pour la régression logistique

Ameliorations scorecard robuste :
    - Monotonicite imposee via Pool Adjacent Violators (PAV)
    - Granularite minimale post-PAV (min_bin_pct = 5%)
    - Bin NaN explicite avec fusion WoE-proximity si sous-peuple
    - Direction auto-detectee par correlation de Spearman
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.config import PD_CONFIG, TARGET


class WoEBinner:
    """Transformateur WoE (Weight of Evidence) pour credit scoring.

    Effectue un binning par quantiles sur les variables numériques,
    calcule le WoE par bin et l'IV globale par variable. Suit le
    pattern fit/transform de scikit-learn.

    Monotonicity enforcement via PAV (Pool Adjacent Violators) :
        - Auto-detection de la direction (Spearman correlation)
        - Fusion des bins adjacents violant la monotonicite
        - Bin NaN explicite avec son propre WoE

    Attributes:
        n_bins: Nombre de bins cible.
        min_bin_pct: Part minimale de la population par bin.
        bins_: Dictionnaire {feature: breakpoints} après fit.
        woe_maps_: Dictionnaire {feature: {bin_label: woe_value}} après fit.
        iv_: Dictionnaire {feature: iv_value} après fit.
        directions_: Dictionnaire {feature: "increasing"|"decreasing"} après fit.
        nan_woe_: Dictionnaire {feature: woe_value} pour le bin NaN.
    """

    def __init__(
        self,
        n_bins: int = PD_CONFIG.n_woe_bins,
        min_bin_pct: float = 0.05,
        epsilon: float = PD_CONFIG.woe_epsilon,
        min_events_per_bin: int = PD_CONFIG.min_events_per_bin,
    ) -> None:
        """Initialise le binner WoE.

        Args:
            n_bins: Nombre de bins pour la discrétisation.
            min_bin_pct: Proportion minimale par bin (protection mono-bin).
            epsilon: Lissage Laplace pour WoE (evite ln(0) si bin vide).
            min_events_per_bin: Nombre minimum de defauts par bin.
                Bins en-dessous sont fusionnes avec le voisin le plus proche.
        """
        self.n_bins = n_bins
        self.min_bin_pct = min_bin_pct
        self.epsilon = epsilon
        self.min_events_per_bin = min_events_per_bin
        self.bins_: Dict[str, np.ndarray] = {}
        self.woe_maps_: Dict[str, Dict[str, float]] = {}
        self.iv_: Dict[str, float] = {}
        self.directions_: Dict[str, str] = {}
        self.nan_woe_: Dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str = TARGET,
    ) -> "WoEBinner":
        """Calcule les bins et les WoE sur le jeu d'entraînement.

        Pour chaque feature :
            1. Sépare les NaN en bin explicite
            2. Crée des bins par quantiles sur les valeurs valides
            3. Calcule le WoE = ln(% non-défauts / % défauts) par bin
            4. Impose la monotonicité via PAV
            5. Calcule l'IV = Σ (% non-défauts - % défauts) × WoE

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

            # --- Etape 2 : Separer les NaN ---
            mask_nan = (
                np.isnan(x)
                if np.issubdtype(x.dtype, np.floating)
                else np.zeros(len(x), dtype=bool)
            )
            x_valid = x[~mask_nan]
            y_valid = y[~mask_nan]
            y_nan = y[mask_nan]

            # WoE du bin NaN
            eps = self.epsilon
            if mask_nan.any():
                nan_events = y_nan.sum()
                nan_non_events = len(y_nan) - nan_events
                pct_ev_nan = (nan_events + eps) / (total_events + 2 * eps)
                pct_nev_nan = (nan_non_events + eps) / (total_non_events + 2 * eps)
                self.nan_woe_[feat] = float(np.log(pct_nev_nan / pct_ev_nan))
            else:
                self.nan_woe_[feat] = 0.0

            # Créer les breakpoints par quantiles (valeurs valides uniquement)
            breakpoints = self._compute_breakpoints(x_valid)
            self.bins_[feat] = breakpoints

            # Assigner les bins
            bin_indices = np.digitize(x_valid, breakpoints[1:-1])
            n_actual_bins = len(breakpoints) - 1

            # Collecter events/non-events par bin
            bin_events = []
            bin_non_events = []
            bin_medians = []
            for b in range(n_actual_bins):
                mask = bin_indices == b
                n_ev = y_valid[mask].sum()
                n_nev = mask.sum() - n_ev
                bin_events.append(n_ev)
                bin_non_events.append(n_nev)
                # Mediane du bin pour detection de direction
                vals_in_bin = x_valid[mask]
                bin_medians.append(
                    float(np.median(vals_in_bin)) if len(vals_in_bin) > 0 else 0.0
                )

            bin_events = np.array(bin_events, dtype=float)
            bin_non_events = np.array(bin_non_events, dtype=float)
            bin_medians = np.array(bin_medians)

            # --- Etape 1b : Fusion bins sous-peuples (min_events_per_bin) ---
            min_ev = self.min_events_per_bin
            if min_ev > 0 and len(bin_events) > 2:
                events_list = list(bin_events)
                nevents_list = list(bin_non_events)
                medians_list = list(bin_medians)
                bps_list = list(breakpoints)

                i = 0
                while i < len(events_list) and len(events_list) > 2:
                    if events_list[i] < min_ev:
                        # Fusionner avec le voisin ayant le moins de defauts
                        if i == 0:
                            merge_with = 1
                        elif i == len(events_list) - 1:
                            merge_with = i - 1
                        else:
                            merge_with = (
                                i - 1
                                if events_list[i - 1] <= events_list[i + 1]
                                else i + 1
                            )
                        lo, hi = min(i, merge_with), max(i, merge_with)
                        events_list[lo] += events_list[hi]
                        nevents_list[lo] += nevents_list[hi]
                        medians_list[lo] = (medians_list[lo] + medians_list[hi]) / 2
                        del events_list[hi]
                        del nevents_list[hi]
                        del medians_list[hi]
                        del bps_list[hi]  # supprime le breakpoint intermediaire
                        i = 0  # recommencer
                    else:
                        i += 1

                bin_events = np.array(events_list, dtype=float)
                bin_non_events = np.array(nevents_list, dtype=float)
                bin_medians = np.array(medians_list)
                breakpoints = np.array(bps_list)
                self.bins_[feat] = breakpoints
                n_actual_bins = len(bin_events)

            # --- Etape 1c : Auto-detection de la direction ---
            # Taux de defaut par bin
            bin_total = bin_events + bin_non_events
            bin_dr = np.where(
                bin_total > 0, bin_events / bin_total, 0.0
            )
            # Spearman entre mediane du bin et taux de defaut
            # corr < 0 : bin_median monte, event_rate baisse => WoE monte => "increasing"
            # corr > 0 : bin_median monte, event_rate monte => WoE baisse => "decreasing"
            if len(bin_medians) >= 3:
                corr, _ = spearmanr(bin_medians, bin_dr)
                direction = "increasing" if corr < 0 else "decreasing"
            elif len(bin_medians) == 2:
                # Avec 2 bins, comparer directement les taux de defaut
                direction = "increasing" if bin_dr[0] >= bin_dr[1] else "decreasing"
            else:
                direction = "increasing"
            self.directions_[feat] = direction

            # --- Etape 1 : PAV (Pool Adjacent Violators) ---
            bin_events, bin_non_events, breakpoints = self._enforce_monotonicity(
                bin_events, bin_non_events, breakpoints, direction,
                total_events, total_non_events,
            )
            self.bins_[feat] = breakpoints
            n_actual_bins = len(bin_events)

            # --- Etape 1d : Granularite minimale post-PAV ---
            # Fusionne les bins contenant < min_bin_pct de la population
            # avec le voisin dont le WoE est le plus proche (evite micro-segments).
            bin_events, bin_non_events, breakpoints = self._enforce_min_bin_pct(
                bin_events, bin_non_events, breakpoints,
                total_events, total_non_events,
            )
            self.bins_[feat] = breakpoints
            n_actual_bins = len(bin_events)

            # Calculer WoE final par bin (apres PAV)
            woe_map: Dict[str, float] = {}
            iv_total = 0.0

            for b in range(n_actual_bins):
                pct_events = (bin_events[b] + eps) / (total_events + 2 * eps)
                pct_non_events = (bin_non_events[b] + eps) / (total_non_events + 2 * eps)

                woe = np.log(pct_non_events / pct_events)
                iv_contrib = (pct_non_events - pct_events) * woe

                bin_label = f"bin_{b}"
                woe_map[bin_label] = float(woe)
                iv_total += iv_contrib

            # --- Bin NaN : fusion WoE-proximity si sous-peuple ---
            # Si le bin NaN contient < min_bin_pct de la population totale,
            # on le fusionne avec le bin numerique dont le WoE est le plus
            # proche (technique "WoE imputation", Anderson 2007).
            if mask_nan.any():
                nan_count = len(y_nan)
                nan_pct_ev = (y_nan.sum() + eps) / (total_events + 2 * eps)
                nan_pct_nev = (nan_count - y_nan.sum() + eps) / (total_non_events + 2 * eps)
                nan_woe = np.log(nan_pct_nev / nan_pct_ev)

                n_total = total_events + total_non_events
                if nan_count < self.min_bin_pct * n_total and woe_map:
                    # Fusionner avec le bin dont le WoE est le plus proche
                    closest_bin = min(woe_map, key=lambda b: abs(woe_map[b] - nan_woe))
                    # Le NaN prendra le WoE du bin le plus proche lors du transform
                    self.nan_woe_[feat] = woe_map[closest_bin]
                else:
                    # Bin NaN suffisamment peuple : conserver son propre WoE
                    iv_total += (nan_pct_nev - nan_pct_ev) * nan_woe

            self.woe_maps_[feat] = woe_map
            self.iv_[feat] = float(iv_total)

        return self

    def _enforce_monotonicity(
        self,
        bin_events: np.ndarray,
        bin_non_events: np.ndarray,
        breakpoints: np.ndarray,
        direction: str,
        total_events: float,
        total_non_events: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Impose la monotonicite WoE via Pool Adjacent Violators (PAV).

        Fusionne les bins adjacents qui ont le meme taux de defaut
        isotonicise. Utilise sklearn.isotonic.IsotonicRegression pour
        obtenir les taux monotones, puis fusionne les bins consecutifs
        ayant le meme taux cible.

        Garantit un minimum de 2 bins pour preserver le pouvoir discriminant.

        Args:
            bin_events: Nombre de defauts par bin.
            bin_non_events: Nombre de non-defauts par bin.
            breakpoints: Bornes des bins.
            direction: "increasing" (WoE croissant) ou "decreasing" (WoE decroissant).
            total_events: Total defauts global.
            total_non_events: Total non-defauts global.

        Returns:
            Tuple (bin_events, bin_non_events, breakpoints) apres fusion.
        """
        from sklearn.isotonic import IsotonicRegression

        n_bins = len(bin_events)
        if n_bins <= 2:
            return bin_events, bin_non_events, breakpoints

        # Taux de defaut par bin
        bin_total = bin_events + bin_non_events
        bin_dr = np.where(bin_total > 0, bin_events / bin_total, 0.0)

        # Isotonic regression sur les taux de defaut
        # WoE increasing => event rate decreasing => isotonic increasing=False
        # WoE decreasing => event rate increasing => isotonic increasing=True
        iso_increasing = (direction == "decreasing")
        iso = IsotonicRegression(increasing=iso_increasing, out_of_bounds="clip")
        x_idx = np.arange(n_bins, dtype=float)
        dr_isotonic = iso.fit_transform(x_idx, bin_dr, sample_weight=bin_total)

        # Fusionner les bins adjacents avec le meme taux isotonique
        events = list(bin_events)
        non_events = list(bin_non_events)
        bps = list(breakpoints)

        i = 0
        while i < len(events) - 1:
            # Trouver la fin du groupe de bins avec le meme dr_isotonic
            j = i + 1
            while j < len(events) and abs(dr_isotonic[j] - dr_isotonic[i]) < 1e-10:
                j += 1

            if j > i + 1:
                # Fusionner bins [i:j] en un seul
                merged_ev = sum(events[i:j])
                merged_nev = sum(non_events[i:j])
                events[i] = merged_ev
                non_events[i] = merged_nev
                del events[i + 1:j]
                del non_events[i + 1:j]
                # Supprimer les breakpoints intermediaires
                del bps[i + 1:j]
                # Mettre a jour dr_isotonic
                dr_isotonic = np.delete(dr_isotonic, range(i + 1, j))

            i += 1

        return (
            np.array(events, dtype=float),
            np.array(non_events, dtype=float),
            np.array(bps),
        )

    def _enforce_min_bin_pct(
        self,
        bin_events: np.ndarray,
        bin_non_events: np.ndarray,
        breakpoints: np.ndarray,
        total_events: float,
        total_non_events: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fusionne les bins sous-peuples apres PAV (granularite minimale).

        Tout bin final contenant moins de min_bin_pct de la population totale
        est fusionne avec le voisin dont le WoE est le plus proche, pour
        eviter le sur-apprentissage sur des micro-segments.

        Ref: Siddiqi (2006), Anderson (2007) — controle post-PAV.

        Args:
            bin_events: Nombre de defauts par bin.
            bin_non_events: Nombre de non-defauts par bin.
            breakpoints: Bornes des bins.
            total_events: Total defauts global.
            total_non_events: Total non-defauts global.

        Returns:
            Tuple (bin_events, bin_non_events, breakpoints) apres fusion.
        """
        n_total = total_events + total_non_events
        min_count = self.min_bin_pct * n_total
        eps = self.epsilon

        events = list(bin_events)
        non_events = list(bin_non_events)
        bps = list(breakpoints)

        changed = True
        while changed and len(events) > 2:
            changed = False
            for i in range(len(events)):
                bin_count = events[i] + non_events[i]
                if bin_count < min_count and len(events) > 2:
                    # Compute WoE of current bin
                    pct_ev_i = (events[i] + eps) / (total_events + 2 * eps)
                    pct_nev_i = (non_events[i] + eps) / (total_non_events + 2 * eps)
                    woe_i = np.log(pct_nev_i / pct_ev_i)

                    # Find neighbor with closest WoE
                    best_neighbor = None
                    best_dist = float("inf")
                    for j in [i - 1, i + 1]:
                        if 0 <= j < len(events):
                            pct_ev_j = (events[j] + eps) / (total_events + 2 * eps)
                            pct_nev_j = (non_events[j] + eps) / (total_non_events + 2 * eps)
                            woe_j = np.log(pct_nev_j / pct_ev_j)
                            dist = abs(woe_i - woe_j)
                            if dist < best_dist:
                                best_dist = dist
                                best_neighbor = j

                    if best_neighbor is not None:
                        lo, hi = min(i, best_neighbor), max(i, best_neighbor)
                        events[lo] += events[hi]
                        non_events[lo] += non_events[hi]
                        del events[hi]
                        del non_events[hi]
                        del bps[hi]
                        changed = True
                        break  # restart scan

        return (
            np.array(events, dtype=float),
            np.array(non_events, dtype=float),
            np.array(bps),
        )

    def transform(
        self,
        df: pd.DataFrame,
        features: List[str],
    ) -> pd.DataFrame:
        """Transforme les features en valeurs WoE.

        Chaque valeur numérique est remplacée par le WoE de son bin.
        Les NaN recoivent le WoE du bin NaN explicite.

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

            # Detecter NaN
            mask_nan = (
                np.isnan(x)
                if np.issubdtype(x.dtype, np.floating)
                else np.zeros(len(x), dtype=bool)
            )

            # Initialiser avec le WoE NaN pour toutes les positions
            woe_values = np.full(len(x), self.nan_woe_.get(feat, 0.0))

            # Pour les valeurs valides, assigner le WoE du bin correspondant
            if (~mask_nan).any():
                x_valid = x[~mask_nan]
                bin_indices = np.digitize(x_valid, breakpoints[1:-1])
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                woe_values[~mask_nan] = np.array([
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

        # Ajouter le bin NaN si present
        if feature in self.nan_woe_ and self.nan_woe_[feature] != 0.0:
            records.append({
                "bin": "bin_nan",
                "lower": np.nan,
                "upper": np.nan,
                "woe": self.nan_woe_[feature],
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
        if len(x) == 0:
            return np.array([-np.inf, 0.0, np.inf])

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
