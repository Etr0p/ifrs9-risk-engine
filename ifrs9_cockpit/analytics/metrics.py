"""Métriques de performance pour les modèles de risque de crédit.

Implémente les 4 métriques obligatoires IFRS 9 :
    - AUC-ROC (Area Under Receiver Operating Characteristic)
    - Gini (= 2×AUC - 1)
    - KS (Kolmogorov-Smirnov)
    - PSI (Population Stability Index)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.config import PD_CONFIG


class ModelMetrics:
    """Calcul centralisé des métriques de discrimination et de stabilité.

    Fournit des méthodes statiques réutilisables par tous les modèles
    PD/LGD et par le module Virtual CRO pour le monitoring.

    Example:
        >>> metrics = ModelMetrics.compute_all(y_true, y_pred_proba)
        >>> print(metrics["auc"], metrics["gini"], metrics["ks"])
    """

    @staticmethod
    def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calcule l'AUC-ROC.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.

        Returns:
            AUC entre 0 et 1.
        """
        return float(roc_auc_score(y_true, y_score))

    @staticmethod
    def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calcule le coefficient de Gini (= 2×AUC - 1).

        Mesure le pouvoir discriminant du modèle.
        Gini = 0 → modèle aléatoire, Gini = 1 → modèle parfait.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.

        Returns:
            Coefficient de Gini entre -1 et 1.
        """
        return 2.0 * roc_auc_score(y_true, y_score) - 1.0

    @staticmethod
    def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calcule la statistique de Kolmogorov-Smirnov.

        Distance maximale entre les CDF cumulées des défauts
        et des non-défauts. KS > 0.40 = excellent pouvoir discriminant.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.

        Returns:
            Statistique KS entre 0 et 1.
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return float(np.max(tpr - fpr))

    @staticmethod
    def psi(
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Calcule le Population Stability Index (PSI).

        Mesure la dérive entre deux distributions de scores.
        PSI < 0.10 → stable, 0.10-0.25 → shift modéré, > 0.25 → drift significatif.

        Args:
            expected: Distribution de référence (train).
            actual: Distribution observée (test/production).
            n_bins: Nombre de bins pour la discrétisation.

        Returns:
            Valeur PSI (>= 0).
        """
        # Créer les bins sur la distribution de référence
        breakpoints = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        # Supprimer les doublons pour éviter les bins vides
        breakpoints = np.unique(breakpoints)

        # Compter les proportions par bin
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # Proportions avec lissage epsilon pour éviter log(0)
        eps = 1e-6
        expected_pct = expected_counts / len(expected) + eps
        actual_pct = actual_counts / len(actual) + eps

        # Formule PSI : Σ (actual% - expected%) × ln(actual% / expected%)
        psi_value = np.sum(
            (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        )
        return float(psi_value)

    @staticmethod
    def roc_curve_data(
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retourne les données de la courbe ROC pour visualisation.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.

        Returns:
            Tuple (fpr, tpr, thresholds).
        """
        return roc_curve(y_true, y_score)

    @staticmethod
    def cap_curve_data(
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne les données de la courbe CAP (Cumulative Accuracy Profile).

        Utilisée pour visualiser le pouvoir de classement du modèle.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.

        Returns:
            Tuple (proportion_population, proportion_defaults_captured).
        """
        n = len(y_true)
        # Trier par score décroissant
        sorted_idx = np.argsort(-y_score)
        sorted_labels = np.array(y_true)[sorted_idx]

        # Cumuler les défauts capturés
        cum_defaults = np.cumsum(sorted_labels)
        total_defaults = cum_defaults[-1]

        prop_pop = np.arange(1, n + 1) / n
        prop_defaults = cum_defaults / total_defaults

        return prop_pop, prop_defaults

    @staticmethod
    def ks_curve_data(
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Retourne les données pour le graphique KS.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.
            n_points: Nombre de points pour les courbes.

        Returns:
            Tuple (thresholds, cdf_default, cdf_non_default, ks_value).
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        thresholds = np.linspace(0, 1, n_points)

        defaults = y_score[y_true == 1]
        non_defaults = y_score[y_true == 0]

        cdf_default = np.array([np.mean(defaults <= t) for t in thresholds])
        cdf_non_default = np.array([np.mean(non_defaults <= t) for t in thresholds])

        ks_value = float(np.max(np.abs(cdf_default - cdf_non_default)))

        return thresholds, cdf_default, cdf_non_default, ks_value

    @staticmethod
    def compute_all(
        y_true: np.ndarray,
        y_score: np.ndarray,
        y_score_ref: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calcule toutes les métriques en une seule passe.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.
            y_score_ref: Scores de référence pour le PSI (si None, PSI = 0).

        Returns:
            Dictionnaire avec les clés 'auc', 'gini', 'ks', 'psi'.
        """
        auc_val = ModelMetrics.auc(y_true, y_score)
        psi_val = 0.0
        if y_score_ref is not None:
            psi_val = ModelMetrics.psi(y_score_ref, y_score)

        return {
            "auc": round(auc_val, 4),
            "gini": round(2.0 * auc_val - 1.0, 4),
            "ks": round(ModelMetrics.ks_statistic(y_true, y_score), 4),
            "psi": round(psi_val, 4),
        }

    @staticmethod
    def classification_table(
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Construit une table de classement par décile de score.

        Utile pour valider la monotonie du modèle : le taux de défaut
        doit croître avec le score prédit.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.
            n_bins: Nombre de déciles.

        Returns:
            DataFrame avec colonnes : bin, count, n_defaults, default_rate,
            avg_score, cumulative_default_rate.
        """
        df = pd.DataFrame({"score": y_score, "default": y_true})
        df["bin"] = pd.qcut(df["score"], q=n_bins, duplicates="drop")

        table = (
            df.groupby("bin", observed=True)
            .agg(
                count=("default", "size"),
                n_defaults=("default", "sum"),
                avg_score=("score", "mean"),
            )
            .reset_index()
        )
        table["default_rate"] = table["n_defaults"] / table["count"]
        table["cumulative_defaults"] = table["n_defaults"].cumsum()
        table["cumulative_default_rate"] = (
            table["cumulative_defaults"] / table["n_defaults"].sum()
        )
        return table
