"""Métriques de performance pour les modèles de risque de crédit.

Implémente les 4 métriques obligatoires IFRS 9 :
    - AUC-ROC (Area Under Receiver Operating Characteristic)
    - Gini (= 2×AUC - 1)
    - KS (Kolmogorov-Smirnov)
    - PSI (Population Stability Index)
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import (
    roc_auc_score, roc_curve, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score, accuracy_score,
)
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.config import PD_CONFIG


class ModelMetrics:
    """Calcul centralisé des métriques de discrimination et de stabilité.

    Fournit des méthodes statiques réutilisables par tous les modèles
    PD/LGD pour le monitoring.

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
    def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calcule le Brier Score (calibration quality).

        Brier = mean((y_score - y_true)^2).
        0 = parfait, 1 = pire. Pour le radar on utilise 1 - Brier.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.

        Returns:
            Brier Score entre 0 et 1 (lower is better).
        """
        return float(brier_score_loss(y_true, y_score))

    @staticmethod
    def logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Calcule la Log Loss (cross-entropy).

        Mesure la qualite probabiliste du modele.
        0 = parfait, +inf = pire. Typiquement < 1.0 pour un bon modele.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.

        Returns:
            Log Loss (lower is better).
        """
        y_clipped = np.clip(y_score, 1e-15, 1 - 1e-15)
        return float(log_loss(y_true, y_clipped))

    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calcule Accuracy, Precision, Recall, F1 a un seuil donne.

        Utilise le seuil optimal (maximisant F1) si threshold=None.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.
            threshold: Seuil de classification.

        Returns:
            Dict avec 'accuracy', 'precision', 'recall', 'f1'.
        """
        y_pred = (y_score >= threshold).astype(int)
        # Guard against edge cases (no positive predictions)
        if y_pred.sum() == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    @staticmethod
    def _optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Trouve le seuil optimal maximisant le Youden's J statistic.

        J = Sensitivity + Specificity - 1 = TPR - FPR.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.

        Returns:
            Seuil optimal.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])

    @staticmethod
    def compute_all(
        y_true: np.ndarray,
        y_score: np.ndarray,
        y_score_ref: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calcule toutes les métriques en une seule passe.

        Retourne 9 metriques : auc, gini, ks, psi (legacy) +
        brier, logloss, precision, recall, f1 (nouvelles).

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.
            y_score_ref: Scores de référence pour le PSI (si None, PSI = 0).

        Returns:
            Dictionnaire avec 9 cles.
        """
        auc_val = ModelMetrics.auc(y_true, y_score)
        psi_val = 0.0
        if y_score_ref is not None:
            psi_val = ModelMetrics.psi(y_score_ref, y_score)

        # Seuil optimal (Youden's J) pour classification metrics
        threshold = ModelMetrics._optimal_threshold(y_true, y_score)
        cls_metrics = ModelMetrics.classification_metrics(y_true, y_score, threshold)

        return {
            "auc": round(auc_val, 4),
            "gini": round(2.0 * auc_val - 1.0, 4),
            "ks": round(ModelMetrics.ks_statistic(y_true, y_score), 4),
            "psi": round(psi_val, 4),
            "brier": round(ModelMetrics.brier(y_true, y_score), 4),
            "logloss": round(ModelMetrics.logloss(y_true, y_score), 4),
            "precision": round(cls_metrics["precision"], 4),
            "recall": round(cls_metrics["recall"], 4),
            "f1": round(cls_metrics["f1"], 4),
        }

    @staticmethod
    def compute_backtesting_metrics(
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_folds: int = 6,
    ) -> pl.DataFrame:
        """Calcule des métriques de backtesting par walk-forward.

        Simule une validation temporelle en découpant les données en
        n_folds séquentiels et en calculant les métriques sur chaque fold.

        Args:
            y_true: Labels binaires (0/1).
            y_score: Probabilités prédites.
            n_folds: Nombre de folds temporels.

        Returns:
            DataFrame avec colonnes 'month', 'auc', 'gini', 'ks', 'psi'.
        """
        n = len(y_true)
        fold_size = n // n_folds
        records = []

        for i in range(n_folds):
            start = i * fold_size
            end = min((i + 1) * fold_size, n)
            y_t = y_true[start:end]
            y_s = y_score[start:end]

            if len(np.unique(y_t)) < 2:
                continue

            auc_val = ModelMetrics.auc(y_t, y_s)
            records.append({
                "month": i + 1,
                "auc": round(auc_val, 4),
                "gini": round(2.0 * auc_val - 1.0, 4),
                "ks": round(ModelMetrics.ks_statistic(y_t, y_s), 4),
                "psi": round(
                    ModelMetrics.psi(y_score[:fold_size], y_s) if i > 0 else 0.0,
                    4,
                ),
            })

        return pl.DataFrame(records)

    @staticmethod
    def hhi(shares: np.ndarray) -> float:
        """Calcule l'indice de Herfindahl-Hirschman (HHI).

        Mesure la concentration du portefeuille. Standard Bâle III / IFRS 7.
        HHI < 0.15 → diversifié, 0.15-0.25 → modéré, > 0.25 → concentré.

        Args:
            shares: Parts de marché (doivent sommer à ~1).

        Returns:
            Valeur HHI entre 0 et 1.
        """
        shares = np.asarray(shares, dtype=float)
        total = shares.sum()
        if total == 0:
            return 0.0
        proportions = shares / total
        return float(np.sum(proportions ** 2))

    @staticmethod
    def classification_table(
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_bins: int = 10,
    ) -> pl.DataFrame:
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
        import pandas as pd  # local import for qcut (no Polars equivalent)

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
        # Convert Interval bin column to string for Polars compatibility
        table["bin"] = table["bin"].astype(str)
        return pl.from_pandas(table)
