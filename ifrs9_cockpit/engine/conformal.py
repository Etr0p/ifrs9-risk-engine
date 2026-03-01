"""Conformal Prediction pour IFRS 9 — intervalles de couverture garantis.

Implémente le Split Conformal Prediction (Vovk et al. 2005, Lei et al. 2018)
pour envelopper les prédictions PD/ECL d'intervalles statistiquement calibrés.

Principe :
    1. Calibration offline : sur un jeu de calibration (hold-out),
       calculer les résidus (nonconformity scores) = |PD_pred - PD_réalisé|.
    2. Inférence online : pour un nouveau client, l'intervalle est
       [PD_pred - q, PD_pred + q] où q = quantile(1-α)(1+1/n) des résidus.

Garantie théorique :
    P(Y ∈ Ĉ(X)) ≥ 1 - α  (couverture marginale exacte)

Distribution-free, model-agnostic, aucune hypothèse gaussienne.

Références :
    - Vovk, Gammerman, Shafer (2005): Algorithmic Learning in a Random World
    - Lei et al. (2018): Distribution-Free Predictive Inference For Regression
    - Romano, Patterson, Candès (2019): Conformalized Quantile Regression
    - Angelopoulos & Bates (2023): Conformal Prediction: A Gentle Introduction
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ConformalResult:
    """Résultat de la prédiction conforme.

    Attributes:
        pd_lower: Borne inférieure de l'intervalle PD (1-α coverage).
        pd_upper: Borne supérieure de l'intervalle PD.
        pd_point: PD point estimate (inchangée).
        ecl_lower: ECL calculée avec pd_lower.
        ecl_upper: ECL calculée avec pd_upper.
        ecl_point: ECL point estimate.
        coverage_level: Niveau de couverture nominal (ex: 0.90).
        quantile_residual: Quantile des résidus utilisé (q_hat).
        n_calibration: Taille du jeu de calibration.
    """
    pd_lower: np.ndarray
    pd_upper: np.ndarray
    pd_point: np.ndarray
    ecl_lower: np.ndarray
    ecl_upper: np.ndarray
    ecl_point: np.ndarray
    coverage_level: float
    quantile_residual: float
    n_calibration: int


class ConformalPredictor:
    """Split Conformal Prediction pour PD et ECL.

    Pipeline :
        1. calibrate(pd_pred_cal, y_cal) — calcule les nonconformity scores
        2. predict(pd_pred_new, lgd, ead, df) — retourne intervalles PD + ECL

    Le nonconformity score est |PD_pred - y_réel| (résidu absolu).
    Pour le staging IFRS 9, on utilise la borne supérieure de l'intervalle
    comme PD conservative (worst-case staging).

    Args:
        alpha: Niveau de miscouverture (0.10 pour 90% de couverture).
        seed: Graine pour reproductibilité du split.
    """

    # Bornes PD physiques
    _PD_MIN: float = 1e-6
    _PD_MAX: float = 0.9999

    def __init__(self, alpha: float = 0.10, seed: int = 42) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha doit être dans ]0, 1[, reçu {alpha}")
        self.alpha = alpha
        self.seed = seed
        self._q_hat: Optional[float] = None
        self._n_cal: int = 0
        self._residuals: Optional[np.ndarray] = None

    @property
    def is_calibrated(self) -> bool:
        """True si le prédicteur a été calibré."""
        return self._q_hat is not None

    @property
    def quantile_residual(self) -> float:
        """Quantile des résidus (q_hat). Lève si non calibré."""
        if self._q_hat is None:
            raise RuntimeError("ConformalPredictor non calibré. Appeler calibrate() d'abord.")
        return self._q_hat

    def calibrate(
        self,
        pd_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Calibre le prédicteur conforme sur un jeu de calibration.

        Calcule les nonconformity scores (résidus absolus) et le quantile
        ajusté pour la couverture finie (correction de Vovk).

        Args:
            pd_pred: PD prédites sur le jeu de calibration.
            y_true: Labels réels (0/1 défaut) du jeu de calibration.

        Returns:
            q_hat : le quantile des résidus.

        Raises:
            ValueError: Si les arrays sont vides ou de tailles différentes.
        """
        pd_pred = np.asarray(pd_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        if len(pd_pred) != len(y_true):
            raise ValueError(
                f"Tailles incompatibles: pd_pred={len(pd_pred)}, y_true={len(y_true)}"
            )
        n = len(pd_pred)
        if n < 10:
            raise ValueError(f"Jeu de calibration trop petit: {n} < 10 minimum")

        # Nonconformity scores = résidu absolu
        self._residuals = np.abs(pd_pred - y_true)
        self._n_cal = n

        # Quantile ajusté pour couverture finie (Vovk 2005)
        # q_hat = quantile de niveau ceil((n+1)(1-alpha)) / n
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)  # Sécurité si n très petit
        self._q_hat = float(np.quantile(self._residuals, level))

        return self._q_hat

    def predict(
        self,
        pd_point: np.ndarray,
        lgd: Optional[np.ndarray] = None,
        ead: Optional[np.ndarray] = None,
        discount_factor: Optional[np.ndarray] = None,
    ) -> ConformalResult:
        """Produit les intervalles conformes pour PD et optionnellement ECL.

        Args:
            pd_point: PD point estimates (modèle actif).
            lgd: LGD par position (pour calculer ECL bands).
            ead: EAD par position.
            discount_factor: Facteur d'actualisation par position.

        Returns:
            ConformalResult avec intervalles PD et ECL.

        Raises:
            RuntimeError: Si non calibré.
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor non calibre.")

        pd_point = np.asarray(pd_point, dtype=float)
        q = self._q_hat

        # Intervalles PD
        pd_lower = np.clip(pd_point - q, self._PD_MIN, self._PD_MAX)
        pd_upper = np.clip(pd_point + q, self._PD_MIN, self._PD_MAX)

        # Intervalles ECL (si composants fournis)
        has_ecl = lgd is not None and ead is not None
        if has_ecl:
            lgd = np.asarray(lgd, dtype=float)
            ead = np.asarray(ead, dtype=float)
            df_arr = np.asarray(discount_factor, dtype=float) if discount_factor is not None else np.ones_like(lgd)

            ecl_point = pd_point * lgd * ead * df_arr
            ecl_lower = pd_lower * lgd * ead * df_arr
            ecl_upper = pd_upper * lgd * ead * df_arr
        else:
            ecl_point = np.zeros_like(pd_point)
            ecl_lower = np.zeros_like(pd_point)
            ecl_upper = np.zeros_like(pd_point)

        return ConformalResult(
            pd_lower=pd_lower,
            pd_upper=pd_upper,
            pd_point=pd_point,
            ecl_lower=ecl_lower,
            ecl_upper=ecl_upper,
            ecl_point=ecl_point,
            coverage_level=1 - self.alpha,
            quantile_residual=self._q_hat,
            n_calibration=self._n_cal,
        )

    def empirical_coverage(
        self,
        pd_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Calcule la couverture empirique sur un jeu de test.

        Args:
            pd_pred: PD prédites.
            y_true: Labels réels.

        Returns:
            Couverture empirique (fraction de y_true dans les intervalles).
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor non calibre.")

        result = self.predict(pd_pred)
        covered = (y_true >= result.pd_lower) & (y_true <= result.pd_upper)
        return float(np.mean(covered))

    def get_diagnostics(self) -> Dict[str, float]:
        """Diagnostics de calibration.

        Returns:
            Dict avec statistiques des résidus et paramètres.
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor non calibre.")

        return {
            "alpha": self.alpha,
            "coverage_target": 1 - self.alpha,
            "q_hat": self._q_hat,
            "n_calibration": self._n_cal,
            "residual_mean": float(np.mean(self._residuals)),
            "residual_median": float(np.median(self._residuals)),
            "residual_std": float(np.std(self._residuals)),
            "residual_max": float(np.max(self._residuals)),
            "residual_p95": float(np.quantile(self._residuals, 0.95)),
        }
