"""Modèle Loss Given Default (LGD) pour le Cockpit IFRS 9.

Implémente une estimation de LGD basée sur la distribution Beta,
avec distinction entre :
    - LGD TTC (Through-The-Cycle) : estimation moyenne long terme
    - LGD Downturn : estimation stressée pour scénarios adverses

La distribution Beta est naturellement bornée [0, 1], ce qui en fait
le choix standard pour modéliser les taux de perte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from typing import Dict, Optional, Tuple

from ifrs9_cockpit.config import LGD_CONFIG, RANDOM_SEED, SEGMENTS


class LGDModel:
    """Modèle LGD avec distribution Beta et ajustement Downturn.

    Le modèle estime la LGD à partir des caractéristiques du prêt
    (collatéral implicite via loan_type, seniority via credit_score)
    et applique un add-on pour le scénario downturn.

    La distribution Beta est paramétrée via la méthode des moments :
        alpha = mu × ((mu × (1 - mu) / sigma² ) - 1)
        beta  = (1 - mu) × ((mu × (1 - mu) / sigma²) - 1)

    Attributes:
        seed: Graine aléatoire.
        rng: Générateur numpy.
        segment_lgd_: LGD moyennes calibrées par segment après fit.
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        """Initialise le modèle LGD.

        Args:
            seed: Graine pour reproductibilité.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.segment_lgd_: Dict[str, float] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "LGDModel":
        """Calibre les paramètres LGD sur les défauts observés.

        Simule des LGD réalisées à partir du modèle latent puis
        calibre les moyennes par segment et type de prêt.

        Args:
            df: DataFrame clients avec colonnes segment, loan_type,
                credit_score, utilization_rate, default_flag.

        Returns:
            Self (pattern fluent).
        """
        defaults = df[df["default_flag"] == 1].copy()

        if len(defaults) == 0:
            # Fallback sur les moyennes config si pas de défauts
            for seg in SEGMENTS:
                self.segment_lgd_[seg.name] = LGD_CONFIG.lgd_ttc_mean
            self._fitted = True
            return self

        # Simuler les LGD réalisées pour les défauts observés
        defaults = defaults.copy()
        defaults["lgd_realized"] = self._simulate_realized_lgd(defaults)

        # Calibrer les moyennes par segment
        for seg in SEGMENTS:
            mask = defaults["segment"] == seg.name
            if mask.sum() > 0:
                self.segment_lgd_[seg.name] = float(defaults.loc[mask, "lgd_realized"].mean())
            else:
                self.segment_lgd_[seg.name] = LGD_CONFIG.lgd_ttc_mean

        self._fitted = True
        return self

    def predict_ttc(self, df: pd.DataFrame) -> np.ndarray:
        """Prédit la LGD Through-The-Cycle pour chaque client.

        La LGD TTC est la perte moyenne attendue sur un cycle
        économique complet. Elle dépend de :
            - Le segment (proxy pour le profil de risque)
            - Le type de prêt (revolving = LGD plus élevée)
            - Le score de crédit (proxy pour la capacité de recouvrement)

        Args:
            df: DataFrame avec colonnes segment, loan_type, credit_score.

        Returns:
            Array de LGD TTC entre 0 et 1.
        """
        n = len(df)
        lgd = np.full(n, LGD_CONFIG.lgd_ttc_mean)

        # Ajustement par segment
        for seg_name, seg_lgd in self.segment_lgd_.items():
            mask = df["segment"].values == seg_name
            lgd[mask] = seg_lgd

        # Ajustement par type de prêt
        # Revolving : LGD plus élevée (pas de collatéral)
        # Term : LGD plus basse (souvent adossé à un actif)
        revolving_mask = df["loan_type"].values == "Revolving"
        lgd[revolving_mask] *= 1.15
        lgd[~revolving_mask] *= 0.90

        # Ajustement par score de crédit (capacité de recouvrement)
        credit_scores = df["credit_score"].values.astype(float)
        credit_adj = np.clip((650 - credit_scores) / 1000, -0.10, 0.10)
        lgd += credit_adj

        # Dispersion Beta autour de la moyenne
        lgd = self._apply_beta_dispersion(lgd)

        return np.clip(lgd, LGD_CONFIG.recovery_rate_floor, 0.95)

    def predict_downturn(self, df: pd.DataFrame) -> np.ndarray:
        """Prédit la LGD Downturn (scénario stressé).

        LGD Downturn = LGD TTC + add-on réglementaire.
        Reflète les conditions de recouvrement dégradées en récession.

        Args:
            df: DataFrame avec colonnes segment, loan_type, credit_score.

        Returns:
            Array de LGD Downturn entre 0 et 1.
        """
        lgd_ttc = self.predict_ttc(df)
        lgd_downturn = lgd_ttc + LGD_CONFIG.downturn_add_on

        # Les segments fragiles subissent un stress additionnel
        for seg in SEGMENTS:
            if seg.unemployment_sensitivity > 1.5:
                mask = df["segment"].values == seg.name
                lgd_downturn[mask] += 0.05  # +5pp additionnel

        return np.clip(lgd_downturn, LGD_CONFIG.recovery_rate_floor, 0.95)

    def predict(
        self,
        df: pd.DataFrame,
        downturn: bool = False,
    ) -> np.ndarray:
        """Interface unifiée de prédiction LGD.

        Args:
            df: DataFrame clients.
            downturn: Si True, retourne la LGD Downturn.

        Returns:
            Array de LGD.
        """
        if downturn:
            return self.predict_downturn(df)
        return self.predict_ttc(df)

    def get_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Résumé des LGD par segment et type de prêt.

        Args:
            df: DataFrame clients.

        Returns:
            DataFrame récapitulatif avec LGD TTC et Downturn moyennes.
        """
        lgd_ttc = self.predict_ttc(df)
        lgd_dt = self.predict_downturn(df)

        summary_df = df[["segment", "loan_type"]].copy()
        summary_df["lgd_ttc"] = lgd_ttc
        summary_df["lgd_downturn"] = lgd_dt

        return (
            summary_df.groupby(["segment", "loan_type"])
            .agg(
                count=("lgd_ttc", "size"),
                lgd_ttc_mean=("lgd_ttc", "mean"),
                lgd_ttc_std=("lgd_ttc", "std"),
                lgd_downturn_mean=("lgd_downturn", "mean"),
            )
            .round(4)
            .reset_index()
        )

    def _simulate_realized_lgd(self, df: pd.DataFrame) -> np.ndarray:
        """Simule des LGD réalisées pour calibrer le modèle.

        Utilise une distribution Beta paramétrée par les
        caractéristiques du prêt.

        Args:
            df: DataFrame des clients en défaut.

        Returns:
            Array de LGD réalisées simulées.
        """
        n = len(df)
        mu = np.full(n, LGD_CONFIG.lgd_ttc_mean)

        # Revolving → LGD plus haute
        revolving = df["loan_type"].values == "Revolving"
        mu[revolving] += 0.08

        # Score bas → recouvrement plus difficile
        scores = df["credit_score"].values.astype(float)
        mu += np.clip((600 - scores) / 2000, -0.05, 0.10)

        # Utilization haute → perte plus importante
        if "utilization_rate" in df.columns:
            util = df["utilization_rate"].values.astype(float)
            mu += (util - 0.5) * 0.08

        mu = np.clip(mu, 0.05, 0.90)
        sigma = LGD_CONFIG.lgd_ttc_std

        return self._sample_beta(mu, sigma, n)

    def _apply_beta_dispersion(self, mu: np.ndarray) -> np.ndarray:
        """Applique une dispersion Beta autour des moyennes.

        Args:
            mu: Moyennes de LGD.

        Returns:
            LGD avec dispersion aléatoire.
        """
        sigma = LGD_CONFIG.lgd_ttc_std * 0.5  # Dispersion réduite pour prédiction
        return self._sample_beta(mu, sigma, len(mu))

    def _sample_beta(
        self,
        mu: np.ndarray,
        sigma: float,
        n: int,
    ) -> np.ndarray:
        """Échantillonne depuis une distribution Beta paramétrée.

        Utilise la méthode des moments pour convertir (mu, sigma)
        en paramètres (alpha, beta) de la distribution Beta.

        Args:
            mu: Moyennes (entre 0 et 1).
            sigma: Écart-type cible.
            n: Nombre d'échantillons.

        Returns:
            Échantillons Beta.
        """
        mu = np.clip(mu, 0.01, 0.99)
        variance = sigma ** 2

        # Méthode des moments : s'assurer que variance < mu*(1-mu)
        max_var = mu * (1 - mu) * 0.95
        variance = np.minimum(variance, max_var)

        kappa = (mu * (1 - mu) / variance) - 1
        kappa = np.maximum(kappa, 2.0)  # Stabilité numérique

        alpha = mu * kappa
        beta = (1 - mu) * kappa

        samples = np.array([
            self.rng.beta(max(a, 0.1), max(b, 0.1))
            for a, b in zip(alpha, beta)
        ])

        return np.clip(samples, 0.01, 0.99)
