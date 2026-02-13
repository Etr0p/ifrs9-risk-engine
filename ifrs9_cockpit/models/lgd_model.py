"""Modele Loss Given Default (LGD) pour le Cockpit IFRS 9.

Implemente une estimation de LGD basee sur la distribution Beta,
avec distinction entre :
    - LGD TTC (Through-The-Cycle) : estimation moyenne long terme
    - LGD Downturn : estimation stressee pour scenarios adverses

La LGD est sensible au HPI via le canal collateral (FR8) :
baisse des prix immobiliers -> hausse LTV -> hausse LGD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional

from ifrs9_cockpit.config import LGD_CONFIG, RANDOM_SEED, SECTORS, SectorConfig


class LGDModel:
    """Modele LGD avec distribution Beta et ajustement Downturn.

    Le modele estime la LGD a partir des caracteristiques du pret
    (collateral implicite via loan_type, seniority via credit_score)
    et applique un add-on pour le scenario downturn.

    Attributes:
        seed: Graine aleatoire.
        rng: Generateur numpy.
        sector_lgd_: LGD moyennes calibrees par secteur apres fit.
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        """Initialise le modele LGD.

        Args:
            seed: Graine pour reproductibilite.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sector_lgd_: Dict[str, float] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> LGDModel:
        """Calibre les parametres LGD sur les defauts observes.

        Args:
            df: DataFrame credit avec colonnes sector, loan_type,
                credit_score, utilization_rate, default_flag.

        Returns:
            Self (pattern fluent).
        """
        defaults = df[df["default_flag"] == 1].copy()

        if len(defaults) == 0:
            for sector in SECTORS:
                self.sector_lgd_[sector.name] = LGD_CONFIG.lgd_ttc_mean
            self._fitted = True
            return self

        # Simuler les LGD realisees pour les defauts observes
        defaults["lgd_realized"] = self._simulate_realized_lgd(defaults)

        # Calibrer les moyennes par secteur
        for sector in SECTORS:
            mask = defaults["sector"] == sector.name
            if mask.sum() > 0:
                self.sector_lgd_[sector.name] = float(
                    defaults.loc[mask, "lgd_realized"].mean()
                )
            else:
                self.sector_lgd_[sector.name] = LGD_CONFIG.lgd_ttc_mean

        self._fitted = True
        return self

    # Constantes de calibration credit score (M1)
    _MEDIAN_SCORE: float = 650.0
    _SCORE_STD: float = 100.0
    _LGD_CREDIT_SCORE_SCALE: float = 0.20

    # Cap LGD a 95% (plafond logique : 5% de recouvrement minimal)
    _LGD_CAP: float = 0.95

    # Ajustement loan_type sur la LGD de base
    _REVOLVING_MULTIPLIER: float = 1.15
    _TERM_MULTIPLIER: float = 0.90

    def _compute_base_lgd(self, df: pd.DataFrame) -> np.ndarray:
        """Calcule la LGD de base (avant dispersion Beta).

        Ajustement credit score calibre (M1) :
            credit_adj = clip((median - score) / (2 * std), -0.10, 0.10)
                         * LGD_CREDIT_SCORE_SCALE

        Args:
            df: DataFrame avec colonnes sector, loan_type, credit_score.

        Returns:
            Array de LGD moyennes deterministes.
        """
        n = len(df)
        lgd = np.full(n, LGD_CONFIG.lgd_ttc_mean)

        # Ajustement par secteur
        for sector_name, sector_lgd in self.sector_lgd_.items():
            mask = df["sector"].values == sector_name
            lgd[mask] = sector_lgd

        # Ajustement par type de pret
        revolving_mask = df["loan_type"].values == "Revolving"
        lgd[revolving_mask] *= self._REVOLVING_MULTIPLIER
        lgd[~revolving_mask] *= self._TERM_MULTIPLIER

        # Ajustement par score de credit — calibre (M1)
        credit_scores = df["credit_score"].values.astype(float)
        credit_adj = (
            np.clip(
                (self._MEDIAN_SCORE - credit_scores) / (2 * self._SCORE_STD),
                -0.10,
                0.10,
            )
            * self._LGD_CREDIT_SCORE_SCALE
        )
        lgd += credit_adj

        return lgd

    def predict_ttc_and_downturn(
        self,
        df: pd.DataFrame,
        z_stress: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calcule LGD TTC et Downturn a partir du meme tirage Beta.

        LGD Downturn via correlation cycle (H3, EBA GL/2019/03) :
            LGD_DT = LGD_TTC * (1 + rho_lgd_cycle * |Z_stress|)

        Ou rho_lgd_cycle est specifique au secteur (SectorConfig).
        Z_stress represente le nombre de sigma du stress :
            - 0 pour le scenario de base
            - 2.0 pour adverse (defaut)
            - -1.0 pour favorable

        Garantit l'invariant LGD Downturn >= LGD TTC pour z_stress >= 0.

        Args:
            df: DataFrame avec colonnes sector, loan_type, credit_score.
            z_stress: Nombre de sigma du stress (0=base, 2=adverse, -1=favorable).

        Returns:
            Tuple (lgd_ttc, lgd_downturn).
        """
        base_lgd = self._compute_base_lgd(df)

        # Dispersion Beta (un seul tirage)
        lgd_ttc = self._apply_beta_dispersion(base_lgd)
        lgd_ttc = np.clip(lgd_ttc, LGD_CONFIG.recovery_rate_floor, self._LGD_CAP)

        # Downturn via correlation cycle (H3)
        # LGD_DT = LGD_TTC * (1 + rho_lgd_cycle * |Z_stress|)
        # Lookup rho_lgd_cycle par secteur depuis SectorConfig
        sector_map = {s.name: s.rho_lgd_cycle for s in SECTORS}
        rho = np.array([
            sector_map.get(s, 0.20) for s in df["sector"].values
        ])
        lgd_downturn = lgd_ttc * (1 + rho * abs(z_stress))

        lgd_downturn = np.clip(lgd_downturn, LGD_CONFIG.recovery_rate_floor, self._LGD_CAP)

        return lgd_ttc, lgd_downturn

    def predict_ttc(self, df: pd.DataFrame) -> np.ndarray:
        """Predit la LGD Through-The-Cycle pour chaque entreprise.

        Args:
            df: DataFrame avec colonnes sector, loan_type, credit_score.

        Returns:
            Array de LGD TTC entre 0 et 1.
        """
        lgd_ttc, _ = self.predict_ttc_and_downturn(df)
        return lgd_ttc

    def predict_downturn(self, df: pd.DataFrame) -> np.ndarray:
        """Predit la LGD Downturn (scenario stresse).

        Args:
            df: DataFrame avec colonnes sector, loan_type, credit_score.

        Returns:
            Array de LGD Downturn entre 0 et 1.
        """
        _, lgd_dt = self.predict_ttc_and_downturn(df)
        return lgd_dt

    def predict(
        self,
        df: pd.DataFrame,
        downturn: bool = False,
        hpi_override: Optional[float] = None,
    ) -> np.ndarray:
        """Interface unifiee de prediction LGD.

        Le HPI impacte la LGD via la valeur du collateral (FR8) :
        baisse des prix immobiliers -> hausse LTV -> hausse LGD.

        Args:
            df: DataFrame credit.
            downturn: Si True, retourne la LGD Downturn.
            hpi_override: Variation des prix immobiliers (%).

        Returns:
            Array de LGD.
        """
        lgd_ttc, lgd_dt = self.predict_ttc_and_downturn(df)
        lgd = lgd_dt if downturn else lgd_ttc

        # Ajustement HPI : baisse des prix -> hausse LGD, hausse -> baisse LGD
        # Bidirectionnel : les scenarios favorables (HPI > 2.0) reduisent la LGD
        if hpi_override is not None:
            hpi_impact = (2.0 - hpi_override) / 100
            for sector in SECTORS:
                mask = df["sector"].values == sector.name
                lgd[mask] += hpi_impact * sector.hpi_sensitivity_credit
            lgd = np.clip(lgd, LGD_CONFIG.recovery_rate_floor, self._LGD_CAP)

        return lgd

    def get_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resume des LGD par secteur et type de pret.

        Args:
            df: DataFrame credit.

        Returns:
            DataFrame recapitulatif avec LGD TTC et Downturn moyennes.
        """
        lgd_ttc, lgd_dt = self.predict_ttc_and_downturn(df)

        summary_df = df[["sector", "loan_type"]].copy()
        summary_df["lgd_ttc"] = lgd_ttc
        summary_df["lgd_downturn"] = lgd_dt

        return (
            summary_df.groupby(["sector", "loan_type"])
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
        """Simule des LGD realisees pour calibrer le modele.

        Args:
            df: DataFrame des entreprises en defaut.

        Returns:
            Array de LGD realisees simulees.
        """
        n = len(df)
        mu = np.full(n, LGD_CONFIG.lgd_ttc_mean)

        # Revolving -> LGD plus haute
        revolving = df["loan_type"].values == "Revolving"
        mu[revolving] += 0.08

        # Score bas -> recouvrement plus difficile
        scores = df["credit_score"].values.astype(float)
        mu += np.clip((self._MEDIAN_SCORE - scores) / (2 * self._SCORE_STD * 10), -0.05, 0.10)

        # Utilization haute -> perte plus importante
        if "utilization_rate" in df.columns:
            util = df["utilization_rate"].values.astype(float)
            mu += (util - 0.5) * 0.08

        mu = np.clip(mu, 0.05, 0.90)
        sigma = LGD_CONFIG.lgd_ttc_std

        return self._sample_beta(mu, sigma)

    def _apply_beta_dispersion(self, mu: np.ndarray) -> np.ndarray:
        """Applique une dispersion Beta autour des moyennes.

        Le RNG est reinitialise avant chaque appel pour garantir que
        le meme portefeuille recoit les memes LGD de base, independamment
        du nombre d'appels precedents (determinisme cross-scenario).

        Args:
            mu: Moyennes de LGD.

        Returns:
            LGD avec dispersion aleatoire (deterministe pour un meme portefeuille).
        """
        # Reinitialiser le RNG pour determinisme cross-scenario
        self.rng = np.random.default_rng(self.seed + 1)
        sigma = LGD_CONFIG.lgd_ttc_std * 0.5
        return self._sample_beta(mu, sigma)

    def _sample_beta(self, mu: np.ndarray, sigma: float) -> np.ndarray:
        """Echantillonne depuis une distribution Beta parametree (vectorise).

        Utilise la methode des moments pour convertir (mu, sigma)
        en parametres (alpha, beta) de la distribution Beta.

        Args:
            mu: Moyennes (entre 0 et 1).
            sigma: Ecart-type cible.

        Returns:
            Echantillons Beta.
        """
        mu = np.clip(mu, 0.01, 0.99)
        variance = sigma ** 2

        # Methode des moments : s'assurer que variance < mu*(1-mu)
        max_var = mu * (1 - mu) * 0.95
        variance = np.minimum(variance, max_var)

        kappa = (mu * (1 - mu) / variance) - 1
        kappa = np.maximum(kappa, 1.05)  # Statistically motivated: var < mu(1-mu) requires kappa > 1

        alpha = np.maximum(mu * kappa, 0.1)
        beta = np.maximum((1 - mu) * kappa, 0.1)

        # Generation vectorisee (Beta accepte des arrays)
        samples = self.rng.beta(alpha, beta)

        return np.clip(samples, 0.01, 0.99)


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset

    print("=" * 60)
    print("IFRS 9 COCKPIT — LGD Model Validation")
    print("=" * 60)

    # 1. Data
    print("\n[1/3] Generation des donnees...")
    df_credit, _, _ = generate_dataset()
    print(f"       {len(df_credit):,} entreprises | DR = {df_credit['default_flag'].mean():.2%}")

    # 2. Fit
    print("\n[2/3] Calibration du modele LGD...")
    model = LGDModel()
    model.fit(df_credit)
    print(f"       LGD moyennes par secteur : {model.sector_lgd_}")

    # 3. Predict
    print("\n[3/3] Prediction LGD TTC et Downturn...")
    for z_label, z_val in [("Base (z=0)", 0.0), ("Adverse (z=2)", 2.0), ("Favorable (z=-1)", -1.0)]:
        lgd_ttc, lgd_dt = model.predict_ttc_and_downturn(df_credit, z_stress=z_val)
        print(f"\n  Scenario {z_label}:")
        print(f"    LGD TTC    : mean={lgd_ttc.mean():.4f}, std={lgd_ttc.std():.4f}")
        print(f"    LGD DT     : mean={lgd_dt.mean():.4f}, std={lgd_dt.std():.4f}")
        print(f"    DT >= TTC  : {(lgd_dt >= lgd_ttc - 1e-10).all()}")

    # Summary
    print("\n--- Resume par secteur/type ---")
    summary = model.get_summary(df_credit)
    print(summary.to_string(index=False))

    print("\nValidation LGD terminee.")
