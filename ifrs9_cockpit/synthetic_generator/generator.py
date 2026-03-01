"""
AdvancedFinancialGenerator — Synthetic Financial Data Generator v4.5.
"""

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import t as t_dist, norm, beta, lognorm, gamma
from scipy.special import expit
from scipy.optimize import brentq
import warnings

from ifrs9_cockpit.synthetic_generator.constants import (
    TARGET, LATENT_PD, NOISE_FEATURES,
    _MACRO, _DEFAULT_STRESS_QUARTERS,
    _SECTOR_WEIGHTS, _SECTOR_FACTOR_SCALES, _FACTOR_LOADINGS,
    _SECTOR_PROFILES, _SECTOR_MARGINAL_SCALES,
    _SECTOR_REVENUE_PARAMS, _SECTOR_DPD_PARAMS,
    _SECTOR_ZOMBIE_RATES, _MACRO_NOISE_SIGMA,
    _SECTOR_DGP_ADJUSTMENTS,
)

warnings.filterwarnings("ignore")


# ===========================================================================
# Generator
# ===========================================================================

class AdvancedFinancialGenerator:
    """
    Generateur de donnees synthetiques financieres pour entrainement Deep Learning.

    Architecture v4.5 :
        1. Assignation secteur (poids non-uniformes) + vintage
        2. t-Copula PAR SECTEUR (correlations differentes) x regime (normal/stress)
        3. 15 marginales continues via PIT (revenue par secteur)
        4. company_size (vectorise, stochastique)
        5. Profils sectoriels (shifts + dispersion + zombie rate quantile-warp)
        6. 3 features discretes (APRES profils, DPD par secteur)
        7. 3 macro (sigma reduit + amplitude reduite = visible dans DGP)
        8. 6 features bruit (5 explicites + nb_credit_lines)
        9. DGP non-stationnaire (signal redistribue, debt x vol reduit)
        10. MNAR (guard missing_rate=0, lognormal + MCAR 2%)

    Total : 27 features + vintage_quarter
    Limitation : pas de structure panel (IID, pas de suivi entreprise inter-trimestres)
    """

    SECTORS = list(_SECTOR_WEIGHTS.keys())
    SIZES = ["PME", "ETI", "GE"]

    # Ground truth DGP v4.5 : realiste 40/35/25
    # (monotones piecewise / amplification conditionnelle / conjonctifs)
    DGP_EFFECTS = {
        # === MONOTONES PIECEWISE (~40%) ===
        "debt_ratio":               "PIECEWISE 3 paliers (0.15/0.80)",
        "net_debt_to_ebitda":       "PIECEWISE 2 paliers (0.12/0.50)",
        "loan_to_value":            "PIECEWISE 2 paliers (0.08/0.30)",
        "ebitda_margin":            "PIECEWISE neg=danger(0.60) pos=protect(0.15)",
        "interest_coverage_ratio":  "PIECEWISE 2 paliers (0.50/0.12)",
        "current_ratio":            "PIECEWISE danger<1.0(0.25) bon>1.5(0.08)",
        "working_capital":          "PIECEWISE neg=danger (0.30)",
        "cf_volatility":            "PIECEWISE 2 paliers (0.12/0.35)",
        "utilization_rate":         "PIECEWISE >50%(0.12) >80%(0.30)",
        "revenue":                  "MONOTONE protective (-0.10 log)",
        "account_age":              "PIECEWISE saturante (0.15/0.05)",
        "nb_incidents":             "PIECEWISE 1er(0.12) puis +0.25",
        "days_past_due":            "PIECEWISE 30j(0.08) 60j+(0.25)",
        # === AMPLIFICATION CONDITIONNELLE (~35%) ===
        "debt x I(vol>0.22)":      "CONDITIONNEL levier en vol (0.50+1.20)",
        "margin x I(vol>0.22)":    "CONDITIONNEL marge en vol (0.80)",
        "debt x I(ICR<2.5)":       "CONDITIONNEL levier en ICR bas (1.00)",
        "util x I(ICR<2.5)":       "CONDITIONNEL util en ICR bas (0.60)",
        "vol x I(debt>0.45)":      "CONDITIONNEL vol chez endettes (0.80)",
        "inc x I(debt>0.45)":      "CONDITIONNEL incidents endettes (0.60)",
        "debt x I(margin<0.08)":   "CONDITIONNEL dette en marge faible (0.80)",
        "WC x I(margin<0.08)":     "CONDITIONNEL WC en marge faible (0.50)",
        "inc x I(CR<1.2)":         "CONDITIONNEL incidents illiquides (0.50)",
        "dpd x I(debt>0.45)":      "CONDITIONNEL DPD chez endettes (0.30)",
        # === CONJONCTIFS 2D/3D (~25%) ===
        "ICR<2 AND margin<0.05":   "CONJONCTIF spirale (1.00)",
        "debt>0.50 AND vol>0.25":  "CONJONCTIF levier instable (0.80)",
        "util>0.70 AND inc>=2":    "CONJONCTIF desperation (0.80)",
        "PME AND debt>0.40":       "CONJONCTIF PME fragile (0.60)",
        "age<3 AND inc>=1":        "CONJONCTIF jeune fragile (0.50)",
        "WC<0 AND debt>0.50":      "CONJONCTIF BFR non-finance (0.50)",
        "debt>0.45 AND vol>0.20 AND ICR<2.5": "3-WAY levier toxique (0.60)",
        "margin<0.05 AND util>0.60 AND inc>=1": "3-WAY spirale (0.50)",
        "debt>0.50 AND margin<0.05 AND CR<1.0": "3-WAY triple detresse (0.50)",
        "NDE > 4 (^2)":            "QUADRATIQUE surendettement (0.30)",
        # === CATEGORIELLES + MACRO ===
        "sector == Retail":         "categorielle (+0.15)",
        "sector == Sante":          "categorielle (-0.10)",
        "size == PME":              "categorielle (+0.12)",
        "gdp_growth":              "macro (-0.15)",
        "unemployment_rate":       "macro (+0.12)",
        "interest_rate_10y":       "macro (+0.08)",
        # === STRUCTURELS ===
        "sector x features":       "INTERACTIONS SECTORIELLES (5 secteurs, coefs forts)",
        "sector_thresholds":        "SEUILS SECTORIELS (debt/ICR/vol differents par secteur)",
        "age_maturity_noise":       "heterogeneite maturite (+0.15)",
        "stress_regime_shift":      "DGP NON-STATIONNAIRE (coefs x1.5 en stress)",
    }

    def __init__(
        self,
        n_rows: int = 50_000,
        degrees_of_freedom: int = 5,
        seed: int = 42,
        n_vintages: int = 24,
        base_default_rate: float = 0.05,
        noise_sigma: float = 1.5,
        stress_df: int = 3,
        stress_corr_amplification: float = 1.3,
        stress_quarters: set[int] | None = None,
        missing_rate: float = 0.10,
    ):
        """
        Args:
            n_rows: nombre d'observations
            degrees_of_freedom: df t-copula regime normal (5=modere, 3=lourd)
            seed: graine RNG
            n_vintages: nombre de trimestres (max 24)
            base_default_rate: taux de defaut cible (0.05 = 5%)
            noise_sigma: ecart-type bruit irreductible
                         (7.0 -> AUC ~0.82 avec coefs v4.5)
            stress_df: df t-copula regime stress
            stress_corr_amplification: amplification correlations en stress
            stress_quarters: indices trimestres stress (defaut: {8,9,10})
            missing_rate: taux MNAR
        """
        assert n_vintages <= 24, "Max 24 vintages dans le cycle macro"
        self.n_rows = n_rows
        self.df_normal = degrees_of_freedom
        self.df_stress = stress_df
        self.stress_amp = stress_corr_amplification
        self.stress_quarters = stress_quarters or _DEFAULT_STRESS_QUARTERS
        self.seed = seed
        self.n_vintages = n_vintages
        self.base_default_rate = base_default_rate
        self.noise_sigma = noise_sigma
        self.missing_rate = missing_rate
        self.rng = np.random.default_rng(seed)

    # -- Correlation (par secteur) -------------------------------------------

    def _build_sector_corr(
        self, factor_scales: list[float], stress: bool = False
    ) -> np.ndarray:
        """
        Matrice de correlation sectorielle via factor model.

        Chaque secteur a des poids differents sur les 4 facteurs latents,
        ce qui modifie la structure de correlation (pas juste la moyenne).
        PSD garanti par construction L @ L^T + Psi.
        """
        L = _FACTOR_LOADINGS * np.array(factor_scales)
        communalities = np.minimum(np.sum(L ** 2, axis=1), 0.95)
        psi = np.diag(1.0 - communalities)
        corr = L @ L.T + psi
        np.fill_diagonal(corr, 1.0)

        if stress:
            off_diag = corr - np.eye(corr.shape[0])
            off_diag = np.clip(off_diag * self.stress_amp, -0.99, 0.99)
            corr = np.eye(corr.shape[0]) + off_diag

        # Toujours projeter sur PSD (les scales sectoriels peuvent
        # pousser des communalites > 1, rendant la matrice non-PSD)
        corr = self._nearest_corr(corr)

        return corr

    @staticmethod
    def _nearest_corr(A: np.ndarray) -> np.ndarray:
        """Projection PSD la plus proche (eigenvalue clamping)."""
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-8)
        B = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(B))
        B = B / np.outer(d, d)
        np.fill_diagonal(B, 1.0)
        return (B + B.T) / 2

    # -- t-Copula -------------------------------------------------------------

    def _generate_t_copula(self, corr: np.ndarray, df: int, n: int) -> np.ndarray:
        """Genere n echantillons uniformes [0,1]^d via t-Copula."""
        dim = corr.shape[0]
        z = self.rng.multivariate_normal(np.zeros(dim), corr, size=n)
        w = self.rng.chisquare(df, size=n) / df
        x_student = z / np.sqrt(w)[:, None]
        return t_dist.cdf(x_student, df=df)

    # -- Marginales continues -------------------------------------------------

    def _apply_marginals(
        self, U: np.ndarray, sector_arr: np.ndarray
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """Transforme les uniformes copula en distributions financieres.

        v4.5: revenue parametrise par secteur (s, scale differents).
        """
        n = U.shape[0]

        # [v4.5 #40] Revenue par secteur
        revenue = lognorm.ppf(U[:, 0], s=0.85, scale=25_000_000)  # fallback
        for sector, params in _SECTOR_REVENUE_PARAMS.items():
            mask = sector_arr == sector
            if mask.any():
                revenue[mask] = lognorm.ppf(
                    U[mask, 0], s=params["s"], scale=params["scale"]
                )

        # Bimodale via inversion CDF mixture (20% zombies, 80% saines)
        mix_w = 0.20
        x_grid = np.linspace(-0.50, 0.50, 10_000)
        cdf_mixture = (
            mix_w * norm.cdf(x_grid, loc=-0.05, scale=0.10)
            + (1 - mix_w) * norm.cdf(x_grid, loc=0.15, scale=0.05)
        )
        ebitda_margin = np.interp(U[:, 1], cdf_mixture, x_grid)

        debt_ratio = gamma.ppf(U[:, 2], a=2.0, scale=0.3)
        cf_volatility = beta.ppf(U[:, 3], a=1.5, b=4.0)
        interest_coverage_ratio = lognorm.ppf(U[:, 4], s=0.7, scale=3.0)
        current_ratio = gamma.ppf(U[:, 5], a=5.0, scale=0.3)
        cash_ratio = beta.ppf(U[:, 6], a=2.0, b=5.0)
        loan_to_value = beta.ppf(U[:, 7], a=2.5, b=3.0)
        net_debt_to_ebitda = gamma.ppf(U[:, 8], a=2.0, scale=1.5)
        capex_to_revenue = beta.ppf(U[:, 9], a=1.5, b=8.0)
        working_capital_ratio = np.clip(
            norm.ppf(U[:, 10], loc=0.15, scale=0.10), -0.30, 0.60
        )
        tangible_assets_ratio = beta.ppf(U[:, 11], a=3.0, b=2.0)
        account_age_months = np.clip(
            stats.expon.ppf(U[:, 12], scale=60), 1, 360
        ).astype(int)
        utilization_rate = beta.ppf(U[:, 13], a=2.0, b=3.0)

        df = pl.DataFrame({
            "revenue": revenue,
            "ebitda_margin": ebitda_margin,
            "debt_ratio": debt_ratio,
            "cf_volatility": cf_volatility,
            "interest_coverage_ratio": interest_coverage_ratio,
            "current_ratio": current_ratio,
            "cash_ratio": cash_ratio,
            "loan_to_value": loan_to_value,
            "net_debt_to_ebitda": net_debt_to_ebitda,
            "capex_to_revenue": capex_to_revenue,
            "working_capital_ratio": working_capital_ratio,
            "tangible_assets_ratio": tangible_assets_ratio,
            "account_age_months": account_age_months,
            "utilization_rate": utilization_rate,
        })

        dpd_latent = U[:, 14]
        return df, dpd_latent

    # -- Features discretes ---------------------------------------------------

    def _add_discrete_features(self, df: pl.DataFrame, dpd_latent: np.ndarray) -> pl.DataFrame:
        """
        Features discretes. Appelees APRES _apply_sector_profiles
        pour que les incidents refletent les profils sectoriels.

        v4.5: DPD threshold et scale par secteur.
        """
        n = len(df)
        sector_arr = df["sector"].to_numpy()

        # [v4.5 #41] Days Past Due : zero-inflated PAR SECTEUR
        dpd_values = np.zeros(n)
        for sector, params in _SECTOR_DPD_PARAMS.items():
            mask = sector_arr == sector
            if not mask.any():
                continue
            sector_latent = dpd_latent[mask]
            has_dpd = sector_latent > params["threshold"]
            n_dpd = has_dpd.sum()
            if n_dpd > 0:
                vals = np.clip(
                    stats.expon.ppf(
                        self.rng.uniform(0, 1, n_dpd), scale=params["scale"]
                    ),
                    1, 365,
                ).astype(int)
                sector_dpd = np.zeros(mask.sum())
                sector_dpd[has_dpd] = vals
                dpd_values[mask] = sector_dpd
        # Fallback pour secteurs non-configures
        uncovered = ~np.isin(sector_arr, list(_SECTOR_DPD_PARAMS.keys()))
        if uncovered.any():
            has_dpd_unc = dpd_latent[uncovered] > 0.70
            n_dpd_unc = has_dpd_unc.sum()
            if n_dpd_unc > 0:
                unc_vals = np.clip(
                    stats.expon.ppf(self.rng.uniform(0, 1, n_dpd_unc), scale=30),
                    1, 365,
                ).astype(int)
                unc_dpd = np.zeros(uncovered.sum())
                unc_dpd[has_dpd_unc] = unc_vals
                dpd_values[uncovered] = unc_dpd

        # Incidents : multi-facteur + bruit multiplicatif log-normal
        margin_stress = np.clip(0.15 - df["ebitda_margin"].to_numpy(), 0, None) / 0.15
        lam = (
            0.3
            + 0.5 * df["debt_ratio"].to_numpy()
            + 0.3 * df["utilization_rate"].to_numpy()
            + 0.2 * margin_stress
        )
        lam *= self.rng.lognormal(0, 0.3, n)
        lam = np.clip(lam, 0.01, None)
        nb_incidents_12m = self.rng.poisson(lam)

        nb_credit_lines = self.rng.poisson(3.0, n)

        df = df.with_columns([
            pl.Series("days_past_due", dpd_values),
            pl.Series("nb_incidents_12m", nb_incidents_12m),
            pl.Series("nb_credit_lines", nb_credit_lines),
        ])
        return df

    # -- company_size (vectorise) [v4.2 #20] ----------------------------------

    def _add_company_size(self, df: pl.DataFrame) -> pl.DataFrame:
        """Taille correlee au revenu. Vectorise (pas de boucle Python).

        v4.5 #46: pentes reduites + bruit additif sur percentile
        pour casser la relation quasi-deterministe revenue->size.
        """
        n = len(df)
        # Polars rank: compute percentile rank manually
        rev_pct = df["revenue"].rank(method="ordinal").to_numpy().astype(float) / n
        # v4.5: bruit additif sur le percentile (casse le gradient deterministe)
        rev_pct_noisy = rev_pct + self.rng.normal(0, 0.15, n)
        rev_pct_noisy = np.clip(rev_pct_noisy, 0, 1)
        # v4.5: pentes reduites (0.50->0.30) pour probabilites moins extremes
        p_pme = np.clip(0.55 - 0.30 * rev_pct_noisy, 0.15, 0.60)
        p_ge = np.clip(0.15 + 0.30 * rev_pct_noisy, 0.15, 0.60)
        p_eti = np.maximum(1.0 - p_pme - p_ge, 0.10)
        probs = np.column_stack([p_pme, p_eti, p_ge])
        probs /= probs.sum(axis=1, keepdims=True)

        # Vectorise : echantillonnage multinomial via cumsum + uniform
        cumprobs = probs.cumsum(axis=1)
        u = self.rng.random(n)[:, None]
        idx = (u >= cumprobs).sum(axis=1)
        idx = np.clip(idx, 0, len(self.SIZES) - 1)
        df = df.with_columns(pl.Series("company_size", np.array(self.SIZES)[idx]))
        return df

    # -- Profils sectoriels ---------------------------------------------------

    def _apply_sector_profiles(self, df: pl.DataFrame) -> pl.DataFrame:
        """Shifts + dispersion sectorielle + zombie rate sur les marginales.

        v4.3 :
          - Dispersion : scaling multiplicatif autour de la mediane par secteur
            (modifie la FORME des distributions, pas juste la location)
          - Zombie rate : ajuste la proportion de marges negatives par secteur
        """
        _BOUNDS = {
            "ebitda_margin": (-0.50, 0.50),
            "debt_ratio": (0.01, None),
            "cf_volatility": (0.01, 1.0),
            "tangible_assets_ratio": (0.01, 0.99),
            "capex_to_revenue": (0.001, 0.50),
            "utilization_rate": (0.01, 0.99),
            "working_capital_ratio": (-0.30, 0.60),
            "current_ratio": (0.10, None),
        }

        # Extract columns that will be mutated to numpy for in-place work
        sector_arr = df["sector"].to_numpy()
        columns_to_mutate = set()
        for sector in self.SECTORS:
            shifts = _SECTOR_PROFILES.get(sector, {})
            for feature in shifts:
                if feature in df.columns:
                    columns_to_mutate.add(feature)
            scales = _SECTOR_MARGINAL_SCALES.get(sector, {})
            for feature in scales:
                if feature in df.columns:
                    columns_to_mutate.add(feature)
        # ebitda_margin is always mutated (zombie rate)
        if "ebitda_margin" in df.columns:
            columns_to_mutate.add("ebitda_margin")
        # Also add bounded columns
        for feature in _BOUNDS:
            if feature in df.columns:
                columns_to_mutate.add(feature)

        # Extract all mutable columns to numpy
        col_arrays = {}
        for col in columns_to_mutate:
            col_arrays[col] = df[col].to_numpy().copy()

        for sector in self.SECTORS:
            mask = sector_arr == sector
            if not mask.any():
                continue

            # 1. Location shift (v4.1)
            shifts = _SECTOR_PROFILES.get(sector, {})
            for feature, shift in shifts.items():
                if feature in col_arrays:
                    col_arrays[feature][mask] = col_arrays[feature][mask] + shift

            # 2. Dispersion scaling autour de la mediane (v4.3)
            scales = _SECTOR_MARGINAL_SCALES.get(sector, {})
            for feature, scale in scales.items():
                if feature in col_arrays:
                    vals = col_arrays[feature][mask]
                    med = np.median(vals)
                    col_arrays[feature][mask] = med + (vals - med) * scale

            # 3. Zombie rate adjustment (v4.4: monotone quantile warp)
            # Au lieu d'un shift global, on compresse/dilate la partie basse
            # pour atteindre le taux zombie cible sans deplacer les saines
            target_zombie = _SECTOR_ZOMBIE_RATES.get(sector, 0.20)
            margin_vals = col_arrays["ebitda_margin"][mask]
            current_zombie = (margin_vals < 0).mean()
            if current_zombie > 0.005 and abs(current_zombie - target_zombie) > 0.02:
                below_zero = margin_vals < 0
                above_zero = ~below_zero
                if below_zero.any() and above_zero.any():
                    # Ratio d'ajustement pour atteindre la cible
                    ratio = target_zombie / max(current_zombie, 0.01)
                    if ratio > 1:
                        # Besoin de PLUS de zombies: etirer les marges proches de 0 vers le bas
                        near_zero = (margin_vals >= 0) & (margin_vals < np.quantile(margin_vals[above_zero], 0.3))
                        n_convert = int(near_zero.sum() * min(ratio - 1, 0.5))
                        if n_convert > 0:
                            idxs = np.where(mask)[0][near_zero]
                            chosen = self.rng.choice(idxs, size=min(n_convert, len(idxs)), replace=False)
                            col_arrays["ebitda_margin"][chosen] *= -0.3
                    else:
                        # Besoin de MOINS de zombies: remonter les marges les moins negatives
                        zombie_idxs = np.where(mask)[0][below_zero]
                        zombie_margins = margin_vals[below_zero]
                        n_rescue = int(len(zombie_idxs) * (1 - ratio))
                        if n_rescue > 0:
                            # Remonter les moins negatives (les plus proches de 0)
                            order = np.argsort(-zombie_margins)[:n_rescue]
                            chosen = zombie_idxs[order]
                            col_arrays["ebitda_margin"][chosen] = np.abs(
                                col_arrays["ebitda_margin"][chosen]
                            ) + 0.01

        for feature, (lo, hi) in _BOUNDS.items():
            if feature in col_arrays:
                col_arrays[feature] = np.clip(col_arrays[feature], lo, hi)

        # Write all mutated columns back
        new_cols = [pl.Series(col, arr) for col, arr in col_arrays.items()]
        df = df.with_columns(new_cols)
        return df

    # -- Temporel + Macro -----------------------------------------------------

    def _add_temporal_macro(self, df: pl.DataFrame, vintage_idx: np.ndarray) -> pl.DataFrame:
        """Macro liees au vintage + bruit intra-trimestre (v4.3).

        Chaque observation recoit le niveau macro de son vintage PLUS un bruit
        gaussien calibre par variable. Empeche le modele d'apprendre la macro
        comme un simple lookup deterministe sur vintage_quarter.
        """
        n = len(df)
        quarters = [f"{2018 + (i // 4)}Q{(i % 4) + 1}" for i in range(self.n_vintages)]
        vintage_quarter = np.array(quarters)[vintage_idx]
        df = df.with_columns(pl.Series("vintage_quarter", vintage_quarter))
        for feat, values in _MACRO.items():
            base = np.array(values[: self.n_vintages])[vintage_idx]
            sigma = _MACRO_NOISE_SIGMA.get(feat, 0.0)
            if sigma > 0:
                base = base + self.rng.normal(0, sigma, n)
            df = df.with_columns(pl.Series(feat, base))
        return df

    # -- Features bruit -------------------------------------------------------

    def _add_noise_features(self, df: pl.DataFrame) -> pl.DataFrame:
        n = len(df)
        # Polars rank: compute percentile rank manually
        debt_rank = df["debt_ratio"].rank(method="ordinal").to_numpy().astype(float) / n
        rev_rank = df["revenue"].rank(method="ordinal").to_numpy().astype(float) / n

        df = df.with_columns([
            pl.Series("noise_gaussian", self.rng.normal(0, 1, n)),
            pl.Series("noise_uniform", self.rng.uniform(0, 1, n)),
            pl.Series("noise_correlated_1", 0.05 * debt_rank + 0.95 * self.rng.normal(0, 1, n)),
            pl.Series("noise_correlated_2", -0.05 * rev_rank + 0.95 * self.rng.normal(0, 1, n)),
            pl.Series("noise_shuffled_debt", self.rng.permutation(df["debt_ratio"].to_numpy())),
        ])
        return df

    # -- DGP structure v4.5 : realiste 50/30/20 monotones/interactions/conj --

    def _structured_dgp(
        self, df: pl.DataFrame, is_stress: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        DGP v4.5 : economiquement realiste, ratio 50/30/20.

        Architecture calibree sur la litterature credit corporate :
        - 50% effets MONOTONES : dette->PD+, marge->PD-, ICR->PD-
          Jamais d'inversion de signe. LogReg capture ~80% de ce signal.
        - 30% INTERACTIONS multiplicatives : debt x vol, margin x ICR, etc.
          LogReg ne capture pas (pas de feature engineering).
        - 20% CONJONCTIFS + SEUILS : zones AND, DPD>60, ICR<1.5.
          Avantage arbres modeste mais reel (~3pp AUC).

        DGP non-stationnaire : coefficients x1.5 en regime stress.
        Ecart LR/XGB attendu : ~2-5pp (realiste portefeuille corporate).
        """
        n = len(df)
        sector_arr = df["sector"].to_numpy()

        # v4.4: facteur d'amplification en stress (non-stationnarite)
        stress_mult = np.where(is_stress, 1.5, 1.0)

        raw = np.zeros(n)

        debt = df["debt_ratio"].to_numpy()
        margin = df["ebitda_margin"].to_numpy()
        vol = df["cf_volatility"].to_numpy()
        icr = df["interest_coverage_ratio"].to_numpy()
        util = df["utilization_rate"].to_numpy()
        nde = df["net_debt_to_ebitda"].to_numpy()
        dpd = df["days_past_due"].to_numpy()
        wc = df["working_capital_ratio"].to_numpy()
        cr = df["current_ratio"].to_numpy()
        ltv = df["loan_to_value"].to_numpy()
        log_rev = np.clip(np.log1p(df["revenue"].to_numpy() / 1e7), 0, 3)
        incidents_capped = np.minimum(df["nb_incidents_12m"].to_numpy(), 5)
        is_pme = (df["company_size"].to_numpy() == "PME").astype(float)
        age_yrs = df["account_age_months"].to_numpy() / 12.0

        # ================================================================
        # BLOC 1 : EFFETS MONOTONES PIECEWISE (~40% du signal)
        #
        # Economiquement realiste : chaque feature a un sens constant
        # (dette+ = risque+) mais avec des PALIERS et ACCELERATIONS.
        # Ex: debt < 0.30 = impact faible (buffer large)
        #     0.30-0.60 = impact modere (zone grise)
        #     debt > 0.60 = impact explosif (zone de detresse)
        # Un coefficient lineaire unique NE PEUT PAS capturer cette
        # forme : il doit moyenner la pente entre les 3 regimes.
        # L'arbre split aux seuils et donne des pentes locales.
        # ================================================================

        # Levier : 3 paliers de risque (debt)
        # Zone safe (<0.30): quasi-zero  |  Grise (0.30-0.60): modere  |  Danger (>0.60): fort
        raw += 0.15 * np.clip(debt - 0.30, 0, 0.30) / 0.30    # pente douce zone grise
        raw += 0.80 * np.clip(debt - 0.60, 0, 0.40) / 0.40    # pente forte zone danger

        # NDE : palier de detresse
        raw += 0.12 * np.clip(nde - 2.0, 0, 3.0) / 3.0        # zone grise 2-5
        raw += 0.50 * np.clip(nde - 5.0, 0, 5.0) / 5.0        # zone danger > 5

        # LTV : palier
        raw += 0.08 * np.clip(ltv - 0.40, 0, 0.20) / 0.20
        raw += 0.30 * np.clip(ltv - 0.60, 0, 0.30) / 0.30

        # Profitabilite : 2 regimes (marge neg = danger, marge pos = normal)
        raw += 0.60 * np.clip(-margin, 0, 0.30) / 0.30         # marge neg = risque fort
        raw -= 0.15 * np.clip(margin - 0.05, 0, 0.30) / 0.30   # marge pos = protection douce

        # ICR : 2 paliers
        raw += 0.50 * np.clip(1.5 - icr, 0, 1.5) / 1.5        # ICR < 1.5 = danger
        raw += 0.12 * np.clip(3.0 - icr, 0, 1.5) / 1.5        # 1.5 < ICR < 3 = zone grise

        # Liquidite : palier (current ratio)
        raw += 0.25 * np.clip(1.0 - cr, 0, 0.5) / 0.5         # CR < 1.0 = danger
        raw -= 0.08 * np.clip(cr - 1.5, 0, 1.0) / 1.0         # CR > 1.5 = bon

        # WC
        raw += 0.30 * np.clip(-wc, 0, 0.30) / 0.30            # WC neg = danger

        # Volatilite : palier
        raw += 0.12 * np.clip(vol - 0.15, 0, 0.15) / 0.15     # vol moderee
        raw += 0.35 * np.clip(vol - 0.30, 0, 0.30) / 0.30     # vol elevee

        # Utilization : palier saturation
        raw += 0.12 * np.clip(util - 0.50, 0, 0.30) / 0.30    # au-dessus de 50%
        raw += 0.30 * np.clip(util - 0.80, 0, 0.20) / 0.20    # > 80% = saturation

        # Taille (protect)
        raw -= 0.10 * log_rev

        # Anciennete : protectrice mais saturante
        raw -= 0.15 * np.clip(age_yrs - 1, 0, 5) / 5.0        # premieres annees
        raw -= 0.05 * np.clip(age_yrs - 5, 0, 15) / 15.0      # annees suivantes (sature)

        # Incidents : paliers (1, puis 3+)
        raw += 0.12 * np.clip(incidents_capped, 0, 1)           # premier incident
        raw += 0.25 * np.clip(incidents_capped - 1, 0, 4) / 4  # incidents supplementaires

        # DPD : palier 30j puis 60j
        raw += 0.08 * np.clip(dpd - 30, 0, 30) / 30            # 30-60j
        raw += 0.25 * np.clip(dpd - 60, 0, 300) / 60           # > 60j explosif

        # ================================================================
        # BLOC 2 : INTERACTIONS REGIME-DEPENDANTES (~35% du signal)
        #
        # Architecture: chaque feature importante a un DOUBLE effet :
        #   - Effet de base faible dans le bloc 1 (LR le capture)
        #   - Effet conditionnel FORT dans un regime specifique
        #
        # L'arbre decouvre les regimes par splits successifs.
        # LR voit un coefficient moyen qui sous-estime l'effet en
        # regime de stress et le surestime en regime sain.
        #
        # NOMBREUSES interactions a coef MODERE : chacune contribue
        # peu individuellement (~0.3-0.5), mais 15+ interactions
        # donnent un avantage cumulatif aux arbres.
        # ================================================================

        # --- Regimes binaires (seuils aux medianes) ---
        hi_debt = (debt > 0.45).astype(float)
        hi_vol = (vol > 0.22).astype(float)
        lo_margin = (margin < 0.08).astype(float)
        lo_icr = (icr < 2.5).astype(float)
        hi_util = (util > 0.50).astype(float)
        lo_cr = (cr < 1.2).astype(float)
        lo_wc = (wc < 0.05).astype(float)
        hi_inc = (incidents_capped >= 2).astype(float)
        hi_dpd = (dpd > 30).astype(float)

        # Debt amplifie en contexte volatile
        raw += 0.5 * np.clip(debt - 0.30, 0, 0.7) * hi_vol * stress_mult
        # Debt amplifie quand ICR faible
        raw += 0.5 * np.clip(debt - 0.30, 0, 0.7) * lo_icr * stress_mult
        # Debt amplifie quand marge faible
        raw += 0.4 * np.clip(debt - 0.30, 0, 0.7) * lo_margin * stress_mult
        # Debt amplifie quand liquidite tendue
        raw += 0.3 * np.clip(debt - 0.30, 0, 0.7) * lo_cr * stress_mult

        # Vol amplifie quand endette
        raw += 0.5 * np.clip(vol - 0.10, 0, 0.5) * hi_debt * stress_mult
        # Vol amplifie quand util elevee
        raw += 0.3 * np.clip(vol - 0.10, 0, 0.5) * hi_util * stress_mult

        # Marge neg amplifie quand endette
        raw += 0.5 * np.clip(-margin, 0, 0.3) * hi_debt * stress_mult
        # Marge neg amplifie quand vol elevee
        raw += 0.3 * np.clip(-margin, 0, 0.3) * hi_vol * stress_mult

        # ICR faible amplifie quand endette
        raw += 0.4 * np.clip(2.0 - icr, 0, 2.0) * hi_debt * stress_mult
        # ICR faible amplifie quand marge faible
        raw += 0.3 * np.clip(2.0 - icr, 0, 2.0) * lo_margin * stress_mult

        # Incidents amplifient quand endette
        raw += 0.4 * incidents_capped * hi_debt * stress_mult
        # Incidents amplifient quand liquidite faible
        raw += 0.3 * incidents_capped * lo_cr * stress_mult

        # Utilization amplifie quand vol elevee
        raw += 0.3 * np.clip(util - 0.50, 0, 0.5) * hi_vol * stress_mult
        # Utilization amplifie quand endette
        raw += 0.3 * np.clip(util - 0.50, 0, 0.5) * hi_debt * stress_mult

        # WC negatif amplifie quand endette
        raw += 0.3 * np.clip(-wc, 0, 0.3) * hi_debt * stress_mult

        # DPD amplifie quand endette et vol elevee
        raw += 0.3 * np.clip(dpd / 90, 0, 3.0) * hi_debt * stress_mult

        # ================================================================
        # BLOC 3 : CONJONCTIFS (AND) (~25% du signal)
        #
        # Combinaisons 2-way et 3-way. Chaque zone rectangulaire
        # est un "pocket" de risque que seul un arbre capture bien.
        # ================================================================

        # 2-way conjunctives
        raw += 0.8 * (lo_icr * lo_margin).astype(float) * stress_mult
        raw += 0.6 * (hi_debt * hi_vol).astype(float) * stress_mult
        raw += 0.6 * (hi_util * hi_inc).astype(float) * stress_mult
        raw += 0.5 * (hi_debt * lo_margin).astype(float) * stress_mult
        raw += 0.5 * (lo_cr * hi_inc).astype(float) * stress_mult
        raw += 0.4 * (lo_wc * hi_debt).astype(float) * stress_mult
        raw += 0.4 * (hi_dpd * hi_debt).astype(float) * stress_mult
        raw += 0.5 * ((is_pme > 0) * hi_debt).astype(float)
        raw += 0.4 * ((age_yrs < 3).astype(float) * hi_inc) * stress_mult

        # 3-way conjunctives (tres dur pour LR)
        raw += 0.6 * (hi_debt * hi_vol * lo_icr).astype(float) * stress_mult
        raw += 0.5 * (lo_margin * hi_util * hi_inc).astype(float) * stress_mult
        raw += 0.5 * (hi_debt * lo_margin * lo_cr).astype(float) * stress_mult
        raw += 0.4 * (hi_vol * lo_icr * lo_margin).astype(float) * stress_mult

        # NDE quadratique > 4 (acceleration du risque)
        raw += 0.3 * np.clip(nde - 4.0, 0, 4.0) ** 2 * stress_mult

        # -- Categorielles --
        raw += 0.15 * (sector_arr == "Retail").astype(float)
        raw -= 0.10 * (sector_arr == "Sante").astype(float)
        raw += 0.12 * (df["company_size"].to_numpy() == "PME").astype(float)

        # -- Macro --
        # v4.5 #43: coefficients renforces (sigma reduit => redeviennent visibles)
        raw -= 0.15 * df["gdp_growth"].to_numpy()
        raw += 0.12 * df["unemployment_rate"].to_numpy()
        raw += 0.08 * df["interest_rate_10y"].to_numpy()

        # -- Interactions secteur x features --
        # Deux mecanismes qui creent de la non-linearite sectorielle :
        # 1. Ajustements continus (coefficients differents par secteur)
        # 2. Seuils sectoriels (les seuils ICR, debt, etc. changent par secteur)
        _EFFECT_MAP = {
            "debt_x_vol":       lambda d: d["debt_ratio"].to_numpy() * d["cf_volatility"].to_numpy(),
            "log_revenue":      lambda d: np.log1p(d["revenue"].to_numpy() / 1e6),
            "icr_threshold":    lambda d: (d["interest_coverage_ratio"].to_numpy() < 1.50).astype(float),
            "margin_x_debt":    lambda d: np.clip(-d["ebitda_margin"].to_numpy(), 0, 0.5) * d["debt_ratio"].to_numpy(),
            "tangible_protect": lambda d: d["tangible_assets_ratio"].to_numpy(),
            "capex_stress":     lambda d: d["capex_to_revenue"].to_numpy() * np.clip(0.10 - d["ebitda_margin"].to_numpy(), 0, 0.3),
            "margin_buffer":    lambda d: np.clip(d["ebitda_margin"].to_numpy(), 0, 0.5),
            "util_high":        lambda d: (d["utilization_rate"].to_numpy() > 0.90).astype(float),
            "dpd_extra":        lambda d: np.clip(d["days_past_due"].to_numpy() - 60, 0, 300) / 30,
            "wc_stress":        lambda d: np.clip(-d["working_capital_ratio"].to_numpy(), 0, 0.5),
            "nde_explosive":    lambda d: np.clip(d["net_debt_to_ebitda"].to_numpy() - 4.0, 0, 4.0) ** 2,
        }

        for sector, adjustments in _SECTOR_DGP_ADJUSTMENTS.items():
            mask = sector_arr == sector
            if not mask.any():
                continue
            for effect_key, coeff in adjustments:
                if effect_key in _EFFECT_MAP:
                    effect_vals = _EFFECT_MAP[effect_key](df)
                    raw[mask] += coeff * effect_vals[mask]

        # Seuils sectoriels : le meme ratio (debt, ICR) a des seuils
        # de danger DIFFERENTS par secteur. LR ne peut pas representer
        # "ICR<1.5 en Tech mais ICR<2.5 en Energie" avec un seul coef.
        _SECTOR_THRESHOLDS = {
            "Tech":       {"debt_danger": 0.40, "icr_danger": 1.2, "vol_danger": 0.35},
            "Industrie":  {"debt_danger": 0.55, "icr_danger": 1.5, "vol_danger": 0.25},
            "Sante":      {"debt_danger": 0.35, "icr_danger": 1.0, "vol_danger": 0.30},
            "Retail":     {"debt_danger": 0.50, "icr_danger": 1.8, "vol_danger": 0.20},
            "Energie":    {"debt_danger": 0.60, "icr_danger": 2.5, "vol_danger": 0.20},
        }
        for sector, thresholds in _SECTOR_THRESHOLDS.items():
            mask = sector_arr == sector
            if not mask.any():
                continue
            # Seuil debt
            raw[mask] += 0.6 * (debt[mask] > thresholds["debt_danger"]).astype(float) * stress_mult[mask]
            # Seuil ICR
            raw[mask] += 0.5 * (icr[mask] < thresholds["icr_danger"]).astype(float) * stress_mult[mask]
            # Seuil vol
            raw[mask] += 0.4 * (vol[mask] > thresholds["vol_danger"]).astype(float) * stress_mult[mask]

        # -- Heterogeneite maturite-dependante --
        age_factor = np.clip(1.0 - df["account_age_months"].to_numpy() / 120, 0, 1.0)
        raw += 0.15 * age_factor * self.rng.standard_normal(n)

        # -- Bruit heteroscedastique Student-t(5) --
        sigma_i = self.noise_sigma * (
            1.0
            + 0.3 * (df["company_size"].to_numpy() == "PME").astype(float)
            + 0.2 * np.clip(df["debt_ratio"].to_numpy() - 0.3, 0, 1.0)
            + 0.2 * np.clip(df["cf_volatility"].to_numpy() - 0.2, 0, 1.0)
        )
        t_noise = self.rng.standard_t(df=5, size=n)
        raw += sigma_i * t_noise * np.sqrt(3.0 / 5.0)

        # -- Calibration intercept --
        def mean_pd(c):
            return expit(c + raw).mean() - self.base_default_rate

        intercept = brentq(mean_pd, -30, 30)
        logit_vals = intercept + raw
        pd_vals = expit(logit_vals)
        default_flag = self.rng.binomial(1, pd_vals)

        return default_flag, pd_vals

    # -- MNAR [v4.2 #22 : + discretes] ----------------------------------------

    def _inject_mnar(self, df: pl.DataFrame) -> pl.DataFrame:
        """MNAR sur continues ET discretes + composante MCAR (v4.3).

        v4.3 :
          - Taux MNAR varie par observation (lognormal) pour casser la regularite
          - Composante MCAR ~2% sur toutes les numeriques (manquant aleatoire pur)
        v4.5 #45:
          - Guard: missing_rate <= 0 => skip MNAR entierement
          - clip(0, 0.40) au lieu de clip(0.01, 0.40) pour respecter missing_rate=0
        """
        if self.missing_rate <= 0:
            return df

        n = len(df)
        is_pme = df["company_size"].to_numpy() == "PME"

        # v4.3 : taux individuel varie autour de self.missing_rate
        rate_i = self.missing_rate * self.rng.lognormal(0, 0.4, n)
        rate_i = np.clip(rate_i, 0.0, 0.40)  # v4.5: 0.01->0.0 (respecte missing_rate faible)

        # Continues : endettes cachent marge, PME cachent ICR
        high_debt = df["debt_ratio"].to_numpy() > df["debt_ratio"].quantile(0.70)

        # Extract columns to numpy for masked NaN injection
        ebitda_margin = df["ebitda_margin"].to_numpy().copy().astype(float)
        icr_arr = df["interest_coverage_ratio"].to_numpy().copy().astype(float)
        ltv_arr = df["loan_to_value"].to_numpy().copy().astype(float)
        tangible_arr = df["tangible_assets_ratio"].to_numpy().copy().astype(float)
        incidents_arr = df["nb_incidents_12m"].to_numpy().copy().astype(float)
        dpd_arr = df["days_past_due"].to_numpy().copy().astype(float)

        ebitda_margin[(self.rng.random(n) < rate_i) & high_debt] = np.nan
        icr_arr[(self.rng.random(n) < rate_i) & is_pme] = np.nan
        ltv_arr[self.rng.random(n) < rate_i * 0.5] = np.nan
        tangible_arr[self.rng.random(n) < rate_i * 0.3] = np.nan

        # [v4.2] Discretes : PME sous-rapportent incidents, DPD parfois manquant
        incidents_arr[(self.rng.random(n) < rate_i * 0.7) & is_pme] = np.nan
        dpd_arr[self.rng.random(n) < rate_i * 0.3] = np.nan

        df = df.with_columns([
            pl.Series("ebitda_margin", ebitda_margin),
            pl.Series("interest_coverage_ratio", icr_arr),
            pl.Series("loan_to_value", ltv_arr),
            pl.Series("tangible_assets_ratio", tangible_arr),
            pl.Series("nb_incidents_12m", incidents_arr),
            pl.Series("days_past_due", dpd_arr),
        ])

        # v4.4 : MCAR ~2% exclut macro (jamais manquant en prod) et noise
        mcar_rate = 0.02
        _MCAR_EXCLUDE = {
            TARGET, LATENT_PD,
            "gdp_growth", "unemployment_rate", "interest_rate_10y",  # macro
            "noise_gaussian", "noise_uniform", "noise_correlated_1",
            "noise_correlated_2", "noise_shuffled_debt",  # bruit
        }
        num_cols = [
            c for c in df.columns
            if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)
            and c not in _MCAR_EXCLUDE
        ]
        mcar_updates = []
        for col in num_cols:
            mcar_mask = self.rng.random(n) < mcar_rate
            if mcar_mask.any():
                col_arr = df[col].to_numpy().copy().astype(float)
                col_arr[mcar_mask] = np.nan
                mcar_updates.append(pl.Series(col, col_arr))
        if mcar_updates:
            df = df.with_columns(mcar_updates)

        return df

    # -- Pipeline principal [v4.2 #16 : ordre corrige] -------------------------

    def generate(self) -> pl.DataFrame:
        """
        Pipeline v4.5 :
            1. Secteurs (non-uniformes) + vintages
            2. Copula PAR SECTEUR x regime
            3. Marginales continues (revenue par secteur)
            4. company_size (vectorise, stochastique)
            5. Profils sectoriels (shifts + dispersion + zombie rate quantile-warp)
            6. Features discretes (DPD par secteur, APRES profils)
            7. Temporal + macro (sigma reduit, amplitude reduite)
            8. Bruit
            9. DGP non-stationnaire (signal redistribue)
            10. MNAR (guard missing_rate=0, lognormal + MCAR 2%)
        """
        # 1. Assignation secteurs + vintages
        sector_names = list(_SECTOR_WEIGHTS.keys())
        sector_probs = list(_SECTOR_WEIGHTS.values())
        sector_arr = self.rng.choice(sector_names, size=self.n_rows, p=sector_probs)

        vintage_idx = self.rng.integers(0, self.n_vintages, size=self.n_rows)
        is_stress = np.isin(vintage_idx, list(self.stress_quarters))

        # 2. Copula par (secteur, regime) -- correlations differentes
        dim = _FACTOR_LOADINGS.shape[0]
        U = np.empty((self.n_rows, dim))

        for sector in sector_names:
            scales = _SECTOR_FACTOR_SCALES[sector]
            corr_normal = self._build_sector_corr(scales, stress=False)
            corr_stress = self._build_sector_corr(scales, stress=True)

            for is_s, corr, df_val in [
                (False, corr_normal, self.df_normal),
                (True, corr_stress, self.df_stress),
            ]:
                mask = (sector_arr == sector) & (is_stress == is_s)
                n_sub = mask.sum()
                if n_sub > 0:
                    U[mask] = self._generate_t_copula(corr, df_val, n_sub)

        # 3. Marginales continues (v4.5: revenue par secteur)
        df, dpd_latent = self._apply_marginals(U, sector_arr)

        # 4. Secteur + company_size
        df = df.with_columns(pl.Series("sector", sector_arr))
        df = self._add_company_size(df)

        # 5. Profils sectoriels (shifts sur marginales)
        df = self._apply_sector_profiles(df)

        # 6. Features discretes (APRES profils -- incidents utilisent marge shiftee)
        df = self._add_discrete_features(df, dpd_latent)

        # 7. Temporal + macro
        df = self._add_temporal_macro(df, vintage_idx)

        # 8. Bruit
        df = self._add_noise_features(df)

        # 9. DGP (v4.4: passe is_stress pour non-stationnarite)
        default_flag, pd_latent = self._structured_dgp(df, is_stress)
        df = df.with_columns([
            pl.Series(TARGET, default_flag),
            pl.Series(LATENT_PD, pd_latent),
        ])

        # 10. MNAR
        df = self._inject_mnar(df)

        return df

    def generate_supply_chain_network(self, df: pl.DataFrame) -> pl.DataFrame:
        """Genere un reseau Supply Chain synthetique bidirectionnel (100% vectorise).

        Algorithme :
            1. Selection de 'Hubs' (top 5% revenue, cap 1000).
            2. Generation batch de tous les liens upstream/downstream.
            3. Affinite sectorielle bidirectionnelle.

        Returns:
            DataFrame [source_id, target_id, weight, type, payment_delay_days].
        """
        _EDGE_DENSITY = 0.00015
        _HUB_MULTIPLIER = 5.0

        n = len(df)
        revenue = df["revenue"].to_numpy()
        sectors = df["sector"].to_numpy()
        ids = np.arange(n)

        n_hubs = max(min(int(n * 0.05), 1000), 1)
        hub_indices = np.argsort(-revenue)[:n_hubs]
        n_edges_total = int(n * _EDGE_DENSITY * 30_000)
        edges_per_hub = max(n_edges_total // n_hubs, 1)

        dpd_array = df["days_past_due"].to_numpy() if "days_past_due" in df.columns else np.zeros(n)

        # Pre-compute sector pools
        sector_pools = {}
        for s in self.SECTORS:
            mask = sectors == s
            if not mask.any():
                continue
            sector_pools[s] = ids[mask]

        sector_list = [s for s in self.SECTORS if s in sector_pools]
        n_sectors = len(sector_list)
        if n_sectors == 0:
            return pl.DataFrame(schema={
                "source_id": pl.Int64,
                "target_id": pl.Int64,
                "weight": pl.Float64,
                "type": pl.Utf8,
                "payment_delay_days": pl.Float64,
            })

        pool_sizes = np.array([len(sector_pools[s]) for s in sector_list])

        upstream_affinity = {
            ("Tech", "Industrie"): 2.0,
            ("Energie", "Industrie"): 2.0,
        }
        downstream_affinity = {
            ("Industrie", "Retail"): 2.0,
            ("Tech", "Retail"): 1.5,
        }

        n_sup = max(1, edges_per_hub // 3)
        n_cli = max(5, edges_per_hub * 2)

        hub_sectors = sectors[hub_indices]
        hub_ids = ids[hub_indices]

        # Build per-hub-sector affinity matrices (n_hubs x n_sectors)
        up_affinity = np.ones((n_hubs, n_sectors))
        for (s_from, s_to), w in upstream_affinity.items():
            j = sector_list.index(s_from) if s_from in sector_list else -1
            if j < 0:
                continue
            hub_mask = hub_sectors == s_to
            up_affinity[hub_mask, j] = w
        up_affinity /= up_affinity.sum(axis=1, keepdims=True)

        down_affinity = np.ones((n_hubs, n_sectors))
        for (s_from, s_to), w in downstream_affinity.items():
            j = sector_list.index(s_to) if s_to in sector_list else -1
            if j < 0:
                continue
            hub_mask = hub_sectors == s_from
            down_affinity[hub_mask, j] = w
        down_affinity /= down_affinity.sum(axis=1, keepdims=True)

        # Vectorized sector assignment for ALL edges at once
        total_up = n_hubs * n_sup
        up_cum = np.cumsum(up_affinity, axis=1)
        up_rand = self.rng.random((n_hubs, n_sup))
        up_sector_idx = np.array([
            np.searchsorted(up_cum[i], up_rand[i]) for i in range(n_hubs)
        ])
        up_sector_idx = np.clip(up_sector_idx, 0, n_sectors - 1)

        total_down = n_hubs * n_cli
        down_cum = np.cumsum(down_affinity, axis=1)
        down_rand = self.rng.random((n_hubs, n_cli))
        down_sector_idx = np.array([
            np.searchsorted(down_cum[i], down_rand[i]) for i in range(n_hubs)
        ])
        down_sector_idx = np.clip(down_sector_idx, 0, n_sectors - 1)

        # Build concatenated pool + offset array
        pool_offsets = np.zeros(n_sectors + 1, dtype=np.intp)
        for i, s in enumerate(sector_list):
            pool_offsets[i + 1] = pool_offsets[i] + len(sector_pools[s])
        all_pool_ids = np.concatenate([sector_pools[s] for s in sector_list])

        # Upstream: uniform sample within sector pool
        up_sector_flat = up_sector_idx.ravel()
        up_pool_sizes = pool_sizes[up_sector_flat]
        up_within_idx = (self.rng.random(total_up) * up_pool_sizes).astype(np.intp)
        up_source_ids = all_pool_ids[pool_offsets[up_sector_flat] + up_within_idx]
        up_hub_ids = np.repeat(hub_ids, n_sup)

        # Downstream: uniform sample within sector pool
        down_sector_flat = down_sector_idx.ravel()
        down_pool_sizes = pool_sizes[down_sector_flat]
        down_within_idx = (self.rng.random(total_down) * down_pool_sizes).astype(np.intp)
        down_target_ids = all_pool_ids[pool_offsets[down_sector_flat] + down_within_idx]
        down_hub_ids = np.repeat(hub_ids, n_cli)

        # Assemble all edges
        sources_arr = np.concatenate([up_source_ids, down_target_ids])
        targets_arr = np.concatenate([up_hub_ids, down_hub_ids])
        types_arr = np.array(
            ["is_supplier_of"] * total_up + ["is_client_of"] * total_down
        )

        # Remove self-loops
        not_self = sources_arr != targets_arr
        sources_arr = sources_arr[not_self]
        targets_arr = targets_arr[not_self]
        types_arr = types_arr[not_self]

        n_edges = len(sources_arr)
        is_supplier = types_arr == "is_supplier_of"
        n_sup_edges = is_supplier.sum()
        n_cli_edges = n_edges - n_sup_edges

        # Vectorized attributes
        weights_arr = np.empty(n_edges)
        weights_arr[is_supplier] = self.rng.uniform(0.15, 0.40, size=n_sup_edges)
        weights_arr[~is_supplier] = self.rng.uniform(0.01, 0.05, size=n_cli_edges)

        target_dpd = dpd_array[targets_arr] if n_edges > 0 else np.array([])
        source_dpd = dpd_array[sources_arr] if n_edges > 0 else np.array([])

        delays_arr = np.empty(n_edges)
        if n_sup_edges > 0:
            delays_arr[is_supplier] = self.rng.uniform(15, 45, size=n_sup_edges) + target_dpd[is_supplier] * 0.8
        if n_cli_edges > 0:
            delays_arr[~is_supplier] = self.rng.uniform(30, 60, size=n_cli_edges) + source_dpd[~is_supplier] * 0.8
        delays_arr = np.minimum(delays_arr, 180.0)

        return pl.DataFrame({
            "source_id": sources_arr,
            "target_id": targets_arr,
            "weight": weights_arr,
            "type": types_arr,
            "payment_delay_days": delays_arr,
        })
