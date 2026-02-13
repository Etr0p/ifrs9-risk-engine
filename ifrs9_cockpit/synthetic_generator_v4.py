"""
Pipeline de Generation de Donnees Synthetiques Financieres v4.5
================================================================
Target: Deep Learning Tabulaire (TabNet, Transformer, XGBoost)

v4.0 (18 corrections) :
  1. Bimodale marge via inversion CDF mixture
  2. DGP structure 17 effets non-lineaires (remplace chaos)
  3. 27 features (20 signal + 5 bruit + 2 categorielles)
  4. Structure temporelle 24 vintages + derive macro
  5. t-Copula factor model PSD + regime stress
  6. Evaluation robuste (temporel + StratifiedKFold + multi-modele)
  7. MNAR ~10% conditionne sur detresse
  8. Bruit avec correlation parasite +/-0.05
  9. Class weighting RF / XGBoost
  10-12. OneHotEncoder, suppression seed global, intercept calibre

v4.1 (anti-biais) :
  13. Profils financiers par secteur (shifts sur marginales copula)
  14. nb_incidents multi-facteur + bruit multiplicatif log-normal
  15. Bruit DGP heteroscedastique Student-t(5)

v4.2 (7 corrections) :
  16. FIX ORDRE : discrete features APRES profils sectoriels
  17. Coefficients DGP /2.5 pour AUC realiste ~0.80 (etait ~0.99)
  18. Distribution sectorielle non-uniforme (realiste)
  19. Copula par secteur (correlations differentes Tech vs Industrie)
  20. company_size vectorise (etait boucle Python 1.5M)
  21. Variable morte v=df.values supprimee
  22. MNAR etendu aux features discretes (incidents, DPD)

v4.3 (8 corrections anti-biais) :
  23. Dispersion sectorielle (scaling variance autour de la mediane)
  24. Zombie rate par secteur (proportion de marges negatives differente)
  25. Macro bruitee (bruit intra-trimestre gaussien)
  26. log(revenue) dans le DGP (taille comme predicteur continu)
  27. Interactions secteur x features (pentes DGP differentes par secteur)
  28. Macro x2 + account_age x1.7 (etaient quasi-invisibles)
  29. MNAR lognormal par observation + MCAR 2% (casse la regularite)
  30. Heterogeneite maturite-dependante (jeunes comptes = plus de bruit)

v4.4 (8 corrections anti-biais finales) :
  31. Macro sigma x5 : vintage non-identifiable depuis macro (81%->~15%)
  32. DGP rebalance : non-linearites renforcees, monotones reduits
      => XGBoost > LogReg (corrige l'inversion)
  33. nb_credit_lines ajoute a NOISE_FEATURES (etait bruit cache)
  34. AUC calibre ~0.82 (noise_sigma ajuste avec nouveaux coefficients)
  35. MCAR exclut macro + noise (macro jamais manquant en prod)
  36. DGP non-stationnaire : coefficients varient en regime stress
  37. Zombie rate via quantile-shift cible (plus propre)
  38. Limitation documentee : pas de structure panel (IID)

v4.5 (8 corrections residuelles) :
  39. CRITIQUE: cycle macro aplati (GDP [-2,+2.5], unemp [7,9]) => regime
      stress non-detectable (etait AUC=0.908 avec [-5,+5])
  40. Revenue par secteur (lognorm s/scale differents)
  41. DPD par secteur (threshold + scale differents)
  42. Redistribution DGP : debt x vol reduit, ICR/marge/WC/revenue renforces
  43. Macro coefficients visibles (sigma reduit car amplitude reduite)
  44. account_age moins conditionnel (debt<0.70, coeff 0.5)
  45. MNAR guard: missing_rate=0 => skip (clip 0.01->0 corrige)
  46. company_size plus stochastique (pentes reduites, bruit additif)

Dependances : numpy, pandas, scipy, scikit-learn
Optionnelles : xgboost
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as t_dist, norm, beta, lognorm, gamma
from scipy.special import expit
from scipy.optimize import brentq
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import warnings

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# ===========================================================================
# Constants
# ===========================================================================

TARGET = "target_default"
LATENT_PD = "pd_latent"

NOISE_FEATURES = [
    "noise_gaussian", "noise_uniform",
    "noise_correlated_1", "noise_correlated_2", "noise_shuffled_debt",
    "nb_credit_lines",  # v4.4: Poisson(3) independant, etait bruit cache
]

# Macro scenario : 24 trimestres generiques (cycle complet)
# v4.5: trajectoire macro REALISTE avec auto-correlation forte
# Principe: les indicateurs macro evoluent lentement (+/-0.25pp max par trimestre)
# Pas de sauts: on ne passe pas de 0.5% a 3% en un trimestre.
# Le stress est un RALENTISSEMENT progressif, pas un choc brutal.
# Cela rend le regime stress quasi-indistinguable des periodes normales.
_MACRO = {
    "gdp_growth": [
        1.5, 1.7, 2.0, 2.2,       # T1-T4  : expansion (+0.2-0.3/T)
        2.2, 2.0, 1.8, 1.5,       # T5-T8  : deceleration (-0.2-0.3/T)
        1.3, 1.0, 0.8, 1.0,       # T9-T12 : creux doux (stress)
        1.2, 1.5, 1.7, 2.0,       # T13-T16: reprise (+0.2-0.3/T)
        2.0, 1.8, 1.5, 1.3,       # T17-T20: re-deceleration
        1.2, 1.3, 1.5, 1.5,       # T21-T24: normalisation
    ],
    "unemployment_rate": [
        7.5, 7.3, 7.2, 7.0,       # T1-T4  : amelioration (-0.1 a -0.2/T)
        7.0, 6.8, 7.0, 7.1,       # T5-T8  : plancher + rebond
        7.2, 7.5, 7.6, 7.5,       # T9-T12 : deterioration moderee (stress)
        7.3, 7.2, 7.0, 6.9,       # T13-T16: amelioration
        6.9, 7.0, 7.1, 7.3,       # T17-T20: re-deterioration
        7.3, 7.2, 7.1, 7.0,       # T21-T24: stabilisation
    ],
    "interest_rate_10y": [
        1.00, 1.25, 1.25, 1.50,    # T1-T4  : hausse graduelle (+0.25 max/T)
        1.50, 1.25, 1.00, 0.75,    # T5-T8  : baisse accommodante (-0.25/T)
        0.75, 0.50, 0.50, 0.75,    # T9-T12 : plancher (stress = taux bas)
        0.75, 1.00, 1.00, 1.25,    # T13-T16: normalisation (+0.25 max/T)
        1.25, 1.50, 1.50, 1.75,    # T17-T20: poursuite (+0.25 max/T)
        1.75, 2.00, 2.00, 2.00,    # T21-T24: plateau (+0.25 max, stabilise)
    ],
}

_DEFAULT_STRESS_QUARTERS = {8, 9, 10}  # T9-T11 : phase recession

# Profils financiers par secteur (shift additif sur les marginales copula)
_SECTOR_PROFILES: dict[str, dict[str, float]] = {
    "Tech": {
        "ebitda_margin": +0.05,
        "cf_volatility": +0.08,
        "tangible_assets_ratio": -0.15,
        "capex_to_revenue": +0.03,
        "debt_ratio": -0.10,
    },
    "Industrie": {
        "tangible_assets_ratio": +0.10,
        "debt_ratio": +0.05,
        "capex_to_revenue": +0.02,
        "current_ratio": -0.10,
    },
    "Sante": {
        "ebitda_margin": +0.03,
        "debt_ratio": -0.08,
        "cf_volatility": -0.05,
        "current_ratio": +0.15,
    },
    "Retail": {
        "ebitda_margin": -0.04,
        "utilization_rate": +0.08,
        "working_capital_ratio": -0.05,
        "debt_ratio": +0.03,
    },
    "Energie": {
        "capex_to_revenue": +0.05,
        "debt_ratio": +0.08,
        "cf_volatility": +0.06,
        "tangible_assets_ratio": +0.08,
    },
}

# [v4.2 #18] Distribution sectorielle non-uniforme (realiste)
_SECTOR_WEIGHTS = {
    "Tech": 0.25,
    "Industrie": 0.20,
    "Sante": 0.12,
    "Retail": 0.28,
    "Energie": 0.15,
}

# [v4.2 #19] Facteurs latents par secteur : modifie la structure de correlation
# Chaque secteur a un profil de dependance different
# [Profitabilite, Levier, Liquidite, Maturite]
_SECTOR_FACTOR_SCALES = {
    "Tech":       [1.20, 0.70, 1.00, 0.80],  # Forte profitabilite, levier faible
    "Industrie":  [0.90, 1.20, 0.80, 1.10],  # Fort levier, liquidite tendue
    "Sante":      [1.00, 0.80, 1.20, 1.00],  # Forte liquidite, levier conservateur
    "Retail":     [0.80, 1.00, 0.90, 0.90],  # Profitabilite plus faible
    "Energie":    [0.90, 1.30, 0.80, 1.00],  # Fort levier (financement projets)
}

# Dispersion sectorielle : scaling multiplicatif autour de la mediane
# Modifie la FORME des distributions (variance), pas juste la location
_SECTOR_MARGINAL_SCALES: dict[str, dict[str, float]] = {
    "Tech": {
        "debt_ratio": 0.85,         # tight (asset-light homogene)
        "cf_volatility": 1.30,      # wide (startups vs GAFAM)
        "ebitda_margin": 1.20,      # wide (bimodalite amplifiee)
    },
    "Industrie": {
        "debt_ratio": 0.80,         # tight (levier stable, collateral)
        "cf_volatility": 0.85,      # narrow (flux predictibles)
    },
    "Sante": {
        "debt_ratio": 0.75,         # very tight (conservateur)
        "cf_volatility": 0.70,      # low variance (reguliers)
        "ebitda_margin": 0.80,      # narrow (marges reglementees)
    },
    "Retail": {
        "debt_ratio": 1.10,         # wide (heterogene)
        "ebitda_margin": 1.15,      # wide (marges fines, variance)
        "utilization_rate": 1.10,   # wide
    },
    "Energie": {
        "debt_ratio": 1.20,         # wide (project finance varie)
        "cf_volatility": 1.25,      # wide (commodity exposure)
    },
}

# [v4.5 #40] Revenue par secteur : parametres lognorm(s, scale) differents
_SECTOR_REVENUE_PARAMS: dict[str, dict[str, float]] = {
    "Tech":       {"s": 1.00, "scale": 30_000_000},  # High dispersion, gros CA moyen
    "Industrie":  {"s": 0.75, "scale": 20_000_000},  # Modere, CA moyen
    "Sante":      {"s": 0.70, "scale": 15_000_000},  # Tight, CA plus petit
    "Retail":     {"s": 0.90, "scale": 12_000_000},  # Large dispersion, petit CA
    "Energie":    {"s": 0.80, "scale": 40_000_000},  # Gros CA (project finance)
}

# [v4.5 #41] DPD par secteur : threshold et scale differents
_SECTOR_DPD_PARAMS: dict[str, dict[str, float]] = {
    "Tech":       {"threshold": 0.75, "scale": 25},  # Peu de DPD, courte duree
    "Industrie":  {"threshold": 0.68, "scale": 35},  # Plus de DPD, plus long
    "Sante":      {"threshold": 0.80, "scale": 20},  # Tres peu de DPD
    "Retail":     {"threshold": 0.60, "scale": 40},  # Beaucoup de DPD, longue duree
    "Energie":    {"threshold": 0.72, "scale": 30},  # Modere
}

# Proportion de zombies (ebitda_margin < seuil) par secteur
# Base copula = ~20% ; ajustement post-generation
_SECTOR_ZOMBIE_RATES: dict[str, float] = {
    "Tech": 0.25,       # Startups cash-burn
    "Industrie": 0.18,  # Proche de la base
    "Sante": 0.05,      # Revenus reglementes
    "Retail": 0.30,     # Marges fines, concurrence
    "Energie": 0.22,    # Cycles commodity
}

# Bruit intra-trimestre sur variables macro (ecart-type)
# v4.5: sigma calibre pour AUC stress < 0.65
# Amplitude GDP ~2.5pp => sigma=2.0 donne overlap > 80% entre regimes
# Macro coefficients restent visibles car coefs DGP renforces
# v4.5: sigma modere car trajectoire auto-correlee (variations +/-0.2/T max)
# Le bruit represente l'heterogeneite intra-trimestre (regional, sectoriel, etc.)
# Amplitude macro reduite (~1pp GDP) => sigma=1.5 suffit pour bon overlap
# v4.5: sigma = heterogeneite intra-trimestre (regional, sectoriel, etc.)
# Les taux necessitent un sigma plus fort car leur tendance monotone
# est facile a detecter sinon.
# v4.5: sigma = heterogeneite intra-trimestre
# Represente : heterogeneite regionale, effets de taux differents par type emprunt,
# conditions specifiques de marche pour chaque entreprise, etc.
_MACRO_NOISE_SIGMA: dict[str, float] = {
    "gdp_growth": 1.80,         # v4.5: GDP cycle + dispersion regionale
    "unemployment_rate": 1.00,   # v4.5: chomage + dispersion sectorielle
    "interest_rate_10y": 1.70,   # v4.5: taux marche + spread credit individuel
}

# Ajustements DGP par secteur : interactions secteur x features
# Coefs forts (0.5-1.5) car c'est ici que les arbres gagnent :
# LR avec OHE pour sector a 5 dummies + 20 features = 25 coefs.
# Mais il ne peut PAS donner un coef different a debt_ratio POUR
# chaque secteur sans feature engineering (sector × debt).
# L'arbre split naturellement sur sector, puis sur debt, et
# donne des pentes locales par secteur.
_SECTOR_DGP_ADJUSTMENTS: dict[str, list[tuple[str, float]]] = {
    "Tech": [
        ("debt_x_vol", -0.8),       # Asset-light: levier x vol = moins dangereux
        ("log_revenue", -0.20),     # Economies d'echelle tres protectrices (GAFAM)
        ("icr_threshold", +0.5),    # Cash burn -> ICR tres critique
        ("margin_x_debt", +0.8),    # Marge negative + dette = spirale rapide
    ],
    "Industrie": [
        ("debt_x_vol", +0.3),       # Actifs physiques + vol = risque
        ("tangible_protect", -0.30), # Collateral protege significativement
        ("capex_stress", +0.5),     # Capex eleve + marge faible = destruction
    ],
    "Sante": [
        ("debt_x_vol", -0.5),       # Flux tres stables = vol moins dangereuse
        ("log_revenue", +0.15),     # Regulation protege les gros
        ("margin_buffer", -0.5),    # Marges resilientes (remboursement garanti)
    ],
    "Retail": [
        ("util_high", +0.6),        # Utilization > 0.9 = signal fort de detresse
        ("margin_x_debt", +0.8),    # Marge fine + dette = spirale
        ("dpd_extra", +0.4),        # DPD tres predictif (comportemental)
        ("wc_stress", +0.5),        # BFR = cl de la survie en retail
    ],
    "Energie": [
        ("debt_x_vol", +1.0),       # Project finance : levier x vol = tres risque
        ("log_revenue", -0.15),     # Taille aide (major vs junior)
        ("icr_threshold", +0.6),    # ICR critique en project finance
        ("nde_explosive", +0.5),    # Surendettement = catastrophe
    ],
}

# Factor loadings : 15 features x 4 facteurs latents
_FACTOR_LOADINGS = np.array([
    #  Prof   Lever  Liquid  Matur
    [ 0.50, -0.15,  0.10,  0.25],   # 0  revenue
    [ 0.65, -0.40,  0.10,  0.10],   # 1  ebitda_margin
    [-0.30,  0.75, -0.20, -0.15],   # 2  debt_ratio
    [-0.40,  0.55, -0.15, -0.10],   # 3  cf_volatility
    [ 0.45, -0.50,  0.25,  0.10],   # 4  interest_coverage_ratio
    [ 0.15, -0.20,  0.65,  0.10],   # 5  current_ratio
    [ 0.10, -0.15,  0.60,  0.05],   # 6  cash_ratio
    [-0.15,  0.55, -0.10, -0.10],   # 7  loan_to_value
    [-0.25,  0.70, -0.15, -0.10],   # 8  net_debt_to_ebitda
    [ 0.30, -0.10,  0.05,  0.15],   # 9  capex_to_revenue
    [ 0.10, -0.10,  0.50,  0.10],   # 10 working_capital_ratio
    [ 0.05,  0.10, -0.05,  0.35],   # 11 tangible_assets_ratio
    [ 0.20, -0.15,  0.10,  0.45],   # 12 account_age_months
    [-0.10,  0.30, -0.35, -0.15],   # 13 utilization_rate
    [-0.25,  0.35, -0.20, -0.30],   # 14 dpd_latent
])


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
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Transforme les uniformes copula en distributions financieres.

        v4.5: revenue parametrise par secteur (s, scale differents).
        """
        df = pd.DataFrame()

        # [v4.5 #40] Revenue par secteur
        df["revenue"] = lognorm.ppf(U[:, 0], s=0.85, scale=25_000_000)  # fallback
        for sector, params in _SECTOR_REVENUE_PARAMS.items():
            mask = sector_arr == sector
            if mask.any():
                df.loc[mask, "revenue"] = lognorm.ppf(
                    U[mask, 0], s=params["s"], scale=params["scale"]
                )

        # Bimodale via inversion CDF mixture (20% zombies, 80% saines)
        mix_w = 0.20
        x_grid = np.linspace(-0.50, 0.50, 10_000)
        cdf_mixture = (
            mix_w * norm.cdf(x_grid, loc=-0.05, scale=0.10)
            + (1 - mix_w) * norm.cdf(x_grid, loc=0.15, scale=0.05)
        )
        df["ebitda_margin"] = np.interp(U[:, 1], cdf_mixture, x_grid)

        df["debt_ratio"] = gamma.ppf(U[:, 2], a=2.0, scale=0.3)
        df["cf_volatility"] = beta.ppf(U[:, 3], a=1.5, b=4.0)
        df["interest_coverage_ratio"] = lognorm.ppf(U[:, 4], s=0.7, scale=3.0)
        df["current_ratio"] = gamma.ppf(U[:, 5], a=5.0, scale=0.3)
        df["cash_ratio"] = beta.ppf(U[:, 6], a=2.0, b=5.0)
        df["loan_to_value"] = beta.ppf(U[:, 7], a=2.5, b=3.0)
        df["net_debt_to_ebitda"] = gamma.ppf(U[:, 8], a=2.0, scale=1.5)
        df["capex_to_revenue"] = beta.ppf(U[:, 9], a=1.5, b=8.0)
        df["working_capital_ratio"] = np.clip(
            norm.ppf(U[:, 10], loc=0.15, scale=0.10), -0.30, 0.60
        )
        df["tangible_assets_ratio"] = beta.ppf(U[:, 11], a=3.0, b=2.0)
        df["account_age_months"] = np.clip(
            stats.expon.ppf(U[:, 12], scale=60), 1, 360
        ).astype(int)
        df["utilization_rate"] = beta.ppf(U[:, 13], a=2.0, b=3.0)

        dpd_latent = U[:, 14]
        return df, dpd_latent

    # -- Features discretes ---------------------------------------------------

    def _add_discrete_features(self, df: pd.DataFrame, dpd_latent: np.ndarray):
        """
        Features discretes. Appelees APRES _apply_sector_profiles
        pour que les incidents refletent les profils sectoriels.

        v4.5: DPD threshold et scale par secteur.
        """
        n = len(df)
        sector_arr = df["sector"].values

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
        df["days_past_due"] = dpd_values

        # Incidents : multi-facteur + bruit multiplicatif log-normal
        margin_stress = np.clip(0.15 - df["ebitda_margin"].values, 0, None) / 0.15
        lam = (
            0.3
            + 0.5 * df["debt_ratio"].values
            + 0.3 * df["utilization_rate"].values
            + 0.2 * margin_stress
        )
        lam *= self.rng.lognormal(0, 0.3, n)
        lam = np.clip(lam, 0.01, None)
        df["nb_incidents_12m"] = self.rng.poisson(lam)

        df["nb_credit_lines"] = self.rng.poisson(3.0, n)

    # -- company_size (vectorise) [v4.2 #20] ----------------------------------

    def _add_company_size(self, df: pd.DataFrame):
        """Taille correlee au revenu. Vectorise (pas de boucle Python).

        v4.5 #46: pentes reduites + bruit additif sur percentile
        pour casser la relation quasi-deterministe revenue->size.
        """
        n = len(df)
        rev_pct = df["revenue"].rank(pct=True).values
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
        df["company_size"] = np.array(self.SIZES)[idx]

    # -- Profils sectoriels ---------------------------------------------------

    def _apply_sector_profiles(self, df: pd.DataFrame):
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

        for sector in self.SECTORS:
            mask = df["sector"].values == sector
            if not mask.any():
                continue

            # 1. Location shift (v4.1)
            shifts = _SECTOR_PROFILES.get(sector, {})
            for feature, shift in shifts.items():
                if feature in df.columns:
                    df.loc[mask, feature] = df.loc[mask, feature] + shift

            # 2. Dispersion scaling autour de la mediane (v4.3)
            scales = _SECTOR_MARGINAL_SCALES.get(sector, {})
            for feature, scale in scales.items():
                if feature in df.columns:
                    vals = df.loc[mask, feature].values
                    med = np.median(vals)
                    df.loc[mask, feature] = med + (vals - med) * scale

            # 3. Zombie rate adjustment (v4.4: monotone quantile warp)
            # Au lieu d'un shift global, on compresse/dilate la partie basse
            # pour atteindre le taux zombie cible sans deplacer les saines
            target_zombie = _SECTOR_ZOMBIE_RATES.get(sector, 0.20)
            margin_vals = df.loc[mask, "ebitda_margin"].values
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
                            df.iloc[chosen, df.columns.get_loc("ebitda_margin")] *= -0.3
                    else:
                        # Besoin de MOINS de zombies: remonter les marges les moins negatives
                        zombie_idxs = np.where(mask)[0][below_zero]
                        zombie_margins = margin_vals[below_zero]
                        n_rescue = int(len(zombie_idxs) * (1 - ratio))
                        if n_rescue > 0:
                            # Remonter les moins negatives (les plus proches de 0)
                            order = np.argsort(-zombie_margins)[:n_rescue]
                            chosen = zombie_idxs[order]
                            df.iloc[chosen, df.columns.get_loc("ebitda_margin")] = np.abs(
                                df.iloc[chosen, df.columns.get_loc("ebitda_margin")]
                            ) + 0.01

        for feature, (lo, hi) in _BOUNDS.items():
            if feature in df.columns:
                df[feature] = np.clip(df[feature].values, lo, hi)

    # -- Temporel + Macro -----------------------------------------------------

    def _add_temporal_macro(self, df: pd.DataFrame, vintage_idx: np.ndarray):
        """Macro liees au vintage + bruit intra-trimestre (v4.3).

        Chaque observation recoit le niveau macro de son vintage PLUS un bruit
        gaussien calibre par variable. Empeche le modele d'apprendre la macro
        comme un simple lookup deterministe sur vintage_quarter.
        """
        n = len(df)
        quarters = pd.period_range("2018Q1", periods=self.n_vintages, freq="Q")
        df["vintage_quarter"] = quarters[vintage_idx].astype(str)
        for feat, values in _MACRO.items():
            base = np.array(values[: self.n_vintages])[vintage_idx]
            sigma = _MACRO_NOISE_SIGMA.get(feat, 0.0)
            if sigma > 0:
                base = base + self.rng.normal(0, sigma, n)
            df[feat] = base

    # -- Features bruit -------------------------------------------------------

    def _add_noise_features(self, df: pd.DataFrame):
        n = len(df)
        df["noise_gaussian"] = self.rng.normal(0, 1, n)
        df["noise_uniform"] = self.rng.uniform(0, 1, n)
        debt_rank = df["debt_ratio"].rank(pct=True).values
        rev_rank = df["revenue"].rank(pct=True).values
        df["noise_correlated_1"] = 0.05 * debt_rank + 0.95 * self.rng.normal(0, 1, n)
        df["noise_correlated_2"] = -0.05 * rev_rank + 0.95 * self.rng.normal(0, 1, n)
        df["noise_shuffled_debt"] = self.rng.permutation(df["debt_ratio"].values)

    # -- DGP structure v4.5 : realiste 50/30/20 monotones/interactions/conj --

    def _structured_dgp(
        self, df: pd.DataFrame, is_stress: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        DGP v4.5 : economiquement realiste, ratio 50/30/20.

        Architecture calibree sur la litterature credit corporate :
        - 50% effets MONOTONES : dette↑=PD↑, marge↑=PD↓, ICR↑=PD↓
          Jamais d'inversion de signe. LogReg capture ~80% de ce signal.
        - 30% INTERACTIONS multiplicatives : debt×vol, margin×ICR, etc.
          LogReg ne capture pas (pas de feature engineering).
        - 20% CONJONCTIFS + SEUILS : zones AND, DPD>60, ICR<1.5.
          Avantage arbres modeste mais reel (~3pp AUC).

        DGP non-stationnaire : coefficients x1.5 en regime stress.
        Ecart LR/XGB attendu : ~2-5pp (realiste portefeuille corporate).
        """
        n = len(df)
        sector_arr = df["sector"].values

        # v4.4: facteur d'amplification en stress (non-stationnarite)
        stress_mult = np.where(is_stress, 1.5, 1.0)

        raw = np.zeros(n)

        debt = df["debt_ratio"].values
        margin = df["ebitda_margin"].values
        vol = df["cf_volatility"].values
        icr = df["interest_coverage_ratio"].values
        util = df["utilization_rate"].values
        nde = df["net_debt_to_ebitda"].values
        dpd = df["days_past_due"].values
        wc = df["working_capital_ratio"].values
        cr = df["current_ratio"].values
        ltv = df["loan_to_value"].values
        log_rev = np.clip(np.log1p(df["revenue"].values / 1e7), 0, 3)
        incidents_capped = np.minimum(df["nb_incidents_12m"].values, 5)
        is_pme = (df["company_size"].values == "PME").astype(float)
        age_yrs = df["account_age_months"].values / 12.0

        # ================================================================
        # BLOC 1 : EFFETS MONOTONES PIECEWISE (~40% du signal)
        #
        # Economiquement realiste : chaque feature a un sens constant
        # (dette↑ = risque↑) mais avec des PALIERS et ACCELERATIONS.
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
        raw += 0.12 * (df["company_size"].values == "PME").astype(float)

        # -- Macro --
        # v4.5 #43: coefficients renforces (sigma reduit => redeviennent visibles)
        raw -= 0.15 * df["gdp_growth"].values
        raw += 0.12 * df["unemployment_rate"].values
        raw += 0.08 * df["interest_rate_10y"].values

        # -- Interactions secteur x features --
        # Deux mecanismes qui creent de la non-linearite sectorielle :
        # 1. Ajustements continus (coefficients differents par secteur)
        # 2. Seuils sectoriels (les seuils ICR, debt, etc. changent par secteur)
        _EFFECT_MAP = {
            "debt_x_vol":       lambda d: d["debt_ratio"].values * d["cf_volatility"].values,
            "log_revenue":      lambda d: np.log1p(d["revenue"].values / 1e6),
            "icr_threshold":    lambda d: (d["interest_coverage_ratio"].values < 1.50).astype(float),
            "margin_x_debt":    lambda d: np.clip(-d["ebitda_margin"].values, 0, 0.5) * d["debt_ratio"].values,
            "tangible_protect": lambda d: d["tangible_assets_ratio"].values,
            "capex_stress":     lambda d: d["capex_to_revenue"].values * np.clip(0.10 - d["ebitda_margin"].values, 0, 0.3),
            "margin_buffer":    lambda d: np.clip(d["ebitda_margin"].values, 0, 0.5),
            "util_high":        lambda d: (d["utilization_rate"].values > 0.90).astype(float),
            "dpd_extra":        lambda d: np.clip(d["days_past_due"].values - 60, 0, 300) / 30,
            "wc_stress":        lambda d: np.clip(-d["working_capital_ratio"].values, 0, 0.5),
            "nde_explosive":    lambda d: np.clip(d["net_debt_to_ebitda"].values - 4.0, 0, 4.0) ** 2,
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
        age_factor = np.clip(1.0 - df["account_age_months"].values / 120, 0, 1.0)
        raw += 0.15 * age_factor * self.rng.standard_normal(n)

        # -- Bruit heteroscedastique Student-t(5) --
        sigma_i = self.noise_sigma * (
            1.0
            + 0.3 * (df["company_size"].values == "PME").astype(float)
            + 0.2 * np.clip(df["debt_ratio"].values - 0.3, 0, 1.0)
            + 0.2 * np.clip(df["cf_volatility"].values - 0.2, 0, 1.0)
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

    def _inject_mnar(self, df: pd.DataFrame):
        """MNAR sur continues ET discretes + composante MCAR (v4.3).

        v4.3 :
          - Taux MNAR varie par observation (lognormal) pour casser la regularite
          - Composante MCAR ~2% sur toutes les numeriques (manquant aleatoire pur)
        v4.5 #45:
          - Guard: missing_rate <= 0 => skip MNAR entierement
          - clip(0, 0.40) au lieu de clip(0.01, 0.40) pour respecter missing_rate=0
        """
        if self.missing_rate <= 0:
            return

        n = len(df)
        is_pme = df["company_size"].values == "PME"

        # v4.3 : taux individuel varie autour de self.missing_rate
        rate_i = self.missing_rate * self.rng.lognormal(0, 0.4, n)
        rate_i = np.clip(rate_i, 0.0, 0.40)  # v4.5: 0.01->0.0 (respecte missing_rate faible)

        # Continues : endettes cachent marge, PME cachent ICR
        high_debt = df["debt_ratio"].values > df["debt_ratio"].quantile(0.70)
        df.loc[(self.rng.random(n) < rate_i) & high_debt, "ebitda_margin"] = np.nan
        df.loc[(self.rng.random(n) < rate_i) & is_pme, "interest_coverage_ratio"] = np.nan
        df.loc[self.rng.random(n) < rate_i * 0.5, "loan_to_value"] = np.nan
        df.loc[self.rng.random(n) < rate_i * 0.3, "tangible_assets_ratio"] = np.nan

        # [v4.2] Discretes : PME sous-rapportent incidents, DPD parfois manquant
        df["nb_incidents_12m"] = df["nb_incidents_12m"].astype(float)
        df["days_past_due"] = df["days_past_due"].astype(float)
        df.loc[(self.rng.random(n) < rate_i * 0.7) & is_pme, "nb_incidents_12m"] = np.nan
        df.loc[self.rng.random(n) < rate_i * 0.3, "days_past_due"] = np.nan

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
            if df[c].dtype in ("float64", "float32", "int64")
            and c not in _MCAR_EXCLUDE
        ]
        for col in num_cols:
            mcar_mask = self.rng.random(n) < mcar_rate
            if mcar_mask.any():
                df.loc[mcar_mask, col] = np.nan

    # -- Pipeline principal [v4.2 #16 : ordre corrige] -------------------------

    def generate(self) -> pd.DataFrame:
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
        df["sector"] = sector_arr
        self._add_company_size(df)

        # 5. Profils sectoriels (shifts sur marginales)
        self._apply_sector_profiles(df)

        # 6. Features discretes (APRES profils -- incidents utilisent marge shiftee)
        self._add_discrete_features(df, dpd_latent)

        # 7. Temporal + macro
        self._add_temporal_macro(df, vintage_idx)

        # 8. Bruit
        self._add_noise_features(df)

        # 9. DGP (v4.4: passe is_stress pour non-stationnarite)
        df[TARGET], df[LATENT_PD] = self._structured_dgp(df, is_stress)

        # 10. MNAR
        self._inject_mnar(df)

        return df


# ===========================================================================
# Evaluator
# ===========================================================================

class RobustEvaluator:
    """
    Evaluation multi-modele :
        - Split temporel (par vintage) OU StratifiedKFold
        - Baseline LogReg + RandomForest + XGBoost
        - Class weighting + OneHotEncoder
        - Metriques : AUC, Average Precision, Brier Score
        - Null-feature importance test
    """

    def __init__(self, df: pd.DataFrame, n_folds: int = 5, temporal_split: bool = True):
        self.df = df.copy()
        self.n_folds = n_folds
        self.temporal_split = temporal_split

        self._meta_cols = [TARGET, LATENT_PD, "vintage_quarter"]
        self._cat_cols = ["sector", "company_size"]
        self._num_cols = [
            c for c in df.columns
            if c not in self._meta_cols + self._cat_cols
            and df[c].dtype != "object"
        ]

    def _preprocessor(self):
        return ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), self._num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self._cat_cols),
        ])

    def _get_models(self, n_pos: int, n_neg: int) -> dict:
        models = {
            "LogReg (baseline)": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=15, class_weight="balanced",
                n_jobs=-1, random_state=42,
            ),
        }
        if _HAS_XGB:
            models["XGBoost"] = XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                scale_pos_weight=min(n_neg / max(n_pos, 1), 10),
                eval_metric="logloss", random_state=42, verbosity=0,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            )
        return models

    def evaluate(self):
        df = self.df
        X = df.drop(columns=[c for c in self._meta_cols if c in df.columns])
        y = df[TARGET]
        n_pos, n_neg = (y == 1).sum(), (y == 0).sum()

        if self.temporal_split and "vintage_quarter" in df.columns:
            vintages = sorted(df["vintage_quarter"].unique())
            split_pt = int(0.70 * len(vintages))
            train_set = set(vintages[:split_pt])
            train_mask = df["vintage_quarter"].isin(train_set)
            folds = [(X[train_mask], y[train_mask], X[~train_mask], y[~train_mask])]
            print(
                f"\n>>> SPLIT TEMPOREL : train {vintages[0]}..{vintages[split_pt-1]}"
                f" | test {vintages[split_pt]}..{vintages[-1]}"
            )
        else:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            folds = [
                (X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te])
                for tr, te in skf.split(X, y)
            ]
            print(f"\n>>> {self.n_folds}-FOLD STRATIFIED CV")

        models = self._get_models(n_pos, n_neg)
        results = {name: {"auc": [], "ap": [], "brier": []} for name in models}

        for X_tr, y_tr, X_te, y_te in folds:
            for name, model in models.items():
                pipe = Pipeline([("prep", self._preprocessor()), ("clf", model)])
                pipe.fit(X_tr, y_tr)
                probs = pipe.predict_proba(X_te)[:, 1]
                results[name]["auc"].append(roc_auc_score(y_te, probs))
                results[name]["ap"].append(average_precision_score(y_te, probs))
                results[name]["brier"].append(brier_score_loss(y_te, probs))

        multi = len(folds) > 1
        print(f"\n{'Model':<20} {'AUC':>12} {'Avg Prec':>12} {'Brier':>12}")
        print("-" * 58)
        for name in models:
            r = results[name]
            if multi:
                print(
                    f"{name:<20} "
                    f"{np.mean(r['auc']):.4f}+/-{np.std(r['auc']):.3f} "
                    f"{np.mean(r['ap']):.4f}+/-{np.std(r['ap']):.3f} "
                    f"{np.mean(r['brier']):.4f}+/-{np.std(r['brier']):.3f}"
                )
            else:
                print(
                    f"{name:<20} "
                    f"{r['auc'][0]:.4f}        "
                    f"{r['ap'][0]:.4f}        "
                    f"{r['brier'][0]:.4f}"
                )

        self._noise_feature_test(X, y)

    def _noise_feature_test(self, X: pd.DataFrame, y: pd.Series):
        # max_depth adapte a la taille : profondeur suffisante pour
        # capturer les interactions 2-3 way mais pas trop pour eviter
        # de surestimer l'importance du bruit
        n = len(X)
        depth = 8 if n < 100_000 else 10 if n < 500_000 else 12
        pipe = Pipeline([
            ("prep", self._preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=depth, class_weight="balanced",
                n_jobs=-1, random_state=42,
            )),
        ])
        pipe.fit(X, y)

        num_names = list(self._num_cols)
        cat_names = list(
            pipe.named_steps["prep"].transformers_[1][1].get_feature_names_out()
        )
        all_names = num_names + cat_names
        importances = pipe.named_steps["clf"].feature_importances_
        sorted_feats = sorted(zip(all_names, importances), key=lambda x: x[1], reverse=True)

        noise_set = set(NOISE_FEATURES)

        print(f"\n>>> FEATURE IMPORTANCE (Top 10 + bruit)")
        print(f"{'Feature':<35} {'Importance':>12} {'Type':>8}")
        print("-" * 58)

        shown_noise = set()
        for rank, (name, imp) in enumerate(sorted_feats):
            if rank < 10:
                tag = "BRUIT" if name in noise_set else "signal"
                print(f"  {name:<33} {imp:.4f}         {tag}")
                if name in noise_set:
                    shown_noise.add(name)

        remaining_noise = noise_set - shown_noise
        if remaining_noise:
            print("  ...")
            for rank, (name, imp) in enumerate(sorted_feats):
                if name in remaining_noise:
                    print(f"  {name:<33} {imp:.4f}         BRUIT  (rang {rank+1}/{len(sorted_feats)})")

        noise_imp = sum(imp for name, imp in sorted_feats if name in noise_set)
        max_noise = max((imp for name, imp in sorted_feats if name in noise_set), default=0)
        n_signal_below = sum(
            1 for name, imp in sorted_feats
            if name not in noise_set and imp < max_noise
        )

        print(f"\n  Importance totale signal : {1 - noise_imp:.4f}")
        print(f"  Importance totale bruit  : {noise_imp:.4f}")
        print(f"  Max importance bruit     : {max_noise:.4f}")
        print(f"  Features signal sous le max bruit : {n_signal_below}")

        if noise_imp < 0.05:
            print("  [OK] Bruit correctement marginalise (<5% importance totale)")
        elif noise_imp < 0.10:
            print("  [WARN] Bruit un peu eleve (5-10%)")
        else:
            print("  [FAIL] Bruit significatif (>10%) -- overfitting")


# ===========================================================================
# Bridge v4.5 → Pipeline IFRS 9
# ===========================================================================
#
# La fonction generate_dataset() wrap le DGP v4.5 et produit un output
# 100% compatible avec le cockpit IFRS 9, SANS modifier le DGP.
# Les colonnes passives (credit_score, loan_amount, collateral, etc.)
# sont generees APRES le DGP avec un RNG separe pour ne pas perturber
# la calibration.
# ===========================================================================

# Mapping secteurs v4 → production config.py
_SECTOR_MAP: dict[str, str] = {
    "Tech": "Technologie",
    "Industrie": "Industrie",
    "Sante": "Sante",
    "Retail": "Services",
    "Energie": "Immobilier",
}

# Credit score par secteur production (mu, sigma)
_BRIDGE_CREDIT_SCORE: dict[str, tuple[int, int]] = {
    "Technologie": (620, 80),
    "Industrie": (660, 60),
    "Sante": (700, 50),
    "Immobilier": (650, 65),
    "Services": (670, 55),
}

# Probabilite revolving par secteur production
_BRIDGE_REVOLVING_PROB: dict[str, float] = {
    "Technologie": 0.40,
    "Industrie": 0.25,
    "Sante": 0.20,
    "Immobilier": 0.15,
    "Services": 0.30,
}

# Fourchette collateral ratio par secteur production
_BRIDGE_COLLATERAL_RATIO: dict[str, tuple[float, float]] = {
    "Technologie": (0.15, 0.50),
    "Industrie": (0.30, 0.70),
    "Sante": (0.25, 0.60),
    "Immobilier": (0.60, 1.00),
    "Services": (0.20, 0.55),
}

# Leverage PE par secteur production
_BRIDGE_LEVERAGE_PE: dict[str, tuple[float, float]] = {
    "Technologie": (0.10, 0.40),
    "Industrie": (0.30, 0.60),
    "Sante": (0.20, 0.50),
    "Immobilier": (0.50, 0.70),
    "Services": (0.20, 0.50),
}


def generate_dataset(
    n_clients: int = 30_000,
    seed: int = 42,
    noise_sigma: float = 7.0,
    base_default_rate: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bridge v4.5 → pipeline IFRS 9.

    Genere les donnees via le DGP v4.5 puis transforme le DataFrame brut
    en 3-tuple (df_credit, df_pe, df_history) compatible avec le cockpit.

    Le DGP n'est PAS modifie : les colonnes passives sont calculees APRES
    avec un RNG separe (seed + 100/200/300).

    Args:
        n_clients: Nombre d'entreprises.
        seed: Graine aleatoire.
        noise_sigma: Ecart-type bruit DGP (7.0 → AUC ~0.82).
        base_default_rate: Taux de defaut cible.

    Returns:
        Tuple (df_credit, df_pe, df_history).
    """
    # Lazy import config to avoid circular dependencies at module level
    from ifrs9_cockpit.config import (
        ALLOWED_SECTORS,
        MACRO_HISTORY_BASELINE as _MACRO_HIST,
        REQUIRED_CREDIT_COLS,
        REQUIRED_PE_COLS,
        SECTORS as _PROD_SECTORS,
    )

    # ── 1. Run DGP v4.5 (untouched) ──
    gen = AdvancedFinancialGenerator(
        n_rows=n_clients,
        seed=seed,
        noise_sigma=noise_sigma,
        base_default_rate=base_default_rate,
        missing_rate=0.0,  # no MNAR for production pipeline
    )
    df_raw = gen.generate()
    n = len(df_raw)

    # ── 2. Map sectors v4 → production ──
    df_raw["sector"] = df_raw["sector"].map(_SECTOR_MAP)

    # ── 3. Rename columns ──
    df_raw.rename(columns={
        "target_default": "default_flag",
        "days_past_due": "dpd",
        "pd_latent": "pd_origination",
    }, inplace=True)

    # ── 4. Add enterprise_id ──
    df_raw["enterprise_id"] = np.arange(n)

    # ── 5. Generate passive credit columns (separate RNG) ──
    rng_credit = np.random.default_rng(seed + 100)
    sectors = df_raw["sector"].values

    # EBITDA = revenue × ebitda_margin
    df_raw["ebitda"] = df_raw["revenue"].values * df_raw["ebitda_margin"].values

    # Credit score — normale tronquee par secteur
    credit_score = np.empty(n)
    for sector_name, (mu, sigma) in _BRIDGE_CREDIT_SCORE.items():
        mask = sectors == sector_name
        count = mask.sum()
        if count > 0:
            raw_cs = rng_credit.normal(mu, sigma, count)
            credit_score[mask] = np.clip(raw_cs, 300, 850).astype(int)
    df_raw["credit_score"] = credit_score.astype(int)

    # Loan type: Revolving / Term
    loan_type = np.full(n, "Term", dtype=object)
    for sector_name, prob in _BRIDGE_REVOLVING_PROB.items():
        mask = sectors == sector_name
        count = mask.sum()
        if count > 0:
            is_revolving = rng_credit.random(count) < prob
            loan_type[mask] = np.where(is_revolving, "Revolving", "Term")
    df_raw["loan_type"] = loan_type

    # Loan amount — revenue × multiplier (differentie par loan_type)
    revenue_arr = df_raw["revenue"].values
    multiplier = np.where(
        loan_type == "Revolving",
        rng_credit.uniform(0.3, 1.5, n),
        rng_credit.uniform(1.0, 4.0, n),
    )
    # v4 revenue is already in raw EUR (not millions), align with production
    # Production generator has revenue in millions, loan_amount = rev_M × mult × 1e6
    # v4 generator has revenue in raw EUR → loan_amount = revenue × mult
    df_raw["loan_amount"] = np.round(revenue_arr * multiplier, 2)

    # Collateral — loan_amount × ratio by sector
    collateral = np.empty(n)
    for sector_name, (lo, hi) in _BRIDGE_COLLATERAL_RATIO.items():
        mask = sectors == sector_name
        count = mask.sum()
        if count > 0:
            ratio = rng_credit.uniform(lo, hi, count)
            collateral[mask] = df_raw["loan_amount"].values[mask] * ratio
    df_raw["collateral"] = np.round(collateral, 2)

    # Engineered features
    df_raw["loan_to_revenue"] = np.round(
        df_raw["loan_amount"].values / np.clip(df_raw["revenue"].values, 1.0, None), 4,
    )
    df_raw["collateral_coverage"] = np.round(
        df_raw["collateral"].values / np.clip(df_raw["loan_amount"].values, 1.0, None), 4,
    )

    # ── 6. Build df_credit ──
    credit_cols = [
        "enterprise_id", "sector", "revenue", "ebitda", "debt_ratio",
        "credit_score", "dpd", "collateral", "loan_amount",
        "utilization_rate", "default_flag", "pd_origination",
        "loan_type", "loan_to_revenue", "collateral_coverage",
    ]
    df_credit = df_raw[credit_cols].copy()

    # ── 7. Build df_pe (separate RNG) ──
    rng_pe = np.random.default_rng(seed + 200)
    sector_config_map = {s.name: s for s in _PROD_SECTORS}
    valuation_map = {s.name: s.valuation_method for s in _PROD_SECTORS}

    entry_multiple = np.empty(n)
    leverage_pe = np.empty(n)
    for sector_name in ALLOWED_SECTORS:
        mask = sectors == sector_name
        count = mask.sum()
        if count == 0:
            continue
        sc = sector_config_map[sector_name]
        entry_multiple[mask] = rng_pe.uniform(
            sc.entry_multiple_range[0], sc.entry_multiple_range[1], count,
        )
        lo, hi = _BRIDGE_LEVERAGE_PE[sector_name]
        leverage_pe[mask] = rng_pe.uniform(lo, hi, count)

    vintage = rng_pe.integers(2018, 2026, size=n)

    # EBITDA PE: floor a 5% du revenue (on n'investit pas en PE dans une
    # entreprise a EBITDA negatif — les zombies credit ne sont pas en PE)
    ebitda_pe = np.maximum(df_raw["ebitda"].values, df_raw["revenue"].values * 0.05)

    df_pe = pd.DataFrame({
        "enterprise_id": np.arange(n),
        "sector": sectors,
        "revenue": df_raw["revenue"].values,
        "ebitda": np.round(ebitda_pe, 2),
        "entry_multiple": np.round(entry_multiple, 2),
        "leverage": np.round(leverage_pe, 4),
        "vintage": vintage,
        "holding_years": 2026 - vintage,
        "valuation_method": pd.Series(sectors).map(valuation_map).values,
    })

    # ── 8. Build df_history (separate RNG) ──
    rng_hist = np.random.default_rng(seed + 300)
    m = 12
    total = n * m

    enterprise_ids_h = np.repeat(np.arange(n), m)
    months = np.tile(np.arange(1, m + 1), n)

    # Macro: tile baseline 12 months for each enterprise
    # MACRO_HISTORY_BASELINE may have more than 12 months (e.g. 60 for covariance),
    # so we take only the last 12 months for the per-enterprise history.
    macro_gdp = np.tile(_MACRO_HIST["gdp_growth"][-m:], n)
    macro_unemp = np.tile(_MACRO_HIST["unemployment_rate"][-m:], n)
    macro_interest = np.tile(_MACRO_HIST["interest_rate"][-m:], n)
    macro_hpi = np.tile(_MACRO_HIST["hpi_growth"][-m:], n)
    macro_inflation = np.tile(_MACRO_HIST["inflation_rate"][-m:], n)

    # Balance: random walk around loan_amount (±5%)
    loan_amounts_h = np.repeat(df_raw["loan_amount"].values, m)
    balance_noise = rng_hist.normal(0, 0.05, total)
    balances = np.maximum(0, loan_amounts_h * (0.85 + balance_noise))

    # DPD mensuel — correle au credit_score
    credit_scores_h = np.repeat(df_raw["credit_score"].values, m)
    dpd_probs = np.clip((650 - credit_scores_h) / 1000, 0, 0.3)
    has_dpd_h = rng_hist.random(total) < dpd_probs

    dpd_choices = np.array([15, 30, 60, 90, 120])
    dpd_choice_probs = np.array([0.40, 0.30, 0.15, 0.10, 0.05])
    dpd_values_h = np.where(
        has_dpd_h,
        rng_hist.choice(dpd_choices, size=total, p=dpd_choice_probs),
        0,
    )

    df_history = pd.DataFrame({
        "enterprise_id": enterprise_ids_h,
        "month": months,
        "balance": np.round(balances, 2),
        "dpd": dpd_values_h.astype(int),
        "gdp_growth": macro_gdp,
        "unemployment_rate": macro_unemp,
        "interest_rate": macro_interest,
        "hpi_growth": macro_hpi,
        "inflation_rate": macro_inflation,
    })

    # ── 9. Validate contracts ──
    assert REQUIRED_CREDIT_COLS <= set(df_credit.columns), (
        f"Missing credit cols: {REQUIRED_CREDIT_COLS - set(df_credit.columns)}"
    )
    assert REQUIRED_PE_COLS <= set(df_pe.columns), (
        f"Missing PE cols: {REQUIRED_PE_COLS - set(df_pe.columns)}"
    )
    assert df_history.shape[0] == n * 12, (
        f"History shape mismatch: {df_history.shape[0]} != {n * 12}"
    )
    assert set(df_credit["sector"].unique()) <= ALLOWED_SECTORS, (
        f"Invalid sectors: {set(df_credit['sector'].unique()) - ALLOWED_SECTORS}"
    )
    assert set(df_credit["loan_type"].unique()) <= {"Revolving", "Term"}, (
        f"Invalid loan types: {set(df_credit['loan_type'].unique())}"
    )

    return df_credit, df_pe, df_history


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Synthetic Financial Data Generator v4.5")
    print("=" * 65)

    gen = AdvancedFinancialGenerator(
        n_rows=1_500_000,
        degrees_of_freedom=5,
        base_default_rate=0.05,     # 5% -> ~75K defauts sur 1.5M
        noise_sigma=7.0,            # AUC ~0.82 (LR ~0.815, XGB ~0.819)
        stress_df=3,
        stress_corr_amplification=1.3,
        missing_rate=0.10,
    )
    df = gen.generate()

    print(f"\nDataset : {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
    print(f"Taux de defaut : {df[TARGET].mean():.2%}")
    n_missing = df.isnull().sum().sum()
    pct_missing = df.isnull().mean().mean() * 100
    print(f"Valeurs manquantes : {n_missing:,} ({pct_missing:.1f}% par colonne en moy.)")
    print(f"Vintages : {df['vintage_quarter'].nunique()} trimestres")

    # Distribution sectorielle
    print(f"\nDistribution sectorielle :")
    for sector, count in df["sector"].value_counts().items():
        dr = df.loc[df["sector"] == sector, TARGET].mean()
        print(f"  {sector:<12} {count:>9,} ({count/len(df)*100:.1f}%)  defaut={dr:.2%}")

    # Default rate par regime
    quarters_sorted = sorted(df["vintage_quarter"].unique())
    stress_labels = {quarters_sorted[i] for i in gen.stress_quarters if i < len(quarters_sorted)}
    dr_stress = df.loc[df["vintage_quarter"].isin(stress_labels), TARGET].mean()
    dr_normal = df.loc[~df["vintage_quarter"].isin(stress_labels), TARGET].mean()
    print(f"\nTaux defaut normal : {dr_normal:.2%} | stress : {dr_stress:.2%}")

    print(f"\n>>> DGP -- {len(gen.DGP_EFFECTS)} effets a retrouver :")
    for effect, desc in gen.DGP_EFFECTS.items():
        print(f"  {effect:<30} {desc}")

    # Evaluation
    evaluator = RobustEvaluator(df, temporal_split=True)
    evaluator.evaluate()

    print("\n" + "=" * 65)
    evaluator_cv = RobustEvaluator(df, n_folds=5, temporal_split=False)
    evaluator_cv.evaluate()
