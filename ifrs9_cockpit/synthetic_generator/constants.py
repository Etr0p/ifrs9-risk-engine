"""
Constants for Synthetic Financial Data Generator v4.5.
"""

import numpy as np


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
# chaque secteur sans feature engineering (sector x debt).
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


# Bridge v4.5 -> Pipeline IFRS 9 constants

# Mapping secteurs v4 -> production config.py
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
