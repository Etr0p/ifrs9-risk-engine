"""Configuration centrale du Cockpit IFRS 9.

Regroupe tous les hyperparamètres, constantes métier, chemins
et paramètres de scénarios utilisés à travers le projet.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ──────────────────────────────────────────────
# REPRODUCTIBILITÉ
# ──────────────────────────────────────────────
RANDOM_SEED: int = 42

# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
N_CLIENTS: int = 10_000
N_MONTHS: int = 12
TRAIN_RATIO: float = 0.7
VALIDATION_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ──────────────────────────────────────────────
# SEGMENTS CLIENTS
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class SegmentConfig:
    """Configuration d'un segment client.

    Attributes:
        name: Nom du segment.
        age_range: Tuple (min, max) de la tranche d'âge.
        proportion: Part du segment dans la population totale.
        base_default_rate: Taux de défaut de base du segment.
        unemployment_sensitivity: Sensibilité au chômage (multiplicateur).
        gdp_sensitivity: Sensibilité au PIB (multiplicateur).
        interest_rate_sensitivity: Sensibilité au taux directeur BCE.
        hpi_sensitivity: Sensibilité aux prix immobiliers.
        inflation_sensitivity: Sensibilité à l'inflation (IPC).
        income_range: Tuple (min, max) du revenu annuel en euros.
        avg_credit_score: Score de crédit moyen du segment.
    """
    name: str
    age_range: Tuple[int, int]
    proportion: float
    base_default_rate: float
    unemployment_sensitivity: float
    gdp_sensitivity: float
    interest_rate_sensitivity: float
    hpi_sensitivity: float
    inflation_sensitivity: float
    income_range: Tuple[int, int]
    avg_credit_score: int


SEGMENTS: List[SegmentConfig] = [
    SegmentConfig(
        name="Jeunes_Actifs",
        age_range=(22, 30),
        proportion=0.25,
        base_default_rate=0.08,
        unemployment_sensitivity=2.5,   # Très sensibles au chômage
        gdp_sensitivity=1.8,
        interest_rate_sensitivity=1.5,  # Endettés, sensibles aux taux
        hpi_sensitivity=1.0,            # Peu de patrimoine immobilier
        inflation_sensitivity=2.0,      # Revenus faibles, peu d'épargne tampon
        income_range=(18_000, 35_000),
        avg_credit_score=580,
    ),
    SegmentConfig(
        name="Mid_Career",
        age_range=(31, 50),
        proportion=0.45,
        base_default_rate=0.035,
        unemployment_sensitivity=1.0,
        gdp_sensitivity=1.0,
        interest_rate_sensitivity=1.0,  # Sensibilité moyenne
        hpi_sensitivity=1.0,            # Propriétaires avec equity
        inflation_sensitivity=1.0,      # Référence
        income_range=(35_000, 75_000),
        avg_credit_score=680,
    ),
    SegmentConfig(
        name="Seniors",
        age_range=(51, 67),
        proportion=0.20,
        base_default_rate=0.025,
        unemployment_sensitivity=0.6,
        gdp_sensitivity=0.5,
        interest_rate_sensitivity=0.3,  # Peu de dette, patrimoine constitué
        hpi_sensitivity=0.5,            # Equity accumulée, LTV faible
        inflation_sensitivity=0.6,      # Pensions partiellement indexées
        income_range=(40_000, 90_000),
        avg_credit_score=720,
    ),
    SegmentConfig(
        name="Primo_Accedants",
        age_range=(25, 35),
        proportion=0.10,
        base_default_rate=0.06,
        unemployment_sensitivity=2.0,
        gdp_sensitivity=1.5,
        interest_rate_sensitivity=2.0,  # Très exposés (achat récent, LTV élevée)
        hpi_sensitivity=1.8,            # Negative equity risk si HPI baisse
        inflation_sensitivity=1.5,      # Revenus modestes, charges immobilières lourdes
        income_range=(22_000, 40_000),
        avg_credit_score=600,
    ),
]

# ──────────────────────────────────────────────
# VARIABLES MACROÉCONOMIQUES
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class MacroScenario:
    """Paramètres d'un scénario macroéconomique.

    Attributes:
        name: Nom du scénario.
        weight: Pondération dans le calcul ECL.
        gdp_growth: Croissance du PIB annualisée (%).
        unemployment_rate: Taux de chômage (%).
        interest_rate: Taux directeur BCE (%).
        hpi_growth: Variation annuelle des prix immobiliers (%).
        inflation_rate: Inflation annuelle IPC (%).
        gdp_shock: Choc PIB appliqué au défaut.
        unemployment_shock: Choc chômage appliqué au défaut.
        interest_rate_shock: Choc taux directeur appliqué au défaut.
        hpi_shock: Choc prix immobiliers appliqué à la LGD.
        inflation_shock: Choc inflation appliqué au défaut.
    """
    name: str
    weight: float
    gdp_growth: float
    unemployment_rate: float
    interest_rate: float
    hpi_growth: float
    inflation_rate: float
    gdp_shock: float
    unemployment_shock: float
    interest_rate_shock: float
    hpi_shock: float
    inflation_shock: float


SCENARIO_BASE = MacroScenario(
    name="Base",
    weight=0.50,
    gdp_growth=1.2,
    unemployment_rate=7.5,
    interest_rate=3.5,
    hpi_growth=2.0,
    inflation_rate=2.5,
    gdp_shock=0.0,
    unemployment_shock=0.0,
    interest_rate_shock=0.0,
    hpi_shock=0.0,
    inflation_shock=0.0,
)

SCENARIO_ADVERSE = MacroScenario(
    name="Adverse",
    weight=0.25,
    gdp_growth=-1.5,
    unemployment_rate=10.5,
    interest_rate=5.0,
    hpi_growth=-8.0,
    inflation_rate=5.5,
    gdp_shock=0.03,
    unemployment_shock=0.05,
    interest_rate_shock=0.02,
    hpi_shock=0.04,
    inflation_shock=0.02,
)

SCENARIO_FAVORABLE = MacroScenario(
    name="Favorable",
    weight=0.25,
    gdp_growth=2.5,
    unemployment_rate=5.5,
    interest_rate=2.0,
    hpi_growth=5.0,
    inflation_rate=1.5,
    gdp_shock=-0.01,
    unemployment_shock=-0.02,
    interest_rate_shock=-0.01,
    hpi_shock=-0.02,
    inflation_shock=-0.01,
)

SCENARIOS: List[MacroScenario] = [SCENARIO_BASE, SCENARIO_ADVERSE, SCENARIO_FAVORABLE]

# Historique macro mensuel (baseline) — 12 mois
MACRO_HISTORY_BASELINE: Dict[str, List[float]] = {
    "gdp_growth": [1.1, 1.0, 1.2, 1.3, 1.1, 0.9, 1.0, 1.2, 1.4, 1.3, 1.2, 1.2],
    "unemployment_rate": [7.8, 7.7, 7.6, 7.5, 7.5, 7.6, 7.7, 7.5, 7.4, 7.3, 7.4, 7.5],
    "interest_rate": [3.0, 3.0, 3.25, 3.25, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5],
    "hpi_growth": [3.0, 2.8, 2.5, 2.3, 2.0, 2.0, 1.8, 2.0, 2.2, 2.0, 2.0, 2.0],
    "inflation_rate": [3.2, 3.0, 2.8, 2.7, 2.6, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
}

# ──────────────────────────────────────────────
# MODÈLES PD
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class PDModelConfig:
    """Configuration des modèles de Probabilité de Défaut.

    Attributes:
        n_woe_bins: Nombre de bins pour le WoE binning.
        lr_C: Régularisation Logistic Regression.
        lr_max_iter: Itérations max LR.
        rf_n_estimators: Nombre d'arbres Random Forest.
        rf_max_depth: Profondeur max RF.
        xgb_n_estimators: Nombre d'arbres XGBoost.
        xgb_max_depth: Profondeur max XGB.
        xgb_learning_rate: Learning rate XGB.
        calibration_method: Méthode de calibration ('isotonic' ou 'sigmoid').
    """
    n_woe_bins: int = 10
    lr_C: float = 1.0
    lr_max_iter: int = 1000
    rf_n_estimators: int = 200
    rf_max_depth: int = 6
    rf_min_samples_leaf: int = 50
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    calibration_method: str = "isotonic"


PD_CONFIG = PDModelConfig()

# ──────────────────────────────────────────────
# MODÈLE LGD
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class LGDConfig:
    """Configuration du modèle Loss Given Default.

    Attributes:
        lgd_ttc_mean: LGD Through-The-Cycle moyenne.
        lgd_ttc_std: Écart-type LGD TTC.
        downturn_add_on: Add-on pour LGD Downturn (%).
        recovery_rate_floor: Plancher du taux de recouvrement.
        cure_rate: Taux de cure (sortie de défaut).
    """
    lgd_ttc_mean: float = 0.35
    lgd_ttc_std: float = 0.15
    downturn_add_on: float = 0.10
    recovery_rate_floor: float = 0.05
    cure_rate: float = 0.15


LGD_CONFIG = LGDConfig()

# ──────────────────────────────────────────────
# MODÈLE EAD
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class EADConfig:
    """Configuration du modèle Exposure At Default.

    Attributes:
        ccf_revolving: Credit Conversion Factor pour lignes revolving.
        ccf_term_loan: CCF pour prêts à terme.
        utilization_draw_stress: Stress sur le tirage en cas de défaut.
    """
    ccf_revolving: float = 0.75
    ccf_term_loan: float = 1.0
    utilization_draw_stress: float = 0.20


EAD_CONFIG = EADConfig()

# ──────────────────────────────────────────────
# MOTEUR IFRS 9
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class IFRS9Config:
    """Configuration du moteur IFRS 9.

    Attributes:
        sicr_threshold_multiplier: Multiplicateur PD pour déclenchement SICR.
        stage3_dpd_threshold: Jours de retard pour passage en Stage 3.
        stage3_pd_threshold: PD seuil pour passage en Stage 3.
        discount_rate: Taux d'actualisation annuel (EIR proxy).
        lifetime_horizon_years: Horizon lifetime pour Stage 2/3 (années).
    """
    sicr_threshold_multiplier: float = 2.0
    stage3_dpd_threshold: int = 90
    stage3_pd_threshold: float = 0.30
    discount_rate: float = 0.045
    lifetime_horizon_years: int = 3


IFRS9_CONFIG = IFRS9Config()

# ──────────────────────────────────────────────
# VIRTUAL CRO — SEUILS D'ALERTE
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class CROAlertConfig:
    """Seuils d'alerte pour le module Virtual CRO.

    Attributes:
        ecl_variation_alert: Variation ECL déclenchant une alerte (%).
        stage2_warning_pct: Part Stage 2 déclenchant un warning (%).
        psi_drift_threshold: PSI seuil pour alerte de drift.
        concentration_threshold: Seuil de concentration sectorielle (%).
    """
    ecl_variation_alert: float = 0.15
    stage2_warning_pct: float = 0.20
    psi_drift_threshold: float = 0.15
    concentration_threshold: float = 0.30


CRO_CONFIG = CROAlertConfig()

# ──────────────────────────────────────────────
# DASHBOARD
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class DashboardConfig:
    """Configuration du dashboard Streamlit.

    Attributes:
        page_title: Titre de la page.
        page_icon: Icône de la page.
        layout: Layout Streamlit ('wide' ou 'centered').
        theme_primary: Couleur primaire hex.
        theme_secondary: Couleur secondaire hex.
        theme_accent: Couleur d'accent hex.
        theme_bg_dark: Fond sombre hex.
        theme_bg_card: Fond de carte hex.
        theme_text: Couleur texte principal hex.
        theme_text_muted: Couleur texte secondaire hex.
        stress_unemployment_range: Range slider chômage (min, max, step).
        stress_gdp_range: Range slider PIB (min, max, step).
        stress_interest_rate_range: Range slider taux directeur (min, max, step).
        stress_hpi_range: Range slider prix immobiliers (min, max, step).
        stress_inflation_range: Range slider inflation (min, max, step).
    """
    page_title: str = "IFRS 9 Risk Cockpit"
    page_icon: str = "\u0024"
    layout: str = "wide"
    theme_primary: str = "#6366F1"
    theme_secondary: str = "#8B5CF6"
    theme_accent: str = "#06D6A0"
    theme_bg_dark: str = "#0F172A"
    theme_bg_card: str = "#1E293B"
    theme_text: str = "#F8FAFC"
    theme_text_muted: str = "#94A3B8"
    stress_unemployment_range: Tuple[float, float, float] = (5.0, 15.0, 0.5)
    stress_gdp_range: Tuple[float, float, float] = (-5.0, 5.0, 0.5)
    stress_interest_rate_range: Tuple[float, float, float] = (0.0, 7.0, 0.25)
    stress_hpi_range: Tuple[float, float, float] = (-15.0, 10.0, 0.5)
    stress_inflation_range: Tuple[float, float, float] = (0.0, 8.0, 0.5)


DASHBOARD_CONFIG = DashboardConfig()

# ──────────────────────────────────────────────
# FEATURES
# ──────────────────────────────────────────────

NUMERICAL_FEATURES: List[str] = [
    "age",
    "income",
    "debt_ratio",
    "credit_score",
    "employment_duration",
    "loan_amount",
    "utilization_rate",
    "nb_past_due_30d",
    "months_since_last_delinquency",
    "nb_credit_lines",
]

CATEGORICAL_FEATURES: List[str] = [
    "segment",
    "loan_type",
    "employment_type",
]

TARGET: str = "default_flag"
