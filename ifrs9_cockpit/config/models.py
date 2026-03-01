"""Configuration — modeles PD/LGD/EAD, IFRS9, SICR, contrats de donnees."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ──────────────────────────────────────────────
# MODELES PD
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class PDModelConfig:
    """Configuration des modeles de Probabilite de Defaut.

    3 familles genuinement differentes :
        - LR_WoE : lineaire (frontiere convexe, interpretable)
        - TabNet : deep learning tabulaire (attention sequentielle, Sparsemax)
        - XGBoost : ensemble d'arbres (frontiere en escalier)

    Attributes:
        n_woe_bins: Nombre de bins pour le WoE binning.
        lr_C: Regularisation Logistic Regression.
        lr_max_iter: Iterations max LR.
        tabnet_n_d: Largeur couche decision TabNet (64 = ~120k params).
        tabnet_n_a: Largeur couche attention TabNet (64 = Gold Standard).
        tabnet_n_steps: Etapes d'attention sequentielle.
        tabnet_gamma: Coefficient relaxation reutilisation features.
        tabnet_lambda_sparse: Penalite sparsite.
        tabnet_lr: Learning rate Adam.
        tabnet_batch_size: Taille batch (8192 pour 1M lignes).
        tabnet_virtual_batch_size: Ghost BN virtual batch.
        tabnet_max_epochs: Epoques max (200 avec patience 20).
        tabnet_patience: Early stopping patience.
        xgb_n_estimators: Nombre d'arbres XGBoost.
        xgb_max_depth: Profondeur max XGB.
        xgb_learning_rate: Learning rate XGB.
        xgb_subsample: Sous-echantillonnage XGB.
        calibration_method: Methode de calibration ('isotonic' ou 'sigmoid').
        iv_min_threshold: Seuil IV minimum pour selection des features (0.02 = non predictif).
        vif_max_threshold: Seuil VIF maximum pour filtrage multicolinearite (5.0 = standard).
            VIF > 5 indique une multicolinearite forte. Applique avant la contrainte beta < 0.
        woe_epsilon: Lissage Laplace pour WoE (evite ln(0) si bin vide).
        min_events_per_bin: Nombre minimum de defauts par bin WoE (robustesse statistique).
        pdo: Points to Double the Odds pour le scaling scorecard.
        target_score: Score cible au point d'ancrage (odds = target_odds).
        target_odds: Ratio de cotes au target_score.
    """

    n_woe_bins: int = 10
    lr_C: float = 1.0
    lr_max_iter: int = 1000
    # --- TabNet Gold Standard (1M lignes / ~120k params) ---
    tabnet_n_d: int = 64
    tabnet_n_a: int = 64
    tabnet_n_steps: int = 7
    tabnet_gamma: float = 1.3
    tabnet_lambda_sparse: float = 1e-3
    tabnet_lr: float = 0.02
    tabnet_batch_size: int = 8192
    tabnet_virtual_batch_size: int = 512
    tabnet_max_epochs: int = 200
    tabnet_patience: int = 20
    # --- TabNet Light (50k lignes / ~8k params, consumer/mortgage) ---
    tabnet_light_n_d: int = 16
    tabnet_light_n_a: int = 16
    tabnet_light_n_steps: int = 3
    tabnet_light_gamma: float = 1.5
    tabnet_light_lambda_sparse: float = 1e-2
    tabnet_light_lr: float = 0.025
    tabnet_light_batch_size: int = 2048
    tabnet_light_virtual_batch_size: int = 256
    tabnet_light_max_epochs: int = 100
    tabnet_light_patience: int = 15
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: int = 3
    xgb_reg_lambda: float = 1.5
    calibration_method: str = "isotonic"
    iv_min_threshold: float = 0.02
    vif_max_threshold: float = 5.0
    woe_epsilon: float = 0.5
    min_events_per_bin: int = 20
    pdo: int = 20
    target_score: int = 600
    target_odds: float = 50.0


PD_CONFIG = PDModelConfig()

# ──────────────────────────────────────────────
# MODELE LGD
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class LGDConfig:
    """Configuration du modele Loss Given Default.

    Attributes:
        lgd_ttc_mean: LGD Through-The-Cycle moyenne.
        lgd_ttc_std: Ecart-type LGD TTC.
        downturn_add_on: Add-on pour LGD Downturn.
        recovery_rate_floor: Plancher du taux de recouvrement.
        cure_rate: Taux de cure (sortie de defaut).
    """

    lgd_ttc_mean: float = 0.35
    lgd_ttc_std: float = 0.15
    downturn_add_on: float = 0.10
    recovery_rate_floor: float = 0.05
    cure_rate: float = 0.15


LGD_CONFIG = LGDConfig()

# ──────────────────────────────────────────────
# MODELE EAD
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class EADConfig:
    """Configuration du modele Exposure At Default.

    Attributes:
        ccf_revolving: Credit Conversion Factor pour lignes revolving.
        ccf_term_loan: CCF pour prets a terme.
        utilization_draw_stress: Stress sur le tirage en cas de defaut.
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
        sicr_threshold_multiplier: Multiplicateur PD pour declenchement SICR.
        stage3_dpd_threshold: Jours de retard pour passage en Stage 3.
        stage3_pd_threshold: PD seuil pour passage en Stage 3.
        discount_rate: Taux d'actualisation annuel (EIR proxy).
        lifetime_horizon_years: Horizon lifetime pour Stage 2/3 (annees).
    """

    sicr_threshold_multiplier: float = 2.0
    stage3_dpd_threshold: int = 90
    stage3_pd_threshold: float = 0.30
    discount_rate: float = 0.045
    lifetime_horizon_years: int = 3


IFRS9_CONFIG = IFRS9Config()


@dataclass(frozen=True)
class SICRConfig:
    """Configuration SICR multi-facteurs (IFRS 9 §B5.5.17).

    Score composite pour le declenchement Stage 2 :
        SICR_score = w_pd_ratio × (PD_current / PD_origination - 1)
                   + w_pd_delta × max(0, PD_current - PD_origination)
                   + w_dpd × (DPD / 30)
                   + w_macro × macro_z_score
        Stage 2 si SICR_score > threshold.

    Attributes:
        w_pd_ratio: Poids du ratio PD relatif (facteur le plus predictif).
        w_pd_delta: Poids du delta PD absolu.
        w_dpd: Poids du retard de paiement normalise.
        w_macro: Poids du Z-score macro (forward-looking).
        threshold: Seuil de declenchement SICR (calibre pour ~10% Stage 2).
    """

    w_pd_ratio: float = 0.25
    w_pd_delta: float = 0.25
    w_dpd: float = 0.20
    w_macro: float = 0.30
    threshold: float = 1.65


SICR_CONFIG = SICRConfig()

# ──────────────────────────────────────────────
# CALIBRATION LOGIT (MERTON-VASICEK)
# ──────────────────────────────────────────────

# Amplitude de transmission macro → logit(PD).
#
# Fondement theorique : dans le modele Vasicek a facteur unique,
# PD_stressed = Phi(Phi^-1(PD) + sqrt(rho) * z_macro).
# Pour la correlation d'actif corporate rho ≈ 0.20 (Bale II mid-point),
# un stress 2-sigma donne un shift logit ≈ 0.89.
#
# RJ audit MEDIUM : Λ=6.0 etait un nombre magique. Desormais calibre via
# _calibrate_logit_amplitude() a partir de 2 points d'ancrage EBA :
#   PD_base=0.06, PD_adverse=0.18 (×3), composite_shock=0.25.
# Formule : Λ = (logit(PD_adv) - logit(PD_base)) / composite_shock.
def _calibrate_logit_amplitude(
    pd_base: float = 0.06,
    pd_adverse: float = 0.18,
    composite_shock: float = 0.25,
) -> float:
    """Calibre Λ depuis 2 points d'ancrage EBA (stress test adverse).

    Args:
        pd_base: PD corporate baseline (defaut EBA mid-point).
        pd_adverse: PD corporate sous scenario adverse (×3 EBA).
        composite_shock: Choc macro composite moyen en scenario adverse.

    Returns:
        Λ tel que logit(PD_base) + Λ × composite_shock ≈ logit(PD_adverse).
    """
    import math
    logit_base = math.log(pd_base / (1 - pd_base))
    logit_adv = math.log(pd_adverse / (1 - pd_adverse))
    return (logit_adv - logit_base) / composite_shock


LOGIT_AMPLITUDE: float = _calibrate_logit_amplitude()  # ≈ 4.94, calibre sur PD 6%→18% (×3 EBA)

# ──────────────────────────────────────────────
# FEATURES (listes de reference pour les modeles)
# ──────────────────────────────────────────────

# Features credit — utilisees par pd_model, lgd_model, ead_model
CREDIT_NUMERICAL_FEATURES: List[str] = [
    "revenue",
    "ebitda",
    "debt_ratio",
    "credit_score",
    "dpd",
    "collateral",
    "loan_amount",
    "utilization_rate",
    "loan_to_revenue",        # engineered
    "collateral_coverage",    # engineered
    "supplier_hhi",
    "customer_count",
    "esg_score",
    "bank_relationship_years",
    # v4.5 DGP features (causal drivers of default)
    "ebitda_margin",
    "interest_coverage_ratio",
    "cf_volatility",
    "current_ratio",
    "working_capital_ratio",
    "net_debt_to_ebitda",
    "nb_incidents_12m",
    "account_age_months",
    # v5.0 interaction features (DGP Block 2/3 capture)
    "debt_x_hi_vol",
    "debt_x_lo_icr",
    "debt_x_lo_margin",
    "margin_neg_x_hi_debt",
    "vol_x_hi_debt",
    "icr_low_x_hi_debt",
    "incidents_x_hi_debt",
    "util_x_hi_vol",
    "icr_low_AND_margin_low",
    "debt_hi_AND_vol_hi",
    "util_hi_AND_incidents",
    "nde_squared_excess",
]

CREDIT_CATEGORICAL_FEATURES: List[str] = [
    "sector",
    "loan_type",
    "company_size",
]

# Features PE — utilisees par pe_model, pe_calculator
PE_NUMERICAL_FEATURES: List[str] = [
    "revenue",
    "ebitda",
    "entry_multiple",
    "leverage",
    "vintage",
    "holding_years",
]

PE_CATEGORICAL_FEATURES: List[str] = [
    "sector",
    "valuation_method",
]

TARGET: str = "default_flag"

# ──────────────────────────────────────────────
# CORPORATE INTERACTION FEATURES (DGP Block 2/3)
# ──────────────────────────────────────────────

# Seuils economiques (utilises par _clip_and_engineer)
DEBT_HI_THRESHOLD: float = 0.45
VOL_HI_THRESHOLD: float = 0.22
ICR_LO_THRESHOLD: float = 2.5
MARGIN_LO_THRESHOLD: float = 0.08
UTIL_HI_THRESHOLD: float = 0.50
NDE_QUAD_THRESHOLD: float = 4.0

CORPORATE_INTERACTION_FEATURES: List[str] = [
    "debt_x_hi_vol",           # debt_ratio * I(vol > 0.22)
    "debt_x_lo_icr",           # debt_ratio * I(icr < 2.5)
    "debt_x_lo_margin",        # debt_ratio * I(margin < 0.08)
    "margin_neg_x_hi_debt",    # max(0, -margin) * I(debt > 0.45)
    "vol_x_hi_debt",           # cf_volatility * I(debt > 0.45)
    "icr_low_x_hi_debt",       # max(0, 2-icr) * I(debt > 0.45)
    "incidents_x_hi_debt",     # nb_incidents * I(debt > 0.45)
    "util_x_hi_vol",           # max(0, util-0.50) * I(vol > 0.22)
    "icr_low_AND_margin_low",  # I(icr<2.5) * I(margin<0.08)
    "debt_hi_AND_vol_hi",      # I(debt>0.45) * I(vol>0.22)
    "util_hi_AND_incidents",   # I(util>0.50) * I(incidents>=2)
    "nde_squared_excess",      # max(0, NDE-4)^2
]

# ──────────────────────────────────────────────
# CONSUMER PD FEATURES (Lending Club)
# ──────────────────────────────────────────────

CONSUMER_NUMERICAL_FEATURES: List[str] = [
    "loan_amount", "remaining_tenor", "interest_rate", "borrower_income",
    "dti", "credit_score", "employment_length", "utilization_rate",
    "revolving_balance", "nb_active_credits", "public_records",
]
CONSUMER_CATEGORICAL_FEATURES: List[str] = [
    "grade", "home_ownership", "loan_purpose", "region",
]
CONSUMER_ENGINEERED_FEATURES: List[str] = [
    "income_to_loan",       # borrower_income / loan_amount
    "revol_to_income",      # revolving_balance / borrower_income
    "payment_burden",       # loan_amount * interest_rate / borrower_income
    "dti_x_util",           # dti * utilization_rate
    "rate_x_dti",           # interest_rate * dti
    "rate_x_util",          # interest_rate * utilization_rate
]

# ──────────────────────────────────────────────
# MORTGAGE PD FEATURES
# ──────────────────────────────────────────────

MORTGAGE_NUMERICAL_FEATURES: List[str] = [
    "property_value", "loan_amount", "ltv", "dti", "borrower_income",
    "borrower_age", "origination_year", "remaining_tenor",
    "interest_rate_margin", "dpd",
]
MORTGAGE_CATEGORICAL_FEATURES: List[str] = [
    "region", "property_type", "rate_type", "dpe_class",
]
MORTGAGE_ENGINEERED_FEATURES: List[str] = [
    "loan_to_income",       # loan_amount / borrower_income
    "payment_to_income",    # loan_amount * interest_rate_margin / borrower_income
    "ltv_x_dti",            # ltv * dti
    "ltv_x_dpd",            # ltv * dpd
    "ltv_squared",          # ltv^2
]

# Aliases de compatibilite avec le code existant (pd_model.py importe ces noms)
NUMERICAL_FEATURES: List[str] = CREDIT_NUMERICAL_FEATURES
CATEGORICAL_FEATURES: List[str] = CREDIT_CATEGORICAL_FEATURES

# ──────────────────────────────────────────────
# PARAMETRES PE SUPPLEMENTAIRES
# ──────────────────────────────────────────────

# Echelle logit pour le score de distress PE.
# Plus faible que credit car les PE sont en equity (junior tranche)
# et les sensibilites PE sont deja plus elevees dans SectorConfig.
PE_DISTRESS_LOGIT_SCALE: float = 3.0

# Ratio charges d'exploitation pour convertir EBITDA en NOI (Net Operating Income)
# pour la methode Cap_rate/NOI en immobilier.
# Source : benchmarks CBRE/JLL (2023), OPEX commercial RE = 12-18% du revenu brut.
NOI_OPEX_RATIO: float = 0.15

# Correlation intra-sectorielle pour le bruit de dispersion NAV.
# Source : Preqin (2023), correlation intra-fonds vintage estimee 0.3-0.5.
PE_NOISE_INTRA_SECTOR_CORR: float = 0.40

# ──────────────────────────────────────────────
# CONTRATS DATAFRAME (AR4)
# ──────────────────────────────────────────────

REQUIRED_CREDIT_COLS: frozenset = frozenset({
    "enterprise_id", "sector", "revenue", "ebitda", "debt_ratio",
    "credit_score", "dpd", "collateral", "loan_amount", "utilization_rate",
    "default_flag", "pd_origination",
})

# ──────────────────────────────────────────────
# CONTRAT DE DONNEES (Enums, Clipping, Features Engineered)
# ──────────────────────────────────────────────

# Domaines de valeurs categorielles (Enums strictes)
def _build_allowed_sectors() -> frozenset:
    """Construit dynamiquement depuis SECTORS (evite import circulaire)."""
    from ifrs9_cockpit.config.sectors import SECTORS
    return frozenset(s.name for s in SECTORS)

ALLOWED_SECTORS: frozenset = _build_allowed_sectors()
ALLOWED_LOAN_TYPES: frozenset = frozenset({"Revolving", "Term"})

# Bornes de clipping outliers (appliquees avant entrainement)
CLIPPING_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "debt_ratio": (0.0, 1.5),
    "credit_score": (300.0, 850.0),
    "utilization_rate": (0.0, 1.2),
}

# Features engineered calculees a la volee
ENGINEERED_FEATURES: List[str] = [
    "loan_to_revenue", "collateral_coverage",
] + CORPORATE_INTERACTION_FEATURES

REQUIRED_PE_COLS: frozenset = frozenset({
    "enterprise_id", "sector", "revenue", "ebitda",
    "entry_multiple", "leverage", "vintage", "holding_years",
    "valuation_method",
})

# Colonnes requises en sortie du pipeline credit (pour le comparateur)
REQUIRED_CREDIT_RESULT_COLS: frozenset = frozenset({
    "enterprise_id", "sector", "pd_12m", "pd_lifetime",
    "lgd", "ead", "ecl_weighted", "stage", "rwa_credit",
})

# Colonnes requises en sortie du pipeline PE (pour le comparateur)
REQUIRED_PE_RESULT_COLS: frozenset = frozenset({
    "enterprise_id", "sector", "nav", "delta_nav",
    "expected_loss_pe", "risk_category", "rwa_pe",
})
