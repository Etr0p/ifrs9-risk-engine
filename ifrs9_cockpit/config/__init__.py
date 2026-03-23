"""Configuration IFRS 9 Risk Cockpit — package re-exports.

Ce package regroupe toutes les constantes de configuration du projet.
Chaque sous-module est thematique :
    - sectors.py   : secteurs, classes d'actifs, constantes globales
    - scenarios.py : scenarios macro, covariance, trajectoires
    - models.py    : modeles PD/LGD/EAD, IFRS9, SICR, contrats de donnees
    - basel.py     : Basel III/CRR3, risk appetite, PE classification, CRO
    - rules.py     : regles d'incoherence macro, validation

Tous les symboles publics sont re-exportes ici pour compatibilite
avec les imports existants ``from ifrs9_cockpit.config import X``.
"""

# ── sectors.py (18 symbols) ──────────────────────────────────
from ifrs9_cockpit.config.sectors import (
    RANDOM_SEED,
    N_CLIENTS,
    N_MONTHS,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    SectorConfig,
    SECTORS,
    SECTOR_NAMES,
    SEGMENTS,
    AssetClassProfile,
    ASSET_CLASSES,
    ASSET_CLASS_NAMES,
    ASSET_CLASS_LABELS,
    ASSET_CLASS_MAP,
    LEVEL1_CLASSES,
    LEVEL2_CLASSES,
    LEVEL3_CLASSES,
)

# ── scenarios.py (12 symbols) ────────────────────────────────
from ifrs9_cockpit.config.scenarios import (
    MacroScenario,
    SCENARIO_BASE,
    SCENARIO_ADVERSE,
    SCENARIO_FAVORABLE,
    ECL_SCENARIOS,
    SCENARIOS,
    PREDEFINED_SCENARIOS,
    MACRO_HISTORY_BASELINE,
    MACRO_STRUCTURAL_EQUILIBRIUM,
    MACRO_MEAN_REVERSION,
    MACRO_COVARIANCE,
    MACRO_VARIABLES_ORDER,
)

# ── models.py (29 symbols) ───────────────────────────────────
from ifrs9_cockpit.config.models import (
    PDModelConfig,
    PD_CONFIG,
    LGDConfig,
    LGD_CONFIG,
    EADConfig,
    EAD_CONFIG,
    IFRS9Config,
    IFRS9_CONFIG,
    SICRConfig,
    SICR_CONFIG,
    LOGIT_AMPLITUDE,
    CREDIT_NUMERICAL_FEATURES,
    CREDIT_CATEGORICAL_FEATURES,
    PE_NUMERICAL_FEATURES,
    PE_CATEGORICAL_FEATURES,
    TARGET,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    PE_DISTRESS_LOGIT_SCALE,
    NOI_OPEX_RATIO,
    PE_NOISE_INTRA_SECTOR_CORR,
    REQUIRED_CREDIT_COLS,
    ALLOWED_SECTORS,
    ALLOWED_LOAN_TYPES,
    CLIPPING_BOUNDS,
    ENGINEERED_FEATURES,
    REQUIRED_PE_COLS,
    REQUIRED_CREDIT_RESULT_COLS,
    REQUIRED_PE_RESULT_COLS,
    # Corporate interaction features
    CORPORATE_INTERACTION_FEATURES,
    DEBT_HI_THRESHOLD,
    VOL_HI_THRESHOLD,
    ICR_LO_THRESHOLD,
    MARGIN_LO_THRESHOLD,
    UTIL_HI_THRESHOLD,
    NDE_QUAD_THRESHOLD,
    # Consumer PD features
    CONSUMER_NUMERICAL_FEATURES,
    CONSUMER_CATEGORICAL_FEATURES,
    CONSUMER_ENGINEERED_FEATURES,
    # Mortgage PD features
    MORTGAGE_NUMERICAL_FEATURES,
    MORTGAGE_CATEGORICAL_FEATURES,
    MORTGAGE_ENGINEERED_FEATURES,
)

# ── basel.py (8 symbols) ─────────────────────────────────────
from ifrs9_cockpit.config.basel import (
    BaselConfig,
    BASEL_CONFIG,
    RiskAppetiteConfig,
    RISK_APPETITE_CONFIG,
    PEClassificationConfig,
    PE_CLASSIFICATION_CONFIG,
    CROAlertConfig,
    CRO_CONFIG,
)

# ── rules.py (3 symbols) ─────────────────────────────────────
from ifrs9_cockpit.config.rules import (
    MacroIncoherenceRule,
    MACRO_INCOHERENCE_RULES,
    validate_config,
)

# ── Private symbols used by engine modules ────────────────────
from ifrs9_cockpit.config.scenarios import (
    _EXPERT_CORR,
    _MACRO_VOLATILITIES,
)

__all__ = [
    # sectors
    "RANDOM_SEED", "N_CLIENTS", "N_MONTHS", "TRAIN_RATIO",
    "VALIDATION_RATIO", "TEST_RATIO",
    "SectorConfig", "SECTORS", "SECTOR_NAMES", "SEGMENTS",
    "AssetClassProfile", "ASSET_CLASSES", "ASSET_CLASS_NAMES",
    "ASSET_CLASS_LABELS", "ASSET_CLASS_MAP",
    "LEVEL1_CLASSES", "LEVEL2_CLASSES", "LEVEL3_CLASSES",
    # scenarios
    "MacroScenario", "SCENARIO_BASE", "SCENARIO_ADVERSE", "SCENARIO_FAVORABLE",
    "ECL_SCENARIOS", "SCENARIOS", "PREDEFINED_SCENARIOS",
    "MACRO_HISTORY_BASELINE", "MACRO_STRUCTURAL_EQUILIBRIUM",
    "MACRO_MEAN_REVERSION", "MACRO_COVARIANCE", "MACRO_VARIABLES_ORDER",
    # models
    "PDModelConfig", "PD_CONFIG", "LGDConfig", "LGD_CONFIG",
    "EADConfig", "EAD_CONFIG", "IFRS9Config", "IFRS9_CONFIG",
    "SICRConfig", "SICR_CONFIG", "LOGIT_AMPLITUDE",
    "CREDIT_NUMERICAL_FEATURES", "CREDIT_CATEGORICAL_FEATURES",
    "PE_NUMERICAL_FEATURES", "PE_CATEGORICAL_FEATURES",
    "TARGET", "NUMERICAL_FEATURES", "CATEGORICAL_FEATURES",
    "PE_DISTRESS_LOGIT_SCALE", "NOI_OPEX_RATIO", "PE_NOISE_INTRA_SECTOR_CORR",
    "REQUIRED_CREDIT_COLS", "ALLOWED_SECTORS", "ALLOWED_LOAN_TYPES",
    "CLIPPING_BOUNDS", "ENGINEERED_FEATURES",
    "REQUIRED_PE_COLS", "REQUIRED_CREDIT_RESULT_COLS", "REQUIRED_PE_RESULT_COLS",
    "CORPORATE_INTERACTION_FEATURES",
    "DEBT_HI_THRESHOLD", "VOL_HI_THRESHOLD", "ICR_LO_THRESHOLD",
    "MARGIN_LO_THRESHOLD", "UTIL_HI_THRESHOLD", "NDE_QUAD_THRESHOLD",
    "CONSUMER_NUMERICAL_FEATURES", "CONSUMER_CATEGORICAL_FEATURES",
    "CONSUMER_ENGINEERED_FEATURES",
    "MORTGAGE_NUMERICAL_FEATURES", "MORTGAGE_CATEGORICAL_FEATURES",
    "MORTGAGE_ENGINEERED_FEATURES",
    # basel
    "BaselConfig", "BASEL_CONFIG", "RiskAppetiteConfig", "RISK_APPETITE_CONFIG",
    "PEClassificationConfig", "PE_CLASSIFICATION_CONFIG",
    "CROAlertConfig", "CRO_CONFIG",
    # rules
    "MacroIncoherenceRule", "MACRO_INCOHERENCE_RULES", "validate_config",
]

# Validation automatique a l'import
validate_config()
