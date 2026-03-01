"""Backward compatibility -- re-exports from synthetic_generator package."""
from ifrs9_cockpit.synthetic_generator import *  # noqa: F401,F403
from ifrs9_cockpit.synthetic_generator import (  # noqa: F401
    generate_dataset,
    AdvancedFinancialGenerator,
    RobustEvaluator,
    TARGET,
    LATENT_PD,
    NOISE_FEATURES,
)
from ifrs9_cockpit.synthetic_generator.constants import (  # noqa: F401
    _MACRO, _DEFAULT_STRESS_QUARTERS,
    _SECTOR_WEIGHTS, _SECTOR_FACTOR_SCALES, _FACTOR_LOADINGS,
    _SECTOR_PROFILES, _SECTOR_MARGINAL_SCALES,
    _SECTOR_REVENUE_PARAMS, _SECTOR_DPD_PARAMS,
    _SECTOR_ZOMBIE_RATES, _MACRO_NOISE_SIGMA,
    _SECTOR_DGP_ADJUSTMENTS,
    _SECTOR_MAP, _BRIDGE_CREDIT_SCORE, _BRIDGE_REVOLVING_PROB,
    _BRIDGE_COLLATERAL_RATIO, _BRIDGE_LEVERAGE_PE,
)
from ifrs9_cockpit.synthetic_generator.evaluator import _HAS_XGB  # noqa: F401
