"""Causal Machine Learning Engine for IFRS 9 Risk Cockpit.

Implements Double Machine Learning (Chernozhukov et al. 2018) with:
- DAG specification and identification (DoWhy)
- Multi-estimator benchmark (DML, CausalForestDML, DR-Learner)
- Causal Survival Analysis (lifelines)
- Sensitivity Analysis (Cinelli-Hazlett inspired)
- True ATE/CATE recovery validation on synthetic DGP

References:
    Chernozhukov et al. (2018) - Double/Debiased ML
    Wager & Athey (2018) - CausalForestDML
    Ahrens et al. (2025) - Model stacking in DML cross-fitting
    Hays & Raghavan (ICML 2025) - DML under shared-state interference
"""

from ifrs9_cockpit.causal.engine import CausalEngine

__all__ = ["CausalEngine"]
