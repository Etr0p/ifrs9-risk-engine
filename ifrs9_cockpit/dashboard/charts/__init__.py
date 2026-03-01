"""Package charts — re-exporte toutes les fonctions pour backward compat.

Usage inchange :
    from ifrs9_cockpit.dashboard.charts import plot_roc_curves
    from ifrs9_cockpit.dashboard import charts  # charts.plot_roc_curves
"""

# ── Base (helpers + constantes) ──
from ifrs9_cockpit.dashboard.charts.base import (  # noqa: F401
    _PRIMARY, _SECONDARY, _ACCENT, _BG, _CARD, _TEXT, _MUTED,
    _SUCCESS, _WARNING, _DANGER, _INFO, _COLORS,
    _FEATURE_LABELS, _SECTOR_PALETTE, _SECTOR_PALETTE_FALLBACK,
    _prettify_feature, _base_layout, _hex_to_rgba,
)

# ── Credit / PD / Staging / ECL ──
from ifrs9_cockpit.dashboard.charts.credit_charts import (  # noqa: F401
    plot_roc_curves,
    plot_feature_importance,
    plot_stage_distribution,
    plot_ecl_by_segment,
    plot_transition_matrix,
    plot_waterfall_ecl,
    plot_ecl_coverage_scatter,
    plot_iv_table,
    plot_model_comparison,
    plot_shap_summary,
    plot_shap_beeswarm,
    plot_calibration_curve,
    plot_hhi_gauge,
    plot_backtesting_auc,
    plot_shap_force_individual,
    plot_score_distribution,
    plot_trajectories_chart,
    plot_stage_sankey,
    plot_pareto_front,
)

# ── PE Fund ──
from ifrs9_cockpit.dashboard.charts.pe_charts import (  # noqa: F401
    plot_pe_nav_by_sector,
    plot_pe_risk_categories,
    plot_pe_moic_drawdown,
    plot_pe_distress_box,
    plot_pe_rwa_by_sector,
    plot_pe_vintage_analysis,
    plot_pe_risk_stacked_bar,
)

# ── Comparator (asymetrie, RAROC, CRR3) ──
from ifrs9_cockpit.dashboard.charts.comparator_charts import (  # noqa: F401
    plot_asymmetry_heatmap,
    plot_raroc_comparison,
    plot_crr3_sensitivity,
    plot_multiclass_raroc_scatter,
    plot_risk_appetite_matrix,
)

# ── Causal ML ──
from ifrs9_cockpit.dashboard.charts.causal_charts import (  # noqa: F401
    plot_cate_heatmap,
    plot_survival_shift,
    plot_bias_comparison,
    plot_sensitivity_contour,
)

# ── Virtual CRO (NeSy MAS) ──
from ifrs9_cockpit.dashboard.charts.vcro_charts import (  # noqa: F401
    plot_vcro_agent_agreement,
    plot_vcro_belief_distribution,
    plot_vcro_qbaf_strengths,
    plot_vcro_pma_comparison,
)

# ── Multi-Asset Balance Sheet ──
from ifrs9_cockpit.dashboard.charts.multi_asset_charts import (  # noqa: F401
    plot_balance_sheet_treemap,
    plot_contagion_network,
    plot_climate_heatmap,
)

# ── Gouvernance Quantitative ──
from ifrs9_cockpit.dashboard.charts.governance_charts import (  # noqa: F401
    plot_conformal_bands,
    plot_sobol_indices,
    plot_vrp_regime,
    plot_rmt_eigenvalues,
    plot_signature_heatmap,
    plot_tda_fragility,
    plot_compliance_gates,
)

# ── Regime Intelligence (HMM + GFlowNet) ──
from ifrs9_cockpit.dashboard.charts.regime_charts import (  # noqa: F401
    plot_regime_gauge,
    plot_gflownet_scatter,
    plot_regime_allocation,
    plot_scenario_weights_pie,
)

# ── Allocation BL-CVaR ──
from ifrs9_cockpit.dashboard.charts.allocation_charts import (  # noqa: F401
    plot_multiclass_allocation_bar,
    plot_sector_allocation_weights,
    plot_sector_allocation_donuts,
    plot_factor_attribution,
    plot_regulatory_constraints,
    plot_sector_signals,
    plot_efficient_frontier,
    plot_capital_sankey,
)
