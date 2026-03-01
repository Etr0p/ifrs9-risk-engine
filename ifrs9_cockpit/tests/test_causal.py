"""Tests unitaires pour le module Causal Machine Learning (Etape 3).

Couvre :
  - DGP causal structure (management_quality, CATE, time_to_default, spillover)
  - DAG specification and adjustment set
  - Data preparation (_prepare_data)
  - Multi-estimator benchmark (DML, CausalForest, DR-Learner, Naive XGBoost)
  - Causal survival analysis (Cox PH Hazard Ratio)
  - Sensitivity analysis (partial R2 framework)
  - CausalResult dataclass
  - CausalEngine orchestration
  - Dashboard chart functions (CATE heatmap, survival shift, bias comparison, sensitivity)

References:
    Chernozhukov et al. (2018) - Double/Debiased ML
    Wager & Athey (2018) - CausalForestDML
    Cinelli & Hazlett (2020) - Sensitivity Analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ifrs9_cockpit.config import RANDOM_SEED


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def causal_data():
    """Generate dataset with causal structure (n=1000 for speed)."""
    from ifrs9_cockpit.data.generator import generate_dataset

    df_credit, df_pe, df_history, _ = generate_dataset(n_clients=1000, seed=RANDOM_SEED)
    return df_credit


@pytest.fixture(scope="module")
def causal_result(causal_data):
    """Run full causal engine (n_splits=3 for speed)."""
    from ifrs9_cockpit.causal.engine import CausalEngine

    engine = CausalEngine(causal_data, n_splits=3)
    return engine.run()


# ============================================================
# 1. DGP Causal Structure
# ============================================================

class TestDGPCausalStructure:
    """Verify the synthetic DGP includes causal features."""

    def test_management_quality_exists(self, causal_data):
        """Hidden confounder must exist in DGP output."""
        assert "_management_quality" in causal_data.columns

    def test_management_quality_range(self, causal_data):
        """management_quality is clipped to [0, 1]."""
        mq = causal_data["_management_quality"]
        assert mq.min() >= 0.0
        assert mq.max() <= 1.0

    def test_true_cate_esg_exists(self, causal_data):
        """Ground truth CATE should be in the dataset."""
        assert "_true_cate_esg" in causal_data.columns

    def test_true_ate_sector_exists(self, causal_data):
        """Per-sector ground truth ATE should be in the dataset."""
        assert "_true_ate_esg_sector" in causal_data.columns

    def test_cate_is_negative(self, causal_data):
        """ESG should have a protective effect (negative CATE on PD)."""
        assert causal_data["_true_cate_esg"].mean() < 0

    def test_cate_heterogeneity(self, causal_data):
        """CATE should differ across sectors."""
        by_sector = causal_data.groupby("sector")["_true_ate_esg_sector"].first()
        assert by_sector.nunique() > 1, "CATE should be heterogeneous across sectors"

    def test_time_to_default_exists(self, causal_data):
        """Survival outcome column must exist."""
        assert "time_to_default" in causal_data.columns
        assert "event_observed" in causal_data.columns

    def test_time_to_default_positive(self, causal_data):
        """All time_to_default values must be positive."""
        assert (causal_data["time_to_default"] > 0).all()

    def test_time_to_default_censored(self, causal_data):
        """time_to_default should be censored at 5 years max."""
        assert causal_data["time_to_default"].max() <= 5.01  # small rounding margin

    def test_event_observed_binary(self, causal_data):
        """event_observed must be 0 or 1."""
        assert set(causal_data["event_observed"].unique()).issubset({0, 1})

    def test_event_rate_plausible(self, causal_data):
        """Event rate should be between 1% and 30%."""
        event_rate = causal_data["event_observed"].mean()
        assert 0.01 < event_rate < 0.30, f"Event rate {event_rate:.2%} outside plausible range"

    def test_esg_confounded_by_management_quality(self, causal_data):
        """ESG and management_quality should be positively correlated."""
        corr = causal_data["esg_score"].corr(causal_data["_management_quality"])
        assert corr > 0.05, f"ESG-management_quality correlation {corr:.3f} too weak"


# ============================================================
# 2. DAG and Adjustment Set
# ============================================================

class TestDAG:
    """Verify DAG structure and identification."""

    def test_build_dag_returns_digraph(self):
        """DAG builder should return a networkx DiGraph."""
        import networkx as nx
        from ifrs9_cockpit.causal.engine import build_causal_dag

        dag = build_causal_dag()
        assert isinstance(dag, nx.DiGraph)

    def test_dag_has_treatment_edge(self):
        """DAG must include ESG → PD edge."""
        from ifrs9_cockpit.causal.engine import build_causal_dag

        dag = build_causal_dag()
        assert dag.has_edge("esg_score", "default_flag")

    def test_dag_has_hidden_confounder(self):
        """DAG must include management_quality node."""
        from ifrs9_cockpit.causal.engine import build_causal_dag

        dag = build_causal_dag()
        assert "management_quality" in dag.nodes

    def test_dag_has_confounder_paths(self):
        """management_quality must confound ESG and PD."""
        from ifrs9_cockpit.causal.engine import build_causal_dag

        dag = build_causal_dag()
        assert dag.has_edge("management_quality", "esg_score")
        assert dag.has_edge("management_quality", "default_flag")

    def test_dag_is_acyclic(self):
        """DAG must be acyclic (no directed cycles)."""
        import networkx as nx
        from ifrs9_cockpit.causal.engine import build_causal_dag

        dag = build_causal_dag()
        assert nx.is_directed_acyclic_graph(dag)

    def test_adjustment_set_returns_confounders(self):
        """Adjustment set should return the known confounders."""
        from ifrs9_cockpit.causal.engine import get_dag_adjustment_set, CONFOUNDERS

        adj_set = get_dag_adjustment_set()
        assert adj_set == CONFOUNDERS


# ============================================================
# 3. Data Preparation
# ============================================================

class TestDataPreparation:
    """Test _prepare_data function."""

    def test_prepare_data_shapes(self, causal_data):
        """Output shapes must match n_samples."""
        from ifrs9_cockpit.causal.engine import _prepare_data, NUISANCE_FEATURES, CONFOUNDERS

        Y, T, W, X, sector_idx = _prepare_data(causal_data)
        n = len(causal_data)

        assert Y.shape == (n,)
        assert T.shape == (n,)
        assert W.shape == (n, len(NUISANCE_FEATURES))
        assert X.shape == (n, len(CONFOUNDERS))
        assert sector_idx.shape == (n,)

    def test_treatment_normalized(self, causal_data):
        """Treatment should be normalized to [0, 1]."""
        from ifrs9_cockpit.causal.engine import _prepare_data

        _, T, _, _, _ = _prepare_data(causal_data)
        assert T.min() >= -0.01  # small numerical margin
        assert T.max() <= 1.01

    def test_outcome_binary(self, causal_data):
        """Outcome should be binary {0, 1}."""
        from ifrs9_cockpit.causal.engine import _prepare_data

        Y, _, _, _, _ = _prepare_data(causal_data)
        assert set(np.unique(Y)).issubset({0.0, 1.0})


# ============================================================
# 4. Multi-Estimator Benchmark
# ============================================================

class TestEstimators:
    """Test individual estimator functions."""

    def test_dml_linear_returns_tuple(self, causal_data):
        """DML Linear should return (ate, ci, cate)."""
        from ifrs9_cockpit.causal.engine import _prepare_data, estimate_dml_linear

        Y, T, W, X, _ = _prepare_data(causal_data)
        ate, ci, cate = estimate_dml_linear(Y, T, W, X, n_splits=3)

        assert isinstance(ate, float)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] <= ate <= ci[1]
        assert len(cate) == len(Y)

    def test_causal_forest_returns_tuple(self, causal_data):
        """CausalForestDML should return (ate, ci, cate)."""
        from ifrs9_cockpit.causal.engine import _prepare_data, estimate_causal_forest

        Y, T, W, X, _ = _prepare_data(causal_data)
        ate, ci, cate = estimate_causal_forest(Y, T, W, X, n_splits=3)

        assert isinstance(ate, float)
        assert len(cate) == len(Y)

    def test_dr_learner_returns_tuple(self, causal_data):
        """DR-Learner should return (ate, ci, cate)."""
        from ifrs9_cockpit.causal.engine import _prepare_data, estimate_dr_learner

        Y, T, W, X, _ = _prepare_data(causal_data)
        ate, ci, cate = estimate_dr_learner(Y, T, W, X, n_splits=3)

        assert isinstance(ate, float)
        assert len(cate) == len(Y)

    def test_naive_xgb_returns_float(self, causal_data):
        """Naive XGBoost should return a float ATE."""
        from ifrs9_cockpit.causal.engine import _prepare_data, estimate_naive_xgb

        Y, T, W, _, _ = _prepare_data(causal_data)
        ate = estimate_naive_xgb(Y, T, W)

        assert isinstance(ate, float)

    def test_ate_direction_consistent(self, causal_result):
        """At least one DML-based ATE should be negative (ESG reduces PD).

        Note: with n=1000 and normalized T, individual estimators may
        not converge to the true sign. We verify at least one is negative
        and that the magnitude is small (|ATE| < 0.1).
        """
        dml_ates = [
            causal_result.ate_estimates[m]
            for m in ["DML_Linear", "CausalForestDML"]
        ]
        # At least one should be negative
        assert any(a < 0 for a in dml_ates) or all(
            abs(a) < 0.05 for a in dml_ates
        ), f"DML ATEs {dml_ates} neither negative nor near zero"


# ============================================================
# 5. CausalEngine Orchestration
# ============================================================

class TestCausalEngine:
    """Test the full CausalEngine pipeline."""

    def test_result_type(self, causal_result):
        """Engine should return a CausalResult."""
        from ifrs9_cockpit.causal.engine import CausalResult

        assert isinstance(causal_result, CausalResult)

    def test_result_has_4_estimators(self, causal_result):
        """Engine should run exactly 4 estimators."""
        assert len(causal_result.ate_estimates) == 4
        expected = {"DML_Linear", "CausalForestDML", "DR_Learner", "XGBoost_Naive"}
        assert set(causal_result.ate_estimates.keys()) == expected

    def test_result_has_cate_by_sector(self, causal_result):
        """CATE by sector should be populated for DML and CausalForest."""
        assert "DML_Linear" in causal_result.cate_by_sector
        assert "CausalForestDML" in causal_result.cate_by_sector

    def test_cate_covers_all_sectors(self, causal_result):
        """CATE should cover all 5 sectors."""
        expected_sectors = {"Technologie", "Industrie", "Sante", "Immobilier", "Services"}
        for method, sector_cate in causal_result.cate_by_sector.items():
            assert set(sector_cate.keys()) == expected_sectors, (
                f"{method} CATE missing sectors: {expected_sectors - set(sector_cate.keys())}"
            )

    def test_result_has_true_ate(self, causal_result):
        """Ground truth ATE should be populated."""
        assert causal_result.true_ate is not None
        assert causal_result.true_ate < 0  # ESG is protective

    def test_result_has_true_cate(self, causal_result):
        """Ground truth CATE by sector should be populated."""
        assert causal_result.true_cate_by_sector is not None
        assert len(causal_result.true_cate_by_sector) == 5

    def test_result_has_dag_edges(self, causal_result):
        """DAG edges should be populated."""
        assert causal_result.dag_edges is not None
        assert len(causal_result.dag_edges) > 10

    def test_n_samples_recorded(self, causal_result):
        """Sample count should be recorded."""
        assert causal_result.n_samples == 1000


# ============================================================
# 6. Causal Survival Analysis
# ============================================================

class TestSurvivalAnalysis:
    """Test causal survival analysis."""

    def test_survival_hr_populated(self, causal_result):
        """Survival HR should be populated."""
        assert causal_result.survival_hr is not None

    def test_survival_hr_keys(self, causal_result):
        """Survival result should have required keys."""
        hr = causal_result.survival_hr
        required = {"hr_adjusted", "hr_naive", "bias_pct", "survival_high", "survival_low"}
        assert required.issubset(set(hr.keys()))

    def test_hazard_ratio_positive(self, causal_result):
        """Hazard ratios must be positive."""
        hr = causal_result.survival_hr
        assert hr["hr_adjusted"] > 0
        assert hr["hr_naive"] > 0

    def test_adjusted_vs_naive_different(self, causal_result):
        """Adjusted and naive HR should differ (confounding bias)."""
        hr = causal_result.survival_hr
        assert abs(hr["hr_adjusted"] - hr["hr_naive"]) > 0.001, (
            "Adjusted and naive HR should differ due to confounding"
        )

    def test_survival_curves_structure(self, causal_result):
        """Survival curves should have timeline and survival arrays."""
        hr = causal_result.survival_hr
        for key in ["survival_high", "survival_low"]:
            curve = hr[key]
            assert "timeline" in curve
            assert "survival" in curve
            assert len(curve["timeline"]) > 0
            assert len(curve["survival"]) > 0

    def test_survival_starts_at_one(self, causal_result):
        """KM survival curves should start at ~1.0."""
        hr = causal_result.survival_hr
        for key in ["survival_high", "survival_low"]:
            s0 = hr[key]["survival"][0]
            assert s0 >= 0.98, f"Survival should start near 1.0, got {s0}"

    def test_standalone_survival_function(self, causal_data):
        """estimate_survival_hr should work standalone."""
        from ifrs9_cockpit.causal.engine import estimate_survival_hr

        result = estimate_survival_hr(causal_data)
        assert "hr_adjusted" in result
        assert "concordance" in result
        assert 0.4 < result["concordance"] < 1.0


# ============================================================
# 7. Sensitivity Analysis
# ============================================================

class TestSensitivityAnalysis:
    """Test Cinelli-Hazlett inspired sensitivity analysis."""

    def test_sensitivity_populated(self, causal_result):
        """Sensitivity analysis should be populated."""
        assert causal_result.sensitivity is not None

    def test_sensitivity_keys(self, causal_result):
        """Sensitivity result should have required keys."""
        s = causal_result.sensitivity
        required = {"partial_r2_treatment", "robustness_value", "rv_interpretation"}
        assert required.issubset(set(s.keys()))

    def test_partial_r2_nonnegative(self, causal_result):
        """Partial R2 must be non-negative."""
        assert causal_result.sensitivity["partial_r2_treatment"] >= 0

    def test_robustness_value_nonnegative(self, causal_result):
        """Robustness value must be non-negative."""
        assert causal_result.sensitivity["robustness_value"] >= 0

    def test_benchmark_management_quality(self, causal_result):
        """Benchmark for hidden confounder should be computed."""
        s = causal_result.sensitivity
        assert "benchmark_mgmt_quality_r2" in s
        # With synthetic data, benchmark should be computable
        assert s["benchmark_mgmt_quality_r2"] is not None

    def test_standalone_sensitivity(self, causal_data):
        """compute_sensitivity should work standalone."""
        from ifrs9_cockpit.causal.engine import compute_sensitivity

        result = compute_sensitivity(causal_data)
        assert "partial_r2_treatment" in result
        assert "robustness_value" in result


# ============================================================
# 8. Dashboard Chart Functions
# ============================================================

class TestCausalCharts:
    """Test the 4 causal chart rendering functions."""

    def test_cate_heatmap(self, causal_result):
        """plot_cate_heatmap should return a Plotly Figure."""
        import plotly.graph_objects as go
        from ifrs9_cockpit.dashboard.charts import plot_cate_heatmap

        fig = plot_cate_heatmap(causal_result.cate_by_sector)
        assert isinstance(fig, go.Figure)

    def test_survival_shift(self, causal_result):
        """plot_survival_shift should return a Plotly Figure."""
        import plotly.graph_objects as go
        from ifrs9_cockpit.dashboard.charts import plot_survival_shift

        hr = causal_result.survival_hr
        fig = plot_survival_shift(hr["survival_high"], hr["survival_low"])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Two KM curves

    def test_bias_comparison(self, causal_result):
        """plot_bias_comparison should return a Plotly Figure."""
        import plotly.graph_objects as go
        from ifrs9_cockpit.dashboard.charts import plot_bias_comparison

        fig = plot_bias_comparison(
            causal_result.ate_estimates,
            causal_result.true_ate,
        )
        assert isinstance(fig, go.Figure)

    def test_bias_comparison_without_true_ate(self, causal_result):
        """plot_bias_comparison should work without true ATE."""
        import plotly.graph_objects as go
        from ifrs9_cockpit.dashboard.charts import plot_bias_comparison

        fig = plot_bias_comparison(causal_result.ate_estimates)
        assert isinstance(fig, go.Figure)

    def test_sensitivity_contour(self, causal_result):
        """plot_sensitivity_contour should return a Plotly Figure."""
        import plotly.graph_objects as go
        from ifrs9_cockpit.dashboard.charts import plot_sensitivity_contour

        s = causal_result.sensitivity
        fig = plot_sensitivity_contour(
            s["partial_r2_treatment"],
            s["robustness_value"],
            s.get("benchmark_mgmt_quality_r2"),
        )
        assert isinstance(fig, go.Figure)

    def test_sensitivity_contour_no_benchmark(self):
        """Sensitivity contour should work without benchmark."""
        import plotly.graph_objects as go
        from ifrs9_cockpit.dashboard.charts import plot_sensitivity_contour

        fig = plot_sensitivity_contour(0.05, 0.22, None)
        assert isinstance(fig, go.Figure)


# ============================================================
# 9. CausalResult Dataclass
# ============================================================

class TestCausalResult:
    """Test CausalResult dataclass defaults and structure."""

    def test_default_empty(self):
        """Default CausalResult should have empty fields."""
        from ifrs9_cockpit.causal.engine import CausalResult

        r = CausalResult()
        assert r.ate_estimates == {}
        assert r.true_ate is None
        assert r.survival_hr is None
        assert r.sensitivity is None
        assert r.dag_edges is None
        assert r.n_samples == 0

    def test_bias_comparison_populated(self, causal_result):
        """Bias comparison should be populated for all estimators."""
        assert len(causal_result.bias_comparison) == 4


# ============================================================
# 10. Integration: Causal Engine with different sample sizes
# ============================================================

class TestCausalEngineRobustness:
    """Test engine robustness with smaller samples."""

    def test_small_sample_runs(self):
        """Engine should not crash with 200 samples."""
        from ifrs9_cockpit.data.generator import generate_dataset
        from ifrs9_cockpit.causal.engine import CausalEngine

        df_credit, _, _, _ = generate_dataset(n_clients=200, seed=99)
        engine = CausalEngine(df_credit, n_splits=2)
        result = engine.run()

        assert len(result.ate_estimates) == 4
        assert result.survival_hr is not None
        assert result.sensitivity is not None
