"""Tests unitaires pour l'optimiseur BL-CVaR 10 cellules (pe-bc).

Teste PebcOptimizerMixin :
  - Matrice de correlation 10x10 (cosine + FTQ + LW + PSD)
  - CVaR Monte Carlo (5000 scenarios)
  - Softmax auto-temp + correlation penalty
  - Spread compression logarithmique
  - Stress diagnostics (asymetric illiquidity, vol, BL confidence, PE band)
  - Normes reglementaires (LCR, NSFR, IRRBB) 10 cellules
  - Allocation complete optimize_allocation_pebc()
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ifrs9_cockpit.config import (
    BASEL_CONFIG,
    SECTORS,
    SECTOR_NAMES,
)
from ifrs9_cockpit.engine.comparator import PortfolioComparator
from ifrs9_cockpit.engine.comparator.regulatory_pebc import (
    compute_lcr_10,
    compute_nsfr_10,
    compute_irrbb_eve_10,
)
from ifrs9_cockpit.engine.comparator.optimizer_pebc import (
    _CELL_NAMES_PEBC,
    _N_CELLS,
)

pytestmark = pytest.mark.pebc


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def comparator(global_pipeline_results):
    return global_pipeline_results["comparator"]


@pytest.fixture(scope="module")
def result_credit(global_pipeline_results):
    return global_pipeline_results["result_credit"]


@pytest.fixture(scope="module")
def result_pe(global_pipeline_results):
    return global_pipeline_results["result_pe"]


@pytest.fixture(scope="module")
def allocation_pebc(comparator):
    return comparator.optimize_allocation_pebc()


# ============================================================
# Correlation matrix 10x10
# ============================================================

class TestCorrelationMatrixPebc:
    """Tests pour _build_corr_matrix_pebc — 10x10 cosine + FTQ + LW + PSD."""

    def test_shape(self):
        corr, lam = PortfolioComparator._build_corr_matrix_pebc()
        assert corr.shape == (10, 10)

    def test_symmetric(self):
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_diagonal_ones(self):
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_positive_semi_definite(self):
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        eigvals = np.linalg.eigvalsh(corr)
        assert np.all(eigvals >= -1e-10), f"Eigenvalues negatives: {eigvals}"

    def test_values_in_range(self):
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        assert np.all(corr >= -1.0 - 1e-6)
        assert np.all(corr <= 1.0 + 1e-6)

    def test_shrinkage_lambda_in_range(self):
        _, lam = PortfolioComparator._build_corr_matrix_pebc()
        assert 0.05 <= lam <= 0.50

    def test_same_sector_cross_canal_correlated(self):
        """Tech Credit et Tech PE doivent etre positivement correles."""
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        # Tech_Credit=0, Tech_PE=5
        assert corr[0, 5] > 0, f"Tech Credit-PE correlation={corr[0, 5]}"


# ============================================================
# CVaR Monte Carlo
# ============================================================

class TestCVaRPebc:
    """Tests pour _compute_cvar_pebc."""

    def test_cvar_positive(self, comparator):
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        vol = np.array([s.market_vol_credit for s in SECTORS]
                      + [s.market_vol_pe for s in SECTORS])
        Sigma = np.outer(vol, vol) * corr
        w = np.ones(10) / 10
        cvar, _, _ = comparator._compute_cvar_pebc(w, Sigma)
        assert cvar > 0

    def test_cvar_reproducible(self, comparator):
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        vol = np.array([s.market_vol_credit for s in SECTORS]
                      + [s.market_vol_pe for s in SECTORS])
        Sigma = np.outer(vol, vol) * corr
        w = np.ones(10) / 10
        cvar1, _, _ = comparator._compute_cvar_pebc(w, Sigma)
        cvar2, _, _ = comparator._compute_cvar_pebc(w, Sigma)
        assert abs(cvar1 - cvar2) < 1e-10

    def test_cvar_monotone_in_volatility(self, comparator):
        corr = np.eye(10)
        w = np.ones(10) / 10
        vol_low = np.full(10, 0.05)
        vol_high = np.full(10, 0.20)
        Sigma_low = np.outer(vol_low, vol_low) * corr
        Sigma_high = np.outer(vol_high, vol_high) * corr
        cvar_low, _, _ = comparator._compute_cvar_pebc(w, Sigma_low)
        cvar_high, _, _ = comparator._compute_cvar_pebc(w, Sigma_high)
        assert cvar_high > cvar_low


# ============================================================
# Softmax auto-temperature
# ============================================================

class TestSoftmaxWeightsPebc:
    """Tests pour _softmax_weights_pebc — temperature auto-calibree."""

    def test_equal_scores_equal_weights(self):
        scores = np.array([0.10] * 10)
        w = PortfolioComparator._softmax_weights_pebc(scores)
        np.testing.assert_allclose(w, 0.10, atol=1e-6)

    def test_best_score_highest_weight(self):
        scores = np.array([0.05, 0.10, 0.30, 0.08, 0.06,
                          0.05, 0.10, 0.30, 0.08, 0.06])
        w = PortfolioComparator._softmax_weights_pebc(scores)
        best_idx = np.argmax(scores)
        assert w[best_idx] == np.max(w)

    def test_floor_respected(self):
        """Le poids minimum est 1/n^2 = 0.01."""
        scores = np.array([1.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0])
        w = PortfolioComparator._softmax_weights_pebc(scores)
        assert np.min(w) >= 1.0 / (10 * 10) - 1e-10

    def test_sums_to_one(self):
        scores = np.random.default_rng(42).standard_normal(10)
        w = PortfolioComparator._softmax_weights_pebc(scores)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_no_nan_on_extreme_scores(self):
        scores = np.array([-100, 100, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        w = PortfolioComparator._softmax_weights_pebc(scores)
        assert np.all(np.isfinite(w))


class TestSoftmaxPebc:
    """Tests pour _softmax_pebc — softmax 2 passes avec penalite de correlation."""

    def test_sums_to_one(self):
        mu = np.array([0.05, 0.15, 0.25, 0.08, 0.12,
                       0.05, 0.15, 0.25, 0.08, 0.12])
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        w = PortfolioComparator._softmax_pebc(mu, corr)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_diversification_vs_naive(self):
        mu = np.ones(10) * 0.10
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        w = PortfolioComparator._softmax_pebc(mu, corr)
        assert np.max(w) > np.min(w), "Penalite devrait creer de l'asymetrie"

    def test_all_positive(self):
        mu = np.random.default_rng(42).uniform(-0.1, 0.3, 10)
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        w = PortfolioComparator._softmax_pebc(mu, corr)
        assert np.all(w > 0)


# ============================================================
# Spread compression
# ============================================================

class TestSpreadCompressionPebc:
    """Tests pour _spread_compression_pebc."""

    def test_zero_weight_no_compression(self):
        w = np.zeros(10)
        mu_base = np.ones(10) * 0.10
        mu_eff = PortfolioComparator._spread_compression_pebc(w, 1e9, mu_base)
        np.testing.assert_allclose(mu_eff, mu_base, atol=1e-10)

    def test_compression_reduces_return(self):
        w = np.ones(10) / 10
        mu_base = np.ones(10) * 0.10
        mu_eff = PortfolioComparator._spread_compression_pebc(w, 1e12, mu_base)
        assert np.all(mu_eff <= mu_base + 1e-10)

    def test_compression_increases_with_weight(self):
        mu_base = np.ones(10) * 0.10
        w_small = np.ones(10) / 10
        w_large = np.zeros(10)
        w_large[0] = 1.0
        mu_small = PortfolioComparator._spread_compression_pebc(w_small, 1e12, mu_base)
        mu_large = PortfolioComparator._spread_compression_pebc(w_large, 1e12, mu_base)
        assert mu_large[0] < mu_small[0]


# ============================================================
# Stress diagnostics
# ============================================================

class TestStressDiagnosticsPebc:
    """Tests stress diagnostics pe-bc."""

    def test_stress_intensity_positive_in_crisis(self):
        s = PortfolioComparator._stress_intensity_pebc(-0.05, -0.10)
        assert s > 0

    def test_stress_intensity_negative_in_expansion(self):
        s = PortfolioComparator._stress_intensity_pebc(0.20, 0.15)
        assert s < 0

    def test_illiquidity_convex(self):
        illiq_1 = PortfolioComparator._asymmetric_illiquidity_pebc(1.0)
        illiq_2 = PortfolioComparator._asymmetric_illiquidity_pebc(2.0)
        base = 0.005
        assert (illiq_2 - base) > 2 * (illiq_1 - base)

    def test_vol_multiplier_above_1_in_stress(self):
        vm = PortfolioComparator._asymmetric_vol_multiplier_pebc(1.0)
        assert vm > 1.0

    def test_bl_confidence_drops_in_stress(self):
        conf_calm_c, conf_calm_p = PortfolioComparator._bl_confidence_pebc(0.0)
        conf_stress_c, conf_stress_p = PortfolioComparator._bl_confidence_pebc(2.0)
        assert conf_stress_c < conf_calm_c
        assert conf_stress_p < conf_calm_p

    def test_pe_band_shrinks_in_stress(self):
        _, pe_max_calm = PortfolioComparator._asymmetric_pe_band_pebc(0.0)
        _, pe_max_stress = PortfolioComparator._asymmetric_pe_band_pebc(3.0)
        assert pe_max_stress < pe_max_calm

    def test_adaptive_step_proportional(self):
        step_small = PortfolioComparator._adaptive_step_pebc(0.01)
        step_large = PortfolioComparator._adaptive_step_pebc(0.10)
        assert step_large > step_small

    def test_adaptive_step_bounded(self):
        for deficit in [0.001, 0.01, 0.05, 0.10, 0.50, 1.0]:
            step = PortfolioComparator._adaptive_step_pebc(deficit)
            assert 0.002 <= step <= 0.02, f"deficit={deficit}: step={step}"


# ============================================================
# Regulatory functions (10-cell standalone)
# ============================================================

class TestLCRPebc:
    """Tests LCR 10 cellules."""

    def test_lcr_ratio_positive(self, allocation_pebc):
        assert allocation_pebc["lcr_ratio"] > 0

    def test_lcr_structure(self):
        n = len(SECTORS) * 2
        w = np.ones(n) / n
        total_ead = 1e9
        lcr = compute_lcr_10(w, _CELL_NAMES_PEBC, SECTORS, total_ead)
        assert "lcr_ratio" in lcr
        assert "hqla" in lcr
        assert "outflows" in lcr

    def test_lcr_compliance_flag(self, allocation_pebc):
        assert isinstance(allocation_pebc["lcr_compliant"], bool)


class TestNSFRPebc:
    """Tests NSFR 10 cellules."""

    def test_nsfr_ratio_positive(self, allocation_pebc):
        assert allocation_pebc["nsfr_ratio"] > 0

    def test_nsfr_structure(self):
        n = len(SECTORS) * 2
        w = np.ones(n) / n
        total_ead = 1e9
        nsfr = compute_nsfr_10(w, _CELL_NAMES_PEBC, SECTORS, total_ead)
        assert "nsfr_ratio" in nsfr
        assert "asf" in nsfr
        assert "rsf" in nsfr

    def test_nsfr_asf_positive(self):
        n = len(SECTORS) * 2
        w = np.ones(n) / n
        nsfr = compute_nsfr_10(w, _CELL_NAMES_PEBC, SECTORS, 1e9)
        assert nsfr["asf"] > 0
        assert nsfr["rsf"] > 0


class TestIRRBBPebc:
    """Tests IRRBB EVE 10 cellules."""

    def test_irrbb_eve_ratio_defined(self, allocation_pebc):
        assert "irrbb_eve_ratio" in allocation_pebc
        assert np.isfinite(allocation_pebc["irrbb_eve_ratio"])

    def test_irrbb_compliant_bool(self, allocation_pebc):
        assert isinstance(allocation_pebc["irrbb_compliant"], bool)

    def test_irrbb_duration_coherent(self):
        n = len(SECTORS) * 2
        w = np.ones(n) / n
        irrbb = compute_irrbb_eve_10(w, _CELL_NAMES_PEBC, SECTORS, 1e9, 1.3e8)
        assert irrbb["weighted_duration"] > 0


# ============================================================
# FTQ correlation adjustments
# ============================================================

class TestFTQPebc:
    """Tests FTQ correlation pe-bc."""

    def test_ftq_sante_immo_reduced(self):
        """corr(Sante_C, Immo_C) reduit apres FTQ."""
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        # Sante_Credit=2, Immobilier_Credit=3
        assert corr[2, 3] < 0.90, "FTQ should reduce Sante-Immo correlation"

    def test_ftq_still_psd(self):
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        eigvals = np.linalg.eigvalsh(corr)
        assert np.all(eigvals >= -1e-10)

    def test_ftq_tech_coinvestment_positive(self):
        corr, _ = PortfolioComparator._build_corr_matrix_pebc()
        # Tech_Credit=0, Tech_PE=5
        assert corr[0, 5] > 0.3, "Tech Credit-PE should have elevated correlation"


# ============================================================
# Full allocation (optimize_allocation_pebc)
# ============================================================

class TestOptimizeAllocationPebc:
    """Tests optimiseur pe-bc BL-CVaR 10 cellules."""

    def test_returns_dict(self, allocation_pebc):
        assert isinstance(allocation_pebc, dict)

    def test_method_is_10c(self, allocation_pebc):
        assert allocation_pebc["method"] == "BL-CVaR-10C"

    def test_required_backward_compat_keys(self, allocation_pebc):
        expected = {
            "credit_allocation", "pe_allocation",
            "sector_weights_credit", "sector_weights_pe",
            "raroc_credit", "raroc_pe",
            "rwa_weighted", "cet1_ratio", "cet1_headroom",
            "headroom_m", "feasible",
        }
        assert expected.issubset(set(allocation_pebc.keys()))

    def test_required_cvar_keys(self, allocation_pebc):
        expected = {
            "method", "class_weights", "cvar_95", "kappa",
            "n_scenarios", "risk_alpha", "portfolio_vol",
            "profit_rate_portfolio", "spread_compression",
            "covariance_shrinkage_lambda", "corr_10x10",
            "phase1_weights", "regulatory_adjustments",
        }
        assert expected.issubset(set(allocation_pebc.keys()))

    def test_allocations_sum_to_1(self, allocation_pebc):
        total = allocation_pebc["credit_allocation"] + allocation_pebc["pe_allocation"]
        assert abs(total - 1.0) < 1e-6

    def test_pe_allocation_respects_max(self, allocation_pebc):
        assert allocation_pebc["pe_allocation"] <= BASEL_CONFIG.pe_max_allocation

    def test_pe_allocation_positive(self, allocation_pebc):
        assert allocation_pebc["pe_allocation"] > 0

    def test_credit_sector_weights_sum_to_1(self, allocation_pebc):
        w = allocation_pebc["sector_weights_credit"]
        total = sum(w.values())
        assert abs(total - 1.0) < 0.01, f"Somme poids credit = {total}"

    def test_pe_sector_weights_sum_to_1(self, allocation_pebc):
        w = allocation_pebc["sector_weights_pe"]
        total = sum(w.values())
        assert abs(total - 1.0) < 0.01, f"Somme poids PE = {total}"

    def test_sector_weights_non_negative(self, allocation_pebc):
        for canal in ["sector_weights_credit", "sector_weights_pe"]:
            for sector, w in allocation_pebc[canal].items():
                assert w >= 0, f"{canal}/{sector}: poids={w} negatif"

    def test_all_5_sectors_in_weights(self, allocation_pebc):
        assert set(allocation_pebc["sector_weights_credit"].keys()) == set(SECTOR_NAMES)
        assert set(allocation_pebc["sector_weights_pe"].keys()) == set(SECTOR_NAMES)

    def test_cet1_ratio_positive(self, allocation_pebc):
        assert allocation_pebc["cet1_ratio"] > 0

    def test_raroc_values_finite(self, allocation_pebc):
        assert np.isfinite(allocation_pebc["raroc_credit"])
        assert np.isfinite(allocation_pebc["raroc_pe"])

    def test_cvar_positive(self, allocation_pebc):
        assert allocation_pebc["cvar_95"] > 0

    def test_kappa_positive(self, allocation_pebc):
        assert allocation_pebc["kappa"] > 0

    def test_class_weights_10_cells(self, allocation_pebc):
        assert len(allocation_pebc["class_weights"]) == 10

    def test_class_weights_sum_to_1(self, allocation_pebc):
        total = sum(allocation_pebc["class_weights"].values())
        assert abs(total - 1.0) < 0.01

    def test_corr_10x10_shape(self, allocation_pebc):
        corr = np.array(allocation_pebc["corr_10x10"])
        assert corr.shape == (10, 10)

    def test_corr_10x10_diagonal_ones(self, allocation_pebc):
        corr = np.array(allocation_pebc["corr_10x10"])
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_spread_compression_10_cells(self, allocation_pebc):
        assert len(allocation_pebc["spread_compression"]) == 10

    def test_stress_diagnostics_present(self, allocation_pebc):
        assert "stress_intensity" in allocation_pebc
        assert "illiquidity_premium" in allocation_pebc
        assert "vol_multiplier" in allocation_pebc
        assert "bl_confidence" in allocation_pebc
        assert "pe_band" in allocation_pebc
        assert isinstance(allocation_pebc["bl_confidence"], tuple)
        assert len(allocation_pebc["bl_confidence"]) == 2

    def test_regulatory_in_allocation(self, allocation_pebc):
        assert "lcr_ratio" in allocation_pebc
        assert "nsfr_ratio" in allocation_pebc
        assert "irrbb_eve_ratio" in allocation_pebc
        adj = allocation_pebc["regulatory_adjustments"]
        assert "lcr_delta" in adj
        assert "nsfr_delta" in adj
        assert "irrbb_delta" in adj

    def test_phase1_weights_sum_to_one(self, allocation_pebc):
        p1 = allocation_pebc.get("phase1_weights", {})
        if p1:
            total = sum(p1.values())
            assert abs(total - 1.0) < 0.01

    def test_covariance_shrinkage_applied(self, allocation_pebc):
        lw = allocation_pebc.get("covariance_shrinkage_lambda", None)
        assert lw is not None
        assert 0 < lw < 1

    def test_n_cells_is_10(self, allocation_pebc):
        assert allocation_pebc["n_classes"] == 10
