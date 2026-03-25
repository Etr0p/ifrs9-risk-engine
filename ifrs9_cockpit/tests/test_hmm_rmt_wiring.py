"""Tests du cablage HMM (cvar_alpha) et RMT (covariance debruitee) dans l'optimiseur.

Couvre :
    - HMM contraction → cvar_alpha=0.80 → allocation plus conservative
    - HMM recovery/expansion → cvar_alpha=0.95 → pas de changement
    - RMT diagnostics presents dans le resultat d'optimisation
    - Non-regression : scenario base → resultats proches de l'ancien hardcode
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.fourteen

from ifrs9_cockpit.engine.hmm_regime import detect_regime, REGIME_CVAR_ALPHA


# ═══════════════════════════════════════════════════════
# HMM REGIME DETECTION
# ═══════════════════════════════════════════════════════


class TestHMMWiring:
    """Tests du cablage HMM → cvar_alpha dans l'optimiseur."""

    def test_contraction_cvar_alpha(self):
        """Scenario adverse (GDP=-4, unemp=12) → contraction → cvar_alpha=0.80."""
        macro_adverse = {
            "unemployment_rate": 12.0,
            "gdp_growth": -4.0,
            "interest_rate": 1.0,
            "hpi_growth": -3.0,
            "inflation_rate": 4.0,
        }
        result = detect_regime(macro_adverse)
        assert result.regime == "contraction"
        assert result.cvar_alpha == REGIME_CVAR_ALPHA["contraction"]
        assert result.cvar_alpha == 0.80

    def test_recovery_cvar_alpha(self):
        """Scenario base (GDP=1.2, unemp=7.5) → recovery → cvar_alpha=0.95."""
        macro_base = {
            "unemployment_rate": 7.5,
            "gdp_growth": 1.2,
            "interest_rate": 2.0,
            "hpi_growth": 2.0,
            "inflation_rate": 2.0,
        }
        result = detect_regime(macro_base)
        assert result.regime == "recovery"
        assert result.cvar_alpha == 0.95

    def test_expansion_cvar_alpha(self):
        """Scenario favorable (GDP=4, unemp=4) → expansion → cvar_alpha=0.95."""
        macro_expansion = {
            "unemployment_rate": 4.0,
            "gdp_growth": 4.0,
            "interest_rate": 2.5,
            "hpi_growth": 5.0,
            "inflation_rate": 1.5,
        }
        result = detect_regime(macro_expansion)
        assert result.regime == "expansion"
        assert result.cvar_alpha == 0.95

    def test_contraction_more_conservative(self, global_comparator):
        """En contraction (cvar_alpha=0.80), l'allocation PE est <= scenario base."""
        opt_base = global_comparator.optimize_allocation(cvar_alpha=0.95)
        opt_contraction = global_comparator.optimize_allocation(cvar_alpha=0.80)

        pe_base = opt_base["class_weights"]["private_equity"]
        pe_contraction = opt_contraction["class_weights"]["private_equity"]

        # Contraction should be at least as conservative (PE <= base)
        assert pe_contraction <= pe_base + 0.01, (
            f"PE contraction={pe_contraction:.4f} should be <= PE base={pe_base:.4f}"
        )

    def test_cvar_alpha_in_output(self, global_comparator):
        """Le cvar_alpha utilise est expose dans le resultat."""
        opt = global_comparator.optimize_allocation(cvar_alpha=0.80)
        assert "cvar_alpha" in opt
        assert opt["cvar_alpha"] == 0.80


# ═══════════════════════════════════════════════════════
# RMT DENOISING IN OPTIMIZER
# ═══════════════════════════════════════════════════════


class TestRMTWiring:
    """Tests du cablage RMT dans l'optimiseur."""

    def test_rmt_diagnostics_in_output(self, global_comparator):
        """Les diagnostics RMT sont presents dans le resultat d'optimisation."""
        opt = global_comparator.optimize_allocation()
        assert "rmt_n_signal" in opt
        assert "rmt_n_noise" in opt
        assert "rmt_noise_fraction" in opt

    def test_rmt_signal_noise_partition(self, global_comparator):
        """n_signal + n_noise = n_classes (14)."""
        opt = global_comparator.optimize_allocation()
        assert opt["rmt_n_signal"] + opt["rmt_n_noise"] == opt["n_classes"]

    def test_rmt_noise_fraction_positive(self, global_comparator):
        """La fraction de bruit est >= 0."""
        opt = global_comparator.optimize_allocation()
        assert opt["rmt_noise_fraction"] >= 0.0
        assert opt["rmt_noise_fraction"] <= 1.0


# ═══════════════════════════════════════════════════════
# NON-REGRESSION
# ═══════════════════════════════════════════════════════


class TestNonRegression:
    """Verifie que l'ajout de HMM/RMT ne casse pas le pipeline."""

    def test_default_still_feasible(self, global_comparator):
        """Avec les valeurs par defaut (cvar_alpha=0.95), l'allocation reste faisable."""
        opt = global_comparator.optimize_allocation()
        assert opt["feasible"]
        assert opt["cet1_ratio"] >= 0.10

    def test_default_regulatory_compliance(self, global_comparator):
        """Les normes reglementaires restent respectees."""
        opt = global_comparator.optimize_allocation()
        assert opt["lcr_compliant"]
        assert opt["nsfr_compliant"]
        assert opt["irrbb_compliant"]

    def test_weights_sum_to_one(self, global_comparator):
        """Les poids d'allocation somment a 1."""
        opt = global_comparator.optimize_allocation()
        total = sum(opt["class_weights"].values())
        assert abs(total - 1.0) < 0.01

    def test_contraction_still_feasible(self, global_comparator):
        """Meme en contraction (cvar_alpha=0.80), l'allocation reste faisable."""
        opt = global_comparator.optimize_allocation(cvar_alpha=0.80)
        assert opt["feasible"]
        assert opt["cet1_ratio"] >= 0.10
        total = sum(opt["class_weights"].values())
        assert abs(total - 1.0) < 0.01


# ═══════════════════════════════════════════════════════
# HMM AUTO-WIRING (macro_params → regime → cvar_alpha)
# ═══════════════════════════════════════════════════════


class TestHMMAutoWiring:
    """Tests that macro_params auto-triggers HMM regime detection."""

    def test_14c_adverse_macro_detects_contraction(self, global_comparator):
        """14C optimizer auto-detects contraction when macro_params is adverse."""
        macro_adverse = {
            "unemployment_rate": 12.0,
            "gdp_growth": -4.0,
            "interest_rate": 1.0,
            "hpi_growth": -3.0,
            "inflation_rate": 4.0,
        }
        opt = global_comparator.optimize_allocation(macro_params=macro_adverse)
        assert opt["hmm_regime"] == "contraction"
        assert opt["cvar_alpha"] == 0.80

    def test_14c_base_macro_detects_recovery(self, global_comparator):
        """14C optimizer auto-detects recovery for base macro conditions."""
        macro_base = {
            "unemployment_rate": 7.5,
            "gdp_growth": 1.2,
            "interest_rate": 2.0,
            "hpi_growth": 2.0,
            "inflation_rate": 2.0,
        }
        opt = global_comparator.optimize_allocation(macro_params=macro_base)
        assert opt["hmm_regime"] == "recovery"
        assert opt["cvar_alpha"] == 0.95

    def test_14c_no_macro_no_regime(self, global_comparator):
        """Without macro_params, hmm_regime is None."""
        opt = global_comparator.optimize_allocation()
        assert opt["hmm_regime"] is None
        assert opt["cvar_alpha"] == 0.95

    def test_14c_explicit_alpha_overrides_hmm(self, global_comparator):
        """Explicit cvar_alpha != 0.95 should bypass HMM detection."""
        macro_adverse = {
            "unemployment_rate": 12.0,
            "gdp_growth": -4.0,
            "interest_rate": 1.0,
            "hpi_growth": -3.0,
            "inflation_rate": 4.0,
        }
        opt = global_comparator.optimize_allocation(
            macro_params=macro_adverse, cvar_alpha=0.90,
        )
        # Explicit alpha overrides: HMM not triggered
        assert opt["cvar_alpha"] == 0.90
        assert opt["hmm_regime"] is None

    def test_10c_adverse_macro_detects_contraction(self, global_comparator):
        """10C pe-bc optimizer auto-detects contraction."""
        macro_adverse = {
            "unemployment_rate": 12.0,
            "gdp_growth": -4.0,
            "interest_rate": 1.0,
            "hpi_growth": -3.0,
            "inflation_rate": 4.0,
        }
        opt = global_comparator.optimize_allocation_pebc(macro_params=macro_adverse)
        assert opt["hmm_regime"] == "contraction"
        assert opt["cvar_alpha"] == 0.80

    def test_10c_no_macro_no_regime(self, global_comparator):
        """10C: without macro_params, hmm_regime is None."""
        opt = global_comparator.optimize_allocation_pebc()
        assert opt.get("hmm_regime") is None
        assert opt["cvar_alpha"] == 0.95
