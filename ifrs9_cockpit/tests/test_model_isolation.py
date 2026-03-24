"""Tests d'isolation fonctionnelle — verifient que les deux modeles coexistent.

Le modele pe-bc (5 secteurs x 2 canaux, BL-CVaR-10C) et le modele 14-couches
(14 classes d'actifs, BL-CVaR-14C) doivent pouvoir s'executer independamment
sur le meme PortfolioComparator sans interference.

Tests fonctionnels uniquement (pas d'introspection AST).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.shared

from ifrs9_cockpit.config import (
    SECTORS,
    SECTOR_NAMES,
    ASSET_CLASSES,
    ASSET_CLASS_NAMES,
    PREDEFINED_SCENARIOS,
)
from ifrs9_cockpit.engine.comparator import PortfolioComparator


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def comparator(global_pipeline_results):
    return global_pipeline_results["comparator"]


@pytest.fixture(scope="module")
def alloc_14c(comparator):
    """Allocation 14 classes (optimiseur principal)."""
    return comparator.optimize_allocation()


@pytest.fixture(scope="module")
def alloc_10c(comparator):
    """Allocation pe-bc 10 cellules."""
    return comparator.optimize_allocation_pebc()


# ============================================================
# Tests d'isolation
# ============================================================


class TestDualOptimizerCoexistence:
    """Les deux optimiseurs coexistent sur le meme objet."""

    def test_both_methods_exist(self, comparator):
        assert hasattr(comparator, "optimize_allocation")
        assert hasattr(comparator, "optimize_allocation_pebc")

    def test_14c_returns_14_classes(self, alloc_14c):
        assert alloc_14c["method"] == "BL-CVaR-14C"
        assert alloc_14c["n_classes"] == 14
        assert len(alloc_14c["class_weights"]) == 14

    def test_10c_returns_10_cells(self, alloc_10c):
        assert alloc_10c["method"] == "BL-CVaR-10C"
        assert len(alloc_10c["class_weights"]) == 10

    def test_different_methods(self, alloc_14c, alloc_10c):
        assert alloc_14c["method"] != alloc_10c["method"]

    def test_both_sum_to_1(self, alloc_14c, alloc_10c):
        total_14c = alloc_14c["credit_allocation"] + alloc_14c["pe_allocation"]
        total_10c = alloc_10c["credit_allocation"] + alloc_10c["pe_allocation"]
        assert abs(total_14c - 1.0) < 1e-6
        assert abs(total_10c - 1.0) < 1e-6

    def test_both_have_regulatory_adjustments(self, alloc_14c, alloc_10c):
        for alloc in [alloc_14c, alloc_10c]:
            ra = alloc.get("regulatory_adjustments", {})
            assert "cet1_delta" in ra
            assert "lcr_delta" in ra
            assert "nsfr_delta" in ra
            assert "irrbb_delta" in ra

    def test_both_have_cvar(self, alloc_14c, alloc_10c):
        for alloc in [alloc_14c, alloc_10c]:
            assert alloc["cvar_95"] > 0
            assert alloc["kappa"] > 0

    def test_corr_matrices_different_sizes(self, alloc_14c, alloc_10c):
        corr_14 = np.array(alloc_14c["corr_10x10"])  # legacy name
        corr_10 = np.array(alloc_10c["corr_10x10"])
        assert corr_14.shape == (14, 14)
        assert corr_10.shape == (10, 10)


class TestNoInterference:
    """Appeler un optimiseur ne corrompt pas l'autre."""

    def test_sequential_calls_independent(self, comparator):
        """14C puis 10C puis 14C de nouveau — resultats stables."""
        r1 = comparator.optimize_allocation()
        _ = comparator.optimize_allocation_pebc()
        r3 = comparator.optimize_allocation()
        # Les cles numeriques doivent etre identiques (tolerance flottante)
        np.testing.assert_allclose(r1["credit_allocation"], r3["credit_allocation"], rtol=1e-10)
        np.testing.assert_allclose(r1["pe_allocation"], r3["pe_allocation"], rtol=1e-10)
        np.testing.assert_allclose(r1["cvar_95"], r3["cvar_95"], rtol=1e-10)

    def test_10c_then_14c_independent(self, comparator):
        """10C puis 14C — 14C non pollue."""
        _ = comparator.optimize_allocation_pebc()
        r = comparator.optimize_allocation()
        assert r["method"] == "BL-CVaR-14C"
        assert r["n_classes"] == 14


class TestSharedInfrastructure:
    """Les deux modeles partagent RAROC et metriques."""

    def test_same_raroc_eva(self, comparator):
        """compute_raroc_eva() est le meme objet pour les deux."""
        raroc1 = comparator.compute_raroc_eva()
        raroc2 = comparator.compute_raroc_eva()
        assert raroc1 is raroc2  # cache

    def test_same_hhi(self, comparator):
        """HHI cross-cell identique (calculé une seule fois)."""
        hhi1 = comparator.compute_hhi_crosscell()
        hhi2 = comparator.compute_hhi_crosscell()
        assert hhi1["hhi_crosscell"] == hhi2["hhi_crosscell"]


class TestConfigIsolation:
    """Verifier que les configs ne se polluent pas."""

    def test_5_sectors_immutable(self):
        assert len(SECTORS) == 5
        assert set(s.name for s in SECTORS) == {"Technologie", "Industrie", "Sante", "Immobilier", "Services"}

    def test_14_asset_classes_present(self):
        assert len(ASSET_CLASSES) == 14

    def test_12_predefined_scenarios(self):
        assert len(PREDEFINED_SCENARIOS) == 12

    def test_sector_dual_channel_attrs(self):
        """Les attributs dual-channel pe-bc sont presents sans corrompre le config 14-couches."""
        for s in SECTORS:
            assert hasattr(s, "market_vol_credit")
            assert hasattr(s, "market_vol_pe")
            assert hasattr(s, "market_capacity_credit_eur")
            assert hasattr(s, "market_capacity_pe_eur")
            assert hasattr(s, "hqla_eligible_credit")
            assert hasattr(s, "rsf_weight_credit")
            assert hasattr(s, "duration_credit")
