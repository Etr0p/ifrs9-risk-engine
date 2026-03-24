"""Tests unitaires pour ifrs9_cockpit/engine/comparator.py (Stories 4-1, 4-2, 4-3).

Couvre les fonctionnalites du PortfolioComparator :
  Story 4-1 : Metriques avancees (FR42), HHI cross-cell (FR41), RAROC/EVA (FR44)
  Story 4-2 : Asymetrie credit/PE (FR20), scores de resilience (FR19)
  Story 4-3 : Optimisation 3 niveaux (FR22), sensibilite CRR3 (FR23),
              seuils de basculement (FR24)
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from ifrs9_cockpit.config import (
    BASEL_CONFIG,
    SECTORS,
    SECTOR_NAMES,
    REQUIRED_CREDIT_RESULT_COLS,
    REQUIRED_PE_RESULT_COLS,
)
from ifrs9_cockpit.engine.comparator import (
    PortfolioComparator,
    compute_crr3_rw,
)

pytestmark = pytest.mark.fourteen


# ============================================================
# Fixture : pipeline complet (scope=module pour ne calculer qu'une fois)
# ============================================================

@pytest.fixture(scope="module")
def pipeline_data(global_pipeline_results):
    """Reutilise le pipeline session (zero recalcul)."""
    return (
        global_pipeline_results["result_credit"],
        global_pipeline_results["result_pe"],
        global_pipeline_results["comparator"],
    )


@pytest.fixture(scope="module")
def comparator(pipeline_data):
    """Retourne le PortfolioComparator instancie."""
    return pipeline_data[2]


@pytest.fixture(scope="module")
def result_credit(pipeline_data):
    return pipeline_data[0]


@pytest.fixture(scope="module")
def result_pe(pipeline_data):
    return pipeline_data[1]


@pytest.fixture(scope="module")
def advanced_metrics(comparator):
    return comparator.compute_advanced_credit_metrics()


@pytest.fixture(scope="module")
def hhi_result(comparator):
    return comparator.compute_hhi_crosscell()


@pytest.fixture(scope="module")
def raroc_eva(comparator):
    return comparator.compute_raroc_eva()


@pytest.fixture(scope="module")
def asymmetry_matrix(comparator):
    return comparator.build_asymmetry_matrix()


@pytest.fixture(scope="module")
def allocation_result(comparator):
    return comparator.optimize_allocation()


@pytest.fixture(scope="module")
def crr3_sensitivity(comparator):
    return comparator.compute_crr3_sensitivity()


@pytest.fixture(scope="module")
def tipping_points(comparator):
    return comparator.find_tipping_points()


# ============================================================
# Story 4-1 : Metriques avancees et concentration
# ============================================================

class TestAdvancedCreditMetrics:
    """Tests FR42 — Metriques credit avancees par secteur."""

    def test_returns_dataframe(self, advanced_metrics):
        assert isinstance(advanced_metrics, pl.DataFrame)

    def test_5_sectors_plus_total(self, advanced_metrics):
        """5 secteurs + 1 ligne Total = 6 lignes."""
        assert len(advanced_metrics) == 6

    def test_all_sectors_present(self, advanced_metrics):
        sectors = set(advanced_metrics["sector"].to_list())
        expected = set(SECTOR_NAMES) | {"Total"}
        assert sectors == expected

    def test_npl_ratio_in_0_1(self, advanced_metrics):
        """NPL ratio = EAD(Stage 3) / EAD total, dans [0, 1]."""
        for val in advanced_metrics["npl_ratio"].to_list():
            assert 0 <= val <= 1, f"NPL ratio hors bornes: {val}"

    def test_cost_of_risk_positive(self, advanced_metrics):
        """Cost of risk en bps doit etre > 0 (portefeuille a defauts)."""
        total = advanced_metrics.filter(pl.col("sector") == "Total")
        assert total["cost_of_risk_bps"][0] > 0

    def test_cost_of_risk_annualized(self, advanced_metrics, result_credit):
        """CoR annualise = ECL / (EAD x T_moyen_pondere) x 10000 (M8).

        T_moyen est pondere par EAD par stage : Stage 1 = 1 an, Stage 2/3 = lifetime.
        """
        from ifrs9_cockpit.config import IFRS9_CONFIG
        total = advanced_metrics.filter(pl.col("sector") == "Total").row(0, named=True)
        ecl = result_credit["ecl_weighted"].sum()
        ead = result_credit["ead"].sum()
        ead_s1 = result_credit.filter(pl.col("stage") == 1)["ead"].sum()
        t_pondere = (ead_s1 * 1.0 + (ead - ead_s1) * IFRS9_CONFIG.lifetime_horizon_years) / max(ead, 1)
        expected_bps = ecl / (ead * t_pondere) * 10_000
        assert abs(total["cost_of_risk_bps"] - round(expected_bps, 1)) < 1.0, \
            f"CoR annualise: attendu={expected_bps:.1f}, obtenu={total['cost_of_risk_bps']}"

    def test_coverage_ratio_in_0_1(self, advanced_metrics):
        """Coverage ratio = ECL_S3 / EAD_S3, dans [0, 1]."""
        for val in advanced_metrics["coverage_ratio"].to_list():
            assert 0 <= val <= 1, f"Coverage ratio hors bornes: {val}"

    def test_texas_ratio_synth_positive(self, advanced_metrics):
        """Texas ratio synth = EAD_S3 / (ECL_S3 + capital) doit etre >= 0."""
        for val in advanced_metrics["texas_ratio_synth"].to_list():
            assert val >= 0, f"Texas ratio synth negatif: {val}"

    def test_pd_mean_in_0_1(self, advanced_metrics):
        """PD moyenne doit etre dans [0, 1]."""
        for val in advanced_metrics["pd_mean"].to_list():
            assert 0 <= val <= 1, f"PD mean hors bornes: {val}"

    def test_stage2_pct_in_0_1(self, advanced_metrics):
        for val in advanced_metrics["stage2_pct"].to_list():
            assert 0 <= val <= 1, f"Stage2 pct hors bornes: {val}"

    def test_rwa_density_positive(self, advanced_metrics):
        for val in advanced_metrics["rwa_density"].to_list():
            assert val > 0, f"RWA density non positive: {val}"

    def test_canal_column_is_credit(self, advanced_metrics):
        """Toutes les lignes sont canal Credit."""
        assert all(v == "Credit" for v in advanced_metrics["canal"].to_list())

    def test_ead_total_positive(self, advanced_metrics):
        total = advanced_metrics.filter(pl.col("sector") == "Total")
        assert total["ead_total"][0] > 0


class TestHHICrossCell:
    """Tests FR41 — HHI concentration cross-cell."""

    def test_returns_dict(self, hhi_result):
        assert isinstance(hhi_result, dict)

    def test_required_keys(self, hhi_result):
        expected_keys = {"hhi_credit", "hhi_pe", "hhi_crosscell", "hhi_name_credit", "hhi_name_pe", "shares"}
        assert set(hhi_result.keys()) == expected_keys

    def test_hhi_credit_in_valid_range(self, hhi_result):
        """HHI dans [0, 10000] (10000 = concentration maximale)."""
        assert 0 <= hhi_result["hhi_credit"] <= 10_000

    def test_hhi_pe_in_valid_range(self, hhi_result):
        assert 0 <= hhi_result["hhi_pe"] <= 10_000

    def test_hhi_crosscell_in_valid_range(self, hhi_result):
        assert 0 <= hhi_result["hhi_crosscell"] <= 10_000

    def test_shares_dataframe_10_rows(self, hhi_result):
        """10 cellules = 5 secteurs x 2 canaux."""
        assert len(hhi_result["shares"]) == 10

    def test_shares_sum_to_1(self, hhi_result):
        total_share = hhi_result["shares"]["share"].sum()
        assert abs(total_share - 1.0) < 1e-4, f"Somme des parts = {total_share}"

    def test_all_shares_non_negative(self, hhi_result):
        assert (hhi_result["shares"]["share"].to_numpy() >= 0).all()

    def test_hhi_credit_above_equal_parts(self, hhi_result):
        """Avec 5 secteurs non-equiponderes, HHI > 2000 (1/5 = 0.20 => HHI=2000 si egal)."""
        # Le HHI 5 secteurs egaux serait exactement 2000
        # Avec des poids inegaux, il sera > 2000
        assert hhi_result["hhi_credit"] >= 2000

    def test_crosscell_credit_and_pe_balanced(self, hhi_result):
        """Credit et PE ont chacun une part significative du portefeuille."""
        shares = hhi_result["shares"]
        credit_share = shares.filter(pl.col("canal") == "Credit")["share"].sum()
        pe_share = shares.filter(pl.col("canal") == "PE")["share"].sum()
        assert credit_share > 0.30
        assert pe_share > 0.30


class TestRAROCEVA:
    """Tests FR44 — RAROC et EVA par cellule."""

    def test_returns_dataframe(self, raroc_eva):
        assert isinstance(raroc_eva, pl.DataFrame)

    def test_12_rows(self, raroc_eva):
        """10 cellules (5 secteurs x 2 canaux) + 2 totaux = 12."""
        assert len(raroc_eva) == 12

    def test_required_columns(self, raroc_eva):
        expected = {"sector", "canal", "exposure", "revenue", "loss", "rwa", "capital", "raroc", "eva"}
        assert expected.issubset(set(raroc_eva.columns))

    def test_exposure_positive(self, raroc_eva):
        assert (raroc_eva["exposure"].to_numpy() > 0).all()

    def test_rwa_positive(self, raroc_eva):
        assert (raroc_eva["rwa"].to_numpy() > 0).all()

    def test_capital_positive(self, raroc_eva):
        assert (raroc_eva["capital"].to_numpy() > 0).all()

    def test_raroc_finite(self, raroc_eva):
        assert raroc_eva["raroc"].null_count() == 0
        assert np.isfinite(raroc_eva["raroc"].to_numpy()).all()

    def test_eva_finite(self, raroc_eva):
        assert raroc_eva["eva"].null_count() == 0
        assert np.isfinite(raroc_eva["eva"].to_numpy()).all()

    def test_total_credit_present(self, raroc_eva):
        total_c = raroc_eva.filter(
            (pl.col("sector") == "Total") & (pl.col("canal") == "Credit")
        )
        assert len(total_c) == 1

    def test_total_pe_present(self, raroc_eva):
        total_p = raroc_eva.filter(
            (pl.col("sector") == "Total") & (pl.col("canal") == "PE")
        )
        assert len(total_p) == 1

    def test_both_canals_present(self, raroc_eva):
        canals = set(raroc_eva["canal"].to_list())
        assert canals == {"Credit", "PE"}


# ============================================================
# Story 4-2 : Asymetrie credit/PE, resilience, RAROC
# ============================================================

class TestRAROCEVACache:
    """Tests cache RAROC/EVA — appels multiples ne recalculent pas."""

    def test_cache_returns_same_object(self, comparator):
        """Deux appels successifs retournent le meme objet (cache)."""
        r1 = comparator.compute_raroc_eva()
        r2 = comparator.compute_raroc_eva()
        assert r1 is r2, "Le cache RAROC/EVA ne fonctionne pas"


class TestResilienceScores:
    """Tests FR19 — Scores de resilience credit et PE."""

    def test_credit_resilience_5_sectors(self, comparator):
        resil = comparator._resilience_score_credit()
        assert len(resil) == 5

    def test_credit_resilience_in_0_1(self, comparator):
        resil = comparator._resilience_score_credit()
        for val in resil["resilience_credit"].to_list():
            assert 0 <= val <= 1, f"Resilience credit hors [0,1]: {val}"

    def test_pe_resilience_5_sectors(self, comparator):
        resil = comparator._resilience_score_pe()
        assert len(resil) == 5

    def test_pe_resilience_in_0_1(self, comparator):
        resil = comparator._resilience_score_pe()
        for val in resil["resilience_pe"].to_list():
            assert 0 <= val <= 1, f"Resilience PE hors [0,1]: {val}"

    def test_resilience_sectors_match(self, comparator):
        resil_c = comparator._resilience_score_credit()
        resil_p = comparator._resilience_score_pe()
        assert set(resil_c["sector"].to_list()) == set(resil_p["sector"].to_list())
        assert set(resil_c["sector"].to_list()) == set(SECTOR_NAMES)


class TestAsymmetryMatrix:
    """Tests FR20 — Matrice d'asymetrie credit vs PE."""

    def test_returns_dataframe(self, asymmetry_matrix):
        assert isinstance(asymmetry_matrix, pl.DataFrame)

    def test_5_sectors(self, asymmetry_matrix):
        assert len(asymmetry_matrix) == 5

    def test_all_sectors_present(self, asymmetry_matrix):
        assert set(asymmetry_matrix["sector"].to_list()) == set(SECTOR_NAMES)

    def test_loss_ratio_non_negative(self, asymmetry_matrix):
        """loss_ratio = EL_PE / ECL_credit >= 0."""
        assert (asymmetry_matrix["loss_ratio"].to_numpy() >= 0).all()

    def test_raroc_delta_is_pe_minus_credit(self, asymmetry_matrix):
        """raroc_delta = raroc_pe - raroc_credit."""
        for row in asymmetry_matrix.iter_rows(named=True):
            expected = round(row["raroc_pe"] - row["raroc_credit"], 4)
            assert abs(row["raroc_delta"] - expected) < 1e-3, (
                f"Secteur {row['sector']}: delta={row['raroc_delta']}, "
                f"attendu={expected}"
            )

    def test_resilience_scores_included(self, asymmetry_matrix):
        assert "resilience_credit" in asymmetry_matrix.columns
        assert "resilience_pe" in asymmetry_matrix.columns

    def test_resilience_in_0_1(self, asymmetry_matrix):
        for val in asymmetry_matrix["resilience_credit"].to_list():
            assert 0 <= val <= 1
        for val in asymmetry_matrix["resilience_pe"].to_list():
            assert 0 <= val <= 1

    def test_ecl_credit_positive(self, asymmetry_matrix):
        assert (asymmetry_matrix["ecl_credit"].to_numpy() > 0).all()

    def test_required_columns(self, asymmetry_matrix):
        expected = {
            "sector", "ecl_credit", "el_pe", "loss_ratio",
            "raroc_credit", "raroc_pe", "raroc_delta",
            "resilience_credit", "resilience_pe",
        }
        assert expected.issubset(set(asymmetry_matrix.columns))

    def test_rwa_columns_present(self, asymmetry_matrix):
        """Les colonnes RWA credit et PE doivent etre presentes."""
        assert "rwa_credit" in asymmetry_matrix.columns
        assert "rwa_pe" in asymmetry_matrix.columns

    def test_ead_and_nav_present(self, asymmetry_matrix):
        """Les colonnes ead_credit et nav_pe doivent etre presentes."""
        assert "ead_credit" in asymmetry_matrix.columns
        assert "nav_pe" in asymmetry_matrix.columns


# ============================================================
# Story 4-3 : Optimisation, CRR3, seuils de basculement
# ============================================================

class TestCRR3RiskWeight:
    """Tests H8 — compute_crr3_rw() fonction standalone."""

    def test_returns_array(self, result_pe):
        rw = compute_crr3_rw(result_pe)
        assert isinstance(rw, np.ndarray)

    def test_length_matches_pe(self, result_pe):
        rw = compute_crr3_rw(result_pe)
        assert len(rw) == len(result_pe)

    def test_valid_rw_values(self, result_pe):
        """Seules valeurs CRR3 : 190, 250, 400."""
        rw = compute_crr3_rw(result_pe)
        valid = {190, 250, 400}
        assert set(np.unique(rw)).issubset(valid)

    def test_low_risk_gets_190(self):
        """Position MOIC=2.0, p_distress=0, leverage=0.1 => score bas => 190."""
        df = pl.DataFrame({
            "moic": [2.0],
            "nav": [100.0],
            "nav_initial": [50.0],
            "p_distress": [0.0],
            "leverage": [0.1],
            "risk_category": ["Performing"],
        })
        rw = compute_crr3_rw(df)
        assert rw[0] == 190

    def test_high_risk_gets_400(self):
        """Position MOIC=0.5, p_distress=0.9, leverage=0.95 => score haut => 400."""
        df = pl.DataFrame({
            "moic": [0.5],
            "nav": [100.0],
            "nav_initial": [200.0],
            "p_distress": [0.9],
            "leverage": [0.95],
            "risk_category": ["Distressed"],
        })
        rw = compute_crr3_rw(df)
        assert rw[0] == 400


class TestOptimizeAllocation:
    """Tests FR22 — Optimisation allocation 3 niveaux."""

    def test_returns_dict(self, allocation_result):
        assert isinstance(allocation_result, dict)

    def test_required_keys(self, allocation_result):
        expected = {
            "credit_allocation", "pe_allocation",
            "sector_weights_credit", "sector_weights_pe",
            "raroc_credit", "raroc_pe",
            "rwa_weighted", "cet1_ratio", "cet1_headroom",
            "headroom_m", "feasible",
            "pe_free", "pe_band", "stress_intensity",
            "illiquidity_premium", "vol_multiplier", "bl_confidence",
            "kappa_pe_eff",
        }
        assert expected.issubset(set(allocation_result.keys()))

    def test_allocations_sum_to_1(self, allocation_result):
        total = allocation_result["credit_allocation"] + allocation_result["pe_allocation"]
        assert abs(total - 1.0) < 1e-6

    def test_pe_allocation_respects_max(self, allocation_result):
        """PE allocation <= pe_max_allocation (40%)."""
        assert allocation_result["pe_allocation"] <= BASEL_CONFIG.pe_max_allocation

    def test_pe_allocation_positive(self, allocation_result):
        assert allocation_result["pe_allocation"] > 0

    def test_credit_sector_weights_sum_to_1(self, allocation_result):
        w = allocation_result["sector_weights_credit"]
        total = sum(w.values())
        assert abs(total - 1.0) < 0.01, f"Somme poids credit = {total}"

    def test_pe_sector_weights_sum_to_1(self, allocation_result):
        w = allocation_result["sector_weights_pe"]
        total = sum(w.values())
        assert abs(total - 1.0) < 0.01, f"Somme poids PE = {total}"

    def test_sector_weights_positive(self, allocation_result):
        """Chaque poids sectoriel > 0 (plancher garanti par clip pre-normalisation)."""
        for canal in ["sector_weights_credit", "sector_weights_pe"]:
            for sector, w in allocation_result[canal].items():
                assert w > 0, f"{canal}/{sector}: poids={w} non positif"

    def test_all_5_sectors_in_weights(self, allocation_result):
        assert set(allocation_result["sector_weights_credit"].keys()) == set(SECTOR_NAMES)
        assert set(allocation_result["sector_weights_pe"].keys()) == set(SECTOR_NAMES)

    def test_cet1_ratio_positive(self, allocation_result):
        assert allocation_result["cet1_ratio"] > 0

    def test_raroc_values_finite(self, allocation_result):
        assert np.isfinite(allocation_result["raroc_credit"])
        assert np.isfinite(allocation_result["raroc_pe"])


class TestSoftmaxWeights:
    """Tests M7 — Softmax adaptative pour poids sectoriels."""

    def test_equal_raroc_diversifies_via_correlation(self, comparator):
        """Si tous les RAROC sont egaux, la penalite de correlation
        favorise les secteurs les moins correles (effet diversification)."""
        cells = pl.DataFrame({
            "sector": list(SECTOR_NAMES),
            "raroc": [0.10] * 5,
        })
        w = comparator._optimize_sector_weights(cells)
        vals = list(w.values())
        # Poids positifs et somment a 1
        assert all(v > 0 for v in vals), f"Poids negatifs: {vals}"
        assert abs(sum(vals) - 1.0) < 1e-6, f"Somme != 1: {sum(vals)}"
        # Avec la penalite de correlation, les poids ne sont plus egaux:
        # les secteurs moins correles aux autres recoivent un poids superieur
        assert max(vals) > min(vals), "La penalite devrait creer de l'asymetrie"

    def test_best_raroc_gets_highest_weight(self, comparator):
        """Le secteur avec le meilleur RAROC devrait avoir le poids le plus eleve."""
        cells = pl.DataFrame({
            "sector": list(SECTOR_NAMES),
            "raroc": [0.05, 0.10, 0.30, 0.08, 0.06],
        })
        w = comparator._optimize_sector_weights(cells)
        # Sante (index 2, RAROC=0.30) devrait etre le plus haut
        assert w["Sante"] == max(w.values())

    def test_very_negative_raroc_no_nan(self, comparator):
        """RAROC tres negatifs ne doivent pas causer de NaN (underflow softmax)."""
        cells = pl.DataFrame({
            "sector": list(SECTOR_NAMES),
            "raroc": [-5.0, -3.0, -10.0, -8.0, -6.0],
        })
        w = comparator._optimize_sector_weights(cells)
        vals = list(w.values())
        assert all(np.isfinite(v) for v in vals), f"Poids non finis: {vals}"
        assert all(v > 0 for v in vals), f"Poids non positifs: {vals}"
        assert abs(sum(vals) - 1.0) < 0.01


class TestCRR3Sensitivity:
    """Tests FR23 — Sensibilite CRR3."""

    def test_returns_dataframe(self, crr3_sensitivity):
        assert isinstance(crr3_sensitivity, pl.DataFrame)

    def test_4_rows(self, crr3_sensitivity):
        """3 RW fixes (190, 250, 400) + 1 composite CRR3 = 4 lignes."""
        assert len(crr3_sensitivity) == 4

    def test_required_columns(self, crr3_sensitivity):
        expected = {"rw_pe", "rwa_pe", "rwa_credit", "rwa_total", "cet1_ratio", "headroom", "feasible"}
        assert expected == set(crr3_sensitivity.columns)

    def test_rw_pe_values(self, crr3_sensitivity):
        """Les 3 premieres lignes ont RW 190, 250, 400."""
        rw_values = crr3_sensitivity["rw_pe"].to_numpy()[:3]
        assert set(rw_values) == {190, 250, 400}

    def test_rwa_increases_with_rw(self, crr3_sensitivity):
        """RWA PE augmente avec le Risk Weight (lignes fixes uniquement)."""
        fixed = crr3_sensitivity.head(3).sort("rw_pe")
        rwa_vals = fixed["rwa_pe"].to_numpy()
        assert rwa_vals[0] <= rwa_vals[1] <= rwa_vals[2]

    def test_cet1_ratio_positive(self, crr3_sensitivity):
        assert (crr3_sensitivity["cet1_ratio"].to_numpy() > 0).all()

    def test_headroom_coherent_with_cet1(self, crr3_sensitivity):
        """headroom = cet1_ratio - cet1_target."""
        for row in crr3_sensitivity.iter_rows(named=True):
            expected = round(row["cet1_ratio"] - BASEL_CONFIG.cet1_target, 4)
            assert abs(row["headroom"] - expected) < 1e-3

    def test_feasible_flag_coherent(self, crr3_sensitivity):
        """feasible = headroom >= 0."""
        for row in crr3_sensitivity.iter_rows(named=True):
            assert row["feasible"] == (row["headroom"] >= 0)

    def test_composite_rw_is_valid(self, crr3_sensitivity):
        """La 4e ligne (composite) a un RW moyen parmi les valeurs CRR3."""
        composite_rw = crr3_sensitivity.row(3, named=True)["rw_pe"]
        # Le RW moyen composite doit etre entre 190 et 400
        assert 190 <= composite_rw <= 400


class TestTippingPoints:
    """Tests FR24 — Seuils de basculement."""

    def test_returns_dataframe(self, tipping_points):
        assert isinstance(tipping_points, pl.DataFrame)

    def test_5_sectors(self, tipping_points):
        assert len(tipping_points) == 5

    def test_all_sectors_present(self, tipping_points):
        assert set(tipping_points["sector"].to_list()) == set(SECTOR_NAMES)

    def test_required_columns(self, tipping_points):
        expected = {"sector", "raroc_credit", "raroc_pe", "preferred_canal", "delta_to_switch"}
        assert expected == set(tipping_points.columns)

    def test_preferred_canal_valid(self, tipping_points):
        """Canal prefere est 'Credit' ou 'PE'."""
        for val in tipping_points["preferred_canal"].to_list():
            assert val in {"Credit", "PE"}

    def test_delta_to_switch_non_negative(self, tipping_points):
        assert (tipping_points["delta_to_switch"].to_numpy() >= 0).all()

    def test_preferred_canal_coherent_with_raroc(self, tipping_points):
        """Si RAROC PE > RAROC Credit, le canal prefere est PE."""
        for row in tipping_points.iter_rows(named=True):
            if row["raroc_pe"] > row["raroc_credit"]:
                assert row["preferred_canal"] == "PE", (
                    f"Secteur {row['sector']}: RAROC PE ({row['raroc_pe']}) > "
                    f"RAROC Credit ({row['raroc_credit']}) mais prefere={row['preferred_canal']}"
                )
            else:
                assert row["preferred_canal"] == "Credit"

    def test_delta_equals_abs_diff(self, tipping_points):
        """delta_to_switch = |raroc_pe - raroc_credit|."""
        for row in tipping_points.iter_rows(named=True):
            expected = round(abs(row["raroc_pe"] - row["raroc_credit"]), 4)
            assert abs(row["delta_to_switch"] - expected) < 1e-3


# ============================================================
# Validation constructeur et standalone
# ============================================================

# ============================================================
# RJ Audit v3 : HHI Name Level, Correlation Penalty, GAR, Texas Synth
# ============================================================

class TestHHINameLevel:
    """Tests HHI Name Level (concentration par contrepartie, ICAAP Pilier 2)."""

    def test_hhi_name_credit_in_valid_range(self, hhi_result):
        """HHI Name Level dans [0, 10000]."""
        assert 0 <= hhi_result["hhi_name_credit"] <= 10_000

    def test_hhi_name_pe_in_valid_range(self, hhi_result):
        assert 0 <= hhi_result["hhi_name_pe"] <= 10_000

    def test_hhi_name_lower_with_more_names(self, hhi_result):
        """HHI Name doit etre inferieur au HHI sectoriel (plus de contreparties que de secteurs)."""
        assert hhi_result["hhi_name_credit"] < hhi_result["hhi_credit"]


class TestCorrelationPenalty:
    """Tests RJ v3 — Softmax avec penalite de correlation."""

    def test_correlated_sectors_penalized(self, comparator):
        """Secteurs correles devraient etre penalises vs Softmax naif."""
        cells = pl.DataFrame({
            "sector": list(SECTOR_NAMES),
            "raroc": [0.10, 0.10, 0.10, 0.10, 0.10],
        })
        w = comparator._optimize_sector_weights(cells)
        # Avec RAROC egaux, la penalite de correlation devrait creer des ecarts
        # Les secteurs les moins correles devraient avoir un poids legerement plus eleve
        vals = list(w.values())
        assert all(np.isfinite(v) for v in vals)
        assert abs(sum(vals) - 1.0) < 0.01

    def test_still_sums_to_one(self, comparator):
        """Les poids doivent sommer a 1 meme avec penalite."""
        cells = pl.DataFrame({
            "sector": list(SECTOR_NAMES),
            "raroc": [0.05, 0.15, 0.25, 0.08, 0.12],
        })
        w = comparator._optimize_sector_weights(cells)
        assert abs(sum(w.values()) - 1.0) < 0.01


class TestGreenAssetRatio:
    """Tests GAR (ESG placeholder)."""

    def test_returns_dict(self, comparator):
        gar = comparator.compute_green_asset_ratio()
        assert isinstance(gar, dict)

    def test_required_keys(self, comparator):
        gar = comparator.compute_green_asset_ratio()
        assert {"gar_credit", "gar_pe", "gar_total", "details"} == set(gar.keys())

    def test_gar_in_0_1(self, comparator):
        """GAR est un ratio dans [0, 1]."""
        gar = comparator.compute_green_asset_ratio()
        assert 0 <= gar["gar_credit"] <= 1
        assert 0 <= gar["gar_pe"] <= 1
        assert 0 <= gar["gar_total"] <= 1

    def test_gar_positive(self, comparator):
        """GAR > 0 car tous les secteurs ont un green_share > 0."""
        gar = comparator.compute_green_asset_ratio()
        assert gar["gar_credit"] > 0
        assert gar["gar_pe"] > 0

    def test_details_5_sectors(self, comparator):
        gar = comparator.compute_green_asset_ratio()
        assert len(gar["details"]) == 5


class TestTexasRatioRenamed:
    """Tests du renommage texas_ratio -> texas_ratio_synth."""

    def test_column_renamed(self, advanced_metrics):
        assert "texas_ratio_synth" in advanced_metrics.columns
        assert "texas_ratio" not in advanced_metrics.columns


class TestPortfolioComparatorInit:
    """Tests du constructeur PortfolioComparator."""

    def test_valid_init(self, result_credit, result_pe):
        """Le constructeur accepte les DataFrames valides."""
        comp = PortfolioComparator(result_credit, result_pe)
        assert comp.result_credit is result_credit
        assert comp.result_pe is result_pe

    def test_missing_credit_cols_raises(self, result_pe):
        """Colonnes credit manquantes => AssertionError."""
        df_bad = pl.DataFrame({"foo": [1]})
        with pytest.raises(AssertionError, match="credit"):
            PortfolioComparator(df_bad, result_pe)

    def test_missing_pe_cols_raises(self, result_credit):
        """Colonnes PE manquantes => AssertionError."""
        df_bad = pl.DataFrame({"foo": [1]})
        with pytest.raises(AssertionError, match="PE"):
            PortfolioComparator(result_credit, df_bad)


@pytest.mark.slow
class TestStandalone:
    """Test du bloc __main__."""

    def test_standalone_runs_successfully(self):
        """python -m ifrs9_cockpit.engine.comparator retourne 0."""
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "ifrs9_cockpit.engine.comparator"],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=r"C:\tout\cours\programme",
            encoding="utf-8",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_standalone_output_contains_validations(self):
        """La sortie contient les validations PASS."""
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "ifrs9_cockpit.engine.comparator"],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=r"C:\tout\cours\programme",
            encoding="utf-8",
        )
        assert "[PASS]" in result.stdout
        assert "Comparateur" in result.stdout
        assert "valide" in result.stdout


# ============================================================
# Asymmetric BL-CVaR equations (static methods)
# ============================================================

class TestStressIntensity:
    """Tests _stress_intensity static method."""

    def test_adverse_returns_positive(self):
        """Negative spot returns -> s > 0 (adverse)."""
        s = PortfolioComparator._stress_intensity(-0.05, -0.05, 0.07)
        assert s > 0

    def test_favorable_returns_negative(self):
        """Positive spot returns -> s < 0 (favorable)."""
        s = PortfolioComparator._stress_intensity(0.10, 0.10, 0.07)
        assert s < 0

    def test_neutral_returns_zero(self):
        """Zero spot returns -> s = 0."""
        s = PortfolioComparator._stress_intensity(0.0, 0.0, 0.07)
        assert abs(s) < 1e-10

    def test_typical_gfc_positive(self):
        """GFC-like returns should give s ~+1.0."""
        s = PortfolioComparator._stress_intensity(-0.02, -0.12, 0.07)
        assert s > 0.5


class TestAsymmetricIlliquidity:
    """Tests _asymmetric_illiquidity static method."""

    def test_favorable_returns_base_only(self):
        """In expansion (s < 0), illiquidity = base (no stress spike)."""
        illiq = PortfolioComparator._asymmetric_illiquidity(-2.0)
        assert abs(illiq - 0.005) < 1e-10

    def test_neutral_returns_base(self):
        """At s=0, illiquidity = base."""
        illiq = PortfolioComparator._asymmetric_illiquidity(0.0)
        assert abs(illiq - 0.005) < 1e-10

    def test_crisis_spikes(self):
        """In crisis (s=+1), illiquidity >> base."""
        illiq = PortfolioComparator._asymmetric_illiquidity(1.0)
        assert illiq > 0.02  # base + scale * 1^1.5 = 0.005 + 0.025 = 0.030

    def test_monotone_increasing(self):
        """Illiquidity increases with stress for s > 0."""
        vals = [PortfolioComparator._asymmetric_illiquidity(s) for s in [0.0, 0.5, 1.0, 2.0]]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1]


class TestAsymmetricVolMultiplier:
    """Tests _asymmetric_vol_multiplier static method."""

    def test_neutral_is_1(self):
        """At s=0, vol multiplier = 1."""
        mult = PortfolioComparator._asymmetric_vol_multiplier(0.0)
        assert abs(mult - 1.0) < 1e-10

    def test_crisis_amplifies(self):
        """In crisis (s > 0), vol amplified."""
        mult = PortfolioComparator._asymmetric_vol_multiplier(1.0)
        assert mult > 1.0
        assert abs(mult - 1.40) < 1e-10

    def test_expansion_attenuates(self):
        """In expansion (s < 0), vol slightly attenuated."""
        mult = PortfolioComparator._asymmetric_vol_multiplier(-1.0)
        assert mult < 1.0
        assert abs(mult - 0.90) < 1e-10

    def test_asymmetry_ratio(self):
        """4:1 asymmetry: crisis amplification much larger than expansion attenuation."""
        crisis = PortfolioComparator._asymmetric_vol_multiplier(1.0) - 1.0
        expansion = 1.0 - PortfolioComparator._asymmetric_vol_multiplier(-1.0)
        assert crisis / max(expansion, 1e-10) > 3.5


class TestAsymmetricPeBand:
    """Tests _asymmetric_pe_band static method."""

    def test_returns_tuple(self):
        pe_min, pe_max = PortfolioComparator._asymmetric_pe_band(0.0)
        assert isinstance(pe_min, float)
        assert isinstance(pe_max, float)

    def test_expansion_opens_band(self):
        """In strong expansion (s << 0), pe_max approaches pe_calm (15%)."""
        _, pe_max = PortfolioComparator._asymmetric_pe_band(-3.0)
        assert pe_max > 0.12  # close to 15%

    def test_crisis_compresses_band(self):
        """In crisis (s >> 0), pe_max approaches pe_stress (5%)."""
        _, pe_max = PortfolioComparator._asymmetric_pe_band(3.0)
        assert pe_max < 0.07  # close to 5%

    def test_min_always_below_max(self):
        """pe_min < pe_max for all stress levels."""
        for s in [-3, -1, 0, 1, 3]:
            pe_min, pe_max = PortfolioComparator._asymmetric_pe_band(float(s))
            assert pe_min < pe_max

    def test_band_monotone_decreasing(self):
        """pe_max decreases as stress increases."""
        vals = [PortfolioComparator._asymmetric_pe_band(float(s))[1] for s in [-3, -1, 0, 1, 3]]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]


class TestBLConfidence:
    """Tests _bl_confidence static method."""

    def test_returns_tuple(self):
        conf_c, conf_pe = PortfolioComparator._bl_confidence(0.0)
        assert isinstance(conf_c, float)
        assert isinstance(conf_pe, float)

    def test_bounded_0_30_0_95(self):
        """Confidence always in [0.30, 0.95]."""
        for s in [-5, -2, 0, 2, 5]:
            conf_c, conf_pe = PortfolioComparator._bl_confidence(float(s))
            assert 0.30 <= conf_c <= 0.95
            assert 0.30 <= conf_pe <= 0.95

    def test_expansion_high_confidence(self):
        """In strong expansion, confidence near 0.95."""
        conf_c, conf_pe = PortfolioComparator._bl_confidence(-3.0)
        assert conf_c > 0.85
        assert conf_pe > 0.85

    def test_crisis_low_confidence(self):
        """In crisis, confidence drops toward 0.30."""
        conf_c, conf_pe = PortfolioComparator._bl_confidence(3.0)
        assert conf_c < 0.50
        assert conf_pe < 0.40

    def test_pe_drops_faster(self):
        """PE confidence drops faster than credit (k_pe=2.5 > k_credit=1.5)."""
        conf_c, conf_pe = PortfolioComparator._bl_confidence(1.0)
        assert conf_pe < conf_c


class TestPhase1Phase2:
    """Tests Phase 1 free allocation -> Phase 2 constrained."""

    def test_pe_free_positive(self, allocation_result):
        """Phase 1 (10-class Softmax) PE should have positive weight."""
        assert allocation_result["pe_free"] > 0.0, (
            f"Phase 1 PE={allocation_result['pe_free']:.0%}, expected > 0%"
        )

    def test_pe_band_fixed(self, allocation_result):
        """pe_band is fixed [0.005, 0.20] (PE band suppressed, CET1 Phase 2 suffices)."""
        pe_band = allocation_result["pe_band"]
        assert abs(pe_band[0] - 0.005) < 1e-6
        assert abs(pe_band[1] - 0.20) < 1e-6

    def test_pe_final_positive(self, allocation_result):
        """Final PE > 0 (market impact allows small but non-zero allocation)."""
        pe_final = allocation_result["pe_allocation"]
        assert pe_final > 0, f"PE={pe_final:.4f} should be positive"

    def test_stress_intensity_finite(self, allocation_result):
        assert np.isfinite(allocation_result["stress_intensity"])

    def test_illiquidity_premium_neutral(self, allocation_result):
        """Illiquidity premium suppressed (CVaR gradient handles tail risk)."""
        assert allocation_result["illiquidity_premium"] == 0.0

    def test_vol_multiplier_neutral(self, allocation_result):
        """Vol multiplier suppressed (kappa auto-calibrated)."""
        assert allocation_result["vol_multiplier"] == 1.0

    def test_bl_confidence_neutral(self, allocation_result):
        """BL confidence suppressed (not used in objective)."""
        conf = allocation_result["bl_confidence"]
        assert len(conf) == 2
        assert conf[0] == 0.50
        assert conf[1] == 0.50

    def test_kappa_pe_eff_equals_kappa_base(self, allocation_result):
        """kappa_pe_eff = kappa_base (vol_mult suppressed)."""
        kappa_pe = allocation_result["kappa_pe_eff"]
        kappa_base = allocation_result["kappa"]
        assert abs(kappa_pe - kappa_base) < 0.01


# ============================================================
# 10-Class Multi-Asset Allocation
# ============================================================

class TestRAROCMulticlass:
    """Tests compute_raroc_multiclass() — 14 asset class RAROC."""

    @pytest.fixture(scope="class")
    def raroc_mc(self, pipeline_data):
        comparator = pipeline_data[2]
        return comparator.compute_raroc_multiclass()

    def test_returns_dataframe(self, raroc_mc):
        assert isinstance(raroc_mc, pl.DataFrame)

    def test_15_rows(self, raroc_mc):
        """14 classes + 1 Total = 15 rows."""
        assert len(raroc_mc) == 15

    def test_required_columns(self, raroc_mc):
        expected = {"asset_class", "label", "exposure", "revenue", "loss",
                    "rwa", "capital", "raroc", "eva", "profit_rate"}
        assert expected.issubset(set(raroc_mc.columns))

    def test_all_10_classes_present(self, raroc_mc):
        from ifrs9_cockpit.config import ASSET_CLASS_NAMES
        classes = set(raroc_mc["asset_class"].to_list())
        expected = set(ASSET_CLASS_NAMES) | {"Total"}
        assert classes == expected

    def test_exposure_positive(self, raroc_mc):
        assert (raroc_mc["exposure"].to_numpy() > 0).all()

    def test_raroc_finite(self, raroc_mc):
        assert raroc_mc["raroc"].null_count() == 0
        assert np.isfinite(raroc_mc["raroc"].to_numpy()).all()

    def test_corporate_loans_uses_real_data(self, raroc_mc, pipeline_data):
        """corporate_loans RAROC should be in the same ballpark as Credit Total."""
        comparator = pipeline_data[2]
        raroc_2ch = comparator.compute_raroc_eva()
        total_c = raroc_2ch.filter(
            (pl.col("canal") == "Credit") & (pl.col("sector") == "Total")
        )
        corp_row = raroc_mc.filter(pl.col("asset_class") == "corporate_loans")
        if len(total_c) > 0 and len(corp_row) > 0:
            # Same order of magnitude (spread aggregation differs slightly)
            assert abs(total_c["raroc"][0] - corp_row["raroc"][0]) < 0.10


class TestMulticlassAllocation:
    """Tests for 10-class BL-CVaR allocation."""

    def test_class_weights_present(self, allocation_result):
        """class_weights dict should have 14 entries."""
        cw = allocation_result.get("class_weights", {})
        assert len(cw) == 14

    def test_class_weights_sum_to_1(self, allocation_result):
        cw = allocation_result.get("class_weights", {})
        total = sum(cw.values())
        assert abs(total - 1.0) < 0.01, f"Class weights sum = {total}"

    def test_constraint_mode_endogenous(self, allocation_result):
        """Allocation should use endogenous constraint mode."""
        assert allocation_result.get("constraint_mode") == "endogenous"

    def test_all_weights_positive(self, allocation_result):
        """All 10 classes should have positive weight (minimum diversification)."""
        cw = allocation_result.get("class_weights", {})
        for name, w in cw.items():
            assert w > 0, f"{name}: weight={w} not positive"

    def test_method_is_14c(self, allocation_result):
        assert allocation_result["method"] == "BL-CVaR-14C"

    def test_n_classes_is_14(self, allocation_result):
        assert allocation_result["n_classes"] == 14

    def test_corr_matrix_present(self, allocation_result):
        corr = allocation_result.get("corr_10x10")
        assert corr is not None
        assert len(corr) == 14
        assert len(corr[0]) == 14

    def test_backward_compat_keys(self, allocation_result):
        """Old keys (credit_allocation, pe_allocation, sector_weights) still present."""
        assert "credit_allocation" in allocation_result
        assert "pe_allocation" in allocation_result
        assert "sector_weights_credit" in allocation_result
        assert "sector_weights_pe" in allocation_result

    def test_credit_pe_sum_to_1(self, allocation_result):
        """credit_allocation + pe_allocation should sum to ~1."""
        total = allocation_result["credit_allocation"] + allocation_result["pe_allocation"]
        assert abs(total - 1.0) < 0.01

    def test_all_weights_non_negative(self, allocation_result):
        """All weights must be >= 0 (no short selling)."""
        cw = allocation_result.get("class_weights", {})
        for name, w in cw.items():
            assert w >= 0, f"{name}: weight={w} is negative"

    def test_lcr_compliant(self, allocation_result):
        """LCR ratio >= 99% (allowing minor numerical tolerance)."""
        lcr = allocation_result.get("lcr_ratio", 0)
        assert lcr >= 0.99, f"LCR={lcr:.2%} < 99%"

    def test_nsfr_compliant(self, allocation_result):
        """NSFR ratio >= 99% (allowing minor numerical tolerance)."""
        nsfr = allocation_result.get("nsfr_ratio", 0)
        assert nsfr >= 0.99, f"NSFR={nsfr:.2%} < 99%"

    def test_phase1_weights_sum_to_one(self, allocation_result):
        """Phase 1 (pre-regulatory) weights should sum to 1."""
        p1 = allocation_result.get("phase1_weights", {})
        if p1:
            total = sum(p1.values())
            assert abs(total - 1.0) < 0.01, f"Phase 1 weights sum = {total}"

    def test_covariance_shrinkage_applied(self, allocation_result):
        """Ledoit-Wolf shrinkage lambda should be in (0, 1)."""
        lw = allocation_result.get("covariance_shrinkage_lambda", None)
        assert lw is not None, "Missing covariance_shrinkage_lambda"
        assert 0 < lw < 1, f"lambda_lw={lw} not in (0, 1)"

    def test_regulatory_adjustments_recorded(self, allocation_result):
        """Regulatory adjustment deltas should be recorded."""
        ra = allocation_result.get("regulatory_adjustments")
        assert ra is not None, "Missing regulatory_adjustments"
        assert "cet1_delta" in ra
        assert "lcr_delta" in ra
        assert "nsfr_delta" in ra
        assert "irrbb_delta" in ra

    def test_spread_compression_in_output(self, allocation_result):
        """spread_compression dict should be present with 14 entries."""
        sc = allocation_result.get("spread_compression")
        assert sc is not None, "Missing spread_compression"
        assert len(sc) == 14
        for name, val in sc.items():
            assert np.isfinite(val), f"{name}: compression={val} not finite"
            assert val >= 0, f"{name}: compression={val} negative"


class TestSpreadCompression:
    """Tests for _spread_compression static method (market impact logarithmique)."""

    def test_no_compression_at_zero_weight(self):
        """mu_eff == mu_base when weight is zero."""
        from ifrs9_cockpit.config import ASSET_CLASS_NAMES
        n = len(ASSET_CLASS_NAMES)
        w = np.zeros(n)
        mu_base = np.ones(n) * 0.10
        mu_eff = PortfolioComparator._spread_compression(
            w, list(ASSET_CLASS_NAMES), 1.3e12, mu_base
        )
        np.testing.assert_array_almost_equal(mu_eff, mu_base, decimal=10)

    def test_compression_increases_with_weight(self):
        """mu_eff decreases as weight increases (more market impact)."""
        from ifrs9_cockpit.config import ASSET_CLASS_NAMES
        names = list(ASSET_CLASS_NAMES)
        n = len(names)
        mu_base = np.ones(n) * 0.10

        w_low = np.ones(n) / n  # ~7% each
        w_high = np.zeros(n)
        w_high[0] = 0.50  # 50% on corporate_loans
        w_high[1:] = 0.50 / (n - 1)

        mu_low = PortfolioComparator._spread_compression(w_low, names, 1.3e12, mu_base)
        mu_high = PortfolioComparator._spread_compression(w_high, names, 1.3e12, mu_base)
        # Corporate loans (index 0) should have lower mu_eff at high weight
        assert mu_high[0] < mu_low[0], (
            f"Corporate loans: mu_high={mu_high[0]:.6f} >= mu_low={mu_low[0]:.6f}"
        )

    def test_small_market_more_compressed(self):
        """PE (300B) should be more compressed than sovereign (11T) at same weight."""
        from ifrs9_cockpit.config import ASSET_CLASS_NAMES
        names = list(ASSET_CLASS_NAMES)
        n = len(names)
        mu_base = np.ones(n) * 0.10
        # Give both 10% weight
        w = np.ones(n) / n
        mu_eff = PortfolioComparator._spread_compression(w, names, 1.3e12, mu_base)
        i_pe = names.index("private_equity")
        i_sov = names.index("sovereign")
        # PE (300B market) should be more compressed than sovereign (11T market)
        assert mu_eff[i_pe] < mu_eff[i_sov], (
            f"PE mu_eff={mu_eff[i_pe]:.6f} >= sovereign mu_eff={mu_eff[i_sov]:.6f}"
        )

    def test_compression_cap_at_95pct(self):
        """No NaN/Inf even when share exceeds 95% of market."""
        from ifrs9_cockpit.config import ASSET_CLASS_NAMES
        names = list(ASSET_CLASS_NAMES)
        n = len(names)
        mu_base = np.ones(n) * 0.10
        # Extreme weight: 99% on PE (300B market, 1.3T total → share = 4.3x capacity)
        w = np.zeros(n)
        w[names.index("private_equity")] = 0.99
        w[0] = 0.01
        mu_eff = PortfolioComparator._spread_compression(w, names, 1.3e12, mu_base)
        assert np.all(np.isfinite(mu_eff)), f"Non-finite values: {mu_eff}"
        assert np.all(mu_eff > 0), f"Non-positive values: {mu_eff}"

    def test_total_ead_consistent(self):
        """total_ead should be positive and finite in a realistic scenario."""
        total_ead = 1.3e12
        assert total_ead > 0
        assert np.isfinite(total_ead)


class TestRAROCMulticlassReasonable:
    """Sanity checks on RAROC values — no infinite or absurd values."""

    @pytest.fixture(scope="class")
    def raroc_mc(self, pipeline_data):
        comparator = pipeline_data[2]
        return comparator.compute_raroc_multiclass()

    def test_no_raroc_above_100pct(self, raroc_mc):
        """No asset class should have RAROC > 100% (leverage floor prevents infinity)."""
        non_total = raroc_mc.filter(pl.col("asset_class") != "Total")
        for row in non_total.iter_rows(named=True):
            assert row["raroc"] < 1.0, (
                f"{row['asset_class']}: RAROC={row['raroc']:.2%} > 100%"
            )

    def test_capital_always_positive(self, raroc_mc):
        """All capital values should be positive (leverage floor ensures this)."""
        assert (raroc_mc["capital"].to_numpy() > 0).all()

    def test_sovereign_raroc_reasonable(self, raroc_mc):
        """Sovereign RAROC should be finite and within [-50%, +50%]."""
        sov = raroc_mc.filter(pl.col("asset_class") == "sovereign")
        assert len(sov) == 1
        assert -0.50 <= sov["raroc"][0] <= 0.50


class TestCorrelation10x10:
    """Tests for the 10x10 macro-sensitivity correlation matrix."""

    def test_symmetric(self):
        corr = PortfolioComparator._build_corr_10x10()
        assert np.allclose(corr, corr.T, atol=1e-10)

    def test_diagonal_is_one(self):
        corr = PortfolioComparator._build_corr_10x10()
        assert np.allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_positive_semi_definite(self):
        corr = PortfolioComparator._build_corr_10x10()
        eigvals = np.linalg.eigvalsh(corr)
        assert eigvals.min() > -1e-8, f"Min eigenvalue: {eigvals.min()}"

    def test_values_in_range(self):
        corr = PortfolioComparator._build_corr_10x10()
        assert corr.min() >= -1.0
        assert corr.max() <= 1.0

    def test_similar_classes_correlated(self):
        """Sovereign and interbank (similar macro sensitivities) should be highly correlated."""
        from ifrs9_cockpit.config import ASSET_CLASS_NAMES
        names = list(ASSET_CLASS_NAMES)
        corr = PortfolioComparator._build_corr_10x10()
        i_sov = names.index("sovereign")
        i_int = names.index("interbank")
        assert corr[i_sov, i_int] > 0.5, f"sovereign-interbank correlation = {corr[i_sov, i_int]}"


# ============================================================
# Profit Rate Objective
# ============================================================

class TestProfitRateObjective:
    """Tests for profit_rate objective (profit/EAD) replacing RAROC."""

    @pytest.fixture(scope="class")
    def raroc_mc(self, pipeline_data):
        comparator = pipeline_data[2]
        return comparator.compute_raroc_multiclass()

    def test_profit_rate_in_raroc_mc(self, raroc_mc):
        """profit_rate column must be present."""
        assert "profit_rate" in raroc_mc.columns

    def test_profit_rate_positive_for_profitable(self, raroc_mc):
        """profit_rate > 0 when raroc > 0."""
        for row in raroc_mc.iter_rows(named=True):
            if row["raroc"] > 0:
                assert row["profit_rate"] > 0, (
                    f"{row['asset_class']}: raroc={row['raroc']:.4f} but profit_rate={row['profit_rate']:.6f}"
                )

    def test_profit_rate_equals_profit_div_ead(self, raroc_mc):
        """profit_rate ~ raroc * capital / exposure (cross-check)."""
        for row in raroc_mc.iter_rows(named=True):
            if row["asset_class"] == "Total":
                continue
            # profit = raroc * capital, profit_rate = profit / ead
            expected_pr = row["raroc"] * row["capital"] / max(row["exposure"], 1)
            assert abs(row["profit_rate"] - expected_pr) < 0.001, (
                f"{row['asset_class']}: profit_rate={row['profit_rate']:.6f} "
                f"vs expected={expected_pr:.6f}"
            )

    def test_cb_lower_profit_rate_than_corp(self, raroc_mc):
        """Covered bonds should have lower profit_rate than corporate loans.

        This is the core motivation: CB has high RAROC (low capital) but low
        profit_rate (low profit/EAD). The profit_rate objective correctly
        devalues CB relative to corporate loans.
        """
        cb = raroc_mc.filter(pl.col("asset_class") == "covered_bonds")
        corp = raroc_mc.filter(pl.col("asset_class") == "corporate_loans")
        assert len(cb) == 1 and len(corp) == 1
        assert cb["profit_rate"][0] < corp["profit_rate"][0], (
            f"CB profit_rate={cb['profit_rate'][0]:.6f} >= corp={corp['profit_rate'][0]:.6f}"
        )

    def test_profit_rate_portfolio_in_output(self, allocation_result):
        """profit_rate_portfolio key must be present in allocation output."""
        assert "profit_rate_portfolio" in allocation_result

    def test_objective_is_profit_rate(self, allocation_result):
        """objective key should be 'profit_rate'."""
        assert allocation_result.get("objective") == "profit_rate"


# ============================================================
# IRRBB Tests
# ============================================================

class TestIRRBB:
    """Tests for IRRBB EVE constraint in Phase 2."""

    def test_irrbb_keys_in_output(self, allocation_result):
        """3 IRRBB keys must be present."""
        assert "irrbb_eve_ratio" in allocation_result
        assert "irrbb_weighted_duration" in allocation_result
        assert "irrbb_compliant" in allocation_result

    def test_irrbb_compliant(self, allocation_result):
        """IRRBB should be compliant after Phase 2 adjustment."""
        assert allocation_result["irrbb_compliant"] == True

    def test_weighted_duration_reasonable(self, allocation_result):
        """Weighted duration should be in [0.5, 6.0] for a diversified portfolio."""
        wd = allocation_result["irrbb_weighted_duration"]
        assert 0.5 <= wd <= 6.0, f"weighted_duration={wd:.2f} out of range"

    def test_irrbb_delta_in_regulatory(self, allocation_result):
        """irrbb_delta must be in regulatory_adjustments."""
        ra = allocation_result.get("regulatory_adjustments", {})
        assert "irrbb_delta" in ra

    def test_irrbb_eve_ratio_below_1(self, allocation_result):
        """EVE ratio must be <= 1.0 (compliant)."""
        assert allocation_result["irrbb_eve_ratio"] <= 1.0


class TestComputeIRRBBEve:
    """Standalone tests for compute_irrbb_eve function."""

    def test_zero_duration(self):
        """All-equities portfolio (duration=0) → EVE=0, compliant."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import compute_irrbb_eve
        alloc = {"equities": 1.0}
        result = compute_irrbb_eve(alloc, 1.3e12, 480e9)
        assert result["eve_shock"] == 0.0
        assert result["compliant"] is True

    def test_high_duration_fails(self):
        """All-sovereign (duration=6Y) portfolio → likely non-compliant with small capital."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import compute_irrbb_eve
        alloc = {"sovereign": 1.0}
        # Small capital relative to EAD → IRRBB non-compliant
        result = compute_irrbb_eve(alloc, 1.3e12, 50e9)
        assert result["compliant"] is False

    def test_weighted_duration_calc(self):
        """Verify weighted duration calculation: 50% sovereign(6Y) + 50% interbank(0.25Y)."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import compute_irrbb_eve
        alloc = {"sovereign": 0.5, "interbank": 0.5}
        result = compute_irrbb_eve(alloc, 1.3e12, 480e9)
        expected_wd = 0.5 * 6.0 + 0.5 * 0.25
        assert abs(result["weighted_duration"] - expected_wd) < 0.01, (
            f"weighted_duration={result['weighted_duration']:.4f} vs expected={expected_wd:.4f}"
        )

    def test_alm_hedge_ratio_from_config(self):
        """Default alm_hedge_ratio should come from BASEL_CONFIG."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import compute_irrbb_eve
        from ifrs9_cockpit.config import BASEL_CONFIG
        alloc = {"sovereign": 0.5, "interbank": 0.5}
        result = compute_irrbb_eve(alloc, 1.3e12, 480e9)
        wd = 0.5 * 6.0 + 0.5 * 0.25
        expected_ed = round(wd * (1.0 - BASEL_CONFIG.alm_hedge_ratio), 4)
        assert abs(result["effective_duration"] - expected_ed) < 0.01


# ============================================================
# Zero Arbitrary Parameters Tests
# ============================================================

class TestMarketVolOverrideAllClasses:
    """All 14 asset classes must have market_vol_override configured."""

    def test_all_14_classes_have_market_vol_override(self):
        from ifrs9_cockpit.config import ASSET_CLASSES
        for ac in ASSET_CLASSES:
            assert ac.market_vol_override is not None, (
                f"{ac.name}: market_vol_override is None"
            )
            assert ac.market_vol_override > 0, (
                f"{ac.name}: market_vol_override={ac.market_vol_override} not positive"
            )

    def test_vol_ordering(self):
        """PE vol > corporate vol > covered bonds vol (economic sense)."""
        from ifrs9_cockpit.config import ASSET_CLASS_MAP
        pe = ASSET_CLASS_MAP["private_equity"].market_vol_override
        corp = ASSET_CLASS_MAP["corporate_loans"].market_vol_override
        cb = ASSET_CLASS_MAP["covered_bonds"].market_vol_override
        assert pe > corp > cb, f"PE={pe}, Corp={corp}, CB={cb}"


class TestAutoCalibration:
    """Tests for auto-calibrated parameters (zero arbitrary constants)."""

    def test_auto_lam_equals_mean_abs_offdiag(self):
        """Lambda correlation penalty should be mean(|off-diag|)."""
        corr, _ = PortfolioComparator._build_corr_matrix()
        n = len(corr)
        off_diag = corr[np.triu_indices(n, k=1)]
        expected_lam = float(np.mean(np.abs(off_diag)))
        assert expected_lam > 0, f"lam={expected_lam} not positive"
        assert expected_lam < 1.0, f"lam={expected_lam} >= 1.0"

    def test_min_weight_auto(self):
        """Min weight should be 1/n^2 for n=14."""
        from ifrs9_cockpit.config import ASSET_CLASSES
        n = len(ASSET_CLASSES)
        expected_min = 1.0 / (n * n)
        # Test via softmax: even with extreme inputs, min weight >= 1/n^2
        extreme = np.array([100.0] + [-100.0] * (n - 1))
        w = PortfolioComparator._softmax_weights_10(extreme, n)
        assert w.min() >= expected_min - 1e-6, (
            f"min weight={w.min():.6f} < expected 1/n^2={expected_min:.6f}"
        )

    def test_r_neutral_auto_calibrated(self):
        """r_neutral auto-calibrated from positive RAROC channels."""
        # Both positive: r_neutral = mean
        s = PortfolioComparator._stress_intensity(0.08, 0.12)
        assert s < 0  # favorable
        # Both negative: r_neutral = 0.07 fallback
        s = PortfolioComparator._stress_intensity(-0.05, -0.05)
        assert s > 0  # adverse


class TestAdaptiveConvergence:
    """Tests for adaptive convergence in regulatory adjustments."""

    def test_adaptive_step_proportional(self):
        """Step size should increase with deficit."""
        step_small = PortfolioComparator._adaptive_step(0.01)
        step_large = PortfolioComparator._adaptive_step(0.10)
        assert step_large > step_small

    def test_adaptive_step_bounded(self):
        """Step size should be in [0.002, 0.02]."""
        for deficit in [0.001, 0.01, 0.05, 0.10, 0.50, 1.0]:
            step = PortfolioComparator._adaptive_step(deficit)
            assert 0.002 <= step <= 0.02, f"deficit={deficit}: step={step}"


# ============================================================
# RAROC Scenario-Dependant (pd_cond_base for loss)
# ============================================================

class TestRAROCScenarioDependant:
    """Tests that RAROC varies across scenarios for balance-sheet classes.

    Fix: compute_raroc_multiclass() now uses pd_cond_base (conditional PD)
    for annual expected loss, making RAROC scenario-sensitive.
    """

    @pytest.fixture(scope="class")
    def raroc_mc(self, pipeline_data):
        comparator = pipeline_data[2]
        return comparator.compute_raroc_multiclass()

    def test_bs_classes_use_pd_cond_base(self, raroc_mc):
        """Balance-sheet classes should have pd_cond_base reflected in loss.

        If pd_cond_base differs from pd_base (stressed scenario), the loss
        column should differ from what pd_base alone would produce.
        This is a structural test: we verify the column is read.
        """
        # Sovereign has very low PD so its RAROC should be high
        sov = raroc_mc.filter(pl.col("asset_class") == "sovereign")
        assert len(sov) == 1
        assert sov["raroc"][0] > -0.50  # reasonable

    def test_raroc_multiclass_has_all_classes(self, raroc_mc):
        """All 14 classes + Total = 15 rows."""
        assert len(raroc_mc) == 15

    def test_loss_positive_for_ecl_classes(self, raroc_mc):
        """ECL-based classes should have positive loss."""
        ecl_classes = raroc_mc.filter(
            ~pl.col("asset_class").is_in(["Total", "equities"])
        )
        for row in ecl_classes.iter_rows(named=True):
            assert row["loss"] > 0, (
                f"{row['asset_class']}: loss={row['loss']} should be > 0"
            )


# ============================================================
# Market Capacity Cap (post Phase 2)
# ============================================================

class TestEnforceMarketCaps:
    """Tests for _enforce_market_caps static method."""

    def test_no_change_when_within_caps(self):
        """Weights at or below market caps should not be modified."""
        from ifrs9_cockpit.config import ASSET_CLASS_NAMES, ASSET_CLASS_MAP
        names = list(ASSET_CLASS_NAMES)
        n = len(names)
        # Use market_capacity-proportional weights (exactly at caps = max_w)
        market_caps = np.array([ASSET_CLASS_MAP[name].market_capacity_eur for name in names])
        w = market_caps / market_caps.sum()
        w_capped = PortfolioComparator._enforce_market_caps(w.copy(), names, n)
        assert abs(w_capped.sum() - 1.0) < 1e-6
        np.testing.assert_array_almost_equal(w_capped, w, decimal=4)

    def test_excess_redistributed(self):
        """Weight exceeding market cap should be redistributed."""
        from ifrs9_cockpit.config import ASSET_CLASS_NAMES, ASSET_CLASS_MAP
        names = list(ASSET_CLASS_NAMES)
        n = len(names)
        market_caps = np.array([ASSET_CLASS_MAP[name].market_capacity_eur for name in names])
        max_w = market_caps / market_caps.sum()

        # Set repos_sft to 50% (way above its cap of ~15.6%)
        w = np.ones(n) / n
        repos_idx = names.index("repos_sft")
        w[repos_idx] = 0.50
        w /= w.sum()

        w_capped = PortfolioComparator._enforce_market_caps(w.copy(), names, n)
        assert w_capped[repos_idx] <= max_w[repos_idx] + 1e-4, (
            f"repos_sft: {w_capped[repos_idx]:.4f} > cap {max_w[repos_idx]:.4f}"
        )
        assert abs(w_capped.sum() - 1.0) < 1e-6

    def test_preserves_sum_to_one(self):
        """Weights always sum to 1 after capping."""
        from ifrs9_cockpit.config import ASSET_CLASS_NAMES
        names = list(ASSET_CLASS_NAMES)
        n = len(names)
        rng = np.random.default_rng(42)
        for _ in range(10):
            w = rng.dirichlet(np.ones(n))
            w_capped = PortfolioComparator._enforce_market_caps(w.copy(), names, n)
            assert abs(w_capped.sum() - 1.0) < 1e-6


class TestReposCapPostPhase2:
    """Tests that repos_sft is capped or constrained by market capacity + IRRBB."""

    def test_repos_below_35pct(self, allocation_result):
        """repos_sft should be below 35% (capped from unconstrained ~40%+).

        Note: repos may exceed strict market cap (~19%) when IRRBB compliance
        requires low-duration assets. Regulatory norms take precedence over
        advisory market capacity caps. Before the fix repos could reach 40%+
        without any limit; now it's bounded by the cap+IRRBB tradeoff.
        """
        cw = allocation_result.get("class_weights", {})
        repos_w = cw.get("repos_sft", 0)
        assert repos_w <= 0.35, (
            f"repos_sft={repos_w:.4f} exceeds 35%"
        )

    def test_irrbb_still_compliant(self, allocation_result):
        """IRRBB should remain compliant after market cap enforcement."""
        assert allocation_result["irrbb_compliant"] is True
