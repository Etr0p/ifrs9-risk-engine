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
import pandas as pd
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


# ============================================================
# Fixture : pipeline complet (scope=module pour ne calculer qu'une fois)
# ============================================================

@pytest.fixture(scope="module")
def pipeline_data():
    """Execute le pipeline complet et retourne (result_credit, result_pe, comparator)."""
    from ifrs9_cockpit.data.generator import generate_dataset
    from ifrs9_cockpit.models.pd_model import PDModelSuite
    from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
    from ifrs9_cockpit.engine.pe_calculator import PECalculator
    from ifrs9_cockpit.models.lgd_model import LGDModel
    from ifrs9_cockpit.models.ead_model import EADModel

    df_credit, df_pe, df_history = generate_dataset()

    pd_suite = PDModelSuite()
    pd_suite.fit(df_credit)
    pd_current = pd_suite.predict_active(df_credit)

    lgd_model = LGDModel()
    ead_model = EADModel()
    ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
    pd_origination = pd_current * 0.8
    result_credit = ecl_calc.calculate(df_credit, pd_current, pd_origination)

    pe_calc = PECalculator()
    result_pe = pe_calc.calculate(df_pe)

    comparator = PortfolioComparator(result_credit, result_pe)

    return result_credit, result_pe, comparator


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
        assert isinstance(advanced_metrics, pd.DataFrame)

    def test_5_sectors_plus_total(self, advanced_metrics):
        """5 secteurs + 1 ligne Total = 6 lignes."""
        assert len(advanced_metrics) == 6

    def test_all_sectors_present(self, advanced_metrics):
        sectors = set(advanced_metrics["sector"].values)
        expected = set(SECTOR_NAMES) | {"Total"}
        assert sectors == expected

    def test_npl_ratio_in_0_1(self, advanced_metrics):
        """NPL ratio = EAD(Stage 3) / EAD total, dans [0, 1]."""
        for val in advanced_metrics["npl_ratio"]:
            assert 0 <= val <= 1, f"NPL ratio hors bornes: {val}"

    def test_cost_of_risk_positive(self, advanced_metrics):
        """Cost of risk en bps doit etre > 0 (portefeuille a defauts)."""
        total = advanced_metrics.loc[advanced_metrics["sector"] == "Total"]
        assert total["cost_of_risk_bps"].values[0] > 0

    def test_coverage_ratio_in_0_1(self, advanced_metrics):
        """Coverage ratio = ECL_S3 / EAD_S3, dans [0, 1]."""
        for val in advanced_metrics["coverage_ratio"]:
            assert 0 <= val <= 1, f"Coverage ratio hors bornes: {val}"

    def test_texas_ratio_positive(self, advanced_metrics):
        """Texas ratio = EAD_S3 / (ECL_S3 + capital) doit etre >= 0."""
        for val in advanced_metrics["texas_ratio"]:
            assert val >= 0, f"Texas ratio negatif: {val}"

    def test_pd_mean_in_0_1(self, advanced_metrics):
        """PD moyenne doit etre dans [0, 1]."""
        for val in advanced_metrics["pd_mean"]:
            assert 0 <= val <= 1, f"PD mean hors bornes: {val}"

    def test_stage2_pct_in_0_1(self, advanced_metrics):
        for val in advanced_metrics["stage2_pct"]:
            assert 0 <= val <= 1, f"Stage2 pct hors bornes: {val}"

    def test_rwa_density_positive(self, advanced_metrics):
        for val in advanced_metrics["rwa_density"]:
            assert val > 0, f"RWA density non positive: {val}"

    def test_canal_column_is_credit(self, advanced_metrics):
        """Toutes les lignes sont canal Credit."""
        assert (advanced_metrics["canal"] == "Credit").all()

    def test_ead_total_positive(self, advanced_metrics):
        total = advanced_metrics.loc[advanced_metrics["sector"] == "Total"]
        assert total["ead_total"].values[0] > 0


class TestHHICrossCell:
    """Tests FR41 — HHI concentration cross-cell."""

    def test_returns_dict(self, hhi_result):
        assert isinstance(hhi_result, dict)

    def test_required_keys(self, hhi_result):
        expected_keys = {"hhi_credit", "hhi_pe", "hhi_crosscell", "shares"}
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
        assert (hhi_result["shares"]["share"] >= 0).all()

    def test_hhi_credit_above_equal_parts(self, hhi_result):
        """Avec 5 secteurs non-equiponderes, HHI > 2000 (1/5 = 0.20 => HHI=2000 si egal)."""
        # Le HHI 5 secteurs egaux serait exactement 2000
        # Avec des poids inegaux, il sera > 2000
        assert hhi_result["hhi_credit"] >= 2000

    def test_crosscell_dominated_by_credit(self, hhi_result):
        """Le portefeuille credit domine largement le PE en exposition."""
        shares = hhi_result["shares"]
        credit_share = shares.loc[shares["canal"] == "Credit", "share"].sum()
        pe_share = shares.loc[shares["canal"] == "PE", "share"].sum()
        assert credit_share > pe_share


class TestRAROCEVA:
    """Tests FR44 — RAROC et EVA par cellule."""

    def test_returns_dataframe(self, raroc_eva):
        assert isinstance(raroc_eva, pd.DataFrame)

    def test_12_rows(self, raroc_eva):
        """10 cellules (5 secteurs x 2 canaux) + 2 totaux = 12."""
        assert len(raroc_eva) == 12

    def test_required_columns(self, raroc_eva):
        expected = {"sector", "canal", "exposure", "revenue", "loss", "rwa", "capital", "raroc", "eva"}
        assert expected.issubset(set(raroc_eva.columns))

    def test_exposure_positive(self, raroc_eva):
        assert (raroc_eva["exposure"] > 0).all()

    def test_rwa_positive(self, raroc_eva):
        assert (raroc_eva["rwa"] > 0).all()

    def test_capital_positive(self, raroc_eva):
        assert (raroc_eva["capital"] > 0).all()

    def test_raroc_finite(self, raroc_eva):
        assert raroc_eva["raroc"].notna().all()
        assert np.isfinite(raroc_eva["raroc"].values).all()

    def test_eva_finite(self, raroc_eva):
        assert raroc_eva["eva"].notna().all()
        assert np.isfinite(raroc_eva["eva"].values).all()

    def test_total_credit_present(self, raroc_eva):
        total_c = raroc_eva.loc[
            (raroc_eva["sector"] == "Total") & (raroc_eva["canal"] == "Credit")
        ]
        assert len(total_c) == 1

    def test_total_pe_present(self, raroc_eva):
        total_p = raroc_eva.loc[
            (raroc_eva["sector"] == "Total") & (raroc_eva["canal"] == "PE")
        ]
        assert len(total_p) == 1

    def test_both_canals_present(self, raroc_eva):
        canals = set(raroc_eva["canal"].values)
        assert canals == {"Credit", "PE"}


# ============================================================
# Story 4-2 : Asymetrie credit/PE, resilience, RAROC
# ============================================================

class TestResilienceScores:
    """Tests FR19 — Scores de resilience credit et PE."""

    def test_credit_resilience_5_sectors(self, comparator):
        resil = comparator._resilience_score_credit()
        assert len(resil) == 5

    def test_credit_resilience_in_0_1(self, comparator):
        resil = comparator._resilience_score_credit()
        for val in resil["resilience_credit"]:
            assert 0 <= val <= 1, f"Resilience credit hors [0,1]: {val}"

    def test_pe_resilience_5_sectors(self, comparator):
        resil = comparator._resilience_score_pe()
        assert len(resil) == 5

    def test_pe_resilience_in_0_1(self, comparator):
        resil = comparator._resilience_score_pe()
        for val in resil["resilience_pe"]:
            assert 0 <= val <= 1, f"Resilience PE hors [0,1]: {val}"

    def test_resilience_sectors_match(self, comparator):
        resil_c = comparator._resilience_score_credit()
        resil_p = comparator._resilience_score_pe()
        assert set(resil_c["sector"]) == set(resil_p["sector"])
        assert set(resil_c["sector"]) == set(SECTOR_NAMES)


class TestAsymmetryMatrix:
    """Tests FR20 — Matrice d'asymetrie credit vs PE."""

    def test_returns_dataframe(self, asymmetry_matrix):
        assert isinstance(asymmetry_matrix, pd.DataFrame)

    def test_5_sectors(self, asymmetry_matrix):
        assert len(asymmetry_matrix) == 5

    def test_all_sectors_present(self, asymmetry_matrix):
        assert set(asymmetry_matrix["sector"]) == set(SECTOR_NAMES)

    def test_loss_ratio_non_negative(self, asymmetry_matrix):
        """loss_ratio = EL_PE / ECL_credit >= 0."""
        assert (asymmetry_matrix["loss_ratio"] >= 0).all()

    def test_raroc_delta_is_pe_minus_credit(self, asymmetry_matrix):
        """raroc_delta = raroc_pe - raroc_credit."""
        for _, row in asymmetry_matrix.iterrows():
            expected = round(row["raroc_pe"] - row["raroc_credit"], 4)
            assert abs(row["raroc_delta"] - expected) < 1e-3, (
                f"Secteur {row['sector']}: delta={row['raroc_delta']}, "
                f"attendu={expected}"
            )

    def test_resilience_scores_included(self, asymmetry_matrix):
        assert "resilience_credit" in asymmetry_matrix.columns
        assert "resilience_pe" in asymmetry_matrix.columns

    def test_resilience_in_0_1(self, asymmetry_matrix):
        for val in asymmetry_matrix["resilience_credit"]:
            assert 0 <= val <= 1
        for val in asymmetry_matrix["resilience_pe"]:
            assert 0 <= val <= 1

    def test_ecl_credit_positive(self, asymmetry_matrix):
        assert (asymmetry_matrix["ecl_credit"] > 0).all()

    def test_required_columns(self, asymmetry_matrix):
        expected = {
            "sector", "ecl_credit", "el_pe", "loss_ratio",
            "raroc_credit", "raroc_pe", "raroc_delta",
            "resilience_credit", "resilience_pe",
        }
        assert expected.issubset(set(asymmetry_matrix.columns))


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
        df = pd.DataFrame({
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
        df = pd.DataFrame({
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
        }
        assert expected == set(allocation_result.keys())

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

    def test_equal_raroc_gives_equal_weights(self, comparator):
        """Si tous les RAROC sont egaux, les poids doivent etre egaux."""
        cells = pd.DataFrame({
            "sector": list(SECTOR_NAMES),
            "raroc": [0.10] * 5,
        })
        w = comparator._optimize_sector_weights(cells)
        vals = list(w.values())
        assert all(abs(v - 0.20) < 0.01 for v in vals), f"Poids non egaux: {vals}"

    def test_best_raroc_gets_highest_weight(self, comparator):
        """Le secteur avec le meilleur RAROC devrait avoir le poids le plus eleve."""
        cells = pd.DataFrame({
            "sector": list(SECTOR_NAMES),
            "raroc": [0.05, 0.10, 0.30, 0.08, 0.06],
        })
        w = comparator._optimize_sector_weights(cells)
        # Sante (index 2, RAROC=0.30) devrait etre le plus haut
        assert w["Sante"] == max(w.values())


class TestCRR3Sensitivity:
    """Tests FR23 — Sensibilite CRR3."""

    def test_returns_dataframe(self, crr3_sensitivity):
        assert isinstance(crr3_sensitivity, pd.DataFrame)

    def test_4_rows(self, crr3_sensitivity):
        """3 RW fixes (190, 250, 400) + 1 composite CRR3 = 4 lignes."""
        assert len(crr3_sensitivity) == 4

    def test_required_columns(self, crr3_sensitivity):
        expected = {"rw_pe", "rwa_pe", "rwa_credit", "rwa_total", "cet1_ratio", "headroom", "feasible"}
        assert expected == set(crr3_sensitivity.columns)

    def test_rw_pe_values(self, crr3_sensitivity):
        """Les 3 premieres lignes ont RW 190, 250, 400."""
        rw_values = crr3_sensitivity["rw_pe"].values[:3]
        assert set(rw_values) == {190, 250, 400}

    def test_rwa_increases_with_rw(self, crr3_sensitivity):
        """RWA PE augmente avec le Risk Weight (lignes fixes uniquement)."""
        fixed = crr3_sensitivity.iloc[:3].sort_values("rw_pe")
        rwa_vals = fixed["rwa_pe"].values
        assert rwa_vals[0] <= rwa_vals[1] <= rwa_vals[2]

    def test_cet1_ratio_positive(self, crr3_sensitivity):
        assert (crr3_sensitivity["cet1_ratio"] > 0).all()

    def test_headroom_coherent_with_cet1(self, crr3_sensitivity):
        """headroom = cet1_ratio - cet1_target."""
        for _, row in crr3_sensitivity.iterrows():
            expected = round(row["cet1_ratio"] - BASEL_CONFIG.cet1_target, 4)
            assert abs(row["headroom"] - expected) < 1e-3

    def test_feasible_flag_coherent(self, crr3_sensitivity):
        """feasible = headroom >= 0."""
        for _, row in crr3_sensitivity.iterrows():
            assert row["feasible"] == (row["headroom"] >= 0)

    def test_composite_rw_is_valid(self, crr3_sensitivity):
        """La 4e ligne (composite) a un RW moyen parmi les valeurs CRR3."""
        composite_rw = crr3_sensitivity.iloc[3]["rw_pe"]
        # Le RW moyen composite doit etre entre 190 et 400
        assert 190 <= composite_rw <= 400


class TestTippingPoints:
    """Tests FR24 — Seuils de basculement."""

    def test_returns_dataframe(self, tipping_points):
        assert isinstance(tipping_points, pd.DataFrame)

    def test_5_sectors(self, tipping_points):
        assert len(tipping_points) == 5

    def test_all_sectors_present(self, tipping_points):
        assert set(tipping_points["sector"]) == set(SECTOR_NAMES)

    def test_required_columns(self, tipping_points):
        expected = {"sector", "raroc_credit", "raroc_pe", "preferred_canal", "delta_to_switch"}
        assert expected == set(tipping_points.columns)

    def test_preferred_canal_valid(self, tipping_points):
        """Canal prefere est 'Credit' ou 'PE'."""
        for val in tipping_points["preferred_canal"]:
            assert val in {"Credit", "PE"}

    def test_delta_to_switch_non_negative(self, tipping_points):
        assert (tipping_points["delta_to_switch"] >= 0).all()

    def test_preferred_canal_coherent_with_raroc(self, tipping_points):
        """Si RAROC PE > RAROC Credit, le canal prefere est PE."""
        for _, row in tipping_points.iterrows():
            if row["raroc_pe"] > row["raroc_credit"]:
                assert row["preferred_canal"] == "PE", (
                    f"Secteur {row['sector']}: RAROC PE ({row['raroc_pe']}) > "
                    f"RAROC Credit ({row['raroc_credit']}) mais prefere={row['preferred_canal']}"
                )
            else:
                assert row["preferred_canal"] == "Credit"

    def test_delta_equals_abs_diff(self, tipping_points):
        """delta_to_switch = |raroc_pe - raroc_credit|."""
        for _, row in tipping_points.iterrows():
            expected = round(abs(row["raroc_pe"] - row["raroc_credit"]), 4)
            assert abs(row["delta_to_switch"] - expected) < 1e-3


# ============================================================
# Validation constructeur et standalone
# ============================================================

class TestPortfolioComparatorInit:
    """Tests du constructeur PortfolioComparator."""

    def test_valid_init(self, result_credit, result_pe):
        """Le constructeur accepte les DataFrames valides."""
        comp = PortfolioComparator(result_credit, result_pe)
        assert comp.result_credit is result_credit
        assert comp.result_pe is result_pe

    def test_missing_credit_cols_raises(self, result_pe):
        """Colonnes credit manquantes => AssertionError."""
        df_bad = pd.DataFrame({"foo": [1]})
        with pytest.raises(AssertionError, match="credit"):
            PortfolioComparator(df_bad, result_pe)

    def test_missing_pe_cols_raises(self, result_credit):
        """Colonnes PE manquantes => AssertionError."""
        df_bad = pd.DataFrame({"foo": [1]})
        with pytest.raises(AssertionError, match="PE"):
            PortfolioComparator(result_credit, df_bad)


class TestStandalone:
    """Test du bloc __main__."""

    def test_standalone_runs_successfully(self):
        """python -m ifrs9_cockpit.engine.comparator retourne 0."""
        result = subprocess.run(
            [sys.executable, "-m", "ifrs9_cockpit.engine.comparator"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=r"C:\tout\cours\programme",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_standalone_output_contains_validations(self):
        """La sortie contient les validations PASS."""
        result = subprocess.run(
            [sys.executable, "-m", "ifrs9_cockpit.engine.comparator"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=r"C:\tout\cours\programme",
        )
        assert "[PASS]" in result.stdout
        assert "Comparateur" in result.stdout
        assert "valide" in result.stdout
