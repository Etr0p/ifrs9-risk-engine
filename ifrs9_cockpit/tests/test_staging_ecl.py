"""Tests unitaires pour ifrs9_cockpit/engine/staging.py et ecl_calculator.py (Story 2-3).

Couvre les taches :
  T1: Staging IFRS 9 (3 stages, SICR multi-facteurs)
  T2: ECL multi-scenarios (Base 50%, Adverse 25%, Favorable 25%)
  T3: ECL >= 0, stages dans {1, 2, 3}
  T4: PD macro-conditionnelle (FR7)
  T5: Modules standalone
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ifrs9_cockpit.config import (
    IFRS9_CONFIG,
    RANDOM_SEED,
    SCENARIOS,
    SICR_CONFIG,
    TARGET,
)
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.engine.staging import StagingEngine, compute_sicr_score, compute_macro_z
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.models.pd_model import PDModelSuite


# ============================================================
# Fixtures partagees
# ============================================================

@pytest.fixture(scope="module")
def pipeline_data():
    """Genere les donnees et entraine tous les modeles."""
    df_credit, _, _ = generate_dataset(n_clients=2000, seed=RANDOM_SEED)

    pd_suite = PDModelSuite(seed=RANDOM_SEED)
    pd_suite.fit(df_credit)
    pd_current = pd_suite.predict_active(df_credit)
    pd_origination = df_credit["pd_origination"].values

    lgd_model = LGDModel(seed=RANDOM_SEED)
    lgd_model.fit(df_credit)

    ead_model = EADModel(seed=RANDOM_SEED)
    ead_model.fit(df_credit)

    return {
        "df_credit": df_credit,
        "pd_current": pd_current,
        "pd_origination": pd_origination,
        "lgd_model": lgd_model,
        "ead_model": ead_model,
    }


@pytest.fixture(scope="module")
def ecl_result(pipeline_data):
    """Calcule l'ECL complet."""
    calc = ECLCalculator(
        lgd_model=pipeline_data["lgd_model"],
        ead_model=pipeline_data["ead_model"],
    )
    result = calc.calculate(
        pipeline_data["df_credit"],
        pipeline_data["pd_current"],
        pipeline_data["pd_origination"],
    )
    return result


# ============================================================
# Staging IFRS 9
# ============================================================

class TestStaging:
    def test_stages_in_123(self, ecl_result):
        """Les stages sont dans {1, 2, 3}."""
        stages = ecl_result["stage"].values
        assert set(np.unique(stages)).issubset({1, 2, 3})

    def test_all_three_stages_present(self, ecl_result):
        """Les 3 stages sont representes (le dataset est assez grand)."""
        stages = ecl_result["stage"].values
        assert len(np.unique(stages)) == 3, f"Stages presentes: {np.unique(stages)}"

    def test_stage3_for_defaults(self, ecl_result):
        """Les defauts averes sont en Stage 3."""
        defaults = ecl_result[ecl_result["default_flag"] == 1]
        if len(defaults) > 0:
            assert (defaults["stage"] == 3).all(), \
                f"Defauts hors Stage 3: {(defaults['stage'] != 3).sum()}"

    def test_stage3_for_high_dpd(self, pipeline_data):
        """DPD >= 90 -> Stage 3."""
        engine = StagingEngine()
        n = 100
        pd_c = np.full(n, 0.05)
        pd_o = np.full(n, 0.05)
        dpd = np.full(n, 95)
        default_flag = np.zeros(n)
        stages = engine.assign_stages(pd_c, pd_o, dpd, default_flag)
        assert (stages == 3).all()

    def test_stage1_for_performing(self):
        """Pas de SICR ni defaut -> Stage 1."""
        engine = StagingEngine()
        n = 100
        pd_c = np.full(n, 0.03)
        pd_o = np.full(n, 0.03)
        dpd = np.zeros(n)
        default_flag = np.zeros(n)
        stages = engine.assign_stages(pd_c, pd_o, dpd, default_flag)
        assert (stages == 1).all()

    def test_sicr_triggers_stage2(self):
        """Score SICR eleve -> Stage 2."""
        engine = StagingEngine()
        n = 100
        # PD doublee : ratio = 1.0, delta = 0.05
        pd_c = np.full(n, 0.10)
        pd_o = np.full(n, 0.03)
        dpd = np.full(n, 35)  # DPD contribue au score
        default_flag = np.zeros(n)
        stages = engine.assign_stages(pd_c, pd_o, dpd, default_flag)
        # Au moins certains devraient etre en Stage 2
        assert (stages == 2).any() or (stages == 3).any()

    def test_compute_sicr_score_shape(self):
        """compute_sicr_score retourne un array de bonne taille."""
        rng = np.random.default_rng(RANDOM_SEED)
        n = 50
        pd_c = rng.uniform(0.01, 0.20, n)
        pd_o = rng.uniform(0.01, 0.10, n)
        dpd = rng.integers(0, 60, n)
        scores = compute_sicr_score(pd_c, pd_o, dpd)
        assert scores.shape == (n,)

    def test_compute_macro_z_base_is_zero(self):
        """Z-score macro au scenario base = 0."""
        from ifrs9_cockpit.config import SCENARIO_BASE
        z = compute_macro_z({
            "unemployment_rate": SCENARIO_BASE.unemployment_rate,
            "gdp_growth": SCENARIO_BASE.gdp_growth,
            "interest_rate": SCENARIO_BASE.interest_rate,
            "hpi_growth": SCENARIO_BASE.hpi_growth,
            "inflation_rate": SCENARIO_BASE.inflation_rate,
        })
        assert abs(z) < 1e-10

    def test_transition_matrix(self):
        """La matrice de transition est 3x3 avec lignes sommant a ~1."""
        engine = StagingEngine()
        stages_t0 = np.array([1, 1, 1, 2, 2, 3])
        stages_t1 = np.array([1, 2, 1, 2, 3, 3])
        matrix = engine.compute_transition_matrix(stages_t0, stages_t1)
        assert matrix.shape == (3, 3)
        # Chaque ligne somme a ~1
        for i in range(3):
            row_sum = matrix.iloc[i].sum()
            if row_sum > 0:
                assert abs(row_sum - 1.0) < 0.01

    def test_stage_summary(self):
        """get_stage_summary retourne le bon format."""
        engine = StagingEngine()
        stages = np.array([1, 1, 1, 2, 3])
        ead = np.array([100.0, 200.0, 150.0, 300.0, 500.0])
        summary = engine.get_stage_summary(stages, ead)
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3
        assert "stage" in summary.columns


# ============================================================
# ECL Calculator
# ============================================================

class TestECLCalculator:
    def test_ecl_weighted_non_negative(self, ecl_result):
        """ECL ponderee >= 0 pour toutes les positions."""
        assert (ecl_result["ecl_weighted"] >= 0).all()

    def test_ecl_base_non_negative(self, ecl_result):
        """ECL base >= 0."""
        assert (ecl_result["ecl_base"] >= 0).all()

    def test_ecl_adverse_non_negative(self, ecl_result):
        """ECL adverse >= 0."""
        assert (ecl_result["ecl_adverse"] >= 0).all()

    def test_ecl_favorable_non_negative(self, ecl_result):
        """ECL favorable >= 0."""
        assert (ecl_result["ecl_favorable"] >= 0).all()

    def test_ecl_weighted_is_scenario_average(self, ecl_result):
        """ECL weighted = 50% Base + 25% Adverse + 25% Favorable."""
        weights = {s.name: s.weight for s in SCENARIOS}
        expected = (
            weights["Base"] * ecl_result["ecl_base"]
            + weights["Adverse"] * ecl_result["ecl_adverse"]
            + weights["Favorable"] * ecl_result["ecl_favorable"]
        )
        np.testing.assert_allclose(
            ecl_result["ecl_weighted"].values,
            np.round(expected.values, 2),
            atol=0.02,
        )

    def test_ecl_result_columns(self, ecl_result):
        """Les colonnes attendues sont presentes dans le resultat."""
        required = {
            "stage", "pd_12m", "pd_lifetime", "lgd", "ead",
            "discount_factor", "ecl_base", "ecl_adverse",
            "ecl_favorable", "ecl_weighted",
        }
        assert required.issubset(set(ecl_result.columns))

    def test_pd_12m_in_01(self, ecl_result):
        """PD 12m dans [0, 1]."""
        assert (ecl_result["pd_12m"] >= 0).all()
        assert (ecl_result["pd_12m"] <= 1).all()

    def test_pd_lifetime_gte_pd_12m(self, ecl_result):
        """PD lifetime >= PD 12m (horizon plus long => cumul plus eleve)."""
        # Pour Stage 2/3, PD lifetime est sur un horizon > 1 an
        stage23 = ecl_result[ecl_result["stage"] >= 2]
        if len(stage23) > 0:
            assert (stage23["pd_lifetime"] >= stage23["pd_12m"] - 1e-6).all()

    def test_lgd_in_01(self, ecl_result):
        """LGD dans [0, 1]."""
        assert (ecl_result["lgd"] >= 0).all()
        assert (ecl_result["lgd"] <= 1).all()

    def test_ead_positive(self, ecl_result):
        """EAD > 0 (ou >= 0 avec bruit)."""
        assert (ecl_result["ead"] >= 0).all()

    def test_discount_factor_bounded(self, ecl_result):
        """Discount factor dans ]0, 1]."""
        df = ecl_result["discount_factor"]
        assert (df > 0).all()
        assert (df <= 1.0 + 1e-6).all()

    def test_ecl_summary(self, ecl_result, pipeline_data):
        """compute_ecl_summary retourne un resume par stage/secteur."""
        calc = ECLCalculator(
            lgd_model=pipeline_data["lgd_model"],
            ead_model=pipeline_data["ead_model"],
        )
        summary = calc.compute_ecl_summary(ecl_result)
        assert isinstance(summary, pd.DataFrame)
        assert "ecl_total" in summary.columns
        assert "coverage_ratio" in summary.columns

    def test_credit_spread_computed(self, ecl_result):
        """Credit spread Merton est calcule et borne."""
        assert "credit_spread" in ecl_result.columns
        spread = ecl_result["credit_spread"]
        assert (spread >= 0.0050).all()  # Min 50 bps
        assert (spread <= 0.2000).all()  # Max 2000 bps

    def test_rwa_credit_computed(self, ecl_result):
        """RWA credit est calcule et >= 0."""
        assert "rwa_credit" in ecl_result.columns
        assert (ecl_result["rwa_credit"] >= 0).all()


# ============================================================
# Standalone
# ============================================================

class TestStandalone:
    def test_staging_module_importable(self):
        """Le module staging.py s'importe sans erreur."""
        result = subprocess.run(
            [sys.executable, "-c",
             "from ifrs9_cockpit.engine.staging import StagingEngine; print('OK')"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
        assert "OK" in result.stdout

    def test_ecl_calculator_module_runs_standalone(self):
        """Le module ecl_calculator.py s'execute sans erreur."""
        result = subprocess.run(
            [sys.executable, "-m", "ifrs9_cockpit.engine.ecl_calculator"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
        assert "Phase 3 valid" in result.stdout
