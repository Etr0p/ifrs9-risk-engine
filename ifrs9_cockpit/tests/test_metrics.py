"""Tests unitaires pour ifrs9_cockpit/analytics/metrics.py (Story 2-4).

Couvre les taches :
  T1: Metriques de discrimination (AUC, Gini, KS)
  T2: Population Stability Index (PSI)
  T3: Courbes ROC, CAP, KS
  T4: Table de classification par decile
  T5: Backtesting walk-forward
  T6: HHI (concentration)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ifrs9_cockpit.analytics.metrics import ModelMetrics


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def binary_data():
    """Genere des donnees binaires avec un modele discriminant."""
    rng = np.random.default_rng(42)
    n = 1000
    y_true = rng.binomial(1, 0.1, n)
    # Scores correles aux labels (modele imparfait)
    y_score = np.where(
        y_true == 1,
        rng.uniform(0.3, 0.9, n),
        rng.uniform(0.0, 0.5, n),
    )
    return y_true, y_score


@pytest.fixture(scope="module")
def random_data():
    """Genere des donnees avec un modele aleatoire (AUC ~ 0.5)."""
    rng = np.random.default_rng(123)
    n = 1000
    y_true = rng.binomial(1, 0.1, n)
    y_score = rng.uniform(0.0, 1.0, n)
    return y_true, y_score


# ============================================================
# T1 — AUC, Gini, KS
# ============================================================

class TestDiscriminationMetrics:
    def test_auc_discriminant_model(self, binary_data):
        """AUC > 0.70 pour un modele discriminant."""
        y_true, y_score = binary_data
        auc = ModelMetrics.auc(y_true, y_score)
        assert 0.70 < auc <= 1.0

    def test_auc_random_model(self, random_data):
        """AUC ~ 0.50 pour un modele aleatoire."""
        y_true, y_score = random_data
        auc = ModelMetrics.auc(y_true, y_score)
        assert 0.35 < auc < 0.65

    def test_gini_equals_2auc_minus_1(self, binary_data):
        """Gini = 2 * AUC - 1."""
        y_true, y_score = binary_data
        auc = ModelMetrics.auc(y_true, y_score)
        gini = ModelMetrics.gini(y_true, y_score)
        assert abs(gini - (2 * auc - 1)) < 1e-6

    def test_gini_positive_for_good_model(self, binary_data):
        """Gini > 0 pour un modele discriminant."""
        y_true, y_score = binary_data
        gini = ModelMetrics.gini(y_true, y_score)
        assert gini > 0

    def test_ks_positive(self, binary_data):
        """KS > 0 pour un modele discriminant."""
        y_true, y_score = binary_data
        ks = ModelMetrics.ks_statistic(y_true, y_score)
        assert ks > 0

    def test_ks_bounded_01(self, binary_data):
        """KS dans [0, 1]."""
        y_true, y_score = binary_data
        ks = ModelMetrics.ks_statistic(y_true, y_score)
        assert 0 <= ks <= 1


# ============================================================
# T2 — PSI
# ============================================================

class TestPSI:
    def test_psi_same_distribution(self, binary_data):
        """PSI ~ 0 pour la meme distribution."""
        _, y_score = binary_data
        psi = ModelMetrics.psi(y_score, y_score)
        assert psi < 0.01

    def test_psi_non_negative(self, binary_data, random_data):
        """PSI >= 0 toujours."""
        _, y_score1 = binary_data
        _, y_score2 = random_data
        psi = ModelMetrics.psi(y_score1, y_score2)
        assert psi >= 0

    def test_psi_shifted_distribution(self):
        """PSI > 0.10 pour une distribution significativement shiftee."""
        rng = np.random.default_rng(42)
        expected = rng.uniform(0.0, 0.5, 1000)
        actual = rng.uniform(0.3, 0.8, 1000)  # Distribution tres shiftee
        psi = ModelMetrics.psi(expected, actual)
        assert psi > 0.10


# ============================================================
# T3 — Courbes ROC, CAP, KS
# ============================================================

class TestCurves:
    def test_roc_curve_data(self, binary_data):
        """roc_curve_data retourne 3 arrays."""
        y_true, y_score = binary_data
        fpr, tpr, thresholds = ModelMetrics.roc_curve_data(y_true, y_score)
        assert len(fpr) == len(tpr) == len(thresholds)
        assert fpr[0] == 0.0
        assert tpr[0] == 0.0

    def test_cap_curve_data(self, binary_data):
        """cap_curve_data retourne 2 arrays monotones."""
        y_true, y_score = binary_data
        prop_pop, prop_def = ModelMetrics.cap_curve_data(y_true, y_score)
        assert len(prop_pop) == len(prop_def)
        # Monotone croissant
        assert (np.diff(prop_pop) >= 0).all()
        assert (np.diff(prop_def) >= -1e-10).all()

    def test_ks_curve_data(self, binary_data):
        """ks_curve_data retourne les CDFs et la valeur KS."""
        y_true, y_score = binary_data
        thresholds, cdf_def, cdf_nondef, ks_val = ModelMetrics.ks_curve_data(y_true, y_score)
        assert len(thresholds) == 100
        assert ks_val > 0
        assert ks_val <= 1


# ============================================================
# T4 — Table de classification
# ============================================================

class TestClassificationTable:
    def test_classification_table_shape(self, binary_data):
        """La table de classification a le bon nombre de bins."""
        y_true, y_score = binary_data
        table = ModelMetrics.classification_table(y_true, y_score, n_bins=10)
        assert isinstance(table, pl.DataFrame)
        assert len(table) <= 10
        assert len(table) >= 5  # Quelques bins au moins

    def test_classification_table_columns(self, binary_data):
        """Les colonnes attendues sont presentes."""
        y_true, y_score = binary_data
        table = ModelMetrics.classification_table(y_true, y_score)
        required_cols = {"count", "n_defaults", "avg_score", "default_rate"}
        assert required_cols.issubset(set(table.columns))

    def test_classification_table_near_monotone(self, binary_data):
        """Le taux de defaut est near-monotone croissant par decile (AC 3.2)."""
        y_true, y_score = binary_data
        table = ModelMetrics.classification_table(y_true, y_score, n_bins=10)
        default_rates = table["default_rate"].to_numpy()
        # Compter les inversions (tolerance : max 2 inversions pour near-monotone)
        inversions = sum(1 for i in range(len(default_rates) - 1)
                        if default_rates[i] > default_rates[i + 1] + 0.01)
        assert inversions <= 2, f"{inversions} inversions dans la calibration table"


# ============================================================
# T5 — Backtesting
# ============================================================

class TestBacktesting:
    def test_backtesting_returns_dataframe(self, binary_data):
        """compute_backtesting_metrics retourne un DataFrame."""
        y_true, y_score = binary_data
        bt = ModelMetrics.compute_backtesting_metrics(y_true, y_score, n_folds=5)
        assert isinstance(bt, pl.DataFrame)
        assert len(bt) > 0

    def test_backtesting_columns(self, binary_data):
        """Les colonnes attendues sont presentes."""
        y_true, y_score = binary_data
        bt = ModelMetrics.compute_backtesting_metrics(y_true, y_score)
        required = {"month", "auc", "gini", "ks", "psi"}
        assert required.issubset(set(bt.columns))

    def test_backtesting_auc_bounded(self, binary_data):
        """AUC par fold dans [0, 1]."""
        y_true, y_score = binary_data
        bt = ModelMetrics.compute_backtesting_metrics(y_true, y_score)
        assert (bt["auc"] >= 0).all()
        assert (bt["auc"] <= 1).all()

    def test_backtesting_auc_stable(self, binary_data):
        """AUC stable entre folds (std < 0.10) — AC 3.3."""
        y_true, y_score = binary_data
        bt = ModelMetrics.compute_backtesting_metrics(y_true, y_score, n_folds=5)
        auc_std = bt["auc"].std()
        assert auc_std < 0.10, f"AUC std = {auc_std:.4f} >= 0.10 (instable)"


# ============================================================
# T6 — HHI
# ============================================================

class TestHHI:
    def test_hhi_equal_shares(self):
        """HHI pour parts egales = 1/N."""
        shares = np.array([100, 100, 100, 100])  # 4 parts egales
        hhi = ModelMetrics.hhi(shares)
        assert abs(hhi - 0.25) < 0.001

    def test_hhi_monopoly(self):
        """HHI pour un monopole = 1.0."""
        shares = np.array([1000, 0, 0, 0])
        hhi = ModelMetrics.hhi(shares)
        assert abs(hhi - 1.0) < 0.001

    def test_hhi_bounded(self):
        """HHI dans [0, 1]."""
        shares = np.array([30, 25, 20, 15, 10])
        hhi = ModelMetrics.hhi(shares)
        assert 0 <= hhi <= 1

    def test_hhi_empty(self):
        """HHI pour un portefeuille vide = 0."""
        shares = np.array([0, 0, 0])
        hhi = ModelMetrics.hhi(shares)
        assert hhi == 0.0


# ============================================================
# compute_all
# ============================================================

class TestComputeAll:
    def test_compute_all_keys(self, binary_data):
        """compute_all retourne les 9 metriques."""
        y_true, y_score = binary_data
        metrics = ModelMetrics.compute_all(y_true, y_score)
        expected = {"auc", "gini", "ks", "psi", "brier", "logloss", "precision", "recall", "f1"}
        assert set(metrics.keys()) == expected

    def test_compute_all_with_ref(self, binary_data, random_data):
        """compute_all avec y_score_ref calcule le PSI."""
        y_true, y_score = binary_data
        _, y_ref = random_data
        metrics = ModelMetrics.compute_all(y_true, y_score, y_score_ref=y_ref)
        assert metrics["psi"] > 0

    def test_compute_all_without_ref(self, binary_data):
        """compute_all sans y_score_ref donne PSI = 0."""
        y_true, y_score = binary_data
        metrics = ModelMetrics.compute_all(y_true, y_score)
        assert metrics["psi"] == 0.0
