"""Tests pour la Mortgage PD Suite (positions synthetiques).

Couvre :
  - Entrainement sur 50k positions
  - AUC > 0.70 (conservateur pour CI)
  - PD dans [0, 1]
  - Pas de leakage (pd_position, payment_status absents)
  - Taille du dataset genere
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.fourteen

from ifrs9_cockpit.config import (
    TARGET,
    MORTGAGE_NUMERICAL_FEATURES,
    MORTGAGE_CATEGORICAL_FEATURES,
    MORTGAGE_ENGINEERED_FEATURES,
)
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.synthetic_generator.mortgage_positions import (
    generate_mortgage_training_data,
)


@pytest.fixture(scope="module")
def mortgage_data():
    """Genere 5000 positions hypothecaires pour les tests."""
    return generate_mortgage_training_data(n_positions=5000, seed=42)


@pytest.fixture(scope="module")
def mortgage_suite(mortgage_data):
    """Entraine la suite mortgage PD (3 modeles dont TabNet Light)."""
    suite = PDModelSuite(
        seed=42,
        numerical_features=MORTGAGE_NUMERICAL_FEATURES + MORTGAGE_ENGINEERED_FEATURES,
        categorical_features=MORTGAGE_CATEGORICAL_FEATURES,
        available_models=("LR_WoE", "TabNet", "XGBoost"),
        clipping_bounds={
            "ltv": (0.0, 1.5),
            "dti": (0.0, 1.0),
        },
        tabnet_variant="light",
    )
    suite.fit(mortgage_data)
    return suite


class TestMortgagePDSuite:
    def test_mortgage_suite_trains(self, mortgage_suite):
        """La suite mortgage s'entraine avec 3 modeles."""
        assert len(mortgage_suite.results) == 3
        assert "LR_WoE" in mortgage_suite.results
        assert "TabNet" in mortgage_suite.results
        assert "XGBoost" in mortgage_suite.results

    def test_mortgage_auc_above_070(self, mortgage_suite):
        """AUC test > 0.70 (conservateur pour CI)."""
        for name, result in mortgage_suite.results.items():
            auc = result.metrics_test["auc"]
            assert auc > 0.70, f"{name}: AUC test = {auc:.4f} < 0.70"

    def test_mortgage_pd_in_01(self, mortgage_suite):
        """PD predites dans [0, 1]."""
        for name, result in mortgage_suite.results.items():
            assert result.y_pred_test.min() >= 0.0, f"{name}: PD min < 0"
            assert result.y_pred_test.max() <= 1.0, f"{name}: PD max > 1"

    def test_mortgage_no_leakage(self, mortgage_suite):
        """pd_position, payment_status, mortgage_id ne sont pas dans les features."""
        all_features = set()
        for result in mortgage_suite.results.values():
            all_features.update(result.feature_names)
        leakage_cols = {"pd_position", "lgd_position", "payment_status", "mortgage_id"}
        found = all_features & leakage_cols
        assert not found, f"Leakage features found: {found}"

    def test_mortgage_50k_positions(self):
        """generate_mortgage_training_data genere le bon nombre de positions."""
        df = generate_mortgage_training_data(n_positions=1000, seed=42)
        assert len(df) == 1000
        assert TARGET in df.columns
        # Leakage cols dropped
        assert "pd_position" not in df.columns
        assert "payment_status" not in df.columns
        assert "mortgage_id" not in df.columns
