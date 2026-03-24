"""Tests pour la Consumer PD Suite (Lending Club).

Couvre :
  - Entrainement sur donnees reelles (skipif no parquet)
  - AUC > 0.75 (conservateur pour CI)
  - PD dans [0, 1]
  - Pas de leakage (payment_status absent)
  - Save/load roundtrip
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
    CONSUMER_NUMERICAL_FEATURES,
    CONSUMER_CATEGORICAL_FEATURES,
    CONSUMER_ENGINEERED_FEATURES,
)
from ifrs9_cockpit.models.pd_model import PDModelSuite

# Parquet path
_PARQUET = Path(__file__).parent.parent / "data" / "consumer_credit.parquet"
_HAS_PARQUET = _PARQUET.exists()


@pytest.fixture(scope="module")
def consumer_data():
    """Charge les donnees consumer Lending Club (split train pour les tests)."""
    if not _HAS_PARQUET:
        pytest.skip("consumer_credit.parquet not found")
    from ifrs9_cockpit.synthetic_generator.consumer_positions import load_consumer_data
    return load_consumer_data(split="train")


@pytest.fixture(scope="module")
def consumer_suite(consumer_data):
    """Entraine la suite consumer PD (3 modeles dont TabNet Light)."""
    suite = PDModelSuite(
        seed=42,
        numerical_features=CONSUMER_NUMERICAL_FEATURES + CONSUMER_ENGINEERED_FEATURES,
        categorical_features=CONSUMER_CATEGORICAL_FEATURES,
        available_models=("LR_WoE", "TabNet", "XGBoost"),
        clipping_bounds={
            "credit_score": (300.0, 850.0),
            "dti": (0.0, 100.0),
            "utilization_rate": (0.0, 1.5),
        },
        tabnet_variant="light",
    )
    suite.fit(consumer_data)
    return suite


@pytest.mark.skipif(not _HAS_PARQUET, reason="consumer_credit.parquet not found")
class TestConsumerPDSuite:
    def test_consumer_suite_trains(self, consumer_suite):
        """La suite consumer s'entraine avec 3 modeles."""
        assert len(consumer_suite.results) == 3
        assert "LR_WoE" in consumer_suite.results
        assert "TabNet" in consumer_suite.results
        assert "XGBoost" in consumer_suite.results

    def test_consumer_auc_above_075(self, consumer_suite):
        """AUC test > 0.75 (conservateur pour CI)."""
        for name, result in consumer_suite.results.items():
            auc = result.metrics_test["auc"]
            assert auc > 0.75, f"{name}: AUC test = {auc:.4f} < 0.75"

    def test_consumer_pd_in_01(self, consumer_suite):
        """PD predites dans [0, 1]."""
        for name, result in consumer_suite.results.items():
            assert result.y_pred_test.min() >= 0.0, f"{name}: PD min < 0"
            assert result.y_pred_test.max() <= 1.0, f"{name}: PD max > 1"

    def test_consumer_no_leakage(self, consumer_suite):
        """payment_status et lgd_observed ne sont pas dans les features."""
        all_features = set()
        for result in consumer_suite.results.values():
            all_features.update(result.feature_names)
        leakage_cols = {"payment_status", "lgd_observed", "pd_position", "lgd_position"}
        found = all_features & leakage_cols
        assert not found, f"Leakage features found: {found}"

    def test_consumer_save_load_roundtrip(self, consumer_suite, consumer_data):
        """Save/load preserve les predictions."""
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        consumer_suite.save(path)
        loaded = PDModelSuite.load(path)
        assert set(loaded.results.keys()) == set(consumer_suite.results.keys())
        # Check AUC is preserved
        for name in consumer_suite.results:
            orig_auc = consumer_suite.results[name].metrics_test["auc"]
            load_auc = loaded.results[name].metrics_test["auc"]
            assert abs(orig_auc - load_auc) < 1e-6, f"{name}: AUC mismatch"
        Path(path).unlink(missing_ok=True)
