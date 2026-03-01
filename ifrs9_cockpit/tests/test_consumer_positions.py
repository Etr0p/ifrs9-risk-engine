"""Tests pour les positions credit consommation (donnees reelles Lending Club).

Couvre :
    - Chargement parquet : existence, schema, taille, qualite
    - Generation : shape, colonnes, scaling, distributions
    - PD scorecard : range, calibration, monotonie score/grade/utilization
    - LGD recovery : range, calibration, bimodalite, floor CRR3
    - Aggregation : EAD, PD/LGD moyens, schema, integration balance sheet
"""

import unittest
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from ifrs9_cockpit.synthetic_generator.consumer_positions import (
    _PARQUET_PATH,
    generate_consumer_positions,
    compute_consumer_pd,
    compute_consumer_lgd,
    aggregate_consumer_to_balance_row,
    load_consumer_data,
)
from ifrs9_cockpit.config import ASSET_CLASS_MAP


# ──────────────────────────────────────────────
# Skip all tests if parquet not available
# ──────────────────────────────────────────────
_PARQUET_EXISTS = _PARQUET_PATH.exists()
_SKIP_MSG = (
    "consumer_credit.parquet not found. "
    "Run: python ifrs9_cockpit/data/fetch_lending_club.py"
)

# Shared fixtures (generated once for module, only if parquet exists)
_N = 5000
_TOTAL_EAD = 1.0e9
_SEED = 542
_DF = None
_DF_FULL = None
_PROFILE = ASSET_CLASS_MAP.get("consumer_credit")

if _PARQUET_EXISTS:
    _DF_FULL = load_consumer_data()
    _DF = generate_consumer_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)


# ──────────────────────────────────────────────
# DATA LOAD
# ──────────────────────────────────────────────

@unittest.skipUnless(_PARQUET_EXISTS, _SKIP_MSG)
class TestConsumerDataLoad(unittest.TestCase):
    """Validation du parquet pre-traite."""

    def test_parquet_exists(self):
        """Fichier consumer_credit.parquet present."""
        self.assertTrue(_PARQUET_PATH.exists())

    def test_parquet_schema(self):
        """Colonnes attendues presentes."""
        required = {
            "consumer_id", "loan_amount", "credit_score", "default_flag",
            "borrower_income", "dti", "grade",
        }
        self.assertTrue(
            required <= set(_DF_FULL.columns),
            f"Missing: {required - set(_DF_FULL.columns)}",
        )

    def test_parquet_size(self):
        """Au moins 1000 lignes (echantillon stratifie)."""
        self.assertGreater(len(_DF_FULL), 1000)

    def test_no_critical_nulls(self):
        """Pas de NaN sur loan_amount, credit_score, default_flag."""
        for col in ["loan_amount", "credit_score", "default_flag"]:
            if col in _DF_FULL.columns:
                self.assertEqual(
                    _DF_FULL[col].is_null().sum(), 0,
                    f"NaN found in critical column {col}",
                )


# ──────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────

@unittest.skipUnless(_PARQUET_EXISTS, _SKIP_MSG)
class TestConsumerGeneration(unittest.TestCase):
    """Validation du generateur de positions consumer."""

    def test_shape(self):
        """n_positions lignes generees."""
        self.assertEqual(len(_DF), _N)

    def test_columns_present(self):
        """Colonnes requises presentes."""
        required = {
            "consumer_id", "loan_amount", "credit_score", "default_flag",
            "borrower_income", "dti", "grade",
            "pd_position", "lgd_position",
        }
        self.assertTrue(
            required <= set(_DF.columns),
            f"Missing: {required - set(_DF.columns)}",
        )

    def test_ead_scaling(self):
        """sum(loan_amount) ~ total_ead."""
        actual = _DF["loan_amount"].sum()
        self.assertAlmostEqual(actual / _TOTAL_EAD, 1.0, places=3)

    def test_credit_score_range(self):
        """Credit score dans [300, 850]."""
        self.assertGreaterEqual(_DF["credit_score"].min(), 300)
        self.assertLessEqual(_DF["credit_score"].max(), 850)

    def test_dti_range(self):
        """DTI dans [0, 100] (valeurs realistes)."""
        if "dti" in _DF.columns:
            valid = _DF["dti"].drop_nulls()
            if len(valid) > 0:
                self.assertGreaterEqual(valid.min(), 0)
                self.assertLessEqual(valid.max(), 100)

    def test_default_rate_raw(self):
        """Taux de defaut brut (Lending Club) dans [5%, 30%]."""
        default_rate = _DF["default_flag"].mean()
        self.assertGreater(default_rate, 0.05,
                           f"Default rate {default_rate:.2%} too low")
        self.assertLess(default_rate, 0.30,
                        f"Default rate {default_rate:.2%} too high")

    def test_reproducibility(self):
        """Meme seed = meme resultat."""
        df2 = generate_consumer_positions(
            n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED,
        )
        assert_frame_equal(_DF, df2)


# ──────────────────────────────────────────────
# PD SCORECARD
# ──────────────────────────────────────────────

@unittest.skipUnless(_PARQUET_EXISTS, _SKIP_MSG)
class TestConsumerPD(unittest.TestCase):
    """Validation du scorecard PD consumer."""

    def test_pd_range(self):
        """PD dans [0.005, 0.30]."""
        self.assertGreaterEqual(_DF["pd_position"].min(), 0.005)
        self.assertLessEqual(_DF["pd_position"].max(), 0.30)

    def test_pd_mean_calibrated(self):
        """PD moyenne ~0.035 +/- 75% (recalibre FR)."""
        pd_mean = _DF["pd_position"].mean()
        self.assertGreater(pd_mean, 0.01,
                           f"PD mean {pd_mean:.4f} too low")
        self.assertLess(pd_mean, 0.08,
                        f"PD mean {pd_mean:.4f} too high")

    def test_pd_monotone_score(self):
        """PD decroissante avec credit_score (par buckets)."""
        cs = _DF["credit_score"].to_numpy()
        pd_pos = _DF["pd_position"].to_numpy()
        bins = [300, 600, 680, 720, 780, 850]
        bucket_means = []
        for i in range(len(bins) - 1):
            mask = (cs >= bins[i]) & (cs < bins[i + 1])
            if mask.sum() > 0:
                bucket_means.append(pd_pos[mask].mean())
        if len(bucket_means) >= 2:
            # PD should decrease: first bucket > last bucket
            self.assertGreater(bucket_means[0], bucket_means[-1])

    def test_pd_grade_ordering(self):
        """PD(G) > PD(F) > ... > PD(A) (approximatif)."""
        if "grade" not in _DF.columns:
            self.skipTest("No grade column")
        grade_pd = (
            _DF.group_by("grade")
            .agg(pl.col("pd_position").mean().alias("pd_mean"))
        )
        grade_dict = dict(zip(
            grade_pd["grade"].to_list(),
            grade_pd["pd_mean"].to_list(),
        ))
        for g_high, g_low in [("G", "A"), ("F", "B"), ("E", "C")]:
            if g_high in grade_dict and g_low in grade_dict:
                self.assertGreater(
                    grade_dict[g_high], grade_dict[g_low],
                    f"PD({g_high})={grade_dict[g_high]:.4f} should > "
                    f"PD({g_low})={grade_dict[g_low]:.4f}",
                )

    def test_pd_utilization_effect(self):
        """High utilization -> higher PD."""
        if "utilization_rate" not in _DF.columns:
            self.skipTest("No utilization_rate column")
        valid = _DF.drop_nulls(subset=["utilization_rate"])
        low_util = valid.filter(pl.col("utilization_rate") < 0.30)["pd_position"].mean()
        high_util = valid.filter(pl.col("utilization_rate") > 0.70)["pd_position"].mean()
        if not np.isnan(low_util) and not np.isnan(high_util):
            self.assertGreater(high_util, low_util)


# ──────────────────────────────────────────────
# LGD RECOVERY
# ──────────────────────────────────────────────

@unittest.skipUnless(_PARQUET_EXISTS, _SKIP_MSG)
class TestConsumerLGD(unittest.TestCase):
    """Validation du modele LGD consumer."""

    def test_lgd_range(self):
        """LGD dans [0.0, 0.95]."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.0)
        self.assertLessEqual(_DF["lgd_position"].max(), 0.95)

    def test_lgd_mean_calibrated(self):
        """LGD moyenne ~0.50 +/- 50% (bimodale: self-cure + loss)."""
        lgd_mean = _DF["lgd_position"].mean()
        self.assertGreater(lgd_mean, 0.25,
                           f"LGD mean {lgd_mean:.4f} too low")
        self.assertLess(lgd_mean, 0.80,
                        f"LGD mean {lgd_mean:.4f} too high")

    def test_lgd_bimodal(self):
        """Presence de self-cures (lgd < 0.10)."""
        n_self_cure = (_DF["lgd_position"] < 0.10).sum()
        pct = n_self_cure / len(_DF)
        self.assertGreater(pct, 0.10,
                           f"Only {pct:.2%} self-cures (expected ~30%)")
        self.assertLess(pct, 0.50,
                        f"{pct:.2%} self-cures (expected ~30%)")

    def test_lgd_floor(self):
        """LGD non-cure >= 0.25 (CRR3 floor unsecured)."""
        non_cure = _DF.filter(pl.col("lgd_position") >= 0.10)
        if len(non_cure) > 0:
            self.assertGreaterEqual(
                non_cure["lgd_position"].min(), 0.25,
                "Non-cure LGD should respect CRR3 floor 25%",
            )

    def test_lgd_from_real_recovery(self):
        """LGD calculee depuis recoveries reelles (pas constante)."""
        lgd_unique = _DF["lgd_position"].n_unique()
        self.assertGreater(lgd_unique, 10,
                           "LGD should have many distinct values (from real data)")


# ──────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────

@unittest.skipUnless(_PARQUET_EXISTS, _SKIP_MSG)
class TestConsumerAggregation(unittest.TestCase):
    """Validation de l'aggregation vers la ligne balance sheet."""

    def setUp(self):
        self.agg = aggregate_consumer_to_balance_row(_DF, _PROFILE)

    def test_aggregate_ead(self):
        """EAD total = sum(loan_amount)."""
        expected = _DF["loan_amount"].sum()
        self.assertAlmostEqual(self.agg["ead_total"], expected, places=0)

    def test_aggregate_pd_range(self):
        """PD agrege ~0.035 +/- 60%."""
        pd_agg = self.agg["pd_base"]
        self.assertGreater(pd_agg, 0.01)
        self.assertLess(pd_agg, 0.08)

    def test_aggregate_lgd_range(self):
        """LGD agrege ~0.50 +/- 50%."""
        lgd_agg = self.agg["lgd_base"]
        self.assertGreater(lgd_agg, 0.25)
        self.assertLess(lgd_agg, 0.80)

    def test_aggregate_schema(self):
        """La ligne contient toutes les colonnes balance sheet."""
        required_keys = {
            "asset_class", "label", "category", "ead_total", "typical_weight",
            "pd_base", "lgd_base", "tenor", "asset_correlation", "rw_crr3",
            "physical_risk", "transition_risk", "green_capex_ratio",
            "scope3_exposure", "absorption_buffer", "hqla_eligible",
            "hqla_level", "exempt_from_staging", "rsf_weight",
        }
        self.assertTrue(required_keys <= set(self.agg.keys()))


if __name__ == "__main__":
    unittest.main()
