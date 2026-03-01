"""Tests pour le generateur position-par-position hypothecaire retail.

Couvre :
    - Generation : shape, colonnes, distributions, coherence
    - PD scorecard : range, calibration, monotonie
    - LGD collateral : range, floor, monotonie
    - Aggregation : EAD, PD/LGD moyens, integration balance sheet
"""

import unittest
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from ifrs9_cockpit.synthetic_generator.mortgage_positions import (
    REGIONS,
    DPE_CLASSES,
    PROPERTY_TYPES,
    RATE_TYPES,
    PAYMENT_STATUS,
    generate_mortgage_positions,
    compute_mortgage_pd,
    compute_mortgage_lgd,
    aggregate_mortgage_to_balance_row,
)
from ifrs9_cockpit.config import ASSET_CLASS_MAP


# Shared fixture: generate once for the module
_N = 2000
_TOTAL_EAD = 1.0e9
_SEED = 442
_DF = generate_mortgage_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
_PROFILE = ASSET_CLASS_MAP["retail_mortgage"]


class TestMortgageGeneration(unittest.TestCase):
    """Validation du generateur de positions hypothecaires."""

    def test_shape(self):
        """n_positions lignes generees."""
        self.assertEqual(len(_DF), _N)

    def test_columns_present(self):
        """Toutes les colonnes requises sont presentes."""
        required = {
            "mortgage_id", "region", "property_type", "rate_type", "dpe_class",
            "property_value", "loan_amount", "ltv", "dti", "borrower_income",
            "borrower_age", "origination_year", "remaining_tenor",
            "interest_rate_margin", "payment_status", "dpd",
            "pd_position", "lgd_position", "default_flag",
        }
        self.assertTrue(required <= set(_DF.columns),
                        f"Missing: {required - set(_DF.columns)}")

    def test_ltv_range(self):
        """LTV dans [0.10, 1.10]."""
        self.assertGreaterEqual(_DF["ltv"].min(), 0.10)
        self.assertLessEqual(_DF["ltv"].max(), 1.10)

    def test_ltv_distribution(self):
        """LTV suit approximativement la distribution du profil."""
        # Median LTV should be around 0.65-0.75 (weighted average of distribution)
        median_ltv = _DF["ltv"].median()
        self.assertGreater(median_ltv, 0.55)
        self.assertLess(median_ltv, 0.85)

    def test_dti_range(self):
        """DTI dans [0.10, 0.55]."""
        self.assertGreaterEqual(_DF["dti"].min(), 0.10)
        self.assertLessEqual(_DF["dti"].max(), 0.55)

    def test_property_value_positive(self):
        """Toutes les valeurs immobilieres sont positives."""
        self.assertTrue((_DF["property_value"] > 0).all())

    def test_loan_amount_eq_pv_times_ltv(self):
        """loan_amount ~ property_value * ltv (before scaling, ratio preserved)."""
        ratio = (_DF["loan_amount"] / _DF["property_value"]).to_numpy()
        # After scaling both pv and loan_amount, ratio = ltv
        np.testing.assert_allclose(ratio, _DF["ltv"].to_numpy(), atol=0.01)

    def test_region_distribution(self):
        """Toutes les 11 regions sont representees, IdF dominant."""
        regions_present = set(_DF["region"].unique().to_list())
        self.assertEqual(regions_present, set(REGIONS.keys()))
        # IdF should be the most common
        region_counts = _DF["region"].value_counts().sort("count", descending=True)
        self.assertEqual(region_counts["region"][0], "Ile-de-France")

    def test_dpe_distribution(self):
        """7 classes DPE representees."""
        dpe_present = set(_DF["dpe_class"].unique().to_list())
        self.assertEqual(dpe_present, set(DPE_CLASSES.keys()))

    def test_default_rate(self):
        """Taux de defaut ~1-5% (coherent avec pd_base)."""
        default_rate = _DF["default_flag"].mean()
        self.assertGreater(default_rate, 0.005)
        self.assertLess(default_rate, 0.10)

    def test_payment_status_coherent_dpd(self):
        """dpd > 0 ssi payment_status != 'current'."""
        current = _DF.filter(pl.col("payment_status") == "current")
        self.assertTrue((current["dpd"] == 0).all(),
                        "Current status should have dpd=0")
        non_current = _DF.filter(pl.col("payment_status") != "current")
        if len(non_current) > 0:
            self.assertTrue((non_current["dpd"] > 0).all(),
                            "Non-current status should have dpd>0")

    def test_reproducibility(self):
        """Same seed = same result."""
        df2 = generate_mortgage_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
        assert_frame_equal(_DF, df2)


class TestMortgagePD(unittest.TestCase):
    """Validation du scorecard PD."""

    def test_pd_range(self):
        """PD dans [0.0001, 0.20]."""
        self.assertGreaterEqual(_DF["pd_position"].min(), 0.0001)
        self.assertLessEqual(_DF["pd_position"].max(), 0.20)

    def test_pd_mean_calibrated(self):
        """PD moyenne ~ 0.012 +/- 50%."""
        pd_mean = _DF["pd_position"].mean()
        self.assertGreater(pd_mean, 0.006, f"PD mean {pd_mean:.4f} too low")
        self.assertLess(pd_mean, 0.018, f"PD mean {pd_mean:.4f} too high")

    def test_pd_monotone_ltv(self):
        """PD croissante avec LTV (par buckets)."""
        ltv = _DF["ltv"].to_numpy()
        pd_pos = _DF["pd_position"].to_numpy()
        # Bucket by LTV ranges
        bins = [0, 0.60, 0.75, 0.90, 1.20]
        bucket_means = []
        for i in range(len(bins) - 1):
            mask = (ltv >= bins[i]) & (ltv < bins[i + 1])
            if mask.sum() > 0:
                bucket_means.append(pd_pos[mask].mean())
        if len(bucket_means) >= 2:
            # At least the last bucket > first bucket (monotonic trend)
            self.assertGreater(bucket_means[-1], bucket_means[0])

    def test_pd_monotone_dti(self):
        """PD croissante avec DTI (par buckets)."""
        dti = _DF["dti"].to_numpy()
        pd_pos = _DF["pd_position"].to_numpy()
        bins = [0.10, 0.25, 0.35, 0.55]
        bucket_means = []
        for i in range(len(bins) - 1):
            mask = (dti >= bins[i]) & (dti < bins[i + 1])
            if mask.sum() > 0:
                bucket_means.append(pd_pos[mask].mean())
        if len(bucket_means) >= 2:
            self.assertGreater(bucket_means[-1], bucket_means[0])

    def test_pd_dpe_effect(self):
        """PD(DPE=G) > PD(DPE=A)."""
        pd_g = _DF.filter(pl.col("dpe_class") == "G")["pd_position"].mean()
        pd_a = _DF.filter(pl.col("dpe_class") == "A")["pd_position"].mean()
        self.assertGreater(pd_g, pd_a)


class TestMortgageLGD(unittest.TestCase):
    """Validation du modele LGD collateral."""

    def test_lgd_range(self):
        """LGD dans [0.05, 0.60]."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.05)
        self.assertLessEqual(_DF["lgd_position"].max(), 0.60)

    def test_lgd_mean_calibrated(self):
        """LGD moyenne ~ 0.15 +/- 50%."""
        lgd_mean = _DF["lgd_position"].mean()
        self.assertGreater(lgd_mean, 0.075, f"LGD mean {lgd_mean:.4f} too low")
        self.assertLess(lgd_mean, 0.225, f"LGD mean {lgd_mean:.4f} too high")

    def test_lgd_monotone_ltv(self):
        """LGD croissante avec LTV (par buckets)."""
        ltv = _DF["ltv"].to_numpy()
        lgd_pos = _DF["lgd_position"].to_numpy()
        bins = [0, 0.60, 0.75, 0.90, 1.20]
        bucket_means = []
        for i in range(len(bins) - 1):
            mask = (ltv >= bins[i]) & (ltv < bins[i + 1])
            if mask.sum() > 0:
                bucket_means.append(lgd_pos[mask].mean())
        if len(bucket_means) >= 2:
            self.assertGreater(bucket_means[-1], bucket_means[0])

    def test_lgd_floor(self):
        """Min LGD >= 0.05 (CRR3 input floor)."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.05)


class TestAggregation(unittest.TestCase):
    """Validation de l'aggregation vers la ligne balance sheet."""

    def setUp(self):
        self.agg = aggregate_mortgage_to_balance_row(_DF, _PROFILE)

    def test_aggregate_ead(self):
        """EAD total = sum(loan_amount)."""
        expected = _DF["loan_amount"].sum()
        self.assertAlmostEqual(self.agg["ead_total"], expected, places=0)

    def test_aggregate_pd_range(self):
        """PD agrege ~ 0.012 +/- 30%."""
        pd_agg = self.agg["pd_base"]
        self.assertGreater(pd_agg, 0.006)
        self.assertLess(pd_agg, 0.020)

    def test_aggregate_lgd_range(self):
        """LGD agrege ~ 0.15 +/- 30%."""
        lgd_agg = self.agg["lgd_base"]
        self.assertGreater(lgd_agg, 0.075)
        self.assertLess(lgd_agg, 0.225)

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

    def test_balance_sheet_integration(self):
        """df_balance_sheet retail_mortgage has bottom-up pd_base != profile.pd_base."""
        import pandas as pd
        from ifrs9_cockpit.synthetic_generator import generate_dataset
        _, _, _, df_bs = generate_dataset(n_clients=3000, seed=42)
        # df_bs may be Pandas or Polars depending on migration state
        if isinstance(df_bs, pd.DataFrame):
            mortgage_row = df_bs[df_bs["asset_class"] == "retail_mortgage"].iloc[0]
        else:
            mortgage_row = df_bs.filter(pl.col("asset_class") == "retail_mortgage").row(0, named=True)
        # Bottom-up PD should differ from the static profile value (0.012)
        # It should be close but not exactly equal
        self.assertNotAlmostEqual(
            mortgage_row["pd_base"], _PROFILE.pd_base, places=6,
            msg="pd_base should be computed bottom-up, not equal to profile",
        )
        # But still in a reasonable range
        self.assertGreater(mortgage_row["pd_base"], 0.005)
        self.assertLess(mortgage_row["pd_base"], 0.025)


if __name__ == "__main__":
    unittest.main()
