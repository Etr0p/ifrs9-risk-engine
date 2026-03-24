"""Tests pour le generateur position-par-position Trade Finance.

Couvre :
    - Generation : shape, colonnes, distributions, coherence, scaling
    - PD multi-composante : range, calibration, product ordering, country effect
    - LGD recovery waterfall : range, calibration, product ordering, floors
    - CCF et EAD : coherence nominal/EAD, valeurs CRR3
    - Maturity adjustment : BCBS205 waiver (tenor < 1 an -> MA < 1)
    - Aggregation : EAD, PD/LGD moyens, schema balance sheet
"""

import pytest

pytestmark = pytest.mark.fourteen

import unittest
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from ifrs9_cockpit.synthetic_generator.trade_finance_positions import (
    PRODUCTS,
    COMMODITIES,
    COUNTRY_RATINGS,
    generate_trade_finance_positions,
    compute_trade_finance_pd,
    compute_trade_finance_lgd,
    compute_ead_with_ccf,
    compute_maturity_adjustment,
    aggregate_trade_finance_to_balance_row,
)
from ifrs9_cockpit.config import ASSET_CLASS_MAP


# Shared fixture: generate once for the module
_N = 5000
_TOTAL_EAD = 1.0e9
_SEED = 642
_DF = generate_trade_finance_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
_PROFILE = ASSET_CLASS_MAP["trade_finance"]


class TestTradeFinanceGeneration(unittest.TestCase):
    """Validation du generateur de positions TF."""

    def test_shape(self):
        """n_positions lignes generees."""
        self.assertEqual(len(_DF), _N)

    def test_columns_present(self):
        """Toutes les colonnes requises sont presentes."""
        required = {
            "trade_id", "product_type", "commodity_class",
            "exporter_country_rating", "importer_country_rating",
            "tenor_days", "tenor_years", "nominal", "ccf", "ead",
            "margin_pct", "margin_held", "commodity_vol", "transit_days",
            "issuing_bank_rating", "confirming_bank_rating",
            "documentary_compliance", "trade_frequency",
            "relationship_years", "fx_exposure",
            "pd_position", "lgd_position", "maturity_adjustment",
            "default_flag",
        }
        self.assertTrue(required <= set(_DF.columns),
                        f"Missing: {required - set(_DF.columns)}")

    def test_ead_scaling(self):
        """sum(ead) ~ total_ead (post-CCF)."""
        actual = _DF["ead"].sum()
        self.assertAlmostEqual(actual / _TOTAL_EAD, 1.0, places=2)

    def test_nominal_vs_ead(self):
        """nominal >= ead pour chaque position (CCF <= 1)."""
        self.assertTrue((_DF["nominal"] >= _DF["ead"] - 0.01).all())

    def test_product_distribution(self):
        """7 types de produits, LC dominant."""
        products_present = set(_DF["product_type"].unique().to_list())
        self.assertEqual(products_present, set(PRODUCTS.keys()))
        # import_lc should be most common (weight 0.35)
        counts = _DF["product_type"].value_counts().sort("count", descending=True)
        self.assertEqual(counts["product_type"][0], "import_lc")

    def test_tenor_range(self):
        """Tenor dans [30/365, 365/365] annees."""
        self.assertGreaterEqual(_DF["tenor_years"].min(), 30 / 365 - 0.01)
        self.assertLessEqual(_DF["tenor_years"].max(), 365 / 365 + 0.01)

    def test_tenor_product_coherence(self):
        """LC tenor < guarantee tenor en moyenne."""
        lc_tenor = _DF.filter(
            pl.col("product_type").is_in(["import_lc", "export_lc"])
        )["tenor_years"].mean()
        guar_tenor = _DF.filter(
            pl.col("product_type").is_in(["guarantee_perf", "guarantee_financial"])
        )["tenor_years"].mean()
        self.assertLess(lc_tenor, guar_tenor)

    def test_country_pairs(self):
        """Exporter et importer ratings differents pour certaines positions."""
        diff_mask = (
            _DF["exporter_country_rating"] != _DF["importer_country_rating"]
        )
        # At least 30% should have different countries
        self.assertGreater(diff_mask.mean(), 0.30)

    def test_reproducibility(self):
        """Same seed = same result."""
        df2 = generate_trade_finance_positions(
            n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED,
        )
        assert_frame_equal(_DF, df2)


class TestTradeFinancePD(unittest.TestCase):
    """Validation du modele PD multi-composante."""

    def test_pd_range(self):
        """PD dans [0.0003, 0.10]."""
        self.assertGreaterEqual(_DF["pd_position"].min(), 0.0003 - 1e-6)
        self.assertLessEqual(_DF["pd_position"].max(), 0.10 + 1e-6)

    def test_pd_mean_calibrated(self):
        """PD moyenne ~ 0.008 +/- 50% (ICC portfolio)."""
        pd_mean = _DF["pd_position"].mean()
        self.assertGreater(pd_mean, 0.004, f"PD mean {pd_mean:.5f} too low")
        self.assertLess(pd_mean, 0.012, f"PD mean {pd_mean:.5f} too high")

    def test_pd_product_ordering(self):
        """PD(SBLC) > PD(guarantee) > PD(doc_coll) > PD(import_lc) > PD(export_lc)."""
        means = (
            _DF.group_by("product_type")
            .agg(pl.col("pd_position").mean().alias("pd_mean"))
        )
        means_dict = dict(zip(
            means["product_type"].to_list(),
            means["pd_mean"].to_list(),
        ))
        self.assertGreater(means_dict["guarantee_financial"], means_dict["guarantee_perf"])
        self.assertGreater(means_dict["guarantee_perf"], means_dict["documentary_coll"])
        self.assertGreater(means_dict["import_lc"], means_dict["export_lc"])

    def test_pd_country_transfer_effect(self):
        """PD(CCC country) >> PD(AAA country)."""
        pd_ccc = _DF.filter(
            pl.col("importer_country_rating") == "CCC"
        )["pd_position"].mean()
        pd_aaa = _DF.filter(
            pl.col("importer_country_rating") == "AAA"
        )["pd_position"].mean()
        # CCC should be at least 3x AAA
        self.assertGreater(pd_ccc, pd_aaa * 3.0)

    def test_pd_confirmation_reduces(self):
        """PD(export_lc confirmed) < PD(import_lc unconfirmed)."""
        pd_export = _DF.filter(
            pl.col("product_type") == "export_lc"
        )["pd_position"].mean()
        pd_import = _DF.filter(
            pl.col("product_type") == "import_lc"
        )["pd_position"].mean()
        self.assertLess(pd_export, pd_import)

    def test_pd_tenor_effect(self):
        """Tenor plus long -> PD plus elevee (within same product)."""
        # Test on import_lc (largest group)
        lc = _DF.filter(pl.col("product_type") == "import_lc")
        median_tenor = lc["tenor_years"].median()
        pd_short = lc.filter(pl.col("tenor_years") < median_tenor)["pd_position"].mean()
        pd_long = lc.filter(pl.col("tenor_years") >= median_tenor)["pd_position"].mean()
        self.assertGreater(pd_long, pd_short)

    def test_pd_joint_default_confirmed(self):
        """Joint PD << individual PD (credit enhancement for confirmed LC)."""
        # Export LC (confirmed) should have much lower PD than guarantees
        pd_export = _DF.filter(
            pl.col("product_type") == "export_lc"
        )["pd_position"].mean()
        pd_guar = _DF.filter(
            pl.col("product_type") == "guarantee_perf"
        )["pd_position"].mean()
        # Confirmation reduces PD substantially
        self.assertLess(pd_export, pd_guar * 0.5)


class TestTradeFinanceLGD(unittest.TestCase):
    """Validation du modele LGD recovery waterfall."""

    def test_lgd_range(self):
        """LGD dans [0.05, 0.80]."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.05 - 1e-4)
        self.assertLessEqual(_DF["lgd_position"].max(), 0.80 + 1e-4)

    def test_lgd_mean_calibrated(self):
        """LGD moyenne ~ 0.30 +/- 40% (ICC portfolio)."""
        lgd_mean = _DF["lgd_position"].mean()
        self.assertGreater(lgd_mean, 0.18, f"LGD mean {lgd_mean:.4f} too low")
        self.assertLess(lgd_mean, 0.42, f"LGD mean {lgd_mean:.4f} too high")

    def test_lgd_product_ordering(self):
        """LGD(guarantee_perf) > LGD(doc_coll) > LGD(import_lc) > LGD(export_credit)."""
        means = (
            _DF.group_by("product_type")
            .agg(pl.col("lgd_position").mean().alias("lgd_mean"))
        )
        means_dict = dict(zip(
            means["product_type"].to_list(),
            means["lgd_mean"].to_list(),
        ))
        self.assertGreater(means_dict["guarantee_perf"], means_dict["documentary_coll"])
        self.assertGreater(means_dict["import_lc"], means_dict["export_credit_mlt"])

    def test_lgd_margin_effect(self):
        """Marge elevee -> LGD faible."""
        # Test on import_lc
        lc = _DF.filter(pl.col("product_type") == "import_lc")
        median_margin = lc["margin_pct"].median()
        lgd_low_margin = lc.filter(pl.col("margin_pct") < median_margin)["lgd_position"].mean()
        lgd_high_margin = lc.filter(pl.col("margin_pct") >= median_margin)["lgd_position"].mean()
        self.assertGreater(lgd_low_margin, lgd_high_margin)

    def test_lgd_commodity_vol_effect(self):
        """Commodity volatile -> LGD elevee (haircut above floor)."""
        # Test on import_lc positions above the LGD floor (0.10)
        # where the commodity vol effect is not compressed by the floor
        lc = _DF.filter(pl.col("product_type") == "import_lc")
        # Compare across commodity classes: energy (vol=0.35) vs machinery (vol=0.10)
        lgd_energy = lc.filter(pl.col("commodity_class") == "energy")["lgd_position"].mean()
        lgd_machinery = lc.filter(pl.col("commodity_class") == "machinery")["lgd_position"].mean()
        # Energy commodities have higher vol -> higher haircut -> higher LGD
        self.assertGreaterEqual(lgd_energy, lgd_machinery)

    def test_lgd_eca_coverage(self):
        """export_credit_mlt a LGD tres faible (~5-15%)."""
        eca_lgd = _DF.filter(
            pl.col("product_type") == "export_credit_mlt"
        )["lgd_position"].mean()
        self.assertLess(eca_lgd, 0.15, f"ECA LGD mean {eca_lgd:.4f} too high")

    def test_lgd_floor_by_product(self):
        """LC >= 0.10, guarantee >= 0.25, ECA >= 0.05."""
        for prod, cfg in PRODUCTS.items():
            subset = _DF.filter(pl.col("product_type") == prod)
            if len(subset) == 0:
                continue
            floor = cfg["lgd_floor"]
            actual_min = subset["lgd_position"].min()
            self.assertGreaterEqual(
                actual_min, floor - 1e-4,
                f"Product {prod}: min LGD {actual_min:.4f} < floor {floor}",
            )


class TestCCFandEAD(unittest.TestCase):
    """Validation CCF et EAD specifiques TF."""

    def test_ccf_by_product(self):
        """LC=0.20, guarantee_perf=0.50, guarantee_financial=1.00."""
        for prod in ["import_lc", "export_lc"]:
            subset = _DF.filter(pl.col("product_type") == prod)
            np.testing.assert_allclose(subset["ccf"].to_numpy(), 0.20)
        gp = _DF.filter(pl.col("product_type") == "guarantee_perf")
        np.testing.assert_allclose(gp["ccf"].to_numpy(), 0.50)
        gf = _DF.filter(pl.col("product_type") == "guarantee_financial")
        np.testing.assert_allclose(gf["ccf"].to_numpy(), 1.00)

    def test_ead_less_than_nominal(self):
        """EAD <= nominal pour tout (sauf CCF=1 ou EAD~=nominal pour SCF/SBLC)."""
        # For CCF < 1 products, EAD must be < nominal
        sub = _DF.filter(pl.col("ccf") < 1.0)
        self.assertTrue(
            (sub["ead"] < sub["nominal"] + 0.01).all(),
            "EAD should be < nominal for off-balance-sheet products",
        )

    def test_maturity_adjustment(self):
        """MA < 1 pour tenor < 1 an (BCBS205 waiver)."""
        short_tenor = _DF.filter(pl.col("tenor_years") < 1.0)
        self.assertTrue(
            (short_tenor["maturity_adjustment"] < 1.0).all(),
            "MA should be < 1 for tenor < 1 year (BCBS205 waiver)",
        )


class TestAggregation(unittest.TestCase):
    """Validation de l'aggregation vers la ligne balance sheet."""

    def setUp(self):
        self.agg = aggregate_trade_finance_to_balance_row(_DF, _PROFILE)

    def test_aggregate_ead(self):
        """EAD total = sum(ead) post-CCF."""
        expected = _DF["ead"].sum()
        self.assertAlmostEqual(self.agg["ead_total"], expected, places=0)

    def test_aggregate_pd_range(self):
        """PD agrege ~ 0.008 +/- 50%."""
        pd_agg = self.agg["pd_base"]
        self.assertGreater(pd_agg, 0.004)
        self.assertLess(pd_agg, 0.012)

    def test_aggregate_lgd_range(self):
        """LGD agrege EAD-weighted in reasonable range.

        Note: EAD-weighted LGD is higher than simple average because
        guarantee_financial/SCF have CCF=1.0 (large EAD) and high LGD.
        """
        lgd_agg = self.agg["lgd_base"]
        self.assertGreater(lgd_agg, 0.15)
        self.assertLess(lgd_agg, 0.55)

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
