"""Tests pour le generateur position-par-position Project Finance.

Couvre :
    - Generation : shape, colonnes, distributions, coherence, scaling
    - PD DSCR-driven : range, calibration, phase/DSCR/sponsor/country/revenue ordering
    - LGD bi-modale : range, calibration, phase ordering, floors, bimodal clusters
    - Slotting CRR3 : categories, RW values, DSCR coherence, EL ordering
    - Aggregation : EAD, PD/LGD/RW blended
    - Stress test (Approche B) : rate hike, GDP shock, PPP immunity, inflation,
      slotting migration, LGD downturn, RW increase
"""

import unittest
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from ifrs9_cockpit.synthetic_generator.project_finance_positions import (
    PROJECT_TYPES,
    PHASES,
    SLOTTING_CATEGORIES,
    REVENUE_STRUCTURES,
    SPONSOR_RATINGS,
    COUNTRY_BUCKETS,
    generate_project_finance_positions,
    compute_project_finance_pd,
    compute_project_finance_lgd,
    assign_slotting_category,
    stress_project_finance_positions,
    aggregate_project_finance_to_balance_row,
)
from ifrs9_cockpit.config import ASSET_CLASS_MAP


# Shared fixture: generate once for the module
_N = 500
_TOTAL_EAD = 1.0e9
_SEED = 742
_DF = generate_project_finance_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
_PROFILE = ASSET_CLASS_MAP["project_finance"]


class TestProjectFinanceGeneration(unittest.TestCase):
    """Validation du generateur de positions PF."""

    def test_shape(self):
        """n_positions lignes generees."""
        self.assertEqual(len(_DF), _N)

    def test_columns_present(self):
        """Toutes les colonnes requises sont presentes."""
        required = {
            "project_id", "project_type", "phase", "country_bucket",
            "sponsor_rating", "revenue_structure", "tenor_years",
            "commitment", "drawn_amount", "ead", "dscr", "llcr",
            "gearing", "epc_fixed_price", "completion_pct",
            "cost_overrun_pct", "dsra_months", "is_green",
            "carbon_intensity", "taxonomy_aligned",
            "slotting_category", "slotting_rw", "slotting_el",
            "pd_position", "lgd_position", "default_flag",
        }
        self.assertTrue(required <= set(_DF.columns),
                        f"Missing: {required - set(_DF.columns)}")

    def test_ead_scaling(self):
        """sum(ead) ~ total_ead."""
        actual = _DF["ead"].sum()
        self.assertAlmostEqual(actual / _TOTAL_EAD, 1.0, places=2)

    def test_project_type_distribution(self):
        """8 types de projets, renewable dominant."""
        types_present = set(_DF["project_type"].unique().to_list())
        self.assertEqual(types_present, set(PROJECT_TYPES.keys()))
        counts = _DF["project_type"].value_counts().sort("count", descending=True)
        # renewable_solar has highest weight (0.20)
        self.assertEqual(counts["project_type"][0], "renewable_solar")

    def test_phase_distribution(self):
        """Operational dominant (~70%)."""
        total = len(_DF)
        phase_counts = _DF["phase"].value_counts()
        oper_row = phase_counts.filter(pl.col("phase") == "operational")
        constr_row = phase_counts.filter(pl.col("phase") == "construction")
        oper_pct = oper_row["count"][0] / total if len(oper_row) > 0 else 0
        constr_pct = constr_row["count"][0] / total if len(constr_row) > 0 else 0
        self.assertGreater(oper_pct, 0.55)
        self.assertLess(constr_pct, 0.30)

    def test_tenor_range(self):
        """Tenor dans [5, 30] ans."""
        self.assertGreaterEqual(_DF["tenor_years"].min(), 5.0 - 0.1)
        self.assertLessEqual(_DF["tenor_years"].max(), 30.0 + 0.1)

    def test_tenor_type_coherence(self):
        """transport/ppp tenor > telecom/extractive en moyenne."""
        long_tenor = _DF.filter(
            pl.col("project_type").is_in(["transport_toll", "social_ppp"])
        )["tenor_years"].mean()
        short_tenor = _DF.filter(
            pl.col("project_type").is_in(["telecom_infra", "extractive"])
        )["tenor_years"].mean()
        self.assertGreater(long_tenor, short_tenor)

    def test_green_ratio(self):
        """~35-65% is_green (renewables + partial PPP)."""
        green_pct = _DF["is_green"].mean()
        self.assertGreater(green_pct, 0.25)
        self.assertLess(green_pct, 0.70)

    def test_revenue_structure_coherence(self):
        """social_ppp → availability dominant, renewable → long_ppa dominant."""
        ppp = _DF.filter(pl.col("project_type") == "social_ppp")
        avail_pct = (ppp["revenue_structure"] == "availability_ppp").mean()
        self.assertGreater(avail_pct, 0.70, "PPP should have >70% availability")

        renewable = _DF.filter(
            pl.col("project_type").is_in(["renewable_solar", "renewable_wind"])
        )
        ppa_pct = renewable["revenue_structure"].is_in(
            ["long_term_ppa", "medium_term_ppa"]
        ).mean()
        self.assertGreater(ppa_pct, 0.50, "Renewables should have >50% PPA")

    def test_reproducibility(self):
        """Same seed = same result."""
        df2 = generate_project_finance_positions(
            n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED,
        )
        assert_frame_equal(_DF, df2)


class TestProjectFinancePD(unittest.TestCase):
    """Validation du modele PD DSCR-driven."""

    def test_pd_range(self):
        """PD dans [0.0005, 0.15]."""
        self.assertGreaterEqual(_DF["pd_position"].min(), 0.0005 - 1e-6)
        self.assertLessEqual(_DF["pd_position"].max(), 0.15 + 1e-6)

    def test_pd_mean_calibrated(self):
        """PD moyenne ~ 0.005-0.015 (Moody's/S&P, bien < corporate)."""
        pd_mean = _DF["pd_position"].mean()
        self.assertGreater(pd_mean, 0.003, f"PD mean {pd_mean:.5f} too low")
        self.assertLess(pd_mean, 0.025, f"PD mean {pd_mean:.5f} too high")

    def test_pd_phase_ordering(self):
        """PD(construction) > PD(ramp_up) > PD(operational)."""
        means = _DF.group_by("phase").agg(pl.col("pd_position").mean())
        means_dict = dict(zip(
            means["phase"].to_list(),
            means["pd_position"].to_list(),
        ))
        if "construction" in means_dict and "ramp_up" in means_dict:
            self.assertGreater(means_dict["construction"], means_dict["ramp_up"])
        if "ramp_up" in means_dict and "operational" in means_dict:
            self.assertGreater(means_dict["ramp_up"], means_dict["operational"])

    def test_pd_dscr_effect(self):
        """DSCR eleve → PD faible (median split, operational only)."""
        oper = _DF.filter(pl.col("phase") == "operational")
        median_dscr = oper["dscr"].median()
        pd_low_dscr = oper.filter(pl.col("dscr") < median_dscr)["pd_position"].mean()
        pd_high_dscr = oper.filter(pl.col("dscr") >= median_dscr)["pd_position"].mean()
        self.assertGreater(pd_low_dscr, pd_high_dscr)

    def test_pd_sponsor_effect(self):
        """IG sponsors → PD faible vs HY."""
        ig = _DF.filter(pl.col("sponsor_rating").is_in(["IG_strong", "IG_weak"]))
        hy = _DF.filter(pl.col("sponsor_rating").is_in(["HY_weak", "unrated"]))
        self.assertLess(ig["pd_position"].mean(), hy["pd_position"].mean())

    def test_pd_country_effect(self):
        """frontier → PD elevee vs OECD."""
        frontier = _DF.filter(pl.col("country_bucket") == "frontier")
        oecd = _DF.filter(pl.col("country_bucket") == "OECD_core")
        if len(frontier) > 5 and len(oecd) > 5:
            self.assertGreater(frontier["pd_position"].mean(), oecd["pd_position"].mean())

    def test_pd_green_vs_brown(self):
        """Renewable PD < extractive PD."""
        renewable = _DF.filter(
            pl.col("project_type").is_in(["renewable_solar", "renewable_wind"])
        )
        brown = _DF.filter(pl.col("project_type") == "extractive")
        self.assertLess(renewable["pd_position"].mean(), brown["pd_position"].mean())

    def test_pd_revenue_structure(self):
        """availability < long_ppa < merchant (operational only)."""
        oper = _DF.filter(pl.col("phase") == "operational")
        means = oper.group_by("revenue_structure").agg(pl.col("pd_position").mean())
        means_dict = dict(zip(
            means["revenue_structure"].to_list(),
            means["pd_position"].to_list(),
        ))
        if "availability_ppp" in means_dict and "long_term_ppa" in means_dict:
            self.assertLess(means_dict["availability_ppp"], means_dict["long_term_ppa"])
        if "long_term_ppa" in means_dict and "merchant" in means_dict:
            self.assertLess(means_dict["long_term_ppa"], means_dict["merchant"])


class TestProjectFinanceLGD(unittest.TestCase):
    """Validation du modele LGD bi-modale."""

    def test_lgd_range(self):
        """LGD dans [0.15, 0.75]."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.15 - 1e-4)
        self.assertLessEqual(_DF["lgd_position"].max(), 0.75 + 1e-4)

    def test_lgd_mean_calibrated(self):
        """LGD moyenne ~ 0.20-0.40 (GCD/EDHEC)."""
        lgd_mean = _DF["lgd_position"].mean()
        self.assertGreater(lgd_mean, 0.15, f"LGD mean {lgd_mean:.4f} too low")
        self.assertLess(lgd_mean, 0.50, f"LGD mean {lgd_mean:.4f} too high")

    def test_lgd_phase_ordering(self):
        """LGD(construction) > LGD(operational)."""
        means = _DF.group_by("phase").agg(pl.col("lgd_position").mean())
        means_dict = dict(zip(
            means["phase"].to_list(),
            means["lgd_position"].to_list(),
        ))
        if "construction" in means_dict and "operational" in means_dict:
            self.assertGreater(means_dict["construction"], means_dict["operational"])

    def test_lgd_sponsor_effect(self):
        """IG sponsors → LGD plus faible (sponsor support)."""
        ig = _DF.filter(pl.col("sponsor_rating") == "IG_strong")
        hy = _DF.filter(pl.col("sponsor_rating") == "HY_weak")
        if len(ig) > 5 and len(hy) > 5:
            self.assertLess(ig["lgd_position"].mean(), hy["lgd_position"].mean())

    def test_lgd_floor(self):
        """min LGD >= 0.15."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.15 - 1e-4)

    def test_lgd_bimodal(self):
        """Existence de deux clusters (low ~15-25% et high ~40-75%)."""
        lgd = _DF["lgd_position"].to_numpy()
        low_cluster = lgd[lgd < 0.30]
        high_cluster = lgd[lgd >= 0.30]
        # Both clusters should have meaningful population
        self.assertGreater(len(low_cluster), _N * 0.10,
                           "Low-loss cluster too small")
        self.assertGreater(len(high_cluster), _N * 0.05,
                           "High-loss cluster too small")


class TestSlotting(unittest.TestCase):
    """Validation du slotting CRR3 Art. 153(5)."""

    def test_slotting_categories_present(self):
        """Au moins 3 des 4 categories (hors default)."""
        cats = set(_DF["slotting_category"].unique().to_list())
        non_default = cats - {"default"}
        self.assertGreaterEqual(len(non_default), 3,
                                f"Only {non_default} categories present")

    def test_slotting_rw_values(self):
        """RW values match Basel CRE 33."""
        for cat, cfg in SLOTTING_CATEGORIES.items():
            if cat == "default":
                continue
            subset = _DF.filter(pl.col("slotting_category") == cat)
            if len(subset) == 0:
                continue
            # Most PF has tenor > 2.5y
            long_tenor = subset.filter(pl.col("tenor_years") >= 2.5)
            if len(long_tenor) > 0:
                expected_rw = cfg["rw_gte_2_5y"]
                np.testing.assert_allclose(
                    long_tenor["slotting_rw"].to_numpy(), expected_rw,
                    err_msg=f"RW mismatch for {cat} (long tenor)",
                )

    def test_slotting_dscr_coherence(self):
        """DSCR >= 1.50 + operational + IG + OECD + contracted → strong."""
        strong = _DF.filter(pl.col("slotting_category") == "strong")
        if len(strong) > 0:
            self.assertTrue((strong["dscr"] >= 1.50).all(),
                            "Strong should have DSCR >= 1.50")
            self.assertTrue((strong["phase"] == "operational").all(),
                            "Strong should be operational")

    def test_slotting_construction_weak(self):
        """Construction + HY/unrated → weak."""
        constr_hy = _DF.filter(
            (pl.col("phase") == "construction") &
            (pl.col("sponsor_rating").is_in(["HY_weak", "unrated"]))
        )
        if len(constr_hy) > 0:
            self.assertTrue(
                (constr_hy["slotting_category"] == "weak").all(),
                "Construction + HY/unrated should be weak",
            )

    def test_slotting_el_ordering(self):
        """EL(strong) < EL(good) < EL(satisfactory) < EL(weak)."""
        means = _DF.group_by("slotting_category").agg(pl.col("slotting_el").mean())
        means_dict = dict(zip(
            means["slotting_category"].to_list(),
            means["slotting_el"].to_list(),
        ))
        cats_present = [c for c in ["strong", "good", "satisfactory", "weak"]
                        if c in means_dict]
        for i in range(len(cats_present) - 1):
            self.assertLess(
                means_dict[cats_present[i]], means_dict[cats_present[i + 1]],
                f"EL({cats_present[i]}) should be < EL({cats_present[i + 1]})",
            )


class TestAggregation(unittest.TestCase):
    """Validation de l'aggregation vers la ligne balance sheet."""

    def setUp(self):
        self.agg = aggregate_project_finance_to_balance_row(_DF, _PROFILE)

    def test_aggregate_ead(self):
        """EAD total = sum(ead)."""
        expected = _DF["ead"].sum()
        self.assertAlmostEqual(self.agg["ead_total"], expected, places=0)

    def test_aggregate_pd_range(self):
        """PD agrege in reasonable range."""
        pd_agg = self.agg["pd_base"]
        self.assertGreater(pd_agg, 0.002)
        self.assertLess(pd_agg, 0.020)

    def test_aggregate_lgd_range(self):
        """LGD agrege EAD-weighted in reasonable range."""
        lgd_agg = self.agg["lgd_base"]
        self.assertGreater(lgd_agg, 0.15)
        self.assertLess(lgd_agg, 0.45)

    def test_aggregate_rw_blended(self):
        """rw_crr3 blende between min and max slotting RW."""
        rw_agg = self.agg["rw_crr3"]
        self.assertGreater(rw_agg, 0.50)
        self.assertLess(rw_agg, 2.50)

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


class TestStress(unittest.TestCase):
    """Validation du stress test position-par-position (Approche B)."""

    def _stress(self, **kwargs):
        """Helper: stress with overrides on base scenario."""
        base = {
            "gdp_growth": 1.2, "unemployment_rate": 7.5,
            "interest_rate": 3.5, "inflation_rate": 2.5, "spread": 200.0,
        }
        base.update(kwargs)
        return stress_project_finance_positions(_DF, base)

    def test_stress_base_unchanged(self):
        """Base scenario → PD/LGD close to unstressed aggregate."""
        result = self._stress()
        agg = aggregate_project_finance_to_balance_row(_DF, _PROFILE)
        # Should be very close (not exact due to DSCR recalculation)
        self.assertAlmostEqual(result["pd_base"], agg["pd_base"], places=4)

    def test_stress_rate_hike_dscr_drops(self):
        """+200bp rate hike → PD augmente."""
        base = self._stress()
        stressed = self._stress(interest_rate=5.5)  # +200bp
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "Rate hike should increase PD")

    def test_stress_gdp_shock_increases_pd(self):
        """GDP -5% → PD augmente (toll roads, merchant exposed)."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-3.8)  # severe recession
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "GDP shock should increase PD")

    def test_stress_ppp_immune_to_gdp(self):
        """GDP shock → PPP/PPA positions quasi-inchangees."""
        ppp_df = _DF.filter(pl.col("revenue_structure") == "availability_ppp")
        if len(ppp_df) < 5:
            self.skipTest("Not enough PPP positions")
        base = stress_project_finance_positions(ppp_df, {
            "gdp_growth": 1.2, "interest_rate": 3.5,
            "inflation_rate": 2.5, "spread": 200.0,
        })
        stressed = stress_project_finance_positions(ppp_df, {
            "gdp_growth": -3.8, "interest_rate": 3.5,
            "inflation_rate": 2.5, "spread": 200.0,
        })
        # PPP PD should change less than 20% relative
        if base["pd_base"] > 0:
            pct_change = abs(stressed["pd_base"] - base["pd_base"]) / base["pd_base"]
            self.assertLess(pct_change, 0.20,
                            f"PPP PD changed by {pct_change:.1%} under GDP shock")

    def test_stress_inflation_drag(self):
        """+5% inflation → PD augmente (merchant non-indexe)."""
        base = self._stress()
        stressed = self._stress(inflation_rate=7.5)  # +5% inflation
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "Inflation should increase PD")

    def test_stress_slotting_migration(self):
        """Stress severe → RW agrege augmente (re-slotting)."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-3.8, interest_rate=5.5)
        self.assertGreater(stressed["rw_crr3"], base["rw_crr3"],
                           "Stress should increase blended RW")

    def test_stress_lgd_downturn(self):
        """GDP negatif → LGD augmente."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-3.8)
        self.assertGreater(stressed["lgd_base"], base["lgd_base"],
                           "GDP shock should increase LGD (downturn addon)")


if __name__ == "__main__":
    unittest.main()
