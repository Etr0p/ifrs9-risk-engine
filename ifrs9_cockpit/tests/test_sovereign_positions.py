"""Tests pour le generateur position-par-position Obligations Souveraines.

Couvre :
    - Generation : shape, colonnes, EAD scaling, distributions emetteurs/maturite/coupon
    - PD composite : range, calibration, ordering par rating, fiscal, marche
    - LGD Cruces-Trebesch : range, calibration, IG vs HY, maturite
    - RW : toujours 0% (CRR3 Art. 114(4))
    - Aggregation : EAD, PD/LGD/RW blended
    - Stress : 3 canaux, amplification peripherique, flight-to-quality
    - HQLA : Level 1, exempt staging, RSF 0%
"""

import pytest

pytestmark = pytest.mark.fourteen

import unittest
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from ifrs9_cockpit.synthetic_generator.sovereign_positions import (
    SOVEREIGN_ISSUERS,
    MATURITY_BUCKETS,
    COUPON_TYPES,
    SPREAD_AMPLIFICATION,
    RATING_PD_MAP,
    generate_sovereign_positions,
    compute_sovereign_pd,
    compute_sovereign_lgd,
    stress_sovereign_positions,
    aggregate_sovereign_to_balance_row,
)
from ifrs9_cockpit.config import ASSET_CLASS_MAP


# Shared fixture: generate once for the module
_N = 180
_TOTAL_EAD = 6.0e9  # 6B EUR
_SEED = 942
_DF = generate_sovereign_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
_PROFILE = ASSET_CLASS_MAP["sovereign"]


class TestSovereignGeneration(unittest.TestCase):
    """Validation du generateur de positions souveraines."""

    def test_shape(self):
        """n_positions lignes generees."""
        self.assertEqual(len(_DF), _N)

    def test_columns_present(self):
        """Toutes les colonnes requises sont presentes."""
        required = {
            "position_id", "country_code", "country", "bond_label",
            "rating", "tenor_years", "maturity_bucket", "coupon_type",
            "coupon_rate", "notional", "mod_duration", "debt_to_gdp",
            "cds_spread_bp", "fiscal_balance", "recovery_rate_hist",
            "spread_amplification", "rw_crr3", "hqla_eligible",
            "hqla_level", "exempt_from_staging", "ead",
            "pd_position", "lgd_position", "yield_to_maturity",
        }
        self.assertTrue(required <= set(_DF.columns),
                        f"Missing: {required - set(_DF.columns)}")

    def test_ead_scaling(self):
        """sum(ead) ~ total_ead."""
        actual = _DF["ead"].sum()
        self.assertAlmostEqual(actual / _TOTAL_EAD, 1.0, places=2)

    def test_issuer_distribution(self):
        """9 pays zone euro, DE en tete."""
        countries = set(_DF["country_code"].unique().to_list())
        expected = set(SOVEREIGN_ISSUERS.keys())
        self.assertEqual(countries, expected)
        counts = _DF["country_code"].value_counts().sort("count", descending=True)
        # DE has highest weight (0.25)
        self.assertEqual(counts["country_code"][0], "DE")

    def test_maturity_distribution(self):
        """Medium dominant (~40%)."""
        vc = _DF["maturity_bucket"].value_counts()
        total = vc["count"].sum()
        vc = vc.with_columns((pl.col("count") / total).alias("proportion"))
        vc_dict = dict(zip(vc["maturity_bucket"].to_list(), vc["proportion"].to_list()))
        self.assertGreater(vc_dict.get("medium", 0), 0.25)

    def test_coupon_type_distribution(self):
        """Fixed dominant (~80%)."""
        vc = _DF["coupon_type"].value_counts()
        total = vc["count"].sum()
        vc = vc.with_columns((pl.col("count") / total).alias("proportion"))
        vc_dict = dict(zip(vc["coupon_type"].to_list(), vc["proportion"].to_list()))
        self.assertGreater(vc_dict.get("fixed", 0), 0.55)

    def test_tenor_range(self):
        """Tenor dans [0.5, 30]."""
        self.assertGreaterEqual(_DF["tenor_years"].min(), 0.5)
        self.assertLessEqual(_DF["tenor_years"].max(), 30.0)

    def test_duration_positive(self):
        """Modified duration > 0."""
        self.assertTrue((_DF["mod_duration"] > 0).all())

    def test_yield_positive(self):
        """Yield > 0."""
        self.assertTrue((_DF["yield_to_maturity"] > 0).all())

    def test_rw_all_zero(self):
        """RW = 0% pour toutes les positions (CRR3 Art. 114(4))."""
        self.assertTrue((_DF["rw_crr3"] == 0.0).all())

    def test_hqla_all_level1(self):
        """Toutes les positions sont HQLA Level 1."""
        self.assertTrue((_DF["hqla_level"] == 1).all())
        self.assertTrue((_DF["hqla_eligible"]).all())

    def test_reproducibility(self):
        """Meme seed -> meme resultat."""
        df2 = generate_sovereign_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
        assert_frame_equal(_DF, df2)


class TestSovereignPD(unittest.TestCase):
    """Validation de la PD composite souveraine."""

    def test_pd_range(self):
        """PD dans [0.0001, 0.05]."""
        self.assertGreaterEqual(_DF["pd_position"].min(), 0.0001)
        self.assertLessEqual(_DF["pd_position"].max(), 0.05)

    def test_pd_mean_calibrated(self):
        """PD moyenne EAD-weighted entre 0.0003 et 0.010 (zone euro core+periphery)."""
        w = _DF["ead"].to_numpy() / _DF["ead"].sum()
        pd_mean = float(np.dot(w, _DF["pd_position"].to_numpy()))
        self.assertGreater(pd_mean, 0.0003)
        self.assertLess(pd_mean, 0.010)

    def test_pd_rating_ordering(self):
        """BBB PD > A PD > AA PD > AAA PD (median per rating)."""
        pd_by_rating = (
            _DF.group_by("rating")
            .agg(pl.col("pd_position").median().alias("pd_median"))
        )
        rating_dict = dict(zip(
            pd_by_rating["rating"].to_list(),
            pd_by_rating["pd_median"].to_list(),
        ))
        if "AAA" in rating_dict and "AA" in rating_dict:
            self.assertLess(rating_dict["AAA"], rating_dict["AA"])
        if "AA" in rating_dict and "A" in rating_dict:
            self.assertLess(rating_dict["AA"], rating_dict["A"])
        if "A" in rating_dict and "BBB" in rating_dict:
            self.assertLess(rating_dict["A"], rating_dict["BBB"])

    def test_pd_germany_lowest(self):
        """Allemagne a la PD mediane la plus basse."""
        pd_by_country = (
            _DF.group_by("country_code")
            .agg(pl.col("pd_position").median().alias("pd_median"))
        )
        min_row = pd_by_country.sort("pd_median").row(0, named=True)
        self.assertEqual(min_row["country_code"], "DE")

    def test_pd_italy_highest(self):
        """Italie a la PD mediane la plus haute."""
        pd_by_country = (
            _DF.group_by("country_code")
            .agg(pl.col("pd_position").median().alias("pd_median"))
        )
        max_row = pd_by_country.sort("pd_median", descending=True).row(0, named=True)
        self.assertEqual(max_row["country_code"], "IT")

    def test_pd_debt_effect(self):
        """Correlation positive entre debt/GDP et PD."""
        corr = float(_DF.select(pl.corr("debt_to_gdp", "pd_position")).item())
        self.assertGreater(corr, 0.3)

    def test_pd_fiscal_surplus(self):
        """Pays a surplus fiscal (IE, NL) ont des PD basses."""
        surplus_countries = _DF.filter(pl.col("fiscal_balance") > 0)
        if len(surplus_countries) > 0:
            pd_surplus = surplus_countries["pd_position"].median()
            pd_overall = _DF["pd_position"].median()
            self.assertLess(pd_surplus, pd_overall)

    def test_pd_market_signal(self):
        """PD correle positivement au CDS spread."""
        corr = float(_DF.select(pl.corr("cds_spread_bp", "pd_position")).item())
        self.assertGreater(corr, 0.5)


class TestSovereignLGD(unittest.TestCase):
    """Validation de la LGD Cruces-Trebesch."""

    def test_lgd_range(self):
        """LGD dans [0.10, 0.65]."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.10)
        self.assertLessEqual(_DF["lgd_position"].max(), 0.65)

    def test_lgd_mean_calibrated(self):
        """LGD moyenne entre 0.20 et 0.50 (Cruces-Trebesch mean 37%)."""
        lgd_mean = _DF["lgd_position"].mean()
        self.assertGreater(lgd_mean, 0.20)
        self.assertLess(lgd_mean, 0.50)

    def test_lgd_ig_lower_than_hy(self):
        """IG (AAA/AA/A) LGD mediane < HY (BBB/BBB+) LGD mediane."""
        ig_ratings = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-"]
        ig = _DF.filter(pl.col("rating").is_in(ig_ratings))
        hy = _DF.filter(~pl.col("rating").is_in(ig_ratings))
        if len(ig) > 5 and len(hy) > 5:
            lgd_ig = ig["lgd_position"].median()
            lgd_hy = hy["lgd_position"].median()
            self.assertLess(lgd_ig, lgd_hy)

    def test_lgd_short_maturity_lower(self):
        """Obligations < 5y ont LGD mediane plus basse (recovery premium)."""
        short = _DF.filter(pl.col("tenor_years") < 5.0)
        long = _DF.filter(pl.col("tenor_years") >= 5.0)
        if len(short) > 10 and len(long) > 10:
            lgd_short = short["lgd_position"].median()
            lgd_long = long["lgd_position"].median()
            self.assertLess(lgd_short, lgd_long)

    def test_lgd_floor(self):
        """LGD >= 10% (floor Cruces-Trebesch)."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.10)

    def test_lgd_country_effect(self):
        """Italie (recovery_hist 65%) a LGD plus haute que Allemagne (95%)."""
        lgd_de = _DF.filter(pl.col("country_code") == "DE")["lgd_position"].median()
        lgd_it = _DF.filter(pl.col("country_code") == "IT")["lgd_position"].median()
        self.assertLess(lgd_de, lgd_it)


class TestSovereignRW(unittest.TestCase):
    """Validation du RW = 0% (CRR3 Art. 114(4))."""

    def test_rw_always_zero(self):
        """RW = 0% pour toutes les positions."""
        self.assertTrue((_DF["rw_crr3"] == 0.0).all())

    def test_rw_aggregate_zero(self):
        """RW agrege = 0%."""
        row = aggregate_sovereign_to_balance_row(_DF, _PROFILE)
        self.assertEqual(row["rw_crr3"], 0.0)


class TestAggregation(unittest.TestCase):
    """Validation de l'aggregation balance sheet."""

    def test_aggregate_ead(self):
        """EAD total agrege ~ total_ead."""
        row = aggregate_sovereign_to_balance_row(_DF, _PROFILE)
        self.assertAlmostEqual(row["ead_total"] / _TOTAL_EAD, 1.0, places=2)

    def test_aggregate_pd_range(self):
        """PD agregee dans une fourchette raisonnable."""
        row = aggregate_sovereign_to_balance_row(_DF, _PROFILE)
        self.assertGreater(row["pd_base"], 0.0001)
        self.assertLess(row["pd_base"], 0.01)

    def test_aggregate_lgd_range(self):
        """LGD agregee dans une fourchette raisonnable."""
        row = aggregate_sovereign_to_balance_row(_DF, _PROFILE)
        self.assertGreater(row["lgd_base"], 0.15)
        self.assertLess(row["lgd_base"], 0.55)

    def test_aggregate_rw_zero(self):
        """RW agrege = 0%."""
        row = aggregate_sovereign_to_balance_row(_DF, _PROFILE)
        self.assertEqual(row["rw_crr3"], 0.0)

    def test_aggregate_schema(self):
        """18 cles presentes dans le dict agrege."""
        row = aggregate_sovereign_to_balance_row(_DF, _PROFILE)
        expected_keys = {
            "asset_class", "label", "category", "ead_total", "typical_weight",
            "pd_base", "lgd_base", "tenor", "asset_correlation", "rw_crr3",
            "physical_risk", "transition_risk", "green_capex_ratio",
            "scope3_exposure", "absorption_buffer", "hqla_eligible",
            "hqla_level", "exempt_from_staging", "rsf_weight",
        }
        self.assertTrue(expected_keys <= set(row.keys()),
                        f"Missing: {expected_keys - set(row.keys())}")


class TestStress(unittest.TestCase):
    """Validation du stress test 3 canaux + amplification peripherique."""

    def _stress(self, **overrides):
        """Helper: stress with base scenario + overrides."""
        base = {
            "gdp_growth": 1.2,
            "unemployment_rate": 7.0,
            "interest_rate": 3.5,
            "hpi_growth": 2.0,
            "inflation_rate": 2.5,
        }
        base.update(overrides)
        return stress_sovereign_positions(_DF, base)

    def test_base_unchanged(self):
        """Scenario base -> PD ~ PD non-stresse."""
        result = self._stress()
        row = aggregate_sovereign_to_balance_row(_DF, _PROFILE)
        # Should be very close to unstressed
        self.assertAlmostEqual(result["pd_base"], row["pd_base"], places=5)

    def test_rate_hike_increases_pd(self):
        """Hausse taux -> PD augmente (spread widening)."""
        base = self._stress()
        stressed = self._stress(interest_rate=6.5)  # +300bp
        self.assertGreater(stressed["pd_base"], base["pd_base"])

    def test_gdp_shock_increases_pd(self):
        """Choc GDP negatif -> PD augmente (fiscal deterioration)."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-3.8)
        self.assertGreater(stressed["pd_base"], base["pd_base"])

    def test_peripheral_amplification(self):
        """PD Italie stresse >> PD Allemagne stresse (amplification peripherique)."""
        macro_stress = {
            "gdp_growth": -3.8,
            "unemployment_rate": 12.0,
            "interest_rate": 5.5,
            "hpi_growth": -4.0,
            "inflation_rate": 4.0,
        }
        # Separate stress for IT and DE positions
        df_it = _DF.filter(pl.col("country_code") == "IT")
        df_de = _DF.filter(pl.col("country_code") == "DE")

        if len(df_it) > 3 and len(df_de) > 3:
            stress_it = stress_sovereign_positions(df_it, macro_stress)
            stress_de = stress_sovereign_positions(df_de, macro_stress)
            # IT should be stressed much more (amplification = 2.0 vs 0.3)
            ratio = stress_it["pd_base"] / max(stress_de["pd_base"], 1e-8)
            self.assertGreater(ratio, 3.0,
                               f"IT/DE PD ratio should be > 3x, got {ratio:.1f}")

    def test_safe_haven_germany(self):
        """Allemagne PD bouge peu sous stress (flight to quality, amp=0.3)."""
        df_de = _DF.filter(pl.col("country_code") == "DE")
        if len(df_de) > 3:
            base = stress_sovereign_positions(df_de, {
                "gdp_growth": 1.2, "interest_rate": 3.5, "inflation_rate": 2.5,
            })
            stressed = stress_sovereign_positions(df_de, {
                "gdp_growth": -2.0, "interest_rate": 5.5, "inflation_rate": 4.0,
            })
            ratio = stressed["pd_base"] / max(base["pd_base"], 1e-8)
            # DE PD should increase less than 3x under moderate stress
            self.assertLess(ratio, 3.0)

    def test_lgd_downturn(self):
        """GDP negatif -> LGD augmente."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-5.0)
        self.assertGreaterEqual(stressed["lgd_base"], base["lgd_base"])

    def test_rw_stays_zero(self):
        """RW reste 0% sous tous les scenarios (CRR3 Art. 114(4))."""
        # Severe combined stress
        result = self._stress(
            gdp_growth=-5.0, interest_rate=7.0, inflation_rate=6.0,
        )
        self.assertEqual(result["rw_crr3"], 0.0)

    def test_severe_combined(self):
        """Stress severe combine -> PD significativement plus haute."""
        base = self._stress()
        severe = self._stress(
            gdp_growth=-5.0, interest_rate=7.0, inflation_rate=6.0,
        )
        self.assertGreater(severe["pd_base"], base["pd_base"] * 1.5,
                           "Severe stress should increase PD by at least 50%")

    def test_positive_gdp_reduces_pd(self):
        """GDP positif -> PD diminue ou stable."""
        base = self._stress()
        boom = self._stress(gdp_growth=4.0)
        self.assertLessEqual(boom["pd_base"], base["pd_base"] * 1.01)

    def test_rate_cut_reduces_pd(self):
        """Baisse taux -> PD diminue (spread compression)."""
        base = self._stress()
        eased = self._stress(interest_rate=1.5)  # -200bp
        self.assertLessEqual(eased["pd_base"], base["pd_base"])

    def test_contagion_relevance(self):
        """PD stresse > 2x base en severe (pertinent pour contagion souverain->bancaire)."""
        base = self._stress()
        severe = self._stress(
            gdp_growth=-5.0, interest_rate=7.0, inflation_rate=6.0,
        )
        ratio = severe["pd_base"] / max(base["pd_base"], 1e-8)
        self.assertGreater(ratio, 2.0,
                           f"Severe stress should at least double PD, got ratio={ratio:.1f}")


class TestHQLAAndStaging(unittest.TestCase):
    """Validation HQLA Level 1 et exemption staging."""

    def test_all_hqla_level1(self):
        """Toutes les positions sont HQLA Level 1."""
        self.assertTrue((_DF["hqla_level"] == 1).all())
        self.assertTrue((_DF["hqla_eligible"]).all())

    def test_exempt_from_staging(self):
        """Toutes les positions exemptees de staging (IFRS 9 B5.5.25)."""
        self.assertTrue((_DF["exempt_from_staging"]).all())

    def test_rsf_weight_zero(self):
        """RSF weight = 0% (HQLA L1, NSFR Basel III)."""
        row = aggregate_sovereign_to_balance_row(_DF, _PROFILE)
        self.assertEqual(row["rsf_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
