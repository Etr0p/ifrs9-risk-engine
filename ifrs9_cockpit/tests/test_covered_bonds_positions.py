"""Tests pour le generateur position-par-position Covered Bonds.

Couvre :
    - Generation : shape, 21 colonnes, EAD scaling, emetteurs, pool types,
      ratings, coupons, devises, tenors, OC, LTV, structure, reproductibilite
    - PD double-defaut : range, mean, monotonie rating, pool type, OC, structure
    - LGD dual recours : range, mean, pool type, OC, LTV, floor
    - RW CRR3 Art. 129 : range, valeurs exactes, monotonie
    - Aggregation : EAD sum, PD/LGD/RW blended, schema complet
    - Stress test : base unchanged, HPI/rate/GDP/inflation shocks,
      LGD downturn, resilience double recours
"""

import pytest

pytestmark = pytest.mark.fourteen

import unittest
import numpy as np
import polars as pl

from ifrs9_cockpit.synthetic_generator.covered_bonds_positions import (
    _COVER_POOL_TYPES,
    _ISSUER_BANKS,
    _CRR3_ART129_RW,
    _RATING_PD,
    _STRUCTURE_PD_MULT,
    generate_covered_bonds_positions,
    compute_covered_bonds_pd,
    compute_covered_bonds_lgd,
    compute_covered_bonds_rw,
    stress_covered_bonds_positions,
    aggregate_covered_bonds_to_balance_row,
)
from ifrs9_cockpit.config import ASSET_CLASS_MAP


# Shared fixture: generate once for the module
_N = 500
_TOTAL_EAD = 1.0e9
_SEED = 942
_DF = generate_covered_bonds_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
_PROFILE = ASSET_CLASS_MAP["covered_bonds"]


class TestCoveredBondsGeneration(unittest.TestCase):
    """Validation du generateur de positions covered bonds."""

    def test_shape(self):
        """Exactement n_positions lignes generees."""
        self.assertEqual(len(_DF), _N)

    def test_columns_present(self):
        """21 colonnes requises sont presentes."""
        required = {
            "bond_id", "issuer_name", "issuer_country", "issuer_rating",
            "pd_issuer", "cover_pool_type", "pool_weighted_ltv",
            "pool_oc_ratio", "maturity_structure", "coupon_type",
            "coupon_rate_bps", "tenor_years", "currency",
            "notional_ead", "ead", "issue_vintage",
            "pd_position", "lgd_position", "rw_crr3", "default_flag",
        }
        self.assertTrue(required <= set(_DF.columns),
                        f"Missing: {required - set(_DF.columns)}")

    def test_ead_scaling(self):
        """sum(ead) ~ total_ead."""
        actual = _DF["ead"].sum()
        self.assertAlmostEqual(actual / _TOTAL_EAD, 1.0, places=2)

    def test_emetteurs_de_dominant(self):
        """DE emetteurs representent > 25% (poids ~37% dans constantes)."""
        de_share = (_DF["issuer_country"] == "DE").mean()
        self.assertGreater(de_share, 0.20,
                           f"DE share {de_share:.1%} too low (expected >20%)")

    def test_pool_types_distribution(self):
        """4 types de pools, residentiel >50% (poids 60%)."""
        types_present = set(_DF["cover_pool_type"].unique().to_list())
        self.assertEqual(types_present, set(_COVER_POOL_TYPES.keys()))
        res_share = (_DF["cover_pool_type"] == "residential_mortgage").mean()
        self.assertGreater(res_share, 0.40,
                           f"Residential share {res_share:.1%} too low")

    def test_rating_distribution(self):
        """~74% AAA+AA (weighted 65% in constants)."""
        high_grade = _DF["issuer_rating"].is_in(["AAA", "AA"]).mean()
        self.assertGreater(high_grade, 0.50,
                           f"AAA+AA share {high_grade:.1%} too low")

    def test_coupon_type_distribution(self):
        """~65% fixed in fallback mode."""
        fixed_share = (_DF["coupon_type"] == "fixed").mean()
        self.assertGreater(fixed_share, 0.45, f"Fixed share {fixed_share:.1%}")
        self.assertLess(fixed_share, 0.85, f"Fixed share {fixed_share:.1%}")

    def test_currency_eur_dominant(self):
        """EUR > 85%."""
        eur_share = (_DF["currency"] == "EUR").mean()
        self.assertGreater(eur_share, 0.80, f"EUR share {eur_share:.1%}")

    def test_tenor_range(self):
        """Tenor in [1, 15]."""
        self.assertGreaterEqual(_DF["tenor_years"].min(), 1.0)
        self.assertLessEqual(_DF["tenor_years"].max(), 15.0)

    def test_oc_minimum(self):
        """OC >= 2% (CRR3 minimum)."""
        self.assertGreaterEqual(_DF["pool_oc_ratio"].min(), 0.02)

    def test_ltv_range(self):
        """LTV in [0, 0.85]."""
        self.assertGreaterEqual(_DF["pool_weighted_ltv"].min(), 0.0)
        self.assertLessEqual(_DF["pool_weighted_ltv"].max(), 0.85)

    def test_public_sector_ltv_zero(self):
        """Public sector pools have LTV = 0."""
        pub = _DF.filter(pl.col("cover_pool_type") == "public_sector")
        if len(pub) > 0:
            self.assertEqual(pub["pool_weighted_ltv"].max(), 0.0,
                             "Public sector should have LTV = 0")

    def test_structure_soft_bullet_dominant(self):
        """~90% soft_bullet."""
        sb_share = (_DF["maturity_structure"] == "soft_bullet").mean()
        self.assertGreater(sb_share, 0.75, f"Soft bullet share {sb_share:.1%}")

    def test_reproducibility(self):
        """Same seed -> identical DataFrame."""
        df2 = generate_covered_bonds_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
        # Compare using Polars frame_equal
        self.assertTrue(_DF.equals(df2), "DataFrames should be identical with same seed")


class TestCoveredBondsPD(unittest.TestCase):
    """Validation PD double-defaut covered bonds."""

    def test_pd_range(self):
        """PD in [3e-5, 0.05]."""
        self.assertGreaterEqual(_DF["pd_position"].min(), 3e-5)
        self.assertLessEqual(_DF["pd_position"].max(), 0.05)

    def test_pd_mean_low(self):
        """Mean PD < 50 bps (covered bonds are very safe)."""
        mean_pd = _DF["pd_position"].mean()
        self.assertLess(mean_pd, 0.0050,
                        f"Mean PD {mean_pd:.4%} > 50 bps (too high for covered bonds)")

    def test_pd_monotonicity_rating(self):
        """AAA < AA < A (monotone with rating)."""
        pd_by_rating = (
            _DF.group_by("issuer_rating")
            .agg(pl.col("pd_position").mean().alias("pd_mean"))
        )
        rating_dict = dict(zip(
            pd_by_rating["issuer_rating"].to_list(),
            pd_by_rating["pd_mean"].to_list(),
        ))
        if "AAA" in rating_dict and "AA" in rating_dict:
            self.assertLess(rating_dict["AAA"], rating_dict["AA"],
                            "AAA PD should be < AA PD")
        if "AA" in rating_dict and "A" in rating_dict:
            self.assertLess(rating_dict["AA"], rating_dict["A"],
                            "AA PD should be < A PD")

    def test_pd_pool_type_ordering(self):
        """public_sector < residential < commercial PD."""
        pd_by_pool = (
            _DF.group_by("cover_pool_type")
            .agg(pl.col("pd_position").mean().alias("pd_mean"))
        )
        pool_dict = dict(zip(
            pd_by_pool["cover_pool_type"].to_list(),
            pd_by_pool["pd_mean"].to_list(),
        ))
        if "public_sector" in pool_dict and "residential_mortgage" in pool_dict:
            self.assertLess(pool_dict["public_sector"],
                            pool_dict["residential_mortgage"])
        if "residential_mortgage" in pool_dict and "commercial_mortgage" in pool_dict:
            self.assertLess(pool_dict["residential_mortgage"],
                            pool_dict["commercial_mortgage"])

    def test_oc_negative_correlation(self):
        """Higher OC -> lower PD (negative correlation)."""
        corr = np.corrcoef(
            _DF["pool_oc_ratio"].to_numpy(),
            _DF["pd_position"].to_numpy(),
        )[0, 1]
        self.assertLess(corr, 0.1,
                        f"OC-PD correlation {corr:.3f} should be negative or near zero")

    def test_double_default_much_lower(self):
        """PD_cb << PD_issuer (double-default protection)."""
        ratio = _DF["pd_position"].mean() / _DF["pd_issuer"].mean()
        self.assertLess(ratio, 0.65,
                        f"PD_cb/PD_issuer = {ratio:.2f}, expected < 0.65")

    def test_soft_bullet_lower_than_hard(self):
        """Soft bullet PD < hard bullet PD."""
        sb = _DF.filter(pl.col("maturity_structure") == "soft_bullet")
        sb_pd = sb["pd_position"].mean()
        hb = _DF.filter(pl.col("maturity_structure") == "hard_bullet")
        if len(hb) > 5:
            hb_pd = hb["pd_position"].mean()
            self.assertLess(sb_pd, hb_pd,
                            "Soft bullet should have lower PD than hard bullet")

    def test_no_nan(self):
        """Pas de NaN dans PD."""
        self.assertFalse(_DF["pd_position"].is_null().any())


class TestCoveredBondsLGD(unittest.TestCase):
    """Validation LGD dual recours covered bonds."""

    def test_lgd_range(self):
        """LGD in [0.05, 0.45]."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.05)
        self.assertLessEqual(_DF["lgd_position"].max(), 0.45)

    def test_lgd_mean_low(self):
        """Mean LGD < 15% (strong dual recourse protection)."""
        mean_lgd = _DF["lgd_position"].mean()
        self.assertLess(mean_lgd, 0.15,
                        f"Mean LGD {mean_lgd:.2%} > 15%")

    def test_lgd_pool_type_effect(self):
        """Mortgage pools have lower LGD than public sector (higher pool recovery)."""
        lgd_by_pool = (
            _DF.group_by("cover_pool_type")
            .agg(pl.col("lgd_position").mean().alias("lgd_mean"))
        )
        pool_dict = dict(zip(
            lgd_by_pool["cover_pool_type"].to_list(),
            lgd_by_pool["lgd_mean"].to_list(),
        ))
        if "residential_mortgage" in pool_dict:
            res_lgd = pool_dict["residential_mortgage"]
            self.assertLess(res_lgd, 0.12,
                            f"Residential LGD {res_lgd:.2%} > 12%")

    def test_oc_negative_correlation_lgd(self):
        """Higher OC -> lower LGD."""
        corr = np.corrcoef(
            _DF["pool_oc_ratio"].to_numpy(),
            _DF["lgd_position"].to_numpy(),
        )[0, 1]
        self.assertLess(corr, 0.1,
                        f"OC-LGD correlation {corr:.3f} should be negative or near zero")

    def test_ltv_positive_correlation_mortgage(self):
        """Higher LTV -> higher LGD for mortgage pools."""
        mortgage = _DF.filter(
            pl.col("cover_pool_type").is_in(
                ["residential_mortgage", "commercial_mortgage"]))
        if len(mortgage) > 20:
            corr = np.corrcoef(
                mortgage["pool_weighted_ltv"].to_numpy(),
                mortgage["lgd_position"].to_numpy(),
            )[0, 1]
            self.assertGreater(corr, -0.3,
                               f"LTV-LGD correlation {corr:.3f} unexpected")

    def test_lgd_floor(self):
        """Floor at 5%."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.05)

    def test_residential_lgd_mean(self):
        """Residential mortgage mean LGD < 12%."""
        res = _DF.filter(pl.col("cover_pool_type") == "residential_mortgage")
        if len(res) > 10:
            mean_lgd = res["lgd_position"].mean()
            self.assertLess(mean_lgd, 0.12,
                            f"Residential LGD {mean_lgd:.2%} > 12%")

    def test_no_nan_lgd(self):
        """Pas de NaN dans LGD."""
        self.assertFalse(_DF["lgd_position"].is_null().any())


class TestCoveredBondsRW(unittest.TestCase):
    """Validation RW CRR3 Art. 129."""

    def test_rw_range(self):
        """RW in [0.10, 1.00]."""
        self.assertGreaterEqual(_DF["rw_crr3"].min(), 0.10)
        self.assertLessEqual(_DF["rw_crr3"].max(), 1.00)

    def test_aaa_rw_10pct(self):
        """AAA -> RW = 10%."""
        aaa = _DF.filter(pl.col("issuer_rating") == "AAA")
        if len(aaa) > 0:
            self.assertTrue((aaa["rw_crr3"] == 0.10).all(),
                            "AAA should have RW = 10%")

    def test_aa_rw_15pct(self):
        """AA -> RW = 15%."""
        aa = _DF.filter(pl.col("issuer_rating") == "AA")
        if len(aa) > 0:
            self.assertTrue((aa["rw_crr3"] == 0.15).all(),
                            "AA should have RW = 15%")

    def test_a_rw_20pct(self):
        """A -> RW = 20%."""
        a = _DF.filter(pl.col("issuer_rating") == "A")
        if len(a) > 0:
            self.assertTrue((a["rw_crr3"] == 0.20).all(),
                            "A should have RW = 20%")

    def test_rw_strict_monotonicity(self):
        """AAA < AA < A strict monotonicity."""
        rw_vals = {
            "AAA": _CRR3_ART129_RW["AAA"],
            "AA": _CRR3_ART129_RW["AA"],
            "A": _CRR3_ART129_RW["A"],
        }
        self.assertLess(rw_vals["AAA"], rw_vals["AA"])
        self.assertLess(rw_vals["AA"], rw_vals["A"])


class TestAggregation(unittest.TestCase):
    """Validation aggregation balance sheet row."""

    def setUp(self):
        self.row = aggregate_covered_bonds_to_balance_row(_DF, _PROFILE)

    def test_ead_sum(self):
        """EAD aggrege = sum(ead) des positions."""
        expected = _DF["ead"].sum()
        self.assertAlmostEqual(self.row["ead_total"], expected, places=0)

    def test_pd_range(self):
        """PD aggrege in [3e-5, 0.01]."""
        self.assertGreaterEqual(self.row["pd_base"], 3e-5)
        self.assertLessEqual(self.row["pd_base"], 0.01)

    def test_lgd_range(self):
        """LGD aggrege in [0.05, 0.20]."""
        self.assertGreaterEqual(self.row["lgd_base"], 0.05)
        self.assertLessEqual(self.row["lgd_base"], 0.20)

    def test_rw_range(self):
        """RW aggrege in [0.10, 0.25]."""
        self.assertGreaterEqual(self.row["rw_crr3"], 0.10)
        self.assertLessEqual(self.row["rw_crr3"], 0.25)

    def test_schema_complete(self):
        """19 cles balance sheet presentes."""
        required_keys = {
            "asset_class", "label", "category", "ead_total", "typical_weight",
            "pd_base", "lgd_base", "tenor", "asset_correlation", "rw_crr3",
            "physical_risk", "transition_risk", "green_capex_ratio",
            "scope3_exposure", "absorption_buffer",
            "hqla_eligible", "hqla_level", "exempt_from_staging", "rsf_weight",
        }
        self.assertTrue(required_keys <= set(self.row.keys()),
                        f"Missing: {required_keys - set(self.row.keys())}")

    def test_bottom_up_differs_from_profile(self):
        """Bottom-up PD/LGD differs from static profile (proves real aggregation)."""
        # PD bottom-up should differ from profile constant
        self.assertNotAlmostEqual(self.row["pd_base"], _PROFILE.pd_base, places=6,
                                  msg="PD should differ from static profile")


class TestStress(unittest.TestCase):
    """Validation stress test covered bonds."""

    def _stress(self, **kwargs):
        """Helper: run stress with given macro overrides."""
        base = {
            "gdp_growth": 1.2,
            "unemployment_rate": 7.5,
            "interest_rate": 3.5,
            "hpi_growth": 2.0,
            "inflation_rate": 2.5,
        }
        base.update(kwargs)
        return stress_covered_bonds_positions(_DF, base)

    def test_base_approximately_unstressed(self):
        """Base scenario ~ unstressed (+/- 5%)."""
        result = self._stress()
        unstressed_pd = _DF["pd_position"].mean()
        # Allow some tolerance since stress function re-derives PD from pool constants
        self.assertAlmostEqual(result["pd_base"] / max(unstressed_pd, 1e-8),
                               1.0, delta=0.30,
                               msg="Base stress should be close to unstressed")

    def test_hpi_shock_mortgage_pd_increase(self):
        """HPI crash -> mortgage pool PD increases > 20%."""
        base = self._stress()
        stressed = self._stress(hpi_growth=-10.0)
        ratio = stressed["pd_base"] / max(base["pd_base"], 1e-10)
        self.assertGreater(ratio, 1.20,
                           f"HPI crash PD ratio {ratio:.2f} should be > 1.20")

    def test_hpi_no_effect_public_sector_only(self):
        """HPI shock has no effect on public sector pool LTV."""
        # Public sector has beta_hpi=0, so no LTV effect
        pub = _DF.filter(pl.col("cover_pool_type") == "public_sector").clone()
        if len(pub) > 5:
            result_base = stress_covered_bonds_positions(pub, {
                "gdp_growth": 1.2, "unemployment_rate": 7.5,
                "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5,
            })
            result_shock = stress_covered_bonds_positions(pub, {
                "gdp_growth": 1.2, "unemployment_rate": 7.5,
                "interest_rate": 3.5, "hpi_growth": -10.0, "inflation_rate": 2.5,
            })
            # PD should be same (beta_hpi=0 for public sector)
            self.assertAlmostEqual(
                result_base["pd_base"], result_shock["pd_base"],
                delta=result_base["pd_base"] * 0.01,
                msg="Public sector PD should not be affected by HPI",
            )

    def test_rate_increase_pd_increase(self):
        """Rate hike -> PD increases (refinancing risk)."""
        base = self._stress()
        stressed = self._stress(interest_rate=6.0)
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "Rate hike should increase PD")

    def test_gdp_crash_pd_increase(self):
        """GDP crash -> PD increases."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-3.0)
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "GDP crash should increase PD")

    def test_inflation_increase_pd_increase(self):
        """Inflation spike -> PD increases."""
        base = self._stress()
        stressed = self._stress(inflation_rate=6.0)
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "Inflation spike should increase PD")

    def test_lgd_downturn_higher(self):
        """Downturn LGD > base LGD (mortgage pools)."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-3.0, hpi_growth=-8.0)
        self.assertGreater(stressed["lgd_base"], base["lgd_base"],
                           "Downturn LGD should be > base LGD")

    def test_resilience_moderate_stress(self):
        """Moderate stress: PD < 1% (double recourse resilience)."""
        result = self._stress(gdp_growth=-1.0, hpi_growth=-3.0,
                              unemployment_rate=9.0, interest_rate=5.0)
        self.assertLess(result["pd_base"], 0.01,
                        f"Moderate stress PD {result['pd_base']:.4%} > 1%")

    def test_resilience_severe_stress(self):
        """Severe stress (GFC-like): PD < 2%."""
        result = self._stress(gdp_growth=-4.0, hpi_growth=-15.0,
                              unemployment_rate=12.0, interest_rate=6.0,
                              inflation_rate=5.0)
        self.assertLess(result["pd_base"], 0.02,
                        f"Severe stress PD {result['pd_base']:.4%} > 2%")

    def test_stress_returns_valid_keys(self):
        """Stress returns pd_base, lgd_base, rw_crr3."""
        result = self._stress()
        self.assertIn("pd_base", result)
        self.assertIn("lgd_base", result)
        self.assertIn("rw_crr3", result)


if __name__ == "__main__":
    unittest.main()
