"""Tests pour le generateur position-par-position Obligations Corporate.

Couvre :
    - Generation : shape, colonnes, EAD scaling, distributions rating/seniority/sector
    - PD : range, calibration, ordering par rating, non-constante
    - LGD : range, calibration, seniority ordering (senior < sub)
    - RW CRR3 : range, monotonie par rating
    - Aggregation : EAD, PD/LGD/RW blended, schema complet (19 cles)
    - Stress : impact macro sur PD/LGD/RW, scenario adverse severe
"""

import unittest
import numpy as np
import polars as pl

from ifrs9_cockpit.synthetic_generator.corporate_bonds_positions import (
    generate_corporate_bonds_positions,
    compute_corporate_bonds_pd,
    compute_corporate_bonds_lgd,
    compute_corporate_bonds_rw,
    stress_corporate_bonds_positions,
    aggregate_corporate_bonds_to_balance_row,
)
from ifrs9_cockpit.config import ASSET_CLASS_MAP


# Shared fixture: generate once for the module
_N = 500
_TOTAL_EAD = 4.0e9  # 4B EUR
_SEED = 972
_DF = generate_corporate_bonds_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
_PROFILE = ASSET_CLASS_MAP["corporate_bonds"]

_BASE_MACRO = {
    "gdp_growth": 1.2,
    "unemployment_rate": 7.5,
    "interest_rate": 3.5,
    "inflation_rate": 2.5,
    "hpi_growth": 2.0,
}
_ADVERSE_MACRO = {
    "gdp_growth": -4.0,
    "unemployment_rate": 12.0,
    "interest_rate": 6.0,
    "inflation_rate": 5.0,
    "hpi_growth": -15.0,
}


class TestGeneration(unittest.TestCase):
    """Validation du generateur de positions obligations corporate."""

    def test_shape(self):
        """n_positions lignes generees."""
        self.assertEqual(len(_DF), _N)

    def test_columns_present(self):
        """Toutes les colonnes requises sont presentes."""
        required = {
            "position_id", "ead", "pd_base", "lgd_base", "rw_crr3",
            "rating", "seniority", "spread_bps", "sector_gics",
        }
        self.assertTrue(required <= set(_DF.columns),
                        f"Missing: {required - set(_DF.columns)}")

    def test_ead_scaling(self):
        """sum(ead) ~ total_ead."""
        actual = _DF["ead"].sum()
        self.assertAlmostEqual(actual / _TOTAL_EAD, 1.0, places=2)

    def test_ead_all_positive(self):
        """Toutes les positions ont un EAD > 0."""
        self.assertTrue((_DF["ead"] > 0).all())

    def test_position_ids_unique(self):
        """IDs de position uniques."""
        self.assertEqual(_DF["position_id"].n_unique(), _N)

    def test_rating_present(self):
        """Ratings valides presents (IG + HY)."""
        ratings = set(_DF["rating"].unique().to_list())
        # At least some IG and HY ratings
        self.assertTrue(len(ratings) >= 3,
                        f"Expected >= 3 distinct ratings, got {len(ratings)}")

    def test_seniority_distribution(self):
        """Senior (secured + unsecured) dominant dans le portefeuille."""
        vc = _DF["seniority"].value_counts()
        total = vc["count"].sum()
        vc = vc.with_columns((pl.col("count") / total).alias("proportion"))
        vc_dict = dict(zip(vc["seniority"].to_list(), vc["proportion"].to_list()))
        # Senior (secured + unsecured) should be dominant
        senior_share = (vc_dict.get("secured", 0) + vc_dict.get("unsecured", 0)
                        + vc_dict.get("senior_secured", 0) + vc_dict.get("senior_unsecured", 0))
        self.assertGreater(senior_share, 0.30,
                           f"Senior share {senior_share:.1%} too low")

    def test_spread_positive(self):
        """Spread > 0 pour toutes les positions."""
        self.assertTrue((_DF["spread_bps"] > 0).all())

    def test_sector_gics_diverse(self):
        """Au moins 3 secteurs GICS distincts."""
        n_sectors = _DF["sector_gics"].n_unique()
        self.assertGreaterEqual(n_sectors, 3,
                                f"Expected >= 3 GICS sectors, got {n_sectors}")

    def test_reproducibility(self):
        """Meme seed -> meme resultat."""
        df2 = generate_corporate_bonds_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
        self.assertTrue(_DF.equals(df2), "DataFrames should be identical with same seed")


class TestPD(unittest.TestCase):
    """Validation de la PD obligations corporate."""

    def test_pd_range(self):
        """PD dans [0.0001, 1.0] (CCC-rated can have PD up to 30%)."""
        self.assertGreaterEqual(_DF["pd_base"].min(), 0.0001)
        self.assertLessEqual(_DF["pd_base"].max(), 1.0)

    def test_pd_not_constant(self):
        """PD non constante (dispersion > 0)."""
        self.assertGreater(_DF["pd_base"].std(), 1e-6,
                           "PD should not be constant across positions")

    def test_pd_mean_calibrated(self):
        """PD moyenne EAD-weighted entre 0.001 et 0.05."""
        w = _DF["ead"].to_numpy() / _DF["ead"].sum()
        pd_mean = float(np.dot(w, _DF["pd_base"].to_numpy()))
        self.assertGreater(pd_mean, 0.001)
        self.assertLess(pd_mean, 0.05)

    def test_pd_rating_ordering(self):
        """BBB PD mediane > A PD mediane > AA PD mediane."""
        pd_by_rating = (
            _DF.group_by("rating")
            .agg(pl.col("pd_base").median().alias("pd_median"))
        )
        rating_dict = dict(zip(
            pd_by_rating["rating"].to_list(),
            pd_by_rating["pd_median"].to_list(),
        ))
        if "AA" in rating_dict and "A" in rating_dict:
            self.assertLess(rating_dict["AA"], rating_dict["A"],
                            "AA PD should be < A PD")
        if "A" in rating_dict and "BBB" in rating_dict:
            self.assertLess(rating_dict["A"], rating_dict["BBB"],
                            "A PD should be < BBB PD")

    def test_pd_spread_correlation(self):
        """PD correle positivement au spread."""
        corr = float(_DF.select(pl.corr("spread_bps", "pd_base")).item())
        self.assertGreater(corr, 0.3,
                           f"Spread-PD correlation {corr:.3f} should be > 0.3")

    def test_pd_no_nan(self):
        """Pas de NaN dans PD."""
        self.assertFalse(_DF["pd_base"].is_null().any())


class TestLGD(unittest.TestCase):
    """Validation de la LGD obligations corporate."""

    def test_lgd_range(self):
        """LGD dans [0.15, 0.90] (subordinated + noise can exceed 0.70)."""
        self.assertGreaterEqual(_DF["lgd_base"].min(), 0.15)
        self.assertLessEqual(_DF["lgd_base"].max(), 0.90)

    def test_lgd_not_all_identical(self):
        """LGD pas identique pour toutes les positions."""
        n_unique = _DF["lgd_base"].n_unique()
        self.assertGreater(n_unique, 1,
                           "LGD should not be identical for all positions")

    def test_lgd_mean_calibrated(self):
        """LGD moyenne entre 0.25 et 0.55 (Moody's historical 40%)."""
        lgd_mean = _DF["lgd_base"].mean()
        self.assertGreater(lgd_mean, 0.25)
        self.assertLess(lgd_mean, 0.55)

    def test_lgd_seniority_ordering(self):
        """Senior LGD < Subordinated LGD."""
        lgd_by_sen = (
            _DF.group_by("seniority")
            .agg(pl.col("lgd_base").median().alias("lgd_median"))
        )
        sen_dict = dict(zip(
            lgd_by_sen["seniority"].to_list(),
            lgd_by_sen["lgd_median"].to_list(),
        ))
        senior_keys = [k for k in sen_dict if "senior" in k.lower()]
        sub_keys = [k for k in sen_dict if "sub" in k.lower()]
        if senior_keys and sub_keys:
            lgd_senior = min(sen_dict[k] for k in senior_keys)
            lgd_sub = max(sen_dict[k] for k in sub_keys)
            self.assertLess(lgd_senior, lgd_sub,
                            "Senior LGD should be < Subordinated LGD")

    def test_lgd_no_nan(self):
        """Pas de NaN dans LGD."""
        self.assertFalse(_DF["lgd_base"].is_null().any())


class TestRW(unittest.TestCase):
    """Validation du RW CRR3 obligations corporate."""

    def test_rw_range(self):
        """RW dans [0.20, 1.50]."""
        self.assertGreaterEqual(_DF["rw_crr3"].min(), 0.20)
        self.assertLessEqual(_DF["rw_crr3"].max(), 1.50)

    def test_rw_not_constant(self):
        """RW non constant."""
        self.assertGreater(_DF["rw_crr3"].n_unique(), 1)

    def test_rw_mean_calibrated(self):
        """RW moyen EAD-weighted entre 0.30 et 1.00."""
        w = _DF["ead"].to_numpy() / _DF["ead"].sum()
        rw_mean = float(np.dot(w, _DF["rw_crr3"].to_numpy()))
        self.assertGreater(rw_mean, 0.30)
        self.assertLess(rw_mean, 1.00)

    def test_rw_rating_ordering(self):
        """RW croissant avec la deterioration du rating."""
        rw_by_rating = (
            _DF.group_by("rating")
            .agg(pl.col("rw_crr3").median().alias("rw_median"))
        )
        rating_dict = dict(zip(
            rw_by_rating["rating"].to_list(),
            rw_by_rating["rw_median"].to_list(),
        ))
        if "AA" in rating_dict and "BBB" in rating_dict:
            self.assertLess(rating_dict["AA"], rating_dict["BBB"],
                            "AA RW should be < BBB RW")


class TestAggregation(unittest.TestCase):
    """Validation de l'aggregation balance sheet."""

    def setUp(self):
        self.row = aggregate_corporate_bonds_to_balance_row(_DF, _PROFILE)

    def test_aggregate_ead(self):
        """EAD total agrege ~ total_ead."""
        self.assertAlmostEqual(self.row["ead_total"] / _TOTAL_EAD, 1.0, places=2)

    def test_aggregate_pd_range(self):
        """PD agregee dans une fourchette raisonnable."""
        self.assertGreater(self.row["pd_base"], 0.0003)
        self.assertLess(self.row["pd_base"], 0.05)

    def test_aggregate_lgd_range(self):
        """LGD agregee dans une fourchette raisonnable."""
        self.assertGreater(self.row["lgd_base"], 0.20)
        self.assertLess(self.row["lgd_base"], 0.60)

    def test_aggregate_rw_range(self):
        """RW agrege dans une fourchette raisonnable."""
        self.assertGreater(self.row["rw_crr3"], 0.20)
        self.assertLess(self.row["rw_crr3"], 1.00)

    def test_aggregate_schema(self):
        """19 cles presentes dans le dict agrege."""
        expected_keys = {
            "asset_class", "label", "category", "ead_total", "typical_weight",
            "pd_base", "lgd_base", "tenor", "asset_correlation", "rw_crr3",
            "physical_risk", "transition_risk", "green_capex_ratio",
            "scope3_exposure", "absorption_buffer", "hqla_eligible",
            "hqla_level", "exempt_from_staging", "rsf_weight",
        }
        self.assertTrue(expected_keys <= set(self.row.keys()),
                        f"Missing: {expected_keys - set(self.row.keys())}")


class TestStress(unittest.TestCase):
    """Validation du stress test obligations corporate."""

    def _stress(self, **overrides):
        """Helper: stress with base scenario + overrides."""
        macro = dict(_BASE_MACRO)
        macro.update(overrides)
        return stress_corporate_bonds_positions(_DF, macro)

    def test_base_approximately_unstressed(self):
        """Scenario base -> PD ~ PD non-stresse."""
        result = self._stress()
        unstressed_pd = _DF["pd_base"].mean()
        self.assertAlmostEqual(result["pd_base"] / max(unstressed_pd, 1e-8),
                               1.0, delta=0.30,
                               msg="Base stress should be close to unstressed")

    def test_gdp_crash_increases_pd(self):
        """Choc GDP negatif -> PD augmente."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-4.0, unemployment_rate=12.0)
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "GDP crash should increase corporate bond PD")

    def test_rate_hike_does_not_decrease_pd(self):
        """Hausse taux -> PD stable ou augmente."""
        base = self._stress()
        stressed = self._stress(interest_rate=6.0)
        self.assertGreaterEqual(stressed["pd_base"], base["pd_base"] * 0.99,
                                "Rate hike should not decrease PD significantly")

    def test_lgd_downturn(self):
        """GDP negatif -> LGD augmente."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-5.0)
        self.assertGreaterEqual(stressed["lgd_base"], base["lgd_base"],
                                "Downturn LGD should increase")

    def test_severe_combined(self):
        """Stress severe combine -> PD significativement plus haute."""
        base = self._stress()
        severe = self._stress(**_ADVERSE_MACRO)
        self.assertGreater(severe["pd_base"], base["pd_base"] * 1.3,
                           "Severe stress should increase PD by at least 30%")

    def test_positive_gdp_stable_or_lower(self):
        """GDP positif -> PD stable ou diminue."""
        base = self._stress()
        boom = self._stress(gdp_growth=4.0)
        self.assertLessEqual(boom["pd_base"], base["pd_base"] * 1.05)

    def test_stress_returns_valid_keys(self):
        """Stress retourne pd_base, lgd_base, rw_crr3."""
        result = self._stress()
        for key in ("pd_base", "lgd_base", "rw_crr3"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_rw_increases_under_stress(self):
        """RW augmente en stress severe (downgrade effect)."""
        base = self._stress()
        severe = self._stress(**_ADVERSE_MACRO)
        self.assertGreaterEqual(severe["rw_crr3"], base["rw_crr3"],
                                "RW should increase under severe stress")


if __name__ == "__main__":
    unittest.main()
