"""Tests pour le generateur position-par-position Repos / SFT.

Couvre :
    - Generation : shape, colonnes, EAD scaling, distributions collateral/tenor
    - PD : range, calibration, collateral effect, non-constante
    - LGD : range, calibration (tres basse, collateralise), overcollateralization
    - RW CRR3 : range, monotonie par qualite collateral
    - Aggregation : EAD, PD/LGD/RW blended, schema complet (19 cles)
    - Stress : haircut spiral, collateral degradation, scenario adverse
"""

import unittest
import numpy as np
import polars as pl

from ifrs9_cockpit.synthetic_generator.repos_sft_positions import (
    generate_repo_positions,
    compute_repo_pd,
    compute_repo_lgd,
    compute_repo_rw,
    stress_repo_positions,
    aggregate_repo_to_balance_row,
)
from ifrs9_cockpit.config import ASSET_CLASS_MAP


# Shared fixture: generate once for the module
_N = 400
_TOTAL_EAD = 3.0e9  # 3B EUR
_SEED = 982
_DF = generate_repo_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
_PROFILE = ASSET_CLASS_MAP["repos_sft"]

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
    """Validation du generateur de positions Repo/SFT."""

    def test_shape(self):
        """n_positions lignes generees."""
        self.assertEqual(len(_DF), _N)

    def test_columns_present(self):
        """Toutes les colonnes requises sont presentes."""
        required = {
            "position_id", "ead", "pd_position", "lgd_position", "rw_crr3",
            "collateral_type", "tenor", "haircut", "overcollateralization",
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

    def test_collateral_types_diverse(self):
        """Au moins 2 types de collateral distincts."""
        n_types = _DF["collateral_type"].n_unique()
        self.assertGreaterEqual(n_types, 2,
                                f"Expected >= 2 collateral types, got {n_types}")

    def test_tenor_range(self):
        """Tenor dans [0, 1] (repos = court terme)."""
        self.assertGreaterEqual(_DF["tenor_years"].min(), 0.0)
        self.assertLessEqual(_DF["tenor_years"].max(), 1.0)

    def test_haircut_range(self):
        """Haircut dans [0, 0.50]."""
        self.assertGreaterEqual(_DF["haircut"].min(), 0.0)
        self.assertLessEqual(_DF["haircut"].max(), 0.50)

    def test_overcollateralization_positive(self):
        """Overcollateralization >= 0 (repos sont collateralises)."""
        self.assertGreaterEqual(_DF["overcollateralization"].min(), 0.0)

    def test_reproducibility(self):
        """Meme seed -> meme resultat."""
        df2 = generate_repo_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
        self.assertTrue(_DF.equals(df2), "DataFrames should be identical with same seed")


class TestPD(unittest.TestCase):
    """Validation de la PD Repo/SFT."""

    def test_pd_range(self):
        """PD dans [0.0001, 0.05] (collateralise = basse PD)."""
        self.assertGreaterEqual(_DF["pd_position"].min(), 0.0001)
        self.assertLessEqual(_DF["pd_position"].max(), 0.05)

    def test_pd_not_constant(self):
        """PD non constante (dispersion > 0)."""
        self.assertGreater(_DF["pd_position"].std(), 1e-6,
                           "PD should not be constant across positions")

    def test_pd_mean_low(self):
        """PD moyenne EAD-weighted basse (collateralise)."""
        w = _DF["ead"].to_numpy() / _DF["ead"].sum()
        pd_mean = float(np.dot(w, _DF["pd_position"].to_numpy()))
        self.assertGreater(pd_mean, 0.0001)
        self.assertLess(pd_mean, 0.02)

    def test_pd_no_nan(self):
        """Pas de NaN dans PD."""
        self.assertFalse(_DF["pd_position"].is_null().any())


class TestLGD(unittest.TestCase):
    """Validation de la LGD Repo/SFT (collateralise = basse LGD)."""

    def test_lgd_range(self):
        """LGD dans [0.005, 0.45] (collateralise, floor=50bp)."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.005)
        self.assertLessEqual(_DF["lgd_position"].max(), 0.45)

    def test_lgd_not_all_identical(self):
        """LGD pas identique pour toutes les positions."""
        n_unique = _DF["lgd_position"].n_unique()
        self.assertGreater(n_unique, 1,
                           "LGD should not be identical for all positions")

    def test_lgd_mean_low(self):
        """LGD moyenne basse (overcollateralization protege)."""
        lgd_mean = _DF["lgd_position"].mean()
        self.assertLess(lgd_mean, 0.20,
                        f"Mean LGD {lgd_mean:.2%} too high for collateralized repos")

    def test_lgd_varies_by_collateral(self):
        """LGD varies by collateral type (govt < equity)."""
        govt_mask = _DF["collateral_type"] == "govt"
        equity_mask = _DF["collateral_type"] == "equity"
        if govt_mask.sum() > 0 and equity_mask.sum() > 0:
            lgd_govt = _DF.filter(govt_mask)["lgd_position"].mean()
            lgd_equity = _DF.filter(equity_mask)["lgd_position"].mean()
            self.assertLess(lgd_govt, lgd_equity,
                            f"Govt LGD {lgd_govt:.4f} should be < equity LGD {lgd_equity:.4f}")

    def test_lgd_no_nan(self):
        """Pas de NaN dans LGD."""
        self.assertFalse(_DF["lgd_position"].is_null().any())


class TestRW(unittest.TestCase):
    """Validation du RW CRR3 Repo/SFT."""

    def test_rw_range(self):
        """RW dans [0.0, 0.50] (collateralise = faible RW)."""
        self.assertGreaterEqual(_DF["rw_crr3"].min(), 0.0)
        self.assertLessEqual(_DF["rw_crr3"].max(), 0.50)

    def test_rw_not_all_zero(self):
        """RW non nul pour au moins certaines positions."""
        self.assertGreater(_DF["rw_crr3"].max(), 0.0)

    def test_rw_mean_low(self):
        """RW moyen bas (repos bien collateralises)."""
        w = _DF["ead"].to_numpy() / _DF["ead"].sum()
        rw_mean = float(np.dot(w, _DF["rw_crr3"].to_numpy()))
        self.assertLess(rw_mean, 0.25)


class TestAggregation(unittest.TestCase):
    """Validation de l'aggregation balance sheet."""

    def setUp(self):
        self.row = aggregate_repo_to_balance_row(_DF, _PROFILE)

    def test_aggregate_ead(self):
        """EAD total agrege ~ total_ead."""
        self.assertAlmostEqual(self.row["ead_total"] / _TOTAL_EAD, 1.0, places=2)

    def test_aggregate_pd_range(self):
        """PD agregee dans une fourchette raisonnable."""
        self.assertGreater(self.row["pd_base"], 0.0001)
        self.assertLess(self.row["pd_base"], 0.02)

    def test_aggregate_lgd_range(self):
        """LGD agregee basse."""
        self.assertGreater(self.row["lgd_base"], 0.01)
        self.assertLess(self.row["lgd_base"], 0.25)

    def test_aggregate_rw_range(self):
        """RW agrege dans une fourchette raisonnable."""
        self.assertGreaterEqual(self.row["rw_crr3"], 0.0)
        self.assertLess(self.row["rw_crr3"], 0.30)

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
    """Validation du stress test Repo/SFT (haircut spiral)."""

    def _stress(self, **overrides):
        """Helper: stress with base scenario + overrides."""
        macro = dict(_BASE_MACRO)
        macro.update(overrides)
        return stress_repo_positions(_DF, macro)

    def test_base_approximately_unstressed(self):
        """Scenario base -> PD ~ PD non-stresse."""
        result = self._stress()
        unstressed_pd = _DF["pd_position"].mean()
        self.assertAlmostEqual(result["pd_base"] / max(unstressed_pd, 1e-8),
                               1.0, delta=0.30,
                               msg="Base stress should be close to unstressed")

    def test_adverse_increases_pd(self):
        """Scenario adverse -> PD augmente."""
        base = self._stress()
        stressed = self._stress(**_ADVERSE_MACRO)
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "Adverse scenario should increase repo PD")

    def test_haircut_spiral_under_stress(self):
        """Stress severe -> LGD augmente (haircut spiral)."""
        base = self._stress()
        stressed = self._stress(**_ADVERSE_MACRO)
        self.assertGreater(stressed["lgd_base"], base["lgd_base"],
                           "Haircut spiral should increase LGD under stress")

    def test_rate_hike_increases_pd(self):
        """Hausse taux -> PD augmente (margin calls, collateral stress)."""
        base = self._stress()
        stressed = self._stress(interest_rate=6.0)
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "Rate hike should increase repo PD")

    def test_gdp_crash_increases_pd(self):
        """Choc GDP -> PD augmente."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-4.0)
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "GDP crash should increase repo PD")

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

    def test_severe_combined_significant(self):
        """Stress severe -> PD significativement plus haute."""
        base = self._stress()
        severe = self._stress(gdp_growth=-5.0, unemployment_rate=14.0,
                              interest_rate=7.0, inflation_rate=6.0,
                              hpi_growth=-20.0)
        self.assertGreater(severe["pd_base"], base["pd_base"] * 1.3,
                           "Severe stress should increase PD by at least 30%")

    def test_lgd_remains_bounded(self):
        """Meme en stress severe, LGD reste < 50% (collateral protege)."""
        result = self._stress(**_ADVERSE_MACRO)
        self.assertLess(result["lgd_base"], 0.50,
                        f"Repo LGD {result['lgd_base']:.2%} > 50% even under stress")


if __name__ == "__main__":
    unittest.main()
