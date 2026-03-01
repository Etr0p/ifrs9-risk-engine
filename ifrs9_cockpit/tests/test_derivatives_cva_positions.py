"""Tests pour le generateur position-par-position Derives / CVA.

Couvre :
    - Generation : shape, colonnes, EAD SA-CCR, distributions desk/product/netting
    - PD : range, calibration, CSA effect, non-constante
    - LGD : range, calibration (netting + CSA reduisent)
    - EAD SA-CCR : positive, calcul coherent
    - RW CRR3 : range, CSA effect
    - Aggregation : EAD, PD/LGD/RW blended, schema complet (19 cles)
    - Stress : impact macro, CVA risk increase, scenario adverse
"""

import unittest
import numpy as np
import polars as pl

from ifrs9_cockpit.synthetic_generator.derivatives_cva_positions import (
    generate_derivative_positions,
    compute_sa_ccr_ead,
    compute_cva_risk,
    compute_derivative_rw,
    stress_derivative_positions,
    aggregate_derivatives_to_balance_row,
)
from ifrs9_cockpit.config import ASSET_CLASS_MAP


# Shared fixture: generate once for the module
_N = 500
_TOTAL_EAD = 2.0e9  # 2B EUR notional
_SEED = 992
_DF = generate_derivative_positions(n_positions=_N, total_notional=_TOTAL_EAD, seed=_SEED)
_PROFILE = ASSET_CLASS_MAP["derivatives_cva"]

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
    """Validation du generateur de positions derives/CVA."""

    def test_shape(self):
        """n_positions lignes generees."""
        self.assertEqual(len(_DF), _N)

    def test_columns_present(self):
        """Toutes les colonnes requises sont presentes."""
        required = {
            "position_id", "ead", "pd_position", "lgd_position", "rw_crr3",
            "desk", "product", "netting_set", "csa_flag", "ead_sa_ccr",
        }
        self.assertTrue(required <= set(_DF.columns),
                        f"Missing: {required - set(_DF.columns)}")

    def test_ead_scaling(self):
        """sum(ead) dans le bon ordre de grandeur (SA-CCR < notional)."""
        actual = _DF["ead"].sum()
        # EAD SA-CCR is typically 10-30% of notional (alpha=1.4, netting reduces)
        self.assertGreater(actual, 0)
        ratio = actual / _TOTAL_EAD
        self.assertGreater(ratio, 0.05, f"EAD/notional ratio too low: {ratio}")
        self.assertLess(ratio, 1.0, f"EAD/notional ratio too high: {ratio}")

    def test_ead_all_positive(self):
        """Toutes les positions ont un EAD > 0."""
        self.assertTrue((_DF["ead"] > 0).all())

    def test_position_ids_unique(self):
        """IDs de position uniques."""
        self.assertEqual(_DF["position_id"].n_unique(), _N)

    def test_desk_diverse(self):
        """Au moins 2 desks distincts."""
        n_desks = _DF["desk"].n_unique()
        self.assertGreaterEqual(n_desks, 2,
                                f"Expected >= 2 desks, got {n_desks}")

    def test_product_diverse(self):
        """Au moins 3 types de produits derives distincts."""
        n_products = _DF["product"].n_unique()
        self.assertGreaterEqual(n_products, 3,
                                f"Expected >= 3 products, got {n_products}")

    def test_netting_set_present(self):
        """Netting sets presents et non-vides."""
        self.assertFalse(_DF["netting_set"].is_null().any())
        self.assertGreater(_DF["netting_set"].n_unique(), 1)

    def test_csa_flag_boolean(self):
        """CSA flag est booleen (True/False ou 0/1)."""
        unique_vals = set(_DF["csa_flag"].unique().to_list())
        # Accept bool or int representation
        self.assertTrue(unique_vals <= {True, False, 0, 1, 0.0, 1.0},
                        f"CSA flag unexpected values: {unique_vals}")

    def test_ead_sa_ccr_positive(self):
        """EAD SA-CCR > 0 pour toutes les positions."""
        self.assertTrue((_DF["ead_sa_ccr"] > 0).all())

    def test_reproducibility(self):
        """Meme seed -> meme resultat."""
        df2 = generate_derivative_positions(n_positions=_N, total_notional=_TOTAL_EAD, seed=_SEED)
        self.assertTrue(_DF.equals(df2), "DataFrames should be identical with same seed")


class TestPD(unittest.TestCase):
    """Validation de la PD derives/CVA."""

    def test_pd_range(self):
        """PD dans [0.0001, 0.10]."""
        self.assertGreaterEqual(_DF["pd_position"].min(), 0.0001)
        self.assertLessEqual(_DF["pd_position"].max(), 0.10)

    def test_pd_not_constant(self):
        """PD non constante (dispersion > 0)."""
        self.assertGreater(_DF["pd_position"].std(), 1e-6,
                           "PD should not be constant across positions")

    def test_pd_mean_calibrated(self):
        """PD moyenne EAD-weighted entre 0.001 et 0.05."""
        w = _DF["ead"].to_numpy() / _DF["ead"].sum()
        pd_mean = float(np.dot(w, _DF["pd_position"].to_numpy()))
        self.assertGreater(pd_mean, 0.001)
        self.assertLess(pd_mean, 0.05)

    def test_pd_csa_lower(self):
        """Positions avec CSA ont PD plus basse (ou egale) en moyenne."""
        csa_true = _DF.filter(pl.col("csa_flag") == True)  # noqa: E712
        csa_false = _DF.filter(pl.col("csa_flag") == False)  # noqa: E712
        if len(csa_true) > 10 and len(csa_false) > 10:
            pd_csa = csa_true["pd_position"].median()
            pd_no_csa = csa_false["pd_position"].median()
            # CSA provides collateral -> lower effective PD
            self.assertLessEqual(pd_csa, pd_no_csa * 1.1,
                                 "CSA positions should have similar or lower PD")

    def test_pd_no_nan(self):
        """Pas de NaN dans PD."""
        self.assertFalse(_DF["pd_position"].is_null().any())


class TestLGD(unittest.TestCase):
    """Validation de la LGD derives/CVA."""

    def test_lgd_range(self):
        """LGD dans [0.10, 0.70]."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.10)
        self.assertLessEqual(_DF["lgd_position"].max(), 0.70)

    def test_lgd_not_all_identical(self):
        """LGD pas identique pour toutes les positions."""
        n_unique = _DF["lgd_position"].n_unique()
        self.assertGreater(n_unique, 1,
                           "LGD should not be identical for all positions")

    def test_lgd_mean_calibrated(self):
        """LGD moyenne entre 0.20 et 0.55 (netting + CSA reduisent)."""
        lgd_mean = _DF["lgd_position"].mean()
        self.assertGreater(lgd_mean, 0.20)
        self.assertLess(lgd_mean, 0.55)

    def test_lgd_csa_lower(self):
        """LGD plus basse pour positions avec CSA (margin calls)."""
        csa_true = _DF.filter(pl.col("csa_flag") == True)  # noqa: E712
        csa_false = _DF.filter(pl.col("csa_flag") == False)  # noqa: E712
        if len(csa_true) > 10 and len(csa_false) > 10:
            lgd_csa = csa_true["lgd_position"].median()
            lgd_no_csa = csa_false["lgd_position"].median()
            self.assertLess(lgd_csa, lgd_no_csa,
                            "CSA positions should have lower LGD")

    def test_lgd_no_nan(self):
        """Pas de NaN dans LGD."""
        self.assertFalse(_DF["lgd_position"].is_null().any())


class TestEAD(unittest.TestCase):
    """Validation de l'EAD SA-CCR."""

    def test_ead_sa_ccr_positive(self):
        """EAD SA-CCR > 0 pour toutes les positions."""
        self.assertTrue((_DF["ead_sa_ccr"] > 0).all())

    def test_ead_sa_ccr_no_nan(self):
        """Pas de NaN dans EAD SA-CCR."""
        self.assertFalse(_DF["ead_sa_ccr"].is_null().any())

    def test_ead_equals_sa_ccr(self):
        """EAD = EAD SA-CCR (ou proportionnel)."""
        # ead should be derived from ead_sa_ccr
        corr = np.corrcoef(
            _DF["ead"].to_numpy(),
            _DF["ead_sa_ccr"].to_numpy(),
        )[0, 1]
        self.assertGreater(corr, 0.8,
                           f"EAD-EAD_SA_CCR correlation {corr:.3f} should be > 0.8")


class TestRW(unittest.TestCase):
    """Validation du RW CRR3 derives/CVA."""

    def test_rw_range(self):
        """RW dans [0, 1.50] (CSA+netting can yield very low RW)."""
        self.assertGreaterEqual(_DF["rw_crr3"].min(), 0.0)
        self.assertLessEqual(_DF["rw_crr3"].max(), 1.50)

    def test_rw_not_constant(self):
        """RW non constant."""
        self.assertGreater(_DF["rw_crr3"].n_unique(), 1)

    def test_rw_mean_calibrated(self):
        """RW moyen entre 0.01 et 0.80 (SA-CVA netting reduces)."""
        w = _DF["ead"].to_numpy() / _DF["ead"].sum()
        rw_mean = float(np.dot(w, _DF["rw_crr3"].to_numpy()))
        self.assertGreater(rw_mean, 0.01)
        self.assertLess(rw_mean, 0.80)


class TestAggregation(unittest.TestCase):
    """Validation de l'aggregation balance sheet."""

    def setUp(self):
        self.row = aggregate_derivatives_to_balance_row(_DF, _PROFILE)

    def test_aggregate_ead(self):
        """EAD total agrege > 0."""
        self.assertGreater(self.row["ead_total"], 0)

    def test_aggregate_pd_range(self):
        """PD agregee dans une fourchette raisonnable."""
        self.assertGreater(self.row["pd_base"], 0.0001)
        self.assertLess(self.row["pd_base"], 0.05)

    def test_aggregate_lgd_range(self):
        """LGD agregee dans une fourchette raisonnable."""
        self.assertGreater(self.row["lgd_base"], 0.10)
        self.assertLess(self.row["lgd_base"], 0.60)

    def test_aggregate_rw_range(self):
        """RW agrege dans une fourchette raisonnable."""
        self.assertGreater(self.row["rw_crr3"], 0.01)
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
    """Validation du stress test derives/CVA."""

    def _stress(self, **overrides):
        """Helper: stress with base scenario + overrides."""
        macro = dict(_BASE_MACRO)
        macro.update(overrides)
        return stress_derivative_positions(_DF, macro)

    def test_base_approximately_unstressed(self):
        """Scenario base -> PD dans le meme ordre de grandeur."""
        result = self._stress()
        unstressed_pd = _DF["pd_position"].mean()
        ratio = result["pd_base"] / max(unstressed_pd, 1e-8)
        self.assertGreater(ratio, 0.3, msg="Base stress PD too low")
        self.assertLess(ratio, 5.0, msg="Base stress PD too high")

    def test_adverse_increases_pd(self):
        """Scenario adverse -> PD augmente."""
        base = self._stress()
        stressed = self._stress(**_ADVERSE_MACRO)
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "Adverse scenario should increase derivative PD")

    def test_rate_hike_increases_pd(self):
        """Hausse taux -> PD augmente (CVA risk + counterparty stress)."""
        base = self._stress()
        stressed = self._stress(interest_rate=6.0)
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "Rate hike should increase derivative PD")

    def test_gdp_crash_increases_pd(self):
        """Choc GDP -> PD augmente."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-4.0)
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "GDP crash should increase derivative PD")

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
        """RW augmente ou stable en stress severe."""
        base = self._stress()
        severe = self._stress(**_ADVERSE_MACRO)
        self.assertGreaterEqual(severe["rw_crr3"], base["rw_crr3"] * 0.95,
                                "RW should not decrease significantly under stress")


if __name__ == "__main__":
    unittest.main()
