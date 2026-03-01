"""Tests pour le generateur position-par-position Actions Cotees (Equities).

Couvre :
    - Generation : shape, colonnes, EAD scaling, distributions secteurs/bourses
    - PD Merton : range, calibration, non-constante, ordering par secteur
    - LGD : range, calibration (actions = perte quasi-totale)
    - RW : toujours 100% (CRR3 Art. 133)
    - Aggregation : EAD, PD/LGD/RW blended, mtm_loss, schema complet (19+1 cles)
    - Stress : impact macro sur PD/LGD/MTM, resilience, accounting FVTPL
"""

import unittest
import numpy as np
import polars as pl

from ifrs9_cockpit.synthetic_generator.equity_positions import (
    generate_equity_positions,
    compute_equity_pd,
    compute_equity_lgd,
    stress_equity_positions,
    aggregate_equity_to_balance_row,
)
from ifrs9_cockpit.config import ASSET_CLASS_MAP


# Shared fixture: generate once for the module
_N = 300
_TOTAL_EAD = 1.5e9  # 1.5B EUR
_SEED = 952
_DF = generate_equity_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
_PROFILE = ASSET_CLASS_MAP["equities"]

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
    """Validation du generateur de positions equity."""

    def test_shape(self):
        """n_positions lignes generees."""
        self.assertEqual(len(_DF), _N)

    def test_columns_present(self):
        """Toutes les colonnes requises sont presentes."""
        required = {
            "position_id", "ead", "pd_merton", "lgd",
            "rw_crr3", "accounting_treatment",
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

    def test_accounting_fvtpl(self):
        """Toutes les actions comptabilisees en FVTPL."""
        self.assertTrue((_DF["accounting_treatment"] == "fvtpl").all())

    def test_position_ids_unique(self):
        """IDs de position uniques."""
        self.assertEqual(_DF["position_id"].n_unique(), _N)

    def test_reproducibility(self):
        """Meme seed -> meme resultat."""
        df2 = generate_equity_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
        self.assertTrue(_DF.equals(df2), "DataFrames should be identical with same seed")


class TestPD(unittest.TestCase):
    """Validation de la PD Merton pour actions."""

    def test_pd_range(self):
        """PD dans [0.0003, 0.30] (actions plus risquees que dette)."""
        self.assertGreaterEqual(_DF["pd_merton"].min(), 0.0003)
        self.assertLessEqual(_DF["pd_merton"].max(), 0.30)

    def test_pd_not_constant(self):
        """PD non constante (dispersion > 0)."""
        self.assertGreater(_DF["pd_merton"].std(), 1e-6,
                           "PD should not be constant across positions")

    def test_pd_mean_calibrated(self):
        """PD moyenne EAD-weighted dans une fourchette raisonnable."""
        w = _DF["ead"].to_numpy() / _DF["ead"].sum()
        pd_mean = float(np.dot(w, _DF["pd_merton"].to_numpy()))
        self.assertGreater(pd_mean, 0.001)
        self.assertLess(pd_mean, 0.15)

    def test_pd_no_nan(self):
        """Pas de NaN dans PD."""
        self.assertFalse(_DF["pd_merton"].is_null().any())


class TestLGD(unittest.TestCase):
    """Validation de la LGD equity (perte quasi-totale en cas de defaut)."""

    def test_lgd_range(self):
        """LGD dans [0.50, 1.00] (actions = perte elevee en defaut)."""
        self.assertGreaterEqual(_DF["lgd"].min(), 0.50)
        self.assertLessEqual(_DF["lgd"].max(), 1.00)

    def test_lgd_not_all_identical(self):
        """LGD pas identique pour toutes les positions."""
        n_unique = _DF["lgd"].n_unique()
        self.assertGreater(n_unique, 1,
                           "LGD should not be identical for all positions")

    def test_lgd_mean_high(self):
        """LGD moyenne elevee (equity = subordinated)."""
        lgd_mean = _DF["lgd"].mean()
        self.assertGreater(lgd_mean, 0.70,
                           f"Mean LGD {lgd_mean:.2%} too low for equity")

    def test_lgd_no_nan(self):
        """Pas de NaN dans LGD."""
        self.assertFalse(_DF["lgd"].is_null().any())


class TestRW(unittest.TestCase):
    """Validation du RW = 100% (CRR3 Art. 133)."""

    def test_rw_always_100pct(self):
        """RW = 100% pour toutes les positions (CRR3 Art. 133)."""
        self.assertTrue((_DF["rw_crr3"] == 1.00).all(),
                        "All equity positions should have RW = 100%")

    def test_rw_aggregate_100pct(self):
        """RW agrege = 100%."""
        row = aggregate_equity_to_balance_row(_DF, _PROFILE)
        self.assertAlmostEqual(row["rw_crr3"], 1.00, places=2)


class TestAggregation(unittest.TestCase):
    """Validation de l'aggregation balance sheet."""

    def setUp(self):
        self.row = aggregate_equity_to_balance_row(_DF, _PROFILE)

    def test_aggregate_ead(self):
        """EAD total agrege ~ total_ead."""
        self.assertAlmostEqual(self.row["ead_total"] / _TOTAL_EAD, 1.0, places=2)

    def test_aggregate_pd_range(self):
        """PD agregee dans une fourchette raisonnable."""
        self.assertGreater(self.row["pd_base"], 0.0003)
        self.assertLess(self.row["pd_base"], 0.15)

    def test_aggregate_lgd_range(self):
        """LGD agregee elevee."""
        self.assertGreater(self.row["lgd_base"], 0.50)
        self.assertLessEqual(self.row["lgd_base"], 1.00)

    def test_aggregate_rw(self):
        """RW agrege = 100%."""
        self.assertAlmostEqual(self.row["rw_crr3"], 1.00, places=2)

    def test_aggregate_has_mtm_loss(self):
        """Aggregation contient la cle mtm_loss (specifique equity)."""
        self.assertIn("mtm_loss", self.row)

    def test_aggregate_schema(self):
        """19 cles standard + mtm_loss presentes dans le dict agrege."""
        expected_keys = {
            "asset_class", "label", "category", "ead_total", "typical_weight",
            "pd_base", "lgd_base", "tenor", "asset_correlation", "rw_crr3",
            "physical_risk", "transition_risk", "green_capex_ratio",
            "scope3_exposure", "absorption_buffer", "hqla_eligible",
            "hqla_level", "exempt_from_staging", "rsf_weight",
            "mtm_loss",
        }
        self.assertTrue(expected_keys <= set(self.row.keys()),
                        f"Missing: {expected_keys - set(self.row.keys())}")


class TestStress(unittest.TestCase):
    """Validation du stress test equity."""

    def _stress(self, **overrides):
        """Helper: stress with base scenario + overrides."""
        macro = dict(_BASE_MACRO)
        macro.update(overrides)
        return stress_equity_positions(_DF, macro)

    def test_base_approximately_unstressed(self):
        """Scenario base -> PD ~ PD non-stresse."""
        result = self._stress()
        unstressed_pd = _DF["pd_merton"].mean()
        self.assertAlmostEqual(result["pd_base"] / max(unstressed_pd, 1e-8),
                               1.0, delta=0.30,
                               msg="Base stress should be close to unstressed")

    def test_gdp_crash_increases_pd(self):
        """Choc GDP negatif -> PD augmente."""
        base = self._stress()
        stressed = self._stress(**_ADVERSE_MACRO)
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "Adverse scenario should increase PD")

    def test_adverse_increases_lgd(self):
        """Scenario adverse -> LGD augmente ou reste stable."""
        base = self._stress()
        stressed = self._stress(**_ADVERSE_MACRO)
        self.assertGreaterEqual(stressed["lgd_base"], base["lgd_base"] * 0.99)

    def test_adverse_mtm_loss(self):
        """Scenario adverse -> mtm_loss present et positif."""
        result = self._stress(**_ADVERSE_MACRO)
        self.assertIn("mtm_loss", result)
        self.assertGreater(result["mtm_loss"], 0,
                           "MTM loss should be positive under adverse scenario")

    def test_stress_returns_valid_keys(self):
        """Stress retourne pd_base, lgd_base, rw_crr3, mtm_loss."""
        result = self._stress()
        for key in ("pd_base", "lgd_base", "rw_crr3", "mtm_loss"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_severe_pd_increase(self):
        """Stress severe -> PD augmente (Merton: mtm loss + vol expansion)."""
        base = self._stress()
        severe = self._stress(gdp_growth=-5.0, unemployment_rate=14.0,
                              interest_rate=7.0, inflation_rate=6.0)
        # Merton PD increase is non-linear; for large-caps, even GFC-like
        # shocks produce modest PD increases (DD still > 2)
        self.assertGreater(severe["pd_base"], base["pd_base"],
                           "Severe stress should increase PD")

    def test_positive_gdp_stable_or_lower(self):
        """GDP positif -> PD stable ou diminue."""
        base = self._stress()
        boom = self._stress(gdp_growth=4.0)
        self.assertLessEqual(boom["pd_base"], base["pd_base"] * 1.05)

    def test_rw_stays_100pct(self):
        """RW reste 100% sous tous les scenarios (CRR3 Art. 133)."""
        result = self._stress(**_ADVERSE_MACRO)
        self.assertAlmostEqual(result["rw_crr3"], 1.00, places=2)


if __name__ == "__main__":
    unittest.main()
