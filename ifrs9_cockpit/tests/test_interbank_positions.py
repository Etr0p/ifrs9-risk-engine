"""Tests pour le generateur position-par-position Interbancaire.

Couvre :
    - Generation : shape, colonnes, EAD scaling, distributions contreparties/tenor/instruments
    - PD multi-composante : range, calibration, ordering, collateral effect
    - LGD waterfall : range, calibration, repo/deposit/CB, BRRD
    - RW CRR3 Art. 120 : range, short-term benefit, collateral reduction
    - Aggregation : EAD, PD/LGD/RW blended, schema
    - Stress : 4 canaux (LIBOR-OIS, GDP, liquidity freeze, contagion),
               GFC simulation, LGD downturn, RW Art. 120(2)
"""

import pytest

pytestmark = pytest.mark.fourteen

import unittest
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from ifrs9_cockpit.synthetic_generator.interbank_positions import (
    INTERBANK_COUNTERPARTIES,
    TENOR_BUCKETS,
    INSTRUMENT_TYPES,
    RATING_PD_MAP,
    CRR3_ART120_RW,
    generate_interbank_positions,
    compute_interbank_pd,
    compute_interbank_lgd,
    compute_interbank_rw,
    stress_interbank_positions,
    aggregate_interbank_to_balance_row,
)
from ifrs9_cockpit.config import ASSET_CLASS_MAP


# Shared fixture: generate once for the module
_N = 1500
_TOTAL_EAD = 2.5e9  # 2.5B EUR
_SEED = 962
_DF = generate_interbank_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
_PROFILE = ASSET_CLASS_MAP["interbank"]


class TestInterbankGeneration(unittest.TestCase):
    """Validation du generateur de positions interbancaires."""

    def test_shape(self):
        """n_positions lignes generees."""
        self.assertEqual(len(_DF), _N)

    def test_columns_present(self):
        """Toutes les colonnes requises sont presentes."""
        required = {
            "position_id", "counterparty_code", "counterparty_name",
            "country", "rating", "tier1_ratio", "tenor_bucket",
            "tenor_days", "tenor_years", "instrument_type",
            "notional", "collateral_ratio", "haircut", "rate",
            "rw_crr3", "hqla_eligible", "hqla_level",
            "exempt_from_staging", "ead", "pd_position",
            "lgd_position",
        }
        self.assertTrue(required <= set(_DF.columns),
                        f"Missing: {required - set(_DF.columns)}")

    def test_ead_scaling(self):
        """sum(ead) ~ total_ead."""
        actual = _DF["ead"].sum()
        self.assertAlmostEqual(actual / _TOTAL_EAD, 1.0, places=2)

    def test_counterparty_distribution(self):
        """25 contreparties, DB/BNP en tete (poids 8%)."""
        counterparties = set(_DF["counterparty_code"].unique().to_list())
        expected = set(INTERBANK_COUNTERPARTIES.keys())
        self.assertEqual(counterparties, expected)
        counts = _DF["counterparty_code"].value_counts().sort("count", descending=True)
        # DB and BNP have highest weight (0.08 each)
        top2 = set(counts["counterparty_code"][:2].to_list())
        self.assertTrue({"DB", "BNP"} & top2,
                        f"Expected DB or BNP in top 2, got {top2}")

    def test_tenor_distribution(self):
        """Overnight dominant (~35%)."""
        vc = _DF["tenor_bucket"].value_counts()
        total = vc["count"].sum()
        vc = vc.with_columns((pl.col("count") / total).alias("proportion"))
        vc_dict = dict(zip(vc["tenor_bucket"].to_list(), vc["proportion"].to_list()))
        self.assertGreater(vc_dict.get("overnight", 0), 0.20)

    def test_instrument_distribution(self):
        """Unsecured deposit dominant (~45%)."""
        vc = _DF["instrument_type"].value_counts()
        total = vc["count"].sum()
        vc = vc.with_columns((pl.col("count") / total).alias("proportion"))
        vc_dict = dict(zip(vc["instrument_type"].to_list(), vc["proportion"].to_list()))
        self.assertGreater(vc_dict.get("unsecured_deposit", 0), 0.30)

    def test_tenor_range(self):
        """Tenor dans [0.003, 0.25]."""
        self.assertGreaterEqual(_DF["tenor_years"].min(), 0.001)
        self.assertLessEqual(_DF["tenor_years"].max(), 0.30)

    def test_collateral_positive_for_repo(self):
        """Repo et swap ont collateral_ratio > 0."""
        repos = _DF.filter(pl.col("instrument_type") == "repo")
        if len(repos) > 0:
            self.assertTrue((repos["collateral_ratio"] > 0).all())
        swaps = _DF.filter(pl.col("instrument_type") == "swap_collateral")
        if len(swaps) > 0:
            self.assertTrue((swaps["collateral_ratio"] > 0).all())

    def test_reproducibility(self):
        """Meme seed -> meme resultat."""
        df2 = generate_interbank_positions(n_positions=_N, total_ead=_TOTAL_EAD, seed=_SEED)
        assert_frame_equal(_DF, df2)

    def test_all_have_rating(self):
        """Toutes les positions ont un rating valide."""
        valid_ratings = set(RATING_PD_MAP.keys())
        self.assertTrue(set(_DF["rating"].unique().to_list()) <= valid_ratings,
                        f"Invalid ratings: {set(_DF['rating'].unique().to_list()) - valid_ratings}")


class TestInterbankPD(unittest.TestCase):
    """Validation de la PD multi-composante interbancaire."""

    def test_pd_range(self):
        """PD dans [0.00001, 0.02]."""
        self.assertGreaterEqual(_DF["pd_position"].min(), 0.00001)
        self.assertLessEqual(_DF["pd_position"].max(), 0.02)

    def test_pd_mean_calibrated(self):
        """PD moyenne EAD-weighted entre 0.0001 et 0.005."""
        w = _DF["ead"].to_numpy() / _DF["ead"].sum()
        pd_mean = float(np.dot(w, _DF["pd_position"].to_numpy()))
        self.assertGreater(pd_mean, 0.0001)
        self.assertLess(pd_mean, 0.005)

    def test_pd_rating_ordering(self):
        """BBB PD mediane > A PD mediane > AA PD mediane."""
        pd_by_rating = (
            _DF.group_by("rating")
            .agg(pl.col("pd_position").median().alias("pd_median"))
        )
        rating_dict = dict(zip(
            pd_by_rating["rating"].to_list(),
            pd_by_rating["pd_median"].to_list(),
        ))
        if "AA-" in rating_dict and "A" in rating_dict:
            self.assertLess(rating_dict["AA-"], rating_dict["A"])
        if "A" in rating_dict and "BBB" in rating_dict:
            self.assertLess(rating_dict["A"], rating_dict["BBB"])

    def test_pd_repo_lower_than_deposit(self):
        """PD repo < PD deposit (collateral protection)."""
        pd_repo = _DF.filter(pl.col("instrument_type") == "repo")["pd_position"].median()
        pd_dep = _DF.filter(pl.col("instrument_type") == "unsecured_deposit")["pd_position"].median()
        self.assertLess(pd_repo, pd_dep)

    def test_pd_overnight_lower(self):
        """PD overnight < PD quarterly (tenor effect)."""
        pd_on = _DF.filter(pl.col("tenor_bucket") == "overnight")["pd_position"].median()
        pd_q = _DF.filter(pl.col("tenor_bucket") == "quarterly")["pd_position"].median()
        self.assertLess(pd_on, pd_q)

    def test_pd_central_bank_lowest(self):
        """PD central_bank_facility la plus basse (alpha_type=0.05)."""
        pd_cb = _DF.filter(pl.col("instrument_type") == "central_bank_facility")["pd_position"].median()
        pd_other = _DF.filter(pl.col("instrument_type") != "central_bank_facility")["pd_position"].median()
        self.assertLess(pd_cb, pd_other)


class TestInterbankLGD(unittest.TestCase):
    """Validation de la LGD waterfall (BRRD-aware)."""

    def test_lgd_range(self):
        """LGD dans [0.02, 0.65]."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.02)
        self.assertLessEqual(_DF["lgd_position"].max(), 0.65)

    def test_lgd_mean_calibrated(self):
        """LGD moyenne entre 0.10 et 0.55 (Lehman/WaMu calibration)."""
        lgd_mean = _DF["lgd_position"].mean()
        self.assertGreater(lgd_mean, 0.10)
        self.assertLess(lgd_mean, 0.55)

    def test_lgd_repo_lowest(self):
        """LGD repo la plus basse (collateral 105%, haircut 2%)."""
        lgd_repo = _DF.filter(pl.col("instrument_type") == "repo")["lgd_position"].median()
        lgd_dep = _DF.filter(pl.col("instrument_type") == "unsecured_deposit")["lgd_position"].median()
        self.assertLess(lgd_repo, lgd_dep)

    def test_lgd_deposit_highest(self):
        """LGD unsecured_deposit la plus haute (pas de collateral)."""
        lgd_dep = _DF.filter(pl.col("instrument_type") == "unsecured_deposit")["lgd_position"].median()
        lgd_repo = _DF.filter(pl.col("instrument_type") == "repo")["lgd_position"].median()
        lgd_swap = _DF.filter(pl.col("instrument_type") == "swap_collateral")["lgd_position"].median()
        self.assertGreater(lgd_dep, lgd_repo)
        self.assertGreater(lgd_dep, lgd_swap)

    def test_lgd_central_bank_near_zero(self):
        """LGD central_bank_facility ~ 0 (fully collateralized)."""
        lgd_cb = _DF.filter(pl.col("instrument_type") == "central_bank_facility")["lgd_position"].median()
        self.assertLess(lgd_cb, 0.10)

    def test_lgd_floor(self):
        """LGD >= 2% (floor)."""
        self.assertGreaterEqual(_DF["lgd_position"].min(), 0.02)


class TestInterbankRW(unittest.TestCase):
    """Validation du RW CRR3 Art. 120 ECRA."""

    def test_rw_range(self):
        """RW dans [0.0, 1.50]."""
        self.assertGreaterEqual(_DF["rw_crr3"].min(), 0.0)
        self.assertLessEqual(_DF["rw_crr3"].max(), 1.50)

    def test_rw_mean_calibrated(self):
        """RW moyen EAD-weighted entre 0.05 et 0.35."""
        w = _DF["ead"].to_numpy() / _DF["ead"].sum()
        rw_mean = float(np.dot(w, _DF["rw_crr3"].to_numpy()))
        self.assertGreater(rw_mean, 0.05)
        self.assertLess(rw_mean, 0.35)

    def test_rw_repo_lower(self):
        """RW repo < RW deposit (CRM Art. 222)."""
        rw_repo = _DF.filter(pl.col("instrument_type") == "repo")["rw_crr3"].median()
        rw_dep = _DF.filter(pl.col("instrument_type") == "unsecured_deposit")["rw_crr3"].median()
        self.assertLess(rw_repo, rw_dep)

    def test_rw_central_bank_zero(self):
        """RW central_bank_facility = 0%."""
        rw_cb = _DF.filter(pl.col("instrument_type") == "central_bank_facility")["rw_crr3"]
        if len(rw_cb) > 0:
            self.assertTrue((rw_cb == 0.0).all())


class TestAggregation(unittest.TestCase):
    """Validation de l'aggregation balance sheet."""

    def test_aggregate_ead(self):
        """EAD total agrege ~ total_ead."""
        row = aggregate_interbank_to_balance_row(_DF, _PROFILE)
        self.assertAlmostEqual(row["ead_total"] / _TOTAL_EAD, 1.0, places=2)

    def test_aggregate_pd_range(self):
        """PD agregee dans une fourchette raisonnable."""
        row = aggregate_interbank_to_balance_row(_DF, _PROFILE)
        self.assertGreater(row["pd_base"], 0.00001)
        self.assertLess(row["pd_base"], 0.01)

    def test_aggregate_lgd_range(self):
        """LGD agregee dans une fourchette raisonnable."""
        row = aggregate_interbank_to_balance_row(_DF, _PROFILE)
        self.assertGreater(row["lgd_base"], 0.05)
        self.assertLess(row["lgd_base"], 0.60)

    def test_aggregate_rw_range(self):
        """RW agrege dans une fourchette raisonnable."""
        row = aggregate_interbank_to_balance_row(_DF, _PROFILE)
        self.assertGreater(row["rw_crr3"], 0.05)
        self.assertLess(row["rw_crr3"], 0.35)

    def test_aggregate_schema(self):
        """19 cles presentes dans le dict agrege."""
        row = aggregate_interbank_to_balance_row(_DF, _PROFILE)
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
    """Validation du stress test 4 canaux (EBA-calibre)."""

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
        return stress_interbank_positions(_DF, base)

    def test_base_unchanged(self):
        """Scenario base -> PD ~ PD non-stresse."""
        result = self._stress()
        row = aggregate_interbank_to_balance_row(_DF, _PROFILE)
        # Should be very close to unstressed
        self.assertAlmostEqual(result["pd_base"], row["pd_base"], places=5)

    def test_rate_hike_increases_pd(self):
        """Hausse taux -> PD augmente (LIBOR-OIS channel)."""
        base = self._stress()
        stressed = self._stress(interest_rate=6.5)  # +300bp
        self.assertGreater(stressed["pd_base"], base["pd_base"])

    def test_gdp_shock_increases_pd(self):
        """Choc GDP negatif -> PD augmente (Glasserman-Young channel)."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-3.8)
        self.assertGreater(stressed["pd_base"], base["pd_base"])

    def test_lgd_downturn(self):
        """GDP negatif -> LGD augmente (+5-10pts en recession)."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-5.0)
        self.assertGreater(stressed["lgd_base"], base["lgd_base"])

    def test_severe_combined(self):
        """Stress severe combine -> PD > 2x base."""
        base = self._stress()
        severe = self._stress(gdp_growth=-5.0, interest_rate=7.0)
        self.assertGreater(severe["pd_base"], base["pd_base"] * 2.0,
                           "Severe stress should at least double PD")

    def test_rate_cut_reduces_pd(self):
        """Baisse taux -> PD diminue."""
        base = self._stress()
        eased = self._stress(interest_rate=1.5)  # -200bp
        self.assertLessEqual(eased["pd_base"], base["pd_base"])

    def test_positive_gdp_stable(self):
        """GDP positif -> PD stable ou diminue."""
        base = self._stress()
        boom = self._stress(gdp_growth=4.0)
        self.assertLessEqual(boom["pd_base"], base["pd_base"] * 1.01)

    def test_interbank_freeze(self):
        """GFC-like (GDP -5%, rates +300bp) -> PD >= 3x base (4 canaux actifs)."""
        base = self._stress()
        gfc = self._stress(gdp_growth=-3.8, interest_rate=6.5)
        ratio = gfc["pd_base"] / max(base["pd_base"], 1e-8)
        self.assertGreater(ratio, 3.0,
                           f"GFC freeze should triple PD, got ratio={ratio:.1f}")

    def test_liquidity_crunch(self):
        """Taux + GDP negatif -> amplification non-lineaire (canal 3)."""
        # Only rate hike
        rate_only = self._stress(interest_rate=6.5)
        # Rate hike + GDP crash (activates freeze_indicator)
        combined = self._stress(interest_rate=6.5, gdp_growth=-3.8)
        # Combined should be more than additive
        self.assertGreater(combined["pd_base"], rate_only["pd_base"] * 1.5,
                           "Liquidity crunch should amplify rate effect")

    def test_short_term_rw_benefit_lost(self):
        """Art. 120(2) short-term 20% benefit perdu en stress freeze."""
        base = self._stress()
        freeze = self._stress(interest_rate=6.5, gdp_growth=-3.8)
        # RW should increase when freeze removes short-term benefit
        self.assertGreaterEqual(freeze["rw_crr3"], base["rw_crr3"])


if __name__ == "__main__":
    unittest.main()
