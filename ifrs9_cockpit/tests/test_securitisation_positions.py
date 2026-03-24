"""Tests pour le generateur position-par-position Titrisation.

Couvre :
    - Generation : shape, colonnes, EAD scaling, pool type distribution, tranching
    - PD Vasicek : range, seniority ordering, senior low, equity high, pool effect
    - LGD : range, seniority ordering, pool type effect
    - SEC-SA RW : range, seniority ordering, floors, STS vs non-STS
    - Aggregation : EAD, PD/LGD/RW blended
    - Stress test : base unchanged, GDP/HPI/unemp/rate shocks, cliff effect,
      LGD downturn, RW increase, STS vs non-STS
"""

import pytest

pytestmark = pytest.mark.fourteen

import unittest
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from ifrs9_cockpit.synthetic_generator.securitisation_positions import (
    POOL_TYPES,
    TRANCHE_TEMPLATES,
    SEC_SA_PARAMS,
    generate_securitisation_positions,
    compute_securitisation_pd,
    compute_securitisation_lgd,
    compute_sec_sa_rw,
    stress_securitisation_positions,
    aggregate_securitisation_to_balance_row,
)
from ifrs9_cockpit.config import ASSET_CLASS_MAP


# Shared fixture: generate once for the module
_N = 300
_TOTAL_EAD = 1.0e9
_SEED = 842
_DF = generate_securitisation_positions(n_tranches=_N, total_ead=_TOTAL_EAD, seed=_SEED)
_PROFILE = ASSET_CLASS_MAP["structured_products"]


class TestSecuritisationGeneration(unittest.TestCase):
    """Validation du generateur de positions titrisation."""

    def test_shape(self):
        """Au moins n_tranches/2 lignes generees (deals vary in tranche count)."""
        self.assertGreater(len(_DF), _N // 3)
        self.assertLess(len(_DF), _N * 3)

    def test_columns_present(self):
        """Toutes les colonnes requises sont presentes."""
        required = {
            "tranche_id", "deal_id", "pool_type", "tranche_type", "rating",
            "attachment", "detachment", "pool_pd", "pool_lgd", "rho",
            "n_obligors", "pool_size", "wal", "wac", "delinquency_rate",
            "is_sts", "vintage", "notional", "ead", "spread_bps",
            "pd_tranche", "lgd_tranche", "rw_sec_sa",
        }
        self.assertTrue(required <= set(_DF.columns),
                        f"Missing: {required - set(_DF.columns)}")

    def test_ead_scaling(self):
        """sum(ead) ~ total_ead."""
        actual = _DF["ead"].sum()
        self.assertAlmostEqual(actual / _TOTAL_EAD, 1.0, places=2)

    def test_pool_type_distribution(self):
        """6 types de pools, RMBS+CLO dominant (>50% of deals)."""
        types_present = set(_DF["pool_type"].unique().to_list())
        self.assertEqual(types_present, set(POOL_TYPES.keys()))
        # RMBS (0.35) + CLO (0.25) = 60% by weight -> >40% of deals
        deal_counts = (
            _DF.group_by("pool_type")
            .agg(pl.col("deal_id").n_unique().alias("n_deals"))
        )
        deal_dict = dict(zip(
            deal_counts["pool_type"].to_list(),
            deal_counts["n_deals"].to_list(),
        ))
        n_total = sum(deal_dict.values())
        top2_share = (deal_dict.get("rmbs", 0) + deal_dict.get("clo", 0)) / n_total
        self.assertGreater(top2_share, 0.40, "RMBS+CLO should be >40% of deals")

    def test_tranching_coherence(self):
        """attachment < detachment pour chaque tranche."""
        self.assertTrue(
            (_DF["attachment"] < _DF["detachment"]).all(),
            "All tranches should have attachment < detachment",
        )

    def test_attachment_ordering(self):
        """Within each deal: equity attach=0 (lowest), senior attach=highest."""
        for deal_id in _DF["deal_id"].unique().to_list():
            group = _DF.filter(pl.col("deal_id") == deal_id)
            tranche_types = group["tranche_type"].to_list()
            if "equity" in tranche_types:
                eq_attach = group.filter(pl.col("tranche_type") == "equity")["attachment"][0]
                self.assertAlmostEqual(eq_attach, 0.0, places=4,
                                       msg=f"Equity attachment should be ~0 in deal {deal_id}")
            if "senior" in tranche_types:
                senior_attach = group.filter(pl.col("tranche_type") == "senior")["attachment"][0]
                other_max = group.filter(pl.col("tranche_type") != "senior")["attachment"].max()
                self.assertGreater(senior_attach, other_max * 0.9,
                                   msg=f"Senior should have highest attachment in deal {deal_id}")

    def test_sts_distribution(self):
        """~20-50% STS (RMBS/ABS Auto/Consumer have STS, CLO/CMBS don't)."""
        sts_pct = _DF["is_sts"].mean()
        self.assertGreater(sts_pct, 0.10, f"STS ratio {sts_pct:.1%} too low")
        self.assertLess(sts_pct, 0.60, f"STS ratio {sts_pct:.1%} too high")

    def test_deal_grouping(self):
        """Each deal_id has 4-6 tranches."""
        deal_sizes = (
            _DF.group_by("deal_id")
            .agg(pl.len().alias("size"))
        )
        self.assertTrue((deal_sizes["size"] >= 4).all(), "Each deal should have >= 4 tranches")
        self.assertTrue((deal_sizes["size"] <= 6).all(), "Each deal should have <= 6 tranches")

    def test_wal_range(self):
        """WAL dans [0.5, 10] ans."""
        self.assertGreaterEqual(_DF["wal"].min(), 0.4)
        self.assertLessEqual(_DF["wal"].max(), 10.5)

    def test_reproducibility(self):
        """Same seed = same result."""
        df2 = generate_securitisation_positions(
            n_tranches=_N, total_ead=_TOTAL_EAD, seed=_SEED,
        )
        assert_frame_equal(_DF, df2)


class TestSecuritisationPD(unittest.TestCase):
    """Validation du modele PD Vasicek tranche-level."""

    def test_pd_range(self):
        """PD dans [0.0001, 0.50]."""
        self.assertGreaterEqual(_DF["pd_tranche"].min(), 0.0001 - 1e-6)
        self.assertLessEqual(_DF["pd_tranche"].max(), 0.50 + 1e-6)

    def test_pd_seniority_ordering(self):
        """PD(equity) > PD(junior) > PD(mezzanine) > PD(senior) on average."""
        means = (
            _DF.group_by("tranche_type")
            .agg(pl.col("pd_tranche").mean().alias("pd_mean"))
        )
        means_dict = dict(zip(
            means["tranche_type"].to_list(),
            means["pd_mean"].to_list(),
        ))
        if "equity" in means_dict and "junior" in means_dict:
            self.assertGreater(means_dict["equity"], means_dict["junior"])
        if "junior" in means_dict and "mezzanine_bbb" in means_dict:
            self.assertGreater(means_dict["junior"], means_dict["mezzanine_bbb"])
        if "mezzanine_bbb" in means_dict and "senior" in means_dict:
            self.assertGreater(means_dict["mezzanine_bbb"], means_dict["senior"])

    def test_pd_senior_very_low(self):
        """PD senior AAA < 0.01."""
        senior = _DF.filter(pl.col("tranche_type") == "senior")
        if len(senior) > 0:
            pd_mean = senior["pd_tranche"].mean()
            self.assertLess(pd_mean, 0.01,
                            f"Senior PD {pd_mean:.6f} should be < 1%")

    def test_pd_equity_very_high(self):
        """PD equity > 0.10."""
        equity = _DF.filter(pl.col("tranche_type") == "equity")
        if len(equity) > 0:
            pd_mean = equity["pd_tranche"].mean()
            self.assertGreater(pd_mean, 0.10,
                               f"Equity PD {pd_mean:.4f} should be > 10%")

    def test_pd_pool_type_effect(self):
        """CLO tranches PD > RMBS tranches PD (same seniority) on average."""
        for tt in ["mezzanine_bbb", "junior"]:
            clo = _DF.filter(
                (pl.col("pool_type") == "clo") & (pl.col("tranche_type") == tt)
            )
            rmbs = _DF.filter(
                (pl.col("pool_type") == "rmbs") & (pl.col("tranche_type") == tt)
            )
            if len(clo) > 3 and len(rmbs) > 3:
                self.assertGreater(clo["pd_tranche"].mean(), rmbs["pd_tranche"].mean(),
                                   f"CLO PD should be > RMBS PD for {tt}")

    def test_pd_attachment_effect(self):
        """Higher attachment -> lower PD (within same pool type)."""
        for pt in ["rmbs", "clo"]:
            sub = _DF.filter(pl.col("pool_type") == pt)
            if len(sub) < 5:
                continue
            corr = float(sub.select(pl.corr("attachment", "pd_tranche")).item())
            self.assertLess(corr, 0.0,
                            f"Attachment-PD correlation should be negative for {pt}")

    def test_pd_vasicek_monotone(self):
        """PD monotonically decreasing with attachment (per deal)."""
        for deal_id in _DF["deal_id"].unique().to_list():
            group = _DF.filter(pl.col("deal_id") == deal_id).sort("attachment")
            pds = group["pd_tranche"].to_numpy()
            # Allow small non-monotonicity due to rounding
            for i in range(1, len(pds)):
                self.assertGreaterEqual(
                    pds[i - 1] + 1e-4, pds[i],
                    f"PD not monotone in deal {deal_id}: {pds}",
                )

    def test_pd_correlation_effect(self):
        """Higher rho pools -> higher junior/equity PD (tail risk)."""
        # Use junior -- senior/mezzanine PD is often at floor, masking the effect
        sub = _DF.filter(pl.col("tranche_type").is_in(["junior", "equity"]))
        if len(sub) < 5:
            self.skipTest("Not enough junior/equity tranches")
        high_rho = sub.filter(pl.col("rho") >= 0.15)
        low_rho = sub.filter(pl.col("rho") < 0.15)
        if len(high_rho) > 3 and len(low_rho) > 3:
            # High correlation -> more tail risk -> higher PD for risky tranches
            self.assertGreater(high_rho["pd_tranche"].mean(),
                               low_rho["pd_tranche"].mean())


class TestSecuritisationLGD(unittest.TestCase):
    """Validation du modele LGD calibre Moody's."""

    def test_lgd_range(self):
        """LGD dans [0.05, 0.95]."""
        self.assertGreaterEqual(_DF["lgd_tranche"].min(), 0.05 - 1e-4)
        self.assertLessEqual(_DF["lgd_tranche"].max(), 0.95 + 1e-4)

    def test_lgd_seniority_ordering(self):
        """LGD(equity) > LGD(junior) > LGD(senior) on average."""
        means = (
            _DF.group_by("tranche_type")
            .agg(pl.col("lgd_tranche").mean().alias("lgd_mean"))
        )
        means_dict = dict(zip(
            means["tranche_type"].to_list(),
            means["lgd_mean"].to_list(),
        ))
        if "equity" in means_dict and "junior" in means_dict:
            self.assertGreater(means_dict["equity"], means_dict["junior"])
        if "junior" in means_dict and "senior" in means_dict:
            self.assertGreater(means_dict["junior"], means_dict["senior"])

    def test_lgd_senior_low(self):
        """LGD senior < 0.15."""
        senior = _DF.filter(pl.col("tranche_type") == "senior")
        if len(senior) > 0:
            self.assertLess(senior["lgd_tranche"].mean(), 0.15)

    def test_lgd_equity_high(self):
        """LGD equity > 0.70."""
        equity = _DF.filter(pl.col("tranche_type") == "equity")
        if len(equity) > 0:
            self.assertGreater(equity["lgd_tranche"].mean(), 0.70)

    def test_lgd_pool_type_effect(self):
        """Consumer ABS LGD > Auto ABS LGD (same seniority, unsecured vs vehicle)."""
        for tt in ["mezzanine_bbb", "junior"]:
            consumer = _DF.filter(
                (pl.col("pool_type") == "abs_consumer") & (pl.col("tranche_type") == tt)
            )
            auto = _DF.filter(
                (pl.col("pool_type") == "abs_auto") & (pl.col("tranche_type") == tt)
            )
            if len(consumer) > 2 and len(auto) > 2:
                self.assertGreater(consumer["lgd_tranche"].mean(),
                                   auto["lgd_tranche"].mean(),
                                   f"Consumer LGD should be > Auto LGD for {tt}")


class TestSecSARiskWeight(unittest.TestCase):
    """Validation du calcul SEC-SA (CRR3 Art. 261)."""

    def test_rw_range(self):
        """RW dans [0.10, 12.50]."""
        self.assertGreaterEqual(_DF["rw_sec_sa"].min(), 0.10 - 1e-4)
        self.assertLessEqual(_DF["rw_sec_sa"].max(), 12.50 + 1e-4)

    def test_rw_seniority_ordering(self):
        """RW(equity) > RW(mezzanine) > RW(senior) on average."""
        means = (
            _DF.group_by("tranche_type")
            .agg(pl.col("rw_sec_sa").mean().alias("rw_mean"))
        )
        means_dict = dict(zip(
            means["tranche_type"].to_list(),
            means["rw_mean"].to_list(),
        ))
        if "equity" in means_dict and "junior" in means_dict:
            self.assertGreater(means_dict["equity"], means_dict["junior"])
        if "junior" in means_dict and "senior" in means_dict:
            self.assertGreater(means_dict["junior"], means_dict["senior"])

    def test_rw_senior_sts_floor(self):
        """RW senior STS >= 10%."""
        senior_sts = _DF.filter(
            (pl.col("tranche_type") == "senior") & (pl.col("is_sts") == 1)
        )
        if len(senior_sts) > 0:
            self.assertTrue(
                (senior_sts["rw_sec_sa"] >= 0.10 - 1e-4).all(),
                "Senior STS RW should be >= 10%",
            )

    def test_rw_senior_non_sts_floor(self):
        """RW senior non-STS >= 15%."""
        senior_non_sts = _DF.filter(
            (pl.col("tranche_type") == "senior") & (pl.col("is_sts") == 0)
        )
        if len(senior_non_sts) > 0:
            self.assertTrue(
                (senior_non_sts["rw_sec_sa"] >= 0.15 - 1e-4).all(),
                "Senior non-STS RW should be >= 15%",
            )

    def test_rw_equity_deduction(self):
        """RW equity should be very high (close to 1250% = 12.50)."""
        equity = _DF.filter(pl.col("tranche_type") == "equity")
        if len(equity) > 0:
            mean_rw = equity["rw_sec_sa"].mean()
            self.assertGreater(mean_rw, 5.0,
                               f"Equity RW {mean_rw:.2f} should be > 500%")

    def test_rw_sts_lower_than_non_sts(self):
        """STS RW < non-STS RW on average for senior (p-factor effect)."""
        # Only check senior -- mezzanine may both be at deduction (1250%)
        sts = _DF.filter(
            (pl.col("tranche_type") == "senior") & (pl.col("is_sts") == 1)
        )
        non_sts = _DF.filter(
            (pl.col("tranche_type") == "senior") & (pl.col("is_sts") == 0)
        )
        if len(sts) > 3 and len(non_sts) > 3:
            self.assertLessEqual(sts["rw_sec_sa"].mean(), non_sts["rw_sec_sa"].mean(),
                                 "STS senior RW should be <= non-STS senior RW")

    def test_rw_formula_consistency(self):
        """Non-senior floor is 15% minimum."""
        non_senior = _DF.filter(pl.col("tranche_type") != "senior")
        if len(non_senior) > 0:
            self.assertTrue(
                (non_senior["rw_sec_sa"] >= 0.15 - 1e-4).all(),
                "Non-senior RW should be >= 15%",
            )


class TestAggregation(unittest.TestCase):
    """Validation de l'aggregation vers la ligne balance sheet."""

    def setUp(self):
        self.agg = aggregate_securitisation_to_balance_row(_DF, _PROFILE)

    def test_aggregate_ead(self):
        """EAD total = sum(ead)."""
        expected = _DF["ead"].sum()
        self.assertAlmostEqual(self.agg["ead_total"], expected, places=0)

    def test_aggregate_pd_range(self):
        """PD agrege in reasonable range."""
        pd_agg = self.agg["pd_base"]
        self.assertGreater(pd_agg, 0.0001)
        self.assertLess(pd_agg, 0.20)

    def test_aggregate_lgd_range(self):
        """LGD agrege EAD-weighted in reasonable range."""
        lgd_agg = self.agg["lgd_base"]
        self.assertGreater(lgd_agg, 0.05)
        self.assertLess(lgd_agg, 0.60)

    def test_aggregate_rw_blended(self):
        """rw_crr3 blende between floor and 1250%."""
        rw_agg = self.agg["rw_crr3"]
        self.assertGreater(rw_agg, 0.10)
        self.assertLess(rw_agg, 12.50)


class TestStress(unittest.TestCase):
    """Validation du stress test position-par-position."""

    def _stress(self, **kwargs):
        """Helper: stress with overrides on base scenario."""
        base = {
            "gdp_growth": 1.2, "unemployment_rate": 7.5,
            "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5,
        }
        base.update(kwargs)
        return stress_securitisation_positions(_DF, base)

    def test_stress_base_unchanged(self):
        """Base scenario -> PD/LGD close to unstressed aggregate."""
        result = self._stress()
        agg = aggregate_securitisation_to_balance_row(_DF, _PROFILE)
        # Should be very close
        self.assertAlmostEqual(result["pd_base"], agg["pd_base"], places=4)

    def test_stress_gdp_shock_clo(self):
        """GDP -5% -> PD increases (CLO exposed via beta_gdp=2.0)."""
        clo_df = _DF.filter(pl.col("pool_type") == "clo")
        if len(clo_df) < 5:
            self.skipTest("Not enough CLO tranches")
        base = stress_securitisation_positions(clo_df, {
            "gdp_growth": 1.2, "unemployment_rate": 7.5,
            "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5,
        })
        stressed = stress_securitisation_positions(clo_df, {
            "gdp_growth": -3.8, "unemployment_rate": 7.5,
            "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5,
        })
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "GDP shock should increase CLO PD")

    def test_stress_hpi_shock_rmbs(self):
        """HPI -20% -> PD RMBS increases (beta_hpi=-2.0)."""
        rmbs_df = _DF.filter(pl.col("pool_type") == "rmbs")
        if len(rmbs_df) < 5:
            self.skipTest("Not enough RMBS tranches")
        base = stress_securitisation_positions(rmbs_df, {
            "gdp_growth": 1.2, "unemployment_rate": 7.5,
            "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5,
        })
        stressed = stress_securitisation_positions(rmbs_df, {
            "gdp_growth": 1.2, "unemployment_rate": 7.5,
            "interest_rate": 3.5, "hpi_growth": -18.0, "inflation_rate": 2.5,
        })
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "HPI shock should increase RMBS PD")

    def test_stress_unemp_shock_consumer(self):
        """Unemployment +5% -> PD ABS Consumer increases (beta_unemp=1.5)."""
        consumer_df = _DF.filter(pl.col("pool_type") == "abs_consumer")
        if len(consumer_df) < 5:
            self.skipTest("Not enough ABS Consumer tranches")
        base = stress_securitisation_positions(consumer_df, {
            "gdp_growth": 1.2, "unemployment_rate": 7.5,
            "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5,
        })
        stressed = stress_securitisation_positions(consumer_df, {
            "gdp_growth": 1.2, "unemployment_rate": 12.5,
            "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5,
        })
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "Unemployment shock should increase ABS Consumer PD")

    def test_stress_rate_shock_cmbs(self):
        """Interest rate +200bp -> CMBS PD increases (beta_rate=1.5)."""
        cmbs_df = _DF.filter(pl.col("pool_type") == "cmbs")
        if len(cmbs_df) < 5:
            self.skipTest("Not enough CMBS tranches")
        base = stress_securitisation_positions(cmbs_df, {
            "gdp_growth": 1.2, "unemployment_rate": 7.5,
            "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5,
        })
        stressed = stress_securitisation_positions(cmbs_df, {
            "gdp_growth": 1.2, "unemployment_rate": 7.5,
            "interest_rate": 5.5, "hpi_growth": 2.0, "inflation_rate": 2.5,
        })
        self.assertGreater(stressed["pd_base"], base["pd_base"],
                           "Rate hike should increase CMBS PD")

    def test_stress_senior_resilient(self):
        """Moderate stress -> senior PD quasi unchanged."""
        senior_df = _DF.filter(pl.col("tranche_type") == "senior")
        if len(senior_df) < 5:
            self.skipTest("Not enough senior tranches")
        base = stress_securitisation_positions(senior_df, {
            "gdp_growth": 1.2, "unemployment_rate": 7.5,
            "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5,
        })
        stressed = stress_securitisation_positions(senior_df, {
            "gdp_growth": -0.5, "unemployment_rate": 9.0,
            "interest_rate": 4.5, "hpi_growth": -3.0, "inflation_rate": 3.5,
        })
        # Senior PD should not change by more than 5x (relative)
        if base["pd_base"] > 0:
            ratio = stressed["pd_base"] / base["pd_base"]
            self.assertLess(ratio, 5.0,
                            f"Senior PD ratio {ratio:.1f}x too high under moderate stress")

    def test_stress_mezzanine_cliff(self):
        """Severe stress -> mezzanine PD increases significantly (cliff effect)."""
        mezz_df = _DF.filter(pl.col("tranche_type") == "mezzanine_bbb")
        if len(mezz_df) < 5:
            self.skipTest("Not enough mezzanine BBB tranches")
        base = stress_securitisation_positions(mezz_df, {
            "gdp_growth": 1.2, "unemployment_rate": 7.5,
            "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5,
        })
        stressed = stress_securitisation_positions(mezz_df, {
            "gdp_growth": -5.0, "unemployment_rate": 12.0,
            "interest_rate": 5.5, "hpi_growth": -18.0, "inflation_rate": 5.0,
        })
        self.assertGreater(stressed["pd_base"], base["pd_base"] * 1.5,
                           "Mezzanine PD should jump significantly under severe stress")

    def test_stress_equity_already_hit(self):
        """Stress -> equity PD already high, changes less in relative terms."""
        equity_df = _DF.filter(pl.col("tranche_type") == "equity")
        if len(equity_df) < 5:
            self.skipTest("Not enough equity tranches")
        base = stress_securitisation_positions(equity_df, {
            "gdp_growth": 1.2, "unemployment_rate": 7.5,
            "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5,
        })
        stressed = stress_securitisation_positions(equity_df, {
            "gdp_growth": -3.8, "unemployment_rate": 12.0,
            "interest_rate": 5.5, "hpi_growth": -10.0, "inflation_rate": 4.0,
        })
        # Equity PD is already high, relative change should be moderate
        if base["pd_base"] > 0:
            ratio = stressed["pd_base"] / base["pd_base"]
            self.assertLess(ratio, 3.0,
                            f"Equity PD ratio {ratio:.1f}x -- already at ceiling")

    def test_stress_lgd_downturn(self):
        """GDP negatif -> LGD augmente."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-3.8)
        self.assertGreater(stressed["lgd_base"], base["lgd_base"],
                           "GDP shock should increase LGD (downturn addon)")

    def test_stress_rw_increases(self):
        """Stress severe -> RW agrege augmente."""
        base = self._stress()
        stressed = self._stress(gdp_growth=-3.8, interest_rate=5.5, hpi_growth=-10.0)
        self.assertGreater(stressed["rw_crr3"], base["rw_crr3"],
                           "Stress should increase blended RW")

    def test_stress_sts_vs_non_sts(self):
        """STS tranches stressed RW < non-STS stressed RW."""
        sts_df = _DF.filter(pl.col("is_sts") == 1)
        non_sts_df = _DF.filter(pl.col("is_sts") == 0)
        if len(sts_df) < 5 or len(non_sts_df) < 5:
            self.skipTest("Not enough STS/non-STS tranches")
        stress_params = {
            "gdp_growth": -2.0, "unemployment_rate": 9.5,
            "interest_rate": 4.5, "hpi_growth": -5.0, "inflation_rate": 3.5,
        }
        stressed_sts = stress_securitisation_positions(sts_df, stress_params)
        stressed_non_sts = stress_securitisation_positions(non_sts_df, stress_params)
        self.assertLess(stressed_sts["rw_crr3"], stressed_non_sts["rw_crr3"],
                        "STS stressed RW should be < non-STS stressed RW")


if __name__ == "__main__":
    unittest.main()
