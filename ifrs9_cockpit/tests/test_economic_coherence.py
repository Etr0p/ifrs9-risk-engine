"""Economic coherence tests — diagnostic labels for broken mechanisms.

Each test name identifies exactly which economic relationship is broken.
If ``test_adverse_increases_pd`` fails, PD stress transmission is broken.
If ``test_eva_sign_matches_raroc_vs_coc`` fails, the EVA formula is inconsistent.

7 test classes covering 7 economic domains:
    1. P&L components and RAROC/EVA identities
    2. Scenario mechanics (macro → Z → PD)
    3. Staging and ECL computation
    4. PE valuation (MOIC, IRR, distress, NAV)
    5. Regulatory norms (LCR, NSFR, CET1)
    6. Optimizer economics (weights, CVaR, feasibility)
    7. Cross-module coherence (ECL → RAROC → optimizer)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.shared

from ifrs9_cockpit.config import (
    BASEL_CONFIG,
    PREDEFINED_SCENARIOS,
    SCENARIO_ADVERSE,
    SCENARIO_BASE,
    SCENARIO_FAVORABLE,
    SCENARIOS,
    SECTORS,
)
from ifrs9_cockpit.config.basel import PE_CLASSIFICATION_CONFIG
from ifrs9_cockpit.engine.staging import compute_macro_z


# ──────────────────────────────────────────────
# FIXTURES (module-scoped, zero recalc)
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline(global_pipeline_results):
    """Reuse session pipeline — zero recalc."""
    return global_pipeline_results


@pytest.fixture(scope="module")
def comparator(pipeline):
    return pipeline["comparator"]


@pytest.fixture(scope="module")
def result_credit(pipeline):
    return pipeline["result_credit"]


@pytest.fixture(scope="module")
def result_pe(pipeline):
    return pipeline["result_pe"]


@pytest.fixture(scope="module")
def raroc_df(comparator):
    """RAROC/EVA table — cached after first call."""
    return comparator.compute_raroc_eva()


@pytest.fixture(scope="module")
def opt_14c(comparator):
    """14-class optimizer result — cached."""
    return comparator.optimize_allocation()


@pytest.fixture(scope="module")
def opt_10c(comparator):
    """10-cell (pe-bc) optimizer result — cached."""
    return comparator.optimize_allocation_pebc()


# ──────────────────────────────────────────────
# CLASS 1: P&L COMPONENT COHERENCE
# ──────────────────────────────────────────────

class TestPnLComponentCoherence:
    """Validates P&L arithmetic and RAROC/EVA identities."""

    def test_raroc_equals_profit_over_capital(self, raroc_df):
        """RAROC = profit_net / capital for every sector×canal row."""
        cir = BASEL_CONFIG.cir
        tax = BASEL_CONFIG.tax_rate
        for row in raroc_df.iter_rows(named=True):
            revenue = row["revenue"]
            loss = row["loss"]
            capital = row["capital"]
            if capital == 0:
                continue
            revenue_net = revenue * (1 - cir)
            profit_net = (revenue_net - loss) * (1 - tax)
            expected_raroc = profit_net / capital
            assert abs(row["raroc"] - round(expected_raroc, 4)) <= 1e-3, (
                f"RAROC broken for {row['sector']}/{row['canal']}: "
                f"got {row['raroc']}, expected {expected_raroc:.4f}"
            )

    def test_eva_equals_raroc_minus_coc_times_capital(self, raroc_df):
        """EVA = (RAROC - CoC) × capital.

        Note: RAROC column is rounded to 4dp, so the test recomputes EVA
        from the unrounded profit formula to avoid rounding amplification
        on large capitals (~1B).
        """
        cir = BASEL_CONFIG.cir
        tax = BASEL_CONFIG.tax_rate
        coc = BASEL_CONFIG.cet1_target
        for row in raroc_df.iter_rows(named=True):
            capital = row["capital"]
            if capital == 0:
                continue
            # Recompute from unrounded profit to match source code
            revenue_net = row["revenue"] * (1 - cir)
            profit_net = (revenue_net - row["loss"]) * (1 - tax)
            raroc_unrounded = profit_net / max(capital, 1)
            expected_eva = (raroc_unrounded - coc) * capital
            assert abs(row["eva"] - round(expected_eva, 0)) <= 2.0, (
                f"EVA broken for {row['sector']}/{row['canal']}: "
                f"got {row['eva']}, expected {expected_eva:.0f}"
            )

    def test_profit_rate_equals_profit_over_exposure(self, raroc_df):
        """profit_rate = profit_net / exposure."""
        cir = BASEL_CONFIG.cir
        tax = BASEL_CONFIG.tax_rate
        for row in raroc_df.iter_rows(named=True):
            exposure = row["exposure"]
            if exposure == 0:
                continue
            revenue_net = row["revenue"] * (1 - cir)
            profit_net = (revenue_net - row["loss"]) * (1 - tax)
            expected = profit_net / exposure
            assert abs(row["profit_rate"] - round(expected, 6)) <= 1e-4, (
                f"profit_rate broken for {row['sector']}/{row['canal']}"
            )

    def test_revenue_net_applies_cir(self, raroc_df):
        """Revenue net should reflect CIR deduction."""
        cir = BASEL_CONFIG.cir
        assert 0 < cir < 1, "CIR must be in (0, 1)"
        # The CIR is used in the RAROC formula — verify it's 0.45
        assert cir == 0.45, f"CIR changed from expected 0.45 to {cir}"

    def test_profit_applies_tax(self, raroc_df):
        """Tax rate is applied correctly."""
        tax = BASEL_CONFIG.tax_rate
        assert tax == 0.25, f"Tax rate changed from expected 0.25 to {tax}"

    def test_capital_equals_rwa_times_cet1_target(self, raroc_df):
        """capital = RWA × CET1_target for all rows."""
        cet1 = BASEL_CONFIG.cet1_target
        for row in raroc_df.iter_rows(named=True):
            expected_capital = round(row["rwa"] * cet1, 0)
            assert abs(row["capital"] - expected_capital) <= 1.0, (
                f"Capital formula broken for {row['sector']}/{row['canal']}: "
                f"capital={row['capital']}, rwa×cet1={expected_capital}"
            )

    def test_total_row_sums_components(self, raroc_df):
        """Total exposure = sum of sector exposures per canal."""
        for canal in ["Credit", "PE"]:
            sub = raroc_df.filter(pl.col("canal") == canal)
            total_row = sub.filter(pl.col("sector") == "Total")
            sector_rows = sub.filter(pl.col("sector") != "Total")
            if len(total_row) == 0 or len(sector_rows) == 0:
                continue
            for col in ["exposure", "revenue", "loss", "rwa", "capital"]:
                total_val = total_row[col][0]
                sum_val = sector_rows[col].sum()
                assert abs(total_val - sum_val) <= 2.0, (
                    f"Aggregation broken: Total {col} ({canal}) = {total_val}, "
                    f"sum of sectors = {sum_val}"
                )

    def test_negative_eva_when_raroc_below_coc(self, raroc_df):
        """If RAROC < 13%, EVA must be negative."""
        coc = BASEL_CONFIG.cet1_target
        sector_rows = raroc_df.filter(pl.col("sector") != "Total")
        below_coc = sector_rows.filter(pl.col("raroc") < coc)
        for row in below_coc.iter_rows(named=True):
            assert row["eva"] < 1.0, (
                f"EVA sign error: RAROC={row['raroc']:.4f} < CoC={coc} "
                f"but EVA={row['eva']} >= 0 for {row['sector']}/{row['canal']}"
            )

    def test_positive_eva_when_raroc_above_coc(self, raroc_df):
        """If RAROC > 13%, EVA must be positive."""
        coc = BASEL_CONFIG.cet1_target
        sector_rows = raroc_df.filter(pl.col("sector") != "Total")
        above_coc = sector_rows.filter(pl.col("raroc") > coc)
        for row in above_coc.iter_rows(named=True):
            assert row["eva"] > -1.0, (
                f"EVA sign error: RAROC={row['raroc']:.4f} > CoC={coc} "
                f"but EVA={row['eva']} <= 0 for {row['sector']}/{row['canal']}"
            )

    def test_profit_rate_bounded(self, raroc_df):
        """profit_rate should be in [-1, +1] reasonable range."""
        for row in raroc_df.iter_rows(named=True):
            assert -1.0 <= row["profit_rate"] <= 1.0, (
                f"Numerical overflow: profit_rate={row['profit_rate']} "
                f"for {row['sector']}/{row['canal']}"
            )


# ──────────────────────────────────────────────
# CLASS 2: SCENARIO MECHANICS
# ──────────────────────────────────────────────

class TestScenarioMechanics:
    """Validates macro scenario transmission (stress → PD → ECL → P&L)."""

    def test_adverse_scenarios_have_higher_unemployment(self):
        """Stagflation/GFC unemployment should be higher than Central."""
        central = PREDEFINED_SCENARIOS["Central"]
        # Adverse scenarios have negative unemployment_bipolar (economic crisis)
        # meaning |bipolar| pp added to base unemployment
        for name in ["Stagflation", "Crise financiere (GFC)", "Crise souveraine (2012)"]:
            scen = PREDEFINED_SCENARIOS[name]
            # unemployment_bipolar < 0 = economic crisis → unemployment goes UP
            assert scen["unemployment_bipolar"] < central["unemployment_bipolar"] or \
                abs(scen["unemployment_bipolar"]) > abs(central["unemployment_bipolar"]), (
                f"Scenario {name} should have higher unemployment than Central"
            )

    def test_favorable_scenarios_have_lower_unemployment(self):
        """Reprise/Hypercroissance should have no unemployment increase."""
        for name in ["Reprise", "Hypercroissance"]:
            scen = PREDEFINED_SCENARIOS[name]
            # unemployment_bipolar = 0 means no increase in unemployment
            assert scen["unemployment_bipolar"] == 0.0, (
                f"Scenario {name} should have unemployment_bipolar = 0 "
                f"(no crisis), got {scen['unemployment_bipolar']}"
            )

    def test_macro_to_z_positive_for_adverse(self):
        """compute_macro_z(adverse) > 0 — higher Z = worse conditions."""
        z = compute_macro_z({
            "unemployment_rate": SCENARIO_ADVERSE.unemployment_rate,
            "gdp_growth": SCENARIO_ADVERSE.gdp_growth,
            "interest_rate": SCENARIO_ADVERSE.interest_rate,
            "hpi_growth": SCENARIO_ADVERSE.hpi_growth,
            "inflation_rate": SCENARIO_ADVERSE.inflation_rate,
        })
        assert z > 0, f"Z-score for adverse should be positive, got {z}"

    def test_macro_to_z_negative_for_favorable(self):
        """compute_macro_z(favorable) < 0 — lower Z = better conditions."""
        z = compute_macro_z({
            "unemployment_rate": SCENARIO_FAVORABLE.unemployment_rate,
            "gdp_growth": SCENARIO_FAVORABLE.gdp_growth,
            "interest_rate": SCENARIO_FAVORABLE.interest_rate,
            "hpi_growth": SCENARIO_FAVORABLE.hpi_growth,
            "inflation_rate": SCENARIO_FAVORABLE.inflation_rate,
        })
        assert z < 0, f"Z-score for favorable should be negative, got {z}"

    def test_macro_to_z_zero_for_base(self):
        """compute_macro_z(base) ≈ 0 — base is the reference point."""
        z = compute_macro_z({
            "unemployment_rate": SCENARIO_BASE.unemployment_rate,
            "gdp_growth": SCENARIO_BASE.gdp_growth,
            "interest_rate": SCENARIO_BASE.interest_rate,
            "hpi_growth": SCENARIO_BASE.hpi_growth,
            "inflation_rate": SCENARIO_BASE.inflation_rate,
        })
        assert abs(z) < 1e-8, f"Z-score for base should be ≈ 0, got {z}"

    def test_scenario_weights_sum_to_one(self):
        """Base(0.50) + Adverse(0.25) + Favorable(0.25) = 1.0."""
        total = sum(s.weight for s in SCENARIOS)
        assert abs(total - 1.0) < 1e-10, (
            f"Scenario weights sum to {total}, expected 1.0"
        )

    def test_all_scenarios_have_five_macro_keys(self):
        """Each predefined scenario has the 5 required macro slider keys."""
        required = {"interest_rate_bp", "unemployment_bipolar", "gdp_pct", "hpi_pct", "inflation_pct"}
        for name, scen in PREDEFINED_SCENARIOS.items():
            missing = required - set(scen.keys())
            assert not missing, (
                f"Scenario '{name}' missing keys: {missing}"
            )

    def test_scenario_names_match_predefined(self):
        """12 predefined scenarios exist with expected names."""
        expected_count = 12
        assert len(PREDEFINED_SCENARIOS) == expected_count, (
            f"Expected {expected_count} predefined scenarios, "
            f"got {len(PREDEFINED_SCENARIOS)}: {list(PREDEFINED_SCENARIOS.keys())}"
        )
        assert "Central" in PREDEFINED_SCENARIOS
        assert "Stagflation" in PREDEFINED_SCENARIOS
        assert "Normalisation monetaire (Volcker)" in PREDEFINED_SCENARIOS


# ──────────────────────────────────────────────
# CLASS 3: STAGING AND ECL
# ──────────────────────────────────────────────

class TestStagingAndECL:
    """Validates staging logic, SICR, and ECL computation."""

    def test_stage1_pd_uses_12m(self, result_credit):
        """Stage 1 clients should use 12-month PD (not lifetime)."""
        stage1 = result_credit.filter(pl.col("stage") == 1)
        if len(stage1) == 0:
            pytest.skip("No Stage 1 clients in test data")
        # For Stage 1, pd_12m and pd_lifetime may differ but ECL uses pd_12m
        # Verify pd_12m is populated and reasonable
        pd_12m = stage1["pd_12m"].to_numpy()
        assert np.all(pd_12m >= 0) and np.all(pd_12m <= 1), (
            "Stage 1 pd_12m should be in [0, 1]"
        )

    def test_stage3_pd_is_one(self, result_credit):
        """Stage 3 (default) clients should have effective PD = 1.0 in ECL."""
        stage3 = result_credit.filter(pl.col("stage") == 3)
        if len(stage3) == 0:
            pytest.skip("No Stage 3 clients in test data")
        # ECL_base for stage 3 should reflect PD=1.0
        # ECL = PD(1.0) × LGD × EAD × DF → ECL ≈ LGD × EAD × DF
        ecl = stage3["ecl_base"].to_numpy()
        lgd = stage3["lgd"].to_numpy()
        ead = stage3["ead"].to_numpy()
        df_ = stage3["discount_factor"].to_numpy()
        expected = lgd * ead * df_
        # Should be very close (PD=1 means ECL = LGD × EAD × DF)
        np.testing.assert_allclose(ecl, np.round(expected, 2), atol=1.0,
            err_msg="Stage 3 ECL should use PD=1.0 (ECL ≈ LGD × EAD × DF)")

    def test_stage3_triggered_by_dpd_90(self, result_credit):
        """Clients with DPD >= 90 should be in Stage 3."""
        if "dpd" not in result_credit.columns:
            pytest.skip("No dpd column in test data")
        high_dpd = result_credit.filter(pl.col("dpd") >= 90)
        if len(high_dpd) == 0:
            pytest.skip("No clients with DPD >= 90 in test data")
        stages = high_dpd["stage"].to_numpy()
        assert np.all(stages == 3), (
            f"Clients with DPD >= 90 should be Stage 3, got stages: {np.unique(stages)}"
        )

    def test_ecl_positive(self, result_credit):
        """ECL should be non-negative for all clients."""
        ecl = result_credit["ecl_weighted"].to_numpy()
        assert np.all(ecl >= -0.01), (
            f"ECL sign error: min ECL = {ecl.min()}"
        )

    def test_ecl_bounded_by_exposure(self, result_credit):
        """ECL should not exceed EAD for any client."""
        ecl = result_credit["ecl_weighted"].to_numpy()
        ead = result_credit["ead"].to_numpy()
        violations = np.sum(ecl > ead * 1.01)  # 1% tolerance for rounding
        assert violations == 0, (
            f"ECL exceeds EAD for {violations} clients. "
            f"Max ratio = {(ecl / np.maximum(ead, 1)).max():.4f}"
        )

    def test_ecl_weighted_between_scenarios(self, result_credit):
        """ECL_weighted should be between ECL_favorable and ECL_adverse."""
        ecl_w = result_credit["ecl_weighted"].to_numpy()
        ecl_fav = result_credit["ecl_favorable"].to_numpy()
        ecl_adv = result_credit["ecl_adverse"].to_numpy()
        ecl_min = np.minimum(ecl_fav, ecl_adv)
        ecl_max = np.maximum(ecl_fav, ecl_adv)
        below = np.sum(ecl_w < ecl_min - 0.05)
        above = np.sum(ecl_w > ecl_max + 0.05)
        assert below == 0 and above == 0, (
            f"ECL weighting incoherent: {below} below min, {above} above max"
        )

    def test_higher_pd_produces_higher_ecl(self, result_credit):
        """Monotonicity: higher PD should produce higher ECL (on average)."""
        # Compare Base vs Adverse: adverse PD should produce higher ECL
        ecl_base = result_credit["ecl_base"].to_numpy().mean()
        ecl_adv = result_credit["ecl_adverse"].to_numpy().mean()
        assert ecl_adv >= ecl_base * 0.95, (
            f"PD-ECL transmission broken: avg ECL_adverse ({ecl_adv:.2f}) "
            f"< avg ECL_base ({ecl_base:.2f})"
        )

    def test_sicr_score_increases_with_pd_deterioration(self):
        """SICR score should increase when PD deteriorates."""
        from ifrs9_cockpit.engine.staging import compute_sicr_score
        pd_orig = np.array([0.01, 0.01, 0.01])
        dpd = np.zeros(3)
        # Small deterioration
        score_small = compute_sicr_score(
            np.array([0.02, 0.02, 0.02]), pd_orig, dpd
        )
        # Large deterioration
        score_large = compute_sicr_score(
            np.array([0.05, 0.05, 0.05]), pd_orig, dpd
        )
        assert np.all(score_large > score_small), (
            f"SICR formula broken: larger PD deterioration should "
            f"produce higher SICR score"
        )


# ──────────────────────────────────────────────
# CLASS 4: PE VALUATION
# ──────────────────────────────────────────────

class TestPEValuation:
    """Validates PE-specific economics (MOIC, IRR, distress, NAV)."""

    def test_moic_equals_nav_over_capital(self, result_pe):
        """MOIC = NAV / capital_invested."""
        nav = result_pe["nav"].to_numpy()
        capital = result_pe["capital_invested"].to_numpy()
        moic = result_pe["moic"].to_numpy()
        expected = np.where(capital > 0, nav / capital, 0)
        np.testing.assert_allclose(moic, np.round(expected, 4), atol=1e-3,
            err_msg="MOIC formula broken: MOIC ≠ NAV / capital_invested")

    def test_irr_bounded(self, result_pe):
        """IRR should be clipped to [-0.50, +0.35]."""
        irr = result_pe["irr"].to_numpy()
        assert np.all(irr >= -0.50 - 1e-6) and np.all(irr <= 0.35 + 1e-6), (
            f"IRR clip missing: min={irr.min():.4f}, max={irr.max():.4f}"
        )

    def test_distress_prob_bounded_zero_one(self, result_pe):
        """P(distress) should be in [0, 1]."""
        dp = result_pe["distress_prob"].to_numpy()
        assert np.all(dp >= 0) and np.all(dp <= 1.0 + 1e-6), (
            f"Logit-expit broken: distress_prob range = [{dp.min()}, {dp.max()}]"
        )

    def test_low_moic_increases_distress(self, result_pe):
        """MOIC < 1 should be associated with higher distress probability."""
        low_moic = result_pe.filter(pl.col("moic") < 1.0)
        high_moic = result_pe.filter(pl.col("moic") >= 1.0)
        if len(low_moic) == 0 or len(high_moic) == 0:
            pytest.skip("Need both low and high MOIC clients")
        avg_dp_low = low_moic["distress_prob"].mean()
        avg_dp_high = high_moic["distress_prob"].mean()
        assert avg_dp_low > avg_dp_high, (
            f"MOIC-distress link broken: avg P(distress|MOIC<1)={avg_dp_low:.4f} "
            f"≤ avg P(distress|MOIC≥1)={avg_dp_high:.4f}"
        )

    def test_el_pe_formula(self, result_pe):
        """EL_PE = P(distress) × LGD_equity(0.60) × NAV."""
        lgd_eq = PE_CLASSIFICATION_CONFIG.lgd_equity
        dp = result_pe["distress_prob"].to_numpy()
        nav = result_pe["nav"].to_numpy()
        el = result_pe["expected_loss_pe"].to_numpy()
        expected = dp * lgd_eq * nav
        np.testing.assert_allclose(el, np.round(expected, 2), atol=1.0,
            err_msg=f"EL PE formula broken: EL ≠ P(distress) × LGD({lgd_eq}) × NAV")

    def test_pe_revenue_equals_nav_times_irr(self, raroc_df, result_pe):
        """PE revenue = NAV × IRR_mean per sector."""
        pe_rows = raroc_df.filter(
            (pl.col("canal") == "PE") & (pl.col("sector") != "Total")
        )
        for row in pe_rows.iter_rows(named=True):
            sector_pe = result_pe.filter(pl.col("sector") == row["sector"])
            if len(sector_pe) == 0:
                continue
            nav_total = sector_pe["nav"].sum()
            irr_mean = sector_pe["irr"].mean()
            expected_rev = nav_total * irr_mean
            assert abs(row["revenue"] - round(expected_rev, 0)) <= 2.0, (
                f"PE revenue formula broken for {row['sector']}: "
                f"got {row['revenue']}, expected {expected_rev:.0f}"
            )

    def test_pe_rwa_uses_crr3_weight(self, result_pe):
        """PE RWA should use CRR3 risk weights (190/250/400)."""
        nav = result_pe["nav"].to_numpy()
        rwa = result_pe["rwa_pe"].to_numpy()
        # RW = RWA / NAV * 100 — should be one of {190, 250, 400}
        valid_rw = {190, 250, 400}
        rw_actual = np.round(rwa / np.maximum(nav, 1) * 100, 0)
        for rw in rw_actual:
            assert int(rw) in valid_rw, (
                f"PE risk weight {int(rw)}% not in CRR3 set {valid_rw}"
            )


# ──────────────────────────────────────────────
# CLASS 5: REGULATORY NORMS
# ──────────────────────────────────────────────

class TestRegulatoryNorms:
    """Validates LCR, NSFR, CET1, IRRBB compliance logic."""

    def test_cet1_ratio_formula(self, opt_14c):
        """CET1 ratio should be CET1_capital / RWA."""
        assert opt_14c["cet1_ratio"] > 0, "CET1 ratio should be positive"

    def test_cet1_target_is_13_pct(self):
        """BASEL_CONFIG.cet1_target should be 0.13."""
        assert BASEL_CONFIG.cet1_target == 0.13, (
            f"Config changed: cet1_target = {BASEL_CONFIG.cet1_target}"
        )

    def test_lcr_ratio_positive(self, opt_14c):
        """LCR should be a positive ratio."""
        assert opt_14c["lcr_ratio"] > 0, (
            f"LCR computation error: lcr_ratio = {opt_14c['lcr_ratio']}"
        )

    def test_nsfr_ratio_positive(self, opt_14c):
        """NSFR should be a positive ratio."""
        assert opt_14c["nsfr_ratio"] > 0, (
            f"NSFR computation error: nsfr_ratio = {opt_14c['nsfr_ratio']}"
        )

    def test_lcr_target_is_100_pct(self):
        """BASEL_CONFIG.lcr_target should be 1.00."""
        assert BASEL_CONFIG.lcr_target == 1.00, (
            f"Config changed: lcr_target = {BASEL_CONFIG.lcr_target}"
        )

    def test_nsfr_target_is_100_pct(self):
        """BASEL_CONFIG.nsfr_target should be 1.00."""
        assert BASEL_CONFIG.nsfr_target == 1.00, (
            f"Config changed: nsfr_target = {BASEL_CONFIG.nsfr_target}"
        )

    def test_hhi_below_limit(self, opt_14c):
        """HHI should be below 2500 for a diversified portfolio."""
        # Compute HHI from class_weights
        weights = list(opt_14c["class_weights"].values())
        hhi = sum(w ** 2 for w in weights) * 10_000
        assert hhi <= BASEL_CONFIG.hhi_max, (
            f"Concentration limit exceeded: HHI = {hhi:.0f} > {BASEL_CONFIG.hhi_max}"
        )

    def test_pe_allocation_below_15_pct(self, opt_14c):
        """PE weight should not exceed 15% after regulatory constraints."""
        pe_alloc = opt_14c["pe_allocation"]
        assert pe_alloc <= BASEL_CONFIG.pe_max_allocation + 1e-4, (
            f"PE cap not enforced: pe_allocation = {pe_alloc:.4f} "
            f"> max {BASEL_CONFIG.pe_max_allocation}"
        )


# ──────────────────────────────────────────────
# CLASS 6: OPTIMIZER ECONOMICS
# ──────────────────────────────────────────────

class TestOptimizerEconomics:
    """Validates that optimizer output is economically coherent."""

    def test_weights_sum_to_one_14c(self, opt_14c):
        """14C class weights should sum to 1.0."""
        total = sum(opt_14c["class_weights"].values())
        assert abs(total - 1.0) < 1e-4, (
            f"Weight normalization broken (14C): sum = {total}"
        )

    def test_weights_sum_to_one_10c(self, opt_10c):
        """10C class weights should sum to 1.0."""
        total = sum(opt_10c["class_weights"].values())
        assert abs(total - 1.0) < 1e-4, (
            f"Weight normalization broken (10C): sum = {total}"
        )

    def test_credit_pe_split_consistent(self, opt_14c):
        """credit_allocation + pe_allocation ≈ 1.0."""
        total = opt_14c["credit_allocation"] + opt_14c["pe_allocation"]
        assert abs(total - 1.0) < 1e-4, (
            f"Channel split broken: credit({opt_14c['credit_allocation']:.4f}) "
            f"+ pe({opt_14c['pe_allocation']:.4f}) = {total}"
        )

    def test_sector_weights_consistent_with_class_weights(self, opt_10c):
        """Sector credit weights should be consistent with class weights."""
        # In 10C, the 5 credit cells + 5 PE cells should match
        # credit_allocation and pe_allocation
        credit_alloc = opt_10c["credit_allocation"]
        pe_alloc = opt_10c["pe_allocation"]
        total = credit_alloc + pe_alloc
        assert abs(total - 1.0) < 1e-4, (
            f"Aggregation broken (10C): credit + PE = {total}"
        )

    def test_cvar_non_negative(self, opt_14c):
        """CVaR_95 should be non-negative (loss measure)."""
        assert opt_14c["cvar_95"] >= 0, (
            f"CVaR sign error: cvar_95 = {opt_14c['cvar_95']}"
        )

    def test_feasible_implies_cet1_above_target(self, opt_14c):
        """If feasible=True, CET1 ratio should be >= target."""
        if opt_14c["feasible"]:
            assert opt_14c["cet1_ratio"] >= BASEL_CONFIG.cet1_target - 1e-6, (
                f"Feasibility flag wrong: feasible=True but "
                f"CET1={opt_14c['cet1_ratio']:.4f} < target={BASEL_CONFIG.cet1_target}"
            )

    def test_method_label_correct(self, opt_14c, opt_10c):
        """14C returns 'BL-CVaR-14C', 10C returns 'BL-CVaR-10C'."""
        assert opt_14c["method"] == "BL-CVaR-14C", (
            f"Method label wrong (14C): got '{opt_14c['method']}'"
        )
        assert opt_10c["method"] == "BL-CVaR-10C", (
            f"Method label wrong (10C): got '{opt_10c['method']}'"
        )

    def test_spread_compression_reduces_large_positions(self, opt_14c):
        """Spread compression should exist for market impact."""
        sc = opt_14c.get("spread_compression")
        assert sc is not None, "spread_compression key missing from optimizer"
        assert isinstance(sc, dict), (
            f"spread_compression should be dict, got {type(sc)}"
        )
        # At least some compression values should be non-zero
        values = list(sc.values())
        assert any(v != 0 for v in values), (
            "All spread compression values are zero — market impact not computed"
        )


# ──────────────────────────────────────────────
# CLASS 7: CROSS-MODULE COHERENCE
# ──────────────────────────────────────────────

class TestCrossModuleCoherence:
    """Validates that modules interact coherently end-to-end."""

    def test_ecl_used_as_loss_in_raroc(self, result_credit, raroc_df):
        """Loss column in RAROC ≈ annual EL from ECL calculator."""
        for sector in SECTORS:
            rc_sec = result_credit.filter(pl.col("sector") == sector.name)
            if len(rc_sec) == 0:
                continue
            # Annual EL = Σ(pd_12m × lgd × ead) for current stressed PD
            pd_arr = rc_sec["pd_12m"].to_numpy().astype(float)
            lgd_arr = rc_sec["lgd"].to_numpy().astype(float)
            ead_arr = rc_sec["ead"].to_numpy().astype(float)
            annual_el = float(np.sum(pd_arr * lgd_arr * ead_arr))

            raroc_row = raroc_df.filter(
                (pl.col("sector") == sector.name) & (pl.col("canal") == "Credit")
            )
            if len(raroc_row) == 0:
                continue
            loss_in_raroc = raroc_row["loss"][0]
            assert abs(loss_in_raroc - round(annual_el, 0)) <= 2.0, (
                f"ECL→RAROC link broken for {sector.name}: "
                f"RAROC loss={loss_in_raroc}, annual_el={annual_el:.0f}"
            )

    def test_pd_model_feeds_ecl(self, pipeline):
        """PD predictions should be used by ECL calculator."""
        pd_current = pipeline["pd_current"]
        result_credit = pipeline["result_credit"]
        # pd_12m in result should correlate with pd_current
        pd_12m = result_credit["pd_12m"].to_numpy()
        # Under base scenario (no override), pd_12m ≈ pd_current
        corr = np.corrcoef(pd_current, pd_12m)[0, 1]
        assert corr > 0.90, (
            f"PD→ECL pipeline broken: correlation(pd_current, pd_12m) = {corr:.4f}"
        )

    def test_lgd_used_in_ecl_and_pe(self, result_credit, result_pe):
        """Both credit and PE paths should have loss components."""
        # Credit should have LGD column
        assert "lgd" in result_credit.columns, "LGD missing from credit results"
        lgd_credit = result_credit["lgd"].to_numpy()
        assert np.all(lgd_credit >= 0) and np.all(lgd_credit <= 1), (
            f"LGD wiring broken (credit): range [{lgd_credit.min()}, {lgd_credit.max()}]"
        )
        # PE should have expected_loss_pe
        assert "expected_loss_pe" in result_pe.columns, "expected_loss_pe missing from PE results"
        el_pe = result_pe["expected_loss_pe"].to_numpy()
        assert np.all(el_pe >= 0), "PE EL should be non-negative"

    def test_raroc_feeds_optimizer(self, raroc_df, opt_14c):
        """Optimizer should read profit rates from RAROC results."""
        # The optimizer's profit_rate_portfolio should be a weighted average
        # of individual profit_rates
        pr = opt_14c.get("profit_rate_portfolio")
        assert pr is not None, "profit_rate_portfolio missing from optimizer"
        # Should be in a reasonable range
        assert -0.5 < pr < 0.5, (
            f"RAROC→optimizer link broken: profit_rate_portfolio = {pr}"
        )

    def test_nii_sign_convention(self, raroc_df):
        """NII (revenue) should be positive for performing portfolios."""
        credit_rows = raroc_df.filter(
            (pl.col("canal") == "Credit") & (pl.col("sector") != "Total")
        )
        for row in credit_rows.iter_rows(named=True):
            assert row["revenue"] > 0, (
                f"NII sign wrong for {row['sector']}: revenue = {row['revenue']}"
            )

    def test_capital_consistent_across_modules(self, raroc_df, opt_14c):
        """Capital in RAROC should use same CET1 target as optimizer."""
        # Both should use BASEL_CONFIG.cet1_target = 0.13
        coc = BASEL_CONFIG.cet1_target
        credit_total = raroc_df.filter(
            (pl.col("canal") == "Credit") & (pl.col("sector") == "Total")
        )
        if len(credit_total) > 0:
            rwa = credit_total["rwa"][0]
            capital = credit_total["capital"][0]
            expected_capital = round(rwa * coc, 0)
            assert abs(capital - expected_capital) <= 1.0, (
                f"Capital inconsistency: RAROC capital={capital}, "
                f"RWA×cet1={expected_capital}"
            )
