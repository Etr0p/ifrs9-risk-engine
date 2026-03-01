"""Diagnostic Phase 1 : allocations BL-CVaR libres sur les 8 scenarios."""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ifrs9_cockpit.config import PREDEFINED_SCENARIOS, SCENARIO_BASE, BASEL_CONFIG
from ifrs9_cockpit.dashboard.cache import load_data, train_pd_models, train_lgd_ead
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.engine.comparator import PortfolioComparator

def slider_to_macro(scenario_dict):
    base = SCENARIO_BASE
    return {
        "unemployment_rate": base.unemployment_rate + abs(scenario_dict["unemployment_bipolar"]),
        "gdp_growth": scenario_dict["gdp_pct"],
        "interest_rate": base.interest_rate + scenario_dict["interest_rate_bp"] / 100,
        "hpi_growth": scenario_dict["hpi_pct"],
        "inflation_rate": scenario_dict["inflation_pct"],
    }

print("Loading data & models...")
df_credit, df_pe, _ = load_data()
pd_suite = train_pd_models()
lgd_model, ead_model = train_lgd_ead()
ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
pe_calc = PECalculator()
pd_predictions = pd_suite.predict(df_credit)
pd_current = pd_predictions["LR_WoE"]
pd_origination = df_credit["pd_origination"].values

print()
print(f"{'Scenario':25s} | {'Regime':10s} | {'Ph1':>5s} | {'Final':>5s} | {'CrSpot':>7s} | {'CrTTC':>7s} | {'PESpot':>7s} | {'PETTC':>7s} | {'BL_Cr':>7s} | {'BL_PE':>7s} | {'Chain'}")
print("-" * 155)

for name, params in PREDEFINED_SCENARIOS.items():
    macro = slider_to_macro(params)
    is_crisis = params["unemployment_bipolar"] < 0

    result_credit = ecl_calc.calculate(
        df_credit, pd_current, pd_origination,
        unemployment_override=macro["unemployment_rate"],
        gdp_override=macro["gdp_growth"],
        interest_rate_override=macro["interest_rate"],
        hpi_override=macro["hpi_growth"],
        inflation_override=macro["inflation_rate"],
    )
    result_pe = pe_calc.calculate(
        df_pe,
        unemployment_override=macro["unemployment_rate"],
        gdp_override=macro["gdp_growth"],
        interest_rate_override=macro["interest_rate"],
        hpi_override=macro["hpi_growth"],
        inflation_override=macro["inflation_rate"],
        unemployment_crisis=is_crisis,
    )

    comp = PortfolioComparator(result_credit, result_pe)
    opt = comp.optimize_allocation()

    r_c_spot = opt.get("raroc_credit_spot", opt.get("raroc_credit", 0))
    r_c_ttc = opt.get("raroc_credit_ttc", 0)
    r_p_spot = opt.get("raroc_pe_spot", 0)
    r_p_ttc = opt.get("raroc_pe_ttc", 0)
    bl = opt.get("bl_posterior", [0, 0])
    phase1 = opt.get("ideal_pe_alloc", 0)
    final_pe = opt.get("optimal_pe_alloc", 0)
    regime = opt.get("regime", "?")
    chain = opt.get("optimal_rationale", "")

    print(f"{name:25s} | {regime:10s} | {phase1:4.0%} | {final_pe:4.0%} | {r_c_spot:6.2%} | {r_c_ttc:6.2%} | {r_p_spot:6.2%} | {r_p_ttc:6.2%} | {bl[0]:6.2%} | {bl[1]:6.2%} | {chain}")

print()
print("=== Config ===")
print(f"  kappa = [credit={BASEL_CONFIG.kappa_credit}, pe={BASEL_CONFIG.kappa_pe}]")
print(f"  credit_LT={BASEL_CONFIG.credit_longterm_raroc:.0%}, credit_h={BASEL_CONFIG.credit_holding_horizon}")
print(f"  pe_LT={BASEL_CONFIG.pe_longterm_raroc:.0%}, illiq_base={BASEL_CONFIG.illiquidity_base:.1%}, illiq_scale={BASEL_CONFIG.illiquidity_scale}, gamma={BASEL_CONFIG.illiquidity_gamma}")
print(f"  lambda_hhi={BASEL_CONFIG.lambda_hhi}, theta={BASEL_CONFIG.pe_mean_reversion_speed}")
print(f"  pe_max={BASEL_CONFIG.pe_max_allocation:.0%}, rwa_budget={BASEL_CONFIG.rwa_budget:.1e}")
