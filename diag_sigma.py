"""Quick sigma diagnostic per scenario — asymmetric equations."""
import sys, os, warnings
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

df_credit, df_pe, _ = load_data()
pd_suite = train_pd_models()
lgd_model, ead_model = train_lgd_ead()
ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
pe_calc = PECalculator()
pd_current = pd_suite.predict(df_credit)["LR_WoE"]
pd_origination = df_credit["pd_origination"].values

print(f"{'Scenario':25s} | {'Regime':10s} | {'stress':>7s} | {'sig_c_raw':>9s} | {'sig_p_raw':>9s} | {'vol_mult':>8s} | {'sig_p_eff':>9s} | {'sig_c>sig_p':>11s} | {'illiq':>7s} | {'Ph1':>5s}")
print("-" * 140)

for name, params in PREDEFINED_SCENARIOS.items():
    macro = slider_to_macro(params)
    is_crisis = params["unemployment_bipolar"] < 0
    result_credit = ecl_calc.calculate(df_credit, pd_current, pd_origination,
        unemployment_override=macro["unemployment_rate"], gdp_override=macro["gdp_growth"],
        interest_rate_override=macro["interest_rate"], hpi_override=macro["hpi_growth"],
        inflation_override=macro["inflation_rate"])
    result_pe = pe_calc.calculate(df_pe,
        unemployment_override=macro["unemployment_rate"], gdp_override=macro["gdp_growth"],
        interest_rate_override=macro["interest_rate"], hpi_override=macro["hpi_growth"],
        inflation_override=macro["inflation_rate"], unemployment_crisis=is_crisis)

    comp = PortfolioComparator(result_credit, result_pe)
    opt = comp.optimize_allocation()
    regime = opt["regime"]
    stress = opt["stress_intensity"]
    vol_mult = opt["vol_multiplier"]
    illiq = opt["illiquidity_premium"]

    # Get raw sigmas
    raroc_df = comp.compute_raroc_eva()
    credit_cells = raroc_df.loc[(raroc_df["canal"] == "Credit") & (raroc_df["sector"] != "Total")]
    pe_cells = raroc_df.loc[(raroc_df["canal"] == "PE") & (raroc_df["sector"] != "Total")]
    sig_c_raw = comp._estimate_canal_volatility(credit_cells)
    sig_p_raw = comp._estimate_canal_volatility(pe_cells)
    sig_p_eff = max(sig_p_raw * vol_mult, BASEL_CONFIG.vol_floor)
    phase1 = opt["ideal_pe_alloc"]

    flag = "ANOMALY" if sig_c_raw > sig_p_eff else ""
    print(f"{name:25s} | {regime:10s} | {stress:+6.2f} | {sig_c_raw:8.2%} | {sig_p_raw:8.2%} | {vol_mult:7.2f}x | {sig_p_eff:8.2%} | {flag:>11s} | {illiq:6.1%} | {phase1:4.0%}")
