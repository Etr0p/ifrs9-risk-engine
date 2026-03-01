"""Debug optimizer: trace CVaR-gradient BL-CVaR direction."""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np
import polars as pl
from ifrs9_cockpit.synthetic_generator import generate_dataset
from ifrs9_cockpit.config import RANDOM_SEED, ASSET_CLASS_MAP, SCENARIO_BASE, PREDEFINED_SCENARIOS
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl
from ifrs9_cockpit.engine.comparator import PortfolioComparator

bundle = generate_dataset(n_clients=5000, seed=RANDOM_SEED)
pd_suite = PDModelSuite(seed=RANDOM_SEED)
pd_suite.fit(bundle.df_credit)
pd_curr = pd_suite.predict_active(bundle.df_credit)
pd_orig = bundle.df_credit["pd_origination"].to_numpy()
lgd_m = LGDModel(); lgd_m.fit(bundle.df_credit)
ead_m = EADModel(); ead_m.fit(bundle.df_credit)
ecl_calc = ECLCalculator(lgd_model=lgd_m, ead_model=ead_m)
pe_calc = PECalculator()

raw = PREDEFINED_SCENARIOS["Central"]
macro = {
    "unemployment_rate": SCENARIO_BASE.unemployment_rate + abs(raw.get("unemployment_bipolar", 0)),
    "gdp_growth": raw.get("gdp_pct", SCENARIO_BASE.gdp_growth),
    "interest_rate": SCENARIO_BASE.interest_rate + raw.get("interest_rate_bp", 0) / 100.0,
    "hpi_growth": raw.get("hpi_pct", SCENARIO_BASE.hpi_growth),
    "inflation_rate": raw.get("inflation_pct", SCENARIO_BASE.inflation_rate),
}
res_c = ecl_calc.calculate(bundle.df_credit, pd_curr, pd_orig,
    unemployment_override=macro["unemployment_rate"], gdp_override=macro["gdp_growth"],
    interest_rate_override=macro["interest_rate"], hpi_override=macro["hpi_growth"],
    inflation_override=macro["inflation_rate"])
res_p = pe_calc.calculate(bundle.df_pe,
    unemployment_override=macro["unemployment_rate"], gdp_override=macro["gdp_growth"],
    interest_rate_override=macro["interest_rate"], hpi_override=macro["hpi_growth"],
    inflation_override=macro["inflation_rate"], unemployment_crisis=False)
bs_ecl = compute_balance_sheet_ecl(bundle.df_balance_sheet, macro)

comp = PortfolioComparator(res_c, res_p, bs_ecl)
r = comp.compute_raroc_multiclass()
classes_df = r.filter(pl.col("asset_class") != "Total")
class_names = classes_df["asset_class"].to_list()
n = len(class_names)
mu_profit = classes_df["profit_rate"].to_numpy().astype(float).copy()
capital_arr = classes_df["capital"].to_numpy().astype(float)
ead_arr = classes_df["exposure"].to_numpy().astype(float)
cap_ead = capital_arr / np.maximum(ead_arr, 1)

# Vol in return-space then convert to profit_rate-space
vol_return = np.zeros(n)
for i, name in enumerate(class_names):
    ac = ASSET_CLASS_MAP[name]
    if name == "private_equity":
        irr_pe = res_p["irr"].mean() if len(res_p) > 0 else 0.08
        vol_return[i] = max(0.15, irr_pe * 1.2)
    elif ac.market_vol_override is not None:
        vol_return[i] = ac.market_vol_override
    else:
        macro_intensity = sum(ac.macro_sensitivities.values()) / 5.0
        vol_return[i] = max(0.02, ac.pd_std * ac.lgd_base * 10 + 0.01 * macro_intensity)
vol = vol_return * cap_ead  # profit_rate space

w_base = np.array([ASSET_CLASS_MAP[nm].typical_weight for nm in class_names])
w_base /= w_base.sum()
corr_14, _ = comp._build_corr_matrix()
total_ead = ead_arr.sum()

# Covariance + Monte Carlo
Sigma = np.outer(vol, vol) * corr_14
Sigma = (Sigma + Sigma.T) / 2
eigvals = np.linalg.eigvalsh(Sigma)
if eigvals.min() < 1e-8:
    Sigma += np.eye(n) * (1e-6 - min(0, eigvals.min()))

rng = np.random.default_rng(42)
L_chol = np.linalg.cholesky(Sigma)
Z = rng.standard_normal((5000, n))
scenarios_0 = Z @ L_chol.T
cutoff = max(int(5000 * 0.05), 1)

# Kappa
mu_eff_base = comp._spread_compression(w_base, class_names, total_ead, mu_profit)
mu_p_base = float(w_base @ mu_eff_base)
port_dev_base = scenarios_0 @ w_base
cvar_base = -float(np.mean(np.sort(port_dev_base)[:cutoff]))
kappa = mu_p_base / max(cvar_base, 1e-10)
print(f"\nmu_p_base={mu_p_base*10000:.1f}bp, CVaR_base={cvar_base*10000:.1f}bp, kappa={kappa:.4f}")

# CVaR gradient at w_base
tail_threshold = np.sort(port_dev_base)[cutoff - 1]
tail_mask = port_dev_base <= tail_threshold
cvar_gradient = -scenarios_0[tail_mask].mean(axis=0)

# BL-CVaR scores
bl_cvar_scores = mu_eff_base - kappa * cvar_gradient

print(f"\n{'Class':20s} | {'mu(bp)':>7s} | {'vol(bp)':>7s} | {'dCVaR(bp)':>9s} | {'score(bp)':>9s} | {'w_opt':>6s} | {'w_base':>6s}")
print("-" * 95)
bl_scores_norm = bl_cvar_scores / max(np.max(np.abs(bl_cvar_scores)), 1e-10)
w_raw = comp._softmax_10(bl_scores_norm, corr_14)
mu_comp = comp._spread_compression(w_raw, class_names, total_ead, mu_profit)
bl_scores_comp = mu_comp - kappa * cvar_gradient
bl_comp_norm = bl_scores_comp / max(np.max(np.abs(bl_scores_comp)), 1e-10)
w_opt = comp._softmax_10(bl_comp_norm, corr_14)

for i, name in enumerate(class_names):
    print(f"{name:20s} | {mu_eff_base[i]*10000:6.1f}  | {vol[i]*10000:6.1f}  | {cvar_gradient[i]*10000:8.2f}  | {bl_cvar_scores[i]*10000:8.2f}  | {w_opt[i]:5.1%} | {w_base[i]:5.1%}")

# Grid search trace
idx_ib = class_names.index("interbank")
idx_cons = class_names.index("consumer_credit")
idx_repo = class_names.index("repos_sft")
idx_eq = class_names.index("equities")
idx_corp = class_names.index("corporate_loans")
print(f"\n{'alpha':>6s} | {'IB':>5s} | {'Cons':>5s} | {'Repo':>5s} |  {'Eq':>5s} | {'Corp':>5s} | {'mu_p(bp)':>8s} | {'CVaR(bp)':>8s} | {'obj(bp)':>8s}")
print("-" * 90)
for a in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.0]:
    w = (1.0 - a) * w_base + a * w_opt
    w = np.maximum(w, 0); w /= w.sum()
    mu_eff = comp._spread_compression(w, class_names, total_ead, mu_profit)
    mu_p = float(w @ mu_eff)
    cvar = -float(np.mean(np.sort(scenarios_0 @ w)[:cutoff]))
    obj = mu_p - kappa * cvar
    print(f"{a:6.2f} | {w[idx_ib]:4.1%} | {w[idx_cons]:4.1%} | {w[idx_repo]:4.1%} | {w[idx_eq]:4.1%} | {w[idx_corp]:4.1%} | {mu_p*10000:7.2f}  | {cvar*10000:7.2f}  | {obj*10000:7.2f}")

# Final result
print("\n--- optimize_allocation result ---")
opt = comp.optimize_allocation(macro)
for name in class_names:
    w = opt["class_weights"].get(name, 0)
    print(f"  {name:20s} {w:5.1%}")
print(f"  risk_alpha = {opt.get('risk_alpha', '?')}")
print(f"  profit_rate_portfolio = {opt.get('profit_rate_portfolio', 0)*10000:.1f}bp")
print(f"  kappa = {opt.get('kappa', 0):.4f}")
