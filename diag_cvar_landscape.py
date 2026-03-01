"""Diagnostic CVaR landscape pour Stagflation — cherche le bug."""
import sys, os, warnings, math
import numpy as np
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

# Run Stagflation
params = PREDEFINED_SCENARIOS["Stagflation"]
macro = slider_to_macro(params)
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
    unemployment_crisis=True,
)

comp = PortfolioComparator(result_credit, result_pe)

# Get the RAROC/EVA to compute intermediate values
raroc_df = comp.compute_raroc_eva()
credit_cells = raroc_df.loc[(raroc_df["canal"] == "Credit") & (raroc_df["sector"] != "Total")]
pe_cells = raroc_df.loc[(raroc_df["canal"] == "PE") & (raroc_df["sector"] != "Total")]

# Recompute BL-CVaR internals
sigma_c = comp._estimate_canal_volatility(credit_cells)
sigma_p = comp._estimate_canal_volatility(pe_cells)
rho_cp = comp._estimate_cross_correlation(credit_cells, pe_cells)

print(f"\n=== Volatility & Correlation ===")
print(f"sigma_credit = {sigma_c:.4f} ({sigma_c:.2%})")
print(f"sigma_pe     = {sigma_p:.4f} ({sigma_p:.2%})")
print(f"rho_credit_pe= {rho_cp:.4f}")

# Get optimization result
opt = comp.optimize_allocation()
mu_BL = np.array(opt["bl_posterior"])
print(f"\n=== BL Posterior ===")
print(f"mu_BL = [{mu_BL[0]:.4%}, {mu_BL[1]:.4%}]")
print(f"  (credit = {mu_BL[0]:.4%}, PE = {mu_BL[1]:.4%})")

# Kappa
kappa = np.array([BASEL_CONFIG.kappa_credit, BASEL_CONFIG.kappa_pe])
print(f"kappa = [{kappa[0]}, {kappa[1]}]")
print(f"kappa-weighted means: credit = {mu_BL[0]*kappa[0]:.4%}, PE = {mu_BL[1]*kappa[1]:.4%}")

# Rebuild Sigma_BL from the opt result
Sigma = np.array([
    [sigma_c ** 2, rho_cp * sigma_c * sigma_p],
    [rho_cp * sigma_c * sigma_p, sigma_p ** 2],
])
theta = BASEL_CONFIG.pe_mean_reversion_speed
h = BASEL_CONFIG.pe_holding_years
decay = math.exp(-theta * h)
delta = BASEL_CONFIG.bl_delta
tau = BASEL_CONFIG.bl_tau

# Recompute views
r_c_ttc = opt.get("raroc_credit_ttc", 0)
r_p_ttc = opt.get("raroc_pe_ttc", 0)
Q = np.array([r_c_ttc, r_p_ttc])
print(f"Views Q = [{Q[0]:.4%}, {Q[1]:.4%}]")

# Approximate Sigma_BL = Sigma + M_bl (use full Sigma as proxy)
# We'll use the same scenarios as the optimizer
S = BASEL_CONFIG.cvar_n_scenarios
alpha = BASEL_CONFIG.cvar_alpha
lam_hhi = BASEL_CONFIG.lambda_hhi

# Compute the approximate Sigma_BL for scenarios
ead_t = float(max(result_credit["ead"].sum(), 1))
nav_t = float(max(result_pe["nav"].sum(), 1))
w_mkt = np.array([ead_t / (ead_t + nav_t), nav_t / (ead_t + nav_t)])
pi = delta * Sigma @ w_mkt

P = np.eye(2)
c_credit = float(np.clip(abs(r_c_ttc - pi[0]) / max(abs(pi[0]) + 0.01, 0.01), 0.30, 0.95))
c_pe = float(np.clip(abs(r_p_ttc - pi[1]) / max(abs(pi[1]) + 0.01, 0.01), 0.30, 0.95))
omega_c = tau * float(P[0] @ Sigma @ P[0]) * (1.0 - c_credit) / max(c_credit, 1e-6)
omega_p = tau * float(P[1] @ Sigma @ P[1]) * (1.0 - c_pe) / max(c_pe, 1e-6)
Omega = np.diag([max(omega_c, 1e-12), max(omega_p, 1e-12)])
tau_Sigma = tau * Sigma
A_bl = P @ tau_Sigma @ P.T + Omega
M_bl = tau_Sigma - tau_Sigma @ P.T @ np.linalg.solve(A_bl, P @ tau_Sigma)
Sigma_BL = Sigma + M_bl

print(f"\n=== Sigma_BL ===")
print(f"  [{Sigma_BL[0,0]:.6f}  {Sigma_BL[0,1]:.6f}]")
print(f"  [{Sigma_BL[1,0]:.6f}  {Sigma_BL[1,1]:.6f}]")
print(f"  Marginal stds: credit={np.sqrt(Sigma_BL[0,0]):.4%}, PE={np.sqrt(Sigma_BL[1,1]):.4%}")

# Generate scenarios
rng = np.random.default_rng(42)
eigvals = np.linalg.eigvalsh(Sigma_BL)
if eigvals.min() < 1e-10:
    Sigma_BL += np.eye(2) * (1e-8 - eigvals.min())
scenarios = rng.multivariate_normal(mu_BL, Sigma_BL, size=S)

print(f"\n=== Scenario Statistics ===")
print(f"scenarios mean = [{scenarios[:,0].mean():.4%}, {scenarios[:,1].mean():.4%}]")
print(f"scenarios std  = [{scenarios[:,0].std():.4%}, {scenarios[:,1].std():.4%}]")
print(f"scenarios corr = {np.corrcoef(scenarios[:,0], scenarios[:,1])[0,1]:.4f}")

# CVaR landscape
k_tail = max(1, int(np.ceil(S * (1.0 - alpha))))
print(f"\n=== CVaR Landscape (alpha={alpha}, k_tail={k_tail}) ===")
print(f"{'PE%':>5s} | {'Objective':>10s} | {'CVaR':>10s} | {'HHI_pen':>10s} | {'Mean_loss':>10s}")
print("-" * 60)

for pe_pct_int in range(0, 101, 5):
    wp = pe_pct_int / 100.0
    w = np.array([1.0 - wp, wp])
    losses = -(scenarios * kappa[np.newaxis, :]) @ w
    top_k = np.partition(losses, -k_tail)[-k_tail:]
    cvar = float(top_k.mean())
    hhi = float(w[0] ** 2 + w[1] ** 2)
    obj = cvar + lam_hhi * hhi
    mean_loss = float(losses.mean())
    print(f"{pe_pct_int:4d}% | {obj:9.6f} | {cvar:9.6f} | {lam_hhi * hhi:9.6f} | {mean_loss:9.6f}")
