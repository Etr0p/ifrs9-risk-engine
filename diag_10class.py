"""Diagnostic complet de l'optimiseur BL-CVaR 10 classes.

Trace chaque etape pour comprendre pourquoi PE n'obtient que 2%.
"""
import numpy as np
import pandas as pd

from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.engine.comparator import PortfolioComparator
from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl
from ifrs9_cockpit.config import (
    ASSET_CLASSES, ASSET_CLASS_MAP, ASSET_CLASS_NAMES,
    BASEL_CONFIG, PREDEFINED_SCENARIOS, SCENARIO_BASE,
    LGD_CONFIG, RANDOM_SEED,
)

print("=" * 80)
print("DIAGNOSTIC OPTIMISEUR BL-CVaR 10 CLASSES")
print("=" * 80)

# ── Generate data + models ──
df_credit, df_pe, df_history, df_bs = generate_dataset(n_clients=5000, seed=RANDOM_SEED)

pd_suite = PDModelSuite(seed=RANDOM_SEED)
pd_suite.fit(df_credit)
pd_current = pd_suite.predict_active(df_credit)
pd_origination = df_credit["pd_origination"].values

lgd_model = LGDModel()
lgd_model.fit(df_credit)
ead_model = EADModel()
ead_model.fit(df_credit)

ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)


def run_scenario(scenario_name, macro_params):
    """Run a single scenario and return comparator + optimization."""
    result_credit = ecl_calc.calculate(
        df_credit, pd_current, pd_origination,
        unemployment_override=macro_params.get("unemployment_rate"),
        gdp_override=macro_params.get("gdp_growth"),
        interest_rate_override=macro_params.get("interest_rate"),
        hpi_override=macro_params.get("hpi_growth"),
        inflation_override=macro_params.get("inflation_rate"),
    )
    pe_calc = PECalculator()
    result_pe = pe_calc.calculate(
        df_pe,
        unemployment_override=macro_params.get("unemployment_rate"),
        gdp_override=macro_params.get("gdp_growth"),
        interest_rate_override=macro_params.get("interest_rate"),
        hpi_override=macro_params.get("hpi_growth"),
        inflation_override=macro_params.get("inflation_rate"),
    )
    df_bs_ecl = compute_balance_sheet_ecl(df_bs, macro_params)
    comp = PortfolioComparator(result_credit, result_pe, df_bs_ecl)
    return comp, result_credit, result_pe


def convert_scenario(raw):
    """Convert slider-based scenario dict to real macro values."""
    return {
        "unemployment_rate": SCENARIO_BASE.unemployment_rate + abs(raw.get("unemployment_bipolar", 0)),
        "gdp_growth": raw.get("gdp_pct", SCENARIO_BASE.gdp_growth),
        "interest_rate": SCENARIO_BASE.interest_rate + raw.get("interest_rate_bp", 0) / 100.0,
        "hpi_growth": raw.get("hpi_pct", SCENARIO_BASE.hpi_growth),
        "inflation_rate": raw.get("inflation_pct", SCENARIO_BASE.inflation_rate),
    }


# ── Central scenario deep dive ──
central_raw = PREDEFINED_SCENARIOS["Central"]
macro = convert_scenario(central_raw)
print(f"\nScenario: Central")
print(f"Slider params: {central_raw}")
print(f"Real macro: {macro}")

comp, result_credit, result_pe = run_scenario("Central", macro)

# ── 1. RAROC multiclass ──
print("\n" + "-" * 60)
print("1. RAROC PAR CLASSE D'ACTIF")
print("-" * 60)
raroc_mc = comp.compute_raroc_multiclass()
for _, row in raroc_mc.iterrows():
    print(f"  {row['label']:25s}  RAROC={row['raroc']:8.2%}  "
          f"Exp={row['exposure']/1e9:7.2f}Md  RWA={row['rwa']/1e9:7.2f}Md  "
          f"Rev={row['revenue']/1e6:8.0f}M  Loss={row['loss']/1e6:8.0f}M")

# ── 2. Stress intensity ──
print("\n" + "-" * 60)
print("2. STRESS INTENSITY")
print("-" * 60)
classes_df = raroc_mc[raroc_mc["asset_class"] != "Total"].copy()
class_names = classes_df["asset_class"].values.tolist()
mu_raroc = classes_df["raroc"].values.astype(float).copy()

idx_corporate = class_names.index("corporate_loans")
idx_pe = class_names.index("private_equity")

r_c_spot = float(mu_raroc[idx_corporate])
irr_pe = result_pe["irr"].mean()
el_pe = result_pe["expected_loss_pe"].sum()
nav_pe = result_pe["nav"].sum()
r_p_spot = irr_pe - el_pe / max(nav_pe, 1)

print(f"  r_c_spot (corporate RAROC) = {r_c_spot:.4f}")
print(f"  irr_pe = {irr_pe:.4f}")
print(f"  el_pe/nav_pe = {el_pe/max(nav_pe,1):.4f}")
print(f"  r_p_spot (PE net return) = {r_p_spot:.4f}")

stress = comp._stress_intensity(r_c_spot, r_p_spot)
s_credit = -r_c_spot / 0.07
s_pe = -r_p_spot / 0.07
print(f"  s_credit = {s_credit:.4f}")
print(f"  s_pe     = {s_pe:.4f}")
print(f"  stress = max(s_c, s_pe) = {stress:.4f}")

# ── 3. Asymmetric equations ──
print("\n" + "-" * 60)
print("3. EQUATIONS ASYMETRIQUES")
print("-" * 60)
illiq = comp._asymmetric_illiquidity(stress)
vol_mult = comp._asymmetric_vol_multiplier(stress)
pe_band = comp._asymmetric_pe_band(stress)
conf_c, conf_pe = comp._bl_confidence(stress)

print(f"  illiquidity_premium = {illiq:.4f}")
print(f"  vol_multiplier = {vol_mult:.4f}")
print(f"  pe_band = [{pe_band[0]:.2%}, {pe_band[1]:.2%}]")
print(f"  bl_confidence = ({conf_c:.2f}, {conf_pe:.2f})")
print(f"  kappa_eff = 0.56 * {vol_mult:.4f} = {0.56 * vol_mult:.4f}")

# ── 4. RAROC after illiquidity penalty ──
print("\n" + "-" * 60)
print("4. RAROC APRES PENALITE ILLIQUIDITE")
print("-" * 60)
_ILLIQUID = {"private_equity", "project_finance", "structured_products"}
mu_adj = mu_raroc.copy()
for i, name in enumerate(class_names):
    if name in _ILLIQUID:
        mu_adj[i] -= illiq
    print(f"  {name:25s}: original={mu_raroc[i]:+.4f}  adjusted={mu_adj[i]:+.4f}")

# ── 5. Volatility per class ──
print("\n" + "-" * 60)
print("5. VOLATILITE PAR CLASSE")
print("-" * 60)
n = len(class_names)
vol = np.zeros(n)
for i, name in enumerate(class_names):
    ac = ASSET_CLASS_MAP[name]
    if name == "private_equity":
        vol[i] = max(0.15, irr_pe * 1.2)
    else:
        macro_intensity = sum(ac.macro_sensitivities.values()) / 5.0
        vol[i] = max(0.02, ac.pd_std * ac.lgd_base * 10 + 0.01 * macro_intensity)
    print(f"  {name:25s}  vol={vol[i]:.4f} ({vol[i]:.1%})")

# ── 6. Correlation matrix ──
print("\n" + "-" * 60)
print("6. MATRICE DE CORRELATION (PE row)")
print("-" * 60)
corr_10 = comp._build_corr_10x10()
for i, name in enumerate(class_names):
    print(f"  corr(PE, {name:25s}) = {corr_10[idx_pe, i]:.3f}")

# ── 7. Softmax weights ──
print("\n" + "-" * 60)
print("7. SOFTMAX RAROC-OPTIMAL")
print("-" * 60)
from ifrs9_cockpit.engine.comparator.optimizer import OptimizerMixin, _STRUCTURAL_FLOORS, _MIN_WEIGHT

# Variance analysis
var_mu = np.var(mu_adj)
scale = np.clip(1.0 / max(var_mu, 1e-6), 2.0, 50.0)
print(f"  var(mu_adj) = {var_mu:.6f}")
print(f"  softmax scale = 1/var = {1/max(var_mu,1e-6):.0f}, clipped to {scale:.1f}")
print(f"  scaled values: min={np.min(mu_adj*scale):.2f} max={np.max(mu_adj*scale):.2f}")
print(f"  range = {np.max(mu_adj*scale) - np.min(mu_adj*scale):.2f}")

w_naive = OptimizerMixin._softmax_weights_10(mu_adj)
print("\n  Naive softmax:")
for i, name in enumerate(class_names):
    print(f"    {name:25s} = {w_naive[i]:.4f} ({w_naive[i]:.1%})")

w_raroc = OptimizerMixin._softmax_10(mu_adj, corr_10)
print("\n  After correlation penalty:")
for i, name in enumerate(class_names):
    print(f"    {name:25s} = {w_raroc[i]:.4f} ({w_raroc[i]:.1%})")

# ── 8. Floor weights ──
print("\n" + "-" * 60)
print("8. POIDS PLANCHER (STRUCTURAL FLOORS)")
print("-" * 60)
w_floor = np.array([
    max(_STRUCTURAL_FLOORS.get(name, _MIN_WEIGHT), _MIN_WEIGHT)
    for name in class_names
])
w_floor_norm = w_floor / w_floor.sum()
for i, name in enumerate(class_names):
    print(f"  {name:25s} floor={w_floor[i]:.2f} norm={w_floor_norm[i]:.4f}")
print(f"  Sum floors: {w_floor.sum():.2f}")

# RAROC-optimal with floors enforced
w_raroc_floored = np.maximum(w_raroc, w_floor)
w_raroc_floored = w_raroc_floored / w_raroc_floored.sum()
print("\n  RAROC-optimal with floors enforced:")
for i, name in enumerate(class_names):
    print(f"    {name:25s} = {w_raroc_floored[i]:.4f} ({w_raroc_floored[i]:.1%})")

# ── 9. Grid search trace ──
print("\n" + "-" * 60)
print("9. GRID SEARCH (alpha trace)")
print("-" * 60)
Sigma = np.outer(vol, vol) * corr_10
Sigma = (Sigma + Sigma.T) / 2
eigvals = np.linalg.eigvalsh(Sigma)
if eigvals.min() < 1e-8:
    Sigma += np.eye(n) * (1e-6 - min(0, eigvals.min()))
L_chol = np.linalg.cholesky(Sigma)

rng = np.random.default_rng(42)
n_scenarios = 5000
Z = rng.standard_normal((n_scenarios, n))
scenarios_0 = Z @ L_chol.T
cutoff = max(int(n_scenarios * 0.05), 1)

kappa_eff = 0.56 * vol_mult
lambda_hhi = 0.15
print(f"  kappa_eff = {kappa_eff:.4f}")
print(f"  lambda_hhi = {lambda_hhi:.4f}")

best_obj = -1e10
best_alpha = 0.5
best_w = w_floor_norm.copy()

for alpha_trial in np.linspace(0.0, 1.0, 201):
    w_trial = (1 - alpha_trial) * w_floor_norm + alpha_trial * w_raroc_floored
    w_trial = np.maximum(w_trial, w_floor)
    w_trial = w_trial / w_trial.sum()
    if w_trial[idx_pe] > pe_band[1]:
        excess = w_trial[idx_pe] - pe_band[1]
        w_trial[idx_pe] = pe_band[1]
        others_mask = np.ones(n, dtype=bool)
        others_mask[idx_pe] = False
        others_sum = w_trial[others_mask].sum()
        if others_sum > 0:
            w_trial[others_mask] += excess * w_trial[others_mask] / others_sum
        w_trial = w_trial / w_trial.sum()

    mu_p = float(w_trial @ mu_adj)
    port_dev = scenarios_0 @ w_trial
    port_sorted = np.sort(port_dev)
    cvar_0 = -float(np.mean(port_sorted[:cutoff]))
    hhi_norm = float(np.sum(w_trial ** 2))
    obj = mu_p - kappa_eff * cvar_0 - lambda_hhi * hhi_norm

    if alpha_trial in [0.0, 0.25, 0.50, 0.75, 1.0] or abs(alpha_trial - 0.01) < 0.003:
        print(f"  alpha={alpha_trial:.2f}: obj={obj:.6f}"
              f"  mu={mu_p:.4f}  CVaR={cvar_0:.4f}  HHI={hhi_norm:.4f}"
              f"  PE_w={w_trial[idx_pe]:.4f}")

    if obj > best_obj:
        best_obj = obj
        best_alpha = alpha_trial
        best_w = w_trial.copy()

print(f"\n  ** BEST alpha = {best_alpha:.3f}, obj = {best_obj:.6f}")
print(f"  ** BEST PE weight = {best_w[idx_pe]:.4f}")
print("\n  Final allocation:")
for i, name in enumerate(class_names):
    print(f"    {name:25s} = {best_w[i]:.4f} ({best_w[i]:.1%})")

# ── 10. Decompose objective ──
print("\n" + "-" * 60)
print("10. DECOMPOSITION DE L'OBJECTIF A L'OPTIMUM")
print("-" * 60)
mu_opt = float(best_w @ mu_adj)
port_dev_opt = scenarios_0 @ best_w
cvar_opt = -float(np.mean(np.sort(port_dev_opt)[:cutoff]))
hhi_opt = float(np.sum(best_w ** 2))

print(f"  mu(w)             = {mu_opt:+.6f}")
print(f"  kappa*CVaR(w)     = {kappa_eff * cvar_opt:+.6f}")
print(f"  lambda*HHI(w)     = {lambda_hhi * hhi_opt:+.6f}")
print(f"  objectif          = {mu_opt - kappa_eff*cvar_opt - lambda_hhi*hhi_opt:+.6f}")
denom = abs(mu_opt) + abs(kappa_eff * cvar_opt) + abs(lambda_hhi * hhi_opt)
print(f"\n  Contribution relative:")
print(f"    mu     : {abs(mu_opt)/denom:.1%}")
print(f"    CVaR   : {abs(kappa_eff*cvar_opt)/denom:.1%}")
print(f"    HHI    : {abs(lambda_hhi*hhi_opt)/denom:.1%}")

# ── 11. Sensitivity: force PE levels ──
print("\n" + "-" * 60)
print("11. SENSIBILITE: OBJECTIF SI ON FORCE PE")
print("-" * 60)
for pe_target in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    w_test = best_w.copy()
    w_test[idx_pe] = pe_target
    others_mask = np.ones(n, dtype=bool)
    others_mask[idx_pe] = False
    others_sum = w_test[others_mask].sum()
    if others_sum > 0:
        w_test[others_mask] *= (1 - pe_target) / others_sum
    w_test = w_test / w_test.sum()
    mu_t = float(w_test @ mu_adj)
    dev_t = scenarios_0 @ w_test
    cvar_t = -float(np.mean(np.sort(dev_t)[:cutoff]))
    hhi_t = float(np.sum(w_test ** 2))
    obj_t = mu_t - kappa_eff * cvar_t - lambda_hhi * hhi_t
    print(f"  PE={pe_target:5.1%}: obj={obj_t:+.6f}  mu={mu_t:.4f}  "
          f"CVaR={cvar_t:.4f}  HHI={hhi_t:.4f}")

# ── 12. Actual optimizer output ──
print("\n" + "-" * 60)
print("12. SORTIE optimize_allocation()")
print("-" * 60)
opt = comp.optimize_allocation(macro)
print(f"  method = {opt['method']}")
print(f"  pe_free = {opt['pe_free']:.2%}")
print(f"  pe_allocation = {opt['pe_allocation']:.2%}")
print(f"  credit_allocation = {opt['credit_allocation']:.2%}")
print(f"  pe_band = {opt['pe_band']}")
print(f"  stress_intensity = {opt['stress_intensity']}")
print(f"  risk_alpha = {opt['risk_alpha']}")
for k, v in opt["class_weights"].items():
    print(f"    {k:25s} = {v:.4f} ({v:.1%})")

# ── 13. Cross-scenario ──
print("\n" + "=" * 80)
print("13. AUDIT CROSS-SCENARIO")
print("=" * 80)
print(f"  {'Scenario':30s} | {'PE':>5s} {'Sov':>5s} {'Corp':>5s} | {'RAROC':>6s} {'CET1':>6s} | {'alpha':>5s} {'stress':>6s}")
print("  " + "-" * 90)

for sc_name, sc_raw in PREDEFINED_SCENARIOS.items():
    macro_s = convert_scenario(sc_raw)
    comp_s, _, _ = run_scenario(sc_name, macro_s)
    opt_s = comp_s.optimize_allocation(macro_s)
    cw = opt_s["class_weights"]
    print(f"  {sc_name:30s} | {cw.get('private_equity',0):5.1%} "
          f"{cw.get('sovereign',0):5.1%} {cw.get('corporate_loans',0):5.1%} "
          f"| {opt_s['raroc_portfolio']:.2%} {opt_s['cet1_ratio']:.2%} "
          f"| {opt_s['risk_alpha']:.2f} {opt_s['stress_intensity']:+.2f}")

print("\n" + "=" * 80)
print("AUDIT TERMINE")
print("=" * 80)
