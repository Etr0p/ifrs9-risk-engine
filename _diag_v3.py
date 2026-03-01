"""Diagnostic post-migration market impact logarithmique (v3)."""
import sys, warnings, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from ifrs9_cockpit.synthetic_generator import generate_dataset
from ifrs9_cockpit.config import (
    RANDOM_SEED, ASSET_CLASS_MAP, PREDEFINED_SCENARIOS,
    SCENARIO_BASE, ASSET_CLASSES,
)
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl
from ifrs9_cockpit.engine.comparator import PortfolioComparator

ORDER = [
    "covered_bonds", "sovereign", "corporate_loans", "retail_mortgage",
    "interbank", "corporate_bonds", "consumer_credit", "private_equity",
    "project_finance", "trade_finance", "repos_sft", "derivatives_cva",
    "equities", "structured_products",
]
SHORT = ["CB", "Sov", "Corp", "Mortg", "IB", "CorpB", "Conso", "PE",
         "PF", "TF", "Repo", "Der", "Eq", "Stru"]

_ADVERSE = {"Crise financiere (GFC)", "Crise souveraine (2012)", "Stagflation",
            "Choc pandemique (COVID)"}
_FAVORABLE = {"Reprise", "Hypercroissance", "Boom immobilier"}


def convert_scenario(raw):
    return {
        "unemployment_rate": SCENARIO_BASE.unemployment_rate + abs(raw.get("unemployment_bipolar", 0)),
        "gdp_growth": raw.get("gdp_pct", SCENARIO_BASE.gdp_growth),
        "interest_rate": SCENARIO_BASE.interest_rate + raw.get("interest_rate_bp", 0) / 100.0,
        "hpi_growth": raw.get("hpi_pct", SCENARIO_BASE.hpi_growth),
        "inflation_rate": raw.get("inflation_pct", SCENARIO_BASE.inflation_rate),
    }


def main():
    t0 = time.time()
    print("=" * 140)
    print("ANALYSE POST-MIGRATION : MARKET IMPACT LOGARITHMIQUE (v3)")
    print("=" * 140)

    # --- Pipeline ---
    bundle = generate_dataset(n_clients=5000, seed=RANDOM_SEED)
    pd_suite = PDModelSuite(seed=RANDOM_SEED)
    pd_suite.fit(bundle.df_credit)
    pd_curr = pd_suite.predict_active(bundle.df_credit)
    pd_orig = bundle.df_credit["pd_origination"].to_numpy()
    lgd_m = LGDModel(); lgd_m.fit(bundle.df_credit)
    ead_m = EADModel(); ead_m.fit(bundle.df_credit)
    ecl_calc = ECLCalculator(lgd_model=lgd_m, ead_model=ead_m)
    pe_calc = PECalculator()

    # --- Run all scenarios ---
    results = {}
    for sc_name, raw in PREDEFINED_SCENARIOS.items():
        macro = convert_scenario(raw)
        res_c = ecl_calc.calculate(
            bundle.df_credit, pd_curr, pd_orig,
            unemployment_override=macro["unemployment_rate"],
            gdp_override=macro["gdp_growth"],
            interest_rate_override=macro["interest_rate"],
            hpi_override=macro["hpi_growth"],
            inflation_override=macro["inflation_rate"],
        )
        res_p = pe_calc.calculate(
            bundle.df_pe,
            unemployment_override=macro["unemployment_rate"],
            gdp_override=macro["gdp_growth"],
            interest_rate_override=macro["interest_rate"],
            hpi_override=macro["hpi_growth"],
            inflation_override=macro["inflation_rate"],
            unemployment_crisis=(raw.get("unemployment_bipolar", 0) < 0),
        )
        bs_ecl = compute_balance_sheet_ecl(bundle.df_balance_sheet, macro)
        comp = PortfolioComparator(res_c, res_p, bs_ecl)
        opt = comp.optimize_allocation(macro)
        results[sc_name] = opt

    elapsed_pipe = time.time() - t0
    print(f"\nPipeline: {elapsed_pipe:.1f}s pour {len(PREDEFINED_SCENARIOS)} scenarios\n")

    # ============ TABLE 1: Allocation par scenario ============
    hdr = "Scenario".ljust(28)
    for s in SHORT:
        hdr += f" {s:>5s}"
    hdr += " |   HHI   LCR  RAROC  alpha"
    print(hdr)
    print("-" * len(hdr))

    for sc_name in PREDEFINED_SCENARIOS:
        opt = results[sc_name]
        cw = opt["class_weights"]
        hhi = sum(w ** 2 for w in cw.values()) * 10000
        lcr = opt["lcr_ratio"]
        raroc = opt["raroc_portfolio"]
        alpha = opt.get("risk_alpha", 0)
        row = sc_name[:28].ljust(28)
        for c in ORDER:
            row += f" {cw.get(c, 0):5.1%}"
        row += f" | {hhi:5.0f} {lcr:5.0%} {raroc:6.2%}  {alpha:.3f}"
        print(row)

    # ============ TABLE 2: Spread compression (Central) ============
    print("\n" + "=" * 90)
    print("SPREAD COMPRESSION (Central)")
    print("=" * 90)
    opt_cen = results["Central"]
    sc = opt_cen["spread_compression"]
    cw_cen = opt_cen["class_weights"]
    # Estimate total_ead
    total_ead_est = 1.3e12  # approximate
    print(f"{'Classe':25s} {'Poids':>7s} {'Mkt(Mds)':>9s} {'Share%':>7s} {'Compr':>7s} {'mu_ratio':>9s}")
    print("-" * 70)
    for c in ORDER:
        w = cw_cen.get(c, 0)
        mc = ASSET_CLASS_MAP[c].market_capacity_eur
        share = w * total_ead_est / mc
        comp_val = sc.get(c, 0)
        ratio = 1.0 / (1.0 + comp_val)
        print(f"{c:25s} {w:6.1%}  {mc / 1e9:7.0f}  {share:6.1%}  {comp_val:6.4f}  {ratio:7.1%}")

    # ============ TABLE 3: RAROC par classe (3 scenarios) ============
    for sc_disp, sc_key in [("Central", "Central"),
                             ("GFC", "Crise financiere (GFC)"),
                             ("Reprise", "Reprise")]:
        if sc_key not in results:
            continue
        opt = results[sc_key]
        cw = opt["class_weights"]
        # We need RAROC per class — use spread_compression + phase1
        print(f"\n--- Poids finaux ({sc_disp}) ---")
        top5 = sorted(cw.items(), key=lambda x: -x[1])[:5]
        for name, w in top5:
            sc_val = opt["spread_compression"].get(name, 0)
            print(f"  {name:25s}  w={w:5.1%}  compression={sc_val:.4f}")

    # ============ DIAGNOSTIC ============
    print("\n" + "=" * 90)
    print("DIAGNOSTIC")
    print("=" * 90)

    n_issues = 0
    for sc_name, opt in results.items():
        cw = opt["class_weights"]
        max_w = max(cw.values())
        max_cls = max(cw, key=cw.get)
        hhi = sum(w ** 2 for w in cw.values()) * 10000
        lcr = opt["lcr_ratio"]
        nsfr = opt["nsfr_ratio"]
        issues = []
        if max_w > 0.40:
            issues.append(f"max_w={max_w:.1%}({max_cls[:8]})")
        if hhi > 2000:
            issues.append(f"HHI={hhi:.0f}")
        if lcr < 1.0:
            issues.append(f"LCR={lcr:.0%}")
        if nsfr < 1.0:
            issues.append(f"NSFR={nsfr:.0%}")
        status = "OK" if not issues else ", ".join(issues)
        flag = "v" if not issues else "!"
        if issues:
            n_issues += 1
        print(f"  {flag} {sc_name[:28]:<28s}  {status}")

    # Flight-to-quality
    print("\n--- Flight-to-quality (Souverain en crise) ---")
    sov_cen = results["Central"]["class_weights"].get("sovereign", 0)
    for sc in ["Crise financiere (GFC)", "Crise souveraine (2012)", "Stagflation"]:
        if sc in results:
            sov_sc = results[sc]["class_weights"].get("sovereign", 0)
            delta = sov_sc - sov_cen
            flag = "v" if delta > 0 else "!"
            print(f"  {flag} {sc[:28]:<28s}  Sov={sov_sc:.1%} (Central={sov_cen:.1%}, delta={delta:+.1%})")

    # Paires identiques
    print("\n--- Allocations identiques (L2 < 0.5%) ---")
    sc_names = list(results.keys())
    n_identical = 0
    for i, name_i in enumerate(sc_names):
        for j, name_j in enumerate(sc_names):
            if j <= i:
                continue
            w_i = np.array([results[name_i]["class_weights"].get(c, 0) for c in ORDER])
            w_j = np.array([results[name_j]["class_weights"].get(c, 0) for c in ORDER])
            dist = np.linalg.norm(w_i - w_j)
            if dist < 0.005:
                n_identical += 1
                print(f"  ! {name_i[:25]:<25s}  ~  {name_j[:25]:<25s}  L2={dist:.6f}")
    if n_identical == 0:
        print("  v Aucune paire identique")

    # Differentiation adverse vs favorable
    print("\n--- Differentiation adverse vs favorable ---")
    adv_vecs = []
    fav_vecs = []
    for sc_name, opt in results.items():
        vec = np.array([opt["class_weights"].get(c, 0) for c in ORDER])
        if sc_name in _ADVERSE:
            adv_vecs.append(vec)
        elif sc_name in _FAVORABLE:
            fav_vecs.append(vec)
    if adv_vecs and fav_vecs:
        mean_adv = np.mean(adv_vecs, axis=0)
        mean_fav = np.mean(fav_vecs, axis=0)
        diff = mean_adv - mean_fav
        l2 = np.linalg.norm(diff)
        print(f"  L2(adverse - favorable) = {l2:.4f}")
        # Top differences
        diffs = [(ORDER[i], diff[i]) for i in range(len(ORDER))]
        diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, d in diffs[:5]:
            print(f"    {name:25s}  delta={d:+.2%}")

    # Phase 2 impact
    print("\n--- Impact Phase 2 (regulatory) ---")
    for sc_name in PREDEFINED_SCENARIOS:
        opt = results[sc_name]
        ra = opt["regulatory_adjustments"]
        total_d = ra["cet1_delta"] + ra["lcr_delta"] + ra["nsfr_delta"]
        if total_d > 0.001:
            print(f"  {sc_name[:28]:<28s}  CET1={ra['cet1_delta']:.4f}  LCR={ra['lcr_delta']:.4f}  NSFR={ra['nsfr_delta']:.4f}  total={total_d:.4f}")

    # Covered bonds analysis
    print("\n--- Covered Bonds : pourquoi dominant? ---")
    cb_weights = [(sc, opt["class_weights"].get("covered_bonds", 0)) for sc, opt in results.items()]
    cb_weights.sort(key=lambda x: -x[1])
    for sc, w in cb_weights:
        print(f"  {sc[:28]:<28s}  CB={w:.1%}")
    print(f"\n  CB profile: PD={ASSET_CLASS_MAP['covered_bonds'].pd_base:.4f}, "
          f"LGD={ASSET_CLASS_MAP['covered_bonds'].lgd_base:.2f}, "
          f"RW={ASSET_CLASS_MAP['covered_bonds'].rw_crr3:.0%}, "
          f"CIR={ASSET_CLASS_MAP['covered_bonds'].cir_class:.0%}, "
          f"mkt_cap={ASSET_CLASS_MAP['covered_bonds'].market_capacity_eur/1e9:.0f}B")

    print(f"\n{'=' * 90}")
    print(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
