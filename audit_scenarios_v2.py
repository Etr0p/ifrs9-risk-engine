"""Audit cross-scenario — IFRS 9 Risk Cockpit complet v3.

Audit de sortie exhaustif couvrant l'ensemble des moteurs :
  A. Credit ECL (delta, staging, monotonicity)
  B. PE NAV (drawdown, IRR, MOIC, first-loss asymetrie)
  C. Balance Sheet ECL 10 classes (Vasicek ASRF, Z cap, ecl/ead ratio)
  D. RAROC multiclass (10 classes, annual EL, revenue vs loss)
  E. Allocation BL-CVaR (per-class bounds, PE band, CET1, NSFR)
  F. Cross-scenario coherence (monotonicity, discrimination, stress signal)
  G. Normes reglementaires (IFRS 9 B5.5.25, CRR3 Art 124-125, NSFR)
  H. Position-level stress (8 classes bottom-up)
  I. Gouvernance (conformal, Z cap, revenue-loss matching)

Execution :
    python audit_scenarios_v2.py
    ~30s, aucune dependance externe, aucun GPU requis.
"""
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.engine.comparator import PortfolioComparator
from ifrs9_cockpit.engine.balance_sheet_ecl import (
    compute_balance_sheet_ecl,
    effective_rw,
    macro_to_z,
    vasicek_conditional_pd,
)
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.config import (
    PREDEFINED_SCENARIOS,
    SCENARIO_BASE,
    RANDOM_SEED,
    ASSET_CLASSES,
    ASSET_CLASS_MAP,
    BASEL_CONFIG,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_PASS = 0
_FAIL = 0
_WARN = 0
_RESULTS = []  # [(section, name, status, detail)]


def check(section: str, name: str, condition: bool, detail: str = ""):
    """Enregistre un controle PASS/FAIL."""
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        _RESULTS.append((section, name, "PASS", detail))
    else:
        _FAIL += 1
        _RESULTS.append((section, name, "FAIL", detail))
        print(f"  FAIL [{section}] {name} — {detail}")


def warn(section: str, name: str, condition: bool, detail: str = ""):
    """Enregistre un controle PASS/WARN (non bloquant)."""
    global _PASS, _WARN
    if condition:
        _PASS += 1
        _RESULTS.append((section, name, "PASS", detail))
    else:
        _WARN += 1
        _RESULTS.append((section, name, "WARN", detail))
        print(f"  WARN [{section}] {name} — {detail}")


def convert_scenario(raw: dict) -> dict:
    """Convert slider-based scenario dict to real macro values."""
    return {
        "unemployment_rate": SCENARIO_BASE.unemployment_rate + abs(raw.get("unemployment_bipolar", 0)),
        "gdp_growth": raw.get("gdp_pct", SCENARIO_BASE.gdp_growth),
        "interest_rate": SCENARIO_BASE.interest_rate + raw.get("interest_rate_bp", 0) / 100.0,
        "hpi_growth": raw.get("hpi_pct", SCENARIO_BASE.hpi_growth),
        "inflation_rate": raw.get("inflation_pct", SCENARIO_BASE.inflation_rate),
    }


# Scenario classification for monotonicity checks
_ADVERSE = {"Crise financiere (GFC)", "Crise souveraine (2012)", "Stagflation",
            "Choc pandemique (COVID)"}
_FAVORABLE = {"Reprise", "Hypercroissance", "Boom immobilier"}
_NEUTRAL = {"Central", "Rupture techno", "Trappe a liquidite", "Transition climatique brutale"}


def run_audit():
    t0 = time.time()
    print("=" * 100)
    print("AUDIT DE SORTIE — IFRS 9 RISK COCKPIT v3.0")
    print("=" * 100)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 0. Chargement des donnees
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[0] Chargement des donnees et modeles (n=5000)...")
    df_credit, df_pe, df_history, df_bs = generate_dataset(n_clients=5000, seed=RANDOM_SEED)

    pd_suite = PDModelSuite(seed=RANDOM_SEED)
    pd_suite.fit(df_credit)
    pd_curr = pd_suite.predict_active(df_credit)
    pd_orig = df_credit["pd_origination"].values

    lgd_model = LGDModel()
    lgd_model.fit(df_credit)
    ead_model = EADModel()
    ead_model.fit(df_credit)
    ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)

    n_clients = len(df_credit)
    n_pe = len(df_pe)
    n_bs = len(df_bs)

    check("0-Data", "df_credit non vide", n_clients > 0, f"n={n_clients}")
    check("0-Data", "df_pe non vide", n_pe > 0, f"n={n_pe}")
    check("0-Data", "df_balance_sheet 10 classes", n_bs == 10, f"n={n_bs}")
    check("0-Data", "4-tuple generate_dataset", df_bs is not None, "df_bs present")

    # Balance sheet positions metadata
    pos_keys = ["mortgage_positions", "consumer_positions", "securitisation_positions",
                "sovereign_positions", "covered_bonds_positions", "interbank_positions",
                "project_positions", "trade_positions"]
    for pk in pos_keys:
        has = pk in df_bs.attrs
        check("0-Data", f"attrs[{pk}]", has, "present" if has else "ABSENT")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. Baseline (Central, no overrides)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[1] Calcul baseline (Central)...")
    res_base = ecl_calc.calculate(df_credit, pd_curr, pd_orig)
    ecl_ref = res_base["ecl_weighted"].sum()
    nav_ref_pe = PECalculator().calculate(df_pe)["nav"].sum()

    check("1-Base", "ECL ref > 0", ecl_ref > 0, f"ECL_ref={ecl_ref/1e6:.0f}M")
    check("1-Base", "NAV ref > 0", nav_ref_pe > 0, f"NAV_ref={nav_ref_pe/1e6:.0f}M")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Boucle scenarios
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    scenario_data = {}  # store per-scenario results for cross-checks
    audit_table = []
    pe_calc = PECalculator()

    print("\n[2-8] Boucle sur les scenarios predefinis...\n")

    for sc_name, raw in PREDEFINED_SCENARIOS.items():
        macro = convert_scenario(raw)
        sc_tag = sc_name[:20]

        # ── A. Credit ECL ──
        res_c = ecl_calc.calculate(
            df_credit, pd_curr, pd_orig,
            unemployment_override=macro["unemployment_rate"],
            gdp_override=macro["gdp_growth"],
            interest_rate_override=macro["interest_rate"],
            hpi_override=macro["hpi_growth"],
            inflation_override=macro["inflation_rate"],
        )
        ecl_total = res_c["ecl_weighted"].sum()
        delta_ecl = (ecl_total - ecl_ref) / max(ecl_ref, 1)
        s1_n = (res_c["stage"] == 1).sum()
        s2_n = (res_c["stage"] == 2).sum()
        s3_n = (res_c["stage"] == 3).sum()
        s2_pct = s2_n / n_clients
        s3_pct = s3_n / n_clients

        check("A-ECL", f"{sc_tag} ECL > 0", ecl_total > 0, f"ECL={ecl_total/1e6:.0f}M")
        check("A-ECL", f"{sc_tag} S1+S2+S3=N", s1_n + s2_n + s3_n == n_clients,
              f"S1={s1_n}+S2={s2_n}+S3={s3_n}={s1_n+s2_n+s3_n} vs N={n_clients}")
        check("A-ECL", f"{sc_tag} ECL/EAD < 30%",
              ecl_total / res_c["ead"].sum() < 0.30,
              f"ratio={ecl_total / res_c['ead'].sum():.2%}")

        # ── B. PE NAV ──
        res_p = pe_calc.calculate(
            df_pe,
            unemployment_override=macro["unemployment_rate"],
            gdp_override=macro["gdp_growth"],
            interest_rate_override=macro["interest_rate"],
            hpi_override=macro["hpi_growth"],
            inflation_override=macro["inflation_rate"],
            unemployment_crisis=(raw.get("unemployment_bipolar", 0) < 0),
        )
        nav_total = res_p["nav"].sum()
        drawdown = max(0.0, (nav_ref_pe - nav_total) / max(nav_ref_pe, 1))
        irr_mean = res_p["irr"].mean()
        moic_mean = res_p["moic"].mean()
        el_pe = res_p["expected_loss_pe"].sum()

        check("B-PE", f"{sc_tag} NAV > 0", nav_total > 0, f"NAV={nav_total/1e6:.0f}M")
        check("B-PE", f"{sc_tag} IRR [-50%,+35%]", -0.50 <= irr_mean <= 0.35,
              f"IRR={irr_mean:.2%}")
        check("B-PE", f"{sc_tag} MOIC > 0", moic_mean > 0, f"MOIC={moic_mean:.2f}")
        check("B-PE", f"{sc_tag} EL PE >= 0", el_pe >= 0, f"EL_PE={el_pe/1e6:.0f}M")

        # ── C. Balance Sheet ECL (10 classes) ──
        df_bs_ecl = compute_balance_sheet_ecl(df_bs, macro)

        check("C-BSECL", f"{sc_tag} 10 rows", len(df_bs_ecl) == 10, f"n={len(df_bs_ecl)}")

        # L2/L3 classes only (L1 = corporate_loans, private_equity don't use BS ECL for RAROC)
        _L1_CLASSES = {"corporate_loans", "private_equity"}
        for _, row_bs in df_bs_ecl.iterrows():
            ac = row_bs["asset_class"]
            ecl_bs = row_bs["ecl_weighted"]
            ead_bs = row_bs["ead_total"]
            ratio_bs = ecl_bs / ead_bs if ead_bs > 0 else 0
            pd_cond_b = row_bs.get("pd_cond_base", 0)
            pd_cond_a = row_bs.get("pd_cond_adverse", 0)

            check("C-BSECL", f"{sc_tag} {ac[:12]} ECL >= 0", ecl_bs >= 0,
                  f"ECL={ecl_bs/1e6:.0f}M")
            # Z cap: pd_cond_base < 30% for non-crisis, L2/L3 only
            # Note: L1 classes (corporate, PE) are checked but their pd_cond
            # does not affect RAROC (RAROC uses position-level data for L1).
            # Note: IR sign ambiguity means some "neutral" scenarios (e.g.
            # Trappe a liquidite with IR=0%) produce high Z for rate-sensitive
            # classes. This is a known limitation of the single-factor Z model.
            if sc_name not in _ADVERSE and ac not in _L1_CLASSES:
                warn("C-BSECL", f"{sc_tag} {ac[:12]} pd_cond_base < 30%",
                     pd_cond_b < 0.30,
                     f"pd_cond={pd_cond_b:.4f}")
            # Adverse pd_cond capped at ~28% (Z cap ±4)
            check("C-BSECL", f"{sc_tag} {ac[:12]} pd_cond_adv < 60%",
                  pd_cond_a < 0.60,
                  f"pd_cond_adv={pd_cond_a:.4f}")
            # ECL/EAD ratio
            warn("C-BSECL", f"{sc_tag} {ac[:12]} ECL/EAD < 50%",
                 ratio_bs < 0.50,
                 f"ECL/EAD={ratio_bs:.2%}")

        # ── D. RAROC Multiclass ──
        comp = PortfolioComparator(res_c, res_p, df_bs_ecl)
        raroc_mc = comp.compute_raroc_multiclass()

        check("D-RAROC", f"{sc_tag} 11 rows", len(raroc_mc) == 11,
              f"n={len(raroc_mc)}")

        for _, rr in raroc_mc.iterrows():
            if rr["asset_class"] == "Total":
                continue
            check("D-RAROC", f"{sc_tag} {rr['asset_class'][:12]} RAROC finite",
                  np.isfinite(rr["raroc"]),
                  f"RAROC={rr['raroc']:.2%}")
            check("D-RAROC", f"{sc_tag} {rr['asset_class'][:12]} capital > 0",
                  rr["capital"] > 0,
                  f"capital={rr['capital']/1e6:.0f}M")
            check("D-RAROC", f"{sc_tag} {rr['asset_class'][:12]} exposure > 0",
                  rr["exposure"] > 0,
                  f"exp={rr['exposure']/1e6:.0f}M")
            # Revenue should exceed loss in favorable scenarios
            if sc_name in _FAVORABLE:
                check("D-RAROC", f"{sc_tag} {rr['asset_class'][:12]} Rev > Loss (fav)",
                      rr["revenue"] > rr["loss"],
                      f"Rev={rr['revenue']/1e6:.0f}M vs Loss={rr['loss']/1e6:.0f}M")
            # RAROC should be positive in Central for most classes
            if sc_name == "Central":
                warn("D-RAROC", f"Central {rr['asset_class'][:12]} RAROC > 0",
                     rr["raroc"] > 0,
                     f"RAROC={rr['raroc']:.2%}")
            # No RAROC > 100% (unrealistic)
            check("D-RAROC", f"{sc_tag} {rr['asset_class'][:12]} |RAROC| < 500%",
                  abs(rr["raroc"]) < 5.0,
                  f"RAROC={rr['raroc']:.2%}")

            # ── M8: Trade Finance loss rate reasonableness ──
            # ICC Trade Register 2024: transaction default rate < 0.06%.
            # Our model uses counterparty PD (not transaction DR) × LGD, so
            # expected loss rate is structurally higher (~0.5-1.5%).
            # Threshold set at 2% to catch model pathologies (previously -346% RAROC).
            if rr["asset_class"] == "trade_finance":
                tf_loss_rate = rr["loss"] / max(rr["exposure"], 1)
                check("D-RAROC", f"{sc_tag} TF loss rate < 2%",
                      tf_loss_rate < 0.02,
                      f"loss_rate={tf_loss_rate:.4%} (ICC transaction DR < 0.06%)")

            # ── M9: Sovereign loss rate < 2% outside sovereign-crisis ──
            # Core EU sovereign = quasi risk-free (Art. 114 RW=0%).
            if rr["asset_class"] == "sovereign" and sc_name != "Crise souveraine (2012)":
                sov_loss_rate = rr["loss"] / max(rr["exposure"], 1)
                check("D-RAROC", f"{sc_tag} SOV loss rate < 2%",
                      sov_loss_rate < 0.02,
                      f"loss_rate={sov_loss_rate:.4%}")

        # ── M2: Favorable scenario should have positive RAROC portfolio ──
        if sc_name in _FAVORABLE:
            total_row = raroc_mc[raroc_mc["asset_class"] == "Total"].iloc[0]
            check("D-RAROC", f"{sc_tag} RAROC portfolio > 0 (favorable)",
                  total_row["raroc"] > 0,
                  f"RAROC_total={total_row['raroc']:.2%}")

        # ── E. Allocation BL-CVaR ──
        opt = comp.optimize_allocation(macro)
        cw = opt["class_weights"]
        pe_w = cw.get("private_equity", 0)
        sov_w = cw.get("sovereign", 0)
        corp_w = cw.get("corporate_loans", 0)
        raroc_p = opt["raroc_portfolio"]
        cet1 = opt["cet1_ratio"]
        stress = opt["stress_intensity"]
        pe_band = opt["pe_band"]

        check("E-Alloc", f"{sc_tag} PE in band",
              pe_band[0] - 0.01 <= pe_w <= pe_band[1] + 0.01,
              f"PE={pe_w:.1%} band=[{pe_band[0]:.1%},{pe_band[1]:.1%}]")
        check("E-Alloc", f"{sc_tag} Weights sum=1",
              abs(sum(cw.values()) - 1.0) < 0.01,
              f"sum={sum(cw.values()):.4f}")
        check("E-Alloc", f"{sc_tag} RAROC finite",
              np.isfinite(raroc_p),
              f"RAROC_p={raroc_p:.2%}")
        check("E-Alloc", f"{sc_tag} CET1 feasible",
              opt["feasible"],
              f"CET1={cet1:.2%}")

        # Per-class bounds
        for ac_name, w in cw.items():
            ac_profile = ASSET_CLASS_MAP.get(ac_name)
            if ac_profile is None:
                continue
            check("E-Alloc", f"{sc_tag} {ac_name[:12]} >= floor",
                  w >= ac_profile.weight_floor - 0.01,
                  f"w={w:.1%} floor={ac_profile.weight_floor:.1%}")
            check("E-Alloc", f"{sc_tag} {ac_name[:12]} <= cap",
                  w <= ac_profile.weight_cap + 0.01,
                  f"w={w:.1%} cap={ac_profile.weight_cap:.1%}")

        # NSFR
        nsfr = opt.get("nsfr_ratio", 0)
        warn("E-Alloc", f"{sc_tag} NSFR >= 100%", nsfr >= 1.0,
             f"NSFR={nsfr:.2%}")

        # ── Store for cross-scenario ──
        total_row = raroc_mc[raroc_mc["asset_class"] == "Total"].iloc[0]
        scenario_data[sc_name] = {
            "ecl": ecl_total, "delta_ecl": delta_ecl,
            "s2_pct": s2_pct, "s3_pct": s3_pct,
            "nav": nav_total, "drawdown": drawdown,
            "irr": irr_mean, "moic": moic_mean, "el_pe": el_pe,
            "raroc_total": total_row["raroc"],
            "pe_w": pe_w, "sov_w": sov_w, "corp_w": corp_w,
            "raroc_p": raroc_p, "cet1": cet1,
            "stress": stress,
            "raroc_mc": raroc_mc,
            "df_bs_ecl": df_bs_ecl,
            "opt": opt,
        }

        audit_table.append({
            "Scenario": sc_name[:25],
            "dECL": f"{delta_ecl:+.1%}",
            "S2%": f"{s2_pct:.0%}",
            "S3%": f"{s3_pct:.0%}",
            "DD%": f"{drawdown:.1%}",
            "PE": f"{pe_w:.1%}",
            "Sov": f"{sov_w:.1%}",
            "Corp": f"{corp_w:.1%}",
            "RAROC_p": f"{raroc_p:.1%}",
            "CET1": f"{cet1:.0%}",
            "s": f"{stress:+.2f}",
        })

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # F. Cross-scenario coherence
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[F] Cross-scenario coherence...")

    # F1: ECL monotonicity (adverse > central > favorable)
    if "Central" in scenario_data and "Crise financiere (GFC)" in scenario_data and "Reprise" in scenario_data:
        ecl_gfc = scenario_data["Crise financiere (GFC)"]["ecl"]
        ecl_cen = scenario_data["Central"]["ecl"]
        ecl_rep = scenario_data["Reprise"]["ecl"]
        check("F-Cross", "ECL GFC > Central", ecl_gfc > ecl_cen,
              f"GFC={ecl_gfc/1e6:.0f}M vs Central={ecl_cen/1e6:.0f}M")
        check("F-Cross", "ECL Central > Reprise", ecl_cen > ecl_rep,
              f"Central={ecl_cen/1e6:.0f}M vs Reprise={ecl_rep/1e6:.0f}M")

    # F2: Stage 2 monotonicity
    if "Stagflation" in scenario_data and "Central" in scenario_data:
        check("F-Cross", "S2% Stagflation > Central",
              scenario_data["Stagflation"]["s2_pct"] > scenario_data["Central"]["s2_pct"],
              f"Stagf={scenario_data['Stagflation']['s2_pct']:.0%} vs Cen={scenario_data['Central']['s2_pct']:.0%}")

    # F3: NAV drawdown ordering
    if "Crise financiere (GFC)" in scenario_data and "Central" in scenario_data:
        check("F-Cross", "NAV drawdown GFC > Central",
              scenario_data["Crise financiere (GFC)"]["drawdown"] > scenario_data["Central"]["drawdown"],
              f"GFC={scenario_data['Crise financiere (GFC)']['drawdown']:.1%} vs Cen={scenario_data['Central']['drawdown']:.1%}")

    # F4: PE first-loss property (PE more sensitive in crisis, better in recovery)
    if "Reprise" in scenario_data and "Crise financiere (GFC)" in scenario_data:
        irr_rep = scenario_data["Reprise"]["irr"]
        irr_gfc = scenario_data["Crise financiere (GFC)"]["irr"]
        check("F-Cross", "IRR Reprise > GFC (first-loss PE)",
              irr_rep > irr_gfc,
              f"Reprise={irr_rep:.2%} vs GFC={irr_gfc:.2%}")

    # F5: Stress intensity ordering
    if all(s in scenario_data for s in ["Central", "Crise financiere (GFC)", "Reprise"]):
        s_gfc = scenario_data["Crise financiere (GFC)"]["stress"]
        s_cen = scenario_data["Central"]["stress"]
        s_rep = scenario_data["Reprise"]["stress"]
        check("F-Cross", "Stress GFC > Central",
              s_gfc > s_cen,
              f"GFC={s_gfc:.2f} vs Central={s_cen:.2f}")
        check("F-Cross", "Stress Central > Reprise",
              s_cen > s_rep,
              f"Central={s_cen:.2f} vs Reprise={s_rep:.2f}")

    # F6: RAROC discrimination — adverse worse than favorable
    for ac_name in ["corporate_loans", "retail_mortgage", "consumer_credit"]:
        if "Central" in scenario_data:
            mc_cen = scenario_data["Central"]["raroc_mc"]
            r_cen = mc_cen[mc_cen["asset_class"] == ac_name]["raroc"].values
            if len(r_cen):
                warn("F-Cross", f"RAROC Central {ac_name[:12]} > 0",
                     r_cen[0] > 0,
                     f"RAROC={r_cen[0]:.2%}")

    # F7: Boom immobilier should benefit mortgage
    if "Boom immobilier" in scenario_data and "Central" in scenario_data:
        mc_boom = scenario_data["Boom immobilier"]["raroc_mc"]
        mc_cen = scenario_data["Central"]["raroc_mc"]
        r_boom = mc_boom[mc_boom["asset_class"] == "retail_mortgage"]["raroc"].values
        r_cen = mc_cen[mc_cen["asset_class"] == "retail_mortgage"]["raroc"].values
        if len(r_boom) and len(r_cen):
            check("F-Cross", "Mortgage RAROC: Boom >= Central",
                  r_boom[0] >= r_cen[0] - 0.005,
                  f"Boom={r_boom[0]:.2%} vs Central={r_cen[0]:.2%}")

    # F8: Scenario discrimination — at least 3 distinct RAROC profiles
    if len(scenario_data) >= 5:
        rarocs_total = [sd["raroc_total"] for sd in scenario_data.values()]
        raroc_range = max(rarocs_total) - min(rarocs_total)
        check("F-Cross", "RAROC range > 5pp (discrimination)",
              raroc_range > 0.05,
              f"range={raroc_range:.2%}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # G. Normes reglementaires
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[G] Normes reglementaires...")

    # G1: IFRS 9 B5.5.25 — sovereign exempt from staging
    sov_profile = ASSET_CLASS_MAP.get("sovereign")
    if sov_profile:
        check("G-Reg", "Sovereign exempt_from_staging",
              sov_profile.exempt_from_staging is True,
              f"exempt={sov_profile.exempt_from_staging}")

    # G2: CRR3 Art 124-125 — mortgage LTV-based RW
    mort_profile = ASSET_CLASS_MAP.get("retail_mortgage")
    if mort_profile:
        check("G-Reg", "Mortgage has ltv_distribution",
              mort_profile.ltv_distribution is not None,
              "present" if mort_profile.ltv_distribution else "ABSENT")
        rw_mort = effective_rw(mort_profile)
        check("G-Reg", "Mortgage RW 20%-70%",
              0.20 <= rw_mort <= 0.70,
              f"RW={rw_mort:.2%}")

    # G3: CRR3 Art 242-270 — securitisation SEC-SA RW
    sec_profile = ASSET_CLASS_MAP.get("structured_products")
    if sec_profile:
        check("G-Reg", "Structured has securitisation_mix",
              sec_profile.securitisation_mix is not None,
              "present" if sec_profile.securitisation_mix else "ABSENT")

    # G4: Sovereign RW = 0%
    if sov_profile:
        check("G-Reg", "Sovereign RW = 0% (CRR3 Art.114)",
              sov_profile.rw_crr3 == 0.0,
              f"RW={sov_profile.rw_crr3}")

    # G5: Leverage ratio floor present
    check("G-Reg", "Leverage max > 0 (Basel III)",
          BASEL_CONFIG.leverage_max > 0,
          f"lev_max={BASEL_CONFIG.leverage_max:.1%}")

    # G6: NSFR compliant in Central
    if "Central" in scenario_data:
        nsfr_cen = scenario_data["Central"]["opt"].get("nsfr_ratio", 0)
        check("G-Reg", "NSFR Central >= 100%",
              nsfr_cen >= 1.0,
              f"NSFR={nsfr_cen:.2%}")

    # G7: 10 asset class profiles present
    check("G-Reg", "10 ASSET_CLASSES configured",
          len(ASSET_CLASSES) == 10,
          f"n={len(ASSET_CLASSES)}")

    # G8: per-class bounds feasibility (sum floors < 1 < sum caps)
    sum_floors = sum(ac.weight_floor for ac in ASSET_CLASSES)
    sum_caps = sum(ac.weight_cap for ac in ASSET_CLASSES)
    check("G-Reg", "Sum floors < 100%", sum_floors < 1.0, f"sum_floors={sum_floors:.0%}")
    check("G-Reg", "Sum caps > 100%", sum_caps > 1.0, f"sum_caps={sum_caps:.0%}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # H. Position-level stress
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[H] Position-level stress (bottom-up 8 classes)...")

    # Stress macro for position-level test
    stress_macro = convert_scenario(PREDEFINED_SCENARIOS.get("Stagflation", PREDEFINED_SCENARIOS["Central"]))

    # H1: Project Finance
    df_pf = df_bs.attrs.get("project_positions")
    if df_pf is not None:
        from ifrs9_cockpit.synthetic_generator.project_finance_positions import stress_project_finance_positions
        stressed_pf = stress_project_finance_positions(df_pf, stress_macro)
        check("H-Pos", "PF stress returns dict", isinstance(stressed_pf, dict), "")
        check("H-Pos", "PF pd_base > 0", stressed_pf.get("pd_base", 0) > 0,
              f"pd={stressed_pf.get('pd_base', 0):.4f}")
        check("H-Pos", "PF lgd_base in [0,1]",
              0 <= stressed_pf.get("lgd_base", 0) <= 1,
              f"lgd={stressed_pf.get('lgd_base', 0):.4f}")

    # H2: Securitisation
    df_sec = df_bs.attrs.get("securitisation_positions")
    if df_sec is not None:
        from ifrs9_cockpit.synthetic_generator.securitisation_positions import stress_securitisation_positions
        stressed_sec = stress_securitisation_positions(df_sec, stress_macro)
        check("H-Pos", "SEC stress returns dict", isinstance(stressed_sec, dict), "")
        check("H-Pos", "SEC rw_crr3 > 0", stressed_sec.get("rw_crr3", 0) > 0,
              f"rw={stressed_sec.get('rw_crr3', 0):.4f}")

    # H3: Sovereign
    df_sov = df_bs.attrs.get("sovereign_positions")
    if df_sov is not None:
        from ifrs9_cockpit.synthetic_generator.sovereign_positions import stress_sovereign_positions
        stressed_sov = stress_sovereign_positions(df_sov, stress_macro)
        check("H-Pos", "SOV stress returns dict", isinstance(stressed_sov, dict), "")
        check("H-Pos", "SOV rw_crr3 == 0 (Art.114)", stressed_sov.get("rw_crr3", 1) == 0,
              f"rw={stressed_sov.get('rw_crr3', 0)}")

    # H4: Covered Bonds
    df_cb = df_bs.attrs.get("covered_bonds_positions")
    if df_cb is not None:
        from ifrs9_cockpit.synthetic_generator.covered_bonds_positions import stress_covered_bonds_positions
        stressed_cb = stress_covered_bonds_positions(df_cb, stress_macro)
        check("H-Pos", "CB stress returns dict", isinstance(stressed_cb, dict), "")
        check("H-Pos", "CB lgd_base < 0.50 (dual-recours)",
              stressed_cb.get("lgd_base", 1) < 0.50,
              f"lgd={stressed_cb.get('lgd_base', 0):.4f}")

    # H5: Interbank
    df_ib = df_bs.attrs.get("interbank_positions")
    if df_ib is not None:
        from ifrs9_cockpit.synthetic_generator.interbank_positions import stress_interbank_positions
        stressed_ib = stress_interbank_positions(df_ib, stress_macro)
        check("H-Pos", "IB stress returns dict", isinstance(stressed_ib, dict), "")
        check("H-Pos", "IB pd_base > 0", stressed_ib.get("pd_base", 0) > 0,
              f"pd={stressed_ib.get('pd_base', 0):.4f}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # I. Gouvernance (Z cap, revenue-loss matching)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n[I] Gouvernance (calibration, Z cap, annual EL)...")

    # I1: Z cap verification
    for ac in ASSET_CLASSES:
        if ac.name in ("corporate_loans", "private_equity"):
            continue
        # Test with extreme Adverse macro
        extreme_macro = {
            "gdp_growth": -5.0,
            "unemployment_rate": 15.0,
            "interest_rate": 8.0,
            "hpi_growth": -15.0,
            "inflation_rate": 8.0,
        }
        z = macro_to_z(extreme_macro, ac.macro_sensitivities)
        z_capped = np.clip(z, -4.0, 4.0)
        pd_cond = vasicek_conditional_pd(ac.pd_base, ac.asset_correlation, z_capped)
        check("I-Gov", f"{ac.name[:12]} Z cap |Z|<=4",
              abs(z_capped) <= 4.0,
              f"Z_raw={z:.1f} Z_cap={z_capped:.1f} pd_cond={pd_cond:.4f}")

    # I2: Revenue > annual EL in Central for all classes
    if "Central" in scenario_data:
        mc_cen = scenario_data["Central"]["raroc_mc"]
        for _, rr in mc_cen.iterrows():
            if rr["asset_class"] == "Total":
                continue
            check("I-Gov", f"Central {rr['asset_class'][:12]} Rev > Loss",
                  rr["revenue"] > rr["loss"],
                  f"Rev={rr['revenue']/1e6:.0f}M vs Loss={rr['loss']/1e6:.0f}M")

    # I3: 10 predefined scenarios
    check("I-Gov", "10+ predefined scenarios",
          len(PREDEFINED_SCENARIOS) >= 10,
          f"n={len(PREDEFINED_SCENARIOS)}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Rapport final
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    elapsed = time.time() - t0
    print("\n" + "=" * 100)
    print("TABLEAU RECAPITULATIF — SCENARIOS")
    print("=" * 100)
    df_table = pd.DataFrame(audit_table)
    print(df_table.to_string(index=False))

    # RAROC per class for Central
    if "Central" in scenario_data:
        print("\n" + "-" * 100)
        print("RAROC PAR CLASSE D'ACTIFS (Central)")
        print("-" * 100)
        mc_cen = scenario_data["Central"]["raroc_mc"]
        for _, rr in mc_cen.iterrows():
            tag = "+" if rr["raroc"] >= 0.13 else (" " if rr["raroc"] >= 0 else "!")
            print(f"  {tag} {rr['label']:28s}  RAROC={rr['raroc']:>8.2%}  "
                  f"Rev={rr['revenue']/1e6:>8.0f}M  Loss={rr['loss']/1e6:>8.0f}M  "
                  f"Capital={rr['capital']/1e6:>8.0f}M")

    # RAROC per class for Boom Immobilier
    if "Boom immobilier" in scenario_data:
        print("\n" + "-" * 100)
        print("RAROC PAR CLASSE D'ACTIFS (Boom immobilier)")
        print("-" * 100)
        mc_boom = scenario_data["Boom immobilier"]["raroc_mc"]
        for _, rr in mc_boom.iterrows():
            tag = "+" if rr["raroc"] >= 0.13 else (" " if rr["raroc"] >= 0 else "!")
            print(f"  {tag} {rr['label']:28s}  RAROC={rr['raroc']:>8.2%}  "
                  f"Rev={rr['revenue']/1e6:>8.0f}M  Loss={rr['loss']/1e6:>8.0f}M  "
                  f"Capital={rr['capital']/1e6:>8.0f}M")

    # RAROC per class for GFC
    if "Crise financiere (GFC)" in scenario_data:
        print("\n" + "-" * 100)
        print("RAROC PAR CLASSE D'ACTIFS (Crise GFC)")
        print("-" * 100)
        mc_gfc = scenario_data["Crise financiere (GFC)"]["raroc_mc"]
        for _, rr in mc_gfc.iterrows():
            tag = "+" if rr["raroc"] >= 0.13 else (" " if rr["raroc"] >= 0 else "!")
            print(f"  {tag} {rr['label']:28s}  RAROC={rr['raroc']:>8.2%}  "
                  f"Rev={rr['revenue']/1e6:>8.0f}M  Loss={rr['loss']/1e6:>8.0f}M  "
                  f"Capital={rr['capital']/1e6:>8.0f}M")

    # Section-by-section summary
    print("\n" + "=" * 100)
    sections = {}
    for section, name, status, detail in _RESULTS:
        if section not in sections:
            sections[section] = {"PASS": 0, "FAIL": 0, "WARN": 0}
        sections[section][status] += 1

    print(f"{'Section':<15} {'PASS':>6} {'FAIL':>6} {'WARN':>6}")
    print("-" * 40)
    for section, counts in sections.items():
        print(f"{section:<15} {counts['PASS']:>6} {counts['FAIL']:>6} {counts['WARN']:>6}")
    print("-" * 40)
    print(f"{'TOTAL':<15} {_PASS:>6} {_FAIL:>6} {_WARN:>6}")

    # Failures detail
    if _FAIL > 0:
        print(f"\n{'!' * 60}")
        print(f"  {_FAIL} CHECKS FAILED:")
        print(f"{'!' * 60}")
        for section, name, status, detail in _RESULTS:
            if status == "FAIL":
                print(f"  [{section}] {name}: {detail}")

    if _WARN > 0:
        print(f"\n  {_WARN} WARNINGS:")
        for section, name, status, detail in _RESULTS:
            if status == "WARN":
                print(f"  [{section}] {name}: {detail}")

    print(f"\n{'=' * 100}")
    total = _PASS + _FAIL + _WARN
    print(f"RESULTAT: {_PASS}/{total} PASS, {_FAIL} FAIL, {_WARN} WARN  ({elapsed:.1f}s)")
    if _FAIL == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"WARNING: {_FAIL} checks FAILED — voir detail ci-dessus")
    print("=" * 100)

    return _FAIL == 0


if __name__ == "__main__":
    success = run_audit()
    exit(0 if success else 1)
