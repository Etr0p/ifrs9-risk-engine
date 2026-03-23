"""Point d'entree pour python -m ifrs9_cockpit.engine.comparator."""

from __future__ import annotations

from ifrs9_cockpit.config import (
    BASEL_CONFIG,
    REQUIRED_CREDIT_RESULT_COLS,
)
from ifrs9_cockpit.engine.comparator import PortfolioComparator, compute_crr3_rw


if __name__ == "__main__":
    import polars as pl
    from ifrs9_cockpit.data.generator import generate_dataset
    from ifrs9_cockpit.models.pd_model import PDModelSuite
    from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
    from ifrs9_cockpit.engine.pe_calculator import PECalculator
    from ifrs9_cockpit.models.lgd_model import LGDModel
    from ifrs9_cockpit.models.ead_model import EADModel
    from ifrs9_cockpit.utils.helpers import format_pct, format_euro

    print("=" * 70)
    print("IFRS 9 COCKPIT — Comparateur : Optimisation & CRR3")
    print("=" * 70)

    # 1. Pipeline credit
    print("\n[1/5] Pipeline credit...")
    df_credit, df_pe, df_history, _ = generate_dataset()
    print(f"       {len(df_credit):,} credits, {len(df_pe):,} PE")

    pd_suite = PDModelSuite()
    pd_suite.fit(df_credit)
    pd_current = pd_suite.predict_active(df_credit)

    lgd_model = LGDModel()
    ead_model = EADModel()
    ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
    pd_origination = pd_current * 0.8  # Proxy : PD origination = 80% de PD courante
    result_credit = ecl_calc.calculate(df_credit, pd_current, pd_origination)

    # 2. Pipeline PE
    print("[2/5] Pipeline PE...")
    pe_calc = PECalculator()
    result_pe = pe_calc.calculate(df_pe)

    # 3. Metriques avancees
    print("\n[3/5] Metriques credit avancees (FR42)...")
    comparator = PortfolioComparator(result_credit, result_pe)
    adv_metrics = comparator.compute_advanced_credit_metrics()
    print(str(adv_metrics))

    # 4. HHI cross-cell
    print("\n[4/5] HHI cross-cell (FR41)...")
    hhi_result = comparator.compute_hhi_crosscell()
    print(f"  HHI Credit       : {hhi_result['hhi_credit']:.2f}")
    print(f"  HHI PE           : {hhi_result['hhi_pe']:.2f}")
    print(f"  HHI Cross-cell   : {hhi_result['hhi_crosscell']:.2f}")
    print(f"  HHI Name Credit  : {hhi_result['hhi_name_credit']:.2f}")
    print(f"  HHI Name PE      : {hhi_result['hhi_name_pe']:.2f}")
    print("\n  --- Parts par cellule ---")
    print(str(hhi_result["shares"]))

    # Green Asset Ratio (ESG placeholder)
    print("\n  --- Green Asset Ratio (ESG) ---")
    gar = comparator.compute_green_asset_ratio()
    print(f"  GAR Credit : {gar['gar_credit']:.2%}")
    print(f"  GAR PE     : {gar['gar_pe']:.2%}")
    print(f"  GAR Total  : {gar['gar_total']:.2%}")

    # 5. RAROC / EVA
    print("\n[5/5] RAROC / EVA par cellule (FR44)...")
    raroc_eva = comparator.compute_raroc_eva()
    print(str(raroc_eva))

    # 6. Matrice d'asymetrie (FR20)
    print("\n[6/7] Matrice d'asymetrie credit vs PE (FR20)...")
    asymmetry = comparator.build_asymmetry_matrix()
    cols_display = ["sector", "ecl_credit", "el_pe", "loss_ratio",
                    "raroc_credit", "raroc_pe", "raroc_delta",
                    "resilience_credit", "resilience_pe"]
    print(str(asymmetry.select(cols_display)))

    # 7. Scores de resilience (FR19)
    print("\n[7/7] Scores de resilience (FR19)...")
    resil_c = comparator._resilience_score_credit()
    resil_p = comparator._resilience_score_pe()
    print("  Credit :")
    for r in resil_c.iter_rows(named=True):
        print(f"    {r['sector']:12s} : {r['resilience_credit']:.4f}")
    print("  PE :")
    for r in resil_p.iter_rows(named=True):
        print(f"    {r['sector']:12s} : {r['resilience_pe']:.4f}")

    # 8. Optimisation BL-CVaR asymetrique (FR22)
    print("\n[8/10] Optimisation BL-CVaR asymetrique (FR22)...")
    optim = comparator.optimize_allocation()
    print(f"  Stress intensity = {optim['stress_intensity']:.2f}")
    print(f"  Illiquidity premium = {optim['illiquidity_premium']:.4f}")
    print(f"  Vol multiplier = {optim['vol_multiplier']:.2f}x")
    print(f"  Kappa PE eff = {optim['kappa_pe_eff']:.2f}")
    print(f"  BL Confidence = credit:{optim['bl_confidence'][0]:.0%} / PE:{optim['bl_confidence'][1]:.0%}")
    print(f"  Phase 1 (free) PE = {optim['pe_free']:.0%}")
    print(f"  Phase 2 band = [{optim['pe_band'][0]:.0%}, {optim['pe_band'][1]:.0%}]")
    print(f"  Final : Credit {optim['credit_allocation']:.0%} | PE {optim['pe_allocation']:.0%}")
    print(f"  RAROC Credit = {optim['raroc_credit']:.4f} | RAROC PE = {optim['raroc_pe']:.4f}")
    print(f"  Poids secteurs credit : {optim['sector_weights_credit']}")
    print(f"  Poids secteurs PE     : {optim['sector_weights_pe']}")
    print(f"  CET1 ratio post-optim = {optim['cet1_ratio']:.4f} "
          f"(headroom = {optim['cet1_headroom']:+.4f})")

    # 9. Sensibilite CRR3 (FR23)
    print("\n[9/10] Sensibilite CRR3 (FR23)...")
    crr3 = comparator.compute_crr3_sensitivity()
    print(str(crr3))

    # 10. Seuils de basculement (FR24)
    print("\n[10/10] Seuils de basculement (FR24)...")
    tipping = comparator.find_tipping_points()
    print(str(tipping))

    # ── Validations ──
    print("\n--- Validations ---")
    all_ok = True

    # V1: rwa_credit present
    ok = "rwa_credit" in result_credit.columns
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] rwa_credit present dans result_credit")
    all_ok &= ok

    # V2: REQUIRED_CREDIT_RESULT_COLS
    missing = REQUIRED_CREDIT_RESULT_COLS - set(result_credit.columns)
    ok = len(missing) == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] REQUIRED_CREDIT_RESULT_COLS (manquantes: {missing if missing else 'aucune'})")
    all_ok &= ok

    # V3: HHI dans [0, 10000]
    ok = 0 <= hhi_result["hhi_crosscell"] <= 10_000
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] HHI cross-cell dans [0, 10000] ({hhi_result['hhi_crosscell']:.2f})")
    all_ok &= ok

    # V4: NPL ratio dans [0, 1]
    npl = adv_metrics.filter(pl.col("sector") == "Total")["npl_ratio"][0]
    ok = 0 <= npl <= 1
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] NPL ratio dans [0, 1] ({npl:.4f})")
    all_ok &= ok

    # V5: RAROC calcule par cellule (10 cellules + 2 totaux)
    ok = len(raroc_eva) == 12
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] RAROC/EVA : 12 lignes (10 cellules + 2 totaux) ({len(raroc_eva)})")
    all_ok &= ok

    # V6: Cost of risk > 0
    cor = adv_metrics.filter(pl.col("sector") == "Total")["cost_of_risk_bps"][0]
    ok = cor > 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Cost of risk > 0 ({cor:.1f} bps)")
    all_ok &= ok

    # V7: Matrice d'asymetrie 5 secteurs
    ok = len(asymmetry) == 5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Matrice asymetrie : 5 secteurs ({len(asymmetry)})")
    all_ok &= ok

    # V8: Resilience scores dans [0, 1]
    resil_vals = resil_c["resilience_credit"].to_list() + resil_p["resilience_pe"].to_list()
    ok = all(0 <= v <= 1 for v in resil_vals)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Resilience scores dans [0, 1] "
          f"(min={min(resil_vals):.4f}, max={max(resil_vals):.4f})")
    all_ok &= ok

    # V9: loss_ratio >= 0
    ok = (asymmetry["loss_ratio"].to_numpy() >= 0).all()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] loss_ratio >= 0 (min={asymmetry['loss_ratio'].min():.4f})")
    all_ok &= ok

    # V10: PE allocation <= max
    ok = optim["pe_allocation"] <= BASEL_CONFIG.pe_max_allocation
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] PE allocation ({optim['pe_allocation']:.0%}) "
          f"<= {BASEL_CONFIG.pe_max_allocation:.0%}")
    all_ok &= ok

    # V11: Somme poids credit = 1
    sum_wc = sum(optim["sector_weights_credit"].values())
    ok = abs(sum_wc - 1.0) < 0.01
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Somme poids credit = {sum_wc:.4f}")
    all_ok &= ok

    # V12: Somme poids PE = 1
    sum_wp = sum(optim["sector_weights_pe"].values())
    ok = abs(sum_wp - 1.0) < 0.01
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Somme poids PE = {sum_wp:.4f}")
    all_ok &= ok

    # V13: CRR3 sensitivity 3 RW fixes + 1 composite CRR3 = 4 lignes
    ok = len(crr3) == 4
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] CRR3 sensitivity : 3 RW fixes + 1 composite ({len(crr3)})")
    all_ok &= ok

    # V14: Tipping points 5 secteurs
    ok = len(tipping) == 5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Tipping points : 5 secteurs ({len(tipping)})")
    all_ok &= ok

    # INFO: HHI credit
    hhi_c = hhi_result["hhi_credit"]
    breach = " (BREACH)" if hhi_c > BASEL_CONFIG.hhi_max else ""
    print(f"  [INFO] HHI credit = {hhi_c:.2f} (seuil optimiseur = {BASEL_CONFIG.hhi_max}){breach}")

    print(f"\n{'=' * 70}")
    if all_ok:
        print("Comparateur (Optimisation & CRR3) valide.")
    else:
        print("ATTENTION : certaines validations ont echoue.")
    print(f"{'=' * 70}")
