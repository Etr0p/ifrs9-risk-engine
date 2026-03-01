"""Callback 3 : Pipeline ECL 5 stages (THE BIG ONE).

Extrait de _callbacks_old.py lignes 205-453.
Toutes les DataFrames restent cote serveur via le cache pipeline.
Seuls les scalaires JSON-serialisables transitent par dcc.Store.
"""

from __future__ import annotations

import traceback

import numpy as np  # noqa: F401 – used inside callback body
import polars as pl

from dash import Input, Output, State, no_update  # noqa: F401

from ifrs9_cockpit.dashboard import ids
from ifrs9_cockpit.dashboard.cache import (
    pipeline_key,
    get_cached_pipeline,
    set_cached_pipeline,
)
from ifrs9_cockpit.config import (
    SCENARIO_BASE,
    BASEL_CONFIG,  # noqa: F401 – available for allocation helpers
    DASHBOARD_CONFIG,  # noqa: F401
)


def _update_bs_row(df_balance_sheet: pl.DataFrame, asset_class_name: str,
                   stressed: dict) -> pl.DataFrame:
    """Update a balance sheet row with stressed values (Polars immutable)."""
    mask = pl.col("asset_class") == asset_class_name
    updates = []
    for key in ("pd_base", "lgd_base", "rw_crr3"):
        if key in stressed:
            updates.append(
                pl.when(mask).then(pl.lit(stressed[key]))
                .otherwise(pl.col(key)).alias(key)
            )
    if updates:
        return df_balance_sheet.with_columns(updates)
    return df_balance_sheet


def register_pipeline(app, df_credit, df_pe, df_history, pd_suite, lgd_model, ead_model,
                       *, df_balance_sheet=None, dataset_bundle=None):
    """Enregistre le callback run_pipeline sur *app*.

    Parameters
    ----------
    app : dash.Dash
    df_credit, df_pe, df_history : pl.DataFrame
    pd_suite : PDModelSuite
    lgd_model : LGDModel
    ead_model : EADModel
    df_balance_sheet : pl.DataFrame | None
    """

    # ══════════════════════════════════════════════
    # CALLBACK 3 : Pipeline Computation (THE BIG ONE)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.STORE_PIPELINE, "data"),
        [Input(ids.SL_INTEREST_RATE, "value"),
         Input(ids.SL_UNEMPLOYMENT, "value"),
         Input(ids.SL_GDP, "value"),
         Input(ids.SL_HPI, "value"),
         Input(ids.SL_INFLATION, "value"),
         Input(ids.PD_MODEL_SELECTOR, "value"),
         Input(ids.PE_ALLOCATION, "data"),
         Input(ids.RW_PE_SELECTOR, "data"),
         Input(ids.RST_ENABLED, "value"),
         Input(ids.RST_TARGET, "value")],
    )
    def run_pipeline(ir_bp, unemp_bipolar, gdp_pct, hpi_pct, inflation_pct,
                     selected_model, pe_alloc, rw_pe, rst_enabled, rst_target):
        """Execute le pipeline ECL 5 stages complet.

        Convertit les sliders en valeurs macro reelles, puis enchaine :
            1. Credit ECL (base + stressed)
            2. PE IFRS 13
            3. Comparaison & metriques avancees
            4. Optimisation allocation
            5. AI Analyst (2 passes)

        Les DataFrames restent cote serveur dans le pipeline cache.
        Seuls les scalaires sont retournes via dcc.Store.
        """
        # Convert sliders to real macro values
        unemployment_rate = SCENARIO_BASE.unemployment_rate + abs(unemp_bipolar or 0)
        unemployment_crisis = (unemp_bipolar or 0) < 0
        interest_rate = SCENARIO_BASE.interest_rate + (ir_bp or 0) / 100.0
        gdp_growth = gdp_pct if gdp_pct is not None else SCENARIO_BASE.gdp_growth
        hpi_growth = hpi_pct if hpi_pct is not None else SCENARIO_BASE.hpi_growth
        inflation_rate = inflation_pct if inflation_pct is not None else SCENARIO_BASE.inflation_rate

        if selected_model is None:
            selected_model = list(pd_suite.results.keys())[0]

        macro_params = {
            "unemployment_rate": unemployment_rate,
            "gdp_growth": gdp_growth,
            "interest_rate": interest_rate,
            "hpi_growth": hpi_growth,
            "inflation_rate": inflation_rate,
        }

        # Check cache (include RST params so toggling RST invalidates cache)
        rw_pe_int = int(rw_pe) if rw_pe else 250
        rst_ecl = (rst_target or 0) * 1e9 if (rst_enabled and "on" in rst_enabled) else None
        key = pipeline_key(macro_params, selected_model, pe_alloc or 20, rw_pe_int, rst_ecl)
        cached = get_cached_pipeline(key)
        if cached is not None and "store_data" in cached:
            return cached["store_data"]

        try:
            from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
            from ifrs9_cockpit.engine.pe_calculator import PECalculator
            from ifrs9_cockpit.engine.comparator import PortfolioComparator
            from ifrs9_cockpit.ai_analyst import CROAnalyst
            from ifrs9_cockpit.analytics.virtual_cro import VirtualCRO

            # ── Stage 1 : Credit ECL ──
            from ifrs9_cockpit.utils.frame_compat import to_pandas
            pd_predictions = pd_suite.predict(to_pandas(df_credit))
            pd_current = pd_predictions[selected_model]
            pd_origination = df_credit["pd_origination"].to_numpy()

            ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
            result_base = ecl_calc.calculate(df_credit, pd_current, pd_origination)
            ecl_base_total = float(result_base["ecl_weighted"].sum())

            result_stressed = ecl_calc.calculate(
                df_credit, pd_current, pd_origination,
                unemployment_override=unemployment_rate,
                gdp_override=gdp_growth,
                interest_rate_override=interest_rate,
                hpi_override=hpi_growth,
                inflation_override=inflation_rate,
            )

            # Alias compat — segment = sector
            if "sector" in result_base.columns and "segment" not in result_base.columns:
                result_base = result_base.with_columns(pl.col("sector").alias("segment"))
            if "sector" in result_stressed.columns and "segment" not in result_stressed.columns:
                result_stressed = result_stressed.with_columns(pl.col("sector").alias("segment"))

            # ── Stage 2 : PE IFRS 13 ──
            pe_calc = PECalculator()
            result_pe_local = pe_calc.calculate(
                df_pe,
                unemployment_override=unemployment_rate,
                gdp_override=gdp_growth,
                interest_rate_override=interest_rate,
                hpi_override=hpi_growth,
                inflation_override=inflation_rate,
                unemployment_crisis=unemployment_crisis,
            )

            # ── Stage 2b : Stress PF positions (Approche B, ~5ms) ──
            _bs = df_balance_sheet.clone() if df_balance_sheet is not None else None
            if _bs is not None and dataset_bundle is not None:
                _df_projects = dataset_bundle.project_positions
                if _df_projects is not None:
                    from ifrs9_cockpit.synthetic_generator.project_finance_positions import (
                        stress_project_finance_positions,
                    )
                    stressed_pf = stress_project_finance_positions(_df_projects, macro_params)
                    _bs = _update_bs_row(_bs, "project_finance", stressed_pf)

            # ── Stage 2b-bis : Stress securitisation positions (~5ms) ──
            if _bs is not None and dataset_bundle is not None:
                _df_tranches = dataset_bundle.securitisation_positions
                if _df_tranches is not None:
                    from ifrs9_cockpit.synthetic_generator.securitisation_positions import (
                        stress_securitisation_positions,
                    )
                    stressed_sec = stress_securitisation_positions(_df_tranches, macro_params)
                    _bs = _update_bs_row(_bs, "structured_products", stressed_sec)

            # ── Stage 2b-ter : Stress covered bonds positions (~3ms) ──
            if _bs is not None and dataset_bundle is not None:
                _df_cb = dataset_bundle.covered_bonds_positions
                if _df_cb is not None:
                    from ifrs9_cockpit.synthetic_generator.covered_bonds_positions import (
                        stress_covered_bonds_positions,
                    )
                    stressed_cb = stress_covered_bonds_positions(_df_cb, macro_params)
                    _bs = _update_bs_row(_bs, "covered_bonds", stressed_cb)

            # ── Stage 2b-quater : Stress sovereign positions (~3ms) ──
            if _bs is not None and dataset_bundle is not None:
                _df_sovereigns = dataset_bundle.sovereign_positions
                if _df_sovereigns is not None:
                    from ifrs9_cockpit.synthetic_generator.sovereign_positions import (
                        stress_sovereign_positions,
                    )
                    stressed_sov = stress_sovereign_positions(_df_sovereigns, macro_params)
                    _bs = _update_bs_row(_bs, "sovereign", stressed_sov)

            # ── Stage 2b-quinquies : Stress interbank positions (~2ms) ──
            if _bs is not None and dataset_bundle is not None:
                _df_interbanks = dataset_bundle.interbank_positions
                if _df_interbanks is not None:
                    from ifrs9_cockpit.synthetic_generator.interbank_positions import (
                        stress_interbank_positions,
                    )
                    stressed_ib = stress_interbank_positions(_df_interbanks, macro_params)
                    _bs = _update_bs_row(_bs, "interbank", stressed_ib)

            # ── Stage 2b-sexies : Stress equity positions (~3ms) ──
            if _bs is not None and dataset_bundle is not None:
                _df_equities = dataset_bundle.equity_positions
                if _df_equities is not None:
                    from ifrs9_cockpit.synthetic_generator.equity_positions import (
                        stress_equity_positions,
                    )
                    stressed_eq = stress_equity_positions(_df_equities, macro_params)
                    _bs = _update_bs_row(_bs, "equities", stressed_eq)

            # ── Stage 2b-septies : Stress corporate bonds positions (~3ms) ──
            if _bs is not None and dataset_bundle is not None:
                _df_corp_bonds = dataset_bundle.corporate_bonds_positions
                if _df_corp_bonds is not None:
                    from ifrs9_cockpit.synthetic_generator.corporate_bonds_positions import (
                        stress_corporate_bonds_positions,
                    )
                    stressed_cb2 = stress_corporate_bonds_positions(_df_corp_bonds, macro_params)
                    _bs = _update_bs_row(_bs, "corporate_bonds", stressed_cb2)

            # ── Stage 2b-octies : Stress repos/SFT positions (~2ms) ──
            if _bs is not None and dataset_bundle is not None:
                _df_repos = dataset_bundle.repos_sft_positions
                if _df_repos is not None:
                    from ifrs9_cockpit.synthetic_generator.repos_sft_positions import (
                        stress_repo_positions,
                    )
                    stressed_repo = stress_repo_positions(_df_repos, macro_params)
                    _bs = _update_bs_row(_bs, "repos_sft", stressed_repo)

            # ── Stage 2b-nonies : Stress derivatives/CVA positions (~5ms) ──
            if _bs is not None and dataset_bundle is not None:
                _df_derivatives = dataset_bundle.derivatives_cva_positions
                if _df_derivatives is not None:
                    from ifrs9_cockpit.synthetic_generator.derivatives_cva_positions import (
                        stress_derivative_positions,
                    )
                    stressed_deriv = stress_derivative_positions(_df_derivatives, macro_params)
                    _bs = _update_bs_row(_bs, "derivatives_cva", stressed_deriv)

            # ── Stage 2c : Balance Sheet ECL (14 classes) ──
            df_bs_ecl = None
            if _bs is not None:
                from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl
                df_bs_ecl = compute_balance_sheet_ecl(_bs, macro_params)

            # ── Stage 3 : Comparaison ──
            comparator = PortfolioComparator(result_stressed, result_pe_local, df_bs_ecl, macro_params)
            advanced_metrics = comparator.compute_advanced_credit_metrics()
            hhi_cross = comparator.compute_hhi_crosscell()
            gar_result = comparator.compute_green_asset_ratio()
            raroc_eva = comparator.compute_raroc_eva()
            raroc_multiclass = comparator.compute_raroc_multiclass()
            asymmetry_matrix = comparator.build_asymmetry_matrix()

            # ── Stage 4 : Optimisation ──
            optimization = comparator.optimize_allocation(macro_params=macro_params)
            crr3_sensitivity = comparator.compute_crr3_sensitivity()

            # ── Stage 5 : AI Analyst ──
            # rst_ecl already computed above for cache key
            cro_analyst_ai = CROAnalyst(
                result_stressed, result_pe_local, macro_params, target_ecl=rst_ecl,
            )
            analytics_state = cro_analyst_ai.analyze()

            # Virtual CRO (alertes regles)
            cro = VirtualCRO()
            psi_value = pd_suite.results[selected_model].metrics_test.get("psi", 0.0)
            alerts = cro.analyze(
                result_stressed,
                ecl_previous=ecl_base_total,
                psi_value=psi_value,
                unemployment_rate=unemployment_rate,
                gdp_growth=gdp_growth,
            )
            summary = cro.get_executive_summary(
                result_stressed, unemployment_rate, gdp_growth,
            )

            # Virtual CRO NeSy MAS (Etape 4)
            from ifrs9_cockpit.virtual_cro import VirtualCROEngine
            vcro_engine = VirtualCROEngine()
            vcro_result = vcro_engine.run(
                result_credit=result_stressed,
                result_pe=result_pe_local,
                macro_params=macro_params,
                analytics_state=analytics_state,
                rst_distance=analytics_state.rst_distance,
                ecl_legal=float(result_stressed["ecl_weighted"].sum()),
            )

            # ── KPI computation ──
            ecl_total = float(summary["ecl_total"])
            ecl_delta = (ecl_total - ecl_base_total) / max(ecl_base_total, 1)

            nav_total = float(result_pe_local["nav"].sum())
            delta_nav = float(result_pe_local["delta_nav"].sum())
            nav_ref = nav_total - delta_nav
            drawdown = max(0.0, -delta_nav) / max(nav_ref, 1.0)

            raroc_row = raroc_eva.filter(
                (pl.col("sector") == "Total") & (pl.col("canal") == "Credit")
            )
            raroc_val = float(raroc_row["raroc"][0]) if len(raroc_row) > 0 else 0.0

            raroc_pe_row = raroc_eva.filter(
                (pl.col("sector") == "Total") & (pl.col("canal") == "PE")
            )
            raroc_pe_val = float(raroc_pe_row["raroc"][0]) if len(raroc_pe_row) > 0 else 0.0

            ra = analytics_state.risk_appetite_matrix
            ra_rouge = 0
            if ra is not None and len(ra) > 0 and "signal" in ra.columns:
                ra_rouge = int((ra["signal"] == "rouge").sum())
            ra_signal = "rouge" if ra_rouge > 3 else "ambre" if ra_rouge > 1 else "vert"

            stage_counts = {
                "1": int(summary.get("stage_1", 0)),
                "2": int(summary.get("stage_2", 0)),
                "3": int(summary.get("stage_3", 0)),
            }

            pe_cats = {}
            if "risk_category" in result_pe_local.columns:
                vc = result_pe_local["risk_category"].value_counts()
                pe_cats = dict(zip(
                    vc["risk_category"].to_list(),
                    vc["count"].to_list(),
                ))
                pe_cats = {str(k): int(v) for k, v in pe_cats.items()}

            ead_total = float(result_stressed["ead"].sum()) if "ead" in result_stressed.columns else 1.0
            ecl_ead_ratio = ecl_total / max(ead_total, 1.0)

            # ── Store FULL results server-side ──
            full_results = {
                "result_stressed": result_stressed,
                "result_base": result_base,
                "result_pe": result_pe_local,
                "asymmetry_matrix": asymmetry_matrix,
                "raroc_eva": raroc_eva,
                "optimization": optimization,
                "crr3_sensitivity": crr3_sensitivity,
                "raroc_multiclass": raroc_multiclass,
                "df_bs_ecl": df_bs_ecl,
                "analytics_state": analytics_state,
                "advanced_metrics": advanced_metrics,
                "hhi_cross": hhi_cross,
                "gar_result": gar_result,
                "alerts": alerts,
                "summary": summary,
                "macro_params": macro_params,
                "ecl_base_total": ecl_base_total,
                "selected_model": selected_model,
                "pd_predictions": pd_predictions,
                "ecl_calc": ecl_calc,
                "cro": cro,
                "psi_value": psi_value,
                "vcro_result": vcro_result,
            }

            # Scalar data for dcc.Store (JSON-serializable)
            store_data = {
                "key": key,
                "ecl_total": ecl_total,
                "ecl_delta": ecl_delta,
                "ecl_base_total": ecl_base_total,
                "raroc_val": raroc_val,
                "raroc_pe_val": raroc_pe_val,
                "drawdown": drawdown,
                "nav_total": nav_total,
                "delta_nav": delta_nav,
                "ra_signal": ra_signal,
                "ra_rouge": ra_rouge,
                "stage_counts": stage_counts,
                "pe_cats": pe_cats,
                "n_clients": int(summary.get("n_clients", 0)),
                "hhi_crosscell": float(hhi_cross.get("hhi_crosscell", 0)),
                "hhi_name_credit": float(hhi_cross.get("hhi_name_credit", 0)),
                "ecl_ead_ratio": ecl_ead_ratio,
                "narrative": analytics_state.narrative if analytics_state.narrative else "",
                "macro_params": macro_params,
                "selected_model": selected_model,
                "scenario_name": "Manuel",
                "interest_rate_bp": float(ir_bp or 0),
                "unemployment_bipolar": float(unemp_bipolar or 0),
                "gdp_pct": float(gdp_pct if gdp_pct is not None else SCENARIO_BASE.gdp_growth),
                "hpi_pct": float(hpi_pct if hpi_pct is not None else SCENARIO_BASE.hpi_growth),
                "inflation_pct": float(inflation_pct if inflation_pct is not None else SCENARIO_BASE.inflation_rate),
                "rst_active": rst_ecl is not None,
                "rst_result": analytics_state.rst_result if analytics_state.rst_result else None,
                "rst_distance": analytics_state.rst_distance,
                "pareto_front": analytics_state.pareto_front,
                "error": None,
            }

            full_results["store_data"] = store_data
            set_cached_pipeline(key, full_results)
            return store_data

        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}
