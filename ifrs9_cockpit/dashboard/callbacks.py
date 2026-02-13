"""Callbacks Dash -- IFRS 9 Risk Cockpit.

12 callbacks gerant la reactivite du dashboard :
    1.  apply_scenario        -- preset sliders depuis dropdown
    2.  check_incoherence     -- alertes macro incoherentes
    3.  run_pipeline          -- pipeline ECL 5 stages (THE BIG ONE)
    4.  update_ui             -- KPI, badges, narrative, banner, cards, arbitrage
    5.  toggle_pe_modal       -- ouverture/fermeture modale PE
    6.  build_pe_modal_content -- contenu lazy de la modale PE
    7.  toggle_credit_modal   -- ouverture/fermeture modale Credit
    8.  build_credit_modal_content -- contenu lazy de la modale Credit
    9.  toggle_collapses      -- sections pliables (Perf, SHAP, Export)
    10. build_perf_content    -- contenu lazy Performance
    11. build_shap_content    -- contenu lazy SHAP global + beeswarm
    12. shap_individual       -- force plot individuel
    + 4 export downloads (Excel, CRO TXT, AI TXT, LaTeX)
    + drill-down sectoriel

Toutes les DataFrames restent cote serveur via le cache pipeline.
Seuls les scalaires JSON-serialisables transitent par dcc.Store.
"""

from __future__ import annotations

import io
import json
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dash import Input, Output, State, callback_context, no_update, html, dcc
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from ifrs9_cockpit.dashboard import ids
from ifrs9_cockpit.dashboard import charts
from ifrs9_cockpit.dashboard.components_dash import (
    build_kpi_row,
    build_classification_row,
    build_ai_narrative_box,
    build_scenario_banner,
    build_pe_score_card,
    build_credit_score_card,
    build_section_title,
    build_arbitrage_insight,
    build_rst_results_panel,
)
from ifrs9_cockpit.dashboard.cache import (
    pipeline_key,
    get_cached_pipeline,
    set_cached_pipeline,
    compute_shap_values,
)
from ifrs9_cockpit.config import (
    PREDEFINED_SCENARIOS,
    MACRO_INCOHERENCE_RULES,
    SCENARIO_BASE,
    BASEL_CONFIG,
    DASHBOARD_CONFIG,
)
from ifrs9_cockpit.utils.helpers import format_euro, format_pct


# ──────────────────────────────────────────────
# TABLE STYLE — reuse across modals
# ──────────────────────────────────────────────
_TABLE_HEADER_STYLE = {
    "backgroundColor": "#1E293B",
    "color": "#F8FAFC",
    "fontWeight": "bold",
}
_TABLE_CELL_STYLE = {
    "backgroundColor": "#0C1222",
    "color": "#F8FAFC",
    "border": "1px solid #334155",
    "fontSize": "0.85rem",
    "padding": "8px",
}
_TABLE_ODD_ROW = [{"if": {"row_index": "odd"}, "backgroundColor": "#131B2E"}]


def register(app, df_credit, df_pe, df_history, pd_suite, lgd_model, ead_model):
    """Enregistre tous les callbacks sur l'app Dash.

    Args:
        app: Instance Dash.
        df_credit: DataFrame portefeuille credit.
        df_pe: DataFrame portefeuille PE.
        df_history: DataFrame historique macro.
        pd_suite: PDModelSuite pre-entrainee (3 familles).
        lgd_model: LGDModel calibre.
        ead_model: EADModel calibre.
    """

    # ══════════════════════════════════════════════
    # CALLBACK 0b : RST toggle (enable/disable target input)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.RST_TARGET, "disabled"),
        Input(ids.RST_ENABLED, "value"),
    )
    def toggle_rst_target(rst_value):
        """Active/desactive le champ cible ECL selon la checkbox RST."""
        return not (rst_value and "on" in rst_value)

    # ══════════════════════════════════════════════
    # CALLBACK 1 : Scenario Preset
    # ══════════════════════════════════════════════
    @app.callback(
        [Output(ids.SL_INTEREST_RATE, "value"),
         Output(ids.SL_UNEMPLOYMENT, "value"),
         Output(ids.SL_GDP, "value"),
         Output(ids.SL_HPI, "value"),
         Output(ids.SL_INFLATION, "value")],
        Input(ids.SCENARIO_SELECTOR, "value"),
        prevent_initial_call=True,
    )
    def apply_scenario(scenario_name):
        """Pre-remplit les 5 sliders depuis un scenario predefined."""
        if scenario_name == "Manuel" or scenario_name not in PREDEFINED_SCENARIOS:
            return no_update, no_update, no_update, no_update, no_update
        preset = PREDEFINED_SCENARIOS[scenario_name]
        return (
            preset["interest_rate_bp"],
            preset["unemployment_bipolar"],
            preset["gdp_pct"],
            preset["hpi_pct"],
            preset["inflation_pct"],
        )

    # ══════════════════════════════════════════════
    # CALLBACK 1b : Slider value display (live update)
    # ══════════════════════════════════════════════
    @app.callback(
        [Output(ids.VAL_INTEREST_RATE, "children"),
         Output(ids.VAL_UNEMPLOYMENT, "children"),
         Output(ids.VAL_GDP, "children"),
         Output(ids.VAL_HPI, "children"),
         Output(ids.VAL_INFLATION, "children"),
         Output(ids.VAL_PE_ALLOC, "children")],
        [Input(ids.SL_INTEREST_RATE, "value"),
         Input(ids.SL_UNEMPLOYMENT, "value"),
         Input(ids.SL_GDP, "value"),
         Input(ids.SL_HPI, "value"),
         Input(ids.SL_INFLATION, "value"),
         Input(ids.PE_ALLOCATION, "value")],
    )
    def update_slider_values(ir, unemp, gdp, hpi, infl, pe_alloc):
        """Met a jour l'affichage de la valeur courante des sliders."""
        return (
            f"{ir}" if ir is not None else "0",
            f"{unemp}" if unemp is not None else "0",
            f"{gdp}" if gdp is not None else "0",
            f"{hpi}" if hpi is not None else "0",
            f"{infl}" if infl is not None else "0",
            f"{pe_alloc}%" if pe_alloc is not None else "20%",
        )

    # ══════════════════════════════════════════════
    # CALLBACK 2 : Incoherence Detection
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.INCOHERENCE_ALERTS, "children"),
        [Input(ids.SL_INTEREST_RATE, "value"),
         Input(ids.SL_UNEMPLOYMENT, "value"),
         Input(ids.SL_GDP, "value"),
         Input(ids.SL_HPI, "value"),
         Input(ids.SL_INFLATION, "value")],
    )
    def check_incoherence(ir_bp, unemp, gdp, hpi, infl):
        """Evalue les regles d'incoherence macro et affiche les alertes."""
        slider_vals = {
            "interest_rate_bp": ir_bp or 0.0,
            "unemployment_bipolar": unemp or 0.0,
            "gdp_pct": gdp if gdp is not None else SCENARIO_BASE.gdp_growth,
            "hpi_pct": hpi if hpi is not None else SCENARIO_BASE.hpi_growth,
            "inflation_pct": infl if infl is not None else SCENARIO_BASE.inflation_rate,
        }
        alerts = []
        for rule in MACRO_INCOHERENCE_RULES:
            all_met = True
            for var, op, threshold in rule.conditions:
                val = slider_vals.get(var, 0.0)
                if op == "gt" and not (val > threshold):
                    all_met = False
                elif op == "lt" and not (val < threshold):
                    all_met = False
            if all_met:
                alerts.append(
                    dbc.Alert(
                        f"Incoherence : {rule.description}",
                        color="warning",
                        className="mb-2",
                    )
                )
        return alerts if alerts else []

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
         Input(ids.PE_ALLOCATION, "value"),
         Input(ids.RW_PE_SELECTOR, "value"),
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
            pd_predictions = pd_suite.predict(df_credit)
            pd_current = pd_predictions[selected_model]
            pd_origination = df_credit["pd_origination"].values

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
            for _df in [result_base, result_stressed]:
                if "sector" in _df.columns and "segment" not in _df.columns:
                    _df["segment"] = _df["sector"]

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

            # ── Stage 3 : Comparaison ──
            comparator = PortfolioComparator(result_stressed, result_pe_local)
            advanced_metrics = comparator.compute_advanced_credit_metrics()
            hhi_cross = comparator.compute_hhi_crosscell()
            gar_result = comparator.compute_green_asset_ratio()
            raroc_eva = comparator.compute_raroc_eva()
            asymmetry_matrix = comparator.build_asymmetry_matrix()

            # ── Stage 4 : Optimisation ──
            optimization = comparator.optimize_allocation()
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

            # ── KPI computation ──
            ecl_total = float(summary["ecl_total"])
            ecl_delta = (ecl_total - ecl_base_total) / max(ecl_base_total, 1)

            nav_total = float(result_pe_local["nav"].sum())
            delta_nav = float(result_pe_local["delta_nav"].sum())
            nav_ref = nav_total - delta_nav
            drawdown = max(0.0, -delta_nav) / max(nav_ref, 1.0)

            raroc_row = raroc_eva[
                (raroc_eva["sector"] == "Total") & (raroc_eva["canal"] == "Credit")
            ]
            raroc_val = float(raroc_row["raroc"].iloc[0]) if len(raroc_row) > 0 else 0.0

            raroc_pe_row = raroc_eva[
                (raroc_eva["sector"] == "Total") & (raroc_eva["canal"] == "PE")
            ]
            raroc_pe_val = float(raroc_pe_row["raroc"].iloc[0]) if len(raroc_pe_row) > 0 else 0.0

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
                pe_cats = result_pe_local["risk_category"].value_counts().to_dict()
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
                "narrative": analytics_state.narrative or "",
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
                "error": None,
            }

            full_results["store_data"] = store_data
            set_cached_pipeline(key, full_results)
            return store_data

        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    # ══════════════════════════════════════════════
    # CALLBACK 4 : UI Update
    # ══════════════════════════════════════════════
    @app.callback(
        [Output(ids.KPI_ROW, "children"),
         Output(ids.CLASSIFICATION_ROW, "children"),
         Output(ids.AI_NARRATIVE, "children"),
         Output(ids.SCENARIO_BANNER, "children"),
         Output(ids.PE_CARD, "children"),
         Output(ids.CREDIT_CARD, "children"),
         Output(ids.ARBITRAGE_SECTION, "children"),
         Output(ids.RST_RESULTS, "children")],
        Input(ids.STORE_PIPELINE, "data"),
    )
    def update_ui(store_data):
        """Met a jour les KPI, badges, narrative, banner, cards, arbitrage et RST."""
        if not store_data or store_data.get("error"):
            error_msg = (
                store_data.get("error", "Pipeline non execute")
                if store_data else "Chargement..."
            )
            error_div = dbc.Alert(f"Erreur: {error_msg}", color="danger")
            empty = html.Div()
            return error_div, empty, empty, empty, empty, empty, empty, empty

        # ── KPI Row ──
        ecl_delta = store_data["ecl_delta"]
        ecl_cls = "negative" if ecl_delta > 0.05 else ("positive" if ecl_delta <= 0 else "neutral")

        raroc_val = store_data["raroc_val"]
        raroc_cls = "positive" if raroc_val > 0.12 else ("neutral" if raroc_val > 0 else "negative")

        drawdown = store_data["drawdown"]
        pe_cls = "negative" if drawdown > 0.15 else ("neutral" if drawdown > 0.05 else "positive")

        ra_signal = store_data["ra_signal"]
        ra_rouge = store_data["ra_rouge"]
        ra_cls = "negative" if ra_signal == "rouge" else ("neutral" if ra_signal == "ambre" else "positive")

        kpi = build_kpi_row([
            {
                "label": "RISK APPETITE",
                "value": ra_signal.upper(),
                "sub_text": (
                    f"{ra_rouge} secteur{'s' if ra_rouge != 1 else ''} en rouge"
                    if ra_rouge > 0 else "Tous les secteurs OK"
                ),
                "sub_class": ra_cls,
            },
            {
                "label": "ECL CREDIT",
                "value": format_euro(store_data["ecl_total"]),
                "sub_text": f"{'+'if ecl_delta > 0 else ''}{ecl_delta:.1%} vs base",
                "sub_class": ecl_cls,
            },
            {
                "label": "RAROC CREDIT",
                "value": f"{raroc_val:.2%}",
                "sub_text": (
                    f"HHI cross: {store_data['hhi_crosscell']:,.0f} "
                    f"| Name: {store_data['hhi_name_credit']:,.0f}"
                ),
                "sub_class": raroc_cls,
            },
            {
                "label": "NAV DRAWDOWN PE",
                "value": f"{drawdown:.1%}",
                "sub_text": f"Delta NAV: {format_euro(store_data['delta_nav'])}",
                "sub_class": pe_cls,
            },
        ])

        # ── Classification row ──
        stage_counts = {int(k): v for k, v in store_data["stage_counts"].items()}
        classif = build_classification_row(
            stage_counts, store_data["pe_cats"], store_data["n_clients"],
        )

        # ── AI Narrative ──
        key = store_data.get("key")
        cached = get_cached_pipeline(key) if key else None
        if cached and "analytics_state" in cached:
            narrative_box = build_ai_narrative_box(
                cached["analytics_state"].narrative or "",
                cached["analytics_state"].recommendations or [],
            )
        else:
            narrative_box = html.Div(store_data.get("narrative", ""))

        # ── Scenario banner ──
        banner = build_scenario_banner(
            store_data["macro_params"],
            store_data.get("scenario_name", "Manuel"),
        )

        # ── Score cards ──
        pe_card = build_pe_score_card(
            store_data["raroc_pe_val"],
            store_data["drawdown"],
            store_data["delta_nav"],
            ra_signal,
            pe_cats=store_data.get("pe_cats", {}),
        )
        credit_card = build_credit_score_card(
            store_data["raroc_val"],
            store_data["ecl_ead_ratio"],
            store_data["ecl_total"],
            ra_signal,
            n_clients=store_data.get("n_clients", 0),
            hhi_cross=int(store_data.get("hhi_crosscell", 0)),
        )

        # ── Arbitrage section ──
        arbitrage_children = [build_section_title("Arbitrage Credit vs PE")]
        if cached:
            opt = cached.get("optimization", {})
            arbitrage_children.append(build_arbitrage_insight(opt))

            # Charts
            asym = cached.get("asymmetry_matrix")
            raroc_eva = cached.get("raroc_eva")
            crr3 = cached.get("crr3_sensitivity")

            row_charts = []
            if asym is not None and isinstance(asym, pd.DataFrame) and len(asym) > 0:
                row_charts.append(
                    dbc.Col(
                        dcc.Graph(
                            figure=charts.plot_asymmetry_heatmap(asym),
                            config={"displayModeBar": False},
                        ),
                        md=6,
                    )
                )
            if raroc_eva is not None and isinstance(raroc_eva, pd.DataFrame) and len(raroc_eva) > 0:
                row_charts.append(
                    dbc.Col(
                        dcc.Graph(
                            figure=charts.plot_raroc_comparison(raroc_eva),
                            config={"displayModeBar": False},
                        ),
                        md=6,
                    )
                )
            if row_charts:
                arbitrage_children.append(dbc.Row(row_charts, className="mt-3"))

            if crr3 is not None and isinstance(crr3, pd.DataFrame) and len(crr3) > 0:
                arbitrage_children.append(
                    html.Details(
                        [
                            html.Summary(
                                "Sensibilite CRR3",
                                style={
                                    "cursor": "pointer",
                                    "color": "#A1B2C8",
                                    "fontSize": "0.9rem",
                                },
                            ),
                            dcc.Graph(
                                figure=charts.plot_crr3_sensitivity(crr3),
                                config={"displayModeBar": False},
                            ),
                        ],
                        className="mt-3",
                    )
                )

        arbitrage = html.Div(arbitrage_children, className="arbitrage-section")

        # ── RST Results panel ──
        rst_panel = html.Div()
        if store_data.get("rst_active") and store_data.get("rst_result"):
            rst_panel = build_rst_results_panel(
                store_data["rst_result"],
                store_data.get("rst_distance", 0.0),
                ecl_current=store_data.get("ecl_total", 0.0),
            )

        return kpi, classif, narrative_box, banner, pe_card, credit_card, arbitrage, rst_panel

    # ══════════════════════════════════════════════
    # CALLBACK 5 : Toggle PE Modal
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.MODAL_PE, "is_open"),
        [Input(ids.PE_CARD, "n_clicks")],
        [State(ids.MODAL_PE, "is_open")],
        prevent_initial_call=True,
    )
    def toggle_pe_modal(n_clicks, is_open):
        """Ouvre/ferme la modale PE au clic sur la score card."""
        if not n_clicks:
            return no_update
        return not is_open

    # ══════════════════════════════════════════════
    # CALLBACK 6 : PE Modal Content (lazy load)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.MODAL_PE_BODY, "children"),
        Input(ids.MODAL_PE, "is_open"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def build_pe_modal_content(is_open, store_data):
        """Construit le contenu de la modale PE a l'ouverture."""
        if not is_open or not store_data or store_data.get("error"):
            return no_update

        key = store_data.get("key")
        cached = get_cached_pipeline(key) if key else None
        if not cached:
            return dbc.Alert("Donnees non disponibles", color="warning")

        result_pe = cached["result_pe"]

        children = [
            build_section_title("Portefeuille Private Equity -- IFRS 13"),
            dcc.Graph(
                figure=charts.plot_pe_risk_stacked_bar(result_pe),
                config={"displayModeBar": False},
            ),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(
                        figure=charts.plot_pe_nav_by_sector(result_pe),
                        config={"displayModeBar": False},
                    ),
                    md=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        figure=charts.plot_pe_risk_categories(result_pe),
                        config={"displayModeBar": False},
                    ),
                    md=6,
                ),
            ]),
            build_section_title("Performance -- MOIC & Drawdown"),
            dcc.Graph(
                figure=charts.plot_pe_moic_drawdown(result_pe),
                config={"displayModeBar": False},
            ),
            build_section_title("Detail par Secteur"),
        ]

        # PE summary table
        agg_dict = {}
        if "enterprise_id" in result_pe.columns:
            agg_dict["positions"] = ("enterprise_id", "count")
        else:
            agg_dict["positions"] = ("sector", "count")
        agg_dict["nav_total"] = ("nav", "sum")
        if "capital_invested" in result_pe.columns:
            agg_dict["capital_investi"] = ("capital_invested", "sum")
        if "moic" in result_pe.columns:
            agg_dict["moic_mean"] = ("moic", "mean")
        if "irr" in result_pe.columns:
            agg_dict["irr_mean"] = ("irr", "mean")
        if "nav_drawdown" in result_pe.columns:
            agg_dict["drawdown_mean"] = ("nav_drawdown", "mean")
        if "expected_loss_pe" in result_pe.columns:
            agg_dict["el_pe_total"] = ("expected_loss_pe", "sum")
        if "rwa_pe" in result_pe.columns:
            agg_dict["rwa_pe_total"] = ("rwa_pe", "sum")

        pe_summary = result_pe.groupby("sector").agg(**agg_dict).reset_index()

        children.append(
            dash_table.DataTable(
                data=pe_summary.round(2).to_dict("records"),
                columns=[{"name": c, "id": c} for c in pe_summary.columns],
                style_header=_TABLE_HEADER_STYLE,
                style_cell=_TABLE_CELL_STYLE,
                style_data_conditional=_TABLE_ODD_ROW,
                page_size=20,
            )
        )

        return children

    # ══════════════════════════════════════════════
    # CALLBACK 7 : Toggle Credit Modal
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.MODAL_CREDIT, "is_open"),
        [Input(ids.CREDIT_CARD, "n_clicks")],
        [State(ids.MODAL_CREDIT, "is_open")],
        prevent_initial_call=True,
    )
    def toggle_credit_modal(n_clicks, is_open):
        """Ouvre/ferme la modale Credit au clic sur la score card."""
        if not n_clicks:
            return no_update
        return not is_open

    # ══════════════════════════════════════════════
    # CALLBACK 8 : Credit Modal Content (lazy load)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.MODAL_CREDIT_BODY, "children"),
        Input(ids.MODAL_CREDIT, "is_open"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def build_credit_modal_content(is_open, store_data):
        """Construit le contenu de la modale Credit a l'ouverture.

        Inclut : ECL par segment, coverage scatter, HHI gauge,
        waterfall ECL, GAR, staging (distribution, transition, Sankey).
        """
        if not is_open or not store_data or store_data.get("error"):
            return no_update

        key = store_data.get("key")
        cached = get_cached_pipeline(key) if key else None
        if not cached:
            return dbc.Alert("Donnees non disponibles", color="warning")

        result_stressed = cached["result_stressed"]
        result_base = cached["result_base"]
        gar_result = cached.get("gar_result", {})

        children = [
            build_section_title("Decomposition ECL"),
        ]

        # ECL by segment + coverage scatter
        chart_row = []
        chart_row.append(
            dbc.Col(
                dcc.Graph(
                    figure=charts.plot_ecl_by_segment(result_stressed),
                    config={"displayModeBar": False},
                ),
                md=6,
            )
        )
        chart_row.append(
            dbc.Col(
                dcc.Graph(
                    figure=charts.plot_ecl_coverage_scatter(result_stressed),
                    config={"displayModeBar": False},
                ),
                md=6,
            )
        )
        children.append(dbc.Row(chart_row))

        # HHI Concentration
        children.append(build_section_title("Concentration du Portefeuille (HHI)"))
        if "segment" in result_stressed.columns and "ead" in result_stressed.columns:
            seg_ead = result_stressed.groupby("segment")["ead"].sum().values
            loan_col = "loan_type" if "loan_type" in result_stressed.columns else "segment"
            loan_ead = result_stressed.groupby(loan_col)["ead"].sum().values
            from ifrs9_cockpit.analytics.metrics import ModelMetrics
            hhi_seg = ModelMetrics.hhi(seg_ead)
            hhi_loan = ModelMetrics.hhi(loan_ead)
            children.append(
                dcc.Graph(
                    figure=charts.plot_hhi_gauge(hhi_seg, hhi_loan),
                    config={"displayModeBar": False},
                )
            )

        # Waterfall ECL (Base vs Stressed)
        children.append(build_section_title("Waterfall ECL (Base vs Stresse)"))
        waterfall_data = []
        sectors = [
            s for s in result_stressed["sector"].unique()
            if s not in ("Total", "TOTAL", "TOTAL_CREDIT")
        ]
        for sector in sectors:
            ecl_b = float(result_base[result_base["sector"] == sector]["ecl_weighted"].sum())
            ecl_s = float(result_stressed[result_stressed["sector"] == sector]["ecl_weighted"].sum())
            waterfall_data.append({
                "sector": sector, "ecl_base": ecl_b,
                "ecl_stressed": ecl_s, "delta": ecl_s - ecl_b,
            })

        if waterfall_data:
            ecl_calc = cached.get("ecl_calc")
            if ecl_calc and hasattr(ecl_calc, "compute_waterfall"):
                try:
                    waterfall_df = ecl_calc.compute_waterfall(
                        ecl_t0=result_base["ecl_weighted"].values,
                        ecl_t1=result_stressed["ecl_weighted"].values,
                        stages_t0=result_base["stage"].values,
                        stages_t1=result_stressed["stage"].values,
                        segments=result_stressed.get("sector", result_stressed.get("segment", pd.Series())).values,
                    )
                    children.append(
                        dcc.Graph(
                            figure=charts.plot_waterfall_ecl(waterfall_df),
                            config={"displayModeBar": False},
                        )
                    )
                except Exception:
                    pass

        # GAR (ESG placeholder)
        if gar_result:
            children.append(build_section_title("Green Asset Ratio (ESG)"))
            gar_metrics = []
            for label, gkey in [("GAR Credit", "gar_credit"), ("GAR PE", "gar_pe"), ("GAR Total", "gar_total")]:
                val = gar_result.get(gkey, 0.0)
                gar_metrics.append(
                    dbc.Col(
                        html.Div([
                            html.Div(label, style={"color": "#A1B2C8", "fontSize": "0.75rem", "textTransform": "uppercase"}),
                            html.Div(f"{val:.1%}", style={"color": "#F8FAFC", "fontSize": "1.4rem", "fontWeight": "700"}),
                        ], className="kpi-card"),
                        md=4,
                    )
                )
            children.append(dbc.Row(gar_metrics, className="mb-3"))

            if "details" in gar_result and isinstance(gar_result["details"], pd.DataFrame):
                children.append(
                    dash_table.DataTable(
                        data=gar_result["details"].round(3).to_dict("records"),
                        columns=[{"name": c, "id": c} for c in gar_result["details"].columns],
                        style_header=_TABLE_HEADER_STYLE,
                        style_cell=_TABLE_CELL_STYLE,
                        style_data_conditional=_TABLE_ODD_ROW,
                        page_size=10,
                    )
                )

        # ── Staging & Transitions ──
        children.append(build_section_title("Staging & Transitions"))
        try:
            from ifrs9_cockpit.engine.staging import StagingEngine
            staging = StagingEngine()

            stages_base = result_base["stage"].values if "stage" in result_base.columns else np.ones(len(result_base))
            stages_stressed = result_stressed["stage"].values if "stage" in result_stressed.columns else np.ones(len(result_stressed))

            # Stage distribution
            if hasattr(staging, "get_stage_summary") and "ead" in result_stressed.columns:
                stage_summary = staging.get_stage_summary(
                    stages_stressed, result_stressed["ead"].values,
                )
                staging_charts = []
                staging_charts.append(
                    dbc.Col(
                        dcc.Graph(
                            figure=charts.plot_stage_distribution(stage_summary),
                            config={"displayModeBar": False},
                        ),
                        md=6,
                    )
                )

                # Transition matrix
                if hasattr(staging, "compute_transition_matrix"):
                    trans_matrix = staging.compute_transition_matrix(stages_base, stages_stressed)
                    if isinstance(trans_matrix, pd.DataFrame) and len(trans_matrix) > 0:
                        staging_charts.append(
                            dbc.Col(
                                dcc.Graph(
                                    figure=charts.plot_transition_matrix(trans_matrix),
                                    config={"displayModeBar": False},
                                ),
                                md=6,
                            )
                        )

                children.append(dbc.Row(staging_charts))

            # Sankey diagram
            children.append(
                dcc.Graph(
                    figure=charts.plot_stage_sankey(stages_base, stages_stressed),
                    config={"displayModeBar": False},
                )
            )
        except Exception:
            children.append(html.Div("Staging non disponible", style={"color": "#A1B2C8"}))

        return children

    # ══════════════════════════════════════════════
    # CALLBACK 9 : Collapse Toggles
    # ══════════════════════════════════════════════
    @app.callback(
        [Output(ids.COLLAPSE_PERF, "is_open"),
         Output(ids.COLLAPSE_SHAP, "is_open"),
         Output(ids.COLLAPSE_EXPORT, "is_open")],
        [Input(ids.BTN_PERF, "n_clicks"),
         Input(ids.BTN_SHAP, "n_clicks"),
         Input(ids.BTN_EXPORT, "n_clicks")],
        [State(ids.COLLAPSE_PERF, "is_open"),
         State(ids.COLLAPSE_SHAP, "is_open"),
         State(ids.COLLAPSE_EXPORT, "is_open")],
        prevent_initial_call=True,
    )
    def toggle_collapses(n_perf, n_shap, n_export, is_perf, is_shap, is_export):
        """Bascule l'etat ouvert/ferme des sections pliables."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if btn_id == ids.BTN_PERF:
            return not is_perf, is_shap, is_export
        elif btn_id == ids.BTN_SHAP:
            return is_perf, not is_shap, is_export
        elif btn_id == ids.BTN_EXPORT:
            return is_perf, is_shap, not is_export
        return no_update, no_update, no_update

    # ══════════════════════════════════════════════
    # CALLBACK 10 : Performance Content (lazy load)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.PERF_CONTENT, "children"),
        Input(ids.COLLAPSE_PERF, "is_open"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def build_perf_content(is_open, store_data):
        """Construit le contenu Performance (ROC, radar, calibration, features)."""
        if not is_open:
            return no_update

        children = [build_section_title("Benchmark des Modeles PD")]

        try:
            from ifrs9_cockpit.analytics.metrics import ModelMetrics

            # ROC Curves
            roc_data = {}
            for name, result in pd_suite.results.items():
                fpr, tpr, _ = ModelMetrics.roc_curve_data(
                    pd_suite.y_test, result.y_pred_test,
                )
                roc_data[name] = (fpr, tpr, result.metrics_test.get("auc", 0.0))

            selected_model = store_data.get("selected_model") if store_data else None
            if selected_model is None:
                selected_model = list(pd_suite.results.keys())[0]

            chart_row = []
            chart_row.append(
                dbc.Col(
                    dcc.Graph(
                        figure=charts.plot_roc_curves(roc_data),
                        config={"displayModeBar": False},
                    ),
                    md=6,
                )
            )

            # Model comparison radar
            comparison_df = pd_suite.get_comparison_table()
            chart_row.append(
                dbc.Col(
                    dcc.Graph(
                        figure=charts.plot_model_comparison(comparison_df),
                        config={"displayModeBar": False},
                    ),
                    md=6,
                )
            )
            children.append(dbc.Row(chart_row))

            # Metriques detaillees table
            children.append(build_section_title("Metriques Detaillees"))
            children.append(
                dash_table.DataTable(
                    data=comparison_df.round(4).to_dict("records"),
                    columns=[{"name": c, "id": c} for c in comparison_df.columns],
                    style_header=_TABLE_HEADER_STYLE,
                    style_cell=_TABLE_CELL_STYLE,
                    style_data_conditional=_TABLE_ODD_ROW,
                )
            )

            # Calibration curve
            children.append(
                html.Details([
                    html.Summary(
                        "Calibration & Backtesting",
                        style={"cursor": "pointer", "color": "#A1B2C8", "fontSize": "0.9rem"},
                    ),
                    dcc.Graph(
                        figure=charts.plot_calibration_curve(
                            pd_suite.y_test,
                            {name: res.y_pred_test for name, res in pd_suite.results.items()},
                        ),
                        config={"displayModeBar": False},
                    ),
                ], className="mt-3")
            )

            # Feature importance
            children.append(
                html.Details([
                    html.Summary(
                        "Analyse des Features",
                        style={"cursor": "pointer", "color": "#A1B2C8", "fontSize": "0.9rem"},
                    ),
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(
                                figure=charts.plot_feature_importance(
                                    pd_suite.get_feature_importance_table(),
                                    selected_model,
                                ),
                                config={"displayModeBar": False},
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            dcc.Graph(
                                figure=charts.plot_iv_table(pd_suite.woe_binner.get_iv_table()),
                                config={"displayModeBar": False},
                            ),
                            md=6,
                        ),
                    ]),
                ], className="mt-3")
            )

            # Scorecard distribution
            lr_result = pd_suite.results.get("LR_WoE")
            if lr_result and getattr(lr_result, "scorecard_params", None):
                scores_test = pd_suite._pd_to_score(
                    lr_result.y_pred_test, lr_result.scorecard_params,
                )
                children.append(
                    html.Details([
                        html.Summary(
                            "Scorecard Distribution",
                            style={"cursor": "pointer", "color": "#A1B2C8", "fontSize": "0.9rem"},
                        ),
                        dcc.Graph(
                            figure=charts.plot_score_distribution(
                                scores_test, pd_suite.y_test, lr_result.scorecard_params,
                            ),
                            config={"displayModeBar": False},
                        ),
                    ], className="mt-3")
                )

        except Exception as e:
            children.append(
                dbc.Alert(f"Erreur construction Performance : {e}", color="danger")
            )

        return children

    # ══════════════════════════════════════════════
    # CALLBACK 11a : SHAP Content (lazy load)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.SHAP_CONTENT, "children"),
        Input(ids.COLLAPSE_SHAP, "is_open"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def build_shap_content(is_open, store_data):
        """Construit le contenu SHAP global (summary + beeswarm)."""
        if not is_open:
            return no_update

        children = [build_section_title("Explainabilite SHAP")]

        try:
            import shap

            selected_model = (
                store_data.get("selected_model") if store_data else None
            ) or list(pd_suite.results.keys())[0]

            model_result = pd_suite.results[selected_model]

            # Prepare features and model for SHAP
            if selected_model == "LR_WoE":
                feature_names = pd_suite._woe_features
                X_shap = pd_suite.X_test[feature_names].values
            else:
                feature_names = pd_suite._raw_features
                X_shap = pd_suite.X_test[feature_names].values

            # Extraire le modele de base du CalibratedClassifierCV
            calibrated = model_result.model
            if hasattr(calibrated, "calibrated_classifiers_"):
                cc = calibrated.calibrated_classifiers_[0]
                inner_model = getattr(cc, "estimator", getattr(cc, "base_estimator", calibrated))
            else:
                inner_model = calibrated

            n_sample = min(1000, len(X_shap))
            X_sample = X_shap[:n_sample]

            shap_vals = compute_shap_values(
                inner_model, X_sample, feature_names, selected_model,
            )

            # Summary + Beeswarm
            children.append(
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            figure=charts.plot_shap_summary(shap_vals, feature_names),
                            config={"displayModeBar": False},
                        ),
                        md=6,
                    ),
                    dbc.Col([
                        dcc.Graph(
                            figure=charts.plot_shap_beeswarm(shap_vals, X_sample, feature_names),
                            config={"displayModeBar": False},
                        ),
                        html.Div(
                            "Couleur : rouge = valeur feature elevee, bleu = valeur feature basse",
                            style={"color": "#A1B2C8", "fontSize": "0.75rem", "textAlign": "center"},
                        ),
                    ], md=6),
                ])
            )

            # Individual selector
            children.append(build_section_title("Explication Entreprise Individuelle"))
            children.append(
                html.Div([
                    html.Label(
                        "Indice de l'entreprise (dans le jeu de test) :",
                        style={"color": "#A1B2C8", "fontSize": "0.82rem"},
                    ),
                    dcc.Input(
                        id=ids.SHAP_CLIENT_SELECTOR,
                        type="number",
                        value=0,
                        min=0,
                        max=n_sample - 1,
                        step=1,
                        style={
                            "width": "120px",
                            "background": "#1E293B",
                            "color": "#F8FAFC",
                            "border": "1px solid rgba(99, 102, 241, 0.25)",
                            "borderRadius": "6px",
                            "padding": "0.3rem 0.6rem",
                            "marginLeft": "0.5rem",
                        },
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "0.5rem", "marginBottom": "1rem"})
            )
            children.append(html.Div(id=ids.SHAP_INDIVIDUAL_OUTPUT))

        except ImportError:
            children.append(
                dbc.Alert(
                    "Le package `shap` n'est pas installe. Executez `pip install shap`.",
                    color="warning",
                )
            )
        except Exception as e:
            children.append(
                dbc.Alert(f"Erreur lors du calcul SHAP : {e}", color="danger")
            )

        return children

    # ══════════════════════════════════════════════
    # CALLBACK 11b : SHAP Individual (force plot)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.SHAP_INDIVIDUAL_OUTPUT, "children"),
        Input(ids.SHAP_CLIENT_SELECTOR, "value"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def shap_individual(client_idx, store_data):
        """Affiche le force plot SHAP pour une entreprise individuelle."""
        if client_idx is None:
            return no_update

        try:
            selected_model = (
                store_data.get("selected_model") if store_data else None
            ) or list(pd_suite.results.keys())[0]

            model_result = pd_suite.results[selected_model]

            if selected_model == "LR_WoE":
                feature_names = pd_suite._woe_features
                X_shap = pd_suite.X_test[feature_names].values
                calibrated = model_result.model
                if hasattr(calibrated, "calibrated_classifiers_"):
                    cc = calibrated.calibrated_classifiers_[0]
                    inner_model = getattr(cc, "estimator", getattr(cc, "base_estimator", calibrated))
                else:
                    inner_model = calibrated
            else:
                feature_names = pd_suite._raw_features
                X_shap = pd_suite.X_test[feature_names].values
                inner_model = model_result.model

            n_sample = min(1000, len(X_shap))
            X_sample = X_shap[:n_sample]

            shap_vals = compute_shap_values(
                inner_model, X_sample, feature_names, selected_model,
            )

            idx = max(0, min(int(client_idx), n_sample - 1))
            client_shap = shap_vals[idx]
            client_features = X_sample[idx]

            children = [
                dcc.Graph(
                    figure=charts.plot_shap_force_individual(
                        client_shap, client_features, feature_names,
                    ),
                    config={"displayModeBar": False},
                ),
            ]

            # Detail table
            client_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": client_shap,
                "Feature Value": client_features,
            }).sort_values("SHAP Value", key=abs, ascending=False).head(10)

            children.append(
                dash_table.DataTable(
                    data=client_df.round(4).to_dict("records"),
                    columns=[{"name": c, "id": c} for c in client_df.columns],
                    style_header=_TABLE_HEADER_STYLE,
                    style_cell=_TABLE_CELL_STYLE,
                    style_data_conditional=_TABLE_ODD_ROW,
                )
            )

            return children

        except Exception as e:
            return dbc.Alert(f"Erreur SHAP individuel : {e}", color="danger")

    # ══════════════════════════════════════════════
    # CALLBACK 11c : Export Content (lazy load)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.EXPORT_CONTENT, "children"),
        Input(ids.COLLAPSE_EXPORT, "is_open"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def build_export_content(is_open, store_data):
        """Construit l'apercu du portefeuille quand la section Export s'ouvre."""
        if not is_open:
            return no_update

        children = []

        try:
            key = store_data.get("key") if store_data else None
            cached = get_cached_pipeline(key) if key else None

            children.append(build_section_title("Apercu du Portefeuille"))

            if cached:
                result_stressed = cached["result_stressed"]
                summary = cached.get("summary", {})

                display_cols = [
                    "enterprise_id", "sector", "credit_score",
                    "debt_ratio", "loan_type", "loan_amount", "stage",
                    "pd_12m", "lgd", "ead", "ecl_weighted",
                ]
                available_cols = [c for c in display_cols if c in result_stressed.columns]
                children.append(
                    dash_table.DataTable(
                        data=result_stressed[available_cols].head(100).round(4).to_dict("records"),
                        columns=[{"name": c, "id": c} for c in available_cols],
                        style_header=_TABLE_HEADER_STYLE,
                        style_cell=_TABLE_CELL_STYLE,
                        style_data_conditional=_TABLE_ODD_ROW,
                        page_size=20,
                    )
                )

                n_clients = summary.get("n_clients", len(result_stressed))
                ecl_total = summary.get("ecl_total", 0)
                ead_total = summary.get("ead_total", 0)
                coverage = summary.get("coverage", 0)
                children.append(
                    html.Div(
                        f"Clients : {n_clients:,}  |  "
                        f"ECL Total : {format_euro(ecl_total)}  |  "
                        f"EAD Total : {format_euro(ead_total)}  |  "
                        f"Coverage : {format_pct(coverage)}",
                        style={
                            "color": "#A1B2C8", "fontSize": "0.82rem",
                            "textAlign": "center", "padding": "0.5rem 0 1rem",
                        },
                    )
                )
            else:
                children.append(
                    html.Div(
                        "Lancez le pipeline pour voir les donnees.",
                        style={"color": "#A1B2C8", "padding": "1rem"},
                    )
                )

        except Exception as e:
            children.append(
                dbc.Alert(f"Erreur construction Export : {e}", color="danger")
            )

        return children

    # ══════════════════════════════════════════════
    # CALLBACK 12a : Export Excel (8 sheets)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.DL_EXCEL, "data"),
        Input(ids.BTN_DL_EXCEL, "n_clicks"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def export_excel(n_clicks, store_data):
        """Genere l'export Excel 8 feuilles (FR50)."""
        if not n_clicks:
            return no_update
        if not store_data or store_data.get("error"):
            return no_update

        key = store_data.get("key")
        cached = get_cached_pipeline(key) if key else None
        if not cached:
            return no_update

        result_stressed = cached["result_stressed"]
        result_base = cached["result_base"]
        result_pe = cached["result_pe"]
        macro_params = cached["macro_params"]
        alerts = cached.get("alerts", [])
        optimization = cached.get("optimization", {})
        asymmetry_matrix = cached.get("asymmetry_matrix", pd.DataFrame())
        selected_model = cached.get("selected_model", "LR_WoE")

        buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                # 1. Macro
                pd.DataFrame([macro_params]).to_excel(
                    writer, sheet_name="1_Macro", index=False,
                )

                # 2. Transitions
                try:
                    from ifrs9_cockpit.engine.staging import StagingEngine
                    staging_exp = StagingEngine()
                    trans_matrix = staging_exp.compute_transition_matrix(
                        result_base["stage"].values, result_stressed["stage"].values,
                    )
                    trans_matrix.to_excel(writer, sheet_name="2_Transitions")
                except Exception:
                    pd.DataFrame().to_excel(writer, sheet_name="2_Transitions")

                # 3. Alertes CRO
                if alerts:
                    alerts_df = pd.DataFrame([
                        {
                            "severity": getattr(a, "severity", ""),
                            "title": getattr(a, "title", ""),
                            "message": getattr(a, "message", ""),
                            "action": getattr(a, "action", ""),
                        }
                        for a in alerts
                    ])
                else:
                    alerts_df = pd.DataFrame(columns=["severity", "title", "message", "action"])
                alerts_df.to_excel(writer, sheet_name="3_Alertes", index=False)

                # 4. SHAP Top 10
                try:
                    _mr = pd_suite.results[selected_model]
                    if selected_model == "LR_WoE":
                        _fn = pd_suite._woe_features
                        _Xs = pd_suite.X_test[_fn].values[:500]
                    else:
                        _fn = pd_suite._raw_features
                        _Xs = pd_suite.X_test[_fn].values[:500]
                    # Extraire le modele de base du CalibratedClassifierCV
                    _cal = _mr.model
                    if hasattr(_cal, "calibrated_classifiers_"):
                        _cc = _cal.calibrated_classifiers_[0]
                        _im = getattr(_cc, "estimator", getattr(_cc, "base_estimator", _cal))
                    else:
                        _im = _cal

                    _sv = compute_shap_values(_im, _Xs, _fn, selected_model)
                    _mean_abs = np.abs(_sv).mean(axis=0)
                    _top_idx = np.argsort(_mean_abs)[::-1][:10]
                    shap_export = pd.DataFrame({
                        "feature": [_fn[i] for i in _top_idx],
                        "mean_abs_shap": [float(_mean_abs[i]) for i in _top_idx],
                    })
                except Exception:
                    shap_export = pd.DataFrame(columns=["feature", "mean_abs_shap"])
                shap_export.to_excel(writer, sheet_name="4_SHAP_Top10", index=False)

                # 5. Credit
                credit_cols = [
                    c for c in [
                        "enterprise_id", "sector", "loan_type", "loan_amount",
                        "stage", "pd_12m", "lgd", "ead", "ecl_weighted",
                    ] if c in result_stressed.columns
                ]
                result_stressed[credit_cols].to_excel(writer, sheet_name="5_Credit", index=False)

                # 6. PE
                pe_cols = [
                    c for c in [
                        "enterprise_id", "sector", "nav", "capital_invested",
                        "moic", "irr", "nav_drawdown", "distress_prob",
                        "risk_category", "expected_loss_pe", "rwa_pe",
                    ] if c in result_pe.columns
                ]
                result_pe[pe_cols].to_excel(writer, sheet_name="6_PE", index=False)

                # 7. Optimisation
                opt_keys = [
                    "credit_allocation", "pe_allocation", "raroc_credit",
                    "raroc_pe", "rwa_weighted", "cet1_ratio", "cet1_headroom",
                ]
                opt_data = {k: optimization.get(k, 0) for k in opt_keys}
                pd.DataFrame([opt_data]).to_excel(writer, sheet_name="7_Optimisation", index=False)

                # 8. Asymetrie
                if isinstance(asymmetry_matrix, pd.DataFrame) and len(asymmetry_matrix) > 0:
                    asymmetry_matrix.to_excel(writer, sheet_name="8_Asymetrie", index=False)

            return dcc.send_bytes(buffer.getvalue(), "ifrs9_cockpit_complet.xlsx")

        except Exception:
            return no_update

    # ══════════════════════════════════════════════
    # CALLBACK 12b : Export CRO TXT
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.DL_CRO_TXT, "data"),
        Input(ids.BTN_DL_CRO, "n_clicks"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def export_cro_txt(n_clicks, store_data):
        """Genere le rapport CRO en texte brut."""
        if not n_clicks:
            return no_update
        if not store_data or store_data.get("error"):
            return no_update

        key = store_data.get("key")
        cached = get_cached_pipeline(key) if key else None
        if not cached:
            return no_update

        try:
            cro = cached.get("cro")
            result_stressed = cached["result_stressed"]
            ecl_base_total = cached["ecl_base_total"]
            psi_value = cached.get("psi_value", 0.0)
            macro_params = cached["macro_params"]

            if cro and hasattr(cro, "generate_report"):
                report = cro.generate_report(
                    result_stressed,
                    ecl_previous=ecl_base_total,
                    psi_value=psi_value,
                    unemployment_rate=macro_params["unemployment_rate"],
                    gdp_growth=macro_params["gdp_growth"],
                )
            else:
                report = "Rapport CRO non disponible."

            return dcc.send_string(report, "rapport_cro.txt")

        except Exception:
            return no_update

    # ══════════════════════════════════════════════
    # CALLBACK 12c : Export AI Analyst TXT
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.DL_AI_TXT, "data"),
        Input(ids.BTN_DL_AI, "n_clicks"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def export_ai_txt(n_clicks, store_data):
        """Exporte la synthese AI Analyst en texte brut."""
        if not n_clicks:
            return no_update
        if not store_data or store_data.get("error"):
            return no_update

        key = store_data.get("key")
        cached = get_cached_pipeline(key) if key else None
        if not cached:
            return no_update

        analytics_state = cached.get("analytics_state")
        narrative = ""
        if analytics_state:
            narrative = analytics_state.narrative or ""
        elif store_data:
            narrative = store_data.get("narrative", "")

        return dcc.send_string(narrative, "ai_analyst_synthese.txt")

    # ══════════════════════════════════════════════
    # CALLBACK 12d : Export LaTeX Audit Trail
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.DL_LATEX, "data"),
        Input(ids.BTN_DL_LATEX, "n_clicks"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def export_latex(n_clicks, store_data):
        """Genere le journal d'audit LaTeX."""
        if not n_clicks:
            return no_update
        if not store_data or store_data.get("error"):
            return no_update

        key = store_data.get("key")
        cached = get_cached_pipeline(key) if key else None
        if not cached:
            return no_update

        try:
            from ifrs9_cockpit.export.audit_trail_latex import generate_audit_latex
            from ifrs9_cockpit.engine.staging import StagingEngine

            result_stressed = cached["result_stressed"]
            result_base = cached["result_base"]

            # Compute transition matrix for LaTeX
            staging_ltx = StagingEngine()
            trans_matrix = staging_ltx.compute_transition_matrix(
                result_base["stage"].values, result_stressed["stage"].values,
            )

            latex_content = generate_audit_latex(
                macro_params=cached["macro_params"],
                selected_model=cached["selected_model"],
                result_base=result_base,
                result_stressed=result_stressed,
                result_pe=cached["result_pe"],
                advanced_metrics=cached.get("advanced_metrics", {}),
                hhi_cross=cached.get("hhi_cross", {}),
                raroc_eva=cached.get("raroc_eva", pd.DataFrame()),
                asymmetry_matrix=cached.get("asymmetry_matrix", pd.DataFrame()),
                optimization=cached.get("optimization", {}),
                crr3_sensitivity=cached.get("crr3_sensitivity", pd.DataFrame()),
                analytics_state=cached.get("analytics_state"),
                trans_matrix=trans_matrix,
            )

            return dcc.send_string(latex_content, "ifrs9_audit_trail.tex")

        except Exception:
            return no_update

    # ══════════════════════════════════════════════
    # CALLBACK 13 : Drill-down Sectoriel
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.DRILL_SECTOR_OUTPUT, "children"),
        Input(ids.DRILL_SECTOR_SELECTOR, "value"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def drill_down_sector(selected_sector, store_data):
        """Affiche le detail sectoriel (allocation, attribution, risk appetite)."""
        if not selected_sector or not store_data or store_data.get("error"):
            return no_update

        key = store_data.get("key")
        cached = get_cached_pipeline(key) if key else None
        if not cached:
            return dbc.Alert("Donnees non disponibles", color="warning")

        analytics_state = cached.get("analytics_state")
        if not analytics_state:
            return html.Div("Analytics non disponibles", style={"color": "#A1B2C8"})

        children = []

        # ── Allocation proportionnelle ──
        alloc = analytics_state.proportional_contributions
        if alloc is not None and isinstance(alloc, pd.DataFrame) and len(alloc) > 0:
            alloc_sec = alloc[alloc["sector"] == selected_sector]
            if len(alloc_sec) > 0:
                display_cols = [
                    c for c in ["canal", "risk_amount", "proportional_share", "rwa"]
                    if c in alloc_sec.columns
                ]
                children.append(
                    html.Div([
                        html.H4(
                            f"Allocation Proportionnelle -- {selected_sector}",
                            style={"color": "#F8FAFC", "fontSize": "0.95rem"},
                        ),
                        dash_table.DataTable(
                            data=alloc_sec[display_cols].round(4).to_dict("records"),
                            columns=[{"name": c, "id": c} for c in display_cols],
                            style_header=_TABLE_HEADER_STYLE,
                            style_cell=_TABLE_CELL_STYLE,
                            style_data_conditional=_TABLE_ODD_ROW,
                        ),
                    ], className="mb-3")
                )

        # ── Factor attribution ──
        factors = analytics_state.factor_attribution
        if factors is not None and isinstance(factors, pd.DataFrame) and len(factors) > 0:
            display_cols = [
                c for c in ["variable", "canal", "delta_from_base", "attribution"]
                if c in factors.columns
            ]
            if display_cols:
                children.append(
                    html.Div([
                        html.H4(
                            "Attribution Factorielle Macro",
                            style={"color": "#F8FAFC", "fontSize": "0.95rem"},
                        ),
                        dash_table.DataTable(
                            data=factors[display_cols].round(4).to_dict("records"),
                            columns=[{"name": c, "id": c} for c in display_cols],
                            style_header=_TABLE_HEADER_STYLE,
                            style_cell=_TABLE_CELL_STYLE,
                            style_data_conditional=_TABLE_ODD_ROW,
                        ),
                    ], className="mb-3")
                )

        # ── Risk appetite for this sector ──
        ra = analytics_state.risk_appetite_matrix
        if ra is not None and isinstance(ra, pd.DataFrame) and len(ra) > 0:
            if "sector" in ra.columns:
                ra_sec = ra[ra["sector"] == selected_sector]
            else:
                ra_sec = ra
            if len(ra_sec) > 0:
                children.append(
                    html.Div([
                        html.H4(
                            f"Risk Appetite -- {selected_sector}",
                            style={"color": "#F8FAFC", "fontSize": "0.95rem"},
                        ),
                        dash_table.DataTable(
                            data=ra_sec.round(4).to_dict("records"),
                            columns=[{"name": c, "id": c} for c in ra_sec.columns],
                            style_header=_TABLE_HEADER_STYLE,
                            style_cell=_TABLE_CELL_STYLE,
                            style_data_conditional=_TABLE_ODD_ROW,
                        ),
                    ], className="mb-3")
                )

                # Risk appetite chart for this sector
                children.append(
                    dcc.Graph(
                        figure=charts.plot_risk_appetite_matrix(ra_sec),
                        config={"displayModeBar": False},
                    )
                )

        # ── Trajectoires prospectives ──
        traj = analytics_state.trajectories
        if traj is not None and isinstance(traj, pd.DataFrame) and len(traj) > 0:
            traj_display = traj.copy()
            if "sector" in traj_display.columns:
                traj_sec = traj_display[traj_display["sector"] == selected_sector]
                if len(traj_sec) > 0:
                    traj_display = traj_sec

            children.append(
                html.Div([
                    html.H4(
                        "Trajectoires Prospectives",
                        style={"color": "#F8FAFC", "fontSize": "0.95rem"},
                    ),
                    dcc.Graph(
                        figure=charts.plot_trajectories_chart(traj_display),
                        config={"displayModeBar": False},
                    ),
                ], className="mb-3")
            )

        if not children:
            children.append(
                html.Div(
                    f"Aucune donnee disponible pour {selected_sector}",
                    style={"color": "#A1B2C8"},
                )
            )

        return html.Div(children)
