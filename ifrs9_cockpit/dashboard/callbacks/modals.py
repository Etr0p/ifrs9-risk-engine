"""Callbacks modales PE et Credit -- toggle + lazy content build."""
from __future__ import annotations

import numpy as np
import polars as pl
import json

from dash import Input, Output, State, no_update, html, dcc
import dash_bootstrap_components as dbc
from dash import dash_table

from ifrs9_cockpit.dashboard import ids
from ifrs9_cockpit.dashboard import charts
from ifrs9_cockpit.dashboard.cache import get_cached_pipeline
from ifrs9_cockpit.dashboard.components_dash import build_section_title
from ifrs9_cockpit.config import DASHBOARD_CONFIG
from ifrs9_cockpit.utils.helpers import format_euro, format_pct

from ifrs9_cockpit.dashboard.callbacks.shared import (
    TABLE_HEADER_STYLE,
    TABLE_CELL_STYLE,
    TABLE_ODD_ROW,
)


def register_modals(app):
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
            build_section_title("Analyse Risque PE"),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(
                        figure=charts.plot_pe_distress_box(result_pe),
                        config={"displayModeBar": False},
                    ),
                    md=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        figure=charts.plot_pe_rwa_by_sector(result_pe),
                        config={"displayModeBar": False},
                    ),
                    md=6,
                ),
            ]),
            build_section_title("Vintage Analysis"),
            dcc.Graph(
                figure=charts.plot_pe_vintage_analysis(result_pe),
                config={"displayModeBar": False},
            ),
            build_section_title("Detail par Secteur"),
        ]

        # PE summary table
        agg_exprs = []
        if "enterprise_id" in result_pe.columns:
            agg_exprs.append(pl.col("enterprise_id").count().alias("positions"))
        else:
            agg_exprs.append(pl.col("sector").count().alias("positions"))
        agg_exprs.append(pl.col("nav").sum().alias("nav_total"))
        if "capital_invested" in result_pe.columns:
            agg_exprs.append(pl.col("capital_invested").sum().alias("capital_investi"))
        if "moic" in result_pe.columns:
            agg_exprs.append(pl.col("moic").mean().alias("moic_mean"))
        if "irr" in result_pe.columns:
            agg_exprs.append(pl.col("irr").mean().alias("irr_mean"))
        if "nav_drawdown" in result_pe.columns:
            agg_exprs.append(pl.col("nav_drawdown").mean().alias("drawdown_mean"))
        if "expected_loss_pe" in result_pe.columns:
            agg_exprs.append(pl.col("expected_loss_pe").sum().alias("el_pe_total"))
        if "rwa_pe" in result_pe.columns:
            agg_exprs.append(pl.col("rwa_pe").sum().alias("rwa_pe_total"))

        pe_summary = result_pe.group_by("sector").agg(agg_exprs)

        children.append(
            dash_table.DataTable(
                data=pe_summary.to_pandas().round(2).to_dict("records"),
                columns=[{"name": c, "id": c} for c in pe_summary.columns],
                style_header=TABLE_HEADER_STYLE,
                style_cell=TABLE_CELL_STYLE,
                style_data_conditional=TABLE_ODD_ROW,
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

        _err_style = {"color": "#F87171", "fontSize": "0.8rem", "padding": "0.5rem"}

        # ECL by segment + coverage scatter
        chart_row = []
        try:
            chart_row.append(
                dbc.Col(
                    dcc.Graph(
                        figure=charts.plot_ecl_by_segment(result_stressed),
                        config={"displayModeBar": False},
                    ),
                    md=6,
                )
            )
        except Exception as e:
            chart_row.append(dbc.Col(html.Div(f"ECL par segment: {e}", style=_err_style), md=6))

        try:
            chart_row.append(
                dbc.Col(
                    dcc.Graph(
                        figure=charts.plot_ecl_coverage_scatter(result_stressed),
                        config={"displayModeBar": False},
                    ),
                    md=6,
                )
            )
        except Exception as e:
            chart_row.append(dbc.Col(html.Div(f"Coverage scatter: {e}", style=_err_style), md=6))

        children.append(dbc.Row(chart_row))

        # HHI Concentration
        children.append(build_section_title("Concentration du Portefeuille (HHI)"))
        sector_col = "sector" if "sector" in result_stressed.columns else "segment"
        if sector_col in result_stressed.columns and "ead" in result_stressed.columns:
            try:
                seg_ead = result_stressed.group_by(sector_col).agg(pl.col("ead").sum())["ead"].to_numpy()
                loan_col = "loan_type" if "loan_type" in result_stressed.columns else sector_col
                loan_ead = result_stressed.group_by(loan_col).agg(pl.col("ead").sum())["ead"].to_numpy()
                from ifrs9_cockpit.analytics.metrics import ModelMetrics
                hhi_seg = ModelMetrics.hhi(seg_ead)
                hhi_loan = ModelMetrics.hhi(loan_ead)
                children.append(
                    dcc.Graph(
                        figure=charts.plot_hhi_gauge(hhi_seg, hhi_loan),
                        config={"displayModeBar": False},
                    )
                )
            except Exception as e:
                children.append(html.Div(f"HHI gauge: {e}", style=_err_style))

        # Waterfall ECL (Base vs Stressed)
        children.append(build_section_title("Waterfall ECL (Base vs Stresse)"))
        try:
            waterfall_data = []
            sectors = [
                s for s in result_stressed["sector"].unique().to_list()
                if s not in ("Total", "TOTAL", "TOTAL_CREDIT")
            ]
            for sector in sectors:
                ecl_b = float(result_base.filter(pl.col("sector") == sector)["ecl_weighted"].sum())
                ecl_s = float(result_stressed.filter(pl.col("sector") == sector)["ecl_weighted"].sum())
                waterfall_data.append({
                    "sector": sector, "ecl_base": ecl_b,
                    "ecl_stressed": ecl_s, "delta": ecl_s - ecl_b,
                })

            if waterfall_data:
                ecl_calc = cached.get("ecl_calc")
                if ecl_calc and hasattr(ecl_calc, "compute_waterfall"):
                    seg_col = "sector" if "sector" in result_stressed.columns else "segment"
                    waterfall_df = ecl_calc.compute_waterfall(
                        ecl_t0=result_base["ecl_weighted"].to_numpy(),
                        ecl_t1=result_stressed["ecl_weighted"].to_numpy(),
                        stages_t0=result_base["stage"].to_numpy(),
                        stages_t1=result_stressed["stage"].to_numpy(),
                        segments=result_stressed[seg_col].to_numpy(),
                    )
                    children.append(
                        dcc.Graph(
                            figure=charts.plot_waterfall_ecl(waterfall_df),
                            config={"displayModeBar": False},
                        )
                    )
        except Exception as e:
            children.append(html.Div(f"Waterfall: {e}", style=_err_style))

        # GAR (ESG placeholder)
        if gar_result:
            children.append(build_section_title("Green Asset Ratio (ESG)"))
            try:
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

                if "details" in gar_result and isinstance(gar_result["details"], pl.DataFrame):
                    children.append(
                        dash_table.DataTable(
                            data=gar_result["details"].to_pandas().round(3).to_dict("records"),
                            columns=[{"name": c, "id": c} for c in gar_result["details"].columns],
                            style_header=TABLE_HEADER_STYLE,
                            style_cell=TABLE_CELL_STYLE,
                            style_data_conditional=TABLE_ODD_ROW,
                            page_size=10,
                        )
                    )
            except Exception as e:
                children.append(html.Div(f"GAR: {e}", style=_err_style))

        # ── Staging & Transitions ──
        children.append(build_section_title("Staging & Transitions"))
        try:
            from ifrs9_cockpit.engine.staging import StagingEngine
            staging = StagingEngine()

            stages_base = result_base["stage"].to_numpy() if "stage" in result_base.columns else np.ones(len(result_base))
            stages_stressed = result_stressed["stage"].to_numpy() if "stage" in result_stressed.columns else np.ones(len(result_stressed))

            # Stage distribution
            if hasattr(staging, "get_stage_summary") and "ead" in result_stressed.columns:
                stage_summary = staging.get_stage_summary(
                    stages_stressed, result_stressed["ead"].to_numpy(),
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
                    if hasattr(trans_matrix, "shape") and len(trans_matrix) > 0:
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
        except Exception as e:
            children.append(html.Div(f"Staging: {e}", style=_err_style))

        return children
