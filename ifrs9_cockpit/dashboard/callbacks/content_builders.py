"""Content-builder callbacks — lazy-loaded collapsible sections.

6 callbacks:
    - build_causal_content       (Causal ML: DML, survie, sensibilite)
    - build_vcro_content         (Virtual CRO: NeSy MAS)
    - build_multiasset_content   (Multi-Asset Balance Sheet)
    - drill_down_sector          (Drill-down sectoriel)
    - build_governance_content   (Gouvernance Quantitative)
    - build_regime_content       (Regime Intelligence: HMM + GFlowNet)
"""

from __future__ import annotations

import json
import traceback

import numpy as np
import polars as pl
from dash import Input, Output, State, callback_context, no_update, html, dcc
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from ifrs9_cockpit.dashboard import ids
from ifrs9_cockpit.dashboard import charts
from ifrs9_cockpit.dashboard.cache import get_cached_pipeline, load_governance_artifacts
from ifrs9_cockpit.dashboard.callbacks.shared import (
    TABLE_HEADER_STYLE,
    TABLE_CELL_STYLE,
    TABLE_ODD_ROW,
)
from ifrs9_cockpit.dashboard.components_dash import build_section_title
from ifrs9_cockpit.config import DASHBOARD_CONFIG, BASEL_CONFIG, SCENARIO_BASE
from ifrs9_cockpit.utils.helpers import format_euro, format_pct

# Aliases matching the original _callbacks_old.py naming convention
_TABLE_HEADER_STYLE = TABLE_HEADER_STYLE
_TABLE_CELL_STYLE = TABLE_CELL_STYLE
_TABLE_ODD_ROW = TABLE_ODD_ROW


def register_content_builders(app, df_credit, df_pe, df_history, pd_suite, *, df_balance_sheet=None):
    """Register 6 content-builder callbacks on *app*.

    Args:
        app: Dash application instance.
        df_credit: Credit portfolio DataFrame.
        df_pe: PE portfolio DataFrame.
        df_history: Macro history DataFrame.
        pd_suite: PDModelSuite (3 families).
        df_balance_sheet: Balance-sheet DataFrame for multi-asset (optional).
    """

    # ══════════════════════════════════════════════
    # CALLBACK 13 : Causal ML Content (lazy load)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.CAUSAL_CONTENT, "children"),
        Input(ids.COLLAPSE_CAUSAL, "is_open"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def build_causal_content(is_open, store_data):
        """Construit le contenu Causal ML (DML, survie, sensibilite)."""
        if not is_open:
            return no_update

        children = [build_section_title("Causal Machine Learning — Double ML")]

        try:
            from ifrs9_cockpit.causal import CausalEngine

            key = store_data.get("key") if store_data else None
            cached = get_cached_pipeline(key) if key else None
            if not cached:
                children.append(
                    dbc.Alert("Lancez le pipeline pour voir les resultats causaux.", color="info")
                )
                return children

            result_stressed = cached["result_stressed"]

            # Subsample pour performance interactive (DML sur 30K est trop lent)
            _MAX_CAUSAL = 3000
            if len(result_stressed) > _MAX_CAUSAL:
                df_causal = result_stressed.sample(
                    n=_MAX_CAUSAL, seed=42,
                )
            else:
                df_causal = result_stressed

            # Run causal engine (subsampled)
            engine = CausalEngine(df_causal, n_splits=3)
            causal_result = engine.run()

            # ── Row 1: CATE Heatmap + Bias Comparison ──
            row1 = []
            if causal_result.cate_by_sector:
                row1.append(
                    dbc.Col(
                        dcc.Graph(
                            figure=charts.plot_cate_heatmap(causal_result.cate_by_sector),
                            config={"displayModeBar": False},
                        ),
                        md=6,
                    )
                )
            if causal_result.ate_estimates and causal_result.true_ate is not None:
                row1.append(
                    dbc.Col(
                        dcc.Graph(
                            figure=charts.plot_bias_comparison(
                                causal_result.ate_estimates,
                                causal_result.true_ate,
                            ),
                            config={"displayModeBar": False},
                        ),
                        md=6,
                    )
                )
            if row1:
                children.append(dbc.Row(row1, className="mt-3"))

            # ── ATE summary table ──
            if causal_result.ate_estimates:
                ate_rows = []
                true_ate = causal_result.true_ate or 0
                for name, ate_val in causal_result.ate_estimates.items():
                    ci = causal_result.ate_ci.get(name, (None, None))
                    bias_pct = (
                        abs(ate_val - true_ate) / max(abs(true_ate), 1e-8) * 100
                        if true_ate else 0
                    )
                    ate_rows.append({
                        "Estimateur": name,
                        "ATE": round(ate_val, 6),
                        "CI_low": round(ci[0], 6) if ci[0] is not None else "N/A",
                        "CI_high": round(ci[1], 6) if ci[1] is not None else "N/A",
                        "Biais (%)": round(bias_pct, 1),
                    })
                if true_ate:
                    ate_rows.append({
                        "Estimateur": "True ATE (DGP)",
                        "ATE": round(true_ate, 6),
                        "CI_low": "-",
                        "CI_high": "-",
                        "Biais (%)": 0.0,
                    })
                children.append(build_section_title("ATE par Estimateur"))
                children.append(
                    dash_table.DataTable(
                        data=ate_rows,
                        columns=[{"name": c, "id": c} for c in ate_rows[0].keys()],
                        style_header=_TABLE_HEADER_STYLE,
                        style_cell=_TABLE_CELL_STYLE,
                        style_data_conditional=_TABLE_ODD_ROW,
                    )
                )

            # ── Row 2: Survival Shift + Sensitivity ──
            row2 = []
            surv_hr = causal_result.survival_hr or {}
            if "survival_high" in surv_hr and "survival_low" in surv_hr:
                row2.append(
                    dbc.Col(
                        dcc.Graph(
                            figure=charts.plot_survival_shift(
                                surv_hr["survival_high"], surv_hr["survival_low"],
                            ),
                            config={"displayModeBar": False},
                        ),
                        md=6,
                    )
                )
            sens = causal_result.sensitivity
            if sens:
                row2.append(
                    dbc.Col(
                        dcc.Graph(
                            figure=charts.plot_sensitivity_contour(
                                sens.get("partial_r2_treatment", 0),
                                sens.get("robustness_value", 0),
                                sens.get("benchmark_mgmt_quality_r2"),
                            ),
                            config={"displayModeBar": False},
                        ),
                        md=6,
                    )
                )
            if row2:
                children.append(dbc.Row(row2, className="mt-3"))

            # ── Survival HR summary ──
            sr = causal_result.survival_hr
            if sr:
                children.append(build_section_title("Analyse de Survie Causale"))
                hr_data = [{
                    "Metrique": "Hazard Ratio (ajuste)",
                    "Valeur": round(sr.get("hr_adjusted", 0), 4),
                }, {
                    "Metrique": "Hazard Ratio (naif)",
                    "Valeur": round(sr.get("hr_naive", 0), 4),
                }, {
                    "Metrique": "Biais confounding (%)",
                    "Valeur": round(sr.get("bias_pct", 0), 1),
                }]
                children.append(
                    dash_table.DataTable(
                        data=hr_data,
                        columns=[{"name": "Metrique", "id": "Metrique"},
                                 {"name": "Valeur", "id": "Valeur"}],
                        style_header=_TABLE_HEADER_STYLE,
                        style_cell=_TABLE_CELL_STYLE,
                        style_data_conditional=_TABLE_ODD_ROW,
                    )
                )

            # ── DAG edges ──
            if causal_result.dag_edges:
                children.append(
                    html.Details([
                        html.Summary(
                            f"DAG Causal ({len(causal_result.dag_edges)} aretes)",
                            style={"cursor": "pointer", "color": "#A1B2C8", "fontSize": "0.9rem"},
                        ),
                        dash_table.DataTable(
                            data=[{"source": e[0], "target": e[1]} for e in causal_result.dag_edges],
                            columns=[{"name": "Source", "id": "source"},
                                     {"name": "Target", "id": "target"}],
                            style_header=_TABLE_HEADER_STYLE,
                            style_cell=_TABLE_CELL_STYLE,
                            style_data_conditional=_TABLE_ODD_ROW,
                            page_size=15,
                        ),
                    ], className="mt-3")
                )

        except ImportError as e:
            children.append(
                dbc.Alert(
                    f"Packages causal non installes : {e}. "
                    "Executez `pip install econml dowhy lifelines`.",
                    color="warning",
                )
            )
        except Exception as e:
            children.append(
                dbc.Alert(f"Erreur Causal ML : {e}", color="danger")
            )

        return children

    # ══════════════════════════════════════════════
    # CALLBACK 13b : Virtual CRO Content (lazy load)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.VCRO_CONTENT, "children"),
        Input(ids.COLLAPSE_VCRO, "is_open"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def build_vcro_content(is_open, store_data):
        """Construit le contenu Virtual CRO (4 charts + resume)."""
        if not is_open:
            return no_update
        if not store_data or store_data.get("error"):
            return html.Div("Pipeline non disponible.", className="text-muted p-3")

        try:
            key = store_data.get("key")
            cached = get_cached_pipeline(key) if key else None
            if not cached:
                return html.Div("Cache expire.", className="text-muted p-3")

            vcro_result = cached.get("vcro_result")
            if vcro_result is None:
                return html.Div(
                    "Virtual CRO non disponible.",
                    className="text-muted p-3",
                )

            # --- Summary banner ---
            rec = vcro_result.recommendation.upper()
            rec_colors = {
                "MAINTENIR": _TABLE_HEADER_STYLE["backgroundColor"],
                "SURVEILLER": "#92400E",
                "REDUIRE": "#9A3412",
                "ESCALADER": "#7F1D1D",
            }
            summary_div = html.Div([
                html.Div([
                    html.Span(
                        f"Recommandation : {rec}",
                        style={
                            "fontWeight": "bold",
                            "fontSize": "1.1rem",
                            "color": "#F8FAFC",
                        },
                    ),
                    html.Span(
                        f" | Force : {vcro_result.recommendation_strength:.2f}"
                        f" | Conflit : {vcro_result.fusion_result.conflict_level:.2f}"
                        f" | Regle : {vcro_result.fusion_result.rule_used}",
                        style={"color": "#A1B2C8", "fontSize": "0.85rem"},
                    ),
                ], style={
                    "padding": "12px 16px",
                    "backgroundColor": rec_colors.get(rec, "#1E293B"),
                    "borderRadius": "8px",
                    "marginBottom": "16px",
                }),
            ])

            # --- 4 Charts ---
            fig_agents = charts.plot_vcro_agent_agreement(
                vcro_result.agent_beliefs,
            )
            fig_beliefs = charts.plot_vcro_belief_distribution(
                vcro_result.fusion_result.fused_masses,
                vcro_result.fusion_result.rule_used,
                vcro_result.fusion_result.conflict_level,
            )
            fig_qbaf = charts.plot_vcro_qbaf_strengths(
                vcro_result.qbaf_result.strengths,
                vcro_result.qbaf_result.symbolic_overrides,
                recommendation=vcro_result.qbaf_result.recommendation,
                recommendation_strength=vcro_result.qbaf_result.recommendation_strength,
                falsification_matrix=vcro_result.qbaf_result.falsification_matrix,
            )
            fig_pma = charts.plot_vcro_pma_comparison(
                vcro_result.pma_result.ecl_legal,
                vcro_result.pma_result.ecl_committee,
                vcro_result.pma_result.pma_amount,
                vcro_result.pma_result.is_valid,
            )

            # --- PMA Justification ---
            justif = vcro_result.pma_result.justification
            justif_div = html.Div([
                html.H6(
                    "Dossier de Justification PMA",
                    style={"color": "#A1B2C8", "marginTop": "16px"},
                ),
                html.Div([
                    html.P([
                        html.Strong("Resume : "),
                        justif.get("resume", ""),
                    ], style={"color": "#F8FAFC", "fontSize": "0.85rem"}),
                    html.P([
                        html.Strong("Consensus : "),
                        justif.get("consensus", ""),
                    ], style={"color": "#F8FAFC", "fontSize": "0.85rem"}),
                    html.P([
                        html.Strong("Reglementaire : "),
                        justif.get("reglementaire", ""),
                    ], style={"color": "#F8FAFC", "fontSize": "0.85rem"}),
                    html.P([
                        html.Strong("Sensibilite : "),
                        justif.get("sensibilite", ""),
                    ], style={"color": "#F8FAFC", "fontSize": "0.85rem"}),
                    html.P([
                        html.Strong("Contexte macro : "),
                        justif.get("contexte_macro", ""),
                    ], style={"color": "#F8FAFC", "fontSize": "0.85rem"}),
                ], style={
                    "backgroundColor": "#131B2E",
                    "padding": "12px 16px",
                    "borderRadius": "6px",
                    "border": "1px solid #334155",
                }),
            ])

            return html.Div([
                summary_div,
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_agents), md=6),
                    dbc.Col(dcc.Graph(figure=fig_beliefs), md=6),
                ], className="g-2"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_qbaf), md=6),
                    dbc.Col(dcc.Graph(figure=fig_pma), md=6),
                ], className="g-2 mt-2"),
                justif_div,
            ])

        except Exception:
            return html.Div(
                "Erreur lors du rendu Virtual CRO.",
                className="text-muted p-3",
            )

    # ══════════════════════════════════════════════
    # CALLBACK 13d : Multi-Asset Balance Sheet (lazy load)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.MULTIASSET_CONTENT, "children"),
        Input(ids.COLLAPSE_MULTIASSET, "is_open"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def build_multiasset_content(is_open, store_data):
        """Construit le contenu Balance Sheet Multi-Asset (treemap, contagion, climat)."""
        if not is_open:
            return no_update
        if not store_data or store_data.get("error"):
            return html.Div("Pipeline non disponible.", className="text-muted p-3")

        try:
            from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl
            from ifrs9_cockpit.engine.contagion import ContagionEngine

            macro_params = store_data.get("macro_params", {})

            # Use injected df_balance_sheet or generate fresh
            bs_df = df_balance_sheet
            if bs_df is None:
                return html.Div(
                    "Balance sheet non disponible.",
                    className="text-muted p-3",
                )

            # Compute Vasicek ECL for all 10 classes
            df_bs_ecl = compute_balance_sheet_ecl(
                bs_df, macro_params,
                carbon_price_shock=0.0,
                physical_severity=0.0,
            )

            # Treemap chart
            fig_treemap = charts.plot_balance_sheet_treemap(df_bs_ecl)

            # Contagion computation
            base_severities = {}
            for row in df_bs_ecl.iter_rows(named=True):
                base_severities[row["asset_class"]] = row["ecl_ead_ratio"]

            contagion_engine = ContagionEngine()
            amplified = contagion_engine.compute(base_severities)
            c_matrix = contagion_engine.contagion_matrix()
            amp_factors = contagion_engine.amplification_factors(base_severities, amplified)

            fig_contagion = charts.plot_contagion_network(
                base_severities, amplified, c_matrix,
            )

            # Climate heatmap
            fig_climate = charts.plot_climate_heatmap(df_bs_ecl)

            # Summary KPIs
            total_ead = df_bs_ecl["ead_total"].sum()
            total_ecl = df_bs_ecl["ecl_weighted"].sum()
            total_rwa = df_bs_ecl["rwa"].sum()
            avg_ecl_ead = total_ecl / total_ead if total_ead > 0 else 0

            max_amp_class = max(amp_factors, key=amp_factors.get)
            max_amp_val = amp_factors[max_amp_class]

            from ifrs9_cockpit.config import ASSET_CLASS_MAP
            max_amp_label = ASSET_CLASS_MAP.get(max_amp_class, None)
            max_amp_label = max_amp_label.label if max_amp_label else max_amp_class

            kpi_row = dbc.Row([
                dbc.Col(html.Div([
                    html.Span("EAD Total", style={"color": "#94A3B8", "fontSize": "0.75rem"}),
                    html.Div(f"{total_ead:,.0f}", style={"color": "#F8FAFC", "fontSize": "1.2rem", "fontWeight": "bold"}),
                ], className="text-center p-2"), md=3),
                dbc.Col(html.Div([
                    html.Span("ECL Total (10 classes)", style={"color": "#94A3B8", "fontSize": "0.75rem"}),
                    html.Div(f"{total_ecl:,.0f}", style={"color": "#F8FAFC", "fontSize": "1.2rem", "fontWeight": "bold"}),
                ], className="text-center p-2"), md=3),
                dbc.Col(html.Div([
                    html.Span("ECL/EAD Moyen", style={"color": "#94A3B8", "fontSize": "0.75rem"}),
                    html.Div(f"{avg_ecl_ead:.2%}", style={"color": "#F8FAFC", "fontSize": "1.2rem", "fontWeight": "bold"}),
                ], className="text-center p-2"), md=3),
                dbc.Col(html.Div([
                    html.Span(f"Max Contagion ({max_amp_label})", style={"color": "#94A3B8", "fontSize": "0.75rem"}),
                    html.Div(f"x{max_amp_val:.2f}", style={"color": "#EF4444" if max_amp_val > 1.5 else "#F8FAFC", "fontSize": "1.2rem", "fontWeight": "bold"}),
                ], className="text-center p-2"), md=3),
            ], className="mb-3")

            return html.Div([
                kpi_row,
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_treemap, config={"displayModeBar": False}), md=12),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_contagion, config={"displayModeBar": False}), md=6),
                    dbc.Col(dcc.Graph(figure=fig_climate, config={"displayModeBar": False}), md=6),
                ], className="g-2"),
            ])

        except Exception:
            import traceback
            return html.Div(
                f"Erreur lors du rendu Multi-Asset: {traceback.format_exc()}",
                className="text-muted p-3",
                style={"whiteSpace": "pre-wrap"},
            )

    # ══════════════════════════════════════════════
    # CALLBACK 13c : Drill-down Sectoriel
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
        if alloc is not None and isinstance(alloc, pl.DataFrame) and len(alloc) > 0:
            alloc_sec = alloc.filter(pl.col("sector") == selected_sector)
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
                            data=alloc_sec.select(display_cols).to_pandas().round(4).to_dict("records"),
                            columns=[{"name": c, "id": c} for c in display_cols],
                            style_header=_TABLE_HEADER_STYLE,
                            style_cell=_TABLE_CELL_STYLE,
                            style_data_conditional=_TABLE_ODD_ROW,
                        ),
                    ], className="mb-3")
                )

        # ── Factor attribution ──
        factors = analytics_state.factor_attribution
        if factors is not None and isinstance(factors, pl.DataFrame) and len(factors) > 0:
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
                            data=factors.select(display_cols).to_pandas().round(4).to_dict("records"),
                            columns=[{"name": c, "id": c} for c in display_cols],
                            style_header=_TABLE_HEADER_STYLE,
                            style_cell=_TABLE_CELL_STYLE,
                            style_data_conditional=_TABLE_ODD_ROW,
                        ),
                    ], className="mb-3")
                )

        # ── Risk appetite for this sector ──
        ra = analytics_state.risk_appetite_matrix
        if ra is not None and isinstance(ra, pl.DataFrame) and len(ra) > 0:
            if "sector" in ra.columns:
                ra_sec = ra.filter(pl.col("sector") == selected_sector)
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
                            data=ra_sec.to_pandas().round(4).to_dict("records"),
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
        if traj is not None and isinstance(traj, pl.DataFrame) and len(traj) > 0:
            traj_display = traj.clone()
            if "sector" in traj_display.columns:
                traj_sec = traj_display.filter(pl.col("sector") == selected_sector)
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

    # ══════════════════════════════════════════════
    # CALLBACK 14 : Gouvernance Quantitative (Phase 1)
    # Conformal Prediction, Sobol, VRP, RMT
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.GOVERNANCE_CONTENT, "children"),
        Input(ids.COLLAPSE_GOVERNANCE, "is_open"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def build_governance_content(is_open, store_data):
        """Construit le contenu Gouvernance Quantitative (lazy load).

        Phase 1: Conformal, Sobol, VRP, RMT.
        Phase 2: Signatures, TDA, Compliance Gates.
        """
        if not is_open:
            return no_update

        try:
            from ifrs9_cockpit.engine.conformal import ConformalPredictor
            from ifrs9_cockpit.engine.sobol_analysis import sobol_analysis
            from ifrs9_cockpit.engine.vrp import simulate_vrp_from_macro
            from ifrs9_cockpit.engine.rmt import denoise_covariance
            from ifrs9_cockpit.engine.signatures import compute_macro_signatures
            from ifrs9_cockpit.engine.tda import compute_macro_fragility
            from ifrs9_cockpit.engine.compliance_gates import ComplianceGates
            from ifrs9_cockpit.config import (
                MACRO_COVARIANCE, MACRO_HISTORY_BASELINE, SCENARIO_BASE,
            )

            key = store_data.get("key") if store_data else None
            cached = get_cached_pipeline(key) if key else None
            if not cached:
                return dbc.Alert("Pipeline non disponible", color="warning")

            result_stressed = cached.get("result_stressed")
            analytics_state = cached.get("analytics_state")

            children = [build_section_title("Gouvernance Quantitative")]

            # Load pre-computed governance artifacts (None if not available)
            artifacts = load_governance_artifacts()

            # ── 1. Conformal Prediction ──
            if result_stressed is not None and "pd_12m" in result_stressed.columns:
                pd_pred = result_stressed["pd_12m"].to_numpy()
                if artifacts and "conformal" in artifacts:
                    conformal = ConformalPredictor(alpha=artifacts["conformal"]["alpha"])
                    conformal.calibrate(pd_pred, pd_pred)  # dummy calibration
                    conformal._q_hat = artifacts["conformal"]["q_hat"]  # restore pre-computed q_hat
                else:
                    conformal = ConformalPredictor(alpha=0.10)
                    y_true = result_stressed["default_flag"].to_numpy() if "default_flag" in result_stressed.columns else (pd_pred > 0.5).astype(float)
                    conformal.calibrate(pd_pred, y_true)
                conf_result = conformal.predict(
                    pd_pred,
                    lgd=result_stressed["lgd"].to_numpy() if "lgd" in result_stressed.columns else None,
                    ead=result_stressed["ead"].to_numpy() if "ead" in result_stressed.columns else None,
                    discount_factor=result_stressed["discount_factor"].to_numpy() if "discount_factor" in result_stressed.columns else None,
                )
                fig_conformal = charts.plot_conformal_bands(conf_result)
                diag = conformal.get_diagnostics()
                children.append(dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_conformal, config={"displayModeBar": False}), md=8),
                    dbc.Col(html.Div([
                        html.Span("Conformal Prediction", style={"color": "#F8FAFC", "fontSize": "0.85rem", "fontWeight": "bold"}),
                        html.Div(f"Couverture: {1 - diag['alpha']:.0%}", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                        html.Div(f"q_hat: {diag['q_hat']:.4f}", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                        html.Div(f"n_cal: {diag['n_calibration']:,}", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                        html.Div(f"Residuel moyen: {diag['residual_mean']:.4f}", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                        html.Div(f"Residuel P95: {diag['residual_p95']:.4f}", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                        html.Div(
                            f"ECL bande: [{conf_result.ecl_lower.sum():,.0f} — {conf_result.ecl_upper.sum():,.0f}]",
                            style={"color": "#F8FAFC", "fontSize": "0.85rem", "marginTop": "8px"},
                        ),
                    ], className="p-3"), md=4),
                ], className="mb-3"))

            # ── 2. VRP ──
            macro_severity = 0.0
            if analytics_state and hasattr(analytics_state, "macro_severity"):
                macro_severity = analytics_state.macro_severity or 0.0
            vrp_result = simulate_vrp_from_macro(macro_severity)
            fig_vrp = charts.plot_vrp_regime(vrp_result)
            children.append(dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_vrp, config={"displayModeBar": False}), md=5),
                dbc.Col(html.Div([
                    html.Span("Variance Risk Premium", style={"color": "#F8FAFC", "fontSize": "0.85rem", "fontWeight": "bold"}),
                    html.Div(f"IV: {vrp_result.implied_vol:.1f}%  |  RV: {vrp_result.realized_vol:.1f}%", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                    html.Div(f"VRP: {vrp_result.vrp:.2f}%", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                    html.Div(f"Regime: {vrp_result.regime}", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                    html.Div(f"Tau BL mult: x{vrp_result.tau_multiplier:.2f}", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                    html.Div(f"Percentile: {vrp_result.percentile:.0%}", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                ], className="p-3"), md=3),
            ], className="mb-3"))

            # ── 3. RMT ──
            if artifacts and "rmt" in artifacts:
                rmt_result = artifacts["rmt"]
            else:
                cov_matrix = np.array(MACRO_COVARIANCE)
                rmt_result = denoise_covariance(cov_matrix, n_observations=60)
            fig_rmt = charts.plot_rmt_eigenvalues(rmt_result)

            # ── 4. Sobol (pre-computed N=512 or inline N=256) ──
            if artifacts and "sobol" in artifacts:
                sobol_result = artifacts["sobol"]
                fig_sobol = charts.plot_sobol_indices(sobol_result)
            elif result_stressed is not None:
                ecl_calc = cached.get("ecl_calculator")
                df_credit = cached.get("df_credit")
                if ecl_calc is not None and df_credit is not None:
                    pd_current = result_stressed["pd_12m"].to_numpy()
                    pd_origination = df_credit["pd_origination"].to_numpy() if "pd_origination" in df_credit.columns else pd_current * 0.8

                    def ecl_fn(params):
                        try:
                            r = ecl_calc.calculate(
                                df_credit, pd_current, pd_origination,
                                unemployment_override=params.get("unemployment_rate"),
                                gdp_override=params.get("gdp_growth"),
                                interest_rate_override=params.get("interest_rate"),
                                hpi_override=params.get("hpi_growth"),
                                inflation_override=params.get("inflation_rate"),
                            )
                            return float(r["ecl_weighted"].sum())
                        except Exception:
                            return 0.0

                    sobol_result = sobol_analysis(ecl_fn, n_samples=256, seed=42)
                    fig_sobol = charts.plot_sobol_indices(sobol_result)
                else:
                    fig_sobol = go.Figure()
                    fig_sobol.update_layout(**charts._base_layout("Sobol — donnees manquantes", 320))
                    sobol_result = None
            else:
                fig_sobol = go.Figure()
                fig_sobol.update_layout(**charts._base_layout("Sobol — pipeline non disponible", 320))
                sobol_result = None

            children.append(dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_rmt, config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_sobol, config={"displayModeBar": False}), md=6),
            ], className="g-2"))

            # ══════════════════════════════════════════════
            # Phase 2: Signatures, TDA, Compliance Gates
            # ══════════════════════════════════════════════
            children.append(html.Hr(style={"borderColor": "#334155", "margin": "24px 0"}))
            children.append(build_section_title("Analyse Avancee (Phase 2)"))

            # ── 5. Path Signatures ──
            try:
                if artifacts and "signatures" in artifacts:
                    sig_result = artifacts["signatures"]
                else:
                    sig_result = compute_macro_signatures(MACRO_HISTORY_BASELINE, order=2)
                fig_sig = charts.plot_signature_heatmap(sig_result)
                children.append(dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_sig, config={"displayModeBar": False}), md=7),
                    dbc.Col(html.Div([
                        html.Span("Path Signatures (Lyons)", style={"color": "#F8FAFC", "fontSize": "0.85rem", "fontWeight": "bold"}),
                        html.Div(f"Ordre: {sig_result.order} | Dims: {sig_result.n_dims}", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                        html.Div(f"Features: {sig_result.n_features}", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                        html.Div(f"Chemin: {sig_result.path_length} mois", style={"color": "#94A3B8", "fontSize": "0.8rem"}),
                    ], className="p-3"), md=5),
                ], className="mb-3"))
            except Exception:
                pass

            # ── 6. TDA Fragility ──
            try:
                if artifacts and "tda" in artifacts:
                    tda_result = artifacts["tda"]
                else:
                    tda_result = compute_macro_fragility(MACRO_HISTORY_BASELINE, window=24)
                fig_tda = charts.plot_tda_fragility(tda_result)
                children.append(dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_tda, config={"displayModeBar": False}), md=12),
                ], className="mb-3"))
            except Exception:
                pass

            # ── 7. Compliance Gates ──
            try:
                summary = cached.get("summary", {})
                optimization = cached.get("optimization", {})
                cg = ComplianceGates()
                gate_result = cg.check_all(
                    cet1_ratio=optimization.get("cet1_ratio"),
                    pma_ratio=summary.get("pma_ratio"),
                    ecl_total=summary.get("ecl_total"),
                    ead_total=summary.get("ead_total"),
                    stage3_pct=summary.get("stage3_pct"),
                    hhi_credit=summary.get("hhi_credit"),
                    hhi_pe=summary.get("hhi_pe"),
                    model_explainability=True,
                    human_override=True,
                )
                fig_gates = charts.plot_compliance_gates(gate_result)
                children.append(dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_gates, config={"displayModeBar": False}), md=12),
                ], className="mb-3"))
            except Exception:
                pass

            return html.Div(children)

        except Exception:
            import traceback
            return html.Div(
                f"Erreur Gouvernance Quantitative: {traceback.format_exc()}",
                className="text-muted p-3",
                style={"whiteSpace": "pre-wrap"},
            )

    # ══════════════════════════════════════════════
    # CALLBACK 15 : Regime Intelligence (HMM + GFlowNet)
    # ══════════════════════════════════════════════
    @app.callback(
        Output(ids.REGIME_CONTENT, "children"),
        Input(ids.COLLAPSE_REGIME, "is_open"),
        State(ids.STORE_PIPELINE, "data"),
        prevent_initial_call=True,
    )
    def build_regime_content(is_open, pipeline_data):
        """Construit le contenu de la section Regime Intelligence."""
        if not is_open or not pipeline_data:
            return no_update

        try:
            from ifrs9_cockpit.engine.hmm_regime import detect_regime
            from ifrs9_cockpit.engine.gflownet import DualGFlowNet
            from ifrs9_cockpit.engine.comparator import PortfolioComparator

            key = pipeline_data.get("key") if pipeline_data else None
            cached = get_cached_pipeline(key) if key else None
            if not cached:
                return html.Div("Lancez le pipeline.", className="text-muted p-3")

            result_credit = cached["result_stressed"]
            result_pe = cached["result_pe"]

            if result_credit.is_empty() or result_pe.is_empty():
                return html.Div("Donnees insuffisantes.", className="text-muted p-3")

            # Load pre-computed governance artifacts
            artifacts = load_governance_artifacts()

            # 1. Detect regime from current macro params
            macro_params = pipeline_data.get("macro_params", {})
            if not macro_params:
                macro_params = {
                    "interest_rate": SCENARIO_BASE.interest_rate,
                    "unemployment_rate": SCENARIO_BASE.unemployment_rate,
                    "gdp_growth": SCENARIO_BASE.gdp_growth,
                    "hpi_growth": SCENARIO_BASE.hpi_growth,
                    "inflation_rate": SCENARIO_BASE.inflation_rate,
                }

            if artifacts and "hmm" in artifacts:
                # Use pre-trained HMM (skip fit, just predict)
                hmm = artifacts["hmm"]
                regime_result = hmm.predict(macro_params)
            else:
                regime_result = detect_regime(macro_params)

            # 2. ECL surrogate for GFlowNet (fast linear proxy)
            ecl_total = pipeline_data.get("ecl_total", 0)
            if ecl_total == 0:
                ecl_total = result_credit["ecl_weighted"].sum() if "ecl_weighted" in result_credit.columns else 1e9

            def ecl_surrogate(params):
                u = params.get("unemployment_rate", 7.5)
                g = params.get("gdp_growth", 1.2)
                r = params.get("interest_rate", 3.5)
                h = params.get("hpi_growth", 2.0)
                base = ecl_total
                return base * (1 + 0.12 * (u - 7.5) - 0.06 * (g - 1.2) + 0.04 * (r - 3.5) - 0.03 * (h - 2.0))

            # 3. Run GFlowNet (pre-trained or lightweight inline)
            if artifacts and "gflownet" in artifacts:
                dual = artifacts["gflownet"]
                gfn_result = dual.run(
                    regime=regime_result.regime,
                    ecl_surrogate=ecl_surrogate,
                    breach_threshold=ecl_total * 1.5,
                    n_samples=30,
                    n_train_steps=10,  # fine-tune only
                )
            else:
                dual = DualGFlowNet(seed=42)
                gfn_result = dual.run(
                    regime=regime_result.regime,
                    ecl_surrogate=ecl_surrogate,
                    breach_threshold=ecl_total * 1.5,
                    n_samples=30,
                    n_train_steps=50,
                )

            # 4. Regime allocation
            comparator = PortfolioComparator(result_credit, result_pe)
            regime_alloc = comparator.compute_regime_allocation(
                regime_result, gflownet_result=gfn_result,
            )

            # 5. Build charts
            children = []

            # KPI row
            regime_label = regime_result.regime.capitalize()
            pe_pct = regime_alloc["pe_allocation"] * 100
            tau = regime_alloc["tau_multiplier"]
            n_relief = regime_alloc["n_capital_relief"]

            children.append(dbc.Row([
                dbc.Col(html.Div([
                    html.H6(regime_label, className="text-center mb-0",
                             style={"fontSize": "1.3rem"}),
                    html.P("Regime HMM", className="text-center text-muted",
                           style={"fontSize": "0.7rem"}),
                ], className="p-2"), md=3),
                dbc.Col(html.Div([
                    html.H6(f"{pe_pct:.0f}%", className="text-center mb-0",
                             style={"fontSize": "1.3rem"}),
                    html.P("PE Allocation", className="text-center text-muted",
                           style={"fontSize": "0.7rem"}),
                ], className="p-2"), md=3),
                dbc.Col(html.Div([
                    html.H6(f"{tau:.1f}x", className="text-center mb-0",
                             style={"fontSize": "1.3rem"}),
                    html.P("Tau BL", className="text-center text-muted",
                           style={"fontSize": "0.7rem"}),
                ], className="p-2"), md=3),
                dbc.Col(html.Div([
                    html.H6(f"{n_relief}", className="text-center mb-0",
                             style={"fontSize": "1.3rem"}),
                    html.P("Capital Relief", className="text-center text-muted",
                           style={"fontSize": "0.7rem"}),
                ], className="p-2"), md=3),
            ], className="mb-3"))

            # Row 1: Regime gauge + GFlowNet scatter
            fig_gauge = charts.plot_regime_gauge(regime_result)
            fig_scatter = charts.plot_gflownet_scatter(gfn_result)
            children.append(dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_gauge, config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_scatter, config={"displayModeBar": False}), md=6),
            ], className="mb-3"))

            # Row 2: Allocation chart + Scenario weights pie
            fig_alloc = charts.plot_regime_allocation(regime_alloc)
            fig_pie = charts.plot_scenario_weights_pie(regime_alloc)
            children.append(dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_alloc, config={"displayModeBar": False}), md=7),
                dbc.Col(dcc.Graph(figure=fig_pie, config={"displayModeBar": False}), md=5),
            ], className="mb-3"))

            return html.Div(children)

        except Exception:
            import traceback
            return html.Div(
                f"Erreur Regime Intelligence: {traceback.format_exc()}",
                className="text-muted p-3",
                style={"whiteSpace": "pre-wrap"},
            )
