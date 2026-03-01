"""Callback update_ui -- KPI, badges, narrative, banner, cards, arbitrage, RST."""
from __future__ import annotations

import json

import numpy as np
import polars as pl
from dash import Input, Output, State, no_update, html, dcc
import dash_bootstrap_components as dbc

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
    build_layer2_allocation,
    build_rst_results_panel,
)
from ifrs9_cockpit.dashboard.cache import get_cached_pipeline
from ifrs9_cockpit.config import DASHBOARD_CONFIG, BASEL_CONFIG
from ifrs9_cockpit.utils.helpers import format_euro, format_pct


def register_ui(app):
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
                cached["analytics_state"].narrative if cached["analytics_state"].narrative else "",
                cached["analytics_state"].recommendations or [],
            )
        else:
            # Fallback: use store_data narrative (may be Dict or str)
            fallback_narrative = store_data.get("narrative", "")
            narrative_box = build_ai_narrative_box(fallback_narrative, [])

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
            asym = cached.get("asymmetry_matrix")
            raroc_eva = cached.get("raroc_eva")
            crr3 = cached.get("crr3_sensitivity")
            hhi_cross_data = cached.get("hhi_cross", {})

            # Row 1: Full arbitrage table — full width
            arbitrage_children.append(
                dbc.Row(dbc.Col(build_arbitrage_insight(opt), md=12), className="mt-2")
            )

            # Row 1b: Efficient frontier — full width
            if opt.get("sigma_credit") and opt.get("sigma_pe"):
                arbitrage_children.append(
                    dbc.Row(
                        dbc.Col(
                            dcc.Graph(
                                figure=charts.plot_efficient_frontier(opt),
                                config={"displayModeBar": False},
                            ),
                            md=12,
                        ),
                        className="mt-2",
                    )
                )

            # Row 1c: Layer 2 — Optimal 2-Layer Allocation (full width)
            analytics_state = cached.get("analytics_state")
            if analytics_state:
                alloc_df = getattr(analytics_state, "proportional_contributions", None)
                factor_df = getattr(analytics_state, "factor_attribution", None)
                if alloc_df is not None and isinstance(alloc_df, pl.DataFrame) and len(alloc_df) > 0:
                    arbitrage_children.append(
                        dbc.Row(
                            dbc.Col(
                                build_layer2_allocation(
                                    alloc_df, factor_df, opt, raroc_eva,
                                ),
                                md=12,
                            ),
                            className="mt-3",
                        )
                    )

            # Row 1d: Sector allocation donuts — full width below Layer 2
            if opt.get("sector_weights_credit") or opt.get("sector_weights_pe"):
                arbitrage_children.append(
                    dbc.Row(
                        dbc.Col(
                            dcc.Graph(
                                figure=charts.plot_sector_allocation_donuts(opt),
                                config={"displayModeBar": False},
                            ),
                            md=12,
                        ),
                        className="mt-2",
                    )
                )

            # Row 2: Asymmetry heatmap — full width
            if asym is not None and isinstance(asym, pl.DataFrame) and len(asym) > 0:
                arbitrage_children.append(
                    dbc.Row(
                        dbc.Col(
                            dcc.Graph(
                                figure=charts.plot_asymmetry_heatmap(asym),
                                config={"displayModeBar": False},
                            ),
                            md=12,
                        ),
                        className="mt-3",
                    )
                )
            # Row 2b: RAROC comparison — full width
            if raroc_eva is not None and isinstance(raroc_eva, pl.DataFrame) and len(raroc_eva) > 0:
                arbitrage_children.append(
                    dbc.Row(
                        dbc.Col(
                            dcc.Graph(
                                figure=charts.plot_raroc_comparison(raroc_eva),
                                config={"displayModeBar": False},
                            ),
                            md=12,
                        ),
                        className="mt-2",
                    )
                )

            # Row 2c: Multi-class allocation bar + RAROC scatter (10 classes)
            raroc_mc = cached.get("raroc_multiclass")
            cw = opt.get("class_weights", {})
            if cw:
                arbitrage_children.append(
                    dbc.Row(
                        dbc.Col(
                            dcc.Graph(
                                figure=charts.plot_multiclass_allocation_bar(opt),
                                config={"displayModeBar": False},
                            ),
                            md=12,
                        ),
                        className="mt-3",
                    )
                )
            if raroc_mc is not None and isinstance(raroc_mc, pl.DataFrame) and len(raroc_mc) > 0:
                arbitrage_children.append(
                    dbc.Row(
                        dbc.Col(
                            dcc.Graph(
                                figure=charts.plot_multiclass_raroc_scatter(raroc_mc, cw),
                                config={"displayModeBar": False},
                            ),
                            md=12,
                        ),
                        className="mt-2",
                    )
                )
                # Capital flow Sankey (source -> classes -> value creation)
                arbitrage_children.append(
                    dbc.Row(
                        dbc.Col(
                            dcc.Graph(
                                figure=charts.plot_capital_sankey(raroc_mc, cw),
                                config={"displayModeBar": False},
                            ),
                            md=12,
                        ),
                        className="mt-2",
                    )
                )

            # Row 3: Regulatory constraints (5 cols) + Sector signals (7 cols)
            if opt or hhi_cross_data:
                arbitrage_children.append(
                    build_section_title("Contraintes & Signaux Sectoriels")
                )
                arb_norm_row = [
                    dbc.Col(
                        dcc.Graph(
                            figure=charts.plot_regulatory_constraints(opt, hhi_cross_data),
                            config={"displayModeBar": False},
                        ),
                        md=5,
                    ),
                ]

                # Sector signals (regime-dependent)
                try:
                    from ifrs9_cockpit.engine.hmm_regime import detect_regime
                    macro_params = cached.get("macro_params", store_data.get("macro_params", {}))
                    regime_result = detect_regime(macro_params)
                    from ifrs9_cockpit.engine.comparator import PortfolioComparator
                    comp = PortfolioComparator(
                        cached["result_stressed"], cached["result_pe"],
                    )
                    regime_alloc = comp.compute_regime_allocation(regime_result)
                    arb_norm_row.append(
                        dbc.Col(
                            dcc.Graph(
                                figure=charts.plot_sector_signals(regime_alloc),
                                config={"displayModeBar": False},
                            ),
                            md=7,
                        )
                    )
                except Exception:
                    pass

                arbitrage_children.append(dbc.Row(arb_norm_row, className="mt-2"))

            # CRR3 Sensitivity (collapsible detail)
            if crr3 is not None and isinstance(crr3, pl.DataFrame) and len(crr3) > 0:
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
            rst_children = [
                build_rst_results_panel(
                    store_data["rst_result"],
                    store_data.get("rst_distance", 0.0),
                    ecl_current=store_data.get("ecl_total", 0.0),
                ),
            ]
            # Pareto front chart (adversarial RST)
            pareto = store_data.get("pareto_front")
            if pareto:
                rst_result = store_data["rst_result"]
                pareto_fig = charts.plot_pareto_front(
                    pareto,
                    ecl_breach=float(rst_result.get("ecl_breach_threshold", 0)),
                    design_point_distance=float(store_data.get("rst_distance", 0)),
                )
                rst_children.append(
                    dcc.Graph(figure=pareto_fig, config={"displayModeBar": False}),
                )
            rst_panel = html.Div(rst_children)

        return kpi, classif, narrative_box, banner, pe_card, credit_card, arbitrage, rst_panel
