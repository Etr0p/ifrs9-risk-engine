"""Callbacks collapsibles -- toggle sections + lazy Performance / SHAP / Export."""
from __future__ import annotations

import numpy as np
import polars as pl
from dash import Input, Output, State, callback_context, no_update, html, dcc
from dash import dash_table
import dash_bootstrap_components as dbc

from ifrs9_cockpit.dashboard import ids
from ifrs9_cockpit.dashboard import charts
from ifrs9_cockpit.dashboard.components_dash import build_section_title
from ifrs9_cockpit.dashboard.cache import get_cached_pipeline, compute_shap_values
from ifrs9_cockpit.dashboard.callbacks.shared import (
    TABLE_HEADER_STYLE,
    TABLE_CELL_STYLE,
    TABLE_ODD_ROW,
)
from ifrs9_cockpit.utils.helpers import format_euro, format_pct


def register_collapsibles(app, pd_suite):
    # ══════════════════════════════════════════════
    # CALLBACK 9 : Toggle Collapses
    # ══════════════════════════════════════════════
    @app.callback(
        [Output(ids.COLLAPSE_PERF, "is_open"),
         Output(ids.COLLAPSE_SHAP, "is_open"),
         Output(ids.COLLAPSE_CAUSAL, "is_open"),
         Output(ids.COLLAPSE_VCRO, "is_open"),
         Output(ids.COLLAPSE_MULTIASSET, "is_open"),
         Output(ids.COLLAPSE_GOVERNANCE, "is_open"),
         Output(ids.COLLAPSE_REGIME, "is_open"),
         Output(ids.COLLAPSE_EXPORT, "is_open")],
        [Input(ids.BTN_PERF, "n_clicks"),
         Input(ids.BTN_SHAP, "n_clicks"),
         Input(ids.BTN_CAUSAL, "n_clicks"),
         Input(ids.BTN_VCRO, "n_clicks"),
         Input(ids.BTN_MULTIASSET, "n_clicks"),
         Input(ids.BTN_GOVERNANCE, "n_clicks"),
         Input(ids.BTN_REGIME, "n_clicks"),
         Input(ids.BTN_EXPORT, "n_clicks")],
        [State(ids.COLLAPSE_PERF, "is_open"),
         State(ids.COLLAPSE_SHAP, "is_open"),
         State(ids.COLLAPSE_CAUSAL, "is_open"),
         State(ids.COLLAPSE_VCRO, "is_open"),
         State(ids.COLLAPSE_MULTIASSET, "is_open"),
         State(ids.COLLAPSE_GOVERNANCE, "is_open"),
         State(ids.COLLAPSE_REGIME, "is_open"),
         State(ids.COLLAPSE_EXPORT, "is_open")],
        prevent_initial_call=True,
    )
    def toggle_collapses(n_perf, n_shap, n_causal, n_vcro, n_multiasset, n_gov, n_regime, n_export,
                         is_perf, is_shap, is_causal, is_vcro, is_multiasset, is_gov, is_regime, is_export):
        """Bascule l'etat ouvert/ferme des sections pliables."""
        ctx = callback_context
        if not ctx.triggered:
            return (no_update,) * 8
        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if btn_id == ids.BTN_PERF:
            return not is_perf, is_shap, is_causal, is_vcro, is_multiasset, is_gov, is_regime, is_export
        elif btn_id == ids.BTN_SHAP:
            return is_perf, not is_shap, is_causal, is_vcro, is_multiasset, is_gov, is_regime, is_export
        elif btn_id == ids.BTN_CAUSAL:
            return is_perf, is_shap, not is_causal, is_vcro, is_multiasset, is_gov, is_regime, is_export
        elif btn_id == ids.BTN_VCRO:
            return is_perf, is_shap, is_causal, not is_vcro, is_multiasset, is_gov, is_regime, is_export
        elif btn_id == ids.BTN_MULTIASSET:
            return is_perf, is_shap, is_causal, is_vcro, not is_multiasset, is_gov, is_regime, is_export
        elif btn_id == ids.BTN_GOVERNANCE:
            return is_perf, is_shap, is_causal, is_vcro, is_multiasset, not is_gov, is_regime, is_export
        elif btn_id == ids.BTN_REGIME:
            return is_perf, is_shap, is_causal, is_vcro, is_multiasset, is_gov, not is_regime, is_export
        elif btn_id == ids.BTN_EXPORT:
            return is_perf, is_shap, is_causal, is_vcro, is_multiasset, is_gov, is_regime, not is_export
        return (no_update,) * 8

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
            comparison_df = pd_suite.get_comparison_table().to_pandas()
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
                    style_header=TABLE_HEADER_STYLE,
                    style_cell=TABLE_CELL_STYLE,
                    style_data_conditional=TABLE_ODD_ROW,
                )
            )

            # Calibration curve (visible by default — prominent)
            children.append(
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            figure=charts.plot_calibration_curve(
                                pd_suite.y_test,
                                {name: res.y_pred_test for name, res in pd_suite.results.items()},
                            ),
                            config={"displayModeBar": False},
                        ),
                        md=12,
                    ),
                ], className="mt-3")
            )

            # Feature importance (visible by default)
            children.append(build_section_title("Analyse des Features"))
            children.append(
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            figure=charts.plot_feature_importance(
                                pd_suite.get_feature_importance_table().to_pandas(),
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
                ], className="mt-2")
            )

            # Scorecard distribution (visible by default)
            lr_result = pd_suite.results.get("LR_WoE")
            if lr_result and getattr(lr_result, "scorecard_params", None):
                scores_test = pd_suite._pd_to_score(
                    lr_result.y_pred_test, lr_result.scorecard_params,
                )
                children.append(build_section_title("Scorecard Distribution"))
                children.append(
                    dcc.Graph(
                        figure=charts.plot_score_distribution(
                            scores_test, pd_suite.y_test, lr_result.scorecard_params,
                        ),
                        config={"displayModeBar": False},
                    )
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
            client_df = pl.DataFrame({
                "Feature": feature_names,
                "SHAP Value": client_shap,
                "Feature Value": client_features,
            }).with_columns(
                pl.col("SHAP Value").abs().alias("_abs_shap")
            ).sort("_abs_shap", descending=True).drop("_abs_shap").head(10)

            children.append(
                dash_table.DataTable(
                    data=client_df.to_pandas().round(4).to_dict("records"),
                    columns=[{"name": c, "id": c} for c in client_df.columns],
                    style_header=TABLE_HEADER_STYLE,
                    style_cell=TABLE_CELL_STYLE,
                    style_data_conditional=TABLE_ODD_ROW,
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
                        data=result_stressed.select(available_cols).head(100).to_pandas().round(4).to_dict("records"),
                        columns=[{"name": c, "id": c} for c in available_cols],
                        style_header=TABLE_HEADER_STYLE,
                        style_cell=TABLE_CELL_STYLE,
                        style_data_conditional=TABLE_ODD_ROW,
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
