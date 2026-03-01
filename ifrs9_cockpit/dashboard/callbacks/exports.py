"""Export callbacks — Excel, CRO TXT, AI TXT, LaTeX audit trail."""

from __future__ import annotations

import io
import json
import traceback

import numpy as np
import pandas as pd
import polars as pl
from dash import Input, Output, State, no_update, dcc

from ifrs9_cockpit.dashboard import ids
from ifrs9_cockpit.dashboard.cache import get_cached_pipeline, compute_shap_values
from ifrs9_cockpit.config import DASHBOARD_CONFIG


def _to_pd(df):
    """Convert Polars DataFrame to Pandas for Excel export."""
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    return df


def register_exports(app, pd_suite):
    """Enregistre les 4 callbacks d'export (Excel, CRO, AI, LaTeX)."""

    # ══════════════════════════════════════════════
    # CALLBACK 12a : Export Excel
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
        asymmetry_matrix = cached.get("asymmetry_matrix", pl.DataFrame())
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
                        result_base["stage"].to_numpy(), result_stressed["stage"].to_numpy(),
                    )
                    _to_pd(trans_matrix).to_excel(writer, sheet_name="2_Transitions", index=False)
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
                _to_pd(result_stressed.select(credit_cols)).to_excel(writer, sheet_name="5_Credit", index=False)

                # 6. PE
                pe_cols = [
                    c for c in [
                        "enterprise_id", "sector", "nav", "capital_invested",
                        "moic", "irr", "nav_drawdown", "distress_prob",
                        "risk_category", "expected_loss_pe", "rwa_pe",
                    ] if c in result_pe.columns
                ]
                _to_pd(result_pe.select(pe_cols)).to_excel(writer, sheet_name="6_PE", index=False)

                # 7. Optimisation
                opt_keys = [
                    "credit_allocation", "pe_allocation", "raroc_credit",
                    "raroc_pe", "rwa_weighted", "cet1_ratio", "cet1_headroom",
                ]
                opt_data = {k: optimization.get(k, 0) for k in opt_keys}
                pd.DataFrame([opt_data]).to_excel(writer, sheet_name="7_Optimisation", index=False)

                # 8. Asymetrie
                if isinstance(asymmetry_matrix, (pd.DataFrame, pl.DataFrame)) and len(asymmetry_matrix) > 0:
                    _to_pd(asymmetry_matrix).to_excel(writer, sheet_name="8_Asymetrie", index=False)

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
            n = analytics_state.narrative
            if isinstance(n, dict):
                # Format structured dict for text export
                sections = [
                    ("DIAGNOSTIC", n.get("diagnostic", "")),
                    ("CONCENTRATION", n.get("concentration", "")),
                    ("FACTEUR MACRO", n.get("facteur", "")),
                    ("RESILIENCE", n.get("resilience", "")),
                    ("ACTION", n.get("action", "")),
                    ("CONFIANCE", n.get("confiance", "")),
                    ("ALTERNATIVES", n.get("alternatives", "")),
                ]
                narrative = "\n\n".join(
                    f"--- {title} ---\n{text}" for title, text in sections if text
                )
            else:
                narrative = n or ""
        elif store_data:
            narrative = str(store_data.get("narrative", ""))

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
                result_base["stage"].to_numpy(), result_stressed["stage"].to_numpy(),
            )

            latex_content = generate_audit_latex(
                macro_params=cached["macro_params"],
                selected_model=cached["selected_model"],
                result_base=result_base,
                result_stressed=result_stressed,
                result_pe=cached["result_pe"],
                advanced_metrics=cached.get("advanced_metrics", {}),
                hhi_cross=cached.get("hhi_cross", {}),
                raroc_eva=cached.get("raroc_eva", pl.DataFrame()),
                asymmetry_matrix=cached.get("asymmetry_matrix", pl.DataFrame()),
                optimization=cached.get("optimization", {}),
                crr3_sensitivity=cached.get("crr3_sensitivity", pl.DataFrame()),
                analytics_state=cached.get("analytics_state"),
                trans_matrix=trans_matrix,
            )

            return dcc.send_string(latex_content, "ifrs9_audit_trail.tex")

        except Exception:
            return no_update
