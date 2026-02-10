"""Application Streamlit principale — IFRS 9 Risk Cockpit.

Point d'entrée du dashboard interactif. Orchestre le pipeline complet :
    1. Génération/chargement des données (cached)
    2. Entraînement des modèles PD (cached)
    3. Calcul ECL avec stress test en temps réel (5 variables macro)
    4. Affichage des KPI, rapport CRO et graphiques
    5. Explainabilité SHAP, backtesting, export Excel

Lancer avec : streamlit run ifrs9_cockpit/app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Permettre le lancement depuis n'importe quel répertoire
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from ifrs9_cockpit.config import (
    BASEL_CONFIG,
    DASHBOARD_CONFIG,
    MACRO_INCOHERENCE_RULES,
    PREDEFINED_SCENARIOS,
    RANDOM_SEED,
    SCENARIO_BASE,
    TARGET,
)
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.engine.staging import StagingEngine
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.engine.comparator import PortfolioComparator
from ifrs9_cockpit.ai_analyst import CROAnalyst
from ifrs9_cockpit.analytics.virtual_cro import VirtualCRO
from ifrs9_cockpit.analytics.ai_analyst import LocalCROAnalyst
from ifrs9_cockpit.analytics.metrics import ModelMetrics
from ifrs9_cockpit.dashboard.styles import get_main_css
from ifrs9_cockpit.dashboard.components import (
    render_header,
    render_kpi_cards,
    render_kpi_row,
    render_classification_row,
    render_ai_narrative_box,
    render_insight_box,
    render_smart_insight_box,
    render_stage_badges,
    render_section_title,
)
from ifrs9_cockpit.dashboard import charts
from ifrs9_cockpit.utils.helpers import format_euro, format_pct


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD_CONFIG.page_title,
    page_icon=DASHBOARD_CONFIG.page_icon,
    layout=DASHBOARD_CONFIG.layout,
)


# ──────────────────────────────────────────────
# CACHED DATA & MODELS
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Génération des données synthétiques...")
def load_data() -> tuple:
    """Génère et cache le dataset (credit + PE + historique)."""
    df_credit, df_pe, df_history = generate_dataset()
    return df_credit, df_pe, df_history


@st.cache_resource(show_spinner="Entraînement des modèles PD...")
def train_pd_models(df_hash: str) -> PDModelSuite:
    """Entraîne et cache la suite de modèles PD."""
    df_credit, _, _ = load_data()
    suite = PDModelSuite()
    suite.fit(df_credit)
    return suite


@st.cache_resource(show_spinner="Calibration LGD & EAD...")
def train_lgd_ead(df_hash: str) -> tuple:
    """Calibre et cache les modèles LGD et EAD."""
    df_credit, _, _ = load_data()
    lgd_model = LGDModel()
    lgd_model.fit(df_credit)
    ead_model = EADModel()
    ead_model.fit(df_credit)
    return lgd_model, ead_model


@st.cache_data(show_spinner="Calcul SHAP values...")
def compute_shap_values(
    _model,
    X_sample: np.ndarray,
    feature_names: list,
    model_name: str,
) -> np.ndarray:
    """Calcule et cache les SHAP values."""
    import shap
    if model_name == "LR_WoE":
        explainer = shap.LinearExplainer(_model, X_sample)
    else:
        explainer = shap.TreeExplainer(_model)
    shap_vals = explainer.shap_values(X_sample)
    # Gérer les différents formats de sortie SHAP :
    # - Liste de 2 arrays (classe 0, classe 1) → prendre classe 1
    # - Array 3D (n_samples, n_features, 2) → prendre [:, :, 1]
    # - Array 2D → utiliser directement
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]
    return shap_vals


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────
def main() -> None:
    """Point d'entrée principal du dashboard."""

    # Inject CSS
    st.markdown(get_main_css(), unsafe_allow_html=True)

    # Load data & models
    df_credit, df_pe, df_history = load_data()
    df_clients = df_credit  # Alias compat tabs existants
    df_hash = str(len(df_credit))

    pd_suite = train_pd_models(df_hash)
    lgd_model, ead_model = train_lgd_ead(df_hash)

    # ── SIDEBAR : Stress Test Controls ──
    with st.sidebar:
        st.markdown(
            f'<h3 style="color:{DASHBOARD_CONFIG.theme_text};">Stress Test</h3>',
            unsafe_allow_html=True,
        )

        # ── Scenarios predefinis (FR34) ──
        def _apply_scenario() -> None:
            """Pre-remplit les sliders depuis le scenario selectionne."""
            sel = st.session_state.get("scenario_selector", "Manuel")
            if sel != "Manuel" and sel in PREDEFINED_SCENARIOS:
                preset = PREDEFINED_SCENARIOS[sel]
                st.session_state["sl_interest_rate_bp"] = preset["interest_rate_bp"]
                st.session_state["sl_unemployment_bipolar"] = preset["unemployment_bipolar"]
                st.session_state["sl_gdp_pct"] = preset["gdp_pct"]
                st.session_state["sl_hpi_pct"] = preset["hpi_pct"]
                st.session_state["sl_inflation_pct"] = preset["inflation_pct"]

        st.selectbox(
            "Scenario predéfini (FR34)",
            options=["Manuel"] + list(PREDEFINED_SCENARIOS.keys()),
            key="scenario_selector",
            on_change=_apply_scenario,
            help="Pre-remplit les sliders ; ajustez ensuite a la main.",
        )

        st.caption("5 variables macro — format bp / bipolaire / pct (FR33).")

        # ── 5 Sliders macro (FR33) ──
        _ir = DASHBOARD_CONFIG.stress_interest_rate_range
        interest_rate_bp = st.slider(
            "Taux BCE (bp vs base)",
            min_value=_ir[0], max_value=_ir[1], value=0.0, step=_ir[2],
            key="sl_interest_rate_bp",
            help=f"Base : {SCENARIO_BASE.interest_rate:.1f}% | +100bp = {SCENARIO_BASE.interest_rate + 1.0:.1f}%",
        )

        _ue = DASHBOARD_CONFIG.stress_unemployment_range
        unemployment_bipolar = st.slider(
            "Chomage (bipolaire, pp)",
            min_value=_ue[0], max_value=_ue[1], value=0.0, step=_ue[2],
            key="sl_unemployment_bipolar",
            help=f"0 = base ({SCENARIO_BASE.unemployment_rate:.1f}%), negatif = hausse chomage",
        )

        _gd = DASHBOARD_CONFIG.stress_gdp_range
        gdp_pct = st.slider(
            "Croissance PIB (%)",
            min_value=_gd[0], max_value=_gd[1], value=1.0, step=_gd[2],
            key="sl_gdp_pct",
            help=f"Base : {SCENARIO_BASE.gdp_growth:.1f}%",
        )

        _hp = DASHBOARD_CONFIG.stress_hpi_range
        hpi_pct = st.slider(
            "Prix immobiliers (%)",
            min_value=_hp[0], max_value=_hp[1], value=2.0, step=_hp[2],
            key="sl_hpi_pct",
            help=f"Base : {SCENARIO_BASE.hpi_growth:+.1f}%",
        )

        _in = DASHBOARD_CONFIG.stress_inflation_range
        inflation_pct = st.slider(
            "Inflation IPC (%)",
            min_value=_in[0], max_value=_in[1], value=2.5, step=_in[2],
            key="sl_inflation_pct",
            help=f"Base : {SCENARIO_BASE.inflation_rate:.1f}%",
        )

        # ── Detection incoherence macro (FR36) ──
        _slider_vals = {
            "interest_rate_bp": interest_rate_bp,
            "unemployment_bipolar": unemployment_bipolar,
            "gdp_pct": gdp_pct,
            "hpi_pct": hpi_pct,
            "inflation_pct": inflation_pct,
        }
        for _rule in MACRO_INCOHERENCE_RULES:
            _all_met = True
            for _var, _op, _threshold in _rule.conditions:
                _val = _slider_vals.get(_var, 0.0)
                if _op == "gt" and not (_val > _threshold):
                    _all_met = False
                elif _op == "lt" and not (_val < _threshold):
                    _all_met = False
            if _all_met:
                st.warning(f"Incoherence : {_rule.description}")

        # ── Conversion sliders → valeurs macro reelles ──
        unemployment_rate = SCENARIO_BASE.unemployment_rate - unemployment_bipolar
        interest_rate = SCENARIO_BASE.interest_rate + interest_rate_bp / 100.0
        gdp_growth = gdp_pct
        hpi_growth = hpi_pct
        inflation_rate = inflation_pct

        st.divider()
        st.markdown(
            f'<h3 style="color:{DASHBOARD_CONFIG.theme_text};">Configuration</h3>',
            unsafe_allow_html=True,
        )
        selected_model = st.selectbox(
            "Modele PD principal",
            options=list(pd_suite.results.keys()),
            index=0,
            help="Modele utilise pour le calcul ECL",
        )

        # ── Configuration PE (FR53) ──
        st.divider()
        st.markdown(
            f'<h3 style="color:{DASHBOARD_CONFIG.theme_text};">Private Equity</h3>',
            unsafe_allow_html=True,
        )
        pe_allocation_pct = st.slider(
            "Allocation PE (%)",
            min_value=0, max_value=int(BASEL_CONFIG.pe_max_allocation * 100),
            value=20, step=5,
            help=f"Max CRR3 : {BASEL_CONFIG.pe_max_allocation:.0%}",
        )
        rw_pe_selected = st.selectbox(
            "Risk Weight PE (CRR3)",
            options=list(BASEL_CONFIG.rw_pe_options),
            index=list(BASEL_CONFIG.rw_pe_options).index(BASEL_CONFIG.rw_pe_default),
            help="190% IRB diversifie, 250% general, 400% speculatif",
        )

        # ── RST personnalise (FR54) ──
        st.divider()
        st.markdown(
            f'<h3 style="color:{DASHBOARD_CONFIG.theme_text};">Reverse Stress Test</h3>',
            unsafe_allow_html=True,
        )
        _capital_cet1 = BASEL_CONFIG.rwa_budget * BASEL_CONFIG.cet1_target
        _default_ecl_target = _capital_cet1 * 0.10
        rst_custom_enabled = st.checkbox(
            "Cible ECL personnalisee (FR54)",
            value=False,
            help="Definir un seuil ECL personnalise pour le RST",
        )
        if rst_custom_enabled:
            rst_target_ecl = st.number_input(
                "Cible ECL rupture (EUR)",
                min_value=1_000_000,
                max_value=int(_capital_cet1),
                value=int(_default_ecl_target),
                step=1_000_000,
                format="%d",
                help=f"Default : {_default_ecl_target:,.0f} (10% du CET1 = {_capital_cet1:,.0f})",
            )
        else:
            rst_target_ecl = None

        st.divider()
        st.caption(
            "IFRS 9 Risk Cockpit v3.0\n\n"
            "Moteur ECL | PE IFRS 13 | Virtual CRO | AI Analyst\n\n"
            "M1 Finance Paris-Saclay"
        )

    # ── HEADER (avant pipeline pour affichage immediat) ──
    render_header()

    # ── PIPELINE COMPLET (FR35 : recalcul automatique, FR52 : progression) ──
    _progress = st.progress(0, text="Initialisation pipeline...")

    macro_params = {
        "unemployment_rate": unemployment_rate,
        "gdp_growth": gdp_growth,
        "interest_rate": interest_rate,
        "hpi_growth": hpi_growth,
        "inflation_rate": inflation_rate,
    }

    # Stage 1/5 — Credit ECL
    _progress.progress(0.05, text="[1/5] Calcul ECL Credit...")
    pd_predictions = pd_suite.predict(df_clients)
    pd_current = pd_predictions[selected_model]
    # pd_origination reelle stockee dans df_credit (H2, Phase C)
    pd_origination = df_clients["pd_origination"].values

    ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
    result_base = ecl_calc.calculate(df_clients, pd_current, pd_origination)
    ecl_base_total = result_base["ecl_weighted"].sum()

    result_stressed = ecl_calc.calculate(
        df_clients, pd_current, pd_origination,
        unemployment_override=unemployment_rate,
        gdp_override=gdp_growth,
        interest_rate_override=interest_rate,
        hpi_override=hpi_growth,
        inflation_override=inflation_rate,
    )

    # Alias compat — tabs existants utilisent "segment" (nettoyage Story 6-4)
    for _df in [result_base, result_stressed]:
        if "sector" in _df.columns and "segment" not in _df.columns:
            _df["segment"] = _df["sector"]

    # Stage 2/5 — PE IFRS 13
    _progress.progress(0.25, text="[2/5] Valorisation PE IFRS 13...")
    pe_calc = PECalculator()
    result_pe = pe_calc.calculate(
        df_pe,
        unemployment_override=unemployment_rate,
        gdp_override=gdp_growth,
        interest_rate_override=interest_rate,
        hpi_override=hpi_growth,
        inflation_override=inflation_rate,
    )

    # Stage 3/5 — Comparaison & Metriques avancees
    _progress.progress(0.45, text="[3/5] Comparaison portefeuille...")
    comparator = PortfolioComparator(result_stressed, result_pe)
    advanced_metrics = comparator.compute_advanced_credit_metrics()
    hhi_cross = comparator.compute_hhi_crosscell()
    raroc_eva = comparator.compute_raroc_eva()
    asymmetry_matrix = comparator.build_asymmetry_matrix()

    # Stage 4/5 — Optimisation allocation
    _progress.progress(0.60, text="[4/5] Optimisation allocation CRR3...")
    optimization = comparator.optimize_allocation()
    crr3_sensitivity = comparator.compute_crr3_sensitivity()

    # Stage 5/5 — AI Analyst (2 passes)
    _progress.progress(0.80, text="[5/5] AI Analyst (2 passes)...")
    cro_analyst_ai = CROAnalyst(result_stressed, result_pe, macro_params, target_ecl=rst_target_ecl)
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

    _progress.progress(1.0, text="Pipeline complet.")
    _progress.empty()

    # ── KPI CARDS (FR45 : Credit, PE, Optimisation, Risk Appetite) ──
    _ecl_total = summary["ecl_total"]
    _ecl_delta = (_ecl_total - ecl_base_total) / max(ecl_base_total, 1)
    _ecl_cls = "negative" if _ecl_delta > 0.05 else "positive" if _ecl_delta <= 0 else "neutral"

    _nav_total = result_pe["nav"].sum()
    _delta_nav = result_pe["delta_nav"].sum()
    _drawdown = abs(_delta_nav / max(_nav_total, 1))
    _pe_cls = "negative" if _drawdown > 0.15 else "neutral" if _drawdown > 0.05 else "positive"

    _raroc_total_row = raroc_eva[raroc_eva["sector"] == "TOTAL_CREDIT"]
    _raroc_val = float(_raroc_total_row["raroc"].iloc[0]) if len(_raroc_total_row) > 0 else 0.0
    _raroc_cls = "positive" if _raroc_val > 0.12 else "neutral" if _raroc_val > 0.0 else "negative"

    _ra = analytics_state.risk_appetite_matrix
    _ra_rouge = int((_ra["signal"] == "rouge").sum()) if _ra is not None and len(_ra) > 0 else 0
    _ra_signal = "rouge" if _ra_rouge > 3 else "ambre" if _ra_rouge > 0 else "vert"
    _ra_cls = "negative" if _ra_signal == "rouge" else "neutral" if _ra_signal == "ambre" else "positive"

    render_kpi_row([
        {"label": "ECL CREDIT", "value": format_euro(_ecl_total),
         "sub_text": f"{'+'if _ecl_delta > 0 else ''}{_ecl_delta:.1%} vs base", "sub_class": _ecl_cls},
        {"label": "NAV DRAWDOWN PE", "value": f"{_drawdown:.1%}",
         "sub_text": f"Delta NAV : {format_euro(_delta_nav)}", "sub_class": _pe_cls},
        {"label": "RAROC CREDIT", "value": f"{_raroc_val:.2%}",
         "sub_text": f"HHI cross-cell : {hhi_cross.get('hhi_crosscell', 0):,}", "sub_class": _raroc_cls},
        {"label": "RISK APPETITE", "value": _ra_signal.upper(),
         "sub_text": f"{_ra_rouge} secteurs en rouge", "sub_class": _ra_cls},
    ])

    # ── CLASSIFICATION BADGES (FR47 : Stage + PE miroir) ──
    stage_counts = {
        1: summary["stage_1"],
        2: summary["stage_2"],
        3: summary["stage_3"],
    }
    pe_cats = result_pe["risk_category"].value_counts().to_dict() if "risk_category" in result_pe.columns else {}
    render_classification_row(stage_counts, pe_cats, summary["n_clients"])

    # ── AI ANALYST INSIGHT BOX (FR46) ──
    render_ai_narrative_box(
        narrative=analytics_state.narrative,
        recommendations=analytics_state.recommendations,
    )

    # ── TABS (8 onglets — FR48, FR55) ──
    (tab_perf, tab_ecl, tab_pe, tab_staging,
     tab_asym, tab_explain, tab_data, tab_cro) = st.tabs([
        "Performance Modeles",
        "Analyse ECL",
        "Analyse PE",
        "Staging & Transitions",
        "Asymetries & Optimisation",
        "Explainabilite",
        "Donnees & Export",
        "Analyse CRO",
    ])

    # ── TAB 1 : Performance Modèles ──
    with tab_perf:
        render_section_title("Benchmark des Modèles PD")

        # ROC Curves
        roc_data = {}
        for name, result in pd_suite.results.items():
            fpr, tpr, _ = ModelMetrics.roc_curve_data(
                pd_suite.y_test, result.y_pred_test,
            )
            roc_data[name] = (fpr, tpr, result.metrics_test["auc"])

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(charts.plot_roc_curves(roc_data), use_container_width=True)
        with col2:
            comparison_df = pd_suite.get_comparison_table()
            st.plotly_chart(charts.plot_model_comparison(comparison_df), use_container_width=True)

        # Courbe de calibration
        render_section_title("Courbe de Calibration")
        calib_predictions = {
            name: res.y_pred_test for name, res in pd_suite.results.items()
        }
        st.plotly_chart(
            charts.plot_calibration_curve(pd_suite.y_test, calib_predictions),
            use_container_width=True,
        )

        # Métriques détaillées
        render_section_title("Métriques Détaillées")
        comparison_styled = pd_suite.get_comparison_table()
        st.dataframe(
            comparison_styled,
            use_container_width=True,
            hide_index=True,
        )

        # Afficher les hyperparamètres XGBoost optimisés si disponibles
        if hasattr(pd_suite, '_xgb_best_params'):
            params = pd_suite._xgb_best_params
            params_str = " · ".join(
                f"**{k}** = {v}" for k, v in sorted(params.items())
            )
            st.markdown(
                f'<div style="color:#94A3B8;font-size:0.82rem;margin-top:0.5rem;">'
                f'XGBoost — Hyperparamètres optimisés (RandomizedSearchCV) : '
                f'{params_str}</div>',
                unsafe_allow_html=True,
            )

        col3, col4 = st.columns(2)
        with col3:
            feat_imp_df = pd_suite.get_feature_importance_table()
            st.plotly_chart(
                charts.plot_feature_importance(feat_imp_df, selected_model),
                use_container_width=True,
            )
        with col4:
            iv_table = pd_suite.woe_binner.get_iv_table()
            st.plotly_chart(charts.plot_iv_table(iv_table), use_container_width=True)

        # Backtesting
        render_section_title("Backtesting — Stabilité Temporelle")
        selected_result = pd_suite.results[selected_model]
        bt_metrics = ModelMetrics.compute_backtesting_metrics(
            pd_suite.y_test, selected_result.y_pred_test, n_folds=6,
        )
        if not bt_metrics.empty:
            st.plotly_chart(
                charts.plot_backtesting_auc(bt_metrics),
                use_container_width=True,
            )
            st.dataframe(bt_metrics, use_container_width=True, hide_index=True)

    # ── TAB 2 : Analyse ECL (FR46) ──
    with tab_ecl:
        render_section_title("Décomposition ECL")

        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(
                charts.plot_ecl_by_segment(result_stressed),
                use_container_width=True,
            )
        with col6:
            st.plotly_chart(
                charts.plot_ecl_coverage_scatter(result_stressed),
                use_container_width=True,
            )

        # HHI Concentration
        render_section_title("Concentration du Portefeuille (HHI)")
        seg_ead = result_stressed.groupby("segment")["ead"].sum().values
        loan_ead = result_stressed.groupby("loan_type")["ead"].sum().values
        hhi_seg = ModelMetrics.hhi(seg_ead)
        hhi_loan = ModelMetrics.hhi(loan_ead)

        st.plotly_chart(
            charts.plot_hhi_gauge(hhi_seg, hhi_loan),
            use_container_width=True,
        )

        col_hhi1, col_hhi2 = st.columns(2)
        with col_hhi1:
            if hhi_seg > 0.25:
                st.warning(f"HHI Segments = {hhi_seg:.4f} — Concentration **ELEVEE**. Revoir la diversification.")
            elif hhi_seg > 0.15:
                st.info(f"HHI Segments = {hhi_seg:.4f} — Concentration **MODEREE**.")
            else:
                st.success(f"HHI Segments = {hhi_seg:.4f} — Portefeuille **diversifié**.")
        with col_hhi2:
            if hhi_loan > 0.25:
                st.warning(f"HHI Prêts = {hhi_loan:.4f} — Concentration **ELEVEE** sur un type de produit.")
            else:
                st.success(f"HHI Prêts = {hhi_loan:.4f} — Mix produits acceptable.")

        # Waterfall
        render_section_title("Waterfall ECL (Base vs Stressé)")
        waterfall_df = ecl_calc.compute_waterfall(
            ecl_t0=result_base["ecl_weighted"].values,
            ecl_t1=result_stressed["ecl_weighted"].values,
            stages_t0=result_base["stage"].values,
            stages_t1=result_stressed["stage"].values,
            segments=df_clients["segment"].values,
        )
        st.plotly_chart(
            charts.plot_waterfall_ecl(waterfall_df),
            use_container_width=True,
        )

        # Résumé ECL
        render_section_title("Résumé ECL par Segment")
        ecl_summary = ecl_calc.compute_ecl_summary(result_stressed)
        st.dataframe(ecl_summary, use_container_width=True, hide_index=True)

    # ── TAB 3 : Analyse PE (FR48) ──
    with tab_pe:
        render_section_title("Portefeuille Private Equity — IFRS 13 Fair Value")

        col_pe1, col_pe2 = st.columns(2)
        with col_pe1:
            st.plotly_chart(
                charts.plot_pe_nav_by_sector(result_pe),
                use_container_width=True,
            )
        with col_pe2:
            st.plotly_chart(
                charts.plot_pe_risk_categories(result_pe),
                use_container_width=True,
            )

        render_section_title("Performance PE — MOIC & Drawdown")
        st.plotly_chart(
            charts.plot_pe_moic_drawdown(result_pe),
            use_container_width=True,
        )

        # Tableau recapitulatif PE
        render_section_title("Detail PE par Secteur")
        pe_summary = result_pe.groupby("sector").agg(
            positions=("enterprise_id", "count"),
            nav_total=("nav", "sum"),
            capital_total=("capital_invested", "sum"),
            moic_mean=("moic", "mean"),
            irr_mean=("irr", "mean"),
            drawdown_mean=("nav_drawdown", "mean"),
            el_pe_total=("expected_loss_pe", "sum"),
            rwa_pe_total=("rwa_pe", "sum"),
        ).reset_index()
        pe_summary.columns = [
            "Secteur", "Positions", "NAV Total", "Capital Investi",
            "MOIC Moyen", "IRR Moyen", "Drawdown Moyen",
            "EL PE Total", "RWA PE Total",
        ]
        st.dataframe(pe_summary, use_container_width=True, hide_index=True)

        # Parametres courants
        st.caption(
            f"Allocation PE : {pe_allocation_pct}% | "
            f"Risk Weight PE : {rw_pe_selected}% (CRR3)"
        )

    # ── TAB 4 : Staging & Transitions ──
    with tab_staging:
        render_section_title("Distribution des Stages")

        staging_engine = StagingEngine()
        stage_summary = staging_engine.get_stage_summary(
            result_stressed["stage"].values,
            result_stressed["ead"].values,
        )

        col7, col8 = st.columns(2)
        with col7:
            st.plotly_chart(
                charts.plot_stage_distribution(stage_summary),
                use_container_width=True,
            )
        with col8:
            # Matrice de transition Base → Stressé
            render_section_title("Matrice de Transition (Base vs Stress)")
            matrix = staging_engine.compute_transition_matrix(
                result_base["stage"].values,
                result_stressed["stage"].values,
            )
            st.plotly_chart(
                charts.plot_transition_matrix(matrix),
                use_container_width=True,
            )

        # Détail par stage
        render_section_title("Détail par Stage")
        st.dataframe(stage_summary, use_container_width=True, hide_index=True)

    # ── TAB 5 : Asymetries & Optimisation (FR55) ──
    with tab_asym:
        render_section_title("Matrice d'Asymetrie Credit vs PE")

        col_as1, col_as2 = st.columns(2)
        with col_as1:
            st.plotly_chart(
                charts.plot_asymmetry_heatmap(asymmetry_matrix),
                use_container_width=True,
            )
        with col_as2:
            st.plotly_chart(
                charts.plot_raroc_comparison(raroc_eva),
                use_container_width=True,
            )

        # Detail asymetrie
        render_section_title("Detail Asymetrie par Secteur")
        asym_display = asymmetry_matrix[[
            "sector", "ecl_credit", "el_pe", "loss_ratio",
            "rwa_credit", "rwa_pe", "rwa_ratio",
            "raroc_credit", "raroc_pe", "raroc_delta",
        ]].copy()
        asym_display.columns = [
            "Secteur", "ECL Credit", "EL PE", "Ratio Perte",
            "RWA Credit", "RWA PE", "Ratio RWA",
            "RAROC Credit", "RAROC PE", "Delta RAROC",
        ]
        st.dataframe(asym_display, use_container_width=True, hide_index=True)

        # CRR3 Sensitivity
        render_section_title("Sensibilite CRR3 — Impact Risk Weight PE")
        col_crr1, col_crr2 = st.columns([2, 1])
        with col_crr1:
            st.plotly_chart(
                charts.plot_crr3_sensitivity(crr3_sensitivity),
                use_container_width=True,
            )
        with col_crr2:
            st.dataframe(crr3_sensitivity, use_container_width=True, hide_index=True)

        # Optimisation
        render_section_title("Optimisation Allocation (3 niveaux)")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            st.markdown(
                f"""
                <div class="insight-box" style="margin-top: 0;">
                    <div class="insight-box-header"><h3>Allocation Optimale</h3></div>
                    <div class="insight-box-content">
                        <b>Credit :</b> {optimization['credit_allocation']:.1%}<br/>
                        <b>PE :</b> {optimization['pe_allocation']:.1%}<br/>
                        <b>RAROC Credit :</b> {optimization['raroc_credit']:.2%}<br/>
                        <b>RAROC PE :</b> {optimization['raroc_pe']:.2%}<br/>
                        <b>RWA Pondere :</b> {format_euro(optimization['rwa_weighted'])}<br/>
                        <b>CET1 Ratio :</b> {optimization['cet1_ratio']:.2%}<br/>
                        <b>CET1 Headroom :</b> {optimization['cet1_headroom']:+.2%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_opt2:
            # Poids sectoriels
            weights_credit = optimization.get("sector_weights_credit", {})
            weights_pe = optimization.get("sector_weights_pe", {})
            if weights_credit:
                weights_df = pd.DataFrame({
                    "Secteur": list(weights_credit.keys()),
                    "Poids Credit": [f"{v:.1%}" for v in weights_credit.values()],
                    "Poids PE": [f"{weights_pe.get(k, 0):.1%}" for k in weights_credit.keys()],
                })
                st.dataframe(weights_df, use_container_width=True, hide_index=True)

        # ── Drill-down sectoriel (FR54) ──
        render_section_title("Drill-down Sectoriel (FR54)")
        _sector_names = asymmetry_matrix["sector"].tolist()
        _selected_sector = st.selectbox(
            "Selectionner un secteur pour le detail",
            options=_sector_names,
            key="drill_sector",
        )

        if _selected_sector and analytics_state.euler_contributions is not None:
            col_dd1, col_dd2 = st.columns(2)

            with col_dd1:
                # Euler pour ce secteur
                _euler = analytics_state.euler_contributions
                _euler_sec = _euler[_euler["sector"] == _selected_sector]
                if len(_euler_sec) > 0:
                    st.markdown(f"**Allocation Proportionnelle — {_selected_sector}**")
                    st.dataframe(_euler_sec[["canal", "risk_amount", "euler_share", "rwa"]],
                                 use_container_width=True, hide_index=True)

                # Factor attribution pour ce secteur
                _factors = analytics_state.factor_attribution
                if _factors is not None and len(_factors) > 0:
                    st.markdown(f"**Attribution Factorielle Macro**")
                    st.dataframe(_factors[["variable", "canal", "delta_from_base", "attribution"]],
                                 use_container_width=True, hide_index=True)

            with col_dd2:
                # Risk appetite pour ce secteur
                _ra = analytics_state.risk_appetite_matrix
                if _ra is not None and len(_ra) > 0:
                    _ra_sec = _ra[_ra["sector"] == _selected_sector] if "sector" in _ra.columns else _ra
                    if len(_ra_sec) > 0:
                        st.markdown(f"**Risk Appetite — {_selected_sector}**")
                        st.dataframe(_ra_sec, use_container_width=True, hide_index=True)

                # Trajectoires pour ce secteur
                _traj = analytics_state.trajectories
                if _traj is not None and len(_traj) > 0:
                    st.markdown(f"**Trajectoires Prospectives**")
                    _traj_display = _traj.copy()
                    if "sector" in _traj_display.columns:
                        _traj_sec = _traj_display[_traj_display["sector"] == _selected_sector]
                        if len(_traj_sec) > 0:
                            st.dataframe(_traj_sec, use_container_width=True, hide_index=True)
                        else:
                            st.dataframe(_traj_display, use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(_traj_display, use_container_width=True, hide_index=True)

    # ── TAB 6 : Explainabilite ──
    with tab_explain:
        render_section_title("Explainabilité SHAP")
        st.caption(
            "Les SHAP values décomposent la prédiction de chaque client en "
            "contributions individuelles par feature. Elles permettent de comprendre "
            "**pourquoi** un client a une PD élevée ou basse."
        )

        try:
            import shap

            # Préparer les données pour SHAP
            model_result = pd_suite.results[selected_model]
            if selected_model == "LR_WoE":
                feature_names = pd_suite._woe_features
                X_shap = pd_suite.X_test[feature_names].values
                # Extraire la LogisticRegression du CalibratedClassifierCV
                calibrated = model_result.model
                if hasattr(calibrated, 'calibrated_classifiers_'):
                    cc = calibrated.calibrated_classifiers_[0]
                    # sklearn >= 1.2 : .estimator, plus ancien : .base_estimator
                    inner_model = getattr(cc, 'estimator', getattr(cc, 'base_estimator', calibrated))
                else:
                    inner_model = calibrated
            else:
                feature_names = pd_suite._raw_features
                X_shap = pd_suite.X_test[feature_names].values
                inner_model = model_result.model

            # Sous-échantillonner pour la performance
            n_sample = min(1000, len(X_shap))
            X_sample = X_shap[:n_sample]

            # Calculer les SHAP values
            shap_vals = compute_shap_values(
                inner_model, X_sample, feature_names, selected_model,
            )

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.plotly_chart(
                    charts.plot_shap_summary(shap_vals, feature_names),
                    use_container_width=True,
                )
            with col_s2:
                st.plotly_chart(
                    charts.plot_shap_beeswarm(shap_vals, X_sample, feature_names),
                    use_container_width=True,
                )

            # SHAP individuel avec @st.fragment (FR49 — pas de recompute pipeline)
            render_section_title("Explication Entreprise Individuelle (FR49)")

            @st.fragment
            def _shap_individual_fragment(
                _shap_vals: np.ndarray,
                _X_sample: np.ndarray,
                _feature_names: list,
                _n_sample: int,
            ) -> None:
                """Fragment isole : selecteur entreprise + force plot."""
                client_idx = st.number_input(
                    "Indice de l'entreprise (dans le jeu de test)",
                    min_value=0,
                    max_value=_n_sample - 1,
                    value=0,
                    step=1,
                    key="shap_client_idx",
                )

                client_shap = _shap_vals[client_idx]
                client_features = _X_sample[client_idx]

                # Force plot visuel
                st.plotly_chart(
                    charts.plot_shap_force_individual(
                        client_shap, client_features, _feature_names,
                    ),
                    use_container_width=True,
                )

                # Tableau detaille
                client_df = pd.DataFrame({
                    "Feature": _feature_names,
                    "SHAP Value": client_shap,
                    "Feature Value": client_features,
                }).sort_values("SHAP Value", key=abs, ascending=False).head(10)
                st.dataframe(client_df, use_container_width=True, hide_index=True)

            _shap_individual_fragment(shap_vals, X_sample, feature_names, n_sample)

        except ImportError:
            st.warning("Le package `shap` n'est pas installé. Exécutez `pip install shap`.")
        except Exception as e:
            st.error(f"Erreur lors du calcul SHAP : {e}")

    # ── TAB 7 : Donnees & Export ──
    with tab_data:
        render_section_title("Aperçu du Portefeuille")

        col9, col10 = st.columns([2, 1])
        with col9:
            display_cols = [
                "client_id", "segment", "age", "credit_score", "income",
                "debt_ratio", "loan_type", "loan_amount", "stage",
                "pd_12m", "lgd", "ead", "ecl_weighted",
            ]
            available_cols = [c for c in display_cols if c in result_stressed.columns]
            st.dataframe(
                result_stressed[available_cols].head(100),
                use_container_width=True,
                hide_index=True,
            )
        with col10:
            st.markdown(
                f"""
                <div class="insight-box" style="margin-top: 0;">
                    <div class="insight-box-header"><h3>Statistiques Clés</h3></div>
                    <div class="insight-box-content">
                        <b>Clients :</b> {summary['n_clients']:,}<br/>
                        <b>ECL Total :</b> {format_euro(summary['ecl_total'])}<br/>
                        <b>EAD Total :</b> {format_euro(summary['ead_total'])}<br/>
                        <b>Coverage :</b> {format_pct(summary['coverage'])}<br/>
                        <b>PD Moyenne :</b> {format_pct(summary['pd_mean'])}<br/>
                        <b>Segment + risqué :</b> {summary['riskiest_segment']}<br/><br/>
                        <b>Macro :</b><br/>
                        Chômage : {unemployment_rate:.1f}%<br/>
                        PIB : {gdp_growth:+.1f}%<br/>
                        Taux BCE : {interest_rate:.2f}%<br/>
                        Immobilier : {hpi_growth:+.1f}%<br/>
                        Inflation : {inflation_rate:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Export Excel 7 feuilles (FR50) ──
        render_section_title("Export Multi-Sheet (FR50)")

        col_exp1, col_exp2, col_exp3 = st.columns(3)
        with col_exp1:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                # 1. Macro — parametres scenario
                pd.DataFrame([macro_params]).to_excel(
                    writer, sheet_name="1_Macro", index=False,
                )
                # 2. Transitions — matrice stage
                staging_exp = StagingEngine()
                trans_matrix = staging_exp.compute_transition_matrix(
                    result_base["stage"].values, result_stressed["stage"].values,
                )
                trans_matrix.to_excel(writer, sheet_name="2_Transitions")
                # 3. Alertes CRO
                alerts_df = pd.DataFrame([
                    {"severity": a.severity, "title": a.title, "message": a.message, "action": a.action or ""}
                    for a in alerts
                ]) if alerts else pd.DataFrame(columns=["severity", "title", "message", "action"])
                alerts_df.to_excel(writer, sheet_name="3_Alertes", index=False)
                # 4. SHAP Top 10
                try:
                    _mr = pd_suite.results[selected_model]
                    if selected_model == "LR_WoE":
                        _fn = pd_suite._woe_features
                        _Xs = pd_suite.X_test[_fn].values[:500]
                        _cal = _mr.model
                        if hasattr(_cal, 'calibrated_classifiers_'):
                            _cc = _cal.calibrated_classifiers_[0]
                            _im = getattr(_cc, 'estimator', getattr(_cc, 'base_estimator', _cal))
                        else:
                            _im = _cal
                    else:
                        _fn = pd_suite._raw_features
                        _Xs = pd_suite.X_test[_fn].values[:500]
                        _im = _mr.model
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
                # 5. Credit — metriques portefeuille
                credit_cols = [c for c in [
                    "enterprise_id", "sector", "loan_type", "loan_amount",
                    "stage", "pd_12m", "lgd", "ead", "ecl_weighted",
                ] if c in result_stressed.columns]
                result_stressed[credit_cols].to_excel(writer, sheet_name="5_Credit", index=False)
                # 6. PE — metriques
                pe_cols = [c for c in [
                    "enterprise_id", "sector", "nav", "capital_invested",
                    "moic", "irr", "nav_drawdown", "distress_prob",
                    "risk_category", "expected_loss_pe", "rwa_pe",
                ] if c in result_pe.columns]
                result_pe[pe_cols].to_excel(writer, sheet_name="6_PE", index=False)
                # 7. Optimisation
                opt_df = pd.DataFrame([{
                    "credit_allocation": optimization["credit_allocation"],
                    "pe_allocation": optimization["pe_allocation"],
                    "raroc_credit": optimization["raroc_credit"],
                    "raroc_pe": optimization["raroc_pe"],
                    "rwa_weighted": optimization["rwa_weighted"],
                    "cet1_ratio": optimization["cet1_ratio"],
                    "cet1_headroom": optimization["cet1_headroom"],
                }])
                opt_df.to_excel(writer, sheet_name="7_Optimisation", index=False)
                # Bonus: asymmetry matrix
                asymmetry_matrix.to_excel(writer, sheet_name="8_Asymetrie", index=False)

            st.download_button(
                label="Export complet (Excel 8 feuilles)",
                data=buffer.getvalue(),
                file_name="ifrs9_cockpit_complet.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with col_exp2:
            # Export rapport CRO texte
            cro_report = cro.generate_report(
                result_stressed,
                ecl_previous=ecl_base_total,
                psi_value=psi_value,
                unemployment_rate=unemployment_rate,
                gdp_growth=gdp_growth,
            )
            st.download_button(
                label="Rapport CRO (TXT)",
                data=cro_report,
                file_name="rapport_cro.txt",
                mime="text/plain",
            )

        with col_exp3:
            # Export synthese AI Analyst
            _ai_narrative = analytics_state.narrative or ""
            st.download_button(
                label="Synthese AI Analyst (TXT)",
                data=_ai_narrative,
                file_name="ai_analyst_synthese.txt",
                mime="text/plain",
                key="export_ai_txt",
            )

        render_section_title("Rapport CRO (regles)")
        st.code(cro_report, language=None)

    # ── TAB 8 : Analyse CRO — AI Analyst (FR55) ──
    with tab_cro:
        render_section_title("Analyse CRO — AI Analyst (2 passes)")
        st.caption(
            "Moteur d'intelligence embarque a 5 couches : "
            "Crossing, Allocation Proportionnelle, Tipping Points, Regime, Prospective. "
            "2 passes iteratives pour affiner les recommandations."
        )

        # ── Risk Appetite Matrix (feux tricolores) ──
        if analytics_state.risk_appetite_matrix is not None and len(analytics_state.risk_appetite_matrix) > 0:
            st.plotly_chart(
                charts.plot_risk_appetite_matrix(analytics_state.risk_appetite_matrix),
                use_container_width=True,
            )

        # ── Regime detecte ──
        render_section_title("Regime Macro Detecte")
        if analytics_state.regime is not None:
            regime = analytics_state.regime
            _regime_name = regime.detected_regime
            _regime_probs = regime.probabilities
            _regime_text = " | ".join(
                f"{k}: {v:.1%}" for k, v in sorted(_regime_probs.items(), key=lambda x: -x[1])
            )
            st.info(f"Regime detecte : **{_regime_name}** ({_regime_text})")
        else:
            st.info("Regime non disponible.")

        # ── Tipping Points ──
        if analytics_state.tipping_points is not None and len(analytics_state.tipping_points) > 0:
            render_section_title("Seuils de Basculement (Tipping Points)")
            st.dataframe(analytics_state.tipping_points, use_container_width=True, hide_index=True)

        # ── RST Distance + Scenario de rupture (FR54) ──
        if analytics_state.rst_distance > 0:
            _rst_color = "#EF4444" if analytics_state.rst_distance < 1.0 else "#F59E0B" if analytics_state.rst_distance < 2.0 else "#06D6A0"
            _rst = analytics_state.rst_result or {}
            _rst_custom = " (cible personnalisee)" if rst_custom_enabled else ""
            st.markdown(
                f'<div style="padding:0.5rem 1rem;border-left:3px solid {_rst_color};margin:0.5rem 0;">'
                f'<b>Distance au point de rupture (RST){_rst_custom} :</b> '
                f'<span style="color:{_rst_color};font-weight:700;">{analytics_state.rst_distance:.2f} sigma</span>'
                f'<br/><span style="color:#94A3B8;font-size:0.85rem;">'
                f'Seuil ECL : {format_euro(_rst.get("ecl_breach_threshold", 0))} | '
                f'Capital CET1 : {format_euro(_rst.get("capital_base", 0))}'
                f'</span></div>',
                unsafe_allow_html=True,
            )
            # Afficher le scenario de rupture si breach trouve
            if _rst.get("breach") and _rst.get("rst_scenario"):
                render_section_title("Scenario de Rupture RST")
                _rst_scen = _rst["rst_scenario"]
                rst_scen_df = pd.DataFrame([{
                    "Variable": var,
                    "Valeur RST": f"{val:.2f}",
                    "Baseline": f"{getattr(SCENARIO_BASE, var):.2f}",
                    "Delta": f"{val - getattr(SCENARIO_BASE, var):+.2f}",
                } for var, val in _rst_scen.items()])
                st.dataframe(rst_scen_df, use_container_width=True, hide_index=True)

        # ── Trajectoires prospectives ──
        if analytics_state.trajectories is not None and len(analytics_state.trajectories) > 0:
            render_section_title("Trajectoires Prospectives (T+3 a T+12)")
            st.dataframe(analytics_state.trajectories, use_container_width=True, hide_index=True)

        # ── Early Warning ──
        if analytics_state.early_warning is not None and len(analytics_state.early_warning) > 0:
            render_section_title("Indicateurs d'Alerte Precoce")
            st.dataframe(analytics_state.early_warning, use_container_width=True, hide_index=True)

        # ── Narrative AI Analyst (synthese texte) ──
        render_section_title("Synthese Narrative (Passe {})".format(analytics_state.pass_number))
        st.markdown(analytics_state.narrative.replace("\n", "  \n") if analytics_state.narrative else "Analyse en cours...")

        # ── Rapport CRO detaille (LocalCROAnalyst — secondaire) ──
        with st.expander("Rapport CRO detaille (moteur analytique local)", expanded=False):
            analyst = LocalCROAnalyst()
            ai_report = analyst.generate_full_report(
                result_stressed=result_stressed,
                result_base=result_base,
                model_comparison=pd_suite.get_comparison_table(),
                macro_params=macro_params,
                psi_value=psi_value,
                selected_model=selected_model,
                shap_top_features=None,
                hhi_segment=ModelMetrics.hhi(
                    result_stressed.groupby("segment")["ead"].sum().values,
                ),
                hhi_loan=ModelMetrics.hhi(
                    result_stressed.groupby("loan_type")["ead"].sum().values,
                ),
                backtesting_df=ModelMetrics.compute_backtesting_metrics(
                    pd_suite.y_test,
                    pd_suite.results[selected_model].y_pred_test,
                    n_folds=6,
                ),
                waterfall_df=ecl_calc.compute_waterfall(
                    ecl_t0=result_base["ecl_weighted"].values,
                    ecl_t1=result_stressed["ecl_weighted"].values,
                    stages_t0=result_base["stage"].values,
                    stages_t1=result_stressed["stage"].values,
                    segments=df_clients["segment"].values,
                ),
                transition_matrix=StagingEngine().compute_transition_matrix(
                    result_base["stage"].values,
                    result_stressed["stage"].values,
                ),
            )
            st.markdown(ai_report)

        # Export
        st.divider()
        col_export_ai1, col_export_ai2 = st.columns(2)
        with col_export_ai1:
            _narrative_export = analytics_state.narrative or ""
            st.download_button(
                label="Telecharger synthese AI (TXT)",
                data=_narrative_export,
                file_name="ai_analyst_synthese.txt",
                mime="text/plain",
                key="ai_report_txt",
            )
        with col_export_ai2:
            st.download_button(
                label="Telecharger synthese AI (Markdown)",
                data=_narrative_export,
                file_name="ai_analyst_synthese.md",
                mime="text/markdown",
                key="ai_report_md",
            )


if __name__ == "__main__":
    main()
