"""Application Streamlit principale — IFRS 9 Risk Cockpit.

Point d'entrée du dashboard interactif. Orchestre le pipeline complet :
    1. Génération/chargement des données (cached)
    2. Entraînement des modèles PD (cached)
    3. Calcul ECL avec stress test en temps réel
    4. Affichage des KPI, rapport CRO et graphiques

Lancer avec : streamlit run ifrs9_cockpit/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ifrs9_cockpit.config import (
    DASHBOARD_CONFIG,
    RANDOM_SEED,
    TARGET,
)
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.engine.staging import StagingEngine
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.analytics.virtual_cro import VirtualCRO
from ifrs9_cockpit.analytics.metrics import ModelMetrics
from ifrs9_cockpit.dashboard.styles import get_main_css
from ifrs9_cockpit.dashboard.components import (
    render_header,
    render_kpi_cards,
    render_insight_box,
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
    """Génère et cache le dataset."""
    df_clients, df_history = generate_dataset()
    return df_clients, df_history


@st.cache_resource(show_spinner="Entraînement des modèles PD...")
def train_pd_models(df_hash: str) -> PDModelSuite:
    """Entraîne et cache la suite de modèles PD."""
    df_clients, _ = load_data()
    suite = PDModelSuite()
    suite.fit(df_clients)
    return suite


@st.cache_resource(show_spinner="Calibration LGD & EAD...")
def train_lgd_ead(df_hash: str) -> tuple:
    """Calibre et cache les modèles LGD et EAD."""
    df_clients, _ = load_data()
    lgd_model = LGDModel()
    lgd_model.fit(df_clients)
    ead_model = EADModel()
    ead_model.fit(df_clients)
    return lgd_model, ead_model


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────
def main() -> None:
    """Point d'entrée principal du dashboard."""

    # Inject CSS
    st.markdown(get_main_css(), unsafe_allow_html=True)

    # Load data & models
    df_clients, df_history = load_data()
    df_hash = str(len(df_clients))

    pd_suite = train_pd_models(df_hash)
    lgd_model, ead_model = train_lgd_ead(df_hash)

    # ── SIDEBAR : Stress Test Controls ──
    with st.sidebar:
        st.markdown(
            f'<h3 style="color:{DASHBOARD_CONFIG.theme_text};">Stress Test</h3>',
            unsafe_allow_html=True,
        )
        st.caption("Ajustez les paramètres macro pour observer l'impact en temps réel sur l'ECL.")

        unemployment_rate = st.slider(
            "Taux de chômage (%)",
            min_value=DASHBOARD_CONFIG.stress_unemployment_range[0],
            max_value=DASHBOARD_CONFIG.stress_unemployment_range[1],
            value=7.5,
            step=DASHBOARD_CONFIG.stress_unemployment_range[2],
            help="Baseline: 7.5%",
        )
        gdp_growth = st.slider(
            "Croissance PIB (%)",
            min_value=DASHBOARD_CONFIG.stress_gdp_range[0],
            max_value=DASHBOARD_CONFIG.stress_gdp_range[1],
            value=1.2,
            step=DASHBOARD_CONFIG.stress_gdp_range[2],
            help="Baseline: 1.2%",
        )

        st.divider()
        st.markdown(
            f'<h3 style="color:{DASHBOARD_CONFIG.theme_text};">Configuration</h3>',
            unsafe_allow_html=True,
        )
        selected_model = st.selectbox(
            "Modèle PD principal",
            options=list(pd_suite.results.keys()),
            index=0,
            help="Modèle utilisé pour le calcul ECL",
        )

        st.divider()
        st.caption(
            "IFRS 9 Risk Cockpit v1.0\n\n"
            "Moteur ECL | Virtual CRO | Stress Testing\n\n"
            "M2 Banque Finance"
        )

    # ── COMPUTE ECL ──
    pd_predictions = pd_suite.predict(df_clients)
    pd_current = pd_predictions[selected_model]
    pd_origination = pd_current * 0.8  # Proxy PD origination

    ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)

    # ECL de base (sans stress, pour comparaison)
    result_base = ecl_calc.calculate(df_clients, pd_current, pd_origination)
    ecl_base_total = result_base["ecl_weighted"].sum()

    # ECL stressé (avec les sliders)
    result_stressed = ecl_calc.calculate(
        df_clients, pd_current, pd_origination,
        unemployment_override=unemployment_rate,
        gdp_override=gdp_growth,
    )

    # ── Virtual CRO ──
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

    # ── HEADER ──
    render_header()

    # ── KPI CARDS ──
    render_kpi_cards(
        ecl_total=summary["ecl_total"],
        pd_mean=summary["pd_mean"],
        stage2_pct=summary["stage_2_pct"],
        coverage=summary["coverage"],
        ecl_previous=ecl_base_total,
    )

    # ── STAGE BADGES ──
    stage_counts = {
        1: summary["stage_1"],
        2: summary["stage_2"],
        3: summary["stage_3"],
    }
    render_stage_badges(stage_counts, summary["n_clients"])

    # ── CRO INSIGHT BOX ──
    render_insight_box(alerts)

    # ── TABS ──
    tab_perf, tab_ecl, tab_staging, tab_data = st.tabs([
        "Performance Modèles",
        "Analyse ECL",
        "Staging & Transitions",
        "Données",
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

        # Métriques détaillées
        render_section_title("Métriques Détaillées")
        comparison_styled = pd_suite.get_comparison_table()
        st.dataframe(
            comparison_styled,
            use_container_width=True,
            hide_index=True,
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

    # ── TAB 2 : Analyse ECL ──
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

    # ── TAB 3 : Staging & Transitions ──
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

    # ── TAB 4 : Données ──
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
                        PIB : {gdp_growth:+.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_section_title("Rapport CRO Complet")
        cro_report = cro.generate_report(
            result_stressed,
            ecl_previous=ecl_base_total,
            psi_value=psi_value,
            unemployment_rate=unemployment_rate,
            gdp_growth=gdp_growth,
        )
        st.code(cro_report, language=None)


if __name__ == "__main__":
    main()
