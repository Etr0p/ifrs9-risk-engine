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
from ifrs9_cockpit.analytics.ai_analyst import LocalCROAnalyst
from ifrs9_cockpit.analytics.metrics import ModelMetrics
from ifrs9_cockpit.dashboard.styles import get_main_css
from ifrs9_cockpit.dashboard.components import (
    render_header,
    render_kpi_cards,
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
        st.caption("Ajustez les 5 paramètres macro pour observer l'impact en temps réel sur l'ECL.")

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
        interest_rate = st.slider(
            "Taux directeur BCE (%)",
            min_value=DASHBOARD_CONFIG.stress_interest_rate_range[0],
            max_value=DASHBOARD_CONFIG.stress_interest_rate_range[1],
            value=3.5,
            step=DASHBOARD_CONFIG.stress_interest_rate_range[2],
            help="Baseline: 3.5%",
        )
        hpi_growth = st.slider(
            "Prix immobiliers (%)",
            min_value=DASHBOARD_CONFIG.stress_hpi_range[0],
            max_value=DASHBOARD_CONFIG.stress_hpi_range[1],
            value=2.0,
            step=DASHBOARD_CONFIG.stress_hpi_range[2],
            help="Baseline: +2.0%",
        )
        inflation_rate = st.slider(
            "Inflation IPC (%)",
            min_value=DASHBOARD_CONFIG.stress_inflation_range[0],
            max_value=DASHBOARD_CONFIG.stress_inflation_range[1],
            value=2.5,
            step=DASHBOARD_CONFIG.stress_inflation_range[2],
            help="Baseline: 2.5%",
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
            "IFRS 9 Risk Cockpit v2.0\n\n"
            "Moteur ECL | Virtual CRO | Stress Testing\n\n"
            "M1 Finance Paris-Saclay"
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
        interest_rate_override=interest_rate,
        hpi_override=hpi_growth,
        inflation_override=inflation_rate,
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

    # ── CRO INSIGHT BOX (Smart) ──
    cro_analyst = LocalCROAnalyst()
    briefing = cro_analyst.generate_executive_briefing(
        result_stressed=result_stressed,
        result_base=result_base,
        macro_params={
            "unemployment_rate": unemployment_rate,
            "gdp_growth": gdp_growth,
            "interest_rate": interest_rate,
            "hpi_growth": hpi_growth,
            "inflation_rate": inflation_rate,
        },
        psi_value=psi_value,
    )
    render_smart_insight_box(briefing)

    # ── TABS ──
    tab_perf, tab_ecl, tab_staging, tab_explain, tab_data, tab_cro = st.tabs([
        "Performance Modèles",
        "Analyse ECL",
        "Staging & Transitions",
        "Explainabilité",
        "Données & Export",
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

    # ── TAB 4 : Explainabilité ──
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

            # SHAP pour un client individuel
            render_section_title("Explication Client Individuel")
            client_idx = st.number_input(
                "Indice du client (dans le jeu de test)",
                min_value=0,
                max_value=n_sample - 1,
                value=0,
                step=1,
            )

            # Afficher la décomposition pour ce client
            client_shap = shap_vals[client_idx]
            client_df = pd.DataFrame({
                "feature": feature_names,
                "shap_value": client_shap,
                "feature_value": X_sample[client_idx],
            }).sort_values("shap_value", key=abs, ascending=False).head(10)

            st.dataframe(client_df, use_container_width=True, hide_index=True)

        except ImportError:
            st.warning("Le package `shap` n'est pas installé. Exécutez `pip install shap`.")
        except Exception as e:
            st.error(f"Erreur lors du calcul SHAP : {e}")

    # ── TAB 5 : Données & Export ──
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

        # ── Export Excel ──
        render_section_title("Export des Résultats")

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            # Export portefeuille complet
            buffer = io.BytesIO()
            export_cols = [c for c in available_cols if c in result_stressed.columns]
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                result_stressed[export_cols].to_excel(
                    writer, sheet_name="Portefeuille", index=False,
                )
                ecl_summary.to_excel(
                    writer, sheet_name="ECL_Summary", index=False,
                )
                pd_suite.get_comparison_table().to_excel(
                    writer, sheet_name="Benchmark_PD", index=False,
                )
            st.download_button(
                label="Télécharger les résultats (Excel)",
                data=buffer.getvalue(),
                file_name="ifrs9_cockpit_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with col_exp2:
            # Export rapport CRO
            cro_report = cro.generate_report(
                result_stressed,
                ecl_previous=ecl_base_total,
                psi_value=psi_value,
                unemployment_rate=unemployment_rate,
                gdp_growth=gdp_growth,
            )
            st.download_button(
                label="Télécharger le rapport CRO (TXT)",
                data=cro_report,
                file_name="rapport_cro.txt",
                mime="text/plain",
            )

        render_section_title("Rapport CRO (règles)")
        st.code(cro_report, language=None)

    # ── TAB 6 : Analyse CRO IA locale ──
    with tab_cro:
        render_section_title("Analyse CRO — Moteur d'Intelligence Embarquée")
        st.caption(
            "Ce module corrèle automatiquement l'ensemble des métriques du "
            "portefeuille (ECL, staging, macro, SHAP, backtesting, HHI, "
            "performance modèles) pour produire un rapport CRO complet. "
            "Aucune dépendance externe — tout le raisonnement est embarqué."
        )

        # Compute all data needed for the AI analyst
        staging_engine_cro = StagingEngine()

        waterfall_cro = ecl_calc.compute_waterfall(
            ecl_t0=result_base["ecl_weighted"].values,
            ecl_t1=result_stressed["ecl_weighted"].values,
            stages_t0=result_base["stage"].values,
            stages_t1=result_stressed["stage"].values,
            segments=df_clients["segment"].values,
        )

        transition_cro = staging_engine_cro.compute_transition_matrix(
            result_base["stage"].values,
            result_stressed["stage"].values,
        )

        seg_ead_cro = result_stressed.groupby("segment")["ead"].sum().values
        loan_ead_cro = result_stressed.groupby("loan_type")["ead"].sum().values
        hhi_seg_cro = ModelMetrics.hhi(seg_ead_cro)
        hhi_loan_cro = ModelMetrics.hhi(loan_ead_cro)

        bt_cro = ModelMetrics.compute_backtesting_metrics(
            pd_suite.y_test,
            pd_suite.results[selected_model].y_pred_test,
            n_folds=6,
        )

        comparison_cro = pd_suite.get_comparison_table()

        # SHAP top features (optional, uses cached function)
        shap_features_cro = None
        try:
            model_result_cro = pd_suite.results[selected_model]
            if selected_model == "LR_WoE":
                feat_names_cro = pd_suite._woe_features
                X_sh_cro = pd_suite.X_test[feat_names_cro].values[:500]
                calibrated_cro = model_result_cro.model
                if hasattr(calibrated_cro, 'calibrated_classifiers_'):
                    cc_cro = calibrated_cro.calibrated_classifiers_[0]
                    inner_cro = getattr(
                        cc_cro, 'estimator',
                        getattr(cc_cro, 'base_estimator', calibrated_cro),
                    )
                else:
                    inner_cro = calibrated_cro
            else:
                feat_names_cro = pd_suite._raw_features
                X_sh_cro = pd_suite.X_test[feat_names_cro].values[:500]
                inner_cro = model_result_cro.model

            shap_v_cro = compute_shap_values(
                inner_cro, X_sh_cro, feat_names_cro, selected_model,
            )
            mean_abs_cro = np.abs(shap_v_cro).mean(axis=0)
            top_idx_cro = np.argsort(mean_abs_cro)[::-1][:10]
            shap_features_cro = [
                (feat_names_cro[i], float(mean_abs_cro[i]))
                for i in top_idx_cro
            ]
        except Exception:
            pass

        macro_params_cro = {
            "unemployment_rate": unemployment_rate,
            "gdp_growth": gdp_growth,
            "interest_rate": interest_rate,
            "hpi_growth": hpi_growth,
            "inflation_rate": inflation_rate,
        }

        # Generate the AI CRO report
        analyst = LocalCROAnalyst()
        ai_report = analyst.generate_full_report(
            result_stressed=result_stressed,
            result_base=result_base,
            model_comparison=comparison_cro,
            macro_params=macro_params_cro,
            psi_value=psi_value,
            selected_model=selected_model,
            shap_top_features=shap_features_cro,
            hhi_segment=hhi_seg_cro,
            hhi_loan=hhi_loan_cro,
            backtesting_df=bt_cro,
            waterfall_df=waterfall_cro,
            transition_matrix=transition_cro,
        )

        # Display the report
        st.markdown(ai_report)

        # Export
        st.divider()
        col_export_ai1, col_export_ai2 = st.columns(2)
        with col_export_ai1:
            st.download_button(
                label="Télécharger le rapport CRO (TXT)",
                data=ai_report,
                file_name="rapport_cro_analyse.txt",
                mime="text/plain",
                key="ai_report_txt",
            )
        with col_export_ai2:
            st.download_button(
                label="Télécharger le rapport CRO (Markdown)",
                data=ai_report,
                file_name="rapport_cro_analyse.md",
                mime="text/markdown",
                key="ai_report_md",
            )


if __name__ == "__main__":
    main()
