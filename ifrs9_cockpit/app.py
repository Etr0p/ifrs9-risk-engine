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
    RISK_APPETITE_CONFIG,
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
from ifrs9_cockpit.export.audit_trail_latex import generate_audit_latex


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


@st.cache_resource(show_spinner="Chargement des modèles PD...")
def train_pd_models(df_hash: str) -> PDModelSuite:
    """Charge des modèles pré-entraînés ou entraîne sur données synthétiques 30K."""
    pretrained_path = Path("ifrs9_cockpit/training/models/pd_suite.joblib")
    if pretrained_path.exists():
        try:
            suite = PDModelSuite.load(str(pretrained_path))
            return suite
        except Exception:
            pass  # Fichier corrompu → fallback entraînement
    # Fallback : entraîner sur données synthétiques 30K
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


def _force_models_cpu(pd_suite: PDModelSuite) -> None:
    """Force tous les modeles sur CPU pour eviter segfault CUDA nightly.

    Le training GPU est cache (st.cache_resource), mais les predictions
    repetees sur GPU nightly Blackwell causent des segfaults lors des
    reruns Streamlit. CPU est assez rapide pour 30K predictions.
    """
    import torch as _torch

    _cpu = _torch.device("cpu")

    def _move_all_tensors(module: _torch.nn.Module) -> None:
        """Deplace params + buffers + tenseurs bruts (group_attention_matrix)."""
        module.to(_cpu)
        for m in module.modules():
            for attr_name in list(m.__dict__.keys()):
                attr = m.__dict__[attr_name]
                if isinstance(attr, _torch.Tensor) and attr.device != _cpu:
                    m.__dict__[attr_name] = attr.to(_cpu)

    for _name, result in pd_suite.results.items():
        model = result.model
        if not hasattr(model, "calibrated_classifiers_"):
            continue
        for cc in model.calibrated_classifiers_:
            inner = getattr(cc, "estimator", getattr(cc, "base_estimator", None))
            if inner is None:
                continue
            # TabNet → deplacer TOUS les tenseurs sur CPU
            if hasattr(inner, "_model") and hasattr(inner._model, "network"):
                try:
                    _move_all_tensors(inner._model.network)
                    inner._model.device_name = "cpu"
                    inner._model.device = _cpu
                except Exception:
                    pass
            # XGBoost → forcer CPU
            if hasattr(inner, "set_params"):
                try:
                    inner.set_params(device="cpu")
                except (TypeError, ValueError):
                    pass


_SHAP_MAX_SAMPLES = 500  # Limiter pour éviter timeout KernelExplainer
_SHAP_BACKGROUND = 50    # Échantillons de fond (KernelExplainer)


@st.cache_data(show_spinner="Calcul SHAP values...")
def compute_shap_values(
    _model,
    X_sample: np.ndarray,
    feature_names: list,
    model_name: str,
) -> np.ndarray:
    """Calcule et cache les SHAP values."""
    import shap
    # Limiter la taille pour éviter les timeouts (surtout KernelExplainer)
    if X_sample.shape[0] > _SHAP_MAX_SAMPLES:
        X_sample = X_sample[:_SHAP_MAX_SAMPLES]
    if model_name == "LR_WoE":
        explainer = shap.LinearExplainer(_model, X_sample)
    elif model_name == "XGBoost":
        explainer = shap.TreeExplainer(_model)
    else:  # TabNet — model-agnostic via KernelExplainer
        n_bg = min(_SHAP_BACKGROUND, X_sample.shape[0])
        background = shap.sample(X_sample, n_bg)
        explainer = shap.KernelExplainer(
            lambda x: _model.predict_proba(x)[:, 1], background,
        )
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

    # Force CPU pour les predictions (evite segfault CUDA nightly Blackwell)
    _force_models_cpu(pd_suite)

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
            "Chomage bipolaire (pp)",
            min_value=_ue[0], max_value=_ue[1], value=0.0, step=_ue[2],
            key="sl_unemployment_bipolar",
            help=(
                f"0 = base ({SCENARIO_BASE.unemployment_rate:.1f}%). "
                "Les deux extremes augmentent le chomage : "
                "positif = rupture techno (PE tech beneficie), "
                "negatif = crise eco (tout souffre)."
            ),
        )

        _gd = DASHBOARD_CONFIG.stress_gdp_range
        gdp_pct = st.slider(
            "Croissance PIB (%)",
            min_value=_gd[0], max_value=_gd[1], value=SCENARIO_BASE.gdp_growth, step=_gd[2],
            key="sl_gdp_pct",
            help=f"Base : {SCENARIO_BASE.gdp_growth:.1f}%",
        )

        _hp = DASHBOARD_CONFIG.stress_hpi_range
        hpi_pct = st.slider(
            "Prix immobiliers (%)",
            min_value=_hp[0], max_value=_hp[1], value=SCENARIO_BASE.hpi_growth, step=_hp[2],
            key="sl_hpi_pct",
            help=f"Base : {SCENARIO_BASE.hpi_growth:+.1f}%",
        )

        _inf = DASHBOARD_CONFIG.stress_inflation_range
        inflation_pct = st.slider(
            "Inflation IPC (%)",
            min_value=_inf[0], max_value=_inf[1], value=SCENARIO_BASE.inflation_rate, step=_inf[2],
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
        # Bipolaire : |slider| = amplitude du choc chomage (toujours hausse),
        # signe = nature (+ = rupture techno, - = crise eco).
        unemployment_rate = SCENARIO_BASE.unemployment_rate + abs(unemployment_bipolar)
        # Nature du chomage pour le canal PE : crise eco → abs(sensitivity)
        _unemployment_crisis = unemployment_bipolar < 0
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

    try:
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
            unemployment_crisis=_unemployment_crisis,
        )

        # Stage 3/5 — Comparaison & Metriques avancees
        _progress.progress(0.45, text="[3/5] Comparaison portefeuille...")
        comparator = PortfolioComparator(result_stressed, result_pe)
        advanced_metrics = comparator.compute_advanced_credit_metrics()
        hhi_cross = comparator.compute_hhi_crosscell()
        gar_result = comparator.compute_green_asset_ratio()
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

    except Exception as _pipeline_err:
        _progress.empty()
        import traceback
        st.error(f"Erreur pipeline : {_pipeline_err}")
        st.code(traceback.format_exc(), language=None)
        st.stop()

    # ── KPI CARDS (FR45 : Credit, PE, Optimisation, Risk Appetite) ──
    _ecl_total = summary["ecl_total"]
    _ecl_delta = (_ecl_total - ecl_base_total) / max(ecl_base_total, 1)
    _ecl_cls = "negative" if _ecl_delta > 0.05 else "positive" if _ecl_delta <= 0 else "neutral"

    _nav_total = result_pe["nav"].sum()
    _delta_nav = result_pe["delta_nav"].sum()
    # Drawdown = perte de NAV par rapport au baseline (pas au stressed)
    _nav_ref_total = _nav_total - _delta_nav  # NAV baseline = NAV_stressed - delta
    _drawdown = max(0, -_delta_nav) / max(_nav_ref_total, 1)
    _pe_cls = "negative" if _drawdown > 0.15 else "neutral" if _drawdown > 0.05 else "positive"

    _raroc_total_row = raroc_eva[(raroc_eva["sector"] == "Total") & (raroc_eva["canal"] == "Credit")]
    _raroc_val = float(_raroc_total_row["raroc"].iloc[0]) if len(_raroc_total_row) > 0 else 0.0
    _raroc_cls = "positive" if _raroc_val > 0.12 else "neutral" if _raroc_val > 0.0 else "negative"

    _ra = analytics_state.risk_appetite_matrix
    _ra_rouge = int((_ra["signal"] == "rouge").sum()) if _ra is not None and len(_ra) > 0 else 0
    _ra_signal = "rouge" if _ra_rouge > 3 else "ambre" if _ra_rouge > 1 else "vert"
    _ra_cls = "negative" if _ra_signal == "rouge" else "neutral" if _ra_signal == "ambre" else "positive"

    render_kpi_row([
        {"label": "RISK APPETITE", "value": _ra_signal.upper(),
         "sub_text": f"{_ra_rouge} secteur{'s' if _ra_rouge != 1 else ''} en rouge" if _ra_rouge > 0
         else "Tous les secteurs OK", "sub_class": _ra_cls},
        {"label": "ECL CREDIT", "value": format_euro(_ecl_total),
         "sub_text": f"{'+'if _ecl_delta > 0 else ''}{_ecl_delta:.1%} vs base", "sub_class": _ecl_cls},
        {"label": "RAROC CREDIT", "value": f"{_raroc_val:.2%}",
         "sub_text": f"HHI cross : {hhi_cross.get('hhi_crosscell', 0):,.0f} | Name : {hhi_cross.get('hhi_name_credit', 0):,.0f}", "sub_class": _raroc_cls},
        {"label": "NAV DRAWDOWN PE", "value": f"{_drawdown:.1%}",
         "sub_text": f"Delta NAV : {format_euro(_delta_nav)}", "sub_class": _pe_cls},
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

    # ── SCENARIO CONTEXT BANNER ──
    _scenario_name = st.session_state.get("scenario_selector", "Manuel")
    _banner_items = [
        f"Scenario : **{_scenario_name}**",
        f"BCE : {interest_rate_bp:+.0f}bp",
        f"Chomage : {unemployment_bipolar:+.1f}pp",
        f"PIB : {gdp_pct:+.1f}%",
        f"HPI : {hpi_pct:+.1f}%",
        f"Inflation : {inflation_pct:.1f}%",
    ]
    st.markdown(
        " | ".join(_banner_items),
        help="Parametres macro du scenario courant (sidebar)",
    )

    # ── TABS (7 onglets — FR48, FR55) ──
    (tab_cro, tab_ecl, tab_pe,
     tab_asym, tab_perf, tab_explain, tab_data) = st.tabs([
        "Synthese CRO",
        "Risque Credit",
        "Private Equity",
        "Optimisation & Bilan",
        "Performance Modeles",
        "Explainabilite",
        "Donnees & Export",
    ])

    # ── TAB 5 : Performance Modeles ──
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
            st.plotly_chart(charts.plot_roc_curves(roc_data), width="stretch")
        with col2:
            comparison_df = pd_suite.get_comparison_table()
            st.plotly_chart(charts.plot_model_comparison(comparison_df), width="stretch")

        # Métriques détaillées
        render_section_title("Metriques Detaillees")
        comparison_styled = pd_suite.get_comparison_table()
        st.dataframe(
            comparison_styled,
            width="stretch",
            hide_index=True,
        )

        # Afficher les hyperparamètres XGBoost optimisés si disponibles
        if hasattr(pd_suite, '_xgb_best_params'):
            params = pd_suite._xgb_best_params
            params_str = " · ".join(
                f"**{k}** = {v}" for k, v in sorted(params.items())
            )
            st.markdown(
                f'<div style="color:{DASHBOARD_CONFIG.theme_text_muted};font-size:0.82rem;margin-top:0.5rem;">'
                f'XGBoost — Hyperparametres optimises (RandomizedSearchCV) : '
                f'{params_str}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Calibration & Backtesting", expanded=False):
            calib_predictions = {
                name: res.y_pred_test for name, res in pd_suite.results.items()
            }
            st.plotly_chart(
                charts.plot_calibration_curve(pd_suite.y_test, calib_predictions),
                width="stretch",
            )
            selected_result = pd_suite.results[selected_model]
            bt_metrics = ModelMetrics.compute_backtesting_metrics(
                pd_suite.y_test, selected_result.y_pred_test, n_folds=6,
            )
            if not bt_metrics.empty:
                st.plotly_chart(
                    charts.plot_backtesting_auc(bt_metrics),
                    width="stretch",
                )
                st.dataframe(bt_metrics, width="stretch", hide_index=True)

        with st.expander("Analyse des Features", expanded=False):
            col3, col4 = st.columns(2)
            with col3:
                feat_imp_df = pd_suite.get_feature_importance_table()
                st.plotly_chart(
                    charts.plot_feature_importance(feat_imp_df, selected_model),
                    width="stretch",
                )
            with col4:
                iv_table = pd_suite.woe_binner.get_iv_table()
                st.plotly_chart(charts.plot_iv_table(iv_table), width="stretch")

        with st.expander("Scorecard Distribution", expanded=False):
            lr_result = pd_suite.results.get("LR_WoE")
            if lr_result and getattr(lr_result, "scorecard_params", None):
                scores_test = pd_suite._pd_to_score(
                    lr_result.y_pred_test, lr_result.scorecard_params,
                )
                st.plotly_chart(
                    charts.plot_score_distribution(
                        scores_test, pd_suite.y_test, lr_result.scorecard_params,
                    ),
                    width="stretch",
                )
            else:
                st.info("Scorecard disponible uniquement pour le modele LR_WoE.")

    # ── TAB 2 : Risque Credit (FR46) ──
    with tab_ecl:
        render_section_title("Décomposition ECL")

        col5, col6 = st.columns(2)
        with col5:
            st.plotly_chart(
                charts.plot_ecl_by_segment(result_stressed),
                width="stretch",
            )
        with col6:
            st.plotly_chart(
                charts.plot_ecl_coverage_scatter(result_stressed),
                width="stretch",
            )

        # HHI Concentration
        render_section_title("Concentration du Portefeuille (HHI)")
        seg_ead = result_stressed.groupby("segment")["ead"].sum().values
        loan_ead = result_stressed.groupby("loan_type")["ead"].sum().values
        hhi_seg = ModelMetrics.hhi(seg_ead)
        hhi_loan = ModelMetrics.hhi(loan_ead)

        st.plotly_chart(
            charts.plot_hhi_gauge(hhi_seg, hhi_loan),
            width="stretch",
        )

        # Waterfall
        render_section_title("Waterfall ECL (Base vs Stressé)")
        waterfall_df = ecl_calc.compute_waterfall(
            ecl_t0=result_base["ecl_weighted"].values,
            ecl_t1=result_stressed["ecl_weighted"].values,
            stages_t0=result_base["stage"].values,
            stages_t1=result_stressed["stage"].values,
            segments=df_clients["sector"].values,
        )
        st.plotly_chart(
            charts.plot_waterfall_ecl(waterfall_df),
            width="stretch",
        )

        # Résumé ECL
        render_section_title("Résumé ECL par Segment")
        ecl_summary = ecl_calc.compute_ecl_summary(result_stressed)
        st.dataframe(ecl_summary, width="stretch", hide_index=True)

        # Green Asset Ratio (ESG placeholder)
        render_section_title("Green Asset Ratio (ESG)")
        st.caption(
            "GAR declaratif par secteur — placeholder conformite BCE 2024. "
            "Les green_share sont estimatives, non auditees."
        )
        col_gar1, col_gar2, col_gar3 = st.columns(3)
        with col_gar1:
            st.metric("GAR Credit", f"{gar_result['gar_credit']:.1%}")
        with col_gar2:
            st.metric("GAR PE", f"{gar_result['gar_pe']:.1%}")
        with col_gar3:
            st.metric("GAR Total", f"{gar_result['gar_total']:.1%}")
        st.dataframe(gar_result["details"], width="stretch", hide_index=True)

        # ── STAGING & TRANSITIONS (merged from former Tab 4) ──
        with st.expander("Staging & Transitions", expanded=True):
            staging_engine = StagingEngine()
            stage_summary = staging_engine.get_stage_summary(
                result_stressed["stage"].values,
                result_stressed["ead"].values,
            )

            col_stg1, col_stg2 = st.columns(2)
            with col_stg1:
                st.plotly_chart(
                    charts.plot_stage_distribution(stage_summary),
                    width="stretch",
                )
            with col_stg2:
                render_section_title("Matrice de Transition (Base vs Stress)")
                matrix = staging_engine.compute_transition_matrix(
                    result_base["stage"].values,
                    result_stressed["stage"].values,
                )
                st.plotly_chart(
                    charts.plot_transition_matrix(matrix),
                    width="stretch",
                )

            # Sankey migration diagram
            st.plotly_chart(
                charts.plot_stage_sankey(
                    result_base["stage"].values,
                    result_stressed["stage"].values,
                ),
                width="stretch",
            )

            render_section_title("Detail par Stage")
            st.dataframe(stage_summary, width="stretch", hide_index=True)

    # ── TAB 3 : Private Equity (FR48) ──
    with tab_pe:
        render_section_title("Portefeuille Private Equity — IFRS 13 Fair Value")

        # Risk classification stacked bar (CVD-safe patterns)
        st.plotly_chart(
            charts.plot_pe_risk_stacked_bar(result_pe),
            width="stretch",
        )

        col_pe1, col_pe2 = st.columns(2)
        with col_pe1:
            st.plotly_chart(
                charts.plot_pe_nav_by_sector(result_pe),
                width="stretch",
            )
        with col_pe2:
            st.plotly_chart(
                charts.plot_pe_risk_categories(result_pe),
                width="stretch",
            )

        render_section_title("Performance PE — MOIC & Drawdown")
        st.plotly_chart(
            charts.plot_pe_moic_drawdown(result_pe),
            width="stretch",
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
        st.dataframe(pe_summary, width="stretch", hide_index=True)

        # Parametres courants
        st.caption(
            f"Allocation PE : {pe_allocation_pct}% | "
            f"Risk Weight PE : {rw_pe_selected}% (CRR3)"
        )

    # ── TAB 4 : Optimisation & Bilan (FR55) ──
    with tab_asym:
        render_section_title("Matrice d'Asymetrie Credit vs PE")

        col_as1, col_as2 = st.columns(2)
        with col_as1:
            st.plotly_chart(
                charts.plot_asymmetry_heatmap(asymmetry_matrix),
                width="stretch",
            )
        with col_as2:
            st.plotly_chart(
                charts.plot_raroc_comparison(raroc_eva),
                width="stretch",
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
        st.dataframe(asym_display, width="stretch", hide_index=True)

        with st.expander("Sensibilite CRR3 — Impact Risk Weight PE", expanded=False):
            col_crr1, col_crr2 = st.columns([2, 1])
            with col_crr1:
                st.plotly_chart(
                    charts.plot_crr3_sensitivity(crr3_sensitivity),
                    width="stretch",
                )
            with col_crr2:
                st.dataframe(crr3_sensitivity, width="stretch", hide_index=True)

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
                st.dataframe(weights_df, width="stretch", hide_index=True)

        # ── Drill-down sectoriel (FR54) ──
        render_section_title("Drill-down Sectoriel (FR54)")
        _sector_names = asymmetry_matrix["sector"].tolist()
        _selected_sector = st.selectbox(
            "Selectionner un secteur pour le detail",
            options=_sector_names,
            key="drill_sector",
        )

        if _selected_sector and analytics_state.proportional_contributions is not None:
            col_dd1, col_dd2 = st.columns(2)

            with col_dd1:
                # Allocation proportionnelle pour ce secteur
                _alloc = analytics_state.proportional_contributions
                _alloc_sec = _alloc[_alloc["sector"] == _selected_sector]
                if len(_alloc_sec) > 0:
                    st.markdown(f"**Allocation Proportionnelle — {_selected_sector}**")
                    st.dataframe(_alloc_sec[["canal", "risk_amount", "proportional_share", "rwa"]],
                                 width="stretch", hide_index=True)

                # Factor attribution pour ce secteur
                _factors = analytics_state.factor_attribution
                if _factors is not None and len(_factors) > 0:
                    st.markdown(f"**Attribution Factorielle Macro**")
                    st.dataframe(_factors[["variable", "canal", "delta_from_base", "attribution"]],
                                 width="stretch", hide_index=True)

            with col_dd2:
                # Risk appetite pour ce secteur
                _ra = analytics_state.risk_appetite_matrix
                if _ra is not None and len(_ra) > 0:
                    _ra_sec = _ra[_ra["sector"] == _selected_sector] if "sector" in _ra.columns else _ra
                    if len(_ra_sec) > 0:
                        st.markdown(f"**Risk Appetite — {_selected_sector}**")
                        st.dataframe(_ra_sec, width="stretch", hide_index=True)

                # Trajectoires pour ce secteur
                _traj = analytics_state.trajectories
                if _traj is not None and len(_traj) > 0:
                    st.markdown(f"**Trajectoires Prospectives**")
                    _traj_display = _traj.copy()
                    if "sector" in _traj_display.columns:
                        _traj_sec = _traj_display[_traj_display["sector"] == _selected_sector]
                        if len(_traj_sec) > 0:
                            st.dataframe(_traj_sec, width="stretch", hide_index=True)
                        else:
                            st.dataframe(_traj_display, width="stretch", hide_index=True)
                    else:
                        st.dataframe(_traj_display, width="stretch", hide_index=True)

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
            else:
                feature_names = pd_suite._raw_features
                X_shap = pd_suite.X_test[feature_names].values

            # Extraire le modèle de base du CalibratedClassifierCV
            # (tous les modeles sont wrappés dans CalibratedClassifierCV)
            calibrated = model_result.model
            if hasattr(calibrated, 'calibrated_classifiers_'):
                cc = calibrated.calibrated_classifiers_[0]
                inner_model = getattr(cc, 'estimator', getattr(cc, 'base_estimator', calibrated))
            else:
                inner_model = calibrated

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
                    width="stretch",
                )
            with col_s2:
                st.plotly_chart(
                    charts.plot_shap_beeswarm(shap_vals, X_sample, feature_names),
                    width="stretch",
                )
                st.caption("Couleur : rouge = valeur feature elevee, bleu = valeur feature basse")

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
                    width="stretch",
                )

                # Tableau detaille
                client_df = pd.DataFrame({
                    "Feature": _feature_names,
                    "SHAP Value": client_shap,
                    "Feature Value": client_features,
                }).sort_values("SHAP Value", key=abs, ascending=False).head(10)
                st.dataframe(client_df, width="stretch", hide_index=True)

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
                width="stretch",
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

        # ── Export Excel 8 feuilles (FR50) ──
        render_section_title("Export Multi-Sheet (FR50)")

        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)
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
                    else:
                        _fn = pd_suite._raw_features
                        _Xs = pd_suite.X_test[_fn].values[:500]
                    # Extraire le modèle de base du CalibratedClassifierCV
                    _cal = _mr.model
                    if hasattr(_cal, 'calibrated_classifiers_'):
                        _cc = _cal.calibrated_classifiers_[0]
                        _im = getattr(_cc, 'estimator', getattr(_cc, 'base_estimator', _cal))
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

            if st.download_button(
                label="Export complet (Excel 8 feuilles)",
                data=buffer.getvalue(),
                file_name="ifrs9_cockpit_complet.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ):
                st.toast("Export Excel genere avec succes", icon=":material/check_circle:")

        with col_exp2:
            # Export rapport CRO texte
            cro_report = cro.generate_report(
                result_stressed,
                ecl_previous=ecl_base_total,
                psi_value=psi_value,
                unemployment_rate=unemployment_rate,
                gdp_growth=gdp_growth,
            )
            if st.download_button(
                label="Rapport CRO (TXT)",
                data=cro_report,
                file_name="rapport_cro.txt",
                mime="text/plain",
            ):
                st.toast("Rapport CRO exporte", icon=":material/check_circle:")

        with col_exp3:
            # Export synthese AI Analyst
            _ai_narrative = analytics_state.narrative or ""
            if st.download_button(
                label="Synthese AI Analyst (TXT)",
                data=_ai_narrative,
                file_name="ai_analyst_synthese.txt",
                mime="text/plain",
                key="export_ai_txt",
            ):
                st.toast("Synthese AI Analyst exportee", icon=":material/check_circle:")

        with col_exp4:
            # Export journal d'audit LaTeX
            _latex_content = generate_audit_latex(
                macro_params=macro_params,
                selected_model=selected_model,
                result_base=result_base,
                result_stressed=result_stressed,
                result_pe=result_pe,
                advanced_metrics=advanced_metrics,
                hhi_cross=hhi_cross,
                raroc_eva=raroc_eva,
                asymmetry_matrix=asymmetry_matrix,
                optimization=optimization,
                crr3_sensitivity=crr3_sensitivity,
                analytics_state=analytics_state,
                trans_matrix=trans_matrix,
            )
            if st.download_button(
                label="Journal d'Audit (LaTeX)",
                data=_latex_content,
                file_name="ifrs9_audit_trail.tex",
                mime="application/x-tex",
                key="export_latex",
            ):
                st.toast("Journal d'audit LaTeX exporte", icon=":material/check_circle:")

        with st.expander("Rapport CRO (regles)", expanded=False):
            st.code(cro_report, language=None)

    # ── TAB 1 : Synthese CRO — AI Analyst (FR55) ──
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
                width="stretch",
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
            st.dataframe(analytics_state.tipping_points, width="stretch", hide_index=True)

        # ── RST Distance + Scenario de rupture (FR54) ──
        if analytics_state.rst_distance > 0:
            _rst_color = DASHBOARD_CONFIG.color_danger if analytics_state.rst_distance < 1.0 else DASHBOARD_CONFIG.color_warning if analytics_state.rst_distance < 2.0 else DASHBOARD_CONFIG.color_success
            _rst = analytics_state.rst_result or {}
            _rst_custom = " (cible personnalisee)" if rst_custom_enabled else ""
            st.markdown(
                f'<div style="padding:0.5rem 1rem;border-left:3px solid {_rst_color};margin:0.5rem 0;">'
                f'<b>Distance au point de rupture (RST){_rst_custom} :</b> '
                f'<span style="color:{_rst_color};font-weight:700;">{analytics_state.rst_distance:.2f} sigma</span>'
                f'<br/><span style="color:{DASHBOARD_CONFIG.theme_text_muted};font-size:0.85rem;">'
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
                    "Baseline": f"{getattr(SCENARIO_BASE, var, 0.0):.2f}",
                    "Delta": f"{val - getattr(SCENARIO_BASE, var, 0.0):+.2f}",
                } for var, val in _rst_scen.items()])
                st.dataframe(rst_scen_df, width="stretch", hide_index=True)

        # ── Trajectoires prospectives ──
        if analytics_state.trajectories is not None and len(analytics_state.trajectories) > 0:
            render_section_title("Trajectoires Prospectives (T+3 a T+12)")
            st.plotly_chart(
                charts.plot_trajectories_chart(analytics_state.trajectories),
                width="stretch",
            )
            with st.expander("Donnees brutes trajectoires"):
                st.dataframe(analytics_state.trajectories, width="stretch", hide_index=True)

        # ── Early Warning ──
        if analytics_state.early_warning is not None and len(analytics_state.early_warning) > 0:
            render_section_title("Indicateurs d'Alerte Precoce")
            st.dataframe(analytics_state.early_warning, width="stretch", hide_index=True)

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
                    segments=df_clients["sector"].values,
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
