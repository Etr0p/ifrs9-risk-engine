"""Tests unitaires pour le dashboard Streamlit — Epic 6 (Stories 6-1 a 6-6).

Couvre les 6 stories :
  6-1: Sidebar, sliders et scenarios predefinis (FR33, FR34, FR36)
  6-2: Orchestration pipeline et progression (FR52)
  6-3: KPI cards, insight box, classifications (FR45, FR46)
  6-4: Design system Steel Blue et 8 onglets (FR48)
  6-5: SHAP individuel et exports Excel (FR49, FR50)
  6-6: RST personnalise et drill-down (FR54, FR55)

Strategie : tester la configuration, les fonctions exportees et la coherence
du design system SANS lancer Streamlit (mock de st pour eviter les side effects).
"""

from __future__ import annotations

import importlib
import inspect
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ifrs9_cockpit.config import (
    CHART_COLORS,
    DASHBOARD_CONFIG,
    DashboardConfig,
    MACRO_INCOHERENCE_RULES,
    PREDEFINED_SCENARIOS,
    SCENARIO_BASE,
    STAGE_COLORS,
)


# ──────────────────────────────────────────────
# STORY 6-1 : Config sliders & scenarios
# ──────────────────────────────────────────────


class TestStory61SidebarSlidersScenarios:
    """Tests pour la sidebar, les sliders et les scenarios predefinis."""

    def test_dashboard_config_exists(self) -> None:
        """DASHBOARD_CONFIG est une instance de DashboardConfig."""
        assert isinstance(DASHBOARD_CONFIG, DashboardConfig)

    def test_dashboard_config_frozen(self) -> None:
        """DashboardConfig est immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            DASHBOARD_CONFIG.page_title = "modified"  # type: ignore[misc]

    def test_slider_interest_rate_range(self) -> None:
        """Range taux BCE : (min, max, step) avec min < 0 < max."""
        rng = DASHBOARD_CONFIG.stress_interest_rate_range
        assert len(rng) == 3
        assert rng[0] < 0.0 < rng[1]
        assert rng[2] > 0.0

    def test_slider_unemployment_range_bipolar(self) -> None:
        """Range chomage bipolaire : min negatif, max positif, symetrique."""
        rng = DASHBOARD_CONFIG.stress_unemployment_range
        assert len(rng) == 3
        assert rng[0] < 0.0
        assert rng[1] > 0.0
        assert abs(rng[0]) == abs(rng[1])  # symetrique

    def test_slider_gdp_range(self) -> None:
        """Range PIB : (min, max, step)."""
        rng = DASHBOARD_CONFIG.stress_gdp_range
        assert len(rng) == 3
        assert rng[0] < 0.0 < rng[1]

    def test_slider_hpi_range(self) -> None:
        """Range HPI : (min, max, step)."""
        rng = DASHBOARD_CONFIG.stress_hpi_range
        assert len(rng) == 3
        assert rng[0] < 0.0
        assert rng[1] > 0.0

    def test_slider_inflation_range(self) -> None:
        """Range inflation : (min, max, step)."""
        rng = DASHBOARD_CONFIG.stress_inflation_range
        assert len(rng) == 3
        assert rng[2] > 0.0

    def test_predefined_scenarios_non_empty(self) -> None:
        """Au moins 3 scenarios predefinis (FR34)."""
        assert len(PREDEFINED_SCENARIOS) >= 3

    def test_predefined_scenario_keys(self) -> None:
        """Chaque scenario contient les 5 variables macro attendues."""
        expected_keys = {
            "interest_rate_bp",
            "unemployment_bipolar",
            "gdp_pct",
            "hpi_pct",
            "inflation_pct",
        }
        for name, scenario in PREDEFINED_SCENARIOS.items():
            assert expected_keys.issubset(
                set(scenario.keys())
            ), f"Scenario '{name}' manque des cles : {expected_keys - set(scenario.keys())}"

    def test_macro_incoherence_rules_non_empty(self) -> None:
        """Au moins 1 regle d'incoherence macro (FR36)."""
        assert len(MACRO_INCOHERENCE_RULES) >= 1

    def test_macro_incoherence_rules_have_conditions(self) -> None:
        """Chaque regle a des conditions et une description."""
        for rule in MACRO_INCOHERENCE_RULES:
            assert hasattr(rule, "conditions")
            assert hasattr(rule, "description")
            assert len(rule.conditions) >= 1
            assert len(rule.description) > 0

    def test_scenario_base_exists(self) -> None:
        """SCENARIO_BASE fournit les valeurs de reference pour conversion."""
        assert hasattr(SCENARIO_BASE, "interest_rate")
        assert hasattr(SCENARIO_BASE, "unemployment_rate")
        assert hasattr(SCENARIO_BASE, "gdp_growth")
        assert hasattr(SCENARIO_BASE, "hpi_growth")
        assert hasattr(SCENARIO_BASE, "inflation_rate")

    def test_slider_conversion_unemployment_bipolar(self) -> None:
        """Conversion chomage bipolaire : 0 = base, negatif = hausse chomage."""
        # Logique : unemployment_rate = SCENARIO_BASE.unemployment_rate - bipolar
        # bipolar = -3 => unemployment = 7.5 - (-3) = 10.5 (hausse)
        # bipolar = +2 => unemployment = 7.5 - 2 = 5.5 (baisse)
        base = SCENARIO_BASE.unemployment_rate
        bipolar_neg = -3.0
        result = base - bipolar_neg
        assert result > base, "Bipolar negatif doit augmenter le chomage"
        bipolar_pos = 2.0
        result_pos = base - bipolar_pos
        assert result_pos < base, "Bipolar positif doit baisser le chomage"

    def test_slider_conversion_interest_rate_bp(self) -> None:
        """Conversion taux BCE : bp/100 ajoute a la base."""
        base = SCENARIO_BASE.interest_rate
        bp = 100.0
        result = base + bp / 100.0
        assert result == base + 1.0, "100bp = +1% en taux"

    def test_predefined_scenario_values_within_slider_ranges(self) -> None:
        """Les valeurs des scenarios predefinis sont dans les ranges des sliders."""
        for name, scenario in PREDEFINED_SCENARIOS.items():
            ir = scenario["interest_rate_bp"]
            rng = DASHBOARD_CONFIG.stress_interest_rate_range
            assert rng[0] <= ir <= rng[1], (
                f"Scenario '{name}' interest_rate_bp={ir} hors range [{rng[0]}, {rng[1]}]"
            )
            gdp = scenario["gdp_pct"]
            rng_g = DASHBOARD_CONFIG.stress_gdp_range
            assert rng_g[0] <= gdp <= rng_g[1], (
                f"Scenario '{name}' gdp_pct={gdp} hors range [{rng_g[0]}, {rng_g[1]}]"
            )

    def test_slider_default_values_use_scenario_base(self) -> None:
        """Les valeurs par defaut des sliders sont coherentes avec SCENARIO_BASE."""
        from pathlib import Path
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        # Verifier que GDP utilise SCENARIO_BASE.gdp_growth
        assert "value=SCENARIO_BASE.gdp_growth" in source, \
            "Slider GDP doit utiliser SCENARIO_BASE.gdp_growth"
        assert "value=SCENARIO_BASE.hpi_growth" in source, \
            "Slider HPI doit utiliser SCENARIO_BASE.hpi_growth"
        assert "value=SCENARIO_BASE.inflation_rate" in source, \
            "Slider inflation doit utiliser SCENARIO_BASE.inflation_rate"


# ──────────────────────────────────────────────
# STORY 6-4 : Design system Steel Blue & palette
# ──────────────────────────────────────────────


class TestStory64DesignSystemSteelBlue:
    """Tests pour le theme Steel Blue et les couleurs du design system."""

    def test_theme_primary_is_steel_blue(self) -> None:
        """Couleur primaire = Steel Blue #3B82F6."""
        assert DASHBOARD_CONFIG.theme_primary == "#3B82F6"

    def test_theme_secondary_steel_blue_dark(self) -> None:
        """Couleur secondaire = Steel Blue 800 #1E40AF."""
        assert DASHBOARD_CONFIG.theme_secondary == "#1E40AF"

    def test_theme_accent_emerald(self) -> None:
        """Couleur accent = Emerald #10B981."""
        assert DASHBOARD_CONFIG.theme_accent == "#10B981"

    def test_theme_bg_dark(self) -> None:
        """Fond sombre = #0C1222."""
        assert DASHBOARD_CONFIG.theme_bg_dark == "#0C1222"

    def test_theme_bg_card(self) -> None:
        """Fond de carte = #1E293B."""
        assert DASHBOARD_CONFIG.theme_bg_card == "#1E293B"

    def test_theme_text_light(self) -> None:
        """Texte principal = #F8FAFC (quasi-blanc)."""
        assert DASHBOARD_CONFIG.theme_text == "#F8FAFC"

    def test_theme_text_muted(self) -> None:
        """Texte secondaire = #94A3B8 (Slate 400)."""
        assert DASHBOARD_CONFIG.theme_text_muted == "#94A3B8"

    def test_semantic_colors(self) -> None:
        """Couleurs semantiques : success, warning, danger, info."""
        assert DASHBOARD_CONFIG.color_success == "#10B981"
        assert DASHBOARD_CONFIG.color_warning == "#F59E0B"
        assert DASHBOARD_CONFIG.color_danger == "#EF4444"
        assert DASHBOARD_CONFIG.color_info == "#06B6D4"

    def test_chart_colors_tuple_length(self) -> None:
        """CHART_COLORS contient 6 couleurs hex."""
        assert len(CHART_COLORS) == 6
        for c in CHART_COLORS:
            assert c.startswith("#")
            assert len(c) == 7

    def test_chart_colors_primary_first(self) -> None:
        """La premiere couleur du chart est Steel Blue primary."""
        assert CHART_COLORS[0] == DASHBOARD_CONFIG.theme_primary

    def test_stage_colors_mapping(self) -> None:
        """STAGE_COLORS couvre stages 1, 2, 3 avec couleurs semantiques."""
        assert set(STAGE_COLORS.keys()) == {1, 2, 3}
        assert STAGE_COLORS[1] == "#10B981"  # Emerald (performing)
        assert STAGE_COLORS[2] == "#F59E0B"  # Amber (watchlist)
        assert STAGE_COLORS[3] == "#EF4444"  # Red (default)

    def test_page_config_values(self) -> None:
        """Page config : titre, icone, layout wide."""
        assert DASHBOARD_CONFIG.page_title == "IFRS 9 Risk Cockpit"
        assert DASHBOARD_CONFIG.layout == "wide"

    def test_all_theme_colors_are_hex(self) -> None:
        """Toutes les couleurs du theme sont des hex valides (#RRGGBB)."""
        color_attrs = [
            "theme_primary", "theme_secondary", "theme_accent",
            "theme_bg_dark", "theme_bg_card", "theme_text", "theme_text_muted",
            "color_success", "color_warning", "color_danger", "color_info",
        ]
        for attr in color_attrs:
            val = getattr(DASHBOARD_CONFIG, attr)
            assert isinstance(val, str), f"{attr} n'est pas une str"
            assert val.startswith("#"), f"{attr}={val} ne commence pas par #"
            assert len(val) == 7, f"{attr}={val} n'a pas 7 caracteres"


# ──────────────────────────────────────────────
# STORY 6-4 : 8 onglets
# ──────────────────────────────────────────────


class TestStory64EightTabs:
    """Tests pour la structure a 8 onglets du dashboard."""

    TAB_NAMES = [
        "Performance Modeles",
        "Analyse ECL",
        "Analyse PE",
        "Staging & Transitions",
        "Asymetries & Optimisation",
        "Explainabilite",
        "Donnees & Export",
        "Analyse CRO",
    ]

    def test_app_defines_8_tabs(self) -> None:
        """app.py definit exactement 8 noms d'onglets via st.tabs()."""
        import ast
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        # Chercher le pattern st.tabs([...]) dans le source
        assert "st.tabs([" in source, "st.tabs([...]) introuvable dans app.py"
        # Compter les noms d'onglets dans l'appel st.tabs
        idx_start = source.index("st.tabs([")
        idx_end = source.index("])", idx_start) + 2
        tabs_call = source[idx_start:idx_end]
        # Compter les strings entre guillemets
        tab_count = tabs_call.count('"') // 2
        assert tab_count == 8, f"Attendu 8 onglets, trouve {tab_count}"

    def test_tab_names_present_in_source(self) -> None:
        """Chaque nom d'onglet est present dans le code source de app.py."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        for name in self.TAB_NAMES:
            assert name in source, f"Onglet '{name}' introuvable dans app.py"


# ──────────────────────────────────────────────
# Charts module : fonctions publiques
# ──────────────────────────────────────────────


class TestChartsModule:
    """Tests pour le module dashboard/charts.py."""

    def test_charts_module_imports(self) -> None:
        """Le module charts s'importe sans erreur."""
        from ifrs9_cockpit.dashboard import charts
        assert charts is not None

    def test_charts_public_functions_exist(self) -> None:
        """Toutes les fonctions publiques de charts sont callables."""
        from ifrs9_cockpit.dashboard import charts

        expected_functions = [
            "plot_roc_curves",
            "plot_feature_importance",
            "plot_stage_distribution",
            "plot_ecl_by_segment",
            "plot_transition_matrix",
            "plot_waterfall_ecl",
            "plot_ecl_coverage_scatter",
            "plot_iv_table",
            "plot_model_comparison",
            "plot_shap_summary",
            "plot_shap_beeswarm",
            "plot_calibration_curve",
            "plot_hhi_gauge",
            "plot_backtesting_auc",
            "plot_pe_nav_by_sector",
            "plot_pe_risk_categories",
            "plot_pe_moic_drawdown",
            "plot_asymmetry_heatmap",
            "plot_raroc_comparison",
            "plot_crr3_sensitivity",
            "plot_risk_appetite_matrix",
            "plot_shap_force_individual",
        ]
        for func_name in expected_functions:
            func = getattr(charts, func_name, None)
            assert func is not None, f"charts.{func_name} introuvable"
            assert callable(func), f"charts.{func_name} n'est pas callable"

    def test_charts_use_dashboard_config_palette(self) -> None:
        """Les constantes de palette dans charts referencent DASHBOARD_CONFIG."""
        from ifrs9_cockpit.dashboard import charts

        # Verifier que _PRIMARY == DASHBOARD_CONFIG.theme_primary
        assert charts._PRIMARY == DASHBOARD_CONFIG.theme_primary
        assert charts._SECONDARY == DASHBOARD_CONFIG.theme_secondary
        assert charts._ACCENT == DASHBOARD_CONFIG.theme_accent
        assert charts._BG == DASHBOARD_CONFIG.theme_bg_dark
        assert charts._TEXT == DASHBOARD_CONFIG.theme_text

    def test_base_layout_returns_dict(self) -> None:
        """_base_layout() retourne un dict de configuration Plotly."""
        from ifrs9_cockpit.dashboard.charts import _base_layout

        layout = _base_layout("Test Title", 500)
        assert isinstance(layout, dict)
        assert "title" in layout or "plot_bgcolor" in layout

    def test_plot_roc_curves_returns_figure(self) -> None:
        """plot_roc_curves() retourne un go.Figure avec des donnees valides."""
        from ifrs9_cockpit.dashboard.charts import plot_roc_curves

        roc_data = {
            "LR_WoE": (
                np.array([0.0, 0.5, 1.0]),
                np.array([0.0, 0.8, 1.0]),
                0.85,
            ),
        }
        fig = plot_roc_curves(roc_data)
        assert isinstance(fig, go.Figure)

    def test_plot_stage_distribution_returns_figure(self) -> None:
        """plot_stage_distribution() retourne un go.Figure."""
        from ifrs9_cockpit.dashboard.charts import plot_stage_distribution

        stage_df = pd.DataFrame({
            "stage": [1, 2, 3],
            "count": [7000, 2000, 1000],
            "total_ead": [5e8, 2e8, 1e8],
        })
        fig = plot_stage_distribution(stage_df)
        assert isinstance(fig, go.Figure)

    def test_plot_hhi_gauge_returns_figure(self) -> None:
        """plot_hhi_gauge() retourne un go.Figure."""
        from ifrs9_cockpit.dashboard.charts import plot_hhi_gauge

        fig = plot_hhi_gauge(0.25, 0.002)
        assert isinstance(fig, go.Figure)

    def test_plot_transition_matrix_returns_figure(self) -> None:
        """plot_transition_matrix() retourne un go.Figure."""
        from ifrs9_cockpit.dashboard.charts import plot_transition_matrix

        matrix_df = pd.DataFrame(
            [[0.9, 0.08, 0.02], [0.05, 0.85, 0.10], [0.01, 0.04, 0.95]],
            index=["Stage 1", "Stage 2", "Stage 3"],
            columns=["Stage 1", "Stage 2", "Stage 3"],
        )
        fig = plot_transition_matrix(matrix_df)
        assert isinstance(fig, go.Figure)

    def test_plot_pe_nav_by_sector_returns_figure(self) -> None:
        """plot_pe_nav_by_sector() retourne un go.Figure (Story 6-4)."""
        from ifrs9_cockpit.dashboard.charts import plot_pe_nav_by_sector

        result_pe = pd.DataFrame({
            "sector": ["Technologie", "Industrie", "Sante"],
            "nav": [1e7, 2e7, 1.5e7],
            "capital_invested": [8e6, 1.5e7, 1.2e7],
            "expected_loss_pe": [5e5, 1e6, 3e5],
        })
        fig = plot_pe_nav_by_sector(result_pe)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # NAV, Capital, EL

    def test_plot_pe_risk_categories_returns_figure(self) -> None:
        """plot_pe_risk_categories() retourne un go.Figure (Story 6-4)."""
        from ifrs9_cockpit.dashboard.charts import plot_pe_risk_categories

        result_pe = pd.DataFrame({
            "risk_category": ["Performing", "Performing", "Watchlist", "Distressed"],
        })
        fig = plot_pe_risk_categories(result_pe)
        assert isinstance(fig, go.Figure)

    def test_plot_pe_moic_drawdown_returns_figure(self) -> None:
        """plot_pe_moic_drawdown() retourne un go.Figure (Story 6-4)."""
        from ifrs9_cockpit.dashboard.charts import plot_pe_moic_drawdown

        result_pe = pd.DataFrame({
            "sector": ["Technologie", "Industrie", "Sante"],
            "moic": [1.5, 1.2, 1.8],
            "nav_drawdown": [0.05, 0.12, 0.03],
        })
        fig = plot_pe_moic_drawdown(result_pe)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Bars + Scatter

    def test_plot_asymmetry_heatmap_returns_figure(self) -> None:
        """plot_asymmetry_heatmap() retourne un go.Figure (Story 6-6)."""
        from ifrs9_cockpit.dashboard.charts import plot_asymmetry_heatmap

        asym_df = pd.DataFrame({
            "sector": ["Technologie", "Industrie"],
            "loss_ratio": [1.5, 0.8],
            "rwa_ratio": [2.0, 1.1],
            "raroc_delta": [-0.03, 0.01],
        })
        fig = plot_asymmetry_heatmap(asym_df)
        assert isinstance(fig, go.Figure)

    def test_plot_raroc_comparison_returns_figure(self) -> None:
        """plot_raroc_comparison() retourne un go.Figure (Story 6-6)."""
        from ifrs9_cockpit.dashboard.charts import plot_raroc_comparison

        raroc_df = pd.DataFrame({
            "sector": ["Technologie", "Industrie", "Technologie", "Industrie"],
            "canal": ["Credit", "Credit", "PE", "PE"],
            "raroc": [0.12, 0.15, 0.08, 0.10],
        })
        fig = plot_raroc_comparison(raroc_df)
        assert isinstance(fig, go.Figure)

    def test_plot_crr3_sensitivity_returns_figure(self) -> None:
        """plot_crr3_sensitivity() retourne un go.Figure (Story 6-6)."""
        from ifrs9_cockpit.dashboard.charts import plot_crr3_sensitivity

        crr3_df = pd.DataFrame({
            "rw_pe": [190, 250, 400],
            "cet1_ratio": [0.14, 0.12, 0.09],
            "feasible": [True, True, False],
        })
        fig = plot_crr3_sensitivity(crr3_df)
        assert isinstance(fig, go.Figure)

    def test_plot_risk_appetite_matrix_returns_figure(self) -> None:
        """plot_risk_appetite_matrix() retourne un go.Figure avec donnees."""
        from ifrs9_cockpit.dashboard.charts import plot_risk_appetite_matrix

        ra_df = pd.DataFrame({
            "sector": ["Technologie", "Technologie", "Industrie", "Industrie"],
            "canal": ["Credit", "PE", "Credit", "PE"],
            "signal": ["vert", "ambre", "rouge", "vert"],
        })
        fig = plot_risk_appetite_matrix(ra_df)
        assert isinstance(fig, go.Figure)

    def test_plot_risk_appetite_matrix_empty_returns_figure(self) -> None:
        """plot_risk_appetite_matrix() gere un DataFrame vide."""
        from ifrs9_cockpit.dashboard.charts import plot_risk_appetite_matrix

        fig = plot_risk_appetite_matrix(pd.DataFrame())
        assert isinstance(fig, go.Figure)

    def test_plot_shap_force_individual_returns_figure(self) -> None:
        """plot_shap_force_individual() retourne un go.Figure (Story 6-5)."""
        from ifrs9_cockpit.dashboard.charts import plot_shap_force_individual

        shap_vals = np.array([0.2, -0.1, 0.05, -0.3, 0.15])
        feat_vals = np.array([750.0, 3.0, 50000.0, 0.35, 120000.0])
        feat_names = ["credit_score", "dpd", "income", "debt_ratio", "loan_amount"]
        fig = plot_shap_force_individual(shap_vals, feat_vals, feat_names)
        assert isinstance(fig, go.Figure)

    def test_plot_shap_force_individual_dimension_mismatch(self) -> None:
        """plot_shap_force_individual() leve ValueError si dimensions incoherentes."""
        from ifrs9_cockpit.dashboard.charts import plot_shap_force_individual

        shap_vals = np.array([0.2, -0.1, 0.05])
        feat_vals = np.array([750.0, 3.0])  # Trop court
        feat_names = ["credit_score", "dpd", "income"]
        with pytest.raises(ValueError, match="Dimensions incoherentes"):
            plot_shap_force_individual(shap_vals, feat_vals, feat_names)

    def test_hex_to_rgba_conversion(self) -> None:
        """_hex_to_rgba() convertit correctement les couleurs hex."""
        from ifrs9_cockpit.dashboard.charts import _hex_to_rgba

        result = _hex_to_rgba("#3B82F6", 0.5)
        assert result == "(59, 130, 246, 0.5)"

    def test_prettify_feature_with_woe_suffix(self) -> None:
        """_prettify_feature() supprime le suffixe _woe avant le lookup."""
        from ifrs9_cockpit.dashboard.charts import _prettify_feature

        assert _prettify_feature("credit_score_woe") == "Score Credit"
        assert _prettify_feature("credit_score") == "Score Credit"
        assert _prettify_feature("unknown_feature") == "unknown_feature"


# ──────────────────────────────────────────────
# Components module : fonctions publiques
# ──────────────────────────────────────────────


class TestComponentsModule:
    """Tests pour le module dashboard/components.py (mocke Streamlit)."""

    def test_components_module_imports(self) -> None:
        """Le module components s'importe sans erreur."""
        from ifrs9_cockpit.dashboard import components
        assert components is not None

    def test_components_public_functions_exist(self) -> None:
        """Toutes les fonctions publiques de components sont callables."""
        from ifrs9_cockpit.dashboard import components

        expected_functions = [
            "render_header",
            "render_kpi_cards",
            "render_insight_box",
            "render_stage_badges",
            "render_smart_insight_box",
            "render_section_title",
            "render_kpi_row",
            "render_classification_row",
            "render_ai_narrative_box",
        ]
        for func_name in expected_functions:
            func = getattr(components, func_name, None)
            assert func is not None, f"components.{func_name} introuvable"
            assert callable(func), f"components.{func_name} n'est pas callable"

    def test_render_header_signature(self) -> None:
        """render_header() ne prend aucun argument requis."""
        from ifrs9_cockpit.dashboard.components import render_header

        sig = inspect.signature(render_header)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
        ]
        assert len(required) == 0

    def test_render_kpi_cards_signature(self) -> None:
        """render_kpi_cards() attend 4 arguments numeriques + 1 optionnel."""
        from ifrs9_cockpit.dashboard.components import render_kpi_cards

        sig = inspect.signature(render_kpi_cards)
        params = list(sig.parameters.values())
        required = [
            p for p in params if p.default is inspect.Parameter.empty
        ]
        assert len(required) >= 4, "render_kpi_cards doit avoir au moins 4 args requis"

    @patch("ifrs9_cockpit.dashboard.components.st")
    def test_render_kpi_row_outputs_html(self, mock_st: MagicMock) -> None:
        """render_kpi_row() genere du HTML avec kpi-container (FR45)."""
        from ifrs9_cockpit.dashboard.components import render_kpi_row

        cards = [
            {"label": "ECL", "value": "1.2M EUR", "sub_text": "+5%", "sub_class": "negative"},
            {"label": "NAV", "value": "10%", "sub_text": "OK", "sub_class": "positive"},
        ]
        render_kpi_row(cards)
        mock_st.markdown.assert_called_once()
        html = mock_st.markdown.call_args[0][0]
        assert "kpi-container" in html
        assert "ECL" in html
        assert "1.2M EUR" in html
        assert "negative" in html

    @patch("ifrs9_cockpit.dashboard.components.st")
    def test_render_classification_row_pe_denominator(self, mock_st: MagicMock) -> None:
        """render_classification_row() utilise le total PE pour les pctages PE (FR47)."""
        from ifrs9_cockpit.dashboard.components import render_classification_row

        stage_counts = {1: 7000, 2: 2000, 3: 1000}
        pe_categories = {"Performing": 8, "Watchlist": 5, "Distressed": 2}
        render_classification_row(stage_counts, pe_categories, 10000)
        html = mock_st.markdown.call_args[0][0]
        # Performing = 8/15 = 53.3%, PAS 8/10000 = 0.1%
        assert "53.3%" in html, "PE pctage doit utiliser le total PE, pas n_total credit"

    @patch("ifrs9_cockpit.dashboard.components.st")
    def test_render_classification_row_empty_pe(self, mock_st: MagicMock) -> None:
        """render_classification_row() gere le cas PE vide sans division par zero."""
        from ifrs9_cockpit.dashboard.components import render_classification_row

        stage_counts = {1: 900, 2: 80, 3: 20}
        pe_categories = {}
        render_classification_row(stage_counts, pe_categories, 1000)
        html = mock_st.markdown.call_args[0][0]
        # Pas de crash — les PE badges affichent 0 (0.0%)
        assert "Performing: 0" in html

    @patch("ifrs9_cockpit.dashboard.components.st")
    def test_render_ai_narrative_box_risk_level_from_status(self, mock_st: MagicMock) -> None:
        """render_ai_narrative_box() determine le niveau depuis risk_appetite_status."""
        from ifrs9_cockpit.dashboard.components import render_ai_narrative_box
        from ifrs9_cockpit.ai_analyst.types import Recommendation

        rec = Recommendation(
            action="Reduire exposition",
            regime="crise",
            trigger="ECL > seuil",
            euler_driver="Credit_Technologie",
            macro_factor="unemployment_rate",
            risk_appetite_status="rouge",
            rst_distance=0.5,
            confidence="high",  # high confidence mais rouge = ELEVE
        )
        render_ai_narrative_box(narrative="Test narrative", recommendations=[rec])
        html = mock_st.markdown.call_args[0][0]
        assert "ELEVE" in html, "risk_appetite_status=rouge doit donner niveau ELEVE"

    @patch("ifrs9_cockpit.dashboard.components.st")
    def test_render_header_contains_cockpit_header(self, mock_st: MagicMock) -> None:
        """render_header() genere le header cockpit."""
        from ifrs9_cockpit.dashboard.components import render_header
        render_header()
        html = mock_st.markdown.call_args[0][0]
        assert "cockpit-header" in html
        assert "IFRS 9 Risk Cockpit" in html

    @patch("ifrs9_cockpit.dashboard.components.st")
    def test_render_section_title_output(self, mock_st: MagicMock) -> None:
        """render_section_title() genere un div section-title."""
        from ifrs9_cockpit.dashboard.components import render_section_title
        render_section_title("Mon Titre")
        html = mock_st.markdown.call_args[0][0]
        assert "section-title" in html
        assert "Mon Titre" in html


# ──────────────────────────────────────────────
# Styles module
# ──────────────────────────────────────────────


class TestStylesModule:
    """Tests pour le module dashboard/styles.py."""

    def test_styles_module_imports(self) -> None:
        """Le module styles s'importe sans erreur."""
        from ifrs9_cockpit.dashboard import styles
        assert styles is not None

    def test_get_main_css_returns_string(self) -> None:
        """get_main_css() retourne un string CSS valide."""
        from ifrs9_cockpit.dashboard.styles import get_main_css

        css = get_main_css()
        assert isinstance(css, str)
        assert "<style>" in css
        assert "</style>" in css

    def test_css_contains_steel_blue_primary(self) -> None:
        """Le CSS contient la couleur Steel Blue primaire."""
        from ifrs9_cockpit.dashboard.styles import get_main_css

        css = get_main_css()
        assert DASHBOARD_CONFIG.theme_primary in css

    def test_css_contains_dark_background(self) -> None:
        """Le CSS contient le fond sombre."""
        from ifrs9_cockpit.dashboard.styles import get_main_css

        css = get_main_css()
        assert DASHBOARD_CONFIG.theme_bg_dark in css

    def test_css_contains_kpi_card_styles(self) -> None:
        """Le CSS definit les styles pour les KPI cards."""
        from ifrs9_cockpit.dashboard.styles import get_main_css

        css = get_main_css()
        assert ".kpi-card" in css
        assert ".kpi-value" in css
        assert ".kpi-label" in css

    def test_css_contains_insight_box_styles(self) -> None:
        """Le CSS definit les styles pour l'insight box CRO."""
        from ifrs9_cockpit.dashboard.styles import get_main_css

        css = get_main_css()
        assert ".insight-box" in css

    def test_css_contains_badge_styles(self) -> None:
        """Le CSS definit les styles pour les badges de stage."""
        from ifrs9_cockpit.dashboard.styles import get_main_css

        css = get_main_css()
        assert ".badge-stage1" in css
        assert ".badge-stage2" in css
        assert ".badge-stage3" in css

    def test_css_contains_sidebar_styles(self) -> None:
        """Le CSS definit les styles pour la sidebar."""
        from ifrs9_cockpit.dashboard.styles import get_main_css

        css = get_main_css()
        assert "stSidebar" in css

    def test_css_contains_tab_styles(self) -> None:
        """Le CSS definit les styles pour les onglets."""
        from ifrs9_cockpit.dashboard.styles import get_main_css

        css = get_main_css()
        assert ".stTabs" in css

    def test_css_custom_config(self) -> None:
        """get_main_css() accepte une config personnalisee."""
        from ifrs9_cockpit.dashboard.styles import get_main_css

        custom = DashboardConfig(theme_primary="#FF0000")
        css = get_main_css(cfg=custom)
        assert "#FF0000" in css


# ──────────────────────────────────────────────
# STORY 6-2 : Pipeline et orchestration (app.py)
# ──────────────────────────────────────────────


class TestStory62PipelineOrchestration:
    """Tests pour l'orchestration du pipeline dans app.py."""

    def test_app_module_source_parseable(self) -> None:
        """app.py est un fichier Python syntaxiquement valide."""
        import ast
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert tree is not None

    def test_app_defines_load_data(self) -> None:
        """app.py definit la fonction load_data()."""
        import ast
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        func_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        assert "load_data" in func_names

    def test_app_defines_train_pd_models(self) -> None:
        """app.py definit la fonction train_pd_models()."""
        import ast
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        func_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        assert "train_pd_models" in func_names

    def test_app_imports_progress_bar(self) -> None:
        """app.py utilise st.progress pour la barre de progression (FR52)."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert "st.progress" in source, "st.progress introuvable (FR52)"

    def test_app_uses_cache_decorators(self) -> None:
        """app.py utilise st.cache_data et st.cache_resource."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert "st.cache_data" in source
        assert "st.cache_resource" in source

    def test_app_pipeline_5_stages(self) -> None:
        """app.py contient les 5 etapes du pipeline avec progression (FR52)."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert "[1/5]" in source, "Etape 1/5 manquante"
        assert "[2/5]" in source, "Etape 2/5 manquante"
        assert "[3/5]" in source, "Etape 3/5 manquante"
        assert "[4/5]" in source, "Etape 4/5 manquante"
        assert "[5/5]" in source, "Etape 5/5 manquante"

    def test_app_progress_cleanup(self) -> None:
        """app.py nettoie la barre de progression apres pipeline (empty)."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert "_progress.empty()" in source, "Nettoyage barre de progression manquant"

    def test_app_imports_all_pipeline_modules(self) -> None:
        """app.py importe les modules critiques du pipeline."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert "PECalculator" in source, "Import PECalculator manquant"
        assert "PortfolioComparator" in source, "Import PortfolioComparator manquant"
        assert "CROAnalyst" in source, "Import CROAnalyst manquant"
        assert "ECLCalculator" in source, "Import ECLCalculator manquant"


# ──────────────────────────────────────────────
# STORY 6-3 : KPI cards & classifications
# ──────────────────────────────────────────────


class TestStory63KpiCardsInsightBox:
    """Tests pour les KPI cards et l'insight box."""

    def test_render_kpi_row_exists(self) -> None:
        """render_kpi_row est importe dans l'espace de noms du dashboard."""
        from ifrs9_cockpit.dashboard.components import render_kpi_row
        assert callable(render_kpi_row)

    def test_render_classification_row_exists(self) -> None:
        """render_classification_row est importe."""
        from ifrs9_cockpit.dashboard.components import render_classification_row
        assert callable(render_classification_row)

    def test_render_ai_narrative_box_exists(self) -> None:
        """render_ai_narrative_box est importe."""
        from ifrs9_cockpit.dashboard.components import render_ai_narrative_box
        assert callable(render_ai_narrative_box)

    def test_render_smart_insight_box_exists(self) -> None:
        """render_smart_insight_box est importe."""
        from ifrs9_cockpit.dashboard.components import render_smart_insight_box
        assert callable(render_smart_insight_box)


# ──────────────────────────────────────────────
# STORY 6-5 : SHAP et export
# ──────────────────────────────────────────────


class TestStory65ShapAndExport:
    """Tests pour SHAP individuel et l'export Excel."""

    def test_shap_force_individual_chart_exists(self) -> None:
        """charts.plot_shap_force_individual est callable."""
        from ifrs9_cockpit.dashboard.charts import plot_shap_force_individual
        assert callable(plot_shap_force_individual)

    def test_app_has_excel_export(self) -> None:
        """app.py contient la logique d'export Excel (FR50)."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert "ExcelWriter" in source or "to_excel" in source, \
            "Export Excel (FR50) introuvable dans app.py"

    def test_app_has_shap_fragment(self) -> None:
        """app.py utilise @st.fragment pour SHAP individuel (FR49)."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert "st.fragment" in source, "@st.fragment introuvable (FR49)"

    def test_app_has_8_sheet_excel_export(self) -> None:
        """app.py contient l'export Excel avec 8 feuilles (FR50)."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        # Verifier la presence des 8 noms de feuilles
        expected_sheets = [
            "1_Macro", "2_Transitions", "3_Alertes", "4_SHAP_Top10",
            "5_Credit", "6_PE", "7_Optimisation", "8_Asymetrie",
        ]
        for sheet in expected_sheets:
            assert sheet in source, f"Feuille Excel '{sheet}' introuvable (FR50)"

    def test_app_excel_export_comment_says_8(self) -> None:
        """Le commentaire de l'export dit '8 feuilles' (pas 7)."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert "8 feuilles" in source, "Le commentaire doit dire '8 feuilles'"


# ──────────────────────────────────────────────
# STORY 6-6 : RST personnalise
# ──────────────────────────────────────────────


class TestStory66RstDrillDown:
    """Tests pour le RST personnalise et le drill-down."""

    def test_app_has_rst_custom_section(self) -> None:
        """app.py contient une section Reverse Stress Test (FR54)."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert "Reverse Stress Test" in source or "RST" in source

    def test_app_has_drill_down_logic(self) -> None:
        """app.py contient une logique drill-down matrice (FR55)."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        # Le drill-down est dans le tab Asymetries
        assert "drill" in source.lower() or "detail" in source.lower() or "selecteur" in source.lower()

    def test_plot_asymmetry_heatmap_exists(self) -> None:
        """charts.plot_asymmetry_heatmap est callable."""
        from ifrs9_cockpit.dashboard.charts import plot_asymmetry_heatmap
        assert callable(plot_asymmetry_heatmap)

    def test_plot_raroc_comparison_exists(self) -> None:
        """charts.plot_raroc_comparison est callable."""
        from ifrs9_cockpit.dashboard.charts import plot_raroc_comparison
        assert callable(plot_raroc_comparison)

    def test_plot_crr3_sensitivity_exists(self) -> None:
        """charts.plot_crr3_sensitivity est callable."""
        from ifrs9_cockpit.dashboard.charts import plot_crr3_sensitivity
        assert callable(plot_crr3_sensitivity)

    def test_plot_risk_appetite_matrix_exists(self) -> None:
        """charts.plot_risk_appetite_matrix est callable."""
        from ifrs9_cockpit.dashboard.charts import plot_risk_appetite_matrix
        assert callable(plot_risk_appetite_matrix)

    def test_app_rst_target_ecl_passthrough(self) -> None:
        """app.py passe rst_target_ecl au CROAnalyst."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert "target_ecl=rst_target_ecl" in source, \
            "Le parametre target_ecl doit etre passe au CROAnalyst"

    def test_app_drill_down_selectbox_in_tab5(self) -> None:
        """app.py contient un selectbox sectoriel dans le tab Asymetries."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert "drill_sector" in source, "Key selectbox drill_sector manquant"

    def test_app_rst_scenario_getattr_safe(self) -> None:
        """app.py utilise getattr avec default pour SCENARIO_BASE (robustesse)."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert "getattr(SCENARIO_BASE, var, 0.0)" in source, \
            "getattr doit avoir un default pour eviter AttributeError"

    def test_app_no_builtin_shadowing(self) -> None:
        """app.py ne masque pas le built-in 'in' avec une variable _in."""
        from pathlib import Path

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        source = app_path.read_text(encoding="utf-8")
        # _in masque le builtin — doit etre renomme _inf ou similaire
        assert "_in = DASHBOARD_CONFIG" not in source, \
            "Variable '_in' masque le builtin Python 'in' — utiliser '_inf'"
