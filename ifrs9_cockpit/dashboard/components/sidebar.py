"""Sidebar and small layout components."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dash import dcc, html
import dash_bootstrap_components as dbc

from ifrs9_cockpit.config import (
    DASHBOARD_CONFIG as _CFG,
    PREDEFINED_SCENARIOS,
    SCENARIO_BASE,
    BASEL_CONFIG,
)
from ifrs9_cockpit.dashboard import ids
from ifrs9_cockpit.dashboard.components.helpers import (
    _sidebar_section_label,
    _slider_label,
    _slider_with_value,
    _rw_label,
)


def build_header() -> html.Div:
    """Return the cockpit header with gradient title and subtitle.

    The gradient effect is achieved via the ``cockpit-header`` CSS class
    defined in ``styles_dash.py``.

    Returns:
        A ``html.Div`` containing the main ``<h1>`` and tagline ``<p>``.
    """
    return html.Div(
        className="cockpit-header",
        children=[
            html.H1("IFRS 9 Risk Cockpit"),
            html.P("Moteur ECL \u2022 AI Analyst \u2022 Stress Testing"),
        ],
    )


def build_section_title(title: str) -> html.H2:
    """Return a styled section title heading.

    Uses semantic ``<h2>`` with the ``section-title`` CSS class that
    adds a Steel Blue underline and proper spacing.

    Args:
        title: Section heading text.

    Returns:
        An ``html.H2`` element.
    """
    return html.H2(title, className="section-title")


def build_scenario_banner(
    macro_params: Dict[str, float],
    scenario_name: str = "Manuel",
) -> html.Div:
    """Build a compact banner showing active macro parameters.

    Displayed above the main content area to provide immediate context
    on the active stress scenario.

    Format example::

        Scenario: Central | BCE: 3.50% | Chomage: 7.50% | PIB: 1.20% |
        HPI: 2.00% | Inflation: 2.50%

    Args:
        macro_params: Dict with keys ``interest_rate_bp``,
            ``unemployment_bipolar``, ``gdp_pct``, ``hpi_pct``,
            ``inflation_pct``.
        scenario_name: Display name of the scenario (default "Manuel").

    Returns:
        A ``html.Div`` with id ``ids.SCENARIO_BANNER``.
    """
    # Extract values with sensible defaults from SCENARIO_BASE
    rate_bp = macro_params.get("interest_rate_bp", 0.0)
    unemp = macro_params.get("unemployment_bipolar", 0.0)
    gdp = macro_params.get("gdp_pct", SCENARIO_BASE.gdp_growth)
    hpi = macro_params.get("hpi_pct", SCENARIO_BASE.hpi_growth)
    infl = macro_params.get("inflation_pct", SCENARIO_BASE.inflation_rate)

    # Convert BCE bp to level: base rate + bp/100
    bce_level = SCENARIO_BASE.interest_rate + rate_bp / 100.0

    # Unemployment: base + |bipolar|
    unemp_level = SCENARIO_BASE.unemployment_rate + abs(unemp)

    parts = [
        html.Strong(f"Scenario: {scenario_name}"),
        html.Span(" | "),
        html.Span(f"BCE: {bce_level:.2f}%"),
        html.Span(" | "),
        html.Span(f"Chomage: {unemp_level:.2f}%"),
        html.Span(" | "),
        html.Span(f"PIB: {gdp:.2f}%"),
        html.Span(" | "),
        html.Span(f"HPI: {hpi:.2f}%"),
        html.Span(" | "),
        html.Span(f"Inflation: {infl:.2f}%"),
    ]

    return html.Div(
        children=parts,
        style={
            "background": "rgba(59, 130, 246, 0.08)",
            "border": "1px solid rgba(59, 130, 246, 0.20)",
            "borderRadius": "8px",
            "padding": "0.5rem 1rem",
            "margin": "0.5rem 0",
            "fontSize": "0.80rem",
            "color": _CFG.theme_text_muted,
            "fontFamily": "monospace",
        },
    )


def build_collapsible_button(
    label: str,
    btn_id: str,
    icon: str = "\u25b8",
) -> dbc.Button:
    """Build a styled toggle button for collapsible sections.

    The button is full-width, dark-themed, and left-aligned with an
    icon prefix.  It is designed to work with ``dbc.Collapse`` via
    a simple callback toggling the ``is_open`` property.

    Example callback::

        @app.callback(
            Output(ids.COLLAPSE_PERF, "is_open"),
            Input(ids.BTN_PERF, "n_clicks"),
            State(ids.COLLAPSE_PERF, "is_open"),
        )
        def toggle(n, is_open):
            return not is_open if n else is_open

    Args:
        label: Button display text.
        btn_id: Dash component id (e.g. ``ids.BTN_PERF``).
        icon: Unicode icon prefix (default right-pointing triangle).

    Returns:
        A ``dbc.Button`` styled for dark theme collapsible sections.
    """
    return dbc.Button(
        children=[
            html.Span(
                icon,
                style={"marginRight": "0.5rem", "transition": "transform 0.2s"},
            ),
            label,
        ],
        id=btn_id,
        n_clicks=0,
        style={
            "width": "100%",
            "textAlign": "left",
            "background": _CFG.theme_bg_card,
            "color": _CFG.theme_text,
            "border": "1px solid rgba(99, 102, 241, 0.20)",
            "borderRadius": "8px",
            "padding": "0.6rem 1rem",
            "fontSize": "0.85rem",
            "fontWeight": "600",
            "marginBottom": "0.3rem",
            "cursor": "pointer",
        },
        className="collapsible-toggle",
    )


def build_sidebar(pd_model_names: List[str]) -> html.Div:
    """Build the full sidebar with all interactive controls.

    Contains three sections:

    1. **Stress Test** -- scenario dropdown + 5 macro sliders
    2. **Configuration** -- PD model selector, PE allocation, RW PE
    3. **Reverse Stress Test** -- RST toggle + ECL target input

    All component IDs are sourced from ``ids.py``.

    Args:
        pd_model_names: Available PD model family names
            (e.g. ``["LR_WoE", "XGBoost", "TabNet"]``).

    Returns:
        A ``html.Div`` containing the complete sidebar layout.
    """
    # Scenario dropdown options
    scenario_options = [{"label": "Manuel", "value": "Manuel"}] + [
        {"label": name, "value": name}
        for name in PREDEFINED_SCENARIOS.keys()
    ]

    # RW PE dropdown options (values as strings for dbc.Select)
    rw_pe_options = [
        {"label": f"{rw}% ({_rw_label(rw)})", "value": str(rw)}
        for rw in BASEL_CONFIG.rw_pe_options
    ]

    # PD model dropdown options
    pd_options = [{"label": name, "value": name} for name in pd_model_names]

    children = [
        # Title
        html.Div(
            "IFRS 9 Risk Cockpit",
            style={
                "color": _CFG.theme_primary,
                "fontSize": "1.0rem",
                "fontWeight": "700",
                "textAlign": "center",
                "padding": "1rem 0 0.5rem 0",
            },
        ),
        html.Hr(style={"borderColor": "rgba(99, 102, 241, 0.15)", "margin": "0"}),

        # ── Section 1: Stress Test ──
        _sidebar_section_label("Stress Test"),

        _slider_label("Scenario predefini"),
        dbc.Select(
            id=ids.SCENARIO_SELECTOR,
            options=scenario_options,
            value="Manuel",
            style={"marginBottom": "1rem"},
        ),

        _slider_with_value("Taux BCE (bp)", ids.VAL_INTEREST_RATE, "0"),
        dcc.Slider(
            id=ids.SL_INTEREST_RATE, min=-200, max=400, step=25, value=0,
            marks={-200: "-200", 0: "0", 200: "200", 400: "400"},
        ),

        _slider_with_value("Chomage bipolaire (pp)", ids.VAL_UNEMPLOYMENT, "0.0"),
        dcc.Slider(
            id=ids.SL_UNEMPLOYMENT, min=-5.0, max=5.0, step=0.5, value=0.0,
            marks={-5: "-5", 0: "0", 5: "5"},
        ),

        _slider_with_value("Croissance PIB (%)", ids.VAL_GDP, f"{SCENARIO_BASE.gdp_growth}"),
        dcc.Slider(
            id=ids.SL_GDP, min=-5.0, max=8.0, step=0.5, value=SCENARIO_BASE.gdp_growth,
            marks={-5: "-5", 0: "0", 5: "5", 8: "8"},
        ),

        _slider_with_value("Prix immobiliers (%)", ids.VAL_HPI, f"{SCENARIO_BASE.hpi_growth}"),
        dcc.Slider(
            id=ids.SL_HPI, min=-20.0, max=15.0, step=1.0, value=SCENARIO_BASE.hpi_growth,
            marks={-20: "-20", 0: "0", 15: "15"},
        ),

        _slider_with_value("Inflation IPC (%)", ids.VAL_INFLATION, f"{SCENARIO_BASE.inflation_rate}"),
        dcc.Slider(
            id=ids.SL_INFLATION, min=0.0, max=8.0, step=0.5, value=SCENARIO_BASE.inflation_rate,
            marks={0: "0", 4: "4", 8: "8"},
        ),

        # ── Section 2: Configuration ──
        _sidebar_section_label("Configuration"),

        _slider_label("Modele PD"),
        dbc.Select(
            id=ids.PD_MODEL_SELECTOR,
            options=pd_options,
            value=pd_model_names[0] if pd_model_names else "LR_WoE",
            style={"marginBottom": "0.8rem"},
        ),

        # PE allocation: auto via BL-CVaR optimizer (plus de slider)
        dcc.Store(id=ids.PE_ALLOCATION, data=20),
        dcc.Store(id=ids.VAL_PE_ALLOC, data="20"),
        # RW PE: auto via compute_crr3_rw() par position (plus de dropdown)
        dcc.Store(id=ids.RW_PE_SELECTOR, data="250"),

        # ── Section 3: Reverse Stress Test ──
        _sidebar_section_label("Reverse Stress Test"),

        dcc.Checklist(
            id=ids.RST_ENABLED,
            options=[{"label": " Activer RST", "value": "on"}],
            value=[],
            style={
                "color": _CFG.theme_text,
                "fontSize": "0.82rem",
                "marginBottom": "0.5rem",
            },
            inputStyle={"marginRight": "0.4rem"},
        ),

        _slider_label("Cible ECL (Md EUR)"),
        dbc.Input(
            id=ids.RST_TARGET,
            type="number",
            value=200,
            min=0,
            step=10,
            placeholder="Ex: 200 (= 200 Md EUR)",
            style={
                "background": _CFG.theme_bg_card,
                "color": _CFG.theme_text,
                "border": "1px solid rgba(99, 102, 241, 0.25)",
                "borderRadius": "6px",
                "fontSize": "0.82rem",
            },
            disabled=True,
        ),

        # Footer
        html.Hr(
            style={
                "borderColor": "rgba(99, 102, 241, 0.15)",
                "marginTop": "1.5rem",
            }
        ),
        html.Div(
            "IFRS 9 Risk Cockpit v3.0",
            style={
                "color": _CFG.theme_text_muted,
                "fontSize": "0.65rem",
                "textAlign": "center",
                "padding": "0.3rem 0",
                "opacity": "0.6",
            },
        ),
    ]

    return html.Div(
        children=children,
        className="sidebar",
    )
