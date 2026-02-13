"""Composants UI Dash natifs pour le dashboard IFRS 9 Risk Cockpit.

Remplace les appels ``st.markdown(unsafe_allow_html=True)`` de
``components.py`` par des constructeurs de composants Dash qui
retournent des arbres ``html.Div`` / ``dbc.*`` manipulables par les
callbacks.

Les classes CSS (kpi-card, kpi-container, badge-stage1, insight-box,
section-title, score-card, etc.) sont definies dans ``styles_dash.py``
et injectees globalement au demarrage de l'application.

Usage typique dans ``layout.py``::

    from ifrs9_cockpit.dashboard.components_dash import (
        build_header, build_kpi_row, build_sidebar,
    )
    header = build_header()
    sidebar = build_sidebar(pd_model_names=["LR_WoE", "XGBoost", "TabNet"])
"""

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
from ifrs9_cockpit.utils.helpers import format_euro, format_pct
from ifrs9_cockpit.dashboard import ids


# ──────────────────────────────────────────────────────────────
# 1. HEADER
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# 2. KPI ROW (generic)
# ──────────────────────────────────────────────────────────────

def build_kpi_row(cards: List[Dict[str, str]]) -> html.Div:
    """Build a row of KPI cards from a list of descriptors.

    Each descriptor is a dict with the following keys:

    * **label** -- upper-case label (e.g. ``"ECL / EAD"``).
    * **value** -- pre-formatted display value (e.g. ``"2.31%"``).
    * **sub_text** -- subtitle / trend annotation.
    * **sub_class** -- one of ``"positive"``, ``"negative"``, ``"neutral"``.

    Accessibility: each card carries ``role="status"`` and an
    ``aria-label`` summarising the KPI for screen readers.

    Args:
        cards: List of KPI descriptor dicts.

    Returns:
        A ``html.Div`` with className ``"kpi-container"`` wrapping
        one ``html.Div(className="kpi-card")`` per descriptor.
    """
    children = []
    for card in cards:
        label = card.get("label", "")
        value = card.get("value", "")
        sub_text = card.get("sub_text", "")
        sub_class = card.get("sub_class", "neutral")
        aria = f"{label}: {value}, {sub_text}"
        children.append(
            html.Div(
                className="kpi-card",
                role="status",
                **{"aria-label": aria},
                children=[
                    html.Div(label, className="kpi-label"),
                    html.Div(value, className="kpi-value"),
                    html.Div(sub_text, className=f"kpi-sub {sub_class}"),
                ],
            )
        )
    return html.Div(className="kpi-container", children=children)


# ──────────────────────────────────────────────────────────────
# 3. CLASSIFICATION ROW (Stage + PE badges)
# ──────────────────────────────────────────────────────────────

def build_classification_row(
    stage_counts: Dict[int, int],
    pe_categories: Dict[str, int],
    n_total: int,
) -> html.Div:
    """Build a badge row for IFRS 9 Stages and PE risk categories.

    Badge CSS classes:

    * ``badge badge-stage1`` -- green (Performing / Stage 1)
    * ``badge badge-stage2`` -- amber (Watchlist / Stage 2)
    * ``badge badge-stage3`` -- red (Distressed / Stage 3)

    Args:
        stage_counts: Mapping ``{1: count, 2: count, 3: count}``.
        pe_categories: Mapping ``{"Performing": n, "Watchlist": n,
            "Distressed": n}``.
        n_total: Total credit positions (denominator for stage %).

    Returns:
        A flex container of ``html.Span`` badge elements.
    """
    badge_cls = {1: "badge-stage1", 2: "badge-stage2", 3: "badge-stage3"}
    pe_cls = {
        "Performing": "badge-stage1",
        "Watchlist": "badge-stage2",
        "Distressed": "badge-stage3",
    }

    children: list[Any] = []

    # Stage badges
    for stage in (1, 2, 3):
        count = stage_counts.get(stage, 0)
        pct = count / n_total if n_total > 0 else 0.0
        children.append(
            html.Span(
                f"Stage {stage}: {count:,} ({pct:.1%})",
                className=f"badge {badge_cls[stage]}",
            )
        )

    # Visual separator
    children.append(
        html.Span(
            "|",
            style={"color": "#475569", "margin": "0 0.3rem", "alignSelf": "center"},
        )
    )

    # PE badges -- denominator = PE total, not n_total
    pe_total = sum(pe_categories.values()) if pe_categories else 0
    for cat in ("Performing", "Watchlist", "Distressed"):
        count = pe_categories.get(cat, 0)
        pct = count / pe_total if pe_total > 0 else 0.0
        children.append(
            html.Span(
                f"{cat}: {count:,} ({pct:.1%})",
                className=f"badge {pe_cls[cat]}",
            )
        )

    return html.Div(
        children=children,
        style={
            "display": "flex",
            "gap": "0.5rem",
            "flexWrap": "wrap",
            "margin": "0.5rem 0",
            "alignItems": "center",
        },
    )


# ──────────────────────────────────────────────────────────────
# 4. AI NARRATIVE BOX
# ──────────────────────────────────────────────────────────────

def build_ai_narrative_box(
    narrative: str,
    recommendations: list,
) -> html.Div:
    """Build the AI Analyst insight box with narrative and recommendations.

    The risk level is determined from the first recommendation's
    ``risk_appetite_status`` attribute (vert / ambre / rouge).

    Args:
        narrative: Multi-line narrative text from CROAnalyst.
        recommendations: List of ``Recommendation`` dataclass instances.

    Returns:
        A styled ``html.Div`` with className ``"insight-box"``.
    """
    # Determine accent color and level label from first recommendation
    if recommendations:
        rec = recommendations[0]
        status = getattr(rec, "risk_appetite_status", "vert")
        if status == "rouge":
            accent = _CFG.color_danger
            icon_char = "\U0001f6a8"
            level = "ELEVE"
        elif status == "ambre":
            accent = _CFG.color_warning
            icon_char = "\u26a0\ufe0f"
            level = "MODERE"
        else:
            accent = _CFG.color_success
            icon_char = "\u2705"
            level = "BON"
    else:
        accent = _CFG.theme_text_muted
        icon_char = "\u2139\ufe0f"
        level = "N/A"

    # Narrative section
    narrative_text = narrative if narrative else "Analyse en cours..."
    narrative_parts = narrative_text.split("\n")
    narrative_children: list[Any] = []
    for i, part in enumerate(narrative_parts):
        if i > 0:
            narrative_children.append(html.Br())
        narrative_children.append(part)

    # Recommendations section
    recs_children: list[Any] = []
    if recommendations:
        recs_children.append(
            html.Div("RECOMMANDATIONS AI ANALYST", className="cro-section-label")
        )
        for r in recommendations:
            recs_children.append(
                html.Div(
                    className="cro-recommendation",
                    children=[
                        html.Span(
                            "ACTION ",
                            style={"color": accent, "fontWeight": "700"},
                        ),
                        r.action,
                    ],
                )
            )
            if r.alternatives:
                for alt in r.alternatives:
                    recs_children.append(
                        html.Div(
                            className="cro-finding",
                            children=[
                                html.Span(
                                    "ALT ",
                                    style={
                                        "color": _CFG.theme_text_muted,
                                        "fontWeight": "600",
                                    },
                                ),
                                alt,
                            ],
                        )
                    )

    return html.Div(
        className="insight-box",
        style={"borderLeftColor": accent},
        children=[
            # Header
            html.Div(
                className="insight-box-header",
                children=[
                    html.Span(icon_char, style={"fontSize": "1.2rem"}),
                    html.H3([
                        "AI Analyst (2 passes) \u2014 Niveau ",
                        html.Span(level, style={"color": accent}),
                    ]),
                ],
            ),
            # Body
            html.Div(
                className="insight-box-content",
                children=[
                    html.Div(narrative_children, className="cro-summary"),
                    *recs_children,
                ],
            ),
        ],
    )


# ──────────────────────────────────────────────────────────────
# 4b. REVERSE STRESS TEST RESULTS PANEL
# ──────────────────────────────────────────────────────────────

def build_rst_results_panel(
    rst_result: Dict[str, Any],
    rst_distance: float,
    ecl_current: float = 0.0,
) -> html.Div:
    """Build a prominent panel showing Reverse Stress Test results.

    Displayed only when RST is active. Shows breach status, distance
    in sigma, target ECL, breach ECL, and the tipping scenario.
    Handles the "already breached at baseline" case (distance=0).

    Args:
        rst_result: Dict from layer3_rst.reverse_stress_test().
        rst_distance: Mahalanobis distance (sigma).
        ecl_current: Current portfolio ECL in EUR (for context).

    Returns:
        A styled ``html.Div`` with RST results.
    """
    if not rst_result:
        return html.Div()

    breach = rst_result.get("breach", False)
    distance_sigma = rst_result.get("rst_distance_sigma", rst_distance)
    ecl_breach = rst_result.get("rst_ecl", 0)
    ecl_threshold = rst_result.get("ecl_breach_threshold", 0)
    capital_base = rst_result.get("capital_base", 0)
    rst_scenario = rst_result.get("rst_scenario", {})

    # Handle inf/nan distance gracefully
    if not isinstance(distance_sigma, (int, float)) or distance_sigma != distance_sigma:
        distance_sigma = 0.0
    if distance_sigma == float("inf"):
        distance_sigma = -1  # sentinel for "no breach found"

    # Detect "already breached at baseline" (breach=True, distance~0)
    already_breached = breach and distance_sigma < 0.15

    # Status styling
    if already_breached:
        border_color = "#FBBF24"  # warning amber
        status_icon = "\u26A0"
        status_text = "DEJA EN BREACH"
        status_color = "#FBBF24"
    elif breach:
        border_color = "#F87171"  # danger
        status_icon = "\u26A0"
        status_text = f"BREACH A {distance_sigma:.1f}\u03C3"
        status_color = "#F87171"
    elif distance_sigma < 0:
        border_color = "#34D399"  # accent
        status_icon = "\u2713"
        status_text = "AUCUN BREACH"
        status_color = "#34D399"
    else:
        border_color = "#34D399"
        status_icon = "\u2713"
        status_text = "PAS DE BREACH"
        status_color = "#34D399"

    # KPI items
    kpi_items = [("ECL Actuel", format_euro(ecl_current), "portefeuille courant")]
    kpi_items.append(("ECL Cible RST", format_euro(ecl_threshold), "seuil de rupture"))
    if breach and not already_breached:
        dist_str = f"{distance_sigma:.1f} \u03C3"
        kpi_items.append(("Distance", dist_str, "Mahalanobis"))
        kpi_items.append(("ECL Rupture", format_euro(ecl_breach), "au point de breach"))
    elif not breach and distance_sigma < 0:
        kpi_items.append(("Distance", "> 10 \u03C3", "aucun breach dans la plage"))

    kpi_children = []
    for label, value, sub in kpi_items:
        kpi_children.append(
            html.Div(
                style={
                    "textAlign": "center",
                    "padding": "0.5rem 1rem",
                    "minWidth": "120px",
                },
                children=[
                    html.Div(label, style={"color": "#A1B2C8", "fontSize": "0.7rem", "textTransform": "uppercase"}),
                    html.Div(value, style={"color": "#F8FAFC", "fontSize": "1.1rem", "fontWeight": "600"}),
                    html.Div(sub, style={"color": "#64748B", "fontSize": "0.65rem"}),
                ],
            )
        )

    # Warning message for "already breached"
    warning_children = []
    if already_breached:
        suggested = int(ecl_current * 1.5 / 1e9)  # 150% of current ECL in Md EUR
        warning_children = [
            html.Div(
                style={
                    "marginTop": "0.75rem",
                    "padding": "0.5rem 0.75rem",
                    "background": "rgba(251, 191, 36, 0.1)",
                    "borderRadius": "8px",
                    "border": "1px solid rgba(251, 191, 36, 0.3)",
                },
                children=[
                    html.Span(
                        "L'ECL actuel du portefeuille depasse deja la cible RST. "
                        f"Augmentez la cible (ex: {suggested:,} Md EUR) pour obtenir "
                        "une distance de Mahalanobis significative.",
                        style={"color": "#FBBF24", "fontSize": "0.8rem"},
                    ),
                ],
            )
        ]

    # Scenario breakdown (only if breach found with real stress)
    scenario_children = []
    if breach and not already_breached and rst_scenario:
        _var_labels = {
            "unemployment_rate": "Chomage",
            "gdp_growth": "PIB",
            "interest_rate": "Taux",
            "hpi_growth": "Immobilier",
            "inflation_rate": "Inflation",
        }
        scenario_items = []
        for var, val in rst_scenario.items():
            label = _var_labels.get(var, var)
            scenario_items.append(
                html.Span(
                    f"{label}: {val:.1f}%",
                    style={
                        "backgroundColor": "#1E293B",
                        "color": "#F8FAFC",
                        "padding": "0.25rem 0.5rem",
                        "borderRadius": "4px",
                        "fontSize": "0.75rem",
                        "marginRight": "0.5rem",
                        "marginBottom": "0.25rem",
                        "display": "inline-block",
                    },
                )
            )
        scenario_children = [
            html.Div(
                style={"marginTop": "0.75rem", "paddingTop": "0.75rem", "borderTop": "1px solid #334155"},
                children=[
                    html.Span(
                        "Scenario de rupture : ",
                        style={"color": "#A1B2C8", "fontSize": "0.75rem", "fontWeight": "600"},
                    ),
                    html.Div(scenario_items, style={"marginTop": "0.25rem"}),
                ],
            )
        ]

    return html.Div(
        style={
            "background": "#0C1222",
            "border": f"1px solid {border_color}",
            "borderLeft": f"4px solid {border_color}",
            "borderRadius": "12px",
            "padding": "1rem 1.25rem",
            "marginBottom": "1rem",
        },
        children=[
            # Header row: icon + title + status
            html.Div(
                style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "marginBottom": "0.75rem"},
                children=[
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "0.5rem"},
                        children=[
                            html.Span(
                                "\U0001F50D",
                                style={"fontSize": "1.1rem"},
                            ),
                            html.Span(
                                "Reverse Stress Test",
                                style={"color": "#F8FAFC", "fontWeight": "600", "fontSize": "0.95rem"},
                            ),
                        ],
                    ),
                    html.Span(
                        f"{status_icon} {status_text}",
                        style={
                            "color": status_color,
                            "fontWeight": "700",
                            "fontSize": "0.85rem",
                            "padding": "0.2rem 0.75rem",
                            "border": f"1px solid {status_color}",
                            "borderRadius": "20px",
                        },
                    ),
                ],
            ),
            # KPI row
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "0.5rem", "justifyContent": "space-around"},
                children=kpi_children,
            ),
            # Warning (already breached)
            *warning_children,
            # Scenario breakdown
            *scenario_children,
        ],
    )


# ──────────────────────────────────────────────────────────────
# 5. SECTION TITLE
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# 6. SCENARIO BANNER
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# 7. PE SCORE CARD (clickable)
# ──────────────────────────────────────────────────────────────

_BORDER_COLORS = {
    "vert": _CFG.color_success,
    "ambre": _CFG.color_warning,
    "rouge": _CFG.color_danger,
}


def build_pe_score_card(
    raroc_pe: float,
    nav_drawdown: float,
    delta_nav: float,
    ra_color: str,
    pe_cats: dict | None = None,
) -> html.Div:
    """Build a clickable score card for the Private Equity pocket."""
    border = _BORDER_COLORS.get(ra_color, _CFG.theme_text_muted)
    raroc_color = _CFG.color_success if raroc_pe >= 0.04 else (
        _CFG.color_warning if raroc_pe >= 0.02 else _CFG.color_danger
    )
    pe_cats = pe_cats or {}
    n_performing = pe_cats.get("Performing", 0)
    n_watchlist = pe_cats.get("Watchlist", 0)
    n_distressed = pe_cats.get("Distressed", 0)
    n_total = n_performing + n_watchlist + n_distressed
    distress_pct = n_distressed / n_total if n_total > 0 else 0.0

    _metric = {"fontSize": "0.82rem", "marginBottom": "0.3rem"}

    return html.Div(
        className="score-card",
        style={
            "borderLeft": f"4px solid {border}",
            "cursor": "pointer",
            "padding": "1.2rem 1.5rem",
        },
        children=[
            html.Div(
                "PRIVATE EQUITY",
                style={
                    "color": _CFG.theme_text_muted,
                    "fontSize": "0.70rem",
                    "textTransform": "uppercase",
                    "letterSpacing": "1.5px",
                    "fontWeight": "700",
                    "marginBottom": "0.5rem",
                },
            ),
            html.Div(
                format_pct(raroc_pe, 1),
                style={
                    "color": raroc_color,
                    "fontSize": "2.0rem",
                    "fontWeight": "800",
                    "lineHeight": "1.2",
                },
            ),
            html.Div(
                "RAROC PE",
                style={
                    "color": _CFG.theme_text_muted,
                    "fontSize": "0.70rem",
                    "marginBottom": "0.8rem",
                },
            ),
            # Separator
            html.Hr(style={"borderColor": "rgba(99,102,241,0.12)", "margin": "0.5rem 0"}),
            html.Div([
                html.Span("NAV Drawdown: ", style={"color": _CFG.theme_text_muted}),
                html.Span(format_pct(nav_drawdown, 1), style={
                    "color": _CFG.color_danger if nav_drawdown > 0.10 else _CFG.theme_text,
                    "fontWeight": "600",
                }),
            ], style=_metric),
            html.Div([
                html.Span("Delta NAV: ", style={"color": _CFG.theme_text_muted}),
                html.Span(format_euro(delta_nav), style={"fontWeight": "600", "color": _CFG.theme_text}),
            ], style=_metric),
            html.Div([
                html.Span("Positions: ", style={"color": _CFG.theme_text_muted}),
                html.Span(f"{n_total:,}", style={"fontWeight": "600", "color": _CFG.theme_text}),
            ], style=_metric),
            html.Div([
                html.Span("Distressed: ", style={"color": _CFG.theme_text_muted}),
                html.Span(
                    f"{n_distressed} ({format_pct(distress_pct, 1)})",
                    style={
                        "fontWeight": "600",
                        "color": _CFG.color_danger if n_distressed > 0 else _CFG.theme_text,
                    },
                ),
            ], style={**_metric, "marginBottom": "0.8rem"}),
            html.Div(
                "Cliquer pour voir les details",
                style={
                    "color": _CFG.theme_primary,
                    "fontSize": "0.68rem",
                    "fontStyle": "italic",
                    "textAlign": "center",
                },
            ),
        ],
    )


# ──────────────────────────────────────────────────────────────
# 8. CREDIT SCORE CARD (clickable)
# ──────────────────────────────────────────────────────────────

def build_credit_score_card(
    raroc_credit: float,
    ecl_ead_ratio: float,
    ecl_total: float,
    ra_color: str,
    n_clients: int = 0,
    hhi_cross: int = 0,
) -> html.Div:
    """Build a clickable score card for the Credit pocket."""
    border = _BORDER_COLORS.get(ra_color, _CFG.theme_text_muted)
    raroc_color = _CFG.color_success if raroc_credit >= 0.04 else (
        _CFG.color_warning if raroc_credit >= 0.02 else _CFG.color_danger
    )

    _metric = {"fontSize": "0.82rem", "marginBottom": "0.3rem"}

    return html.Div(
        className="score-card",
        style={
            "borderLeft": f"4px solid {border}",
            "cursor": "pointer",
            "padding": "1.2rem 1.5rem",
        },
        children=[
            html.Div(
                "CREDIT BANCAIRE",
                style={
                    "color": _CFG.theme_text_muted,
                    "fontSize": "0.70rem",
                    "textTransform": "uppercase",
                    "letterSpacing": "1.5px",
                    "fontWeight": "700",
                    "marginBottom": "0.5rem",
                },
            ),
            html.Div(
                format_pct(raroc_credit, 1),
                style={
                    "color": raroc_color,
                    "fontSize": "2.0rem",
                    "fontWeight": "800",
                    "lineHeight": "1.2",
                },
            ),
            html.Div(
                "RAROC Credit",
                style={
                    "color": _CFG.theme_text_muted,
                    "fontSize": "0.70rem",
                    "marginBottom": "0.8rem",
                },
            ),
            html.Hr(style={"borderColor": "rgba(99,102,241,0.12)", "margin": "0.5rem 0"}),
            html.Div([
                html.Span("ECL / EAD: ", style={"color": _CFG.theme_text_muted}),
                html.Span(format_pct(ecl_ead_ratio, 2), style={
                    "color": _CFG.color_danger if ecl_ead_ratio > 0.04 else _CFG.theme_text,
                    "fontWeight": "600",
                }),
            ], style=_metric),
            html.Div([
                html.Span("ECL Total: ", style={"color": _CFG.theme_text_muted}),
                html.Span(format_euro(ecl_total), style={"fontWeight": "600", "color": _CFG.theme_text}),
            ], style=_metric),
            html.Div([
                html.Span("Clients: ", style={"color": _CFG.theme_text_muted}),
                html.Span(f"{n_clients:,}", style={"fontWeight": "600", "color": _CFG.theme_text}),
            ], style=_metric),
            html.Div([
                html.Span("HHI Concentration: ", style={"color": _CFG.theme_text_muted}),
                html.Span(f"{hhi_cross:,}", style={"fontWeight": "600", "color": _CFG.theme_text}),
            ], style={**_metric, "marginBottom": "0.8rem"}),
            html.Div(
                "Cliquer pour voir les details",
                style={
                    "color": _CFG.theme_primary,
                    "fontSize": "0.68rem",
                    "fontStyle": "italic",
                    "textAlign": "center",
                },
            ),
        ],
    )


# ──────────────────────────────────────────────────────────────
# 9. SIDEBAR (full controls panel)
# ──────────────────────────────────────────────────────────────

def _sidebar_section_label(text: str) -> html.Div:
    """Return a styled sidebar section label.

    Args:
        text: Label text (e.g. "Stress Test").

    Returns:
        A ``html.Div`` styled as an uppercase section label.
    """
    return html.Div(
        text,
        style={
            "color": _CFG.theme_text_muted,
            "fontSize": "0.68rem",
            "textTransform": "uppercase",
            "letterSpacing": "1.5px",
            "fontWeight": "700",
            "marginTop": "1.2rem",
            "marginBottom": "0.6rem",
            "paddingBottom": "0.3rem",
            "borderBottom": f"1px solid rgba(99, 102, 241, 0.20)",
        },
    )


def _slider_label(text: str) -> html.Label:
    """Return a styled slider label.

    Args:
        text: Label text for the slider.

    Returns:
        An ``html.Label`` with muted text styling.
    """
    return html.Label(
        text,
        style={
            "color": _CFG.theme_text_muted,
            "fontSize": "0.78rem",
            "fontWeight": "500",
            "marginBottom": "0.2rem",
            "display": "block",
        },
    )


def _slider_with_value(label: str, value_id: str, default: str) -> html.Div:
    """Label row with slider name on left and live value on right."""
    return html.Div(
        [
            html.Span(
                label,
                style={
                    "color": _CFG.theme_text_muted,
                    "fontSize": "0.78rem",
                    "fontWeight": "500",
                },
            ),
            html.Span(
                default,
                id=value_id,
                style={
                    "color": _CFG.color_info,
                    "fontSize": "0.78rem",
                    "fontWeight": "700",
                },
            ),
        ],
        style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "marginBottom": "0.2rem",
            "marginTop": "0.5rem",
        },
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

        _slider_with_value("Allocation PE (%)", ids.VAL_PE_ALLOC, "20"),
        dcc.Slider(
            id=ids.PE_ALLOCATION,
            min=0, max=int(BASEL_CONFIG.pe_max_allocation * 100), step=5, value=20,
            marks={0: "0%", 20: "20%", 40: "40%"},
        ),

        _slider_label("Risk Weight PE (CRR3)"),
        dbc.Select(
            id=ids.RW_PE_SELECTOR,
            options=rw_pe_options,
            value=str(BASEL_CONFIG.rw_pe_default),
            style={"marginBottom": "0.8rem"},
        ),

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


def _rw_label(rw: int) -> str:
    """Return a human-readable label for a CRR3 risk weight.

    Args:
        rw: Risk weight in percent (190, 250, or 400).

    Returns:
        Short CRR3 classification label.
    """
    labels = {
        190: "IRB diversifie",
        250: "General equity",
        400: "Speculatif",
    }
    return labels.get(rw, f"{rw}%")


# ──────────────────────────────────────────────────────────────
# 10. ARBITRAGE INSIGHT
# ──────────────────────────────────────────────────────────────

def build_arbitrage_insight(optimization: Dict[str, Any]) -> html.Div:
    """Build the allocation optimization result panel.

    Displays the optimal credit/PE split with associated RAROC,
    RWA, CET1 ratio, and headroom.

    The ``optimization`` dict is expected to contain::

        {
            "w_credit": 0.80,
            "w_pe": 0.20,
            "raroc_credit": 0.045,
            "raroc_pe": 0.08,
            "rwa_weighted": 15_000_000_000,
            "cet1_ratio": 0.135,
            "headroom_m": 120.5,
            "feasible": True,
        }

    Args:
        optimization: Dict from ``comparator.optimize_allocation()``.

    Returns:
        A styled ``html.Div`` with the optimization summary.
    """
    w_credit = optimization.get("credit_allocation", optimization.get("w_credit", 0.0))
    w_pe = optimization.get("pe_allocation", optimization.get("w_pe", 0.0))
    raroc_credit = optimization.get("raroc_credit", 0.0)
    raroc_pe = optimization.get("raroc_pe", 0.0)
    rwa_weighted = optimization.get("rwa_weighted", 0.0)
    cet1_ratio = optimization.get("cet1_ratio", 0.0)
    # cet1_headroom is a ratio (e.g. 0.05 = 5pp), convert to M EUR
    cet1_headroom = optimization.get("cet1_headroom", 0.0)
    headroom_m = optimization.get("headroom_m", cet1_headroom * rwa_weighted / 1e6)
    feasible = optimization.get("feasible", cet1_headroom >= 0)

    feasible_color = _CFG.color_success if feasible else _CFG.color_danger
    feasible_text = "FAISABLE" if feasible else "NON FAISABLE"

    def _metric_row(label: str, value: str, color: str = _CFG.theme_text) -> html.Div:
        """Build a single metric row within the insight panel."""
        return html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "padding": "0.3rem 0",
                "borderBottom": "1px solid rgba(148, 163, 184, 0.08)",
            },
            children=[
                html.Span(
                    label,
                    style={
                        "color": _CFG.theme_text_muted,
                        "fontSize": "0.80rem",
                    },
                ),
                html.Span(
                    value,
                    style={
                        "color": color,
                        "fontSize": "0.80rem",
                        "fontWeight": "600",
                    },
                ),
            ],
        )

    return html.Div(
        className="insight-box",
        children=[
            html.Div(
                className="insight-box-header",
                children=[
                    html.Span("\u2696\ufe0f", style={"fontSize": "1.2rem"}),
                    html.H3("Allocation Optimale"),
                ],
            ),
            html.Div(
                className="insight-box-content",
                children=[
                    _metric_row("Credit", format_pct(w_credit, 1)),
                    _metric_row("Private Equity", format_pct(w_pe, 1)),
                    _metric_row("RAROC Credit", format_pct(raroc_credit, 2)),
                    _metric_row("RAROC PE", format_pct(raroc_pe, 2)),
                    _metric_row("RWA Pondere", format_euro(rwa_weighted)),
                    _metric_row("Ratio CET1", format_pct(cet1_ratio, 2)),
                    _metric_row(
                        "Headroom",
                        f"{headroom_m:+.1f}M EUR",
                        color=_CFG.color_success
                        if headroom_m >= 0
                        else _CFG.color_danger,
                    ),
                    html.Div(
                        style={
                            "textAlign": "center",
                            "marginTop": "0.6rem",
                            "padding": "0.3rem",
                        },
                        children=[
                            html.Span(
                                feasible_text,
                                style={
                                    "color": feasible_color,
                                    "fontSize": "0.75rem",
                                    "fontWeight": "700",
                                    "letterSpacing": "1px",
                                },
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# ──────────────────────────────────────────────────────────────
# 11. COLLAPSIBLE BUTTON
# ──────────────────────────────────────────────────────────────

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
