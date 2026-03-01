"""AI narrative and RST panel builders."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dash import dcc, html
import dash_bootstrap_components as dbc

from ifrs9_cockpit.config import DASHBOARD_CONFIG as _CFG
from ifrs9_cockpit.utils.helpers import format_euro, format_pct
from ifrs9_cockpit.dashboard import ids


def build_ai_narrative_box(
    narrative,
    recommendations: list,
) -> html.Div:
    """Build the AI Analyst insight box with structured narrative and recommendations.

    Handles both Dict (structured, v2) and str (legacy) narrative formats.
    When narrative is a Dict, renders 7 sections with muted headers and prose
    paragraphs. Color-coded recommendation with RST escalation logic.

    Args:
        narrative: Dict with keys (diagnostic, concentration, facteur,
            resilience, action, confiance, alternatives) or legacy str.
        recommendations: List of ``Recommendation`` dataclass instances.

    Returns:
        A styled ``html.Div`` with className ``"insight-box"``.
    """
    # Determine accent color, level label, and effective status
    if recommendations:
        rec = recommendations[0]
        status = getattr(rec, "risk_appetite_status", "vert")
        rst_dist = getattr(rec, "rst_distance", 99.0)
        # RST escalation: ambre + RST<2σ → rouge
        effective_status = status
        if rst_dist < 2.0 and status == "ambre":
            effective_status = "rouge"
        elif rst_dist < 2.0 and status == "vert":
            effective_status = "ambre"

        if effective_status == "rouge":
            accent = _CFG.color_danger
            icon_char = "\U0001f6a8"
            level = "ELEVE"
        elif effective_status == "ambre":
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
        effective_status = "N/A"

    # ── Build narrative body ──
    _section_header_style = {
        "color": _CFG.theme_text_muted,
        "fontSize": "0.68rem",
        "textTransform": "uppercase",
        "letterSpacing": "1.5px",
        "fontWeight": "700",
        "marginTop": "0.8rem",
        "marginBottom": "0.3rem",
    }
    _section_text_style = {
        "color": _CFG.theme_text,
        "fontSize": "0.82rem",
        "lineHeight": "1.5",
        "marginBottom": "0.4rem",
    }

    narrative_children: list[Any] = []

    if isinstance(narrative, dict):
        # Structured Dict format (v2)
        _sections = [
            ("Diagnostic", "diagnostic"),
            ("Concentration", "concentration"),
            ("Facteur macro dominant", "facteur"),
            ("Resilience", "resilience"),
        ]
        for label, key in _sections:
            text = narrative.get(key, "")
            if text:
                narrative_children.append(
                    html.Div(label, style=_section_header_style)
                )
                narrative_children.append(
                    html.P(text, style=_section_text_style)
                )

        # Confiance section
        confiance = narrative.get("confiance", "")
        if confiance:
            narrative_children.append(
                html.Div("Confiance", style=_section_header_style)
            )
            narrative_children.append(
                html.P(confiance, style={**_section_text_style, "fontStyle": "italic"})
            )
    else:
        # Legacy string format
        narrative_text = narrative if narrative else "Analyse en cours..."
        for i, part in enumerate(narrative_text.split("\n")):
            if i > 0:
                narrative_children.append(html.Br())
            narrative_children.append(part)

    # ── Recommendations section ──
    recs_children: list[Any] = []
    if recommendations:
        recs_children.append(html.Hr(style={
            "borderColor": "rgba(99, 102, 241, 0.15)",
            "margin": "0.8rem 0",
        }))
        recs_children.append(
            html.Div("RECOMMANDATIONS AI ANALYST", className="cro-section-label")
        )

        # Action with color-coded border
        action_text = ""
        if isinstance(narrative, dict):
            action_text = narrative.get("action", "")
        if not action_text and recommendations:
            action_text = recommendations[0].action

        action_color = {
            "rouge": _CFG.color_danger,
            "ambre": _CFG.color_warning,
            "vert": _CFG.color_success,
        }.get(effective_status, _CFG.theme_text_muted)

        recs_children.append(
            html.Div(
                style={
                    "borderLeft": f"3px solid {action_color}",
                    "paddingLeft": "0.75rem",
                    "marginBottom": "0.5rem",
                    "marginTop": "0.4rem",
                },
                children=[
                    html.Span(
                        "ACTION ",
                        style={"color": action_color, "fontWeight": "700", "fontSize": "0.78rem"},
                    ),
                    html.Span(
                        action_text,
                        style={"color": _CFG.theme_text, "fontSize": "0.82rem"},
                    ),
                ],
            )
        )

        # Alternatives
        alts = []
        if isinstance(narrative, dict) and narrative.get("alternatives"):
            alts = narrative["alternatives"].split(" | ")
        elif recommendations and recommendations[0].alternatives:
            alts = recommendations[0].alternatives

        for alt in alts:
            recs_children.append(
                html.Div(
                    style={
                        "borderLeft": f"3px solid {_CFG.theme_text_muted}",
                        "paddingLeft": "0.75rem",
                        "marginBottom": "0.3rem",
                    },
                    children=[
                        html.Span(
                            "ALT ",
                            style={
                                "color": _CFG.theme_text_muted,
                                "fontWeight": "600",
                                "fontSize": "0.75rem",
                            },
                        ),
                        html.Span(
                            alt,
                            style={"color": _CFG.theme_text, "fontSize": "0.80rem"},
                        ),
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
