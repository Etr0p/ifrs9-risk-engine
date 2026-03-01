"""KPI and score card builders."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dash import dcc, html
import dash_bootstrap_components as dbc

from ifrs9_cockpit.config import DASHBOARD_CONFIG as _CFG
from ifrs9_cockpit.utils.helpers import format_euro, format_pct
from ifrs9_cockpit.dashboard import ids


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
