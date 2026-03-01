"""Private helpers used by other component builders."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dash import dcc, html
import dash_bootstrap_components as dbc

from ifrs9_cockpit.config import DASHBOARD_CONFIG as _CFG
from ifrs9_cockpit.dashboard import ids


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
