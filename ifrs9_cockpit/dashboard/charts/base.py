"""Constantes partagees et helpers pour tous les modules de charts."""

from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.config import DASHBOARD_CONFIG


# Palette coherente avec le CSS — WCAG AAA
_PRIMARY = DASHBOARD_CONFIG.theme_primary
_SECONDARY = DASHBOARD_CONFIG.theme_secondary
_ACCENT = DASHBOARD_CONFIG.theme_accent
_BG = DASHBOARD_CONFIG.theme_bg_dark
_CARD = DASHBOARD_CONFIG.theme_bg_card
_TEXT = DASHBOARD_CONFIG.theme_text
_MUTED = DASHBOARD_CONFIG.theme_text_muted
_SUCCESS = DASHBOARD_CONFIG.color_success
_WARNING = DASHBOARD_CONFIG.color_warning
_DANGER = DASHBOARD_CONFIG.color_danger
_INFO = DASHBOARD_CONFIG.color_info
_COLORS = [_PRIMARY, _ACCENT, _WARNING, _DANGER, _SECONDARY, _INFO]

# Mapping noms techniques → labels lisibles (francais)
_FEATURE_LABELS: Dict[str, str] = {
    "credit_score": "Score Credit",
    "nb_past_due_30d": "Retards 30j",
    "income": "Revenu",
    "age": "Age",
    "months_since_last_delinquency": "Delai Dern. Incident",
    "employment_duration": "Anciennete Emploi",
    "debt_ratio": "Ratio Endettement",
    "loan_amount": "Montant Pret",
    "utilization_rate": "Taux Utilisation",
    "nb_credit_lines": "Nb Lignes Credit",
}

# Vibrant sector palette — each sector owns a distinct hue
_SECTOR_PALETTE = {
    "Technologie": "#818CF8",   # soft indigo
    "Industrie": "#34D399",     # emerald
    "Sante": "#F472B6",         # rose
    "Immobilier": "#FBBF24",    # amber
    "Services": "#22D3EE",      # cyan
}
_SECTOR_PALETTE_FALLBACK = ["#818CF8", "#34D399", "#F472B6", "#FBBF24", "#22D3EE"]


def _prettify_feature(name: str) -> str:
    """Convertit un nom technique de feature en label lisible.

    Gere les suffixes _woe en les supprimant avant le lookup.

    Args:
        name: Nom technique (ex: 'credit_score_woe').

    Returns:
        Label lisible (ex: 'Score Credit').
    """
    base = name.replace("_woe", "")
    return _FEATURE_LABELS.get(base, name)


def _base_layout(title: str = "", height: int = 400) -> dict:
    """Layout Plotly de base avec theme sombre.

    Args:
        title: Titre du graphique.
        height: Hauteur en pixels.

    Returns:
        Dictionnaire de layout.
    """
    return dict(
        title=dict(text=title, font=dict(color=_TEXT, size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_MUTED, size=11),
        height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(
            font=dict(color=_MUTED, size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.1)",
            zerolinecolor="rgba(148,163,184,0.15)",
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.1)",
            zerolinecolor="rgba(148,163,184,0.15)",
        ),
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convertit une couleur hex en tuple RGBA string.

    Args:
        hex_color: Couleur hexadecimale (#RRGGBB).
        alpha: Opacite (0-1).

    Returns:
        String "(r, g, b, a)".
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"({r}, {g}, {b}, {alpha})"
