"""Configuration — dashboard UI, couleurs, features PE."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# DASHBOARD
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class DashboardConfig:
    """Configuration du dashboard Streamlit.

    Palette Steel Blue sur fond sombre, conforme au design system
    institutional defini dans dashboard-ui-ux-research-2026-02-10.md.

    Slider ranges conformes au PRD FR33.

    Attributes:
        page_title: Titre de la page.
        page_icon: Icone de la page.
        layout: Layout Streamlit ('wide' ou 'centered').
        theme_primary: Couleur primaire hex (Steel Blue).
        theme_secondary: Couleur secondaire hex (Steel Blue 800).
        theme_accent: Couleur d'accent hex (Emerald).
        theme_bg_dark: Fond sombre hex.
        theme_bg_card: Fond de carte hex.
        theme_text: Couleur texte principal hex.
        theme_text_muted: Couleur texte secondaire hex.
        stress_interest_rate_range: Range slider taux BCE (min_bp, max_bp, step_bp).
        stress_unemployment_range: Range slider chomage bipolaire (min, max, step).
        stress_gdp_range: Range slider PIB (min_pct, max_pct, step_pct).
        stress_hpi_range: Range slider HPI (min_pct, max_pct, step_pct).
        stress_inflation_range: Range slider inflation (min_pct, max_pct, step_pct).
    """

    page_title: str = "IFRS 9 Risk Cockpit"
    page_icon: str = "$"
    layout: str = "wide"
    # Palette Steel Blue sur fond sombre — WCAG AAA (7:1+ sur #0C1222)
    theme_primary: str = "#3B82F6"
    theme_primary_text: str = "#60A5FA"  # Blue 400 pour texte (8.2:1)
    theme_secondary: str = "#1E40AF"
    theme_accent: str = "#34D399"  # Emerald 400 (9.6:1)
    theme_bg_dark: str = "#0C1222"
    theme_bg_card: str = "#1E293B"
    theme_text: str = "#F8FAFC"
    theme_text_muted: str = "#A1B2C8"  # Slate clair (7.2:1)
    # Couleurs semantiques — WCAG AAA
    color_success: str = "#34D399"  # Emerald 400 (9.6:1)
    color_warning: str = "#FBBF24"  # Amber 400 (11.2:1)
    color_danger: str = "#F87171"  # Red 400 (7.5:1)
    color_info: str = "#22D3EE"  # Cyan 400 (10.1:1)
    # Slider ranges (FR33) — taux en bp, chomage bipolaire, reste en pct
    stress_interest_rate_range: Tuple[float, float, float] = (-400.0, 650.0, 25.0)
    stress_unemployment_range: Tuple[float, float, float] = (-7.0, 7.0, 1.0)
    stress_gdp_range: Tuple[float, float, float] = (-8.0, 8.0, 0.5)
    stress_hpi_range: Tuple[float, float, float] = (-30.0, 20.0, 1.0)
    stress_inflation_range: Tuple[float, float, float] = (-2.0, 8.0, 0.5)


DASHBOARD_CONFIG = DashboardConfig()

# Palette de donnees pour les graphiques Plotly (6 couleurs, dans cet ordre)
CHART_COLORS: Tuple[str, ...] = (
    "#3B82F6",  # Steel Blue (primary)
    "#34D399",  # Emerald 400 (success) — WCAG AAA
    "#FBBF24",  # Amber 400 (warning) — WCAG AAA
    "#F87171",  # Red 400 (danger) — WCAG AAA
    "#22D3EE",  # Cyan 400 (info) — WCAG AAA
    "#8B5CF6",  # Purple (accent)
)

# Palette CVD-safe IBM (daltoniens) — pour séries de données multi-catégories
CVD_SAFE_COLORS: Tuple[str, ...] = (
    "#648FFF",  # Blue
    "#785EF0",  # Purple
    "#DC267F",  # Magenta
    "#FE6100",  # Orange
    "#FFB000",  # Gold
)

STAGE_COLORS: Dict[int, str] = {
    1: "#34D399",  # Emerald 400 — WCAG AAA
    2: "#FBBF24",  # Amber 400 — WCAG AAA
    3: "#F87171",  # Red 400 — WCAG AAA
}

PE_CATEGORY_COLORS: Dict[str, str] = {
    "Performing": "#34D399",
    "Watchlist": "#FBBF24",
    "Distressed": "#F87171",
}

