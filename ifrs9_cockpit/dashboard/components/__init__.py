"""Dashboard components package — split from components_dash.py.

Re-exports all public functions so that
``from ifrs9_cockpit.dashboard.components import build_header`` works.

Also re-exports legacy Streamlit-based render functions from the original
``components.py`` (now ``components/legacy.py``) for backward compatibility.
"""

# ── New Dash component builders (from components_dash.py split) ──
from ifrs9_cockpit.dashboard.components.cards import (
    build_kpi_row,
    build_classification_row,
    build_pe_score_card,
    build_credit_score_card,
)
from ifrs9_cockpit.dashboard.components.narrative import (
    build_ai_narrative_box,
    build_rst_results_panel,
)
from ifrs9_cockpit.dashboard.components.sidebar import (
    build_header,
    build_section_title,
    build_scenario_banner,
    build_collapsible_button,
    build_sidebar,
)
from ifrs9_cockpit.dashboard.components.arbitrage import (
    build_arbitrage_insight,
    build_layer2_allocation,
)

# ── Legacy Streamlit render functions (from original components.py) ──
# Only imported when Streamlit is available (not needed by Dash app).
try:
    import streamlit as st  # noqa: F401
    from ifrs9_cockpit.dashboard.components.legacy import (
        render_header,
        render_kpi_cards,
        render_insight_box,
        render_stage_badges,
        render_smart_insight_box,
        render_section_title,
        render_kpi_row,
        render_classification_row,
        render_ai_narrative_box,
    )
except (ImportError, ModuleNotFoundError):
    pass

__all__ = [
    # Dash builders
    "build_kpi_row",
    "build_classification_row",
    "build_pe_score_card",
    "build_credit_score_card",
    "build_ai_narrative_box",
    "build_rst_results_panel",
    "build_header",
    "build_section_title",
    "build_scenario_banner",
    "build_collapsible_button",
    "build_sidebar",
    "build_arbitrage_insight",
    "build_layer2_allocation",
]

# Extend __all__ with legacy names only if they were successfully imported
_LEGACY_NAMES = [
    "render_header", "render_kpi_cards", "render_insight_box",
    "render_stage_badges", "render_smart_insight_box", "render_section_title",
    "render_kpi_row", "render_classification_row", "render_ai_narrative_box",
]
__all__.extend(n for n in _LEGACY_NAMES if n in dir())
