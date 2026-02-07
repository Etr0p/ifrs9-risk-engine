"""Styles CSS custom pour le dashboard IFRS 9.

3 composants principaux avec design fintech moderne :
    1. KPI Cards — métriques clés avec icônes et tendances
    2. Insight Box — rapport CRO avec fond dégradé
    3. Badges — indicateurs de stage et sévérité
"""

from __future__ import annotations

from ifrs9_cockpit.config import DashboardConfig, DASHBOARD_CONFIG


def get_main_css(cfg: DashboardConfig = DASHBOARD_CONFIG) -> str:
    """Retourne le CSS global du dashboard.

    Args:
        cfg: Configuration du dashboard (couleurs, polices).

    Returns:
        CSS complet en string, prêt pour st.markdown().
    """
    return f"""
    <style>
    /* ─── GLOBAL ─── */
    .stApp {{
        background-color: {cfg.theme_bg_dark};
    }}

    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }}

    /* ─── HEADER ─── */
    .cockpit-header {{
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }}
    .cockpit-header h1 {{
        background: linear-gradient(135deg, {cfg.theme_primary}, {cfg.theme_secondary});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
    }}
    .cockpit-header p {{
        color: {cfg.theme_text_muted};
        font-size: 0.9rem;
        margin: 0;
    }}

    /* ─── KPI CARDS ─── */
    .kpi-container {{
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }}
    .kpi-card {{
        background: linear-gradient(145deg, {cfg.theme_bg_card}, #253348);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        flex: 1;
        transition: transform 0.2s, border-color 0.2s;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        border-color: {cfg.theme_primary};
    }}
    .kpi-label {{
        color: {cfg.theme_text_muted};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }}
    .kpi-value {{
        color: {cfg.theme_text};
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }}
    .kpi-sub {{
        font-size: 0.78rem;
        font-weight: 500;
    }}
    .kpi-sub.positive {{
        color: {cfg.theme_accent};
    }}
    .kpi-sub.negative {{
        color: #EF4444;
    }}
    .kpi-sub.neutral {{
        color: {cfg.theme_text_muted};
    }}

    /* ─── INSIGHT BOX (CRO Report) ─── */
    .insight-box {{
        background: linear-gradient(135deg,
            rgba(99, 102, 241, 0.12),
            rgba(139, 92, 246, 0.08),
            rgba(6, 214, 160, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-left: 4px solid {cfg.theme_primary};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }}
    .insight-box-header {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.8rem;
    }}
    .insight-box-header h3 {{
        color: {cfg.theme_text};
        font-size: 1rem;
        font-weight: 700;
        margin: 0;
    }}
    .insight-box-content {{
        color: {cfg.theme_text_muted};
        font-size: 0.85rem;
        line-height: 1.65;
    }}
    .insight-box-content .alert-line {{
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }}
    .insight-box-content .alert-line:last-child {{
        border-bottom: none;
    }}
    .insight-box-content .severity-CRITICAL {{
        color: #EF4444;
        font-weight: 700;
    }}
    .insight-box-content .severity-ALERT {{
        color: #F59E0B;
        font-weight: 600;
    }}
    .insight-box-content .severity-WARNING {{
        color: #F97316;
        font-weight: 600;
    }}
    .insight-box-content .severity-INFO {{
        color: {cfg.theme_accent};
        font-weight: 500;
    }}
    .insight-box-content .action-text {{
        color: {cfg.theme_primary};
        font-style: italic;
        font-size: 0.82rem;
    }}

    /* ─── BADGES ─── */
    .badge {{
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    .badge-stage1 {{
        background: rgba(6, 214, 160, 0.15);
        color: {cfg.theme_accent};
        border: 1px solid rgba(6, 214, 160, 0.3);
    }}
    .badge-stage2 {{
        background: rgba(249, 115, 22, 0.15);
        color: #F97316;
        border: 1px solid rgba(249, 115, 22, 0.3);
    }}
    .badge-stage3 {{
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }}
    .badge-info {{
        background: rgba(99, 102, 241, 0.15);
        color: {cfg.theme_primary};
        border: 1px solid rgba(99, 102, 241, 0.3);
    }}

    /* ─── SIDEBAR ─── */
    section[data-testid="stSidebar"] {{
        background-color: #0B1120;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }}
    section[data-testid="stSidebar"] .stSlider > div > div {{
        color: {cfg.theme_text_muted};
    }}

    /* ─── TABS ─── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {cfg.theme_bg_card};
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        color: {cfg.theme_text_muted};
        padding: 0.5rem 1rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {cfg.theme_primary}, {cfg.theme_secondary}) !important;
        color: white !important;
        border-color: transparent !important;
    }}

    /* ─── PLOTLY CHARTS ─── */
    .stPlotlyChart {{
        background-color: transparent;
    }}

    /* ─── SECTION TITLE ─── */
    .section-title {{
        color: {cfg.theme_text};
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid {cfg.theme_primary};
        display: inline-block;
    }}
    </style>
    """
