"""Styles CSS pour le dashboard Dash -- IFRS 9 Risk Cockpit.

Adapte les styles Streamlit (styles.py) pour Dash :
    - Suppression des selecteurs Streamlit (.stApp, .stTabs, etc.)
    - Ajout de styles pour les composants Dash natifs (dbc.Modal, DataTable, etc.)
    - Nouvelles sections : score cards, modals, arbitrage, collapse, sidebar sliders

La fonction get_dash_css() retourne une chaine CSS brute (sans balises <style>).
Dash injecte le CSS via app.index_string ou un fichier assets/.

Palette Steel Blue sur fond sombre, conforme WCAG AAA (7:1+ sur #0C1222).
"""

from __future__ import annotations

from typing import Optional

from ifrs9_cockpit.config import DashboardConfig, DASHBOARD_CONFIG


def get_dash_css(cfg: Optional[DashboardConfig] = None) -> str:
    """Retourne le CSS global du dashboard Dash.

    Args:
        cfg: Configuration du dashboard (couleurs, polices).
            Defaut : DASHBOARD_CONFIG depuis config.py.

    Returns:
        CSS complet en string brute (sans balises <style>).
    """
    if cfg is None:
        cfg = DASHBOARD_CONFIG

    return f"""
/* ═══════════════════════════════════════════════
   0. DASH 4.0 DESIGN TOKENS (override light defaults)
   Dash 4.0 defaults: white bg (#fff), dark text — must override for dark theme.
   We set variables AND direct property overrides for maximum reliability.
   ═══════════════════════════════════════════════ */
:root {{
    --Dash-Fill-Inverse-Strong: {cfg.theme_bg_card};
    --Dash-Fill-Interactive-Strong: {cfg.theme_primary};
    --Dash-Fill-Interactive-Weak: rgba(59, 130, 246, 0.15);
    --Dash-Fill-Disabled: rgba(99, 102, 241, 0.12);
    --Dash-Fill-Primary-Active: rgba(59, 130, 246, 0.15);
    --Dash-Fill-Primary-Hover: rgba(59, 130, 246, 0.08);
    --Dash-Text-Strong: {cfg.theme_text};
    --Dash-Text-Primary: {cfg.theme_text};
    --Dash-Text-Weak: {cfg.theme_text_muted};
    --Dash-Text-Disabled: {cfg.theme_text_muted};
    --Dash-Stroke-Strong: rgba(99, 102, 241, 0.25);
    --Dash-Stroke-Weak: rgba(99, 102, 241, 0.12);
    --Dash-Shading-Strong: rgba(0, 0, 0, 0.5);
    --Dash-Shading-Weak: rgba(0, 0, 0, 0.3);
    --Dash-Spacing: 4px;
}}

/* ═══════════════════════════════════════════════
   1. GLOBAL
   ═══════════════════════════════════════════════ */
*,
*::before,
*::after {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    color: {cfg.theme_text};
    background-color: {cfg.theme_bg_dark};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

#app-container {{
    background-color: {cfg.theme_bg_dark};
    min-height: 100vh;
}}

a {{
    color: {cfg.theme_primary};
    text-decoration: none;
}}

a:hover {{
    color: {cfg.color_info};
    text-decoration: underline;
}}


/* ═══════════════════════════════════════════════
   2. SIDEBAR (fixed left panel, 280px)
   ═══════════════════════════════════════════════ */
.sidebar {{
    position: fixed;
    top: 0;
    left: 0;
    width: 280px;
    height: 100vh;
    background-color: #0B1120;
    border-right: 1px solid rgba(99, 102, 241, 0.15);
    padding: 1.5rem 1rem;
    overflow-y: auto;
    z-index: 100;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}}

.sidebar-title {{
    color: {cfg.theme_text};
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(99, 102, 241, 0.25);
}}

.sidebar-label {{
    color: {cfg.theme_text_muted};
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.25rem;
    margin-top: 0.5rem;
}}

.sidebar-section {{
    margin-bottom: 0.75rem;
}}

.sidebar-divider {{
    border: none;
    border-top: 1px solid rgba(99, 102, 241, 0.12);
    margin: 0.75rem 0;
}}


/* ═══════════════════════════════════════════════
   3. MAIN CONTENT (offset by sidebar width)
   ═══════════════════════════════════════════════ */
.main-content {{
    margin-left: 280px;
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1400px;
    min-height: 100vh;
}}


/* ═══════════════════════════════════════════════
   4. HEADER (gradient text)
   ═══════════════════════════════════════════════ */
.cockpit-header {{
    text-align: center;
    padding: 1rem 0 0.5rem 0;
}}

.cockpit-header h1 {{
    background: linear-gradient(135deg, {cfg.theme_primary}, {cfg.theme_secondary});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
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


/* ═══════════════════════════════════════════════
   5. KPI CARDS
   ═══════════════════════════════════════════════ */
.kpi-container {{
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}}

.kpi-card {{
    background: linear-gradient(145deg, {cfg.theme_bg_card}, #253348);
    border: 1px solid rgba(99, 102, 241, 0.20);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    flex: 1;
    min-width: 180px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4), 0 1px 2px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}}

.kpi-card:hover {{
    transform: translateY(-2px);
    border-color: {cfg.theme_primary};
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.35), 0 2px 4px rgba(0, 0, 0, 0.25);
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
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.3;
    margin-bottom: 0.25rem;
}}

.kpi-sub {{
    font-size: 0.78rem;
    font-weight: 500;
}}

.kpi-sub.positive {{
    color: {cfg.theme_accent};
}}

.kpi-sub.negative {{
    color: {cfg.color_danger};
}}

.kpi-sub.neutral {{
    color: {cfg.theme_text_muted};
}}


/* ═══════════════════════════════════════════════
   6. INSIGHT BOX (CRO Report)
   ═══════════════════════════════════════════════ */
.insight-box {{
    background: linear-gradient(135deg,
        rgba(99, 102, 241, 0.12),
        rgba(139, 92, 246, 0.06));
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-left: 4px solid {cfg.theme_primary};
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.35), 0 2px 4px rgba(0, 0, 0, 0.25);
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
    color: {cfg.color_danger};
    font-weight: 700;
}}

.insight-box-content .severity-ALERT {{
    color: {cfg.color_warning};
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

.insight-box-content .cro-summary {{
    color: {cfg.theme_text};
    font-size: 0.88rem;
    line-height: 1.6;
    padding-bottom: 0.6rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
}}

.insight-box-content .cro-section-label {{
    color: {cfg.theme_text_muted};
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-top: 0.6rem;
    margin-bottom: 0.3rem;
}}

.insight-box-content .cro-finding {{
    color: {cfg.theme_text_muted};
    font-size: 0.84rem;
    line-height: 1.6;
    padding: 0.5rem 0.6rem;
    border-left: 2px solid rgba(99, 102, 241, 0.25);
    margin-bottom: 0.4rem;
    text-align: justify;
}}

.insight-box-content .cro-recommendation {{
    color: {cfg.theme_text};
    font-size: 0.84rem;
    line-height: 1.6;
    padding: 0.5rem 0.6rem;
    border-left: 2px solid rgba(6, 214, 160, 0.35);
    margin-bottom: 0.4rem;
    text-align: justify;
}}


/* ═══════════════════════════════════════════════
   7. BADGES (stage and severity indicators)
   ═══════════════════════════════════════════════ */
.badge {{
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
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
    background: rgba(248, 113, 113, 0.15);
    color: {cfg.color_danger};
    border: 1px solid rgba(248, 113, 113, 0.3);
}}

.badge-info {{
    background: rgba(99, 102, 241, 0.15);
    color: {cfg.theme_primary};
    border: 1px solid rgba(99, 102, 241, 0.3);
}}

.badge-success {{
    background: rgba(6, 214, 160, 0.15);
    color: {cfg.color_success};
    border: 1px solid rgba(6, 214, 160, 0.3);
}}

.badge-warning {{
    background: rgba(251, 191, 36, 0.15);
    color: {cfg.color_warning};
    border: 1px solid rgba(251, 191, 36, 0.3);
}}

.badge-danger {{
    background: rgba(248, 113, 113, 0.15);
    color: {cfg.color_danger};
    border: 1px solid rgba(248, 113, 113, 0.3);
}}


/* ═══════════════════════════════════════════════
   8. SECTION TITLES
   ═══════════════════════════════════════════════ */
h2.section-title {{
    color: {cfg.theme_text};
    font-size: 1.125rem;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid {cfg.theme_primary};
    display: inline-block;
}}


/* ═══════════════════════════════════════════════
   9. SCORE CARDS (clickable summary, credit & PE)
   ═══════════════════════════════════════════════ */
.score-card {{
    background: linear-gradient(145deg, {cfg.theme_bg_card}, #1A2332);
    border: 1px solid rgba(99, 102, 241, 0.20);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    cursor: pointer;
    transition: transform 0.2s ease, border-color 0.2s ease,
                box-shadow 0.25s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.35);
    position: relative;
    overflow: hidden;
}}

.score-card::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, {cfg.theme_primary}, {cfg.theme_secondary});
    opacity: 0;
    transition: opacity 0.25s ease;
}}

.score-card:hover {{
    transform: translateY(-3px);
    border-color: {cfg.theme_primary};
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4), 0 4px 8px rgba(0, 0, 0, 0.3);
}}

.score-card:hover::before {{
    opacity: 1;
}}

.score-card:active {{
    transform: translateY(-1px);
}}

.score-card-title {{
    color: {cfg.theme_text_muted};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    font-weight: 600;
    margin-bottom: 0.5rem;
}}

.score-card-value {{
    color: {cfg.theme_text};
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}}

.score-card-detail {{
    color: {cfg.theme_text_muted};
    font-size: 0.8rem;
    line-height: 1.5;
}}

.score-card-arrow {{
    position: absolute;
    right: 1.25rem;
    top: 50%;
    transform: translateY(-50%);
    color: {cfg.theme_text_muted};
    font-size: 1.2rem;
    opacity: 0.4;
    transition: opacity 0.2s ease, transform 0.2s ease;
}}

.score-card:hover .score-card-arrow {{
    opacity: 0.8;
    transform: translateY(-50%) translateX(3px);
}}


/* ═══════════════════════════════════════════════
   10. MODAL (detail overlays with blur backdrop)
   ═══════════════════════════════════════════════ */
@keyframes modal-slide-up {{
    from {{
        opacity: 0;
        transform: translateY(30px) scale(0.97);
    }}
    to {{
        opacity: 1;
        transform: translateY(0) scale(1);
    }}
}}

.modal-backdrop.show {{
    background: rgba(0, 0, 0, 0.6) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}}

.modal-content {{
    background: {cfg.theme_bg_card};
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    width: 90vw;
    max-width: 1100px;
    max-height: 85vh;
    overflow-y: auto;
    padding: 2rem;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 8px 24px rgba(0, 0, 0, 0.4);
    animation: modal-slide-up 0.3s ease-out;
}}

.modal-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(99, 102, 241, 0.15);
}}

.modal-header h2 {{
    color: {cfg.theme_text};
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
}}

.modal-close {{
    background: none;
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 8px;
    color: {cfg.theme_text_muted};
    font-size: 1.2rem;
    cursor: pointer;
    padding: 0.3rem 0.6rem;
    transition: background-color 0.15s ease, color 0.15s ease;
    line-height: 1;
}}

.modal-close:hover {{
    background-color: rgba(248, 113, 113, 0.15);
    color: {cfg.color_danger};
    border-color: {cfg.color_danger};
}}

.modal-body {{
    color: {cfg.theme_text_muted};
    font-size: 0.88rem;
    line-height: 1.6;
}}

/* Override for dbc.Modal — fill the area right of the sidebar */
.modal {{
    padding-left: 280px !important;
}}

.modal-dialog {{
    max-width: calc(100% - 2rem) !important;
    width: calc(100% - 2rem) !important;
    margin: 1.75rem auto !important;
}}

.modal .modal-content {{
    background: {cfg.theme_bg_card} !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 16px !important;
    color: {cfg.theme_text} !important;
    animation: modal-slide-up 0.3s ease-out;
}}

.modal .modal-header {{
    border-bottom: 1px solid rgba(99, 102, 241, 0.15) !important;
}}

.modal .modal-header .btn-close {{
    filter: invert(1);
}}

.modal .modal-body {{
    color: {cfg.theme_text_muted} !important;
    padding: 1.5rem 2rem;
    scrollbar-width: none;           /* Firefox */
    -ms-overflow-style: none;        /* IE/Edge */
}}

.modal .modal-body::-webkit-scrollbar {{
    display: none;                   /* Chrome/Safari */
}}

/* ═══════════════════════════════════════════════
   11. ARBITRAGE SECTION (drill-down sectoriel)
   ═══════════════════════════════════════════════ */
.arbitrage-section {{
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 1.5rem 0;
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.5), rgba(12, 18, 34, 0.5));
}}

.arbitrage-section-title {{
    color: {cfg.theme_text};
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.arbitrage-detail {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}}

.arbitrage-metric {{
    background: {cfg.theme_bg_card};
    border-radius: 8px;
    padding: 0.75rem 1rem;
    border: 1px solid rgba(99, 102, 241, 0.12);
}}

.arbitrage-metric-label {{
    color: {cfg.theme_text_muted};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}}

.arbitrage-metric-value {{
    color: {cfg.theme_text};
    font-size: 1.15rem;
    font-weight: 700;
    margin-top: 0.25rem;
}}


/* ═══════════════════════════════════════════════
   12. COLLAPSE SECTIONS (progressive disclosure)
   ═══════════════════════════════════════════════ */
.collapse-toggle {{
    background: linear-gradient(145deg, {cfg.theme_bg_card}, #1A2332);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 10px;
    color: {cfg.theme_text};
    font-size: 0.9rem;
    font-weight: 600;
    padding: 0.75rem 1.25rem;
    cursor: pointer;
    width: 100%;
    text-align: left;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: background-color 0.15s ease, border-color 0.2s ease;
    margin-top: 1rem;
}}

.collapse-toggle:hover {{
    border-color: {cfg.theme_primary};
    background: linear-gradient(145deg, #253348, {cfg.theme_bg_card});
}}

.collapse-toggle:focus {{
    outline: 2px solid {cfg.theme_primary};
    outline-offset: 2px;
}}

.collapse-toggle-icon {{
    color: {cfg.theme_text_muted};
    font-size: 0.8rem;
    transition: transform 0.25s ease;
}}

.collapse-toggle.open .collapse-toggle-icon {{
    transform: rotate(180deg);
}}

.collapse-body {{
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.35s ease-out, opacity 0.25s ease-out;
    opacity: 0;
}}

.collapse-body.open {{
    max-height: 5000px;
    opacity: 1;
    transition: max-height 0.45s ease-in, opacity 0.3s ease-in;
}}

.collapse-inner {{
    padding: 1.25rem 0.5rem;
}}


/* ═══════════════════════════════════════════════
   13. SIDEBAR SLIDERS (dark theme)
   ═══════════════════════════════════════════════ */
/* Dash Core Components sliders */
.rc-slider {{
    margin: 0.5rem 0 1rem 0;
}}

.rc-slider-rail {{
    background-color: rgba(99, 102, 241, 0.15) !important;
    height: 4px !important;
}}

.rc-slider-track {{
    background-color: {cfg.theme_primary} !important;
    height: 4px !important;
}}

.rc-slider-handle {{
    border-color: {cfg.theme_primary} !important;
    background-color: {cfg.theme_bg_card} !important;
    width: 16px !important;
    height: 16px !important;
    margin-top: -6px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4) !important;
    opacity: 1 !important;
}}

.rc-slider-handle:hover {{
    border-color: {cfg.color_info} !important;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2) !important;
}}

.rc-slider-handle:active {{
    box-shadow: 0 0 0 6px rgba(59, 130, 246, 0.3) !important;
}}

.rc-slider-mark-text {{
    color: {cfg.theme_text_muted} !important;
    font-size: 0.7rem !important;
}}

.rc-slider-dot {{
    border-color: rgba(99, 102, 241, 0.25) !important;
    background-color: {cfg.theme_bg_dark} !important;
}}

/* Slider tooltip (value popup on hover) */
.rc-slider-tooltip {{
    z-index: 1000 !important;
}}

.rc-slider-tooltip-inner {{
    background-color: {cfg.theme_bg_card} !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    color: {cfg.theme_text} !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 0.2rem 0.5rem !important;
    border-radius: 6px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    min-width: 30px !important;
    text-align: center !important;
}}

.rc-slider-tooltip-arrow {{
    border-top-color: {cfg.theme_bg_card} !important;
}}

/* ─── Dash 4.0 dcc.Dropdown — dark theme (direct property overrides) ─── */
/* Variables in :root may be overwritten by Dash JS at runtime.            */
/* Direct !important on properties guarantees dark theme regardless.       */

/* Dropdown wrapper — dark background, light text */
.dash-dropdown {{
    background: {cfg.theme_bg_card} !important;
    color: {cfg.theme_text} !important;
    border-radius: 6px !important;
}}

/* Trigger (closed state — shows selected value) */
.dash-dropdown-trigger {{
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 6px !important;
    background: {cfg.theme_bg_card} !important;
}}

.dash-dropdown-trigger:hover {{
    border-color: {cfg.theme_primary} !important;
}}

/* Selected value text — THE key fix: white on dark */
.dash-dropdown-value {{
    color: {cfg.theme_text} !important;
    font-size: 0.82rem !important;
}}

/* Placeholder */
.dash-dropdown-placeholder {{
    color: {cfg.theme_text_muted} !important;
}}

/* Trigger arrow icon */
.dash-dropdown-trigger-icon {{
    color: {cfg.theme_text_muted} !important;
}}

.dash-dropdown-trigger-icon svg {{
    fill: {cfg.theme_text_muted} !important;
}}

/* Dropdown menu (popover — may render outside .sidebar via portal) */
.dash-dropdown-content {{
    background: #1A2332 !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 6px !important;
    z-index: 200 !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
}}

/* Options in the menu */
.dash-dropdown-option {{
    color: {cfg.theme_text} !important;
    background: #1A2332 !important;
    font-size: 0.82rem !important;
}}

.dash-dropdown-option:hover {{
    background: rgba(59, 130, 246, 0.2) !important;
}}

/* Search input */
.dash-dropdown-search {{
    color: {cfg.theme_text} !important;
    background: transparent !important;
}}

.dash-dropdown-search-container {{
    background: {cfg.theme_bg_card} !important;
    border-color: rgba(99, 102, 241, 0.25) !important;
}}

.dash-dropdown-search-icon {{
    color: {cfg.theme_text_muted} !important;
}}

/* Clear button */
.dash-dropdown-clear {{
    color: {cfg.theme_text_muted} !important;
}}

.dash-dropdown-clear:hover {{
    color: {cfg.color_danger} !important;
}}

/* Multi-select value items */
.dash-dropdown-value-item {{
    background: rgba(59, 130, 246, 0.15) !important;
    color: {cfg.theme_text} !important;
    border-radius: 4px !important;
}}

/* Actions bar */
.dash-dropdown-actions {{
    border-top-color: rgba(99, 102, 241, 0.15) !important;
}}

.dash-dropdown-action-button {{
    color: {cfg.theme_text_muted} !important;
}}

.dash-dropdown-action-button:hover {{
    color: {cfg.theme_text} !important;
}}


/* ═══════════════════════════════════════════════
   14. PLOTLY CHART CONTAINERS
   ═══════════════════════════════════════════════ */
.js-plotly-plot,
.plotly {{
    background-color: transparent !important;
}}

.js-plotly-plot .plotly .main-svg {{
    background: transparent !important;
}}

.chart-container {{
    background: transparent;
    border-radius: 12px;
    margin: 0.75rem 0;
}}


/* ═══════════════════════════════════════════════
   15. DATATABLE (Dash DataTable dark theme)
   ═══════════════════════════════════════════════ */
.dash-table-container {{
    border-radius: 10px;
    overflow: hidden;
}}

.dash-spreadsheet-container {{
    background-color: {cfg.theme_bg_card} !important;
}}

.dash-spreadsheet-container .dash-spreadsheet-inner th {{
    background-color: #0F1A2E !important;
    color: {cfg.theme_text} !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    border-bottom: 2px solid {cfg.theme_primary} !important;
    padding: 0.6rem 0.8rem !important;
}}

.dash-spreadsheet-container .dash-spreadsheet-inner td {{
    background-color: {cfg.theme_bg_card} !important;
    color: {cfg.theme_text_muted} !important;
    font-size: 0.82rem !important;
    border-bottom: 1px solid rgba(99, 102, 241, 0.08) !important;
    padding: 0.5rem 0.8rem !important;
}}

.dash-spreadsheet-container .dash-spreadsheet-inner tr:hover td {{
    background-color: rgba(59, 130, 246, 0.08) !important;
}}

.dash-spreadsheet-container .dash-spreadsheet-inner td.cell--selected {{
    background-color: rgba(59, 130, 246, 0.15) !important;
    border: 1px solid {cfg.theme_primary} !important;
}}

.dash-spreadsheet-container .previous-next-container {{
    background-color: {cfg.theme_bg_card} !important;
    color: {cfg.theme_text_muted} !important;
}}

.dash-spreadsheet-container .previous-next-container button {{
    color: {cfg.theme_primary} !important;
}}

/* Filter row */
.dash-spreadsheet-container .dash-filter {{
    background-color: #0F1A2E !important;
    color: {cfg.theme_text} !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
}}


/* ═══════════════════════════════════════════════
   16. TOAST NOTIFICATIONS
   ═══════════════════════════════════════════════ */
.toast-container {{
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    z-index: 2000;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}}

.toast {{
    background: {cfg.theme_bg_card};
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-left: 4px solid {cfg.theme_primary};
    border-radius: 10px;
    padding: 0.75rem 1.25rem;
    min-width: 280px;
    max-width: 420px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
    animation: toast-slide-in 0.3s ease-out;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}}

.toast-success {{
    border-left-color: {cfg.color_success};
}}

.toast-warning {{
    border-left-color: {cfg.color_warning};
}}

.toast-error {{
    border-left-color: {cfg.color_danger};
}}

.toast-message {{
    color: {cfg.theme_text};
    font-size: 0.85rem;
    font-weight: 500;
    flex: 1;
}}

.toast-close {{
    background: none;
    border: none;
    color: {cfg.theme_text_muted};
    cursor: pointer;
    font-size: 1rem;
    padding: 0;
    line-height: 1;
}}

.toast-close:hover {{
    color: {cfg.theme_text};
}}

@keyframes toast-slide-in {{
    from {{
        opacity: 0;
        transform: translateX(30px);
    }}
    to {{
        opacity: 1;
        transform: translateX(0);
    }}
}}


/* ═══════════════════════════════════════════════
   17. SCROLLBAR (thin, dark theme)
   ═══════════════════════════════════════════════ */
::-webkit-scrollbar {{
    width: 8px;
    height: 8px;
}}

::-webkit-scrollbar-track {{
    background: {cfg.theme_bg_dark};
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb {{
    background: rgba(99, 102, 241, 0.25);
    border-radius: 4px;
    border: 2px solid {cfg.theme_bg_dark};
}}

::-webkit-scrollbar-thumb:hover {{
    background: rgba(99, 102, 241, 0.4);
}}

/* Firefox scrollbar */
* {{
    scrollbar-width: thin;
    scrollbar-color: rgba(99, 102, 241, 0.25) {cfg.theme_bg_dark};
}}


/* ═══════════════════════════════════════════════
   UTILITY CLASSES
   ═══════════════════════════════════════════════ */
.text-muted {{
    color: {cfg.theme_text_muted};
}}

.text-success {{
    color: {cfg.color_success};
}}

.text-warning {{
    color: {cfg.color_warning};
}}

.text-danger {{
    color: {cfg.color_danger};
}}

.text-info {{
    color: {cfg.color_info};
}}

.text-primary {{
    color: {cfg.theme_primary};
}}

.mt-1 {{ margin-top: 0.5rem; }}
.mt-2 {{ margin-top: 1rem; }}
.mt-3 {{ margin-top: 1.5rem; }}
.mb-1 {{ margin-bottom: 0.5rem; }}
.mb-2 {{ margin-bottom: 1rem; }}
.mb-3 {{ margin-bottom: 1.5rem; }}
.gap-1 {{ gap: 0.5rem; }}
.gap-2 {{ gap: 1rem; }}

.flex-row {{
    display: flex;
    flex-direction: row;
    align-items: center;
}}

.flex-col {{
    display: flex;
    flex-direction: column;
}}

.flex-wrap {{
    flex-wrap: wrap;
}}

.flex-1 {{
    flex: 1;
}}

/* Scenario context banner */
.scenario-banner {{
    background: linear-gradient(135deg,
        rgba(59, 130, 246, 0.08),
        rgba(139, 92, 246, 0.04));
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 10px;
    padding: 0.6rem 1.25rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
    font-size: 0.82rem;
    color: {cfg.theme_text_muted};
}}

.scenario-banner-label {{
    color: {cfg.theme_text};
    font-weight: 700;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

.scenario-banner-param {{
    color: {cfg.theme_text_muted};
    font-size: 0.82rem;
}}

/* Incoherence alerts */
.incoherence-alert {{
    background: rgba(248, 113, 113, 0.08);
    border: 1px solid rgba(248, 113, 113, 0.25);
    border-left: 3px solid {cfg.color_danger};
    border-radius: 8px;
    padding: 0.5rem 1rem;
    margin-bottom: 0.5rem;
    color: {cfg.color_danger};
    font-size: 0.82rem;
    font-weight: 500;
}}

/* Loading overlay */
.loading-overlay {{
    position: fixed;
    top: 0;
    left: 280px;
    width: calc(100vw - 280px);
    height: 100vh;
    background: rgba(12, 18, 34, 0.85);
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
    z-index: 500;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1rem;
}}

.loading-spinner {{
    width: 40px;
    height: 40px;
    border: 3px solid rgba(99, 102, 241, 0.15);
    border-top-color: {cfg.theme_primary};
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}}

@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}

.loading-text {{
    color: {cfg.theme_text_muted};
    font-size: 0.88rem;
    font-weight: 500;
}}

/* ARIA focus indicators */
:focus-visible {{
    outline: 2px solid {cfg.theme_primary};
    outline-offset: 2px;
}}

/* Risk appetite symbols (CVD accessible) */
.risk-vert {{
    color: {cfg.color_success};
    font-weight: 700;
}}

.risk-ambre {{
    color: {cfg.color_warning};
    font-weight: 700;
}}

.risk-rouge {{
    color: {cfg.color_danger};
    font-weight: 700;
}}
"""
