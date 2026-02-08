"""Composants UI réutilisables pour le dashboard Streamlit.

Fournit les briques de construction du layout :
    - Header cockpit avec titre dégradé
    - KPI cards (ECL, PD, Stage 2%, Coverage)
    - Insight box pour le rapport Virtual CRO
    - Badges de stage et sévérité
"""

from __future__ import annotations

import streamlit as st
from typing import Any, Dict, List, Optional

from ifrs9_cockpit.analytics.virtual_cro import CROAlert
from ifrs9_cockpit.utils.helpers import format_euro, format_pct


def render_header() -> None:
    """Affiche le header du cockpit avec titre dégradé."""
    st.markdown(
        """
        <div class="cockpit-header">
            <h1>IFRS 9 Risk Cockpit</h1>
            <p>Moteur ECL &bull; Virtual CRO &bull; Stress Testing</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(
    ecl_total: float,
    pd_mean: float,
    stage2_pct: float,
    coverage: float,
    ecl_previous: Optional[float] = None,
) -> None:
    """Affiche les 4 KPI cards en haut du dashboard.

    Args:
        ecl_total: ECL pondéré total en euros.
        pd_mean: PD moyenne du portefeuille.
        stage2_pct: Part du portefeuille en Stage 2.
        coverage: Ratio ECL / EAD.
        ecl_previous: ECL de la période précédente (pour la tendance).
    """
    # Calcul de la tendance ECL
    if ecl_previous and ecl_previous > 0:
        ecl_delta = (ecl_total - ecl_previous) / ecl_previous
        ecl_trend_class = "negative" if ecl_delta > 0 else "positive"
        ecl_trend_text = f"{'+'if ecl_delta > 0 else ''}{ecl_delta:.1%} vs période préc."
    else:
        ecl_trend_class = "neutral"
        ecl_trend_text = "Période initiale"

    # Classification Stage 2
    if stage2_pct > 0.20:
        stage2_class = "negative"
        stage2_comment = "Au-dessus du seuil"
    elif stage2_pct > 0.10:
        stage2_class = "neutral"
        stage2_comment = "Sous surveillance"
    else:
        stage2_class = "positive"
        stage2_comment = "Dans les normes"

    # Classification Coverage
    if coverage > 0.05:
        cov_class = "negative"
        cov_comment = "Provisionnement élevé"
    elif coverage > 0.02:
        cov_class = "neutral"
        cov_comment = "Niveau standard"
    else:
        cov_class = "positive"
        cov_comment = "Provisionnement faible"

    html = f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">ECL Pondéré</div>
            <div class="kpi-value">{format_euro(ecl_total)}</div>
            <div class="kpi-sub {ecl_trend_class}">{ecl_trend_text}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">PD Moyenne</div>
            <div class="kpi-value">{format_pct(pd_mean)}</div>
            <div class="kpi-sub neutral">Portefeuille global</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Stage 2</div>
            <div class="kpi-value">{format_pct(stage2_pct, 1)}</div>
            <div class="kpi-sub {stage2_class}">{stage2_comment}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Coverage Ratio</div>
            <div class="kpi-value">{format_pct(coverage)}</div>
            <div class="kpi-sub {cov_class}">{cov_comment}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_insight_box(alerts: List[CROAlert]) -> None:
    """Affiche l'insight box avec le rapport Virtual CRO.

    Args:
        alerts: Liste des alertes générées par le VirtualCRO.
    """
    # Construire le contenu HTML des alertes
    if not alerts:
        content_html = '<div class="alert-line severity-INFO">Aucune alerte active. Portefeuille nominal.</div>'
    else:
        alert_lines = []
        for alert in alerts:
            icon = _severity_icon_html(alert.severity)
            line = (
                f'<div class="alert-line">'
                f'<span class="severity-{alert.severity}">{icon} [{alert.severity}] {alert.title}</span>'
                f'<br/>{alert.message}'
            )
            if alert.action:
                line += f'<br/><span class="action-text">&rarr; {alert.action}</span>'
            line += '</div>'
            alert_lines.append(line)
        content_html = "\n".join(alert_lines)

    # Déterminer le titre selon la sévérité max
    severities = [a.severity for a in alerts] if alerts else ["INFO"]
    if "CRITICAL" in severities:
        header_icon = "\U0001f6a8"
        header_text = "Virtual CRO — Alertes Critiques"
    elif "ALERT" in severities:
        header_icon = "\U0001f534"
        header_text = "Virtual CRO — Alertes Actives"
    elif "WARNING" in severities:
        header_icon = "\u26a0\ufe0f"
        header_text = "Virtual CRO — Points de Vigilance"
    else:
        header_icon = "\u2705"
        header_text = "Virtual CRO — Situation Nominale"

    html = f"""
    <div class="insight-box">
        <div class="insight-box-header">
            <span style="font-size: 1.2rem;">{header_icon}</span>
            <h3>{header_text}</h3>
        </div>
        <div class="insight-box-content">
            {content_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_stage_badges(stage_counts: Dict[int, int], total: int) -> None:
    """Affiche les badges de répartition des stages.

    Args:
        stage_counts: Dictionnaire {stage: count}.
        total: Nombre total de clients.
    """
    badges_html = ""
    badge_classes = {1: "badge-stage1", 2: "badge-stage2", 3: "badge-stage3"}

    for stage in [1, 2, 3]:
        count = stage_counts.get(stage, 0)
        pct = count / total if total > 0 else 0
        css_class = badge_classes[stage]
        badges_html += (
            f'<span class="badge {css_class}">'
            f'Stage {stage}: {count:,} ({pct:.1%})'
            f'</span>&nbsp;&nbsp;'
        )

    st.markdown(badges_html, unsafe_allow_html=True)


def render_smart_insight_box(briefing: Dict[str, Any]) -> None:
    """Affiche l'insight box avec analyse CRO en prose professionnelle.

    Remplace les alertes de seuil par un briefing exécutif
    produit par le LocalCROAnalyst : diagnostic narratif,
    constats avec indicateurs de sévérité colorés, et
    préconisations avec priorité.

    Args:
        briefing: Dictionnaire du LocalCROAnalyst.generate_executive_briefing().
    """
    risk_level = briefing["risk_level"]
    risk_score = briefing["risk_score"]
    risk_color = briefing["risk_color"]
    diagnostic = briefing["diagnostic"]
    findings = briefing["findings"]
    recommendations = briefing["recommendations"]

    # Header icon & color
    color_map = {
        "rouge": ("#EF4444", "\U0001f6a8"),
        "orange": ("#F59E0B", "\u26a0\ufe0f"),
        "vert": ("#06D6A0", "\u2705"),
    }
    accent_color, header_icon = color_map.get(
        risk_color, ("#94A3B8", "\u2753")
    )

    border_color = accent_color

    # Severity color mapping
    sev_colors = {
        "CRITICAL": "#EF4444",
        "ALERT": "#F59E0B",
        "WARNING": "#F97316",
        "INFO": "#06D6A0",
    }

    # Priority color mapping
    prio_colors = {
        "HAUTE": "#EF4444",
        "MOYENNE": "#F59E0B",
        "STANDARD": "#06D6A0",
        "INFO": "#06D6A0",
    }

    # Findings HTML (prose paragraphs with colored severity prefix)
    findings_html = ""
    if findings:
        findings_lines = []
        for f in findings:
            sev = f["severity"]
            color = sev_colors.get(sev, "#94A3B8")
            findings_lines.append(
                f'<div class="cro-finding">'
                f'<span style="color:{color};font-weight:700;">'
                f'({sev})</span> {f["text"]}</div>'
            )
        findings_html = (
            '<div class="cro-section-label">CONSTATS</div>'
            + "".join(findings_lines)
        )

    # Recommendations HTML (prose paragraphs with colored priority prefix)
    recs_html = ""
    if recommendations:
        recs_lines = []
        for r in recommendations:
            prio = r["priority"]
            color = prio_colors.get(prio, "#94A3B8")
            recs_lines.append(
                f'<div class="cro-recommendation">'
                f'<span style="color:{color};font-weight:700;">'
                f'({prio})</span> {r["text"]}</div>'
            )
        recs_html = (
            '<div class="cro-section-label">PRECONISATIONS</div>'
            + "".join(recs_lines)
        )

    # Build HTML without indentation to avoid Markdown code blocks
    body_parts = [f'<div class="cro-summary">{diagnostic}</div>']
    if findings_html:
        body_parts.append(findings_html)
    if recs_html:
        body_parts.append(recs_html)
    body = "".join(body_parts)

    html = (
        f'<div class="insight-box" style="border-left-color:{border_color};">'
        f'<div class="insight-box-header">'
        f'<span style="font-size:1.2rem;">{header_icon}</span>'
        f'<h3>Analyse CRO &mdash; Niveau de risque '
        f'<span style="color:{accent_color};">{risk_level}</span> '
        f'<span style="color:#94A3B8;font-size:0.8rem;font-weight:400;">'
        f'(score {risk_score}/10)</span></h3>'
        f'</div>'
        f'<div class="insight-box-content">{body}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_section_title(title: str) -> None:
    """Affiche un titre de section stylisé.

    Args:
        title: Texte du titre.
    """
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def _severity_icon_html(severity: str) -> str:
    """Retourne l'icône HTML pour un niveau de sévérité.

    Args:
        severity: Niveau de sévérité.

    Returns:
        Icône unicode.
    """
    icons = {
        "INFO": "\u2139\ufe0f",
        "WARNING": "\u26a0\ufe0f",
        "ALERT": "\U0001f534",
        "CRITICAL": "\U0001f6a8",
    }
    return icons.get(severity, "")
