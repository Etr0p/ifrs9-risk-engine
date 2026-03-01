"""Charts Virtual CRO — NeSy MAS agents, DS fusion, QBAF, PMA."""

from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.dashboard.charts.base import (
    _PRIMARY, _SECONDARY, _ACCENT, _BG, _CARD, _TEXT, _MUTED,
    _SUCCESS, _WARNING, _DANGER, _INFO, _COLORS,
    _base_layout, _hex_to_rgba,
)


# ──────────────────────────────────────────────
# Virtual CRO — NeSy MAS (Etape 4)
# ──────────────────────────────────────────────

def plot_vcro_agent_agreement(
    agent_beliefs: Dict,
) -> go.Figure:
    """Radar chart des confidences et croyances dominantes par agent.

    Args:
        agent_beliefs: Dict agent_name -> BeliefMass.
    """
    agents = list(agent_beliefs.keys())
    confidences = [agent_beliefs[a].confidence for a in agents]
    agents_closed = agents + [agents[0]]
    conf_closed = confidences + [confidences[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=conf_closed,
        theta=agents_closed,
        fill="toself",
        fillcolor=f"rgba{_hex_to_rgba(_PRIMARY, 0.2)}",
        line=dict(color=_PRIMARY, width=2),
        name="Confiance",
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
    ))

    layout = _base_layout("Accord Inter-Agents", height=350)
    layout["polar"] = dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(
            range=[0, 1], showticklabels=True,
            tickfont=dict(color=_MUTED, size=9),
            gridcolor="rgba(255,255,255,0.1)",
        ),
        angularaxis=dict(
            tickfont=dict(color=_TEXT, size=11),
            gridcolor="rgba(255,255,255,0.1)",
        ),
    )
    fig.update_layout(**layout)
    return fig


def plot_vcro_belief_distribution(
    fused_masses: Dict[str, float],
    rule_used: str,
    conflict: float,
) -> go.Figure:
    """Bar chart des masses fusionnees par hypothese.

    Args:
        fused_masses: Dict hypothese -> masse fusionnee.
        rule_used: Regle de combinaison utilisee.
        conflict: Niveau de conflit.
    """
    hyps = sorted(k for k in fused_masses if k != "uncertainty")
    hyps.append("uncertainty")
    values = [fused_masses.get(h, 0) for h in hyps]

    colors = []
    for h in hyps:
        if h == "favorable":
            colors.append(_SUCCESS)
        elif h == "defavorable":
            colors.append(_DANGER)
        elif h == "uncertainty":
            colors.append(_MUTED)
        else:
            colors.append(_PRIMARY)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hyps,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(color=_TEXT, size=10),
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))

    layout = _base_layout(
        f"Masses Fusionnees (regle: {rule_used}, conflit: {conflict:.2f})",
        height=300,
    )
    layout["yaxis"]["range"] = [0, max(values) * 1.3] if values else [0, 1]
    layout["xaxis"]["tickangle"] = -30
    fig.update_layout(**layout)
    return fig


def plot_vcro_qbaf_strengths(
    strengths: Dict[str, float],
    symbolic_overrides: List[str],
    recommendation: str = "",
    recommendation_strength: float = 0.0,
    falsification_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> go.Figure:
    """SHAP-style waterfall force-plot for QBAF argument contributions.

    Shows how each argument's force flows through supports/attacks
    to produce the winning recommendation. Inspired by Lundberg's
    SHAP force plots (NeurIPS 2017) adapted to bipolar argumentation.

    Args:
        strengths: Dict argument -> force finale.
        symbolic_overrides: Liste des arguments symboliques actifs.
        recommendation: Recommandation gagnante.
        recommendation_strength: Force de la recommandation gagnante.
        falsification_matrix: Impact de chaque argument sur les recommandations.
    """
    recommendations = ["maintenir", "surveiller", "reduire", "escalader"]
    winner = recommendation or max(
        recommendations, key=lambda r: strengths.get(r, 0),
    )
    final_strength = recommendation_strength or strengths.get(winner, 0)

    # --- Build waterfall contributions toward the winner ---
    source_args = sorted(
        [k for k in strengths if k not in recommendations],
        key=lambda k: abs(strengths.get(k, 0)),
        reverse=True,
    )

    # Compute net contribution of each source arg toward the winner
    # via falsification matrix (counterfactual impact) or raw strength
    contributions = []
    for arg in source_args:
        s = strengths.get(arg, 0)
        if s == 0:
            continue
        if falsification_matrix and arg in falsification_matrix:
            # Counterfactual: removing this arg changes winner by delta
            delta = -falsification_matrix[arg].get(winner, 0)
            contributions.append((arg, delta, arg in symbolic_overrides))
        else:
            # Fallback: use raw strength as proxy (positive = supports winner)
            contributions.append((arg, s * 0.5, arg in symbolic_overrides))

    # Sort: positive contributions first (top), then negative (bottom)
    contributions.sort(key=lambda x: -x[1])

    if not contributions:
        # Fallback to simple bar chart if no contribution data
        contributions = [(a, strengths.get(a, 0), a in symbolic_overrides)
                         for a in source_args if strengths.get(a, 0) != 0]

    # --- Waterfall geometry ---
    labels = [c[0] for c in contributions]
    deltas = [c[1] for c in contributions]
    is_symbolic = [c[2] for c in contributions]

    # Cumulative: start from 0, each bar is a step
    cumulative = [0.0]
    for d in deltas:
        cumulative.append(cumulative[-1] + d)

    fig = go.Figure()

    # Arrow-like waterfall bars
    for i, (label, delta, sym) in enumerate(contributions):
        base = cumulative[i]
        top = cumulative[i + 1]

        # Color: red=negative/attack, teal=positive/support, bright red=symbolic
        if sym:
            bar_color = _DANGER
            edge_color = "#FF6B6B"
        elif delta >= 0:
            bar_color = f"rgba{_hex_to_rgba('#06B6D4', 0.85)}"
            edge_color = "#06B6D4"
        else:
            bar_color = f"rgba{_hex_to_rgba('#F43F5E', 0.85)}"
            edge_color = "#F43F5E"

        # Waterfall bar (horizontal)
        fig.add_trace(go.Bar(
            y=[label],
            x=[delta],
            base=[base],
            orientation="h",
            marker=dict(
                color=bar_color,
                line=dict(color=edge_color, width=1.5),
            ),
            text=[f"{delta:+.3f}"],
            textposition="outside",
            textfont=dict(color=edge_color, size=10),
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"{'SYMBOLIQUE' if sym else 'Neural'}<br>"
                f"Contribution: {delta:+.3f}<br>"
                f"Cumul: {top:.3f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

        # Connector line to next bar
        if i < len(contributions) - 1:
            fig.add_shape(
                type="line",
                y0=i + 0.4, y1=i + 0.6,
                x0=top, x1=top,
                line=dict(color=_MUTED, width=1, dash="dot"),
                yref="y", xref="x",
            )

    # --- Final result marker ---
    fig.add_trace(go.Scatter(
        x=[cumulative[-1]],
        y=[labels[-1] if labels else ""],
        mode="markers+text",
        marker=dict(
            symbol="diamond",
            size=14,
            color=_ACCENT,
            line=dict(color="#FFF", width=1.5),
        ),
        text=[f" {winner.upper()} = {final_strength:+.2f}"],
        textposition="middle right",
        textfont=dict(color=_ACCENT, size=12, family="monospace"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # --- Zero baseline ---
    fig.add_vline(x=0, line=dict(color=_MUTED, width=1, dash="dot"))

    # --- Layout ---
    x_range_max = max(abs(v) for v in cumulative) * 1.4
    layout = _base_layout(
        f"QBAF Force-Plot \u2192 {winner.upper()}",
        height=max(320, len(labels) * 45 + 60),
    )
    layout["xaxis"]["range"] = [-x_range_max, x_range_max]
    layout["xaxis"]["title"] = dict(
        text="Contribution cumulative", font=dict(color=_MUTED, size=10),
    )
    layout["yaxis"]["autorange"] = "reversed"

    # Legend annotations
    fig.add_annotation(
        x=0.01, y=1.0, xref="paper", yref="paper",
        text=(
            '<span style="color:#06B6D4">\u25CF Support</span>'
            '  <span style="color:#F43F5E">\u25CF Attaque</span>'
            '  <span style="color:#EF4444">\u25CF Symbolique</span>'
        ),
        showarrow=False,
        font=dict(size=10),
        align="left",
    )

    fig.update_layout(**layout)
    return fig


def plot_vcro_pma_comparison(
    ecl_legal: float,
    ecl_committee: float,
    pma_amount: float,
    is_valid: bool,
) -> go.Figure:
    """Bar chart comparant ECL legal vs ECL committee avec delta PMA.

    Args:
        ecl_legal: ECL reglementaire.
        ecl_committee: ECL recommande par le comite.
        pma_amount: PMA en EUR.
        is_valid: True si le PMA est valide.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["ECL Legal"],
        y=[ecl_legal],
        name="ECL Legal (IFRS 9)",
        marker_color=_PRIMARY,
        text=[f"{ecl_legal:,.0f}"],
        textposition="outside",
        textfont=dict(color=_TEXT, size=11),
        hovertemplate="ECL Legal: %{y:,.0f} EUR<extra></extra>",
    ))

    committee_color = _ACCENT if is_valid else f"rgba{_hex_to_rgba(_WARNING, 0.6)}"
    fig.add_trace(go.Bar(
        x=["ECL Comite"],
        y=[ecl_committee],
        name="ECL Comite (MAS)",
        marker_color=committee_color,
        text=[f"{ecl_committee:,.0f}"],
        textposition="outside",
        textfont=dict(color=_TEXT, size=11),
        hovertemplate="ECL Comite: %{y:,.0f} EUR<extra></extra>",
    ))

    pma_color = _DANGER if pma_amount > 0 else _SUCCESS
    pma_label = f"PMA : {pma_amount:+,.0f} EUR"
    if not is_valid:
        pma_label += " (rejete)"
    fig.add_annotation(
        x=1.5, y=max(ecl_legal, ecl_committee) * 0.5,
        text=pma_label,
        showarrow=False,
        font=dict(color=pma_color, size=14),
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor=pma_color,
        borderwidth=1,
        borderpad=6,
    )

    layout = _base_layout("ECL Legal vs Comite — PMA", height=350)
    layout["barmode"] = "group"
    layout["showlegend"] = True
    layout["legend"] = dict(
        orientation="h", y=-0.15,
        font=dict(color=_MUTED),
    )
    fig.update_layout(**layout)
    return fig
