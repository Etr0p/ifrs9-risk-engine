"""Charts Regime — HMM gauge, GFlowNet scatter, allocation regime."""

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


def plot_regime_gauge(regime_result: object) -> go.Figure:
    """Gauge HMM montrant le regime detecte et les probabilites.

    Affiche une jauge en 3 zones (contraction/recovery/expansion)
    avec l'aiguille sur la probabilite du regime courant.
    """
    regime = getattr(regime_result, "regime", "recovery")
    probs = getattr(regime_result, "probabilities", [0.33, 0.34, 0.33])

    regime_colors = {
        "contraction": _DANGER,
        "recovery": _WARNING,
        "expansion": _SUCCESS,
    }
    regime_labels = {
        "contraction": "Contraction",
        "recovery": "Recovery",
        "expansion": "Expansion",
    }

    # Value: 0-1 scale mapped from regime probabilities
    # contraction = low, expansion = high
    value = probs[1] * 0.5 + probs[2] * 1.0  # weighted toward expansion

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(value, 3),
        number=dict(suffix="", font=dict(color=regime_colors.get(regime, _PRIMARY), size=28)),
        title=dict(
            text=f"Regime: {regime_labels.get(regime, regime)}",
            font=dict(color=_MUTED, size=14),
        ),
        gauge=dict(
            axis=dict(range=[0, 1], tickfont=dict(color=_MUTED, size=9)),
            bar=dict(color=regime_colors.get(regime, _PRIMARY)),
            steps=[
                dict(range=[0, 0.33], color=f"rgba{_hex_to_rgba(_DANGER, 0.15)}"),
                dict(range=[0.33, 0.66], color=f"rgba{_hex_to_rgba(_WARNING, 0.15)}"),
                dict(range=[0.66, 1.0], color=f"rgba{_hex_to_rgba(_SUCCESS, 0.15)}"),
            ],
            threshold=dict(
                line=dict(color=_MUTED, width=2),
                thickness=0.8,
                value=value,
            ),
        ),
    ))

    layout = _base_layout("Detection de Regime HMM", height=280)
    fig.update_layout(**layout)
    return fig


def plot_gflownet_scatter(gflownet_result: object) -> go.Figure:
    """Scatter des scenarios GFlowNet : Mahalanobis vs Reward.

    Taille des points proportionnelle au reward, couleur par mode.
    """
    z_vectors = getattr(gflownet_result, "z_vectors", np.array([]))
    rewards = getattr(gflownet_result, "rewards", np.array([]))
    mode = getattr(gflownet_result, "mode", "defensive")
    best_idx = int(np.argmax(rewards)) if len(rewards) > 0 else 0

    if len(z_vectors) == 0:
        fig = go.Figure()
        layout = _base_layout("GFlowNet — Aucun scenario", height=350)
        fig.update_layout(**layout)
        return fig

    distances = np.linalg.norm(z_vectors, axis=1)
    color = _DANGER if mode == "defensive" else _SUCCESS

    fig = go.Figure()

    # All scenarios
    fig.add_trace(go.Scatter(
        x=distances, y=rewards,
        mode="markers",
        marker=dict(
            size=np.clip(rewards / max(rewards.max(), 1e-8) * 15, 4, 20),
            color=color,
            opacity=0.6,
            line=dict(width=0.5, color=_MUTED),
        ),
        name=f"Scenarios ({mode})",
        hovertemplate="Mahalanobis: %{x:.2f}<br>Reward: %{y:.4f}<extra></extra>",
    ))

    # Best scenario (diamond)
    fig.add_trace(go.Scatter(
        x=[distances[best_idx]], y=[rewards[best_idx]],
        mode="markers+text",
        marker=dict(size=16, color=_ACCENT, symbol="diamond", line=dict(width=2, color=_PRIMARY)),
        text=["Best"],
        textposition="top center",
        textfont=dict(color=_ACCENT, size=10),
        name="Meilleur scenario",
        showlegend=True,
    ))

    layout = _base_layout(
        f"GFlowNet — {mode.capitalize()} ({len(rewards)} scenarios)",
        height=350,
    )
    layout["xaxis"]["title"] = dict(text="Distance Mahalanobis (sigma)", font=dict(color=_MUTED, size=11))
    layout["yaxis"]["title"] = dict(text="Reward", font=dict(color=_MUTED, size=11))
    layout["legend"] = dict(orientation="h", y=-0.2, font=dict(color=_MUTED))
    fig.update_layout(**layout)
    return fig


def plot_regime_allocation(regime_alloc: dict) -> go.Figure:
    """Barres horizontales : signaux sectoriels conditionnes par le regime.

    Couleur par signal (surponderer=vert, maintenir=bleu, sous-ponderer=rouge).
    """
    signals = regime_alloc.get("sector_signals", [])
    if not signals:
        fig = go.Figure()
        layout = _base_layout("Allocation Regime — Aucune donnee", height=300)
        fig.update_layout(**layout)
        return fig

    sectors = [s["sector"] for s in signals]
    rarocs_c = [s["raroc_credit"] for s in signals]
    rarocs_p = [s["raroc_pe"] for s in signals]
    signal_colors = {
        "surponderer": _SUCCESS,
        "maintenir": _PRIMARY,
        "sous-ponderer": _DANGER,
    }

    fig = go.Figure()

    # Credit RAROC
    fig.add_trace(go.Bar(
        y=sectors, x=rarocs_c,
        orientation="h",
        name="RAROC Credit",
        marker_color=_PRIMARY,
        opacity=0.8,
    ))

    # PE RAROC
    fig.add_trace(go.Bar(
        y=sectors, x=rarocs_p,
        orientation="h",
        name="RAROC PE",
        marker_color=_ACCENT,
        opacity=0.8,
    ))

    # Signal markers
    for i, s in enumerate(signals):
        color = signal_colors.get(s["signal"], _MUTED)
        max_raroc = max(abs(s["raroc_credit"]), abs(s["raroc_pe"]), 0.01)
        fig.add_annotation(
            x=max_raroc + 0.005,
            y=s["sector"],
            text=s["signal"],
            showarrow=False,
            font=dict(color=color, size=9),
            xanchor="left",
        )

    regime = regime_alloc.get("regime", "?")
    pe_pct = regime_alloc.get("pe_allocation", 0) * 100

    layout = _base_layout(
        f"Allocation Regime ({regime}) — PE {pe_pct:.0f}%",
        height=max(250, len(sectors) * 50),
    )
    layout["xaxis"]["title"] = dict(text="RAROC", font=dict(color=_MUTED, size=11))
    layout["barmode"] = "group"
    layout["legend"] = dict(orientation="h", y=-0.2, font=dict(color=_MUTED))
    fig.update_layout(**layout)
    return fig


def plot_scenario_weights_pie(regime_alloc: dict) -> go.Figure:
    """Pie chart des ponderations scenarios (base/adverse/favorable) par regime."""
    weights = regime_alloc.get("scenario_weights", {})
    if not weights:
        fig = go.Figure()
        layout = _base_layout("Ponderations — Aucune donnee", height=300)
        fig.update_layout(**layout)
        return fig

    labels = list(weights.keys())
    values = list(weights.values())

    color_map = {
        "base": _PRIMARY,
        "adverse": _DANGER,
        "favorable": _SUCCESS,
    }
    colors = [color_map.get(l.lower(), _MUTED) for l in labels]

    fig = go.Figure(go.Pie(
        labels=[l.capitalize() for l in labels],
        values=values,
        marker=dict(colors=colors, line=dict(color=_BG, width=2)),
        textinfo="label+percent",
        textfont=dict(color=_MUTED, size=11),
        hovertemplate="%{label}: %{value:.0%}<extra></extra>",
        hole=0.4,
    ))

    regime = regime_alloc.get("regime", "?")
    layout = _base_layout(
        f"Ponderations Scenarios — {regime.capitalize()}",
        height=300,
    )
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig
