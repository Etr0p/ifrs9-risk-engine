"""Charts Causal ML — CATE, survie, biais, Cinelli-Hazlett."""

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


def plot_cate_heatmap(cate_by_sector: Dict[str, Dict[str, float]]) -> go.Figure:
    """Heatmap CATE par secteur × estimateur.

    Montre l'heterogeneite de l'effet causal de l'ESG sur la PD.

    Args:
        cate_by_sector: Dict[method_name -> Dict[sector -> cate_value]].
    """
    methods = sorted(cate_by_sector.keys())
    sectors = ["Technologie", "Industrie", "Sante", "Immobilier", "Services"]

    z = []
    for method in methods:
        row = [cate_by_sector[method].get(s, 0) for s in sectors]
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=sectors,
        y=methods,
        colorscale=[
            [0, _DANGER],
            [0.5, _MUTED],
            [1, _SUCCESS],
        ],
        text=[[f"{v:.4f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        hovertemplate=(
            "Secteur: %{x}<br>"
            "Methode: %{y}<br>"
            "CATE: %{z:.5f}<br>"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="CATE", font=dict(color=_MUTED)),
            tickfont=dict(color=_MUTED),
        ),
    ))

    layout = _base_layout("CATE ESG → PD par Secteur", height=300)
    fig.update_layout(**layout)
    return fig


def plot_survival_shift(
    survival_high: Dict,
    survival_low: Dict,
) -> go.Figure:
    """Courbes de survie Kaplan-Meier : High ESG vs Low ESG.

    Args:
        survival_high: Dict with 'timeline' and 'survival' lists.
        survival_low: Dict with 'timeline' and 'survival' lists.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=survival_high["timeline"],
        y=survival_high["survival"],
        mode="lines",
        name="ESG > mediane",
        line=dict(color=_SUCCESS, width=2),
        hovertemplate="t=%{x:.1f}a S(t)=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=survival_low["timeline"],
        y=survival_low["survival"],
        mode="lines",
        name="ESG ≤ mediane",
        line=dict(color=_DANGER, width=2),
        hovertemplate="t=%{x:.1f}a S(t)=%{y:.3f}<extra></extra>",
    ))

    layout = _base_layout("Courbes de Survie — Effet Causal ESG", height=340)
    layout["xaxis"]["title"] = "Temps (annees)"
    layout["yaxis"]["title"] = "Probabilite de survie S(t)"
    layout["showlegend"] = True
    layout["legend"] = dict(
        orientation="h", y=-0.25,
        font=dict(color=_MUTED),
    )
    fig.update_layout(**layout)
    return fig


def plot_bias_comparison(
    ate_estimates: Dict[str, float],
    true_ate: Optional[float] = None,
) -> go.Figure:
    """Barre chart : biais de chaque estimateur vs True ATE.

    Args:
        ate_estimates: Dict[method_name -> ATE value].
        true_ate: Ground truth ATE from DGP (if available).
    """
    methods = list(ate_estimates.keys())
    values = [ate_estimates[m] for m in methods]

    # Color: causal methods in primary, naive in danger
    colors = [
        _DANGER if "Naive" in m or "naive" in m else _PRIMARY
        for m in methods
    ]

    fig = go.Figure(data=go.Bar(
        x=methods,
        y=values,
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(color=_TEXT, size=10),
        hovertemplate="Methode: %{x}<br>ATE: %{y:.5f}<extra></extra>",
    ))

    if true_ate is not None:
        fig.add_hline(
            y=true_ate,
            line=dict(color=_SUCCESS, width=2, dash="dash"),
            annotation_text=f"Vrai ATE: {true_ate:.4f}",
            annotation_font=dict(color=_SUCCESS, size=10),
            annotation_position="top left",
        )

    layout = _base_layout("Benchmark Estimateurs — ATE ESG → PD", height=340)
    layout["yaxis"]["title"] = "ATE (effet sur PD)"
    fig.update_layout(**layout)
    return fig


def plot_sensitivity_contour(
    partial_r2: float,
    robustness_value: float,
    benchmark_r2: Optional[float] = None,
) -> go.Figure:
    """Contour plot : robustesse de l'effet causal aux confounders caches.

    Inspired by Cinelli-Hazlett (2020) Austen plot.
    Shows the region where the causal effect is robust.

    Args:
        partial_r2: Partial R2 of treatment on outcome.
        robustness_value: Minimum confounder strength to nullify effect.
        benchmark_r2: Strength of the hidden management_quality confounder.
    """
    # Generate contour grid
    r2_t = np.linspace(0, 0.2, 50)  # R2 of confounder on treatment
    r2_y = np.linspace(0, 0.2, 50)  # R2 of confounder on outcome
    R2_T, R2_Y = np.meshgrid(r2_t, r2_y)

    # Bias surface: sqrt(R2_T * R2_Y) — simplified Cinelli-Hazlett
    bias = np.sqrt(R2_T * R2_Y)

    fig = go.Figure()

    # Contour of bias strength
    fig.add_trace(go.Contour(
        x=r2_t,
        y=r2_y,
        z=bias,
        colorscale=[
            [0, "rgba(76, 175, 80, 0.3)"],    # Green = robust
            [0.5, "rgba(255, 193, 7, 0.5)"],   # Yellow = moderate
            [1, "rgba(244, 67, 54, 0.7)"],     # Red = nullified
        ],
        contours=dict(
            start=0, end=0.15, size=0.01,
            showlabels=True,
            labelfont=dict(size=10, color="#1a1a2e"),
        ),
        colorbar=dict(
            title=dict(text="Bias Strength", font=dict(color=_MUTED)),
            tickfont=dict(color=_MUTED),
        ),
        hovertemplate=(
            "R²(U,T|W): %{x:.3f}<br>"
            "R²(U,Y|W): %{y:.3f}<br>"
            "Bias: %{z:.4f}<br>"
            "<extra></extra>"
        ),
    ))

    # Robustness value line (hyperbola: R2_T * R2_Y = RV^2)
    r2_curve = np.linspace(0.001, 0.2, 100)
    rv_curve = np.minimum(robustness_value ** 2 / r2_curve, 0.2)
    fig.add_trace(go.Scatter(
        x=r2_curve,
        y=rv_curve,
        mode="lines",
        name=f"RV = {robustness_value:.3f}",
        line=dict(color=_WARNING, width=2, dash="dash"),
        hovertemplate="RV boundary<extra></extra>",
    ))

    # Benchmark point (management_quality)
    if benchmark_r2 is not None:
        fig.add_trace(go.Scatter(
            x=[benchmark_r2],
            y=[benchmark_r2],
            mode="markers+text",
            name="management_quality",
            marker=dict(color=_ACCENT, size=12, symbol="diamond"),
            text=["mgmt_quality"],
            textposition="top right",
            textfont=dict(color=_ACCENT, size=9),
            hovertemplate=(
                f"Benchmark confounder<br>"
                f"R² = {benchmark_r2:.4f}<extra></extra>"
            ),
        ))

    layout = _base_layout("Analyse de Sensibilite — Cinelli-Hazlett", height=380)
    layout["xaxis"]["title"] = "R²(Confounder, Traitement | W)"
    layout["yaxis"]["title"] = "R²(Confounder, Outcome | W)"
    layout["showlegend"] = True
    layout["legend"] = dict(
        orientation="h", y=-0.25,
        font=dict(color=_MUTED),
    )
    fig.update_layout(**layout)
    return fig
