"""Charts PE — NAV, risk categories, MOIC, vintage, RWA."""

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


def plot_pe_nav_by_sector(result_pe: pl.DataFrame) -> go.Figure:
    """Barres NAV et Expected Loss PE par secteur.

    Args:
        result_pe: DataFrame resultat PECalculator.

    Returns:
        Figure Plotly grouped bar.
    """
    cols = [c for c in ["nav", "expected_loss_pe", "capital_invested"] if c in result_pe.columns]
    agg_exprs = [pl.col(c).sum().alias(c) for c in cols]
    seg = result_pe.group_by("sector").agg(agg_exprs).sort("nav")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=seg["sector"], x=seg["nav"], name="NAV",
        orientation="h", marker_color=_PRIMARY, opacity=0.9,
    ))
    fig.add_trace(go.Bar(
        y=seg["sector"], x=seg["capital_invested"], name="Capital Investi",
        orientation="h", marker_color=_SECONDARY, opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        y=seg["sector"], x=seg["expected_loss_pe"], name="Expected Loss PE",
        orientation="h", marker_color=_DANGER, opacity=0.85,
    ))

    layout = _base_layout("NAV & Pertes PE par Secteur", height=380)
    layout["barmode"] = "group"
    layout["xaxis"]["title"] = "Montant (EUR)"
    fig.update_layout(**layout)
    return fig


def plot_pe_risk_categories(result_pe: pl.DataFrame) -> go.Figure:
    """Pie chart des categories de risque PE.

    Args:
        result_pe: DataFrame resultat PECalculator.

    Returns:
        Figure Plotly pie.
    """
    counts = result_pe["risk_category"].value_counts()
    cat_order = ["Performing", "Watchlist", "Distressed"]
    count_map = dict(zip(
        counts["risk_category"].to_list(),
        counts["count"].to_list(),
    ))
    cat_labels = cat_order
    cat_values = [count_map.get(c, 0) for c in cat_order]
    colors = [_ACCENT, _WARNING, _DANGER]

    fig = go.Figure(go.Pie(
        labels=cat_labels,
        values=cat_values,
        marker=dict(colors=colors),
        textinfo="label+percent+value",
        textfont=dict(color=_TEXT, size=11),
        hole=0.45,
    ))
    layout = _base_layout("Classification PE (IPEV)", height=380)
    layout["showlegend"] = True
    fig.update_layout(**layout)
    return fig


def plot_pe_moic_drawdown(result_pe: pl.DataFrame) -> go.Figure:
    """MOIC moyen et drawdown moyen par secteur (double axe).

    Args:
        result_pe: DataFrame resultat PECalculator.

    Returns:
        Figure Plotly.
    """
    seg = result_pe.group_by("sector").agg(
        pl.col("moic").mean().alias("moic_mean"),
        pl.col("nav_drawdown").mean().alias("drawdown_mean"),
    ).sort("moic_mean")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        y=seg["sector"], x=seg["moic_mean"], name="MOIC moyen",
        orientation="h", marker_color=_PRIMARY, opacity=0.9,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        y=seg["sector"], x=seg["drawdown_mean"], name="Drawdown moyen",
        mode="markers+lines", marker=dict(size=10, color=_DANGER),
        line=dict(color=_DANGER, width=2),
    ), secondary_y=True)

    layout = _base_layout("MOIC & Drawdown par Secteur", height=380)
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="MOIC / Drawdown")
    fig.update_yaxes(title_text="Secteur")
    return fig


def plot_pe_distress_box(result_pe: pl.DataFrame) -> go.Figure:
    """Box plot de la probabilite de distress par secteur.

    Args:
        result_pe: DataFrame resultat PECalculator.

    Returns:
        Figure Plotly box plot.
    """
    sectors = sorted(result_pe["sector"].unique().to_list())
    fig = go.Figure()

    for i, sector in enumerate(sectors):
        subset = result_pe.filter(pl.col("sector") == sector)
        fig.add_trace(go.Box(
            y=subset["distress_prob"].to_numpy(),
            name=sector,
            marker_color=_COLORS[i % len(_COLORS)],
            opacity=0.6,
            boxpoints="outliers",
            marker=dict(size=3),
        ))

    # Seuils watchlist / distressed
    fig.add_hline(
        y=0.10, line=dict(color=_WARNING, width=1, dash="dash"),
        annotation_text="Watchlist", annotation_position="right",
        annotation_font=dict(color=_WARNING, size=9),
    )
    fig.add_hline(
        y=0.30, line=dict(color=_DANGER, width=1, dash="dash"),
        annotation_text="Distressed", annotation_position="right",
        annotation_font=dict(color=_DANGER, size=9),
    )

    layout = _base_layout("Distribution P(Distress) par Secteur", height=380)
    layout["yaxis"]["title"] = dict(text="P(Distress)", font=dict(color=_MUTED, size=11))
    fig.update_layout(**layout)
    return fig


def plot_pe_rwa_by_sector(result_pe: pl.DataFrame) -> go.Figure:
    """Barres RWA PE + EL PE par secteur.

    Args:
        result_pe: DataFrame resultat PECalculator.

    Returns:
        Figure Plotly grouped bars.
    """
    agg = result_pe.group_by("sector").agg(
        pl.col("rwa_pe").sum().alias("rwa"),
        pl.col("expected_loss_pe").sum().alias("el"),
        pl.col("nav").sum().alias("nav"),
    ).sort("rwa")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=agg["sector"],
        x=agg["rwa"],
        name="RWA PE",
        orientation="h",
        marker_color=_PRIMARY,
        text=[f"{v:,.0f}" for v in agg["rwa"]],
        textposition="auto",
        textfont=dict(color=_TEXT, size=9),
    ))

    fig.add_trace(go.Bar(
        y=agg["sector"],
        x=agg["el"],
        name="EL PE",
        orientation="h",
        marker_color=_DANGER,
        text=[f"{v:,.0f}" for v in agg["el"]],
        textposition="auto",
        textfont=dict(color=_TEXT, size=9),
    ))

    layout = _base_layout("RWA & Expected Loss PE par Secteur", height=350)
    layout["barmode"] = "group"
    layout["xaxis"]["title"] = dict(text="Montant (EUR)", font=dict(color=_MUTED, size=11))
    fig.update_layout(**layout)
    return fig


def plot_pe_vintage_analysis(result_pe: pl.DataFrame) -> go.Figure:
    """Scatter plot MOIC vs holding years (vintage analysis) colore par risque.

    Args:
        result_pe: DataFrame resultat PECalculator.

    Returns:
        Figure Plotly scatter.
    """
    if "holding_years" not in result_pe.columns:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Vintage — donnees manquantes", height=300))
        return fig

    # Subsample for readability
    n = len(result_pe)
    if n > 500:
        sample = result_pe.sample(500, seed=42)
    else:
        sample = result_pe

    color_map = {"Performing": _SUCCESS, "Watchlist": _WARNING, "Distressed": _DANGER}
    fig = go.Figure()

    for cat in ["Performing", "Watchlist", "Distressed"]:
        subset = sample.filter(pl.col("risk_category") == cat)
        if len(subset) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=subset["holding_years"].to_numpy(),
            y=subset["moic"].to_numpy(),
            mode="markers",
            name=cat,
            marker=dict(
                size=5,
                color=color_map.get(cat, _PRIMARY),
                opacity=0.5,
            ),
            hovertemplate=(
                "Holding: %{x:.1f}a<br>"
                "MOIC: %{y:.2f}<br>"
                "<extra></extra>"
            ),
        ))

    fig.add_hline(
        y=1.0, line=dict(color=_MUTED, width=1, dash="dot"),
        annotation_text="MOIC=1 (breakeven)",
        annotation_position="right",
        annotation_font=dict(color=_MUTED, size=9),
    )

    layout = _base_layout("Vintage Analysis — MOIC vs Holding Period", height=380)
    layout["xaxis"]["title"] = dict(text="Holding (annees)", font=dict(color=_MUTED, size=11))
    layout["yaxis"]["title"] = dict(text="MOIC", font=dict(color=_MUTED, size=11))
    layout["showlegend"] = True
    layout["legend"] = dict(orientation="h", y=-0.2, font=dict(color=_MUTED))
    fig.update_layout(**layout)
    return fig


def plot_pe_risk_stacked_bar(result_pe: pl.DataFrame) -> go.Figure:
    """Barre horizontale empilee pour categories de risque PE.

    Remplace le pie chart pour meilleure lisibilite.

    Args:
        result_pe: DataFrame PE avec colonne risk_category.

    Returns:
        Figure Plotly horizontal stacked bar.
    """
    from ifrs9_cockpit.config import PE_CATEGORY_COLORS

    counts_df = result_pe["risk_category"].value_counts()
    count_map = dict(zip(
        counts_df["risk_category"].to_list(),
        counts_df["count"].to_list(),
    ))
    total = sum(count_map.values())

    fig = go.Figure()
    cat_order = ["Performing", "Watchlist", "Distressed"]
    patterns = ["", "/", "x"]  # CVD-safe patterns

    for i, cat in enumerate(cat_order):
        count = count_map.get(cat, 0)
        pct = count / total if total > 0 else 0
        fig.add_trace(go.Bar(
            y=["Portefeuille PE"],
            x=[pct],
            name=f"{cat} ({count})",
            orientation="h",
            marker=dict(
                color=PE_CATEGORY_COLORS.get(cat, _MUTED),
                pattern_shape=patterns[i],
            ),
            text=f"{pct:.0%}",
            textposition="inside",
            textfont=dict(color="white", size=12),
            hovertemplate=f"<b>{cat}</b><br>Count: {count}<br>Part: {pct:.1%}<extra></extra>",
        ))

    layout = _base_layout("Classification Risque PE", height=180)
    layout["barmode"] = "stack"
    layout["showlegend"] = True
    layout["legend"] = dict(orientation="h", y=-0.3)
    fig.update_layout(**layout)
    return fig
