"""Charts Comparator — asymetrie, RAROC, CRR3, risk appetite."""

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


def plot_asymmetry_heatmap(asym_df: pl.DataFrame) -> go.Figure:
    """Heatmap de la matrice d'asymetrie Credit vs PE.

    Args:
        asym_df: DataFrame du PortfolioComparator.build_asymmetry_matrix().

    Returns:
        Figure Plotly heatmap.
    """
    metrics = ["loss_ratio", "rwa_ratio", "raroc_delta"]
    labels = ["Ratio Perte PE/Credit", "Ratio RWA PE/Credit", "Delta RAROC (PE-Credit)"]

    z = []
    for m in metrics:
        if m in asym_df.columns:
            z.append(asym_df[m].to_list())
        else:
            z.append([0.0] * len(asym_df))

    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=asym_df["sector"].to_list(),
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=13, color=_TEXT),
        colorscale=[
            [0, _ACCENT],
            [0.5, _CARD],
            [1, _DANGER],
        ],
        showscale=True,
        colorbar=dict(tickfont=dict(color=_MUTED)),
    ))

    layout = _base_layout("Matrice d'Asymetrie Credit vs PE", height=320)
    fig.update_layout(**layout)
    return fig


def plot_raroc_comparison(raroc_df: pl.DataFrame) -> go.Figure:
    """Barres groupees RAROC Credit vs PE par secteur.

    Args:
        raroc_df: DataFrame du PortfolioComparator.compute_raroc_eva().

    Returns:
        Figure Plotly grouped bar.
    """
    # Filtrer les totaux
    df = raroc_df.filter(~pl.col("sector").str.starts_with("TOTAL"))

    credit = df.filter(pl.col("canal") == "Credit").sort("sector")
    pe = df.filter(pl.col("canal") == "PE").sort("sector")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=credit["sector"], y=credit["raroc"], name="RAROC Credit",
        marker_color=_PRIMARY, opacity=0.9,
    ))
    if len(pe) > 0:
        fig.add_trace(go.Bar(
            x=pe["sector"], y=pe["raroc"], name="RAROC PE",
            marker_color=_ACCENT, opacity=0.9,
        ))

    layout = _base_layout("RAROC Credit vs PE par Secteur", height=400)
    layout["barmode"] = "group"
    layout["yaxis"]["title"] = "RAROC"
    layout["yaxis"]["tickformat"] = ".1%"
    fig.update_layout(**layout)
    # Target reference line (Risk Appetite green threshold)
    fig.add_hline(
        y=0.04, line_dash="dash", line_color=_SUCCESS, line_width=1.5,
        annotation_text="Cible RAROC (4%)",
        annotation_font_color=_SUCCESS,
        annotation_font_size=10,
    )
    return fig


def plot_crr3_sensitivity(crr3_df: pl.DataFrame) -> go.Figure:
    """Barres CET1 ratio par scenario RW PE (CRR3).

    Args:
        crr3_df: DataFrame du PortfolioComparator.compute_crr3_sensitivity().

    Returns:
        Figure Plotly.
    """
    fig = go.Figure()

    colors = []
    for row in crr3_df.iter_rows(named=True):
        colors.append(_ACCENT if row.get("feasible", True) else _DANGER)

    x_labels = [f"{v}%" for v in crr3_df["rw_pe"].to_list()]

    fig.add_trace(go.Bar(
        x=x_labels,
        y=crr3_df["cet1_ratio"],
        marker_color=colors,
        text=[f"{v:.2%}" for v in crr3_df["cet1_ratio"].to_list()],
        textposition="outside",
        textfont=dict(color=_TEXT, size=12),
    ))

    # Seuil CET1 minimum (10.5% Pillar 1+2)
    fig.add_hline(
        y=0.105, line_dash="dash", line_color=_DANGER,
        annotation_text="CET1 min (10.5%)",
        annotation_font_color=_DANGER,
    )

    layout = _base_layout("Sensibilite CRR3 — CET1 par Risk Weight PE", height=380)
    layout["xaxis"]["title"] = "Risk Weight PE"
    layout["yaxis"]["title"] = "CET1 Ratio"
    layout["yaxis"]["tickformat"] = ".1%"
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


def plot_multiclass_raroc_scatter(raroc_mc: pl.DataFrame, class_weights: dict = None) -> go.Figure:
    """Diverging horizontal bar — RAROC vs Hurdle Rate (10 classes).

    Visual encoding:
        - Y axis: Class labels (sorted by RAROC descending, best on top)
        - X axis: RAROC % (centered on k_e=13%)
        - Bar color: Divergent (red < k_e < green)
        - RAROC value annotated at bar end
        - Color marker stripe on left = class identity
        - Hover: full detail (EVA, Exposure, RWA, Capital, Allocation)

    Args:
        raroc_mc: DataFrame de compute_raroc_multiclass() (10 rows + Total).
        class_weights: Dict {asset_class: weight} (optionnel).

    Returns:
        Figure Plotly diverging horizontal bars.
    """
    from ifrs9_cockpit.config import ASSET_CLASS_MAP

    _CLASS_COLORS = {
        "corporate_loans": "#6366F1",
        "private_equity": "#10B981",
        "sovereign": "#F59E0B",
        "retail_mortgage": "#3B82F6",
        "project_finance": "#8B5CF6",
        "covered_bonds": "#14B8A6",
        "consumer_credit": "#F97316",
        "trade_finance": "#06B6D4",
        "interbank": "#EC4899",
        "structured_products": "#EF4444",
        "equities": "#A855F7",
        "corporate_bonds": "#0EA5E9",
        "repos_sft": "#84CC16",
        "derivatives_cva": "#F472B6",
    }

    df = raroc_mc.filter(pl.col("asset_class") != "Total")
    if len(df) == 0:
        fig = go.Figure()
        fig.update_layout(**_base_layout("RAROC Multi-Actif — Aucune donnee", height=200))
        return fig

    if class_weights:
        weight_map = class_weights
        df = df.with_columns(
            pl.col("asset_class").replace_strict(weight_map, default=0.05).alias("weight"),
        )
    else:
        df = df.with_columns(pl.lit(0.10).alias("weight"))

    # Sort by RAROC ascending (bottom=worst, top=best)
    df = df.sort("raroc")

    k_e = 0.13
    k_e_pct = k_e * 100
    rarocs_pct = df["raroc"].to_numpy() * 100
    labels = df["label"].to_numpy()
    weights = df["weight"].to_numpy()
    asset_classes = df["asset_class"].to_numpy()

    # X-range: clip outliers for readable scale, rest visible in hover
    clipped_rarocs = np.clip(rarocs_pct, k_e_pct - 25, k_e_pct + 25)
    x_lo = min(clipped_rarocs.min(), k_e_pct - 10) - 3
    x_hi = max(clipped_rarocs.max(), k_e_pct + 10) + 5

    def _bar_color(rpct: float) -> str:
        delta = rpct - k_e_pct
        if delta >= 5:
            return _SUCCESS
        elif delta >= 0:
            t = delta / 5.0
            return f"rgba(52,211,153,{0.50 + t * 0.40})"
        elif delta >= -5:
            t = abs(delta) / 5.0
            return f"rgba({int(251 - 3*t)},{int(191 - 78*t)},{int(36 + 77*t)},0.85)"
        else:
            return _DANGER

    fig = go.Figure()
    n = len(df)

    for i in range(n):
        rpct = clipped_rarocs[i]
        rpct_real = rarocs_pct[i]
        bar_len = rpct - k_e_pct
        ac = asset_classes[i]
        w = weights[i]
        row_i = df.row(i, named=True)
        exp = row_i.get("exposure", 0)
        eva = row_i.get("eva", 0)
        rwa = row_i.get("rwa", 0)
        cap = row_i.get("capital", 0)

        # Format EVA for hover
        if abs(eva) >= 1e9:
            eva_str = f"{eva/1e9:+.1f} Md"
        else:
            eva_str = f"{eva/1e6:+,.0f} M"

        hover = (
            f"<b>{labels[i]}</b><br>"
            f"RAROC: {rpct_real/100:.1%}<br>"
            f"Allocation: {w:.1%}<br>"
            f"Exposure: {exp/1e9:.1f} Md<br>"
            f"RWA: {rwa/1e9:.1f} Md<br>"
            f"Capital: {cap/1e9:.1f} Md<br>"
            f"EVA: {eva_str}"
        )

        fig.add_trace(go.Bar(
            x=[bar_len],
            y=[i],
            base=k_e_pct,
            orientation="h",
            marker=dict(color=_bar_color(rpct), line=dict(color=_BG, width=1)),
            width=0.6,
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
            name=labels[i],
        ))

        # RAROC % annotation at bar end
        x_anno = rpct + (0.8 if bar_len >= 0 else -0.8)
        anchor = "left" if bar_len >= 0 else "right"
        fig.add_annotation(
            x=x_anno, y=i,
            text=f"<b>{rpct_real:+.1f}%</b>",
            showarrow=False,
            font=dict(size=10, color=_bar_color(rpct)),
            xanchor=anchor, yanchor="middle",
        )

        # Class color stripe (paper-left edge)
        class_color = _CLASS_COLORS.get(ac, _MUTED)
        fig.add_shape(
            type="rect",
            x0=0, x1=0.012, y0=(i - 0.3) / n, y1=(i + 0.3) / n,
            xref="paper", yref="paper",
            fillcolor=class_color, line=dict(width=0),
        )

    # ── Hurdle rate k_e vertical line ──
    fig.add_vline(
        x=k_e_pct,
        line=dict(dash="dash", color=_WARNING, width=2),
    )
    fig.add_annotation(
        x=k_e_pct, y=1.04, yref="paper", xref="x",
        text=f"k<sub>e</sub> = {k_e_pct:.0f}%",
        showarrow=False, font=dict(size=11, color=_WARNING),
    )

    # ── Quadrant labels (paper-relative, no overlap) ──
    fig.add_annotation(
        x=0.98, y=0.02, xref="paper", yref="paper",
        text="Createurs de valeur  \u25b6",
        showarrow=False, font=dict(size=9, color=_SUCCESS),
        xanchor="right", yanchor="bottom",
    )
    fig.add_annotation(
        x=0.02, y=0.02, xref="paper", yref="paper",
        text="\u25c0  Destructeurs de valeur",
        showarrow=False, font=dict(size=9, color=_DANGER),
        xanchor="left", yanchor="bottom",
    )

    # ── Layout ──
    layout = _base_layout("RAROC vs Hurdle Rate — 10 Classes", height=max(420, n * 42))
    layout["xaxis"] = dict(
        title=dict(text="RAROC (%)", font=dict(color=_MUTED, size=11)),
        ticksuffix="%",
        gridcolor="rgba(148,163,184,0.06)",
        zerolinecolor="rgba(148,163,184,0.1)",
        tickfont=dict(color=_MUTED, size=10),
        range=[x_lo, x_hi],
    )
    layout["yaxis"] = dict(
        title=None,
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=labels,
        tickfont=dict(color=_TEXT, size=11),
        showgrid=False,
    )
    layout["margin"] = dict(l=170, r=70, t=55, b=45)
    layout["bargap"] = 0.15
    layout["showlegend"] = False
    fig.update_layout(**layout)

    return fig


def plot_risk_appetite_matrix(ra_df: pl.DataFrame) -> go.Figure:
    """Matrice risk appetite (traffic lights) par secteur et canal.

    Args:
        ra_df: DataFrame analytics_state.risk_appetite_matrix.

    Returns:
        Figure Plotly heatmap.
    """
    if ra_df is None or len(ra_df) == 0:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Risk Appetite — Aucune donnee", height=200))
        return fig

    signal_map = {"vert": 0, "ambre": 1, "rouge": 2}
    # CVD-safe: symbols + text for colorblind accessibility
    signal_labels = {"vert": "\u2713 Vert", "ambre": "\u26a0 Ambre", "rouge": "\u2717 Rouge"}

    # Pivoter pour avoir secteurs en lignes, canaux en colonnes
    if "canal" in ra_df.columns and "sector" in ra_df.columns:
        pivot = ra_df.pivot(on="canal", index="sector", values="signal").fill_null("vert")
        canal_cols = [c for c in pivot.columns if c != "sector"]
        sectors = pivot["sector"].to_list()

        z = []
        text = []
        for row in pivot.iter_rows(named=True):
            z_row = [signal_map.get(row.get(c, "vert"), 0) for c in canal_cols]
            t_row = [signal_labels.get(row.get(c, "vert"), row.get(c, "?")) for c in canal_cols]
            z.append(z_row)
            text.append(t_row)

        fig = go.Figure(go.Heatmap(
            z=z,
            x=canal_cols,
            y=sectors,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=14, color=_TEXT),
            colorscale=[
                [0, _SUCCESS],
                [0.5, _WARNING],
                [1, _DANGER],
            ],
            showscale=False,
            zmin=0, zmax=2,
        ))
    else:
        # Fallback — simple list
        z = [[signal_map.get(str(row.get("signal", "vert")), 0) for row in ra_df.iter_rows(named=True)]]
        text = [[signal_labels.get(str(row.get("signal", "vert")), "?") for row in ra_df.iter_rows(named=True)]]
        fig = go.Figure(go.Heatmap(
            z=z, text=text, texttemplate="%{text}",
            textfont=dict(size=14, color=_TEXT),
            colorscale=[[0, _SUCCESS], [0.5, _WARNING], [1, _DANGER]],
            showscale=False, zmin=0, zmax=2,
        ))

    layout = _base_layout("Matrice Risk Appetite (Feux Tricolores)", height=350)
    fig.update_layout(**layout)
    return fig
