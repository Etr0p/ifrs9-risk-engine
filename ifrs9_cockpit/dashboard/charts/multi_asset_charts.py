"""Charts Multi-Asset — balance sheet treemap, contagion, climat."""

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


def _fmt_ead(v: float) -> str:
    """Format EAD with compact suffixes: 1.37T, 683B, 42M."""
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"{v / 1e12:.2f}T"
    if abs_v >= 1e9:
        return f"{v / 1e9:.0f}B"
    if abs_v >= 1e6:
        return f"{v / 1e6:.0f}M"
    return f"{v:,.0f}"


def plot_balance_sheet_treemap(df_bs_ecl: pl.DataFrame) -> go.Figure:
    """Treemap des 10 classes d'actifs (surface = EAD, couleur = ECL/EAD).

    Design v3 "Bento" — modern card-grid aesthetic with generous gaps,
    large rounded corners, no visible borders, sans-serif typography,
    and a subtle dark-to-warm colorscale that highlights risk without
    screaming. Cells float like cards on a dark void background.

    Args:
        df_bs_ecl: DataFrame avec colonnes asset_class, label, category,
            ead_total, ecl_weighted, ecl_ead_ratio.
    """
    _VOID = "#0B1120"  # gap/background color (near-black navy)
    _FONT = "Inter, DM Sans, Segoe UI, -apple-system, system-ui, sans-serif"

    cat_labels = {
        "level1_detailed": "Level 1 (Detail)",
        "level2_parametric": "Level 2 (Enrichi)",
        "level3_simple": "Level 3 (Parametrique)",
    }

    total_ead = float(df_bs_ecl["ead_total"].sum())
    total_ecl = (
        float(df_bs_ecl["ecl_weighted"].sum())
        if "ecl_weighted" in df_bs_ecl.columns
        else 0.0
    )
    # Pre-compute column availability
    _has_ecl_weighted = "ecl_weighted" in df_bs_ecl.columns
    _has_rw_crr3 = "rw_crr3" in df_bs_ecl.columns
    total_ratio = total_ecl / total_ead if total_ead > 0 else 0.0

    labels = ["Bilan Total"]
    parents = [""]
    values = [total_ead]
    colors = [0.0]
    custom_text = [
        f"<b style='font-size:15px'>{_fmt_ead(total_ead)}</b><br>"
        f"ECL/EAD {total_ratio:.2%}"
    ]
    hover_texts = [
        f"<b>Bilan Consolide</b><br><br>"
        f"EAD  {_fmt_ead(total_ead)}<br>"
        f"ECL  {_fmt_ead(total_ecl)}<br>"
        f"Ratio  {total_ratio:.2%}"
    ]

    for cat_key, cat_label in cat_labels.items():
        cat_rows = df_bs_ecl.filter(pl.col("category") == cat_key)
        cat_ead = float(cat_rows["ead_total"].sum())
        avg_ratio = (
            float(cat_rows["ecl_ead_ratio"].mean()) if len(cat_rows) > 0 else 0.0
        )
        n_classes = len(cat_rows)
        pct = cat_ead / total_ead * 100 if total_ead > 0 else 0

        labels.append(cat_label)
        parents.append("Bilan Total")
        values.append(cat_ead)
        colors.append(avg_ratio)
        custom_text.append(
            f"<b>{cat_label}</b><br>"
            f"{_fmt_ead(cat_ead)}  {pct:.0f}%"
        )
        hover_texts.append(
            f"<b>{cat_label}</b><br><br>"
            f"EAD  {_fmt_ead(cat_ead)}  ({pct:.1f}%)<br>"
            f"ECL/EAD moy  {avg_ratio:.2%}<br>"
            f"Classes  {n_classes}"
        )

    for row in df_bs_ecl.iter_rows(named=True):
        cat_label = cat_labels.get(row["category"], row["category"])
        ead = row["ead_total"]
        ecl = row.get("ecl_weighted", 0) if _has_ecl_weighted else 0
        ratio = row["ecl_ead_ratio"]
        rw = row.get("rw_crr3", 0) if _has_rw_crr3 else 0
        pct_total = ead / total_ead * 100 if total_ead > 0 else 0

        labels.append(row["label"])
        parents.append(cat_label)
        values.append(ead)
        colors.append(ratio)

        # Inline risk dot — color only, no text label
        if ratio < 0.015:
            dot = "<span style='color:#22D3EE'>\u2022</span>"
        elif ratio < 0.035:
            dot = "<span style='color:#94A3B8'>\u2022</span>"
        elif ratio < 0.07:
            dot = "<span style='color:#F59E0B'>\u2022</span>"
        else:
            dot = "<span style='color:#EF4444'>\u2022</span>"

        custom_text.append(
            f"<b>{row['label']}</b><br>"
            f"<span style='font-size:13px'>{_fmt_ead(ead)}</span>"
            f"  {dot} {ratio:.1%}"
        )
        hover_texts.append(
            f"<b>{row['label']}</b><br><br>"
            f"EAD  {_fmt_ead(ead)}  ({pct_total:.1f}%)<br>"
            f"ECL  {_fmt_ead(ecl)}<br>"
            f"ECL/EAD  {ratio:.2%}<br>"
            f"RW CRR3  {rw:.0%}<br><br>"
            f"<span style='color:#475569'>{ead:,.0f} EUR</span>"
        )

    # --- Colorscale: ultra-subtle dark gradient ---
    # Low risk = cool dark slate, high risk = warm dark rose.
    # All stops stay dark so the gap color (_VOID) blends seamlessly.
    # Risk differences are visible but never garish.
    _colorscale = [
        [0.00, "#131C2E"],  # cold void
        [0.20, "#162340"],  # deep indigo
        [0.40, "#1E2D45"],  # steel dusk
        [0.55, "#2B2E3A"],  # neutral slate
        [0.70, "#372A30"],  # warm graphite
        [0.85, "#442828"],  # dark burgundy
        [1.00, "#4F2222"],  # muted crimson
    ]

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale=_colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="ECL/EAD",
                    font=dict(color="#64748B", size=9),
                    side="right",
                ),
                tickfont=dict(color="#475569", size=8),
                tickformat=".1%",
                thickness=4,
                len=0.35,
                yanchor="middle",
                y=0.5,
                outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
                x=1.005,
            ),
            cornerradius=8,
            line=dict(width=2, color=_VOID),
            pad=dict(t=38, l=6, r=6, b=6),
        ),
        textinfo="text",
        text=custom_text,
        textfont=dict(
            color="#CBD5E1",
            size=12,
            family=_FONT,
        ),
        hovertext=hover_texts,
        hoverinfo="text",
        hoverlabel=dict(
            bgcolor="rgba(15,23,42,0.96)",
            bordercolor="rgba(99,102,241,0.20)",
            font=dict(color="#E2E8F0", size=12, family=_FONT),
        ),
        branchvalues="total",
        pathbar=dict(visible=False),
        tiling=dict(squarifyratio=1.0),
    ))

    fig.update_layout(
        paper_bgcolor=_VOID,
        plot_bgcolor=_VOID,
        font=dict(family=_FONT, color="#94A3B8"),
        title=dict(
            text=(
                "<span style='color:#64748B;font-size:11px'>BALANCE SHEET</span>"
                "  <span style='color:#334155'>|</span>  "
                "<span style='color:#94A3B8;font-size:11px'>"
                "Exposure par Classe d'Actifs</span>"
            ),
            font=dict(size=11, family=_FONT),
            x=0.01,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        height=580,
        margin=dict(l=0, r=0, t=32, b=0),
    )
    return fig


def plot_contagion_network(
    base_severities: Dict[str, float],
    amplified_severities: Dict[str, float],
    contagion_matrix: Dict[str, Dict[str, float]],
) -> go.Figure:
    """Graphe de contagion inter-classes (barres horizontales avant/apres).

    Args:
        base_severities: Severites de base par classe.
        amplified_severities: Severites post-contagion.
        contagion_matrix: Matrice {source: {target: weight}}.
    """
    classes = sorted(
        amplified_severities.keys(),
        key=lambda c: amplified_severities.get(c, 0) / max(base_severities.get(c, 0), 1e-8),
        reverse=True,
    )

    from ifrs9_cockpit.config import ASSET_CLASS_MAP
    label_map = {k: v.label for k, v in ASSET_CLASS_MAP.items()}

    labels = [label_map.get(c, c) for c in classes]
    base_vals = [base_severities.get(c, 0) for c in classes]
    amplified_vals = [amplified_severities.get(c, 0) for c in classes]
    amp_factors = [
        amplified_severities.get(c, 0) / max(base_severities.get(c, 0), 1e-8)
        for c in classes
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels,
        x=base_vals,
        name="Severite Base",
        orientation="h",
        marker_color=_PRIMARY,
        opacity=0.5,
        hovertemplate="%{y}: %{x:.4f}<extra>Base</extra>",
    ))

    fig.add_trace(go.Bar(
        y=labels,
        x=amplified_vals,
        name="Post-Contagion",
        orientation="h",
        marker_color=_DANGER,
        opacity=0.7,
        hovertemplate="%{y}: %{x:.4f}<extra>Post-Contagion</extra>",
    ))

    for i, (label, factor) in enumerate(zip(labels, amp_factors)):
        if factor > 1.01:
            fig.add_annotation(
                x=amplified_vals[i],
                y=label,
                text=f"x{factor:.2f}",
                showarrow=False,
                xanchor="left",
                font=dict(color=_DANGER, size=10),
                xshift=5,
            )

    layout = _base_layout("Contagion Eisenberg-Noe -- Amplification", height=450)
    layout["barmode"] = "overlay"
    layout["showlegend"] = True
    layout["legend"] = dict(
        orientation="h", y=-0.12,
        font=dict(color=_MUTED),
    )
    layout["xaxis"] = dict(
        title=dict(text="Severite", font=dict(color=_MUTED)),
        gridcolor="rgba(100,116,139,0.15)",
    )
    fig.update_layout(**layout)
    return fig


def plot_climate_heatmap(df_bs_ecl: pl.DataFrame) -> go.Figure:
    """Heatmap risque climatique : PD base vs PD climate-adjusted par classe.

    Args:
        df_bs_ecl: DataFrame avec colonnes asset_class, label, pd_base,
            pd_climate, ecl_ead_ratio.
    """
    labels = df_bs_ecl["label"].to_list()
    pd_base = df_bs_ecl["pd_base"].to_list()
    pd_climate = df_bs_ecl["pd_climate"].to_list()
    ecl_ratios = df_bs_ecl["ecl_ead_ratio"].to_list()

    climate_impacts = [
        (pc - pb) / pb if pb > 1e-8 else 0.0
        for pc, pb in zip(pd_climate, pd_base)
    ]

    z = np.array([pd_base, pd_climate, climate_impacts, ecl_ratios]).T
    col_labels = ["PD Base", "PD Climate", "Impact Climat (%)", "ECL/EAD"]

    text = []
    for i in range(len(labels)):
        row_text = [
            f"{pd_base[i]:.3%}",
            f"{pd_climate[i]:.3%}",
            f"{climate_impacts[i]:+.1%}",
            f"{ecl_ratios[i]:.2%}",
        ]
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=col_labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(color=_TEXT, size=10),
        colorscale=[
            [0.0, _SUCCESS],
            [0.3, _PRIMARY],
            [0.6, _WARNING],
            [1.0, _DANGER],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Intensite", font=dict(color=_MUTED)),
            tickfont=dict(color=_MUTED),
        ),
        hovertemplate="%{y} - %{x}: %{text}<extra></extra>",
    ))

    layout = _base_layout("Risque Climatique -- PD & ECL par Classe", height=450)
    layout["xaxis"] = dict(
        tickfont=dict(color=_MUTED),
        side="top",
    )
    layout["yaxis"] = dict(
        tickfont=dict(color=_MUTED),
        autorange="reversed",
    )
    layout["margin"] = dict(l=130, r=30, t=70, b=30)
    fig.update_layout(**layout)
    return fig
