"""Charts Allocation — BL-CVaR, sector weights, donuts, frontier, regulatory."""

from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.dashboard.charts.base import (
    _PRIMARY, _SECONDARY, _ACCENT, _BG, _CARD, _TEXT, _MUTED,
    _SUCCESS, _WARNING, _DANGER, _INFO, _COLORS,
    _SECTOR_PALETTE, _SECTOR_PALETTE_FALLBACK,
    _base_layout, _hex_to_rgba,
)


# Color palette for 14 classes — shared across allocation charts
_CLASS_COLORS = {
    "corporate_loans": "#6366F1",      # indigo
    "private_equity": "#10B981",       # emerald
    "sovereign": "#F59E0B",            # amber
    "retail_mortgage": "#3B82F6",      # blue
    "project_finance": "#8B5CF6",      # violet
    "covered_bonds": "#14B8A6",        # teal
    "consumer_credit": "#F97316",      # orange
    "trade_finance": "#06B6D4",        # cyan
    "interbank": "#EC4899",            # pink
    "structured_products": "#EF4444",  # red
    "equities": "#A855F7",            # purple
    "corporate_bonds": "#0EA5E9",     # sky blue
    "repos_sft": "#84CC16",           # lime
    "derivatives_cva": "#F472B6",     # rose
}


def plot_multiclass_allocation_bar(optimization: dict) -> go.Figure:
    """Horizontal lollipop — allocation des 10 classes d'actifs.

    Chaque classe affichee comme un lollipop horizontal avec plancher
    structurel en trait pointille. Lisible meme avec de grands ecarts
    de ponderation (2% vs 30%).
    """
    from ifrs9_cockpit.config import ASSET_CLASS_MAP

    class_weights = optimization.get("class_weights", {})
    if not class_weights:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Allocation Multi-Actif — Aucune donnee", height=200))
        return fig

    typical = optimization.get("typical_weights", {})

    # Sort by weight ascending (smallest at top → largest at bottom for readability)
    sorted_classes = sorted(class_weights.items(), key=lambda x: x[1])
    names = [s[0] for s in sorted_classes]
    weights = [s[1] for s in sorted_classes]
    typicals = [typical.get(s[0], ac.typical_weight if (ac := ASSET_CLASS_MAP.get(s[0])) else 0) for s in sorted_classes]
    labels = []
    colors = []
    for name in names:
        ac = ASSET_CLASS_MAP.get(name)
        labels.append(ac.label if ac else name)
        colors.append(_CLASS_COLORS.get(name, _MUTED))

    fig = go.Figure()

    # Typical weight markers (indicative reference, not constraints)
    fig.add_trace(go.Scatter(
        x=[t * 100 for t in typicals],
        y=labels,
        mode="markers",
        marker=dict(
            symbol="line-ns", size=14, color="rgba(148,163,184,0.5)",
            line=dict(width=2, color="rgba(148,163,184,0.5)"),
        ),
        name="Poids Typique",
        hovertemplate="Typique: %{x:.1f}%<extra></extra>",
    ))

    # Horizontal bars (actual allocation)
    fig.add_trace(go.Bar(
        x=[w * 100 for w in weights],
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color=_BG, width=1),
            opacity=0.85,
        ),
        text=[f"  {w:.1%}" for w in weights],
        textposition="outside",
        textfont=dict(color=_TEXT, size=11),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Poids: %{x:.1f}%<br>"
            "<extra></extra>"
        ),
        name="Allocation",
        showlegend=False,
    ))

    max_pct = max(weights) * 100
    layout = _base_layout("Allocation Optimale — 10 Classes", height=max(380, len(labels) * 38))
    layout["margin"] = dict(l=140, r=60, t=45, b=30)
    layout["barmode"] = "overlay"
    layout["xaxis"] = dict(
        range=[0, max_pct * 1.20],
        ticksuffix="%",
        gridcolor="rgba(148,163,184,0.08)",
        zeroline=False,
        tickfont=dict(color=_MUTED, size=10),
        title=dict(text="Poids (%)", font=dict(color=_MUTED, size=10)),
    )
    layout["yaxis"] = dict(
        showgrid=False, zeroline=False,
        tickfont=dict(color=_TEXT, size=11),
        automargin=True,
    )
    layout["legend"] = dict(
        orientation="h", y=-0.12, x=0.5, xanchor="center",
        font=dict(color=_MUTED, size=10), bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(**layout)

    return fig


def plot_sector_allocation_weights(optimization: dict) -> go.Figure:
    """Vibrant stacked allocation bars — Credit vs PE sector breakdown.

    Two proportional stacked bars with bold sector colors, clean inline labels,
    and a refined layout matching the institutional dark theme.
    """
    w_credit = optimization.get("sector_weights_credit", {})
    w_pe = optimization.get("sector_weights_pe", {})

    if not w_credit and not w_pe:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Poids Sectoriels — Aucune donnee", height=200))
        return fig

    w_c_total = optimization.get("credit_allocation", optimization.get("w_credit", 0))
    w_p_total = optimization.get("pe_allocation", optimization.get("w_pe", 0))

    # Consistent sector ordering across both rows
    all_sectors = sorted(set(list(w_credit.keys()) + list(w_pe.keys())))

    channels = [
        (f"Credit ({w_c_total:.0%})", w_credit, all_sectors),
        (f"PE ({w_p_total:.0%})", w_pe, all_sectors),
    ]

    fig = go.Figure()

    for row_idx, (ch_label, weights, sectors) in enumerate(channels):
        vals = [weights.get(s, 0) for s in sectors]
        for i, (sector, val) in enumerate(zip(sectors, vals)):
            color = _SECTOR_PALETTE.get(
                sector, _SECTOR_PALETTE_FALLBACK[i % len(_SECTOR_PALETTE_FALLBACK)]
            )
            # Only show text if segment is wide enough
            if val > 0.06:
                text_label = f"<b>{sector}</b>  {val:.0%}"
            elif val > 0.02:
                text_label = f"{val:.0%}"
            else:
                text_label = ""

            fig.add_trace(go.Bar(
                x=[val * 100],
                y=[ch_label],
                orientation="h",
                name=sector if row_idx == 0 else None,
                legendgroup=sector,
                showlegend=(row_idx == 0),
                marker=dict(
                    color=color,
                    line=dict(color=_BG, width=1.5),
                ),
                text=[text_label],
                textposition="inside",
                textfont=dict(color="white", size=11),
                hovertemplate=f"<b>{sector}</b>: {val:.1%}<extra>{ch_label}</extra>",
            ))

    layout = _base_layout("", height=180)
    layout["barmode"] = "stack"
    layout["margin"] = dict(l=90, r=20, t=25, b=10)
    layout["legend"] = dict(
        orientation="h", y=-0.25, x=0.5, xanchor="center",
        font=dict(color=_TEXT, size=11),
        bgcolor="rgba(0,0,0,0)",
        traceorder="normal",
    )
    layout["yaxis"] = dict(
        showgrid=False, zeroline=False,
        tickfont=dict(color=_TEXT, size=12),
    )
    layout["xaxis"] = dict(
        showticklabels=False, showgrid=False, zeroline=False,
        range=[0, 100],
    )
    fig.update_layout(**layout)
    return fig


# ──────────────────────────────────────────────
# SECTOR ALLOCATION DONUTS
# ──────────────────────────────────────────────


def plot_sector_allocation_donuts(optimization: dict) -> go.Figure:
    """Side-by-side donut charts: Credit (left) and PE (right) sector breakdown.

    Each donut shows optimal sector weights with channel total % in the center.
    Monochrome per channel: Credit = indigo shades, PE = emerald shades.

    Args:
        optimization: Dict from optimize_allocation().

    Returns:
        Plotly Figure with two donuts via subplots.
    """
    from plotly.subplots import make_subplots

    w_credit = optimization.get("sector_weights_credit", {})
    w_pe = optimization.get("sector_weights_pe", {})
    c_alloc = optimization.get("credit_allocation", 0)
    p_alloc = optimization.get("pe_allocation", 0)

    all_sectors = sorted(set(list(w_credit.keys()) + list(w_pe.keys())))
    if not all_sectors:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Optimal Sector Allocation", height=300))
        return fig

    # Monochrome palettes — dark shades only (white text must stay readable)
    n = len(all_sectors)
    # Credit: indigo shades from dark (#312E81) to medium (#6366F1)
    credit_colors = [
        f"rgba({49 + int(i * (99 - 49) / max(n - 1, 1))},"
        f"{46 + int(i * (102 - 46) / max(n - 1, 1))},"
        f"{129 + int(i * (241 - 129) / max(n - 1, 1))},0.90)"
        for i in range(n)
    ]
    # PE: emerald shades from dark (#064E3B) to medium (#059669)
    pe_colors = [
        f"rgba({6 + int(i * (5 - 6) / max(n - 1, 1))},"
        f"{78 + int(i * (150 - 78) / max(n - 1, 1))},"
        f"{59 + int(i * (105 - 59) / max(n - 1, 1))},0.90)"
        for i in range(n)
    ]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=[None, None],
    )

    # Credit donut (left)
    c_vals = [w_credit.get(s, 0) for s in all_sectors]
    fig.add_trace(
        go.Pie(
            labels=all_sectors,
            values=[v * 100 for v in c_vals],
            hole=0.55,
            marker=dict(colors=credit_colors, line=dict(color=_BG, width=2)),
            textinfo="label+percent",
            textposition="inside",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{label}</b><br>%{percent}<extra>Credit</extra>",
            showlegend=False,
            sort=False,
        ),
        row=1, col=1,
    )

    # PE donut (right)
    p_vals = [w_pe.get(s, 0) for s in all_sectors]
    fig.add_trace(
        go.Pie(
            labels=all_sectors,
            values=[v * 100 for v in p_vals],
            hole=0.55,
            marker=dict(colors=pe_colors, line=dict(color=_BG, width=2)),
            textinfo="label+percent",
            textposition="inside",
            textfont=dict(size=11, color="white"),
            hovertemplate="<b>%{label}</b><br>%{percent}<extra>PE</extra>",
            showlegend=False,
            sort=False,
        ),
        row=1, col=2,
    )

    # Center labels
    fig.add_annotation(
        text=f"<b>Credit</b><br>{c_alloc:.0%}",
        x=0.20, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(color=_TEXT, size=16),
    )
    fig.add_annotation(
        text=f"<b>PE</b><br>{p_alloc:.0%}",
        x=0.80, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(color=_TEXT, size=16),
    )

    # Title
    fig.add_annotation(
        text="Optimal Sector Allocation",
        x=0, y=1.05, xref="paper", yref="paper",
        showarrow=False, font=dict(color=_TEXT, size=13),
        xanchor="left",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ──────────────────────────────────────────────
# FACTOR ATTRIBUTION (Layer 2)
# ──────────────────────────────────────────────


def plot_factor_attribution(factor_df) -> go.Figure:
    """Horizontal bar chart of macro factor attribution (Layer 2).

    Shows the contribution of each macro variable to portfolio risk,
    split by Credit/PE channel.

    Args:
        factor_df: DataFrame with columns
            [variable, canal, attribution, regime_adjusted].

    Returns:
        Plotly Figure.
    """
    if factor_df is None or len(factor_df) == 0:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Attribution Factorielle", height=200))
        return fig

    variables = factor_df["variable"].unique().to_list()
    fig = go.Figure()
    for canal, color, name in [
        ("Credit", _PRIMARY, "Credit"),
        ("PE", _ACCENT, "PE"),
    ]:
        canal_data = factor_df.filter(pl.col("canal") == canal)
        vals = []
        for v in variables:
            row = canal_data.filter(pl.col("variable") == v)
            vals.append(float(row["attribution"].to_numpy()[0]) * 100 if len(row) > 0 else 0)

        labels = [v.replace("_", " ").title() for v in variables]
        fig.add_trace(go.Bar(
            y=labels, x=vals,
            orientation="h", name=name,
            marker=dict(color=color, opacity=0.8),
            texttemplate="%{x:+.2f}%",
            textposition="outside",
            textfont=dict(size=9),
        ))

    layout = _base_layout("Attribution Factorielle Macro", height=max(220, len(variables) * 50))
    layout["barmode"] = "group"
    layout["bargap"] = 0.25
    layout["bargroupgap"] = 0.1
    layout["margin"] = dict(l=110, r=50, t=40, b=30)
    layout["xaxis"]["title"] = dict(text="Attribution (%)", font=dict(color=_MUTED, size=10))
    layout["xaxis"]["zeroline"] = True
    layout["xaxis"]["zerolinecolor"] = "rgba(148,163,184,0.2)"
    layout["legend"] = dict(
        orientation="h", y=-0.2, x=0.5, xanchor="center",
        font=dict(color=_MUTED, size=9), bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(**layout)
    return fig


# ──────────────────────────────────────────────
# REGULATORY CONSTRAINTS (INSTITUTIONAL GAUGES)
# ──────────────────────────────────────────────

def plot_regulatory_constraints(
    optimization: dict,
    hhi_cross: dict,
) -> go.Figure:
    """Regulatory constraints — angular gauge dials, 2 rows (3 + 2 centered).

    Row 1: CET1 | Leverage | HHI Cross   (evenly spaced across full width)
    Row 2:   HHI Credit | HHI PE         (centered, occupying middle 2/3)
    Uses explicit domain positioning for perfect centering.
    """
    cet1 = optimization.get("cet1_ratio", 0)
    leverage = optimization.get("leverage_ratio", cet1 * 0.4)

    # (label, value, threshold, max_range, unit, higher_better)
    constraints = [
        ("CET1 Ratio", cet1, 0.105, 0.25, "%", True),
        ("Leverage Ratio", leverage, 0.03, 0.10, "%", True),
        ("HHI Cross-Cell", hhi_cross.get("hhi_crosscell", 0), 3000, 6000, "pts", False),
        ("HHI Credit", hhi_cross.get("hhi_credit", 0), 4000, 8000, "pts", False),
        ("HHI PE", hhi_cross.get("hhi_pe", 0), 4000, 8000, "pts", False),
    ]

    # Explicit domain positions with generous gaps to prevent tick overlap.
    # Each gauge is ~28% wide with ~5% gap between them.
    # Row 1 (y 0.48–1.0): 3 gauges   |  Row 2 (y 0.0–0.42): 2 centered
    w1 = 0.28  # width per gauge row 1
    g1 = (1.0 - 3 * w1) / 2  # gap between row 1 gauges
    w2 = 0.30  # width per gauge row 2
    g2 = 0.10  # gap between row 2 gauges
    r2_start = (1.0 - 2 * w2 - g2) / 2  # center offset for row 2
    domains = [
        # Row 1: 3 columns with gaps
        {"x": [0.0, w1], "y": [0.48, 1.0]},
        {"x": [w1 + g1, 2 * w1 + g1], "y": [0.48, 1.0]},
        {"x": [1.0 - w1, 1.0], "y": [0.48, 1.0]},
        # Row 2: 2 columns, centered
        {"x": [r2_start, r2_start + w2], "y": [0.0, 0.42]},
        {"x": [r2_start + w2 + g2, r2_start + 2 * w2 + g2], "y": [0.0, 0.42]},
    ]

    fig = go.Figure()
    n = len(constraints)
    n_pass = 0

    for i, (label, value, threshold, max_val, unit, higher_better) in enumerate(constraints):
        passed = value >= threshold if higher_better else value <= threshold
        if passed:
            n_pass += 1

        if unit == "%":
            display_val = value * 100
            thr_display = threshold * 100
            max_display = max_val * 100
            number_fmt = ".1f"
            suffix = "%"
        else:
            display_val = value
            thr_display = threshold
            max_display = max_val
            number_fmt = ",.0f"
            suffix = ""

        if higher_better:
            steps = [
                {"range": [0, thr_display * 0.7], "color": "rgba(248,113,113,0.20)"},
                {"range": [thr_display * 0.7, thr_display], "color": "rgba(251,191,36,0.18)"},
                {"range": [thr_display, max_display], "color": "rgba(52,211,153,0.15)"},
            ]
        else:
            steps = [
                {"range": [0, thr_display * 0.7], "color": "rgba(52,211,153,0.15)"},
                {"range": [thr_display * 0.7, thr_display], "color": "rgba(251,191,36,0.18)"},
                {"range": [thr_display, max_display], "color": "rgba(248,113,113,0.20)"},
            ]

        bar_color = _SUCCESS if passed else _DANGER
        status_text = "PASS" if passed else "BREACH"
        status_color = "#34D399" if passed else "#F87171"

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=display_val,
                domain=domains[i],
                number=dict(
                    font=dict(color=bar_color, size=26),
                    suffix=suffix,
                    valueformat=number_fmt,
                ),
                title=dict(
                    text=(
                        f"<b>{label}</b><br>"
                        f"<span style='font-size:11px;color:{status_color}'>"
                        f"{status_text}</span>"
                        f"<span style='font-size:9px;color:{_MUTED}'>"
                        f"  seuil: {thr_display:g}{suffix}</span>"
                    ),
                    font=dict(color=_TEXT, size=13),
                ),
                gauge=dict(
                    axis=dict(
                        range=[0, max_display],
                        tickfont=dict(color=_MUTED, size=8),
                        tickcolor="rgba(148,163,184,0.2)",
                        nticks=4,
                        tickwidth=1,
                    ),
                    bar=dict(color=bar_color, thickness=0.65),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    steps=steps,
                    threshold=dict(
                        line=dict(color=_WARNING, width=3),
                        thickness=0.85,
                        value=thr_display,
                    ),
                ),
            ),
        )

    badge_color = _SUCCESS if n_pass == n else (_WARNING if n_pass >= n - 1 else _DANGER)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_MUTED, size=11),
        height=480,
        margin=dict(l=30, r=30, t=65, b=20),
    )

    fig.add_annotation(
        text="Contraintes Reglementaires",
        xref="paper", yref="paper", x=0, y=1.06,
        showarrow=False, font=dict(color=_TEXT, size=14), xanchor="left",
    )
    fig.add_annotation(
        text=f"{n_pass}/{n} conformes",
        xref="paper", yref="paper", x=1.0, y=1.06,
        showarrow=False, font=dict(color=badge_color, size=13), xanchor="right",
    )
    return fig


# ──────────────────────────────────────────────
# SECTOR SIGNALS (REGIME-DEPENDENT ARBITRAGE)
# ──────────────────────────────────────────────

def plot_sector_signals(regime_alloc: dict) -> go.Figure:
    """Sector signals — dumbbell chart comparing Credit vs PE RAROC.

    Each sector shown as a horizontal line connecting two dots:
    Credit (blue) and PE (green). The visual gap immediately shows
    the RAROC differential. Signal badges on the right.
    """
    signals = regime_alloc.get("sector_signals", [])
    if not signals:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Signaux Sectoriels", height=250))
        return fig

    _signal_styles = {
        "surponderer": (_SUCCESS, "\u25b2 Surponderer"),
        "maintenir": (_MUTED, "\u25c6 Maintenir"),
        "sous-ponderer": (_DANGER, "\u25bc Sous-ponderer"),
    }

    sectors = [s["sector"] for s in signals]
    raroc_c = [s.get("raroc_credit", 0) * 100 for s in signals]
    raroc_p = [s.get("raroc_pe", 0) * 100 for s in signals]
    sig_values = [s.get("signal", "maintenir") for s in signals]
    capital_relief = [s.get("capital_relief_candidate", False) for s in signals]

    fig = go.Figure()

    # Connecting lines (dumbbell stems)
    for i, sector in enumerate(sectors):
        lo = min(raroc_c[i], raroc_p[i])
        hi = max(raroc_c[i], raroc_p[i])
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[sector, sector],
            mode="lines",
            line=dict(color="rgba(148,163,184,0.3)", width=3),
            showlegend=False, hoverinfo="skip",
        ))

    # Position text on outside of each dot to avoid overlap
    credit_text_pos = []
    pe_text_pos = []
    for rc, rp in zip(raroc_c, raroc_p):
        if rc <= rp:
            credit_text_pos.append("middle left")
            pe_text_pos.append("middle right")
        else:
            credit_text_pos.append("middle right")
            pe_text_pos.append("middle left")

    # Credit dots (blue)
    fig.add_trace(go.Scatter(
        x=raroc_c, y=sectors, mode="markers+text",
        name="RAROC Credit",
        marker=dict(color=_PRIMARY, size=13, line=dict(color=_BG, width=2)),
        text=[f" {v:.1f}% " for v in raroc_c],
        textposition=credit_text_pos,
        textfont=dict(color=_PRIMARY, size=10),
        hovertemplate="<b>%{y}</b><br>Credit RAROC: %{x:.1f}%<extra></extra>",
    ))

    # PE dots (green diamond)
    fig.add_trace(go.Scatter(
        x=raroc_p, y=sectors, mode="markers+text",
        name="RAROC PE",
        marker=dict(color=_ACCENT, size=13, symbol="diamond",
                    line=dict(color=_BG, width=2)),
        text=[f" {v:.1f}% " for v in raroc_p],
        textposition=pe_text_pos,
        textfont=dict(color=_ACCENT, size=10),
        hovertemplate="<b>%{y}</b><br>PE RAROC: %{x:.1f}%<extra></extra>",
    ))

    # Signal badges on right
    max_raroc = max(max(raroc_c), max(raroc_p)) if raroc_c and raroc_p else 10
    for sector, sig, cr in zip(sectors, sig_values, capital_relief):
        color, label = _signal_styles.get(sig, _signal_styles["maintenir"])
        if cr:
            label += " *"
        fig.add_annotation(
            x=1.02, xref="paper", y=sector,
            text=f"<b>{label}</b>",
            showarrow=False, xanchor="left",
            font=dict(color=color, size=10),
        )

    # Cost of equity reference line
    k_e_pct = 13.0
    fig.add_vline(
        x=k_e_pct, line_dash="dot",
        line_color="rgba(251,191,36,0.25)", line_width=1,
    )
    fig.add_annotation(
        x=k_e_pct, y=sectors[-1], yshift=-28,
        text=f"k_e {k_e_pct:.0f}%",
        showarrow=False, font=dict(color="rgba(251,191,36,0.5)", size=8),
    )

    regime = regime_alloc.get("regime", "?")
    n_relief = regime_alloc.get("n_capital_relief", 0)

    min_raroc = min(min(raroc_c), min(raroc_p)) if raroc_c and raroc_p else 0
    layout = _base_layout("", height=max(320, len(sectors) * 75))
    layout["margin"] = dict(l=110, r=140, t=50, b=55)
    layout["legend"] = dict(
        orientation="h", y=-0.18, x=0.5, xanchor="center",
        font=dict(color=_MUTED, size=11), bgcolor="rgba(0,0,0,0)",
        itemsizing="constant", tracegroupgap=30,
    )
    layout["yaxis"] = dict(
        showgrid=False, zeroline=False,
        tickfont=dict(color=_TEXT, size=12),
    )
    layout["xaxis"] = dict(
        gridcolor="rgba(148,163,184,0.06)",
        zerolinecolor="rgba(148,163,184,0.08)",
        title=dict(text="RAROC (%)", font=dict(color=_MUTED, size=10)),
        range=[max(0, min_raroc - 2.5), max_raroc * 1.30],
        ticksuffix="%", tickfont=dict(color=_MUTED, size=10),
        side="top",
    )
    fig.update_layout(**layout)

    # Title
    relief_note = f" \u2014 {n_relief} titrisation" if n_relief > 0 else ""
    fig.add_annotation(
        text=f"Signaux Regime ({regime.capitalize()}){relief_note}",
        xref="paper", yref="paper", x=0, y=1.06,
        showarrow=False, font=dict(color=_TEXT, size=13), xanchor="left",
    )

    if any(capital_relief):
        fig.add_annotation(
            text="* Candidat titrisation",
            xref="paper", yref="paper", x=1.0, y=-0.15,
            showarrow=False, font=dict(color=_MUTED, size=9), xanchor="right",
        )
    return fig


def plot_efficient_frontier(optimization: dict) -> go.Figure:
    """Efficient frontier: RAROC vs PE Allocation with dual CET1 axis.

    Dual-axis chart showing the tradeoff between increasing PE allocation
    (higher RAROC) and decreasing CET1 ratio (regulatory constraint).

    Args:
        optimization: Dict from PortfolioComparator.optimize_allocation().

    Returns:
        Figure Plotly with secondary y-axis.
    """
    from plotly.subplots import make_subplots

    raroc_c = optimization.get("raroc_credit", 0.03)
    raroc_p = optimization.get("raroc_pe", 0.05)
    pe_free = optimization.get("pe_free", 0.20)
    pe_alloc = optimization.get("pe_allocation", 0.10)
    cet1_target = optimization.get("k_e", 0.13)
    rwa_weighted = optimization.get("rwa_weighted", 1e12)
    credit_alloc = optimization.get("credit_allocation", 0.90)

    # Back-calculate RWA components from optimization dict
    cet1_capital = rwa_weighted * optimization.get("cet1_ratio", 0.14)
    if pe_alloc > 0 and credit_alloc > 0 and abs(pe_alloc + credit_alloc - 1.0) < 0.01:
        sigma_c = optimization.get("sigma_credit", 0.02)
        sigma_p = optimization.get("sigma_pe", 0.08)
        ratio = max(sigma_p / max(sigma_c, 1e-6), 1.5)
        rwa_c_total = rwa_weighted / (credit_alloc + pe_alloc * ratio)
        rwa_p_total = rwa_c_total * ratio
    else:
        rwa_c_total = rwa_weighted
        rwa_p_total = rwa_weighted

    # ── Sweep PE allocation 0% → 95% ──
    pe_range = np.linspace(0.0, 0.95, 200)
    raroc_sweep = (1.0 - pe_range) * raroc_c + pe_range * raroc_p
    rwa_sweep = (1.0 - pe_range) * rwa_c_total + pe_range * rwa_p_total
    cet1_sweep = cet1_capital / np.maximum(rwa_sweep, 1.0)

    # Infeasible boundary
    infeasible_idx = np.where(cet1_sweep < cet1_target)[0]
    pe_infeasible = pe_range[infeasible_idx[0]] if len(infeasible_idx) > 0 else 1.0

    # RAROC at key points
    raroc_at_alloc = credit_alloc * raroc_c + pe_alloc * raroc_p
    raroc_at_free = (1.0 - pe_free) * raroc_c + pe_free * raroc_p

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ── 1. Infeasible zone (red shaded area) ──
    if pe_infeasible < 0.95:
        y_lo = min(raroc_sweep) * 100 - 1
        y_hi = max(raroc_sweep) * 100 + 1
        fig.add_shape(
            type="rect",
            x0=pe_infeasible * 100, x1=97,
            y0=y_lo, y1=y_hi,
            fillcolor="rgba(248,113,113,0.12)",
            line=dict(width=0),
            layer="below",
        )
        fig.add_annotation(
            x=(pe_infeasible * 100 + 95) / 2,
            y=y_hi - 0.5,
            text="Zone Infaisable",
            showarrow=False,
            font=dict(color=_DANGER, size=11),
        )

    # ── 2. Weighted RAROC (blue solid) ──
    fig.add_trace(
        go.Scatter(
            x=pe_range * 100, y=raroc_sweep * 100,
            mode="lines", name="Weighted RAROC",
            line=dict(color=_PRIMARY, width=2.5),
            hovertemplate="PE: %{x:.0f}%<br>RAROC: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )

    # ── 3. CET1 Ratio (green dotted) ──
    fig.add_trace(
        go.Scatter(
            x=pe_range * 100, y=cet1_sweep * 100,
            mode="lines", name="CET1 Ratio",
            line=dict(color=_ACCENT, width=2, dash="dot"),
            hovertemplate="PE: %{x:.0f}%<br>CET1: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    # ── 4. CET1 threshold (red dashed) ──
    fig.add_trace(
        go.Scatter(
            x=[0, 95], y=[cet1_target * 100, cet1_target * 100],
            mode="lines",
            name=f"Seuil CET1 ({cet1_target:.0%})",
            line=dict(color=_DANGER, width=1.5, dash="dash"),
            hoverinfo="skip",
        ),
        secondary_y=True,
    )

    # ── 5. Ideal Markowitz (diamond) ──
    fig.add_trace(
        go.Scatter(
            x=[pe_free * 100], y=[raroc_at_free * 100],
            mode="markers+text",
            name=f"Ideal: {pe_free:.0%}",
            marker=dict(size=12, symbol="diamond", color=_WARNING,
                        line=dict(color=_BG, width=1.5)),
            text=[f"Ideal: {pe_free:.0%}"],
            textposition="top center",
            textfont=dict(color=_WARNING, size=10),
            hovertemplate=f"<b>Ideal Markowitz</b><br>PE: {pe_free:.0%}<br>RAROC: {raroc_at_free:.2%}<extra></extra>",
        ),
        secondary_y=False,
    )

    # ── 6. Final Allocation (star) ──
    fig.add_trace(
        go.Scatter(
            x=[pe_alloc * 100], y=[raroc_at_alloc * 100],
            mode="markers+text",
            name="Allocation Finale",
            marker=dict(size=14, symbol="star", color=_TEXT,
                        line=dict(color=_BG, width=1)),
            text=[f"{raroc_at_alloc:.2%}"],
            textposition="bottom center",
            textfont=dict(color=_TEXT, size=10),
            hovertemplate=f"<b>Allocation Finale</b><br>PE: {pe_alloc:.0%}<br>RAROC: {raroc_at_alloc:.2%}<extra></extra>",
        ),
        secondary_y=False,
    )

    # ── Layout — clean, no double gridlines ──
    fig.update_layout(
        title=dict(
            text="Frontiere Efficiente : RAROC vs PE Allocation",
            font=dict(color=_TEXT, size=14),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_MUTED, size=11),
        height=400,
        margin=dict(l=60, r=60, t=45, b=65),
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center",
            font=dict(color=_MUTED, size=9),
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    # X-axis
    fig.update_xaxes(
        title=dict(text="PE Allocation (%)", font=dict(color=_MUTED, size=11)),
        range=[0, 95], dtick=10, ticksuffix="%",
        gridcolor="rgba(148,163,184,0.08)",
        zerolinecolor="rgba(148,163,184,0.1)",
    )

    # Left Y — RAROC (with subtle grid)
    fig.update_yaxes(
        title=dict(text="Weighted RAROC (%)", font=dict(color=_PRIMARY, size=11)),
        tickformat=".1f", ticksuffix="%",
        gridcolor="rgba(148,163,184,0.08)",
        zerolinecolor="rgba(148,163,184,0.1)",
        secondary_y=False,
    )

    # Right Y — CET1 (NO grid to avoid overlap)
    fig.update_yaxes(
        title=dict(text="CET1 Ratio (%)", font=dict(color=_ACCENT, size=11)),
        tickformat=".0f", ticksuffix="%",
        showgrid=False,
        secondary_y=True,
    )

    return fig


# ──────────────────────────────────────────────
# SANKEY CAPITAL FLOW (capital story)
# ──────────────────────────────────────────────

def plot_capital_sankey(raroc_mc: pl.DataFrame, class_weights: dict = None) -> go.Figure:
    """Sankey diagram: capital flow from source to asset classes to value outcome.

    Left: Total Capital
    Middle: 10 Asset Classes (flow width = capital consumed)
    Right: Value Created (RAROC > k_e) vs Value Destroyed (RAROC < k_e)

    Args:
        raroc_mc: DataFrame from compute_raroc_multiclass() (10 rows + Total).
        class_weights: Optional weights dict (unused, kept for API consistency).

    Returns:
        Plotly Figure with Sankey diagram.
    """
    from ifrs9_cockpit.config import ASSET_CLASS_MAP

    df = raroc_mc.filter(pl.col("asset_class") != "Total")
    if len(df) == 0:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Capital Flow — Aucune donnee", height=200))
        return fig

    k_e = 0.13  # hurdle rate

    # Node indices:
    # 0: Capital Total
    # 1..N: Asset classes
    # N+1: Value Created
    # N+2: Value Destroyed
    n = len(df)
    idx_source = 0
    idx_created = n + 1
    idx_destroyed = n + 2

    # Short labels for Sankey nodes (long names don't fit)
    _SHORT_LABELS = {
        "corporate_loans": "Corporate",
        "private_equity": "PE",
        "sovereign": "Souverain",
        "retail_mortgage": "Hypothec.",
        "project_finance": "Projet",
        "covered_bonds": "Covered",
        "consumer_credit": "Conso.",
        "trade_finance": "Trade Fin.",
        "interbank": "Interbanc.",
        "structured_products": "Structure",
        "equities": "Equity",
        "corporate_bonds": "Corp Bonds",
        "repos_sft": "Repos",
        "derivatives_cva": "Derives",
    }

    # Build nodes
    node_labels = ["Capital Total"]
    node_colors = ["rgba(99,102,241,0.7)"]  # indigo for source

    _has_label = "label" in df.columns
    _has_capital = "capital" in df.columns
    for row in df.iter_rows(named=True):
        ac_key = row["asset_class"]
        lbl = _SHORT_LABELS.get(ac_key)
        if lbl is None:
            ac = ASSET_CLASS_MAP.get(ac_key)
            lbl = ac.label if ac else (row.get("label", ac_key) if _has_label else ac_key)
        # Append capital amount for context
        cap_md = (row.get("capital", 0) if _has_capital else 0) / 1e9
        node_labels.append(f"{lbl}\n{cap_md:.1f} Md")
        color = _CLASS_COLORS.get(ac_key, _MUTED)
        node_colors.append(color)

    node_labels.append("Valeur Creee")
    node_colors.append(_SUCCESS)
    node_labels.append("Valeur Detruite")
    node_colors.append(_DANGER)

    # Build links
    sources = []
    targets = []
    values = []
    link_colors = []

    total_capital = float(df["capital"].sum())
    if total_capital <= 0:
        total_capital = 1.0  # fallback

    for i, row in enumerate(df.iter_rows(named=True)):
        cap = max(row.get("capital", 0), 0.001)
        raroc = row.get("raroc", 0)

        # Source -> Asset class
        sources.append(idx_source)
        targets.append(i + 1)
        values.append(cap / 1e9)  # in billions
        ac_color = _CLASS_COLORS.get(row["asset_class"], _MUTED)
        # Fade the color for the flow
        hex_c = ac_color.lstrip("#")
        r, g, b = int(hex_c[:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
        link_colors.append(f"rgba({r},{g},{b},0.35)")

        # Asset class -> Value Created or Destroyed
        if raroc >= k_e:
            sources.append(i + 1)
            targets.append(idx_created)
            values.append(cap / 1e9)
            link_colors.append("rgba(52,211,153,0.30)")
        else:
            sources.append(i + 1)
            targets.append(idx_destroyed)
            values.append(cap / 1e9)
            link_colors.append("rgba(248,113,113,0.30)")

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=22,
            line=dict(color=_BG, width=1),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
        textfont=dict(color=_TEXT, size=11),
    ))

    # Count value creators / destroyers
    n_created = sum(1 for r in df.iter_rows(named=True) if r.get("raroc", 0) >= k_e)
    n_destroyed = n - n_created

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_TEXT, size=11),
        height=420,
        margin=dict(l=20, r=20, t=55, b=30),
    )

    fig.add_annotation(
        text="Flux de Capital — Source, Allocation, Creation de Valeur",
        xref="paper", yref="paper", x=0, y=1.06,
        showarrow=False, font=dict(color=_TEXT, size=13), xanchor="left",
    )
    fig.add_annotation(
        text=f"{n_created} creatrices | {n_destroyed} destructrices (seuil k_e={k_e:.0%})",
        xref="paper", yref="paper", x=1.0, y=1.06,
        showarrow=False, font=dict(color=_MUTED, size=10), xanchor="right",
    )

    return fig
