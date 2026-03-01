"""Arbitrage insight builders."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from dash import dcc, html
import dash_bootstrap_components as dbc

from ifrs9_cockpit.config import (
    DASHBOARD_CONFIG as _CFG,
    BASEL_CONFIG,
)
from ifrs9_cockpit.utils.helpers import format_euro, format_pct
from ifrs9_cockpit.dashboard import ids
from ifrs9_cockpit.dashboard.components.helpers import _rw_label


def build_arbitrage_insight(optimization: Dict[str, Any]) -> html.Div:
    """Build the full allocation optimization table with BL-CVaR details.

    Two-step display matching institutional CRO report format:
        Step 1 — BL-CVaR Libre: unconstrained allocation (what the bank WANTS)
        Step 2 — Normes CET1: constrained allocation (what the bank CAN DO)

    Args:
        optimization: Dict from ``comparator.optimize_allocation()``.

    Returns:
        A styled ``html.Div`` with the full optimization breakdown.
    """
    # Extract all fields
    w_credit = optimization.get("credit_allocation", 0.0)
    w_pe = optimization.get("pe_allocation", 0.0)
    pe_free = optimization.get("pe_free", w_pe)
    pe_band_val = optimization.get("pe_band", [0.02, BASEL_CONFIG.pe_max_allocation])
    pe_band_hi = pe_band_val[1] if isinstance(pe_band_val, (list, tuple)) and len(pe_band_val) > 1 else BASEL_CONFIG.pe_max_allocation
    pe_band_lo = pe_band_val[0] if isinstance(pe_band_val, (list, tuple)) else 0.02
    raroc_portfolio = optimization.get("raroc_portfolio", 0.0)
    raroc_credit = optimization.get("raroc_credit", 0.0)
    raroc_pe = optimization.get("raroc_pe", 0.0)
    sharpe = optimization.get("sharpe", 0.0)
    port_vol = optimization.get("portfolio_vol", 0.0)
    sigma_credit = optimization.get("sigma_credit", 0.0)
    sigma_pe = optimization.get("sigma_pe", 0.0)
    rho = optimization.get("correlation", 0.0)
    k_e = optimization.get("k_e", 0.12)
    rwa_weighted = optimization.get("rwa_weighted", 0.0)
    cet1_ratio = optimization.get("cet1_ratio", 0.0)
    headroom_m = optimization.get("headroom_m", 0.0)
    feasible = optimization.get("feasible", True)
    method = optimization.get("method", "BL-CVaR")
    cvar_95 = optimization.get("cvar_95", None)
    pe_max = BASEL_CONFIG.pe_max_allocation

    # Phase 1 metrics (free allocation)
    credit_free = 1.0 - pe_free
    raroc_free = credit_free * raroc_credit + pe_free * raroc_pe
    port_vol_free = float(
        (credit_free**2 * sigma_credit**2
         + pe_free**2 * sigma_pe**2
         + 2 * credit_free * pe_free * rho * sigma_credit * sigma_pe) ** 0.5
    ) if sigma_credit > 0 else port_vol
    sharpe_free = raroc_free / max(port_vol_free, 1e-9)

    # Phase 2 metrics (constrained)
    raroc_constrained = w_credit * raroc_credit + w_pe * raroc_pe
    sharpe_constrained = raroc_constrained / max(port_vol, 1e-9)

    _muted = _CFG.theme_text_muted
    _text = _CFG.theme_text
    _green = _CFG.color_success
    _yellow = _CFG.color_warning
    _red = _CFG.color_danger

    def _row(label: str, value: str, color: str = _text) -> html.Div:
        return html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "padding": "0.25rem 0",
                "borderBottom": "1px solid rgba(148, 163, 184, 0.06)",
            },
            children=[
                html.Span(label, style={"color": _muted, "fontSize": "0.78rem"}),
                html.Span(value, style={"color": color, "fontSize": "0.78rem", "fontWeight": "600"}),
            ],
        )

    def _section_header(text: str) -> html.Div:
        return html.Div(
            text,
            style={
                "color": _CFG.color_info,
                "fontSize": "0.70rem",
                "fontWeight": "700",
                "letterSpacing": "1.5px",
                "textTransform": "uppercase",
                "padding": "0.6rem 0 0.2rem",
                "borderBottom": "1px solid rgba(59, 130, 246, 0.15)",
                "marginBottom": "0.15rem",
            },
        )

    children = []

    # ── STEP 1 — BL-CVaR LIBRE ──
    children.append(_section_header("Step 1 \u2014 BL-CVaR Libre"))
    children.append(_row("Credit", f"{credit_free:.1%}"))
    children.append(_row("Private Equity", f"{pe_free:.1%}", color=_yellow))
    children.append(_row("RAROC", f"{raroc_free:.2%}"))
    children.append(_row("Sharpe (mu/sigma)", f"{sharpe_free:.2f}", color=_green if sharpe_free > 1 else _text))
    children.append(_row("Portfolio Volatility", f"{port_vol_free:.2%}"))

    # ── STEP 2 — NORMES CET1 ──
    children.append(_section_header("Step 2 \u2014 Normes CET1"))
    children.append(_row("Credit", f"{w_credit:.1%}"))
    children.append(_row("Private Equity", f"{w_pe:.1%}", color=_green if w_pe <= pe_max else _yellow))
    children.append(_row("Adjusted RAROC", f"{raroc_constrained:.2%}"))
    children.append(_row("Adjusted Sharpe", f"{sharpe_constrained:.2f}"))
    if cvar_95 is not None:
        children.append(_row("CVaR 95%", f"{cvar_95:.2%}"))

    # Constraint note — show what reduced PE
    if pe_free > w_pe + 0.005:
        constraint_binding = f"pe_max({pe_max:.0%})"
        if pe_band_hi < pe_max - 0.005:
            constraint_binding = f"pe_band({pe_band_hi:.0%})"
        note = f"PE reduced from {pe_free:.0%} to {w_pe:.0%} ({constraint_binding}, CET1 >= {BASEL_CONFIG.cet1_target:.0%})"
        children.append(
            html.Div(
                note,
                style={
                    "color": _yellow,
                    "fontSize": "0.72rem",
                    "fontStyle": "italic",
                    "padding": "0.3rem 0 0.1rem",
                },
            )
        )

    # ── RAROC BY CHANNEL ──
    children.append(_section_header("RAROC by Channel"))
    children.append(_row("RAROC Credit", f"{raroc_credit:.2%}", color=_green))
    children.append(_row("RAROC PE", f"{raroc_pe:.2%}", color=_green))

    # ── RISK PARAMETERS ──
    children.append(_section_header("Risk Parameters"))
    children.append(_row("sigma Credit", f"{sigma_credit:.2%}"))
    children.append(_row("sigma PE", f"{sigma_pe:.2%}"))
    children.append(_row("Correlation (rho)", f"{rho:.2f}"))
    children.append(_row("k_e (cost of equity)", f"{k_e:.1%}"))

    # ── REGULATORY CONSTRAINTS ──
    children.append(_section_header("Regulatory Constraints"))
    children.append(_row("Weighted RWA", f"{rwa_weighted / 1e9:.1f}Md EUR"))
    children.append(_row("CET1 Ratio", f"{cet1_ratio:.2%}"))
    headroom_md = headroom_m / 1000  # M EUR → Md EUR
    children.append(_row(
        "Headroom",
        f"{headroom_md:+.1f}Md EUR",
        color=_green if headroom_md >= 0 else _red,
    ))

    # Feasible badge
    feasible_color = _green if feasible else _red
    feasible_text = "FEASIBLE" if feasible else "NON FEASIBLE"
    children.append(
        html.Div(
            feasible_text,
            style={
                "textAlign": "center",
                "color": feasible_color,
                "fontSize": "0.80rem",
                "fontWeight": "700",
                "letterSpacing": "2px",
                "padding": "0.8rem 0 0.3rem",
            },
        )
    )

    # Footnote
    pe_note = f"BL-CVaR libre: {pe_free:.0%} PE -> pe_max({pe_band_hi:.0%}) -> final: {w_pe:.0%} PE"
    children.append(
        html.Div(
            pe_note,
            style={"color": _muted, "fontSize": "0.62rem", "fontStyle": "italic", "padding": "0.2rem 0"},
        )
    )

    # Method badge
    method_badge = html.Span(
        method,
        style={
            "backgroundColor": "rgba(59, 130, 246, 0.15)",
            "color": _CFG.color_info,
            "padding": "0.15rem 0.5rem",
            "borderRadius": "10px",
            "fontSize": "0.65rem",
            "fontWeight": "600",
            "marginLeft": "0.5rem",
        },
    )

    return html.Div(
        className="insight-box",
        children=[
            html.Div(
                className="insight-box-header",
                children=[
                    html.Span("\u2696\ufe0f", style={"fontSize": "1.2rem"}),
                    html.H3(["Credit / PE Arbitrage", method_badge]),
                ],
            ),
            html.Div(
                className="insight-box-content",
                children=children,
            ),
        ],
    )


def build_layer2_allocation(
    proportional_df: "pl.DataFrame",
    factor_df: "pl.DataFrame | None" = None,
    optimization: "Dict[str, Any] | None" = None,
    raroc_eva: "pl.DataFrame | None" = None,
) -> html.Div:
    """Build Optimal 2-Layer Allocation panel.

    Matches institutional CRO format with:
        1. Portfolio Arbitrage (Credit vs PE top-level split)
        2. Sector Arbitrage per channel (Current, Optimal, Total Share, RAROC, Action)

    Args:
        proportional_df: 10-cell allocation (sector, canal, proportional_share, risk_amount, rwa).
        factor_df: Macro factor attribution (unused in table, kept for signature compat).
        optimization: Dict from optimize_allocation() with sector_weights and allocations.
        raroc_eva: DataFrame with per-sector RAROC (sector, canal, raroc).

    Returns:
        Styled html.Div.
    """
    import polars as pl

    opt = optimization or {}
    _muted = _CFG.theme_text_muted
    _text = _CFG.theme_text
    _green = _CFG.color_success
    _yellow = _CFG.color_warning
    _red = _CFG.color_danger
    _blue = _CFG.color_info

    w_credit = opt.get("credit_allocation", 0.0)
    w_pe = opt.get("pe_allocation", 0.0)
    w_opt_credit = opt.get("sector_weights_credit", {})
    w_opt_pe = opt.get("sector_weights_pe", {})

    # ── Helpers ──
    _cell = {"fontSize": "0.78rem", "padding": "0.3rem 0.5rem", "textAlign": "center"}
    _cell_left = {**_cell, "textAlign": "left"}
    _hdr_cell = {
        **_cell, "color": _muted, "fontSize": "0.65rem", "fontWeight": "700",
        "letterSpacing": "1px", "textTransform": "uppercase",
        "borderBottom": "1px solid rgba(148,163,184,0.15)",
        "padding": "0.4rem 0.5rem",
    }
    _row_border = "1px solid rgba(148,163,184,0.06)"

    def _kv_row(label: str, value: str, color: str = _text) -> html.Div:
        return html.Div(
            style={"display": "flex", "justifyContent": "space-between",
                    "padding": "0.3rem 0", "borderBottom": _row_border},
            children=[
                html.Span(label, style={"color": _muted, "fontSize": "0.82rem"}),
                html.Span(value, style={"color": color, "fontSize": "0.82rem", "fontWeight": "600"}),
            ],
        )

    def _section_divider(icon: str, text: str) -> html.Div:
        return html.Div(
            style={"padding": "0.7rem 0 0.3rem", "borderBottom": "1px solid rgba(59,130,246,0.2)",
                    "marginBottom": "0.3rem"},
            children=[
                html.Span(icon, style={"marginRight": "0.4rem", "fontSize": "0.9rem"}),
                html.Span(text, style={"color": _text, "fontSize": "0.85rem", "fontWeight": "600"}),
            ],
        )

    def _action_signal(current: float, optimal: float) -> html.Span:
        """Determine Surponderer / Maintenir / Sous-ponderer."""
        diff = optimal - current
        if diff > 0.05:
            return html.Span(
                ["\u25b2 Surponderer"],
                style={"color": _green, "fontSize": "0.75rem", "fontWeight": "600"},
            )
        elif diff < -0.05:
            return html.Span(
                ["\u25bc Sous-ponderer"],
                style={"color": _red, "fontSize": "0.75rem", "fontWeight": "600"},
            )
        return html.Span(
            ["\u2022 Maintenir"],
            style={"color": _muted, "fontSize": "0.75rem"},
        )

    def _get_raroc(sector: str, canal: str) -> float:
        if raroc_eva is None or not isinstance(raroc_eva, pl.DataFrame):
            return 0.0
        row = raroc_eva.filter(
            (pl.col("sector") == sector) & (pl.col("canal") == canal)
        )
        return float(row["raroc"].to_numpy()[0]) if len(row) > 0 else 0.0

    def _sector_table(canal: str, channel_alloc: float, w_opt: dict) -> html.Div:
        """Build a sector table for one channel (Credit or PE)."""
        canal_rows = proportional_df.filter(pl.col("canal") == canal)
        if len(canal_rows) == 0:
            return html.Div()

        label = "Credit" if canal == "Credit" else "Private Equity"
        color = _green if canal == "Credit" else _yellow

        header = html.Div(
            [html.Span(f"{label}", style={"color": _text, "fontWeight": "700", "fontSize": "0.85rem"}),
             html.Span(f" \u2014 {channel_alloc:.1%} of portfolio",
                        style={"color": color, "fontSize": "0.80rem", "fontStyle": "italic"})],
            style={"padding": "0.5rem 0 0.3rem"},
        )

        # Column headers
        cols = ["SECTOR", "CURRENT", "OPTIMAL", "TOTAL SHARE", "RAROC", "ACTION"]
        th_row = html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr 1fr 1fr 1fr 1.5fr"},
            children=[html.Span(c, style={**_hdr_cell, "textAlign": "left" if c == "SECTOR" else "center"}) for c in cols],
        )

        data_rows = []
        for r in canal_rows.iter_rows(named=True):
            sector = str(r["sector"])
            current = float(r.get("proportional_share", 0))
            optimal = float(w_opt.get(sector, 1.0 / max(len(canal_rows), 1)))
            total_share = channel_alloc * optimal
            raroc = _get_raroc(sector, canal)

            opt_color = _text
            if optimal > current + 0.05:
                opt_color = _green
            elif optimal < current - 0.05:
                opt_color = _red

            data_rows.append(html.Div(
                style={"display": "grid", "gridTemplateColumns": "2fr 1fr 1fr 1fr 1fr 1.5fr",
                        "borderBottom": _row_border, "alignItems": "center"},
                children=[
                    html.Span(sector, style={**_cell_left, "color": _text}),
                    html.Span(f"{current:.1%}", style={**_cell, "color": _muted}),
                    html.Span(f"{optimal:.1%}", style={**_cell, "color": opt_color, "fontWeight": "700"}),
                    html.Span(f"{total_share:.1%}", style={**_cell, "color": _muted}),
                    html.Span(
                        f"{raroc:.2%}",
                        style={**_cell, "color": _green if raroc > 0 else _red},
                    ),
                    html.Span(
                        _action_signal(current, optimal),
                        style={**_cell},
                    ),
                ],
            ))

        return html.Div([header, th_row] + data_rows)

    # ── Build component ──
    children = []

    # 1. Portfolio Arbitrage
    children.append(_section_divider("\u2460", "Portfolio Arbitrage"))
    children.append(_kv_row("Credit", f"{w_credit:.1%}", color=_green))
    children.append(_kv_row("Private Equity", f"{w_pe:.1%}", color=_yellow))

    # 2. Sector Arbitrage
    children.append(_section_divider("\u2461", "Sector Arbitrage"))
    children.append(_sector_table("Credit", w_credit, w_opt_credit))
    children.append(_sector_table("PE", w_pe, w_opt_pe))

    return html.Div(
        className="insight-box",
        children=[
            html.Div(
                className="insight-box-header",
                children=[
                    html.Span("\U0001f4ca", style={"fontSize": "1.2rem"}),
                    html.H3("Optimal 2-Layer Allocation"),
                ],
            ),
            html.Div(
                className="insight-box-content",
                children=children,
            ),
        ],
    )
