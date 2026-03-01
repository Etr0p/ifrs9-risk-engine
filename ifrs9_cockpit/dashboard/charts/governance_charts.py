"""Charts Gouvernance — conformal, Sobol, VRP, RMT, signatures, TDA, compliance."""

from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.dashboard.charts.base import (
    _PRIMARY, _SECONDARY, _ACCENT, _BG, _CARD, _TEXT, _MUTED,
    _SUCCESS, _WARNING, _DANGER, _INFO, _COLORS,
    _base_layout, _hex_to_rgba,
)


def plot_conformal_bands(
    conformal_result,
    n_display: int = 200,
) -> go.Figure:
    """Bandes de couverture conforme PD (90%) triées par PD croissante.

    Affiche PD point estimate + intervalle [PD_lower, PD_upper] pour
    un sous-échantillon de positions, triées par PD pour lisibilité.

    Args:
        conformal_result: ConformalResult du module engine/conformal.py.
        n_display: Nombre de positions à afficher (sous-échantillon).

    Returns:
        Figure Plotly avec bandes conformes.
    """
    pd_pt = conformal_result.pd_point
    pd_lo = conformal_result.pd_lower
    pd_hi = conformal_result.pd_upper
    coverage = conformal_result.coverage_level

    # Sous-échantillonner si nécessaire
    n = len(pd_pt)
    if n > n_display:
        step = n // n_display
        idx = np.arange(0, n, step)[:n_display]
    else:
        idx = np.arange(n)

    # Trier par PD croissante
    sort_idx = np.argsort(pd_pt[idx])
    x = np.arange(len(sort_idx))
    pd_pt_s = pd_pt[idx][sort_idx]
    pd_lo_s = pd_lo[idx][sort_idx]
    pd_hi_s = pd_hi[idx][sort_idx]

    fig = go.Figure()

    # Bande conforme (fill between)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([pd_hi_s, pd_lo_s[::-1]]),
        fill="toself",
        fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"Intervalle {coverage:.0%}",
        hoverinfo="skip",
    ))

    # PD point estimate
    fig.add_trace(go.Scatter(
        x=x,
        y=pd_pt_s,
        mode="lines",
        line=dict(color=_PRIMARY, width=1.5),
        name="PD point",
    ))

    # Bornes
    fig.add_trace(go.Scatter(
        x=x,
        y=pd_hi_s,
        mode="lines",
        line=dict(color=_ACCENT, width=0.8, dash="dot"),
        name="PD upper",
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=pd_lo_s,
        mode="lines",
        line=dict(color=_ACCENT, width=0.8, dash="dot"),
        name="PD lower",
        showlegend=False,
    ))

    layout = _base_layout(
        f"Conformal Prediction — Couverture {coverage:.0%} (q={conformal_result.quantile_residual:.4f})",
        height=350,
    )
    layout["xaxis"]["title"] = dict(text="Positions (triees par PD)", font=dict(color=_MUTED, size=11))
    layout["yaxis"]["title"] = dict(text="PD", font=dict(color=_MUTED, size=11))
    fig.update_layout(**layout)

    return fig


def plot_sobol_indices(sobol_result) -> go.Figure:
    """Barres horizontales S1 (premier ordre) et ST (total) par variable.

    Visualise la décomposition de variance de l'ECL : quelle variable
    macro explique quel % de la variance, et quelles sont les interactions.

    Args:
        sobol_result: SobolResult du module engine/sobol_analysis.py.

    Returns:
        Figure Plotly avec S1 et ST en barres groupées.
    """
    var_names = sobol_result.variable_names
    s1_vals = [sobol_result.s1.get(v, 0) for v in var_names]
    st_vals = [sobol_result.st.get(v, 0) for v in var_names]

    # Labels français
    _labels = {
        "unemployment_rate": "Chomage",
        "gdp_growth": "PIB",
        "interest_rate": "Taux directeur",
        "hpi_growth": "Prix immo (HPI)",
        "inflation_rate": "Inflation",
    }
    labels = [_labels.get(v, v) for v in var_names]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=labels,
        x=s1_vals,
        name="S1 (1er ordre)",
        orientation="h",
        marker_color=_PRIMARY,
        text=[f"{v:.1%}" for v in s1_vals],
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        y=labels,
        x=st_vals,
        name="ST (total)",
        orientation="h",
        marker_color=_ACCENT,
        text=[f"{v:.1%}" for v in st_vals],
        textposition="auto",
    ))

    layout = _base_layout("Indices de Sobol — Decomposition variance ECL", height=320)
    layout["xaxis"]["title"] = dict(text="Part de variance expliquee", font=dict(color=_MUTED, size=11))
    layout["barmode"] = "group"
    fig.update_layout(**layout)

    return fig


def plot_vrp_regime(vrp_result) -> go.Figure:
    """Jauge VRP avec zones de régime colorées.

    Affiche le VRP courant sur une échelle avec 4 zones :
    complaisance (bleu), normal (vert), stress (ambre), panique (rouge).

    Args:
        vrp_result: VRPResult du module engine/vrp.py.

    Returns:
        Figure Plotly indicator gauge.
    """
    vrp_val = vrp_result.vrp
    regime = vrp_result.regime

    regime_colors = {
        "complaisance": _INFO,
        "normal": _SUCCESS,
        "stress": _WARNING,
        "panique": _DANGER,
    }

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=vrp_val,
        number=dict(suffix="%", font=dict(color=_TEXT, size=28)),
        delta=dict(
            reference=4.0,
            increasing=dict(color=_DANGER),
            decreasing=dict(color=_SUCCESS),
        ),
        title=dict(
            text=f"VRP — Regime: {regime.capitalize()}",
            font=dict(color=_TEXT, size=14),
        ),
        gauge=dict(
            axis=dict(range=[-4, 16], tickfont=dict(color=_MUTED, size=10)),
            bar=dict(color=regime_colors.get(regime, _PRIMARY)),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(148,163,184,0.3)",
            steps=[
                dict(range=[-4, 0], color="rgba(59,130,246,0.15)"),
                dict(range=[0, 4], color="rgba(34,197,94,0.15)"),
                dict(range=[4, 8], color="rgba(251,191,36,0.15)"),
                dict(range=[8, 16], color="rgba(239,68,68,0.15)"),
            ],
            threshold=dict(
                line=dict(color=_TEXT, width=2),
                thickness=0.8,
                value=vrp_val,
            ),
        ),
    ))

    layout = _base_layout(height=280)
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(**layout)

    return fig


def plot_rmt_eigenvalues(rmt_result) -> go.Figure:
    """Spectre des valeurs propres avant/après débruitage RMT.

    Superpose les valeurs propres brutes et nettoyées, avec la borne
    Marchenko-Pastur en ligne horizontale.

    Args:
        rmt_result: RMTResult du module engine/rmt.py.

    Returns:
        Figure Plotly avec barres eigenvalues + ligne MP.
    """
    eig_raw = rmt_result.eigenvalues_raw
    eig_clean = rmt_result.eigenvalues_clean
    mp_upper = rmt_result.mp_upper
    n = len(eig_raw)

    x_labels = [f"\u03bb{i+1}" for i in range(n)]

    fig = go.Figure()

    # Barres valeurs propres brutes
    fig.add_trace(go.Bar(
        x=x_labels,
        y=eig_raw,
        name="Brutes",
        marker_color=_DANGER,
        opacity=0.5,
    ))

    # Barres valeurs propres nettoyées
    fig.add_trace(go.Bar(
        x=x_labels,
        y=eig_clean,
        name="Nettoyees (RMT)",
        marker_color=_PRIMARY,
        opacity=0.8,
    ))

    # Borne Marchenko-Pastur
    fig.add_hline(
        y=mp_upper,
        line=dict(color=_WARNING, width=2, dash="dash"),
        annotation_text=f"\u03bb_max MP = {mp_upper:.2f}",
        annotation_position="top right",
        annotation_font=dict(color=_WARNING, size=11),
    )

    layout = _base_layout(
        f"RMT — {rmt_result.n_signal} signal / {rmt_result.n_noise} bruit "
        f"({rmt_result.noise_fraction:.0%} variance bruit)",
        height=320,
    )
    layout["xaxis"]["title"] = dict(text="Valeurs propres", font=dict(color=_MUTED, size=11))
    layout["yaxis"]["title"] = dict(text="Magnitude", font=dict(color=_MUTED, size=11))
    layout["barmode"] = "group"
    fig.update_layout(**layout)

    return fig


def plot_signature_heatmap(signature_result) -> go.Figure:
    """Heatmap des coefficients de signature ordre 2.

    Affiche la matrice d*d des produits iteres de Chen pour
    visualiser les interactions temporelles entre variables macro.

    Args:
        signature_result: SignatureResult du module engine/signatures.py.

    Returns:
        Figure Plotly heatmap.
    """
    d = signature_result.n_dims
    features = signature_result.features

    # Extraire la sous-matrice d'ordre 2 (apres les d features d'ordre 1)
    if signature_result.order >= 2 and len(features) >= d + d * d:
        s2 = features[d:d + d * d].reshape(d, d)
    else:
        s2 = np.diag(features[:d])

    names = signature_result.feature_names[:d]
    labels = [n.replace("sig1_", "") for n in names]

    fig = go.Figure(go.Heatmap(
        z=s2,
        x=labels,
        y=labels,
        colorscale=[
            [0, _DANGER],
            [0.5, _BG],
            [1, _SUCCESS],
        ],
        text=[[f"{v:.2f}" for v in row] for row in s2],
        texttemplate="%{text}",
        textfont=dict(size=10, color=_TEXT),
        hovertemplate="S2(%{y}, %{x}) = %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="S2", font=dict(color=_MUTED)),
            tickfont=dict(color=_MUTED),
        ),
        zmid=0,
    ))

    layout = _base_layout("Signature Ordre 2 — Interactions Temporelles Macro", height=350)
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**layout)
    return fig


def plot_tda_fragility(tda_result) -> go.Figure:
    """Jauge de fragilite topologique + diagramme de persistance.

    Args:
        tda_result: TDAResult du module engine/tda.py.

    Returns:
        Figure Plotly avec gauge + scatter (H0 + H1).
    """
    fig = go.Figure()

    # ── Gauge indicator (left panel) ──
    frag = tda_result.fragility_index
    gauge_color = _SUCCESS if frag < 0.4 else (_WARNING if frag < 0.7 else _DANGER)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=frag,
        number=dict(font=dict(color=_TEXT, size=28)),
        title=dict(text="Fragilite Topologique", font=dict(color=_TEXT, size=12)),
        domain=dict(x=[0.0, 0.38], y=[0.12, 0.88]),
        gauge=dict(
            axis=dict(range=[0, 1], tickfont=dict(color=_MUTED, size=9)),
            bar=dict(color=gauge_color),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(148,163,184,0.3)",
            steps=[
                dict(range=[0, 0.4], color="rgba(34,197,94,0.15)"),
                dict(range=[0.4, 0.7], color="rgba(251,191,36,0.15)"),
                dict(range=[0.7, 1], color="rgba(239,68,68,0.15)"),
            ],
        ),
    ))

    # ── Persistence diagram (right panel) ──
    # Collect all bars (H0 + H1) for a richer diagram
    all_bars = []
    for b in tda_result.bars_h0:
        if b.death < 1.0:  # skip the infinite bar
            all_bars.append(("H0", b))
    for b in tda_result.bars_h1:
        all_bars.append(("H1", b))

    if all_bars:
        # H0 bars (composantes)
        h0_bars = [b for dim, b in all_bars if dim == "H0"]
        if h0_bars:
            fig.add_trace(go.Scatter(
                x=[b.birth for b in h0_bars],
                y=[b.death for b in h0_bars],
                mode="markers",
                marker=dict(
                    size=[max(6, b.persistence * 25) for b in h0_bars],
                    color=_PRIMARY,
                    opacity=0.6,
                    symbol="circle",
                ),
                hovertemplate="H0<br>Naissance: %{x:.3f}<br>Mort: %{y:.3f}<extra></extra>",
                name="H0 (composantes)",
                xaxis="x", yaxis="y",
            ))

        # H1 bars (trous)
        h1_bars = [b for dim, b in all_bars if dim == "H1"]
        if h1_bars:
            fig.add_trace(go.Scatter(
                x=[b.birth for b in h1_bars],
                y=[b.death for b in h1_bars],
                mode="markers",
                marker=dict(
                    size=[max(8, b.persistence * 30) for b in h1_bars],
                    color=_ACCENT,
                    symbol="diamond",
                ),
                hovertemplate="H1<br>Naissance: %{x:.3f}<br>Mort: %{y:.3f}<extra></extra>",
                name="H1 (trous)",
                xaxis="x", yaxis="y",
            ))

        # Diagonal (birth=death line)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color=_MUTED, width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
            xaxis="x", yaxis="y",
        ))
    else:
        fig.add_annotation(
            text="Aucune structure topologique detectee",
            xref="x", yref="y",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=_MUTED, size=12),
        )

    layout = _base_layout(
        f"TDA Fragilite — {tda_result.n_components_at_threshold} composantes a eps=0.5"
        f" | corr moy: {tda_result.mean_correlation:.2f}",
        height=340,
    )
    layout["xaxis"] = dict(
        title=dict(text="Naissance (eps)", font=dict(color=_MUTED, size=10)),
        gridcolor="rgba(148,163,184,0.1)",
        range=[0, 1],
        domain=[0.45, 0.98],
    )
    layout["yaxis"] = dict(
        title=dict(text="Mort (eps)", font=dict(color=_MUTED, size=10)),
        gridcolor="rgba(148,163,184,0.1)",
        range=[0, 1],
    )
    layout["legend"] = dict(
        font=dict(color=_MUTED, size=9),
        bgcolor="rgba(0,0,0,0)",
        x=0.70, y=0.25,
    )
    fig.update_layout(**layout)
    return fig


def plot_compliance_gates(compliance_result) -> go.Figure:
    """Barre horizontale des gates de conformite avec couleurs PASS/WARN/FAIL.

    Args:
        compliance_result: ComplianceResult du module engine/compliance_gates.py.

    Returns:
        Figure Plotly horizontal bars.
    """
    gates = compliance_result.gates
    if not gates:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Compliance Gates — Aucun gate", height=200))
        return fig

    gate_names = [g.gate_name for g in gates]
    statuses = [g.status.value for g in gates]

    scores = []
    colors = []
    for g in gates:
        if g.status.value == "pass":
            scores.append(1.0)
            colors.append(_SUCCESS)
        elif g.status.value == "warning":
            scores.append(0.5)
            colors.append(_WARNING)
        else:
            scores.append(0.0)
            colors.append(_DANGER)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=gate_names,
        x=scores,
        orientation="h",
        marker_color=colors,
        text=[s.upper() for s in statuses],
        textposition="inside",
        textfont=dict(color="white", size=10),
        hovertemplate=[
            f"<b>{g.gate_name}</b><br>"
            f"Statut: {g.status.value.upper()}<br>"
            f"Valeur: {g.value:.4f}<br>"
            f"Seuil: {g.threshold:.4f}<br>"
            f"{g.message}<br>"
            f"Ref: {g.regulation}<extra></extra>"
            for g in gates
        ],
    ))

    n_pass = compliance_result.n_pass
    n_warn = compliance_result.n_warning
    n_fail = compliance_result.n_fail
    status_text = "CONFORME" if compliance_result.is_compliant else "NON CONFORME"
    status_color = _SUCCESS if compliance_result.is_compliant else _DANGER

    layout = _base_layout(
        f"Compliance Gates — {status_text} ({n_pass}P / {n_warn}W / {n_fail}F)",
        height=max(250, len(gates) * 35),
    )
    layout["xaxis"]["range"] = [0, 1.1]
    layout["xaxis"]["title"] = dict(text="Score", font=dict(color=_MUTED, size=10))
    layout["yaxis"]["autorange"] = "reversed"
    layout["title"]["font"]["color"] = status_color
    fig.update_layout(**layout)
    return fig
