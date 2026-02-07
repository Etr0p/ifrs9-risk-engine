"""Graphiques Plotly pour le dashboard IFRS 9.

Tous les graphiques utilisent le thème sombre cohérent avec le CSS
et retournent des objets plotly.graph_objects.Figure prêts à afficher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.config import DASHBOARD_CONFIG


# Palette cohérente avec le CSS
_PRIMARY = DASHBOARD_CONFIG.theme_primary
_SECONDARY = DASHBOARD_CONFIG.theme_secondary
_ACCENT = DASHBOARD_CONFIG.theme_accent
_BG = DASHBOARD_CONFIG.theme_bg_dark
_CARD = DASHBOARD_CONFIG.theme_bg_card
_TEXT = DASHBOARD_CONFIG.theme_text
_MUTED = DASHBOARD_CONFIG.theme_text_muted
_COLORS = [_PRIMARY, _ACCENT, "#F59E0B", "#EF4444", _SECONDARY, "#06B6D4"]


def _base_layout(title: str = "", height: int = 400) -> dict:
    """Layout Plotly de base avec thème sombre.

    Args:
        title: Titre du graphique.
        height: Hauteur en pixels.

    Returns:
        Dictionnaire de layout.
    """
    return dict(
        title=dict(text=title, font=dict(color=_TEXT, size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_MUTED, size=11),
        height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(
            font=dict(color=_MUTED, size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.1)",
            zerolinecolor="rgba(148,163,184,0.15)",
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.1)",
            zerolinecolor="rgba(148,163,184,0.15)",
        ),
    )


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
) -> go.Figure:
    """Courbes ROC comparatives des modèles PD.

    Args:
        roc_data: Dict {model_name: (fpr, tpr, auc)}.

    Returns:
        Figure Plotly avec les courbes ROC superposées.
    """
    fig = go.Figure()

    for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{name} (AUC={auc_val:.3f})",
            line=dict(color=_COLORS[i % len(_COLORS)], width=2.5),
        ))

    # Diagonale de référence
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color=_MUTED, width=1, dash="dash"),
        showlegend=False,
    ))

    layout = _base_layout("ROC Curves — Benchmark PD Models", height=420)
    layout["xaxis"]["title"] = "False Positive Rate"
    layout["yaxis"]["title"] = "True Positive Rate"
    fig.update_layout(**layout)

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str = "LR_WoE",
    top_n: int = 10,
) -> go.Figure:
    """Barres horizontales d'importance des features.

    Args:
        importance_df: DataFrame avec colonnes 'model', 'feature', 'importance'.
        model_name: Modèle à afficher.
        top_n: Nombre de features à afficher.

    Returns:
        Figure Plotly.
    """
    df = (
        importance_df[importance_df["model"] == model_name]
        .nlargest(top_n, "importance")
        .sort_values("importance")
    )

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["importance"],
            colorscale=[[0, _SECONDARY], [1, _PRIMARY]],
        ),
        text=df["importance"].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
        textfont=dict(color=_TEXT, size=10),
    ))

    layout = _base_layout(f"Feature Importance — {model_name}", height=380)
    layout["xaxis"]["title"] = "Importance relative"
    fig.update_layout(**layout)

    return fig


def plot_stage_distribution(
    stage_summary: pd.DataFrame,
) -> go.Figure:
    """Graphique de distribution des stages (count + EAD).

    Args:
        stage_summary: DataFrame du StagingEngine.get_stage_summary().

    Returns:
        Figure Plotly avec double barres.
    """
    colors = [_ACCENT, "#F59E0B", "#EF4444"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Répartition Clients", "Répartition EAD"),
        specs=[[{"type": "pie"}, {"type": "pie"}]],
    )

    fig.add_trace(go.Pie(
        labels=stage_summary["stage"],
        values=stage_summary["count"],
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(color=_TEXT, size=11),
        hole=0.45,
    ), row=1, col=1)

    fig.add_trace(go.Pie(
        labels=stage_summary["stage"],
        values=stage_summary["total_ead"],
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(color=_TEXT, size=11),
        hole=0.45,
    ), row=1, col=2)

    layout = _base_layout("Distribution des Stages IFRS 9", height=380)
    layout["showlegend"] = False
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(color=_TEXT, size=12))

    return fig


def plot_ecl_by_segment(result_df: pd.DataFrame) -> go.Figure:
    """Barres ECL par segment avec décomposition Base/Adverse.

    Args:
        result_df: DataFrame résultat du ECLCalculator.

    Returns:
        Figure Plotly.
    """
    segments = result_df.groupby("segment").agg(
        ecl_base=("ecl_base", "sum"),
        ecl_adverse=("ecl_adverse", "sum"),
        ecl_weighted=("ecl_weighted", "sum"),
    ).reset_index()
    segments = segments.sort_values("ecl_weighted", ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=segments["segment"],
        x=segments["ecl_base"],
        name="Scénario Base (70%)",
        orientation="h",
        marker_color=_PRIMARY,
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        y=segments["segment"],
        x=segments["ecl_adverse"] - segments["ecl_base"],
        name="Add-on Adverse (30%)",
        orientation="h",
        marker_color="#EF4444",
        opacity=0.7,
    ))

    layout = _base_layout("ECL par Segment — Décomposition Scénarios", height=350)
    layout["barmode"] = "stack"
    layout["xaxis"]["title"] = "ECL (EUR)"
    fig.update_layout(**layout)

    return fig


def plot_transition_matrix(matrix_df: pd.DataFrame) -> go.Figure:
    """Heatmap de la matrice de transition des stages.

    Args:
        matrix_df: DataFrame 3×3 de probabilités de transition.

    Returns:
        Figure Plotly heatmap.
    """
    labels = ["Stage 1", "Stage 2", "Stage 3"]
    z = matrix_df.values

    text = [[f"{val:.1%}" for val in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=14, color=_TEXT),
        colorscale=[
            [0, _CARD],
            [0.5, _PRIMARY],
            [1, "#EF4444"],
        ],
        showscale=False,
    ))

    layout = _base_layout("Matrice de Transition", height=350)
    layout["xaxis"]["title"] = "Stage (t+1)"
    layout["yaxis"]["title"] = "Stage (t)"
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**layout)

    return fig


def plot_waterfall_ecl(waterfall_df: pd.DataFrame) -> go.Figure:
    """Waterfall chart de variation ECL.

    Args:
        waterfall_df: DataFrame du ECLCalculator.compute_waterfall().

    Returns:
        Figure Plotly waterfall.
    """
    measures = []
    for _, row in waterfall_df.iterrows():
        comp = row["component"]
        if comp in ("ECL Ouverture", "ECL Clôture"):
            measures.append("total")
        elif comp == "Variation nette":
            measures.append("total")
        else:
            measures.append("relative")

    colors = []
    for _, row in waterfall_df.iterrows():
        if row["component"] in ("ECL Ouverture", "ECL Clôture"):
            colors.append(_PRIMARY)
        elif row["amount"] >= 0:
            colors.append("#EF4444")
        else:
            colors.append(_ACCENT)

    fig = go.Figure(go.Waterfall(
        name="ECL",
        orientation="v",
        measure=measures,
        x=waterfall_df["component"],
        y=waterfall_df["amount"],
        connector=dict(line=dict(color=_MUTED, width=1)),
        increasing=dict(marker_color="#EF4444"),
        decreasing=dict(marker_color=_ACCENT),
        totals=dict(marker_color=_PRIMARY),
        textposition="outside",
        texttemplate="%{y:,.0f}",
        textfont=dict(color=_TEXT, size=10),
    ))

    layout = _base_layout("Waterfall ECL — Décomposition de la Variation", height=420)
    layout["yaxis"]["title"] = "ECL (EUR)"
    layout["showlegend"] = False
    fig.update_layout(**layout)

    return fig


def plot_ecl_coverage_scatter(result_df: pd.DataFrame) -> go.Figure:
    """Scatter PD vs Coverage par segment.

    Args:
        result_df: DataFrame résultat ECL.

    Returns:
        Figure Plotly scatter.
    """
    seg_data = result_df.groupby("segment").agg(
        pd_mean=("pd_12m", "mean"),
        coverage=("ecl_weighted", "sum"),
        ead_total=("ead", "sum"),
        count=("ecl_weighted", "size"),
    ).reset_index()
    seg_data["coverage_ratio"] = seg_data["coverage"] / seg_data["ead_total"]

    fig = go.Figure()

    for i, row in seg_data.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["pd_mean"]],
            y=[row["coverage_ratio"]],
            mode="markers+text",
            name=row["segment"],
            text=[row["segment"]],
            textposition="top center",
            textfont=dict(color=_TEXT, size=10),
            marker=dict(
                size=max(10, row["count"] / 100),
                color=_COLORS[i % len(_COLORS)],
                line=dict(width=1, color=_TEXT),
            ),
        ))

    layout = _base_layout("PD Moyenne vs Coverage Ratio par Segment", height=400)
    layout["xaxis"]["title"] = "PD Moyenne"
    layout["yaxis"]["title"] = "Coverage Ratio"
    layout["xaxis"]["tickformat"] = ".1%"
    layout["yaxis"]["tickformat"] = ".1%"
    layout["showlegend"] = False
    fig.update_layout(**layout)

    return fig


def plot_iv_table(iv_df: pd.DataFrame) -> go.Figure:
    """Bar chart de l'Information Value par feature.

    Args:
        iv_df: DataFrame du WoEBinner.get_iv_table().

    Returns:
        Figure Plotly.
    """
    df = iv_df.sort_values("iv", ascending=True)

    # Couleur selon la force
    color_map = {
        "Non predictif": _MUTED,
        "Faible": "#06B6D4",
        "Moyen": _ACCENT,
        "Fort": _PRIMARY,
        "Suspect": "#F59E0B",
    }
    colors = [color_map.get(s, _MUTED) for s in df["strength"]]

    fig = go.Figure(go.Bar(
        x=df["iv"],
        y=df["feature"],
        orientation="h",
        marker_color=colors,
        text=df.apply(lambda r: f'{r["iv"]:.3f} ({r["strength"]})', axis=1),
        textposition="outside",
        textfont=dict(color=_TEXT, size=10),
    ))

    layout = _base_layout("Information Value — Pouvoir Prédictif", height=380)
    layout["xaxis"]["title"] = "IV"
    fig.update_layout(**layout)

    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Radar chart comparant les 3 modèles PD.

    Args:
        comparison_df: DataFrame du PDModelSuite.get_comparison_table().

    Returns:
        Figure Plotly radar.
    """
    metrics = ["auc_test", "gini_test", "ks_test"]
    metric_labels = ["AUC", "Gini", "KS"]

    fig = go.Figure()

    for i, (_, row) in enumerate(comparison_df.iterrows()):
        values = [row[m] for m in metrics]
        values.append(values[0])  # Fermer le polygone

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels + [metric_labels[0]],
            fill="toself",
            name=row["model"],
            line=dict(color=_COLORS[i % len(_COLORS)], width=2),
            fillcolor=f"rgba{_hex_to_rgba(_COLORS[i % len(_COLORS)], 0.1)}",
        ))

    layout = _base_layout("Benchmark Modèles PD", height=400)
    layout["polar"] = dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(
            visible=True,
            range=[0.5, 1.0],
            gridcolor="rgba(148,163,184,0.15)",
            color=_MUTED,
        ),
        angularaxis=dict(
            gridcolor="rgba(148,163,184,0.15)",
            color=_TEXT,
        ),
    )
    fig.update_layout(**layout)

    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convertit une couleur hex en tuple RGBA string.

    Args:
        hex_color: Couleur hexadécimale (#RRGGBB).
        alpha: Opacité (0-1).

    Returns:
        String "(r, g, b, a)".
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"({r}, {g}, {b}, {alpha})"
