"""Charts Credit — PD models, staging, ECL, SHAP, backtesting."""

from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.dashboard.charts.base import (
    _PRIMARY, _SECONDARY, _ACCENT, _BG, _CARD, _TEXT, _MUTED,
    _SUCCESS, _WARNING, _DANGER, _INFO, _COLORS,
    _FEATURE_LABELS, _prettify_feature, _base_layout, _hex_to_rgba,
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
    importance_df: pl.DataFrame,
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
        importance_df.filter(pl.col("model") == model_name)
        .sort("importance", descending=True)
        .head(top_n)
        .sort("importance")
    )

    labels = [_prettify_feature(f) for f in df["feature"].to_list()]

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=labels,
        orientation="h",
        marker=dict(
            color=df["importance"].to_numpy(),
            colorscale=[[0, _SECONDARY], [1, _PRIMARY]],
        ),
        text=[f"{v:.3f}" for v in df["importance"].to_list()],
        textposition="outside",
        textfont=dict(color=_TEXT, size=10),
    ))

    layout = _base_layout(f"Feature Importance — {model_name}", height=380)
    layout["xaxis"]["title"] = "Importance relative"
    fig.update_layout(**layout)

    return fig


def plot_stage_distribution(
    stage_summary,
) -> go.Figure:
    """Graphique de distribution des stages (count + EAD).

    Args:
        stage_summary: DataFrame (Polars or Pandas) du StagingEngine.get_stage_summary().

    Returns:
        Figure Plotly avec double barres.
    """
    colors = [_ACCENT, _WARNING, _DANGER]

    # Convert Polars Series to list for Plotly compatibility
    _stages = stage_summary["stage"].to_list() if hasattr(stage_summary["stage"], "to_list") else stage_summary["stage"]
    _counts = stage_summary["count"].to_list() if hasattr(stage_summary["count"], "to_list") else stage_summary["count"]
    _eads = stage_summary["total_ead"].to_list() if hasattr(stage_summary["total_ead"], "to_list") else stage_summary["total_ead"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Répartition Clients", "Répartition EAD"),
        specs=[[{"type": "pie"}, {"type": "pie"}]],
    )

    fig.add_trace(go.Pie(
        labels=_stages,
        values=_counts,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(color=_TEXT, size=11),
        hole=0.45,
    ), row=1, col=1)

    fig.add_trace(go.Pie(
        labels=_stages,
        values=_eads,
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


def plot_ecl_by_segment(result_df: pl.DataFrame) -> go.Figure:
    """Barres ECL par segment avec décomposition multi-scénarios.

    Args:
        result_df: DataFrame résultat du ECLCalculator.

    Returns:
        Figure Plotly.
    """
    # Détection dynamique des colonnes de scénarios
    scenario_cols = [c for c in result_df.columns if c.startswith("ecl_") and c != "ecl_weighted"]
    agg_exprs = [pl.col(c).sum().alias(c) for c in scenario_cols]
    agg_exprs.append(pl.col("ecl_weighted").sum())

    grp_col = "sector" if "sector" in result_df.columns else "segment"
    segments = result_df.group_by(grp_col).agg(agg_exprs).sort("ecl_weighted")

    fig = go.Figure()

    scenario_colors = {
        "ecl_base": (_PRIMARY, "Base (50%)"),
        "ecl_adverse": (_DANGER, "Adverse (25%)"),
        "ecl_favorable": (_ACCENT, "Favorable (25%)"),
    }

    for col in scenario_cols:
        color, label = scenario_colors.get(col, (_SECONDARY, col))
        fig.add_trace(go.Bar(
            y=segments[grp_col],
            x=segments[col],
            name=f"Scénario {label}",
            orientation="h",
            marker_color=color,
            opacity=0.85,
        ))

    layout = _base_layout("ECL par Segment — Décomposition Scénarios", height=350)
    layout["barmode"] = "group"
    layout["xaxis"]["title"] = "ECL (EUR)"
    fig.update_layout(**layout)

    return fig


def plot_transition_matrix(matrix_df) -> go.Figure:
    """Heatmap de la matrice de transition des stages.

    Args:
        matrix_df: DataFrame (Polars or Pandas) de probabilités de transition.

    Returns:
        Figure Plotly heatmap.
    """
    import polars as pl
    labels = ["Stage 1", "Stage 2", "Stage 3"]
    # Support both Polars (from_stage + 3 cols) and Pandas (3x3 indexed)
    if isinstance(matrix_df, pl.DataFrame):
        z = matrix_df.select(labels).to_numpy()
    else:
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
            [1, _DANGER],
        ],
        showscale=False,
    ))

    layout = _base_layout("Matrice de Transition", height=350)
    layout["xaxis"]["title"] = "Stage (t+1)"
    layout["yaxis"]["title"] = "Stage (t)"
    layout["yaxis"]["autorange"] = "reversed"
    fig.update_layout(**layout)

    return fig


def plot_waterfall_ecl(waterfall_df: pl.DataFrame) -> go.Figure:
    """Waterfall chart de variation ECL.

    Args:
        waterfall_df: DataFrame du ECLCalculator.compute_waterfall().

    Returns:
        Figure Plotly waterfall.
    """
    measures = []
    for row in waterfall_df.iter_rows(named=True):
        comp = row["component"]
        if comp in ("ECL Ouverture", "ECL Clôture"):
            measures.append("total")
        elif comp == "Variation nette":
            measures.append("total")
        else:
            measures.append("relative")

    colors = []
    for row in waterfall_df.iter_rows(named=True):
        if row["component"] in ("ECL Ouverture", "ECL Clôture"):
            colors.append(_PRIMARY)
        elif row["amount"] >= 0:
            colors.append(_DANGER)
        else:
            colors.append(_ACCENT)

    fig = go.Figure(go.Waterfall(
        name="ECL",
        orientation="v",
        measure=measures,
        x=waterfall_df["component"],
        y=waterfall_df["amount"],
        connector=dict(line=dict(color=_MUTED, width=1)),
        increasing=dict(marker_color=_DANGER),
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


def plot_ecl_coverage_scatter(result_df: pl.DataFrame) -> go.Figure:
    """Scatter PD vs Coverage par segment.

    Args:
        result_df: DataFrame résultat ECL.

    Returns:
        Figure Plotly scatter.
    """
    grp_col = "sector" if "sector" in result_df.columns else "segment"
    seg_data = result_df.group_by(grp_col).agg(
        pl.col("pd_12m").mean().alias("pd_mean"),
        pl.col("ecl_weighted").sum().alias("coverage"),
        pl.col("ead").sum().alias("ead_total"),
        pl.col("ecl_weighted").count().alias("count"),
    ).with_columns(
        (pl.col("coverage") / pl.col("ead_total")).alias("coverage_ratio"),
    )

    fig = go.Figure()

    for i, row in enumerate(seg_data.iter_rows(named=True)):
        fig.add_trace(go.Scatter(
            x=[row["pd_mean"]],
            y=[row["coverage_ratio"]],
            mode="markers+text",
            name=row[grp_col],
            text=[row[grp_col]],
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


def plot_iv_table(iv_df: pl.DataFrame) -> go.Figure:
    """Bar chart de l'Information Value par feature.

    Args:
        iv_df: DataFrame du WoEBinner.get_iv_table().

    Returns:
        Figure Plotly.
    """
    df = iv_df.sort("iv")

    # Couleur selon la force
    color_map = {
        "Non predictif": _MUTED,
        "Faible": _INFO,
        "Moyen": _ACCENT,
        "Fort": _PRIMARY,
        "Suspect": _WARNING,
    }
    colors = [color_map.get(s, _MUTED) for s in df["strength"].to_list()]

    labels = [_prettify_feature(f) for f in df["feature"].to_list()]

    fig = go.Figure(go.Bar(
        x=df["iv"],
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f'{r["iv"]:.3f} ({r["strength"]})' for r in df.iter_rows(named=True)],
        textposition="outside",
        textfont=dict(color=_TEXT, size=10),
    ))

    layout = _base_layout("Information Value — Pouvoir Prédictif", height=380)
    layout["xaxis"]["title"] = "IV"
    fig.update_layout(**layout)

    return fig


def plot_model_comparison(comparison_df: pl.DataFrame) -> go.Figure:
    """Radar chart 7 axes comparant les 3 modeles PD.

    Axes (tous normalises [0, 1], higher = better) :
        AUC, Gini, KS — discrimination
        1-Brier — calibration accuracy
        1-LogLoss — probabilistic calibration
        Stabilite — 1 - PSI (temporal drift)
        Robustesse — 1 - overfit_gap (generalization)

    Args:
        comparison_df: DataFrame du PDModelSuite.get_comparison_table().

    Returns:
        Figure Plotly radar 7 branches.
    """
    # 7 axes: (column_key, display_label, transform)
    # transform: None=raw [0,1], "invert"=1-v, "overfit"=1-|v|*5
    _RADAR_AXES = [
        ("auc_test",       "AUC",         None),
        ("gini_test",      "Gini",        None),
        ("ks_test",        "KS",          None),
        ("brier_test",     "1-Brier",     "invert"),
        ("logloss_test",   "1-LogLoss",   "invert"),
        ("psi",            "Stabilite",   "invert"),
        ("overfit_gap",    "Robustesse",  "overfit"),
    ]

    metric_labels = [a[1] for a in _RADAR_AXES]
    col_names = comparison_df.columns

    fig = go.Figure()

    for i, row in enumerate(comparison_df.iter_rows(named=True)):
        values = []
        for key, _, transform in _RADAR_AXES:
            v = float(row.get(key, 0.0)) if key in col_names else 0.0
            if transform == "invert":
                v = max(0.0, 1.0 - min(abs(v), 1.0))
            elif transform == "overfit":
                # Scale overfit gap: 0.02 gap → 0.90, 0.10 gap → 0.50, 0.20 → 0.0
                v = max(0.0, 1.0 - min(abs(v) * 5.0, 1.0))
            else:
                v = max(0.0, min(v, 1.0))
            values.append(round(v, 4))
        values.append(values[0])  # Fermer le polygone

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels + [metric_labels[0]],
            fill="toself",
            name=row["model"],
            line=dict(color=_COLORS[i % len(_COLORS)], width=2),
            fillcolor=f"rgba{_hex_to_rgba(_COLORS[i % len(_COLORS)], 0.1)}",
        ))

    layout = _base_layout("Benchmark Modeles PD", height=480)
    layout["polar"] = dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(
            visible=True,
            range=[0.0, 1.0],
            tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
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


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: list,
    top_n: int = 12,
) -> go.Figure:
    """Bar chart des SHAP values moyennes (importance globale).

    Args:
        shap_values: Matrice SHAP (n_samples × n_features).
        feature_names: Noms des features.
        top_n: Nombre de features à afficher.

    Returns:
        Figure Plotly.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_abs_shap)[-top_n:]

    fig = go.Figure(go.Bar(
        x=mean_abs_shap[indices],
        y=[feature_names[i] for i in indices],
        orientation="h",
        marker=dict(
            color=mean_abs_shap[indices],
            colorscale=[[0, _SECONDARY], [1, _PRIMARY]],
        ),
        text=[f"{v:.4f}" for v in mean_abs_shap[indices]],
        textposition="outside",
        textfont=dict(color=_TEXT, size=10),
    ))

    layout = _base_layout("SHAP — Importance Globale des Features", height=420)
    layout["xaxis"]["title"] = "Mean |SHAP value|"
    fig.update_layout(**layout)

    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    top_n: int = 10,
) -> go.Figure:
    """Beeswarm plot des SHAP values (direction de l'impact).

    Args:
        shap_values: Matrice SHAP (n_samples × n_features).
        X: Matrice de features (pour la coloration).
        feature_names: Noms des features.
        top_n: Nombre de features à afficher.

    Returns:
        Figure Plotly.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    fig = go.Figure()

    for rank, idx in enumerate(top_indices):
        feat_shap = shap_values[:, idx]
        feat_vals = X[:, idx]

        # Normaliser les valeurs de feature pour la couleur
        fmin, fmax = feat_vals.min(), feat_vals.max()
        if fmax > fmin:
            normalized = (feat_vals - fmin) / (fmax - fmin)
        else:
            normalized = np.zeros_like(feat_vals)

        # Sous-échantillonner pour la performance
        n_sample = min(500, len(feat_shap))
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(feat_shap), n_sample, replace=False)

        # Jitter sur y (increased from 0.12 for better visibility)
        jitter = rng.normal(0, 0.20, n_sample)

        # Use RdBu_r inspired palette for CVD accessibility
        colors = [
            f"rgb({int(59 + 189 * v)}, {int(130 - 50 * abs(v - 0.5))}, {int(246 - 175 * v)})"
            for v in normalized[sample_idx]
        ]

        fig.add_trace(go.Scatter(
            x=feat_shap[sample_idx],
            y=[rank + j for j in jitter],
            mode="markers",
            marker=dict(size=3, color=colors, opacity=0.6),
            showlegend=False,
            hovertemplate=(
                f"<b>{feature_names[idx]}</b><br>"
                "SHAP: %{x:.4f}<br>"
                "<extra></extra>"
            ),
        ))

    layout = _base_layout("SHAP — Beeswarm (Impact Directionnel)", height=450)
    layout["xaxis"]["title"] = "SHAP value"
    layout["yaxis"]["tickvals"] = list(range(len(top_indices)))
    layout["yaxis"]["ticktext"] = [feature_names[i] for i in top_indices]
    fig.update_layout(**layout)

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    n_bins: int = 10,
    min_bin_count: int = 15,
) -> go.Figure:
    """Calibration deviation plot — Observed/Expected ratio par decile.

    Montre le ratio (taux observe) / (PD predite) par bin. Un modele
    parfaitement calibre = ligne horizontale a 1.0. Les ecarts de
    calibration sont amplifies et lisibles meme sur donnees synthetiques.

    Subplot 2 : nombre d'observations par bin (histogramme) pour
    montrer la representativite statistique de chaque point.

    Args:
        y_true: Labels binaires (0/1).
        predictions: Dict {model_name: y_pred_proba}.
        n_bins: Nombre de bins (deciles).
        min_bin_count: Nombre min d'observations par bin.

    Returns:
        Figure Plotly (2 subplots).
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
    )

    for i, (name, y_pred) in enumerate(predictions.items()):
        color = _COLORS[i % len(_COLORS)]
        bin_centers = []
        oe_ratios = []
        ci_lo = []
        ci_hi = []
        bin_counts = []

        # Bins quantiles par modele — equalise ~N/n_bins obs par bin
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.unique(np.percentile(y_pred, quantiles))
        n_actual = len(bin_edges) - 1

        for j in range(n_actual):
            if j < n_actual - 1:
                mask = (y_pred >= bin_edges[j]) & (y_pred < bin_edges[j + 1])
            else:
                mask = y_pred >= bin_edges[j]
            n_obs = int(mask.sum())
            if n_obs < min_bin_count:
                continue
            pred_mean = float(y_pred[mask].mean())
            obs_rate = float(y_true[mask].mean())
            if pred_mean < 1e-6:
                continue
            oe = obs_rate / pred_mean
            bin_centers.append(pred_mean)
            oe_ratios.append(oe)
            bin_counts.append(n_obs)
            # IC Wilson sur obs_rate, puis diviser par pred_mean
            z = 1.96
            denom = 1 + z**2 / n_obs
            centre = (obs_rate + z**2 / (2 * n_obs)) / denom
            half = z * np.sqrt((obs_rate * (1 - obs_rate) + z**2 / (4 * n_obs)) / n_obs) / denom
            ci_lo.append(max(0, centre - half) / pred_mean)
            ci_hi.append((centre + half) / pred_mean)

        if not bin_centers:
            continue

        # Bande de confiance (gris neutre, identique pour tous les modeles)
        fig.add_trace(go.Scatter(
            x=bin_centers + bin_centers[::-1],
            y=ci_hi + ci_lo[::-1],
            fill="toself",
            fillcolor="rgba(148,163,184,0.10)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ), row=1, col=1)

        # Courbe O/E
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=oe_ratios,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2.5),
            marker=dict(size=7, line=dict(color=_BG, width=1)),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "PD predite: %{x:.3f}<br>"
                "O/E ratio: %{y:.2f}<br>"
                "<extra></extra>"
            ),
        ), row=1, col=1)

        # Histogramme observations
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=bin_counts,
            name=name,
            marker=dict(color=color, opacity=0.5),
            showlegend=False,
            hovertemplate=f"<b>{name}</b><br>N=%{{y:,.0f}}<extra></extra>",
        ), row=2, col=1)

    # Ligne de reference O/E = 1.0 (calibration parfaite)
    fig.add_hline(
        y=1.0, line_dash="dash", line_color=_MUTED, line_width=1.5,
        row=1, col=1,
    )
    # Bande acceptable +/-20%
    fig.add_hrect(
        y0=0.8, y1=1.2,
        fillcolor="rgba(52,211,153,0.06)",
        line_width=0, row=1, col=1,
    )

    layout = _base_layout("Calibration — Ratio Observe / Predit par Decile", height=520)
    layout["legend"] = dict(
        x=0.02, y=0.98, xanchor="left", yanchor="top",
        bgcolor="rgba(15,23,42,0.8)",
        bordercolor="rgba(148,163,184,0.15)",
        borderwidth=1,
        font=dict(color=_TEXT, size=11),
    )
    fig.update_layout(**layout)

    # Axe X log — etale les bins basses PD, compacte les hautes
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_xaxes(
        type="log",
        title=dict(text="PD Predite (echelle log)", font=dict(color=_MUTED, size=11)),
        row=2, col=1,
    )
    fig.update_yaxes(
        title=dict(text="O/E Ratio", font=dict(color=_MUTED, size=11)),
        row=1, col=1,
    )
    fig.update_yaxes(
        title=dict(text="N obs", font=dict(color=_MUTED, size=9)),
        row=2, col=1,
    )

    # Annotation bande verte
    fig.add_annotation(
        text="±20%",
        x=1.0, xref="paper", y=1.2, yref="y",
        showarrow=False, font=dict(color=_SUCCESS, size=9),
        xanchor="right",
    )

    return fig


def plot_hhi_gauge(hhi_by_segment: float, hhi_by_loan: float) -> go.Figure:
    """Jauge de concentration HHI (Herfindahl-Hirschman Index).

    Args:
        hhi_by_segment: HHI sur les segments.
        hhi_by_loan: HHI sur les types de prêts.

    Returns:
        Figure Plotly avec deux jauges.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("HHI Segments", "HHI Types de Prêt"),
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
    )

    for col, (value, title) in enumerate([(hhi_by_segment, "Segments"), (hhi_by_loan, "Prêts")], 1):
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(font=dict(color=_TEXT, size=28), valueformat=".4f"),
            gauge=dict(
                axis=dict(range=[0, 1], tickcolor=_MUTED),
                bar=dict(color=_PRIMARY),
                bgcolor=_CARD,
                steps=[
                    dict(range=[0, 0.15], color="rgba(6,214,160,0.2)"),
                    dict(range=[0.15, 0.25], color="rgba(249,115,22,0.2)"),
                    dict(range=[0.25, 1], color="rgba(239,68,68,0.2)"),
                ],
                threshold=dict(
                    line=dict(color=_DANGER, width=2),
                    thickness=0.8,
                    value=0.25,
                ),
            ),
        ), row=1, col=col)

    layout = _base_layout("Indice de Concentration HHI", height=280)
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(color=_TEXT, size=12))

    return fig


def plot_backtesting_auc(monthly_metrics: pl.DataFrame) -> go.Figure:
    """Graphique d'évolution temporelle des métriques (backtesting).

    Args:
        monthly_metrics: DataFrame avec colonnes 'month', 'auc', 'gini', 'ks'.

    Returns:
        Figure Plotly.
    """
    fig = go.Figure()

    metric_styles = {
        "auc": (_PRIMARY, "AUC"),
        "gini": (_ACCENT, "Gini"),
        "ks": (_WARNING, "KS"),
    }

    for metric, (color, label) in metric_styles.items():
        if metric in monthly_metrics.columns:
            fig.add_trace(go.Scatter(
                x=monthly_metrics["month"],
                y=monthly_metrics[metric],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=5),
            ))

    layout = _base_layout("Backtesting — Stabilité Temporelle des Métriques", height=380)
    layout["xaxis"]["title"] = "Mois"
    layout["yaxis"]["title"] = "Valeur"
    layout["yaxis"]["range"] = [0.4, 1.0]
    fig.update_layout(**layout)

    return fig


def plot_shap_force_individual(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list,
    base_value: float = 0.0,
    top_n: int = 10,
) -> go.Figure:
    """Force plot individuel SHAP pour une entreprise (FR49).

    Barres horizontales montrant la contribution de chaque feature
    a la prediction individuelle, triees par impact absolu.

    Args:
        shap_values: SHAP values pour un individu (1D array).
        feature_values: Valeurs des features pour cet individu.
        feature_names: Noms des features.
        base_value: Valeur de base (expected value du modele).
        top_n: Nombre de features a afficher.

    Returns:
        Figure Plotly waterfall-like.
    """
    # Validation des dimensions
    n_features = len(shap_values)
    if len(feature_names) != n_features or len(feature_values) != n_features:
        raise ValueError(
            f"Dimensions incoherentes : shap_values({n_features}), "
            f"feature_names({len(feature_names)}), feature_values({len(feature_values)})"
        )

    # Trier par impact absolu
    indices = np.argsort(np.abs(shap_values))[::-1][:top_n]
    indices = indices[::-1]  # Inverser pour afficher le plus important en haut

    names = [f"{feature_names[i]} = {feature_values[i]:.2f}" for i in indices]
    values = [shap_values[i] for i in indices]
    colors = [_ACCENT if v < 0 else _DANGER for v in values]

    fig = go.Figure(go.Bar(
        y=names,
        x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
        textfont=dict(color=_TEXT, size=10),
    ))

    # Ligne de reference a 0
    fig.add_vline(x=0, line_color=_MUTED, line_width=1)

    pred_value = base_value + sum(shap_values)
    layout = _base_layout(
        f"SHAP Force Plot — Prediction : {pred_value:.4f} (base : {base_value:.4f})",
        height=max(300, top_n * 32),
    )
    layout["xaxis"]["title"] = "SHAP value (contribution)"
    fig.update_layout(**layout)
    return fig


def plot_score_distribution(
    scores: np.ndarray,
    y_true: np.ndarray,
    scorecard_params: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """Histogramme de la distribution des scores scorecard (LR_WoE).

    Affiche la distribution des scores pour les bons et mauvais dossiers
    avec les parametres du scoring.

    Args:
        scores: Array de scores.
        y_true: Array de labels binaires (0/1).
        scorecard_params: Parametres scorecard (pdo, target_score, etc.).

    Returns:
        Figure Plotly.
    """
    fig = go.Figure()

    # Bons dossiers (default=0)
    fig.add_trace(go.Histogram(
        x=scores[y_true == 0],
        name="Non-défaut",
        marker_color=_PRIMARY,
        opacity=0.7,
        nbinsx=40,
    ))

    # Mauvais dossiers (default=1)
    fig.add_trace(go.Histogram(
        x=scores[y_true == 1],
        name="Défaut",
        marker_color=_DANGER,
        opacity=0.7,
        nbinsx=40,
    ))

    title = "Distribution des Scores Scorecard (LR_WoE)"
    if scorecard_params:
        title += (
            f"<br><span style='font-size:11px;color:{_MUTED}'>"
            f"PDO={scorecard_params.get('pdo', 20):.0f} | "
            f"Target={scorecard_params.get('target_score', 600):.0f} pts "
            f"@ odds {scorecard_params.get('target_odds', 50):.0f}:1</span>"
        )

    layout = _base_layout(title, height=400)
    layout["barmode"] = "overlay"
    layout["xaxis"]["title"] = "Score"
    layout["yaxis"]["title"] = "Nombre d'entreprises"
    fig.update_layout(**layout)

    return fig


def plot_trajectories_chart(trajectories_df: pl.DataFrame) -> go.Figure:
    """Graphique en lignes des trajectoires macro prospectives.

    Args:
        trajectories_df: DataFrame avec colonnes horizon + variables macro.

    Returns:
        Figure Plotly multi-line.
    """
    fig = go.Figure()

    # Identifier les colonnes de variables macro (exclure horizon, sector, etc.)
    meta_cols = {"horizon", "sector", "period", "t"}
    var_cols = [c for c in trajectories_df.columns if c not in meta_cols]

    # Axe X : horizon ou row index
    has_horizon = "horizon" in trajectories_df.columns
    x_vals = trajectories_df["horizon"] if has_horizon else list(range(len(trajectories_df)))

    colors = list(_COLORS) + [_WARNING, _INFO, _DANGER]
    for i, col in enumerate(var_cols):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=trajectories_df[col],
            mode="lines+markers",
            name=col.replace("_", " ").title(),
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
        ))

    layout = _base_layout("Trajectoires Macro Prospectives (Ornstein-Uhlenbeck)", height=400)
    layout["xaxis"]["title"] = "Horizon (mois)"
    layout["yaxis"]["title"] = "Valeur"
    layout["showlegend"] = True
    fig.update_layout(**layout)
    return fig


def plot_stage_sankey(
    stages_base: np.ndarray,
    stages_stressed: np.ndarray,
) -> go.Figure:
    """Diagramme Sankey des migrations de stage (Base -> Stress).

    Args:
        stages_base: Array des stages avant stress.
        stages_stressed: Array des stages apres stress.

    Returns:
        Figure Plotly Sankey.
    """
    from ifrs9_cockpit.config import STAGE_COLORS

    labels = [
        "Stage 1 (Base)", "Stage 2 (Base)", "Stage 3 (Base)",
        "Stage 1 (Stress)", "Stage 2 (Stress)", "Stage 3 (Stress)",
    ]

    # Couleurs des noeuds
    node_colors = [
        STAGE_COLORS[1], STAGE_COLORS[2], STAGE_COLORS[3],
        STAGE_COLORS[1], STAGE_COLORS[2], STAGE_COLORS[3],
    ]

    # Calculer les flux
    sources, targets, values = [], [], []
    for s_from in [1, 2, 3]:
        for s_to in [1, 2, 3]:
            count = int(((stages_base == s_from) & (stages_stressed == s_to)).sum())
            if count > 0:
                sources.append(s_from - 1)  # index 0-2 = base
                targets.append(s_to + 2)    # index 3-5 = stress
                values.append(count)

    # Couleurs des liens (transparentes, basees sur la source)
    link_colors = [
        f"rgba{_hex_to_rgba(STAGE_COLORS[s + 1], 0.3)}"
        for s in sources
    ]

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    ))

    layout = _base_layout("Migrations de Stage (Base vs Stress)", height=400)
    fig.update_layout(**layout)
    return fig


def plot_pareto_front(
    pareto_front: List[Dict],
    ecl_breach: float = 0.0,
    design_point_distance: float = 0.0,
) -> go.Figure:
    """Front de Pareto adversarial RST : plausibilite x severite.

    Scatter plot des points Pareto-optimaux generes par l'Evolution
    Differentielle dans l'espace Cholesky. Chaque point represente
    le pire scenario ECL atteignable a un budget Mahalanobis donne.

    References :
        - Traccucci et al. (2019), "A Triptych Approach"
        - Hurlin et al. (2026), arXiv:2601.03983

    Args:
        pareto_front: Liste de dicts avec sigma_budget, ecl, distance_sigma, breach.
        ecl_breach: Seuil ECL de rupture (ligne horizontale).
        design_point_distance: Distance du design point (marqueur special).

    Returns:
        Figure Plotly.
    """
    fig = go.Figure()

    sigmas = [p["sigma_budget"] for p in pareto_front]
    ecls = [p["ecl"] / 1e9 for p in pareto_front]  # En milliards
    breaches = [p["breach"] for p in pareto_front]
    colors = [_DANGER if b else _PRIMARY for b in breaches]

    # Courbe Pareto
    fig.add_trace(go.Scatter(
        x=sigmas,
        y=ecls,
        mode="lines+markers",
        marker=dict(size=12, color=colors, line=dict(color="white", width=1.5)),
        line=dict(color=_MUTED, width=1.5, dash="dot"),
        name="Pareto front",
        hovertemplate=(
            "<b>Budget: %{x:.0f}\u03c3</b><br>"
            "ECL: %{y:,.1f} Md\u20ac<br>"
            "<extra></extra>"
        ),
    ))

    # Seuil de breach
    if ecl_breach > 0:
        fig.add_hline(
            y=ecl_breach / 1e9,
            line_dash="dash",
            line_color=_DANGER,
            annotation_text="Seuil breach",
            annotation_position="top right",
            annotation_font_color=_DANGER,
        )

    # Design point (marqueur diamant)
    if design_point_distance > 0:
        # Interpoler l'ECL au design point depuis le front Pareto
        dp_ecl = None
        for p in pareto_front:
            if p["breach"]:
                dp_ecl = p["ecl"] / 1e9
                break
        if dp_ecl is not None:
            fig.add_trace(go.Scatter(
                x=[design_point_distance],
                y=[dp_ecl],
                mode="markers",
                marker=dict(
                    size=16, symbol="diamond", color=_WARNING,
                    line=dict(color="white", width=2),
                ),
                name=f"Design point ({design_point_distance:.1f}\u03c3)",
                hovertemplate=(
                    "<b>Design Point</b><br>"
                    f"Distance: {design_point_distance:.1f}\u03c3<br>"
                    f"ECL: {dp_ecl:,.1f} Md\u20ac<br>"
                    "<extra></extra>"
                ),
            ))

    layout = _base_layout("Front de Pareto \u2014 RST Adversarial (DE Cholesky)", height=340)
    layout["xaxis"]["title"] = "Budget Mahalanobis (\u03c3)"
    layout["yaxis"]["title"] = "ECL Maximum (Md\u20ac)"
    layout["showlegend"] = True
    layout["legend"] = dict(orientation="h", y=-0.25)
    fig.update_layout(**layout)
    return fig
