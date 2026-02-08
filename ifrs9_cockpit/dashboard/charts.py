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

# Mapping noms techniques → labels lisibles (français)
_FEATURE_LABELS: Dict[str, str] = {
    "credit_score": "Score Credit",
    "nb_past_due_30d": "Retards 30j",
    "income": "Revenu",
    "age": "Age",
    "months_since_last_delinquency": "Delai Dern. Incident",
    "employment_duration": "Anciennete Emploi",
    "debt_ratio": "Ratio Endettement",
    "loan_amount": "Montant Pret",
    "utilization_rate": "Taux Utilisation",
    "nb_credit_lines": "Nb Lignes Credit",
}


def _prettify_feature(name: str) -> str:
    """Convertit un nom technique de feature en label lisible.

    Gère les suffixes _woe en les supprimant avant le lookup.

    Args:
        name: Nom technique (ex: 'credit_score_woe').

    Returns:
        Label lisible (ex: 'Score Credit').
    """
    base = name.replace("_woe", "")
    return _FEATURE_LABELS.get(base, name)


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

    labels = df["feature"].apply(_prettify_feature)

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=labels,
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
    """Barres ECL par segment avec décomposition multi-scénarios.

    Args:
        result_df: DataFrame résultat du ECLCalculator.

    Returns:
        Figure Plotly.
    """
    # Détection dynamique des colonnes de scénarios
    scenario_cols = [c for c in result_df.columns if c.startswith("ecl_") and c != "ecl_weighted"]
    agg_dict = {c: (c, "sum") for c in scenario_cols}
    agg_dict["ecl_weighted"] = ("ecl_weighted", "sum")

    segments = result_df.groupby("segment").agg(**agg_dict).reset_index()
    segments = segments.sort_values("ecl_weighted", ascending=True)

    fig = go.Figure()

    scenario_colors = {
        "ecl_base": (_PRIMARY, "Base (50%)"),
        "ecl_adverse": ("#EF4444", "Adverse (25%)"),
        "ecl_favorable": (_ACCENT, "Favorable (25%)"),
    }

    for col in scenario_cols:
        color, label = scenario_colors.get(col, (_SECONDARY, col))
        fig.add_trace(go.Bar(
            y=segments["segment"],
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

    labels = df["feature"].apply(_prettify_feature)

    fig = go.Figure(go.Bar(
        x=df["iv"],
        y=labels,
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

        # Jitter sur y
        jitter = rng.normal(0, 0.12, n_sample)

        colors = [
            f"rgb({int(255 * v)}, {int(80 * (1 - v))}, {int(255 * (1 - v))})"
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
) -> go.Figure:
    """Courbe de calibration (reliability diagram) pour les modèles PD.

    Compare la PD prédite à la fréquence de défaut observée par décile.
    Un modèle bien calibré suit la diagonale.

    Args:
        y_true: Labels binaires (0/1).
        predictions: Dict {model_name: y_pred_proba}.
        n_bins: Nombre de bins.

    Returns:
        Figure Plotly.
    """
    fig = go.Figure()

    # Diagonale de calibration parfaite
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Calibration parfaite",
        line=dict(color=_MUTED, width=1, dash="dash"),
        showlegend=True,
    ))

    for i, (name, y_pred) in enumerate(predictions.items()):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        observed_rates = []

        for j in range(n_bins):
            mask = (y_pred >= bin_edges[j]) & (y_pred < bin_edges[j + 1])
            if mask.sum() > 0:
                bin_centers.append(y_pred[mask].mean())
                observed_rates.append(y_true[mask].mean())

        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=observed_rates,
            mode="lines+markers",
            name=name,
            line=dict(color=_COLORS[i % len(_COLORS)], width=2),
            marker=dict(size=6),
        ))

    layout = _base_layout("Courbe de Calibration (Reliability Diagram)", height=420)
    layout["xaxis"]["title"] = "PD Prédite (moyenne par bin)"
    layout["yaxis"]["title"] = "Taux de Défaut Observé"
    layout["xaxis"]["range"] = [0, max(0.3, 1)]
    layout["yaxis"]["range"] = [0, max(0.3, 1)]
    fig.update_layout(**layout)

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
                    line=dict(color="#EF4444", width=2),
                    thickness=0.8,
                    value=0.25,
                ),
            ),
        ), row=1, col=col)

    layout = _base_layout("Indice de Concentration HHI", height=280)
    fig.update_layout(**layout)
    fig.update_annotations(font=dict(color=_TEXT, size=12))

    return fig


def plot_backtesting_auc(monthly_metrics: pd.DataFrame) -> go.Figure:
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
        "ks": ("#F59E0B", "KS"),
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
