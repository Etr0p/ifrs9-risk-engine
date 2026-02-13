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


# Palette cohérente avec le CSS — WCAG AAA
_PRIMARY = DASHBOARD_CONFIG.theme_primary
_SECONDARY = DASHBOARD_CONFIG.theme_secondary
_ACCENT = DASHBOARD_CONFIG.theme_accent
_BG = DASHBOARD_CONFIG.theme_bg_dark
_CARD = DASHBOARD_CONFIG.theme_bg_card
_TEXT = DASHBOARD_CONFIG.theme_text
_MUTED = DASHBOARD_CONFIG.theme_text_muted
_SUCCESS = DASHBOARD_CONFIG.color_success
_WARNING = DASHBOARD_CONFIG.color_warning
_DANGER = DASHBOARD_CONFIG.color_danger
_INFO = DASHBOARD_CONFIG.color_info
_COLORS = [_PRIMARY, _ACCENT, _WARNING, _DANGER, _SECONDARY, _INFO]

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
    colors = [_ACCENT, _WARNING, _DANGER]

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
        "ecl_adverse": (_DANGER, "Adverse (25%)"),
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
        "Faible": _INFO,
        "Moyen": _ACCENT,
        "Fort": _PRIMARY,
        "Suspect": _WARNING,
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
    """Courbe de calibration (reliability diagram) pour les modèles PD.

    Compare la PD prédite à la fréquence de défaut observée par décile.
    Un modèle bien calibré suit la diagonale.

    Utilise des bins à population égale (quantiles) pour éviter les artefacts
    dans les bins à haute PD avec peu d'observations.

    Args:
        y_true: Labels binaires (0/1).
        predictions: Dict {model_name: y_pred_proba}.
        n_bins: Nombre de bins (quantiles à population égale).
        min_bin_count: Nombre min d'observations par bin pour l'afficher.

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
        # Bins a largeur egale dans le range reel des predictions
        # (pas [0,1] qui laisse trop de bins vides en haute PD)
        pred_max = float(np.percentile(y_pred, 99.5))
        bin_edges = np.linspace(0, max(pred_max, 0.05), n_bins + 1)
        bin_centers = []
        observed_rates = []

        for j in range(n_bins):
            if j < n_bins - 1:
                mask = (y_pred >= bin_edges[j]) & (y_pred < bin_edges[j + 1])
            else:
                mask = y_pred >= bin_edges[j]
            if mask.sum() >= min_bin_count:
                bin_centers.append(float(y_pred[mask].mean()))
                observed_rates.append(float(y_true[mask].mean()))

        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=observed_rates,
            mode="lines+markers",
            name=name,
            line=dict(color=_COLORS[i % len(_COLORS)], width=2),
            marker=dict(size=6),
        ))

    # Adapter l'axe au range effectif des donnees
    all_preds = np.concatenate(list(predictions.values()))
    x_max = min(1.0, max(0.3, float(np.percentile(all_preds, 99.5)) * 1.3))
    layout = _base_layout("Courbe de Calibration (Reliability Diagram)", height=420)
    layout["xaxis"]["title"] = "PD Prédite (moyenne par bin)"
    layout["yaxis"]["title"] = "Taux de Défaut Observé"
    layout["xaxis"]["range"] = [0, x_max]
    layout["yaxis"]["range"] = [0, x_max]
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


def plot_pe_nav_by_sector(result_pe: pd.DataFrame) -> go.Figure:
    """Barres NAV et Expected Loss PE par secteur.

    Args:
        result_pe: DataFrame resultat PECalculator.

    Returns:
        Figure Plotly grouped bar.
    """
    cols = ["nav", "expected_loss_pe", "capital_invested"]
    agg = {c: "sum" for c in cols if c in result_pe.columns}
    seg = result_pe.groupby("sector").agg(**{c: (c, "sum") for c in agg}).reset_index()
    seg = seg.sort_values("nav", ascending=True)

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


def plot_pe_risk_categories(result_pe: pd.DataFrame) -> go.Figure:
    """Pie chart des categories de risque PE.

    Args:
        result_pe: DataFrame resultat PECalculator.

    Returns:
        Figure Plotly pie.
    """
    cats = result_pe["risk_category"].value_counts().reindex(
        ["Performing", "Watchlist", "Distressed"], fill_value=0,
    )
    colors = [_ACCENT, _WARNING, _DANGER]

    fig = go.Figure(go.Pie(
        labels=cats.index,
        values=cats.values,
        marker=dict(colors=colors),
        textinfo="label+percent+value",
        textfont=dict(color=_TEXT, size=11),
        hole=0.45,
    ))
    layout = _base_layout("Classification PE (IPEV)", height=380)
    layout["showlegend"] = True
    fig.update_layout(**layout)
    return fig


def plot_pe_moic_drawdown(result_pe: pd.DataFrame) -> go.Figure:
    """MOIC moyen et drawdown moyen par secteur (double axe).

    Args:
        result_pe: DataFrame resultat PECalculator.

    Returns:
        Figure Plotly.
    """
    seg = result_pe.groupby("sector").agg(
        moic_mean=("moic", "mean"),
        drawdown_mean=("nav_drawdown", "mean"),
    ).reset_index().sort_values("moic_mean", ascending=True)

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


def plot_asymmetry_heatmap(asym_df: pd.DataFrame) -> go.Figure:
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
            z.append(asym_df[m].values.tolist())
        else:
            z.append([0.0] * len(asym_df))

    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=asym_df["sector"].tolist(),
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


def plot_raroc_comparison(raroc_df: pd.DataFrame) -> go.Figure:
    """Barres groupees RAROC Credit vs PE par secteur.

    Args:
        raroc_df: DataFrame du PortfolioComparator.compute_raroc_eva().

    Returns:
        Figure Plotly grouped bar.
    """
    # Filtrer les totaux
    df = raroc_df[~raroc_df["sector"].str.startswith("TOTAL")].copy()

    credit = df[df["canal"] == "Credit"].sort_values("sector")
    pe = df[df["canal"] == "PE"].sort_values("sector")

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


def plot_crr3_sensitivity(crr3_df: pd.DataFrame) -> go.Figure:
    """Barres CET1 ratio par scenario RW PE (CRR3).

    Args:
        crr3_df: DataFrame du PortfolioComparator.compute_crr3_sensitivity().

    Returns:
        Figure Plotly.
    """
    fig = go.Figure()

    colors = []
    for _, row in crr3_df.iterrows():
        colors.append(_ACCENT if row.get("feasible", True) else _DANGER)

    fig.add_trace(go.Bar(
        x=crr3_df["rw_pe"].astype(str) + "%",
        y=crr3_df["cet1_ratio"],
        marker_color=colors,
        text=crr3_df["cet1_ratio"].apply(lambda v: f"{v:.2%}"),
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


def plot_risk_appetite_matrix(ra_df: pd.DataFrame) -> go.Figure:
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
        pivot = ra_df.pivot_table(
            index="sector", columns="canal", values="signal",
            aggfunc="first",
        ).fillna("vert")
        z = pivot.map(lambda v: signal_map.get(v, 0)).values
        text = pivot.map(lambda v: signal_labels.get(v, v)).values

        fig = go.Figure(go.Heatmap(
            z=z,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
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
        z = [[signal_map.get(str(row.get("signal", "vert")), 0) for _, row in ra_df.iterrows()]]
        text = [[signal_labels.get(str(row.get("signal", "vert")), "?") for _, row in ra_df.iterrows()]]
        fig = go.Figure(go.Heatmap(
            z=z, text=text, texttemplate="%{text}",
            textfont=dict(size=14, color=_TEXT),
            colorscale=[[0, _SUCCESS], [0.5, _WARNING], [1, _DANGER]],
            showscale=False, zmin=0, zmax=2,
        ))

    layout = _base_layout("Matrice Risk Appetite (Feux Tricolores)", height=350)
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


# ──────────────────────────────────────────────
# NOUVEAUX CHARTS — UX Redesign Phase 2
# ──────────────────────────────────────────────


def plot_trajectories_chart(trajectories_df: pd.DataFrame) -> go.Figure:
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

    # Axe X : horizon ou index
    x_col = "horizon" if "horizon" in trajectories_df.columns else trajectories_df.index

    colors = list(_COLORS) + [_WARNING, _INFO, _DANGER]
    for i, col in enumerate(var_cols):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=trajectories_df[x_col] if isinstance(x_col, str) else x_col,
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


def plot_pe_risk_stacked_bar(result_pe: pd.DataFrame) -> go.Figure:
    """Barre horizontale empilee pour categories de risque PE.

    Remplace le pie chart pour meilleure lisibilite.

    Args:
        result_pe: DataFrame PE avec colonne risk_category.

    Returns:
        Figure Plotly horizontal stacked bar.
    """
    from ifrs9_cockpit.config import PE_CATEGORY_COLORS

    cats = result_pe["risk_category"].value_counts()
    total = cats.sum()

    fig = go.Figure()
    cat_order = ["Performing", "Watchlist", "Distressed"]
    patterns = ["", "/", "x"]  # CVD-safe patterns

    for i, cat in enumerate(cat_order):
        count = cats.get(cat, 0)
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
