"""Module Analyste CRO Local — Intelligence d'analyse embarquée IFRS 9.

Moteur d'analyse statistique autonome qui corrèle automatiquement
l'ensemble des métriques du portefeuille (ECL, staging, macro, SHAP,
backtesting, HHI, performance modèles) pour produire un rapport CRO
structuré de niveau Direction des Risques.

Contrairement au VirtualCRO (5 règles de seuil), ce module :
    - Décompose mathématiquement la variation d'ECL (attribution)
    - Calcule les élasticités macro-crédit par segment
    - Identifie automatiquement les corrélations et causalités
    - Analyse les flux de migration et la vulnérabilité par segment
    - Évalue la gouvernance des modèles (performance, drift, stabilité)
    - Produit un rapport narratif corrélé avec recommandations priorisées

Aucune dépendance externe (pas d'API). Tout le raisonnement est
embarqué dans le moteur d'analyse statistique.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ifrs9_cockpit.config import SEGMENTS, SCENARIOS


class LocalCROAnalyst:
    """Analyste CRO embarqué — analyse corrélée sans dépendance externe.

    Remplace le système rules-based à 5 règles par un moteur d'analyse
    statistique complet qui :
        - Décompose mathématiquement la variation d'ECL
        - Calcule les élasticités macro-crédit par segment
        - Identifie automatiquement les corrélations et les causalités
        - Produit un rapport narratif de niveau Direction des Risques

    Example:
        >>> analyst = LocalCROAnalyst()
        >>> report = analyst.generate_full_report(
        ...     result_stressed, result_base, model_comparison,
        ...     macro_params, psi_value, selected_model,
        ... )
        >>> print(report)
    """

    # Baselines macro de référence
    _MACRO_BASELINES: Dict[str, Tuple[str, float]] = {
        "unemployment_rate": ("Taux de chômage", 7.5),
        "gdp_growth": ("Croissance PIB", 1.2),
        "interest_rate": ("Taux directeur BCE", 3.5),
        "hpi_growth": ("Prix immobiliers (YoY)", 2.0),
        "inflation_rate": ("Inflation IPC", 2.5),
    }

    # Variables adverses quand elles augmentent
    _ADVERSE_UP = {"unemployment_rate", "interest_rate", "inflation_rate"}
    # Variables adverses quand elles diminuent
    _ADVERSE_DOWN = {"gdp_growth", "hpi_growth"}

    # Labels avec article pour la prose française
    _MACRO_LABELS_FR: Dict[str, str] = {
        "unemployment_rate": "le taux de chomage",
        "gdp_growth": "la croissance du PIB",
        "interest_rate": "le taux directeur BCE",
        "hpi_growth": "les prix immobiliers",
        "inflation_rate": "l'inflation",
    }

    def generate_full_report(
        self,
        result_stressed: pd.DataFrame,
        result_base: pd.DataFrame,
        model_comparison: pd.DataFrame,
        macro_params: Dict[str, float],
        psi_value: float,
        selected_model: str = "LR_WoE",
        shap_top_features: Optional[List[Tuple[str, float]]] = None,
        hhi_segment: Optional[float] = None,
        hhi_loan: Optional[float] = None,
        backtesting_df: Optional[pd.DataFrame] = None,
        waterfall_df: Optional[pd.DataFrame] = None,
        transition_matrix: Optional[pd.DataFrame] = None,
    ) -> str:
        """Génère le rapport CRO complet avec analyse corrélée.

        Agrège toutes les métriques, calcule les corrélations,
        et produit un rapport narratif structuré en 8 sections.

        Args:
            result_stressed: DataFrame ECL stressé (avec sliders).
            result_base: DataFrame ECL baseline (sans stress).
            model_comparison: Table de comparaison des modèles PD.
            macro_params: Paramètres macro courants (5 variables).
            psi_value: PSI du modèle PD sélectionné.
            selected_model: Nom du modèle PD sélectionné.
            shap_top_features: Top features SHAP [(name, mean_abs_shap)].
            hhi_segment: HHI par segment.
            hhi_loan: HHI par type de prêt.
            backtesting_df: Métriques de backtesting walk-forward.
            waterfall_df: Décomposition waterfall ECL.
            transition_matrix: Matrice de transition des stages.

        Returns:
            Rapport complet en Markdown structuré.
        """
        # Phase 1 : Calcul de tous les indicateurs analytiques
        analytics = self._compute_core_analytics(
            result_stressed, result_base, macro_params,
        )

        # Phase 2 : Génération du rapport section par section
        sections = [
            self._section_header(),
            self._section_executive_summary(
                analytics, result_stressed, macro_params,
                psi_value, hhi_segment,
            ),
            self._section_ecl_attribution(
                analytics, waterfall_df,
            ),
            self._section_macro_sensitivity(
                analytics, macro_params,
            ),
            self._section_staging_analysis(
                analytics, result_stressed, result_base,
                transition_matrix,
            ),
            self._section_model_governance(
                model_comparison, psi_value, selected_model,
                backtesting_df, shap_top_features,
            ),
            self._section_concentration(
                hhi_segment, hhi_loan, result_stressed,
            ),
            self._section_forward_looking(
                analytics, macro_params, result_stressed,
            ),
            self._section_recommendations(
                analytics, macro_params, psi_value,
                hhi_segment, result_stressed,
            ),
        ]

        return "\n\n".join(sections)

    # ──────────────────────────────────────────────
    # CORE ANALYTICS ENGINE
    # ──────────────────────────────────────────────

    def _compute_core_analytics(
        self,
        result_stressed: pd.DataFrame,
        result_base: pd.DataFrame,
        macro_params: Dict[str, float],
    ) -> Dict[str, Any]:
        """Calcule tous les indicateurs analytiques centraux.

        Produit un dictionnaire complet avec :
            - Métriques ECL globales et variation
            - Analytics par segment (ECL, PD, staging, macro impacts)
            - Ranking des segments par impact
            - Élasticités macro-crédit

        Args:
            result_stressed: DataFrame ECL stressé.
            result_base: DataFrame ECL baseline.
            macro_params: Paramètres macro courants.

        Returns:
            Dictionnaire d'analytics pour toutes les sections.
        """
        ecl_stressed = result_stressed["ecl_weighted"].sum()
        ecl_base = result_base["ecl_weighted"].sum()
        ead_stressed = result_stressed["ead"].sum()

        ecl_variation = (
            (ecl_stressed - ecl_base) / ecl_base if ecl_base > 0 else 0
        )

        # Analyse par segment
        segment_analytics: Dict[str, Dict[str, Any]] = {}

        for seg in SEGMENTS:
            mask_s = result_stressed["segment"] == seg.name
            mask_b = result_base["segment"] == seg.name

            seg_ecl_s = result_stressed.loc[mask_s, "ecl_weighted"].sum()
            seg_ecl_b = result_base.loc[mask_b, "ecl_weighted"].sum()
            seg_ead_s = result_stressed.loc[mask_s, "ead"].sum()
            seg_pd_s = result_stressed.loc[mask_s, "pd_12m"].mean()
            seg_pd_b = result_base.loc[mask_b, "pd_12m"].mean()

            seg_ecl_var = (
                (seg_ecl_s - seg_ecl_b) / seg_ecl_b
                if seg_ecl_b > 0 else 0
            )
            ecl_contribution = (
                seg_ecl_s / ecl_stressed if ecl_stressed > 0 else 0
            )
            coverage = seg_ecl_s / seg_ead_s if seg_ead_s > 0 else 0

            # Distribution des stages (stressé et baseline)
            stages_s = result_stressed.loc[mask_s, "stage"]
            stages_b = result_base.loc[mask_b, "stage"]

            # Élasticités macro-crédit par canal de transmission
            sensitivity_map = {
                "unemployment_rate": seg.unemployment_sensitivity_credit,
                "gdp_growth": seg.gdp_sensitivity_credit,
                "interest_rate": seg.interest_rate_sensitivity_credit,
                "hpi_growth": seg.hpi_sensitivity_credit,
                "inflation_rate": seg.inflation_sensitivity_credit,
            }

            macro_impacts: Dict[str, Dict[str, Any]] = {}
            for var, (_, baseline) in self._MACRO_BASELINES.items():
                current = macro_params.get(var, baseline)
                delta = current - baseline
                sensitivity = sensitivity_map[var]

                # Choc effectif (même logique que ecl_calculator)
                if var in self._ADVERSE_DOWN:
                    shock = max(0, baseline - current) / 100
                else:
                    shock = max(0, current - baseline) / 100

                estimated_pd_mult = 1 + shock * sensitivity * 10

                is_adverse = (
                    (var in self._ADVERSE_UP and delta > 0)
                    or (var in self._ADVERSE_DOWN and delta < 0)
                )

                macro_impacts[var] = {
                    "delta": delta,
                    "sensitivity": sensitivity,
                    "estimated_pd_multiplier": estimated_pd_mult,
                    "direction": (
                        "adverse" if is_adverse
                        else "favorable" if delta != 0
                        else "neutre"
                    ),
                }

            # Driver macro dominant (celui avec le plus grand impact)
            adverse_impacts = {
                k: v["estimated_pd_multiplier"] - 1
                for k, v in macro_impacts.items()
                if v["estimated_pd_multiplier"] > 1
            }
            dominant_driver = (
                max(adverse_impacts, key=adverse_impacts.get)
                if adverse_impacts else None
            )

            segment_analytics[seg.name] = {
                "ecl_stressed": seg_ecl_s,
                "ecl_base": seg_ecl_b,
                "ecl_delta": seg_ecl_s - seg_ecl_b,
                "ecl_variation": seg_ecl_var,
                "ecl_contribution": ecl_contribution,
                "ead": seg_ead_s,
                "coverage": coverage,
                "pd_stressed": seg_pd_s,
                "pd_base": seg_pd_b,
                "pd_variation": (
                    (seg_pd_s - seg_pd_b) / seg_pd_b
                    if seg_pd_b > 0 else 0
                ),
                "stage_dist_stressed": {
                    s: float((stages_s == s).mean()) for s in [1, 2, 3]
                },
                "stage_dist_base": {
                    s: float((stages_b == s).mean()) for s in [1, 2, 3]
                },
                "macro_impacts": macro_impacts,
                "dominant_driver": dominant_driver,
                "count": int(mask_s.sum()),
                "base_default_rate": seg.base_default_rate,
            }

        # Classement des segments par impact ECL
        ranked_segments = sorted(
            segment_analytics.keys(),
            key=lambda s: abs(segment_analytics[s]["ecl_variation"]),
            reverse=True,
        )

        top_ecl_segment = max(
            segment_analytics.keys(),
            key=lambda s: segment_analytics[s]["ecl_contribution"],
        )

        return {
            "ecl_stressed": ecl_stressed,
            "ecl_base": ecl_base,
            "ecl_variation": ecl_variation,
            "ecl_delta": ecl_stressed - ecl_base,
            "ead_stressed": ead_stressed,
            "coverage": (
                ecl_stressed / ead_stressed if ead_stressed > 0 else 0
            ),
            "pd_mean_stressed": float(result_stressed["pd_12m"].mean()),
            "pd_mean_base": float(result_base["pd_12m"].mean()),
            "n_clients": len(result_stressed),
            "segment_analytics": segment_analytics,
            "ranked_segments": ranked_segments,
            "top_ecl_segment": top_ecl_segment,
        }

    # ──────────────────────────────────────────────
    # REPORT SECTIONS
    # ──────────────────────────────────────────────

    def _section_header(self) -> str:
        """En-tête du rapport."""
        date = datetime.now().strftime("%d/%m/%Y %H:%M")
        return (
            "# RAPPORT D'ANALYSE CRO — IFRS 9 RISK COCKPIT\n"
            f"**Date :** {date}  \n"
            "**Type :** Analyse corrélée automatique  \n"
            "**Classification :** Confidentiel — Direction des Risques"
        )

    def _section_executive_summary(
        self,
        analytics: Dict[str, Any],
        result_stressed: pd.DataFrame,
        macro_params: Dict[str, float],
        psi_value: float,
        hhi_segment: Optional[float],
    ) -> str:
        """Section 1 : Synthèse exécutive avec évaluation du risque global."""
        lines = ["## 1. Synthèse Exécutive"]

        ecl_var = analytics["ecl_variation"]
        ecl_s = analytics["ecl_stressed"]
        ecl_b = analytics["ecl_base"]
        coverage = analytics["coverage"]
        top_seg = analytics["top_ecl_segment"]
        seg_data = analytics["segment_analytics"][top_seg]

        # Scoring du risque global (somme pondérée d'indicateurs)
        risk_score = 0
        risk_flags: List[str] = []

        if abs(ecl_var) > 0.15:
            risk_score += 3
            risk_flags.append(f"variation ECL {ecl_var:+.1%}")
        elif abs(ecl_var) > 0.05:
            risk_score += 1

        s2_pct = float((result_stressed["stage"] == 2).mean())
        if s2_pct > 0.20:
            risk_score += 2
            risk_flags.append(f"Stage 2 a {s2_pct:.1%}")
        elif s2_pct > 0.15:
            risk_score += 1

        if psi_value > 0.25:
            risk_score += 2
            risk_flags.append(f"PSI a {psi_value:.3f}")
        elif psi_value > 0.10:
            risk_score += 1

        n_adverse_macro = sum(
            1 for key in self._MACRO_BASELINES
            if self._is_adverse(key, macro_params)
        )
        if n_adverse_macro >= 3:
            risk_score += 2
            risk_flags.append(f"{n_adverse_macro} variables macro adverses")
        elif n_adverse_macro >= 1:
            risk_score += 1

        if risk_score >= 5:
            risk_level, risk_color = "ELEVE", "rouge"
        elif risk_score >= 2:
            risk_level, risk_color = "MODERE", "orange"
        else:
            risk_level, risk_color = "MAITRISE", "vert"

        lines.append(
            f"\n**Niveau de risque global : {risk_level}** "
            f"(indicateur {risk_color}, score {risk_score}/10)"
        )

        # Paragraphe principal
        if abs(ecl_var) > 0.01:
            direction = "hausse" if ecl_var > 0 else "baisse"
            lines.append(
                f"\nLe portefeuille affiche un ECL stresse de "
                f"**{ecl_s:,.0f} EUR** contre {ecl_b:,.0f} EUR en baseline, "
                f"soit une {direction} de **{ecl_var:+.1%}** "
                f"({analytics['ecl_delta']:+,.0f} EUR). "
                f"Le coverage ratio s'etablit a **{coverage:.2%}** sur un "
                f"EAD de {analytics['ead_stressed']:,.0f} EUR pour "
                f"{analytics['n_clients']:,} clients. "
                f"La PD moyenne passe de {analytics['pd_mean_base']:.2%} a "
                f"{analytics['pd_mean_stressed']:.2%}."
            )
        else:
            lines.append(
                f"\nL'ECL reste stable a **{ecl_s:,.0f} EUR** "
                f"(coverage : {coverage:.2%}, "
                f"EAD : {analytics['ead_stressed']:,.0f} EUR, "
                f"{analytics['n_clients']:,} clients)."
            )

        # Points d'attention
        concerns: List[str] = []

        ranked = analytics["ranked_segments"]
        if ranked:
            top = ranked[0]
            top_d = analytics["segment_analytics"][top]
            if abs(top_d["ecl_variation"]) > 0.05:
                concerns.append(
                    f"Le segment **{top}** concentre "
                    f"{top_d['ecl_contribution']:.0%} de l'ECL total "
                    f"avec une variation de {top_d['ecl_variation']:+.1%} "
                    f"(coverage : {top_d['coverage']:.2%})"
                )

        macro_concerns = []
        for key, (label, baseline) in self._MACRO_BASELINES.items():
            current = macro_params.get(key, baseline)
            delta = current - baseline
            if self._is_adverse(key, macro_params) and abs(delta) > 0.5:
                macro_concerns.append(
                    f"{label} a {current:.1f}% ({delta:+.1f}pp)"
                )
        if macro_concerns:
            concerns.append(
                f"Stress macroeconomique actif : {', '.join(macro_concerns)}"
            )

        if psi_value > 0.10:
            action = (
                "recalibration urgente requise"
                if psi_value > 0.25
                else "monitoring renforce"
            )
            concerns.append(f"PSI du modele a {psi_value:.3f} — {action}")

        if concerns:
            lines.append("\n**Points d'attention :**")
            for c in concerns:
                lines.append(f"- {c}")
        else:
            lines.append(
                "\nAucun point d'attention majeur. Le portefeuille reste "
                "dans les limites de l'appetit au risque."
            )

        return "\n".join(lines)

    def _section_ecl_attribution(
        self,
        analytics: Dict[str, Any],
        waterfall_df: Optional[pd.DataFrame],
    ) -> str:
        """Section 2 : Décomposition ECL et attribution par driver."""
        lines = ["## 2. Analyse ECL et Decomposition"]

        ecl_var = analytics["ecl_variation"]
        ecl_delta = analytics["ecl_delta"]

        if abs(ecl_var) < 0.001:
            lines.append(
                "\nL'ECL est stable entre le baseline et le scenario "
                "stresse. Les parametres macro n'ont pas d'impact "
                "significatif a ces niveaux."
            )
            return "\n".join(lines)

        direction = "hausse" if ecl_delta > 0 else "baisse"

        lines.append(
            f"\nVariation ECL totale : **{ecl_delta:+,.0f} EUR** "
            f"({ecl_var:+.1%}) en {direction}."
        )

        # Décomposition waterfall
        if waterfall_df is not None:
            lines.append("\n**Decomposition du mouvement :**")
            migration_impact = 0.0
            param_impact = 0.0

            for _, row in waterfall_df.iterrows():
                comp = row["component"]
                amount = row["amount"]
                if comp in (
                    "ECL Ouverture", "ECL Cloture", "Variation nette"
                ):
                    continue
                pct = (
                    amount / abs(ecl_delta) * 100
                    if ecl_delta != 0 else 0
                )
                lines.append(
                    f"- **{comp}** : {amount:+,.0f} EUR "
                    f"({pct:+.0f}% du mouvement)"
                )

                if "migration" in comp.lower():
                    migration_impact += amount
                elif "param" in comp.lower():
                    param_impact += amount

            # Interprétation
            if abs(migration_impact) > abs(param_impact) and migration_impact != 0:
                lines.append(
                    f"\nLe mouvement est principalement porte par les "
                    f"**migrations de stage** ({migration_impact:+,.0f} EUR), "
                    f"indiquant une deterioration de la qualite de credit "
                    f"sous stress. Les downgrades (Stage 1 vers 2 ou 2 "
                    f"vers 3) declenchent le passage au calcul ECL lifetime, "
                    f"ce qui amplifie mecaniquement la provision."
                )
            elif abs(param_impact) > 0:
                lines.append(
                    f"\nLe mouvement est principalement porte par les "
                    f"**revisions de parametres** (PD/LGD/EAD : "
                    f"{param_impact:+,.0f} EUR), refletant l'impact direct "
                    f"du choc macro sur les estimations de risque a stage "
                    f"constant."
                )

        # Analyse par segment
        lines.append("\n**Analyse par segment :**")
        seg_analytics = analytics["segment_analytics"]

        for seg_name in analytics["ranked_segments"]:
            seg = seg_analytics[seg_name]
            if abs(seg["ecl_variation"]) < 0.001:
                continue

            lines.append(
                f"\n- **{seg_name}** ({seg['count']} clients) : "
                f"ECL {seg['ecl_stressed']:,.0f} EUR "
                f"({seg['ecl_variation']:+.1%}), "
                f"contribution = {seg['ecl_contribution']:.0%} du total. "
                f"PD moyenne : {seg['pd_base']:.2%} -> "
                f"{seg['pd_stressed']:.2%} "
                f"({seg['pd_variation']:+.1%}). "
                f"Coverage : {seg['coverage']:.2%}."
            )

            # Causalité : identifier pourquoi ce segment bouge
            if seg["dominant_driver"]:
                driver = seg["dominant_driver"]
                driver_label = self._MACRO_BASELINES[driver][0]
                driver_data = seg["macro_impacts"][driver]
                lines.append(
                    f"  Driver principal : **{driver_label}** "
                    f"(sensibilite {driver_data['sensitivity']}x, "
                    f"ecart {driver_data['delta']:+.1f}pp)."
                )

        return "\n".join(lines)

    def _section_macro_sensitivity(
        self,
        analytics: Dict[str, Any],
        macro_params: Dict[str, float],
    ) -> str:
        """Section 3 : Analyse de sensibilité macroéconomique."""
        lines = ["## 3. Sensibilite Macroeconomique"]

        # Tableau des paramètres
        lines.append("\n**Parametres macroeconomiques actuels :**\n")
        lines.append(
            "| Variable | Baseline | Courant | Ecart | Direction |"
        )
        lines.append(
            "|----------|----------|---------|-------|-----------|"
        )

        active_stresses: List[Tuple[str, str, float]] = []
        for key, (label, baseline) in self._MACRO_BASELINES.items():
            current = macro_params.get(key, baseline)
            delta = current - baseline
            is_adv = self._is_adverse(key, macro_params)
            direction = (
                "Defavorable" if is_adv
                else "Favorable" if delta != 0
                else "Neutre"
            )
            lines.append(
                f"| {label} | {baseline:.1f}% | {current:.1f}% | "
                f"{delta:+.1f}pp | {direction} |"
            )
            if is_adv:
                active_stresses.append((key, label, delta))

        # Canaux de transmission
        if active_stresses:
            lines.append("\n**Canaux de transmission macro -> credit :**\n")

            for key, label, delta in active_stresses:
                # Segment le plus sensible à cette variable
                seg_sens = []
                for seg in SEGMENTS:
                    sens_map = {
                        "unemployment_rate": seg.unemployment_sensitivity_credit,
                        "gdp_growth": seg.gdp_sensitivity_credit,
                        "interest_rate": seg.interest_rate_sensitivity_credit,
                        "hpi_growth": seg.hpi_sensitivity_credit,
                        "inflation_rate": seg.inflation_sensitivity_credit,
                    }
                    seg_sens.append((seg.name, sens_map[key]))

                seg_sens.sort(key=lambda x: x[1], reverse=True)
                top_seg, top_sensitivity = seg_sens[0]
                seg_data = analytics["segment_analytics"][top_seg]

                lines.append(
                    f"- **{label}** ({delta:+.1f}pp) -> Impact maximal "
                    f"sur **{top_seg}** (sensibilite {top_sensitivity}x). "
                    f"Ce segment affiche une variation d'ECL de "
                    f"{seg_data['ecl_variation']:+.1%} et concentre "
                    f"{seg_data['ecl_contribution']:.0%} de l'ECL total."
                )

                explanation = self._macro_explanation(key, top_seg)
                if explanation:
                    lines.append(f"  *{explanation}*")
        else:
            lines.append(
                "\nAucun stress macroeconomique defavorable actif. "
                "Les parametres sont au niveau du baseline ou en zone "
                "favorable."
            )

        # Effet combiné
        if len(active_stresses) >= 2:
            stress_desc = " et ".join(
                f"{label.lower()} ({delta:+.1f}pp)"
                for _, label, delta in active_stresses
            )
            lines.append(
                f"\n**Effet combine :** La conjonction de {stress_desc} "
                f"cree un effet multiplicatif sur les segments les plus "
                f"vulnerables. L'ECL global varie de "
                f"{analytics['ecl_variation']:+.1%}, soit "
                f"{analytics['ecl_delta']:+,.0f} EUR. Les segments a "
                f"double exposition (sensibilite elevee a plusieurs "
                f"variables) sont particulierement touches."
            )

            # Identifier les segments à double/triple exposition
            multi_exposed = []
            for seg in SEGMENTS:
                n_exposures = sum(
                    1 for key, _, _ in active_stresses
                    if analytics["segment_analytics"][seg.name][
                        "macro_impacts"
                    ][key]["sensitivity"] > 1.5
                )
                if n_exposures >= 2:
                    multi_exposed.append((seg.name, n_exposures))

            if multi_exposed:
                segs_str = ", ".join(
                    f"{name} ({n}x)" for name, n in multi_exposed
                )
                lines.append(
                    f"  Segments a exposition multiple : {segs_str}."
                )

        return "\n".join(lines)

    def _section_staging_analysis(
        self,
        analytics: Dict[str, Any],
        result_stressed: pd.DataFrame,
        result_base: pd.DataFrame,
        transition_matrix: Optional[pd.DataFrame],
    ) -> str:
        """Section 4 : Qualité de crédit et staging."""
        lines = ["## 4. Qualite de Credit et Staging"]

        # Distribution
        lines.append("\n**Distribution des stages :**\n")
        lines.append(
            "| Stage | Baseline | Stresse | Migration nette |"
        )
        lines.append(
            "|-------|----------|---------|-----------------|"
        )

        for s in [1, 2, 3]:
            base_n = int((result_base["stage"] == s).sum())
            stress_n = int((result_stressed["stage"] == s).sum())
            base_pct = base_n / len(result_base)
            stress_pct = stress_n / len(result_stressed)
            delta_n = stress_n - base_n
            lines.append(
                f"| Stage {s} | {base_n:,} ({base_pct:.1%}) | "
                f"{stress_n:,} ({stress_pct:.1%}) | {delta_n:+,} |"
            )

        # Matrice de transition
        if transition_matrix is not None:
            lines.append(
                "\n**Matrice de transition (Baseline -> Stresse) :**\n"
            )
            labels = ["Stage 1", "Stage 2", "Stage 3"]
            lines.append(
                "| De / Vers | Stage 1 | Stage 2 | Stage 3 |"
            )
            lines.append(
                "|-----------|---------|---------|---------|"
            )
            for i, label in enumerate(labels):
                row_vals = " | ".join(
                    f"{transition_matrix.values[i][j]:.1%}"
                    for j in range(3)
                )
                lines.append(f"| {label} | {row_vals} |")

            # Interprétation des flux
            dg_12 = transition_matrix.values[0][1]
            dg_23 = transition_matrix.values[1][2]
            ug_21 = transition_matrix.values[1][0]

            lines.append("\n**Analyse des flux :**")

            if dg_12 > 0.05:
                intensity = (
                    "significatif et reflete une deterioration marquee"
                    if dg_12 > 0.15
                    else "modere mais a surveiller"
                )
                lines.append(
                    f"- **{dg_12:.1%}** des clients Stage 1 migrent vers "
                    f"Stage 2 sous stress (SICR detecte). Ce flux de "
                    f"downgrade est {intensity}."
                )

            if dg_23 > 0.05:
                lines.append(
                    f"- **{dg_23:.1%}** des clients Stage 2 passent en "
                    f"Stage 3 (defaut). L'ECL lifetime sur ces clients "
                    f"passe a PD=100%, amplifiant fortement la provision."
                )

            if ug_21 > 0.01:
                lines.append(
                    f"- **{ug_21:.1%}** de cure rate (Stage 2 -> Stage 1), "
                    f"indiquant une certaine resilience du portefeuille."
                )

        # Vulnérabilité par segment
        lines.append("\n**Vulnerabilite par segment :**")

        seg_vuln: List[Tuple[str, float, float, float]] = []
        for seg_name in result_stressed["segment"].unique():
            seg_a = analytics["segment_analytics"].get(seg_name)
            if seg_a is None:
                continue
            s2_base = seg_a["stage_dist_base"][2]
            s2_stress = seg_a["stage_dist_stressed"][2]
            s3_base = seg_a["stage_dist_base"][3]
            s3_stress = seg_a["stage_dist_stressed"][3]
            downgrade = (s2_stress - s2_base) + (s3_stress - s3_base)
            seg_vuln.append((seg_name, downgrade, s2_stress, s3_stress))

        seg_vuln.sort(key=lambda x: x[1], reverse=True)

        for seg_name, intensity, s2, s3 in seg_vuln:
            if abs(intensity) > 0.005:
                lines.append(
                    f"- **{seg_name}** : Stage 2 = {s2:.1%}, "
                    f"Stage 3 = {s3:.1%} "
                    f"(intensite de downgrade : {intensity:+.1%})"
                )
            else:
                lines.append(
                    f"- **{seg_name}** : Staging stable "
                    f"(S2={s2:.1%}, S3={s3:.1%})"
                )

        return "\n".join(lines)

    def _section_model_governance(
        self,
        model_comparison: pd.DataFrame,
        psi_value: float,
        selected_model: str,
        backtesting_df: Optional[pd.DataFrame],
        shap_top_features: Optional[List[Tuple[str, float]]],
    ) -> str:
        """Section 5 : Gouvernance des modèles."""
        lines = ["## 5. Gouvernance des Modeles"]

        # Performance
        lines.append(
            f"\n**Modele selectionne pour l'ECL : {selected_model}**\n"
        )
        lines.append("| Modele | AUC | Gini | KS | Verdict |")
        lines.append("|--------|-----|------|----|---------|")

        best_auc = 0.0
        best_model = ""
        for _, row in model_comparison.iterrows():
            auc = float(row.get("auc_test", 0))
            gini = float(row.get("gini_test", 0))
            ks = float(row.get("ks_test", 0))

            if auc > best_auc:
                best_auc = auc
                best_model = str(row["model"])

            verdict = (
                "Excellent" if auc >= 0.80
                else "Bon" if auc >= 0.70
                else "Acceptable" if auc >= 0.60
                else "Insuffisant"
            )
            marker = " *" if str(row["model"]) == selected_model else ""
            lines.append(
                f"| {row['model']}{marker} | {auc:.4f} | "
                f"{gini:.4f} | {ks:.4f} | {verdict} |"
            )

        if best_model != selected_model and best_auc > 0:
            lines.append(
                f"\n> Note : Le modele **{best_model}** affiche la "
                f"meilleure AUC ({best_auc:.4f}). Envisager son "
                f"utilisation pour le calcul ECL."
            )

        # PSI
        lines.append(
            f"\n**Stabilite du modele (PSI) :** {psi_value:.4f}"
        )
        if psi_value < 0.10:
            lines.append(
                "-> Distribution stable. Aucune action requise."
            )
        elif psi_value < 0.25:
            lines.append(
                "-> Shift modere detecte. Renforcer la frequence de "
                "monitoring et preparer une analyse de sensibilite."
            )
        else:
            lines.append(
                "-> **DRIFT SIGNIFICATIF.** La distribution des scores a "
                "substantiellement evolue. Recalibration du modele "
                "necessaire."
            )

        # Backtesting
        if backtesting_df is not None and not backtesting_df.empty:
            auc_values = backtesting_df["auc"].values
            auc_mean = float(auc_values.mean())
            auc_std = float(auc_values.std())
            auc_min = float(auc_values.min())
            auc_max = float(auc_values.max())

            lines.append(
                f"\n**Backtesting walk-forward "
                f"({len(auc_values)} folds) :**"
            )
            lines.append(
                f"- AUC : moyenne {auc_mean:.4f}, "
                f"ecart-type {auc_std:.4f}, "
                f"range [{auc_min:.4f} - {auc_max:.4f}]"
            )

            if auc_std < 0.015:
                lines.append(
                    "-> Stabilite **excellente** du modele dans le temps."
                )
            elif auc_std < 0.03:
                lines.append(
                    "-> Stabilite **acceptable**. Legere variabilite "
                    "temporelle."
                )
            else:
                lines.append(
                    "-> **Instabilite detectee.** La performance varie "
                    "significativement selon la periode. Investiguer les "
                    "drivers de cette instabilite."
                )

            # Trend detection
            if len(auc_values) >= 3:
                recent = auc_values[-2:].mean()
                earlier = auc_values[:2].mean()
                if recent < earlier - 0.02:
                    lines.append(
                        f"  Tendance baissiere detectee : AUC recent "
                        f"({recent:.4f}) < AUC initial ({earlier:.4f}). "
                        f"Le modele perd en pouvoir discriminant."
                    )

        # SHAP
        if shap_top_features:
            lines.append(
                "\n**Features les plus influentes (SHAP) :**\n"
            )
            lines.append("| Rang | Feature | Impact moyen |")
            lines.append("|------|---------|--------------|")

            for i, (name, val) in enumerate(
                shap_top_features[:8], 1
            ):
                lines.append(f"| {i} | {name} | {val:.4f} |")

            # Cohérence économique
            top_feat = shap_top_features[0][0]
            expected = [
                "credit_score", "debt_ratio", "income",
                "nb_past_due_30d", "utilization_rate",
                "loan_amount",
            ]
            if any(f in top_feat for f in expected):
                lines.append(
                    "\n-> La feature la plus influente est coherente "
                    "avec la theorie du risque de credit."
                )
            else:
                lines.append(
                    f"\n-> La feature dominante ({top_feat}) merite "
                    f"une analyse approfondie pour verifier sa "
                    f"coherence economique."
                )

        return "\n".join(lines)

    def _section_concentration(
        self,
        hhi_segment: Optional[float],
        hhi_loan: Optional[float],
        result_stressed: pd.DataFrame,
    ) -> str:
        """Section 6 : Concentration et diversification."""
        lines = ["## 6. Concentration et Diversification"]

        if hhi_segment is not None:
            lines.append("\n**Indice HHI :**")
            hhi_label = (
                "CONCENTRE (> 0.25)" if hhi_segment > 0.25
                else "MODERE (0.15 - 0.25)" if hhi_segment > 0.15
                else "DIVERSIFIE (< 0.15)"
            )
            lines.append(
                f"- Segments : **{hhi_segment:.4f}** - {hhi_label}"
            )
            if hhi_loan is not None:
                hhi_l = (
                    "CONCENTRE" if hhi_loan > 0.25
                    else "MODERE" if hhi_loan > 0.15
                    else "DIVERSIFIE"
                )
                lines.append(
                    f"- Types de pret : **{hhi_loan:.4f}** - {hhi_l}"
                )

        # Contribution ECL par segment
        seg_ecl = (
            result_stressed.groupby("segment")["ecl_weighted"]
            .sum()
            .sort_values(ascending=False)
        )
        ecl_total = seg_ecl.sum()

        lines.append("\n**Contribution ECL par segment :**\n")
        lines.append("| Segment | ECL | Part | Cumul |")
        lines.append("|---------|-----|------|-------|")

        cumul = 0.0
        for seg_name, ecl_val in seg_ecl.items():
            pct = ecl_val / ecl_total if ecl_total > 0 else 0
            cumul += pct
            lines.append(
                f"| {seg_name} | {ecl_val:,.0f} EUR | "
                f"{pct:.1%} | {cumul:.1%} |"
            )

        # Alerte concentration
        top_seg_pct = (
            float(seg_ecl.iloc[0]) / ecl_total
            if ecl_total > 0 else 0
        )
        if top_seg_pct > 0.40:
            lines.append(
                f"\n**Alerte concentration :** Le segment "
                f"{seg_ecl.index[0]} represente **{top_seg_pct:.0%}** "
                f"de l'ECL total. Une diversification ou un renforcement "
                f"des limites est recommande."
            )

        # Par type de produit
        loan_ecl = (
            result_stressed.groupby("loan_type")["ecl_weighted"]
            .sum()
            .sort_values(ascending=False)
        )

        lines.append("\n**ECL par type de produit :**")
        for loan_name, ecl_val in loan_ecl.items():
            pct = ecl_val / ecl_total if ecl_total > 0 else 0
            lines.append(
                f"- {loan_name} : {ecl_val:,.0f} EUR ({pct:.1%})"
            )

        return "\n".join(lines)

    def _section_forward_looking(
        self,
        analytics: Dict[str, Any],
        macro_params: Dict[str, float],
        result_stressed: pd.DataFrame,
    ) -> str:
        """Section 7 : Perspectives forward-looking."""
        lines = ["## 7. Perspectives Forward-Looking"]

        unemployment = macro_params.get("unemployment_rate", 7.5)
        gdp = macro_params.get("gdp_growth", 1.2)
        interest_rate = macro_params.get("interest_rate", 3.5)
        hpi = macro_params.get("hpi_growth", 2.0)
        inflation = macro_params.get("inflation_rate", 2.5)

        risks: List[str] = []

        if unemployment > 8.0:
            risks.append(
                f"**Chomage persistant ({unemployment:.1f}%)** : "
                f"Si le chomage se maintient au-dessus de 8%, la "
                f"migration Stage 1 -> 2 continuera de s'accelerer, "
                f"en particulier sur Jeunes_Actifs (sensibilite 2.5x). "
                f"Risque de vague de defauts en T+2/T+3."
            )

        if interest_rate > 4.0:
            risks.append(
                f"**Taux eleves ({interest_rate:.1f}%)** : Le "
                f"resserrement monetaire pese sur les emprunteurs a "
                f"taux variable et les Primo_Accedants "
                f"(sensibilite 2.0x). Risque de refinancement couteux "
                f"et de pression sur les prix immobiliers."
            )

        if hpi < 0:
            risks.append(
                f"**Marche immobilier en contraction "
                f"({hpi:+.1f}%)** : La baisse des prix reduit la "
                f"valeur du collateral et augmente la LGD. Les "
                f"Primo_Accedants (sensibilite 1.8x) sont en premiere "
                f"ligne avec un risque de negative equity."
            )

        if inflation > 3.5:
            risks.append(
                f"**Inflation elevee ({inflation:.1f}%)** : "
                f"L'erosion du pouvoir d'achat touche principalement "
                f"les menages a revenus modestes (Jeunes_Actifs, "
                f"sensibilite 2.0x). Risque de spirale : inflation -> "
                f"baisse consommation -> ralentissement PIB -> hausse "
                f"chomage."
            )

        if gdp < 0:
            risks.append(
                f"**Recession ({gdp:+.1f}%)** : La contraction du "
                f"PIB annonce une degradation generalisee de la qualite "
                f"de credit. Prevoir une hausse de la provision ECL de "
                f"15-25% sur les 2 prochains trimestres."
            )

        # Stress combiné
        n_adverse = sum([
            unemployment > 8.5,
            gdp < 0.5,
            interest_rate > 4.5,
            hpi < 0,
            inflation > 4.0,
        ])

        if n_adverse >= 3:
            risks.append(
                f"**Stress combine ({n_adverse} variables "
                f"defavorables)** : La conjonction de plusieurs chocs "
                f"macro cree un risque systemique. L'effet combine est "
                f"superieur a la somme des parties en raison des boucles "
                f"de retroaction (ex : taux eleves -> baisse immobilier "
                f"-> hausse LGD ; chomage -> defaut -> pertes bancaires "
                f"-> restriction credit -> chomage)."
            )

        if risks:
            lines.append("\n**Risques emergents identifies :**\n")
            for r in risks:
                lines.append(f"{r}\n")
        else:
            lines.append(
                "\nLe contexte macro est favorable ou neutre. Aucun "
                "risque emergent majeur identifie a ce stade. Maintenir "
                "le monitoring standard."
            )

        # Pipeline Stage 2
        s2_pct = float((result_stressed["stage"] == 2).mean())
        if s2_pct > 0.15:
            s2_ecl = result_stressed.loc[
                result_stressed["stage"] == 2, "ecl_weighted"
            ].sum()
            lines.append(
                f"\n**Pipeline Stage 2 :** Avec {s2_pct:.1%} du "
                f"portefeuille en Stage 2, il existe un reservoir "
                f"significatif de migrations potentielles vers Stage 3 "
                f"en cas d'aggravation macro. Estimation : si 20% du "
                f"Stage 2 migre en Stage 3, l'impact ECL additionnel "
                f"serait de l'ordre de {s2_ecl * 0.5:,.0f} EUR."
            )

        return "\n".join(lines)

    def _section_recommendations(
        self,
        analytics: Dict[str, Any],
        macro_params: Dict[str, float],
        psi_value: float,
        hhi_segment: Optional[float],
        result_stressed: pd.DataFrame,
    ) -> str:
        """Section 8 : Recommandations priorisées."""
        lines = ["## 8. Recommandations"]

        recs: List[Dict[str, str]] = []

        ecl_var = analytics["ecl_variation"]

        # 1. ECL
        if ecl_var > 0.20:
            recs.append({
                "priority": "HAUTE",
                "action": "Escalade Direction des Risques",
                "detail": (
                    f"Variation ECL de {ecl_var:+.1%}. Convoquer un "
                    f"comite des risques extraordinaire sous 48h. "
                    f"Preparer une note d'impact pour le regulateur."
                ),
                "kpi": f"ECL : {analytics['ecl_delta']:+,.0f} EUR",
            })
        elif ecl_var > 0.10:
            recs.append({
                "priority": "MOYENNE",
                "action": "Revue de l'appetit au risque",
                "detail": (
                    f"L'ECL a augmente de {ecl_var:+.1%}. Verifier "
                    f"le budget de provisionnement. Preparer un "
                    f"scenario de stress additionnel."
                ),
                "kpi": f"ECL : {analytics['ecl_delta']:+,.0f} EUR",
            })

        # 2. Segment
        for seg_name in analytics["ranked_segments"][:2]:
            seg = analytics["segment_analytics"][seg_name]
            if (
                seg["ecl_contribution"] > 0.35
                and seg["ecl_variation"] > 0.10
            ):
                recs.append({
                    "priority": "HAUTE",
                    "action": (
                        f"Renforcer les criteres d'octroi sur "
                        f"{seg_name}"
                    ),
                    "detail": (
                        f"Le segment concentre "
                        f"{seg['ecl_contribution']:.0%} de l'ECL avec "
                        f"un coverage de {seg['coverage']:.2%}. Durcir "
                        f"le scoring d'entree, abaisser le seuil "
                        f"d'acceptation, revoir les limites de "
                        f"concentration."
                    ),
                    "kpi": f"Coverage : {seg['coverage']:.2%}",
                })

        # 3. Modèle
        if psi_value > 0.25:
            recs.append({
                "priority": "HAUTE",
                "action": "Recalibration du modele PD",
                "detail": (
                    f"PSI = {psi_value:.3f} > 0.25. Le modele n'est "
                    f"plus representatif de la population actuelle. "
                    f"Lancer un chantier de recalibration et soumettre "
                    f"au comite de validation des modeles."
                ),
                "kpi": f"PSI : {psi_value:.4f}",
            })
        elif psi_value > 0.10:
            recs.append({
                "priority": "MOYENNE",
                "action": "Monitoring renforce du modele PD",
                "detail": (
                    f"PSI = {psi_value:.3f}. Passer le monitoring en "
                    f"frequence mensuelle. Preparer un plan de "
                    f"recalibration si degradation supplementaire."
                ),
                "kpi": f"PSI : {psi_value:.4f}",
            })

        # 4. Concentration
        if hhi_segment is not None and hhi_segment > 0.25:
            recs.append({
                "priority": "MOYENNE",
                "action": "Diversification du portefeuille",
                "detail": (
                    f"HHI segments = {hhi_segment:.4f}. Concentration "
                    f"elevee. Revoir les limites de concentration "
                    f"sectorielle et les objectifs commerciaux."
                ),
                "kpi": f"HHI : {hhi_segment:.4f}",
            })

        # 5. Staging
        s2_pct = float((result_stressed["stage"] == 2).mean())
        if s2_pct > 0.20:
            recs.append({
                "priority": "MOYENNE",
                "action": (
                    "Renforcer le suivi des watchlists Stage 2"
                ),
                "detail": (
                    f"Stage 2 = {s2_pct:.1%} du portefeuille. "
                    f"Analyser les flux d'entree par vintage et "
                    f"segment. Mettre en place des actions preventives "
                    f"(restructuration, moratoire) pour limiter la "
                    f"migration vers Stage 3."
                ),
                "kpi": f"Stage 2 : {s2_pct:.1%}",
            })

        # 6. Macro-spécifique
        unemployment = macro_params.get("unemployment_rate", 7.5)
        if unemployment > 9.0:
            recs.append({
                "priority": "HAUTE",
                "action": "Plan de contingence emploi",
                "detail": (
                    f"Chomage a {unemployment:.1f}%. Preparer un "
                    f"plan de restructuration preventive pour les "
                    f"emprunteurs des secteurs les plus touches. "
                    f"Envisager des moratoires cibles."
                ),
                "kpi": f"Chomage : {unemployment:.1f}%",
            })

        # Default si aucune alerte haute
        if not any(r["priority"] == "HAUTE" for r in recs):
            recs.append({
                "priority": "STANDARD",
                "action": "Maintien du monitoring standard",
                "detail": (
                    "Aucune alerte critique. Poursuivre le monitoring "
                    "trimestriel et les comites de suivi mensuels."
                ),
                "kpi": f"ECL variation : {ecl_var:+.1%}",
            })

        # Tri par priorité
        priority_order = {"HAUTE": 0, "MOYENNE": 1, "STANDARD": 2}
        recs.sort(key=lambda r: priority_order.get(r["priority"], 3))

        lines.append(
            f"\n{len(recs)} recommandation(s) identifiee(s) :\n"
        )

        priority_icons = {
            "HAUTE": "[!!!]",
            "MOYENNE": "[!!]",
            "STANDARD": "[OK]",
        }

        for i, rec in enumerate(recs, 1):
            icon = priority_icons.get(rec["priority"], "[?]")
            lines.append(
                f"### {i}. {icon} [{rec['priority']}] "
                f"{rec['action']}\n"
                f"{rec['detail']}  \n"
                f"*KPI de suivi : {rec['kpi']}*\n"
            )

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # UTILITY METHODS
    # ──────────────────────────────────────────────

    def _is_adverse(
        self,
        key: str,
        macro_params: Dict[str, float],
    ) -> bool:
        """Vérifie si un paramètre macro est en zone adverse."""
        baseline = self._MACRO_BASELINES[key][1]
        current = macro_params.get(key, baseline)
        delta = current - baseline
        if key in self._ADVERSE_UP:
            return delta > 0.1
        return delta < -0.1

    @staticmethod
    def _macro_explanation(key: str, top_seg: str) -> str:
        """Retourne l'explication économique d'un canal de transmission."""
        explanations = {
            "unemployment_rate": (
                f"La hausse du chomage fragilise directement la capacite "
                f"de remboursement des emprunteurs, en particulier les "
                f"menages a revenus modestes du segment {top_seg}."
            ),
            "gdp_growth": (
                f"Le ralentissement de la croissance reduit les "
                f"perspectives de revenus et d'emploi, amplifiant le "
                f"risque de defaut sur {top_seg}."
            ),
            "interest_rate": (
                f"Le resserrement monetaire augmente la charge de "
                f"dette des emprunteurs a taux variable et reduit la "
                f"solvabilite des primo-accedants ({top_seg})."
            ),
            "hpi_growth": (
                f"La baisse des prix immobiliers reduit la valeur du "
                f"collateral (hausse LGD) et peut creer des situations "
                f"de negative equity sur {top_seg}."
            ),
            "inflation_rate": (
                f"La hausse de l'inflation erode le pouvoir d'achat "
                f"reel des menages et augmente le cout de la vie, "
                f"reduisant la capacite d'epargne de {top_seg}."
            ),
        }
        return explanations.get(key, "")

    def generate_executive_briefing(
        self,
        result_stressed: pd.DataFrame,
        result_base: pd.DataFrame,
        macro_params: Dict[str, float],
        psi_value: float,
    ) -> Dict[str, Any]:
        """Génère un briefing exécutif en prose pour l'insight box.

        Produit une note d'analyste structurée en paragraphes
        professionnels, comme le ferait un CRO senior.

        Args:
            result_stressed: DataFrame ECL stressé.
            result_base: DataFrame ECL baseline.
            macro_params: Paramètres macro courants.
            psi_value: PSI du modèle sélectionné.

        Returns:
            Dictionnaire structuré pour le rendu :
                - risk_level: str ('ELEVE', 'MODERE', 'MAITRISE')
                - risk_score: int (0-10)
                - risk_color: str ('rouge', 'orange', 'vert')
                - diagnostic: str (paragraphe d'analyse principal)
                - findings: list[str] (constats en prose)
                - recommendations: list[str] (préconisations en prose)
        """
        analytics = self._compute_core_analytics(
            result_stressed, result_base, macro_params,
        )

        ecl_var = analytics["ecl_variation"]
        ecl_s = analytics["ecl_stressed"]
        ecl_b = analytics["ecl_base"]
        ecl_delta = analytics["ecl_delta"]
        coverage = analytics["coverage"]
        seg_a = analytics["segment_analytics"]
        ranked = analytics["ranked_segments"]
        top_seg = analytics["top_ecl_segment"]

        # ── Risk scoring ──
        risk_score = 0
        s2_pct = float((result_stressed["stage"] == 2).mean())
        s3_pct = float((result_stressed["stage"] == 3).mean())

        if abs(ecl_var) > 0.15:
            risk_score += 3
        elif abs(ecl_var) > 0.05:
            risk_score += 1
        if s2_pct > 0.20:
            risk_score += 2
        elif s2_pct > 0.15:
            risk_score += 1
        if psi_value > 0.25:
            risk_score += 2
        elif psi_value > 0.10:
            risk_score += 1

        n_adverse = sum(
            1 for key in self._MACRO_BASELINES
            if self._is_adverse(key, macro_params)
        )
        if n_adverse >= 3:
            risk_score += 2
        elif n_adverse >= 1:
            risk_score += 1

        if risk_score >= 5:
            risk_level, risk_color = "ELEVE", "rouge"
        elif risk_score >= 2:
            risk_level, risk_color = "MODERE", "orange"
        else:
            risk_level, risk_color = "MAITRISE", "vert"

        # ── Diagnostic paragraph (prose) ──
        diagnostic = self._build_diagnostic_prose(
            analytics, macro_params, s2_pct, s3_pct, n_adverse,
        )

        # ── Findings (prose paragraphs) ──
        findings = self._build_findings_prose(
            analytics, macro_params, psi_value, s2_pct,
        )

        # ── Recommendations (prose) ──
        recommendations = self._build_recommendations_prose(
            analytics, macro_params, psi_value, s2_pct,
        )

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_color": risk_color,
            "diagnostic": diagnostic,
            "findings": findings,
            "recommendations": recommendations,
        }

    def _build_diagnostic_prose(
        self,
        analytics: Dict[str, Any],
        macro_params: Dict[str, float],
        s2_pct: float,
        s3_pct: float,
        n_adverse: int,
    ) -> str:
        """Construit le paragraphe de diagnostic en prose professionnelle."""
        ecl_var = analytics["ecl_variation"]
        ecl_s = analytics["ecl_stressed"]
        ecl_b = analytics["ecl_base"]
        coverage = analytics["coverage"]
        pd_base = analytics["pd_mean_base"]
        pd_stress = analytics["pd_mean_stressed"]
        n_clients = analytics["n_clients"]
        top_seg = analytics["top_ecl_segment"]
        top_d = analytics["segment_analytics"][top_seg]

        parts: List[str] = []

        if abs(ecl_var) < 0.01:
            parts.append(
                f"Le portefeuille de {n_clients:,} clients presente "
                f"un niveau de risque stable, avec un ECL de "
                f"{ecl_s:,.0f} EUR et un taux de couverture de "
                f"{coverage:.2%}. Les parametres macroeconomiques "
                f"actuels n'induisent pas de stress significatif "
                f"sur les provisions."
            )
            return " ".join(parts)

        direction = "en hausse" if ecl_var > 0 else "en baisse"

        parts.append(
            f"Le portefeuille de {n_clients:,} clients affiche un "
            f"ECL de {ecl_s:,.0f} EUR sous les conditions de stress "
            f"actuelles, {direction} de {abs(ecl_var):.1%} par rapport "
            f"au scenario de base ({ecl_b:,.0f} EUR)."
        )

        parts.append(
            f"La probabilite de defaut moyenne est passee de "
            f"{pd_base:.2%} a {pd_stress:.2%}, portant le taux de "
            f"couverture a {coverage:.2%}."
        )

        # Macro context
        active_labels: List[str] = []
        for key, (label, baseline) in self._MACRO_BASELINES.items():
            if self._is_adverse(key, macro_params):
                current = macro_params.get(key, baseline)
                label_fr = self._MACRO_LABELS_FR.get(key, label.lower())
                active_labels.append(f"{label_fr} a {current:.1f}%")

        if active_labels:
            if n_adverse >= 3 and len(active_labels) > 1:
                parts.append(
                    f"Cette deterioration s'inscrit dans un contexte "
                    f"de stress macroeconomique generalise, avec "
                    f"{', '.join(active_labels[:-1])} et "
                    f"{active_labels[-1]}."
                )
            elif len(active_labels) == 1:
                parts.append(
                    f"Le principal facteur de stress identifie est "
                    f"{active_labels[0]}."
                )
            else:
                parts.append(
                    f"Les facteurs de stress identifies sont "
                    f"{' et '.join(active_labels)}."
                )

        # Top segment
        if top_d["ecl_contribution"] > 0.30 and abs(ecl_var) > 0.05:
            parts.append(
                f"Le segment {top_seg} concentre "
                f"{top_d['ecl_contribution']:.0%} de l'ECL total "
                f"avec un taux de couverture de "
                f"{top_d['coverage']:.2%}, ce qui en fait le "
                f"principal contributeur au risque du portefeuille."
            )

        return " ".join(parts)

    def _build_findings_prose(
        self,
        analytics: Dict[str, Any],
        macro_params: Dict[str, float],
        psi_value: float,
        s2_pct: float,
    ) -> List[Dict[str, str]]:
        """Construit les constats en phrases complètes avec sévérité.

        Returns:
            Liste de dicts {"severity": str, "text": str}.
        """
        findings: List[Dict[str, str]] = []
        seg_a = analytics["segment_analytics"]
        ecl_var = analytics["ecl_variation"]

        if abs(ecl_var) < 0.01:
            return findings

        # Macro → Segment causal chains (diversified by driver)
        active_keys = [
            key for key in self._MACRO_BASELINES
            if self._is_adverse(key, macro_params)
        ]

        driver_findings: List[Tuple[float, str, str]] = []
        for key in active_keys:
            label_fr = self._MACRO_LABELS_FR.get(
                key, self._MACRO_BASELINES[key][0].lower(),
            )
            baseline = self._MACRO_BASELINES[key][1]
            current = macro_params.get(key, baseline)
            delta = current - baseline

            best_seg = None
            best_impact = 0.0
            for seg_name, s in seg_a.items():
                eff = s["macro_impacts"][key]["estimated_pd_multiplier"] - 1
                if eff > best_impact:
                    best_impact = eff
                    best_seg = seg_name

            if best_seg and best_impact > 0.01:
                s = seg_a[best_seg]
                sens = s["macro_impacts"][key]["sensitivity"]

                severity = (
                    "CRITICAL" if best_impact > 0.5
                    else "ALERT" if best_impact > 0.15
                    else "WARNING"
                )

                text = (
                    f"L'ecart observe sur {label_fr} "
                    f"({delta:+.1f} points par rapport au baseline) "
                    f"impacte en premier lieu le segment {best_seg}, "
                    f"dont la sensibilite a cette variable "
                    f"({sens}x) amplifie la degradation du risque. "
                    f"Ce segment represente {s['ecl_contribution']:.0%} "
                    f"de l'ECL total avec une PD moyenne de "
                    f"{s['pd_stressed']:.2%} et un taux de couverture "
                    f"de {s['coverage']:.2%}."
                )
                driver_findings.append((best_impact, severity, text))

        driver_findings.sort(key=lambda x: x[0], reverse=True)
        for _, severity, text in driver_findings[:3]:
            findings.append({"severity": severity, "text": text})

        # Stage 2 pipeline
        if s2_pct > 0.15:
            sev = "ALERT" if s2_pct > 0.20 else "WARNING"
            findings.append({
                "severity": sev,
                "text": (
                    f"La part du portefeuille classee en Stage 2 "
                    f"(SICR detecte) s'eleve a {s2_pct:.1%}, ce qui "
                    f"constitue un reservoir significatif de migrations "
                    f"potentielles vers le defaut en cas d'aggravation "
                    f"du contexte macroeconomique. Une attention "
                    f"particuliere doit etre portee a ces expositions "
                    f"pour prevenir le passage en Stage 3."
                ),
            })

        # PSI
        if psi_value > 0.10:
            if psi_value > 0.25:
                findings.append({
                    "severity": "CRITICAL",
                    "text": (
                        f"Le Population Stability Index du modele PD "
                        f"atteint {psi_value:.3f}, bien au-dela du seuil "
                        f"de 0,25. La distribution des scores a "
                        f"significativement evolue depuis la calibration "
                        f"initiale, ce qui remet en question la fiabilite "
                        f"des estimations de risque produites par le "
                        f"modele. Une recalibration est necessaire."
                    ),
                })
            else:
                findings.append({
                    "severity": "WARNING",
                    "text": (
                        f"Le PSI du modele PD s'etablit a "
                        f"{psi_value:.3f}, au-dessus du seuil de "
                        f"vigilance de 0,10. Cette derive moderee de la "
                        f"distribution des scores justifie un renforcement "
                        f"du dispositif de monitoring."
                    ),
                })

        # Combined stress
        n_adverse = sum(
            1 for key in self._MACRO_BASELINES
            if self._is_adverse(key, macro_params)
        )
        if n_adverse >= 3:
            sev = "CRITICAL" if n_adverse >= 4 else "ALERT"
            findings.append({
                "severity": sev,
                "text": (
                    f"Le portefeuille est soumis a un stress simultane "
                    f"sur {n_adverse} des 5 variables macroeconomiques "
                    f"suivies. Cette conjonction de facteurs defavorables "
                    f"genere des effets de second tour et des boucles de "
                    f"retroaction qui amplifient l'impact au-dela de la "
                    f"somme des chocs individuels."
                ),
            })

        return findings

    def _build_recommendations_prose(
        self,
        analytics: Dict[str, Any],
        macro_params: Dict[str, float],
        psi_value: float,
        s2_pct: float,
    ) -> List[Dict[str, str]]:
        """Construit les recommandations en prose avec priorité.

        Returns:
            Liste de dicts {"priority": str, "text": str}.
        """
        recs: List[Dict[str, str]] = []
        ecl_var = analytics["ecl_variation"]
        seg_a = analytics["segment_analytics"]
        ranked = analytics["ranked_segments"]

        if abs(ecl_var) < 0.01 and s2_pct < 0.15 and psi_value < 0.10:
            recs.append({
                "priority": "INFO",
                "text": (
                    "Les indicateurs de risque sont dans les limites "
                    "de l'appetit au risque. Il est recommande de "
                    "maintenir le dispositif de suivi a frequence "
                    "trimestrielle."
                ),
            })
            return recs

        if ecl_var > 0.20:
            recs.append({
                "priority": "HAUTE",
                "text": (
                    f"Compte tenu de la hausse de {ecl_var:.0%} de "
                    f"l'ECL, il est recommande de proceder a une "
                    f"escalade aupres de la Direction des Risques dans "
                    f"un delai de 48 heures et de preparer une note "
                    f"d'impact a destination du regulateur."
                ),
            })
        elif ecl_var > 0.10:
            recs.append({
                "priority": "MOYENNE",
                "text": (
                    f"La hausse de {ecl_var:.0%} de l'ECL justifie une "
                    f"revue de l'adequation du budget de "
                    f"provisionnement et la preparation d'un scenario "
                    f"de stress additionnel pour le prochain comite "
                    f"des risques."
                ),
            })

        for seg_name in ranked[:2]:
            s = seg_a[seg_name]
            if s["ecl_contribution"] > 0.30 and s["ecl_variation"] > 0.10:
                recs.append({
                    "priority": "HAUTE",
                    "text": (
                        f"Le segment {seg_name}, qui concentre "
                        f"{s['ecl_contribution']:.0%} de l'ECL avec un "
                        f"taux de couverture de {s['coverage']:.2%}, "
                        f"necessite un durcissement des criteres "
                        f"d'octroi et une revue des limites de "
                        f"concentration."
                    ),
                })
                break

        if psi_value > 0.25:
            recs.append({
                "priority": "HAUTE",
                "text": (
                    f"Le drift du modele PD (PSI a {psi_value:.3f}) "
                    f"impose une recalibration dans les meilleurs "
                    f"delais et une soumission au comite de validation "
                    f"des modeles."
                ),
            })

        if s2_pct > 0.20:
            recs.append({
                "priority": "MOYENNE",
                "text": (
                    f"Avec {s2_pct:.1%} du portefeuille en Stage 2, "
                    f"il est preconise de renforcer le suivi des "
                    f"watchlists et de mettre en place des actions "
                    f"preventives (restructuration, moratoire) pour "
                    f"limiter les migrations vers le defaut."
                ),
            })

        unemployment = macro_params.get("unemployment_rate", 7.5)
        if unemployment > 9.0:
            recs.append({
                "priority": "HAUTE",
                "text": (
                    f"Le niveau de chomage a {unemployment:.1f}% "
                    f"appelle la mise en place d'un plan de "
                    f"contingence cible sur les emprunteurs les plus "
                    f"exposes, incluant des dispositifs de "
                    f"restructuration preventive."
                ),
            })

        # Fallback si aucune recommandation spécifique
        if not recs:
            recs.append({
                "priority": "INFO",
                "text": (
                    "Les indicateurs de risque restent dans des "
                    "limites acceptables. Il est recommande de "
                    "maintenir le dispositif de monitoring courant "
                    "et de rester vigilant sur l'evolution des "
                    "parametres macroeconomiques."
                ),
            })

        return recs
