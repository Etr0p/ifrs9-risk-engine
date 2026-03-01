"""SectionsMixin — les 8 sections du rapport CRO."""

from __future__ import annotations

import numpy as np
import polars as pl
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ifrs9_cockpit.config import SEGMENTS


class SectionsMixin:
    """Mixin : generation des 8 sections du rapport CRO."""

    # ──────────────────────────────────────────────
    # REPORT SECTIONS
    # ──────────────────────────────────────────────

    def _section_header(self) -> str:
        """En-tete du rapport."""
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
        result_stressed: pl.DataFrame,
        macro_params: Dict[str, float],
        psi_value: float,
        hhi_segment: Optional[float],
    ) -> str:
        """Section 1 : Synthese executive avec evaluation du risque global."""
        lines = ["## 1. Synthèse Exécutive"]

        ecl_var = analytics["ecl_variation"]
        ecl_s = analytics["ecl_stressed"]
        ecl_b = analytics["ecl_base"]
        coverage = analytics["coverage"]
        top_seg = analytics["top_ecl_segment"]
        seg_data = analytics["segment_analytics"][top_seg]

        # Scoring du risque global (somme ponderee d'indicateurs)
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
            risk_level, risk_color = "BON", "vert"

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
        waterfall_df: Optional[pl.DataFrame],
    ) -> str:
        """Section 2 : Decomposition ECL et attribution par driver."""
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

        # Decomposition waterfall
        if waterfall_df is not None:
            lines.append("\n**Decomposition du mouvement :**")
            migration_impact = 0.0
            param_impact = 0.0

            for row in waterfall_df.iter_rows(named=True):
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

            # Interpretation
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

            # Causalite : identifier pourquoi ce segment bouge
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
        """Section 3 : Analyse de sensibilite macroeconomique."""
        lines = ["## 3. Sensibilite Macroeconomique"]

        # Tableau des parametres
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
                # Segment le plus sensible a cette variable
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

        # Effet combine
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

            # Identifier les segments a double/triple exposition
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
        result_stressed: pl.DataFrame,
        result_base: pl.DataFrame,
        transition_matrix: Optional[pl.DataFrame],
    ) -> str:
        """Section 4 : Qualite de credit et staging."""
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
            base_pct = base_n / result_base.height
            stress_pct = stress_n / result_stressed.height
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
            # Extract numeric matrix (Polars: select cols; Pandas: .values)
            if hasattr(transition_matrix, "select"):
                _tm_vals = transition_matrix.select(labels).to_numpy()
            else:
                _tm_vals = transition_matrix.values
            lines.append(
                "| De / Vers | Stage 1 | Stage 2 | Stage 3 |"
            )
            lines.append(
                "|-----------|---------|---------|---------|"
            )
            for i, label in enumerate(labels):
                row_vals = " | ".join(
                    f"{_tm_vals[i][j]:.1%}"
                    for j in range(3)
                )
                lines.append(f"| {label} | {row_vals} |")

            # Interpretation des flux
            dg_12 = _tm_vals[0][1]
            dg_23 = _tm_vals[1][2]
            ug_21 = _tm_vals[1][0]

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

        # Vulnerabilite par segment
        lines.append("\n**Vulnerabilite par segment :**")

        seg_vuln: List[Tuple[str, float, float, float]] = []
        for seg_name in result_stressed["segment"].unique().to_list():
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
        model_comparison: pl.DataFrame,
        psi_value: float,
        selected_model: str,
        backtesting_df: Optional[pl.DataFrame],
        shap_top_features: Optional[List[Tuple[str, float]]],
    ) -> str:
        """Section 5 : Gouvernance des modeles."""
        lines = ["## 5. Gouvernance des Modeles"]

        # Performance
        lines.append(
            f"\n**Modele selectionne pour l'ECL : {selected_model}**\n"
        )
        lines.append("| Modele | AUC | Gini | KS | Verdict |")
        lines.append("|--------|-----|------|----|---------|")

        best_auc = 0.0
        best_model = ""
        for row in model_comparison.iter_rows(named=True):
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
        if backtesting_df is not None and len(backtesting_df) > 0:
            auc_values = backtesting_df["auc"].to_numpy()
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

            # Coherence economique
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
        result_stressed: pl.DataFrame,
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
        seg_ecl_df = (
            result_stressed.group_by("segment")
            .agg(pl.col("ecl_weighted").sum())
            .sort("ecl_weighted", descending=True)
        )
        ecl_total = seg_ecl_df["ecl_weighted"].sum()

        lines.append("\n**Contribution ECL par segment :**\n")
        lines.append("| Segment | ECL | Part | Cumul |")
        lines.append("|---------|-----|------|-------|")

        cumul = 0.0
        for row in seg_ecl_df.iter_rows(named=True):
            seg_name = row["segment"]
            ecl_val = row["ecl_weighted"]
            pct = ecl_val / ecl_total if ecl_total > 0 else 0
            cumul += pct
            lines.append(
                f"| {seg_name} | {ecl_val:,.0f} EUR | "
                f"{pct:.1%} | {cumul:.1%} |"
            )

        # Alerte concentration
        top_seg_pct = (
            float(seg_ecl_df[0, "ecl_weighted"]) / ecl_total
            if ecl_total > 0 else 0
        )
        if top_seg_pct > 0.40:
            lines.append(
                f"\n**Alerte concentration :** Le segment "
                f"{seg_ecl_df[0, 'segment']} represente **{top_seg_pct:.0%}** "
                f"de l'ECL total. Une diversification ou un renforcement "
                f"des limites est recommande."
            )

        # Par type de produit
        loan_ecl_df = (
            result_stressed.group_by("loan_type")
            .agg(pl.col("ecl_weighted").sum())
            .sort("ecl_weighted", descending=True)
        )

        lines.append("\n**ECL par type de produit :**")
        for row in loan_ecl_df.iter_rows(named=True):
            loan_name = row["loan_type"]
            ecl_val = row["ecl_weighted"]
            pct = ecl_val / ecl_total if ecl_total > 0 else 0
            lines.append(
                f"- {loan_name} : {ecl_val:,.0f} EUR ({pct:.1%})"
            )

        return "\n".join(lines)

    def _section_forward_looking(
        self,
        analytics: Dict[str, Any],
        macro_params: Dict[str, float],
        result_stressed: pl.DataFrame,
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

        # Stress combine
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
            s2_ecl = result_stressed.filter(
                pl.col("stage") == 2
            )["ecl_weighted"].sum()
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
        result_stressed: pl.DataFrame,
    ) -> str:
        """Section 8 : Recommandations priorisees."""
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

        # 3. Modele
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

        # 6. Macro-specifique
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

        # Tri par priorite
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
