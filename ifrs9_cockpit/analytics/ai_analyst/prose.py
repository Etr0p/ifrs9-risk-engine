"""ProseMixin — generation de prose et briefing executif."""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class ProseMixin:
    """Mixin : prose generation et executive briefing."""

    def generate_executive_briefing(
        self,
        result_stressed,
        result_base,
        macro_params: Dict[str, float],
        psi_value: float,
    ) -> Dict[str, Any]:
        """Genere un briefing executif en prose pour l'insight box.

        Produit une note d'analyste structuree en paragraphes
        professionnels, comme le ferait un CRO senior.

        Args:
            result_stressed: DataFrame ECL stresse.
            result_base: DataFrame ECL baseline.
            macro_params: Parametres macro courants.
            psi_value: PSI du modele selectionne.

        Returns:
            Dictionnaire structure pour le rendu :
                - risk_level: str ('ELEVE', 'MODERE', 'BON')
                - risk_score: int (0-10)
                - risk_color: str ('rouge', 'orange', 'vert')
                - diagnostic: str (paragraphe d'analyse principal)
                - findings: list[str] (constats en prose)
                - recommendations: list[str] (preconisations en prose)
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
            risk_level, risk_color = "BON", "vert"

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
        """Construit les constats en phrases completes avec severite.

        Returns:
            Liste de dicts {"severity": str, "text": str}.
        """
        findings: List[Dict[str, str]] = []
        seg_a = analytics["segment_analytics"]
        ecl_var = analytics["ecl_variation"]

        if abs(ecl_var) < 0.01:
            return findings

        # Macro -> Segment causal chains (diversified by driver)
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
        """Construit les recommandations en prose avec priorite.

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

        # Fallback si aucune recommandation specifique
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
