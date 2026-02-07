"""Module Virtual CRO — Le différenciateur du Cockpit IFRS 9.

Génère automatiquement des rapports textuels business à destination
du Chief Risk Officer. Analyse les indicateurs clés et produit des
commentaires contextualisés avec 5 règles d'alerte :

    1. ECL variation > 15%   → Ton "ALERTE"
    2. Segment touché + driver macro identifié
    3. Stage 2 > 20%         → Warning concentration
    4. PSI > 0.15            → Drift alert modèle
    5. Action recommandée concrète
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ifrs9_cockpit.config import CRO_CONFIG, SEGMENTS


@dataclass
class CROAlert:
    """Alerte générée par le Virtual CRO.

    Attributes:
        severity: Niveau de sévérité ('INFO', 'WARNING', 'ALERT', 'CRITICAL').
        rule_id: Identifiant de la règle déclenchée (1-5).
        title: Titre court de l'alerte.
        message: Message détaillé avec contexte business.
        metric_value: Valeur de la métrique ayant déclenché l'alerte.
        threshold: Seuil de déclenchement.
        action: Action recommandée.
    """
    severity: str
    rule_id: int
    title: str
    message: str
    metric_value: float
    threshold: float
    action: str


class VirtualCRO:
    """Moteur d'analyse automatique générant des rapports CRO.

    Analyse l'état du portefeuille et produit des commentaires
    business contextualisés, comme le ferait un Risk Officer senior.

    Attributes:
        ecl_variation_threshold: Seuil de variation ECL pour alerte.
        stage2_warning_pct: Part Stage 2 déclenchant un warning.
        psi_drift_threshold: PSI seuil pour alerte drift.
    """

    def __init__(
        self,
        ecl_variation_threshold: float = CRO_CONFIG.ecl_variation_alert,
        stage2_warning_pct: float = CRO_CONFIG.stage2_warning_pct,
        psi_drift_threshold: float = CRO_CONFIG.psi_drift_threshold,
    ) -> None:
        """Initialise le Virtual CRO.

        Args:
            ecl_variation_threshold: Seuil variation ECL (défaut 15%).
            stage2_warning_pct: Seuil Stage 2 (défaut 20%).
            psi_drift_threshold: Seuil PSI (défaut 0.15).
        """
        self.ecl_variation_threshold = ecl_variation_threshold
        self.stage2_warning_pct = stage2_warning_pct
        self.psi_drift_threshold = psi_drift_threshold

    def analyze(
        self,
        result_df: pd.DataFrame,
        ecl_previous: Optional[float] = None,
        psi_value: float = 0.0,
        unemployment_rate: float = 7.5,
        gdp_growth: float = 1.2,
    ) -> List[CROAlert]:
        """Analyse complète du portefeuille et génération des alertes.

        Args:
            result_df: DataFrame résultat du ECLCalculator.calculate().
            ecl_previous: ECL de la période précédente (pour variation).
            psi_value: PSI du modèle PD courant.
            unemployment_rate: Taux de chômage courant.
            gdp_growth: Croissance PIB courante.

        Returns:
            Liste d'alertes CRO triées par sévérité.
        """
        alerts: List[CROAlert] = []

        # Règle 1 : Variation ECL
        alerts.extend(self._rule_ecl_variation(result_df, ecl_previous))

        # Règle 2 : Segment touché + driver macro
        alerts.extend(self._rule_segment_analysis(
            result_df, unemployment_rate, gdp_growth,
        ))

        # Règle 3 : Concentration Stage 2
        alerts.extend(self._rule_stage2_concentration(result_df))

        # Règle 4 : PSI drift
        alerts.extend(self._rule_psi_drift(psi_value))

        # Règle 5 : Actions recommandées (synthèse)
        alerts.extend(self._rule_recommended_actions(result_df, alerts))

        # Trier par sévérité
        severity_order = {"CRITICAL": 0, "ALERT": 1, "WARNING": 2, "INFO": 3}
        alerts.sort(key=lambda a: severity_order.get(a.severity, 4))

        return alerts

    def generate_report(
        self,
        result_df: pd.DataFrame,
        ecl_previous: Optional[float] = None,
        psi_value: float = 0.0,
        unemployment_rate: float = 7.5,
        gdp_growth: float = 1.2,
    ) -> str:
        """Génère un rapport textuel complet pour le CRO.

        Args:
            result_df: DataFrame résultat ECL.
            ecl_previous: ECL précédent.
            psi_value: PSI du modèle.
            unemployment_rate: Chômage courant.
            gdp_growth: PIB courant.

        Returns:
            Rapport formaté en texte structuré.
        """
        alerts = self.analyze(
            result_df, ecl_previous, psi_value,
            unemployment_rate, gdp_growth,
        )

        ecl_total = result_df["ecl_weighted"].sum()
        ead_total = result_df["ead"].sum()
        coverage = ecl_total / ead_total if ead_total > 0 else 0

        # En-tête
        lines = [
            f"RAPPORT RISK — {datetime.now().strftime('%d/%m/%Y')}",
            "",
            f"ECL Total : {ecl_total:,.0f} EUR  |  "
            f"EAD : {ead_total:,.0f} EUR  |  "
            f"Coverage : {coverage:.2%}",
            "",
        ]

        if not alerts:
            lines.append("Aucune alerte. Portefeuille dans les limites normales.")
            return "\n".join(lines)

        # Alertes par sévérité
        for alert in alerts:
            icon = self._severity_icon(alert.severity)
            lines.append(f"{icon} [{alert.severity}] {alert.title}")
            lines.append(f"   {alert.message}")
            if alert.action:
                lines.append(f"   >> Action : {alert.action}")
            lines.append("")

        return "\n".join(lines)

    def get_executive_summary(
        self,
        result_df: pd.DataFrame,
        unemployment_rate: float = 7.5,
        gdp_growth: float = 1.2,
    ) -> Dict[str, object]:
        """Résumé exécutif structuré pour le dashboard.

        Args:
            result_df: DataFrame résultat ECL.
            unemployment_rate: Chômage courant.
            gdp_growth: PIB courant.

        Returns:
            Dictionnaire avec les KPI et insights clés.
        """
        ecl_total = result_df["ecl_weighted"].sum()
        ead_total = result_df["ead"].sum()
        n_total = len(result_df)

        stages = result_df["stage"].values
        stage_dist = {
            f"stage_{s}": int((stages == s).sum())
            for s in [1, 2, 3]
        }
        stage_pct = {
            f"stage_{s}_pct": round((stages == s).mean(), 4)
            for s in [1, 2, 3]
        }

        # ECL par segment
        ecl_by_segment = (
            result_df.groupby("segment")["ecl_weighted"]
            .sum()
            .to_dict()
        )

        # Segment le plus risqué
        coverage_by_segment = {}
        for seg in result_df["segment"].unique():
            mask = result_df["segment"] == seg
            seg_ecl = result_df.loc[mask, "ecl_weighted"].sum()
            seg_ead = result_df.loc[mask, "ead"].sum()
            coverage_by_segment[seg] = seg_ecl / seg_ead if seg_ead > 0 else 0

        riskiest_segment = max(coverage_by_segment, key=coverage_by_segment.get)

        return {
            "ecl_total": round(ecl_total, 2),
            "ead_total": round(ead_total, 2),
            "coverage": round(ecl_total / ead_total, 4) if ead_total > 0 else 0,
            "n_clients": n_total,
            "pd_mean": round(result_df["pd_12m"].mean(), 4),
            **stage_dist,
            **stage_pct,
            "ecl_by_segment": ecl_by_segment,
            "coverage_by_segment": coverage_by_segment,
            "riskiest_segment": riskiest_segment,
            "unemployment_rate": unemployment_rate,
            "gdp_growth": gdp_growth,
        }

    # ──────────────────────────────────────────
    # RÈGLES D'ALERTE
    # ──────────────────────────────────────────

    def _rule_ecl_variation(
        self,
        result_df: pd.DataFrame,
        ecl_previous: Optional[float],
    ) -> List[CROAlert]:
        """Règle 1 : Variation ECL > seuil → ALERTE.

        Args:
            result_df: DataFrame résultat ECL.
            ecl_previous: ECL période précédente.

        Returns:
            Liste d'alertes (0 ou 1).
        """
        if ecl_previous is None or ecl_previous == 0:
            return []

        ecl_current = result_df["ecl_weighted"].sum()
        variation = (ecl_current - ecl_previous) / ecl_previous

        if abs(variation) > self.ecl_variation_threshold:
            direction = "hausse" if variation > 0 else "baisse"
            severity = "ALERT" if variation > 0 else "WARNING"
            return [CROAlert(
                severity=severity,
                rule_id=1,
                title=f"Variation ECL significative ({variation:+.1%})",
                message=(
                    f"L'ECL total est passé de {ecl_previous:,.0f} EUR à "
                    f"{ecl_current:,.0f} EUR, soit une {direction} de "
                    f"{abs(variation):.1%} sur la période. Ce mouvement "
                    f"dépasse le seuil d'alerte de {self.ecl_variation_threshold:.0%}."
                ),
                metric_value=round(variation, 4),
                threshold=self.ecl_variation_threshold,
                action=(
                    "Convoquer le comité des risques pour analyse approfondie "
                    "des drivers de variation."
                    if variation > 0.30
                    else "Documenter les drivers et reporter en comité mensuel."
                ),
            )]
        return []

    def _rule_segment_analysis(
        self,
        result_df: pd.DataFrame,
        unemployment_rate: float,
        gdp_growth: float,
    ) -> List[CROAlert]:
        """Règle 2 : Identification du segment touché et driver macro.

        Args:
            result_df: DataFrame résultat ECL.
            unemployment_rate: Chômage courant.
            gdp_growth: PIB courant.

        Returns:
            Liste d'alertes pour les segments à risque.
        """
        alerts: List[CROAlert] = []

        for seg in SEGMENTS:
            mask = result_df["segment"] == seg.name
            if mask.sum() == 0:
                continue

            seg_data = result_df[mask]
            seg_ecl = seg_data["ecl_weighted"].sum()
            seg_ead = seg_data["ead"].sum()
            coverage = seg_ecl / seg_ead if seg_ead > 0 else 0
            stage2_pct = (seg_data["stage"] == 2).mean()
            stage3_pct = (seg_data["stage"] == 3).mean()
            total_ecl = result_df["ecl_weighted"].sum()
            ecl_contribution = seg_ecl / total_ecl if total_ecl > 0 else 0

            # Identifier si le segment concentre trop de risque
            if coverage > 0.10 or ecl_contribution > 0.40:
                # Identifier le driver macro principal
                if seg.unemployment_sensitivity > 1.5 and unemployment_rate > 8.0:
                    driver = (
                        f"chômage élevé ({unemployment_rate:.1f}%) combiné à "
                        f"une sensibilité segment de {seg.unemployment_sensitivity}x"
                    )
                elif seg.gdp_sensitivity > 1.2 and gdp_growth < 0.5:
                    driver = (
                        f"croissance PIB atone ({gdp_growth:.1f}%) avec "
                        f"sensibilité segment de {seg.gdp_sensitivity}x"
                    )
                else:
                    driver = (
                        f"profil structurel du segment (score moyen "
                        f"{seg.avg_credit_score}, DR de base {seg.base_default_rate:.1%})"
                    )

                severity = "ALERT" if coverage > 0.15 else "WARNING"
                alerts.append(CROAlert(
                    severity=severity,
                    rule_id=2,
                    title=f"Concentration risque sur {seg.name}",
                    message=(
                        f"Le segment {seg.name} représente {ecl_contribution:.0%} "
                        f"de l'ECL total avec un coverage de {coverage:.2%} "
                        f"(Stage 2 : {stage2_pct:.1%}, Stage 3 : {stage3_pct:.1%}). "
                        f"Driver identifié : {driver}."
                    ),
                    metric_value=round(coverage, 4),
                    threshold=0.10,
                    action=(
                        f"Durcir les critères d'octroi sur le segment {seg.name}. "
                        f"Revoir les limites de concentration sectorielle."
                    ),
                ))

        return alerts

    def _rule_stage2_concentration(
        self,
        result_df: pd.DataFrame,
    ) -> List[CROAlert]:
        """Règle 3 : Part Stage 2 > seuil → WARNING.

        Args:
            result_df: DataFrame résultat ECL.

        Returns:
            Liste d'alertes (0 ou 1).
        """
        stage2_pct = (result_df["stage"] == 2).mean()

        if stage2_pct > self.stage2_warning_pct:
            return [CROAlert(
                severity="WARNING",
                rule_id=3,
                title=f"Concentration Stage 2 élevée ({stage2_pct:.1%})",
                message=(
                    f"{stage2_pct:.1%} du portefeuille est classé en Stage 2 "
                    f"(SICR détecté), dépassant le seuil de vigilance de "
                    f"{self.stage2_warning_pct:.0%}. Cela signale une dégradation "
                    f"significative de la qualité de crédit du book."
                ),
                metric_value=round(stage2_pct, 4),
                threshold=self.stage2_warning_pct,
                action=(
                    "Analyser les flux d'entrée en Stage 2 par vintage et "
                    "segment. Renforcer le suivi des watchlists."
                ),
            )]
        return []

    def _rule_psi_drift(self, psi_value: float) -> List[CROAlert]:
        """Règle 4 : PSI > seuil → ALERT drift modèle.

        Args:
            psi_value: PSI du modèle PD.

        Returns:
            Liste d'alertes (0 ou 1).
        """
        if psi_value > self.psi_drift_threshold:
            severity = "CRITICAL" if psi_value > 0.25 else "ALERT"
            return [CROAlert(
                severity=severity,
                rule_id=4,
                title=f"Drift modèle détecté (PSI = {psi_value:.3f})",
                message=(
                    f"Le Population Stability Index du modèle PD atteint "
                    f"{psi_value:.3f}, dépassant le seuil de {self.psi_drift_threshold:.2f}. "
                    f"La distribution des scores a significativement évolué "
                    f"depuis la calibration. "
                    + (
                        "Le modèle nécessite une recalibration urgente."
                        if psi_value > 0.25
                        else "Un monitoring renforcé est recommandé."
                    )
                ),
                metric_value=round(psi_value, 4),
                threshold=self.psi_drift_threshold,
                action=(
                    "Lancer une recalibration du modèle PD et soumettre "
                    "au comité de validation des modèles."
                    if psi_value > 0.25
                    else "Renforcer la fréquence du backtesting. "
                    "Préparer une analyse de sensibilité."
                ),
            )]
        return []

    def _rule_recommended_actions(
        self,
        result_df: pd.DataFrame,
        existing_alerts: List[CROAlert],
    ) -> List[CROAlert]:
        """Règle 5 : Synthèse des actions recommandées.

        Génère une action de synthèse si plusieurs alertes sont actives.

        Args:
            result_df: DataFrame résultat ECL.
            existing_alerts: Alertes déjà générées par les règles 1-4.

        Returns:
            Liste d'alertes de synthèse.
        """
        n_alerts = len([a for a in existing_alerts if a.severity in ("ALERT", "CRITICAL")])
        n_warnings = len([a for a in existing_alerts if a.severity == "WARNING"])

        if n_alerts == 0 and n_warnings == 0:
            # Tout est nominal
            ecl_total = result_df["ecl_weighted"].sum()
            ead_total = result_df["ead"].sum()
            coverage = ecl_total / ead_total if ead_total > 0 else 0
            return [CROAlert(
                severity="INFO",
                rule_id=5,
                title="Portefeuille dans les normes",
                message=(
                    f"Aucune alerte significative. Coverage ratio à {coverage:.2%}, "
                    f"les indicateurs sont dans les limites de l'appétit au risque."
                ),
                metric_value=0,
                threshold=0,
                action="Maintenir le monitoring standard. Prochain comité : J+30.",
            )]

        if n_alerts >= 2:
            segments_touched = set()
            for a in existing_alerts:
                if a.rule_id == 2:
                    for seg in SEGMENTS:
                        if seg.name in a.title:
                            segments_touched.add(seg.name)

            segments_str = ", ".join(segments_touched) if segments_touched else "multiple"
            return [CROAlert(
                severity="CRITICAL",
                rule_id=5,
                title="Plan d'action immédiat requis",
                message=(
                    f"{n_alerts} alerte(s) et {n_warnings} warning(s) actifs. "
                    f"Segments impactés : {segments_str}. "
                    f"La situation requiert une escalade immédiate."
                ),
                metric_value=float(n_alerts),
                threshold=2.0,
                action=(
                    "1) Escalade Direction des Risques sous 24h. "
                    "2) Gel des octrois sur segments impactés. "
                    "3) Stress test additionnel sur scénario sévère. "
                    "4) Préparer note au régulateur si ECL > budget."
                ),
            )]

        return [CROAlert(
            severity="WARNING",
            rule_id=5,
            title="Vigilance renforcée recommandée",
            message=(
                f"{n_alerts} alerte(s) et {n_warnings} warning(s) actifs. "
                f"La situation est sous contrôle mais nécessite un suivi rapproché."
            ),
            metric_value=float(n_alerts + n_warnings),
            threshold=1.0,
            action=(
                "Augmenter la fréquence de monitoring (hebdomadaire). "
                "Préparer une analyse d'impact pour le prochain comité."
            ),
        )]

    @staticmethod
    def _severity_icon(severity: str) -> str:
        """Retourne l'icône associée à un niveau de sévérité.

        Args:
            severity: Niveau ('INFO', 'WARNING', 'ALERT', 'CRITICAL').

        Returns:
            Icône unicode.
        """
        icons = {
            "INFO": "\u2139\ufe0f",
            "WARNING": "\u26a0\ufe0f",
            "ALERT": "\U0001f534",
            "CRITICAL": "\U0001f6a8",
        }
        return icons.get(severity, "\u2753")
