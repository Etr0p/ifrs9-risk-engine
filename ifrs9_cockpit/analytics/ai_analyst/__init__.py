"""Module Analyste CRO Local — Intelligence d'analyse embarquee IFRS 9.

Moteur d'analyse statistique autonome qui correle automatiquement
l'ensemble des metriques du portefeuille (ECL, staging, macro, SHAP,
backtesting, HHI, performance modeles) pour produire un rapport CRO
structure de niveau Direction des Risques.

Contrairement au VirtualCRO (5 regles de seuil), ce module :
    - Decompose mathematiquement la variation d'ECL (attribution)
    - Calcule les elasticites macro-credit par segment
    - Identifie automatiquement les correlations et causalites
    - Analyse les flux de migration et la vulnerabilite par segment
    - Evalue la gouvernance des modeles (performance, drift, stabilite)
    - Produit un rapport narratif correle avec recommandations priorisees

Aucune dependance externe (pas d'API). Tout le raisonnement est
embarque dans le moteur d'analyse statistique.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from ifrs9_cockpit.analytics.ai_analyst.core import CoreMixin
from ifrs9_cockpit.analytics.ai_analyst.sections import SectionsMixin
from ifrs9_cockpit.analytics.ai_analyst.prose import ProseMixin


class LocalCROAnalyst(CoreMixin, SectionsMixin, ProseMixin):
    """Analyste CRO embarque — analyse correlee sans dependance externe.

    Remplace le systeme rules-based a 5 regles par un moteur d'analyse
    statistique complet qui :
        - Decompose mathematiquement la variation d'ECL
        - Calcule les elasticites macro-credit par segment
        - Identifie automatiquement les correlations et les causalites
        - Produit un rapport narratif de niveau Direction des Risques

    Example:
        >>> analyst = LocalCROAnalyst()
        >>> report = analyst.generate_full_report(
        ...     result_stressed, result_base, model_comparison,
        ...     macro_params, psi_value, selected_model,
        ... )
        >>> print(report)
    """

    def generate_full_report(
        self,
        result_stressed: pl.DataFrame,
        result_base: pl.DataFrame,
        model_comparison: pl.DataFrame,
        macro_params: Dict[str, float],
        psi_value: float,
        selected_model: str = "LR_WoE",
        shap_top_features: Optional[List[Tuple[str, float]]] = None,
        hhi_segment: Optional[float] = None,
        hhi_loan: Optional[float] = None,
        backtesting_df: Optional[pl.DataFrame] = None,
        waterfall_df: Optional[pl.DataFrame] = None,
        transition_matrix: Optional[pl.DataFrame] = None,
    ) -> str:
        """Genere le rapport CRO complet avec analyse correlee.

        Agrege toutes les metriques, calcule les correlations,
        et produit un rapport narratif structure en 8 sections.

        Args:
            result_stressed: DataFrame ECL stresse (avec sliders).
            result_base: DataFrame ECL baseline (sans stress).
            model_comparison: Table de comparaison des modeles PD.
            macro_params: Parametres macro courants (5 variables).
            psi_value: PSI du modele PD selectionne.
            selected_model: Nom du modele PD selectionne.
            shap_top_features: Top features SHAP [(name, mean_abs_shap)].
            hhi_segment: HHI par segment.
            hhi_loan: HHI par type de pret.
            backtesting_df: Metriques de backtesting walk-forward.
            waterfall_df: Decomposition waterfall ECL.
            transition_matrix: Matrice de transition des stages.

        Returns:
            Rapport complet en Markdown structure.
        """
        # Phase 1 : Calcul de tous les indicateurs analytiques
        analytics = self._compute_core_analytics(
            result_stressed, result_base, macro_params,
        )

        # Phase 2 : Generation du rapport section par section
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


__all__ = ["LocalCROAnalyst"]
