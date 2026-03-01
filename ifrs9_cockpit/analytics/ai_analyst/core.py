"""CoreMixin — moteur analytique central et utilitaires."""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Any, Dict, List, Optional, Tuple

from ifrs9_cockpit.config import SEGMENTS, SCENARIOS


class CoreMixin:
    """Mixin : core analytics engine et methodes utilitaires."""

    # Baselines macro de reference
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

    # Labels avec article pour la prose francaise
    _MACRO_LABELS_FR: Dict[str, str] = {
        "unemployment_rate": "le taux de chomage",
        "gdp_growth": "la croissance du PIB",
        "interest_rate": "le taux directeur BCE",
        "hpi_growth": "les prix immobiliers",
        "inflation_rate": "l'inflation",
    }

    # ──────────────────────────────────────────────
    # CORE ANALYTICS ENGINE
    # ──────────────────────────────────────────────

    def _compute_core_analytics(
        self,
        result_stressed: pl.DataFrame,
        result_base: pl.DataFrame,
        macro_params: Dict[str, float],
    ) -> Dict[str, Any]:
        """Calcule tous les indicateurs analytiques centraux.

        Produit un dictionnaire complet avec :
            - Metriques ECL globales et variation
            - Analytics par segment (ECL, PD, staging, macro impacts)
            - Ranking des segments par impact
            - Elasticites macro-credit

        Args:
            result_stressed: DataFrame ECL stresse.
            result_base: DataFrame ECL baseline.
            macro_params: Parametres macro courants.

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
            stressed_seg = result_stressed.filter(pl.col("segment") == seg.name)
            base_seg = result_base.filter(pl.col("segment") == seg.name)

            seg_ecl_s = stressed_seg["ecl_weighted"].sum()
            seg_ecl_b = base_seg["ecl_weighted"].sum()
            seg_ead_s = stressed_seg["ead"].sum()
            seg_pd_s = stressed_seg["pd_12m"].mean()
            seg_pd_b = base_seg["pd_12m"].mean()

            seg_ecl_var = (
                (seg_ecl_s - seg_ecl_b) / seg_ecl_b
                if seg_ecl_b > 0 else 0
            )
            ecl_contribution = (
                seg_ecl_s / ecl_stressed if ecl_stressed > 0 else 0
            )
            coverage = seg_ecl_s / seg_ead_s if seg_ead_s > 0 else 0

            # Distribution des stages (stresse et baseline)
            stages_s = stressed_seg["stage"]
            stages_b = base_seg["stage"]

            # Elasticites macro-credit par canal de transmission
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

                # Choc effectif (meme logique que ecl_calculator)
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
                "count": stressed_seg.height,
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
            "n_clients": result_stressed.height,
            "segment_analytics": segment_analytics,
            "ranked_segments": ranked_segments,
            "top_ecl_segment": top_ecl_segment,
        }

    # ──────────────────────────────────────────────
    # UTILITY METHODS
    # ──────────────────────────────────────────────

    def _is_adverse(
        self,
        key: str,
        macro_params: Dict[str, float],
    ) -> bool:
        """Verifie si un parametre macro est en zone adverse."""
        baseline = self._MACRO_BASELINES[key][1]
        current = macro_params.get(key, baseline)
        delta = current - baseline
        if key in self._ADVERSE_UP:
            return delta > 0.1
        return delta < -0.1

    @staticmethod
    def _macro_explanation(key: str, top_seg: str) -> str:
        """Retourne l'explication economique d'un canal de transmission."""
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
