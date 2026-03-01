"""PMA Engine — Post-Model Adjustment automatise.

Quantifie l'ecart entre l'ECL calcule par le moteur reglementaire (staging IFRS 9)
et l'ECL recommande par le comite d'agents (fusion + QBAF), puis genere
un dossier de justification conforme aux exigences de gouvernance.

Architecture :
    ECL_legal : ECL calcule par le moteur classique (staging hard)
    ECL_committee : ECL ajuste selon la recommandation du comite d'agents
    PMA = ECL_committee - ECL_legal (en EUR)
    Si PMA > 0 : le comite recommande de provisionner davantage
    Si PMA < 0 : le comite juge la provision excessive (rare en pratique)

Conditions de validite :
    - Confiance du comite > seuil minimum
    - Pas de veto symbolique (IFRS 9 hard constraints)
    - PMA dans les limites de tolerance (+/- 25% de l'ECL_legal)

References :
    - PMADS (arXiv 2510.17108) : post-model adjustments automatises
    - EBA/GL/2021/02 : guidelines on MoC and PMA governance
    - IFRS 9 par. B5.5.52 : management overlay allowance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ifrs9_cockpit.virtual_cro.qbaf import QBAFResult, RECOMMENDATIONS


# ──────────────────────────────────────────────
# PMA Result
# ──────────────────────────────────────────────

@dataclass
class PMAResult:
    """Resultat du calcul PMA.

    Attributes:
        ecl_legal: ECL reglementaire (moteur classique).
        ecl_committee: ECL recommande par le comite d'agents.
        pma_amount: PMA en EUR (ecl_committee - ecl_legal).
        pma_ratio: PMA / ECL_legal en pourcentage.
        direction: Direction du PMA ('increase', 'decrease', 'neutral').
        is_valid: True si le PMA passe les conditions de validite.
        justification: Dossier de justification structure.
        rejection_reasons: Raisons de rejet si is_valid = False.
        confidence: Confiance du comite dans le PMA.
    """
    ecl_legal: float
    ecl_committee: float
    pma_amount: float
    pma_ratio: float
    direction: str
    is_valid: bool
    justification: Dict[str, str]
    rejection_reasons: List[str]
    confidence: float


# ──────────────────────────────────────────────
# PMA Engine
# ──────────────────────────────────────────────

# Facteurs d'ajustement ECL par recommandation
_ADJUSTMENT_FACTORS = {
    "maintenir": 1.00,     # Pas d'ajustement
    "surveiller": 1.05,    # +5% provision prudentielle
    "reduire": 1.15,       # +15% provision renforcee
    "escalader": 1.25,     # +25% provision de crise
}


class PMAEngine:
    """Moteur PMA : calcule le Post-Model Adjustment et genere le dossier.

    Le PMA traduit la recommandation qualitative du comite d'agents
    en un ajustement quantitatif de l'ECL en euros.
    """

    # Limites de tolerance (EBA GL 2021/02)
    _MAX_PMA_RATIO = 0.25       # +/- 25% de l'ECL_legal
    _MIN_CONFIDENCE = 0.40      # Confiance minimale du comite
    _MIN_BELIEF = 0.30          # Croyance minimale de la fusion

    def compute(
        self,
        ecl_legal: float,
        qbaf_result: QBAFResult,
        fusion_belief: float,
        fusion_conflict: float,
        macro_severity: float = 0.0,
    ) -> PMAResult:
        """Calcule le PMA et genere le dossier de justification.

        Args:
            ecl_legal: ECL reglementaire (EUR).
            qbaf_result: Resultat du NeSy QBAF.
            fusion_belief: Croyance totale de la fusion (1 - uncertainty).
            fusion_conflict: Niveau de conflit inter-agents.
            macro_severity: Severite macro [0, 1].

        Returns:
            PMAResult complet avec justification.
        """
        recommendation = qbaf_result.recommendation
        rec_strength = qbaf_result.recommendation_strength

        # Facteur d'ajustement de base
        base_factor = _ADJUSTMENT_FACTORS.get(recommendation, 1.0)

        # Modulation par la force de la recommandation
        # Si rec_strength est faible, le facteur est attenue vers 1.0
        modulated_factor = 1.0 + (base_factor - 1.0) * np.clip(rec_strength, 0, 1)

        # Ajustement pour severite macro (en crise, on provisionne plus)
        if recommendation in ("reduire", "escalader"):
            macro_boost = 1.0 + 0.05 * macro_severity  # jusqu'a +5% supplementaire
            modulated_factor *= macro_boost

        # ECL committee
        ecl_committee = ecl_legal * modulated_factor

        # PMA
        pma_amount = ecl_committee - ecl_legal
        pma_ratio = pma_amount / max(ecl_legal, 1.0)

        # Direction
        if abs(pma_ratio) < 0.005:
            direction = "neutral"
        elif pma_ratio > 0:
            direction = "increase"
        else:
            direction = "decrease"

        # Confiance du comite
        confidence = float(np.clip(
            fusion_belief * (1 - fusion_conflict * 0.5) * rec_strength,
            0, 1,
        ))

        # Conditions de validite
        rejection_reasons = []
        if confidence < self._MIN_CONFIDENCE:
            rejection_reasons.append(
                f"Confiance insuffisante ({confidence:.2f} < {self._MIN_CONFIDENCE})"
            )
        if fusion_belief < self._MIN_BELIEF:
            rejection_reasons.append(
                f"Croyance fusion trop faible ({fusion_belief:.2f} < {self._MIN_BELIEF})"
            )
        if abs(pma_ratio) > self._MAX_PMA_RATIO:
            rejection_reasons.append(
                f"PMA hors tolerance ({pma_ratio:+.1%} vs +/- {self._MAX_PMA_RATIO:.0%})"
            )
        if qbaf_result.symbolic_overrides and recommendation not in ("escalader", "reduire"):
            rejection_reasons.append(
                f"Incoherence : veto symbolique actif mais recommandation '{recommendation}'"
            )

        is_valid = len(rejection_reasons) == 0

        # Dossier de justification
        justification = self._build_justification(
            recommendation=recommendation,
            rec_strength=rec_strength,
            pma_amount=pma_amount,
            pma_ratio=pma_ratio,
            ecl_legal=ecl_legal,
            ecl_committee=ecl_committee,
            fusion_belief=fusion_belief,
            fusion_conflict=fusion_conflict,
            symbolic_overrides=qbaf_result.symbolic_overrides,
            falsification_matrix=qbaf_result.falsification_matrix,
            macro_severity=macro_severity,
        )

        return PMAResult(
            ecl_legal=ecl_legal,
            ecl_committee=ecl_committee,
            pma_amount=pma_amount,
            pma_ratio=pma_ratio,
            direction=direction,
            is_valid=is_valid,
            justification=justification,
            rejection_reasons=rejection_reasons,
            confidence=confidence,
        )

    @staticmethod
    def _build_justification(
        recommendation: str,
        rec_strength: float,
        pma_amount: float,
        pma_ratio: float,
        ecl_legal: float,
        ecl_committee: float,
        fusion_belief: float,
        fusion_conflict: float,
        symbolic_overrides: List[str],
        falsification_matrix: Dict[str, Dict[str, float]],
        macro_severity: float,
    ) -> Dict[str, str]:
        """Genere le dossier de justification du PMA."""

        # Section 1 : Resume executif
        if pma_amount > 0:
            direction_text = "a la hausse"
            reason = "provisions supplementaires"
        elif pma_amount < 0:
            direction_text = "a la baisse"
            reason = "provisions excessives"
        else:
            direction_text = "nul"
            reason = "aucun ajustement"

        resume = (
            f"Le comite d'agents recommande un ajustement {direction_text} "
            f"de {abs(pma_amount):,.0f} EUR ({pma_ratio:+.1%} de l'ECL legal). "
            f"Recommandation : '{recommendation}' (force : {rec_strength:.2f}). "
            f"Motif : {reason}."
        )

        # Section 2 : Analyse du consensus
        if fusion_conflict < 0.3:
            consensus_text = "Fort consensus inter-agents"
        elif fusion_conflict < 0.6:
            consensus_text = "Consensus modere avec divergences"
        else:
            consensus_text = "Divergence significative entre agents"
        consensus = (
            f"{consensus_text}. "
            f"Croyance totale : {fusion_belief:.2f}. "
            f"Conflit residuel : {fusion_conflict:.2f}."
        )

        # Section 3 : Contraintes reglementaires
        if symbolic_overrides:
            overrides_text = ", ".join(symbolic_overrides)
            reglementaire = (
                f"Contraintes IFRS 9 actives : {overrides_text}. "
                f"Ces contraintes sont non-negociables et priment sur "
                f"les recommandations des agents neuronaux."
            )
        else:
            reglementaire = (
                "Aucune contrainte reglementaire stricte activee. "
                "Le PMA repose entierement sur l'analyse multi-agents."
            )

        # Section 4 : Sensibilite
        sensitivity_lines = []
        for arg, deltas in falsification_matrix.items():
            max_delta = max(abs(v) for v in deltas.values()) if deltas else 0
            if max_delta > 0.1:
                sensitivity_lines.append(
                    f"  - {arg} : impact max = {max_delta:.2f}"
                )
        sensitivity = (
            "Arguments les plus influents :\n" + "\n".join(sensitivity_lines)
            if sensitivity_lines
            else "Aucun argument individuellement determinant."
        )

        # Section 5 : Contexte macro
        if macro_severity > 0.5:
            macro_text = (
                f"Environnement macro severe (indice {macro_severity:.2f}). "
                f"L'ajustement inclut un supplement de provision de crise."
            )
        else:
            macro_text = (
                f"Environnement macro modere (indice {macro_severity:.2f}). "
                f"Pas de supplement de crise applique."
            )

        return {
            "resume": resume,
            "consensus": consensus,
            "reglementaire": reglementaire,
            "sensibilite": sensitivity,
            "contexte_macro": macro_text,
        }
