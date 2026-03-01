"""VirtualCROEngine — Orchestrateur du systeme multi-agents NeSy.

Pipeline complet en 4 etapes :
    1. Agents neuronaux : 4 MLP specialises produisent des masses de croyance
    2. Fusion DS : Jousselme + bBPA + regime-switching
    3. QBAF NeSy : argumentation bipolaire avec contraintes symboliques IFRS 9
    4. PMA Engine : Post-Model Adjustment quantifie en euros

Usage::

    from ifrs9_cockpit.virtual_cro import VirtualCROEngine
    vcro = VirtualCROEngine()
    result = vcro.run(result_credit, result_pe, macro_params,
                      analytics_state=state, rst_distance=3.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ifrs9_cockpit.virtual_cro.agents import (
    MacroAgent, QuantAgent, PEAgent, ContrarianAgent,
    BeliefMass,
)
from ifrs9_cockpit.virtual_cro.fusion import DSFusion, FusionResult
from ifrs9_cockpit.virtual_cro.qbaf import NeSyQBAF, QBAFResult
from ifrs9_cockpit.virtual_cro.pma import PMAEngine, PMAResult


# ──────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────

@dataclass
class VirtualCROResult:
    """Resultat complet du Virtual CRO NeSy MAS.

    Attributes:
        agent_beliefs: Masses de croyance des 4 agents.
        fusion_result: Resultat de la fusion DS.
        qbaf_result: Resultat du NeSy QBAF.
        pma_result: Resultat du PMA.
        recommendation: Recommandation finale.
        recommendation_strength: Force de la recommandation.
        macro_severity: Severite macro calculee.
        summary: Resume executif textuel.
    """
    agent_beliefs: Dict[str, BeliefMass]
    fusion_result: FusionResult
    qbaf_result: QBAFResult
    pma_result: PMAResult
    recommendation: str
    recommendation_strength: float
    macro_severity: float
    summary: str = ""


# ──────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────

class VirtualCROEngine:
    """Orchestrateur NeSy MAS pour le Virtual CRO.

    Enchaine les 4 couches dans l'ordre :
        Agents -> Fusion -> QBAF -> PMA

    Thread-safe : chaque appel a run() cree ses propres instances.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialise les agents neuronaux.

        Args:
            seed: Graine pour reproductibilite.
        """
        self.macro_agent = MacroAgent(seed=seed)
        self.quant_agent = QuantAgent(seed=seed + 1)
        self.pe_agent = PEAgent(seed=seed + 2)
        self.contrarian_agent = ContrarianAgent(seed=seed + 57)

    def run(
        self,
        result_credit,
        result_pe,
        macro_params: Dict[str, float],
        analytics_state=None,
        rst_distance: float = 5.0,
        ecl_legal: Optional[float] = None,
    ) -> VirtualCROResult:
        """Execute le pipeline complet du Virtual CRO.

        Args:
            result_credit: DataFrame resultat ECLCalculator.
            result_pe: DataFrame resultat PECalculator.
            macro_params: Dict des 5 variables macro.
            analytics_state: AnalyticsState (optionnel, pour enrichir).
            rst_distance: Distance RST en sigma.
            ecl_legal: ECL reglementaire total. Si None, calcule depuis result_credit.

        Returns:
            VirtualCROResult complet.
        """
        # ── Severite macro ──
        macro_severity = self._compute_macro_severity(macro_params)

        # ── Couche 1 : Agents neuronaux ──
        belief_macro = self.macro_agent.predict(macro_params)
        belief_quant = self.quant_agent.predict(result_credit)
        belief_pe = self.pe_agent.predict(result_pe)
        belief_contrarian = self.contrarian_agent.predict(
            belief_macro, belief_quant, belief_pe,
        )

        agent_beliefs = {
            "macro": belief_macro,
            "quant": belief_quant,
            "pe": belief_pe,
            "contrarian": belief_contrarian,
        }

        # ── Couche 2 : Fusion DS ──
        # On fusionne les 3 agents principaux (pas le contrarian qui est meta)
        fusion = DSFusion(macro_severity=macro_severity)
        # Projeter sur un cadre commun avant fusion
        unified_beliefs = self._unify_frames(
            [belief_macro, belief_quant, belief_pe],
        )
        fusion_result = fusion.fuse(unified_beliefs)

        # ── Couche 3 : QBAF NeSy ──
        stages = result_credit["stage"].to_numpy()
        stage3_pct = float((stages == 3).mean())
        sicr_pct = float(((stages == 2) | (stages == 3)).mean())

        ecl_total = float(result_credit["ecl_weighted"].sum())
        ead_total = float(result_credit["ead"].sum())
        ecl_ead_ratio = ecl_total / max(ead_total, 1.0)

        contrarian_wrong_prob = belief_contrarian.masses.get("consensus_wrong", 0.0)

        qbaf = NeSyQBAF()
        qbaf.build_graph(
            fusion_result=fusion_result,
            contrarian_wrong_prob=contrarian_wrong_prob,
            stage3_pct=stage3_pct,
            sicr_pct=sicr_pct,
            rst_distance=rst_distance,
            ecl_ead_ratio=ecl_ead_ratio,
        )
        qbaf_result = qbaf.evaluate()

        # ── Couche 4 : PMA ──
        if ecl_legal is None:
            ecl_legal = ecl_total

        pma_engine = PMAEngine()
        pma_result = pma_engine.compute(
            ecl_legal=ecl_legal,
            qbaf_result=qbaf_result,
            fusion_belief=fusion_result.belief,
            fusion_conflict=fusion_result.conflict_level,
            macro_severity=macro_severity,
        )

        # ── Summary ──
        summary = self._build_summary(
            qbaf_result=qbaf_result,
            pma_result=pma_result,
            fusion_result=fusion_result,
            belief_contrarian=belief_contrarian,
            macro_severity=macro_severity,
        )

        return VirtualCROResult(
            agent_beliefs=agent_beliefs,
            fusion_result=fusion_result,
            qbaf_result=qbaf_result,
            pma_result=pma_result,
            recommendation=qbaf_result.recommendation,
            recommendation_strength=qbaf_result.recommendation_strength,
            macro_severity=macro_severity,
            summary=summary,
        )

    @staticmethod
    def _compute_macro_severity(macro_params: Dict[str, float]) -> float:
        """Calcule un indice de severite macro normalise [0, 1].

        Combine les 5 variables macro en un score unique :
        - Chomage eleve = severe
        - PIB negatif = severe
        - Taux eleves + inflation = severe (stagflation)
        - HPI negatif = severe
        """
        from ifrs9_cockpit.config import SCENARIO_BASE

        base = SCENARIO_BASE
        scores = []

        # Chomage : elevation = adverse
        unemp = macro_params.get("unemployment_rate", base.unemployment_rate)
        scores.append(np.clip((unemp - base.unemployment_rate) / 5.0, -0.5, 1.0))

        # PIB : baisse = adverse
        gdp = macro_params.get("gdp_growth", base.gdp_growth)
        scores.append(np.clip((base.gdp_growth - gdp) / 5.0, -0.5, 1.0))

        # Taux : elevation combinee a l'inflation = adverse (stagflation)
        ir = macro_params.get("interest_rate", base.interest_rate)
        infl = macro_params.get("inflation_rate", base.inflation_rate)
        ir_stress = np.clip((ir - base.interest_rate) / 3.0, -0.3, 0.5)
        infl_stress = np.clip((infl - base.inflation_rate) / 4.0, -0.3, 0.5)
        scores.append(ir_stress + infl_stress)

        # HPI : baisse = adverse
        hpi = macro_params.get("hpi_growth", base.hpi_growth)
        scores.append(np.clip((base.hpi_growth - hpi) / 10.0, -0.3, 1.0))

        severity = float(np.clip(np.mean(scores), 0.0, 1.0))
        return severity

    @staticmethod
    def _unify_frames(beliefs: List[BeliefMass]) -> List[BeliefMass]:
        """Projette les masses sur un cadre commun unifie.

        Cadre unifie : {favorable, neutre, defavorable, uncertainty}
        Mapping :
            expansion, sain, amelioration -> favorable
            normal, stable -> neutre
            recession, crise, degradation, distress, watchlist -> defavorable
        """
        _FAVORABLE = {"expansion", "sain", "amelioration"}
        _NEUTRE = {"normal", "stable"}
        _DEFAVORABLE = {"recession", "crise", "degradation", "distress", "watchlist"}

        unified_frame = ["favorable", "neutre", "defavorable"]
        unified_beliefs = []

        for belief in beliefs:
            masses = {"favorable": 0.0, "neutre": 0.0, "defavorable": 0.0}

            for hyp, mass in belief.masses.items():
                if hyp == "uncertainty":
                    continue
                elif hyp in _FAVORABLE:
                    masses["favorable"] += mass
                elif hyp in _NEUTRE:
                    masses["neutre"] += mass
                elif hyp in _DEFAVORABLE:
                    masses["defavorable"] += mass
                else:
                    # Unknown hypothesis — map to neutre
                    masses["neutre"] += mass

            masses["uncertainty"] = belief.masses.get("uncertainty", 0.0)

            unified_beliefs.append(BeliefMass(
                frame=unified_frame,
                masses=masses,
                agent_name=belief.agent_name,
                confidence=belief.confidence,
            ))

        return unified_beliefs

    @staticmethod
    def _build_summary(
        qbaf_result: QBAFResult,
        pma_result: PMAResult,
        fusion_result: FusionResult,
        belief_contrarian: BeliefMass,
        macro_severity: float,
    ) -> str:
        """Construit le resume executif du Virtual CRO."""
        lines = []

        # 1. Recommandation
        rec = qbaf_result.recommendation
        strength = qbaf_result.recommendation_strength
        lines.append(
            f"Recommandation du comite : {rec.upper()} "
            f"(force : {strength:.2f})"
        )

        # 2. Consensus
        rule = fusion_result.rule_used
        conflict = fusion_result.conflict_level
        rule_labels = {
            "dempster": "Dempster (consensus fort)",
            "pcr": "PCR6 (conflit modere)",
            "yager": "Yager fail-safe (conflit fort)",
        }
        lines.append(
            f"Fusion : {rule_labels.get(rule, rule)} | "
            f"Conflit : {conflict:.2f}"
        )

        # 3. Contrarian
        p_wrong = belief_contrarian.masses.get("consensus_wrong", 0)
        if p_wrong > 0.3:
            lines.append(
                f"Alerte Contrarian : P(consensus errone) = {p_wrong:.1%}"
            )

        # 4. Contraintes IFRS 9
        if qbaf_result.symbolic_overrides:
            lines.append(
                f"Contraintes IFRS 9 actives : "
                f"{', '.join(qbaf_result.symbolic_overrides)}"
            )

        # 5. PMA
        if pma_result.is_valid and abs(pma_result.pma_ratio) > 0.005:
            lines.append(
                f"PMA : {pma_result.pma_amount:+,.0f} EUR "
                f"({pma_result.pma_ratio:+.1%}) | "
                f"Confiance : {pma_result.confidence:.2f}"
            )
        elif not pma_result.is_valid:
            lines.append(
                f"PMA rejete : {'; '.join(pma_result.rejection_reasons)}"
            )

        # 6. Severite macro
        if macro_severity > 0.5:
            lines.append(f"Environnement macro severe ({macro_severity:.2f})")

        return " | ".join(lines)
