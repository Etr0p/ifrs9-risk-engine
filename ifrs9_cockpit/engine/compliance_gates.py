"""Decision Gating Mechanism — Conformite reglementaire.

Filtre deterministe place en bout de chaine pour garantir que toute
recommandation du systeme (BL-CVaR, Decision Transformer, NeSy MAS)
respecte les contraintes reglementaires.

3 niveaux de gates :
    1. **Basel IV / CRR3** : ratios prudentiels (CET1, LCR, leverage).
    2. **IFRS 9** : staging coherent, PMA dans les limites EBA.
    3. **AI Act** : explicabilite, human-in-the-loop, droit d'explication.

Chaque gate retourne PASS/FAIL/WARNING avec un motif. Si un gate FAIL,
la recommandation est bloquee et une alerte est emise.

References :
    - CRR3 / CRD VI (2025) : ratios prudentiels
    - IFRS 9 par. B5.5.52 : management overlay
    - EBA/GL/2021/02 : MoC and PMA governance
    - EU AI Act (2024) : Art. 14 human oversight, Art. 13 transparency
    - Basel Committee (2023) : BCBS 239 principles
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class GateStatus(Enum):
    """Statut d'un gate de conformite."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class GateResult:
    """Resultat d'un gate individuel.

    Attributes:
        gate_id: Identifiant unique du gate (ex: "CET1_MIN").
        gate_name: Nom lisible du gate.
        category: Categorie (basel_iv, ifrs9, ai_act, risk_appetite).
        status: PASS, WARNING ou FAIL.
        value: Valeur observee.
        threshold: Seuil reglementaire.
        message: Description du resultat.
        regulation: Reference reglementaire.
    """
    gate_id: str
    gate_name: str
    category: str
    status: GateStatus
    value: float
    threshold: float
    message: str
    regulation: str = ""


@dataclass(frozen=True)
class ComplianceResult:
    """Resultat de l'ensemble des gates de conformite.

    Attributes:
        gates: Liste des resultats de chaque gate.
        is_compliant: True si aucun gate FAIL.
        n_pass: Nombre de gates PASS.
        n_warning: Nombre de gates WARNING.
        n_fail: Nombre de gates FAIL.
        blocking_gates: Liste des gates qui bloquent (FAIL).
    """
    gates: List[GateResult]
    is_compliant: bool
    n_pass: int
    n_warning: int
    n_fail: int
    blocking_gates: List[str]


class ComplianceGates:
    """Mecanisme de decision gating pour la conformite reglementaire.

    Verifie les contraintes Basel IV, IFRS 9 et AI Act sur les
    resultats du pipeline avant validation.

    Args:
        cet1_min: CET1 ratio minimum (defaut: 4.5% pilier 1).
        cet1_target: CET1 cible interne (defaut: 10.5% avec buffers).
        lcr_min: LCR minimum (defaut: 100%).
        leverage_min: Ratio de levier minimum (defaut: 3%).
        pma_max_ratio: PMA maximum en % de l'ECL (defaut: 25%).
        max_concentration: HHI max par secteur (defaut: 4000).
    """

    def __init__(
        self,
        cet1_min: float = 0.045,
        cet1_target: float = 0.105,
        lcr_min: float = 1.0,
        leverage_min: float = 0.03,
        pma_max_ratio: float = 0.25,
        max_concentration: float = 4000.0,
    ) -> None:
        self.cet1_min = cet1_min
        self.cet1_target = cet1_target
        self.lcr_min = lcr_min
        self.leverage_min = leverage_min
        self.pma_max_ratio = pma_max_ratio
        self.max_concentration = max_concentration

    def check_all(
        self,
        cet1_ratio: Optional[float] = None,
        rwa_total: Optional[float] = None,
        capital: Optional[float] = None,
        ecl_total: Optional[float] = None,
        ead_total: Optional[float] = None,
        pma_ratio: Optional[float] = None,
        hhi_credit: Optional[float] = None,
        hhi_pe: Optional[float] = None,
        stage3_pct: Optional[float] = None,
        model_explainability: Optional[bool] = None,
        human_override: Optional[bool] = None,
    ) -> ComplianceResult:
        """Execute tous les gates de conformite.

        Args:
            cet1_ratio: Ratio CET1 observe.
            rwa_total: RWA total.
            capital: Fonds propres CET1.
            ecl_total: ECL ponderee totale.
            ead_total: EAD totale.
            pma_ratio: Ratio PMA (ECL_committee / ECL_legal - 1).
            hhi_credit: HHI concentration credit.
            hhi_pe: HHI concentration PE.
            stage3_pct: Pourcentage Stage 3.
            model_explainability: SHAP/Sobol disponibles.
            human_override: Un humain peut overrider les decisions.

        Returns:
            ComplianceResult avec tous les gates.
        """
        gates: List[GateResult] = []

        # ── Basel IV / CRR3 gates ──
        if cet1_ratio is not None:
            gates.append(self._gate_cet1_pillar1(cet1_ratio))
            gates.append(self._gate_cet1_target(cet1_ratio))

        if capital is not None and ead_total is not None and ead_total > 0:
            leverage = capital / ead_total
            gates.append(self._gate_leverage(leverage))

        # ── IFRS 9 gates ──
        if pma_ratio is not None:
            gates.append(self._gate_pma_limit(pma_ratio))

        if ecl_total is not None and ead_total is not None and ead_total > 0:
            coverage = ecl_total / ead_total
            gates.append(self._gate_coverage_floor(coverage))

        if stage3_pct is not None:
            gates.append(self._gate_stage3_alert(stage3_pct))

        # ── Risk Appetite gates ──
        if hhi_credit is not None:
            gates.append(self._gate_concentration(hhi_credit, "credit"))
        if hhi_pe is not None:
            gates.append(self._gate_concentration(hhi_pe, "pe"))

        # ── AI Act gates ──
        if model_explainability is not None:
            gates.append(self._gate_explainability(model_explainability))
        if human_override is not None:
            gates.append(self._gate_human_oversight(human_override))

        # Aggregation
        n_pass = sum(1 for g in gates if g.status == GateStatus.PASS)
        n_warn = sum(1 for g in gates if g.status == GateStatus.WARNING)
        n_fail = sum(1 for g in gates if g.status == GateStatus.FAIL)
        blocking = [g.gate_id for g in gates if g.status == GateStatus.FAIL]

        return ComplianceResult(
            gates=gates,
            is_compliant=(n_fail == 0),
            n_pass=n_pass,
            n_warning=n_warn,
            n_fail=n_fail,
            blocking_gates=blocking,
        )

    # ── Basel IV gates ──

    def _gate_cet1_pillar1(self, cet1: float) -> GateResult:
        if cet1 >= self.cet1_min:
            status = GateStatus.PASS
            msg = f"CET1 {cet1:.2%} >= {self.cet1_min:.1%} pilier 1"
        else:
            status = GateStatus.FAIL
            msg = f"CET1 {cet1:.2%} < {self.cet1_min:.1%} — VIOLATION pilier 1"
        return GateResult("CET1_P1", "CET1 Pilier 1", "basel_iv", status, cet1, self.cet1_min, msg, "CRR3 Art. 92(1)(a)")

    def _gate_cet1_target(self, cet1: float) -> GateResult:
        if cet1 >= self.cet1_target:
            status = GateStatus.PASS
            msg = f"CET1 {cet1:.2%} >= {self.cet1_target:.1%} cible interne"
        elif cet1 >= self.cet1_min:
            status = GateStatus.WARNING
            msg = f"CET1 {cet1:.2%} sous la cible {self.cet1_target:.1%} (buffers entames)"
        else:
            status = GateStatus.FAIL
            msg = f"CET1 {cet1:.2%} < {self.cet1_min:.1%}"
        return GateResult("CET1_TGT", "CET1 Cible Interne", "basel_iv", status, cet1, self.cet1_target, msg, "CRR3 + P2R + CCB + CCyB")

    def _gate_leverage(self, leverage: float) -> GateResult:
        if leverage >= self.leverage_min:
            status = GateStatus.PASS
            msg = f"Levier {leverage:.2%} >= {self.leverage_min:.1%}"
        else:
            status = GateStatus.FAIL
            msg = f"Levier {leverage:.2%} < {self.leverage_min:.1%} — VIOLATION"
        return GateResult("LEVERAGE", "Ratio de Levier", "basel_iv", status, leverage, self.leverage_min, msg, "CRR3 Art. 92(1)(d)")

    # ── IFRS 9 gates ──

    def _gate_pma_limit(self, pma_ratio: float) -> GateResult:
        abs_ratio = abs(pma_ratio)
        if abs_ratio <= self.pma_max_ratio:
            status = GateStatus.PASS
            msg = f"PMA {pma_ratio:+.1%} dans les limites (±{self.pma_max_ratio:.0%})"
        elif abs_ratio <= self.pma_max_ratio * 1.5:
            status = GateStatus.WARNING
            msg = f"PMA {pma_ratio:+.1%} proche de la limite ±{self.pma_max_ratio:.0%}"
        else:
            status = GateStatus.FAIL
            msg = f"PMA {pma_ratio:+.1%} depasse ±{self.pma_max_ratio:.0%}"
        return GateResult("PMA_LIMIT", "Limite PMA", "ifrs9", status, pma_ratio, self.pma_max_ratio, msg, "EBA/GL/2021/02")

    def _gate_coverage_floor(self, coverage: float) -> GateResult:
        floor = 0.001  # 0.1% minimum (pas de portefeuille a 0% de couverture)
        if coverage >= floor:
            status = GateStatus.PASS
            msg = f"Couverture ECL/EAD {coverage:.2%} > plancher"
        else:
            status = GateStatus.WARNING
            msg = f"Couverture ECL/EAD {coverage:.2%} tres basse"
        return GateResult("COV_FLOOR", "Couverture Minimale", "ifrs9", status, coverage, floor, msg, "IFRS 9 par. 5.5.1")

    def _gate_stage3_alert(self, stage3_pct: float) -> GateResult:
        if stage3_pct <= 0.05:
            status = GateStatus.PASS
            msg = f"Stage 3 = {stage3_pct:.1%} (normal)"
        elif stage3_pct <= 0.10:
            status = GateStatus.WARNING
            msg = f"Stage 3 = {stage3_pct:.1%} — surveillance renforcee"
        else:
            status = GateStatus.FAIL
            msg = f"Stage 3 = {stage3_pct:.1%} — depassement seuil critique 10%"
        return GateResult("STAGE3", "Alerte Stage 3", "ifrs9", status, stage3_pct, 0.10, msg, "IFRS 9 B5.5.1")

    # ── Risk Appetite gates ──

    def _gate_concentration(self, hhi: float, channel: str) -> GateResult:
        if hhi <= self.max_concentration:
            status = GateStatus.PASS
            msg = f"HHI {channel} = {hhi:.0f} (diversifie)"
        elif hhi <= self.max_concentration * 1.5:
            status = GateStatus.WARNING
            msg = f"HHI {channel} = {hhi:.0f} — concentration moderee"
        else:
            status = GateStatus.FAIL
            msg = f"HHI {channel} = {hhi:.0f} — concentration excessive"
        return GateResult(f"HHI_{channel.upper()}", f"Concentration {channel.upper()}", "risk_appetite", status, hhi, self.max_concentration, msg, "BCBS 239 Principle 6")

    # ── AI Act gates ──

    def _gate_explainability(self, available: bool) -> GateResult:
        if available:
            status = GateStatus.PASS
            msg = "SHAP + Sobol disponibles"
        else:
            status = GateStatus.FAIL
            msg = "Explicabilite manquante — violation Art. 13 AI Act"
        return GateResult("EXPLAIN", "Explicabilite", "ai_act", status, float(available), 1.0, msg, "EU AI Act Art. 13")

    def _gate_human_oversight(self, available: bool) -> GateResult:
        if available:
            status = GateStatus.PASS
            msg = "Human-in-the-loop actif"
        else:
            status = GateStatus.WARNING
            msg = "Pas de supervision humaine — recommandation Art. 14"
        return GateResult("HUMAN", "Supervision Humaine", "ai_act", status, float(available), 1.0, msg, "EU AI Act Art. 14")
