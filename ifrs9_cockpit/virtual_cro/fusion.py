"""Fusion Dempster-Shafer avec distance de Jousselme et regime-switching.

Combine les masses de croyance des 4 agents neuronaux en utilisant :
    1. Distance de Jousselme (2001) pour mesurer le conflit inter-agents
    2. bBPA discounting (confidence-weighted) pour attenuer les agents peu fiables
    3. 3 regles de combinaison selon le niveau de conflit :
       - Faible conflit (d_J < tau) : regle de Dempster classique
       - Conflit moyen : PCR6 (Proportional Conflict Redistribution)
       - Conflit fort (d_J > 0.8) : Yager fail-safe (masse vers uncertainty)
    4. Seuil dynamique tau = f(macro_severity) + conflit residuel

References :
    - Jousselme et al. (2001) : distance evidence
    - Dempster (1967), Shafer (1976) : theorie des croyances
    - Smarandache & Dezert (2009) : PCR rules
    - Yager (1987) : regle fail-safe
    - Zadeh (1979) : paradoxe de Zadeh (motivation PCR/Yager)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ifrs9_cockpit.virtual_cro.agents import BeliefMass


# ──────────────────────────────────────────────
# Fusion result
# ──────────────────────────────────────────────

class FusionResult:
    """Resultat de la fusion DS multi-agents.

    Attributes:
        fused_masses: Masses fusionnees par hypothese.
        conflict_level: Niveau de conflit global (0 = accord, 1 = Zadeh).
        rule_used: Regle de combinaison utilisee (dempster/pcr/yager).
        pairwise_distances: Matrice de Jousselme inter-agents.
        dynamic_threshold: Seuil tau dynamique utilise.
        agent_beliefs: Masses individuelles des agents (pre-fusion).
    """

    def __init__(
        self,
        fused_masses: Dict[str, float],
        conflict_level: float,
        rule_used: str,
        pairwise_distances: Dict[str, float],
        dynamic_threshold: float,
        agent_beliefs: List[BeliefMass],
    ) -> None:
        self.fused_masses = fused_masses
        self.conflict_level = conflict_level
        self.rule_used = rule_used
        self.pairwise_distances = pairwise_distances
        self.dynamic_threshold = dynamic_threshold
        self.agent_beliefs = agent_beliefs

    @property
    def dominant_hypothesis(self) -> str:
        """Hypothese dominante (masse max, hors uncertainty)."""
        focal = {k: v for k, v in self.fused_masses.items() if k != "uncertainty"}
        if not focal:
            return "uncertainty"
        return max(focal, key=focal.get)

    @property
    def belief(self) -> float:
        """Croyance totale (1 - uncertainty)."""
        return 1.0 - self.fused_masses.get("uncertainty", 0.0)


# ──────────────────────────────────────────────
# DS Fusion engine
# ──────────────────────────────────────────────

class DSFusion:
    """Moteur de fusion Dempster-Shafer avec regime-switching.

    Combine N masses de croyance en gerant le conflit via
    la distance de Jousselme et un seuil dynamique tau.
    """

    # Seuils de conflit pour le regime-switching
    _TAU_BASE = 0.45       # seuil de base (regime normal)
    _TAU_CRISIS_BOOST = 0.15  # reduction en crise (plus tolerant)
    _YAGER_THRESHOLD = 0.80   # au-dela, Yager fail-safe

    def __init__(self, macro_severity: float = 0.0) -> None:
        """Initialise le moteur de fusion.

        Args:
            macro_severity: Severite macro normalisee [0, 1].
                0 = conditions normales, 1 = crise severe.
                Utilise pour le seuil dynamique tau.
        """
        self.macro_severity = np.clip(macro_severity, 0.0, 1.0)

    @property
    def dynamic_threshold(self) -> float:
        """Seuil dynamique tau = f(macro_severity).

        En crise, le seuil baisse (plus de tolerance au conflit)
        car un desaccord inter-agents est attendu.
        """
        return self._TAU_BASE - self._TAU_CRISIS_BOOST * self.macro_severity

    def fuse(self, beliefs: List[BeliefMass]) -> FusionResult:
        """Fusionne N masses de croyance avec regime-switching.

        Args:
            beliefs: Liste des masses de croyance des agents.

        Returns:
            FusionResult avec masses fusionnees et diagnostics.
        """
        if not beliefs:
            return FusionResult(
                fused_masses={"uncertainty": 1.0},
                conflict_level=0.0,
                rule_used="none",
                pairwise_distances={},
                dynamic_threshold=self.dynamic_threshold,
                agent_beliefs=[],
            )

        if len(beliefs) == 1:
            return FusionResult(
                fused_masses=dict(beliefs[0].masses),
                conflict_level=0.0,
                rule_used="single",
                pairwise_distances={},
                dynamic_threshold=self.dynamic_threshold,
                agent_beliefs=beliefs,
            )

        # 1. bBPA discounting (attenuer les agents peu confiants)
        discounted = [self._bbpa_discount(b) for b in beliefs]

        # 2. Distances de Jousselme pairwise
        pairwise = self._pairwise_jousselme(discounted)
        max_conflict = max(pairwise.values()) if pairwise else 0.0

        # 3. Regime-switching selon le conflit
        tau = self.dynamic_threshold
        if max_conflict >= self._YAGER_THRESHOLD:
            rule = "yager"
            fused = self._yager_combine(discounted)
        elif max_conflict >= tau:
            rule = "pcr"
            fused = self._pcr_combine(discounted)
        else:
            rule = "dempster"
            fused = self._dempster_combine(discounted)

        return FusionResult(
            fused_masses=fused,
            conflict_level=max_conflict,
            rule_used=rule,
            pairwise_distances=pairwise,
            dynamic_threshold=tau,
            agent_beliefs=beliefs,
        )

    # ──────────────────────────────────────────
    # bBPA discounting
    # ──────────────────────────────────────────

    @staticmethod
    def _bbpa_discount(belief: BeliefMass) -> BeliefMass:
        """Applique le discounting bBPA selon la confiance de l'agent.

        m_discounted(A) = alpha * m(A) pour A != Omega
        m_discounted(Omega) = 1 - alpha * (1 - m(Omega))
        ou alpha = confidence de l'agent.
        """
        alpha = belief.confidence
        discounted_masses = {}
        mass_sum = 0.0

        for hyp, mass in belief.masses.items():
            if hyp == "uncertainty":
                continue
            discounted_masses[hyp] = mass * alpha
            mass_sum += discounted_masses[hyp]

        discounted_masses["uncertainty"] = 1.0 - mass_sum

        return BeliefMass(
            frame=belief.frame,
            masses=discounted_masses,
            agent_name=belief.agent_name,
            confidence=alpha,
        )

    # ──────────────────────────────────────────
    # Distance de Jousselme
    # ──────────────────────────────────────────

    @staticmethod
    def jousselme_distance(m1: Dict[str, float], m2: Dict[str, float]) -> float:
        """Distance de Jousselme entre deux masses de croyance.

        d_J(m1, m2) = sqrt(0.5 * (m1 - m2)^T * D * (m1 - m2))
        ou D est la matrice de Jaccard : D_ij = |Ai inter Aj| / |Ai union Aj|.

        Pour des masses focales singletons, D est l'identite sauf D_ii = 1.
        """
        # Union des hypotheses (singletons only)
        all_hyp = sorted(set(list(m1.keys()) + list(m2.keys())))

        # Vecteurs de masses
        v1 = np.array([m1.get(h, 0.0) for h in all_hyp])
        v2 = np.array([m2.get(h, 0.0) for h in all_hyp])
        diff = v1 - v2

        # Pour des singletons, D = I (Jaccard inter singleton = {A} si meme, {} sinon)
        # d_J = sqrt(0.5 * ||diff||^2) = ||diff|| / sqrt(2)
        return float(np.sqrt(0.5 * np.dot(diff, diff)))

    def _pairwise_jousselme(self, beliefs: List[BeliefMass]) -> Dict[str, float]:
        """Calcule les distances de Jousselme pairwise."""
        distances = {}
        for i in range(len(beliefs)):
            for j in range(i + 1, len(beliefs)):
                key = f"{beliefs[i].agent_name}-{beliefs[j].agent_name}"
                distances[key] = self.jousselme_distance(
                    beliefs[i].masses, beliefs[j].masses,
                )
        return distances

    # ──────────────────────────────────────────
    # Regles de combinaison
    # ──────────────────────────────────────────

    def _dempster_combine(self, beliefs: List[BeliefMass]) -> Dict[str, float]:
        """Regle de Dempster classique (normalisation du conflit)."""
        result = dict(beliefs[0].masses)

        for i in range(1, len(beliefs)):
            result = self._dempster_pair(result, beliefs[i].masses)

        return result

    @staticmethod
    def _dempster_pair(m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
        """Combine deux masses par la regle de Dempster."""
        all_hyp = sorted(set(list(m1.keys()) + list(m2.keys())))
        combined = {h: 0.0 for h in all_hyp}
        conflict = 0.0

        for h1, v1 in m1.items():
            for h2, v2 in m2.items():
                product = v1 * v2
                if h1 == "uncertainty":
                    combined[h2] = combined.get(h2, 0.0) + product
                elif h2 == "uncertainty":
                    combined[h1] = combined.get(h1, 0.0) + product
                elif h1 == h2:
                    combined[h1] += product
                else:
                    conflict += product

        # Normalisation
        norm = 1.0 - conflict
        if norm < 1e-10:
            # Conflit total — fallback Yager
            return {"uncertainty": 1.0}

        result = {}
        for h, v in combined.items():
            result[h] = v / norm

        return result

    def _pcr_combine(self, beliefs: List[BeliefMass]) -> Dict[str, float]:
        """Regle PCR6 : redistribution proportionnelle du conflit.

        Le conflit est redistribue proportionnellement aux masses
        individuelles (plus equitable que Dempster en cas de conflit moyen).
        """
        result = dict(beliefs[0].masses)

        for i in range(1, len(beliefs)):
            result = self._pcr_pair(result, beliefs[i].masses)

        return result

    @staticmethod
    def _pcr_pair(m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
        """Combine deux masses par PCR6."""
        all_hyp = sorted(set(list(m1.keys()) + list(m2.keys())))
        combined = {h: 0.0 for h in all_hyp}
        conflict_parts = []

        for h1, v1 in m1.items():
            for h2, v2 in m2.items():
                product = v1 * v2
                if h1 == "uncertainty":
                    combined[h2] = combined.get(h2, 0.0) + product
                elif h2 == "uncertainty":
                    combined[h1] = combined.get(h1, 0.0) + product
                elif h1 == h2:
                    combined[h1] += product
                else:
                    # Conflit — PCR6 redistribue proportionnellement
                    conflict_parts.append((h1, v1, h2, v2, product))

        # Redistribution proportionnelle du conflit
        for h1, v1, h2, v2, prod in conflict_parts:
            total_mass = v1 + v2
            if total_mass > 1e-10:
                combined[h1] = combined.get(h1, 0.0) + prod * v1 / total_mass
                combined[h2] = combined.get(h2, 0.0) + prod * v2 / total_mass
            else:
                # Si les deux masses sont nulles, redistribuer vers uncertainty
                combined["uncertainty"] = combined.get("uncertainty", 0.0) + prod

        # Normalisation (somme = 1)
        total = sum(combined.values())
        if total > 1e-10:
            combined = {k: v / total for k, v in combined.items()}

        return combined

    @staticmethod
    def _yager_combine(beliefs: List[BeliefMass]) -> Dict[str, float]:
        """Regle de Yager : conflit -> uncertainty (fail-safe conservateur).

        En cas de conflit fort, tout le conflit est transfere vers
        l'ignorance totale Omega plutot que normalise.
        """
        result = dict(beliefs[0].masses)

        for i in range(1, len(beliefs)):
            all_hyp = sorted(set(list(result.keys()) + list(beliefs[i].masses.keys())))
            combined = {h: 0.0 for h in all_hyp}
            conflict = 0.0

            for h1, v1 in result.items():
                for h2, v2 in beliefs[i].masses.items():
                    product = v1 * v2
                    if h1 == "uncertainty":
                        combined[h2] = combined.get(h2, 0.0) + product
                    elif h2 == "uncertainty":
                        combined[h1] = combined.get(h1, 0.0) + product
                    elif h1 == h2:
                        combined[h1] += product
                    else:
                        conflict += product

            # Yager : conflit -> uncertainty
            combined["uncertainty"] = combined.get("uncertainty", 0.0) + conflict
            result = combined

        return result
