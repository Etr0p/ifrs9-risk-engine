"""NeSy QBAF — Quantitative Bipolar Argumentation Framework Neuro-Symbolique.

Combine des arguments symboliques (regles IFRS 9, non-differentiables, force=inf)
avec des arguments neuronaux (differentiables, apprenables) dans un graphe
d'argumentation bipolaire quantitatif.

Architecture NeSy :
    - Arguments symboliques : IFRS 9 staging rules (Stage 3 si PD > 30% ou DPD > 90)
      → force = +inf, NON modifiables par l'apprentissage
    - Arguments neuronaux : sorties des agents MLP
      → force = f(confidence, belief), differentiables

Le modele QE (Quadratic Energy) resout les cycles dans le QBAF par
iteration de point fixe avec damping (alpha = 0.85).

CE-QArg : Counterfactual Explanations via perturbation des entrees macro
et observation de la variation des forces des arguments.

References :
    - Dung (1995) : Abstract Argumentation Frameworks
    - Baroni et al. (2019) : QBAF semantics
    - Potyka (2018, 2021) : QE model for cyclic QBAF
    - Albini et al. (2021) : CE-QArg counterfactual explanations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ifrs9_cockpit.virtual_cro.fusion import FusionResult


# ──────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────

@dataclass
class Argument:
    """Argument dans le QBAF.

    Attributes:
        name: Identifiant unique.
        base_strength: Force de base tau(a) dans [-1, 1].
        is_symbolic: True = argument IFRS 9 (non-differentiable, force=inf).
        description: Description lisible.
    """
    name: str
    base_strength: float
    is_symbolic: bool = False
    description: str = ""


@dataclass
class Attack:
    """Relation d'attaque dans le QBAF.

    Attributes:
        source: Argument attaquant.
        target: Argument attaque.
        weight: Poids de l'attaque w dans [0, 1].
    """
    source: str
    target: str
    weight: float = 1.0


@dataclass
class Support:
    """Relation de support dans le QBAF.

    Attributes:
        source: Argument supportant.
        target: Argument supporte.
        weight: Poids du support w dans [0, 1].
    """
    source: str
    target: str
    weight: float = 1.0


@dataclass
class QBAFResult:
    """Resultat de l'evaluation du NeSy QBAF.

    Attributes:
        strengths: Forces finales de chaque argument.
        recommendation: Recommandation finale (argument dominant).
        recommendation_strength: Force de la recommandation.
        symbolic_overrides: Arguments symboliques ayant force l'issue.
        convergence_iterations: Nombre d'iterations QE pour la convergence.
        counterfactuals: Explication contrefactuelle (CE-QArg).
        falsification_matrix: Matrice de falsification (quel argument fait basculer).
    """
    strengths: Dict[str, float]
    recommendation: str
    recommendation_strength: float
    symbolic_overrides: List[str]
    convergence_iterations: int
    counterfactuals: Dict[str, float] = field(default_factory=dict)
    falsification_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ──────────────────────────────────────────────
# NeSy QBAF engine
# ──────────────────────────────────────────────

# Recommendations possibles
RECOMMENDATIONS = [
    "maintenir",     # Pas de changement
    "surveiller",    # Monitoring renforce
    "reduire",       # Reduire l'exposition
    "escalader",     # Escalade direction des risques
]


class NeSyQBAF:
    """Moteur NeSy QBAF avec resolution QE et arguments symboliques.

    Architecture du graphe :
        Arguments de recommandation : maintenir, surveiller, reduire, escalader
        Arguments neuronaux : fusion_favorable, fusion_defavorable, contrarian_alert
        Arguments symboliques : ifrs9_hard_stage3, ifrs9_hard_sicr, rst_proximity

    Relations :
        fusion_favorable --support--> maintenir
        fusion_defavorable --support--> reduire
        fusion_defavorable --attack--> maintenir
        contrarian_alert --attack--> maintenir (si consensus_wrong)
        ifrs9_hard_stage3 --support--> escalader (force=inf)
        ifrs9_hard_sicr --support--> surveiller
        rst_proximity --support--> escalader
        rst_proximity --attack--> maintenir
    """

    # QE model params
    DAMPING = 0.85       # Facteur d'amortissement
    MAX_ITER = 100       # Iterations max QE
    TOLERANCE = 1e-6     # Convergence

    def __init__(self) -> None:
        self.arguments: Dict[str, Argument] = {}
        self.attacks: List[Attack] = []
        self.supports: List[Support] = []

    def build_graph(
        self,
        fusion_result: FusionResult,
        contrarian_wrong_prob: float,
        stage3_pct: float,
        sicr_pct: float,
        rst_distance: float,
        ecl_ead_ratio: float,
    ) -> None:
        """Construit le graphe QBAF a partir des resultats de fusion.

        Args:
            fusion_result: Resultat de la fusion DS.
            contrarian_wrong_prob: P(consensus_wrong) du Contrarian.
            stage3_pct: Pourcentage de positions en Stage 3.
            sicr_pct: Pourcentage de positions avec SICR (Stage 2+).
            rst_distance: Distance RST en sigma.
            ecl_ead_ratio: Ratio ECL/EAD du portefeuille.
        """
        self.arguments.clear()
        self.attacks.clear()
        self.supports.clear()

        # --- Arguments de recommandation ---
        for rec in RECOMMENDATIONS:
            self.arguments[rec] = Argument(
                name=rec, base_strength=0.0, is_symbolic=False,
                description=f"Recommandation : {rec}",
            )

        # --- Arguments neuronaux (issus de la fusion) ---
        # Force = masse fusionnee dominante
        fused = fusion_result.fused_masses
        belief = fusion_result.belief
        conflict = fusion_result.conflict_level

        # Favorable : expansion + sain + amelioration
        favorable_mass = sum(
            fused.get(h, 0.0)
            for h in ["expansion", "sain", "amelioration", "normal"]
        )
        self.arguments["fusion_favorable"] = Argument(
            name="fusion_favorable",
            base_strength=np.clip(favorable_mass, -1, 1),
            description=f"Consensus favorable (masse={favorable_mass:.2f})",
        )

        # Defavorable : recession + crise + degradation + distress + watchlist
        defavorable_mass = sum(
            fused.get(h, 0.0)
            for h in ["recession", "crise", "degradation", "distress", "watchlist"]
        )
        self.arguments["fusion_defavorable"] = Argument(
            name="fusion_defavorable",
            base_strength=np.clip(defavorable_mass, -1, 1),
            description=f"Consensus defavorable (masse={defavorable_mass:.2f})",
        )

        # Contrarian alert
        self.arguments["contrarian_alert"] = Argument(
            name="contrarian_alert",
            base_strength=np.clip(contrarian_wrong_prob, 0, 1),
            description=f"Alerte contrarian (P(wrong)={contrarian_wrong_prob:.2f})",
        )

        # --- Arguments symboliques (IFRS 9 — force infinie) ---
        # Hard Stage 3 : si > 5% du book en Stage 3, escalade non-negociable
        stage3_trigger = stage3_pct > 0.05
        self.arguments["ifrs9_hard_stage3"] = Argument(
            name="ifrs9_hard_stage3",
            base_strength=1.0 if stage3_trigger else 0.0,
            is_symbolic=True,
            description=f"IFRS 9 Stage 3 > 5% ({stage3_pct:.1%})",
        )

        # Hard SICR : si > 25% en Stage 2+, surveillance obligatoire
        sicr_trigger = sicr_pct > 0.25
        self.arguments["ifrs9_hard_sicr"] = Argument(
            name="ifrs9_hard_sicr",
            base_strength=1.0 if sicr_trigger else 0.0,
            is_symbolic=True,
            description=f"IFRS 9 SICR > 25% ({sicr_pct:.1%})",
        )

        # RST proximity : si < 2 sigma, danger imminent
        rst_danger = rst_distance < 2.0
        rst_strength = np.clip(1.0 - rst_distance / 5.0, 0, 1)
        self.arguments["rst_proximity"] = Argument(
            name="rst_proximity",
            base_strength=rst_strength if rst_danger else rst_strength * 0.3,
            is_symbolic=rst_danger,  # Symbolique seulement si < 2 sigma
            description=f"RST distance = {rst_distance:.1f} sigma",
        )

        # ECL/EAD severity
        ecl_severity = np.clip(ecl_ead_ratio / 0.10, 0, 1)  # 10% = pleine force
        self.arguments["ecl_severity"] = Argument(
            name="ecl_severity",
            base_strength=ecl_severity,
            description=f"ECL/EAD = {ecl_ead_ratio:.2%}",
        )

        # --- Relations ---
        # Supports
        self.supports.append(Support("fusion_favorable", "maintenir", 0.8))
        self.supports.append(Support("fusion_defavorable", "reduire", 0.8))
        self.supports.append(Support("fusion_defavorable", "surveiller", 0.5))
        self.supports.append(Support("ecl_severity", "reduire", 0.6))
        self.supports.append(Support("ecl_severity", "surveiller", 0.4))
        self.supports.append(Support("ifrs9_hard_stage3", "escalader", 1.0))
        self.supports.append(Support("ifrs9_hard_sicr", "surveiller", 0.9))
        self.supports.append(Support("rst_proximity", "escalader", 0.9))

        # Attacks
        self.attacks.append(Attack("fusion_defavorable", "maintenir", 0.7))
        self.attacks.append(Attack("contrarian_alert", "maintenir", 0.5))
        self.attacks.append(Attack("fusion_favorable", "reduire", 0.5))
        self.attacks.append(Attack("fusion_favorable", "escalader", 0.3))
        self.attacks.append(Attack("rst_proximity", "maintenir", 0.8))
        self.attacks.append(Attack("ifrs9_hard_stage3", "maintenir", 1.0))

    def evaluate(self) -> QBAFResult:
        """Evalue le QBAF par le modele QE (Quadratic Energy).

        Resolution par iteration de point fixe avec damping :
            sigma(a) = tau(a) + alpha * [sum(w_s * sigma(s)) - sum(w_a * sigma(a_att))]
            Clip dans [-1, 1].

        Les arguments symboliques ne sont PAS mis a jour (force fixe).

        Returns:
            QBAFResult avec forces finales et recommandation.
        """
        # Initialiser les forces avec les base_strengths
        strengths = {
            name: arg.base_strength
            for name, arg in self.arguments.items()
        }

        # Iteration QE
        n_iter = 0
        for iteration in range(self.MAX_ITER):
            new_strengths = {}
            max_change = 0.0

            for name, arg in self.arguments.items():
                if arg.is_symbolic:
                    # Arguments symboliques : force fixe (non-differentiable)
                    new_strengths[name] = arg.base_strength
                    continue

                # Somme des supports entrants
                support_sum = 0.0
                for s in self.supports:
                    if s.target == name:
                        support_sum += s.weight * max(0, strengths.get(s.source, 0))

                # Somme des attaques entrantes
                attack_sum = 0.0
                for a in self.attacks:
                    if a.target == name:
                        attack_sum += a.weight * max(0, strengths.get(a.source, 0))

                # QE update avec damping
                new_val = arg.base_strength + self.DAMPING * (support_sum - attack_sum)
                new_val = np.clip(new_val, -1.0, 1.0)
                new_strengths[name] = float(new_val)

                max_change = max(max_change, abs(new_val - strengths.get(name, 0)))

            strengths = new_strengths
            n_iter = iteration + 1

            if max_change < self.TOLERANCE:
                break

        # Identifier la recommandation dominante
        rec_strengths = {
            r: strengths.get(r, 0.0) for r in RECOMMENDATIONS
        }
        recommendation = max(rec_strengths, key=rec_strengths.get)
        rec_strength = rec_strengths[recommendation]

        # Arguments symboliques ayant influence l'issue
        symbolic_overrides = [
            name for name, arg in self.arguments.items()
            if arg.is_symbolic and arg.base_strength > 0.5
        ]

        # Si un argument symbolique force l'escalade, il domine
        if "ifrs9_hard_stage3" in symbolic_overrides:
            recommendation = "escalader"
            rec_strength = strengths.get("escalader", 1.0)

        # Construire la matrice de falsification
        falsification = self._compute_falsification_matrix(strengths)

        return QBAFResult(
            strengths=strengths,
            recommendation=recommendation,
            recommendation_strength=rec_strength,
            symbolic_overrides=symbolic_overrides,
            convergence_iterations=n_iter,
            counterfactuals={},
            falsification_matrix=falsification,
        )

    def _compute_falsification_matrix(
        self,
        current_strengths: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """Matrice de falsification : quel argument fait basculer la recommandation.

        Pour chaque argument neural, on simule sa desactivation (force=0)
        et on observe l'impact sur les recommandations.

        Returns:
            Dict[argument_name -> Dict[recommendation -> delta_strength]]
        """
        matrix = {}
        neural_args = [
            name for name, arg in self.arguments.items()
            if not arg.is_symbolic and name in [
                "fusion_favorable", "fusion_defavorable",
                "contrarian_alert", "ecl_severity",
            ]
        ]

        for arg_name in neural_args:
            # Sauvegarder la force originale
            original_strength = self.arguments[arg_name].base_strength
            self.arguments[arg_name].base_strength = 0.0

            # Re-evaluer (pas de recursion — direct QE)
            strengths_modified = self._quick_evaluate()

            # Restaurer
            self.arguments[arg_name].base_strength = original_strength

            # Delta par recommandation
            deltas = {}
            for rec in RECOMMENDATIONS:
                deltas[rec] = strengths_modified.get(rec, 0.0) - current_strengths.get(rec, 0.0)
            matrix[arg_name] = deltas

        return matrix

    def _quick_evaluate(self) -> Dict[str, float]:
        """Evaluation rapide du QBAF (sans construire QBAFResult)."""
        strengths = {
            name: arg.base_strength
            for name, arg in self.arguments.items()
        }

        for _ in range(self.MAX_ITER):
            new_strengths = {}
            max_change = 0.0

            for name, arg in self.arguments.items():
                if arg.is_symbolic:
                    new_strengths[name] = arg.base_strength
                    continue

                support_sum = sum(
                    s.weight * max(0, strengths.get(s.source, 0))
                    for s in self.supports if s.target == name
                )
                attack_sum = sum(
                    a.weight * max(0, strengths.get(a.source, 0))
                    for a in self.attacks if a.target == name
                )

                new_val = float(np.clip(
                    arg.base_strength + self.DAMPING * (support_sum - attack_sum),
                    -1.0, 1.0,
                ))
                new_strengths[name] = new_val
                max_change = max(max_change, abs(new_val - strengths.get(name, 0)))

            strengths = new_strengths
            if max_change < self.TOLERANCE:
                break

        return strengths

    def compute_counterfactuals(
        self,
        macro_params: Dict[str, float],
        ecl_proxy_fn,
        current_result: QBAFResult,
    ) -> Dict[str, float]:
        """CE-QArg : quelles perturbations macro changeraient la recommandation.

        Pour chaque variable macro, on teste un choc de +/- 1 ecart-type
        et on mesure l'impact sur la force de la recommandation actuelle.

        Args:
            macro_params: Variables macro actuelles.
            ecl_proxy_fn: Fonction ECL proxy.
            current_result: Resultat QBAF actuel.

        Returns:
            Dict[variable -> sensitivity] indiquant la sensibilite
            de la recommandation a chaque variable macro.
        """
        from ifrs9_cockpit.config import _MACRO_VOLATILITIES

        base_ecl = ecl_proxy_fn(macro_params)
        sensitivities = {}

        for var, sigma in _MACRO_VOLATILITIES.items():
            # Choc +1 sigma
            params_up = dict(macro_params)
            params_up[var] = macro_params.get(var, 0) + sigma
            ecl_up = ecl_proxy_fn(params_up)

            # Choc -1 sigma
            params_down = dict(macro_params)
            params_down[var] = macro_params.get(var, 0) - sigma
            ecl_down = ecl_proxy_fn(params_down)

            # Sensibilite = (ECL_up - ECL_down) / (2 * sigma * ECL_base)
            if base_ecl > 0:
                sensitivities[var] = (ecl_up - ecl_down) / (2 * sigma * base_ecl)
            else:
                sensitivities[var] = 0.0

        return sensitivities
