"""4 Agents neuronaux specialises pour le Virtual CRO NeSy MAS.

Chaque agent est un MLP leger (numpy, pas de PyTorch pour l'inference)
qui produit des masses de croyance Dempster-Shafer sur un cadre de
discernement adapte a sa specialite.

Architecture :
    - MacroAgent  : MLP(5 vars macro) -> P(regime) sur 4 classes
    - QuantAgent  : MLP(PD + features credit) -> P(stage_shift) sur 3 classes
    - PEAgent     : MLP(features PE) -> P(distress_level) sur 3 classes
    - ContrarianAgent : discriminateur(3 outputs agents) -> P(consensus_wrong)

Les poids sont initialises par Xavier/Glorot et peuvent etre entraines
par descente de gradient (backprop numpy). En production, les poids
seraient charges depuis un fichier .npz.

References :
    - FinCon (NeurIPS 2024) : agents specialises multi-niveaux
    - MASCA (ACL 2025) : consensus credit multi-agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────

@dataclass
class BeliefMass:
    """Masse de croyance Dempster-Shafer sur un cadre de discernement.

    Attributes:
        frame: Noms des hypotheses (ex: ["expansion", "normal", "recession", "crise"]).
        masses: Dictionnaire hypothese -> masse m(A). Inclut "uncertainty" pour Omega.
        agent_name: Nom de l'agent source.
        confidence: Confiance auto-evaluee [0, 1] (pour bBPA discounting).
    """
    frame: List[str]
    masses: Dict[str, float]
    agent_name: str
    confidence: float = 1.0


# ──────────────────────────────────────────────
# MLP base class (numpy)
# ──────────────────────────────────────────────

class _MLPBase:
    """MLP leger a 2 couches cachees (numpy).

    Architecture : Input -> Dense(h1, ReLU) -> Dense(h2, ReLU) -> Dense(out, Softmax)
    Initialisation Xavier/Glorot pour une convergence rapide.
    """

    def __init__(
        self,
        input_dim: int,
        hidden1: int,
        hidden2: int,
        output_dim: int,
        seed: int = 42,
    ) -> None:
        rng = np.random.RandomState(seed)
        # Xavier init
        self.W1 = rng.randn(input_dim, hidden1) * np.sqrt(2.0 / (input_dim + hidden1))
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.randn(hidden1, hidden2) * np.sqrt(2.0 / (hidden1 + hidden2))
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.randn(hidden2, output_dim) * np.sqrt(2.0 / (hidden2 + output_dim))
        self.b3 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass : x (input_dim,) -> softmax probabilities (output_dim,)."""
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        logits = h2 @ self.W3 + self.b3
        # Numerically stable softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return all parameters (for serialization)."""
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W3": self.W3, "b3": self.b3,
        }

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Load parameters (from .npz file)."""
        self.W1 = params["W1"]
        self.b1 = params["b1"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]
        self.W3 = params["W3"]
        self.b3 = params["b3"]


# ──────────────────────────────────────────────
# Agent 1 : Macro
# ──────────────────────────────────────────────

# Cadre de discernement macro
MACRO_FRAME = ["expansion", "normal", "recession", "crise"]


class MacroAgent:
    """Agent Macro : MLP(5 vars macro) -> P(regime) sur 4 classes.

    Entrees (normalisees) :
        [unemployment_rate, gdp_growth, interest_rate, hpi_growth, inflation_rate]

    Sorties (softmax) :
        P(expansion), P(normal), P(recession), P(crise)

    La normalisation utilise les ranges historiques zone euro (2000-2024).
    """

    # Ranges historiques pour normalisation [min, max]
    _RANGES = {
        "unemployment_rate": (3.0, 13.0),
        "gdp_growth": (-8.0, 6.0),
        "interest_rate": (-0.5, 5.0),
        "hpi_growth": (-15.0, 15.0),
        "inflation_rate": (-1.0, 10.0),
    }

    _VAR_ORDER = [
        "unemployment_rate", "gdp_growth", "interest_rate",
        "hpi_growth", "inflation_rate",
    ]

    def __init__(self, seed: int = 42) -> None:
        self.mlp = _MLPBase(
            input_dim=5, hidden1=16, hidden2=8,
            output_dim=len(MACRO_FRAME), seed=seed,
        )
        self.name = "macro"

    def _normalize(self, macro_params: Dict[str, float]) -> np.ndarray:
        """Normalise les 5 variables macro dans [0, 1]."""
        x = np.zeros(5)
        for i, var in enumerate(self._VAR_ORDER):
            lo, hi = self._RANGES[var]
            val = macro_params.get(var, (lo + hi) / 2)
            x[i] = np.clip((val - lo) / (hi - lo), 0.0, 1.0)
        return x

    def predict(self, macro_params: Dict[str, float]) -> BeliefMass:
        """Produit une masse de croyance sur le regime macro.

        Args:
            macro_params: Dict des 5 variables macro.

        Returns:
            BeliefMass sur MACRO_FRAME + uncertainty.
        """
        x = self._normalize(macro_params)
        probs = self.mlp.forward(x)

        # Convertir softmax en masses DS (avec masse residuelle d'incertitude)
        # Regle : si max(probs) < 0.4, l'agent est peu confiant
        max_prob = float(np.max(probs))
        confidence = np.clip(max_prob * 2.0, 0.3, 0.95)

        # Masses focales = probs * confidence, reste = uncertainty
        masses = {}
        for i, hyp in enumerate(MACRO_FRAME):
            masses[hyp] = float(probs[i]) * confidence
        masses["uncertainty"] = 1.0 - confidence

        return BeliefMass(
            frame=MACRO_FRAME,
            masses=masses,
            agent_name=self.name,
            confidence=confidence,
        )


# ──────────────────────────────────────────────
# Agent 2 : Quant (consomme PD existante)
# ──────────────────────────────────────────────

QUANT_FRAME = ["amelioration", "stable", "degradation"]


class QuantAgent:
    """Agent Quant : MLP(PD + features credit) -> P(stage_shift).

    Consomme la PD du modele existant (LR_WoE / TabNet / XGBoost)
    comme input principal + metriques portfolio.

    Entrees (6 features normalisees) :
        [pd_mean, pd_std, stage2_pct, stage3_pct, ecl_ead_ratio, sicr_score_mean]

    Sorties (softmax) :
        P(amelioration), P(stable), P(degradation)
    """

    def __init__(self, seed: int = 43) -> None:
        self.mlp = _MLPBase(
            input_dim=6, hidden1=12, hidden2=8,
            output_dim=len(QUANT_FRAME), seed=seed,
        )
        self.name = "quant"

    def _extract_features(self, result_credit) -> np.ndarray:
        """Extrait 6 features du resultat credit normalise."""
        pd_vals = result_credit["pd_12m"].to_numpy()
        stages = result_credit["stage"].to_numpy()
        ecl = result_credit["ecl_weighted"].to_numpy()
        ead = result_credit["ead"].to_numpy()

        x = np.array([
            np.clip(pd_vals.mean() / 0.20, 0, 1),           # pd_mean normalise
            np.clip(pd_vals.std() / 0.15, 0, 1),            # pd_std normalise
            np.clip((stages == 2).mean() / 0.40, 0, 1),     # stage2_pct normalise
            np.clip((stages == 3).mean() / 0.20, 0, 1),     # stage3_pct normalise
            np.clip(ecl.sum() / max(ead.sum(), 1) / 0.10, 0, 1),  # ecl_ead normalise
            np.clip(pd_vals.mean() / 0.30, 0, 1),           # proxy sicr_score
        ])
        return x

    def predict(self, result_credit) -> BeliefMass:
        """Produit une masse de croyance sur la tendance credit.

        Args:
            result_credit: DataFrame resultat ECLCalculator.

        Returns:
            BeliefMass sur QUANT_FRAME + uncertainty.
        """
        x = self._extract_features(result_credit)
        probs = self.mlp.forward(x)

        max_prob = float(np.max(probs))
        confidence = np.clip(max_prob * 1.8, 0.3, 0.90)

        masses = {}
        for i, hyp in enumerate(QUANT_FRAME):
            masses[hyp] = float(probs[i]) * confidence
        masses["uncertainty"] = 1.0 - confidence

        return BeliefMass(
            frame=QUANT_FRAME,
            masses=masses,
            agent_name=self.name,
            confidence=confidence,
        )


# ──────────────────────────────────────────────
# Agent 3 : PE
# ──────────────────────────────────────────────

PE_FRAME = ["sain", "watchlist", "distress"]


class PEAgent:
    """Agent PE : MLP(features PE) -> P(distress_level).

    Entrees (5 features normalisees) :
        [nav_drawdown, pct_distressed, pct_watchlist, avg_leverage, vintage_avg]

    Sorties (softmax) :
        P(sain), P(watchlist), P(distress)
    """

    def __init__(self, seed: int = 44) -> None:
        self.mlp = _MLPBase(
            input_dim=5, hidden1=12, hidden2=8,
            output_dim=len(PE_FRAME), seed=seed,
        )
        self.name = "pe"

    def _extract_features(self, result_pe) -> np.ndarray:
        """Extrait 5 features du resultat PE normalise."""
        nav = result_pe["nav"].to_numpy()
        delta = result_pe["delta_nav"].to_numpy()
        cats = result_pe["risk_category"].to_numpy()

        nav_ref = nav - delta
        drawdown = float(np.clip(-delta.sum() / max(nav_ref.sum(), 1), 0, 1))
        pct_distressed = float((cats == "Distressed").mean())
        pct_watchlist = float((cats == "Watchlist").mean())

        avg_leverage = 0.5  # default normalise
        if "leverage" in result_pe.columns:
            avg_leverage = float(np.clip(result_pe["leverage"].mean() / 5.0, 0, 1))

        vintage_avg = 0.5
        if "vintage" in result_pe.columns:
            vintage_avg = float(np.clip(result_pe["vintage"].mean() / 10.0, 0, 1))

        return np.array([drawdown, pct_distressed, pct_watchlist, avg_leverage, vintage_avg])

    def predict(self, result_pe) -> BeliefMass:
        """Produit une masse de croyance sur l'etat du portefeuille PE.

        Args:
            result_pe: DataFrame resultat PECalculator.

        Returns:
            BeliefMass sur PE_FRAME + uncertainty.
        """
        x = self._extract_features(result_pe)
        probs = self.mlp.forward(x)

        max_prob = float(np.max(probs))
        confidence = np.clip(max_prob * 1.8, 0.3, 0.90)

        masses = {}
        for i, hyp in enumerate(PE_FRAME):
            masses[hyp] = float(probs[i]) * confidence
        masses["uncertainty"] = 1.0 - confidence

        return BeliefMass(
            frame=PE_FRAME,
            masses=masses,
            agent_name=self.name,
            confidence=confidence,
        )


# ──────────────────────────────────────────────
# Agent 4 : Contrarian (discriminateur adversarial)
# ──────────────────────────────────────────────

CONTRARIAN_FRAME = ["consensus_correct", "consensus_wrong"]


class ContrarianAgent:
    """Agent Contrarian : discriminateur adversarial.

    Prend en entree les confidences des 3 autres agents et evalue
    la probabilite que le consensus soit errone (ACPO concept).

    Anti-sycophancy by design :
        - Query neutralization : ne voit PAS la recommandation finale
        - Confidence hiding : ne recoit que les confidences, pas les masses
        - Heterogene : architecture et seed differents

    Entrees (3 confidences + 3 entropies) :
        [conf_macro, conf_quant, conf_pe, H_macro, H_quant, H_pe]

    Sorties (softmax) :
        P(consensus_correct), P(consensus_wrong)
    """

    def __init__(self, seed: int = 99) -> None:
        self.mlp = _MLPBase(
            input_dim=6, hidden1=8, hidden2=4,
            output_dim=len(CONTRARIAN_FRAME), seed=seed,
        )
        self.name = "contrarian"

    @staticmethod
    def _entropy(masses: Dict[str, float]) -> float:
        """Entropie de Shannon des masses focales (hors uncertainty)."""
        focal = [v for k, v in masses.items() if k != "uncertainty" and v > 1e-10]
        if not focal:
            return 0.0
        total = sum(focal)
        if total < 1e-10:
            return 0.0
        probs = [f / total for f in focal]
        return float(-sum(p * np.log(p + 1e-10) for p in probs))

    def predict(
        self,
        belief_macro: BeliefMass,
        belief_quant: BeliefMass,
        belief_pe: BeliefMass,
    ) -> BeliefMass:
        """Evalue si le consensus des 3 agents est fiable.

        Args:
            belief_macro: Masse de croyance agent Macro.
            belief_quant: Masse de croyance agent Quant.
            belief_pe: Masse de croyance agent PE.

        Returns:
            BeliefMass P(consensus_correct) vs P(consensus_wrong).
        """
        x = np.array([
            belief_macro.confidence,
            belief_quant.confidence,
            belief_pe.confidence,
            self._entropy(belief_macro.masses),
            self._entropy(belief_quant.masses),
            self._entropy(belief_pe.masses),
        ])

        probs = self.mlp.forward(x)

        confidence = float(np.clip(np.max(probs) * 1.5, 0.3, 0.85))

        masses = {}
        for i, hyp in enumerate(CONTRARIAN_FRAME):
            masses[hyp] = float(probs[i]) * confidence
        masses["uncertainty"] = 1.0 - confidence

        return BeliefMass(
            frame=CONTRARIAN_FRAME,
            masses=masses,
            agent_name=self.name,
            confidence=confidence,
        )
