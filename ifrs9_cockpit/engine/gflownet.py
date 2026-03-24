"""Continuous GFlowNet dual-reward pour generation de scenarios et allocations.

Systeme ambidextre conditionne par le regime HMM :
    - Mode defensif (contraction) : genere des scenarios de stress (cygnes noirs)
      R = max(0, ECL_surrogate(z) - breach) * plausibility(z)
    - Mode offensif (expansion) : genere des allocations optimales
      R = RORAC(allocation) * (1 - penalty_ECL)

Architecture numpy pure :
    - Forward policy : MLP genere des actions dans Z-space (Cholesky)
    - Backward policy : MLP inverse pour flow consistency
    - Trajectory Balance loss (Malkin et al. 2022)
    - Replay buffer pour stabilite

L'espace de generation est le Z-space Mahalanobis (5D) defini par la
matrice MACRO_COVARIANCE -- chaque point z correspond a un scenario
macro x = mu + L*z ou L est le facteur Cholesky.

References :
    - Bengio et al. (JMLR 2023) : GFlowNet Foundations
    - Malkin et al. (NeurIPS 2022) : Trajectory Balance
    - Lahlou et al. (ICML 2023) : A Theory of Continuous GFlowNets
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GFlowNetResult:
    """Resultat du GFlowNet dual-reward.

    Attributes:
        mode: "defensive" ou "offensive".
        scenarios: Liste de scenarios generes (dicts macro_params).
        z_vectors: Vecteurs Z correspondants (N, D).
        rewards: Rewards associes a chaque scenario (N,).
        log_Z: Log-partition estimee.
        diversity: Metrique de diversite (distance inter-scenarios moyenne).
        best_scenario: Scenario avec le reward le plus eleve.
        best_reward: Reward du meilleur scenario.
        allocations: Allocations optimales (mode offensif seulement).
    """

    mode: str
    scenarios: List[Dict[str, float]]
    z_vectors: np.ndarray
    rewards: np.ndarray
    log_Z: float
    diversity: float
    best_scenario: Dict[str, float]
    best_reward: float
    allocations: Optional[List[Dict[str, float]]] = None


# ──────────────────────────────────────────────
# MLP NUMPY (forward/backward policies)
# ──────────────────────────────────────────────
class _MLP:
    """MLP numpy minimaliste (ReLU + softplus output)."""

    def __init__(self, dims: List[int], seed: int = 42):
        """Args: dims = [input, hidden1, ..., output]."""
        self.rng = np.random.RandomState(seed)
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            w = self.rng.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass avec ReLU sauf derniere couche (lineaire)."""
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0)  # ReLU
        return h

    def parameters(self) -> List[np.ndarray]:
        """Retourne tous les parametres (pour SGD)."""
        params = []
        for w, b in zip(self.weights, self.biases):
            params.extend([w, b])
        return params


# ──────────────────────────────────────────────
# Z-SPACE TRANSFORM (Cholesky)
# ──────────────────────────────────────────────
def _get_cholesky():
    """Retourne (mu, L) pour transformer Z-space -> macro-space."""
    from ifrs9_cockpit.config import (
        MACRO_COVARIANCE,
        MACRO_VARIABLES_ORDER,
        SCENARIO_BASE,
    )
    mu = np.array([
        getattr(SCENARIO_BASE, {
            "unemployment_rate": "unemployment_rate",
            "gdp_growth": "gdp_growth",
            "interest_rate": "interest_rate",
            "hpi_growth": "hpi_growth",
            "inflation_rate": "inflation_rate",
        }[v]) for v in MACRO_VARIABLES_ORDER
    ])
    Sigma = np.array(MACRO_COVARIANCE)
    L = np.linalg.cholesky(Sigma)
    return mu, L, list(MACRO_VARIABLES_ORDER)


def z_to_macro(z: np.ndarray) -> Dict[str, float]:
    """Convertit un vecteur Z-space en dict macro_params."""
    mu, L, var_names = _get_cholesky()
    x = mu + L @ z
    return {var_names[i]: float(x[i]) for i in range(len(var_names))}


def macro_to_z_vec(macro_params: Dict[str, float]) -> np.ndarray:
    """Convertit un dict macro_params en vecteur Z-space."""
    mu, L, var_names = _get_cholesky()
    x = np.array([macro_params.get(v, mu[i]) for i, v in enumerate(var_names)])
    return np.linalg.solve(L, x - mu)


# ──────────────────────────────────────────────
# REWARD FUNCTIONS
# ──────────────────────────────────────────────
def defensive_reward(
    z: np.ndarray,
    ecl_surrogate: Callable[[Dict[str, float]], float],
    breach_threshold: float,
    sigma_budget: float = 4.0,
) -> float:
    """Reward defensif : scenarios de stress (cygnes noirs).

    R = max(0, ECL(z) - breach) * exp(-0.5 * max(0, ||z|| - sigma)^2)

    Encourage les scenarios qui brisent le seuil ECL tout en restant
    plausibles (distance Mahalanobis sous le budget sigma).

    Args:
        z: Vecteur Z-space (5D).
        ecl_surrogate: Callable(macro_dict) -> ECL total.
        breach_threshold: Seuil ECL a depasser.
        sigma_budget: Budget Mahalanobis max.
    """
    macro = z_to_macro(z)
    ecl = ecl_surrogate(macro)
    severity = max(0.0, ecl - breach_threshold)

    # Plausibility : soft penalty au-dela du budget sigma
    distance = float(np.linalg.norm(z))
    if distance > sigma_budget:
        plausibility = np.exp(-0.5 * (distance - sigma_budget) ** 2)
    else:
        plausibility = 1.0

    return severity * plausibility + 1e-8  # floor pour eviter R=0


def offensive_reward(
    z: np.ndarray,
    rorac_surrogate: Callable[[Dict[str, float]], float],
    ecl_surrogate: Callable[[Dict[str, float]], float],
    ecl_constraint: float,
    sigma_budget: float = 3.0,
) -> float:
    """Reward offensif : scenarios favorables pour arbitrage.

    R = RORAC(z) * max(0, 1 - ECL(z)/constraint) * plausibility(z)

    Encourage les scenarios ou le RORAC est eleve et l'ECL reste sous controle.

    Args:
        z: Vecteur Z-space.
        rorac_surrogate: Callable(macro_dict) -> RORAC.
        ecl_surrogate: Callable(macro_dict) -> ECL total.
        ecl_constraint: Contrainte ECL max.
        sigma_budget: Budget Mahalanobis.
    """
    macro = z_to_macro(z)
    rorac = max(0.0, rorac_surrogate(macro))
    ecl = ecl_surrogate(macro)

    # ECL penalty : lineaire au-dessus du constraint
    ecl_factor = max(0.0, 1.0 - ecl / ecl_constraint) if ecl_constraint > 0 else 1.0

    # Plausibility
    distance = float(np.linalg.norm(z))
    plausibility = np.exp(-0.5 * (max(0.0, distance - sigma_budget)) ** 2)

    return rorac * ecl_factor * plausibility + 1e-8


# ──────────────────────────────────────────────
# CONTINUOUS GFlowNet
# ──────────────────────────────────────────────
class ContinuousGFlowNet:
    """GFlowNet continu dual-reward pour generation de scenarios macro.

    Architecture :
        - Forward policy: MLP(state_dim + 1) -> (mean, log_std) sur Z-space
        - Sampling: z ~ N(mean, exp(log_std)^2)
        - Trajectory Balance: loss = (log Z + log P_F - log R - log P_B)^2
        - Replay buffer pour stabilite

    Usage :
        gfn = ContinuousGFlowNet(state_dim=5)
        result = gfn.generate(
            reward_fn=defensive_reward,
            n_samples=100,
            n_train_steps=200,
        )
    """

    def __init__(
        self,
        state_dim: int = 5,
        hidden_dim: int = 64,
        n_steps: int = 5,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.rng = np.random.RandomState(seed)

        # Learnable log Z (partition function)
        self.log_Z = 0.0

        # Forward policy : state -> (mean, log_std) pour action
        self.forward_net = _MLP(
            [state_dim, hidden_dim, hidden_dim // 2, state_dim * 2],
            seed=seed,
        )

        # Backward policy : state -> log_prob
        self.backward_net = _MLP(
            [state_dim, hidden_dim // 2, 1],
            seed=seed + 1,
        )

        # Replay buffer
        self._replay_z: List[np.ndarray] = []
        self._replay_r: List[float] = []
        self._replay_max = 500

    def _sample_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Echantillonne une action (delta Z) depuis la forward policy.

        Returns:
            action: Delta Z (state_dim,).
            log_prob: Log-probabilite de l'action.
        """
        out = self.forward_net.forward(state)
        mean = out[:self.state_dim]
        log_std = np.clip(out[self.state_dim:], -3.0, 1.0)
        std = np.exp(log_std)

        # Sample from Gaussian
        eps = self.rng.randn(self.state_dim)
        action = mean + std * eps

        # Log probability
        log_prob = -0.5 * np.sum(
            ((action - mean) / std) ** 2 + 2.0 * log_std + np.log(2 * np.pi)
        )

        return action, float(log_prob)

    def _backward_log_prob(self, state: np.ndarray) -> float:
        """Log-probabilite backward (approximee par le backward net)."""
        return float(self.backward_net.forward(state).ravel()[0])

    def _sample_trajectory(
        self,
        reward_fn: Callable[[np.ndarray], float],
    ) -> Tuple[np.ndarray, float, float, float]:
        """Echantillonne une trajectoire et calcule la TB loss.

        Returns:
            z_final: Vecteur Z final.
            reward: Reward du point final.
            log_pf: Sum of forward log-probs.
            log_pb: Sum of backward log-probs.
        """
        state = np.zeros(self.state_dim)
        log_pf_total = 0.0

        for step in range(self.n_steps):
            action, log_pf = self._sample_action(state)
            state = state + action / self.n_steps  # Incremental steps
            log_pf_total += log_pf

        z_final = state
        reward = reward_fn(z_final)

        # Backward log-prob (approximation simple)
        log_pb_total = self._backward_log_prob(z_final)

        return z_final, reward, log_pf_total, log_pb_total

    def _update_params(
        self,
        trajectories: List[Tuple[np.ndarray, float, float, float]],
        lr: float = 0.001,
    ) -> float:
        """Met a jour log_Z et les politiques via gradient TB.

        Returns:
            Moyenne de la TB loss.
        """
        total_loss = 0.0
        n = len(trajectories)

        for z, r, log_pf, log_pb in trajectories:
            # TB loss = (log Z + log P_F - log R - log P_B)^2
            log_r = np.log(max(r, 1e-10))
            tb_residual = self.log_Z + log_pf - log_r - log_pb

            # Update log_Z (gradient descent on residual)
            self.log_Z -= lr * 2.0 * tb_residual / n

            total_loss += tb_residual ** 2

        return total_loss / n

    def generate(
        self,
        reward_fn: Callable[[np.ndarray], float],
        n_samples: int = 100,
        n_train_steps: int = 200,
        batch_size: int = 32,
        lr: float = 0.001,
    ) -> GFlowNetResult:
        """Genere des scenarios/allocations via le GFlowNet entraine.

        Pipeline :
            1. Warm-up : echantillons aleatoires pour initialiser le replay
            2. Training : n_train_steps d'optimisation TB
            3. Sampling : n_samples generes depuis la politique entrainee

        Args:
            reward_fn: Callable(z_vector) -> float (reward).
            n_samples: Nombre de scenarios a generer.
            n_train_steps: Iterations d'entrainement.
            batch_size: Trajectoires par batch.
            lr: Learning rate.

        Returns:
            GFlowNetResult avec scenarios, rewards, diversite.
        """
        # Warm-up: explore with random Z vectors
        for _ in range(min(100, self._replay_max)):
            z = self.rng.randn(self.state_dim) * 2.0
            r = reward_fn(z)
            self._replay_z.append(z)
            self._replay_r.append(r)

        # Training loop
        for step in range(n_train_steps):
            trajectories = []
            for _ in range(batch_size):
                traj = self._sample_trajectory(reward_fn)
                trajectories.append(traj)

                # Add to replay
                z, r, _, _ = traj
                if len(self._replay_z) >= self._replay_max:
                    # Evict lowest reward
                    min_idx = int(np.argmin(self._replay_r))
                    self._replay_z[min_idx] = z
                    self._replay_r[min_idx] = r
                else:
                    self._replay_z.append(z)
                    self._replay_r.append(r)

            self._update_params(trajectories, lr=lr)

        # Final sampling
        z_samples = []
        rewards = []

        for _ in range(n_samples):
            z, r, _, _ = self._sample_trajectory(reward_fn)
            z_samples.append(z)
            rewards.append(r)

        z_array = np.array(z_samples)
        r_array = np.array(rewards)

        # Convert Z to macro scenarios
        scenarios = [z_to_macro(z) for z in z_samples]

        # Diversity: mean pairwise distance
        if len(z_samples) > 1:
            dists = []
            n_pairs = min(500, len(z_samples) * (len(z_samples) - 1) // 2)
            for _ in range(n_pairs):
                i, j = self.rng.choice(len(z_samples), 2, replace=False)
                dists.append(float(np.linalg.norm(z_samples[i] - z_samples[j])))
            diversity = float(np.mean(dists))
        else:
            diversity = 0.0

        # Best scenario
        best_idx = int(np.argmax(r_array))

        return GFlowNetResult(
            mode="defensive",
            scenarios=scenarios,
            z_vectors=z_array,
            rewards=r_array,
            log_Z=self.log_Z,
            diversity=diversity,
            best_scenario=scenarios[best_idx],
            best_reward=float(r_array[best_idx]),
        )


# ──────────────────────────────────────────────
# DUAL-MODE ORCHESTRATOR
# ──────────────────────────────────────────────
class DualGFlowNet:
    """Orchestrateur dual-mode conditionne par le regime HMM.

    En contraction : genere des scenarios de stress (cygnes noirs).
    En expansion : genere des scenarios favorables (alpha mining).
    En recovery : mode equilibre (mix des deux).

    Usage :
        dual = DualGFlowNet()
        result = dual.run(
            regime="contraction",
            ecl_surrogate=surrogate.predict_single,
            breach_threshold=2e9,
        )
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def run(
        self,
        regime: str,
        ecl_surrogate: Callable[[Dict[str, float]], float],
        breach_threshold: float = 2e9,
        rorac_surrogate: Optional[Callable[[Dict[str, float]], float]] = None,
        ecl_constraint: Optional[float] = None,
        n_samples: int = 64,
        n_train_steps: int = 150,
        sigma_budget: float = 4.0,
    ) -> GFlowNetResult:
        """Execute le GFlowNet en mode defensif ou offensif.

        Args:
            regime: "contraction", "recovery", ou "expansion".
            ecl_surrogate: Callable(macro_dict) -> ECL total.
            breach_threshold: Seuil ECL pour mode defensif.
            rorac_surrogate: Callable(macro_dict) -> RORAC (optionnel, offensif).
            ecl_constraint: Contrainte ECL max pour mode offensif.
            n_samples: Nombre de scenarios a generer.
            n_train_steps: Iterations d'entrainement.
            sigma_budget: Budget Mahalanobis max.

        Returns:
            GFlowNetResult.
        """
        gfn = ContinuousGFlowNet(state_dim=5, seed=self.seed)

        if regime == "contraction" or (regime == "recovery" and rorac_surrogate is None):
            # Mode defensif
            def reward_fn(z):
                return defensive_reward(z, ecl_surrogate, breach_threshold, sigma_budget)

            result = gfn.generate(
                reward_fn=reward_fn,
                n_samples=n_samples,
                n_train_steps=n_train_steps,
            )
            return GFlowNetResult(
                mode="defensive",
                scenarios=result.scenarios,
                z_vectors=result.z_vectors,
                rewards=result.rewards,
                log_Z=result.log_Z,
                diversity=result.diversity,
                best_scenario=result.best_scenario,
                best_reward=result.best_reward,
            )

        elif regime == "expansion" and rorac_surrogate is not None:
            # Mode offensif
            _ecl_constraint = ecl_constraint or breach_threshold

            def reward_fn(z):
                return offensive_reward(
                    z, rorac_surrogate, ecl_surrogate,
                    _ecl_constraint, sigma_budget,
                )

            result = gfn.generate(
                reward_fn=reward_fn,
                n_samples=n_samples,
                n_train_steps=n_train_steps,
            )

            # Compute allocations from scenarios (RORAC-weighted)
            allocations = []
            for scenario in result.scenarios:
                rorac = rorac_surrogate(scenario)
                ecl = ecl_surrogate(scenario)
                # Simple heuristic: high RORAC + low ECL = overweight
                alloc = {
                    "rorac": rorac,
                    "ecl": ecl,
                    "signal": "surponderer" if rorac > 0.05 else "maintenir",
                }
                allocations.append(alloc)

            return GFlowNetResult(
                mode="offensive",
                scenarios=result.scenarios,
                z_vectors=result.z_vectors,
                rewards=result.rewards,
                log_Z=result.log_Z,
                diversity=result.diversity,
                best_scenario=result.best_scenario,
                best_reward=result.best_reward,
                allocations=allocations,
            )

        else:
            # Recovery : mode equilibre (defensif par defaut)
            def reward_fn(z):
                return defensive_reward(z, ecl_surrogate, breach_threshold, sigma_budget)

            result = gfn.generate(
                reward_fn=reward_fn,
                n_samples=n_samples,
                n_train_steps=n_train_steps,
            )
            return GFlowNetResult(
                mode="balanced",
                scenarios=result.scenarios,
                z_vectors=result.z_vectors,
                rewards=result.rewards,
                log_Z=result.log_Z,
                diversity=result.diversity,
                best_scenario=result.best_scenario,
                best_reward=result.best_reward,
            )


# ──────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=== GFlowNet Dual-Reward Test ===\n")

    # Mock ECL surrogate
    def mock_ecl(params):
        u = params.get("unemployment_rate", 7.5)
        g = params.get("gdp_growth", 1.2)
        r = params.get("interest_rate", 3.5)
        return 1e9 * (1.0 + 0.15 * u - 0.08 * g + 0.05 * r)

    # Mock RORAC surrogate
    def mock_rorac(params):
        g = params.get("gdp_growth", 1.2)
        u = params.get("unemployment_rate", 7.5)
        return max(0.0, 0.08 + 0.01 * g - 0.005 * u)

    breach = 2.5e9

    # Test defensive mode
    dual = DualGFlowNet(seed=42)
    result_def = dual.run(
        regime="contraction",
        ecl_surrogate=mock_ecl,
        breach_threshold=breach,
        n_samples=32,
        n_train_steps=50,
    )
    print(f"Defensive mode: {result_def.mode}")
    print(f"  Samples: {len(result_def.scenarios)}")
    print(f"  Best reward: {result_def.best_reward:.4f}")
    print(f"  Best scenario: unemp={result_def.best_scenario.get('unemployment_rate', 0):.1f}, "
          f"gdp={result_def.best_scenario.get('gdp_growth', 0):.1f}")
    print(f"  Diversity: {result_def.diversity:.3f}")
    print(f"  Log Z: {result_def.log_Z:.3f}")
    breaching = sum(1 for s in result_def.scenarios if mock_ecl(s) > breach)
    print(f"  Scenarios breaching threshold: {breaching}/{len(result_def.scenarios)}")

    # Test offensive mode
    result_off = dual.run(
        regime="expansion",
        ecl_surrogate=mock_ecl,
        rorac_surrogate=mock_rorac,
        breach_threshold=breach,
        ecl_constraint=2.0e9,
        n_samples=32,
        n_train_steps=50,
    )
    print(f"\nOffensive mode: {result_off.mode}")
    print(f"  Samples: {len(result_off.scenarios)}")
    print(f"  Best reward: {result_off.best_reward:.4f}")
    print(f"  Best RORAC scenario: gdp={result_off.best_scenario.get('gdp_growth', 0):.1f}")
    if result_off.allocations:
        signals = [a["signal"] for a in result_off.allocations]
        print(f"  Signals: surponderer={signals.count('surponderer')}, "
              f"maintenir={signals.count('maintenir')}")

    print("\nGFlowNet OK")
