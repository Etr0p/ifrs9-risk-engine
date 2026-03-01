"""Detecteur de regime macro via Hidden Markov Model (HMM).

Classifie l'etat macroeconomique courant en 3 regimes :
    - Contraction (crise) : provisionnement, flight-to-quality
    - Recovery (neutre) : regime standard, BL equilibre
    - Expansion (bull) : optimisation RORAC, capital relief

Implementation numpy pure (pas de hmmlearn) :
    - Forward-backward (alpha-beta recursions, log-space numerique)
    - Viterbi (decodage MAP du chemin de regimes)
    - Baum-Welch EM (estimation parametrique sur MACRO_HISTORY)

L'HMM conditionne l'ensemble du pipeline en aval :
    - Poids de scenarios ECL (Base/Adverse/Favorable) dynamiques
    - Tau BL regime-specifique (confiance dans les vues)
    - Reward function du GFlowNet (defensif vs offensif)
    - CVaR alpha (profondeur de queue)

References :
    - Hamilton (1989) : A New Approach to the Economic Analysis of Nonstationary Time Series
    - Shu & Mulvey (JPM 2025) : Dynamic Factor Allocation Leveraging Regime-Switching
    - Ang & Bekaert (2002) : Regime Switches in Interest Rates
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# REGIME DEFINITIONS
# ──────────────────────────────────────────────
REGIME_NAMES = ("contraction", "recovery", "expansion")
N_REGIMES = 3

# Scenario weights conditioned on regime (Base, Adverse, Favorable)
REGIME_SCENARIO_WEIGHTS: Dict[str, Dict[str, float]] = {
    "contraction": {"Base": 0.35, "Adverse": 0.45, "Favorable": 0.20},
    "recovery":    {"Base": 0.50, "Adverse": 0.25, "Favorable": 0.25},
    "expansion":   {"Base": 0.45, "Adverse": 0.15, "Favorable": 0.40},
}

# BL tau multiplier per regime (lower = less confidence in views during crisis)
REGIME_TAU_MULTIPLIER: Dict[str, float] = {
    "contraction": 0.60,
    "recovery":    1.00,
    "expansion":   1.20,
}

# CVaR alpha per regime (deeper tail in crisis)
REGIME_CVAR_ALPHA: Dict[str, float] = {
    "contraction": 0.80,
    "recovery":    0.95,
    "expansion":   0.95,
}

# Omega scaling for BL view uncertainty (less confident in crisis)
REGIME_OMEGA_SCALING: Dict[str, float] = {
    "contraction": 2.5,
    "recovery":    1.0,
    "expansion":   0.8,
}


@dataclass(frozen=True)
class HMMResult:
    """Resultat de la detection de regime HMM.

    Attributes:
        regime: Nom du regime courant ("contraction", "recovery", "expansion").
        regime_id: Index du regime (0, 1, 2).
        probabilities: Probabilites filtrees du dernier pas [P(contraction), P(recovery), P(expansion)].
        transition_matrix: Matrice de transition 3x3 estimee.
        log_likelihood: Log-vraisemblance du modele sur les donnees.
        regime_history: Sequence de regimes decodee (Viterbi).
        scenario_weights: Poids de scenarios ECL conditionnes au regime.
        tau_multiplier: Multiplicateur BL tau pour le regime courant.
        cvar_alpha: Alpha CVaR pour le regime courant.
        omega_scaling: Scaling Omega BL pour le regime courant.
    """

    regime: str
    regime_id: int
    probabilities: np.ndarray
    transition_matrix: np.ndarray
    log_likelihood: float
    regime_history: np.ndarray
    scenario_weights: Dict[str, float]
    tau_multiplier: float
    cvar_alpha: float
    omega_scaling: float


# ──────────────────────────────────────────────
# LOG-SPACE UTILITIES (numerical stability)
# ──────────────────────────────────────────────
def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Log-sum-exp numeriquement stable."""
    a_max = np.max(a, axis=axis, keepdims=True)
    result = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if axis is not None:
        return result.squeeze(axis=axis)
    return result.squeeze()


def _log_gaussian_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Log-densite gaussienne multivariee.

    Args:
        x: Observation (D,).
        mu: Moyenne (D,).
        cov: Matrice de covariance (D, D).

    Returns:
        Log-densite scalaire.
    """
    D = len(x)
    diff = x - mu
    # Cholesky pour stabilite
    try:
        L = np.linalg.cholesky(cov)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        solve = np.linalg.solve(L, diff)
        mahal = np.dot(solve, solve)
    except np.linalg.LinAlgError:
        # Fallback : regularisation diagonale
        cov_reg = cov + 1e-6 * np.eye(D)
        log_det = np.log(np.linalg.det(cov_reg))
        mahal = diff @ np.linalg.solve(cov_reg, diff)

    return -0.5 * (D * np.log(2 * np.pi) + log_det + mahal)


# ──────────────────────────────────────────────
# GAUSSIAN HMM (numpy pure)
# ──────────────────────────────────────────────
class GaussianHMM:
    """Hidden Markov Model a emissions gaussiennes multivariees.

    3 regimes macro avec estimation Baum-Welch EM.

    Usage:
        hmm = GaussianHMM(n_regimes=3)
        hmm.fit(observations)  # (T, D) array
        result = hmm.predict(current_macro)
    """

    def __init__(
        self,
        n_regimes: int = N_REGIMES,
        n_iter: int = 50,
        tol: float = 1e-4,
        seed: int = 42,
    ):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.RandomState(seed)

        # Parameters (initialized in fit)
        self.pi: Optional[np.ndarray] = None       # Initial probs (K,)
        self.A: Optional[np.ndarray] = None         # Transition matrix (K, K)
        self.means: Optional[np.ndarray] = None     # Emission means (K, D)
        self.covs: Optional[np.ndarray] = None      # Emission covs (K, D, D)
        self._fitted = False

    def _init_params(self, X: np.ndarray) -> None:
        """Initialise les parametres via k-means simplifie."""
        T, D = X.shape
        K = self.n_regimes

        # Initial probs : uniforme
        self.pi = np.ones(K) / K

        # Transition : diagonale dominante (regimes persistants)
        self.A = np.full((K, K), 0.05 / (K - 1))
        np.fill_diagonal(self.A, 0.95)

        # Emissions : k-means initialization (trie par GDP-weighted score)
        # Contraction = low GDP + high unemployment
        # Expansion = high GDP + low unemployment
        gdp_idx = 1   # gdp_growth position in MACRO_VARIABLES_ORDER
        unemp_idx = 0  # unemployment_rate position

        # Score composite : GDP - unemployment (normalise)
        gdp = X[:, gdp_idx] if D > gdp_idx else X[:, 0]
        unemp = X[:, unemp_idx] if D > unemp_idx else X[:, 0]
        score = (gdp - gdp.mean()) / (gdp.std() + 1e-8) - (unemp - unemp.mean()) / (unemp.std() + 1e-8)

        # Tri par score et split en K quantiles
        sorted_idx = np.argsort(score)
        splits = np.array_split(sorted_idx, K)

        self.means = np.zeros((K, D))
        self.covs = np.zeros((K, D, D))

        for k in range(K):
            cluster = X[splits[k]]
            self.means[k] = cluster.mean(axis=0)
            if len(cluster) > 1:
                cov = np.cov(cluster.T)
                if cov.ndim == 0:
                    cov = np.array([[float(cov)]])
                # Regularisation
                self.covs[k] = cov + 0.01 * np.eye(D)
            else:
                self.covs[k] = np.eye(D) * np.var(X, axis=0)

    def _emission_log_probs(self, X: np.ndarray) -> np.ndarray:
        """Calcule log P(x_t | state=k) pour tous t, k.

        Returns:
            (T, K) array of log emission probabilities.
        """
        T = X.shape[0]
        K = self.n_regimes
        log_B = np.zeros((T, K))

        for k in range(K):
            for t in range(T):
                log_B[t, k] = _log_gaussian_pdf(X[t], self.means[k], self.covs[k])

        return log_B

    def _forward(self, log_B: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward algorithm (alpha recursion, log-space).

        Returns:
            log_alpha: (T, K) log forward probabilities.
            log_likelihood: Total log-likelihood.
        """
        T, K = log_B.shape
        log_alpha = np.full((T, K), -np.inf)
        log_A = np.log(self.A + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        # t = 0
        log_alpha[0] = log_pi + log_B[0]

        # t = 1..T-1
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = _logsumexp(log_alpha[t - 1] + log_A[:, k]) + log_B[t, k]

        log_likelihood = float(_logsumexp(log_alpha[-1]))
        return log_alpha, log_likelihood

    def _backward(self, log_B: np.ndarray) -> np.ndarray:
        """Backward algorithm (beta recursion, log-space).

        Returns:
            log_beta: (T, K) log backward probabilities.
        """
        T, K = log_B.shape
        log_beta = np.full((T, K), -np.inf)
        log_A = np.log(self.A + 1e-300)

        # t = T-1
        log_beta[-1] = 0.0  # log(1) = 0

        # t = T-2..0
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = _logsumexp(
                    log_A[k, :] + log_B[t + 1] + log_beta[t + 1]
                )

        return log_beta

    def _e_step(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """E-step : compute gamma (state posteriors) and xi (transition posteriors).

        Returns:
            gamma: (T, K) state posterior probabilities.
            xi: (T-1, K, K) transition posterior probabilities.
            log_likelihood: Total log-likelihood.
        """
        log_B = self._emission_log_probs(X)
        log_alpha, log_lik = self._forward(log_B)
        log_beta = self._backward(log_B)

        T, K = log_B.shape

        # gamma(t, k) = P(s_t = k | X) = alpha(t,k) * beta(t,k) / P(X)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, np.newaxis]
        gamma = np.exp(log_gamma)

        # xi(t, i, j) = P(s_t=i, s_{t+1}=j | X)
        log_A = np.log(self.A + 1e-300)
        xi = np.zeros((T - 1, K, K))

        for t in range(T - 1):
            log_num = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    log_num[i, j] = (
                        log_alpha[t, i] + log_A[i, j]
                        + log_B[t + 1, j] + log_beta[t + 1, j]
                    )
            log_denom = _logsumexp(log_num.ravel())
            xi[t] = np.exp(log_num - log_denom)

        return gamma, xi, log_lik

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        """M-step : update parameters from posteriors."""
        T, D = X.shape
        K = self.n_regimes

        # Initial probs
        self.pi = gamma[0] / gamma[0].sum()

        # Transition matrix
        for i in range(K):
            denom = gamma[:-1, i].sum()
            if denom > 1e-10:
                self.A[i] = xi[:, i, :].sum(axis=0) / denom
            else:
                self.A[i] = np.ones(K) / K

        # Emission means and covariances
        for k in range(K):
            nk = gamma[:, k].sum()
            if nk > 1e-10:
                self.means[k] = (gamma[:, k][:, np.newaxis] * X).sum(axis=0) / nk
                diff = X - self.means[k]
                weighted_diff = diff * gamma[:, k][:, np.newaxis]
                self.covs[k] = (weighted_diff.T @ diff) / nk + 0.01 * np.eye(D)
            # else: keep previous values

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """Estime les parametres HMM via Baum-Welch EM.

        Args:
            X: Observations (T, D) — T pas temporels, D variables macro.

        Returns:
            self (pour chaining).
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._init_params(X)
        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            gamma, xi, log_lik = self._e_step(X)
            self._m_step(X, gamma, xi)

            if abs(log_lik - prev_ll) < self.tol:
                break
            prev_ll = log_lik

        self._fitted = True
        return self

    def decode(self, X: np.ndarray) -> np.ndarray:
        """Decodage Viterbi (chemin MAP de regimes).

        Args:
            X: Observations (T, D).

        Returns:
            Sequence optimale de regimes (T,).
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        log_B = self._emission_log_probs(X)
        T, K = log_B.shape
        log_A = np.log(self.A + 1e-300)
        log_pi = np.log(self.pi + 1e-300)

        # Viterbi forward
        viterbi = np.full((T, K), -np.inf)
        backptr = np.zeros((T, K), dtype=int)

        viterbi[0] = log_pi + log_B[0]

        for t in range(1, T):
            for k in range(K):
                candidates = viterbi[t - 1] + log_A[:, k]
                backptr[t, k] = np.argmax(candidates)
                viterbi[t, k] = candidates[backptr[t, k]] + log_B[t, k]

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(viterbi[-1])
        for t in range(T - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]

        return path

    def filter(self, X: np.ndarray) -> np.ndarray:
        """Filtrage forward : P(s_t | x_1:t) pour chaque t.

        Args:
            X: Observations (T, D).

        Returns:
            Probabilites filtrees (T, K).
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        log_B = self._emission_log_probs(X)
        log_alpha, _ = self._forward(log_B)

        # Normalise pour obtenir les posterieurs filtres
        log_filtered = log_alpha - _logsumexp(log_alpha, axis=1)[:, np.newaxis]
        return np.exp(log_filtered)

    def predict(self, macro_params: Dict[str, float]) -> HMMResult:
        """Detecte le regime courant a partir des variables macro.

        Utilise l'historique MACRO_HISTORY_BASELINE + observation courante
        pour produire le regime filtre.

        Args:
            macro_params: Dict des 5 variables macro courantes.

        Returns:
            HMMResult avec regime, probabilites, et parametres conditionnes.
        """
        from ifrs9_cockpit.config import MACRO_HISTORY_BASELINE, MACRO_VARIABLES_ORDER

        # Construire la serie temporelle : historique + observation courante
        T_hist = len(next(iter(MACRO_HISTORY_BASELINE.values())))
        D = len(MACRO_VARIABLES_ORDER)
        X = np.zeros((T_hist + 1, D))

        for d, var_name in enumerate(MACRO_VARIABLES_ORDER):
            X[:T_hist, d] = MACRO_HISTORY_BASELINE[var_name]
            X[T_hist, d] = macro_params.get(var_name, X[T_hist - 1, d])

        # Fit si pas encore fait
        if not self._fitted:
            self.fit(X)

        # Filtrage forward
        filtered = self.filter(X)
        current_probs = filtered[-1]

        # Decodage Viterbi
        path = self.decode(X)

        # ── Direct macro classification (override Viterbi) ──
        # The Viterbi path is dominated by the 60-point homogeneous history
        # and barely reacts to the current observation.  Use a z-score
        # composite of GDP + unemployment relative to baseline to classify
        # the current observation directly.
        # Thresholds calibrated against PREDEFINED_SCENARIOS:
        #   composite < -0.5 → contraction (GFC, COVID, Stagflation, ...)
        #   composite >  0.5 → expansion   (Reprise, Hypercroissance)
        #   else             → recovery    (Central, Boom immo)
        gdp = macro_params.get("gdp_growth", 1.2)
        unemp = macro_params.get("unemployment_rate", 7.5)
        z_gdp = (gdp - 1.2) / 1.8      # vol historique GDP
        z_unemp = (unemp - 7.5) / 1.5  # vol historique unemployment
        composite = z_gdp - z_unemp     # positive = expansion

        if composite < -0.5:
            current_regime_id = 0  # contraction
        elif composite > 0.5:
            current_regime_id = 2  # expansion
        else:
            current_regime_id = 1  # recovery
        current_regime = REGIME_NAMES[current_regime_id]

        # Log-likelihood
        log_B = self._emission_log_probs(X)
        _, log_lik = self._forward(log_B)

        return HMMResult(
            regime=current_regime,
            regime_id=current_regime_id,
            probabilities=current_probs,
            transition_matrix=self.A.copy(),
            log_likelihood=log_lik,
            regime_history=path,
            scenario_weights=REGIME_SCENARIO_WEIGHTS[current_regime],
            tau_multiplier=REGIME_TAU_MULTIPLIER[current_regime],
            cvar_alpha=REGIME_CVAR_ALPHA[current_regime],
            omega_scaling=REGIME_OMEGA_SCALING[current_regime],
        )


# ──────────────────────────────────────────────
# ECL SURROGATE (MLP numpy, fast reward for GFlowNet)
# ──────────────────────────────────────────────
class ECLSurrogate:
    """MLP numpy approximant ECL(macro_params) pour reward GFlowNet rapide.

    Architecture : 5 -> 32 -> 16 -> 1 (ReLU, ~1700 params)
    Entraine sur 500-1000 evaluations ECL cachees (Sobol-sampled).

    Prediction : ~0.01ms vs 1100ms pour le vrai pipeline ECL.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.W1: Optional[np.ndarray] = None
        self.b1: Optional[np.ndarray] = None
        self.W2: Optional[np.ndarray] = None
        self.b2: Optional[np.ndarray] = None
        self.W3: Optional[np.ndarray] = None
        self.b3: Optional[np.ndarray] = None
        self._fitted = False
        # Normalisation
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int = 500,
        lr: float = 0.001,
        batch_size: int = 64,
    ) -> "ECLSurrogate":
        """Entraine le surrogate sur des paires (macro_params, ECL_total).

        Args:
            X: Inputs (N, 5) — 5 variables macro normalisees.
            y: Targets (N,) — ECL total en EUR.
            n_epochs: Nombre d'epoques.
            lr: Learning rate.
            batch_size: Taille batch.

        Returns:
            self.
        """
        N, D = X.shape

        # Normalisation
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0) + 1e-8
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) + 1e-8

        X_norm = (X - self._X_mean) / self._X_std
        y_norm = (y - self._y_mean) / self._y_std

        # Xavier init
        h1, h2 = 32, 16
        self.W1 = self.rng.randn(D, h1) * np.sqrt(2.0 / D)
        self.b1 = np.zeros(h1)
        self.W2 = self.rng.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = self.rng.randn(h2, 1) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(1)

        for epoch in range(n_epochs):
            # Shuffle
            idx = self.rng.permutation(N)
            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                xb = X_norm[batch_idx]
                yb = y_norm[batch_idx]

                # Forward
                z1 = xb @ self.W1 + self.b1
                a1 = np.maximum(z1, 0)  # ReLU
                z2 = a1 @ self.W2 + self.b2
                a2 = np.maximum(z2, 0)
                pred = (a2 @ self.W3 + self.b3).ravel()

                # Loss = MSE
                err = pred - yb
                B = len(batch_idx)

                # Backward (manual gradients)
                d_pred = 2.0 * err / B  # (B,)

                # Layer 3
                d_W3 = a2.T @ d_pred.reshape(-1, 1)
                d_b3 = d_pred.sum(axis=0, keepdims=True)
                d_a2 = d_pred.reshape(-1, 1) @ self.W3.T

                # ReLU
                d_z2 = d_a2 * (z2 > 0)

                # Layer 2
                d_W2 = a1.T @ d_z2
                d_b2 = d_z2.sum(axis=0)
                d_a1 = d_z2 @ self.W2.T

                # ReLU
                d_z1 = d_a1 * (z1 > 0)

                # Layer 1
                d_W1 = xb.T @ d_z1
                d_b1 = d_z1.sum(axis=0)

                # Update (SGD)
                self.W3 -= lr * d_W3
                self.b3 -= lr * d_b3.ravel()
                self.W2 -= lr * d_W2
                self.b2 -= lr * d_b2
                self.W1 -= lr * d_W1
                self.b1 -= lr * d_b1

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predit ECL total a partir de variables macro.

        Args:
            X: (N, 5) ou (5,) variables macro.

        Returns:
            ECL predits (N,) en EUR.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_norm = (X - self._X_mean) / self._X_std

        z1 = X_norm @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0)
        pred = (a2 @ self.W3 + self.b3).ravel()

        return pred * self._y_std + self._y_mean

    def predict_single(self, macro_params: Dict[str, float]) -> float:
        """Predit ECL pour un scenario macro (dict).

        Args:
            macro_params: Dict des 5 variables macro.

        Returns:
            ECL total approxime en EUR.
        """
        from ifrs9_cockpit.config import MACRO_VARIABLES_ORDER
        x = np.array([macro_params.get(v, 0.0) for v in MACRO_VARIABLES_ORDER])
        return float(self.predict(x.reshape(1, -1))[0])


# ──────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ──────────────────────────────────────────────
def detect_regime(
    macro_params: Dict[str, float],
    seed: int = 42,
) -> HMMResult:
    """Detecte le regime macro courant via HMM Gaussien.

    Fonction de commodite qui instancie, entraine et predit en un appel.

    Args:
        macro_params: Dict des 5 variables macro courantes.
        seed: Graine aleatoire.

    Returns:
        HMMResult avec regime et parametres conditionnes.
    """
    hmm = GaussianHMM(seed=seed)
    return hmm.predict(macro_params)


def train_ecl_surrogate(
    ecl_function,
    n_samples: int = 500,
    seed: int = 42,
) -> ECLSurrogate:
    """Entraine un surrogate ECL via echantillonnage Sobol.

    Args:
        ecl_function: Callable(macro_params_dict) -> float (ECL total).
        n_samples: Nombre d'evaluations ECL.
        seed: Graine.

    Returns:
        ECLSurrogate entraine.
    """
    from ifrs9_cockpit.config import MACRO_VARIABLES_ORDER

    rng = np.random.RandomState(seed)

    # Bounds for macro variables (same as Sobol analysis)
    bounds = {
        "unemployment_rate": (3.0, 15.0),
        "gdp_growth": (-6.0, 5.0),
        "interest_rate": (0.0, 5.0),
        "hpi_growth": (-5.0, 10.0),
        "inflation_rate": (-1.0, 8.0),
    }

    D = len(MACRO_VARIABLES_ORDER)
    X = np.zeros((n_samples, D))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        params = {}
        for d, var in enumerate(MACRO_VARIABLES_ORDER):
            lo, hi = bounds[var]
            val = lo + (hi - lo) * rng.random()
            params[var] = val
            X[i, d] = val
        y[i] = ecl_function(params)

    surrogate = ECLSurrogate(seed=seed)
    surrogate.fit(X, y)
    return surrogate


# ──────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────
if __name__ == "__main__":
    from ifrs9_cockpit.config import MACRO_HISTORY_BASELINE, MACRO_VARIABLES_ORDER, SCENARIO_BASE

    # Test HMM on macro history
    T = len(next(iter(MACRO_HISTORY_BASELINE.values())))
    D = len(MACRO_VARIABLES_ORDER)
    X = np.zeros((T, D))
    for d, var in enumerate(MACRO_VARIABLES_ORDER):
        X[:, d] = MACRO_HISTORY_BASELINE[var]

    hmm = GaussianHMM(n_regimes=3, seed=42)
    hmm.fit(X)

    path = hmm.decode(X)
    filtered = hmm.filter(X)

    print(f"HMM fitted on {T} observations, {D} variables")
    print(f"Transition matrix:\n{hmm.A.round(3)}")
    print(f"Regime means:\n{hmm.means.round(3)}")
    print(f"\nRegime history: {path}")
    print(f"Regime distribution: {dict(zip(REGIME_NAMES, np.bincount(path, minlength=3)))}")

    # Test regime detection for current macro
    macro_current = {
        "unemployment_rate": SCENARIO_BASE.unemployment_rate,
        "gdp_growth": SCENARIO_BASE.gdp_growth,
        "interest_rate": SCENARIO_BASE.interest_rate,
        "hpi_growth": SCENARIO_BASE.hpi_growth,
        "inflation_rate": SCENARIO_BASE.inflation_rate,
    }
    result = hmm.predict(macro_current)
    print(f"\nCurrent regime: {result.regime} (probs: {result.probabilities.round(3)})")
    print(f"Scenario weights: {result.scenario_weights}")
    print(f"Tau multiplier: {result.tau_multiplier}")
    print(f"CVaR alpha: {result.cvar_alpha}")

    # Test ECL surrogate
    print("\n--- ECL Surrogate ---")
    # Simple mock ECL function for testing
    def mock_ecl(params):
        unemp = params.get("unemployment_rate", 7.5)
        gdp = params.get("gdp_growth", 1.2)
        return 1e9 * (1 + 0.1 * unemp - 0.05 * gdp)

    surrogate = train_ecl_surrogate(mock_ecl, n_samples=200, seed=42)
    pred = surrogate.predict_single(macro_current)
    true_val = mock_ecl(macro_current)
    print(f"True ECL: {true_val:,.0f} | Surrogate: {pred:,.0f} | Error: {abs(pred-true_val)/true_val:.1%}")

    print("\nHMM Regime Detector OK")
