"""Path Signatures — compression geometrique de series temporelles.

Implémente les signatures tronquées (Rough Path Theory, Terry Lyons 1998)
en numpy pur pour transformer des trajectoires macro (60 mois, 5 variables)
en vecteurs statiques de taille fixe.

Avantages :
    - Capture l'ordre des événements (contrairement aux statistiques agrégées).
    - Invariant au reparamétrage temporel (seule la géométrie compte).
    - Vecteur de taille fixe → compatible XGBoost, MLP, tout classifieur tabulaire.
    - Rend les RNN/LSTM inutiles pour cette tâche.

Formule (signature tronquée ordre K d'un chemin X: [0,T] → R^d) :
    S(X)^K = (1, S¹, S², ..., Sᴷ)

    S¹_i = ∫ dX^i_t                           (incréments cumulés)
    S²_ij = ∫∫ dX^i_s dX^j_t  (s < t)          (produits itérés)

Dimensions :
    - Ordre 1 : d features
    - Ordre 2 : d² features
    - Total (K=2) : d + d² = 5 + 25 = 30 features pour 5 variables macro

Références :
    - Lyons (1998): Differential Equations Driven by Rough Signals
    - Chevyrev & Kormilitzin (2016): A Primer on the Signature Method
    - Kidger & Lyons (2021): Signatory: differentiable computations of the signature
    - Morrill et al. (2021): A Generalised Signature Method for Multivariate Time Series
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SignatureResult:
    """Résultat du calcul de signature.

    Attributes:
        features: Vecteur de signature (taille fixe).
        feature_names: Noms des features pour interprétabilité.
        order: Ordre de troncature.
        n_features: Nombre total de features.
        path_length: Longueur du chemin (nombre de pas de temps).
        n_dims: Dimensionnalité du chemin.
    """
    features: np.ndarray
    feature_names: List[str]
    order: int
    n_features: int
    path_length: int
    n_dims: int


def _compute_signature_order1(increments: np.ndarray) -> np.ndarray:
    """Signature d'ordre 1 : intégrales de Chen (incréments cumulés).

    S¹_i = Σ_t ΔX^i_t

    Args:
        increments: Array (T-1, d) des incréments ΔX_t = X_{t+1} - X_t.

    Returns:
        Array (d,) de la signature d'ordre 1.
    """
    return np.sum(increments, axis=0)


def _compute_signature_order2(increments: np.ndarray) -> np.ndarray:
    """Signature d'ordre 2 : produits itérés de Chen.

    S²_ij = Σ_{s<t} ΔX^i_s × ΔX^j_t

    Calcul efficace via somme cumulée :
        S²_ij = Σ_t ΔX^j_t × (Σ_{s<t} ΔX^i_s)

    Args:
        increments: Array (T-1, d) des incréments.

    Returns:
        Array (d, d) = (d²,) aplati de la signature d'ordre 2.
    """
    T, d = increments.shape
    cumsum = np.cumsum(increments, axis=0)

    # Somme cumulée décalée (s < t strict)
    # cumsum_shifted[t] = Σ_{s=0}^{t-1} ΔX^i_s
    cumsum_shifted = np.zeros_like(cumsum)
    cumsum_shifted[1:] = cumsum[:-1]

    # S²_ij = Σ_t ΔX^j_t × cumsum_shifted^i_t
    s2 = np.zeros((d, d))
    for t in range(T):
        s2 += np.outer(cumsum_shifted[t], increments[t])

    return s2.flatten()


def _compute_signature_order3(increments: np.ndarray) -> np.ndarray:
    """Signature d'ordre 3 : triple produits itérés.

    S³_ijk = Σ_{r<s<t} ΔX^i_r × ΔX^j_s × ΔX^k_t

    Plus coûteux (O(T² × d³)), on utilise une approximation efficace.

    Args:
        increments: Array (T-1, d) des incréments.

    Returns:
        Array (d³,) de la signature d'ordre 3.
    """
    T, d = increments.shape
    s3 = np.zeros((d, d, d))

    # Accumulateurs
    cum1 = np.zeros(d)  # Σ_{r<s} ΔX^i_r
    cum2 = np.zeros((d, d))  # Σ_{r<s<t} ΔX^i_r × ΔX^j_s

    for t in range(T):
        # S³ += cum2 ⊗ ΔX_t
        for k in range(d):
            s3[:, :, k] += cum2 * increments[t, k]
        # Mise à jour cum2 += cum1 ⊗ ΔX_t
        for j in range(d):
            cum2[:, j] += cum1 * increments[t, j]
        # Mise à jour cum1
        cum1 += increments[t]

    return s3.flatten()


def compute_path_signature(
    path: np.ndarray,
    order: int = 2,
    normalize: bool = True,
    var_names: Optional[List[str]] = None,
) -> SignatureResult:
    """Calcule la signature tronquée d'un chemin multidimensionnel.

    Args:
        path: Array (T, d) — série temporelle multivariée.
            T = nombre de pas de temps, d = nombre de variables.
        order: Ordre de troncature (1, 2, ou 3).
        normalize: Si True, normalise par la longueur du chemin.
        var_names: Noms des variables. Si None, utilise x0, x1, ...

    Returns:
        SignatureResult avec le vecteur de features et les métadonnées.

    Raises:
        ValueError: Si l'ordre n'est pas dans {1, 2, 3} ou le chemin trop court.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim == 1:
        path = path.reshape(-1, 1)
    T, d = path.shape

    if T < 2:
        raise ValueError(f"Chemin trop court: T={T} < 2 minimum")
    if order not in (1, 2, 3):
        raise ValueError(f"Ordre doit etre 1, 2 ou 3, recu {order}")

    if var_names is None:
        var_names = [f"x{i}" for i in range(d)]

    # Incréments
    increments = np.diff(path, axis=0)  # (T-1, d)

    # Normalisation optionnelle (stabilise les échelles)
    if normalize and T > 2:
        scale = np.std(increments, axis=0)
        scale = np.where(scale > 1e-10, scale, 1.0)
        increments = increments / scale

    # Ordre 1
    s1 = _compute_signature_order1(increments)
    features = [s1]
    names = [f"sig1_{var_names[i]}" for i in range(d)]

    # Ordre 2
    if order >= 2:
        s2 = _compute_signature_order2(increments)
        features.append(s2)
        for i in range(d):
            for j in range(d):
                names.append(f"sig2_{var_names[i]}_{var_names[j]}")

    # Ordre 3
    if order >= 3:
        s3 = _compute_signature_order3(increments)
        features.append(s3)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    names.append(f"sig3_{var_names[i]}_{var_names[j]}_{var_names[k]}")

    feature_vector = np.concatenate(features)

    return SignatureResult(
        features=feature_vector,
        feature_names=names,
        order=order,
        n_features=len(feature_vector),
        path_length=T,
        n_dims=d,
    )


def compute_macro_signatures(
    macro_history: Dict[str, List[float]],
    order: int = 2,
    normalize: bool = True,
) -> SignatureResult:
    """Calcule les signatures à partir de l'historique macro du cockpit.

    Prend le dict MACRO_HISTORY_BASELINE (5 variables × 60 mois)
    et retourne un vecteur de features statiques.

    Args:
        macro_history: Dict {var_name: [v_1, ..., v_T]} historique macro.
        order: Ordre de troncature (défaut: 2 → 30 features).
        normalize: Normaliser les incréments.

    Returns:
        SignatureResult avec features pour XGBoost/agents.
    """
    _MACRO_VARS = [
        "unemployment_rate",
        "gdp_growth",
        "interest_rate",
        "hpi_growth",
        "inflation_rate",
    ]

    var_names = [v for v in _MACRO_VARS if v in macro_history]
    if not var_names:
        raise ValueError("Aucune variable macro trouvee dans l'historique")

    # Construire la matrice (T, d)
    arrays = [np.array(macro_history[v]) for v in var_names]
    T = min(len(a) for a in arrays)
    path = np.column_stack([a[:T] for a in arrays])

    # Labels courts pour les features
    _SHORT = {
        "unemployment_rate": "unemp",
        "gdp_growth": "gdp",
        "interest_rate": "ir",
        "hpi_growth": "hpi",
        "inflation_rate": "infl",
    }
    short_names = [_SHORT.get(v, v) for v in var_names]

    return compute_path_signature(path, order=order, normalize=normalize, var_names=short_names)


def rolling_signatures(
    path: np.ndarray,
    window: int = 12,
    order: int = 2,
    var_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Calcule les signatures sur fenêtres glissantes.

    Pour chaque pas de temps t, calcule la signature du sous-chemin
    [t-window+1, ..., t]. Utile pour le suivi temporel de la dynamique.

    Args:
        path: Array (T, d).
        window: Taille de la fenêtre glissante.
        order: Ordre de troncature.
        var_names: Noms des variables.

    Returns:
        Array (T - window + 1, n_features) des signatures glissantes.
    """
    T, d = path.shape
    if T < window:
        raise ValueError(f"Chemin trop court ({T}) pour fenetre {window}")

    results = []
    for t in range(window, T + 1):
        sub_path = path[t - window:t]
        sig = compute_path_signature(sub_path, order=order, normalize=True, var_names=var_names)
        results.append(sig.features)

    return np.array(results)
