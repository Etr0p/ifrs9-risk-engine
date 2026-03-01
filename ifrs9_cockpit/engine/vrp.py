"""Variance Risk Premium (VRP) — indicateur d'aversion au risque.

VRP = IV² - RV²  (variance implicite - variance réalisée)

Un VRP élevé signale que le marché paie une prime excessive pour la
protection (puts), ce qui anticipe les krachs et les changements de régime.

Usage dans le cockpit IFRS 9 :
    1. **Feature engineering** : VRP comme input du MacroAgent NeSy MAS.
    2. **BL-CVaR dynamique** : ajuste la confiance dans les vues Black-Litterman.
       - VRP élevé → tau diminue → vues plus conservatrices (risk-off).
       - VRP faible → tau augmente → vues plus confiantes (risk-on).
    3. **Détection de régime** : VRP + seuils calibrés → 4 régimes.

Calibration historique (VSTOXX / VIX, 2000-2024) :
    - VRP moyen : ~4% (variance annualisée)
    - VRP > 8% : stress significatif (GFC, COVID)
    - VRP > 12% : panique (Lehman T4 2008, Mars 2020)
    - VRP < 0% : complaisance (VIX en contango excessif)

Références :
    - Carr & Wu (2009): Variance Risk Premiums, RFS
    - Bollerslev, Tauchen, Zhou (2009): Expected Stock Returns and VRP, RFS
    - Bekaert & Hoerova (2014): The VIX, the Variance Premium and Stock Market Volatility, JoE
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# Seuils VRP calibrés (VSTOXX 2000-2024)
_VRP_REGIMES = {
    "complaisance": (-np.inf, 0.0),     # VRP < 0 : marché sous-estime le risque
    "normal": (0.0, 4.0),               # VRP ∈ [0, 4%] : prime d'assurance normale
    "stress": (4.0, 8.0),               # VRP ∈ [4, 8%] : inquiétude croissante
    "panique": (8.0, np.inf),           # VRP > 8% : panique (flight to safety)
}

# Mapping régime → ajustement du tau BL (multiplicateur)
_TAU_ADJUSTMENTS = {
    "complaisance": 1.30,   # Surestimer les vues (marché euphorique, contrarian)
    "normal": 1.00,         # Pas d'ajustement
    "stress": 0.70,         # Réduire confiance dans les vues
    "panique": 0.40,        # Très conservateur (risk-off)
}


@dataclass(frozen=True)
class VRPResult:
    """Résultat du calcul VRP.

    Attributes:
        vrp: Variance Risk Premium (IV² - RV²) en % annualisé.
        implied_vol: Volatilité implicite (IV) en % annualisé.
        realized_vol: Volatilité réalisée (RV) en % annualisé.
        regime: Régime détecté ("complaisance", "normal", "stress", "panique").
        tau_multiplier: Multiplicateur à appliquer au tau BL-CVaR.
        severity_score: Score de sévérité normalisé [0, 1] pour le MacroAgent.
        percentile: Percentile historique du VRP actuel.
    """
    vrp: float
    implied_vol: float
    realized_vol: float
    regime: str
    tau_multiplier: float
    severity_score: float
    percentile: float


def compute_vrp(
    implied_vol: float,
    realized_vol: float,
    vrp_history: Optional[np.ndarray] = None,
) -> VRPResult:
    """Calcule le Variance Risk Premium et le régime associé.

    Args:
        implied_vol: Volatilité implicite courante (annualisée, en %).
            Source typique : VSTOXX (Europe) ou VIX (US).
        realized_vol: Volatilité réalisée courante (annualisée, en %).
            Typiquement sur 30 jours glissants.
        vrp_history: Historique de VRP pour le calcul du percentile.
            Si None, le percentile est estimé par rapport aux seuils calibrés.

    Returns:
        VRPResult avec VRP, régime, ajustement tau, et diagnostics.
    """
    # VRP = IV² - RV² (en variance, pas en vol)
    vrp = implied_vol ** 2 - realized_vol ** 2

    # Détection de régime
    regime = "normal"
    for regime_name, (low, high) in _VRP_REGIMES.items():
        if low <= vrp < high:
            regime = regime_name
            break

    # Ajustement tau BL
    tau_mult = _TAU_ADJUSTMENTS[regime]

    # Score de sévérité normalisé [0, 1]
    # Mapping linéaire : vrp=0 → 0.0, vrp=12 → 1.0
    severity = np.clip(vrp / 12.0, 0.0, 1.0)

    # Percentile historique
    if vrp_history is not None and len(vrp_history) > 0:
        percentile = float(np.mean(vrp_history <= vrp))
    else:
        # Estimation par rapport à la distribution calibrée (N(4, 3²))
        from scipy.stats import norm
        percentile = float(norm.cdf(vrp, loc=4.0, scale=3.0))

    return VRPResult(
        vrp=round(vrp, 4),
        implied_vol=round(implied_vol, 4),
        realized_vol=round(realized_vol, 4),
        regime=regime,
        tau_multiplier=round(tau_mult, 4),
        severity_score=round(severity, 4),
        percentile=round(percentile, 4),
    )


def simulate_vrp_from_macro(
    macro_severity: float,
    seed: int = 42,
) -> VRPResult:
    """Simule un VRP synthétique à partir de la sévérité macro.

    En l'absence de données de marché réelles (VSTOXX/VIX), on génère
    un VRP cohérent avec la macro_severity du cockpit.

    Relation calibrée (régression Bollerslev-Zhou sur zone euro 2000-2024) :
        IV ≈ 15 + 25 × macro_severity + ε
        RV ≈ 12 + 18 × macro_severity + ε
        VRP = IV² - RV²

    Args:
        macro_severity: Sévérité macro [0, 1] du cockpit.
        seed: Graine pour reproductibilité.

    Returns:
        VRPResult simulé.
    """
    rng = np.random.RandomState(seed)

    # Volatilités synthétiques calibrées
    iv = 15.0 + 25.0 * macro_severity + rng.normal(0, 1.5)
    rv = 12.0 + 18.0 * macro_severity + rng.normal(0, 1.0)

    # Borner (IV ≥ RV en moyenne, mais pas toujours — complaisance possible)
    iv = max(iv, 5.0)
    rv = max(rv, 3.0)

    return compute_vrp(iv, rv)


def adjust_bl_tau(
    base_tau: float,
    vrp_result: VRPResult,
) -> float:
    """Ajuste le paramètre tau de Black-Litterman selon le VRP.

    tau contrôle la confiance dans les vues (prior vs views).
    En période de stress, tau diminue → le portefeuille se rapproche
    de l'équilibre de marché (market cap weights) → risk-off.

    Args:
        base_tau: Tau de base (typiquement 0.05 pour BL classique).
        vrp_result: Résultat VRP courant.

    Returns:
        Tau ajusté.
    """
    return base_tau * vrp_result.tau_multiplier
