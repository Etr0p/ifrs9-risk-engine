"""Fonctions utilitaires transverses au projet IFRS 9."""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, TypeVar

import numpy as np

from ifrs9_cockpit.config import RANDOM_SEED
from ifrs9_cockpit.utils.logging import get_logger

_logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fixe la graine aléatoire pour numpy et les librairies dépendantes.

    Args:
        seed: Valeur de la graine.
    """
    np.random.seed(seed)


def timer(func: F) -> F:
    """Décorateur mesurant le temps d'exécution d'une fonction.

    Args:
        func: Fonction à chronométrer.

    Returns:
        Fonction wrappée avec logging du temps.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _logger.info("timer", function=func.__qualname__, elapsed_s=f"{elapsed:.3f}")
        return result
    return wrapper  # type: ignore[return-value]


def safe_divide(
    numerator: np.ndarray | float,
    denominator: np.ndarray | float,
    fill: float = 0.0,
) -> np.ndarray | float:
    """Division sécurisée évitant les divisions par zéro.

    Args:
        numerator: Numérateur.
        denominator: Dénominateur.
        fill: Valeur de remplacement si dénominateur nul.

    Returns:
        Résultat de la division, avec fill là où dénominateur est nul.
    """
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else fill
    return np.where(denominator != 0, numerator / denominator, fill)


def format_euro(value: float) -> str:
    """Formate un montant en euros lisible.

    Args:
        value: Montant en euros.

    Returns:
        Chaîne formatée (ex: '1.2M EUR' ou '450K EUR').
    """
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}Md EUR"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M EUR"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}K EUR"
    return f"{value:,.0f} EUR"


def format_pct(value: float, decimals: int = 2) -> str:
    """Formate un ratio en pourcentage.

    Args:
        value: Valeur décimale (ex: 0.0523).
        decimals: Nombre de décimales.

    Returns:
        Chaîne formatée (ex: '5.23%').
    """
    return f"{value * 100:.{decimals}f}%"


def format_bps(value: float) -> str:
    """Formate un ratio en points de base.

    Args:
        value: Valeur décimale.

    Returns:
        Chaîne formatée (ex: '52 bps').
    """
    return f"{value * 10_000:.0f} bps"


def logit(p: np.ndarray | float) -> np.ndarray | float:
    """Calcule le logit (log-odds) d'une probabilite.

    logit(p) = log(p / (1 - p))

    Transformation standard en modelisation credit (Merton-Vasicek).
    Mappe [0, 1] -> R, permettant des operations additives sur les
    probabilites sans risque de debordement.

    Args:
        p: Probabilite(s) dans ]0, 1[.

    Returns:
        Log-odds dans R.
    """
    p_safe = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p_safe / (1 - p_safe))


def expit(x: np.ndarray | float) -> np.ndarray | float:
    """Calcule l'inverse du logit (fonction logistique / sigmoide).

    expit(x) = 1 / (1 + exp(-x))

    Implementation numeriquement stable (evite overflow pour |x| grand).

    Args:
        x: Valeur(s) en espace logit (log-odds).

    Returns:
        Probabilite(s) dans ]0, 1[.
    """
    x = np.asarray(x, dtype=float)
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )
