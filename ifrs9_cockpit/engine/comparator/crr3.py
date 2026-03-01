"""CRR3 Risk Weight composite (Art. 133) pour PE."""

from __future__ import annotations

import numpy as np
import polars as pl

from ifrs9_cockpit.config import BASEL_CONFIG


def compute_crr3_rw(result_pe: pl.DataFrame) -> np.ndarray:
    """Score CRR3 composite (Art. 133) pour Risk Weight PE.

    Combine 4 dimensions de risque pour attribuer un RW gradue :
        - Performance financiere (MOIC)
        - Qualite du portefeuille (P(distress))
        - Risque de levier
        - Environnement macro

    Score = 0.30 x financial_perf + 0.25 x portfolio_quality
          + 0.25 x leverage_risk + 0.20 x macro_env

    Classification CRR3 :
        - score < 0.25 -> 190% (IRB diversifie)
        - score < 0.50 -> 250% (equity general)
        - score >= 0.50 -> 400% (speculatif)

    Args:
        result_pe: DataFrame PE avec colonnes moic, nav, p_distress/distress_prob,
                   leverage, risk_category.

    Returns:
        Array de Risk Weights (190, 250 ou 400) par position.
    """
    n = len(result_pe)

    # Performance financiere (MOIC-based)
    if "moic" in result_pe.columns:
        moic = result_pe["moic"].to_numpy().astype(float)
    else:
        moic = np.ones(n)

    if "nav_initial" in result_pe.columns:
        nav_initial = result_pe["nav_initial"].to_numpy().astype(float)
    elif "capital_invested" in result_pe.columns:
        nav_initial = result_pe["capital_invested"].to_numpy().astype(float)
    else:
        nav_initial = result_pe["nav"].to_numpy().astype(float)

    effective_moic = np.where(
        moic > 0,
        moic,
        result_pe["nav"].to_numpy() / np.maximum(nav_initial, 1.0),
    )
    financial_perf = np.where(
        effective_moic < 1.5,
        np.clip(1.0 - effective_moic, 0, 1),
        0.0,
    )

    # Qualite du portefeuille (P(distress))
    if "p_distress" in result_pe.columns:
        portfolio_quality = result_pe["p_distress"].to_numpy().astype(float)
    elif "distress_prob" in result_pe.columns:
        portfolio_quality = result_pe["distress_prob"].to_numpy().astype(float)
    else:
        portfolio_quality = np.zeros(n)

    # Risque de levier (normalise par 0.95)
    if "leverage" in result_pe.columns:
        leverage = result_pe["leverage"].to_numpy().astype(float)
    else:
        leverage = np.full(n, 0.5)
    leverage_risk = np.clip(leverage / 0.95, 0, 1)

    # Environnement macro : combine leverage et financial_perf comme proxy
    # distinct de portfolio_quality pour eviter le double-comptage de P(distress)
    macro_env = np.clip(0.5 * leverage_risk + 0.5 * financial_perf, 0, 1)

    score = (
        0.30 * financial_perf
        + 0.25 * portfolio_quality
        + 0.25 * leverage_risk
        + 0.20 * macro_env
    )

    rw = np.where(score < 0.25, 190, np.where(score < 0.50, 250, 400))
    return rw
