"""Couche 1 — Croisement : asymetries secteur x canal (FR26).

Scanne les asymetries credit vs PE par paire secteur x variable macro
et calcule les contributions marginales au risque.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from ifrs9_cockpit.config import SECTORS, SCENARIO_BASE


def analyze_crossings(
    result_credit: pd.DataFrame,
    result_pe: pd.DataFrame,
    macro_params: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scanne les asymetries credit vs PE (FR26).

    Args:
        result_credit: Resultat ECLCalculator.
        result_pe: Resultat PECalculator.
        macro_params: Variables macro actuelles (5 cles).

    Returns:
        Tuple (asymmetry_matrix, marginal_contributions).
        - asymmetry_matrix : 5 secteurs x metriques comparatives
        - marginal_contributions : 10 cellules (secteur x canal) avec contribution marginale
    """
    # ── Matrice d'asymetrie ──
    asym_records = []
    for sector in SECTORS:
        name = sector.name
        mask_c = result_credit["sector"].values == name
        mask_p = result_pe["sector"].values == name

        ecl = result_credit.loc[mask_c, "ecl_weighted"].sum()
        ead = result_credit.loc[mask_c, "ead"].sum()
        el_pe = result_pe.loc[mask_p, "expected_loss_pe"].sum()
        nav = result_pe.loc[mask_p, "nav"].sum()

        # Taux de perte normalise
        loss_rate_credit = ecl / max(ead, 1)
        loss_rate_pe = el_pe / max(nav, 1)

        # Asymetrie = difference de taux de perte
        asymmetry = loss_rate_pe - loss_rate_credit

        # Stress delta par rapport au baseline, pondere par les sensibilites
        # du secteur (credit) pour refleter l'exposition sectorielle reelle.
        stress_delta = 0.0
        _sens_map = {
            "unemployment_rate": "unemployment_sensitivity_credit",
            "gdp_growth": "gdp_sensitivity_credit",
            "interest_rate": "interest_rate_sensitivity_credit",
            "hpi_growth": "hpi_sensitivity_credit",
            "inflation_rate": "inflation_sensitivity_credit",
        }
        for var in _sens_map:
            base_val = getattr(SCENARIO_BASE, var)
            current_val = macro_params.get(var, base_val)
            sens = getattr(sector, _sens_map[var])
            stress_delta += abs(current_val - base_val) * sens

        asym_records.append({
            "sector": name,
            "ecl": round(ecl, 0),
            "ead": round(ead, 0),
            "el_pe": round(el_pe, 0),
            "nav": round(nav, 0),
            "loss_rate_credit": round(loss_rate_credit, 6),
            "loss_rate_pe": round(loss_rate_pe, 6),
            "asymmetry": round(asymmetry, 6),
            "stress_intensity": round(stress_delta, 2),
        })

    asymmetry_matrix = pd.DataFrame(asym_records)

    # ── Contributions marginales (10 cellules) ──
    total_risk = (
        result_credit["ecl_weighted"].sum()
        + result_pe["expected_loss_pe"].sum()
    )

    marginal_records = []
    for sector in SECTORS:
        name = sector.name

        # Credit
        mask_c = result_credit["sector"].values == name
        ecl_sec = result_credit.loc[mask_c, "ecl_weighted"].sum()
        contrib_c = ecl_sec / max(total_risk, 1)

        marginal_records.append({
            "sector": name,
            "canal": "Credit",
            "risk_amount": round(ecl_sec, 0),
            "marginal_contribution": round(contrib_c, 6),
        })

        # PE
        mask_p = result_pe["sector"].values == name
        el_sec = result_pe.loc[mask_p, "expected_loss_pe"].sum()
        contrib_p = el_sec / max(total_risk, 1)

        marginal_records.append({
            "sector": name,
            "canal": "PE",
            "risk_amount": round(el_sec, 0),
            "marginal_contribution": round(contrib_p, 6),
        })

    marginal_contributions = pd.DataFrame(marginal_records)

    return asymmetry_matrix, marginal_contributions
