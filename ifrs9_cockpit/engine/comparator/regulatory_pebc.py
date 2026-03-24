"""Fonctions regulatoires LCR/NSFR/IRRBB pour l'optimiseur pe-bc 10 cellules.

Adapte aux 10 cellules (5 secteurs x 2 canaux : Credit + PE).
Les fonctions lisent les attributs dual-channel de SectorConfig
(hqla_eligible_credit/pe, rsf_weight_credit/pe, duration_credit/pe).

Sources : Basel III LCR (Art. 428r), NSFR (CRR2), IRRBB (BCBS 368).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ifrs9_cockpit.config import BASEL_CONFIG, SECTORS


def _parse_cell_name(
    cell_name: str, sectors: list
) -> Tuple[str, bool, object]:
    """Parse un nom de cellule en (sector_name, is_pe, SectorConfig).

    Args:
        cell_name: e.g. "Technologie_Credit" ou "Industrie_PE".
        sectors: Liste de SectorConfig.

    Returns:
        (sector_name, is_pe, sector_config) ou (name, is_pe, None) si introuvable.
    """
    is_pe = "_PE" in cell_name
    sector_name = cell_name.replace("_Credit", "").replace("_PE", "")
    sector = next((s for s in sectors if s.name == sector_name), None)
    return sector_name, is_pe, sector


def compute_lcr_10(
    w: np.ndarray,
    cell_names: Tuple[str, ...],
    sectors: list,
    total_ead: float,
) -> Dict[str, object]:
    """LCR = HQLA / Net cash outflows (adapte 10 cellules).

    Dans pe-bc, aucune cellule n'est HQLA (pas de souverain/couvert).
    Le LCR est donc determine par le ratio HQLA exogene (depots) / outflows.

    Args:
        w: Poids d'allocation (10,).
        cell_names: Noms des 10 cellules.
        sectors: Liste de SectorConfig.
        total_ead: Exposition totale du portefeuille (EUR).

    Returns:
        Dict avec lcr_ratio, hqla, outflows.
    """
    hqla = 0.0
    outflows = 0.0
    for i, cname in enumerate(cell_names):
        ead_i = w[i] * total_ead
        _, is_pe, sector = _parse_cell_name(cname, sectors)
        if sector is None:
            continue
        if is_pe:
            eligible = sector.hqla_eligible_pe
            level = sector.hqla_level_pe
        else:
            eligible = sector.hqla_eligible_credit
            level = sector.hqla_level_credit
        if eligible:
            haircuts = {1: 0.0, 2: 0.15, 3: 0.50}
            haircut = haircuts.get(level, 0.50)
            hqla += ead_i * (1 - haircut)
        # Outflows : 10% du portefeuille credit (wholesale funding proxy),
        # 5% PE (illiquide, pas de demande de remboursement immediate)
        outflow_rate = 0.05 if is_pe else 0.10
        outflows += ead_i * outflow_rate

    # HQLA exogene : depots stables couvrent une partie des outflows
    hqla_exog = total_ead * BASEL_CONFIG.asf_deposit_coverage * 0.10
    hqla_total = hqla + hqla_exog
    lcr_ratio = hqla_total / max(outflows, 1.0)

    return {
        "lcr_ratio": float(lcr_ratio),
        "hqla": float(hqla_total),
        "outflows": float(outflows),
    }


def compute_nsfr_10(
    w: np.ndarray,
    cell_names: Tuple[str, ...],
    sectors: list,
    total_ead: float,
) -> Dict[str, object]:
    """NSFR = ASF / RSF (adapte 10 cellules).

    ASF = Available Stable Funding (depots, capitaux propres).
    RSF = Required Stable Funding (pondere par rsf_weight par cellule).

    Args:
        w: Poids d'allocation (10,).
        cell_names: Noms des 10 cellules.
        sectors: Liste de SectorConfig.
        total_ead: Exposition totale du portefeuille (EUR).

    Returns:
        Dict avec nsfr_ratio, asf, rsf.
    """
    rsf = 0.0
    for i, cname in enumerate(cell_names):
        ead_i = w[i] * total_ead
        _, is_pe, sector = _parse_cell_name(cname, sectors)
        if sector is None:
            continue
        rsf_w = sector.rsf_weight_pe if is_pe else sector.rsf_weight_credit
        rsf += ead_i * rsf_w

    # ASF : depots stables + capitaux propres + dette long terme
    asf = total_ead * BASEL_CONFIG.asf_deposit_coverage
    nsfr_ratio = asf / max(rsf, 1.0)

    return {
        "nsfr_ratio": float(nsfr_ratio),
        "asf": float(asf),
        "rsf": float(rsf),
    }


def compute_irrbb_eve_10(
    w: np.ndarray,
    cell_names: Tuple[str, ...],
    sectors: list,
    total_ead: float,
    cet1_capital: float,
) -> Dict[str, object]:
    """IRRBB EVE (adapte 10 cellules).

    EVE variation = sum(w_i * EAD_i * duration_i * delta_rate) / CET1_capital.
    Choc standard BCBS 368 : +200bp.

    Args:
        w: Poids d'allocation (10,).
        cell_names: Noms des 10 cellules.
        sectors: Liste de SectorConfig.
        total_ead: Exposition totale du portefeuille (EUR).
        cet1_capital: Capital CET1 (EUR).

    Returns:
        Dict avec eve_ratio, weighted_duration, compliant.
    """
    delta_rate = 0.02  # +200bp choc standard BCBS
    weighted_dur = 0.0
    for i, cname in enumerate(cell_names):
        ead_i = w[i] * total_ead
        _, is_pe, sector = _parse_cell_name(cname, sectors)
        if sector is None:
            continue
        dur = sector.duration_pe if is_pe else sector.duration_credit
        # ALM hedge reduit l'exposition duration
        hedge = BASEL_CONFIG.alm_hedge_ratio
        weighted_dur += ead_i * dur * (1 - hedge)

    eve_change = weighted_dur * delta_rate
    eve_limit = 0.15  # BCBS standard : 15% du Tier 1
    eve_ratio = eve_change / max(cet1_capital * eve_limit, 1.0)
    compliant = eve_ratio <= 1.0

    return {
        "eve_ratio": float(eve_ratio),
        "weighted_duration": float(weighted_dur),
        "compliant": bool(compliant),
    }
