"""Configuration — Basel III/CRR3, risk appetite, PE classification, CRO alerts."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BaselConfig:
    """Contraintes prudentielles Bale III et grille CRR3.

    Le risk weight PE n'est pas un 400% fixe : la CRR3 (jan. 2025)
    prevoit une grille graduee (190% IRB diversifie, 250% general equity,
    400% speculatif). L'optimiseur teste les 3 classifications.

    Ref: BCBS d424 (2017), CRR3 Art. 133 (SA equity RW), Art. 155 (IRB equity).

    Attributes:
        cet1_target: Ratio CET1 cible.
        rwa_budget: Budget RWA en EUR.
        leverage_max: Ratio de levier max Bale III.
        rw_credit: Risk Weight moyen credit corporate (SA).
        rw_pe_default: Risk Weight PE par defaut (general equity CRR3).
        rw_pe_options: Tuple des 3 classifications CRR3 testees.
        pe_max_allocation: Allocation maximale en PE (fraction du portefeuille).
        hhi_max: HHI maximum acceptable (seuil de concentration).
        min_sectors_above_5pct: Nombre minimum de secteurs > 5% (diversification).
        cir: Cost/Income Ratio pour le calcul RAROC complet.
        tax_rate: Taux d'imposition effectif pour le profit net RAROC.
        liquidity_premium_bps: Prime de liquidite en bps ajoutee au spread Merton.
        commercial_margin_bps: Marge commerciale bancaire en bps ajoutee au NII.
            En pratique, les banques facturent 1.5-2.5x le spread risk-neutral.
            150 bps est un proxy mid-market corporate (EBA 2023 benchmarks).
    """

    cet1_target: float = 0.13
    # Budget RWA total de la banque (3.69T EUR).
    # CET1 capital = rwa_budget × cet1_target = 479.7 Md EUR.
    # Contraint l'allocation PE a ~15-20% max.
    rwa_budget: float = 3_690_000_000_000.0
    leverage_max: float = 0.033
    rw_credit: float = 1.0
    rw_pe_default: int = 250
    rw_pe_options: Tuple[int, ...] = (190, 250, 400)
    pe_max_allocation: float = 0.15
    hhi_max: int = 2500
    min_sectors_above_5pct: int = 3
    # RAROC complet (H5) : CIR et impots
    cir: float = 0.45
    tax_rate: float = 0.25
    # Spread Merton (H4) : prime de liquidite
    liquidity_premium_bps: int = 50
    # Marge commerciale bancaire (NII = spread Merton + liquidite + marge)
    commercial_margin_bps: int = 200
    # Penalite de correlation pour l'optimiseur Softmax (RJ audit v3).
    # Score_i = RAROC_i - lambda * sum(w_j * rho_ij).
    # lambda > 0 penalise les secteurs correles aux autres.
    lambda_correlation: float = 0.5
    # IRRBB ALM hedge ratio (EBA 2024: large EU banks hedge 30-50% duration)
    alm_hedge_ratio: float = 0.30
    # NSFR Basel III (Net Stable Funding Ratio)
    nsfr_target: float = 1.00       # NSFR minimum reglementaire (100%)
    asf_deposit_coverage: float = 0.90  # Part stable des depots (ASF)
    # LCR Basel III (Liquidity Coverage Ratio)
    lcr_target: float = 1.00
    tnco_fraction: float = 0.15     # Net Cash Outflows = 15% du EAD total


BASEL_CONFIG = BaselConfig()

# ──────────────────────────────────────────────
# RISK APPETITE — SEUILS TRICOLORES
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class RiskAppetiteConfig:
    """Seuils de l'appetit au risque (cadre FSB 2013, feux tricolores).

    Convention : vert = acceptable, ambre = vigilance, rouge = breach.
    Les seuils definissent la frontiere vert/ambre et ambre/rouge.

    Attributes:
        ecl_ead_green: Seuil ECL/EAD vert (< seuil = vert).
        ecl_ead_amber: Seuil ECL/EAD ambre (< seuil = ambre, >= = rouge).
        raroc_green: Seuil RAROC vert (> seuil = vert).
        raroc_amber: Seuil RAROC ambre (> seuil = ambre, <= = rouge).
        hhi_green: Seuil HHI vert (< seuil = diversifie).
        hhi_amber: Seuil HHI ambre (< seuil = modere, >= = concentre).
        nav_drawdown_green: Seuil NAV drawdown PE vert.
        nav_drawdown_amber: Seuil NAV drawdown PE ambre.
    """

    ecl_ead_green: float = 0.020
    ecl_ead_amber: float = 0.040
    raroc_green: float = 0.04
    raroc_amber: float = 0.02
    hhi_green: int = 1500
    hhi_amber: int = 2500
    nav_drawdown_green: float = 0.10
    nav_drawdown_amber: float = 0.25
    # HHI Name Level (concentration par contrepartie, ICAAP Pilier 2)
    # Echelle 10 000. Seuil 50 = alerte concentration idiosyncratique.
    hhi_name_green: int = 30
    hhi_name_amber: int = 50


RISK_APPETITE_CONFIG = RiskAppetiteConfig()

# ──────────────────────────────────────────────
# CLASSIFICATION PE (Performing/Watchlist/Distressed)
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class PEClassificationConfig:
    """Seuils de classification des participations PE.

    Calibration sources :
        - distress thresholds : quartiles historiques de defaut PE.
          Preqin (2022) : ~8% des fonds en distress (Q3), ~25% en watchlist.
          Seuils 10%/30% sont conservateurs vs. benchmarks industriels.
        - lgd_equity : Moody's Recovery & LGD study (2023), equity tranche
          recovery rate ~40% en moyenne -> LGD = 60%.
        - secondary_discount : decote marche secondaire PE, Jefferies/Lazard
          (2023) : median 8-12% pour buyouts mid-market.
        - dlom_vintage_factor : DLOM (Discount for Lack of Marketability)
          ajuste par la maturite. Fonds jeunes = moins liquides.
          Lit. AICPA (2013), Pratt & Grabowski (2014) : DLOM 15-30% PE.
          Le facteur 0.50 donne une fourchette 10-15% (conservateur).

    Attributes:
        distress_threshold_performing: P(distress) max pour rester Performing.
        distress_threshold_watchlist: P(distress) max pour Watchlist (au-dela = Distressed).
        secondary_discount: Decote de marche secondaire de base (~10% buyout).
        lgd_equity: LGD sur les investissements en equity (perte en cas de distress).
        dlom_vintage_factor: Facteur multiplicatif pour l'ajustement DLOM vintage.
            DLOM_effectif = secondary_discount × (1 + factor × max(0, threshold - holding) / threshold)
        dlom_vintage_threshold: Seuil de maturite (annees) au-dela duquel pas de surcharge DLOM.
    """

    distress_threshold_performing: float = 0.10
    distress_threshold_watchlist: float = 0.30
    secondary_discount: float = 0.10
    lgd_equity: float = 0.60
    dlom_vintage_factor: float = 0.50
    dlom_vintage_threshold: float = 5.0


PE_CLASSIFICATION_CONFIG = PEClassificationConfig()

# ──────────────────────────────────────────────
# SEUILS D'ALERTE CRO
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class CROAlertConfig:
    """Seuils d'alerte CRO.

    Attributes:
        ecl_variation_alert: Variation ECL declenchant une alerte.
        stage2_warning_pct: Part Stage 2 declenchant un warning.
        psi_drift_threshold: PSI seuil pour alerte de drift.
        concentration_threshold: Seuil de concentration sectorielle.
    """

    ecl_variation_alert: float = 0.15
    stage2_warning_pct: float = 0.20
    psi_drift_threshold: float = 0.15
    concentration_threshold: float = 0.30


CRO_CONFIG = CROAlertConfig()
