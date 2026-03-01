"""Generateur position-par-position pour le portefeuille Derives/CVA.

Genere ~500 positions derivees OTC (IRS, FX forwards, CDS) avec calcul
SA-CCR (CRR3 Art. 274-280a) pour l'EAD et SA-CVA (CRR3 Art. 382-386)
pour le risk-weight. Le stress test couvre 5 canaux de transmission.

Le pipeline aval (ECL Vasicek, comparator RAROC, BL-CVaR) consomme alors
des valeurs calculees bottom-up plutot que des constantes parametriques.

Calibration :
    - BIS OTC Derivatives Statistics (H1 2024) : notional 667T USD,
      gross MTM 20.7T, netting ~85%, composition IRS/FX/CDS
    - ISDA Margin Survey 2024 : CSA coverage ~90% IM+VM
    - BIS Quarterly Review (Dec 2023) : SA-CCR impact study
    - Solum Financial (2024) : CVA capital charge calibration
    - BCBS d457 (2020) : SA-CVA standardised approach
    - S&P Global (1981-2023) : taux de defaut institutions financieres
    - Pykhtin-Zhu (2007) : wrong-way risk modelling, alpha = 1.4

References :
    - CRR3 Art. 274-280a : SA-CCR (RC, PFE, multiplier, alpha=1.4)
    - CRR3 Art. 275 : Replacement Cost = max(V - C, 0) per netting set
    - CRR3 Art. 277-280 : Add-on by hedging set (supervisory factors)
    - CRR3 Art. 280a : Supervisory delta adjustments
    - CRR3 Art. 382-386 : SA-CVA (capital charge pour risque de contrepartie)
    - CRR3 Art. 383a : SA-CVA reduced (simplified) pour expositions < 100B
    - CRR3 Art. 384 : Basic CVA approach (floor for SA-CVA)
    - Basel FRTB : SA-CVA hedging set correlation
    - BRRD Art. 44(2) : bail-in implications pour LGD contrepartie
    - Basel III NSFR : RSF 85% for derivative assets
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


# ──────────────────────────────────────────────
# CONSTANTES (calibrees BIS/ISDA/CRR3/BCBS)
# ──────────────────────────────────────────────

# Data directory (for potential real-data loading)
_DATA_DIR: Path = Path(__file__).parent.parent / "data"

# SA-CCR alpha multiplier (CRR3 Art. 274(2))
ALPHA_SA_CCR: float = 1.4

# ── Desk mix (BIS OTC Statistics H1 2024) ──
# IRS = 80% of notional but ~60% of count (low risk per unit)
# FX = ~17% of notional, ~25% of count (shorter maturities, more tickets)
# CDS = ~3% of notional, ~15% of count (concentrated, higher risk per unit)
DESK_MIX: Dict[str, float] = {
    "irs": 0.60,
    "fx": 0.25,
    "cds": 0.15,
}

# ── Maturity mix (BIS, semi-annual survey) ──
MATURITY_MIX: Dict[str, float] = {
    "LT1Y": 0.40,
    "1Y5Y": 0.35,
    "GT5Y": 0.25,
}

MATURITY_YEARS: Dict[str, float] = {
    "LT1Y": 0.5,
    "1Y5Y": 3.0,
    "GT5Y": 7.0,
}

# ── SA-CCR Supervisory Factors (CRR3 Art. 280a, Table 2) ──
# Per hedging set, annualised percentage of notional
SA_CCR_SF: Dict[str, float] = {
    "irs": 0.005,        # Interest Rate: 0.50%
    "fx": 0.04,          # Foreign Exchange: 4.0%
    "cds_ig": 0.0038,    # Credit IG: 0.38%
    "cds_hy": 0.0054,    # Credit HY: 0.54%
    "equity_idx": 0.20,  # Equity (index): 20% — not used here but reference
}

# ── SA-CVA Spread buckets (CRR3 Art. 383, BCBS d457) ──
# RW on counterparty spread for CVA capital charge
SA_CVA_SPREAD: Dict[str, float] = {
    "IG": 0.005,   # 50bp spread (IG counterparty)
    "HY": 0.02,    # 200bp spread (HY counterparty)
}

# ── SA-CVA Risk Weights by CQS (CRR3 Art. 383a, Table 3) ──
SA_CVA_RW: Dict[str, float] = {
    "AAA": 0.005,
    "AA+": 0.005, "AA": 0.005, "AA-": 0.005,
    "A+": 0.01, "A": 0.01, "A-": 0.01,
    "BBB+": 0.02, "BBB": 0.03, "BBB-": 0.03,
    "BB+": 0.05, "BB": 0.05,
}

# ── Product specifications per desk ──
DESK_PRODUCTS: Dict[str, List[Dict]] = {
    "irs": [
        {"product": "vanilla_swap",     "weight": 0.50, "delta_sign": 1.0},
        {"product": "basis_swap",       "weight": 0.20, "delta_sign": 1.0},
        {"product": "xccy_swap",        "weight": 0.15, "delta_sign": 1.0},
        {"product": "swaption",         "weight": 0.10, "delta_sign": 0.5},
        {"product": "cap_floor",        "weight": 0.05, "delta_sign": 0.5},
    ],
    "fx": [
        {"product": "fx_forward",       "weight": 0.45, "delta_sign": 1.0},
        {"product": "fx_swap",          "weight": 0.30, "delta_sign": 1.0},
        {"product": "fx_option",        "weight": 0.15, "delta_sign": 0.5},
        {"product": "fx_ndf",           "weight": 0.10, "delta_sign": 1.0},
    ],
    "cds": [
        {"product": "cds_single_name",  "weight": 0.55, "delta_sign": -1.0},
        {"product": "cds_index",        "weight": 0.35, "delta_sign": -1.0},
        {"product": "cds_tranche",      "weight": 0.10, "delta_sign": -1.0},
    ],
}

# ── 20 EU bank/dealer counterparties (EBA Transparency 2024 + G-SIBs) ──
DERIVATIVE_COUNTERPARTIES: Dict[str, Dict] = {
    "DB":    {"name": "Deutsche Bank",       "country": "DE", "weight": 0.10, "rating": "A",    "pd_base": 0.0006, "netting_sets": 5, "csa_flag": True,  "ig_flag": True},
    "BNP":   {"name": "BNP Paribas",         "country": "FR", "weight": 0.10, "rating": "AA-",  "pd_base": 0.0003, "netting_sets": 4, "csa_flag": True,  "ig_flag": True},
    "SG":    {"name": "Societe Generale",     "country": "FR", "weight": 0.08, "rating": "A",    "pd_base": 0.0005, "netting_sets": 4, "csa_flag": True,  "ig_flag": True},
    "JPM":   {"name": "JPMorgan",            "country": "US", "weight": 0.08, "rating": "AA-",  "pd_base": 0.0002, "netting_sets": 6, "csa_flag": True,  "ig_flag": True},
    "GS":    {"name": "Goldman Sachs",        "country": "US", "weight": 0.07, "rating": "A+",   "pd_base": 0.0003, "netting_sets": 5, "csa_flag": True,  "ig_flag": True},
    "BARC":  {"name": "Barclays",            "country": "GB", "weight": 0.06, "rating": "A",    "pd_base": 0.0005, "netting_sets": 4, "csa_flag": True,  "ig_flag": True},
    "HSBC":  {"name": "HSBC",               "country": "GB", "weight": 0.06, "rating": "AA-",  "pd_base": 0.0002, "netting_sets": 4, "csa_flag": True,  "ig_flag": True},
    "C":     {"name": "Citi",                "country": "US", "weight": 0.05, "rating": "A+",   "pd_base": 0.0003, "netting_sets": 5, "csa_flag": True,  "ig_flag": True},
    "UBS":   {"name": "UBS",                "country": "CH", "weight": 0.05, "rating": "AA-",  "pd_base": 0.0002, "netting_sets": 4, "csa_flag": True,  "ig_flag": True},
    "UCG":   {"name": "UniCredit",           "country": "IT", "weight": 0.04, "rating": "BBB",  "pd_base": 0.0012, "netting_sets": 3, "csa_flag": True,  "ig_flag": True},
    "ING":   {"name": "ING",                "country": "NL", "weight": 0.04, "rating": "A+",   "pd_base": 0.0004, "netting_sets": 3, "csa_flag": True,  "ig_flag": True},
    "SAN":   {"name": "Santander",           "country": "ES", "weight": 0.04, "rating": "A",    "pd_base": 0.0005, "netting_sets": 3, "csa_flag": True,  "ig_flag": True},
    "NDA":   {"name": "Nordea",             "country": "FI", "weight": 0.03, "rating": "AA-",  "pd_base": 0.0002, "netting_sets": 2, "csa_flag": True,  "ig_flag": True},
    "ISP":   {"name": "Intesa",             "country": "IT", "weight": 0.03, "rating": "BBB+", "pd_base": 0.0010, "netting_sets": 3, "csa_flag": True,  "ig_flag": True},
    "CABK":  {"name": "CaixaBank",          "country": "ES", "weight": 0.03, "rating": "BBB+", "pd_base": 0.0008, "netting_sets": 2, "csa_flag": True,  "ig_flag": True},
    "CBK":   {"name": "Commerzbank",        "country": "DE", "weight": 0.03, "rating": "BBB+", "pd_base": 0.0008, "netting_sets": 2, "csa_flag": True,  "ig_flag": True},
    "KBC":   {"name": "KBC",               "country": "BE", "weight": 0.02, "rating": "A+",   "pd_base": 0.0003, "netting_sets": 2, "csa_flag": True,  "ig_flag": True},
    "ABN":   {"name": "ABN Amro",           "country": "NL", "weight": 0.02, "rating": "A",    "pd_base": 0.0004, "netting_sets": 2, "csa_flag": True,  "ig_flag": True},
    "STAN":  {"name": "Standard Chartered",  "country": "GB", "weight": 0.02, "rating": "A",    "pd_base": 0.0005, "netting_sets": 2, "csa_flag": False, "ig_flag": True},
    "MPS":   {"name": "Banca MPS",          "country": "IT", "weight": 0.05, "rating": "BB+",  "pd_base": 0.0050, "netting_sets": 1, "csa_flag": False, "ig_flag": False},
}

# ── EAD calibration target ──
# Post-netting, post-CSA EAD should be ~15-30% of gross notional
# (BIS: for mixed margined/unmargined books with moderate netting)
_TARGET_EAD_RATIO: float = 0.22  # ~22% of gross notional

# ── CDS reference entity mix (IG vs HY) ──
_CDS_IG_SHARE: float = 0.70  # 70% IG reference entities
_CDS_HY_SHARE: float = 0.30  # 30% HY reference entities

# ── Notional parameters by desk+maturity (EUR millions, median) ──
_NOTIONAL_MEDIAN: Dict[str, float] = {
    "irs": 100e6,    # IRS: large notional, small risk
    "fx": 30e6,      # FX: medium notional, short tenor
    "cds": 15e6,     # CDS: smaller notional, higher risk
}

# ── MTM parameters ──
# Net MTM as % of notional, by desk (BIS: gross MTM ~3% notional, net ~0.5%)
_MTM_PCT_MEAN: Dict[str, float] = {
    "irs": 0.005,    # ~50bp of notional
    "fx": 0.015,     # ~150bp of notional (shorter, more volatile)
    "cds": 0.020,    # ~200bp of notional (credit-sensitive)
}
_MTM_PCT_STD: Dict[str, float] = {
    "irs": 0.015,
    "fx": 0.025,
    "cds": 0.035,
}

# ── CSA collateral parameters (ISDA Margin Survey 2024) ──
_CSA_THRESHOLD: float = 0.0        # Modern CSA: zero threshold (VM)
_CSA_MTA: float = 500_000.0        # Minimum Transfer Amount EUR 500K
_CSA_HAIRCUT: float = 0.02         # 2% haircut on cash collateral

# ── PD/LGD parameters ──
_PD_FLOOR: float = 0.0003         # CRR3 3bp floor for financial institutions
_PD_CAP: float = 0.02
_LGD_FLOOR: float = 0.25          # CRR3 Art. 161 unsecured senior
_LGD_CAP: float = 0.65
_LGD_SENIOR_UNSECURED: float = 0.45  # F-IRB default
_LGD_NOISE_STD: float = 0.02

# ── Stress base scenario ──
_BASE_GDP: float = 1.2
_BASE_RATE: float = 3.5
_BASE_UNEMPLOYMENT: float = 7.5

# ── LGD downturn ──
_LGD_DOWNTURN_COEFF: float = 0.02   # +2pp LGD per -1% GDP
_LGD_DOWNTURN_CAP: float = 0.10

# ── Wrong-way risk (Pykhtin-Zhu 2007) ──
_WWR_ALPHA_BASE: float = 0.10       # Base alpha addon for wrong-way risk
_WWR_MAX_ADDON: float = 0.30        # Maximum WWR addon

# ── SA-CVA discount rate ──
_CVA_DISCOUNT_RATE: float = 0.03    # 3% risk-free rate for CVA discounting


# ──────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────

def generate_derivative_positions(
    n_positions: int = 500,
    total_notional: float = 50e9,
    seed: int = 123,
) -> pl.DataFrame:
    """Genere un portefeuille de derives OTC position par position.

    Etape A: sample counterparty (20 banks/dealers), desk, product, maturity.
    Etape B: generate notional, MTM, delta pour chaque position.
    Etape C: assign netting sets, CSA flags.
    Etape D: compute PD, LGD, EAD (SA-CCR) pour chaque position.

    Args:
        n_positions: Nombre de positions a generer (~500).
        total_notional: Notionnel total du portefeuille derives.
        seed: Graine aleatoire.

    Returns:
        DataFrame avec ~25 colonnes (une ligne par position).
    """
    rng = np.random.default_rng(seed)

    # ── Sample desks ──
    desk_names = list(DESK_MIX.keys())
    desk_weights = np.array([DESK_MIX[d] for d in desk_names])
    desk_weights = desk_weights / desk_weights.sum()
    sampled_desks = rng.choice(desk_names, size=n_positions, p=desk_weights)

    # ── Sample counterparties ──
    cp_codes = list(DERIVATIVE_COUNTERPARTIES.keys())
    cp_weights = np.array([DERIVATIVE_COUNTERPARTIES[c]["weight"] for c in cp_codes])
    cp_weights = cp_weights / cp_weights.sum()
    sampled_cps = rng.choice(cp_codes, size=n_positions, p=cp_weights)

    # ── Sample maturity buckets ──
    mat_names = list(MATURITY_MIX.keys())
    mat_weights = np.array([MATURITY_MIX[m] for m in mat_names])
    mat_weights = mat_weights / mat_weights.sum()
    sampled_mats = rng.choice(mat_names, size=n_positions, p=mat_weights)

    # ── Build positions ──
    records: List[Dict] = []
    for i in range(n_positions):
        desk = sampled_desks[i]
        cp_code = sampled_cps[i]
        cp = DERIVATIVE_COUNTERPARTIES[cp_code]
        mat_bucket = sampled_mats[i]
        maturity = MATURITY_YEARS[mat_bucket]

        # Sample product within desk
        products = DESK_PRODUCTS[desk]
        prod_weights = np.array([p["weight"] for p in products])
        prod_weights = prod_weights / prod_weights.sum()
        prod_idx = rng.choice(len(products), p=prod_weights)
        product = products[prod_idx]

        # Notional (lognormal around desk median)
        median_notional = _NOTIONAL_MEDIAN[desk]
        notional = rng.lognormal(np.log(median_notional), 0.6)

        # MTM (normal, can be positive or negative)
        mtm_pct = rng.normal(
            _MTM_PCT_MEAN[desk],
            _MTM_PCT_STD[desk],
        )
        mtm = notional * mtm_pct

        # Supervisory delta (CRR3 Art. 280a)
        # Long risk: delta > 0, short risk: delta < 0
        # Options get delta=0.5 * sign, linear get sign
        delta = product["delta_sign"]
        # Add small randomisation for options
        if abs(delta) < 1.0:
            delta = delta * rng.uniform(0.3, 0.7)

        # CSA and netting set assignment
        csa_flag = cp["csa_flag"]
        n_netting = cp["netting_sets"]
        netting_set_id = f"{cp_code}_NS{rng.integers(1, n_netting + 1)}"

        # CDS IG/HY classification
        is_cds = desk == "cds"
        cds_quality = ""
        if is_cds:
            cds_quality = "IG" if rng.random() < _CDS_IG_SHARE else "HY"

        # Collateral (if CSA is active)
        # Modern bilateral CSA: VM covers ~90% of positive MTM
        if csa_flag and mtm > 0:
            collateral = mtm * rng.uniform(0.80, 0.95)
        elif csa_flag:
            collateral = 0.0  # Out-of-money: no collateral to post
        else:
            collateral = 0.0  # No CSA: uncollateralised

        # DV01 for IRS (dollar value of 1bp rate move)
        # DV01 ~ notional * maturity * 1bp
        dv01 = 0.0
        if desk == "irs":
            dv01 = notional * maturity * 0.0001

        # CS01 for CDS (credit spread 01)
        # CS01 ~ notional * risky_duration * 1bp
        cs01 = 0.0
        if desk == "cds":
            risky_duration = maturity * 0.95  # approximate
            cs01 = notional * risky_duration * 0.0001

        records.append({
            "position_id": i,
            "desk": desk,
            "product": product["product"],
            "counterparty_code": cp_code,
            "counterparty_name": cp["name"],
            "country": cp["country"],
            "rating": cp["rating"],
            "ig_flag": cp["ig_flag"],
            "netting_set": netting_set_id,
            "csa_flag": csa_flag,
            "notional": round(notional, 2),
            "maturity_bucket": mat_bucket,
            "maturity": maturity,
            "mtm": round(mtm, 2),
            "delta": round(delta, 4),
            "collateral": round(collateral, 2),
            "cds_quality": cds_quality,
            "dv01": round(dv01, 2),
            "cs01": round(cs01, 2),
            "pd_base": cp["pd_base"],
            "lgd_base": 0.0,    # placeholder
            "ead_sa_ccr": 0.0,  # placeholder
            "rw_crr3": 0.0,     # placeholder
        })

    df = pl.DataFrame(records)

    # ── Scale notionals to target total ──
    raw_total = df["notional"].sum()
    if raw_total > 0:
        scale = total_notional / raw_total
        df = df.with_columns([
            (pl.col("notional") * scale).alias("notional"),
            (pl.col("mtm") * scale).alias("mtm"),
            (pl.col("collateral") * scale).alias("collateral"),
            (pl.col("dv01") * scale).alias("dv01"),
            (pl.col("cs01") * scale).alias("cs01"),
        ])

    # ── Compute PD (counterparty-level) ──
    df = df.with_columns(
        pl.Series("pd_position", _compute_derivative_pd(df, rng)),
    )

    # ── Compute LGD (waterfall) ──
    df = df.with_columns(
        pl.Series("lgd_position", _compute_derivative_lgd(df, rng)),
    )

    # ── Compute SA-CCR EAD (per netting set, then attributed to positions) ──
    df = compute_sa_ccr_ead(df)

    # ── Calibrate EAD to target ratio ──
    # SA-CCR with high CSA coverage produces very low EAD/notional (~1-3%).
    # Real bank portfolios with mixed margined/unmargined and imperfect netting
    # show EAD/notional of ~15-30% (BIS QIS). Scale to match empirical target.
    current_ead = df["ead_sa_ccr"].sum()
    target_ead = df["notional"].sum() * _TARGET_EAD_RATIO
    if current_ead > 0:
        ead_scale = target_ead / current_ead
        df = df.with_columns([
            (pl.col("ead_sa_ccr") * ead_scale).alias("ead_sa_ccr"),
            (pl.col("ead") * ead_scale).alias("ead"),
        ])

    # ── Compute SA-CVA RW ──
    df = compute_derivative_rw(df)

    return df


# ──────────────────────────────────────────────
# PD MULTI-COMPOSANTE (counterparty credit risk)
# ──────────────────────────────────────────────

def _compute_derivative_pd(
    df: pl.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """Calcule la PD multi-composante pour chaque position derivee.

    3 composantes multiplicatives :
        1. PD rating contrepartie (S&P 1981-2023, institutions financieres)
        2. Alpha maturity (tenor plus long -> PD plus elevee, CRR3 MA)
        3. Alpha desk (CDS = protection vendue -> risque supplementaire)

    La PD est au niveau contrepartie, mais ajustee par desk/maturity
    car le risque effectif depend du profil d'exposition.

    Args:
        df: DataFrame from generate_derivative_positions().
        rng: Random generator for noise.

    Returns:
        Array de PD par position.
    """
    n = len(df)
    pd_base_arr = df["pd_base"].to_numpy()
    maturities = df["maturity"].to_numpy()
    desks = df["desk"].to_numpy()

    # 1. Base PD from counterparty rating
    pd_result = pd_base_arr.copy().astype(float)

    # 2. Maturity adjustment (CRR3 Art. 153(1))
    # b(PD) = (0.11852 - 0.05478 * ln(PD))^2
    # MA = (1 + (M-2.5) * b) / (1 - 1.5 * b)
    # Simplified: longer maturity -> higher PD multiplier
    for i in range(n):
        mat = maturities[i]
        # Linear approximation: +15% per year above 1Y
        alpha_maturity = 1.0 + max(0.0, mat - 1.0) * 0.15
        pd_result[i] *= alpha_maturity

    # 3. Desk adjustment
    # CDS sellers have additional jump-to-default risk
    _desk_alpha = {"irs": 1.0, "fx": 1.0, "cds": 1.5}
    for i in range(n):
        pd_result[i] *= _desk_alpha.get(desks[i], 1.0)

    # 4. Small noise
    pd_result += rng.normal(0.0, 0.00005, n)

    return np.clip(pd_result, _PD_FLOOR, _PD_CAP)


# ──────────────────────────────────────────────
# LGD WATERFALL (close-out netting + CSA)
# ──────────────────────────────────────────────

def _compute_derivative_lgd(
    df: pl.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Calcule la LGD pour chaque position derivee.

    Recovery waterfall (3 sources) :
        1. Close-out netting recovery (ISDA Master Agreement)
           - CSA positions: recover ~60-80% via collateral
           - Non-CSA: ~10% seniority recovery only
        2. BRRD bail-in adjustment
           - Derivative claims are senior unsecured under BRRD
        3. CDS-specific: bought protection provides partial hedge

    Calibration :
        - Lehman Brothers (2008) : derivative claims LGD ~65-75%
        - MF Global (2011) : customer accounts LGD ~20%
        - CRR3 Art. 161 F-IRB : LGD = 45% senior unsecured

    Args:
        df: DataFrame from generate_derivative_positions().
        rng: Optional random generator for noise.

    Returns:
        Array de LGD par position.
    """
    n = len(df)
    lgd = np.zeros(n)

    csa_flags = df["csa_flag"].to_numpy()
    desks = df["desk"].to_numpy()
    mtm_arr = df["mtm"].to_numpy()
    collateral_arr = df["collateral"].to_numpy()

    for i in range(n):
        csa = csa_flags[i]
        desk = desks[i]
        mtm = mtm_arr[i]
        collateral = collateral_arr[i]

        if csa:
            # CSA: collateral covers most of positive exposure
            if mtm > 0 and collateral > 0:
                coverage = min(1.0, collateral / mtm)
                rr_collateral = coverage * 0.95  # 5% friction on close-out
            else:
                # Negative MTM or zero collateral
                rr_collateral = 0.0
            # Seniority: senior claim in insolvency
            rr_seniority = 0.10
            # Total recovery (capped at 1.0)
            rr = min(1.0, rr_collateral + rr_seniority)
        else:
            # No CSA: unsecured derivative claim
            # Recovery: Lehman average ~30% for derivatives
            rr = 0.30

        # CDS desk: if protection buyer, additional recovery
        if desk == "cds":
            # Mixed book: some bought, some sold
            rr = min(1.0, rr + 0.05)

        lgd[i] = 1.0 - rr

    # Add noise
    if rng is not None:
        lgd += rng.normal(0.0, _LGD_NOISE_STD, n)

    return np.clip(lgd, _LGD_FLOOR, _LGD_CAP)


# ──────────────────────────────────────────────
# SA-CCR EAD (CRR3 Art. 274-280a)
# ──────────────────────────────────────────────

def compute_sa_ccr_ead(df: pl.DataFrame) -> pl.DataFrame:
    """Calcule l'EAD SA-CCR pour chaque position derivee.

    SA-CCR formula (CRR3 Art. 274):
        EAD = alpha * (RC + PFE)

    Computed at netting set level, then attributed proportionally.

    Per netting set:
        RC  = max(V_net - C_net, 0)                           (Art. 275)
        PFE = multiplier * AddOn_aggregate                     (Art. 278)
        multiplier = min(1, floor + (1-floor)*exp(V/(2*k*AddOn)))  (Art. 278(3))
        AddOn_hs = SF * |sum(delta_i * d_i * MF_i * notional_i)|  (Art. 280)
        MF_i = sqrt(min(M_i, 1) / 1)  for unmargined              (Art. 279(1))
             = 1.5*sqrt(MPOR/250) for margined                    (Art. 279(3))

    Args:
        df: DataFrame from generate_derivative_positions().

    Returns:
        DataFrame with ead_sa_ccr column updated.
    """
    n = len(df)
    netting_sets = df["netting_set"].to_list()
    notionals = df["notional"].to_numpy()
    mtm_arr = df["mtm"].to_numpy()
    collateral_arr = df["collateral"].to_numpy()
    deltas = df["delta"].to_numpy()
    maturities = df["maturity"].to_numpy()
    desks = df["desk"].to_numpy()
    csa_flags = df["csa_flag"].to_numpy()
    cds_qualities = df["cds_quality"].to_list()

    # ── Group positions by netting set ──
    ns_map: Dict[str, List[int]] = {}
    for i in range(n):
        ns = netting_sets[i]
        if ns not in ns_map:
            ns_map[ns] = []
        ns_map[ns].append(i)

    ead_arr = np.zeros(n)

    for ns_id, indices in ns_map.items():
        idx = np.array(indices)

        # ── Replacement Cost (RC) ──
        # V_net = sum of MTM in the netting set
        v_net = mtm_arr[idx].sum()
        # C_net = sum of collateral received
        c_net = collateral_arr[idx].sum()
        rc = max(v_net - c_net, 0.0)

        # ── PFE: Add-on per hedging set ──
        # Group by hedging set within netting set
        # IRS, FX, CDS are separate hedging sets
        addon_total = 0.0
        for hs_desk in ["irs", "fx", "cds"]:
            hs_mask = desks[idx] == hs_desk
            if not hs_mask.any():
                continue

            hs_idx = idx[hs_mask]
            hs_notionals = notionals[hs_idx]
            hs_deltas = deltas[hs_idx]
            hs_maturities = maturities[hs_idx]
            hs_csa = csa_flags[hs_idx]

            # Maturity factor (CRR3 Art. 279)
            is_margined = hs_csa.any()
            if is_margined:
                # MPOR = 10 business days for bilateral CSA
                mpor = 10.0
                mf = 1.5 * np.sqrt(mpor / 250.0)
                mf_arr = np.full(len(hs_idx), mf)
            else:
                # Unmargined: MF = sqrt(min(M, 1))
                mf_arr = np.sqrt(np.minimum(hs_maturities, 1.0))
                mf_arr = np.maximum(mf_arr, 0.2)  # floor

            # Supervisory duration (CRR3 Art. 279(2))
            # d_i = (exp(-0.05*S_i) - exp(-0.05*E_i)) / 0.05
            # Simplified: start=0, end=maturity
            d_arr = (1.0 - np.exp(-0.05 * hs_maturities)) / 0.05

            # Effective notional per position
            eff_notional = hs_deltas * d_arr * mf_arr * hs_notionals

            # Net effective notional for hedging set (with partial offset)
            net_eff = np.abs(eff_notional.sum())

            # Supervisory factor
            if hs_desk == "cds":
                # Split IG/HY within CDS hedging set
                hs_cds_qual = [cds_qualities[j] for j in hs_idx]
                addon_ig = 0.0
                addon_hy = 0.0
                for k_local, j in enumerate(hs_idx):
                    qual = cds_qualities[j]
                    eff_n = abs(eff_notional[k_local])
                    if qual == "HY":
                        addon_hy += eff_n
                    else:
                        addon_ig += eff_n
                addon_desk = (
                    SA_CCR_SF["cds_ig"] * addon_ig
                    + SA_CCR_SF["cds_hy"] * addon_hy
                )
            else:
                sf = SA_CCR_SF.get(hs_desk, 0.005)
                addon_desk = sf * net_eff

            addon_total += addon_desk

        # ── PFE multiplier (CRR3 Art. 278(3)) ──
        floor = 0.05  # 5% floor
        if addon_total > 0 and v_net < 0:
            # Out of money: multiplier < 1
            multiplier = floor + (1.0 - floor) * np.exp(
                v_net / (2.0 * (1.0 - floor) * addon_total)
            )
            multiplier = min(1.0, max(floor, multiplier))
        else:
            multiplier = 1.0

        pfe = multiplier * addon_total

        # ── EAD = alpha * (RC + PFE) ──
        ead_netting_set = ALPHA_SA_CCR * (rc + pfe)

        # ── Attribute EAD proportionally to positions in netting set ──
        ns_notionals = notionals[idx]
        total_ns_notional = ns_notionals.sum()
        if total_ns_notional > 0:
            weights = ns_notionals / total_ns_notional
        else:
            weights = np.ones(len(idx)) / len(idx)

        for k_local, j in enumerate(indices):
            ead_arr[j] = ead_netting_set * weights[k_local]

    # Update the DataFrame
    df = df.with_columns(pl.Series("ead_sa_ccr", ead_arr))

    # Also add ead column for compatibility with aggregation
    df = df.with_columns(pl.col("ead_sa_ccr").alias("ead"))

    return df


# ──────────────────────────────────────────────
# CVA RISK (CRR3 Art. 382-386)
# ──────────────────────────────────────────────

def compute_cva_risk(df: pl.DataFrame) -> pl.DataFrame:
    """Calcule le risque CVA pour chaque position derivee.

    CVA (Credit Valuation Adjustment) represents the expected loss
    from counterparty default on derivative exposures.

    Per counterparty:
        CVA_c = sum_t [ PD_c(t) * EPE_c(t) * LGD_c * DF(t) ]

    Where:
        PD_c(t) = marginal probability of default at time t
        EPE_c(t) = Expected Positive Exposure (EAD * (1 + WWR_alpha))
        LGD_c = Loss Given Default of counterparty
        DF(t) = discount factor = exp(-r * t)

    Wrong-way risk addon (Pykhtin-Zhu 2007):
        WWR_alpha = alpha_base * max(0, corr(exposure, pd))
        Typically 10-30% addon for positive correlation

    Args:
        df: DataFrame from generate_derivative_positions().

    Returns:
        DataFrame with cva_charge column added.
    """
    n = len(df)
    cva_arr = np.zeros(n)

    ead_arr = df["ead_sa_ccr"].to_numpy()
    pd_arr = df["pd_position"].to_numpy()
    lgd_arr = df["lgd_position"].to_numpy()
    maturities = df["maturity"].to_numpy()
    desks = df["desk"].to_numpy()
    csa_flags = df["csa_flag"].to_numpy()

    for i in range(n):
        pd_i = pd_arr[i]
        ead_i = ead_arr[i]
        lgd_i = lgd_arr[i]
        mat_i = maturities[i]
        desk = desks[i]
        csa = csa_flags[i]

        # Wrong-way risk addon
        # Higher for CDS (exposure and PD positively correlated)
        # Lower for IRS (less systematic), moderate for FX
        if desk == "cds":
            wwr_alpha = _WWR_ALPHA_BASE * 2.5  # CDS: strong WWR
        elif desk == "fx":
            wwr_alpha = _WWR_ALPHA_BASE * 1.5  # FX: moderate WWR
        else:
            wwr_alpha = _WWR_ALPHA_BASE * 0.8  # IRS: mild WWR

        # CSA reduces WWR via frequent margining
        if csa:
            wwr_alpha *= 0.5

        wwr_alpha = min(wwr_alpha, _WWR_MAX_ADDON)

        # EPE (Expected Positive Exposure)
        epe_i = ead_i * (1.0 + wwr_alpha)

        # Discount factor
        df_i = np.exp(-_CVA_DISCOUNT_RATE * mat_i)

        # CVA = PD * EPE * LGD * DF
        # Marginal PD over the maturity (approximate cumulative)
        pd_marginal = 1.0 - (1.0 - pd_i) ** mat_i
        cva_arr[i] = pd_marginal * epe_i * lgd_i * df_i

    df = df.with_columns(pl.Series("cva_charge", cva_arr))
    return df


# ──────────────────────────────────────────────
# SA-CVA RISK WEIGHT (CRR3 Art. 382-386)
# ──────────────────────────────────────────────

def compute_derivative_rw(df: pl.DataFrame) -> pl.DataFrame:
    """Calcule le Risk Weight SA-CVA pour chaque position derivee.

    SA-CVA (CRR3 Art. 382-386) combines:
        1. Counterparty credit spread component (systematic)
        2. Idiosyncratic component
        3. Hedging benefit (if CVA hedges exist)

    SA-CVA RW formula (simplified, no hedging):
        K_CVA = sqrt(rho^2 * (sum(RW_c * M_c * EAD_c * DF_c))^2
                     + (1-rho^2) * sum((RW_c * M_c * EAD_c * DF_c)^2))

    Where:
        rho = 0.5 (supervisory correlation, Art. 383)
        RW_c = counterparty risk weight (by CQS)
        M_c = effective maturity
        DF_c = supervisory discount factor = (1-exp(-0.05*M)) / (0.05*M)

    For position-level attribution:
        RW_i = K_CVA_i / EAD_i (synthetic per-position RW)

    Args:
        df: DataFrame from generate_derivative_positions().

    Returns:
        DataFrame with rw_crr3 column updated.
    """
    n = len(df)
    rw_arr = np.zeros(n)

    ratings = df["rating"].to_list()
    maturities = df["maturity"].to_numpy()
    ead_arr = df["ead_sa_ccr"].to_numpy()

    # Supervisory correlation (CRR3 Art. 383)
    rho = 0.50

    # Compute weighted contribution per counterparty, then attribute back
    # Group by counterparty
    cp_codes = df["counterparty_code"].to_list()
    cp_map: Dict[str, List[int]] = {}
    for i in range(n):
        cp = cp_codes[i]
        if cp not in cp_map:
            cp_map[cp] = []
        cp_map[cp].append(i)

    # Step 1: Compute per-counterparty S_c (sensitivity)
    s_c_map: Dict[str, float] = {}
    for cp_id, indices in cp_map.items():
        idx = np.array(indices)
        rating = ratings[indices[0]]
        rw_c = SA_CVA_RW.get(rating, 0.03)

        total_s = 0.0
        for j in indices:
            mat_j = maturities[j]
            ead_j = ead_arr[j]
            # Supervisory discount factor
            if mat_j > 0:
                df_j = (1.0 - np.exp(-0.05 * mat_j)) / (0.05 * mat_j)
            else:
                df_j = 1.0
            total_s += rw_c * mat_j * ead_j * df_j

        s_c_map[cp_id] = total_s

    # Step 2: Compute K_CVA (portfolio-level)
    s_values = np.array(list(s_c_map.values()))
    systematic = rho * s_values.sum()
    idiosyncratic_sq = (1.0 - rho ** 2) * (s_values ** 2).sum()
    k_cva_total = np.sqrt(systematic ** 2 + idiosyncratic_sq)

    # Step 3: Attribute RW to positions
    # RW_i = (S_c_i / sum(S_c)) * K_CVA / EAD_i
    total_s_all = abs(s_values.sum()) if abs(s_values.sum()) > 0 else 1.0

    for cp_id, indices in cp_map.items():
        s_c = s_c_map[cp_id]
        for j in indices:
            ead_j = ead_arr[j]
            if ead_j > 0:
                # Position's contribution to CVA capital
                position_k = k_cva_total * (abs(s_c) / total_s_all) * (
                    ead_j / max(1.0, sum(ead_arr[k] for k in indices))
                )
                rw_arr[j] = position_k / ead_j
            else:
                rw_arr[j] = 0.0

    # Floor at counterparty spread-based minimum
    for i in range(n):
        rating = ratings[i]
        rw_min = SA_CVA_RW.get(rating, 0.03)
        rw_arr[i] = max(rw_arr[i], rw_min)

    # Cap to avoid extreme values
    rw_arr = np.clip(rw_arr, 0.0, 1.50)

    df = df.with_columns(pl.Series("rw_crr3", rw_arr))
    return df


# ──────────────────────────────────────────────
# STRESS TEST (5 canaux, calibre EBA/BIS)
# ──────────────────────────────────────────────

def stress_derivative_positions(
    df: pl.DataFrame,
    macro_params: Dict[str, float],
) -> Dict[str, float]:
    """Stress test des positions derivees et re-aggregation.

    5 canaux de transmission :
        1. Rate curve shift (parallel) -> IRS MTM via DV01
        2. FX volatility spike -> FX PFE increase
        3. Credit spread widening -> CDS MTM via CS01
        4. Counterparty PD stress -> same canal as interbank
        5. Wrong-way risk amplification -> alpha addon on correlated exposure

    Calibration cible :
        - GFC (GDP -5%, rates +300bp) -> EAD ~2.5x base, PD ~3x
        - Souveraine (2012) -> CDS spread widening, PD ~2x for periphery
        - COVID (2020) -> FX vol spike, cross-asset correlation ~1

    Args:
        df: DataFrame from generate_derivative_positions().
        macro_params: Dict avec gdp_growth, interest_rate, unemployment_rate, etc.

    Returns:
        Dict avec pd_base, lgd_base, rw_crr3 stresses (EAD-weighted).
    """
    delta_gdp = macro_params.get("gdp_growth", _BASE_GDP) - _BASE_GDP
    delta_rate = macro_params.get("interest_rate", _BASE_RATE) - _BASE_RATE
    delta_unemp = macro_params.get("unemployment_rate", _BASE_UNEMPLOYMENT) - _BASE_UNEMPLOYMENT

    n = len(df)
    ead_arr = df["ead_sa_ccr"].to_numpy()
    pd_base_arr = df["pd_position"].to_numpy()
    lgd_base_arr = df["lgd_position"].to_numpy()
    rw_base_arr = df["rw_crr3"].to_numpy()
    desks = df["desk"].to_numpy()
    maturities = df["maturity"].to_numpy()
    dv01_arr = df["dv01"].to_numpy()
    cs01_arr = df["cs01"].to_numpy()
    mtm_arr = df["mtm"].to_numpy()
    notionals = df["notional"].to_numpy()
    csa_flags = df["csa_flag"].to_numpy()

    # ── Canal 1 : Rate curve shift -> IRS MTM ──
    # delta_rate in pp: e.g. +1.5pp -> +150bp parallel shift
    # MTM impact = DV01 * delta_rate * 100 (convert pp to bp)
    rate_shift_bp = delta_rate * 100.0
    mtm_stressed = mtm_arr.copy()
    for i in range(n):
        if desks[i] == "irs":
            # Rate increase -> payer swaps gain, receiver swaps lose
            # Net effect depends on delta sign
            mtm_impact = dv01_arr[i] * rate_shift_bp
            mtm_stressed[i] += mtm_impact

    # ── Canal 2 : FX volatility spike -> FX PFE ──
    # GDP drop / rate hike -> FX vol increases
    stress_intensity = max(0.0, -delta_gdp) + max(0.0, delta_rate) * 0.5
    fx_vol_mult = 1.0 + 0.5 * stress_intensity  # 50% PFE increase per unit stress
    ead_stressed = ead_arr.copy()
    for i in range(n):
        if desks[i] == "fx":
            ead_stressed[i] *= fx_vol_mult

    # ── Canal 3 : Credit spread widening -> CDS MTM ──
    # Adverse macro -> spreads widen -> protection sellers lose
    spread_widening_bp = max(0.0, -delta_gdp * 80.0 + delta_unemp * 30.0)
    for i in range(n):
        if desks[i] == "cds":
            mtm_impact_cds = -cs01_arr[i] * spread_widening_bp  # sellers lose
            mtm_stressed[i] += mtm_impact_cds
            # Re-estimate EAD for CDS with new MTM
            # Simplified: EAD increase proportional to spread widening
            spread_mult = 1.0 + spread_widening_bp / 100.0
            ead_stressed[i] *= min(3.0, spread_mult)  # cap at 3x

    # ── Canal 4 : Counterparty PD stress ──
    # Same mechanism as interbank: GDP + unemployment driven
    gdp_mult = 1.0 + max(0.0, -delta_gdp) * 0.30     # 30% per 1pp GDP drop
    unemp_mult = 1.0 + max(0.0, delta_unemp) * 0.15   # 15% per 1pp unemployment rise
    contagion_mult = 1.0 + max(0.0, -delta_gdp) * 0.10  # Interbank contagion

    pd_stressed = pd_base_arr * gdp_mult * unemp_mult * contagion_mult
    pd_stressed = np.clip(pd_stressed, _PD_FLOOR, _PD_CAP)

    # ── Canal 5 : Wrong-way risk amplification ──
    # In stress: exposure and PD become more correlated
    # -> Additional EAD addon for positions with WWR
    wwr_stress = max(0.0, stress_intensity) * 0.10  # 10% per unit stress
    for i in range(n):
        if desks[i] == "cds":
            # CDS: strong WWR in stress
            ead_stressed[i] *= (1.0 + wwr_stress * 2.0)
        elif desks[i] == "fx":
            # FX: moderate WWR (EM counterparties)
            ead_stressed[i] *= (1.0 + wwr_stress * 1.0)
        else:
            # IRS: mild WWR
            ead_stressed[i] *= (1.0 + wwr_stress * 0.5)

    # ── EAD floor: SA-CCR recalculation effect ──
    # In stress, RC increases (MTM moves adverse)
    # Simplified: recalculate RC contribution at position level
    for i in range(n):
        if mtm_stressed[i] > mtm_arr[i] and not csa_flags[i]:
            # Positive MTM increase without CSA -> RC increases
            rc_addon = (mtm_stressed[i] - mtm_arr[i]) * ALPHA_SA_CCR
            ead_stressed[i] += max(0.0, rc_addon)

    # Cap EAD stress multiplier
    ead_stressed = np.clip(ead_stressed, 0.0, ead_arr * 5.0)

    # ── LGD downturn ──
    lgd_addon = np.clip(
        max(0.0, -delta_gdp) * _LGD_DOWNTURN_COEFF,
        0.0,
        _LGD_DOWNTURN_CAP,
    )
    lgd_stressed = np.clip(lgd_base_arr + lgd_addon, _LGD_FLOOR, _LGD_CAP)

    # ── RW stress: SA-CVA increases with spread widening ──
    # RW multiplier from credit spread widening
    rw_spread_mult = 1.0 + spread_widening_bp / 200.0  # 50% RW increase per 100bp spread
    rw_stressed = np.clip(rw_base_arr * rw_spread_mult, 0.0, 1.50)

    # ── EAD-weighted aggregation ──
    total_ead = ead_stressed.sum()
    if total_ead <= 0:
        return {"pd_base": 0.0005, "lgd_base": 0.45, "rw_crr3": 0.10}

    w = ead_stressed / total_ead
    pd_agg = float(np.dot(w, pd_stressed))
    lgd_agg = float(np.dot(w, lgd_stressed))
    rw_agg = float(np.dot(w, rw_stressed))

    return {
        "pd_base": round(pd_agg, 6),
        "lgd_base": round(lgd_agg, 6),
        "rw_crr3": round(rw_agg, 4),
    }


# ──────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────

def aggregate_derivatives_to_balance_row(
    df_positions: pl.DataFrame,
    profile: "AssetClassProfile",
) -> dict:
    """Agrege les positions derivees en une ligne balance sheet.

    EAD-weighted PD/LGD/tenor/RW.
    NOT HQLA eligible, rsf_weight=0.85 (derivative assets),
    NOT exempt from staging.

    Args:
        df_positions: DataFrame from generate_derivative_positions().
        profile: AssetClassProfile for derivatives.

    Returns:
        Dict avec toutes les colonnes balance sheet (19 cles).
    """
    weights = df_positions["ead_sa_ccr"].to_numpy()
    total_ead = weights.sum()

    if total_ead <= 0:
        pd_agg = profile.pd_base
        lgd_agg = profile.lgd_base
        tenor_agg = profile.tenor
        rw_agg = profile.rw_crr3
    else:
        w = weights / total_ead
        pd_agg = float(np.dot(w, df_positions["pd_position"].to_numpy()))
        lgd_agg = float(np.dot(w, df_positions["lgd_position"].to_numpy()))
        tenor_agg = float(np.dot(w, df_positions["maturity"].to_numpy()))
        rw_agg = float(np.dot(w, df_positions["rw_crr3"].to_numpy()))

    return {
        "asset_class": profile.name,
        "label": profile.label,
        "category": profile.category,
        "ead_total": round(total_ead, 2),
        "typical_weight": profile.typical_weight,
        "pd_base": round(pd_agg, 6),
        "lgd_base": round(lgd_agg, 6),
        "tenor": round(tenor_agg, 4),
        "asset_correlation": profile.asset_correlation,
        "rw_crr3": round(rw_agg, 4),
        "physical_risk": profile.physical_risk,
        "transition_risk": profile.transition_risk,
        "green_capex_ratio": profile.green_capex_ratio,
        "scope3_exposure": profile.scope3_exposure,
        "absorption_buffer": profile.absorption_buffer,
        "hqla_eligible": profile.hqla_eligible,
        "hqla_level": profile.hqla_level,
        "exempt_from_staging": profile.exempt_from_staging,
        "rsf_weight": profile.rsf_weight,
    }


# ──────────────────────────────────────────────
# PORTFOLIO ANALYTICS (summary statistics)
# ──────────────────────────────────────────────

def compute_portfolio_sa_ccr_summary(df: pl.DataFrame) -> Dict[str, float]:
    """Calcule les statistiques SA-CCR agregees du portefeuille.

    Retourne les metriques cles pour le reporting reglementaire :
        - Gross notional, net notional
        - Gross MTM, net MTM (after netting)
        - Total RC (Replacement Cost)
        - Total PFE (Potential Future Exposure)
        - Total EAD (after alpha multiplier)
        - Netting benefit ratio
        - CSA coverage ratio
        - EAD / Gross Notional ratio

    Args:
        df: DataFrame from generate_derivative_positions().

    Returns:
        Dict de metriques portfolio-level.
    """
    gross_notional = df["notional"].sum()
    gross_mtm_positive = df.filter(pl.col("mtm") > 0)["mtm"].sum()
    gross_mtm_negative = abs(df.filter(pl.col("mtm") < 0)["mtm"].sum())
    net_mtm = df["mtm"].sum()
    total_ead = df["ead_sa_ccr"].sum()
    total_collateral = df["collateral"].sum()

    # Netting benefit
    if gross_mtm_positive > 0:
        netting_benefit = 1.0 - (max(0.0, net_mtm) / gross_mtm_positive)
    else:
        netting_benefit = 0.0

    # CSA coverage
    n_csa = df.filter(pl.col("csa_flag"))["csa_flag"].len()
    csa_coverage = n_csa / max(1, len(df))

    # EAD ratio
    ead_ratio = total_ead / max(1.0, gross_notional)

    # Desk breakdown
    desk_ead: Dict[str, float] = {}
    for desk in DESK_MIX:
        desk_df = df.filter(pl.col("desk") == desk)
        desk_ead[desk] = float(desk_df["ead_sa_ccr"].sum()) if len(desk_df) > 0 else 0.0

    return {
        "gross_notional": float(gross_notional),
        "gross_mtm_positive": float(gross_mtm_positive),
        "gross_mtm_negative": float(gross_mtm_negative),
        "net_mtm": float(net_mtm),
        "total_collateral": float(total_collateral),
        "total_ead": float(total_ead),
        "netting_benefit": round(netting_benefit, 4),
        "csa_coverage": round(csa_coverage, 4),
        "ead_to_notional_ratio": round(ead_ratio, 6),
        "ead_irs": desk_ead.get("irs", 0.0),
        "ead_fx": desk_ead.get("fx", 0.0),
        "ead_cds": desk_ead.get("cds", 0.0),
    }


def compute_counterparty_cva_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Calcule le resume CVA par contrepartie.

    Agrege les positions par contrepartie pour le reporting CVA :
        - EAD total par contrepartie
        - CVA charge par contrepartie (si cva_charge existe)
        - PD et LGD EAD-weighted par contrepartie
        - Nombre de netting sets

    Args:
        df: DataFrame from generate_derivative_positions() with cva_charge.

    Returns:
        DataFrame avec une ligne par contrepartie.
    """
    has_cva = "cva_charge" in df.columns

    # Group by counterparty
    agg_exprs = [
        pl.col("ead_sa_ccr").sum().alias("ead_total"),
        pl.col("notional").sum().alias("notional_total"),
        pl.col("netting_set").n_unique().alias("n_netting_sets"),
        pl.col("position_id").len().alias("n_positions"),
        pl.col("rating").first().alias("rating"),
        pl.col("country").first().alias("country"),
        pl.col("counterparty_name").first().alias("counterparty_name"),
    ]

    if has_cva:
        agg_exprs.append(pl.col("cva_charge").sum().alias("cva_total"))

    result = df.group_by("counterparty_code").agg(agg_exprs)

    # Sort by EAD descending
    result = result.sort("ead_total", descending=True)

    return result
