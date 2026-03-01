"""Generateur position-par-position pour le portefeuille Interbancaire.

Genere ~1500 depots/prets interbancaires individuels (25 contreparties
bancaires) avec PD multi-composante (rating + tenor + type + LFI),
LGD waterfall (collateral + seniorite + BRRD bail-in), RW CRR3 Art. 120
(ECRA), et stress test 4 canaux (LIBOR-OIS spread, GDP contagion,
funding liquidity freeze, reseau Eisenberg-Noe simplifie).

Le pipeline aval (ECL Vasicek, comparator RAROC, BL-CVaR) consomme alors
des valeurs calculees bottom-up plutot que des constantes parametriques.

Donnees reelles (priorite) :
    - ECB MMSR (Money Market Statistical Reporting) : taux, volumes
    - EURIBOR GitHub : term structure 1W-12M
    - Fallback : constantes hardcodees si parquet absent

Calibration :
    - S&P Global (1981-2023) : taux de defaut institutions financieres
    - Lehman Brothers (2008) : recovery 28.1% -> LGD = 71.9%
    - Washington Mutual (2008) : recovery 57% -> LGD = 43%
    - Thornton (2009), Taylor-Williams (2009) : LIBOR-OIS comme signal
    - Eisenberg-Noe (2001) : clearing vector, cascade de defauts
    - Glasserman-Young (2015) : amplification par fire sales

References :
    - CRR3 Art. 120 ECRA : RW 20/30/50/100% par CQS
    - CRR3 Art. 120(2) : short-term <=3M : CQS 1-3 = 20%
    - CRR3 Art. 161 F-IRB : LGD = 45% senior, 75% subordinated
    - CRR3 Art. 153(3) : multiplicateur x1.25 pour FI > 70B EUR
    - CRR3 Art. 222 CRM : repo/swap collatéralise -> RW reduit
    - BRRD Dir. 2014/59/EU Art. 44(2) : depots <7j exclus du bail-in
    - Basel III LCR : HQLA Level 2B (50% haircut)
    - Basel III NSFR : RSF = 0% pour O/N, 15% pour <6M
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import polars as pl


# ──────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────

# Rating -> PD TTC (S&P Global 1981-2023, institutions financieres)
RATING_PD_MAP: Dict[str, float] = {
    "AAA": 0.0001, "AA+": 0.0001, "AA": 0.0002, "AA-": 0.0003,
    "A+": 0.0004, "A": 0.0006, "A-": 0.0008,
    "BBB+": 0.0010, "BBB": 0.0018, "BBB-": 0.0030,
    "BB+": 0.0050, "BB": 0.0068,
}

# 25 contreparties bancaires (EBA Transparency Exercise 2024)
INTERBANK_COUNTERPARTIES: Dict[str, Dict] = {
    "DB":    {"name": "Deutsche Bank",     "country": "DE", "weight": 0.08, "rating": "A",    "pd_base": 0.0005, "tier1": 0.145},
    "BNP":   {"name": "BNP Paribas",       "country": "FR", "weight": 0.08, "rating": "AA-",  "pd_base": 0.0003, "tier1": 0.138},
    "SG":    {"name": "Societe Generale",   "country": "FR", "weight": 0.06, "rating": "A",    "pd_base": 0.0005, "tier1": 0.135},
    "CA":    {"name": "Credit Agricole",    "country": "FR", "weight": 0.06, "rating": "AA-",  "pd_base": 0.0003, "tier1": 0.170},
    "BPCE":  {"name": "BPCE/Natixis",       "country": "FR", "weight": 0.04, "rating": "A",    "pd_base": 0.0004, "tier1": 0.155},
    "ING":   {"name": "ING",                "country": "NL", "weight": 0.05, "rating": "A+",   "pd_base": 0.0004, "tier1": 0.148},
    "SAN":   {"name": "Santander",           "country": "ES", "weight": 0.05, "rating": "A",    "pd_base": 0.0005, "tier1": 0.125},
    "BBVA":  {"name": "BBVA",                "country": "ES", "weight": 0.04, "rating": "A",    "pd_base": 0.0005, "tier1": 0.128},
    "UCG":   {"name": "UniCredit",           "country": "IT", "weight": 0.05, "rating": "BBB",  "pd_base": 0.0012, "tier1": 0.150},
    "ISP":   {"name": "Intesa",              "country": "IT", "weight": 0.04, "rating": "BBB+", "pd_base": 0.0010, "tier1": 0.135},
    "JPM":   {"name": "JPMorgan",            "country": "US", "weight": 0.05, "rating": "AA-",  "pd_base": 0.0002, "tier1": 0.155},
    "C":     {"name": "Citi",                "country": "US", "weight": 0.04, "rating": "A+",   "pd_base": 0.0003, "tier1": 0.136},
    "HSBC":  {"name": "HSBC",                "country": "GB", "weight": 0.04, "rating": "AA-",  "pd_base": 0.0002, "tier1": 0.148},
    "BARC":  {"name": "Barclays",            "country": "GB", "weight": 0.03, "rating": "A",    "pd_base": 0.0005, "tier1": 0.139},
    "UBS":   {"name": "UBS",                 "country": "CH", "weight": 0.03, "rating": "AA-",  "pd_base": 0.0002, "tier1": 0.143},
    "GS":    {"name": "Goldman Sachs",       "country": "US", "weight": 0.02, "rating": "A+",   "pd_base": 0.0003, "tier1": 0.149},
    "CBK":   {"name": "Commerzbank",         "country": "DE", "weight": 0.03, "rating": "BBB+", "pd_base": 0.0008, "tier1": 0.142},
    "ABN":   {"name": "ABN Amro",            "country": "NL", "weight": 0.02, "rating": "A",    "pd_base": 0.0004, "tier1": 0.137},
    "KBC":   {"name": "KBC",                 "country": "BE", "weight": 0.02, "rating": "A+",   "pd_base": 0.0003, "tier1": 0.152},
    "NDA":   {"name": "Nordea",              "country": "FI", "weight": 0.02, "rating": "AA-",  "pd_base": 0.0002, "tier1": 0.180},
    "RABO":  {"name": "Rabobank",            "country": "NL", "weight": 0.02, "rating": "AA-",  "pd_base": 0.0002, "tier1": 0.165},
    "EBS":   {"name": "Erste",               "country": "AT", "weight": 0.02, "rating": "A",    "pd_base": 0.0004, "tier1": 0.160},
    "CABK":  {"name": "CaixaBank",           "country": "ES", "weight": 0.02, "rating": "BBB+", "pd_base": 0.0008, "tier1": 0.123},
    "DANSKE": {"name": "Danske",             "country": "DK", "weight": 0.02, "rating": "A",    "pd_base": 0.0004, "tier1": 0.190},
    "STAN":  {"name": "Standard Chartered",  "country": "GB", "weight": 0.02, "rating": "A",    "pd_base": 0.0005, "tier1": 0.138},
}

# 4 tenor buckets
TENOR_BUCKETS: Dict[str, Dict] = {
    "overnight": {"weight": 0.35, "tenor_days": 1,  "tenor_years": 0.003},
    "weekly":    {"weight": 0.25, "tenor_days": 7,  "tenor_years": 0.019},
    "monthly":   {"weight": 0.25, "tenor_days": 30, "tenor_years": 0.082},
    "quarterly": {"weight": 0.15, "tenor_days": 90, "tenor_years": 0.25},
}

# 4 instrument types
INSTRUMENT_TYPES: Dict[str, Dict] = {
    "unsecured_deposit": {"weight": 0.45, "haircut": 1.0,  "collateral_ratio": 0.0},
    "repo":              {"weight": 0.25, "haircut": 0.02, "collateral_ratio": 1.05},
    "swap_collateral":   {"weight": 0.20, "haircut": 0.0,  "collateral_ratio": 0.90},
    "central_bank_facility": {"weight": 0.10, "haircut": 0.0, "collateral_ratio": 1.0},
}

# CRR3 Art. 120 ECRA — Rating -> RW
CRR3_ART120_RW: Dict[str, float] = {
    "AAA": 0.20, "AA+": 0.20, "AA": 0.20, "AA-": 0.20,  # CQS 1
    "A+": 0.30, "A": 0.30, "A-": 0.30,                    # CQS 2
    "BBB+": 0.50, "BBB": 0.50, "BBB-": 0.50,              # CQS 3
    "BB+": 1.00, "BB": 1.00,                                # CQS 4
}

# Instrument type -> PD multiplier (collateral reduces credit risk)
_ALPHA_TYPE: Dict[str, float] = {
    "unsecured_deposit": 1.0,
    "repo": 0.3,
    "swap_collateral": 0.5,
    "central_bank_facility": 0.05,
}

# LGD parameters
_LGD_SENIOR = 0.45        # CRR3 Art. 161 F-IRB senior unsecured
_LGD_NOISE_STD = 0.02
_LGD_FLOOR = 0.02
_LGD_CAP = 0.65

# PD parameters
_PD_FLOOR = 0.00001
_PD_CAP = 0.02

# Stress base scenario
_BASE_GDP = 1.2
_BASE_RATE = 3.5

# LGD downturn
_LGD_DOWNTURN_COEFF = 0.03   # +3pp LGD per -1% GDP (interbank higher than sov)
_LGD_DOWNTURN_CAP = 0.10

# Notional parameters by tenor
_NOTIONAL_BY_TENOR: Dict[str, float] = {
    "overnight": 20e6,
    "weekly": 35e6,
    "monthly": 55e6,
    "quarterly": 80e6,
}

# Fallback interbank rates (EUR-STR based, Feb 2026)
_FALLBACK_RATES: Dict[str, float] = {
    "overnight": 0.0365,
    "weekly": 0.0368,
    "monthly": 0.0372,
    "quarterly": 0.0380,
}


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_interbank_market_data(path: Optional[str] = None) -> pl.DataFrame:
    """Charge les donnees marche interbancaire ECB MMSR.

    Args:
        path: Chemin vers le parquet. Si None, utilise le defaut.

    Returns:
        DataFrame avec tenor, rate_unsecured, rate_secured, volume, spread.

    Raises:
        FileNotFoundError: Si le parquet n'existe pas.
    """
    if path is None:
        path = str(Path(__file__).parent.parent / "data" / "ecb_interbank.parquet")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ECB interbank parquet not found: {p}")
    return pl.read_parquet(p)


def _get_rate_for_tenor(
    market_data: Optional[pl.DataFrame],
    tenor_years: float,
) -> float:
    """Interpolate interbank rate from market data or use fallback."""
    if market_data is not None and len(market_data) > 0:
        tenors = market_data["tenor_years"].to_numpy()
        rates = market_data["rate_unsecured"].to_numpy()
        if len(tenors) >= 2:
            return float(np.interp(tenor_years, tenors, rates))
    # Fallback: flat + slight slope
    return 0.0365 + tenor_years * 0.006


# ──────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────

def generate_interbank_positions(
    n_positions: int = 1500,
    total_ead: float = 2.5e9,
    seed: int = 962,
) -> pl.DataFrame:
    """Genere un portefeuille de depots/prets interbancaires position par position.

    Etape A: tente de charger les donnees ECB MMSR (taux, volumes).
             Si absent, fallback aux constantes.
    Etape B: sample counterparty (25 banques), tenor bucket, instrument type.
    Etape C: compute PD, LGD, RW pour chaque position.

    Args:
        n_positions: Nombre de positions a generer (~1500).
        total_ead: EAD total du portefeuille interbancaire.
        seed: Graine aleatoire.

    Returns:
        DataFrame avec ~20 colonnes (une ligne par position).
    """
    rng = np.random.default_rng(seed)

    # ── Etape A: Load market data (real or fallback) ──
    market_data = None
    try:
        market_data = load_interbank_market_data()
        if len(market_data) < 2:
            market_data = None
    except FileNotFoundError:
        pass

    # ── Sample counterparties ──
    cp_codes = list(INTERBANK_COUNTERPARTIES.keys())
    cp_weights = np.array([INTERBANK_COUNTERPARTIES[c]["weight"] for c in cp_codes])
    cp_weights = cp_weights / cp_weights.sum()
    sampled_cps = rng.choice(cp_codes, size=n_positions, p=cp_weights)

    # ── Sample tenor buckets ──
    bucket_names = list(TENOR_BUCKETS.keys())
    bucket_weights = np.array([TENOR_BUCKETS[b]["weight"] for b in bucket_names])
    bucket_weights = bucket_weights / bucket_weights.sum()
    sampled_buckets = rng.choice(bucket_names, size=n_positions, p=bucket_weights)

    # ── Sample instrument types ──
    instr_names = list(INSTRUMENT_TYPES.keys())
    instr_weights = np.array([INSTRUMENT_TYPES[i]["weight"] for i in instr_names])
    instr_weights = instr_weights / instr_weights.sum()
    sampled_instrs = rng.choice(instr_names, size=n_positions, p=instr_weights)

    # ── Generate individual positions ──
    records = []
    for i in range(n_positions):
        cp_code = sampled_cps[i]
        cp = INTERBANK_COUNTERPARTIES[cp_code]
        bucket_name = sampled_buckets[i]
        bucket = TENOR_BUCKETS[bucket_name]
        instr_name = sampled_instrs[i]
        instr = INSTRUMENT_TYPES[instr_name]

        tenor_days = bucket["tenor_days"]
        tenor_years = bucket["tenor_years"]

        # Notional (lognormal, scaled by tenor)
        median_notional = _NOTIONAL_BY_TENOR.get(bucket_name, 50e6)
        notional = rng.lognormal(np.log(median_notional), 0.5)

        # Collateral ratio
        collateral_ratio = instr["collateral_ratio"]

        # Rate
        base_rate = _get_rate_for_tenor(market_data, tenor_years)
        # Counterparty spread from rating (BBB pays more than AA)
        pd_base = cp["pd_base"]
        cp_spread = pd_base * 5.0  # ~spread = 5x PD (rough approximation)
        rate = base_rate + cp_spread

        records.append({
            "position_id": i,
            "counterparty_code": cp_code,
            "counterparty_name": cp["name"],
            "country": cp["country"],
            "rating": cp["rating"],
            "tier1_ratio": cp["tier1"],
            "tenor_bucket": bucket_name,
            "tenor_days": tenor_days,
            "tenor_years": round(tenor_years, 4),
            "instrument_type": instr_name,
            "notional": round(notional, 2),
            "collateral_ratio": collateral_ratio,
            "haircut": instr["haircut"],
            "rate": round(rate, 6),
            "rw_crr3": 0.0,  # placeholder, computed below
            "hqla_eligible": True,
            "hqla_level": 3,  # Level 2B, 50% haircut LCR
            "exempt_from_staging": False,  # NOT sovereign
        })

    df = pl.DataFrame(records)

    # ── Scale notionals to total_ead ──
    raw_total = df["notional"].sum()
    if raw_total > 0:
        df = df.with_columns((pl.col("notional") * (total_ead / raw_total)).alias("ead"))
    else:
        df = df.with_columns(pl.Series("ead", [total_ead / n_positions] * len(df)))

    # ── Compute PD, LGD, RW ──
    df = df.with_columns(pl.Series("pd_position", compute_interbank_pd(df)))
    df = df.with_columns(pl.Series("lgd_position", compute_interbank_lgd(df, rng)))
    df = df.with_columns(pl.Series("rw_crr3", compute_interbank_rw(df)))

    return df


# ──────────────────────────────────────────────
# PD MULTI-COMPOSANTE (S&P calibration)
# ──────────────────────────────────────────────

def compute_interbank_pd(df: pl.DataFrame) -> np.ndarray:
    """Calcule la PD multi-composante pour chaque position interbancaire.

    4 composantes multiplicatives :
        1. PD rating (S&P 1981-2023, institutions financieres)
        2. Alpha tenor (overnight quasi-nul, quarterly +50%)
        3. Alpha type (collateral reduces credit risk)
        4. Alpha LFI (CRR3 Art. 153(3) : x1.25 pour FI > 70B EUR)

    Args:
        df: DataFrame from generate_interbank_positions().

    Returns:
        Array de PD par position.
    """
    n = len(df)
    pd_result = np.zeros(n)

    ratings = df["rating"].to_numpy()
    tenor_years_arr = df["tenor_years"].to_numpy()
    instr_types = df["instrument_type"].to_numpy()

    for i in range(n):
        rating = ratings[i]
        tenor_years = tenor_years_arr[i]
        instr_type = instr_types[i]

        # 1. PD rating
        pd_rating = RATING_PD_MAP.get(rating, 0.0006)

        # 2. Tenor adjustment: overnight quasi-nul, quarterly +50%
        alpha_tenor = 1.0 + 0.5 * (tenor_years / 0.25)

        # 3. Instrument type (collateral reduces credit risk)
        alpha_type = _ALPHA_TYPE.get(instr_type, 1.0)

        # 4. Large FI multiplier (CRR3 Art. 153(3))
        # All 25 counterparties are > 70B EUR => always 1.25
        alpha_lfi = 1.25

        pd_result[i] = pd_rating * alpha_tenor * alpha_type * alpha_lfi

    return np.clip(pd_result, _PD_FLOOR, _PD_CAP)


# ──────────────────────────────────────────────
# LGD WATERFALL (BRRD-aware)
# ──────────────────────────────────────────────

def compute_interbank_lgd(
    df: pl.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Calcule la LGD pour chaque position interbancaire.

    Recovery waterfall (4 sources, empirique Lehman/WaMu) :
        1. Collateral recovery : min(1.0, collateral_ratio * (1 - haircut))
        2. Seniority recovery : 0.10 (senior unsecured)
        3. BRRD exclusion : 0.05 (depots <7j exclus du bail-in, Art. 44(2))

    LGD = 1.0 - total_recovery + noise

    Historique : Lehman LGD=72%, WaMu=43%, moyenne=60%
    CRR3 Art. 161 : F-IRB LGD = 45% senior unsecured

    Args:
        df: DataFrame from generate_interbank_positions().
        rng: Optional random generator for noise.

    Returns:
        Array de LGD par position.
    """
    n = len(df)
    lgd = np.zeros(n)

    collateral_ratios = df["collateral_ratio"].to_numpy()
    haircuts = df["haircut"].to_numpy()
    instr_types = df["instrument_type"].to_numpy()
    tenor_days_arr = df["tenor_days"].to_numpy()

    for i in range(n):
        collateral_ratio = collateral_ratios[i]
        haircut = haircuts[i]
        instr_type = instr_types[i]
        tenor_days = tenor_days_arr[i]

        # Recovery from collateral
        rr_collateral = min(1.0, collateral_ratio * (1.0 - haircut))

        # Recovery from seniority (senior unsecured)
        rr_seniority = 0.10 if collateral_ratio == 0.0 else 0.0

        # BRRD Art. 44(2): deposits < 7 days excluded from bail-in
        rr_brrd = 0.05 if (tenor_days < 7 and instr_type == "unsecured_deposit") else 0.0

        # Total recovery (capped at 1.0)
        rr = min(1.0, rr_collateral + rr_seniority + rr_brrd)

        lgd[i] = 1.0 - rr

    # Add noise
    if rng is not None:
        lgd += rng.normal(0.0, _LGD_NOISE_STD, n)

    return np.clip(lgd, _LGD_FLOOR, _LGD_CAP)


# ──────────────────────────────────────────────
# RW CRR3 ART. 120 ECRA
# ──────────────────────────────────────────────

def compute_interbank_rw(df: pl.DataFrame) -> np.ndarray:
    """Calcule le Risk Weight CRR3 Art. 120 pour chaque position.

    ECRA (External Credit Risk Assessment Approach) :
        - CQS 1 (AAA/AA) = 20%
        - CQS 2 (A) = 30%
        - CQS 3 (BBB) = 50%
        - CQS 4 (BB) = 100%

    Ajustements :
        - Art. 120(2) : short-term <=3M -> CQS 1-3 = 20%
        - Art. 222 CRM : repo/swap collatéralise -> 50% reduction
        - Central bank facility -> RW = 0%

    Args:
        df: DataFrame from generate_interbank_positions().

    Returns:
        Array de RW par position.
    """
    n = len(df)
    rw = np.zeros(n)

    ratings = df["rating"].to_numpy()
    tenor_days_arr = df["tenor_days"].to_numpy()
    instr_types = df["instrument_type"].to_numpy()

    for i in range(n):
        rating = ratings[i]
        tenor_days = tenor_days_arr[i]
        instr_type = instr_types[i]

        # Base RW from rating (ECRA)
        base_rw = CRR3_ART120_RW.get(rating, 1.00)

        # Art. 120(2): short-term claims <= 3M -> CQS 1-3 benefit (max 20%)
        if tenor_days <= 90 and base_rw <= 0.50:
            base_rw = min(base_rw, 0.20)

        # Central bank facility -> 0% RW
        if instr_type == "central_bank_facility":
            rw[i] = 0.0
            continue

        # Art. 222 CRM: collateralized (repo/swap) -> 50% reduction
        if instr_type in ("repo", "swap_collateral"):
            base_rw = base_rw * 0.50

        rw[i] = base_rw

    return np.clip(rw, 0.0, 1.50)


# ──────────────────────────────────────────────
# STRESS TEST (4 canaux EBA-calibre)
# ──────────────────────────────────────────────

def stress_interbank_positions(
    df: pl.DataFrame,
    macro_params: Dict[str, float],
) -> Dict[str, float]:
    """Stress test des positions interbancaires et re-aggregation.

    4 canaux de transmission :
        1. LIBOR-OIS spread (Thornton 2009) — taux -> credit interbancaire
        2. GDP -> credit counterpartie (Glasserman-Young 2015)
        3. Funding liquidity freeze (taux + GDP negatif -> amplification)
        4. Contagion reseau (Eisenberg-Noe simplifie)

    LGD downturn : lgd_addon = clip(-delta_gdp * 0.03, 0, 0.10)
    RW : Art. 120(2) short-term benefit perdu en freeze

    Calibration cible : GFC (GDP -5%, rates +300bp) -> PD >= 3x base

    Args:
        df: DataFrame from generate_interbank_positions().
        macro_params: Dict avec gdp_growth, interest_rate, etc.

    Returns:
        Dict avec pd_base, lgd_base, rw_crr3 stresses (EAD-weighted).
    """
    # Deltas (in percentage points: interest_rate=6.5 vs base=3.5 → delta=3.0)
    delta_gdp = macro_params.get("gdp_growth", _BASE_GDP) - _BASE_GDP
    delta_rate = macro_params.get("interest_rate", _BASE_RATE) - _BASE_RATE

    n = len(df)
    ead = df["ead"].to_numpy()
    pd_base_arr = df["pd_position"].to_numpy()

    # ── Canal 1 : LIBOR-OIS spread (Thornton 2009) ──
    # delta_rate en pp -> LIBOR-OIS implied ~65bp per 1pp rate increase
    # Normal ~10bp, GFC 364bp, COVID 140bp, souveraine 100bp
    libor_ois_bps = max(0.0, delta_rate * 65.0)
    # Doublement tous les 50bp de spread
    spread_mult = 1.0 + libor_ois_bps / 50.0

    # ── Canal 2 : GDP -> credit counterpartie (Glasserman-Young) ──
    # delta_gdp in pp: GDP from 1.2 to -3.8 → delta = -5.0
    gdp_mult = 1.0 + max(0.0, -delta_gdp) * 0.25  # 25% per 1pp GDP drop

    # ── Canal 3 : Funding liquidity (volume crunch) ──
    # Gel interbancaire : taux ET GDP negatif ensemble -> amplification non-lineaire
    freeze_indicator = max(0.0, delta_rate) * max(0.0, -delta_gdp)
    liquidity_mult = 1.0 + freeze_indicator * 0.15

    # ── Canal 4 : Contagion reseau (Eisenberg-Noe simplifie) ──
    # Si PD du portefeuille credit augmente -> contagion -> PD interbank
    contagion_mult = 1.0 + max(0.0, -delta_gdp) * 0.10

    # ── Combine (multiplicatif — Jensen-correct) ──
    pd_stressed = (
        pd_base_arr
        * spread_mult
        * gdp_mult
        * max(1.0, liquidity_mult)
        * contagion_mult
    )
    pd_stressed = np.clip(pd_stressed, _PD_FLOOR, _PD_CAP)

    # ── LGD downturn ──
    # delta_gdp in pp (already computed above)
    lgd_addon = np.clip(
        max(0.0, -delta_gdp) * 0.015,  # +1.5pp LGD per 1pp GDP drop
        0.0,
        _LGD_DOWNTURN_CAP,
    )
    lgd_stressed = np.clip(df["lgd_position"].to_numpy() + lgd_addon, _LGD_FLOOR, _LGD_CAP)

    # ── RW : Art. 120(2) short-term benefit lost in freeze ──
    rw_base = df["rw_crr3"].to_numpy().copy()
    if freeze_indicator > 0.5:
        # When interbank market freezes, short-term rollover is impossible
        # -> remove Art. 120(2) short-term benefit
        tenor_days_arr = df["tenor_days"].to_numpy()
        ratings_arr = df["rating"].to_numpy()
        instr_types_arr = df["instrument_type"].to_numpy()
        for i in range(n):
            if tenor_days_arr[i] <= 90:
                rating = ratings_arr[i]
                instr_type = instr_types_arr[i]
                full_rw = CRR3_ART120_RW.get(rating, 1.00)
                if instr_type == "central_bank_facility":
                    full_rw = 0.0
                elif instr_type in ("repo", "swap_collateral"):
                    full_rw = full_rw * 0.50
                rw_base[i] = full_rw

    # ── EAD-weighted aggregation ──
    total_ead = ead.sum()
    if total_ead <= 0:
        return {"pd_base": 0.0005, "lgd_base": 0.45, "rw_crr3": 0.20}

    w = ead / total_ead
    pd_agg = float(np.dot(w, pd_stressed))
    lgd_agg = float(np.dot(w, lgd_stressed))
    rw_agg = float(np.dot(w, rw_base))

    return {
        "pd_base": round(pd_agg, 6),
        "lgd_base": round(lgd_agg, 6),
        "rw_crr3": round(rw_agg, 4),
    }


# ──────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────

def aggregate_interbank_to_balance_row(
    df_positions: pl.DataFrame,
    profile: "AssetClassProfile",
) -> dict:
    """Agrege les positions interbancaires en une ligne balance sheet.

    EAD-weighted PD/LGD/tenor/RW.
    HQLA Level 2B (50% haircut), rsf_weight=0.0, NOT exempt from staging.

    Args:
        df_positions: DataFrame from generate_interbank_positions().
        profile: AssetClassProfile for interbank.

    Returns:
        Dict avec toutes les colonnes balance sheet (19 cles).
    """
    weights = df_positions["ead"].to_numpy()
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
        tenor_agg = float(np.dot(w, df_positions["tenor_years"].to_numpy()))
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
