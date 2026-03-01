"""Generateur position-par-position pour le portefeuille Obligations Corporate.

Genere ~200 obligations corporate individuelles (30 emetteurs europeens)
avec PD rating-based + secteur overlay, LGD par seniorite (Moody's URD),
RW CRR3 ECRA, et stress test 4 canaux position-par-position.

Le pipeline aval (ECL Vasicek, comparator RAROC, BL-CVaR) consomme alors
des valeurs calculees bottom-up plutot que des constantes parametriques.

Deux modes :
    - Mode reel (parquet present) : echantillonne depuis le parquet
      pre-traite, enrichit avec PD/LGD/spread calcules
    - Mode fallback (parquet absent) : generation 100% parametrique depuis
      constantes calibrees CORPORATE_ISSUERS

Calibration :
    - Moody's 2024 Annual Default Study : taux de defaut par rating
    - Moody's Ultimate Recovery Database : recovery par seniorite
    - ICE BofA Corporate Index : spreads par rating
    - S&P Global Fixed Income Research : transition matrices
    - EBA GL/2019/03 : LGD downturn add-on methodology

References :
    - CRR3 Art. 120 ECRA : RW par External Credit Assessment
    - CRR3 Art. 161 : LGD F-IRB floors (45% senior unsecured, 75% sub)
    - CRR3 Art. 153(1) : PD input floor 0.03% (3 bps)
    - IFRS 9 5.5.9 : SICR assessment via credit quality deterioration
    - Basel III NSFR : RSF = 65% for IG corporate bonds, 85% for HY
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from ifrs9_cockpit.utils.helpers import logit, expit


# ──────────────────────────────────────────────
# CONSTANTES (calibrees Moody's 2024 / ICE BofA / CRR3)
# ──────────────────────────────────────────────

# --- Rating -> PD TTC (Moody's 2024 Annual Default Study) ---
# Annualized one-year default rates (1983-2023, investment grade + HY)
RATING_PD_MAP: Dict[str, float] = {
    "AAA":  0.0001,   # 0.01%
    "AA":   0.0002,   # 0.02%
    "A":    0.0005,   # 0.05%
    "BBB":  0.0025,   # 0.25%
    "BB":   0.0150,   # 1.50%
    "B":    0.0300,   # 3.00%
    "CCC":  0.1000,   # 10.00%
}

# --- Seniority -> LGD (Moody's Ultimate Recovery Database 1987-2023) ---
# Issuer-weighted ultimate recovery rates, LGD = 1 - Recovery
SENIORITY_LGD_MAP: Dict[str, float] = {
    "secured":      0.35,   # Senior secured: RR ~65%
    "unsecured":    0.45,   # Senior unsecured: RR ~55%
    "subordinated": 0.70,   # Subordinated: RR ~30%
}

# --- Rating -> Credit Spread (bps, ICE BofA EUR Corporate Index, 5yr) ---
RATING_SPREAD_BPS: Dict[str, float] = {
    "AAA":  30.0,
    "AA":   40.0,
    "A":    60.0,
    "BBB": 130.0,
    "BB":  300.0,
    "B":   500.0,
    "CCC": 900.0,
}

# --- CRR3 ECRA Risk Weights (Art. 120, corporates) ---
# External Credit Assessment Approach
RATING_RW_MAP: Dict[str, float] = {
    "AAA": 0.20,   # CQS 1
    "AA":  0.20,   # CQS 1
    "A":   0.50,   # CQS 2
    "BBB": 0.75,   # CQS 3
    "BB":  1.00,   # CQS 4
    "B":   1.50,   # CQS 5
    "CCC": 1.50,   # CQS 6
}

# --- GICS Sector Classification (8 sectors) ---
SECTOR_GICS: List[str] = [
    "Financials",
    "Industrials",
    "Utilities",
    "Technology",
    "Healthcare",
    "Consumer",
    "Energy",
    "Telecom",
]

# --- Sector-specific overlay factors ---
# GDP sensitivity for PD stress (higher = more cyclical)
_SECTOR_GDP_SENSITIVITY: Dict[str, float] = {
    "Financials":  1.2,
    "Industrials": 1.4,
    "Utilities":   0.6,
    "Technology":  1.0,
    "Healthcare":  0.5,
    "Consumer":    1.3,
    "Energy":      1.5,
    "Telecom":     0.8,
}

# Sector PD overlay (multiplicative, relative to pure rating PD)
_SECTOR_PD_OVERLAY: Dict[str, float] = {
    "Financials":  1.05,   # Slightly above due to systemic risk
    "Industrials": 1.10,   # Cyclical
    "Utilities":   0.80,   # Regulated, defensive
    "Technology":  0.95,   # Mixed: high growth offset by vol
    "Healthcare":  0.75,   # Defensive
    "Consumer":    1.15,   # Cyclical discretionary
    "Energy":      1.20,   # Commodity + transition risk
    "Telecom":     0.90,   # Regulated, stable cash flows
}

# Sector-level spread overlay (additive bps)
_SECTOR_SPREAD_ADDON_BPS: Dict[str, float] = {
    "Financials":   5.0,
    "Industrials": 10.0,
    "Utilities":   -5.0,
    "Technology":   0.0,
    "Healthcare":  -8.0,
    "Consumer":    12.0,
    "Energy":      15.0,
    "Telecom":     -3.0,
}

# --- 30 European Corporate Issuers (calibrated, diversified) ---
# (name, country, rating, sector, typical_debt_eur, seniority_mix)
# seniority_mix: {secured: weight, unsecured: weight, subordinated: weight}
CORPORATE_ISSUERS: List[Dict] = [
    # Financials (5)
    {"name": "Allianz SE",             "country": "DE", "rating": "AA",  "sector": "Financials",  "typical_debt": 25e9, "seniority_mix": {"secured": 0.10, "unsecured": 0.70, "subordinated": 0.20}, "weight": 0.050},
    {"name": "AXA SA",                 "country": "FR", "rating": "A",   "sector": "Financials",  "typical_debt": 22e9, "seniority_mix": {"secured": 0.05, "unsecured": 0.65, "subordinated": 0.30}, "weight": 0.045},
    {"name": "Zurich Insurance",       "country": "CH", "rating": "AA",  "sector": "Financials",  "typical_debt": 18e9, "seniority_mix": {"secured": 0.10, "unsecured": 0.70, "subordinated": 0.20}, "weight": 0.035},
    {"name": "Munich Re",              "country": "DE", "rating": "AA",  "sector": "Financials",  "typical_debt": 15e9, "seniority_mix": {"secured": 0.15, "unsecured": 0.65, "subordinated": 0.20}, "weight": 0.030},
    {"name": "Generali",               "country": "IT", "rating": "BBB", "sector": "Financials",  "typical_debt": 20e9, "seniority_mix": {"secured": 0.05, "unsecured": 0.60, "subordinated": 0.35}, "weight": 0.030},
    # Industrials (5)
    {"name": "Siemens AG",             "country": "DE", "rating": "A",   "sector": "Industrials", "typical_debt": 35e9, "seniority_mix": {"secured": 0.20, "unsecured": 0.70, "subordinated": 0.10}, "weight": 0.050},
    {"name": "Schneider Electric",     "country": "FR", "rating": "A",   "sector": "Industrials", "typical_debt": 18e9, "seniority_mix": {"secured": 0.25, "unsecured": 0.65, "subordinated": 0.10}, "weight": 0.040},
    {"name": "Airbus SE",              "country": "NL", "rating": "A",   "sector": "Industrials", "typical_debt": 22e9, "seniority_mix": {"secured": 0.15, "unsecured": 0.75, "subordinated": 0.10}, "weight": 0.040},
    {"name": "Saint-Gobain",           "country": "FR", "rating": "BBB", "sector": "Industrials", "typical_debt": 14e9, "seniority_mix": {"secured": 0.20, "unsecured": 0.65, "subordinated": 0.15}, "weight": 0.030},
    {"name": "ThyssenKrupp",           "country": "DE", "rating": "BB",  "sector": "Industrials", "typical_debt": 8e9,  "seniority_mix": {"secured": 0.30, "unsecured": 0.55, "subordinated": 0.15}, "weight": 0.020},
    # Utilities (4)
    {"name": "Engie SA",               "country": "FR", "rating": "A",   "sector": "Utilities",   "typical_debt": 30e9, "seniority_mix": {"secured": 0.15, "unsecured": 0.75, "subordinated": 0.10}, "weight": 0.045},
    {"name": "Enel SpA",               "country": "IT", "rating": "BBB", "sector": "Utilities",   "typical_debt": 45e9, "seniority_mix": {"secured": 0.10, "unsecured": 0.70, "subordinated": 0.20}, "weight": 0.045},
    {"name": "RWE AG",                 "country": "DE", "rating": "BBB", "sector": "Utilities",   "typical_debt": 12e9, "seniority_mix": {"secured": 0.20, "unsecured": 0.65, "subordinated": 0.15}, "weight": 0.030},
    {"name": "Iberdrola",              "country": "ES", "rating": "BBB", "sector": "Utilities",   "typical_debt": 42e9, "seniority_mix": {"secured": 0.10, "unsecured": 0.75, "subordinated": 0.15}, "weight": 0.040},
    # Technology (3)
    {"name": "SAP SE",                 "country": "DE", "rating": "A",   "sector": "Technology",  "typical_debt": 12e9, "seniority_mix": {"secured": 0.05, "unsecured": 0.85, "subordinated": 0.10}, "weight": 0.040},
    {"name": "ASML Holding",           "country": "NL", "rating": "A",   "sector": "Technology",  "typical_debt": 6e9,  "seniority_mix": {"secured": 0.10, "unsecured": 0.80, "subordinated": 0.10}, "weight": 0.035},
    {"name": "Nokia Oyj",              "country": "FI", "rating": "BBB", "sector": "Technology",  "typical_debt": 5e9,  "seniority_mix": {"secured": 0.10, "unsecured": 0.70, "subordinated": 0.20}, "weight": 0.025},
    # Healthcare (3)
    {"name": "Roche Holding",          "country": "CH", "rating": "AA",  "sector": "Healthcare",  "typical_debt": 20e9, "seniority_mix": {"secured": 0.10, "unsecured": 0.80, "subordinated": 0.10}, "weight": 0.040},
    {"name": "Novartis AG",            "country": "CH", "rating": "AA",  "sector": "Healthcare",  "typical_debt": 25e9, "seniority_mix": {"secured": 0.10, "unsecured": 0.80, "subordinated": 0.10}, "weight": 0.040},
    {"name": "Sanofi SA",              "country": "FR", "rating": "A",   "sector": "Healthcare",  "typical_debt": 18e9, "seniority_mix": {"secured": 0.05, "unsecured": 0.80, "subordinated": 0.15}, "weight": 0.035},
    # Consumer (4)
    {"name": "LVMH SE",               "country": "FR", "rating": "A",   "sector": "Consumer",    "typical_debt": 28e9, "seniority_mix": {"secured": 0.05, "unsecured": 0.85, "subordinated": 0.10}, "weight": 0.045},
    {"name": "Unilever",               "country": "NL", "rating": "A",   "sector": "Consumer",    "typical_debt": 24e9, "seniority_mix": {"secured": 0.10, "unsecured": 0.80, "subordinated": 0.10}, "weight": 0.035},
    {"name": "Danone SA",              "country": "FR", "rating": "BBB", "sector": "Consumer",    "typical_debt": 16e9, "seniority_mix": {"secured": 0.10, "unsecured": 0.75, "subordinated": 0.15}, "weight": 0.025},
    {"name": "Casino Guichard",        "country": "FR", "rating": "CCC", "sector": "Consumer",    "typical_debt": 6e9,  "seniority_mix": {"secured": 0.40, "unsecured": 0.40, "subordinated": 0.20}, "weight": 0.010},
    # Energy (3)
    {"name": "TotalEnergies SE",       "country": "FR", "rating": "AA",  "sector": "Energy",      "typical_debt": 40e9, "seniority_mix": {"secured": 0.15, "unsecured": 0.75, "subordinated": 0.10}, "weight": 0.045},
    {"name": "Shell plc",              "country": "NL", "rating": "AA",  "sector": "Energy",      "typical_debt": 50e9, "seniority_mix": {"secured": 0.10, "unsecured": 0.80, "subordinated": 0.10}, "weight": 0.040},
    {"name": "Repsol SA",              "country": "ES", "rating": "BBB", "sector": "Energy",      "typical_debt": 10e9, "seniority_mix": {"secured": 0.20, "unsecured": 0.65, "subordinated": 0.15}, "weight": 0.025},
    # Telecom (3)
    {"name": "Deutsche Telekom",       "country": "DE", "rating": "BBB", "sector": "Telecom",     "typical_debt": 60e9, "seniority_mix": {"secured": 0.10, "unsecured": 0.75, "subordinated": 0.15}, "weight": 0.040},
    {"name": "Orange SA",              "country": "FR", "rating": "BBB", "sector": "Telecom",     "typical_debt": 25e9, "seniority_mix": {"secured": 0.10, "unsecured": 0.70, "subordinated": 0.20}, "weight": 0.030},
    {"name": "Telefonica SA",          "country": "ES", "rating": "BBB", "sector": "Telecom",     "typical_debt": 30e9, "seniority_mix": {"secured": 0.15, "unsecured": 0.65, "subordinated": 0.20}, "weight": 0.030},
]

# --- Maturity buckets ---
_MATURITY_BUCKETS: Dict[str, Dict] = {
    "short":      {"weight": 0.20, "tenor_min": 0.5, "tenor_max": 3.0},
    "medium":     {"weight": 0.45, "tenor_min": 3.0, "tenor_max": 7.0},
    "long":       {"weight": 0.25, "tenor_min": 7.0, "tenor_max": 12.0},
    "ultra_long": {"weight": 0.10, "tenor_min": 12.0, "tenor_max": 30.0},
}

# --- Coupon types ---
_COUPON_TYPES: Dict[str, float] = {
    "fixed":    0.75,
    "floating": 0.20,
    "zero":     0.05,
}

# --- Rating transition probabilities (1yr, Moody's 2024) ---
# For stress: probability of downgrade (1 notch) under adverse conditions
_RATING_DOWNGRADE_PROB: Dict[str, float] = {
    "AAA": 0.02,
    "AA":  0.04,
    "A":   0.06,
    "BBB": 0.10,
    "BB":  0.15,
    "B":   0.20,
    "CCC": 0.30,
}

# Rating ladder for migration (one-notch downgrade)
_RATING_LADDER: List[str] = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]

# --- Scalaires ---
_PD_FLOOR = 0.0003          # CRR3 Art. 153(1) : PD input floor 3 bps
_PD_CAP = 0.30              # cap for distressed
_LGD_FLOOR_CRR3 = 0.25      # CRR3 input floor (unsecured)
_LGD_CAP = 0.95
_LGD_NOISE_STD = 0.03       # idiosyncratic noise on LGD
_LGD_DOWNTURN_COEFF = 0.15  # EBA GL/2019/03 : LGD downturn multiplier

# Stress base values (from SCENARIO_BASE)
_BASE_GDP = 1.2
_BASE_RATE = 3.5
_BASE_UNEMP = 7.5
_BASE_INFLATION = 2.5

# Target mean PD for portfolio (~1.2%, matching typical EUR IG/HY blend)
_TARGET_MEAN_PD = 0.012

# --- Parquet location ---
_DATA_DIR = Path(__file__).parent.parent / "data"
_PARQUET_PATH = _DATA_DIR / "corporate_bonds.parquet"


# ──────────────────────────────────────────────
# CHARGEMENT DONNEES REELLES
# ──────────────────────────────────────────────

def load_corporate_bonds_data(path: Optional[str] = None) -> pl.DataFrame:
    """Charge le parquet pre-traite si present.

    Tente de lire ``data/corporate_bonds.parquet`` contenant des donnees
    d'obligations corporate europeennes (pre-traitees a partir de sources
    de marche : ICE BofA, Bloomberg LEAG, etc.).

    Args:
        path: Chemin vers le parquet. Par defaut : ``data/corporate_bonds.parquet``.

    Returns:
        DataFrame Polars avec donnees reelles.

    Raises:
        FileNotFoundError: Si le parquet n'existe pas.
    """
    if path is None:
        p = _PARQUET_PATH
    else:
        p = Path(path)

    if not p.exists():
        raise FileNotFoundError(
            f"Corporate bonds parquet not found: {p}. "
            "Using parametric fallback."
        )
    return pl.read_parquet(p)


# ──────────────────────────────────────────────
# GENERATION (REEL + ENRICHISSEMENT OU FALLBACK)
# ──────────────────────────────────────────────

def generate_corporate_bonds_positions(
    n_positions: int = 200,
    total_ead: float = 5.0e9,
    seed: int = 123,
) -> pl.DataFrame:
    """Generate individual corporate bond positions.

    Two modes:
    - Real mode (parquet present): sample from real data, enrich with
      PD/LGD/spread computed from rating + sector overlay
    - Fallback mode (parquet absent): 100% parametric from CORPORATE_ISSUERS

    Distribution target: ~70% IG (AAA/AA/A/BBB) / ~30% HY (BB/B/CCC)
    to match typical European corporate bond index composition.

    Args:
        n_positions: Number of positions to generate (~200).
        total_ead: Target total EAD for the corporate bond portfolio.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with ~20 columns per position (one row per bond).
    """
    rng = np.random.default_rng(seed)

    # Try loading real data
    df_real: Optional[pl.DataFrame] = None
    try:
        df_real = load_corporate_bonds_data()
    except FileNotFoundError:
        pass

    if df_real is not None and len(df_real) > 0:
        df = _generate_from_real(df_real, n_positions, total_ead, rng)
    else:
        df = _generate_fallback(n_positions, total_ead, rng)

    # Compute PD, LGD, RW, spread
    df = compute_corporate_bonds_pd(df)
    df = compute_corporate_bonds_lgd(df, rng)
    df = compute_corporate_bonds_rw(df)

    # Spread from rating + sector
    df = _compute_spread(df)

    # Default flag (PD-based)
    pd_arr = df["pd_base"].to_numpy()
    default_flags = rng.binomial(1, np.minimum(pd_arr, 0.50))
    df = df.with_columns(pl.Series("default_flag", default_flags))

    return df


def _generate_from_real(
    df_real: pl.DataFrame,
    n_positions: int,
    total_ead: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Generate positions from real parquet data with parametric enrichment.

    Samples from the real dataset and maps columns to the position schema.
    Enriches with sector overlay and seniority from CORPORATE_ISSUERS
    when available.

    Args:
        df_real: Real data DataFrame from parquet.
        n_positions: Number of positions to generate.
        total_ead: Target total EAD.
        rng: Random number generator.

    Returns:
        DataFrame with base columns (PD/LGD/RW computed later).
    """
    replace = n_positions > len(df_real)
    idx = rng.choice(len(df_real), size=n_positions, replace=replace)
    df_sampled = df_real[idx.tolist()]

    col_names = set(df_sampled.columns)
    records = []

    seniority_names = list(SENIORITY_LGD_MAP.keys())
    seniority_default_weights = np.array([0.20, 0.65, 0.15])

    for i in range(n_positions):
        row = df_sampled.row(i, named=True)

        # Extract real data fields with fallback
        issuer_name = str(row.get("issuer_name", row.get("issuer", f"Corp_{i:04d}")))
        country = str(row.get("country", row.get("issuer_country", "FR")))
        rating = str(row.get("rating", row.get("implied_rating", "BBB")))
        # Normalize rating to our map
        if rating not in RATING_PD_MAP:
            # Try stripping +/- for approximate match
            rating_base = rating.rstrip("+-")
            rating = rating_base if rating_base in RATING_PD_MAP else "BBB"

        sector = str(row.get("sector_gics", row.get("sector", "Industrials")))
        if sector not in SECTOR_GICS:
            sector = rng.choice(SECTOR_GICS)

        # Tenor
        tenor = float(row.get("tenor_years", row.get("remaining_maturity", 5.0)))
        tenor = float(np.clip(tenor, 0.5, 30.0))

        # Coupon
        coupon = float(row.get("coupon_rate", row.get("coupon", 0.035)))
        if coupon > 1.0:
            coupon = coupon / 100.0  # Convert from pct to decimal
        coupon = float(np.clip(coupon, 0.0, 0.15))

        # Seniority
        seniority_str = str(row.get("seniority", ""))
        if seniority_str.lower() in SENIORITY_LGD_MAP:
            seniority = seniority_str.lower()
        else:
            seniority = rng.choice(seniority_names, p=seniority_default_weights)

        records.append({
            "position_id": i,
            "issuer": issuer_name,
            "country": country,
            "rating": rating,
            "sector_gics": sector,
            "seniority": seniority,
            "tenor_years": round(tenor, 2),
            "coupon": round(coupon, 6),
            "notional": 0.0,  # filled after
            "ead": 0.0,
        })

    df_out = pl.DataFrame(records)

    # Scale EAD (lognormal then scale to target)
    raw_ead = rng.lognormal(mean=np.log(30e6), sigma=0.7, size=n_positions)
    raw_total = raw_ead.sum()
    scale = total_ead / raw_total if raw_total > 0 else 1.0
    scaled_ead = np.round(raw_ead * scale, 2)
    df_out = df_out.with_columns(pl.Series("notional", scaled_ead))
    df_out = df_out.with_columns(pl.col("notional").alias("ead"))

    return df_out


def _generate_fallback(
    n_positions: int,
    total_ead: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Generate 100% parametric corporate bond positions from constants.

    Distribution target: ~70% Investment Grade / ~30% High Yield.
    Issuers drawn from CORPORATE_ISSUERS weighted by ``weight``.
    Seniority sampled from each issuer's ``seniority_mix``.

    Args:
        n_positions: Number of positions to generate.
        total_ead: Target total EAD.
        rng: Random number generator.

    Returns:
        DataFrame with base columns (PD/LGD/RW computed later).
    """
    # Issuer distribution (weighted)
    issuer_weights = np.array([iss["weight"] for iss in CORPORATE_ISSUERS])
    issuer_weights = issuer_weights / issuer_weights.sum()
    issuer_idx = rng.choice(len(CORPORATE_ISSUERS), size=n_positions, p=issuer_weights)

    # Maturity buckets
    bucket_names = list(_MATURITY_BUCKETS.keys())
    bucket_weights = np.array([_MATURITY_BUCKETS[b]["weight"] for b in bucket_names])
    bucket_weights = bucket_weights / bucket_weights.sum()
    sampled_buckets = rng.choice(bucket_names, size=n_positions, p=bucket_weights)

    # Coupon types
    coupon_names = list(_COUPON_TYPES.keys())
    coupon_weights = np.array([_COUPON_TYPES[c] for c in coupon_names])
    coupon_weights = coupon_weights / coupon_weights.sum()
    sampled_coupons = rng.choice(coupon_names, size=n_positions, p=coupon_weights)

    records = []
    for i in range(n_positions):
        idx = issuer_idx[i]
        iss = CORPORATE_ISSUERS[idx]
        bucket = _MATURITY_BUCKETS[sampled_buckets[i]]

        # Tenor uniform in bucket range
        tenor = rng.uniform(bucket["tenor_min"], bucket["tenor_max"])

        # Seniority from issuer's seniority mix
        sen_names = list(iss["seniority_mix"].keys())
        sen_weights = np.array([iss["seniority_mix"][s] for s in sen_names])
        sen_weights = sen_weights / sen_weights.sum()
        seniority = rng.choice(sen_names, p=sen_weights)

        # Coupon rate: base from rating spread + small noise
        rating = iss["rating"]
        base_spread_pct = RATING_SPREAD_BPS.get(rating, 130.0) / 10000.0
        risk_free_approx = 0.035  # ~3.5% EUR swap rate
        coupon_type = sampled_coupons[i]

        if coupon_type == "fixed":
            coupon = risk_free_approx + base_spread_pct + rng.normal(0, 0.003)
        elif coupon_type == "floating":
            coupon = base_spread_pct + rng.normal(0, 0.002)  # spread over benchmark
        else:  # zero coupon
            coupon = 0.0

        coupon = float(np.clip(coupon, 0.0, 0.15))

        records.append({
            "position_id": i,
            "issuer": iss["name"],
            "country": iss["country"],
            "rating": rating,
            "sector_gics": iss["sector"],
            "seniority": seniority,
            "tenor_years": round(float(tenor), 2),
            "coupon": round(float(coupon), 6),
            "notional": 0.0,  # filled after
            "ead": 0.0,
        })

    df = pl.DataFrame(records)

    # Scale EAD (lognormal then scale to target)
    raw_ead = rng.lognormal(mean=np.log(30e6), sigma=0.7, size=n_positions)
    raw_total = raw_ead.sum()
    scale = total_ead / raw_total if raw_total > 0 else 1.0
    scaled_ead = np.round(raw_ead * scale, 2)
    df = df.with_columns(pl.Series("notional", scaled_ead))
    df = df.with_columns(pl.col("notional").alias("ead"))

    return df


# ──────────────────────────────────────────────
# PD : RATING-BASED + SECTOR OVERLAY
# ──────────────────────────────────────────────

def compute_corporate_bonds_pd(df: pl.DataFrame) -> pl.DataFrame:
    """Compute position-level PD from rating with sector overlay.

    Model:
        PD_base = RATING_PD_MAP[rating] x sector_overlay x alpha_maturity

    Where:
        - sector_overlay: multiplicative sector cyclicality factor
        - alpha_maturity: slight uplift for longer tenor (1 + 0.01 * max(0, tenor - 5))

    The intercept is calibrated for portfolio mean PD ~1.2% which reflects
    a typical EUR IG/HY mix (70/30).

    CRR3 input floor applied: PD >= 3 bps (Art. 153(1)).

    Args:
        df: DataFrame with corporate bond positions.

    Returns:
        DataFrame with added ``pd_base`` column.
    """
    n = len(df)
    ratings = df["rating"].to_numpy()
    sectors = df["sector_gics"].to_numpy()
    tenors = df["tenor_years"].to_numpy()

    pd_values = np.zeros(n, dtype=float)

    for i in range(n):
        rating = ratings[i]
        sector = sectors[i]
        tenor = tenors[i]

        # Base PD from rating
        pd_rating = RATING_PD_MAP.get(rating, 0.0025)

        # Sector overlay
        sector_overlay = _SECTOR_PD_OVERLAY.get(sector, 1.0)

        # Maturity adjustment: longer bonds have slightly higher PD
        # (cumulative default probability effect)
        alpha_maturity = 1.0 + 0.01 * max(0.0, tenor - 5.0)

        pd_values[i] = pd_rating * sector_overlay * alpha_maturity

    # Intercept calibration: shift to target ~1.2% mean
    # Compute current mean and adjust
    current_mean = pd_values.mean()
    if current_mean > 0:
        calibration_factor = _TARGET_MEAN_PD / current_mean
        pd_values = pd_values * calibration_factor

    # Apply CRR3 floors and caps
    pd_values = np.clip(pd_values, _PD_FLOOR, _PD_CAP)

    return df.with_columns(pl.Series("pd_base", np.round(pd_values, 6)))


# ──────────────────────────────────────────────
# LGD : SENIORITY-BASED + NOISE
# ──────────────────────────────────────────────

def compute_corporate_bonds_lgd(
    df: pl.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Compute position-level LGD from seniority with idiosyncratic noise.

    Model (Moody's Ultimate Recovery Database):
        LGD = SENIORITY_LGD_MAP[seniority] + noise

    Where:
        - Secured: LGD ~0.35 (RR ~65%), senior liens
        - Unsecured: LGD ~0.45 (RR ~55%), senior unsecured
        - Subordinated: LGD ~0.70 (RR ~30%), junior claims

    CRR3 input floor applied: LGD >= 25% (Art. 161(1)).

    Args:
        df: DataFrame with corporate bond positions.
        rng: Random generator for idiosyncratic noise.

    Returns:
        DataFrame with added ``lgd_base`` column.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(df)
    seniorities = df["seniority"].to_numpy()
    ratings = df["rating"].to_numpy()

    lgd_values = np.zeros(n, dtype=float)

    _HY_RATINGS = {"BB", "B", "CCC"}

    for i in range(n):
        seniority = seniorities[i]
        rating = ratings[i]

        # Base LGD from seniority
        base_lgd = SENIORITY_LGD_MAP.get(seniority, 0.45)

        # Rating adjustment: HY names have slightly higher LGD (less recovery)
        # due to smaller, less diversified balance sheets (Moody's URD evidence)
        if rating in _HY_RATINGS:
            base_lgd += 0.05

        lgd_values[i] = base_lgd

    # Add idiosyncratic noise
    lgd_values += rng.normal(0.0, _LGD_NOISE_STD, n)

    # Apply CRR3 floor and cap
    lgd_values = np.clip(lgd_values, _LGD_FLOOR_CRR3, _LGD_CAP)

    return df.with_columns(pl.Series("lgd_base", np.round(lgd_values, 4)))


# ──────────────────────────────────────────────
# RW : CRR3 ECRA (ART. 120)
# ──────────────────────────────────────────────

def compute_corporate_bonds_rw(df: pl.DataFrame) -> pl.DataFrame:
    """Compute CRR3 ECRA risk weight from rating.

    Risk weights per CRR3 Art. 120 External Credit Assessment Approach:
        AAA/AA -> 20% (CQS 1)
        A      -> 50% (CQS 2)
        BBB    -> 75% (CQS 3)
        BB     -> 100% (CQS 4)
        B/CCC  -> 150% (CQS 5-6)

    For unrated corporates (not in this portfolio), 100% would apply
    under CRR3 Art. 122.

    Args:
        df: DataFrame with corporate bond positions.

    Returns:
        DataFrame with added ``rw_crr3`` column.
    """
    rw_values = np.array([
        RATING_RW_MAP.get(r, 1.00)
        for r in df["rating"].to_numpy()
    ], dtype=float)

    return df.with_columns(pl.Series("rw_crr3", np.round(rw_values, 4)))


# ──────────────────────────────────────────────
# SPREAD : RATING + SECTOR OVERLAY
# ──────────────────────────────────────────────

def _compute_spread(df: pl.DataFrame) -> pl.DataFrame:
    """Compute credit spread (bps) from rating and sector.

    Spread = RATING_SPREAD_BPS[rating] + sector_addon

    Args:
        df: DataFrame with corporate bond positions.

    Returns:
        DataFrame with added ``spread_bps`` column.
    """
    n = len(df)
    ratings = df["rating"].to_numpy()
    sectors = df["sector_gics"].to_numpy()

    spread_values = np.zeros(n, dtype=float)
    for i in range(n):
        base_spread = RATING_SPREAD_BPS.get(ratings[i], 130.0)
        sector_addon = _SECTOR_SPREAD_ADDON_BPS.get(sectors[i], 0.0)
        spread_values[i] = max(5.0, base_spread + sector_addon)

    return df.with_columns(pl.Series("spread_bps", np.round(spread_values, 1)))


# ──────────────────────────────────────────────
# STRESS TEST (4 canaux)
# ──────────────────────────────────────────────

def stress_corporate_bonds_positions(
    df: pl.DataFrame,
    macro_params: Dict[str, float],
) -> Dict[str, float]:
    """Stress test des positions corporate bonds et re-aggregation.

    4 transmission channels:

    1. **Spread widening** (mark-to-market loss proxy):
       ``delta_spread = -50 * delta_gdp + 30 * delta_rate + sector_shock``
       Captures market repricing under stress.

    2. **Migration shock** (rating downgrade probability):
       ``migration_shock = p_downgrade * (pd_next_rating - pd_current)``
       Stressed downgrade probability scales with GDP deterioration.

    3. **Default increase** (PD stress via GDP sensitivity):
       ``pd_stressed = pd * (1 + migration_shock_factor) * exp(-gdp_sens * delta_gdp)``
       Sector-specific GDP sensitivity for cyclical PD amplification.

    4. **LGD downturn** (EBA GL/2019/03):
       ``lgd_stressed = lgd * (1 + 0.15 * max(0, -delta_gdp))``
       Procyclical LGD increase when GDP contracts.

    Args:
        df: DataFrame from generate_corporate_bonds_positions().
        macro_params: Dict with gdp_growth, interest_rate, unemployment_rate,
            inflation_rate.

    Returns:
        Dict with stressed pd_base, lgd_base, rw_crr3 (EAD-weighted).
    """
    # Deltas from base scenario
    delta_gdp = (macro_params.get("gdp_growth", _BASE_GDP) - _BASE_GDP) / 100.0
    delta_rate = (macro_params.get("interest_rate", _BASE_RATE) - _BASE_RATE) / 100.0
    delta_unemp = (macro_params.get("unemployment_rate", _BASE_UNEMP) - _BASE_UNEMP) / 100.0
    delta_inflation = (macro_params.get("inflation_rate", _BASE_INFLATION) - _BASE_INFLATION) / 100.0

    n = len(df)
    ead = df["ead"].to_numpy()
    ratings = df["rating"].to_numpy()
    sectors = df["sector_gics"].to_numpy()
    pd_base = df["pd_base"].to_numpy().copy()
    lgd_base = df["lgd_base"].to_numpy().copy()

    # ── Channel 1: Spread widening ──
    # Not directly PD, but used as auxiliary signal for migration probability
    sector_shock = np.array([
        _SECTOR_GDP_SENSITIVITY.get(s, 1.0) * (-delta_gdp) * 50.0
        for s in sectors
    ], dtype=float)

    spread_widening = -50.0 * delta_gdp + 30.0 * delta_rate + sector_shock
    spread_widening = np.maximum(spread_widening, -200.0)  # floor: max tightening

    # ── Channel 2: Migration shock ──
    # Under stress, downgrade probability increases proportional to GDP/unemp
    stress_factor = max(0.0, -delta_gdp * 10.0 + delta_unemp * 5.0)

    migration_pd_delta = np.zeros(n, dtype=float)
    for i in range(n):
        rating = ratings[i]
        base_downgrade_prob = _RATING_DOWNGRADE_PROB.get(rating, 0.10)

        # Stressed downgrade probability (baseline + GDP/unemployment amplification)
        p_downgrade = min(1.0, base_downgrade_prob * (1.0 + stress_factor))

        # PD of next lower rating
        rating_idx = _RATING_LADDER.index(rating) if rating in _RATING_LADDER else 3
        next_rating_idx = min(rating_idx + 1, len(_RATING_LADDER) - 1)
        next_rating = _RATING_LADDER[next_rating_idx]

        pd_current = RATING_PD_MAP.get(rating, 0.0025)
        pd_next = RATING_PD_MAP.get(next_rating, pd_current * 3.0)

        # Migration contribution to PD
        migration_pd_delta[i] = p_downgrade * (pd_next - pd_current)

    # ── Channel 3: Default increase (GDP-driven, sector-specific) ──
    gdp_sensitivity = np.array([
        _SECTOR_GDP_SENSITIVITY.get(s, 1.0) for s in sectors
    ], dtype=float)

    # PD stress multiplier: exp(-sensitivity * delta_gdp * 10)
    # negative delta_gdp (recession) -> multiplier > 1
    pd_stress_mult = np.exp(-gdp_sensitivity * delta_gdp * 10.0)

    # Combine: base PD * GDP multiplier + migration delta
    pd_stressed = pd_base * pd_stress_mult + migration_pd_delta

    # Inflation channel: persistent inflation erodes corporate margins
    # Mild effect: +5% PD per +1pp inflation above base
    inflation_pd_mult = 1.0 + max(0.0, delta_inflation) * 5.0
    pd_stressed = pd_stressed * inflation_pd_mult

    # Apply floors and caps
    pd_stressed = np.clip(pd_stressed, _PD_FLOOR, _PD_CAP)

    # ── Channel 4: LGD downturn (EBA GL/2019/03) ──
    # LGD increases when GDP contracts (fire-sale effect on collateral,
    # lower recovery rates in recession)
    lgd_downturn_addon = _LGD_DOWNTURN_COEFF * np.maximum(0.0, -delta_gdp * 100.0)
    lgd_stressed = lgd_base * (1.0 + lgd_downturn_addon)

    # Unemployment channel: higher unemployment -> lower recovery
    lgd_unemp_addon = max(0.0, delta_unemp * 100.0) * 0.002  # +0.2% LGD per +1pp unemp
    lgd_stressed = lgd_stressed + lgd_unemp_addon

    lgd_stressed = np.clip(lgd_stressed, _LGD_FLOOR_CRR3, _LGD_CAP)

    # ── RW stays rating-based (no stressed RW migration for now) ──
    rw = np.array([
        RATING_RW_MAP.get(r, 1.00) for r in ratings
    ], dtype=float)

    # ── Spread stress for downstream use ──
    base_spread = np.array([
        RATING_SPREAD_BPS.get(r, 130.0) for r in ratings
    ], dtype=float)
    spread_stressed = np.maximum(5.0, base_spread + spread_widening)

    # ── EAD-weighted aggregation ──
    total_ead = ead.sum()
    if total_ead <= 0:
        return {
            "pd_base": 0.012,
            "lgd_base": 0.45,
            "rw_crr3": 0.75,
        }

    w = ead / total_ead
    pd_agg = float(np.dot(w, pd_stressed))
    lgd_agg = float(np.dot(w, lgd_stressed))
    rw_agg = float(np.dot(w, rw))

    return {
        "pd_base": round(pd_agg, 6),
        "lgd_base": round(lgd_agg, 6),
        "rw_crr3": round(rw_agg, 4),
    }


# ──────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────

def aggregate_corporate_bonds_to_balance_row(
    df_positions: pl.DataFrame,
    profile: "AssetClassProfile",
) -> dict:
    """Aggregate individual corporate bond positions into a single balance sheet row.

    Returns a dict compatible with the df_balance_sheet schema.
    PD, LGD, RW, tenor are EAD-weighted averages (bottom-up).

    Args:
        df_positions: DataFrame from generate_corporate_bonds_positions().
        profile: AssetClassProfile for the corporate bonds class.

    Returns:
        Dict with all 19 balance sheet columns.
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
        pd_agg = float(np.dot(w, df_positions["pd_base"].to_numpy()))
        lgd_agg = float(np.dot(w, df_positions["lgd_base"].to_numpy()))
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
        "tenor": round(tenor_agg, 2),
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
