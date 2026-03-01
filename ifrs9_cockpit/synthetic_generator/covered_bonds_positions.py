"""Generateur position-par-position pour le portefeuille Covered Bonds.

Genere ~500 obligations securisees (covered bonds / Pfandbriefe / cedulas)
avec mecanisme de double recours (emetteur + cover pool), PD/LGD tres faibles,
et stress test position-par-position (4 canaux, ~3ms).

Deux modes :
    - Mode reel (parquet present) : echantillonne depuis le parquet ECB,
      enrichit avec cover pool genere parametriquement
    - Mode fallback (parquet absent) : generation 100% parametrique depuis
      constantes calibrees ECBC 2024

Calibration :
    - ECBC European Covered Bond Fact Book 2024 : encours EUR 3.31T,
      composition par type de pool, OC ratios, structures
    - 200+ ans sans defaut (Pfandbriefe depuis 1769, ECBC 2024)
    - Cover pool : 60% residentiel, 30% public, 7% commercial, 3% mixte
    - Moody's Covered Bond Monitor : PD emetteur, LGD couverture
    - vdp Pfandbrief Study 2024 : OC ratios DE 15-30%

References :
    - CRR3 Art. 129 : RW preferentiels (10% AAA, 15% AA, 20% A, 35% BBB)
    - UCITS 52(4) : definition legale covered bond
    - LCR Delegated Act : HQLA Level 2A (15% haircut)
    - Basel III NSFR : RSF 15% pour HQLA L2A
    - CBPP3 (ECB) : programme d'achat (eligibilite)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import polars as pl


# ──────────────────────────────────────────────
# CONSTANTES (calibrees ECBC 2024 / Moody's / CRR3)
# ──────────────────────────────────────────────

# --- 4 types de cover pool (ECBC 2024) ---
_COVER_POOL_TYPES: Dict[str, Dict] = {
    "residential_mortgage": {
        "weight": 0.60,
        "pool_pd": 0.012,
        "pool_lgd": 0.15,
        "avg_ltv": 0.62,
        "oc_typical": 0.08,
        "beta_hpi": 1.5,
        "beta_rate": 0.3,
        "beta_gdp": 0.2,
        "beta_inflation": 0.1,
    },
    "public_sector": {
        "weight": 0.30,
        "pool_pd": 0.0005,
        "pool_lgd": 0.45,
        "avg_ltv": 0.0,
        "oc_typical": 0.05,
        "beta_hpi": 0.0,
        "beta_rate": 0.5,
        "beta_gdp": 0.3,
        "beta_inflation": 0.2,
    },
    "commercial_mortgage": {
        "weight": 0.07,
        "pool_pd": 0.020,
        "pool_lgd": 0.25,
        "avg_ltv": 0.55,
        "oc_typical": 0.10,
        "beta_hpi": 2.0,
        "beta_rate": 0.5,
        "beta_gdp": 0.5,
        "beta_inflation": 0.2,
    },
    "mixed": {
        "weight": 0.03,
        "pool_pd": 0.008,
        "pool_lgd": 0.20,
        "avg_ltv": 0.50,
        "oc_typical": 0.07,
        "beta_hpi": 1.0,
        "beta_rate": 0.4,
        "beta_gdp": 0.3,
        "beta_inflation": 0.15,
    },
}

# --- 18 emetteurs fallback (pas de parquet) ---
_ISSUER_BANKS = [
    # (name, country, weight, rating)
    ("Deutsche Pfandbriefbank", "DE", 0.10, "AAA"),
    ("Muenchener Hyp", "DE", 0.09, "AAA"),
    ("Berlin Hyp", "DE", 0.09, "AAA"),
    ("DZ HYP", "DE", 0.09, "AA"),
    ("Credit Mutuel", "FR", 0.08, "AA"),
    ("BPCE", "FR", 0.07, "AA"),
    ("BNP Paribas", "FR", 0.06, "AA"),
    ("Nordea", "FI", 0.05, "AA"),
    ("Danske Bank", "DK", 0.04, "AA"),
    ("ABN AMRO", "NL", 0.04, "AA"),
    ("ING Bank", "NL", 0.04, "AA"),
    ("Caixabank", "ES", 0.04, "A"),
    ("Unicredit", "IT", 0.04, "A"),
    ("Intesa Sanpaolo", "IT", 0.03, "A"),
    ("Erste Group", "AT", 0.03, "A"),
    ("Raiffeisen", "AT", 0.03, "A"),
    ("SpareBank 1", "NO", 0.03, "AA"),
    ("Belfius", "BE", 0.05, "A"),
]

# --- CRR3 Art. 129 Risk Weights ---
_CRR3_ART129_RW: Dict[str, float] = {
    "AAA": 0.10,
    "AA": 0.15,
    "A": 0.20,
    "BBB": 0.35,
}
_CRR3_ART129_RW_DEFAULT = 1.00

# --- Rating -> PD emetteur (Moody's, annualized TTC) ---
_RATING_PD: Dict[str, float] = {
    "AAA": 0.0001,
    "AA": 0.0003,
    "A": 0.0010,
    "BBB": 0.0030,
}

# --- Pool type -> correlation factor for conditional PD ---
_POOL_CORRELATION: Dict[str, float] = {
    "residential_mortgage": 1.5,
    "public_sector": 0.5,
    "commercial_mortgage": 2.0,
    "mixed": 1.0,
}

# --- Scalaires ---
_PD_FLOOR = 0.00003
_PD_CAP = 0.05
_LGD_FLOOR = 0.05
_LGD_CAP = 0.45
_OC_EFFECTIVENESS = 0.80
_OC_MIN_CRR3 = 0.02

# --- Structure maturity ---
_STRUCTURE_WEIGHTS = {"soft_bullet": 0.90, "hard_bullet": 0.09, "cpt": 0.01}
_STRUCTURE_PD_MULT = {"soft_bullet": 1.0, "hard_bullet": 1.3, "cpt": 0.7}

# --- Parquet location ---
_PARQUET_PATH = Path(__file__).parent.parent / "data" / "covered_bonds.parquet"


# ──────────────────────────────────────────────
# CHARGEMENT DONNEES REELLES
# ──────────────────────────────────────────────

def load_covered_bond_data(path: Optional[Path] = None) -> Optional[pl.DataFrame]:
    """Charge le parquet pre-traite si present.

    Args:
        path: Chemin vers le parquet. Par defaut : data/covered_bonds.parquet.

    Returns:
        DataFrame avec donnees ECB reelles, ou None si absent.
    """
    p = path or _PARQUET_PATH
    if not p.exists():
        return None
    try:
        return pl.read_parquet(p)
    except Exception:
        return None


# ──────────────────────────────────────────────
# GENERATION (REEL + ENRICHISSEMENT OU FALLBACK)
# ──────────────────────────────────────────────

def generate_covered_bonds_positions(
    n_positions: int = 500,
    total_ead: float = 1.0e9,
    seed: int = 942,
) -> pl.DataFrame:
    """Generate individual covered bond positions.

    Two modes:
    - Real mode (parquet present): sample from ECB data, enrich with
      parametric cover pool characteristics
    - Fallback mode (parquet absent): 100% parametric from _ISSUER_BANKS

    Args:
        n_positions: Number of positions to generate.
        total_ead: Target total EAD.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with 21 columns per position.
    """
    rng = np.random.default_rng(seed)

    # Try loading real data
    df_real = load_covered_bond_data()

    if df_real is not None and len(df_real) > 0:
        df = _generate_from_real(df_real, n_positions, total_ead, rng)
    else:
        df = _generate_fallback(n_positions, total_ead, rng)

    # Compute PD, LGD, RW
    df = df.with_columns(pl.Series("pd_position", compute_covered_bonds_pd(df)))
    df = df.with_columns(pl.Series("lgd_position", compute_covered_bonds_lgd(df)))
    df = df.with_columns(pl.Series("rw_crr3", compute_covered_bonds_rw(df)))

    # Default flag (PD-based, very rare for covered bonds)
    df = df.with_columns(pl.Series("default_flag", rng.binomial(1, np.minimum(df["pd_position"].to_numpy(), 0.01))))

    return df


def _generate_from_real(
    df_real: pl.DataFrame,
    n_positions: int,
    total_ead: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Generate from real ECB parquet data with parametric enrichment."""
    # Sample with replacement if needed
    replace = n_positions > len(df_real)
    idx = rng.choice(len(df_real), size=n_positions, replace=replace)
    df = df_real[idx.tolist()]

    # Extract columns to numpy arrays BEFORE the loop
    col_names = df.columns
    bond_id_arr = df["bond_id"].to_numpy() if "bond_id" in col_names else None
    issuer_name_arr = df["issuer_name"].to_numpy() if "issuer_name" in col_names else None
    issuer_country_arr = df["issuer_country"].to_numpy() if "issuer_country" in col_names else None
    implied_rating_arr = df["implied_rating"].to_numpy() if "implied_rating" in col_names else None
    coupon_type_arr = df["coupon_type"].to_numpy() if "coupon_type" in col_names else None
    coupon_rate_arr = df["coupon_rate_pct"].to_numpy() if "coupon_rate_pct" in col_names else None
    tenor_arr = df["tenor_years"].to_numpy() if "tenor_years" in col_names else None
    denomination_arr = df["denomination"].to_numpy() if "denomination" in col_names else None
    issuance_date_arr = df["issuance_date"].to_numpy() if "issuance_date" in col_names else None

    # Map real columns to position schema
    records = []
    pool_names = list(_COVER_POOL_TYPES.keys())
    pool_weights = np.array([_COVER_POOL_TYPES[k]["weight"] for k in pool_names])
    pool_weights = pool_weights / pool_weights.sum()

    for i in range(n_positions):
        # Issuer info from real data
        bond_id = str(bond_id_arr[i]) if bond_id_arr is not None else f"CB_{i:06d}"
        issuer_name = str(issuer_name_arr[i]) if issuer_name_arr is not None else "Unknown"
        issuer_country = str(issuer_country_arr[i]) if issuer_country_arr is not None else "DE"
        issuer_rating = str(implied_rating_arr[i]) if implied_rating_arr is not None else "AA"
        if issuer_rating not in _RATING_PD:
            issuer_rating = "AA"

        # Cover pool (enriched parametrically)
        pool_type = rng.choice(pool_names, p=pool_weights)
        pool_cfg = _COVER_POOL_TYPES[pool_type]

        ltv = _generate_ltv(pool_type, pool_cfg, rng)
        oc = float(np.clip(rng.normal(pool_cfg["oc_typical"], 0.02), _OC_MIN_CRR3, 0.30))

        # Structure
        structure = rng.choice(
            list(_STRUCTURE_WEIGHTS.keys()),
            p=list(_STRUCTURE_WEIGHTS.values()),
        )

        # Coupon from real data
        coupon_type = str(coupon_type_arr[i]) if coupon_type_arr is not None else "fixed"
        coupon_rate_bps = float(coupon_rate_arr[i]) * 100 if coupon_rate_arr is not None else 250.0

        # Tenor from real data
        tenor = float(tenor_arr[i]) if tenor_arr is not None else 5.5
        tenor = float(np.clip(tenor, 1.0, 15.0))

        currency = str(denomination_arr[i]) if denomination_arr is not None else "EUR"

        # Issue vintage
        issuance = issuance_date_arr[i] if issuance_date_arr is not None else None
        if issuance is not None:
            try:
                import datetime as _dt
                if hasattr(issuance, 'year'):
                    vintage = int(issuance.year)
                elif isinstance(issuance, (np.datetime64,)):
                    vintage = int(np.datetime64(issuance, 'Y').astype(int) + 1970)
                else:
                    vintage = int(rng.integers(2018, 2026))
            except Exception:
                vintage = int(rng.integers(2018, 2026))
        else:
            vintage = int(rng.integers(2018, 2026))

        records.append({
            "bond_id": bond_id,
            "issuer_name": issuer_name,
            "issuer_country": issuer_country,
            "issuer_rating": issuer_rating,
            "pd_issuer": _RATING_PD[issuer_rating],
            "cover_pool_type": pool_type,
            "pool_weighted_ltv": ltv,
            "pool_oc_ratio": oc,
            "maturity_structure": structure,
            "coupon_type": coupon_type,
            "coupon_rate_bps": coupon_rate_bps,
            "tenor_years": tenor,
            "currency": currency,
            "notional_ead": 0.0,  # filled after
            "ead": 0.0,
            "issue_vintage": vintage,
        })

    df_out = pl.DataFrame(records)

    # Scale EAD
    raw_ead = rng.lognormal(mean=18.0, sigma=0.8, size=n_positions)
    raw_total = raw_ead.sum()
    scale = total_ead / raw_total if raw_total > 0 else 1.0
    df_out = df_out.with_columns(pl.Series("notional_ead", np.round(raw_ead * scale, 2)))
    df_out = df_out.with_columns(pl.col("notional_ead").alias("ead"))

    return df_out


def _generate_fallback(
    n_positions: int,
    total_ead: float,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Generate 100% parametric from constants."""
    # Issuer distribution
    issuer_names = [b[0] for b in _ISSUER_BANKS]
    issuer_countries = [b[1] for b in _ISSUER_BANKS]
    issuer_weights = np.array([b[2] for b in _ISSUER_BANKS])
    issuer_ratings = [b[3] for b in _ISSUER_BANKS]
    issuer_weights = issuer_weights / issuer_weights.sum()

    issuer_idx = rng.choice(len(_ISSUER_BANKS), size=n_positions, p=issuer_weights)

    # Pool type distribution
    pool_names = list(_COVER_POOL_TYPES.keys())
    pool_weights = np.array([_COVER_POOL_TYPES[k]["weight"] for k in pool_names])
    pool_weights = pool_weights / pool_weights.sum()
    pool_types = rng.choice(pool_names, size=n_positions, p=pool_weights)

    records = []
    for i in range(n_positions):
        idx = issuer_idx[i]
        pool_type = pool_types[i]
        pool_cfg = _COVER_POOL_TYPES[pool_type]
        rating = issuer_ratings[idx]

        ltv = _generate_ltv(pool_type, pool_cfg, rng)
        oc = float(np.clip(rng.normal(pool_cfg["oc_typical"], 0.02), _OC_MIN_CRR3, 0.30))

        structure = rng.choice(
            list(_STRUCTURE_WEIGHTS.keys()),
            p=list(_STRUCTURE_WEIGHTS.values()),
        )

        # Coupon
        is_fixed = rng.random() < 0.65
        coupon_type = "fixed" if is_fixed else "floating"
        if is_fixed:
            coupon_rate_bps = float(np.clip(rng.normal(250, 50), 50, 600))
        else:
            coupon_rate_bps = float(np.clip(rng.normal(100, 30), 10, 300))

        # Tenor
        tenor = float(np.clip(rng.normal(5.5, 1.5), 1.0, 15.0))

        # Currency
        currency = "EUR" if rng.random() < 0.92 else rng.choice(["GBP", "SEK", "NOK", "DKK"])

        # Vintage
        vintage = int(rng.integers(2018, 2026))

        records.append({
            "bond_id": f"CB_{i:06d}",
            "issuer_name": issuer_names[idx],
            "issuer_country": issuer_countries[idx],
            "issuer_rating": rating,
            "pd_issuer": _RATING_PD[rating],
            "cover_pool_type": pool_type,
            "pool_weighted_ltv": ltv,
            "pool_oc_ratio": oc,
            "maturity_structure": structure,
            "coupon_type": coupon_type,
            "coupon_rate_bps": coupon_rate_bps,
            "tenor_years": tenor,
            "currency": currency,
            "notional_ead": 0.0,
            "ead": 0.0,
            "issue_vintage": vintage,
        })

    df = pl.DataFrame(records)

    # Scale EAD (lognormal then scale)
    raw_ead = rng.lognormal(mean=18.0, sigma=0.8, size=n_positions)
    raw_total = raw_ead.sum()
    scale = total_ead / raw_total if raw_total > 0 else 1.0
    df = df.with_columns(pl.Series("notional_ead", np.round(raw_ead * scale, 2)))
    df = df.with_columns(pl.col("notional_ead").alias("ead"))

    return df


def _generate_ltv(pool_type: str, pool_cfg: Dict, rng: np.random.Generator) -> float:
    """Generate LTV for a cover pool type."""
    if pool_type == "public_sector":
        return 0.0  # No physical collateral
    avg_ltv = pool_cfg["avg_ltv"]
    return float(np.clip(rng.normal(avg_ltv, 0.08), 0.0, 0.85))


# ──────────────────────────────────────────────
# PD : DOUBLE-DEFAUT + ABSORPTION OC
# ──────────────────────────────────────────────

def compute_covered_bonds_pd(df: pl.DataFrame) -> np.ndarray:
    """Compute position-level PD via double-default + OC absorption model.

    PD_cb = PD_issuer x alpha_dd x (1 + pool_factor) x alpha_structure

    Where:
    - PD_issuer: from rating (AAA=0.01%, AA=0.03%, A=0.10%, BBB=0.30%)
    - alpha_dd = 0.50 : double-default discount (200+ yrs history, ECBC)
    - pool_factor = PD_pool_cond x oc_factor x 8
      - PD_pool_cond = pool_pd x alpha_correlation
      - oc_factor = max(0.20, 1 - OC x effectiveness x 3)
    - alpha_structure: soft_bullet=1.0, hard_bullet=1.3, cpt=0.7

    Target: portfolio 1-5 bps EAD-weighted.

    Args:
        df: DataFrame with covered bond positions.

    Returns:
        Array of PD values per position.
    """
    # PD issuer
    pd_issuer = np.array([
        _RATING_PD.get(r, 0.0010) for r in df["issuer_rating"].to_numpy()
    ], dtype=float)

    # Pool conditional PD
    pool_types = df["cover_pool_type"].to_numpy()
    pool_pd_base = np.array([
        _COVER_POOL_TYPES[pt]["pool_pd"] for pt in pool_types
    ], dtype=float)

    alpha_corr = np.array([
        _POOL_CORRELATION.get(pt, 1.0) for pt in pool_types
    ], dtype=float)

    pd_pool_cond = pool_pd_base * alpha_corr

    # OC absorption factor: higher OC -> lower pool contribution
    oc = df["pool_oc_ratio"].to_numpy()
    oc_factor = np.maximum(0.20, 1.0 - oc * _OC_EFFECTIVENESS * 3.0)

    # Pool contribution to PD
    pool_factor = pd_pool_cond * oc_factor * 8.0

    # Structure multiplier (refinancing risk)
    structures = df["maturity_structure"].to_numpy()
    alpha_struct = np.array([
        _STRUCTURE_PD_MULT.get(s, 1.0) for s in structures
    ], dtype=float)

    # Double-default formula: issuer PD scaled by dual recourse
    alpha_dd = 0.50  # historical discount (200+ yrs no default, ECBC 2024)
    pd_cb = pd_issuer * alpha_dd * (1.0 + pool_factor) * alpha_struct

    return np.clip(pd_cb, _PD_FLOOR, _PD_CAP)


# ──────────────────────────────────────────────
# LGD : WATERFALL DUAL RECOURS
# ──────────────────────────────────────────────

def compute_covered_bonds_lgd(df: pl.DataFrame) -> np.ndarray:
    """Compute position-level LGD via dual recourse waterfall.

    RR_pool = (1 - pool_lgd) x (1 - haircut_ltv)
        haircut_ltv = max(0, ltv - 0.60) x 0.5  (residential)
        haircut_ltv = 0.0  (public sector)
    RR_oc = OC x 0.80
    RR_issuer = 0.10 x (1 - pd_issuer)
    LGD = 1 - min(1.0, RR_pool + RR_oc + RR_issuer)

    Clip [0.05, 0.45]. Target portfolio: 5-10%.

    Args:
        df: DataFrame with covered bond positions.

    Returns:
        Array of LGD values per position.
    """
    pool_types = df["cover_pool_type"].to_numpy()
    ltv = df["pool_weighted_ltv"].to_numpy()
    oc = df["pool_oc_ratio"].to_numpy()

    # Pool LGD by type
    pool_lgd = np.array([
        _COVER_POOL_TYPES[pt]["pool_lgd"] for pt in pool_types
    ], dtype=float)

    # LTV haircut (only for mortgage pools)
    haircut_ltv = np.zeros(len(df), dtype=float)
    for i, pt in enumerate(pool_types):
        if pt in ("residential_mortgage", "commercial_mortgage", "mixed"):
            haircut_ltv[i] = max(0.0, ltv[i] - 0.60) * 0.5

    # Recovery rate from pool
    rr_pool = (1.0 - pool_lgd) * (1.0 - haircut_ltv)

    # Recovery rate from OC
    rr_oc = oc * _OC_EFFECTIVENESS

    # Recovery rate from issuer (unsecured senior claim)
    pd_issuer = np.array([
        _RATING_PD.get(r, 0.0010) for r in df["issuer_rating"].to_numpy()
    ], dtype=float)
    rr_issuer = 0.10 * (1.0 - pd_issuer)

    # Total recovery
    total_rr = np.minimum(1.0, rr_pool + rr_oc + rr_issuer)

    lgd = 1.0 - total_rr
    return np.clip(lgd, _LGD_FLOOR, _LGD_CAP).round(4)


# ──────────────────────────────────────────────
# RW : CRR3 ART. 129
# ──────────────────────────────────────────────

def compute_covered_bonds_rw(df: pl.DataFrame) -> np.ndarray:
    """Compute CRR3 Art. 129 risk weight from issuer rating.

    AAA → 10%, AA → 15%, A → 20%, BBB → 35%, other → 100%.

    Args:
        df: DataFrame with covered bond positions.

    Returns:
        Array of RW values per position.
    """
    rw = np.array([
        _CRR3_ART129_RW.get(r, _CRR3_ART129_RW_DEFAULT)
        for r in df["issuer_rating"].to_numpy()
    ], dtype=float)
    return rw


# ──────────────────────────────────────────────
# STRESS TEST POSITION-PAR-POSITION
# ──────────────────────────────────────────────

def stress_covered_bonds_positions(
    df: pl.DataFrame,
    macro_params: Dict[str, float],
) -> Dict[str, float]:
    """Stress covered bonds positions under macro scenario and re-aggregate.

    4 transmission channels:
    1. HPI -> cover pool: pool_pd_stressed, ltv_stressed (mortgage pools only)
    2. Rate -> refinancing: pd_issuer_stressed
    3. GDP/unemployment -> issuer: factor on pd_issuer
    4. Inflation -> erosion: factor on pd_issuer

    LGD downturn for mortgage pools: HPI and GDP addons.

    Args:
        df: DataFrame from generate_covered_bonds_positions().
        macro_params: Dict with keys gdp_growth, unemployment_rate,
            interest_rate, hpi_growth, inflation_rate.

    Returns:
        Dict with stressed pd_base, lgd_base, rw_crr3 (EAD-weighted).
    """
    # Scenario base values
    base_gdp = 1.2
    base_unemp = 7.5
    base_rate = 3.5
    base_hpi = 2.0
    base_inflation = 2.5

    # Deltas (percentage points -> fraction for exp transforms)
    delta_gdp = (macro_params.get("gdp_growth", base_gdp) - base_gdp) / 100.0
    delta_unemp = (macro_params.get("unemployment_rate", base_unemp) - base_unemp) / 100.0
    delta_rate = (macro_params.get("interest_rate", base_rate) - base_rate) / 100.0
    delta_hpi = (macro_params.get("hpi_growth", base_hpi) - base_hpi) / 100.0
    delta_inflation = (macro_params.get("inflation_rate", base_inflation) - base_inflation) / 100.0

    n = len(df)
    pool_types = df["cover_pool_type"].to_numpy()
    ead = df["ead"].to_numpy()

    # Base values
    pd_issuer_base = df["pd_issuer"].to_numpy().copy()
    pool_pd_base = np.array([
        _COVER_POOL_TYPES[pt]["pool_pd"] for pt in pool_types
    ], dtype=float)
    ltv_base = df["pool_weighted_ltv"].to_numpy().copy()
    oc_base = df["pool_oc_ratio"].to_numpy().copy()

    # --- Channel 1: HPI -> cover pool (mortgage pools) ---
    betas_hpi = np.array([
        _COVER_POOL_TYPES[pt]["beta_hpi"] for pt in pool_types
    ], dtype=float)

    pool_pd_stressed = pool_pd_base * np.exp(-delta_hpi * betas_hpi * 0.5)
    ltv_stressed = ltv_base * np.exp(-delta_hpi * betas_hpi)
    # Public sector: no HPI effect on LTV (already 0)
    for i in range(n):
        if pool_types[i] == "public_sector":
            ltv_stressed[i] = 0.0

    pool_pd_stressed = np.clip(pool_pd_stressed, 0.0001, 0.10)
    ltv_stressed = np.clip(ltv_stressed, 0.0, 0.85)

    # --- Channel 2: Rate -> refinancing risk ---
    betas_rate = np.array([
        _COVER_POOL_TYPES[pt]["beta_rate"] for pt in pool_types
    ], dtype=float)
    pd_issuer_stressed = pd_issuer_base * np.exp(delta_rate * betas_rate)

    # --- Channel 2b: HPI -> issuer solvency ---
    # Covered bond issuers are typically mortgage banks — HPI decline
    # weakens their solvency (doom loop: cover pool deterioration + issuer stress)
    factor_hpi_issuer = np.exp(-delta_hpi * betas_hpi * 1.5)
    pd_issuer_stressed = pd_issuer_stressed * factor_hpi_issuer

    # --- Channel 3: GDP/unemployment -> issuer ---
    betas_gdp = np.array([
        _COVER_POOL_TYPES[pt]["beta_gdp"] for pt in pool_types
    ], dtype=float)
    factor_gdp = np.exp(-delta_gdp * betas_gdp + delta_unemp * 0.5)

    # --- Channel 4: Inflation -> erosion ---
    betas_infl = np.array([
        _COVER_POOL_TYPES[pt]["beta_inflation"] for pt in pool_types
    ], dtype=float)
    factor_infl = np.exp(delta_inflation * betas_infl)

    pd_issuer_stressed = pd_issuer_stressed * factor_gdp * factor_infl
    pd_issuer_stressed = np.clip(pd_issuer_stressed, 0.00001, 0.05)

    # --- Recalculate PD with stressed values (same formula as compute_covered_bonds_pd) ---
    alpha_corr = np.array([
        _POOL_CORRELATION.get(pt, 1.0) for pt in pool_types
    ], dtype=float)
    pd_pool_cond_stressed = pool_pd_stressed * alpha_corr

    oc_factor = np.maximum(0.20, 1.0 - oc_base * _OC_EFFECTIVENESS * 3.0)
    pool_factor = pd_pool_cond_stressed * oc_factor * 8.0

    structures = df["maturity_structure"].to_numpy()
    alpha_struct = np.array([
        _STRUCTURE_PD_MULT.get(s, 1.0) for s in structures
    ], dtype=float)

    alpha_dd = 0.50
    pd_stressed = pd_issuer_stressed * alpha_dd * (1.0 + pool_factor) * alpha_struct
    pd_stressed = np.clip(pd_stressed, _PD_FLOOR, _PD_CAP)

    # --- LGD downturn (mortgage pools) ---
    # Build stressed LGD via waterfall with stressed LTV
    pool_lgd = np.array([
        _COVER_POOL_TYPES[pt]["pool_lgd"] for pt in pool_types
    ], dtype=float)

    haircut_ltv_stressed = np.zeros(n, dtype=float)
    for i, pt in enumerate(pool_types):
        if pt in ("residential_mortgage", "commercial_mortgage", "mixed"):
            haircut_ltv_stressed[i] = max(0.0, ltv_stressed[i] - 0.60) * 0.5

    rr_pool = (1.0 - pool_lgd) * (1.0 - haircut_ltv_stressed)
    rr_oc = oc_base * _OC_EFFECTIVENESS
    rr_issuer = 0.10 * (1.0 - pd_issuer_stressed)
    total_rr = np.minimum(1.0, rr_pool + rr_oc + rr_issuer)
    lgd_stressed = np.clip(1.0 - total_rr, _LGD_FLOOR, _LGD_CAP)

    # GDP downturn addon for mortgage pools
    is_mortgage = np.array([
        pt in ("residential_mortgage", "commercial_mortgage", "mixed")
        for pt in pool_types
    ])
    lgd_addon_gdp = np.clip(-delta_gdp * 0.005, 0, 0.08)
    lgd_addon_hpi = np.zeros(n, dtype=float)
    lgd_addon_hpi[is_mortgage] = np.clip(-delta_hpi * 0.015, 0, 0.05)
    lgd_stressed = np.clip(lgd_stressed + lgd_addon_gdp + lgd_addon_hpi, _LGD_FLOOR, _LGD_CAP)

    # RW stays based on rating (not stressed)
    rw = np.array([
        _CRR3_ART129_RW.get(r, _CRR3_ART129_RW_DEFAULT)
        for r in df["issuer_rating"].to_numpy()
    ], dtype=float)

    # --- EAD-weighted aggregation ---
    total_ead = ead.sum()
    if total_ead <= 0:
        return {"pd_base": 0.0003, "lgd_base": 0.10, "rw_crr3": 0.15}

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

def aggregate_covered_bonds_to_balance_row(
    df_positions: pl.DataFrame,
    profile: "AssetClassProfile",
) -> dict:
    """Aggregate individual covered bond positions into a single balance sheet row.

    Returns a dict compatible with the df_balance_sheet schema.
    PD, LGD, RW, tenor are EAD-weighted averages (bottom-up).

    Args:
        df_positions: DataFrame from generate_covered_bonds_positions().
        profile: AssetClassProfile for covered_bonds.

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
