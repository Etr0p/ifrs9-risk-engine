"""Generateur position-par-position pour le portefeuille Obligations Souveraines.

Genere ~180 obligations souveraines individuelles (9 emetteurs zone euro)
avec PD composite rating+fiscal+marche, LGD Cruces-Trebesch (2013 AER),
et stress test 3 canaux avec amplification peripherique (BTP vs Bund).

Le pipeline aval (ECL Vasicek, comparator RAROC, BL-CVaR) consomme alors
des valeurs calculees bottom-up plutot que des constantes parametriques.

Donnees reelles (priorite) :
    - Damodaran (NYU Stern) : CDS spreads, ratings pour 143+ pays
    - ECB SDW : courbes de taux AAA/all-govt, 7 maturites
    - Fallback : constantes SOVEREIGN_ISSUERS si parquets absents

Calibration :
    - Cruces & Trebesch (2013 AER) : 187 restructurations, mean haircut 37%
    - Moody's Sovereign Default Study (2003-2023)
    - BIS Working Paper 2014 : flight-to-quality correlations
    - ECB Financial Stability Review 2012 : doom loop
    - EBA Stress Test 2023 : sovereign shock calibration

References :
    - CRR3 Art. 114(4) : RW = 0% dette souveraine zone euro en devise domestique
    - IFRS 9 B5.5.25 : souverain AAA/AA exempt de staging (Stage 1 only)
    - Basel III LCR : HQLA Level 1 (0% haircut)
    - Basel III NSFR : RSF = 0% pour HQLA Level 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import polars as pl


# ──────────────────────────────────────────────
# CONSTANTES FALLBACK (utilisees si parquets absents)
# ──────────────────────────────────────────────

# Mapping rating → PD TTC (Moody's Sovereign Default Study)
RATING_PD_MAP: Dict[str, float] = {
    "AAA": 0.0001,
    "AA+": 0.0002,
    "AA":  0.0003,
    "AA-": 0.0004,
    "A+":  0.0006,
    "A":   0.0008,
    "A-":  0.0012,
    "BBB+": 0.0015,
    "BBB":  0.0020,
    "BBB-": 0.0035,
    "BB+":  0.0060,
    "BB":   0.0100,
}

# 9 emetteurs zone euro (fallback si Damodaran parquet absent)
SOVEREIGN_ISSUERS: Dict[str, Dict] = {
    "DE": {
        "country": "Germany", "label": "Bund", "weight": 0.25,
        "rating": "AAA", "pd_base": 0.0001, "debt_to_gdp": 0.64,
        "cds_spread_bp": 8.0, "fiscal_balance": -0.015,
        "recovery_rate_hist": 0.95, "spread_bp": 0.0,
    },
    "FR": {
        "country": "France", "label": "OAT", "weight": 0.20,
        "rating": "AA", "pd_base": 0.0003, "debt_to_gdp": 1.12,
        "cds_spread_bp": 28.0, "fiscal_balance": -0.055,
        "recovery_rate_hist": 0.90, "spread_bp": 20.0,
    },
    "IT": {
        "country": "Italy", "label": "BTP", "weight": 0.15,
        "rating": "BBB", "pd_base": 0.0020, "debt_to_gdp": 1.37,
        "cds_spread_bp": 110.0, "fiscal_balance": -0.045,
        "recovery_rate_hist": 0.65, "spread_bp": 100.0,
    },
    "ES": {
        "country": "Spain", "label": "Bonos", "weight": 0.12,
        "rating": "A", "pd_base": 0.0008, "debt_to_gdp": 1.07,
        "cds_spread_bp": 48.0, "fiscal_balance": -0.035,
        "recovery_rate_hist": 0.75, "spread_bp": 40.0,
    },
    "BE": {
        "country": "Belgium", "label": "OLO", "weight": 0.08,
        "rating": "AA", "pd_base": 0.0003, "debt_to_gdp": 1.05,
        "cds_spread_bp": 22.0, "fiscal_balance": -0.042,
        "recovery_rate_hist": 0.88, "spread_bp": 15.0,
    },
    "NL": {
        "country": "Netherlands", "label": "DSL", "weight": 0.08,
        "rating": "AAA", "pd_base": 0.0001, "debt_to_gdp": 0.47,
        "cds_spread_bp": 10.0, "fiscal_balance": 0.005,
        "recovery_rate_hist": 0.95, "spread_bp": 2.0,
    },
    "AT": {
        "country": "Austria", "label": "RAGB", "weight": 0.05,
        "rating": "AA+", "pd_base": 0.0002, "debt_to_gdp": 0.77,
        "cds_spread_bp": 18.0, "fiscal_balance": -0.030,
        "recovery_rate_hist": 0.90, "spread_bp": 10.0,
    },
    "PT": {
        "country": "Portugal", "label": "OT", "weight": 0.04,
        "rating": "BBB+", "pd_base": 0.0015, "debt_to_gdp": 0.99,
        "cds_spread_bp": 52.0, "fiscal_balance": -0.010,
        "recovery_rate_hist": 0.70, "spread_bp": 45.0,
    },
    "IE": {
        "country": "Ireland", "label": "NTMA", "weight": 0.03,
        "rating": "A+", "pd_base": 0.0006, "debt_to_gdp": 0.44,
        "cds_spread_bp": 25.0, "fiscal_balance": 0.015,
        "recovery_rate_hist": 0.80, "spread_bp": 18.0,
    },
}

# Maturity buckets
MATURITY_BUCKETS: Dict[str, Dict] = {
    "short":      {"weight": 0.25, "tenor_min": 0.5, "tenor_max": 3.0},
    "medium":     {"weight": 0.40, "tenor_min": 3.0, "tenor_max": 7.0},
    "long":       {"weight": 0.25, "tenor_min": 7.0, "tenor_max": 15.0},
    "ultra_long": {"weight": 0.10, "tenor_min": 15.0, "tenor_max": 30.0},
}

# Coupon types
COUPON_TYPES: Dict[str, float] = {
    "fixed": 0.80,
    "inflation_linked": 0.15,
    "floating": 0.05,
}

# Stress amplification multipliers (BIS WP 2014, ECB FSR 2012)
# Flight-to-quality: core compresses, periphery widens
SPREAD_AMPLIFICATION: Dict[str, float] = {
    "DE": 0.3,   # flight to quality — spread compresses
    "NL": 0.3,
    "AT": 0.8,
    "FR": 1.0,   # core
    "BE": 1.0,
    "IE": 1.5,   # semi-periphery
    "ES": 1.5,
    "IT": 2.0,   # periphery — amplification cle (doom loop)
    "PT": 2.0,
}

# Modified duration approximation factor
_MOD_DURATION_FACTOR = 0.92  # empirical (convexity adjustment)

# LGD parameters (Cruces & Trebesch 2013 AER)
_LGD_IG_BASE = 0.35       # investment grade base
_LGD_HY_BASE = 0.50       # high yield base
_LGD_MATURITY_ADJ = -0.10  # short-dated recovery premium (tenor < 5y)
_LGD_FLOOR = 0.10
_LGD_CAP = 0.65           # PSI grec 53.5% + marge
_LGD_NOISE_STD = 0.03

# PD parameters
_PD_FLOOR = 0.0001
_PD_CAP = 0.05
_MAASTRICHT_THRESHOLD = 0.60  # Maastricht debt/GDP reference

# Stress parameters
_BASE_GDP = 1.2
_BASE_RATE = 3.5
_LGD_DOWNTURN_COEFF = 0.01   # +1pp LGD per -1% GDP (lighter for sovereign)
_LGD_DOWNTURN_CAP = 0.08     # max +8pp


# ──────────────────────────────────────────────
# DATA LOADING (real data priority)
# ──────────────────────────────────────────────

def load_sovereign_risk_data(path: Optional[str] = None) -> pl.DataFrame:
    """Charge les donnees risque souverain Damodaran.

    Args:
        path: Chemin vers le parquet. Si None, utilise le defaut.

    Returns:
        DataFrame avec country_code, moody_rating, cds_spread_bp, etc.

    Raises:
        FileNotFoundError: Si le parquet n'existe pas.
    """
    if path is None:
        path = str(Path(__file__).parent.parent / "data" / "damodaran_sovereign.parquet")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Damodaran sovereign parquet not found: {p}")
    return pl.read_parquet(p)


def load_yield_curves(path: Optional[str] = None) -> Dict[str, float]:
    """Charge les courbes de taux ECB.

    Args:
        path: Chemin vers le parquet. Si None, utilise le defaut.

    Returns:
        Dict {tenor_years: yield_decimal} pour interpolation.

    Raises:
        FileNotFoundError: Si le parquet n'existe pas.
    """
    if path is None:
        path = str(Path(__file__).parent.parent / "data" / "ecb_sovereign_yields.parquet")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ECB sovereign yields parquet not found: {p}")
    df = pl.read_parquet(p)
    # Use all-govt curve as base (more representative than AAA-only)
    yield_col = "yield_all_govt" if "yield_all_govt" in df.columns else "yield_aaa"
    result = {}
    for row in df.iter_rows(named=True):
        tenor = float(row["tenor_years"])
        y = float(row[yield_col])
        if not np.isnan(y):
            result[tenor] = y
    return result


def _interpolate_yield(yield_curve: Dict[str, float], tenor: float) -> float:
    """Linear interpolation of yield for a given tenor."""
    tenors = sorted(yield_curve.keys())
    yields = [yield_curve[t] for t in tenors]
    if tenor <= tenors[0]:
        return yields[0]
    if tenor >= tenors[-1]:
        return yields[-1]
    # Find bracketing tenors
    for i in range(len(tenors) - 1):
        if tenors[i] <= tenor <= tenors[i + 1]:
            alpha = (tenor - tenors[i]) / (tenors[i + 1] - tenors[i])
            return yields[i] + alpha * (yields[i + 1] - yields[i])
    return yields[-1]


def _build_fallback_issuer_data() -> Dict[str, Dict]:
    """Return fallback issuer constants when Damodaran parquet is absent."""
    return SOVEREIGN_ISSUERS.copy()


# ──────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────

def generate_sovereign_positions(
    n_positions: int = 180,
    total_ead: float = 1e10,
    seed: int = 942,
) -> pl.DataFrame:
    """Genere un portefeuille d'obligations souveraines position par position.

    Etape A: tente de charger les donnees reelles Damodaran (CDS, ratings).
             Si absent, fallback aux constantes SOVEREIGN_ISSUERS.
    Etape B: tente de charger les courbes ECB (yields par maturite).
             Si absent, fallback courbe plate (base_rate + spread).

    Args:
        n_positions: Nombre d'obligations a generer (~180).
        total_ead: EAD total du portefeuille souverain.
        seed: Graine aleatoire.

    Returns:
        DataFrame avec ~25 colonnes (une ligne par obligation).
    """
    rng = np.random.default_rng(seed)

    # ── Etape A: Load issuer data (real or fallback) ──
    issuer_data = _build_fallback_issuer_data()
    try:
        df_risk = load_sovereign_risk_data()
        # Override fallback with real Damodaran data
        for row in df_risk.iter_rows(named=True):
            cc = row.get("country_code", "")
            if cc in issuer_data:
                if row.get("cds_spread_bp") is not None:
                    issuer_data[cc]["cds_spread_bp"] = float(row["cds_spread_bp"])
                if row.get("default_spread_bp") is not None:
                    issuer_data[cc]["spread_bp"] = float(row["default_spread_bp"])
                if row.get("moody_rating") is not None:
                    issuer_data[cc]["rating"] = str(row["moody_rating"]).strip()
                if row.get("debt_to_gdp") is not None:
                    issuer_data[cc]["debt_to_gdp"] = float(row["debt_to_gdp"])
                if row.get("fiscal_balance") is not None:
                    issuer_data[cc]["fiscal_balance"] = float(row["fiscal_balance"])
                if row.get("recovery_rate_hist") is not None:
                    issuer_data[cc]["recovery_rate_hist"] = float(row["recovery_rate_hist"])
    except FileNotFoundError:
        pass  # Use fallback constants

    # ── Etape B: Load yield curves (real or fallback) ──
    yield_curve = None
    try:
        yield_curve = load_yield_curves()
        if len(yield_curve) < 3:
            yield_curve = None  # insufficient data
    except FileNotFoundError:
        pass  # Fallback: flat curve

    # ── Sample issuers ──
    issuer_codes = list(issuer_data.keys())
    issuer_weights = np.array([issuer_data[cc]["weight"] for cc in issuer_codes])
    issuer_weights = issuer_weights / issuer_weights.sum()
    sampled_codes = rng.choice(issuer_codes, size=n_positions, p=issuer_weights)

    # ── Sample maturity buckets ──
    bucket_names = list(MATURITY_BUCKETS.keys())
    bucket_weights = np.array([MATURITY_BUCKETS[b]["weight"] for b in bucket_names])
    bucket_weights = bucket_weights / bucket_weights.sum()
    sampled_buckets = rng.choice(bucket_names, size=n_positions, p=bucket_weights)

    # ── Sample coupon types ──
    coupon_names = list(COUPON_TYPES.keys())
    coupon_weights = np.array([COUPON_TYPES[c] for c in coupon_names])
    coupon_weights = coupon_weights / coupon_weights.sum()
    sampled_coupons = rng.choice(coupon_names, size=n_positions, p=coupon_weights)

    # ── Generate individual positions ──
    records = []
    for i in range(n_positions):
        cc = sampled_codes[i]
        iss = issuer_data[cc]
        bucket = MATURITY_BUCKETS[sampled_buckets[i]]

        # Tenor uniform in bucket range
        tenor = rng.uniform(bucket["tenor_min"], bucket["tenor_max"])

        # Coupon rate
        coupon_type = sampled_coupons[i]
        base_rate = 0.035  # default flat rate

        # Country spread
        country_spread = iss.get("spread_bp", 0.0) / 10000.0

        if yield_curve is not None:
            base_yield = _interpolate_yield(yield_curve, tenor)
            coupon_rate = base_yield + country_spread
        else:
            coupon_rate = base_rate + country_spread

        # Inflation-linked bonds have lower nominal coupon
        if coupon_type == "inflation_linked":
            coupon_rate = max(0.001, coupon_rate - 0.015)  # real yield ~1.5% lower
        elif coupon_type == "floating":
            coupon_rate = base_rate + country_spread * 0.5  # floating resets

        coupon_rate = max(0.001, coupon_rate)

        # Notional (lognormal, ~50M median)
        notional = rng.lognormal(np.log(50e6), 0.6)

        # Modified duration approximation
        if coupon_rate > 0.001:
            mod_duration = tenor * _MOD_DURATION_FACTOR * (1.0 - coupon_rate / max(coupon_rate + 0.01, 0.02))
        else:
            mod_duration = tenor * _MOD_DURATION_FACTOR * 0.95
        mod_duration = max(0.1, min(mod_duration, 28.0))

        records.append({
            "position_id": i,
            "country_code": cc,
            "country": iss["country"],
            "bond_label": iss["label"],
            "rating": iss["rating"],
            "tenor_years": round(tenor, 2),
            "maturity_bucket": sampled_buckets[i],
            "coupon_type": coupon_type,
            "coupon_rate": round(coupon_rate, 6),
            "notional": round(notional, 2),
            "mod_duration": round(mod_duration, 2),
            "debt_to_gdp": iss["debt_to_gdp"],
            "cds_spread_bp": iss["cds_spread_bp"],
            "fiscal_balance": iss.get("fiscal_balance", -0.03),
            "recovery_rate_hist": iss.get("recovery_rate_hist", 0.80),
            "spread_amplification": SPREAD_AMPLIFICATION.get(cc, 1.0),
            "rw_crr3": 0.0,  # CRR3 Art. 114(4) : toujours 0%
            "hqla_eligible": True,
            "hqla_level": 1,
            "exempt_from_staging": True,  # IFRS 9 B5.5.25
        })

    df = pl.DataFrame(records)

    # ── Scale notionals to total_ead ──
    raw_total = df["notional"].sum()
    if raw_total > 0:
        df = df.with_columns((pl.col("notional") * (total_ead / raw_total)).alias("ead"))
    else:
        df = df.with_columns(pl.lit(total_ead / n_positions).alias("ead"))

    # ── Compute PD and LGD ──
    df = df.with_columns(pl.Series("pd_position", compute_sovereign_pd(df)))
    df = df.with_columns(pl.Series("lgd_position", compute_sovereign_lgd(df, rng)))

    # ── Yield (for downstream spread analysis) ──
    df = df.with_columns(pl.col("coupon_rate").alias("yield_to_maturity"))  # simplified: coupon ~ YTM at par

    return df


# ──────────────────────────────────────────────
# PD COMPOSITE (rating + fiscal + marche)
# ──────────────────────────────────────────────

def compute_sovereign_pd(df: pl.DataFrame) -> np.ndarray:
    """Calcule la PD composite pour chaque obligation souveraine.

    Modele 3 composantes :
        1. PD fondamentale = PD_rating × alpha_debt × alpha_fiscal
        2. PD marche = CDS_spread / (10000 × (1 - recovery))
        3. PD composite = 0.70 × fondamentale + 0.30 × marche

    Args:
        df: DataFrame from generate_sovereign_positions().

    Returns:
        Array de PD par position.
    """
    n = len(df)
    pd_composite = np.zeros(n)

    rating_arr = df["rating"].to_numpy()
    debt_gdp_arr = df["debt_to_gdp"].to_numpy()
    fiscal_arr = df["fiscal_balance"].to_numpy()
    cds_bp_arr = df["cds_spread_bp"].to_numpy()
    recovery_arr = df["recovery_rate_hist"].to_numpy()

    for i in range(n):
        rating = rating_arr[i]
        debt_gdp = debt_gdp_arr[i]
        fiscal = fiscal_arr[i]
        cds_bp = cds_bp_arr[i]
        recovery = recovery_arr[i]

        # 1. PD fondamentale
        pd_rating = RATING_PD_MAP.get(rating, 0.0008)
        alpha_debt = np.exp(1.2 * max(0.0, debt_gdp - _MAASTRICHT_THRESHOLD))
        alpha_fiscal = 1.0 - 0.5 * max(0.0, fiscal)  # surplus reduces PD
        pd_fundamental = pd_rating * alpha_debt * alpha_fiscal

        # 2. PD marche (CDS-implied)
        pd_market = cds_bp / 10000.0 / max(1.0 - recovery, 0.20)

        # 3. Composite
        pd_composite[i] = 0.70 * pd_fundamental + 0.30 * pd_market

    return np.clip(pd_composite, _PD_FLOOR, _PD_CAP)


# ──────────────────────────────────────────────
# LGD CRUCES-TREBESCH (2013 AER)
# ──────────────────────────────────────────────

def compute_sovereign_lgd(
    df: pl.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Calcule la LGD pour chaque obligation souveraine.

    Calibration Cruces & Trebesch (2013 AER) :
        - 187 restructurations souveraines 1970-2010
        - Mean haircut 37%, median 27%
        - IG base : 0.35, HY base : 0.50
        - Ajustement maturite : short-dated recuperent plus (-10% pour tenor < 5y)
        - Ajustement pays : modulation par recovery_rate_hist

    Args:
        df: DataFrame from generate_sovereign_positions().
        rng: Optional random generator for noise.

    Returns:
        Array de LGD par position.
    """
    n = len(df)
    lgd = np.zeros(n)

    _HY_RATINGS = {"BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-"}

    rating_arr = df["rating"].to_numpy()
    tenor_arr = df["tenor_years"].to_numpy()
    recovery_hist_arr = df["recovery_rate_hist"].to_numpy()

    for i in range(n):
        rating = rating_arr[i]
        tenor = tenor_arr[i]
        recovery_hist = recovery_hist_arr[i]

        # Base LGD by credit quality
        if rating in _HY_RATINGS:
            base = _LGD_HY_BASE
        else:
            base = _LGD_IG_BASE

        # Maturity adjustment: short-dated bonds recover more
        mat_adj = _LGD_MATURITY_ADJ if tenor < 5.0 else 0.0

        # Country adjustment: modulate by historical recovery
        country_adj = (0.80 - recovery_hist) * 0.30  # 80% = neutral recovery

        lgd[i] = base + mat_adj + country_adj

    # Add noise
    if rng is not None:
        lgd += rng.normal(0.0, _LGD_NOISE_STD, n)

    return np.clip(lgd, _LGD_FLOOR, _LGD_CAP)


# ──────────────────────────────────────────────
# STRESS TEST (3 canaux + amplification peripherique)
# ──────────────────────────────────────────────

def stress_sovereign_positions(
    df: pl.DataFrame,
    macro_params: Dict[str, float],
) -> Dict[str, float]:
    """Stress test des positions souveraines et re-aggregation.

    3 canaux de transmission :
        1. Taux → spread (avec amplification peripherique)
        2. GDP → fiscal → dette/GDP → PD
        3. LGD downturn (leger pour souverain)

    RW reste TOUJOURS 0% (CRR3 Art. 114(4)).

    Args:
        df: DataFrame from generate_sovereign_positions().
        macro_params: Dict avec gdp_growth, interest_rate, etc.

    Returns:
        Dict avec pd_base, lgd_base, rw_crr3 stresses (EAD-weighted).
    """
    # Deltas
    delta_gdp = (macro_params.get("gdp_growth", _BASE_GDP) - _BASE_GDP) / 100.0
    delta_rate = (macro_params.get("interest_rate", _BASE_RATE) - _BASE_RATE) / 100.0

    n = len(df)
    ead = df["ead"].to_numpy()
    country_codes = df["country_code"].to_numpy()

    # ── Canal 1: Taux → spread (amplification peripherique) ──
    amplification = np.array([
        SPREAD_AMPLIFICATION.get(cc, 1.0) for cc in country_codes
    ])
    # Spread shock as PD multiplier: higher rates → wider spreads
    # Calibration: EBA 2023 severe (+300bp) → sovereign PD 3-5x for periphery
    # +100bp rate × amp 2.0 (periphery) → PD ×2.0 (~100% increase)
    # +100bp rate × amp 0.3 (safe haven) → PD ×1.09 (~9% increase)
    spread_pd_mult = 1.0 + delta_rate * amplification * 15.0

    # ── Canal 2: GDP → fiscal deterioration → PD ──
    # GDP negative → fiscal balance worsens → debt/GDP increases
    # Calibration: -5% GDP → +5pp deficit → ~25pp debt/GDP increase (EBA severe)
    fiscal_deterioration = -0.5 * delta_gdp  # -1% GDP → +0.5% deficit
    stressed_debt_gdp = df["debt_to_gdp"].to_numpy() + fiscal_deterioration * 5.0  # deficit → debt accumulation

    # Re-compute alpha_debt with stressed debt/GDP
    alpha_debt_stressed = np.exp(1.2 * np.maximum(0.0, stressed_debt_gdp - _MAASTRICHT_THRESHOLD))
    alpha_debt_base = np.exp(1.2 * np.maximum(0.0, df["debt_to_gdp"].to_numpy() - _MAASTRICHT_THRESHOLD))

    # PD multiplier from fiscal channel
    fiscal_pd_mult = np.where(
        alpha_debt_base > 0,
        alpha_debt_stressed / np.maximum(alpha_debt_base, 1e-6),
        1.0,
    )

    # ── Combine PD stress (multiplicative channels) ──
    pd_base_arr = df["pd_position"].to_numpy()
    pd_stressed = pd_base_arr * fiscal_pd_mult * np.maximum(spread_pd_mult, 0.5)
    pd_stressed = np.clip(pd_stressed, _PD_FLOOR, _PD_CAP)

    # ── Canal 3: LGD downturn ──
    gdp_growth_pct = macro_params.get("gdp_growth", _BASE_GDP)
    lgd_addon = np.clip(
        -(gdp_growth_pct - _BASE_GDP) / 100.0 * _LGD_DOWNTURN_COEFF,
        0.0,
        _LGD_DOWNTURN_CAP,
    )
    lgd_stressed = np.clip(df["lgd_position"].to_numpy() + lgd_addon, _LGD_FLOOR, _LGD_CAP)

    # ── RW always 0% ──
    rw_stressed = 0.0

    # ── EAD-weighted aggregation ──
    total_ead = ead.sum()
    if total_ead <= 0:
        return {"pd_base": 0.0005, "lgd_base": 0.35, "rw_crr3": 0.0}

    w = ead / total_ead
    pd_agg = float(np.dot(w, pd_stressed))
    lgd_agg = float(np.dot(w, lgd_stressed))

    return {
        "pd_base": round(pd_agg, 6),
        "lgd_base": round(lgd_agg, 6),
        "rw_crr3": round(rw_stressed, 4),
    }


# ──────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────

def aggregate_sovereign_to_balance_row(
    df_positions: pl.DataFrame,
    profile: "AssetClassProfile",
) -> dict:
    """Agrege les positions souveraines en une ligne balance sheet.

    Pattern identique a project_finance : EAD-weighted PD/LGD/tenor.
    RW = 0.0 toujours (CRR3 Art. 114(4)).

    Args:
        df_positions: DataFrame from generate_sovereign_positions().
        profile: AssetClassProfile for sovereign.

    Returns:
        Dict avec toutes les colonnes balance sheet (18 cles).
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
        rw_agg = 0.0  # always 0%

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
