"""Generateur position-par-position pour le portefeuille Titrisation.

Genere ~300 tranches (50-60 deals) avec mecanismes de risque specifiques :
PD Vasicek tranche-level, LGD calibree Moody's par seniorite, SEC-SA RW
(CRR3 Art. 261), et stress test position-par-position (full online ~5ms).

6 types de pools : RMBS, CLO, ABS Auto, ABS Consumer, CMBS, SME ABS.
4 produits sur donnees reelles (parquet), 2 parametriques (CLO, SME ABS).
Fallback parametrique si parquet absent (zero regression).

Calibration :
    - AFME Q2 2025 : encours 1270 Mds EUR, composition par type
    - ESMA EU Securitisation Overview : STS ~30-40%
    - BIS QR Dec 2014 (Antoniades) : cliff effect, sous-capitalisation mezzanine
    - Vasicek (1987/2002) : LHP model, pool loss distribution
    - Moody's Structured Finance Default Study : LGD par seniorite
    - S&P Default Study (1978-2009) : PD par rating, zero AAA default
    - TwentyFour AM ABS Outlook 2025 : spreads par type/rating
    - Freddie Mac SFLLD : pool-level RMBS (PD/LGD/CPR reels)
    - SEC EDGAR ABS-EE : loan-level ABS Auto (Reg AB II)
    - Freddie Mac Multifamily LPD : CMBS proxy

References :
    - CRR3 Art. 261 : SEC-SA formula (K_SSFA, p-factor, floors)
    - CRR3 Art. 242-270 : securitisation framework
    - Basel CRE 42 : SEC-ERBA tables
    - Basel IRB : correlation formulas corporate/retail
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import polars as pl
from scipy.stats import norm


# ──────────────────────────────────────────────
# CONSTANTES (calibrees AFME/ESMA/Moody's/S&P/BIS)
# ──────────────────────────────────────────────

# --- 6 types de pools ---
POOL_TYPES: Dict[str, Dict] = {
    "rmbs": {
        "weight": 0.35,
        "pool_pd": 0.010,
        "pool_lgd": 0.15,
        "rho": 0.12,
        "wal_mean": 5.5,
        "wac": 0.028,
        "n_obligors": 15000,
        "sts_prob": 0.50,
        "tranches_per_deal": 5,
        "beta_hpi": -2.0,
        "beta_unemp": 0.30,
        "beta_gdp": 0.20,
        "beta_rate": 0.0,
        "beta_inflation": 0.10,
    },
    "clo": {
        "weight": 0.25,
        "pool_pd": 0.030,
        "pool_lgd": 0.40,
        "rho": 0.20,
        "wal_mean": 5.0,
        "wac": 0.050,
        "n_obligors": 200,
        "sts_prob": 0.00,
        "tranches_per_deal": 6,
        "beta_hpi": 0.0,
        "beta_unemp": 0.10,
        "beta_gdp": 2.0,
        "beta_rate": 0.0,
        "beta_inflation": 0.0,
    },
    "abs_auto": {
        "weight": 0.15,
        "pool_pd": 0.012,
        "pool_lgd": 0.35,
        "rho": 0.10,
        "wal_mean": 2.5,
        "wac": 0.040,
        "n_obligors": 30000,
        "sts_prob": 0.60,
        "tranches_per_deal": 4,
        "beta_hpi": 0.0,
        "beta_unemp": 1.50,
        "beta_gdp": 0.30,
        "beta_rate": 0.0,
        "beta_inflation": 0.10,
    },
    "abs_consumer": {
        "weight": 0.10,
        "pool_pd": 0.030,
        "pool_lgd": 0.50,
        "rho": 0.08,
        "wal_mean": 3.0,
        "wac": 0.065,
        "n_obligors": 20000,
        "sts_prob": 0.40,
        "tranches_per_deal": 4,
        "beta_hpi": 0.0,
        "beta_unemp": 1.50,
        "beta_gdp": 0.20,
        "beta_rate": 0.0,
        "beta_inflation": 0.80,
    },
    "cmbs": {
        "weight": 0.08,
        "pool_pd": 0.020,
        "pool_lgd": 0.30,
        "rho": 0.20,
        "wal_mean": 5.0,
        "wac": 0.040,
        "n_obligors": 50,
        "sts_prob": 0.00,
        "tranches_per_deal": 6,
        "beta_hpi": -1.0,
        "beta_unemp": 0.20,
        "beta_gdp": 1.0,
        "beta_rate": 1.50,
        "beta_inflation": 0.10,
    },
    "sme_abs": {
        "weight": 0.07,
        "pool_pd": 0.025,
        "pool_lgd": 0.45,
        "rho": 0.15,
        "wal_mean": 3.5,
        "wac": 0.045,
        "n_obligors": 5000,
        "sts_prob": 0.30,
        "tranches_per_deal": 4,
        "beta_hpi": 0.0,
        "beta_unemp": 0.50,
        "beta_gdp": 2.50,
        "beta_rate": 0.80,
        "beta_inflation": 0.30,
    },
}

# --- 6 niveaux de seniorite (templates de tranches) ---
TRANCHE_TEMPLATES: Dict[str, Dict] = {
    "senior":        {"rating": "AAA", "attach_mult": 3.0,  "spread_bps": 80,   "lgd": 0.08},
    "mezzanine_aa":  {"rating": "AA",  "attach_mult": 2.0,  "spread_bps": 150,  "lgd": 0.15},
    "mezzanine_a":   {"rating": "A",   "attach_mult": 1.5,  "spread_bps": 220,  "lgd": 0.25},
    "mezzanine_bbb": {"rating": "BBB", "attach_mult": 1.0,  "spread_bps": 320,  "lgd": 0.35},
    "junior":        {"rating": "BB",  "attach_mult": 0.6,  "spread_bps": 580,  "lgd": 0.55},
    "equity":        {"rating": "NR",  "attach_mult": 0.0,  "spread_bps": 1200, "lgd": 0.90},
}

# Spread adjustment par type de pool (base = CLO)
SPREAD_ADJUSTMENT: Dict[str, float] = {
    "rmbs": 0.60,
    "clo": 1.00,
    "abs_auto": 0.55,
    "abs_consumer": 0.75,
    "cmbs": 0.90,
    "sme_abs": 0.80,
}

# SEC-SA parametres (CRR3 Art. 261)
SEC_SA_PARAMS: Dict[str, float] = {
    "p_sts": 0.5,
    "p_non_sts": 1.0,
    "floor_senior_sts": 0.10,
    "floor_senior_non_sts": 0.15,
    "floor_non_senior": 0.15,
    "deduction_rw": 12.50,
}

# LGD adjustment par type de pool (multiplicateur sur LGD calibree Moody's)
_LGD_POOL_ADJUSTMENT: Dict[str, float] = {
    "rmbs": 0.80,          # -20% car collateral immobilier
    "clo": 1.00,           # reference
    "abs_auto": 0.90,      # -10% car collateral vehicule
    "abs_consumer": 1.15,  # +15% unsecured
    "cmbs": 0.85,          # -15% car immobilier commercial
    "sme_abs": 1.05,       # +5% PME
}

# Stress parameters
_STRESS_LGD_GDP_COEFF = 0.02      # +2pp LGD par -1% GDP
_STRESS_LGD_HPI_COEFF = 0.01      # +1pp LGD par -1% HPI (RMBS)
_STRESS_LGD_DOWNTURN_CAP = 0.15   # max +15pp downturn addon

# PD floors/caps
_PD_FLOOR = 0.0001
_PD_CAP = 0.50

# Parquet paths for real data
_DATA_DIR = Path(__file__).parent.parent / "data"
_PARQUET_RMBS = _DATA_DIR / "freddie_mac_rmbs.parquet"
_PARQUET_ABS_AUTO = _DATA_DIR / "abs_auto_loans.parquet"
_PARQUET_CMBS = _DATA_DIR / "cmbs_multifamily.parquet"


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _weighted_choice(
    rng: np.random.Generator,
    categories: Dict[str, Dict],
    n: int,
    weight_key: str = "weight",
) -> np.ndarray:
    """Sample from weighted categories."""
    names = list(categories.keys())
    weights = np.array([categories[k][weight_key] for k in names])
    weights = weights / weights.sum()
    return rng.choice(names, size=n, p=weights)


def _vasicek_var(pool_pd: float, pool_lgd: float, rho: float,
                 confidence: float = 0.999) -> float:
    """Vasicek LHP portfolio VaR at given confidence level."""
    z = norm.ppf(confidence)
    cond_pd = norm.cdf(
        (norm.ppf(pool_pd) + np.sqrt(rho) * z) / np.sqrt(1.0 - rho)
    )
    return pool_lgd * cond_pd


def _load_real_pool_stats(pool_type: str) -> Optional[Dict[str, float]]:
    """Load pool statistics from real data parquet if available."""
    parquet_map = {
        "rmbs": _PARQUET_RMBS,
        "abs_auto": _PARQUET_ABS_AUTO,
        "cmbs": _PARQUET_CMBS,
    }
    path = parquet_map.get(pool_type)
    if path is None or not path.exists():
        return None
    try:
        df = pl.read_parquet(path)
        stats = {}
        if "default_rate" in df.columns:
            stats["pool_pd"] = float(np.clip(df["default_rate"].mean(), 0.001, 0.10))
        if "loss_rate" in df.columns:
            stats["pool_lgd"] = float(np.clip(df["loss_rate"].mean(), 0.05, 0.70))
        return stats if stats else None
    except Exception:
        return None


# ──────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────

def generate_securitisation_positions(
    n_tranches: int = 300,
    total_ead: float = 1.0e9,
    seed: int = 842,
) -> pl.DataFrame:
    """Generate individual securitisation tranche positions.

    Creates ~50-60 deals with 4-6 tranches each, covering 6 pool types.
    If real data parquets are available (RMBS, ABS Auto, CMBS), pool
    statistics are calibrated from them; otherwise parametric fallback.

    Args:
        n_tranches: Approximate number of tranches to generate.
        total_ead: Target total EAD across all tranches.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with ~25 columns per tranche position.
    """
    rng = np.random.default_rng(seed)

    # --- 1. Determine number of deals per pool type ---
    pool_names = list(POOL_TYPES.keys())
    pool_weights = np.array([POOL_TYPES[k]["weight"] for k in pool_names])
    pool_weights = pool_weights / pool_weights.sum()

    avg_tranches = np.mean([POOL_TYPES[k]["tranches_per_deal"] for k in pool_names])
    n_deals = max(10, int(n_tranches / avg_tranches))

    deal_pool_types = rng.choice(pool_names, size=n_deals, p=pool_weights)

    # --- 2. Generate deal-level pool characteristics ---
    records = []
    deal_id = 0

    for pool_type in deal_pool_types:
        cfg = POOL_TYPES[pool_type]

        # Try loading real data stats for this pool type
        real_stats = _load_real_pool_stats(pool_type)

        # Pool PD (with noise)
        base_pd = real_stats.get("pool_pd", cfg["pool_pd"]) if real_stats else cfg["pool_pd"]
        pool_pd = float(np.clip(base_pd + rng.normal(0, 0.002), 0.001, 0.10))

        # Pool LGD (with noise)
        base_lgd = real_stats.get("pool_lgd", cfg["pool_lgd"]) if real_stats else cfg["pool_lgd"]
        pool_lgd = float(np.clip(base_lgd + rng.normal(0, 0.03), 0.05, 0.70))

        # Pool characteristics
        rho = cfg["rho"]
        n_obligors = max(10, int(cfg["n_obligors"] + rng.normal(0, cfg["n_obligors"] * 0.1)))
        wal = float(np.clip(cfg["wal_mean"] + rng.normal(0, 0.5), 0.5, 10.0))
        wac = float(np.clip(cfg["wac"] + rng.normal(0, 0.003), 0.01, 0.10))
        pool_size = float(np.exp(rng.normal(20.5, 0.6)))  # median ~800M EUR
        delinquency_rate = float(np.clip(
            pool_pd * 1.5 * rng.uniform(0.8, 1.2), 0.001, 0.20,
        ))
        is_sts = int(rng.random() < cfg["sts_prob"])
        vintage = int(rng.integers(2018, 2026))

        # --- 3. Compute attachment/detachment via Vasicek VaR ---
        var_999 = _vasicek_var(pool_pd, pool_lgd, rho)

        # Determine which tranches this deal has
        n_deal_tranches = cfg["tranches_per_deal"]
        tranche_names = list(TRANCHE_TEMPLATES.keys())[:n_deal_tranches]

        # Build attachment/detachment waterfall
        attachments = []
        for t_name in tranche_names:
            tmpl = TRANCHE_TEMPLATES[t_name]
            attach = var_999 * tmpl["attach_mult"]
            attachments.append((t_name, attach))

        # Sort by attachment (ascending: equity=0, ..., senior=highest)
        attachments.sort(key=lambda x: x[1])

        for idx, (t_name, attach) in enumerate(attachments):
            if idx < len(attachments) - 1:
                detach = attachments[idx + 1][1]
            else:
                detach = 1.0  # senior tranche goes to 100%

            # Tranche thickness
            thickness = max(detach - attach, 0.001)
            notional = thickness * pool_size

            tmpl = TRANCHE_TEMPLATES[t_name]
            spread_adj = SPREAD_ADJUSTMENT.get(pool_type, 1.0)

            records.append({
                "tranche_id": len(records),
                "deal_id": deal_id,
                "pool_type": pool_type,
                "tranche_type": t_name,
                "rating": tmpl["rating"],
                "attachment": round(attach, 6),
                "detachment": round(detach, 6),
                "pool_pd": round(pool_pd, 6),
                "pool_lgd": round(pool_lgd, 6),
                "rho": rho,
                "n_obligors": n_obligors,
                "pool_size": round(pool_size, 2),
                "wal": round(wal, 2),
                "wac": round(wac, 6),
                "delinquency_rate": round(delinquency_rate, 6),
                "is_sts": is_sts,
                "vintage": vintage,
                "notional": round(notional, 2),
                "ead": round(notional, 2),
                "spread_bps": round(tmpl["spread_bps"] * spread_adj, 1),
            })

        deal_id += 1

    df = pl.DataFrame(records)

    # --- 4. Scale EAD to total_ead ---
    raw_total = df["ead"].sum()
    if raw_total > 0:
        scale = total_ead / raw_total
        df = df.with_columns(
            (pl.col("ead") * scale).round(2).alias("ead"),
            (pl.col("notional") * scale).round(2).alias("notional"),
            (pl.col("pool_size") * scale).round(2).alias("pool_size"),
        )

    # --- 5. Compute PD, LGD, SEC-SA RW ---
    df = df.with_columns(pl.Series("pd_tranche", compute_securitisation_pd(df)))
    df = df.with_columns(pl.Series("lgd_tranche", compute_securitisation_lgd(df)))
    df = df.with_columns(pl.Series("rw_sec_sa", compute_sec_sa_rw(df)))

    return df


# ──────────────────────────────────────────────
# PD VASICEK TRANCHE (CRR3 / Basel IRB)
# ──────────────────────────────────────────────

def compute_securitisation_pd(df: pl.DataFrame) -> np.ndarray:
    """Compute tranche-level PD via Vasicek LHP model.

    PD_tranche = P(pool_loss > attachment)
               = 1 - Phi((sqrt(1-rho) * Phi^-1(A/pool_lgd) - Phi^-1(pool_pd)) / sqrt(rho))

    Clip: [0.0001, 0.50].
    """
    pool_pd = df["pool_pd"].to_numpy()
    pool_lgd = df["pool_lgd"].to_numpy()
    rho = df["rho"].to_numpy()
    attach = df["attachment"].to_numpy()

    # Normalized attachment point (fraction of pool loss)
    # A / pool_lgd gives the attachment as fraction of max loss
    attach_norm = np.clip(attach / np.maximum(pool_lgd, 0.01), 0.0001, 0.9999)

    sqrt_rho = np.sqrt(np.maximum(rho, 1e-6))
    sqrt_1_rho = np.sqrt(np.maximum(1.0 - rho, 1e-6))

    # Vasicek inverse: P(L > A)
    z_a = norm.ppf(attach_norm)
    z_pd = norm.ppf(np.clip(pool_pd, 1e-6, 1 - 1e-6))

    arg = (sqrt_1_rho * z_a - z_pd) / sqrt_rho
    pd_tranche = 1.0 - norm.cdf(arg)

    return np.clip(pd_tranche, _PD_FLOOR, _PD_CAP)


# ──────────────────────────────────────────────
# LGD TRANCHE (calibree Moody's par seniorite)
# ──────────────────────────────────────────────

def compute_securitisation_lgd(df: pl.DataFrame) -> np.ndarray:
    """Compute tranche-level LGD calibrated on Moody's recovery data.

    Base LGD from tranche template, adjusted by pool type:
    - RMBS: -20% (immobilier collateral)
    - ABS Consumer: +15% (unsecured)
    """
    tranche_types = df["tranche_type"].to_numpy()
    pool_types = df["pool_type"].to_numpy()

    lgd = np.array([
        TRANCHE_TEMPLATES[t]["lgd"] for t in tranche_types
    ], dtype=float)

    # Pool type adjustment
    adj = np.array([
        _LGD_POOL_ADJUSTMENT.get(pt, 1.0) for pt in pool_types
    ], dtype=float)

    lgd = lgd * adj
    return np.clip(lgd, 0.05, 0.95).round(4)


# ──────────────────────────────────────────────
# SEC-SA RISK WEIGHT (CRR3 Art. 261)
# ──────────────────────────────────────────────

def compute_sec_sa_rw(df: pl.DataFrame) -> np.ndarray:
    """Compute SEC-SA risk weight per CRR3 Art. 261 K_SSFA formula.

    Steps:
    1. K_G = 0.08 x average pool RW (using pool PD -> SA corporate RW proxy)
    2. W = delinquency_rate
    3. K_A = (1-W)*K_G + 0.5*W
    4. Apply K_SSFA formula with p-factor (STS vs non-STS)
    5. Floors: 10% senior STS, 15% otherwise
    6. Deduction 1250% if D <= K_A
    """
    n = len(df)
    pool_pd = df["pool_pd"].to_numpy()
    attach = df["attachment"].to_numpy()
    detach = df["detachment"].to_numpy()
    delinq = df["delinquency_rate"].to_numpy()
    is_sts = df["is_sts"].to_numpy()
    tranche_types = df["tranche_type"].to_numpy()

    # K_G: pool capital requirement (8% x SA RW)
    # SA corporate RW proxy: ~75% for typical pool
    # More precisely: RW = max(20%, min(150%, 12.5 * (pool_pd * 8)))
    rw_pool = np.clip(12.5 * pool_pd * 8.0, 0.20, 1.50)
    k_g = 0.08 * rw_pool

    # W: delinquency/default ratio
    w = np.clip(delinq, 0.0, 1.0)

    # K_A: attachment point of full deduction
    k_a = (1.0 - w) * k_g + 0.5 * w

    # p-factor
    p = np.where(is_sts == 1, SEC_SA_PARAMS["p_sts"], SEC_SA_PARAMS["p_non_sts"])

    rw = np.zeros(n, dtype=float)

    for i in range(n):
        a_i = attach[i]
        d_i = detach[i]
        ka_i = k_a[i]
        p_i = p[i]

        if d_i <= ka_i:
            # Full deduction
            rw[i] = SEC_SA_PARAMS["deduction_rw"]
        elif a_i >= ka_i:
            # Entire tranche above K_A
            a_param = -1.0 / (p_i * ka_i) if ka_i > 1e-8 else -100.0
            u = d_i - ka_i
            l = a_i - ka_i
            if abs(u - l) < 1e-10:
                rw[i] = SEC_SA_PARAMS["deduction_rw"]
            else:
                k_ssfa = (np.exp(a_param * u) - np.exp(a_param * l)) / (a_param * (u - l))
                rw[i] = k_ssfa * 12.50
        else:
            # Mixed: A < K_A < D
            part_deducted = (ka_i - a_i) / (d_i - a_i) if (d_i - a_i) > 1e-10 else 1.0
            part_formula = 1.0 - part_deducted

            a_param = -1.0 / (p_i * ka_i) if ka_i > 1e-8 else -100.0
            u_upper = d_i - ka_i
            if u_upper > 1e-10:
                k_ssfa_upper = (np.exp(a_param * u_upper) - 1.0) / (a_param * u_upper)
            else:
                k_ssfa_upper = 1.0

            rw[i] = part_deducted * SEC_SA_PARAMS["deduction_rw"] + part_formula * k_ssfa_upper * 12.50

    # Apply floors
    is_senior = np.array([t == "senior" for t in tranche_types])
    floor_sts_senior = SEC_SA_PARAMS["floor_senior_sts"]
    floor_non_sts_senior = SEC_SA_PARAMS["floor_senior_non_sts"]
    floor_non_senior = SEC_SA_PARAMS["floor_non_senior"]

    for i in range(n):
        if is_senior[i]:
            if is_sts[i]:
                rw[i] = max(rw[i], floor_sts_senior)
            else:
                rw[i] = max(rw[i], floor_non_sts_senior)
        else:
            rw[i] = max(rw[i], floor_non_senior)

    return np.round(rw, 4)


# ──────────────────────────────────────────────
# STRESS TEST POSITION-PAR-POSITION (FULL ONLINE)
# ──────────────────────────────────────────────

def stress_securitisation_positions(
    df: pl.DataFrame,
    macro_params: Dict[str, float],
) -> Dict[str, float]:
    """Stress securitisation positions under macro scenario and re-aggregate.

    Full online, ~5ms for 300 tranches. Implements 3 transmission channels:
    1. Pool PD stressed by type-specific macro betas
    2. Pool LGD downturn addon (GDP, HPI)
    3. Tranche PD/LGD/RW re-calculated with stressed pool parameters

    Args:
        df: DataFrame from generate_securitisation_positions().
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

    # Deltas (percentage points -> fraction)
    delta_gdp = (macro_params.get("gdp_growth", base_gdp) - base_gdp) / 100.0
    delta_unemp = (macro_params.get("unemployment_rate", base_unemp) - base_unemp) / 100.0
    delta_rate = (macro_params.get("interest_rate", base_rate) - base_rate) / 100.0
    delta_hpi = (macro_params.get("hpi_growth", base_hpi) - base_hpi) / 100.0
    delta_inflation = (macro_params.get("inflation_rate", base_inflation) - base_inflation) / 100.0

    n = len(df)
    pool_types = df["pool_type"].to_numpy()
    ead = df["ead"].to_numpy()
    pool_pd_base = df["pool_pd"].to_numpy().copy()
    pool_lgd_base = df["pool_lgd"].to_numpy().copy()

    # --- Canal 1: Pool PD stressed by type-specific betas ---
    stress_exponent = np.zeros(n, dtype=float)
    for i in range(n):
        cfg = POOL_TYPES[pool_types[i]]
        stress_exponent[i] = (
            cfg["beta_hpi"] * delta_hpi
            + cfg["beta_unemp"] * delta_unemp
            + cfg["beta_gdp"] * abs(min(0.0, delta_gdp))
            + cfg["beta_rate"] * delta_rate
            + cfg["beta_inflation"] * delta_inflation
        )

    pool_pd_stressed = np.clip(pool_pd_base * np.exp(stress_exponent), 0.001, 0.30)

    # --- Canal 2: Pool LGD downturn ---
    lgd_addon_gdp = np.clip(-delta_gdp * _STRESS_LGD_GDP_COEFF, 0.0, _STRESS_LGD_DOWNTURN_CAP)
    lgd_addon_hpi = np.zeros(n, dtype=float)
    is_rmbs = np.array([pt == "rmbs" for pt in pool_types])
    is_cmbs = np.array([pt == "cmbs" for pt in pool_types])
    lgd_addon_hpi[is_rmbs | is_cmbs] = np.clip(
        -delta_hpi * _STRESS_LGD_HPI_COEFF, 0.0, 0.05,
    )
    pool_lgd_stressed = np.clip(
        pool_lgd_base + lgd_addon_gdp + lgd_addon_hpi, 0.05, 0.70,
    )

    # --- Canal 3: Recalculate tranche PD/LGD/RW with stressed pool params ---
    df_stressed = df.clone()
    df_stressed = df_stressed.with_columns(pl.Series("pool_pd", pool_pd_stressed))
    df_stressed = df_stressed.with_columns(pl.Series("pool_lgd", pool_lgd_stressed))

    # Update delinquency rate proportionally
    if "delinquency_rate" in df_stressed.columns:
        ratio = pool_pd_stressed / np.maximum(pool_pd_base, 1e-6)
        df_stressed = df_stressed.with_columns(
            pl.Series("delinquency_rate", np.clip(
                df["delinquency_rate"].to_numpy() * ratio, 0.001, 0.20,
            ))
        )

    pd_stressed = compute_securitisation_pd(df_stressed)
    lgd_stressed = compute_securitisation_lgd(df_stressed)
    rw_stressed = compute_sec_sa_rw(df_stressed)

    # LGD tranche also gets downturn addon (propagated from pool)
    lgd_tranche_addon = lgd_addon_gdp + lgd_addon_hpi
    lgd_stressed = np.clip(lgd_stressed + lgd_tranche_addon, 0.05, 0.95)

    # --- EAD-weighted aggregation ---
    total_ead = ead.sum()
    if total_ead <= 0:
        return {"pd_base": 0.010, "lgd_base": 0.40, "rw_crr3": 1.00}

    w = ead / total_ead
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

def aggregate_securitisation_to_balance_row(
    df_positions: pl.DataFrame,
    profile: "AssetClassProfile",
) -> dict:
    """Aggregate individual securitisation tranches into a single balance sheet row.

    Returns a dict compatible with the df_balance_sheet schema.
    PD, LGD, RW are EAD-weighted averages (bottom-up SEC-SA).

    Args:
        df_positions: DataFrame from generate_securitisation_positions().
        profile: AssetClassProfile for structured_products.

    Returns:
        Dict with all balance sheet columns.
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
        pd_agg = float(np.dot(w, df_positions["pd_tranche"].to_numpy()))
        lgd_agg = float(np.dot(w, df_positions["lgd_tranche"].to_numpy()))
        tenor_agg = float(np.dot(w, df_positions["wal"].to_numpy()))
        rw_agg = float(np.dot(w, df_positions["rw_sec_sa"].to_numpy()))

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
