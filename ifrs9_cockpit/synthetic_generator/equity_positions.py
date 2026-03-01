"""Generateur position-par-position pour le portefeuille Actions Cotees.

Genere ~100 positions actions europeennes (30 large-caps EU) avec PD
Merton (distance-to-default structurelle), LGD first-loss [0.70, 1.00],
stress test 4 canaux (index_shock, rate_sensitivity, sector_rotation,
vol_expansion GJR-GARCH), et MTM loss au lieu d'ECL (traitement FVTPL).

Le pipeline aval (comparator RAROC, BL-CVaR) consomme des valeurs
calculees bottom-up. Pas de staging IFRS 9 (FVTPL, pas d'ECL).

Donnees reelles (priorite) :
    - STOXX 600 (parquet pre-traite) : market_cap, total_debt, beta, vol
    - Fallback : constantes EU_LARGE_CAPS si parquet absent

Calibration :
    - Merton (1974) : modele structurel, distance-to-default (DD)
    - KMV/Moody's Analytics : EDF calibration, recalibration factor ~5.0
    - EBA 2023 Stress Test : equity haircut = beta × (-2.5×dGDP - 0.8×dUnemp)
    - GJR-GARCH (1993) : leverage effect, vol expansion asymetrique
    - VSTOXX : reference de volatilite europeenne, long-run ~20%
    - Dimson-Marsh-Staunton (2023 DMS Yearbook) : equity risk premium EU

References :
    - CRR3 Art. 133 : RW = 100% pour actions cotees (listed equity)
    - IFRS 9 par. 4.1.4 : FVTPL = pas d'ECL, impact P&L via mark-to-market
    - IFRS 9 par. 5.7.1 : gains/pertes en resultat
    - Basel III NSFR : RSF = 85% pour equities (illiquide)
    - BCBS 352 (FRTB) : sensibility-based approach equity risk class
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


# ──────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data"

# --- 30 European Large-Cap Stocks (STOXX 50 / STOXX 600 core) ---
# Market data as of Q4 2025 (approximations for parametric fallback)
# market_cap in EUR billions, total_debt in EUR billions
EU_LARGE_CAPS: List[Dict] = [
    # ── Germany (7) ──
    {
        "ticker": "SAP.DE", "name": "SAP SE",
        "sector_gics": "Technology", "country": "DE",
        "market_cap": 260.0, "total_debt": 32.0,
        "beta": 1.15, "volatility_252d": 0.26, "dividend_yield": 0.008,
    },
    {
        "ticker": "SIE.DE", "name": "Siemens AG",
        "sector_gics": "Industrials", "country": "DE",
        "market_cap": 145.0, "total_debt": 45.0,
        "beta": 1.10, "volatility_252d": 0.24, "dividend_yield": 0.025,
    },
    {
        "ticker": "ALV.DE", "name": "Allianz SE",
        "sector_gics": "Financials", "country": "DE",
        "market_cap": 115.0, "total_debt": 55.0,
        "beta": 1.15, "volatility_252d": 0.22, "dividend_yield": 0.045,
    },
    {
        "ticker": "DTE.DE", "name": "Deutsche Telekom AG",
        "sector_gics": "Telecom", "country": "DE",
        "market_cap": 130.0, "total_debt": 95.0,
        "beta": 0.65, "volatility_252d": 0.18, "dividend_yield": 0.035,
    },
    {
        "ticker": "MUV2.DE", "name": "Munich Re",
        "sector_gics": "Financials", "country": "DE",
        "market_cap": 65.0, "total_debt": 15.0,
        "beta": 1.05, "volatility_252d": 0.20, "dividend_yield": 0.032,
    },
    {
        "ticker": "BAS.DE", "name": "BASF SE",
        "sector_gics": "Materials", "country": "DE",
        "market_cap": 40.0, "total_debt": 22.0,
        "beta": 1.20, "volatility_252d": 0.30, "dividend_yield": 0.055,
    },
    {
        "ticker": "MBG.DE", "name": "Mercedes-Benz Group",
        "sector_gics": "Consumer", "country": "DE",
        "market_cap": 70.0, "total_debt": 100.0,
        "beta": 1.25, "volatility_252d": 0.32, "dividend_yield": 0.065,
    },
    # ── France (7) ──
    {
        "ticker": "MC.PA", "name": "LVMH",
        "sector_gics": "Consumer", "country": "FR",
        "market_cap": 350.0, "total_debt": 30.0,
        "beta": 0.95, "volatility_252d": 0.28, "dividend_yield": 0.015,
    },
    {
        "ticker": "TTE.PA", "name": "TotalEnergies SE",
        "sector_gics": "Energy", "country": "FR",
        "market_cap": 150.0, "total_debt": 48.0,
        "beta": 1.05, "volatility_252d": 0.25, "dividend_yield": 0.050,
    },
    {
        "ticker": "SAN.PA", "name": "Sanofi SA",
        "sector_gics": "Healthcare", "country": "FR",
        "market_cap": 120.0, "total_debt": 25.0,
        "beta": 0.70, "volatility_252d": 0.20, "dividend_yield": 0.038,
    },
    {
        "ticker": "AIR.PA", "name": "Airbus SE",
        "sector_gics": "Industrials", "country": "FR",
        "market_cap": 110.0, "total_debt": 18.0,
        "beta": 1.30, "volatility_252d": 0.30, "dividend_yield": 0.012,
    },
    {
        "ticker": "BNP.PA", "name": "BNP Paribas SA",
        "sector_gics": "Financials", "country": "FR",
        "market_cap": 80.0, "total_debt": 120.0,
        "beta": 1.30, "volatility_252d": 0.28, "dividend_yield": 0.060,
    },
    {
        "ticker": "SU.PA", "name": "Schneider Electric SE",
        "sector_gics": "Industrials", "country": "FR",
        "market_cap": 120.0, "total_debt": 15.0,
        "beta": 1.05, "volatility_252d": 0.24, "dividend_yield": 0.016,
    },
    {
        "ticker": "ORA.PA", "name": "Orange SA",
        "sector_gics": "Telecom", "country": "FR",
        "market_cap": 28.0, "total_debt": 32.0,
        "beta": 0.55, "volatility_252d": 0.18, "dividend_yield": 0.065,
    },
    # ── Netherlands (4) ──
    {
        "ticker": "ASML.AS", "name": "ASML Holding NV",
        "sector_gics": "Technology", "country": "NL",
        "market_cap": 380.0, "total_debt": 5.0,
        "beta": 1.35, "volatility_252d": 0.35, "dividend_yield": 0.006,
    },
    {
        "ticker": "INGA.AS", "name": "ING Group NV",
        "sector_gics": "Financials", "country": "NL",
        "market_cap": 55.0, "total_debt": 45.0,
        "beta": 1.25, "volatility_252d": 0.28, "dividend_yield": 0.055,
    },
    {
        "ticker": "UNA.AS", "name": "Unilever NV",
        "sector_gics": "Consumer", "country": "NL",
        "market_cap": 130.0, "total_debt": 25.0,
        "beta": 0.50, "volatility_252d": 0.16, "dividend_yield": 0.035,
    },
    {
        "ticker": "PHIA.AS", "name": "Philips NV",
        "sector_gics": "Healthcare", "country": "NL",
        "market_cap": 25.0, "total_debt": 8.0,
        "beta": 1.10, "volatility_252d": 0.35, "dividend_yield": 0.015,
    },
    # ── Italy (4) ──
    {
        "ticker": "ISP.MI", "name": "Intesa Sanpaolo SpA",
        "sector_gics": "Financials", "country": "IT",
        "market_cap": 65.0, "total_debt": 80.0,
        "beta": 1.35, "volatility_252d": 0.30, "dividend_yield": 0.070,
    },
    {
        "ticker": "ENI.MI", "name": "ENI SpA",
        "sector_gics": "Energy", "country": "IT",
        "market_cap": 45.0, "total_debt": 30.0,
        "beta": 1.10, "volatility_252d": 0.28, "dividend_yield": 0.060,
    },
    {
        "ticker": "ENEL.MI", "name": "ENEL SpA",
        "sector_gics": "Utilities", "country": "IT",
        "market_cap": 68.0, "total_debt": 60.0,
        "beta": 0.65, "volatility_252d": 0.25, "dividend_yield": 0.060,
    },
    {
        "ticker": "UCG.MI", "name": "UniCredit SpA",
        "sector_gics": "Financials", "country": "IT",
        "market_cap": 60.0, "total_debt": 70.0,
        "beta": 1.40, "volatility_252d": 0.32, "dividend_yield": 0.055,
    },
    # ── Spain (4) ──
    {
        "ticker": "SAN.MC", "name": "Banco Santander SA",
        "sector_gics": "Financials", "country": "ES",
        "market_cap": 75.0, "total_debt": 90.0,
        "beta": 1.30, "volatility_252d": 0.30, "dividend_yield": 0.040,
    },
    {
        "ticker": "IBE.MC", "name": "Iberdrola SA",
        "sector_gics": "Utilities", "country": "ES",
        "market_cap": 85.0, "total_debt": 50.0,
        "beta": 0.55, "volatility_252d": 0.20, "dividend_yield": 0.042,
    },
    {
        "ticker": "ITX.MC", "name": "Inditex SA",
        "sector_gics": "Consumer", "country": "ES",
        "market_cap": 130.0, "total_debt": 5.0,
        "beta": 0.90, "volatility_252d": 0.25, "dividend_yield": 0.025,
    },
    {
        "ticker": "TEF.MC", "name": "Telefonica SA",
        "sector_gics": "Telecom", "country": "ES",
        "market_cap": 22.0, "total_debt": 38.0,
        "beta": 0.60, "volatility_252d": 0.22, "dividend_yield": 0.070,
    },
    # ── Nordic / Other (4) ──
    {
        "ticker": "NOVOB.CO", "name": "Novo Nordisk A/S",
        "sector_gics": "Healthcare", "country": "DK",
        "market_cap": 420.0, "total_debt": 12.0,
        "beta": 0.75, "volatility_252d": 0.30, "dividend_yield": 0.010,
    },
    {
        "ticker": "NESN.SW", "name": "Nestle SA",
        "sector_gics": "Consumer", "country": "CH",
        "market_cap": 250.0, "total_debt": 40.0,
        "beta": 0.55, "volatility_252d": 0.16, "dividend_yield": 0.030,
    },
    {
        "ticker": "NOVN.SW", "name": "Novartis AG",
        "sector_gics": "Healthcare", "country": "CH",
        "market_cap": 210.0, "total_debt": 28.0,
        "beta": 0.60, "volatility_252d": 0.18, "dividend_yield": 0.032,
    },
    {
        "ticker": "ABI.BR", "name": "AB InBev SA/NV",
        "sector_gics": "Consumer", "country": "BE",
        "market_cap": 110.0, "total_debt": 65.0,
        "beta": 0.85, "volatility_252d": 0.24, "dividend_yield": 0.010,
    },
]

# --- GICS Sector -> Average Beta (empirical EU large-cap) ---
SECTOR_BETA: Dict[str, float] = {
    "Technology": 1.30,
    "Financials": 1.20,
    "Industrials": 1.10,
    "Utilities": 0.60,
    "Healthcare": 0.80,
    "Consumer": 1.00,
    "Energy": 1.10,
    "Telecom": 0.70,
    "Materials": 1.15,
}

# --- GICS Sector -> Rate Sensitivity Multiplier ---
# Positive = rises hurt, negative = rises help (or neutral)
# Tech high-duration (long-dated cash flows); Utilities bond-proxy
# Financials benefit from rate rises (NIM expansion)
# Energy largely rate-indifferent (commodity-driven)
SECTOR_RATE_SENSITIVITY: Dict[str, float] = {
    "Technology": -1.50,     # Long-duration growth stocks — most rate-sensitive
    "Utilities": -0.30,      # Bond-proxy but regulated → moderate
    "Healthcare": -0.80,     # Moderate duration, some growth
    "Consumer": -0.60,       # Mixed, discretionary more than staples
    "Industrials": -0.50,    # Moderate, capex-sensitive
    "Energy": -0.20,         # Commodity-driven, low rate sensitivity
    "Telecom": -0.40,        # High debt but stable cash flows
    "Financials": 0.80,      # Banks benefit from rate rises (NIM)
    "Materials": -0.30,      # Commodity-driven, moderate
}

# --- GICS Sector -> Inflation Sensitivity ---
# Positive = inflation benefits (pricing power, real assets)
# Negative = inflation hurts (margin compression)
_SECTOR_INFLATION_SENSITIVITY: Dict[str, float] = {
    "Energy": 0.60,          # Commodity hedge, pricing power
    "Materials": 0.40,       # Commodity producers
    "Utilities": -0.20,      # Regulated tariffs lag inflation
    "Technology": -0.50,     # Long-duration → real discount rate rises
    "Healthcare": -0.30,     # Cost pressure but pricing power in pharma
    "Consumer": -0.40,       # Margin compression (staples less, discretionary more)
    "Industrials": 0.10,     # Mixed, capex can lag
    "Telecom": -0.20,        # Regulated, stable
    "Financials": 0.20,      # Rate rises often accompany inflation
}

# --- Merton Model Parameters ---
_RISK_FREE_RATE = 0.02            # EUR risk-free rate approximation
_TIME_HORIZON = 1.0               # 1-year PD horizon (IFRS 9 standard)

# KMV EDF mapping: PD = exp(a - b * DD)
# Pure norm.cdf(-DD) yields near-zero PD for large-caps (DD >> 3).
# Moody's KMV uses an empirical power-law mapping instead.
# Calibrated so that EAD-weighted mean PD ~ 2.0% on EU large-cap universe:
#   DD ~4 (leveraged financials) → PD ~8%
#   DD ~10 (median large-cap)    → PD ~1.8%
#   DD ~15 (low-debt tech)       → PD ~0.5%
_KMV_INTERCEPT = -1.60            # a: intercept of log-PD mapping
_KMV_SLOPE = 0.231                # b: slope (PD sensitivity to DD)

_PD_FLOOR = 0.0003                # CRR3 input floor
_PD_CAP = 0.20                    # Extreme cap for equities (higher than debt)
_PD_MERTON_BASE_CAP = 0.10       # Cap for base (unstressed) PD

# --- LGD Parameters (equity = first loss) ---
_LGD_MEAN = 0.85                  # Equity absorbs losses first (DMS 2023)
_LGD_FLOOR = 0.70                 # Minimum: some franchise value remains
_LGD_CAP = 1.00                   # Maximum: total loss possible
_LGD_STD = 0.06                   # Dispersion around mean

# --- RW CRR3 Art. 133 ---
_RW_LISTED_EQUITY = 1.00          # CRR3 Art. 133(2)(a): listed equity = 100%

# --- Stress Parameters (EBA 2023 calibration) ---
_BASE_GDP = 1.2                   # SCENARIO_BASE.gdp_growth
_BASE_UNEMP = 7.5                 # SCENARIO_BASE.unemployment_rate
_BASE_RATE = 3.5                  # SCENARIO_BASE.interest_rate
_BASE_INFLATION = 2.5             # SCENARIO_BASE.inflation_rate
_BASE_HPI = 2.0                   # SCENARIO_BASE.hpi_growth

# EBA 2023 severe: STOXX 600 → -40%, Tier 1 banks → -55%
_INDEX_SHOCK_GDP_COEFF = -2.50    # per pp of GDP deviation
_INDEX_SHOCK_UNEMP_COEFF = -0.80  # per pp of unemployment deviation

# Equity duration proxy (Dechow-Sloan 1997, PV of growth opportunities)
_DURATION_EQUITY_BASE = 15.0      # years (long-duration asset)

# GJR-GARCH vol expansion coefficient
_VOL_EXPANSION_COEFF = 0.30       # 30% vol increase per unit stress intensity

# Stress PD multiplier bounds
_PD_STRESS_MULT_FLOOR = 0.5       # PD can halve in best case
_PD_STRESS_MULT_CAP = 8.0         # PD can 8x in severe stress


# ──────────────────────────────────────────────
# DATA LOADING (real data priority)
# ──────────────────────────────────────────────

def load_equity_data(path: Optional[str] = None) -> pl.DataFrame:
    """Charge les donnees equities depuis le parquet pre-traite.

    Le parquet attendu contient des colonnes : ticker, name, sector_gics,
    country, market_cap, total_debt, beta, volatility_252d, dividend_yield.

    Args:
        path: Chemin vers le parquet. Si None, utilise le defaut
              ``data/stoxx_equities.parquet``.

    Returns:
        DataFrame avec donnees equities reelles.

    Raises:
        FileNotFoundError: Si le parquet n'existe pas.
    """
    if path is None:
        path = str(_DATA_DIR / "stoxx_equities.parquet")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"STOXX equities parquet not found: {p}")
    return pl.read_parquet(p)


def _build_fallback_equity_data() -> List[Dict]:
    """Return fallback EU_LARGE_CAPS constants when parquet is absent."""
    return [d.copy() for d in EU_LARGE_CAPS]


# ──────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────

def generate_equity_positions(
    n_positions: int = 100,
    total_ead: float = 2.0e9,
    seed: int = 123,
) -> pl.DataFrame:
    """Genere un portefeuille d'actions cotees position par position.

    Etape A : tente de charger les donnees reelles (STOXX parquet).
              Si absent, fallback aux constantes EU_LARGE_CAPS.
    Etape B : echantillonne n_positions parmi l'univers, avec poids
              proportionnels aux market caps (capitalisation-weighted).
    Etape C : calcule PD Merton et LGD pour chaque position.

    Le traitement est FVTPL : pas de staging IFRS 9, pas d'ECL.
    Le stress test produit un MTM loss au lieu d'un ECL.

    Args:
        n_positions: Nombre de positions a generer (~100).
        total_ead: EAD total du portefeuille actions (mark-to-market).
        seed: Graine aleatoire.

    Returns:
        DataFrame avec ~20 colonnes (une ligne par position).
    """
    rng = np.random.default_rng(seed)

    # ── Etape A: Load equity universe (real or fallback) ──
    equity_universe: List[Dict] = []
    try:
        df_real = load_equity_data()
        # Convert to list of dicts for uniform processing
        for row in df_real.iter_rows(named=True):
            equity_universe.append({
                "ticker": str(row.get("ticker", "UNK")),
                "name": str(row.get("name", "Unknown")),
                "sector_gics": str(row.get("sector_gics", "Industrials")),
                "country": str(row.get("country", "DE")),
                "market_cap": float(row.get("market_cap", 50.0)),
                "total_debt": float(row.get("total_debt", 20.0)),
                "beta": float(row.get("beta", 1.0)),
                "volatility_252d": float(row.get("volatility_252d", 0.25)),
                "dividend_yield": float(row.get("dividend_yield", 0.03)),
            })
    except FileNotFoundError:
        equity_universe = _build_fallback_equity_data()

    if len(equity_universe) == 0:
        equity_universe = _build_fallback_equity_data()

    # ── Etape B: Sample positions (market-cap weighted) ──
    n_universe = len(equity_universe)
    mcaps = np.array([eq["market_cap"] for eq in equity_universe], dtype=float)
    mcaps = np.maximum(mcaps, 1.0)  # floor to avoid zero weights
    weights_sampling = mcaps / mcaps.sum()

    sampled_idx = rng.choice(
        n_universe, size=n_positions, replace=True, p=weights_sampling,
    )

    # ── Build position records ──
    records = []
    for i in range(n_positions):
        eq = equity_universe[sampled_idx[i]]

        ticker = eq["ticker"]
        name = eq["name"]
        sector_gics = eq["sector_gics"]
        country = eq.get("country", "DE")

        # Market cap with noise (+/-15% to create position-level variation)
        mcap_noise = rng.normal(1.0, 0.15)
        mcap_noise = max(0.5, min(mcap_noise, 1.5))
        market_cap = eq["market_cap"] * mcap_noise

        # Total debt with noise (+/-10%)
        debt_noise = rng.normal(1.0, 0.10)
        debt_noise = max(0.5, min(debt_noise, 1.5))
        total_debt = max(0.1, eq["total_debt"] * debt_noise)

        # Beta: stock-specific with noise, constrained by sector average
        sector_beta = SECTOR_BETA.get(sector_gics, 1.0)
        beta = eq["beta"] + rng.normal(0.0, 0.08)
        beta = max(0.2, min(beta, 2.5))

        # Volatility: annualized 252d, with noise
        vol_base = eq["volatility_252d"]
        vol = vol_base * max(0.7, min(rng.normal(1.0, 0.10), 1.3))
        vol = max(0.08, min(vol, 0.60))

        # Dividend yield with noise
        div_yield = eq.get("dividend_yield", 0.03)
        div_yield = max(0.0, div_yield + rng.normal(0.0, 0.005))
        div_yield = min(div_yield, 0.12)

        # Position notional (lognormal, ~20M median)
        notional = rng.lognormal(np.log(20e6), 0.7)

        records.append({
            "position_id": i,
            "ticker": ticker,
            "name": name,
            "sector_gics": sector_gics,
            "country": country,
            "market_cap": round(market_cap, 2),
            "total_debt": round(total_debt, 2),
            "beta": round(beta, 4),
            "volatility": round(vol, 4),
            "dividend_yield": round(div_yield, 4),
            "notional": round(notional, 2),
            "rw_crr3": _RW_LISTED_EQUITY,
            "accounting_treatment": "fvtpl",
        })

    df = pl.DataFrame(records)

    # ── Scale notionals to total_ead ──
    raw_total = df["notional"].sum()
    if raw_total > 0:
        df = df.with_columns(
            (pl.col("notional") * (total_ead / raw_total)).alias("ead"),
        )
    else:
        df = df.with_columns(pl.lit(total_ead / n_positions).alias("ead"))

    # ── Compute PD (Merton) and LGD ──
    df = compute_equity_pd(df)
    df = compute_equity_lgd(df, rng)

    return df


# ──────────────────────────────────────────────
# PD MERTON (modele structurel)
# ──────────────────────────────────────────────

def _compute_distance_to_default(
    market_cap: np.ndarray,
    total_debt: np.ndarray,
    vol_equity: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Merton distance-to-default and stressed asset volatility.

    Internal helper shared by compute_equity_pd and stress_equity_positions.

    Args:
        market_cap: Array of equity market capitalizations (EUR bn).
        total_debt: Array of total debt (EUR bn).
        vol_equity: Array of annualized equity volatilities.

    Returns:
        Tuple (dd, sigma_asset) — distance-to-default and de-leveraged vol.
    """
    # Asset value (Merton: V = E + D)
    asset_value = market_cap + total_debt

    # De-leveraged asset volatility (Ito's lemma approximation)
    # sigma_A = sigma_E * E / V (simplified, ignores option curvature)
    leverage_ratio = np.clip(market_cap / np.maximum(asset_value, 1.0), 0.05, 1.0)
    sigma_asset = vol_equity * leverage_ratio

    # Ensure minimum vol for numerical stability
    sigma_asset = np.maximum(sigma_asset, 0.01)

    # Distance to Default
    # DD = (ln(V/D) + (r - sigma^2/2) * T) / (sigma * sqrt(T))
    r = _RISK_FREE_RATE
    t = _TIME_HORIZON

    # Default point: total_debt (senior + junior claims)
    # Floor debt at 1% of asset value to avoid log(inf)
    default_point = np.maximum(total_debt, asset_value * 0.01)

    numerator = (
        np.log(asset_value / default_point)
        + (r - 0.5 * sigma_asset**2) * t
    )
    denominator = sigma_asset * np.sqrt(t)
    dd = numerator / np.maximum(denominator, 0.001)

    return dd, sigma_asset


def compute_equity_pd(df: pl.DataFrame) -> pl.DataFrame:
    """Calcule la PD Merton pour chaque position action.

    Modele structurel de Merton (1974) :
        V = market_cap + total_debt  (asset value)
        sigma_asset = volatility x market_cap / V  (de-leverage Ito)
        DD = (ln(V / D) + (r - sigma^2/2) * T) / (sigma * sqrt(T))

    Mapping DD → EDF (Moody's KMV empirical) :
        PD = exp(_KMV_INTERCEPT - _KMV_SLOPE * DD)

    Le mapping exponentiel remplace le theorique Phi(-DD) qui donne des
    PD quasi-nulles pour les large-caps europeennes (DD >> 3). Le mapping
    est calibre pour produire une PD moyenne EAD-ponderee ~ 2.0%
    (coherent avec pd_base=0.020 dans config).

    Args:
        df: DataFrame with market_cap, total_debt, volatility columns.

    Returns:
        DataFrame with pd_merton column added.
    """
    market_cap = df["market_cap"].to_numpy().astype(float)
    total_debt = df["total_debt"].to_numpy().astype(float)
    vol_equity = df["volatility"].to_numpy().astype(float)

    dd, _ = _compute_distance_to_default(market_cap, total_debt, vol_equity)

    # KMV empirical EDF mapping: PD = exp(a - b * DD)
    pd_raw = np.exp(_KMV_INTERCEPT - _KMV_SLOPE * dd)

    # Clip to [floor, cap]
    pd_merton = np.clip(pd_raw, _PD_FLOOR, _PD_MERTON_BASE_CAP)

    return df.with_columns(pl.Series("pd_merton", np.round(pd_merton, 6)))


# ──────────────────────────────────────────────
# LGD (equity = first loss, very high)
# ──────────────────────────────────────────────

def compute_equity_lgd(
    df: pl.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> pl.DataFrame:
    """Calcule la LGD pour chaque position action.

    Actions = premiere tranche de perte (residual claim apres dette).
    En cas de defaut de l'emetteur, les actionnaires perdent presque tout.

    Calibration :
        - DMS Yearbook (2023) : mean equity loss in default ~ 85%
        - Recovery rate residuelle ~ 15% (franchise value, breakup)
        - Dispersion : secteur defensif (Utilities, Healthcare) → LGD plus basse
        - Variance intra-position : sigma = 0.06

    Args:
        df: DataFrame with sector_gics column.
        rng: Optional random generator for noise.

    Returns:
        DataFrame with lgd column added.
    """
    n = len(df)
    sectors = df["sector_gics"].to_numpy()

    # Sector-specific base LGD (franchise value differences)
    _SECTOR_LGD_OFFSET: Dict[str, float] = {
        "Technology": -0.02,    # IP / brand value
        "Healthcare": -0.03,    # Patent portfolio, pipeline value
        "Consumer": -0.01,      # Brand equity
        "Utilities": -0.05,     # Regulated assets, always worth something
        "Industrials": 0.00,    # Average
        "Financials": 0.03,     # Pro-cyclical, counterparty losses amplify
        "Energy": 0.02,         # Stranded asset risk
        "Telecom": -0.02,       # Spectrum / infrastructure
        "Materials": 0.01,      # Commodity cycle
    }

    lgd = np.full(n, _LGD_MEAN, dtype=float)
    for i in range(n):
        sector = sectors[i]
        lgd[i] += _SECTOR_LGD_OFFSET.get(sector, 0.0)

    # Add noise
    if rng is not None:
        lgd += rng.normal(0.0, _LGD_STD, n)

    lgd = np.clip(lgd, _LGD_FLOOR, _LGD_CAP)

    return df.with_columns(pl.Series("lgd", np.round(lgd, 4)))


# ──────────────────────────────────────────────
# STRESS TEST (4 canaux, ~5ms, position-par-position)
# ──────────────────────────────────────────────

def stress_equity_positions(
    df: pl.DataFrame,
    macro_params: Dict[str, float],
) -> Dict[str, float]:
    """Stress test des positions actions et re-aggregation.

    4 canaux de transmission (non-lineaires, asymetriques) :

    1. Index shock (EBA 2023 calibration) :
       haircut_i = beta_i * (-2.5 * delta_gdp - 0.8 * delta_unemp)
       Captures systematic equity risk via CAPM beta.
       EBA 2023 severe: STOXX 600 ~ -40% (beta=1.0).

    2. Rate sensitivity (sector-differentiated, duration equity) :
       rate_effect_i = -duration_equity * delta_rate * sector_mult_i
       Tech stocks (long-duration) suffer most from rate rises.
       Financials benefit (NIM expansion).

    3. Sector rotation (inflation / idiosyncratic) :
       GICS-specific: Energy +inflation, Tech -rates.
       Captures relative sector performance under regime shifts.

    4. Vol expansion (GJR-GARCH leverage effect) :
       vol_stressed_i = vol_base_i * (1 + 0.30 * stress_intensity)
       Asymmetric: downside vol expands more than upside contracts.
       Affects PD via Merton model (higher vol → lower DD → higher PD).

    FVTPL : no ECL — computes MTM loss instead.

    Args:
        df: DataFrame from generate_equity_positions().
        macro_params: Dict with gdp_growth, unemployment_rate, interest_rate,
            inflation_rate, hpi_growth.

    Returns:
        Dict with pd_base, lgd_base, rw_crr3 (EAD-weighted) + mtm_loss.
    """
    # Macro deltas (convert pp to fraction for coefficient compatibility)
    delta_gdp = (macro_params.get("gdp_growth", _BASE_GDP) - _BASE_GDP) / 100.0
    delta_unemp = (macro_params.get("unemployment_rate", _BASE_UNEMP) - _BASE_UNEMP) / 100.0
    delta_rate = (macro_params.get("interest_rate", _BASE_RATE) - _BASE_RATE) / 100.0
    delta_inflation = (macro_params.get("inflation_rate", _BASE_INFLATION) - _BASE_INFLATION) / 100.0
    delta_hpi = (macro_params.get("hpi_growth", _BASE_HPI) - _BASE_HPI) / 100.0

    # Stress intensity (scalar): magnitude of macro deviation
    # Used for GJR-GARCH vol expansion
    stress_intensity = np.sqrt(
        delta_gdp**2 + delta_unemp**2 + delta_rate**2
        + delta_inflation**2 + delta_hpi**2
    )
    # Normalize to [0, ~3] range (severe stress ~2.5, base=0)
    stress_intensity = min(stress_intensity, 3.0)

    n = len(df)
    ead = df["ead"].to_numpy()
    betas = df["beta"].to_numpy()
    vols_base = df["volatility"].to_numpy()
    sectors = df["sector_gics"].to_numpy()
    market_caps = df["market_cap"].to_numpy()
    total_debts = df["total_debt"].to_numpy()

    # ── Canal 1: Index Shock (EBA-calibrated) ──
    # haircut_i = beta_i * (-2.5 * delta_gdp - 0.8 * delta_unemp)
    # Negative delta_gdp (recession) → positive haircut (loss)
    index_haircut = betas * (
        _INDEX_SHOCK_GDP_COEFF * delta_gdp
        + _INDEX_SHOCK_UNEMP_COEFF * delta_unemp
    )

    # ── Canal 2: Rate Sensitivity (duration × sector) ──
    sector_rate_mult = np.array([
        SECTOR_RATE_SENSITIVITY.get(s, -0.50) for s in sectors
    ], dtype=float)

    # Duration of equity differs by sector (growth vs value)
    # Growth stocks (Tech): higher duration → more rate-sensitive
    # Value stocks (Financials): lower effective duration, benefit from rates
    rate_effect = -_DURATION_EQUITY_BASE * delta_rate * sector_rate_mult / 100.0

    # ── Canal 3: Sector Rotation (inflation / macro regime) ──
    sector_inflation_mult = np.array([
        _SECTOR_INFLATION_SENSITIVITY.get(s, 0.0) for s in sectors
    ], dtype=float)

    # Inflation effect: Energy/Materials benefit, Tech/Consumer suffer
    inflation_effect = delta_inflation * sector_inflation_mult * 0.5

    # HPI effect (limited to Real Estate proxies / Financials)
    _SECTOR_HPI_SENSITIVITY: Dict[str, float] = {
        "Financials": 0.30,   # Mortgage-backed → HPI matters
        "Consumer": 0.10,     # Wealth effect
        "Utilities": 0.05,    # Asset base
    }
    hpi_effect = np.array([
        delta_hpi * _SECTOR_HPI_SENSITIVITY.get(s, 0.0) for s in sectors
    ], dtype=float)

    sector_rotation = inflation_effect + hpi_effect

    # ── Canal 4: Vol Expansion / Contraction (GJR-GARCH) ──
    # Asymmetric leverage effect (Glosten-Jagannathan-Runkle 1993):
    #   - Adverse shocks expand vol (leverage effect, panic, margin calls)
    #   - Favorable shocks contract vol (calm markets, low VIX)
    # Detect regime: adverse = recession (GDP down or unemployment up)
    is_adverse = (delta_gdp < 0) or (delta_unemp > 0)
    if is_adverse:
        # Vol expands: GJR leverage effect × 1.5
        vol_multiplier = 1.0 + _VOL_EXPANSION_COEFF * stress_intensity * 1.5
    else:
        # Vol contracts: calmer markets (VSTOXX mean-reversion to ~15%)
        vol_multiplier = 1.0 - _VOL_EXPANSION_COEFF * stress_intensity * 0.6

    vol_stressed = vols_base * np.clip(vol_multiplier, 0.5, 3.0)

    # ── Combine MTM haircut (4 channels) ──
    # Total haircut per position (additive across channels, bounded)
    total_haircut = index_haircut + rate_effect + sector_rotation
    # Clip total return to [-0.80, +0.60] (cannot lose more than 80% in one year)
    total_haircut = np.clip(total_haircut, -0.60, 0.80)

    # ── Recompute Stressed PD via Merton DD + KMV mapping ──
    # Stressed market cap (MTM haircut reduces equity value)
    market_cap_stressed = market_caps * (1.0 - total_haircut)
    market_cap_stressed = np.maximum(market_cap_stressed, market_caps * 0.05)

    # Recompute DD with stressed equity value and expanded vol
    dd_stressed, _ = _compute_distance_to_default(
        market_cap_stressed, total_debts, vol_stressed,
    )

    # KMV EDF mapping (same as compute_equity_pd)
    pd_stressed = np.exp(_KMV_INTERCEPT - _KMV_SLOPE * dd_stressed)
    pd_stressed = np.clip(pd_stressed, _PD_FLOOR, _PD_CAP)

    # ── LGD stressed (slightly higher in adverse scenarios) ──
    lgd_base_arr = df["lgd"].to_numpy()
    lgd_addon = 0.0
    if is_adverse:
        # In recessions, recovery rates are lower (fire-sale discounts)
        lgd_addon = min(0.10, stress_intensity * 0.03)
    lgd_stressed = np.clip(lgd_base_arr + lgd_addon, _LGD_FLOOR, _LGD_CAP)

    # ── RW always 100% for listed equity ──
    rw_stressed = _RW_LISTED_EQUITY

    # ── MTM loss (FVTPL: loss goes straight to P&L) ──
    # Position MTM loss = EAD × haircut (positive = loss)
    mtm_loss_per_position = ead * np.maximum(total_haircut, 0.0)
    total_mtm_loss = float(mtm_loss_per_position.sum())

    # ── EAD-weighted aggregation ──
    total_ead = ead.sum()
    if total_ead <= 0:
        return {
            "pd_base": 0.020,
            "lgd_base": 0.85,
            "rw_crr3": _RW_LISTED_EQUITY,
            "mtm_loss": 0.0,
        }

    w = ead / total_ead
    pd_agg = float(np.dot(w, pd_stressed))
    lgd_agg = float(np.dot(w, lgd_stressed))

    return {
        "pd_base": round(pd_agg, 6),
        "lgd_base": round(lgd_agg, 6),
        "rw_crr3": round(rw_stressed, 4),
        "mtm_loss": round(total_mtm_loss, 2),
    }


# ──────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────

def aggregate_equity_to_balance_row(
    df_positions: pl.DataFrame,
    profile: "AssetClassProfile",
) -> dict:
    """Agrege les positions actions en une ligne balance sheet.

    Pattern identique aux autres generateurs : EAD-weighted PD/LGD.
    RW = 1.00 toujours (CRR3 Art. 133 listed equity).

    Pour FVTPL : pas d'ECL en staging IFRS 9. Le champ mtm_loss additionnel
    est calcule comme la somme des pertes mark-to-market positives.

    Args:
        df_positions: DataFrame from generate_equity_positions().
        profile: AssetClassProfile for equities.

    Returns:
        Dict avec toutes les colonnes balance sheet (19+ cles) + mtm_loss.
    """
    weights = df_positions["ead"].to_numpy()
    total_ead = weights.sum()

    if total_ead <= 0:
        pd_agg = profile.pd_base
        lgd_agg = profile.lgd_base
        rw_agg = profile.rw_crr3
    else:
        w = weights / total_ead
        pd_agg = float(np.dot(w, df_positions["pd_merton"].to_numpy()))
        lgd_agg = float(np.dot(w, df_positions["lgd"].to_numpy()))
        rw_agg = _RW_LISTED_EQUITY  # always 100%

    # Base MTM loss: zero in base scenario (no stress)
    mtm_loss = 0.0

    return {
        "asset_class": profile.name,
        "label": profile.label,
        "category": profile.category,
        "ead_total": round(total_ead, 2),
        "typical_weight": profile.typical_weight,
        "pd_base": round(pd_agg, 6),
        "lgd_base": round(lgd_agg, 6),
        "tenor": 0.0,  # equity = perpetual
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
        "mtm_loss": round(mtm_loss, 2),
    }
