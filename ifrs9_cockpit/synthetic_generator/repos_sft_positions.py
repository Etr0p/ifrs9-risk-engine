"""Generateur position-par-position pour le portefeuille Repos / SFT.

Genere ~300 transactions repo (Securities Financing Transactions)
avec PD contrepartie, LGD collateral-driven, RW CRR3 Art. 222-224
(CRM supervisory haircuts), et stress test 3 canaux
(haircut spiral, counterparty contagion, rollover risk).

Le pipeline aval (ECL Vasicek, comparator RAROC, BL-CVaR) consomme alors
des valeurs calculees bottom-up plutot que des constantes parametriques.

Calibration (ICMA European Repo Survey #48, Dec 2024) :
    - Encours EUR 10.4T, 80% govt collateral, 35% overnight
    - Haircuts : 2% govt, 5% corp IG, 15% equity, 10% other
    - Tenor : 35% O/N, 20% 1W, 25% 1M, 15% 3M, 5% >3M
    - 25 contreparties EU (memes que interbank — meme infrastructure)

Calibration stress (Brunnermeier & Pedersen, RFS 2009) :
    - Haircut spiral : funding liquidity + market liquidity doom loop
    - Govt repos stables sous stress (Duffie 2010, safe haven)
    - Non-govt repos : haircuts spiralent (Gorton & Metrick AER 2012)
    - ABX index (2007-2009) : haircuts 0% -> 60% en 18 mois
    - LTCM (1998) : rollover freeze sur non-govt, govt epargne

References :
    - CRR3 Art. 222 : CRM — comprehensive method, supervisory haircuts
    - CRR3 Art. 223 : Conditions d'eligibilite collatéral financier
    - CRR3 Art. 224 : Supervisory volatility adjustments (Table 2)
    - CRR3 Art. 220(2) : Repo eligible pour master netting (E* formula)
    - SFTR (EU 2015/2365) : reporting obligations SFT
    - Basel III LCR : reverse repos HQLA eligible (0-50% haircut)
    - Basel III NSFR : RSF 0-15% par tenor
    - GMRA 2011 : Global Master Repurchase Agreement (ICMA)
    - Brunnermeier & Pedersen (RFS 2009) : Market/funding liquidity spiral
    - Gorton & Metrick (AER 2012) : Securitized banking and the run on repo
    - Duffie (2010) : How big banks fail and what to do about it
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import polars as pl


# ──────────────────────────────────────────────
# CONSTANTES (ICMA European Repo Survey #48, Dec 2024)
# ──────────────────────────────────────────────

# --- Collateral mix (ICMA #48 : govt 80%, corp IG 10%, equity 7%, other 3%) ---
COLLATERAL_MIX: Dict[str, float] = {
    "govt": 0.80,
    "corp_ig": 0.10,
    "equity": 0.07,
    "other": 0.03,
}

# --- Tenor mix (ICMA #48 : ON dominant, decreasing) ---
TENOR_MIX: Dict[str, float] = {
    "ON": 0.35,
    "1W": 0.20,
    "1M": 0.25,
    "3M": 0.15,
    "GT3M": 0.05,
}

# --- Tenor in days (for maturity calculations) ---
TENOR_DAYS: Dict[str, int] = {
    "ON": 1,
    "1W": 7,
    "1M": 30,
    "3M": 90,
    "GT3M": 180,
}

# --- Tenor in years (for annualized calculations) ---
TENOR_YEARS: Dict[str, float] = {
    "ON": 1.0 / 365.0,
    "1W": 7.0 / 365.0,
    "1M": 30.0 / 365.0,
    "3M": 90.0 / 365.0,
    "GT3M": 180.0 / 365.0,
}

# --- Base haircuts (CRR3 Art. 224, Table 2, residual maturity <=1y) ---
BASE_HAIRCUTS: Dict[str, float] = {
    "govt": 0.02,
    "corp_ig": 0.05,
    "equity": 0.15,
    "other": 0.10,
}

# --- Post-CRM Risk Weights by collateral type ---
# CRR3 Art. 222-224 comprehensive method:
#   E* = max(0, E - C_va) where C_va = C * (1 - H_c - H_fx)
#   Govt collateral: E* ~ 0 -> RW ~ 0%
#   Corp IG: residual exposure after CRM -> 8%
#   Equity: higher haircuts -> residual 15%
#   Other: mixed -> 12%
COLLATERAL_RW: Dict[str, float] = {
    "govt": 0.00,
    "corp_ig": 0.08,
    "equity": 0.15,
    "other": 0.12,
}

# --- 25 EU bank counterparties (same as interbank module, EBA 2024) ---
REPO_COUNTERPARTIES: Dict[str, Dict] = {
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

# --- Overcollateralization parameters ---
# GMRA 2011 standard : initial margin = 1 + haircut
# Typical OC = 2-5% above haircut (ICMA #48 survey)
_OC_BASE: Dict[str, float] = {
    "govt": 0.02,
    "corp_ig": 0.03,
    "equity": 0.05,
    "other": 0.04,
}

# --- Median notional by tenor (EUR, ICMA #48) ---
_NOTIONAL_BY_TENOR: Dict[str, float] = {
    "ON": 50e6,
    "1W": 40e6,
    "1M": 60e6,
    "3M": 80e6,
    "GT3M": 100e6,
}

# --- Repo rate approximation (EUR-STR based, Feb 2026) ---
# Govt repo: EUR-STR minus ~ 5-10bp (special collateral premium)
# Non-govt: EUR-STR plus spread
_REPO_RATE_BASE: Dict[str, float] = {
    "govt": 0.0360,      # slightly below EUR-STR (GC premium)
    "corp_ig": 0.0380,   # EUR-STR + 15bp
    "equity": 0.0400,    # EUR-STR + 35bp
    "other": 0.0390,     # EUR-STR + 25bp
}

# --- PD parameters ---
_PD_FLOOR = 0.0003       # 3bp CRR3 minimum
_PD_CAP = 0.02

# --- LGD parameters ---
_LGD_FLOOR = 0.005       # 50bp (repo is collateralized)
_LGD_FLOOR_CRR3 = 0.05   # 5% CRR3 floor for collateralized exposures
_LGD_CAP = 0.45
_LGD_NOISE_STD = 0.005

# --- Stress parameters ---
_BASE_GDP = 1.2
_BASE_RATE = 3.5
_LGD_DOWNTURN_COEFF = 0.008   # +0.8pp LGD per -1% GDP (lighter than unsecured)
_LGD_DOWNTURN_CAP = 0.06      # max +6pp

# --- Haircut spiral parameters (Brunnermeier & Pedersen 2009) ---
_HAIRCUT_SPIRAL_GOVT = 0.5       # govt repos: spiral attenuated (safe haven)
_HAIRCUT_SPIRAL_NON_GOVT = 3.0   # non-govt: full spiral amplification
_HAIRCUT_SPIRAL_CAP_GOVT = 0.06  # govt haircuts max 6% even in GFC
_HAIRCUT_SPIRAL_CAP_OTHER = 0.60 # ABX-type collateral haircuts capped at 60%

# --- Data directory (for consistency, even if not used) ---
_DATA_DIR = Path(__file__).parent.parent / "data"


# ──────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────

def generate_repo_positions(
    n_positions: int = 300,
    total_ead: float = 2.0e9,
    seed: int = 123,
) -> pl.DataFrame:
    """Genere un portefeuille de transactions repo position par position.

    Parametric generation (ICMA-calibrated constants, no parquet needed).
    Samples counterparty (25 EU banks), collateral type, tenor bucket,
    then computes PD, LGD, RW for each position.

    Args:
        n_positions: Nombre de positions a generer (~300).
        total_ead: EAD total du portefeuille repo.
        seed: Graine aleatoire.

    Returns:
        DataFrame avec ~18 colonnes (une ligne par position).
    """
    rng = np.random.default_rng(seed)

    # ── Sample counterparties ──
    cp_codes = list(REPO_COUNTERPARTIES.keys())
    cp_weights = np.array([REPO_COUNTERPARTIES[c]["weight"] for c in cp_codes])
    cp_weights = cp_weights / cp_weights.sum()
    sampled_cps = rng.choice(cp_codes, size=n_positions, p=cp_weights)

    # ── Sample collateral types ──
    coll_names = list(COLLATERAL_MIX.keys())
    coll_weights = np.array([COLLATERAL_MIX[c] for c in coll_names])
    coll_weights = coll_weights / coll_weights.sum()
    sampled_collaterals = rng.choice(coll_names, size=n_positions, p=coll_weights)

    # ── Sample tenor buckets ──
    tenor_names = list(TENOR_MIX.keys())
    tenor_weights = np.array([TENOR_MIX[t] for t in tenor_names])
    tenor_weights = tenor_weights / tenor_weights.sum()
    sampled_tenors = rng.choice(tenor_names, size=n_positions, p=tenor_weights)

    # ── Generate individual positions ──
    records = []
    for i in range(n_positions):
        cp_code = sampled_cps[i]
        cp = REPO_COUNTERPARTIES[cp_code]
        coll_type = sampled_collaterals[i]
        tenor_name = sampled_tenors[i]

        tenor_days = TENOR_DAYS[tenor_name]
        tenor_years = TENOR_YEARS[tenor_name]

        # Base haircut with slight noise
        base_h = BASE_HAIRCUTS[coll_type]
        haircut = float(np.clip(
            rng.normal(base_h, base_h * 0.15),
            base_h * 0.5,
            base_h * 2.0,
        ))

        # Overcollateralization (above haircut level)
        oc_base = _OC_BASE[coll_type]
        overcollateralization = float(np.clip(
            rng.normal(oc_base, oc_base * 0.3),
            0.005,
            0.10,
        ))

        # Notional (lognormal, scaled by tenor)
        median_notional = _NOTIONAL_BY_TENOR.get(tenor_name, 60e6)
        notional = rng.lognormal(np.log(median_notional), 0.5)

        # Repo rate (collateral-dependent)
        base_rate = _REPO_RATE_BASE.get(coll_type, 0.0370)
        # Counterparty spread (lower quality counterparty pays more)
        cp_spread = cp["pd_base"] * 3.0
        # Tenor premium (longer repos pay more)
        tenor_premium = tenor_years * 0.002
        repo_rate = base_rate + cp_spread + tenor_premium

        records.append({
            "position_id": i,
            "counterparty_code": cp_code,
            "counterparty_name": cp["name"],
            "country": cp["country"],
            "rating": cp["rating"],
            "tier1_ratio": cp["tier1"],
            "collateral_type": coll_type,
            "tenor": tenor_name,
            "tenor_days": tenor_days,
            "tenor_years": round(tenor_years, 6),
            "notional": round(notional, 2),
            "haircut": round(haircut, 6),
            "overcollateralization": round(overcollateralization, 6),
            "repo_rate": round(repo_rate, 6),
            "pd_base": cp["pd_base"],
            "lgd_base": 0.0,     # placeholder, computed below
            "rw_crr3": 0.0,      # placeholder, computed below
            "ead": 0.0,          # placeholder, computed below
            "hqla_eligible": coll_type == "govt",
            "hqla_level": 1 if coll_type == "govt" else (2 if coll_type == "corp_ig" else 0),
            "exempt_from_staging": False,
        })

    df = pl.DataFrame(records)

    # ── Scale notionals to total_ead ──
    raw_total = df["notional"].sum()
    if raw_total > 0:
        df = df.with_columns(
            (pl.col("notional") * (total_ead / raw_total)).alias("ead"),
        )
    else:
        df = df.with_columns(
            pl.lit(total_ead / n_positions).alias("ead"),
        )

    # ── Compute PD, LGD, RW ──
    df = df.with_columns(pl.Series("pd_position", compute_repo_pd(df)))
    df = df.with_columns(pl.Series("lgd_position", compute_repo_lgd(df, rng)))
    df = df.with_columns(pl.Series("rw_crr3", compute_repo_rw(df)))

    return df


# ──────────────────────────────────────────────
# PD CONTREPARTIE (counterparty default risk)
# ──────────────────────────────────────────────

def compute_repo_pd(df: pl.DataFrame) -> np.ndarray:
    """Calcule la PD contrepartie pour chaque position repo.

    Pour un repo, le risque de credit est la PD de la contrepartie
    (pas du collateral — le collateral est detenu en garantie).
    La PD est ajustee par :
        1. PD rating (S&P, meme base que interbank)
        2. Alpha tenor (court terme = PD quasi-nulle, conforme CRR3)
        3. Alpha collateral (meilleur collatéral => contrepartie plus fiable)

    La PD minimale est 3bp (CRR3 floor).

    Args:
        df: DataFrame from generate_repo_positions().

    Returns:
        Array de PD par position.
    """
    n = len(df)
    pd_result = np.zeros(n)

    pd_base_arr = df["pd_base"].to_numpy()
    tenor_years_arr = df["tenor_years"].to_numpy()
    coll_types = df["collateral_type"].to_numpy()

    # Collateral quality -> counterparty quality proxy
    # Dealers posting govt collateral tend to be higher quality
    _ALPHA_COLLATERAL: Dict[str, float] = {
        "govt": 0.7,        # best quality counterparties
        "corp_ig": 0.9,     # good quality
        "equity": 1.2,      # higher risk (margin lending)
        "other": 1.1,       # mixed
    }

    for i in range(n):
        pd_base = pd_base_arr[i]
        tenor_years = tenor_years_arr[i]
        coll_type = coll_types[i]

        # 1. Tenor adjustment: overnight PD is negligible, 3M+ pays more
        # PD annualisee * tenor_fraction (maturity-scaled PD)
        # But we floor at a minimum — even O/N has settlement risk
        alpha_tenor = max(0.10, min(1.0, tenor_years / 0.25))

        # 2. Collateral quality proxy
        alpha_coll = _ALPHA_COLLATERAL.get(coll_type, 1.0)

        # 3. Combined PD (counterparty PD, tenor-adjusted)
        pd_result[i] = pd_base * alpha_tenor * alpha_coll

    return np.clip(pd_result, _PD_FLOOR, _PD_CAP)


# ──────────────────────────────────────────────
# LGD COLLATERAL-DRIVEN
# ──────────────────────────────────────────────

def compute_repo_lgd(
    df: pl.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Calcule la LGD pour chaque position repo.

    LGD = max(0, 1 - collateral_value*(1-haircut) / exposure) + noise

    Pour un repo, la LGD est tres faible car le collatéral est
    detenu en garantie avec haircut et overcollateralisation :
        - Govt collateral : LGD ~2% (near-zero loss)
        - Corp IG : LGD ~5% (small residual gap risk)
        - Equity : LGD ~12% (volatility + liquidation discount)
        - Other : LGD ~8% (mixed collateral)

    Floor CRR3 : 5% pour expositions collateralisees (Art. 161(1d)).
    Mais certains repos masters nettés ont floor inférieur (Art. 220(2)).

    La formule est :
        collateral_value = (1 + overcollateralization) = GMRA initial margin
        effective_coverage = collateral_value * (1 - haircut) / exposure
        lgd_residual = max(0, 1 - effective_coverage)

    Args:
        df: DataFrame from generate_repo_positions().
        rng: Optional random generator for noise.

    Returns:
        Array de LGD par position.
    """
    n = len(df)
    lgd = np.zeros(n)

    haircuts = df["haircut"].to_numpy()
    oc_arr = df["overcollateralization"].to_numpy()
    coll_types = df["collateral_type"].to_numpy()

    # Target LGD by collateral type (calibration ICMA/EBA)
    _TARGET_LGD: Dict[str, float] = {
        "govt": 0.02,
        "corp_ig": 0.05,
        "equity": 0.12,
        "other": 0.08,
    }

    for i in range(n):
        haircut = haircuts[i]
        oc = oc_arr[i]
        coll_type = coll_types[i]

        # Collateral coverage: (1 + OC) * (1 - haircut) per unit of exposure
        # Exposure = 1.0 (normalized), collateral = 1 + OC
        collateral_value = 1.0 + oc
        effective_coverage = collateral_value * (1.0 - haircut)

        # Residual LGD = shortfall after collateral liquidation
        lgd_residual = max(0.0, 1.0 - effective_coverage)

        # Add a gap risk component (price moves during closeout period)
        # Calibrated to match target LGD per collateral type
        target = _TARGET_LGD.get(coll_type, 0.05)
        gap_risk = max(0.0, target - lgd_residual)

        lgd[i] = lgd_residual + gap_risk

    # Add noise
    if rng is not None:
        lgd += rng.normal(0.0, _LGD_NOISE_STD, n)

    # Floor: repos eligible under GMRA master netting can have sub-5% LGD
    # but we still apply a 0.5% absolute floor (settlement risk)
    return np.clip(lgd, _LGD_FLOOR, _LGD_CAP)


# ──────────────────────────────────────────────
# RW CRR3 ART. 222-224 (CRM SUPERVISORY HAIRCUTS)
# ──────────────────────────────────────────────

def compute_repo_rw(df: pl.DataFrame) -> np.ndarray:
    """Calcule le Risk Weight CRR3 Art. 222-224 pour chaque position repo.

    CRM comprehensive method (Art. 222) :
        E* = max(0, E × (1 + He) - C × (1 - Hc - Hfx))

    Where:
        E = exposure value, He = volatility adjustment for exposure
        C = collateral value, Hc = volatility adjustment for collateral
        Hfx = FX mismatch haircut (0 for same currency)

    For repos under master netting (GMRA) :
        - Govt collateral : E* ~ 0 -> RW ~ 0%
        - Corp IG : residual E* -> RW 8%
        - Equity : higher E* -> RW 15%
        - Other : mixed -> RW 12%

    Additional adjustments :
        - Art. 224 Table 2 : supervisory haircuts by residual maturity
        - Short-term (<= 90 days) : reduced haircuts
        - Master netting benefit : further 50% reduction (Art. 220(2))

    Args:
        df: DataFrame from generate_repo_positions().

    Returns:
        Array de RW par position.
    """
    n = len(df)
    rw = np.zeros(n)

    coll_types = df["collateral_type"].to_numpy()
    tenor_days_arr = df["tenor_days"].to_numpy()
    haircuts = df["haircut"].to_numpy()
    oc_arr = df["overcollateralization"].to_numpy()

    for i in range(n):
        coll_type = coll_types[i]
        tenor_days = tenor_days_arr[i]
        haircut = haircuts[i]
        oc = oc_arr[i]

        # Base RW from collateral type (post-CRM)
        base_rw = COLLATERAL_RW.get(coll_type, 0.12)

        # Art. 224 short-term adjustment: repos <= 90 days benefit
        # from reduced supervisory haircuts -> lower E*
        if tenor_days <= 90:
            base_rw = base_rw * 0.70  # 30% reduction for short-term

        # Master netting benefit (Art. 220(2)):
        # For repos under GMRA with daily margining -> 50% further reduction
        # We assume all positions benefit from master netting
        netting_benefit = 0.50
        base_rw = base_rw * netting_benefit

        # Overcollateralization further reduces residual RW
        # Higher OC -> lower E* -> lower RW
        oc_reduction = max(0.0, 1.0 - oc * 5.0)
        base_rw = base_rw * max(0.10, oc_reduction)

        rw[i] = base_rw

    return np.clip(rw, 0.0, 1.50)


# ──────────────────────────────────────────────
# STRESS TEST (3 canaux, Brunnermeier-Pedersen calibre)
# ──────────────────────────────────────────────

def stress_repo_positions(
    df: pl.DataFrame,
    macro_params: Dict[str, float],
) -> Dict[str, float]:
    """Stress test des positions repo et re-aggregation.

    3 canaux de transmission :

    1. Haircut spiral (Brunnermeier & Pedersen RFS 2009) :
        H_stressed = H_base * (1 + gamma * max(0, -delta_gdp))
        - Govt repos : gamma = 0.5 (safe haven, Duffie 2010)
        - Non-govt repos : gamma = 3.0 (full spiral, Gorton-Metrick 2012)
        - ABX 2007-09 : haircuts 0% -> 60% en 18 mois
        - Impact LGD : higher haircuts -> lower collateral coverage -> higher LGD

    2. Counterparty contagion (Eisenberg-Noe 2001) :
        PD_stressed = PD_base * stress_mult
        - stress_mult = f(delta_gdp, delta_rate) — same as interbank
        - CRR3 Art. 153(3) LFI multiplier applies

    3. Rollover risk (maturity mismatch) :
        rollover_stress = tenor_mismatch * stress_intensity
        - Short-term repos face rollover freeze under stress
        - Overnight repos : highest rollover risk (daily roll)
        - Long-term repos : lower rollover risk but higher rate risk
        - LTCM 1998, Bear Stearns 2008 : repo rollover freeze

    LGD downturn : recalculated from stressed haircuts (channel 1 flows
    through to LGD mechanically via reduced coverage ratio).

    Args:
        df: DataFrame from generate_repo_positions().
        macro_params: Dict avec gdp_growth, interest_rate, etc.

    Returns:
        Dict avec pd_base, lgd_base, rw_crr3 stresses (EAD-weighted).
    """
    # Deltas (percentage points)
    delta_gdp = macro_params.get("gdp_growth", _BASE_GDP) - _BASE_GDP
    delta_rate = macro_params.get("interest_rate", _BASE_RATE) - _BASE_RATE

    n = len(df)
    ead = df["ead"].to_numpy()
    pd_base_arr = df["pd_position"].to_numpy()
    coll_types = df["collateral_type"].to_numpy()
    haircuts_base = df["haircut"].to_numpy().copy()
    oc_arr = df["overcollateralization"].to_numpy()
    tenor_days_arr = df["tenor_days"].to_numpy()

    # ── Canal 1 : Haircut spiral (Brunnermeier-Pedersen 2009) ──
    # GDP decline triggers margin calls -> forced selling -> wider haircuts
    # Govt collateral remains stable (flight to quality / safe haven)
    # Non-govt haircuts spiral: H_stressed = H_base * (1 + gamma * max(0, -delta_gdp))
    haircut_stressed = np.zeros(n)
    for i in range(n):
        coll_type = coll_types[i]
        h_base = haircuts_base[i]

        if coll_type == "govt":
            # Safe haven: minimal haircut increase
            gamma = _HAIRCUT_SPIRAL_GOVT
            h_cap = _HAIRCUT_SPIRAL_CAP_GOVT
        else:
            # Non-govt: full spiral amplification
            gamma = _HAIRCUT_SPIRAL_NON_GOVT
            h_cap = _HAIRCUT_SPIRAL_CAP_OTHER

        h_stressed = h_base * (1.0 + gamma * max(0.0, -delta_gdp))

        # Rate hikes also widen haircuts (bond price volatility)
        if coll_type in ("govt", "corp_ig"):
            rate_effect = max(0.0, delta_rate) * 0.01  # +1pp haircut per +1pp rate
            h_stressed += rate_effect

        haircut_stressed[i] = min(h_stressed, h_cap)

    # LGD stressed from haircut spiral (mechanical transmission)
    lgd_stressed = np.zeros(n)
    _TARGET_LGD_STRESS: Dict[str, float] = {
        "govt": 0.02,
        "corp_ig": 0.05,
        "equity": 0.12,
        "other": 0.08,
    }
    for i in range(n):
        oc = oc_arr[i]
        h_stress = haircut_stressed[i]
        coll_type = coll_types[i]

        collateral_value = 1.0 + oc
        effective_coverage = collateral_value * (1.0 - h_stress)
        lgd_residual = max(0.0, 1.0 - effective_coverage)

        target = _TARGET_LGD_STRESS.get(coll_type, 0.05)
        gap_risk = max(0.0, target - lgd_residual)
        lgd_stressed[i] = lgd_residual + gap_risk

    # GDP downturn addon
    lgd_addon = np.clip(
        max(0.0, -delta_gdp) * _LGD_DOWNTURN_COEFF,
        0.0,
        _LGD_DOWNTURN_CAP,
    )
    lgd_stressed = np.clip(lgd_stressed + lgd_addon, _LGD_FLOOR, _LGD_CAP)

    # ── Canal 2 : Counterparty contagion ──
    # Same multiplier structure as interbank (financial institutions)
    # GDP drop -> counterparty PD increase
    gdp_mult = 1.0 + max(0.0, -delta_gdp) * 0.25

    # Rate shock -> funding stress -> counterparty PD increase
    rate_mult = 1.0 + max(0.0, delta_rate) * 0.15

    # Interbank freeze amplification (non-linear, rate AND GDP together)
    freeze_indicator = max(0.0, delta_rate) * max(0.0, -delta_gdp)
    contagion_mult = 1.0 + freeze_indicator * 0.10

    pd_stressed = pd_base_arr * gdp_mult * rate_mult * contagion_mult

    # ── Canal 3 : Rollover risk (maturity mismatch) ──
    # Short-term repos face rollover freeze: counterparty refuses to renew
    # This is modeled as additional PD stress for short tenors under adverse
    # conditions (Bear Stearns March 2008: overnight repo run)
    stress_intensity = max(0.0, -delta_gdp) * max(0.0, delta_rate)
    for i in range(n):
        tenor_days = tenor_days_arr[i]
        # Maturity mismatch: shorter tenor = higher rollover risk
        if tenor_days <= 1:
            rollover_factor = 1.0 + stress_intensity * 0.20  # O/N: highest
        elif tenor_days <= 7:
            rollover_factor = 1.0 + stress_intensity * 0.15
        elif tenor_days <= 30:
            rollover_factor = 1.0 + stress_intensity * 0.10
        elif tenor_days <= 90:
            rollover_factor = 1.0 + stress_intensity * 0.05
        else:
            rollover_factor = 1.0  # >3M: term-locked, no rollover risk
        pd_stressed[i] *= rollover_factor

    pd_stressed = np.clip(pd_stressed, _PD_FLOOR, _PD_CAP)

    # ── RW stressed (haircut-dependent) ──
    # Wider haircuts mean higher E* -> higher post-CRM RW
    rw_stressed = np.zeros(n)
    for i in range(n):
        coll_type = coll_types[i]
        tenor_days = tenor_days_arr[i]
        oc = oc_arr[i]
        h_stress = haircut_stressed[i]

        base_rw = COLLATERAL_RW.get(coll_type, 0.12)

        # Stressed: short-term benefit partially lost in freeze
        if tenor_days <= 90 and freeze_indicator < 0.5:
            base_rw = base_rw * 0.70
        # In freeze: no short-term benefit (rollover impossible)

        # Master netting still applies (contractual)
        base_rw = base_rw * 0.50

        # OC reduction diminished by haircut spiral
        # Higher stressed haircut erodes overcollateralization benefit
        effective_oc = max(0.0, oc - max(0.0, h_stress - haircuts_base[i]))
        oc_reduction = max(0.0, 1.0 - effective_oc * 5.0)
        base_rw = base_rw * max(0.10, oc_reduction)

        rw_stressed[i] = base_rw

    rw_stressed = np.clip(rw_stressed, 0.0, 1.50)

    # ── EAD-weighted aggregation ──
    total_ead = ead.sum()
    if total_ead <= 0:
        return {"pd_base": 0.0003, "lgd_base": 0.03, "rw_crr3": 0.02}

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

def aggregate_repo_to_balance_row(
    df_positions: pl.DataFrame,
    profile: "AssetClassProfile",
) -> dict:
    """Agrege les positions repo en une ligne balance sheet.

    EAD-weighted PD/LGD/tenor/RW (bottom-up).
    HQLA : depends on collateral mix (govt = Level 1, corp IG = Level 2A).
    NSFR : RSF 0% for O/N, 10% for <6M, 15% for >6M.

    Args:
        df_positions: DataFrame from generate_repo_positions().
        profile: AssetClassProfile for repos/SFT.

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
