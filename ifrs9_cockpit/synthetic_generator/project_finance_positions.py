"""Generateur position-par-position pour le portefeuille Project Finance.

Genere ~500 positions PF individuelles (SPV infrastructure) avec mecanismes
de risque specifiques : PD DSCR-driven multi-facteur, LGD bi-modale avec
recovery waterfall 5 sources, slotting superviseur CRR3 Art. 153(5),
et stress test position-par-position (Approche B, full online ~5ms).

Le pipeline aval (ECL Vasicek, comparator RAROC, BL-CVaR) consomme alors
des valeurs calculees bottom-up plutot que des constantes parametriques.

Calibration :
    - Moody's PF Default Study 1983-2020 (6389 prets, CDR par secteur)
    - S&P 2023 Infrastructure Default Study (CDR 2y = 0.9% infra)
    - GCD Interactive Dashboard (350K+ defaults, senior secured LGD ~22%)
    - EDHEC Infrastructure (Blanc-Brude), recovery >85%, LGD bi-modale
    - Flyvbjerg (16000+ projets, arXiv:1409.0003), cost overrun par type
    - Oxford/iScience 2025 (8144 tx PF, spread -100bp renewables)

References :
    - CRR3 Art. 153(5) : slotting superviseur, PD floor 0.05%
    - Basel CRE 33 : 5 categories, RW table (70/90/115/250)
    - EBA RTS 2016 : 5 facteurs de classification
    - BCBS205 : maturity adjustment
    - NGFS Phase V : stranded assets, transition risk
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import polars as pl


# ──────────────────────────────────────────────
# CONSTANTES (calibrees Moody's/S&P/Flyvbjerg/Oxford)
# ──────────────────────────────────────────────

# --- 8 types de projets ---
PROJECT_TYPES: Dict[str, Dict] = {
    "renewable_solar": {
        "weight": 0.20,
        "pd_operational": 0.003,
        "pd_construction": 0.008,
        "lgd_operational": 0.20,
        "lgd_construction": 0.35,
        "tenor_mean": 20,
        "green_ratio": 1.0,
        "cost_overrun_mean": 0.01,
        "cost_overrun_p90": 0.10,
        "typical_gearing": 0.80,
    },
    "renewable_wind": {
        "weight": 0.15,
        "pd_operational": 0.004,
        "pd_construction": 0.009,
        "lgd_operational": 0.22,
        "lgd_construction": 0.37,
        "tenor_mean": 20,
        "green_ratio": 1.0,
        "cost_overrun_mean": 0.13,
        "cost_overrun_p90": 0.35,
        "typical_gearing": 0.75,
    },
    "power_conventional": {
        "weight": 0.12,
        "pd_operational": 0.005,
        "pd_construction": 0.012,
        "lgd_operational": 0.25,
        "lgd_construction": 0.40,
        "tenor_mean": 18,
        "green_ratio": 0.0,
        "cost_overrun_mean": 0.16,
        "cost_overrun_p90": 0.40,
        "typical_gearing": 0.70,
    },
    "transport_toll": {
        "weight": 0.15,
        "pd_operational": 0.004,
        "pd_construction": 0.010,
        "lgd_operational": 0.28,
        "lgd_construction": 0.42,
        "tenor_mean": 25,
        "green_ratio": 0.0,
        "cost_overrun_mean": 0.20,
        "cost_overrun_p90": 0.50,
        "typical_gearing": 0.75,
    },
    "transport_port": {
        "weight": 0.07,
        "pd_operational": 0.004,
        "pd_construction": 0.010,
        "lgd_operational": 0.25,
        "lgd_construction": 0.38,
        "tenor_mean": 20,
        "green_ratio": 0.0,
        "cost_overrun_mean": 0.20,
        "cost_overrun_p90": 0.50,
        "typical_gearing": 0.70,
    },
    "telecom_infra": {
        "weight": 0.10,
        "pd_operational": 0.005,
        "pd_construction": 0.012,
        "lgd_operational": 0.22,
        "lgd_construction": 0.35,
        "tenor_mean": 15,
        "green_ratio": 0.2,
        "cost_overrun_mean": 0.15,
        "cost_overrun_p90": 0.35,
        "typical_gearing": 0.65,
    },
    "extractive": {
        "weight": 0.08,
        "pd_operational": 0.006,
        "pd_construction": 0.015,
        "lgd_operational": 0.32,
        "lgd_construction": 0.50,
        "tenor_mean": 12,
        "green_ratio": 0.0,
        "cost_overrun_mean": 0.16,
        "cost_overrun_p90": 0.40,
        "typical_gearing": 0.60,
    },
    "social_ppp": {
        "weight": 0.13,
        "pd_operational": 0.002,
        "pd_construction": 0.006,
        "lgd_operational": 0.18,
        "lgd_construction": 0.30,
        "tenor_mean": 25,
        "green_ratio": 0.3,
        "cost_overrun_mean": 0.15,
        "cost_overrun_p90": 0.35,
        "typical_gearing": 0.85,
    },
}

# --- 3 phases ---
PHASES: Dict[str, Dict] = {
    "construction": {
        "weight": 0.15,
        "dscr_range": (0.0, 0.8),
        "drawn_range": (0.30, 0.90),
    },
    "ramp_up": {
        "weight": 0.15,
        "dscr_range": (0.80, 1.30),
        "drawn_range": (0.85, 1.00),
    },
    "operational": {
        "weight": 0.70,
        "dscr_range": (1.10, 2.50),
        "drawn_range": (1.00, 1.00),
    },
}

# --- Slotting CRR3 (Basel CRE 33 EXACT) ---
SLOTTING_CATEGORIES: Dict[str, Dict] = {
    "strong":       {"rw_gte_2_5y": 0.70, "rw_lt_2_5y": 0.50, "el": 0.004},
    "good":         {"rw_gte_2_5y": 0.90, "rw_lt_2_5y": 0.70, "el": 0.008},
    "satisfactory": {"rw_gte_2_5y": 1.15, "rw_lt_2_5y": 1.15, "el": 0.028},
    "weak":         {"rw_gte_2_5y": 2.50, "rw_lt_2_5y": 2.50, "el": 0.080},
    "default":      {"rw_gte_2_5y": 0.00, "rw_lt_2_5y": 0.00, "el": 0.500},
}

# --- 6 structures de revenus (Oxford/iScience 2025) ---
REVENUE_STRUCTURES: Dict[str, Dict] = {
    "availability_ppp":  {"weight": 0.15, "pd_mult": 0.50, "spread_bp": 125},
    "long_term_ppa":     {"weight": 0.30, "pd_mult": 0.70, "spread_bp": 150},
    "medium_term_ppa":   {"weight": 0.25, "pd_mult": 1.00, "spread_bp": 200},
    "short_contract":    {"weight": 0.15, "pd_mult": 1.50, "spread_bp": 300},
    "merchant":          {"weight": 0.10, "pd_mult": 2.50, "spread_bp": 425},
    "full_merchant":     {"weight": 0.05, "pd_mult": 3.00, "spread_bp": 500},
}

# --- 5 niveaux sponsors ---
SPONSOR_RATINGS: Dict[str, Dict] = {
    "IG_strong":  {"weight": 0.25, "pd_mult": 0.70, "sponsor_support": 0.15},
    "IG_weak":    {"weight": 0.30, "pd_mult": 0.85, "sponsor_support": 0.08},
    "HY_strong":  {"weight": 0.25, "pd_mult": 1.10, "sponsor_support": 0.04},
    "HY_weak":    {"weight": 0.15, "pd_mult": 1.30, "sponsor_support": 0.02},
    "unrated":    {"weight": 0.05, "pd_mult": 1.50, "sponsor_support": 0.01},
}

# --- 5 buckets pays ---
COUNTRY_BUCKETS: Dict[str, Dict] = {
    "OECD_core":   {"weight": 0.30, "pd_politique": 0.001, "has_pri": False},
    "OECD_other":  {"weight": 0.25, "pd_politique": 0.003, "has_pri": False},
    "EM_IG":       {"weight": 0.25, "pd_politique": 0.008, "has_pri": True},
    "EM_subIG":    {"weight": 0.12, "pd_politique": 0.020, "has_pri": True},
    "frontier":    {"weight": 0.08, "pd_politique": 0.050, "has_pri": True},
}

# PRI (MIGA) parameters
_PRI_COVERAGE = 0.95
_PRI_EFFECTIVENESS = 0.80  # PRI covers political, not commercial

# DSCR PD sensitivity (EthiFinance methodology)
_DSCR_REFERENCE = 1.30
_DSCR_SENSITIVITY = 1.5

# PD/LGD floors and caps (CRR3)
_PD_FLOOR = 0.0005   # CRR3 Art. 153 (0.05%)
_PD_CAP = 0.15
_LGD_FLOOR_OPERATIONAL = 0.15
_LGD_FLOOR_CONSTRUCTION = 0.25
_LGD_CAP = 0.75

# Bimodal LGD proportions (GCD/EDHEC/Moody's)
_PROB_RESTRUCTURE = 0.75
_LGD_MULT_RESTRUCTURE = 0.50
_LGD_MULT_ABANDON = 2.00

# Stress parameters (research-calibrated)
GDP_REVENUE_ELASTICITY: Dict[str, float] = {
    "availability_ppp":  0.00,
    "long_term_ppa":     0.00,
    "medium_term_ppa":   0.10,
    "short_contract":    0.30,
    "merchant":          0.50,
    "full_merchant":     0.70,
}
GDP_REVENUE_ELASTICITY_OVERRIDE: Dict[str, float] = {
    "transport_toll":  0.90,
    "transport_port":  0.70,
    "extractive":      0.90,
}
INDEXATION_RATIO: Dict[str, float] = {
    "availability_ppp":  0.45,
    "long_term_ppa":     0.75,
    "medium_term_ppa":   0.65,
    "short_contract":    0.50,
    "merchant":          0.30,
    "full_merchant":     0.20,
}
_OPEX_SHARE = 0.25
_HEDGE_RATIO_CONSTRUCTION = 1.00
_HEDGE_RATIO_OPERATIONAL = 0.80
_LGD_DOWNTURN_COEFF = 0.015   # +1.5pp LGD per -1% GDP
_LGD_DOWNTURN_CAP = 0.10      # max +10pp (Moody's)

# Revenue structure conditioning by project type
_REVENUE_STRUCTURE_BIAS: Dict[str, Dict[str, float]] = {
    "social_ppp": {"availability_ppp": 0.90, "long_term_ppa": 0.05,
                   "medium_term_ppa": 0.05},
    "renewable_solar": {"long_term_ppa": 0.50, "medium_term_ppa": 0.25,
                        "short_contract": 0.15, "merchant": 0.08,
                        "full_merchant": 0.02},
    "renewable_wind": {"long_term_ppa": 0.50, "medium_term_ppa": 0.25,
                       "short_contract": 0.15, "merchant": 0.08,
                       "full_merchant": 0.02},
    "extractive": {"short_contract": 0.30, "merchant": 0.40,
                   "full_merchant": 0.20, "medium_term_ppa": 0.10},
    "power_conventional": {"medium_term_ppa": 0.30, "short_contract": 0.30,
                           "merchant": 0.25, "full_merchant": 0.15},
}


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


def _conditioned_revenue_structure(
    rng: np.random.Generator,
    project_types: np.ndarray,
) -> np.ndarray:
    """Sample revenue structure conditioned on project type."""
    n = len(project_types)
    result = np.empty(n, dtype=object)
    rev_names = list(REVENUE_STRUCTURES.keys())
    default_weights = np.array([REVENUE_STRUCTURES[k]["weight"] for k in rev_names])
    default_weights = default_weights / default_weights.sum()

    for pt in np.unique(project_types):
        mask = project_types == pt
        count = mask.sum()
        if pt in _REVENUE_STRUCTURE_BIAS:
            bias = _REVENUE_STRUCTURE_BIAS[pt]
            w = np.array([bias.get(k, 0.0) for k in rev_names])
            w = w / w.sum()
        else:
            w = default_weights
        result[mask] = rng.choice(rev_names, size=count, p=w)
    return result


# ──────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────

def generate_project_finance_positions(
    n_positions: int = 500,
    total_ead: float = 1.0e9,
    seed: int = 742,
) -> pl.DataFrame:
    """Generate individual project finance positions with PF-specific features.

    Args:
        n_positions: Number of PF projects to generate.
        total_ead: Target total EAD (drawn amounts).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with ~30 columns per position.
    """
    rng = np.random.default_rng(seed)
    n = n_positions

    # --- 1. Project type ---
    project_types = _weighted_choice(rng, PROJECT_TYPES, n)

    # --- 2. Phase ---
    phase_names = list(PHASES.keys())
    phase_weights = np.array([PHASES[p]["weight"] for p in phase_names])
    phase_weights = phase_weights / phase_weights.sum()
    phases = rng.choice(phase_names, size=n, p=phase_weights)

    # --- 3. Country bucket ---
    country_buckets = _weighted_choice(rng, COUNTRY_BUCKETS, n)

    # --- 4. Sponsor rating ---
    sponsor_ratings = _weighted_choice(rng, SPONSOR_RATINGS, n)

    # --- 5. Revenue structure (conditioned on project type) ---
    revenue_structures = _conditioned_revenue_structure(rng, project_types)

    # --- 6. Tenor (type-specific + noise) ---
    tenor_means = np.array([
        PROJECT_TYPES[pt]["tenor_mean"] for pt in project_types
    ], dtype=float)
    tenor_noise = rng.normal(0, 2.0, size=n)
    tenor_years = np.clip(tenor_means + tenor_noise, 5.0, 30.0)

    # --- 7. Commitment (lognormal, scale to total_ead) ---
    log_commitment = rng.normal(19.1, 0.8, size=n)  # median ~200M EUR
    commitment = np.exp(log_commitment)
    commitment = np.maximum(commitment, 10e6)

    # --- 8. Drawn amount (phase-dependent) ---
    drawn_pct = np.empty(n, dtype=float)
    for phase_name, cfg in PHASES.items():
        mask = phases == phase_name
        lo, hi = cfg["drawn_range"]
        drawn_pct[mask] = rng.uniform(lo, hi, size=mask.sum())
    drawn_amount = commitment * drawn_pct

    # --- 9. EAD = drawn amount (on-balance, no CCF) ---
    ead = drawn_amount.copy()
    # Scale to match total_ead
    raw_total = ead.sum()
    if raw_total > 0:
        scale = total_ead / raw_total
        commitment *= scale
        drawn_amount *= scale
        ead *= scale

    # --- 10. DSCR (phase-specific) ---
    dscr = np.zeros(n, dtype=float)
    # Construction: DSCR = 0 (no cash flows)
    mask_constr = phases == "construction"
    dscr[mask_constr] = 0.0
    # Ramp-up: Uniform(0.80, 1.30)
    mask_ramp = phases == "ramp_up"
    dscr[mask_ramp] = rng.uniform(0.80, 1.30, size=mask_ramp.sum())
    # Operational: Gamma(shape=8, scale=0.18) + 1.0 -> median ~1.35
    mask_oper = phases == "operational"
    n_oper = mask_oper.sum()
    if n_oper > 0:
        dscr[mask_oper] = rng.gamma(8.0, 0.045, size=n_oper) + 1.0

    # --- 11. LLCR (always >= DSCR) ---
    remaining_years = tenor_years * rng.uniform(0.3, 0.8, size=n)
    llcr = dscr * (1.0 + 0.15 * remaining_years / np.maximum(tenor_years, 1.0))
    llcr = np.maximum(llcr, dscr)

    # --- 12. Gearing (type-specific + noise) ---
    gearing_means = np.array([
        PROJECT_TYPES[pt]["typical_gearing"] for pt in project_types
    ], dtype=float)
    gearing = np.clip(gearing_means + rng.normal(0, 0.05, size=n), 0.40, 0.90)

    # --- 13. Construction features ---
    epc_fixed_price = rng.binomial(1, 0.80, size=n)
    completion_pct = np.where(
        mask_constr, rng.uniform(0.10, 0.90, size=n),
        np.where(mask_ramp, rng.uniform(0.90, 1.00, size=n), 1.00),
    )
    # Cost overrun: Beta-distributed, type-specific mean
    overrun_means = np.array([
        PROJECT_TYPES[pt]["cost_overrun_mean"] for pt in project_types
    ], dtype=float)
    # Beta parameterization: mean=a/(a+b), use a=2 for moderate spread
    a_beta = 2.0
    b_beta = a_beta * (1.0 / np.maximum(overrun_means, 0.01) - 1.0)
    b_beta = np.clip(b_beta, 1.0, 200.0)
    cost_overrun_pct = rng.beta(a_beta, b_beta)
    dsra_months = np.full(n, 6.0)

    # --- 14. Green features ---
    green_ratios = np.array([
        PROJECT_TYPES[pt]["green_ratio"] for pt in project_types
    ], dtype=float)
    is_green = rng.random(size=n) < green_ratios
    # Carbon intensity: high for extractive/conventional, low for renewables
    carbon_base = np.where(is_green, 50.0, 400.0)
    carbon_intensity = np.clip(
        carbon_base + rng.normal(0, 50, size=n), 10, 800,
    )
    is_oecd = np.isin(country_buckets, ["OECD_core", "OECD_other"])
    taxonomy_aligned = is_green & is_oecd

    # --- 15. Interest rate features (for stress) ---
    base_rate = 0.035  # SCENARIO_BASE interest_rate
    spread_bp = np.array([
        REVENUE_STRUCTURES[rs]["spread_bp"] for rs in revenue_structures
    ], dtype=float)
    all_in_rate = base_rate + spread_bp / 10000.0

    # --- Build DataFrame ---
    df = pl.DataFrame({
        "project_id": np.arange(n),
        "project_type": project_types,
        "phase": phases,
        "country_bucket": country_buckets,
        "sponsor_rating": sponsor_ratings,
        "revenue_structure": revenue_structures,
        "tenor_years": np.round(tenor_years, 2),
        "commitment": np.round(commitment, 2),
        "drawn_amount": np.round(drawn_amount, 2),
        "ead": np.round(ead, 2),
        "dscr": np.round(dscr, 4),
        "llcr": np.round(llcr, 4),
        "gearing": np.round(gearing, 4),
        "epc_fixed_price": epc_fixed_price,
        "completion_pct": np.round(completion_pct, 4),
        "cost_overrun_pct": np.round(cost_overrun_pct, 4),
        "dsra_months": dsra_months,
        "is_green": is_green.astype(int),
        "carbon_intensity": np.round(carbon_intensity, 1),
        "taxonomy_aligned": taxonomy_aligned.astype(int),
        "base_rate": base_rate,
        "spread_bp": spread_bp,
        "all_in_rate": np.round(all_in_rate, 6),
    })

    # --- Compute PD, LGD, slotting ---
    pd_position = compute_project_finance_pd(df)
    lgd_position = compute_project_finance_lgd(df, rng)
    slotting = assign_slotting_category(df)

    # Default flag: stochastic based on PD
    default_flag = rng.binomial(1, pd_position)

    df = df.with_columns([
        pl.Series("pd_position", pd_position),
        pl.Series("lgd_position", lgd_position),
        pl.Series("slotting_category", slotting["category"].to_numpy()),
        pl.Series("slotting_rw", slotting["rw"].to_numpy()),
        pl.Series("slotting_el", slotting["el"].to_numpy()),
        pl.Series("default_flag", default_flag),
    ])

    return df


# ──────────────────────────────────────────────
# PD DSCR-DRIVEN (SPECIFIQUE PF)
# ──────────────────────────────────────────────

def compute_project_finance_pd(df: pl.DataFrame) -> np.ndarray:
    """Compute position-level PD via DSCR-driven multi-factor model.

    PD = PD_base(type, phase) * alpha_dscr * alpha_revenue * alpha_sponsor * alpha_country

    Clip: [0.0005, 0.15] (CRR3 Art. 153).
    """
    n = len(df)
    project_types = df["project_type"].to_numpy()
    phases = df["phase"].to_numpy()

    # --- PD base (type, phase) ---
    pd_base = np.empty(n, dtype=float)
    for i in range(n):
        pt_cfg = PROJECT_TYPES[project_types[i]]
        if phases[i] == "construction":
            pd_base[i] = pt_cfg["pd_construction"]
        elif phases[i] == "ramp_up":
            pd_base[i] = pt_cfg["pd_operational"] * 1.5
        else:
            pd_base[i] = pt_cfg["pd_operational"]

    # --- Alpha DSCR (EthiFinance methodology) ---
    dscr = df["dscr"].to_numpy()
    alpha_dscr = np.where(
        dscr > 0,
        np.exp(-_DSCR_SENSITIVITY * (dscr - _DSCR_REFERENCE)),
        2.0,  # construction: flat multiplier
    )

    # --- Alpha revenue ---
    revenue_structures = df["revenue_structure"].to_numpy()
    alpha_revenue = np.array([
        REVENUE_STRUCTURES[rs]["pd_mult"] for rs in revenue_structures
    ], dtype=float)

    # --- Alpha sponsor ---
    sponsor_ratings = df["sponsor_rating"].to_numpy()
    alpha_sponsor = np.array([
        SPONSOR_RATINGS[sr]["pd_mult"] for sr in sponsor_ratings
    ], dtype=float)

    # --- Alpha country (additive political risk) ---
    country_buckets = df["country_bucket"].to_numpy()
    pd_politique = np.array([
        COUNTRY_BUCKETS[cb]["pd_politique"] for cb in country_buckets
    ], dtype=float)
    has_pri = np.array([
        COUNTRY_BUCKETS[cb]["has_pri"] for cb in country_buckets
    ], dtype=bool)
    alpha_country = 1.0 + pd_politique / np.maximum(pd_base, 1e-6)
    # PRI mitigation
    pri_mitigation = np.where(has_pri, 1.0 - _PRI_COVERAGE * _PRI_EFFECTIVENESS, 1.0)
    alpha_country *= pri_mitigation

    # --- Combine ---
    pd_final = pd_base * alpha_dscr * alpha_revenue * alpha_sponsor * alpha_country

    return np.clip(pd_final, _PD_FLOOR, _PD_CAP)


# ──────────────────────────────────────────────
# LGD BI-MODALE RECOVERY WATERFALL (SPECIFIQUE PF)
# ──────────────────────────────────────────────

def compute_project_finance_lgd(
    df: pl.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute position-level LGD via bimodal SPV recovery waterfall.

    75% restructure (LGD ~ base * 0.50) / 25% abandon (LGD ~ base * 2.00).
    Recovery waterfall: assets + CF NPV + sponsor + PRI + DSRA.
    Floors: 0.15 (operational IG), 0.25 (construction). Cap: 0.75.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(df)
    project_types = df["project_type"].to_numpy()
    phases = df["phase"].to_numpy()

    # --- Base LGD (type, phase) ---
    lgd_type_phase = np.empty(n, dtype=float)
    for i in range(n):
        pt_cfg = PROJECT_TYPES[project_types[i]]
        if phases[i] == "construction":
            lgd_type_phase[i] = pt_cfg["lgd_construction"]
        elif phases[i] == "ramp_up":
            lgd_type_phase[i] = (pt_cfg["lgd_operational"] + pt_cfg["lgd_construction"]) / 2
        else:
            lgd_type_phase[i] = pt_cfg["lgd_operational"]

    # --- Bimodal draw ---
    is_restructure = rng.random(size=n) < _PROB_RESTRUCTURE
    lgd_mode = np.where(
        is_restructure,
        lgd_type_phase * _LGD_MULT_RESTRUCTURE,
        lgd_type_phase * _LGD_MULT_ABANDON,
    )

    # --- Recovery waterfall ---
    # RR_assets: tangible asset recovery
    asset_recovery = np.where(
        phases == "operational",
        rng.uniform(0.50, 0.70, size=n),
        np.where(
            phases == "ramp_up",
            rng.uniform(0.35, 0.55, size=n),
            rng.uniform(0.20, 0.40, size=n),  # construction
        ),
    )
    # Weight by completion for construction
    completion = df["completion_pct"].to_numpy()
    asset_recovery *= np.where(phases == "construction", completion, 1.0)
    rr_assets = asset_recovery * (1.0 - lgd_mode)  # assets partially offset loss

    # RR_cashflow: NPV of future CF post-default
    revenue_structures = df["revenue_structure"].to_numpy()
    revenue_contracted = np.array([
        1.0 - REVENUE_STRUCTURES[rs]["pd_mult"] / 3.0  # higher merchant = less contracted
        for rs in revenue_structures
    ], dtype=float)
    revenue_contracted = np.clip(revenue_contracted, 0.0, 1.0)
    npv_factor = np.where(phases == "operational", 0.30, np.where(phases == "ramp_up", 0.10, 0.0))
    rr_cashflow = npv_factor * revenue_contracted

    # RR_sponsor: limited support (non-recourse)
    sponsor_ratings = df["sponsor_rating"].to_numpy()
    sponsor_support = np.array([
        SPONSOR_RATINGS[sr]["sponsor_support"] for sr in sponsor_ratings
    ], dtype=float)
    rr_sponsor = sponsor_support

    # RR_PRI: political risk insurance
    country_buckets = df["country_bucket"].to_numpy()
    has_pri = np.array([
        COUNTRY_BUCKETS[cb]["has_pri"] for cb in country_buckets
    ], dtype=bool)
    rr_pri = np.where(has_pri, _PRI_COVERAGE * 0.10, 0.0)  # covers ~10% of total loss

    # RR_DSRA: debt service reserve account
    rr_dsra = np.minimum(df["dsra_months"].to_numpy() / 12.0 * 0.10, 0.05)

    # --- Total recovery ---
    total_rr = rr_assets + rr_cashflow + rr_sponsor + rr_pri + rr_dsra
    total_rr = np.minimum(total_rr, 0.85)  # cap total recovery at 85%

    lgd = lgd_mode * (1.0 - total_rr)

    # --- Floors (CRR3) ---
    is_construction = phases == "construction"
    lgd_floor = np.where(is_construction, _LGD_FLOOR_CONSTRUCTION, _LGD_FLOOR_OPERATIONAL)
    lgd = np.maximum(lgd, lgd_floor)

    # --- Cap ---
    lgd = np.minimum(lgd, _LGD_CAP)

    return np.round(lgd, 4)


# ──────────────────────────────────────────────
# SLOTTING CRR3 Art. 153(5) — BASEL CRE 33
# ──────────────────────────────────────────────

def assign_slotting_category(df: pl.DataFrame) -> pl.DataFrame:
    """Assign supervisory slotting category based on 5 EBA RTS 2016 factors.

    Returns DataFrame with columns: category, rw, el.
    """
    n = len(df)
    dscr = df["dscr"].to_numpy()
    phases = df["phase"].to_numpy()
    sponsors = df["sponsor_rating"].to_numpy()
    countries = df["country_bucket"].to_numpy()
    revenues = df["revenue_structure"].to_numpy()
    tenor_years = df["tenor_years"].to_numpy()
    default_flags = (
        df["default_flag"].to_numpy()
        if "default_flag" in df.columns
        else np.zeros(n)
    )

    categories = np.full(n, "satisfactory", dtype=object)

    is_ig = np.isin(sponsors, ["IG_strong", "IG_weak"])
    is_oecd = np.isin(countries, ["OECD_core", "OECD_other"])
    is_contracted = np.isin(revenues, ["availability_ppp", "long_term_ppa"])
    is_operational = phases == "operational"
    is_construction = phases == "construction"
    is_hy_or_unrated = np.isin(sponsors, ["HY_weak", "unrated"])

    # Rule 5 (lowest priority): DSCR < 1.00 or construction + HY/unrated → weak
    categories[dscr < 1.00] = "weak"
    categories[is_construction & is_hy_or_unrated] = "weak"

    # Rule 4: DSCR >= 1.00 → satisfactory
    categories[(dscr >= 1.00) & ~is_construction] = "satisfactory"

    # Rule 6: construction + IG → satisfactory (completion guarantee)
    categories[is_construction & is_ig] = "satisfactory"

    # Rule 3: DSCR >= 1.25 and not construction → good
    categories[(dscr >= 1.25) & ~is_construction] = "good"

    # Rule 2: DSCR >= 1.50 + operational + IG + OECD + contracted → strong
    strong_mask = (
        (dscr >= 1.50) & is_operational & is_ig & is_oecd & is_contracted
    )
    categories[strong_mask] = "strong"

    # Rule 1 (highest priority): default_flag → default
    categories[default_flags == 1] = "default"

    # Map to RW and EL
    long_tenor = tenor_years >= 2.5
    rw = np.array([
        SLOTTING_CATEGORIES[c]["rw_gte_2_5y"] if lt else SLOTTING_CATEGORIES[c]["rw_lt_2_5y"]
        for c, lt in zip(categories, long_tenor)
    ], dtype=float)
    el = np.array([
        SLOTTING_CATEGORIES[c]["el"] for c in categories
    ], dtype=float)

    return pl.DataFrame({"category": categories, "rw": rw, "el": el})


# ──────────────────────────────────────────────
# STRESS TEST POSITION-PAR-POSITION (APPROCHE B)
# ──────────────────────────────────────────────

def stress_project_finance_positions(
    df: pl.DataFrame,
    macro_params: Dict[str, float],
) -> Dict[str, float]:
    """Stress PF positions under macro scenario and re-aggregate.

    Full online, ~5ms for 500 positions. Implements 4 transmission channels:
    1. Interest rate → debt service (direct, hedged)
    2. GDP → revenue (elasticity by revenue structure / project type)
    3. Inflation → cost/revenue mismatch (indexation ratio)
    4. Spread → refinancing cost (construction only)

    Args:
        df: DataFrame from generate_project_finance_positions().
        macro_params: Dict with keys gdp_growth, unemployment_rate,
            interest_rate, inflation_rate, spread (bps).

    Returns:
        Dict with stressed pd_base, lgd_base, rw_crr3 (EAD-weighted).
    """
    # Scenario base values
    base_gdp = 1.2
    base_rate = 3.5
    base_inflation = 2.5
    base_spread = 200.0  # bps

    # Deltas (percentage points)
    delta_gdp = (macro_params.get("gdp_growth", base_gdp) - base_gdp) / 100.0
    delta_rate = (macro_params.get("interest_rate", base_rate) - base_rate) / 100.0
    delta_inflation = (macro_params.get("inflation_rate", base_inflation) - base_inflation) / 100.0
    delta_spread = macro_params.get("spread", base_spread) - base_spread  # bps

    n = len(df)
    phases = df["phase"].to_numpy()
    project_types = df["project_type"].to_numpy()
    revenue_structures = df["revenue_structure"].to_numpy()
    ead = df["ead"].to_numpy()
    dscr_base = df["dscr"].to_numpy().copy()
    gearing = df["gearing"].to_numpy()

    # --- Canal 1: Interest rate → debt service ---
    hedge_ratio = np.where(
        phases == "construction", _HEDGE_RATIO_CONSTRUCTION, _HEDGE_RATIO_OPERATIONAL,
    )
    effective_rate_shock = delta_rate * (1.0 - hedge_ratio)
    # DS multiplier: (rate_new / rate_old) for interest component
    # Approximate: DSCR_stressed / DSCR_base ~ 1 / (1 + gearing * effective_shock / all_in_rate)
    all_in_rate = df["all_in_rate"].to_numpy()
    ds_multiplier = 1.0 / (1.0 + gearing * effective_rate_shock / np.maximum(all_in_rate, 0.01))

    # --- Canal 2: GDP → revenue ---
    gdp_elasticity = np.array([
        GDP_REVENUE_ELASTICITY_OVERRIDE.get(
            project_types[i],
            GDP_REVENUE_ELASTICITY.get(revenue_structures[i], 0.0),
        )
        for i in range(n)
    ], dtype=float)
    revenue_shock = 1.0 + delta_gdp * gdp_elasticity

    # --- Canal 3: Inflation → mismatch ---
    indexation = np.array([
        INDEXATION_RATIO.get(revenue_structures[i], 0.50)
        for i in range(n)
    ], dtype=float)
    inflation_drag = 1.0 - delta_inflation * (1.0 - indexation) * _OPEX_SHARE

    # --- Canal 4: Spread → refinancing (construction only) ---
    spread_impact = np.where(
        phases == "construction",
        1.0 - delta_spread / 10000.0 * gearing * 0.5,  # half-year avg exposure
        1.0,
    )

    # --- CFADS stressed / DS stressed → DSCR stressed ---
    # For operational/ramp_up: DSCR_stressed = DSCR_base * revenue * inflation * spread * ds
    # For construction: DSCR remains 0
    cfads_factor = revenue_shock * inflation_drag * spread_impact
    dscr_stressed = np.where(
        dscr_base > 0,
        dscr_base * cfads_factor * ds_multiplier,
        0.0,
    )
    dscr_stressed = np.maximum(dscr_stressed, 0.0)

    # --- Re-compute PD with stressed DSCR ---
    # Build temporary df with stressed DSCR
    df_stressed = df.clone()
    df_stressed = df_stressed.with_columns(pl.Series("dscr", dscr_stressed))
    pd_stressed = compute_project_finance_pd(df_stressed)

    # --- LGD downturn addon ---
    # GDP negative → LGD increases
    gdp_growth_pct = macro_params.get("gdp_growth", base_gdp)
    lgd_addon = np.clip(-((gdp_growth_pct - base_gdp) / 100.0) * _LGD_DOWNTURN_COEFF,
                        0.0, _LGD_DOWNTURN_CAP)
    lgd_stressed = np.clip(df["lgd_position"].to_numpy() + lgd_addon, _LGD_FLOOR_OPERATIONAL, _LGD_CAP)

    # --- Re-slotting ---
    df_stressed = df_stressed.with_columns([
        pl.Series("pd_position", pd_stressed),
        pl.lit(0).alias("default_flag"),
    ])
    slotting_stressed = assign_slotting_category(df_stressed)
    rw_stressed = slotting_stressed["rw"].to_numpy()

    # --- EAD-weighted aggregation ---
    total_ead = ead.sum()
    if total_ead <= 0:
        return {"pd_base": 0.006, "lgd_base": 0.25, "rw_crr3": 1.15}

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

def aggregate_project_finance_to_balance_row(
    df_positions: pl.DataFrame,
    profile: "AssetClassProfile",
) -> dict:
    """Aggregate individual PF positions into a single balance sheet row.

    Returns a dict compatible with the df_balance_sheet schema.
    PD, LGD, RW are EAD-weighted averages.

    Args:
        df_positions: DataFrame from generate_project_finance_positions().
        profile: AssetClassProfile for project_finance.

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
        pd_agg = float(np.dot(w, df_positions["pd_position"].to_numpy()))
        lgd_agg = float(np.dot(w, df_positions["lgd_position"].to_numpy()))
        tenor_agg = float(np.dot(w, df_positions["tenor_years"].to_numpy()))
        rw_agg = float(np.dot(w, df_positions["slotting_rw"].to_numpy()))

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
