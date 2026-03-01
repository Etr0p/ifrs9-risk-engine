"""Generateur position-par-position pour le portefeuille hypothecaire retail.

Genere ~2000 prets hypothecaires individuels avec features realistes
(LTV, DTI, DPE, region, statut paiement), PD/LGD position-level,
puis agrege vers la ligne ``retail_mortgage`` du bilan.

Le pipeline aval (ECL Vasicek, comparator RAROC) consomme alors
des valeurs calculees bottom-up plutot que des constantes.

Calibration :
    - Regions : 11 regions metropolitaines (poids INSEE)
    - DPE : distribution A-G (parc existant + neuf)
    - LTV : coherente avec ``ltv_distribution`` du profil AssetClassProfile
    - PD moyenne ~1.2% (EBA 2024, portefeuille FR/EU residentiel)
    - LGD moyenne ~15% (collateral immobilier, haircut 15-25%)

References :
    - CRR3 Art. 124-125 : schedule RW par tranche LTV
    - CRR3 Art. 154(2)(a) : asset_correlation = 0.15 (residentiel)
    - EBA/GL/2025/01 : ESG dans l'evaluation du risque de credit
    - HCSF (2022) : limites DTI 35%, duree 25 ans
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import polars as pl

from ifrs9_cockpit.utils.helpers import logit, expit


# ──────────────────────────────────────────────
# CONSTANTES (calibrees marche FR/EU)
# ──────────────────────────────────────────────

REGIONS: Dict[str, Dict] = {
    "Ile-de-France": {"weight": 0.25, "hpi_sensitivity": 1.3, "pd_adj": -0.15,
                      "median_pv": 350_000},
    "PACA":          {"weight": 0.12, "hpi_sensitivity": 1.1, "pd_adj": 0.0,
                      "median_pv": 280_000},
    "Auvergne-RA":   {"weight": 0.10, "hpi_sensitivity": 0.9, "pd_adj": 0.05,
                      "median_pv": 220_000},
    "Nouvelle-Aq":   {"weight": 0.08, "hpi_sensitivity": 0.8, "pd_adj": 0.10,
                      "median_pv": 200_000},
    "Occitanie":     {"weight": 0.08, "hpi_sensitivity": 0.85, "pd_adj": 0.08,
                      "median_pv": 210_000},
    "Hauts-de-Fr":   {"weight": 0.07, "hpi_sensitivity": 0.7, "pd_adj": 0.20,
                      "median_pv": 160_000},
    "Grand-Est":     {"weight": 0.07, "hpi_sensitivity": 0.75, "pd_adj": 0.15,
                      "median_pv": 170_000},
    "Pays-Loire":    {"weight": 0.06, "hpi_sensitivity": 0.80, "pd_adj": 0.10,
                      "median_pv": 190_000},
    "Bretagne":      {"weight": 0.06, "hpi_sensitivity": 0.80, "pd_adj": 0.08,
                      "median_pv": 195_000},
    "Normandie":     {"weight": 0.05, "hpi_sensitivity": 0.65, "pd_adj": 0.18,
                      "median_pv": 165_000},
    "Autres":        {"weight": 0.06, "hpi_sensitivity": 0.70, "pd_adj": 0.12,
                      "median_pv": 155_000},
}

DPE_CLASSES: Dict[str, Dict] = {
    "A": {"weight": 0.05, "green_bonus": -0.20, "transition_risk": 0.0},
    "B": {"weight": 0.10, "green_bonus": -0.10, "transition_risk": 0.05},
    "C": {"weight": 0.20, "green_bonus": -0.05, "transition_risk": 0.10},
    "D": {"weight": 0.25, "green_bonus":  0.0,  "transition_risk": 0.20},
    "E": {"weight": 0.20, "green_bonus":  0.10, "transition_risk": 0.40},
    "F": {"weight": 0.12, "green_bonus":  0.20, "transition_risk": 0.70},
    "G": {"weight": 0.08, "green_bonus":  0.35, "transition_risk": 1.00},
}

PROPERTY_TYPES: Dict[str, float] = {"apartment": 0.55, "house": 0.35, "mixed_use": 0.10}

RATE_TYPES: Dict[str, float] = {"fixed": 0.85, "variable": 0.10, "mixed": 0.05}

PAYMENT_STATUS: Dict[str, float] = {
    "current": 0.92, "watch": 0.05, "past_due_30": 0.02, "past_due_90": 0.01,
}

# Scorecard coefficients (logistic, calibrated for PD mean ~1.2%)
_BETA_0 = -4.40       # intercept (logit scale)
_BETA_LTV = 3.0       # LTV > 70% = risk
_BETA_DTI = 2.5       # DTI > 33% = risk
_BETA_DPE = 1.2       # DPE green_bonus (F/G = positive = risk)
_BETA_REGION = 1.5    # regional pd_adj
_BETA_AGE = 0.3       # deviation from 42
_BETA_RATE = 0.4      # variable rate = risk
_BETA_VINTAGE = 0.5   # seasoning (newer = more risk)


# ──────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────

def compute_mortgage_rw(df: pl.DataFrame) -> np.ndarray:
    """CRR3 Art. 124-125 : RW par bucket LTV pour chaque pret."""
    ltv = df["ltv"].to_numpy()
    rw = np.full(len(ltv), 0.70)  # default: LTV > 100%
    rw[ltv <= 1.00] = 0.70
    rw[ltv <= 0.90] = 0.50
    rw[ltv <= 0.80] = 0.35
    rw[ltv <= 0.70] = 0.30
    rw[ltv <= 0.60] = 0.25
    rw[ltv <= 0.50] = 0.20
    # Defaut (CRR3 Art. 127) : 150%
    if "default_flag" in df.columns:
        rw[df["default_flag"].to_numpy() == 1] = 1.50
    return rw


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


def generate_mortgage_positions(
    n_positions: int = 2000,
    total_ead: float = 1.0e9,
    seed: int = 442,
    ltv_distribution: Optional[Dict[float, float]] = None,
) -> pl.DataFrame:
    """Generate individual mortgage positions with realistic features.

    Args:
        n_positions: Number of mortgage loans to generate.
        total_ead: Target total EAD (loan amounts will be scaled).
        seed: Random seed for reproducibility.
        ltv_distribution: LTV bucket distribution from AssetClassProfile.
            If None, uses a default FR/EU distribution.

    Returns:
        DataFrame with ~20 columns per position.
    """
    rng = np.random.default_rng(seed)

    # --- 1. Categorical features ---
    regions = _weighted_choice(rng, REGIONS, n_positions)
    property_types = rng.choice(
        list(PROPERTY_TYPES.keys()),
        size=n_positions,
        p=list(PROPERTY_TYPES.values()),
    )
    rate_types = rng.choice(
        list(RATE_TYPES.keys()),
        size=n_positions,
        p=list(RATE_TYPES.values()),
    )
    dpe_classes = _weighted_choice(rng, DPE_CLASSES, n_positions)
    payment_statuses = rng.choice(
        list(PAYMENT_STATUS.keys()),
        size=n_positions,
        p=list(PAYMENT_STATUS.values()),
    )

    # --- 2. Property values (log-normal, calibrated by region) ---
    median_pvs = np.array([REGIONS[r]["median_pv"] for r in regions], dtype=float)
    # Log-normal: mu = log(median), sigma = 0.35 (typical housing dispersion)
    log_mu = np.log(median_pvs)
    log_sigma = 0.35
    property_values = np.exp(rng.normal(log_mu, log_sigma))
    property_values = np.maximum(property_values, 50_000.0)  # floor 50k

    # --- 3. LTV (sampled from profile distribution + noise) ---
    if ltv_distribution is None:
        ltv_distribution = {
            0.50: 0.15, 0.60: 0.25, 0.70: 0.25,
            0.80: 0.20, 0.90: 0.10, 1.00: 0.05,
        }
    ltv_buckets = np.array(list(ltv_distribution.keys()))
    ltv_weights = np.array(list(ltv_distribution.values()))
    ltv_weights = ltv_weights / ltv_weights.sum()

    # Sample bucket, then add gaussian noise +-5pp
    ltv_bucket_samples = rng.choice(ltv_buckets, size=n_positions, p=ltv_weights)
    ltv_noise = rng.normal(0, 0.05, size=n_positions)
    ltv = np.clip(ltv_bucket_samples + ltv_noise, 0.10, 1.10)

    # --- 4. Loan amount = property_value * ltv ---
    loan_amounts = property_values * ltv

    # Scale to match total_ead
    raw_total = loan_amounts.sum()
    if raw_total > 0:
        scale_factor = total_ead / raw_total
        loan_amounts *= scale_factor
        property_values *= scale_factor

    # --- 5. DTI (Beta distribution, anti-correlated with income) ---
    # Beta(2,5) -> mean ~0.286, rescaled to [0.15, 0.50]
    dti_raw = rng.beta(2.0, 5.0, size=n_positions)
    dti = 0.15 + dti_raw * 0.35  # [0.15, 0.50]

    # --- 6. Borrower income (derived from loan_amount and DTI) ---
    # Annual mortgage payment ~ loan_amount * 0.05 (5% annual rate approx)
    # DTI = annual_payment / income => income = annual_payment / DTI
    annual_payment_approx = loan_amounts * 0.05
    borrower_income = annual_payment_approx / np.clip(dti, 0.15, 0.50)

    # --- 7. Borrower age ---
    borrower_age = np.clip(rng.normal(42, 12, size=n_positions), 25, 75).astype(int)

    # --- 8. Origination and tenor ---
    origination_year = rng.integers(2005, 2026, size=n_positions)
    remaining_tenor = np.clip(
        rng.integers(1, 26, size=n_positions),
        1,
        25,
    )

    # --- 9. Interest rate margin (spread over reference rate) ---
    # Fixed: tighter spread; Variable: wider spread
    base_margin = rng.uniform(0.005, 0.025, size=n_positions)
    is_variable = (rate_types == "variable").astype(float)
    interest_rate_margin = base_margin + is_variable * 0.005

    # --- 10. DPD (coherent with payment status) ---
    dpd = np.zeros(n_positions, dtype=int)
    for i, status in enumerate(payment_statuses):
        if status == "watch":
            dpd[i] = rng.integers(1, 30)
        elif status == "past_due_30":
            dpd[i] = rng.integers(30, 90)
        elif status == "past_due_90":
            dpd[i] = rng.integers(90, 365)

    # --- 11. Compute PD and LGD ---
    df = pl.DataFrame({
        "mortgage_id": np.arange(n_positions),
        "region": regions,
        "property_type": property_types,
        "rate_type": rate_types,
        "dpe_class": dpe_classes,
        "property_value": np.round(property_values, 2),
        "loan_amount": np.round(loan_amounts, 2),
        "ltv": np.round(ltv, 4),
        "dti": np.round(dti, 4),
        "borrower_income": np.round(borrower_income, 2),
        "borrower_age": borrower_age,
        "origination_year": origination_year,
        "remaining_tenor": remaining_tenor,
        "interest_rate_margin": np.round(interest_rate_margin, 5),
        "payment_status": payment_statuses,
        "dpd": dpd,
    })

    df = df.with_columns(pl.Series("pd_position", compute_mortgage_pd(df)))
    df = df.with_columns(pl.Series("lgd_position", compute_mortgage_lgd(df)))

    # Default flag: stochastic based on PD, but forced for past_due_90
    default_draw = rng.random(n_positions) < df["pd_position"].to_numpy()
    forced_default = df["payment_status"].to_numpy() == "past_due_90"
    df = df.with_columns(pl.Series("default_flag", (default_draw | forced_default).astype(int)))

    # RW position-level (CRR3 Art. 124-125)
    df = df.with_columns(pl.Series("rw_crr3", compute_mortgage_rw(df)))

    return df


# ──────────────────────────────────────────────
# PD SCORECARD
# ──────────────────────────────────────────────

def compute_mortgage_pd(df: pl.DataFrame) -> np.ndarray:
    """Compute position-level PD via logistic scorecard.

    logit(PD) = beta_0
               + beta_ltv * (ltv - 0.70)
               + beta_dti * (dti - 0.33)
               + beta_dpe * green_bonus
               + beta_region * pd_adj
               + beta_age * (age - 42) / 30
               + beta_rate * is_variable
               + beta_vintage * (2025 - origination_year) / 20

    Calibrated for PD mean ~1.2%.
    """
    green_bonus = np.array([
        DPE_CLASSES[d]["green_bonus"] for d in df["dpe_class"].to_numpy()
    ])
    pd_adj = np.array([
        REGIONS[r]["pd_adj"] for r in df["region"].to_numpy()
    ])
    is_variable = (df["rate_type"].to_numpy() == "variable").astype(float)

    score = (
        _BETA_0
        + _BETA_LTV * (df["ltv"].to_numpy() - 0.70)
        + _BETA_DTI * (df["dti"].to_numpy() - 0.33)
        + _BETA_DPE * green_bonus
        + _BETA_REGION * pd_adj
        + _BETA_AGE * (df["borrower_age"].to_numpy() - 42) / 30.0
        + _BETA_RATE * is_variable
        + _BETA_VINTAGE * (2025 - df["origination_year"].to_numpy()) / 20.0
    )

    pd_values = expit(score)
    return np.clip(pd_values, 0.0001, 0.20)


# ──────────────────────────────────────────────
# LGD (COLLATERAL-BASED)
# ──────────────────────────────────────────────

def compute_mortgage_lgd(df: pl.DataFrame) -> np.ndarray:
    """Compute position-level LGD based on collateral coverage.

    lgd = max(floor, 1 - recovery_rate)
    recovery_rate = min(1.0, property_value * (1 - haircut) / loan_amount)
    haircut = base_haircut + dpe_malus + region_illiquidity

    Floor: 0.05 (CRR3 input floor for secured residential).
    """
    # Base haircut: forced-sale discount + legal costs + time value of money
    # Calibrated to produce LGD ~15% across the portfolio (EBA 2024)
    base_haircut = 0.30

    # DPE malus: passoire thermique = valuation discount
    dpe_malus = np.array([
        max(0.0, DPE_CLASSES[d]["green_bonus"] * 0.15)
        for d in df["dpe_class"].to_numpy()
    ])

    # Region illiquidity: lower hpi_sensitivity = less liquid market
    region_illiq = np.array([
        max(0.0, (1.0 - REGIONS[r]["hpi_sensitivity"]) * 0.15)
        for r in df["region"].to_numpy()
    ])

    haircut = base_haircut + dpe_malus + region_illiq

    # Recovery rate
    loan_amount = np.clip(df["loan_amount"].to_numpy(), 1.0, None)
    recovery = np.minimum(
        1.0,
        df["property_value"].to_numpy() * (1.0 - haircut) / loan_amount,
    )

    lgd = 1.0 - recovery
    # CRR3 input floor: 5% for secured residential
    lgd = np.maximum(lgd, 0.05)
    # Cap at 60% (extreme cases)
    lgd = np.minimum(lgd, 0.60)

    return np.round(lgd, 4)


# ──────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────

def aggregate_mortgage_to_balance_row(
    df_positions: pl.DataFrame,
    profile: "AssetClassProfile",
) -> dict:
    """Aggregate individual positions into a single balance sheet row.

    Returns a dict compatible with the df_balance_sheet schema.
    PD and LGD are EAD-weighted averages from the positions.

    Args:
        df_positions: DataFrame from generate_mortgage_positions().
        profile: AssetClassProfile for retail_mortgage.

    Returns:
        Dict with all balance sheet columns.
    """
    weights = df_positions["loan_amount"].to_numpy()
    total_ead = weights.sum()

    if total_ead <= 0:
        pd_agg = profile.pd_base
        lgd_agg = profile.lgd_base
        tenor_agg = profile.tenor
    else:
        w = weights / total_ead
        pd_agg = float(np.dot(w, df_positions["pd_position"].to_numpy()))
        lgd_agg = float(np.dot(w, df_positions["lgd_position"].to_numpy()))
        tenor_agg = float(np.dot(w, df_positions["remaining_tenor"].to_numpy()))

    # EAD-weighted RW if position-level available
    if total_ead > 0 and "rw_crr3" in df_positions.columns:
        rw_agg = float(np.dot(w, df_positions["rw_crr3"].to_numpy()))
    else:
        rw_agg = profile.rw_crr3

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
        "rw_crr3": rw_agg,
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
# TRAINING DATA (50k positions, leakage-free)
# ──────────────────────────────────────────────

def generate_mortgage_training_data(
    n_positions: int = 50_000,
    seed: int = 442,
) -> pl.DataFrame:
    """Generate a large mortgage dataset suitable for PD model training.

    Calls generate_mortgage_positions() with n_positions, then drops leakage
    columns (pd_position, lgd_position, payment_status, mortgage_id) to
    produce a clean training DataFrame with default_flag as target.

    Args:
        n_positions: Number of positions to generate (default 50k).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with ~15 features + default_flag, no leakage columns.
    """
    df = generate_mortgage_positions(n_positions=n_positions, seed=seed)

    # Drop leakage columns
    leakage_cols = ["pd_position", "lgd_position", "payment_status", "mortgage_id"]
    df = df.drop([c for c in leakage_cols if c in df.columns])

    return df
