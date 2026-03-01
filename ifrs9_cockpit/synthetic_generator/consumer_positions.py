"""Chargement et sampling de donnees reelles Lending Club pour le credit consommation.

Charge le fichier ``consumer_credit.parquet`` pre-traite par
``ifrs9_cockpit/data/fetch_lending_club.py``, sample N positions,
scale vers le total_ead cible, et calcule PD/LGD position-level
depuis les features reelles.

Le pipeline aval (ECL Vasicek, comparator RAROC) consomme alors
des valeurs calculees bottom-up plutot que des constantes.

Calibration :
    - PD moyenne ~3.5% (Lending Club ~14%, recalibre FR via intercept shift)
    - LGD moyenne ~0.65 (unsecured consumer, bimodal : self-cure + queue)
    - Distribution FICO 300-850 (reelle, pas simulee)
    - Correlations inter-features reelles (dti vs income, grade vs int_rate)

References :
    - CRR3 Art. 123 : RW 75% retail
    - CRR3 Art. 154(2)(b) : asset_correlation = 0.04 (revolving)
    - CRR3 input floor : PD 5bp, LGD 25% unsecured
    - EBA/GL/2023/02 : origination standards consumer credit
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from ifrs9_cockpit.utils.helpers import logit, expit


# ──────────────────────────────────────────────
# CONSTANTES (calibrees marche FR/EU)
# ──────────────────────────────────────────────

# Scorecard coefficients (logistic, calibrated for PD mean ~3.5% FR consumer)
# Lending Club raw default ~14% -> intercept shift to match FR market
_BETA_0 = -4.20       # intercept (logit scale, target ~3.5%)
_BETA_SCORE = -3.5     # credit_score (normalized, higher = safer)
_BETA_DTI = 2.0        # dti (higher = riskier)
_BETA_INCOME = -0.8    # log(income) (higher = safer)
_BETA_UTIL = 1.5       # utilization_rate (higher = riskier)
_BETA_DELINQ = 1.0     # delinquencies_2y (more = riskier)
_BETA_INQ = 0.4        # inquiries_6m (more = riskier)
_BETA_EMP = -0.3       # employment_length (longer = safer)

# LGD parameters
_LGD_SELF_CURE_PROB = 0.30     # 30% self-cure (LGD ~0)
_LGD_SELF_CURE_MEAN = 0.05    # mean LGD for self-cures
_LGD_FLOOR_CRR3 = 0.25        # CRR3 input floor unsecured
_LGD_CAP = 0.95               # extreme cap

# Grade -> average PD (Lending Club empirical, pre-recalibration)
_GRADE_PD_RAW = {
    "A": 0.03, "B": 0.08, "C": 0.13,
    "D": 0.18, "E": 0.25, "F": 0.30, "G": 0.35,
}

# Parquet locations (train/test disjoints pour eviter le data leakage)
_DATA_DIR = Path(__file__).parent.parent / "data"
_PARQUET_TRAIN = _DATA_DIR / "consumer_credit_train.parquet"
_PARQUET_TEST = _DATA_DIR / "consumer_credit_test.parquet"
_PARQUET_LEGACY = _DATA_DIR / "consumer_credit.parquet"
# Backward compat alias (tests importent ce nom)
_PARQUET_PATH = _PARQUET_LEGACY


# ──────────────────────────────────────────────
# CHARGEMENT
# ──────────────────────────────────────────────

def compute_consumer_rw(df: pl.DataFrame) -> np.ndarray:
    """CRR3 Art. 123 : regulatory retail RW.
    - Transacteur revolving (utilization < 50%) : 45% (Art. 123(2))
    - Autre retail : 75%
    - Defaut : 150% (Art. 127)
    """
    n = len(df)
    rw = np.full(n, 0.75)
    if "utilization_rate" in df.columns:
        util = df["utilization_rate"].fill_null(0.50).fill_nan(0.50).to_numpy().astype(float)
        rw[util < 0.50] = 0.45
    if "default_flag" in df.columns:
        rw[df["default_flag"].fill_null(0).to_numpy() == 1] = 1.50
    return rw


def load_consumer_data(
    path: Optional[Path] = None,
    split: str = "train",
) -> pl.DataFrame:
    """Charge le parquet pre-traite.

    Args:
        path: Chemin vers le parquet (override, ignore split).
        split: "train" (50k, pour entrainement) ou "test" (15k, pour runtime/holdout).

    Returns:
        DataFrame avec ~25 colonnes de donnees reelles Lending Club.

    Raises:
        FileNotFoundError: Si le parquet n'existe pas.
            Executez d'abord : python ifrs9_cockpit/data/fetch_lending_club.py
    """
    if path is not None:
        p = path
    elif split == "test":
        p = _PARQUET_TEST
    elif split == "train":
        p = _PARQUET_TRAIN
    else:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    # Fallback vers le fichier legacy si le fichier split n'existe pas
    if not p.exists() and path is None:
        if _PARQUET_LEGACY.exists():
            p = _PARQUET_LEGACY
        else:
            raise FileNotFoundError(
                f"Fichier {p} introuvable. "
                "Executez d'abord : python ifrs9_cockpit/data/fetch_lending_club.py"
            )
    elif not p.exists():
        raise FileNotFoundError(
            f"Fichier {p} introuvable. "
            "Executez d'abord : python ifrs9_cockpit/data/fetch_lending_club.py"
        )
    return pl.read_parquet(p)


# ──────────────────────────────────────────────
# GENERATION (SAMPLE + SCALE)
# ──────────────────────────────────────────────

def generate_consumer_positions(
    n_positions: int = 5000,
    total_ead: float = 1.0e9,
    seed: int = 542,
    path: Optional[Path] = None,
) -> pl.DataFrame:
    """Sample n_positions depuis les donnees reelles Lending Club.

    1. Charge le parquet
    2. Sample avec remplacement (si n > taille dataset)
    3. Scale loan_amount pour matcher total_ead
    4. Recalcule PD/LGD position-level depuis les features reelles

    Args:
        n_positions: Nombre de positions a generer.
        total_ead: EAD total cible (les montants seront scales).
        seed: Graine aleatoire pour reproductibilite.
        path: Chemin vers le parquet (defaut: data/consumer_credit.parquet).

    Returns:
        DataFrame avec ~25 colonnes par position.
    """
    df_full = load_consumer_data(path, split="test")
    rng = np.random.default_rng(seed)

    # Sample (avec remplacement si n > dataset)
    replace = n_positions > len(df_full)
    idx = rng.choice(len(df_full), size=n_positions, replace=replace)
    df = df_full[idx.tolist()]
    df = df.with_columns(pl.Series("consumer_id", np.arange(n_positions)))

    # Scale montants vers total_ead
    raw_total = df["loan_amount"].sum()
    if raw_total > 0:
        scale = total_ead / raw_total
        df = df.with_columns((pl.col("loan_amount") * scale).alias("loan_amount"))
        # Scale income proportionnellement (coherence DTI)
        if "borrower_income" in df.columns:
            df = df.with_columns((pl.col("borrower_income") * scale).alias("borrower_income"))

    # PD position-level (scorecard sur features reelles)
    df = df.with_columns(pl.Series("pd_position", compute_consumer_pd(df)))

    # LGD position-level (depuis recovery data reel + modele)
    df = df.with_columns(pl.Series("lgd_position", compute_consumer_lgd(df, rng)))

    # RW position-level (CRR3 Art. 123)
    df = df.with_columns(pl.Series("rw_crr3", compute_consumer_rw(df)))

    return df


# ──────────────────────────────────────────────
# PD SCORECARD
# ──────────────────────────────────────────────

def compute_consumer_pd(df: pl.DataFrame) -> np.ndarray:
    """Compute position-level PD via logistic scorecard on real features.

    logit(PD) = beta_0
               + beta_score * (credit_score - 700) / 100
               + beta_dti * (dti - 20) / 20
               + beta_income * (log(income) - 11) / 2
               + beta_util * (utilization - 0.50)
               + beta_delinq * min(delinq, 5) / 5
               + beta_inq * min(inq, 10) / 10
               + beta_emp * (emp_length - 5) / 5

    Calibrated for PD mean ~3.5% (FR consumer market).
    """
    n = len(df)

    # credit_score: normalized around 700
    cs = df["credit_score"].fill_null(700).fill_nan(700).to_numpy().astype(float)
    x_score = (cs - 700.0) / 100.0

    # dti: normalized around 20
    dti = df["dti"].fill_null(20.0).fill_nan(20.0).to_numpy().astype(float)
    x_dti = (dti - 20.0) / 20.0

    # income: log-normalized
    inc = df["borrower_income"].fill_null(60000).fill_nan(60000).to_numpy().astype(float)
    x_income = (np.log(np.maximum(inc, 1.0)) - 11.0) / 2.0

    # utilization_rate
    if "utilization_rate" in df.columns:
        x_util = df["utilization_rate"].fill_null(0.50).fill_nan(0.50).to_numpy().astype(float) - 0.50
    else:
        x_util = np.zeros(n)

    # delinquencies_2y
    if "delinquencies_2y" in df.columns:
        x_delinq = np.minimum(df["delinquencies_2y"].fill_null(0).fill_nan(0).to_numpy().astype(float), 5.0) / 5.0
    else:
        x_delinq = np.zeros(n)

    # inquiries_6m
    if "inquiries_6m" in df.columns:
        x_inq = np.minimum(df["inquiries_6m"].fill_null(0).fill_nan(0).to_numpy().astype(float), 10.0) / 10.0
    else:
        x_inq = np.zeros(n)

    # employment_length
    if "employment_length" in df.columns:
        x_emp = (df["employment_length"].fill_null(5.0).fill_nan(5.0).to_numpy().astype(float) - 5.0) / 5.0
    else:
        x_emp = np.zeros(n)

    score = (
        _BETA_0
        + _BETA_SCORE * x_score
        + _BETA_DTI * x_dti
        + _BETA_INCOME * x_income
        + _BETA_UTIL * x_util
        + _BETA_DELINQ * x_delinq
        + _BETA_INQ * x_inq
        + _BETA_EMP * x_emp
    )

    pd_values = expit(score)
    # CRR3 input floor 5bp, cap at 30% (extreme consumer)
    return np.clip(pd_values, 0.005, 0.30)


# ──────────────────────────────────────────────
# LGD (RECOVERY-BASED)
# ──────────────────────────────────────────────

def compute_consumer_lgd(
    df: pl.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Compute position-level LGD from real recovery data.

    Pour les prets en defaut dans le dataset, la LGD est directement
    observee (lgd_observed). Pour les prets performing, on estime
    la LGD via un modele simplifie (grade + DTI proxy).

    Distribution bimodale reelle : ~30% self-cure + queue [0.25, 0.90].

    Args:
        df: DataFrame with consumer positions.
        rng: Random generator (for self-cure stochastic draw).

    Returns:
        Array of LGD values per position.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(df)
    lgd = np.full(n, 0.65)  # default prior

    # Use observed LGD for defaulted loans (from real recovery data)
    if "lgd_observed" in df.columns and "default_flag" in df.columns:
        defaults = df["default_flag"].to_numpy() == 1
        observed = df["lgd_observed"].to_numpy()
        valid_obs = defaults & np.isfinite(observed)
        lgd[valid_obs] = observed[valid_obs]

    # For performing loans: model-based LGD
    # Grade effect: A → lower LGD, G → higher LGD
    if "grade" in df.columns:
        grade_lgd_shift = {
            "A": -0.10, "B": -0.05, "C": 0.0,
            "D": 0.05, "E": 0.10, "F": 0.12, "G": 0.15,
        }
        if "default_flag" in df.columns:
            performing = ~(df["default_flag"].to_numpy().astype(bool))
        else:
            performing = np.ones(n, dtype=bool)
        for grade, shift in grade_lgd_shift.items():
            mask = performing & (df["grade"].to_numpy() == grade)
            lgd[mask] = 0.65 + shift

    # Self-cure mechanism (bimodal distribution)
    # ~30% of positions have very low LGD (full recovery)
    self_cure = rng.random(n) < _LGD_SELF_CURE_PROB
    lgd[self_cure] = rng.beta(1.5, 20.0, size=self_cure.sum()) * 0.20  # ~0.05 mean

    # Apply CRR3 floor for non-cure positions
    non_cure = ~self_cure
    lgd[non_cure] = np.maximum(lgd[non_cure], _LGD_FLOOR_CRR3)

    # Cap
    lgd = np.minimum(lgd, _LGD_CAP)

    return np.round(lgd, 4)


# ──────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────

def aggregate_consumer_to_balance_row(
    df_positions: pl.DataFrame,
    profile: "AssetClassProfile",
) -> dict:
    """Aggregate individual positions into a single balance sheet row.

    Returns a dict compatible with the df_balance_sheet schema.
    PD and LGD are EAD-weighted averages from the positions.

    Args:
        df_positions: DataFrame from generate_consumer_positions().
        profile: AssetClassProfile for consumer_credit.

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
        # Tenor from remaining_tenor if available, else profile default
        if "remaining_tenor" in df_positions.columns:
            tenor_vals = df_positions["remaining_tenor"].fill_null(profile.tenor).to_numpy()
            tenor_agg = float(np.dot(w, tenor_vals))
        else:
            tenor_agg = profile.tenor

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
