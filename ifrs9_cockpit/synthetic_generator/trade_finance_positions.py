"""Generateur position-par-position pour le portefeuille Trade Finance.

Genere ~5000 transactions TF individuelles avec mecanismes de risque
specifiques : PD multi-composante (counterparty + transfer + confirmation),
LGD recovery waterfall (goods, margin, confirmation, ECA), EAD = nominal * CCF,
et maturite effective sans floor 1 an (BCBS205 waiver).

Le pipeline aval (ECL Vasicek, comparator RAROC) consomme alors
des valeurs calculees bottom-up plutot que des constantes.

Calibration :
    - ICC Trade Register 2024 (52M transactions, USD 25.7T, 22 banques)
    - PD moyenne ~0.008 (portefeuille TF global ICC)
    - LGD moyenne ~0.30 (recovery waterfall multi-source)
    - CCF CRR3 Art. 111 + Annex I : 20% LC, 50% guarantee, 100% SBLC/SCF

References :
    - CRR3 Art. 111, Annex I : CCF off-balance-sheet
    - BCBS205 (2014) : maturity waiver for self-liquidating TF
    - UCP 600 (ICC) : documentary credit rules
    - ICC Trade Register 2024 : default and loss statistics
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import polars as pl

from ifrs9_cockpit.utils.helpers import logit, expit


# ──────────────────────────────────────────────
# CONSTANTES (calibrees ICC Trade Register 2024 + CRR3)
# ──────────────────────────────────────────────

# --- 7 produits TF ---
PRODUCTS: Dict[str, Dict] = {
    "import_lc": {
        "weight": 0.35, "pd_icc": 0.0008, "lgd_icc": 0.37,
        "ccf": 0.20, "has_goods": True, "has_confirmation": False,
        "is_eca": False, "tenor_scale": 180, "alpha_product": 0.25,
        "lgd_floor": 0.10,
    },
    "export_lc": {
        "weight": 0.25, "pd_icc": 0.0004, "lgd_icc": 0.36,
        "ccf": 0.20, "has_goods": True, "has_confirmation": True,
        "is_eca": False, "tenor_scale": 180, "alpha_product": 0.12,
        "lgd_floor": 0.10,
    },
    "guarantee_perf": {
        "weight": 0.15, "pd_icc": 0.0019, "lgd_icc": 0.58,
        "ccf": 0.50, "has_goods": False, "has_confirmation": False,
        "is_eca": False, "tenor_scale": 365, "alpha_product": 0.60,
        "lgd_floor": 0.25,
    },
    "guarantee_financial": {
        "weight": 0.05, "pd_icc": 0.0024, "lgd_icc": 0.45,
        "ccf": 1.00, "has_goods": False, "has_confirmation": False,
        "is_eca": False, "tenor_scale": 365, "alpha_product": 0.75,
        "lgd_floor": 0.25,
    },
    "documentary_coll": {
        "weight": 0.10, "pd_icc": 0.0021, "lgd_icc": 0.40,
        "ccf": 0.20, "has_goods": False, "has_confirmation": False,
        "is_eca": False, "tenor_scale": 120, "alpha_product": 0.65,
        "lgd_floor": 0.25,
    },
    "supply_chain_fin": {
        "weight": 0.07, "pd_icc": 0.0010, "lgd_icc": 0.25,
        "ccf": 1.00, "has_goods": False, "has_confirmation": False,
        "is_eca": False, "tenor_scale": 60, "alpha_product": 0.30,
        "lgd_floor": 0.25,
    },
    "export_credit_mlt": {
        "weight": 0.03, "pd_icc": 0.0044, "lgd_icc": 0.053,
        "ccf": 0.50, "has_goods": False, "has_confirmation": False,
        "is_eca": True, "tenor_scale": 365, "alpha_product": 1.40,
        "lgd_floor": 0.05,
    },
}

# --- 6 classes de commodites ---
COMMODITIES: Dict[str, Dict] = {
    "agricultural":  {"weight": 0.25, "volatility": 0.30, "h_liquidity": 0.05},
    "machinery":     {"weight": 0.20, "volatility": 0.10, "h_liquidity": 0.02},
    "chemicals":     {"weight": 0.15, "volatility": 0.20, "h_liquidity": 0.03},
    "metals":        {"weight": 0.15, "volatility": 0.30, "h_liquidity": 0.04},
    "textiles":      {"weight": 0.10, "volatility": 0.15, "h_liquidity": 0.03},
    "energy":        {"weight": 0.15, "volatility": 0.35, "h_liquidity": 0.05},
}

# --- 7 buckets risque pays ---
COUNTRY_RATINGS: Dict[str, Dict] = {
    "AAA": {"weight": 0.05, "pd_sovereign": 0.0001, "tc_factor": 0.0, "lgd_adj": -0.05},
    "AA":  {"weight": 0.15, "pd_sovereign": 0.0004, "tc_factor": 0.0, "lgd_adj": -0.03},
    "A":   {"weight": 0.30, "pd_sovereign": 0.0008, "tc_factor": 0.1, "lgd_adj":  0.00},
    "BBB": {"weight": 0.25, "pd_sovereign": 0.0020, "tc_factor": 0.3, "lgd_adj":  0.05},
    "BB":  {"weight": 0.15, "pd_sovereign": 0.0080, "tc_factor": 0.5, "lgd_adj":  0.10},
    "B":   {"weight": 0.07, "pd_sovereign": 0.0250, "tc_factor": 0.7, "lgd_adj":  0.15},
    "CCC": {"weight": 0.03, "pd_sovereign": 0.0800, "tc_factor": 1.0, "lgd_adj":  0.20},
}

# Rating numeric mapping (higher = riskier)
_RATING_NUMERIC = {"AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6, "CCC": 7}

# PD counterparty scorecard coefficients
_BETA_0 = -4.80       # intercept (logit scale, calibrated ~0.008 portfolio)
_BETA_RATING = 0.40   # importer rating (higher numeric = riskier)
_BETA_FREQ = -0.15    # trade frequency (more = safer, established relationship)

# Copula correlation for confirmed LC joint default
_RHO_SAME_COUNTRY = 0.40
_RHO_DIFF_COUNTRY = 0.15

# ECA coverage for export_credit_mlt
_ECA_COVERAGE = 0.95

# z_99 for commodity haircut
_Z99 = 2.326


# ──────────────────────────────────────────────
# GENERATION
# ──────────────────────────────────────────────

_TF_RW_BASE = {
    "import_lc": 0.20,           # LC court terme, contrepartie bancaire
    "export_lc": 0.20,           # LC confirmee
    "guarantee_perf": 0.50,      # Garantie de performance
    "guarantee_financial": 1.00, # Garantie financiere, RW corporate
    "documentary_coll": 0.20,    # Encaissement documentaire
    "supply_chain_fin": 1.00,    # SCF, contrepartie corporate
    "export_credit_mlt": 0.50,   # Couverture ECA
}

_RATING_ADJ = {"AAA": 0.5, "AA": 0.5, "A": 0.75, "BBB": 1.0, "BB": 1.25, "B": 1.5, "CCC": 2.0}


def compute_trade_finance_rw(df: pl.DataFrame) -> np.ndarray:
    """CRR3 Art. 111 : RW par type de produit TF + ajustement rating."""
    products = df["product_type"].to_numpy()
    rw = np.array([_TF_RW_BASE.get(p, 1.00) for p in products])
    # Ajustement rating contrepartie pour garanties
    if "importer_country_rating" in df.columns:
        ratings = df["importer_country_rating"].to_numpy()
        for i, (p, r) in enumerate(zip(products, ratings)):
            if p in ("guarantee_perf", "guarantee_financial"):
                rw[i] *= _RATING_ADJ.get(r, 1.0)
    if "default_flag" in df.columns:
        rw[df["default_flag"].to_numpy() == 1] = 1.50
    return np.clip(rw, 0.0, 2.50)


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


def generate_trade_finance_positions(
    n_positions: int = 5000,
    total_ead: float = 1.0e9,
    seed: int = 642,
) -> pl.DataFrame:
    """Generate individual trade finance positions with TF-specific features.

    Args:
        n_positions: Number of TF transactions to generate.
        total_ead: Target total EAD (post-CCF; nominals will be back-computed).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with ~25 columns per position.
    """
    rng = np.random.default_rng(seed)
    n = n_positions

    # --- 1. Product type ---
    product_types = _weighted_choice(rng, PRODUCTS, n)

    # --- 2. Commodity class ---
    commodity_classes = _weighted_choice(rng, COMMODITIES, n)

    # --- 3. Country risk (exporter + importer, may differ) ---
    exporter_ratings = _weighted_choice(rng, COUNTRY_RATINGS, n)
    importer_ratings = _weighted_choice(rng, COUNTRY_RATINGS, n)

    # --- 4. Tenor (product-specific scale, exponential) ---
    tenor_scales = np.array([PRODUCTS[p]["tenor_scale"] for p in product_types], dtype=float)
    raw_tenor_days = rng.exponential(scale=tenor_scales)
    tenor_days = np.clip(raw_tenor_days, 30, 365).astype(int)
    tenor_years = tenor_days / 365.0

    # --- 5. Nominal amount (log-normal, then scale) ---
    log_nominal = rng.normal(13.0, 1.5, size=n)
    nominal = np.exp(log_nominal)
    nominal = np.maximum(nominal, 10_000.0)  # floor 10k

    # --- 6. EAD = nominal * CCF ---
    ccf_arr = np.array([PRODUCTS[p]["ccf"] for p in product_types], dtype=float)
    ead = nominal * ccf_arr

    # Scale to match total_ead (post-CCF)
    raw_ead_total = ead.sum()
    if raw_ead_total > 0:
        scale_factor = total_ead / raw_ead_total
        nominal *= scale_factor
        ead *= scale_factor

    # --- 7. Margin deposit (5-30% of nominal) ---
    # Higher for LC, lower for guarantees; better-rated = higher margin
    imp_rating_num = np.array([_RATING_NUMERIC[r] for r in importer_ratings], dtype=float)
    base_margin = rng.uniform(0.05, 0.30, size=n)
    # LC products have higher margin deposits
    is_lc = np.isin(product_types, ["import_lc", "export_lc"]).astype(float)
    margin_adj = is_lc * 0.05 - (imp_rating_num - 3) * 0.02
    margin_pct = np.clip(base_margin + margin_adj, 0.05, 0.30)
    margin_held = nominal * margin_pct

    # --- 8. Commodity volatility (class-specific + idiosyncratic) ---
    base_vol = np.array([COMMODITIES[c]["volatility"] for c in commodity_classes], dtype=float)
    idio_noise = rng.normal(0, 0.03, size=n)
    commodity_vol = np.clip(base_vol + idio_noise, 0.05, 0.50)

    # Transit days for commodity haircut (subset of tenor)
    transit_days = np.clip(rng.normal(tenor_days * 0.5, 15), 15, tenor_days).astype(int)

    # --- 9. Multi-party features ---
    # Issuing bank rating: centered on A, capped by importer country
    issuing_raw = rng.normal(3.0, 0.8, size=n)  # centered on A=3
    issuing_num = np.clip(issuing_raw, 1, np.minimum(7, imp_rating_num + 1)).astype(int)
    _NUM_TO_RATING = {v: k for k, v in _RATING_NUMERIC.items()}
    issuing_bank_ratings = np.array([_NUM_TO_RATING[int(x)] for x in issuing_num])

    # Confirming bank rating: centered on AA (large banks)
    confirming_raw = rng.normal(2.0, 0.5, size=n)  # centered on AA=2
    confirming_num = np.clip(confirming_raw, 1, 4).astype(int)  # max BBB
    confirming_bank_ratings = np.array([_NUM_TO_RATING[int(x)] for x in confirming_num])

    # Documentary compliance: 70% compliant on first presentation
    documentary_compliance = rng.binomial(1, 0.70, size=n)

    # --- 10. Counterparty features ---
    trade_frequency = rng.poisson(lam=8, size=n)
    relationship_years = np.clip(rng.exponential(5.0, size=n), 0.5, 30.0).round(1)
    fx_exposure = rng.uniform(0.0, 1.0, size=n).round(2)  # fraction of trade in foreign currency

    # --- Build DataFrame ---
    df = pl.DataFrame({
        "trade_id": np.arange(n),
        "product_type": product_types,
        "commodity_class": commodity_classes,
        "exporter_country_rating": exporter_ratings,
        "importer_country_rating": importer_ratings,
        "tenor_days": tenor_days,
        "tenor_years": np.round(tenor_years, 4),
        "nominal": np.round(nominal, 2),
        "ccf": ccf_arr,
        "ead": np.round(ead, 2),
        "margin_pct": np.round(margin_pct, 4),
        "margin_held": np.round(margin_held, 2),
        "commodity_vol": np.round(commodity_vol, 4),
        "transit_days": transit_days,
        "issuing_bank_rating": issuing_bank_ratings,
        "confirming_bank_rating": confirming_bank_ratings,
        "documentary_compliance": documentary_compliance,
        "trade_frequency": trade_frequency,
        "relationship_years": relationship_years,
        "fx_exposure": fx_exposure,
    })

    # --- Compute PD, LGD, maturity adjustment ---
    pd_position = compute_trade_finance_pd(df)
    lgd_position = compute_trade_finance_lgd(df)
    df = df.with_columns(pl.Series("pd_position", pd_position))
    df = df.with_columns(pl.Series("lgd_position", lgd_position))
    df = df.with_columns(pl.Series("maturity_adjustment", compute_maturity_adjustment(df)))

    # Default flag: stochastic based on PD
    df = df.with_columns(pl.Series("default_flag", rng.binomial(1, df["pd_position"].to_numpy())))

    # RW position-level (CRR3 Art. 111)
    df = df.with_columns(pl.Series("rw_crr3", compute_trade_finance_rw(df)))

    return df


# ──────────────────────────────────────────────
# PD MULTI-COMPOSANTE (SPECIFIQUE TF)
# ──────────────────────────────────────────────

def compute_trade_finance_pd(df: pl.DataFrame) -> np.ndarray:
    """Compute position-level PD via TF multi-component model.

    PD_TF = PD_counterparty * alpha_product * alpha_tenor
             + PD_transfer * (1 - confirmation_coverage)

    For confirmed export LC: joint default via Gaussian copula.
    Clip: [0.0003, 0.10].
    """
    n = len(df)
    product_types = df["product_type"].to_numpy()

    # --- PD counterparty ---
    imp_rating_arr = df["importer_country_rating"].to_numpy()
    imp_rating_num = np.array([
        _RATING_NUMERIC[r] for r in imp_rating_arr
    ], dtype=float)
    freq_adj = np.log1p(df["trade_frequency"].to_numpy()) / 3.0  # normalize
    score = _BETA_0 + _BETA_RATING * imp_rating_num + _BETA_FREQ * freq_adj
    pd_counterparty = expit(score)

    # --- Alpha product ---
    alpha_product = np.array([
        PRODUCTS[p]["alpha_product"] for p in product_types
    ], dtype=float)

    # --- Alpha tenor (sub-linear, reflects auto-liquidation) ---
    tenor_years = df["tenor_years"].to_numpy()
    alpha_tenor = np.power(np.maximum(tenor_years, 0.01) / 1.0, 0.7)

    # --- PD transfer (country risk) ---
    pd_sovereign = np.array([
        COUNTRY_RATINGS[r]["pd_sovereign"]
        for r in imp_rating_arr
    ], dtype=float)
    tc_factor = np.array([
        COUNTRY_RATINGS[r]["tc_factor"]
        for r in imp_rating_arr
    ], dtype=float)
    pd_transfer = pd_sovereign * tc_factor

    # --- Confirmation coverage ---
    # Only export_lc has confirmation coverage
    is_export_lc = (product_types == "export_lc")
    confirmation_coverage = np.where(is_export_lc, 1.0, 0.0)

    # --- Base PD ---
    pd_base = (
        pd_counterparty * alpha_product * alpha_tenor
        + pd_transfer * (1.0 - confirmation_coverage)
    )

    # --- Joint default for confirmed export LC (Gaussian copula) ---
    # PD_joint = Phi_2(G(PD_issuing), G(PD_confirming); rho)
    # Approximation: PD_joint ~ PD_issuing * PD_confirming / (1 - rho * ...)
    # Simplified: PD_joint = PD_issuing * PD_confirming * (1 + rho)
    # This is conservative but captures the credit enhancement.
    if is_export_lc.any():
        issuing_rating_arr = df["issuing_bank_rating"].to_numpy()
        confirming_rating_arr = df["confirming_bank_rating"].to_numpy()
        exporter_rating_arr = df["exporter_country_rating"].to_numpy()

        issuing_num = np.array([
            _RATING_NUMERIC[r] for r in issuing_rating_arr
        ], dtype=float)
        confirming_num = np.array([
            _RATING_NUMERIC[r] for r in confirming_rating_arr
        ], dtype=float)

        # PD for issuing and confirming banks
        pd_issuing = expit(-5.0 + 0.50 * issuing_num)
        pd_confirming = expit(-5.0 + 0.50 * confirming_num)

        # Correlation: same country vs different country
        same_country = (
            exporter_rating_arr == imp_rating_arr
        )
        rho = np.where(same_country, _RHO_SAME_COUNTRY, _RHO_DIFF_COUNTRY)

        # Joint PD approximation (bivariate normal lower bound)
        pd_joint = pd_issuing * pd_confirming * (1.0 + rho)

        # For export_lc, PD is the joint default probability
        # (both issuing and confirming must fail)
        pd_base[is_export_lc] = pd_joint[is_export_lc]

    return np.clip(pd_base, 0.0003, 0.10)


# ──────────────────────────────────────────────
# LGD RECOVERY WATERFALL (SPECIFIQUE TF)
# ──────────────────────────────────────────────

def compute_trade_finance_lgd(df: pl.DataFrame) -> np.ndarray:
    """Compute position-level LGD via TF recovery waterfall.

    LGD = 1 - min(1.0, RR_goods + RR_margin + RR_confirmation + RR_eca)

    Product-specific floors: LC=0.10, guarantee=0.25, ECA=0.05.
    Cap: 0.80.
    """
    n = len(df)
    product_types = df["product_type"].to_numpy()
    ead = np.maximum(df["ead"].to_numpy(), 1.0)

    # --- RR_goods: recovery on goods in transit ---
    has_goods = np.array([
        PRODUCTS[p]["has_goods"] for p in product_types
    ], dtype=float)
    commodity_vol = df["commodity_vol"].to_numpy()
    transit_days = df["transit_days"].to_numpy()
    h_liquidity = np.array([
        COMMODITIES[c]["h_liquidity"] for c in df["commodity_class"].to_numpy()
    ], dtype=float)

    # Commodity haircut: H = z_99 * sigma * sqrt(T/252) + H_liquidity
    h_commodity = _Z99 * commodity_vol * np.sqrt(transit_days / 252.0) + h_liquidity
    h_commodity = np.clip(h_commodity, 0.0, 0.90)

    # Value of goods ~ nominal (for LC with documents of title)
    nominal = df["nominal"].to_numpy()
    v_goods = nominal * (1.0 - h_commodity)
    rr_goods = has_goods * v_goods / np.maximum(ead, 1.0)
    rr_goods = np.clip(rr_goods, 0.0, 1.0)

    # --- RR_margin: cash margin deposit ---
    margin_held = df["margin_held"].to_numpy()
    rr_margin = margin_held / np.maximum(ead, 1.0)
    rr_margin = np.clip(rr_margin, 0.0, 1.0)

    # --- RR_confirmation: recovery via confirming bank ---
    is_confirmed = np.array([
        PRODUCTS[p]["has_confirmation"] for p in product_types
    ], dtype=float)
    confirming_num = np.array([
        _RATING_NUMERIC[r] for r in df["confirming_bank_rating"].to_numpy()
    ], dtype=float)
    pd_confirming = expit(-5.0 + 0.50 * confirming_num)
    coverage_ratio = 0.90  # confirming bank covers ~90% of exposure
    rr_confirmation = is_confirmed * (1.0 - pd_confirming) * coverage_ratio

    # --- RR_eca: ECA coverage ---
    is_eca = np.array([
        PRODUCTS[p]["is_eca"] for p in product_types
    ], dtype=float)
    rr_eca = is_eca * _ECA_COVERAGE

    # --- Total recovery ---
    total_rr = rr_goods + rr_margin + rr_confirmation + rr_eca
    total_rr = np.minimum(total_rr, 1.0)

    lgd = 1.0 - total_rr

    # --- Country LGD adjustment ---
    lgd_adj = np.array([
        COUNTRY_RATINGS[r]["lgd_adj"]
        for r in df["importer_country_rating"].to_numpy()
    ], dtype=float)
    lgd = lgd + lgd_adj

    # --- Product-specific floors ---
    lgd_floor = np.array([
        PRODUCTS[p]["lgd_floor"] for p in product_types
    ], dtype=float)
    lgd = np.maximum(lgd, lgd_floor)

    # Cap
    lgd = np.minimum(lgd, 0.80)

    return np.round(lgd, 4)


# ──────────────────────────────────────────────
# EAD avec CCF (deja calcule dans generate, expose pour tests)
# ──────────────────────────────────────────────

def compute_ead_with_ccf(df: pl.DataFrame) -> np.ndarray:
    """Compute EAD = nominal * CCF (CRR3 Art. 111, Annex I).

    Exposed for independent testing; the generator already computes this.
    """
    ccf_arr = np.array([
        PRODUCTS[p]["ccf"] for p in df["product_type"].to_numpy()
    ], dtype=float)
    return df["nominal"].to_numpy() * ccf_arr


# ──────────────────────────────────────────────
# MATURITY ADJUSTMENT (Basel IRB, BCBS205 waiver)
# ──────────────────────────────────────────────

def compute_maturity_adjustment(df: pl.DataFrame) -> np.ndarray:
    """Compute Basel IRB maturity adjustment.

    b = (0.11852 - 0.05478 * ln(PD))^2
    MA = (1 + (M - 2.5) * b) / (1 - 1.5 * b)

    For TF: M = tenor reel (no 1-year floor, BCBS205 waiver).
    """
    pd_vals = np.clip(df["pd_position"].to_numpy(), 1e-6, 1.0)
    m = df["tenor_years"].to_numpy()

    b = (0.11852 - 0.05478 * np.log(pd_vals)) ** 2
    ma = (1.0 + (m - 2.5) * b) / (1.0 - 1.5 * b)

    return np.round(ma, 4)


# ──────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────

def aggregate_trade_finance_to_balance_row(
    df_positions: pl.DataFrame,
    profile: "AssetClassProfile",
) -> dict:
    """Aggregate individual TF positions into a single balance sheet row.

    Returns a dict compatible with the df_balance_sheet schema.
    PD and LGD are EAD-weighted averages (EAD = post-CCF).

    Args:
        df_positions: DataFrame from generate_trade_finance_positions().
        profile: AssetClassProfile for trade_finance.

    Returns:
        Dict with all balance sheet columns.
    """
    weights = df_positions["ead"].to_numpy()
    total_ead = weights.sum()

    if total_ead <= 0:
        pd_agg = profile.pd_base
        lgd_agg = profile.lgd_base
        tenor_agg = profile.tenor
    else:
        w = weights / total_ead
        pd_agg = float(np.dot(w, df_positions["pd_position"].to_numpy()))
        lgd_agg = float(np.dot(w, df_positions["lgd_position"].to_numpy()))
        tenor_agg = float(np.dot(w, df_positions["tenor_years"].to_numpy()))

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
