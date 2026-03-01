"""Pipeline de Generation de Donnees Synthetiques Financieres v4.5."""

import numpy as np
import polars as pl

from ifrs9_cockpit.synthetic_generator.constants import *  # noqa: F401,F403
from ifrs9_cockpit.synthetic_generator.constants import (
    TARGET, LATENT_PD, NOISE_FEATURES,
    _SECTOR_MAP, _BRIDGE_CREDIT_SCORE, _BRIDGE_REVOLVING_PROB,
    _BRIDGE_COLLATERAL_RATIO, _BRIDGE_LEVERAGE_PE,
)
from ifrs9_cockpit.synthetic_generator.generator import AdvancedFinancialGenerator  # noqa: F401
from ifrs9_cockpit.synthetic_generator.evaluator import RobustEvaluator  # noqa: F401
from ifrs9_cockpit.synthetic_generator.mortgage_positions import (  # noqa: F401
    generate_mortgage_positions,
    aggregate_mortgage_to_balance_row,
)
from ifrs9_cockpit.synthetic_generator.consumer_positions import (  # noqa: F401
    generate_consumer_positions,
    aggregate_consumer_to_balance_row,
)
from ifrs9_cockpit.synthetic_generator.trade_finance_positions import (  # noqa: F401
    generate_trade_finance_positions,
    aggregate_trade_finance_to_balance_row,
)
from ifrs9_cockpit.synthetic_generator.project_finance_positions import (  # noqa: F401
    generate_project_finance_positions,
    aggregate_project_finance_to_balance_row,
    stress_project_finance_positions,
)
from ifrs9_cockpit.synthetic_generator.securitisation_positions import (  # noqa: F401
    generate_securitisation_positions,
    aggregate_securitisation_to_balance_row,
    stress_securitisation_positions,
)
from ifrs9_cockpit.synthetic_generator.sovereign_positions import (  # noqa: F401
    generate_sovereign_positions,
    aggregate_sovereign_to_balance_row,
    stress_sovereign_positions,
)
from ifrs9_cockpit.synthetic_generator.covered_bonds_positions import (  # noqa: F401
    generate_covered_bonds_positions,
    aggregate_covered_bonds_to_balance_row,
    stress_covered_bonds_positions,
)
from ifrs9_cockpit.synthetic_generator.interbank_positions import (  # noqa: F401
    generate_interbank_positions,
    aggregate_interbank_to_balance_row,
    stress_interbank_positions,
)
from ifrs9_cockpit.synthetic_generator.equity_positions import (  # noqa: F401
    generate_equity_positions,
    aggregate_equity_to_balance_row,
    stress_equity_positions,
)
from ifrs9_cockpit.synthetic_generator.corporate_bonds_positions import (  # noqa: F401
    generate_corporate_bonds_positions,
    aggregate_corporate_bonds_to_balance_row,
    stress_corporate_bonds_positions,
)
from ifrs9_cockpit.synthetic_generator.repos_sft_positions import (  # noqa: F401
    generate_repo_positions,
    aggregate_repo_to_balance_row,
    stress_repo_positions,
)
from ifrs9_cockpit.synthetic_generator.derivatives_cva_positions import (  # noqa: F401
    generate_derivative_positions,
    aggregate_derivatives_to_balance_row,
    stress_derivative_positions,
)
from ifrs9_cockpit.utils.dataset_bundle import DatasetBundle

__all__ = [
    "AdvancedFinancialGenerator", "RobustEvaluator", "generate_dataset",
    "generate_mortgage_positions", "aggregate_mortgage_to_balance_row",
    "generate_consumer_positions", "aggregate_consumer_to_balance_row",
    "generate_trade_finance_positions", "aggregate_trade_finance_to_balance_row",
    "generate_project_finance_positions", "aggregate_project_finance_to_balance_row",
    "stress_project_finance_positions",
    "generate_securitisation_positions", "aggregate_securitisation_to_balance_row",
    "stress_securitisation_positions",
    "generate_sovereign_positions", "aggregate_sovereign_to_balance_row",
    "stress_sovereign_positions",
    "generate_covered_bonds_positions", "aggregate_covered_bonds_to_balance_row",
    "stress_covered_bonds_positions",
    "generate_interbank_positions", "aggregate_interbank_to_balance_row",
    "stress_interbank_positions",
    "generate_equity_positions", "aggregate_equity_to_balance_row",
    "stress_equity_positions",
    "generate_corporate_bonds_positions", "aggregate_corporate_bonds_to_balance_row",
    "stress_corporate_bonds_positions",
    "generate_repo_positions", "aggregate_repo_to_balance_row",
    "stress_repo_positions",
    "generate_derivative_positions", "aggregate_derivatives_to_balance_row",
    "stress_derivative_positions",
    "DatasetBundle",
    "TARGET", "LATENT_PD", "NOISE_FEATURES",
]


def generate_dataset(
    n_clients: int = 30_000,
    seed: int = 123,
    noise_sigma: float = 7.0,
    base_default_rate: float = 0.05,
) -> DatasetBundle:
    """Bridge v4.5 -> pipeline IFRS 9.

    Genere les donnees via le DGP v4.5 puis transforme le DataFrame brut
    en DatasetBundle (df_credit, df_pe, df_history, df_balance_sheet +
    8 positions par classe d'actif) compatible avec le cockpit.

    Le DGP n'est PAS modifie : les colonnes passives sont calculees APRES
    avec un RNG separe (seed + 100/200/300).

    Le df_balance_sheet (14 lignes) est une vue parametrique de la balance
    bancaire complete (14 classes d'actifs, 3 niveaux de profondeur).
    Les classes Level 1 (corporate, PE, equities, corporate_bonds) sont
    agregees depuis les donnees position par position ; les classes Level 2/3
    sont generees parametriquement a partir des AssetClassProfile.

    Args:
        n_clients: Nombre d'entreprises.
        seed: Graine aleatoire.
        noise_sigma: Ecart-type bruit DGP (7.0 -> AUC ~0.82).
        base_default_rate: Taux de defaut cible.

    Returns:
        DatasetBundle with (df_credit, df_pe, df_history, df_balance_sheet)
        and 8 position DataFrames.
    """
    # Lazy import config to avoid circular dependencies at module level
    from ifrs9_cockpit.config import (
        ALLOWED_SECTORS,
        ASSET_CLASSES,
        MACRO_HISTORY_BASELINE as _MACRO_HIST,
        REQUIRED_CREDIT_COLS,
        REQUIRED_PE_COLS,
        SECTORS as _PROD_SECTORS,
    )

    # -- 1. Run DGP v4.5 (returns pl.DataFrame, convert to pandas for manipulation) --
    gen = AdvancedFinancialGenerator(
        n_rows=n_clients,
        seed=seed,
        noise_sigma=noise_sigma,
        base_default_rate=base_default_rate,
        missing_rate=0.0,  # no MNAR for production pipeline
    )
    import pandas as pd
    df_raw_pl = gen.generate()  # pl.DataFrame from generator
    df_raw_pd = df_raw_pl.to_pandas()
    n = len(df_raw_pd)

    # -- 1b. Generate Supply Chain Graph (before sector remap) --
    df_links_pl = gen.generate_supply_chain_network(df_raw_pl)
    df_links_pd = df_links_pl.to_pandas()

    # -- 2. Map sectors v4 -> production --
    sectors_mapped = np.array([
        _SECTOR_MAP.get(s, s) for s in df_raw_pd["sector"].values
    ])
    df_raw_pd["sector"] = sectors_mapped

    # -- 3. Rename columns --
    df_raw_pd.rename(columns={
        "target_default": "default_flag",
        "days_past_due": "dpd",
        "pd_latent": "pd_origination",
    }, inplace=True)

    # -- 4. Add enterprise_id --
    df_raw_pd["enterprise_id"] = np.arange(n)

    # -- 4c. Compute Network Node Features (HHI & Customer Count) --
    supplier_hhi = np.zeros(n)
    customer_count = np.zeros(n)

    if len(df_links_pd) > 0:
        suppliers = df_links_pd[df_links_pd["type"] == "is_supplier_of"].copy()
        if not suppliers.empty:
            total_purchases = suppliers.groupby("target_id")["weight"].transform("sum")
            suppliers["norm_weight"] = suppliers["weight"] / total_purchases
            suppliers["sq_weight"] = (suppliers["norm_weight"] * 100) ** 2
            hhi_series = suppliers.groupby("target_id")["sq_weight"].sum()
            hhi_dict = hhi_series.to_dict()
            for idx in range(n):
                if idx in hhi_dict:
                    supplier_hhi[idx] = hhi_dict[idx]

        clients = df_links_pd[df_links_pd["type"] == "is_client_of"]
        if not clients.empty:
            client_counts = clients.groupby("target_id").size()
            cc_dict = client_counts.to_dict()
            for idx in range(n):
                if idx in cc_dict:
                    customer_count[idx] = cc_dict[idx]

    df_raw_pd["supplier_hhi"] = supplier_hhi
    df_raw_pd["customer_count"] = customer_count

    # -- 4d. Causal Features (DGP v4.6 -- Causal Structure) --
    rng_causal = np.random.default_rng(seed + 50)

    # --- 4d.1 Hidden Confounder : management_quality ---
    _size_map = {"PME": 0, "ETI": 1, "GE": 2}
    size_numeric = np.array(
        [_size_map.get(s, 0) for s in df_raw_pd["company_size"].values], dtype=float,
    )
    size_norm = size_numeric / max(size_numeric.max(), 1e-6)
    management_quality = np.clip(
        rng_causal.normal(0.5, 0.15, size=n) + size_norm * 0.15, 0, 1,
    )
    df_raw_pd["_management_quality"] = management_quality

    # --- 4d.2 ESG Score (confounded par management_quality + size) ---
    base_esg = (
        rng_causal.normal(50, 12, size=n)
        + size_norm * 15
        + management_quality * 20
    )
    df_raw_pd["esg_score"] = np.clip(base_esg, 10, 100).round(1)

    # --- 4d.3 CATE heterogene : ESG -> PD ---
    _CAUSAL_ESG_CATE = {
        "Technologie": -0.0002,
        "Industrie": -0.0008,
        "Sante": -0.0004,
        "Immobilier": -0.0007,
        "Services": -0.0005,
    }
    esg_centered = df_raw_pd["esg_score"].values - 50.0
    sector_cate = np.array([
        _CAUSAL_ESG_CATE.get(s, -0.0004) for s in df_raw_pd["sector"].values
    ])
    causal_esg_effect = esg_centered * sector_cate
    causal_mgmt_effect = -management_quality * 0.03

    from ifrs9_cockpit.utils.helpers import logit as _logit, expit as _expit
    pd_orig = df_raw_pd["pd_origination"].values
    logit_pd = _logit(np.clip(pd_orig, 1e-6, 1 - 1e-6))
    logit_pd_causal = logit_pd + causal_esg_effect + causal_mgmt_effect
    df_raw_pd["pd_origination"] = _expit(logit_pd_causal)
    df_raw_pd["default_flag"] = rng_causal.binomial(
        1, df_raw_pd["pd_origination"].values,
    )

    # --- 4d.4 Spillover sectoriel (violation SUTVA legere) ---
    sector_avg_pd = df_raw_pd.groupby("sector")["pd_origination"].transform("mean")
    spillover_effect = (sector_avg_pd - df_raw_pd["pd_origination"].mean()) * 0.10
    logit_pd_spill = _logit(np.clip(df_raw_pd["pd_origination"].values, 1e-6, 1 - 1e-6))
    df_raw_pd["pd_origination"] = _expit(logit_pd_spill + spillover_effect)

    # --- 4d.5 Time-to-default (survival outcome) ---
    pd_final = df_raw_pd["pd_origination"].values
    hazard = -np.log(np.clip(1 - pd_final, 1e-6, 1))
    raw_ttd = rng_causal.exponential(1.0 / np.clip(hazard, 1e-6, None))
    horizon = 5.0
    df_raw_pd["time_to_default"] = np.round(np.minimum(raw_ttd, horizon), 4)
    df_raw_pd["event_observed"] = (raw_ttd <= horizon).astype(int)

    # --- 4d.6 Ground Truth CATE (pour validation) ---
    df_raw_pd["_true_cate_esg"] = (sector_cate * 100).round(4)
    _ate_map = {s: v * 100 for s, v in _CAUSAL_ESG_CATE.items()}
    df_raw_pd["_true_ate_esg_sector"] = np.array([
        _ate_map.get(s, -0.04) for s in df_raw_pd["sector"].values
    ])

    # Bank relationship (pas de role causal -- juste features)
    df_raw_pd["bank_relationship_years"] = rng_causal.uniform(1.0, 25.0, size=n).round(1)

    # -- 5. Generate passive credit columns (separate RNG) --
    rng_credit = np.random.default_rng(seed + 100)
    sectors = df_raw_pd["sector"].values

    # EBITDA = revenue x ebitda_margin
    df_raw_pd["ebitda"] = df_raw_pd["revenue"].values * df_raw_pd["ebitda_margin"].values

    # Credit score -- normale tronquee par secteur
    credit_score = np.empty(n)
    for sector_name, (mu, sigma) in _BRIDGE_CREDIT_SCORE.items():
        mask = sectors == sector_name
        count = mask.sum()
        if count > 0:
            raw_cs = rng_credit.normal(mu, sigma, count)
            credit_score[mask] = np.clip(raw_cs, 300, 850).astype(int)
    df_raw_pd["credit_score"] = credit_score.astype(int)

    # Loan type: Revolving / Term
    loan_type = np.full(n, "Term", dtype=object)
    for sector_name, prob in _BRIDGE_REVOLVING_PROB.items():
        mask = sectors == sector_name
        count = mask.sum()
        if count > 0:
            is_revolving = rng_credit.random(count) < prob
            loan_type[mask] = np.where(is_revolving, "Revolving", "Term")
    df_raw_pd["loan_type"] = loan_type

    # Loan amount -- revenue x multiplier
    revenue_arr = df_raw_pd["revenue"].values
    multiplier = np.where(
        loan_type == "Revolving",
        rng_credit.uniform(0.3, 1.5, n),
        rng_credit.uniform(1.0, 4.0, n),
    )
    df_raw_pd["loan_amount"] = np.round(revenue_arr * multiplier, 2)

    # Collateral -- loan_amount x ratio by sector
    collateral = np.empty(n)
    for sector_name, (lo, hi) in _BRIDGE_COLLATERAL_RATIO.items():
        mask = sectors == sector_name
        count = mask.sum()
        if count > 0:
            ratio = rng_credit.uniform(lo, hi, count)
            collateral[mask] = df_raw_pd["loan_amount"].values[mask] * ratio
    df_raw_pd["collateral"] = np.round(collateral, 2)

    # Engineered features
    df_raw_pd["loan_to_revenue"] = np.round(
        df_raw_pd["loan_amount"].values / np.clip(df_raw_pd["revenue"].values, 1.0, None), 4,
    )
    df_raw_pd["collateral_coverage"] = np.round(
        df_raw_pd["collateral"].values / np.clip(df_raw_pd["loan_amount"].values, 1.0, None), 4,
    )

    # -- 6. Build df_credit (Polars) --
    credit_cols = [
        "enterprise_id", "sector", "revenue", "ebitda", "debt_ratio",
        "credit_score", "dpd", "collateral", "loan_amount",
        "utilization_rate", "default_flag", "pd_origination",
        "loan_type", "loan_to_revenue", "collateral_coverage",
        # v4.5 DGP features (causal drivers of default)
        "ebitda_margin", "interest_coverage_ratio", "cf_volatility",
        "current_ratio", "working_capital_ratio", "net_debt_to_ebitda",
        "nb_incidents_12m", "account_age_months", "company_size",
        # Network & Causal Features
        "supplier_hhi", "customer_count", "esg_score", "bank_relationship_years",
        # Survival (Causal Engine v4.6)
        "time_to_default", "event_observed",
        # Ground truth (prefixed _ = hidden from models, used for validation)
        "_true_cate_esg", "_true_ate_esg_sector", "_management_quality",
    ]
    df_credit = pl.from_pandas(df_raw_pd[credit_cols])

    # -- 7. Build df_pe (separate RNG) --
    rng_pe = np.random.default_rng(seed + 200)
    sector_config_map = {s.name: s for s in _PROD_SECTORS}
    valuation_map = {s.name: s.valuation_method for s in _PROD_SECTORS}

    entry_multiple = np.empty(n)
    leverage_pe = np.empty(n)
    for sector_name in ALLOWED_SECTORS:
        mask = sectors == sector_name
        count = mask.sum()
        if count == 0:
            continue
        sc = sector_config_map[sector_name]
        entry_multiple[mask] = rng_pe.uniform(
            sc.entry_multiple_range[0], sc.entry_multiple_range[1], count,
        )
        lo, hi = _BRIDGE_LEVERAGE_PE[sector_name]
        leverage_pe[mask] = rng_pe.uniform(lo, hi, count)

    vintage = rng_pe.integers(2018, 2026, size=n)

    # EBITDA PE: floor a 5% du revenue
    ebitda_pe = np.maximum(df_raw_pd["ebitda"].values, df_raw_pd["revenue"].values * 0.05)

    valuation_method_arr = np.array([
        valuation_map.get(s, "dcf") for s in sectors
    ])

    df_pe = pl.DataFrame({
        "enterprise_id": np.arange(n),
        "sector": sectors,
        "revenue": df_raw_pd["revenue"].values,
        "ebitda": np.round(ebitda_pe, 2),
        "entry_multiple": np.round(entry_multiple, 2),
        "leverage": np.round(leverage_pe, 4),
        "vintage": vintage,
        "holding_years": 2026 - vintage,
        "valuation_method": valuation_method_arr,
    })

    # -- 8. Build df_history (separate RNG) --
    rng_hist = np.random.default_rng(seed + 300)
    m = 12
    total = n * m

    enterprise_ids_h = np.repeat(np.arange(n), m)
    months = np.tile(np.arange(1, m + 1), n)

    macro_gdp = np.tile(_MACRO_HIST["gdp_growth"][-m:], n)
    macro_unemp = np.tile(_MACRO_HIST["unemployment_rate"][-m:], n)
    macro_interest = np.tile(_MACRO_HIST["interest_rate"][-m:], n)
    macro_hpi = np.tile(_MACRO_HIST["hpi_growth"][-m:], n)
    macro_inflation = np.tile(_MACRO_HIST["inflation_rate"][-m:], n)

    # Balance: random walk around loan_amount (+/-5%)
    loan_amounts_h = np.repeat(df_raw_pd["loan_amount"].values, m)
    balance_noise = rng_hist.normal(0, 0.05, total)
    balances = np.maximum(0, loan_amounts_h * (0.85 + balance_noise))

    # DPD mensuel -- correle au credit_score
    credit_scores_h = np.repeat(df_raw_pd["credit_score"].values, m)
    dpd_probs = np.clip((650 - credit_scores_h) / 1000, 0, 0.3)
    has_dpd_h = rng_hist.random(total) < dpd_probs

    dpd_choices = np.array([15, 30, 60, 90, 120])
    dpd_choice_probs = np.array([0.40, 0.30, 0.15, 0.10, 0.05])
    dpd_values_h = np.where(
        has_dpd_h,
        rng_hist.choice(dpd_choices, size=total, p=dpd_choice_probs),
        0,
    )

    df_history = pl.DataFrame({
        "enterprise_id": enterprise_ids_h,
        "month": months,
        "balance": np.round(balances, 2),
        "dpd": dpd_values_h.astype(int),
        "gdp_growth": macro_gdp,
        "unemployment_rate": macro_unemp,
        "interest_rate": macro_interest,
        "hpi_growth": macro_hpi,
        "inflation_rate": macro_inflation,
    })

    # -- 9. Validate contracts --
    credit_col_set = set(df_credit.columns)
    assert REQUIRED_CREDIT_COLS <= credit_col_set, (
        f"Missing credit cols: {REQUIRED_CREDIT_COLS - credit_col_set}"
    )
    pe_col_set = set(df_pe.columns)
    assert REQUIRED_PE_COLS <= pe_col_set, (
        f"Missing PE cols: {REQUIRED_PE_COLS - pe_col_set}"
    )
    assert df_history.height == n * 12, (
        f"History shape mismatch: {df_history.height} != {n * 12}"
    )
    credit_sectors = set(df_credit["sector"].unique().to_list())
    assert credit_sectors <= ALLOWED_SECTORS, (
        f"Invalid sectors: {credit_sectors - ALLOWED_SECTORS}"
    )
    credit_loan_types = set(df_credit["loan_type"].unique().to_list())
    assert credit_loan_types <= {"Revolving", "Term"}, (
        f"Invalid loan types: {credit_loan_types}"
    )

    # -- 10. Build df_balance_sheet (14 classes d'actifs) --
    total_credit_ead = df_credit["loan_amount"].sum()
    total_bank_ead = total_credit_ead / 0.30

    # -- 10a. Generate mortgage positions (bottom-up for retail_mortgage) --
    mortgage_ead = total_bank_ead * 0.20
    df_mortgages = generate_mortgage_positions(
        n_positions=max(200, n_clients // 15),
        total_ead=mortgage_ead,
        seed=seed + 400,
    )

    # -- 10a-bis. Generate consumer positions --
    df_consumers = None
    try:
        consumer_ead = total_bank_ead * 0.07
        df_consumers = generate_consumer_positions(
            n_positions=max(500, n_clients // 6),
            total_ead=consumer_ead,
            seed=seed + 500,
        )
    except FileNotFoundError:
        pass

    # -- 10a-ter. Generate trade finance positions --
    trade_ead = total_bank_ead * 0.05
    df_trades = generate_trade_finance_positions(
        n_positions=max(500, n_clients // 6),
        total_ead=trade_ead,
        seed=seed + 600,
    )

    # -- 10a-quater. Generate project finance positions --
    project_ead = total_bank_ead * 0.05
    df_projects = generate_project_finance_positions(
        n_positions=max(100, n_clients // 60),
        total_ead=project_ead,
        seed=seed + 700,
    )

    # -- 10a-quinquies. Generate securitisation positions --
    structured_ead = total_bank_ead * 0.03
    df_securitisations = generate_securitisation_positions(
        n_tranches=max(50, n_clients // 100),
        total_ead=structured_ead,
        seed=seed + 800,
    )

    # -- 10a-sexies. Generate sovereign positions --
    sovereign_ead = total_bank_ead * 0.20
    df_sovereigns = generate_sovereign_positions(
        n_positions=max(30, n_clients // 170),
        total_ead=sovereign_ead,
        seed=seed + 900,
    )

    # -- 10a-septies. Generate covered bonds positions --
    covered_bonds_ead = total_bank_ead * 0.10
    df_covered_bonds = generate_covered_bonds_positions(
        n_positions=max(100, n_clients // 60),
        total_ead=covered_bonds_ead,
        seed=seed + 950,
    )

    # -- 10a-octies. Generate interbank positions --
    interbank_ead = total_bank_ead * 0.08
    df_interbanks = generate_interbank_positions(
        n_positions=max(200, n_clients // 20),
        total_ead=interbank_ead,
        seed=seed + 960,
    )

    # -- 10a-nonies. Generate equity positions --
    equity_ead = total_bank_ead * 0.04
    df_equities = generate_equity_positions(
        n_positions=max(30, n_clients // 100),
        total_ead=equity_ead,
        seed=seed + 1000,
    )

    # -- 10a-decies. Generate corporate bonds positions --
    corp_bonds_ead = total_bank_ead * 0.08
    df_corp_bonds = generate_corporate_bonds_positions(
        n_positions=max(50, n_clients // 60),
        total_ead=corp_bonds_ead,
        seed=seed + 1100,
    )

    # -- 10a-undecies. Generate repos/SFT positions --
    repos_ead = total_bank_ead * 0.06
    df_repos = generate_repo_positions(
        n_positions=max(50, n_clients // 60),
        total_ead=repos_ead,
        seed=seed + 1200,
    )

    # -- 10a-duodecies. Generate derivatives/CVA positions --
    df_derivatives = generate_derivative_positions(
        n_positions=max(50, n_clients // 60),
        seed=seed + 1300,
    )

    # -- 10b. Build balance rows --
    balance_rows = []
    for ac in ASSET_CLASSES:
        if ac.name == "retail_mortgage":
            row = aggregate_mortgage_to_balance_row(df_mortgages, ac)
        elif ac.name == "consumer_credit" and df_consumers is not None:
            row = aggregate_consumer_to_balance_row(df_consumers, ac)
        elif ac.name == "trade_finance":
            row = aggregate_trade_finance_to_balance_row(df_trades, ac)
        elif ac.name == "project_finance":
            row = aggregate_project_finance_to_balance_row(df_projects, ac)
        elif ac.name == "structured_products":
            row = aggregate_securitisation_to_balance_row(df_securitisations, ac)
        elif ac.name == "sovereign":
            row = aggregate_sovereign_to_balance_row(df_sovereigns, ac)
        elif ac.name == "covered_bonds":
            row = aggregate_covered_bonds_to_balance_row(df_covered_bonds, ac)
        elif ac.name == "interbank":
            row = aggregate_interbank_to_balance_row(df_interbanks, ac)
        elif ac.name == "equities":
            row = aggregate_equity_to_balance_row(df_equities, ac)
        elif ac.name == "corporate_bonds":
            row = aggregate_corporate_bonds_to_balance_row(df_corp_bonds, ac)
        elif ac.name == "repos_sft":
            row = aggregate_repo_to_balance_row(df_repos, ac)
        elif ac.name == "derivatives_cva":
            row = aggregate_derivatives_to_balance_row(df_derivatives, ac)
        elif ac.name == "corporate_loans":
            ead = total_credit_ead
            row = _build_balance_row(ac, ead)
        elif ac.name == "private_equity":
            ead = total_bank_ead * ac.typical_weight
            row = _build_balance_row(ac, ead)
        else:
            ead = total_bank_ead * ac.typical_weight
            row = _build_balance_row(ac, ead)
        balance_rows.append(row)

    df_balance_sheet = pl.DataFrame(balance_rows)

    return DatasetBundle(
        df_credit=df_credit,
        df_pe=df_pe,
        df_history=df_history,
        df_balance_sheet=df_balance_sheet,
        mortgage_positions=df_mortgages,
        consumer_positions=df_consumers,
        trade_positions=df_trades,
        project_positions=df_projects,
        securitisation_positions=df_securitisations,
        sovereign_positions=df_sovereigns,
        covered_bonds_positions=df_covered_bonds,
        interbank_positions=df_interbanks,
        equity_positions=df_equities,
        corporate_bonds_positions=df_corp_bonds,
        repos_sft_positions=df_repos,
        derivatives_cva_positions=df_derivatives,
    )


def _build_balance_row(ac: "AssetClassProfile", ead: float) -> dict:
    """Build a parametric balance sheet row from an AssetClassProfile."""
    return {
        "asset_class": ac.name,
        "label": ac.label,
        "category": ac.category,
        "ead_total": round(ead, 2),
        "typical_weight": ac.typical_weight,
        "pd_base": ac.pd_base,
        "lgd_base": ac.lgd_base,
        "tenor": ac.tenor,
        "asset_correlation": ac.asset_correlation,
        "rw_crr3": ac.rw_crr3,
        "physical_risk": ac.physical_risk,
        "transition_risk": ac.transition_risk,
        "green_capex_ratio": ac.green_capex_ratio,
        "scope3_exposure": ac.scope3_exposure,
        "absorption_buffer": ac.absorption_buffer,
        "hqla_eligible": ac.hqla_eligible,
        "hqla_level": ac.hqla_level,
        "exempt_from_staging": ac.exempt_from_staging,
        "rsf_weight": ac.rsf_weight,
    }
