"""Tests unitaires pour ifrs9_cockpit/data/generator.py (Story 1-2).

Couvre les 9 tâches :
  T1: SyntheticDataGenerator retourne 3-tuple
  T2: Features communes entreprise
  T3: Features spécifiques crédit
  T4: Features spécifiques PE
  T5: Default flag via modèle latent
  T6: Historique panel 12 mois vectorisé
  T7: Contrats DataFrame AR4
  T8: Réalisme (PD, multiples, corrélations, proportions)
  T9: Utilitaires et reproductibilité
"""

from __future__ import annotations

import hashlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.shared

from ifrs9_cockpit.config import (
    MACRO_HISTORY_BASELINE,
    N_CLIENTS,
    N_MONTHS,
    RANDOM_SEED,
    REQUIRED_CREDIT_COLS,
    REQUIRED_PE_COLS,
    SECTORS,
    SECTOR_NAMES,
)
from ifrs9_cockpit.data.generator import SyntheticDataGenerator, generate_dataset


# Fixture partagée — petit dataset pour les tests rapides
@pytest.fixture(scope="module")
def dataset():
    gen = SyntheticDataGenerator(n_clients=2000, seed=RANDOM_SEED)
    return gen.generate()


@pytest.fixture(scope="module")
def df_credit(dataset):
    return dataset[0]


@pytest.fixture(scope="module")
def df_pe(dataset):
    return dataset[1]


@pytest.fixture(scope="module")
def df_history(dataset):
    return dataset[2]


# ============================================================
# T1 — SyntheticDataGenerator retourne 3-tuple
# ============================================================

class TestGeneratorSignature:
    def test_returns_3_tuple(self, dataset):
        assert len(dataset) == 3

    def test_all_dataframes(self, dataset):
        for df in dataset:
            assert isinstance(df, pd.DataFrame)

    def test_rng_isolated(self):
        gen = SyntheticDataGenerator(seed=42)
        assert hasattr(gen, "rng")
        assert isinstance(gen.rng, np.random.Generator)


# ============================================================
# T2 — Features communes entreprise
# ============================================================

class TestCommonFeatures:
    def test_enterprise_id_unique(self, df_credit):
        assert df_credit["enterprise_id"].nunique() == len(df_credit)

    def test_sector_in_config(self, df_credit):
        assert set(df_credit["sector"].unique()).issubset(set(SECTOR_NAMES))

    def test_revenue_positive(self, df_credit):
        assert (df_credit["revenue"].dropna() > 0).all()

    def test_ebitda_exists(self, df_credit):
        assert "ebitda" in df_credit.columns

    def test_debt_ratio_bounded(self, df_credit):
        dr = df_credit["debt_ratio"]
        assert (dr >= 0).all() and (dr <= 1).all()

    def test_shared_between_credit_and_pe(self, df_credit, df_pe):
        """Enterprise IDs et secteurs identiques entre crédit et PE.

        Note: revenue peut diverger pour ~0.5% d'outliers injectés post-génération.
        """
        common_ids = set(df_credit["enterprise_id"]) & set(df_pe["enterprise_id"])
        assert len(common_ids) == len(df_credit)
        # Secteurs identiques
        merged = df_credit[["enterprise_id", "sector"]].merge(
            df_pe[["enterprise_id", "sector"]], on="enterprise_id", suffixes=("_c", "_p")
        )
        assert (merged["sector_c"] == merged["sector_p"]).all()


# ============================================================
# T3 — Features spécifiques crédit
# ============================================================

class TestCreditFeatures:
    def test_credit_score_range(self, df_credit):
        cs = df_credit["credit_score"]
        assert (cs >= 300).all() and (cs <= 850).all()

    def test_dpd_non_negative(self, df_credit):
        assert (df_credit["dpd"] >= 0).all()

    def test_loan_amount_positive(self, df_credit):
        assert (df_credit["loan_amount"] > 0).all()

    def test_loan_type_values(self, df_credit):
        assert set(df_credit["loan_type"].unique()).issubset({"Revolving", "Term"})

    def test_utilization_rate_bounded(self, df_credit):
        ur = df_credit["utilization_rate"].dropna()
        assert (ur >= 0).all() and (ur <= 1).all()

    def test_collateral_mostly_positive(self, df_credit):
        coll = df_credit["collateral"].dropna()
        assert (coll > 0).all()


# ============================================================
# T4 — Features spécifiques PE
# ============================================================

class TestPEFeatures:
    def test_entry_multiple_positive(self, df_pe):
        assert (df_pe["entry_multiple"] > 0).all()

    def test_leverage_bounded(self, df_pe):
        lev = df_pe["leverage"]
        assert (lev >= 0).all() and (lev <= 1).all()

    def test_holding_years_positive(self, df_pe):
        assert (df_pe["holding_years"] >= 1).all()

    def test_vintage_range(self, df_pe):
        assert (df_pe["vintage"] >= 2018).all()
        assert (df_pe["vintage"] <= 2025).all()

    def test_valuation_method_from_config(self, df_pe):
        valid_methods = {s.valuation_method for s in SECTORS}
        assert set(df_pe["valuation_method"].unique()).issubset(valid_methods)

    def test_multiples_in_sector_range(self, df_pe):
        """Multiples PE dans les fourchettes IPEV par secteur."""
        for sector in SECTORS:
            mask = df_pe["sector"] == sector.name
            multiples = df_pe.loc[mask, "entry_multiple"]
            lo, hi = sector.entry_multiple_range
            assert (multiples >= lo - 0.01).all(), f"{sector.name} min: {multiples.min()}"
            assert (multiples <= hi + 0.01).all(), f"{sector.name} max: {multiples.max()}"


# ============================================================
# T5 — Default flag via modèle latent
# ============================================================

class TestDefaultFlag:
    def test_binary(self, df_credit):
        assert set(df_credit["default_flag"].unique()).issubset({0, 1})

    def test_pd_latent_bounded(self, df_credit):
        pd_lat = df_credit["pd_latent"]
        assert (pd_lat >= 0).all() and (pd_lat <= 1).all()

    def test_pd_origination_exists(self, df_credit):
        assert "pd_origination" in df_credit.columns
        assert (df_credit["pd_origination"] >= 0).all()

    def test_global_default_rate_reasonable(self, df_credit):
        dr = df_credit["default_flag"].mean()
        assert 0.005 <= dr <= 0.08, f"Taux de défaut global: {dr:.2%}"

    def test_tech_higher_default_rate(self, df_credit):
        """Technologie doit avoir un taux de défaut élevé (poche de vulnérabilité)."""
        tech_dr = df_credit[df_credit["sector"] == "Technologie"]["default_flag"].mean()
        sante_dr = df_credit[df_credit["sector"] == "Sante"]["default_flag"].mean()
        assert tech_dr > sante_dr


# ============================================================
# T6 — Historique panel vectorisé
# ============================================================

class TestHistoryPanel:
    def test_shape(self, df_credit, df_history):
        expected = len(df_credit) * N_MONTHS
        assert len(df_history) == expected

    def test_months_1_to_12(self, df_history):
        assert set(df_history["month"].unique()) == set(range(1, N_MONTHS + 1))

    def test_balance_non_negative(self, df_history):
        assert (df_history["balance"] >= 0).all()

    def test_dpd_non_negative(self, df_history):
        assert (df_history["dpd"] >= 0).all()

    def test_macro_variables_present(self, df_history):
        for var in ["gdp_growth", "unemployment_rate", "interest_rate",
                     "hpi_growth", "inflation_rate"]:
            assert var in df_history.columns

    def test_enterprise_ids_match(self, df_credit, df_history):
        assert set(df_history["enterprise_id"].unique()) == set(df_credit["enterprise_id"])


# ============================================================
# T7 — Contrats DataFrame AR4
# ============================================================

class TestContracts:
    def test_credit_cols(self, df_credit):
        assert REQUIRED_CREDIT_COLS.issubset(set(df_credit.columns))

    def test_pe_cols(self, df_pe):
        assert REQUIRED_PE_COLS.issubset(set(df_pe.columns))

    def test_enterprise_ids_match(self, df_credit, df_pe):
        assert set(df_credit["enterprise_id"]) == set(df_pe["enterprise_id"])


# ============================================================
# T8 — Réalisme
# ============================================================

class TestRealism:
    def test_sector_proportions(self, df_credit):
        """Proportions sectorielles ±5% de la config."""
        for sector in SECTORS:
            obs = (df_credit["sector"] == sector.name).mean()
            assert abs(obs - sector.proportion) < 0.05, \
                f"{sector.name}: obs={obs:.2%} vs cfg={sector.proportion:.0%}"

    def test_correlation_credit_score_debt_ratio(self, df_credit):
        corr = df_credit[["credit_score", "debt_ratio"]].corr().iloc[0, 1]
        assert corr < 0, f"Corr credit_score/debt_ratio devrait être négative: {corr:.3f}"

    def test_correlation_revenue_ebitda(self, df_credit):
        corr = df_credit[["revenue", "ebitda"]].corr().iloc[0, 1]
        assert corr > 0.5, f"Corr revenue/ebitda devrait être > 0.5: {corr:.3f}"

    def test_default_rate_per_sector(self, df_credit):
        """PD par secteur entre 1% et 20%."""
        for sector in SECTORS:
            mask = df_credit["sector"] == sector.name
            dr = df_credit.loc[mask, "default_flag"].mean()
            assert 0.005 <= dr <= 0.20, f"{sector.name}: DR={dr:.2%}"


# ============================================================
# T9 — Utilitaires et reproductibilité
# ============================================================

class TestUtilitiesAndReproducibility:
    def test_generate_dataset_function(self):
        result = generate_dataset(n_clients=100, seed=42)
        assert len(result) == 4  # df_credit, df_pe, df_history, df_balance_sheet
        assert len(result[0]) == 100
        assert len(result[3]) == 14  # 14 asset classes

    def test_reproducibility(self):
        """Deux exécutions avec le même seed produisent le même résultat."""
        gen1 = SyntheticDataGenerator(n_clients=500, seed=42)
        gen2 = SyntheticDataGenerator(n_clients=500, seed=42)
        df1, _, _ = gen1.generate()
        df2, _, _ = gen2.generate()
        hash1 = hashlib.md5(pd.util.hash_pandas_object(df1).values.tobytes()).hexdigest()
        hash2 = hashlib.md5(pd.util.hash_pandas_object(df2).values.tobytes()).hexdigest()
        assert hash1 == hash2

    def test_different_seed_different_result(self):
        gen1 = SyntheticDataGenerator(n_clients=500, seed=42)
        gen2 = SyntheticDataGenerator(n_clients=500, seed=123)
        df1, _, _ = gen1.generate()
        df2, _, _ = gen2.generate()
        hash1 = hashlib.md5(pd.util.hash_pandas_object(df1).values.tobytes()).hexdigest()
        hash2 = hashlib.md5(pd.util.hash_pandas_object(df2).values.tobytes()).hexdigest()
        assert hash1 != hash2

    def test_standalone_runs(self):
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "ifrs9_cockpit.data.generator"],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Generation valide" in result.stdout
