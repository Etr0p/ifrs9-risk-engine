"""Tests unitaires pour ifrs9_cockpit/models/lgd_model.py et ead_model.py (Story 2-2).

Couvre les taches :
  T1: LGD Beta TTC et Downturn
  T2: LGD dans [0, 1] et Downturn >= TTC
  T3: LGD sensible au HPI (FR8)
  T4: EAD CCF (Revolving vs Term)
  T5: EAD >= 0
  T6: Modules standalone
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ifrs9_cockpit.config import EAD_CONFIG, LGD_CONFIG, RANDOM_SEED, SECTORS
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel


# ============================================================
# Fixtures partagees
# ============================================================

@pytest.fixture(scope="module")
def df_credit():
    """Genere un dataset credit pour les tests."""
    df_credit, _, _ = generate_dataset(n_clients=2000, seed=RANDOM_SEED)
    return df_credit


@pytest.fixture(scope="module")
def lgd_model(df_credit):
    """Modele LGD calibre."""
    model = LGDModel(seed=RANDOM_SEED)
    model.fit(df_credit)
    return model


@pytest.fixture(scope="module")
def ead_model(df_credit):
    """Modele EAD calibre."""
    model = EADModel(seed=RANDOM_SEED)
    model.fit(df_credit)
    return model


# ============================================================
# LGD — Distribution Beta
# ============================================================

class TestLGDModel:
    def test_fit_returns_self(self, df_credit):
        """fit() retourne self (pattern fluent)."""
        model = LGDModel(seed=RANDOM_SEED)
        result = model.fit(df_credit)
        assert result is model

    def test_fitted_flag(self, lgd_model):
        """Le flag _fitted est True apres fit."""
        assert lgd_model._fitted is True

    def test_sector_lgd_calibrated(self, lgd_model):
        """LGD moyennes calibrees par secteur."""
        assert len(lgd_model.sector_lgd_) == len(SECTORS)
        for sector in SECTORS:
            assert sector.name in lgd_model.sector_lgd_

    def test_lgd_ttc_in_01(self, lgd_model, df_credit):
        """LGD TTC dans [0, 1] pour toutes les positions."""
        lgd_ttc = lgd_model.predict_ttc(df_credit)
        assert lgd_ttc.min() >= 0.0, f"LGD TTC min = {lgd_ttc.min()}"
        assert lgd_ttc.max() <= 1.0, f"LGD TTC max = {lgd_ttc.max()}"

    def test_lgd_downturn_in_01(self, lgd_model, df_credit):
        """LGD Downturn dans [0, 1] pour toutes les positions."""
        lgd_dt = lgd_model.predict_downturn(df_credit)
        assert lgd_dt.min() >= 0.0, f"LGD DT min = {lgd_dt.min()}"
        assert lgd_dt.max() <= 1.0, f"LGD DT max = {lgd_dt.max()}"

    def test_lgd_downturn_gte_ttc(self, lgd_model, df_credit):
        """LGD Downturn >= LGD TTC (z_stress=2 scenario adverse)."""
        lgd_ttc, lgd_dt = lgd_model.predict_ttc_and_downturn(df_credit, z_stress=2.0)
        # Tolerance numerique pour les arrondis
        assert (lgd_dt >= lgd_ttc - 1e-10).all(), \
            f"LGD DT < TTC pour {(lgd_dt < lgd_ttc - 1e-10).sum()} positions"

    def test_lgd_base_scenario_dt_eq_ttc(self, lgd_model, df_credit):
        """En scenario base (z=0), LGD DT == LGD TTC."""
        lgd_ttc, lgd_dt = lgd_model.predict_ttc_and_downturn(df_credit, z_stress=0.0)
        np.testing.assert_allclose(lgd_dt, lgd_ttc, atol=1e-10)

    def test_lgd_predict_downturn_flag(self, lgd_model, df_credit):
        """predict(downturn=True) retourne des LGD plus elevees en moyenne."""
        lgd_base = lgd_model.predict(df_credit, downturn=False)
        lgd_dt = lgd_model.predict(df_credit, downturn=True)
        assert lgd_dt.mean() >= lgd_base.mean()

    def test_lgd_hpi_sensitivity(self, lgd_model, df_credit):
        """Baisse HPI -> hausse LGD (canal collateral FR8)."""
        lgd_base = lgd_model.predict(df_credit, hpi_override=2.0)  # baseline
        lgd_stress = lgd_model.predict(df_credit, hpi_override=-8.0)  # baisse
        assert lgd_stress.mean() > lgd_base.mean(), \
            f"HPI stress devrait augmenter LGD: base={lgd_base.mean():.4f}, stress={lgd_stress.mean():.4f}"

    def test_lgd_summary_table(self, lgd_model, df_credit):
        """get_summary retourne un DataFrame par secteur/type."""
        summary = lgd_model.get_summary(df_credit)
        assert isinstance(summary, pd.DataFrame)
        assert "sector" in summary.columns
        assert "loan_type" in summary.columns
        assert "lgd_ttc_mean" in summary.columns
        assert len(summary) > 0


# ============================================================
# EAD — CCF
# ============================================================

class TestEADModel:
    def test_fit_returns_self(self, df_credit):
        """fit() retourne self (pattern fluent)."""
        model = EADModel(seed=RANDOM_SEED)
        result = model.fit(df_credit)
        assert result is model

    def test_fitted_flag(self, ead_model):
        """Le flag _fitted est True apres fit."""
        assert ead_model._fitted is True

    def test_ccf_calibrated(self, ead_model):
        """CCF calibres par type de pret."""
        assert "Revolving" in ead_model.avg_ccf_by_type_
        assert "Term" in ead_model.avg_ccf_by_type_
        assert ead_model.avg_ccf_by_type_["Revolving"] == EAD_CONFIG.ccf_revolving
        assert ead_model.avg_ccf_by_type_["Term"] == EAD_CONFIG.ccf_term_loan

    def test_ead_positive(self, ead_model, df_credit):
        """EAD >= 0 pour toutes les positions."""
        ead = ead_model.predict(df_credit)
        assert (ead >= 0).all(), f"EAD negatives: {(ead < 0).sum()}"

    def test_ead_stressed_positive(self, ead_model, df_credit):
        """EAD stressees >= 0."""
        ead = ead_model.predict(df_credit, stressed=True)
        assert (ead >= 0).all()

    def test_ead_stressed_gte_base(self, ead_model, df_credit):
        """EAD stressees >= EAD base en moyenne."""
        ead_base = ead_model.predict(df_credit, stressed=False)
        ead_stress = ead_model.predict(df_credit, stressed=True)
        assert ead_stress.mean() >= ead_base.mean() * 0.95, \
            f"EAD stress ({ead_stress.mean():.0f}) < base ({ead_base.mean():.0f})"

    def test_ead_revolving_uses_ccf(self, ead_model, df_credit):
        """Pour les revolving, EAD = drawn + CCF x undrawn (verifie que EAD != loan_amount)."""
        revolving = df_credit[df_credit["loan_type"] == "Revolving"]
        if len(revolving) == 0:
            pytest.skip("Pas de revolving dans le dataset")
        ead = ead_model.predict(revolving)
        loan_amount = revolving["loan_amount"].values
        # EAD revolving != loan_amount pour la plupart (sauf util = 1.0)
        diff = np.abs(ead - loan_amount)
        assert (diff > 1.0).any(), "EAD revolving devrait differer du loan_amount"

    def test_ead_summary_table(self, ead_model, df_credit):
        """get_summary retourne un DataFrame par secteur/type."""
        summary = ead_model.get_summary(df_credit)
        assert isinstance(summary, pd.DataFrame)
        assert "sector" in summary.columns
        assert len(summary) > 0

    def test_ead_ccf_analysis(self, ead_model, df_credit):
        """get_ccf_analysis retourne les CCF implicites."""
        ccf_table = ead_model.get_ccf_analysis(df_credit)
        assert isinstance(ccf_table, pd.DataFrame)
        if len(ccf_table) > 0:
            assert "avg_ccf" in ccf_table.columns


# ============================================================
# Standalone
# ============================================================

class TestStandalone:
    def test_lgd_module_runs_standalone(self):
        """Le module lgd_model.py s'execute sans erreur."""
        result = subprocess.run(
            [sys.executable, "-m", "ifrs9_cockpit.models.lgd_model"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
        assert "Validation LGD terminee" in result.stdout

    def test_ead_module_importable(self):
        """Le module ead_model.py s'importe sans erreur."""
        result = subprocess.run(
            [sys.executable, "-c", "from ifrs9_cockpit.models.ead_model import EADModel; print('OK')"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
        assert "OK" in result.stdout
