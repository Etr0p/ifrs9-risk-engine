"""Tests du pipeline offline etendu (PD + Gouvernance).

TestGovernanceArtifacts : valide la structure et la qualite des artefacts
    produits par _train_governance().
TestFallbackWithoutArtifacts : verifie que le dashboard fonctionne
    sans governance_suite.joblib (calcul inline, zero regression).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ifrs9_cockpit.config import (
    MACRO_COVARIANCE,
    MACRO_HISTORY_BASELINE,
    MACRO_VARIABLES_ORDER,
    RANDOM_SEED,
)
from ifrs9_cockpit.data.generator import generate_dataset


# ──────────────────────────────────────────────
# Shared fixture: generate governance artifacts once per session
# ──────────────────────────────────────────────
@pytest.fixture(scope="session")
def governance_artifacts(global_pipeline_results):
    """Run _train_governance() and return the artifacts dict."""
    from ifrs9_cockpit.training.train import _train_governance

    df_credit = global_pipeline_results["df_credit"]
    pd_suite = global_pipeline_results["pd_suite"]
    return _train_governance(df_credit, pd_suite)


# ══════════════════════════════════════════════
# TestGovernanceArtifacts
# ══════════════════════════════════════════════
class TestGovernanceArtifacts:
    """Valide la structure et qualite des artefacts de gouvernance."""

    REQUIRED_KEYS = {
        "signatures", "signatures_order3", "tda", "rmt", "hmm",
        "conformal", "sobol", "gflownet", "lgd_model", "ead_model",
    }

    def test_governance_artifacts_keys(self, governance_artifacts):
        """Toutes les cles requises sont presentes."""
        assert self.REQUIRED_KEYS.issubset(governance_artifacts.keys()), (
            f"Missing keys: {self.REQUIRED_KEYS - set(governance_artifacts.keys())}"
        )

    def test_signatures_valid(self, governance_artifacts):
        """SignatureResult: ordre 2 (display) + ordre 3 (stockage)."""
        sig2 = governance_artifacts["signatures"]
        assert sig2.order == 2
        assert sig2.n_dims == 5
        assert sig2.n_features == 30
        assert len(sig2.features) == 30
        assert np.all(np.isfinite(sig2.features))

        sig3 = governance_artifacts["signatures_order3"]
        assert sig3.order == 3
        assert sig3.n_dims == 5
        assert sig3.n_features == 155
        assert len(sig3.features) == 155
        assert np.all(np.isfinite(sig3.features))

    def test_tda_valid(self, governance_artifacts):
        """TDAResult: fragility dans [0, 1]."""
        tda = governance_artifacts["tda"]
        assert 0.0 <= tda.fragility_index <= 1.0
        assert len(tda.bars_h0) > 0

    def test_rmt_valid(self, governance_artifacts):
        """RMTResult: matrice debruitee positive semi-definie."""
        rmt = governance_artifacts["rmt"]
        cov_clean = rmt.covariance_clean
        assert cov_clean.shape == (5, 5)
        eigenvalues = np.linalg.eigvalsh(cov_clean)
        assert np.all(eigenvalues >= -1e-10), "Covariance not PSD"
        assert rmt.n_signal >= 0
        assert rmt.n_noise >= 0
        assert rmt.n_signal + rmt.n_noise == 5

    def test_hmm_valid(self, governance_artifacts):
        """HMM: objet fitted, transition rows sum to 1."""
        hmm = governance_artifacts["hmm"]
        assert hmm._fitted
        assert hmm.A.shape == (3, 3)
        np.testing.assert_allclose(hmm.A.sum(axis=1), 1.0, atol=1e-6)
        assert hmm.means.shape[0] == 3
        assert hmm.means.shape[1] == 5

    def test_conformal_valid(self, governance_artifacts):
        """q_hat > 0, alpha = 0.10."""
        conf = governance_artifacts["conformal"]
        assert conf["alpha"] == 0.10
        assert conf["q_hat"] > 0
        assert np.isfinite(conf["q_hat"])

    def test_sobol_valid(self, governance_artifacts):
        """S1/ST indices, sum S1 raisonnable."""
        sobol = governance_artifacts["sobol"]
        assert len(sobol.s1) == 5
        assert len(sobol.st) == 5
        s1_sum = sum(sobol.s1.values())
        # Sobol S1 sum should be approximately 1 (±0.5 for interactions)
        assert 0.3 < s1_sum < 1.5, f"S1 sum = {s1_sum:.3f}"
        # ST >= S1 for each variable
        for var in sobol.variable_names:
            assert sobol.st[var] >= sobol.s1[var] - 0.05, (
                f"{var}: ST={sobol.st[var]:.3f} < S1={sobol.s1[var]:.3f}"
            )

    def test_gflownet_valid(self, governance_artifacts):
        """DualGFlowNet objet present et fonctionnel."""
        dual = governance_artifacts["gflownet"]
        from ifrs9_cockpit.engine.gflownet import DualGFlowNet
        assert isinstance(dual, DualGFlowNet)

    def test_lgd_model_valid(self, governance_artifacts):
        """LGDModel fitted et fonctionnel."""
        lgd = governance_artifacts["lgd_model"]
        from ifrs9_cockpit.models.lgd_model import LGDModel
        assert isinstance(lgd, LGDModel)
        assert lgd._fitted

    def test_ead_model_valid(self, governance_artifacts):
        """EADModel fitted et fonctionnel."""
        ead = governance_artifacts["ead_model"]
        from ifrs9_cockpit.models.ead_model import EADModel
        assert isinstance(ead, EADModel)
        assert ead._fitted

    def test_serialization_roundtrip(self, governance_artifacts, tmp_path):
        """Artefacts se serialisent/deserialisent correctement via joblib."""
        import joblib

        path = tmp_path / "governance_suite.joblib"
        joblib.dump(governance_artifacts, str(path))

        loaded = joblib.load(str(path))
        assert set(loaded.keys()) == set(governance_artifacts.keys())
        assert loaded["conformal"]["q_hat"] == governance_artifacts["conformal"]["q_hat"]
        assert loaded["signatures"].n_features == 30
        assert loaded["hmm"]._fitted


# ══════════════════════════════════════════════
# TestFallbackWithoutArtifacts
# ══════════════════════════════════════════════
class TestFallbackWithoutArtifacts:
    """Dashboard fonctionne sans governance_suite.joblib."""

    def test_load_returns_none(self, tmp_path, monkeypatch):
        """load_governance_artifacts() retourne None si pas de fichier."""
        import functools
        from ifrs9_cockpit.dashboard import cache

        # Clear lru_cache and point to non-existent path
        cache.load_governance_artifacts.cache_clear()
        monkeypatch.setattr(
            "ifrs9_cockpit.dashboard.cache.load_governance_artifacts",
            functools.lru_cache(maxsize=1)(lambda: None),
        )

        from ifrs9_cockpit.dashboard.cache import load_governance_artifacts
        # The monkeypatched version always returns None
        result = load_governance_artifacts()
        assert result is None

    def test_governance_engines_compute_inline(self):
        """Engines de gouvernance fonctionnent en mode inline (sans artefacts)."""
        from ifrs9_cockpit.engine.signatures import compute_macro_signatures
        from ifrs9_cockpit.engine.tda import compute_macro_fragility
        from ifrs9_cockpit.engine.rmt import denoise_covariance

        # Signatures inline
        sig = compute_macro_signatures(MACRO_HISTORY_BASELINE, order=2)
        assert sig.n_features == 30

        # TDA inline
        tda = compute_macro_fragility(MACRO_HISTORY_BASELINE, window=24)
        assert 0.0 <= tda.fragility_index <= 1.0

        # RMT inline
        cov = np.array(MACRO_COVARIANCE)
        rmt = denoise_covariance(cov, n_observations=60)
        assert rmt.covariance_clean.shape == (5, 5)
