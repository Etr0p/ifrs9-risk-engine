"""Tests Phase 1 — Gouvernance Quantitative.

Conformal Prediction, Sobol indices, VRP, RMT.
Tests rapides uniquement (pas d'entraînement de modèles).
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════
# CONFORMAL PREDICTION
# ═══════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════
# SOBOL INDICES
# ═══════════════════════════════════════════════════════


class TestSobolAnalysis:
    """Tests du module engine/sobol_analysis.py."""

    def test_sobol_linear_function(self):
        """Pour f(x) = 3*x1 + x2, S1(x1) >> S1(x2)."""
        from ifrs9_cockpit.engine.sobol_analysis import sobol_analysis

        def linear_fn(params):
            return 3.0 * params["x1"] + 1.0 * params["x2"]

        result = sobol_analysis(
            linear_fn,
            bounds={"x1": (0, 1), "x2": (0, 1)},
            n_samples=512,
            variable_names=["x1", "x2"],
        )

        assert result.s1["x1"] > result.s1["x2"]
        assert result.s1["x1"] > 0.5  # x1 domine

    def test_sobol_s1_sum_bounded(self):
        """Somme des S1 <= 1 (pas de sur-décomposition)."""
        from ifrs9_cockpit.engine.sobol_analysis import sobol_analysis

        def quadratic_fn(params):
            x = params["a"]
            y = params["b"]
            return x ** 2 + x * y + y

        result = sobol_analysis(
            quadratic_fn,
            bounds={"a": (0, 1), "b": (0, 1)},
            n_samples=256,
            variable_names=["a", "b"],
        )

        s1_sum = sum(result.s1.values())
        assert s1_sum <= 1.05  # tolérance numérique

    def test_sobol_st_ge_s1(self):
        """ST_i >= S1_i pour toute variable (par définition)."""
        from ifrs9_cockpit.engine.sobol_analysis import sobol_analysis

        def nonlinear_fn(params):
            return params["x"] * params["y"] + params["x"] ** 2

        result = sobol_analysis(
            nonlinear_fn,
            bounds={"x": (0, 1), "y": (0, 1)},
            n_samples=256,
            variable_names=["x", "y"],
        )

        for v in ["x", "y"]:
            assert result.st[v] >= result.s1[v] - 0.05  # tolérance

    def test_sobol_constant_function(self):
        """Fonction constante → S1 = ST = 0."""
        from ifrs9_cockpit.engine.sobol_analysis import sobol_analysis

        result = sobol_analysis(
            lambda p: 42.0,
            bounds={"a": (0, 1), "b": (0, 1)},
            n_samples=64,
            variable_names=["a", "b"],
        )

        for v in ["a", "b"]:
            assert result.s1[v] == 0
            assert result.st[v] == 0

    def test_sobol_result_structure(self):
        """Vérifier que SobolResult contient les bons champs."""
        from ifrs9_cockpit.engine.sobol_analysis import sobol_analysis

        result = sobol_analysis(
            lambda p: p["x"] + p["y"],
            bounds={"x": (0, 1), "y": (0, 1)},
            n_samples=64,
            variable_names=["x", "y"],
        )

        assert len(result.variable_names) == 2
        assert result.n_samples == 64
        assert result.ecl_var >= 0
        assert len(result.s1_conf) == 2
        assert len(result.st_conf) == 2


# ═══════════════════════════════════════════════════════
# RMT (RANDOM MATRIX THEORY)
# ═══════════════════════════════════════════════════════


class TestRMT:
    """Tests du module engine/rmt.py."""

    def test_mp_bounds_basic(self):
        """Bornes MP pour N=5, T=60."""
        from ifrs9_cockpit.engine.rmt import marchenko_pastur_bounds

        lo, hi = marchenko_pastur_bounds(5, 60)
        assert lo >= 0
        assert hi > lo
        assert hi < 5  # Pas dégénéré

    def test_mp_bounds_symmetry(self):
        """MP(N,T) symétrique en q = N/T."""
        from ifrs9_cockpit.engine.rmt import marchenko_pastur_bounds

        _, hi1 = marchenko_pastur_bounds(5, 50)
        _, hi2 = marchenko_pastur_bounds(50, 5)
        assert hi1 == pytest.approx(hi2, rel=1e-10)

    def test_denoise_identity(self):
        """Matrice identité → aucun signal (tout est bruit si T petit)."""
        from ifrs9_cockpit.engine.rmt import denoise_covariance

        I5 = np.eye(5)
        result = denoise_covariance(I5, n_observations=10)

        assert result.n_signal >= 0
        assert result.n_noise >= 0
        assert result.n_signal + result.n_noise == 5
        # La matrice nettoyée doit rester symétrique
        np.testing.assert_allclose(
            result.covariance_clean,
            result.covariance_clean.T,
            atol=1e-10,
        )

    def test_denoise_preserves_trace(self):
        """Le débruitage 'constant' préserve la trace (risque total)."""
        from ifrs9_cockpit.engine.rmt import denoise_covariance

        rng = np.random.RandomState(42)
        # Matrice de corrélation réaliste
        A = rng.randn(5, 5) * 0.3
        corr = A @ A.T
        np.fill_diagonal(corr, 1.0)

        result = denoise_covariance(corr, n_observations=60, method="constant")

        trace_raw = np.trace(corr)
        trace_clean = np.trace(result.correlation_clean)
        assert trace_clean == pytest.approx(trace_raw, rel=0.05)

    def test_denoise_eigenvalues_ordered(self):
        """Les valeurs propres doivent être triées décroissantes."""
        from ifrs9_cockpit.engine.rmt import denoise_covariance

        cov = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.2],
            [0.3, 0.2, 1.0],
        ])
        result = denoise_covariance(cov, n_observations=30)

        assert np.all(np.diff(result.eigenvalues_raw) <= 1e-10)  # décroissant
        assert np.all(np.diff(result.eigenvalues_clean) <= 1e-10)

    def test_denoise_with_real_macro_covariance(self):
        """Test avec la vraie matrice de covariance macro du cockpit."""
        from ifrs9_cockpit.engine.rmt import denoise_covariance
        from ifrs9_cockpit.config import MACRO_COVARIANCE

        cov = np.array(MACRO_COVARIANCE)
        result = denoise_covariance(cov, n_observations=60)

        assert result.covariance_clean.shape == (5, 5)
        assert result.noise_fraction >= 0
        assert result.noise_fraction <= 1
        assert result.mp_upper > 0

    def test_denoise_non_square_raises(self):
        """Matrice non carrée doit lever ValueError."""
        from ifrs9_cockpit.engine.rmt import denoise_covariance

        with pytest.raises(ValueError):
            denoise_covariance(np.ones((3, 4)))

    def test_denoise_shrink_method(self):
        """Méthode 'shrink' doit fonctionner."""
        from ifrs9_cockpit.engine.rmt import denoise_covariance

        cov = np.eye(4) + 0.1 * np.ones((4, 4))
        result = denoise_covariance(cov, n_observations=20, method="shrink")
        assert result.covariance_clean.shape == (4, 4)

    def test_rmt_result_fields(self):
        """RMTResult doit avoir tous les champs."""
        from ifrs9_cockpit.engine.rmt import denoise_covariance

        result = denoise_covariance(np.eye(3), n_observations=15)
        assert hasattr(result, "covariance_clean")
        assert hasattr(result, "correlation_clean")
        assert hasattr(result, "eigenvalues_raw")
        assert hasattr(result, "eigenvalues_clean")
        assert hasattr(result, "mp_upper")
        assert hasattr(result, "mp_lower")
        assert hasattr(result, "n_signal")
        assert hasattr(result, "n_noise")
        assert hasattr(result, "noise_fraction")


# ═══════════════════════════════════════════════════════
# PHASE 2 — PATH SIGNATURES (Rough Paths)
# ═══════════════════════════════════════════════════════


class TestSignatures:
    """Tests du module engine/signatures.py."""

    def test_signature_order1_dimensions(self):
        """Ordre 1 : d features pour un chemin d-dimensionnel."""
        from ifrs9_cockpit.engine.signatures import compute_path_signature

        rng = np.random.RandomState(42)
        path = rng.randn(20, 3)  # 20 pas de temps, 3 dimensions
        result = compute_path_signature(path, order=1)

        assert result.n_features == 3
        assert result.n_dims == 3
        assert result.path_length == 20
        assert result.order == 1
        assert len(result.features) == 3

    def test_signature_order2_dimensions(self):
        """Ordre 2 : d + d^2 features."""
        from ifrs9_cockpit.engine.signatures import compute_path_signature

        path = np.random.RandomState(10).randn(30, 5)
        result = compute_path_signature(path, order=2)

        assert result.n_features == 5 + 25  # d + d^2 = 30
        assert result.order == 2
        assert len(result.feature_names) == 30

    def test_signature_order3_dimensions(self):
        """Ordre 3 : d + d^2 + d^3 features."""
        from ifrs9_cockpit.engine.signatures import compute_path_signature

        path = np.random.RandomState(7).randn(15, 3)
        result = compute_path_signature(path, order=3)

        # 3 + 9 + 27 = 39
        assert result.n_features == 39

    def test_signature_constant_path(self):
        """Chemin constant -> signature d'ordre 1 = 0."""
        from ifrs9_cockpit.engine.signatures import compute_path_signature

        path = np.ones((10, 2))  # constant
        result = compute_path_signature(path, order=1, normalize=False)

        np.testing.assert_allclose(result.features, 0.0, atol=1e-12)

    def test_signature_linear_path(self):
        """Chemin lineaire : S1 capture les increments totaux."""
        from ifrs9_cockpit.engine.signatures import compute_path_signature

        path = np.column_stack([
            np.linspace(0, 10, 50),  # x: 0 -> 10
            np.linspace(0, 5, 50),   # y: 0 -> 5
        ])
        result = compute_path_signature(path, order=1, normalize=False)

        assert result.features[0] == pytest.approx(10.0, rel=0.01)
        assert result.features[1] == pytest.approx(5.0, rel=0.01)

    def test_macro_signatures(self):
        """compute_macro_signatures fonctionne avec MACRO_HISTORY_BASELINE."""
        from ifrs9_cockpit.engine.signatures import compute_macro_signatures
        from ifrs9_cockpit.config import MACRO_HISTORY_BASELINE

        result = compute_macro_signatures(MACRO_HISTORY_BASELINE, order=2)

        assert result.n_dims == 5  # 5 variables macro
        assert result.n_features == 30  # 5 + 25
        assert result.path_length == 60  # 60 mois

    def test_rolling_signatures(self):
        """rolling_signatures retourne la bonne forme."""
        from ifrs9_cockpit.engine.signatures import rolling_signatures

        path = np.random.RandomState(99).randn(30, 3)
        result = rolling_signatures(path, window=12, order=1)

        assert result.shape == (19, 3)  # 30 - 12 + 1 = 19 fenetres

    def test_signature_too_short_raises(self):
        """Chemin < 2 pas de temps doit lever ValueError."""
        from ifrs9_cockpit.engine.signatures import compute_path_signature

        with pytest.raises(ValueError, match="trop court"):
            compute_path_signature(np.array([[1.0]]), order=1)

    def test_signature_invalid_order_raises(self):
        """Ordre != 1,2,3 doit lever ValueError."""
        from ifrs9_cockpit.engine.signatures import compute_path_signature

        with pytest.raises(ValueError, match="Ordre"):
            compute_path_signature(np.random.randn(10, 2), order=5)

    def test_signature_1d_path(self):
        """Chemin 1D (vecteur) doit etre auto-reshape."""
        from ifrs9_cockpit.engine.signatures import compute_path_signature

        path = np.array([1, 2, 4, 7, 11], dtype=float)
        result = compute_path_signature(path, order=2)

        assert result.n_dims == 1
        assert result.n_features == 2  # 1 + 1


# ═══════════════════════════════════════════════════════
# PHASE 2 — TDA (Topological Data Analysis)
# ═══════════════════════════════════════════════════════


class TestTDA:
    """Tests du module engine/tda.py."""

    def test_identity_correlation_fragility(self):
        """Matrice identite (secteurs independants) -> fragilite basse."""
        from ifrs9_cockpit.engine.tda import compute_fragility_index

        corr = np.eye(5)
        result = compute_fragility_index(corr)

        # Correlation moyenne = 0 -> composante corr = 0
        assert result.mean_correlation == 0.0
        assert result.fragility_index < 0.6  # pas fragile

    def test_full_correlation_fragility(self):
        """Matrice tout-a-1 (parfaite correlation) -> fragilite haute."""
        from ifrs9_cockpit.engine.tda import compute_fragility_index

        corr = np.ones((5, 5))
        result = compute_fragility_index(corr)

        assert result.mean_correlation == pytest.approx(1.0, abs=0.01)
        assert result.fragility_index > 0.8  # tres fragile

    def test_fragility_bounded(self):
        """Indice de fragilite dans [0, 1]."""
        from ifrs9_cockpit.engine.tda import compute_fragility_index

        rng = np.random.RandomState(42)
        A = rng.randn(5, 5) * 0.5
        corr = A @ A.T
        np.fill_diagonal(corr, 1.0)
        # Normaliser en correlation
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

        result = compute_fragility_index(corr)
        assert 0.0 <= result.fragility_index <= 1.0

    def test_h0_bars_count(self):
        """N secteurs -> N barres H0 (N-1 fusions + 1 immortelle)."""
        from ifrs9_cockpit.engine.tda import compute_fragility_index

        corr = np.eye(4) + 0.1 * np.ones((4, 4))
        np.fill_diagonal(corr, 1.0)
        result = compute_fragility_index(corr)

        assert len(result.bars_h0) == 4  # 3 fusions + 1 immortelle

    def test_distance_matrix_symmetry(self):
        """La matrice de distance doit etre symetrique et d(i,i)=0."""
        from ifrs9_cockpit.engine.tda import compute_fragility_index

        corr = np.array([
            [1.0, 0.5, -0.3],
            [0.5, 1.0, 0.2],
            [-0.3, 0.2, 1.0],
        ])
        result = compute_fragility_index(corr, sector_names=["A", "B", "C"])

        D = result.distance_matrix
        np.testing.assert_allclose(D, D.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)

    def test_macro_fragility(self):
        """compute_macro_fragility fonctionne avec MACRO_HISTORY_BASELINE."""
        from ifrs9_cockpit.engine.tda import compute_macro_fragility
        from ifrs9_cockpit.config import MACRO_HISTORY_BASELINE

        result = compute_macro_fragility(MACRO_HISTORY_BASELINE, window=24)

        assert 0.0 <= result.fragility_index <= 1.0
        assert len(result.sector_names) == 5

    def test_non_square_raises(self):
        """Matrice non carree doit lever ValueError."""
        from ifrs9_cockpit.engine.tda import compute_fragility_index

        with pytest.raises(ValueError, match="carree"):
            compute_fragility_index(np.ones((3, 4)))

    def test_n_components_at_threshold(self):
        """n_components_at_threshold >= 1."""
        from ifrs9_cockpit.engine.tda import compute_fragility_index

        corr = np.eye(5)
        result = compute_fragility_index(corr)
        assert result.n_components_at_threshold >= 1


