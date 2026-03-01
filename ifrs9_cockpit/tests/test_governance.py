"""Tests Phase 1 — Gouvernance Quantitative.

Conformal Prediction, Sobol indices, VRP, RMT.
Tests rapides uniquement (pas d'entraînement de modèles).
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════
# CONFORMAL PREDICTION
# ═══════════════════════════════════════════════════════


class TestConformalPredictor:
    """Tests du module engine/conformal.py."""

    def test_calibration_basic(self):
        """Calibration sur résidus simples."""
        from ifrs9_cockpit.engine.conformal import ConformalPredictor

        cp = ConformalPredictor(alpha=0.10)
        rng = np.random.RandomState(42)
        pd_pred = rng.uniform(0.01, 0.20, size=500)
        y_true = (rng.uniform(0, 1, size=500) < pd_pred).astype(float)

        q = cp.calibrate(pd_pred, y_true)
        assert cp.is_calibrated
        assert q > 0
        assert q < 1.0
        assert cp._n_cal == 500

    def test_coverage_guarantee(self):
        """La couverture empirique doit être >= 1 - alpha (propriété théorique).

        Pour un modèle bien calibré sur données continues, la couverture
        est garantie. Ici on utilise PD + bruit comme proxy de 'vraie PD'.
        """
        from ifrs9_cockpit.engine.conformal import ConformalPredictor

        cp = ConformalPredictor(alpha=0.10)
        rng = np.random.RandomState(123)
        n = 2000
        # PD 'vraie' = PD prédite + bruit (résidus continus, pas binaires)
        pd_pred = rng.uniform(0.01, 0.30, size=n)
        noise = rng.normal(0, 0.03, size=n)
        y_true_continuous = np.clip(pd_pred + noise, 0, 1)

        # Split: 50% calibration, 50% test
        n_cal = n // 2
        cp.calibrate(pd_pred[:n_cal], y_true_continuous[:n_cal])
        coverage = cp.empirical_coverage(pd_pred[n_cal:], y_true_continuous[n_cal:])

        # Couverture >= 85% (90% nominal, tolérance pour variance finie)
        assert coverage >= 0.85, f"Couverture {coverage:.2%} < 85%"

    def test_predict_pd_bounds(self):
        """Les bornes PD doivent encadrer le point estimate."""
        from ifrs9_cockpit.engine.conformal import ConformalPredictor

        cp = ConformalPredictor(alpha=0.10)
        rng = np.random.RandomState(77)
        pd_pred_cal = rng.uniform(0.01, 0.40, size=50)
        y_cal = np.clip(pd_pred_cal + rng.normal(0, 0.02, 50), 0, 1)
        cp.calibrate(pd_pred_cal, y_cal)

        pd_pred = np.array([0.05, 0.10, 0.20, 0.50, 0.80])

        result = cp.predict(pd_pred)
        assert np.all(result.pd_lower <= result.pd_point)
        assert np.all(result.pd_upper >= result.pd_point)
        assert np.all(result.pd_lower >= 0)
        assert np.all(result.pd_upper <= 1)

    def test_predict_ecl_bands(self):
        """Les ECL bands doivent être monotones avec PD bands."""
        from ifrs9_cockpit.engine.conformal import ConformalPredictor

        cp = ConformalPredictor(alpha=0.10)
        pd_pred = np.linspace(0.01, 0.30, 100)
        y_true = (np.random.RandomState(42).uniform(0, 1, 100) < pd_pred).astype(float)
        cp.calibrate(pd_pred, y_true)

        lgd = np.full(100, 0.35)
        ead = np.full(100, 100_000.0)
        df = np.full(100, 0.95)

        result = cp.predict(pd_pred, lgd=lgd, ead=ead, discount_factor=df)
        assert np.all(result.ecl_lower <= result.ecl_point + 1)  # tolérance arrondi
        assert np.all(result.ecl_upper >= result.ecl_point - 1)
        assert result.ecl_upper.sum() > result.ecl_lower.sum()

    def test_not_calibrated_raises(self):
        """predict() avant calibrate() doit lever RuntimeError."""
        from ifrs9_cockpit.engine.conformal import ConformalPredictor

        cp = ConformalPredictor()
        with pytest.raises(RuntimeError, match="non calibre"):
            cp.predict(np.array([0.1]))

    def test_diagnostics(self):
        """get_diagnostics() retourne les bonnes clés."""
        from ifrs9_cockpit.engine.conformal import ConformalPredictor

        cp = ConformalPredictor(alpha=0.05)
        cp.calibrate(np.linspace(0.01, 0.30, 50), np.zeros(50))
        diag = cp.get_diagnostics()
        assert "q_hat" in diag
        assert "n_calibration" in diag
        assert diag["alpha"] == 0.05
        assert diag["coverage_target"] == 0.95

    def test_small_calibration_set_raises(self):
        """Jeu de calibration < 10 doit lever ValueError."""
        from ifrs9_cockpit.engine.conformal import ConformalPredictor

        cp = ConformalPredictor()
        with pytest.raises(ValueError, match="trop petit"):
            cp.calibrate(np.array([0.1] * 5), np.array([0] * 5))


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
# VRP (VARIANCE RISK PREMIUM)
# ═══════════════════════════════════════════════════════


class TestVRP:
    """Tests du module engine/vrp.py."""

    def test_vrp_normal_regime(self):
        """IV=18%, RV=16% → VRP = 18²-16² = 68 → 'panique' (>8%)."""
        from ifrs9_cockpit.engine.vrp import compute_vrp

        result = compute_vrp(implied_vol=18.0, realized_vol=16.0)
        assert result.vrp == pytest.approx(18**2 - 16**2, abs=0.01)
        # VRP = 324 - 256 = 68 → panique (>8)
        assert result.regime == "panique"
        assert 0 <= result.severity_score <= 1
        assert 0 < result.tau_multiplier <= 1.3

    def test_vrp_complaisance(self):
        """RV > IV → VRP < 0 → complaisance."""
        from ifrs9_cockpit.engine.vrp import compute_vrp

        result = compute_vrp(implied_vol=10.0, realized_vol=12.0)
        assert result.vrp < 0
        assert result.regime == "complaisance"
        assert result.tau_multiplier == 1.30

    def test_vrp_panique(self):
        """IV très élevée → panique."""
        from ifrs9_cockpit.engine.vrp import compute_vrp

        result = compute_vrp(implied_vol=40.0, realized_vol=15.0)
        assert result.vrp > 8
        assert result.regime == "panique"
        assert result.tau_multiplier == 0.40

    def test_simulate_vrp_from_macro(self):
        """Simulation VRP cohérente avec la sévérité macro."""
        from ifrs9_cockpit.engine.vrp import simulate_vrp_from_macro

        vrp_calm = simulate_vrp_from_macro(0.0)
        vrp_crisis = simulate_vrp_from_macro(1.0)

        # En crise, IV et VRP plus élevés
        assert vrp_crisis.implied_vol > vrp_calm.implied_vol
        assert vrp_crisis.severity_score >= vrp_calm.severity_score

    def test_adjust_bl_tau(self):
        """Le tau ajusté doit être cohérent avec le régime."""
        from ifrs9_cockpit.engine.vrp import adjust_bl_tau, compute_vrp

        base_tau = 0.05

        # Normal
        vrp_norm = compute_vrp(16.0, 15.0)  # VRP = 31, stress
        tau_adj = adjust_bl_tau(base_tau, vrp_norm)
        assert tau_adj <= base_tau  # Stress → tau diminue

    def test_vrp_percentile_with_history(self):
        """Le percentile avec historique doit être cohérent."""
        from ifrs9_cockpit.engine.vrp import compute_vrp

        history = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = compute_vrp(16.0, 14.0, vrp_history=history)
        # VRP = 256 - 196 = 60 → au-dessus de tout l'historique
        assert result.percentile == 1.0

    def test_vrp_result_fields(self):
        """VRPResult doit avoir tous les champs."""
        from ifrs9_cockpit.engine.vrp import compute_vrp

        result = compute_vrp(20.0, 15.0)
        assert hasattr(result, "vrp")
        assert hasattr(result, "implied_vol")
        assert hasattr(result, "realized_vol")
        assert hasattr(result, "regime")
        assert hasattr(result, "tau_multiplier")
        assert hasattr(result, "severity_score")
        assert hasattr(result, "percentile")


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
# CHARTS (vérification que les figures se construisent)
# ═══════════════════════════════════════════════════════


class TestGovernanceCharts:
    """Tests que les 4 charts se construisent sans erreur."""

    def test_plot_conformal_bands(self):
        """plot_conformal_bands retourne une Figure."""
        from ifrs9_cockpit.dashboard.charts import plot_conformal_bands
        from ifrs9_cockpit.engine.conformal import ConformalPredictor

        cp = ConformalPredictor(alpha=0.10)
        pd_pred = np.linspace(0.01, 0.30, 100)
        y_true = np.zeros(100)
        cp.calibrate(pd_pred, y_true)
        result = cp.predict(pd_pred)

        fig = plot_conformal_bands(result)
        assert fig is not None
        assert len(fig.data) >= 3  # fill + line + bounds

    def test_plot_sobol_indices(self):
        """plot_sobol_indices retourne une Figure."""
        from ifrs9_cockpit.dashboard.charts import plot_sobol_indices
        from ifrs9_cockpit.engine.sobol_analysis import sobol_analysis

        result = sobol_analysis(
            lambda p: p["x"] + 2 * p["y"],
            bounds={"x": (0, 1), "y": (0, 1)},
            n_samples=64,
            variable_names=["x", "y"],
        )
        fig = plot_sobol_indices(result)
        assert fig is not None
        assert len(fig.data) == 2  # S1 + ST bars

    def test_plot_vrp_regime(self):
        """plot_vrp_regime retourne une Figure."""
        from ifrs9_cockpit.dashboard.charts import plot_vrp_regime
        from ifrs9_cockpit.engine.vrp import compute_vrp

        result = compute_vrp(20.0, 15.0)
        fig = plot_vrp_regime(result)
        assert fig is not None

    def test_plot_rmt_eigenvalues(self):
        """plot_rmt_eigenvalues retourne une Figure."""
        from ifrs9_cockpit.dashboard.charts import plot_rmt_eigenvalues
        from ifrs9_cockpit.engine.rmt import denoise_covariance

        result = denoise_covariance(np.eye(5), n_observations=30)
        fig = plot_rmt_eigenvalues(result)
        assert fig is not None
        assert len(fig.data) == 2  # raw + clean bars


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


# ═══════════════════════════════════════════════════════
# PHASE 2 — COMPLIANCE GATES
# ═══════════════════════════════════════════════════════


class TestComplianceGates:
    """Tests du module engine/compliance_gates.py."""

    def test_all_pass(self):
        """Tous les gates passent pour valeurs saines."""
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates

        cg = ComplianceGates()
        result = cg.check_all(
            cet1_ratio=0.12,
            pma_ratio=0.05,
            ecl_total=100_000,
            ead_total=10_000_000,
            stage3_pct=0.03,
            hhi_credit=2000,
            model_explainability=True,
            human_override=True,
        )

        assert result.is_compliant
        assert result.n_fail == 0
        assert result.n_pass > 0
        assert len(result.blocking_gates) == 0

    def test_cet1_violation(self):
        """CET1 < 4.5% -> FAIL pilier 1."""
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates

        cg = ComplianceGates()
        result = cg.check_all(cet1_ratio=0.03)

        assert not result.is_compliant
        assert "CET1_P1" in result.blocking_gates

    def test_cet1_warning_zone(self):
        """CET1 entre 4.5% et 10.5% -> WARNING cible."""
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates, GateStatus

        cg = ComplianceGates()
        result = cg.check_all(cet1_ratio=0.07)

        statuses = {g.gate_id: g.status for g in result.gates}
        assert statuses["CET1_P1"] == GateStatus.PASS
        assert statuses["CET1_TGT"] == GateStatus.WARNING

    def test_pma_limit_fail(self):
        """PMA > ±37.5% -> FAIL."""
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates

        cg = ComplianceGates()
        result = cg.check_all(pma_ratio=0.50)

        assert not result.is_compliant
        assert "PMA_LIMIT" in result.blocking_gates

    def test_pma_limit_pass(self):
        """PMA dans les limites -> PASS."""
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates, GateStatus

        cg = ComplianceGates()
        result = cg.check_all(pma_ratio=0.10)

        pma_gate = [g for g in result.gates if g.gate_id == "PMA_LIMIT"][0]
        assert pma_gate.status == GateStatus.PASS

    def test_stage3_alert_levels(self):
        """Stage 3 : <5% PASS, 5-10% WARNING, >10% FAIL."""
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates, GateStatus

        cg = ComplianceGates()

        r1 = cg.check_all(stage3_pct=0.03)
        s1 = [g for g in r1.gates if g.gate_id == "STAGE3"][0]
        assert s1.status == GateStatus.PASS

        r2 = cg.check_all(stage3_pct=0.07)
        s2 = [g for g in r2.gates if g.gate_id == "STAGE3"][0]
        assert s2.status == GateStatus.WARNING

        r3 = cg.check_all(stage3_pct=0.15)
        s3 = [g for g in r3.gates if g.gate_id == "STAGE3"][0]
        assert s3.status == GateStatus.FAIL

    def test_explainability_fail(self):
        """Pas d'explicabilite -> FAIL AI Act."""
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates

        cg = ComplianceGates()
        result = cg.check_all(model_explainability=False)

        assert not result.is_compliant
        assert "EXPLAIN" in result.blocking_gates

    def test_human_override_warning(self):
        """Pas de supervision humaine -> WARNING (pas FAIL)."""
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates, GateStatus

        cg = ComplianceGates()
        result = cg.check_all(human_override=False)

        h_gate = [g for g in result.gates if g.gate_id == "HUMAN"][0]
        assert h_gate.status == GateStatus.WARNING
        assert result.is_compliant  # WARNING ne bloque pas

    def test_concentration_hhi(self):
        """HHI > 4000 -> WARNING, > 6000 -> FAIL."""
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates, GateStatus

        cg = ComplianceGates()

        r1 = cg.check_all(hhi_credit=3000)
        hhi1 = [g for g in r1.gates if g.gate_id == "HHI_CREDIT"][0]
        assert hhi1.status == GateStatus.PASS

        r2 = cg.check_all(hhi_credit=5000)
        hhi2 = [g for g in r2.gates if g.gate_id == "HHI_CREDIT"][0]
        assert hhi2.status == GateStatus.WARNING

        r3 = cg.check_all(hhi_credit=7000)
        hhi3 = [g for g in r3.gates if g.gate_id == "HHI_CREDIT"][0]
        assert hhi3.status == GateStatus.FAIL

    def test_no_inputs_empty(self):
        """Aucun parametre -> aucun gate, compliant par defaut."""
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates

        cg = ComplianceGates()
        result = cg.check_all()

        assert result.is_compliant
        assert len(result.gates) == 0

    def test_gate_result_fields(self):
        """GateResult a tous les champs attendus."""
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates

        cg = ComplianceGates()
        result = cg.check_all(cet1_ratio=0.12)

        gate = result.gates[0]
        assert hasattr(gate, "gate_id")
        assert hasattr(gate, "gate_name")
        assert hasattr(gate, "category")
        assert hasattr(gate, "status")
        assert hasattr(gate, "value")
        assert hasattr(gate, "threshold")
        assert hasattr(gate, "message")
        assert hasattr(gate, "regulation")

    def test_compliance_result_counts(self):
        """n_pass + n_warning + n_fail == len(gates)."""
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates

        cg = ComplianceGates()
        result = cg.check_all(
            cet1_ratio=0.07,  # WARNING
            pma_ratio=0.50,   # FAIL
            stage3_pct=0.03,  # PASS
        )

        assert result.n_pass + result.n_warning + result.n_fail == len(result.gates)


# ═══════════════════════════════════════════════════════
# PHASE 2 — CHARTS (verification construction)
# ═══════════════════════════════════════════════════════


class TestPhase2Charts:
    """Tests que les 3 charts Phase 2 se construisent sans erreur."""

    def test_plot_signature_heatmap(self):
        """plot_signature_heatmap retourne une Figure."""
        from ifrs9_cockpit.dashboard.charts import plot_signature_heatmap
        from ifrs9_cockpit.engine.signatures import compute_path_signature

        path = np.random.RandomState(42).randn(20, 4)
        sig = compute_path_signature(path, order=2)
        fig = plot_signature_heatmap(sig)

        assert fig is not None
        assert len(fig.data) >= 1

    def test_plot_tda_fragility(self):
        """plot_tda_fragility retourne une Figure."""
        from ifrs9_cockpit.dashboard.charts import plot_tda_fragility
        from ifrs9_cockpit.engine.tda import compute_fragility_index

        corr = np.eye(5) + 0.3 * np.ones((5, 5))
        np.fill_diagonal(corr, 1.0)
        tda = compute_fragility_index(corr)
        fig = plot_tda_fragility(tda)

        assert fig is not None

    def test_plot_compliance_gates(self):
        """plot_compliance_gates retourne une Figure."""
        from ifrs9_cockpit.dashboard.charts import plot_compliance_gates
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates

        cg = ComplianceGates()
        result = cg.check_all(
            cet1_ratio=0.12,
            pma_ratio=0.10,
            stage3_pct=0.04,
            model_explainability=True,
            human_override=True,
        )
        fig = plot_compliance_gates(result)

        assert fig is not None
        assert len(fig.data) >= 1

    def test_plot_compliance_gates_empty(self):
        """plot_compliance_gates avec aucun gate retourne une Figure vide."""
        from ifrs9_cockpit.dashboard.charts import plot_compliance_gates
        from ifrs9_cockpit.engine.compliance_gates import ComplianceGates

        cg = ComplianceGates()
        result = cg.check_all()  # aucun param -> aucun gate
        fig = plot_compliance_gates(result)

        assert fig is not None
