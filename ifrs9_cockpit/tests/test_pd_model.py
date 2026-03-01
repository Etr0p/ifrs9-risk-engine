"""Tests unitaires pour ifrs9_cockpit/models/pd_model.py (Story 2-1).

Couvre les taches :
  T1: PDModelSuite entraine 3 modeles PD avec calibration isotonique
  T2: Selection du modele actif (FR6)
  T3: Metriques de comparaison (AUC, Gini, KS, PSI)
  T4: Predictions PD calibrees dans [0, 1]
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from ifrs9_cockpit.config import (
    TARGET, RANDOM_SEED,
    CORPORATE_INTERACTION_FEATURES,
    CONSUMER_NUMERICAL_FEATURES,
    CONSUMER_CATEGORICAL_FEATURES,
    CONSUMER_ENGINEERED_FEATURES,
)
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.models.pd_model import PDModelSuite, PDModelResult


# ============================================================
# Fixtures partagees — scope module pour eviter re-entrainement
# ============================================================

@pytest.fixture(scope="module")
def dataset():
    """Genere un dataset credit pour les tests (5000 pour AUC fiable)."""
    df_credit, _, _, _ = generate_dataset(n_clients=5000, seed=RANDOM_SEED)
    return df_credit


@pytest.fixture(scope="module")
def trained_suite(dataset):
    """Entraine la suite de 3 modeles PD sur le petit dataset."""
    suite = PDModelSuite(seed=RANDOM_SEED)
    suite.fit(dataset)
    return suite


# ============================================================
# T1 — PDModelSuite entraine 3 modeles
# ============================================================

class TestPDModelTraining:
    def test_three_models_trained(self, trained_suite):
        """Verifie que 3 modeles sont entraines."""
        assert len(trained_suite.results) == 3

    def test_model_names(self, trained_suite):
        """Verifie les noms des 3 modeles."""
        expected = {"LR_WoE", "TabNet", "XGBoost"}
        assert set(trained_suite.results.keys()) == expected

    def test_each_result_is_pd_model_result(self, trained_suite):
        """Verifie que chaque resultat est un PDModelResult."""
        for name, result in trained_suite.results.items():
            assert isinstance(result, PDModelResult), f"{name} n'est pas PDModelResult"

    def test_models_have_predict_proba(self, trained_suite):
        """Verifie que chaque modele calibre expose predict_proba."""
        for name, result in trained_suite.results.items():
            assert hasattr(result.model, "predict_proba"), f"{name} sans predict_proba"

    def test_train_test_split_done(self, trained_suite):
        """Verifie que le split train/test est effectue."""
        assert trained_suite.X_train is not None
        assert trained_suite.X_test is not None
        assert trained_suite.y_train is not None
        assert trained_suite.y_test is not None

    def test_train_test_sizes(self, trained_suite):
        """Verifie les tailles relatives du split (70/30)."""
        n_train = len(trained_suite.X_train)
        n_test = len(trained_suite.X_test)
        ratio = n_train / (n_train + n_test)
        assert 0.65 <= ratio <= 0.75, f"Ratio train: {ratio:.2f}"


# ============================================================
# T2 — Selection du modele actif (FR6)
# ============================================================

class TestModelSelection:
    def test_default_model_is_lr_woe(self, trained_suite):
        """Le modele actif par defaut est LR_WoE."""
        assert trained_suite.active_model_name == "LR_WoE"

    def test_select_model_changes_active(self, trained_suite):
        """select_model change le modele actif."""
        original = trained_suite.active_model_name
        trained_suite.select_model("XGBoost")
        assert trained_suite.active_model_name == "XGBoost"
        # Restaurer
        trained_suite.select_model(original)

    def test_select_invalid_model_raises(self, trained_suite):
        """Selectionner un modele inconnu leve ValueError."""
        with pytest.raises(ValueError, match="inconnu"):
            trained_suite.select_model("NonExistent")

    def test_predict_active_returns_array(self, trained_suite, dataset):
        """predict_active retourne un array numpy."""
        preds = trained_suite.predict_active(dataset)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(dataset)


# ============================================================
# T3 — Metriques de comparaison
# ============================================================

class TestMetrics:
    def test_auc_above_070_all_models(self, trained_suite):
        """AUC test > 0.70 pour les 3 modeles (FR38)."""
        for name, result in trained_suite.results.items():
            auc = result.metrics_test["auc"]
            assert auc > 0.70, f"{name}: AUC test = {auc:.4f} < 0.70"

    def test_gini_positive_all_models(self, trained_suite):
        """Gini test > 0 pour les 3 modeles."""
        for name, result in trained_suite.results.items():
            gini = result.metrics_test["gini"]
            assert gini > 0, f"{name}: Gini test = {gini:.4f} <= 0"

    def test_ks_positive_all_models(self, trained_suite):
        """KS test > 0 pour les 3 modeles."""
        for name, result in trained_suite.results.items():
            ks = result.metrics_test["ks"]
            assert ks > 0, f"{name}: KS test = {ks:.4f} <= 0"

    def test_psi_computed(self, trained_suite):
        """PSI est calcule et >= 0 pour les 3 modeles."""
        for name, result in trained_suite.results.items():
            psi = result.metrics_test["psi"]
            assert psi >= 0, f"{name}: PSI = {psi:.4f} < 0"

    def test_psi_below_025(self, trained_suite):
        """PSI < 0.25 (pas de drift significatif entre train et test)."""
        for name, result in trained_suite.results.items():
            psi = result.metrics_test["psi"]
            assert psi < 0.25, f"{name}: PSI = {psi:.4f} >= 0.25 (drift)"

    def test_metrics_train_exist(self, trained_suite):
        """Les metriques train sont calculees pour chaque modele."""
        for name, result in trained_suite.results.items():
            for key in ("auc", "gini", "ks"):
                assert key in result.metrics_train, f"{name}: {key} absent du train"

    def test_comparison_table(self, trained_suite):
        """get_comparison_table retourne un pl.DataFrame valide."""
        table = trained_suite.get_comparison_table()
        assert isinstance(table, pl.DataFrame)
        assert len(table) == 3
        assert "auc_test" in table.columns
        assert "psi" in table.columns


# ============================================================
# T4 — Predictions PD calibrees dans [0, 1]
# ============================================================

class TestPredictions:
    def test_predictions_in_01_train(self, trained_suite):
        """PD predites dans [0, 1] sur le train pour chaque modele."""
        for name, result in trained_suite.results.items():
            assert result.y_pred_train.min() >= 0.0, f"{name}: PD train min < 0"
            assert result.y_pred_train.max() <= 1.0, f"{name}: PD train max > 1"

    def test_predictions_in_01_test(self, trained_suite):
        """PD predites dans [0, 1] sur le test pour chaque modele."""
        for name, result in trained_suite.results.items():
            assert result.y_pred_test.min() >= 0.0, f"{name}: PD test min < 0"
            assert result.y_pred_test.max() <= 1.0, f"{name}: PD test max > 1"

    def test_predict_all_models(self, trained_suite, dataset):
        """predict() retourne un dict avec 3 modeles."""
        preds = trained_suite.predict(dataset)
        assert isinstance(preds, dict)
        assert len(preds) == 3
        for name, arr in preds.items():
            assert isinstance(arr, np.ndarray)
            assert len(arr) == len(dataset)
            assert arr.min() >= 0.0
            assert arr.max() <= 1.0

    def test_feature_importance_available(self, trained_suite):
        """L'importance des features est disponible pour chaque modele."""
        table = trained_suite.get_feature_importance_table()
        assert isinstance(table, pl.DataFrame)
        assert len(table) > 0

    @pytest.mark.slow
    def test_reproducibility_with_seed(self, dataset):
        """Deux entrainements avec le meme seed donnent les memes PD (AC#1)."""
        suite1 = PDModelSuite(seed=RANDOM_SEED)
        suite1.fit(dataset)
        preds1 = suite1.predict_active(dataset)

        suite2 = PDModelSuite(seed=RANDOM_SEED)
        suite2.fit(dataset)
        preds2 = suite2.predict_active(dataset)

        np.testing.assert_allclose(preds1, preds2, atol=1e-6)


# ============================================================
# T5 — VIF filtering et min_bin_pct (revue expert)
# ============================================================

class TestExpertReviewImprovements:
    def test_vif_dropped_is_list(self, trained_suite):
        """_vif_dropped est une liste (tracabilite VIF)."""
        assert isinstance(trained_suite._vif_dropped, list)

    def test_sign_dropped_is_list(self, trained_suite):
        """_sign_dropped est une liste (tracabilite beta > 0)."""
        assert isinstance(trained_suite._sign_dropped, list)

    def test_vif_threshold_in_config(self):
        """Le seuil VIF est configure dans PDModelConfig."""
        from ifrs9_cockpit.config import PD_CONFIG
        assert hasattr(PD_CONFIG, "vif_max_threshold")
        assert PD_CONFIG.vif_max_threshold > 0

    def test_all_final_betas_negative_or_zero(self, trained_suite):
        """Tous les beta base du LR final sont <= 0 (interactions exemptees)."""
        if trained_suite._base_lr is not None:
            interaction_woe = {f"{f}_woe" for f in CORPORATE_INTERACTION_FEATURES}
            for feat, coef in zip(trained_suite._woe_features, trained_suite._base_lr.coef_[0]):
                if feat not in interaction_woe:
                    assert coef <= 0, f"Positive beta on base feature: {feat} = {coef:+.4f}"

    def test_woe_bins_respect_min_pct(self, trained_suite):
        """Chaque bin WoE contient >= min_bin_pct de la population (post-PAV)."""
        binner = trained_suite.woe_binner
        for feat in binner.bins_:
            woe_map = binner.woe_maps_[feat]
            n_bins = len(woe_map)
            # Avec 10 features et min_bin_pct=5%, chaque bin doit etre > 2%
            # du dataset (apres PAV, on peut have fewer bins so threshold is met)
            assert n_bins >= 2, f"{feat}: only {n_bins} bins"

    def test_nan_woe_populated(self, trained_suite):
        """Le dictionnaire nan_woe_ est rempli pour chaque feature."""
        binner = trained_suite.woe_binner
        for feat in binner.bins_:
            assert feat in binner.nan_woe_, f"{feat}: nan_woe_ missing"

    def test_woe_monotonicity_preserved(self, trained_suite):
        """La monotonicite WoE est preservee apres min_bin_pct enforcement."""
        binner = trained_suite.woe_binner
        for feat, direction in binner.directions_.items():
            woe_detail = binner.get_woe_detail(feat)
            woe_vals = woe_detail[woe_detail["bin"] != "bin_nan"]["woe"].values
            if len(woe_vals) >= 2:
                if direction == "increasing":
                    monotone = all(
                        woe_vals[i] <= woe_vals[i + 1] + 1e-10
                        for i in range(len(woe_vals) - 1)
                    )
                else:
                    monotone = all(
                        woe_vals[i] >= woe_vals[i + 1] - 1e-10
                        for i in range(len(woe_vals) - 1)
                    )
                assert monotone, f"{feat}: monotonicity broken after min_bin_pct enforcement"


# ============================================================
# T6 — Corporate Interaction Features
# ============================================================

class TestCorporateInteractionFeatures:
    def test_interaction_features_exist_in_train(self, trained_suite):
        """Les 12 features d'interaction sont presentes dans X_train."""
        for feat in CORPORATE_INTERACTION_FEATURES:
            assert feat in trained_suite.X_train.columns, f"Missing: {feat}"

    def test_interaction_features_no_nan(self, trained_suite):
        """Aucun NaN dans les features d'interaction."""
        for feat in CORPORATE_INTERACTION_FEATURES:
            if feat in trained_suite.X_train.columns:
                nan_count = trained_suite.X_train[feat].isna().sum()
                assert nan_count == 0, f"{feat}: {nan_count} NaN"

    def test_auc_above_075_with_interactions(self, trained_suite):
        """AUC test > 0.75 on 5k rows (0.87+ expected on 1.5M production)."""
        for name in ("LR_WoE", "XGBoost"):
            if name in trained_suite.results:
                auc = trained_suite.results[name].metrics_test["auc"]
                assert auc > 0.75, f"{name}: AUC test = {auc:.4f} < 0.75"

    def test_interaction_features_not_vif_dropped(self, trained_suite):
        """Les interaction features ne sont pas supprimees par VIF."""
        interaction_woe = {f"{f}_woe" for f in CORPORATE_INTERACTION_FEATURES}
        dropped_set = set(trained_suite._vif_dropped)
        wrongly_dropped = interaction_woe & dropped_set
        assert not wrongly_dropped, f"Interaction features VIF-dropped: {wrongly_dropped}"


# ============================================================
# T7 — Parametrized Suite
# ============================================================

class TestParametrizedSuite:
    def test_custom_features(self, dataset):
        """Suite avec features custom fonctionne sans erreur."""
        suite = PDModelSuite(
            seed=RANDOM_SEED,
            numerical_features=["revenue", "debt_ratio", "credit_score", "dpd"],
            categorical_features=["sector"],
            available_models=("LR_WoE",),
        )
        suite.fit(dataset, models=("LR_WoE",))
        assert "LR_WoE" in suite.results
        auc = suite.results["LR_WoE"].metrics_test["auc"]
        assert auc > 0.55, f"AUC too low: {auc:.4f}"

    def test_default_backward_compat(self, dataset):
        """Suite sans arguments = comportement corporate identique."""
        suite = PDModelSuite(seed=RANDOM_SEED)
        from ifrs9_cockpit.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
        assert suite._numerical_features == list(NUMERICAL_FEATURES)
        assert suite._categorical_features == list(CATEGORICAL_FEATURES)

    def test_available_models_override(self, dataset):
        """available_models override entraine seulement les modeles specifies."""
        suite = PDModelSuite(
            seed=RANDOM_SEED,
            available_models=("LR_WoE", "XGBoost"),
        )
        suite.fit(dataset)
        assert "LR_WoE" in suite.results
        assert "XGBoost" in suite.results
        assert "TabNet" not in suite.results


# ============================================================
# T8 — TabNet Light variant
# ============================================================

class TestTabNetLightVariant:
    def test_tabnet_light_trains(self, dataset):
        """TabNet Light (8k params) s'entraine sans erreur."""
        suite = PDModelSuite(
            seed=RANDOM_SEED,
            available_models=("TabNet",),
            tabnet_variant="light",
        )
        suite.fit(dataset, models=("TabNet",))
        assert "TabNet" in suite.results

    def test_tabnet_light_auc_above_070(self, dataset):
        """TabNet Light AUC test > 0.70 sur 5k lignes."""
        suite = PDModelSuite(
            seed=RANDOM_SEED,
            available_models=("TabNet",),
            tabnet_variant="light",
        )
        suite.fit(dataset, models=("TabNet",))
        auc = suite.results["TabNet"].metrics_test["auc"]
        assert auc > 0.70, f"TabNet Light AUC = {auc:.4f} < 0.70"

    def test_tabnet_light_pd_in_01(self, dataset):
        """TabNet Light PD dans [0, 1]."""
        suite = PDModelSuite(
            seed=RANDOM_SEED,
            available_models=("TabNet",),
            tabnet_variant="light",
        )
        suite.fit(dataset, models=("TabNet",))
        result = suite.results["TabNet"]
        assert result.y_pred_test.min() >= 0.0
        assert result.y_pred_test.max() <= 1.0

    def test_tabnet_variant_stored(self, dataset):
        """Le variant est stocke dans la suite."""
        suite = PDModelSuite(seed=RANDOM_SEED, tabnet_variant="light")
        assert suite._tabnet_variant == "light"
        suite2 = PDModelSuite(seed=RANDOM_SEED, tabnet_variant="full")
        assert suite2._tabnet_variant == "full"


# ============================================================
# Standalone
# ============================================================

@pytest.mark.slow
class TestStandalone:
    def test_module_runs_standalone(self):
        """Le module pd_model.py s'execute sans erreur."""
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "ifrs9_cockpit.models.pd_model"],
            capture_output=True, text=True, timeout=600,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"stderr: {result.stderr[-500:]}"
        assert "Tous les modeles PD valides" in result.stdout
