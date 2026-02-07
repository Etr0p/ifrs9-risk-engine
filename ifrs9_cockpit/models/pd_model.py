"""Modèles de Probabilité de Défaut (PD) pour le Cockpit IFRS 9.

Implémente 3 algorithmes avec benchmark comparatif :
    1. Logistic Regression sur features WoE + calibration isotonic
    2. Random Forest (benchmark non-linéaire)
    3. XGBoost (benchmark gradient boosting)

La classe PDModelSuite encapsule le pipeline complet :
    data split → WoE binning → fit 3 modèles → calibration → métriques → comparaison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from ifrs9_cockpit.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PD_CONFIG,
    RANDOM_SEED,
    TARGET,
    TRAIN_RATIO,
    TEST_RATIO,
)
from ifrs9_cockpit.models.woe import WoEBinner
from ifrs9_cockpit.analytics.metrics import ModelMetrics


@dataclass
class PDModelResult:
    """Résultat d'un modèle PD individuel.

    Attributes:
        name: Nom du modèle (LR_WoE, RandomForest, XGBoost).
        model: Objet modèle entraîné.
        metrics_train: Métriques sur le jeu d'entraînement.
        metrics_test: Métriques sur le jeu de test.
        y_pred_train: Probabilités prédites (train).
        y_pred_test: Probabilités prédites (test).
        feature_names: Noms des features utilisées.
        feature_importance: Importance des features (si disponible).
    """
    name: str
    model: Any
    metrics_train: Dict[str, float]
    metrics_test: Dict[str, float]
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    feature_names: List[str]
    feature_importance: Optional[Dict[str, float]] = None


class PDModelSuite:
    """Suite de modèles PD avec benchmark comparatif.

    Orchestre le pipeline complet de modélisation PD :
        1. Split train/test stratifié
        2. Encodage des catégorielles + WoE binning
        3. Entraînement de 3 algorithmes
        4. Calibration isotonic pour la Logistic Regression
        5. Calcul des métriques (AUC, Gini, KS, PSI)
        6. Comparaison et export des résultats

    Attributes:
        seed: Graine aléatoire.
        woe_binner: Instance WoEBinner fittée.
        label_encoders: Encodeurs pour les variables catégorielles.
        results: Dictionnaire {model_name: PDModelResult}.
        X_train: Features d'entraînement.
        X_test: Features de test.
        y_train: Labels d'entraînement.
        y_test: Labels de test.
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        """Initialise la suite de modèles.

        Args:
            seed: Graine pour reproductibilité.
        """
        self.seed = seed
        self.woe_binner = WoEBinner()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.results: Dict[str, PDModelResult] = {}

        # Données (remplies par fit)
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        # Features WoE pour la LR
        self._woe_features: List[str] = []
        # Features brutes pour RF/XGB
        self._raw_features: List[str] = []

    def fit(self, df: pd.DataFrame) -> "PDModelSuite":
        """Pipeline principal : split, encode, train, calibrate, evaluate.

        Args:
            df: DataFrame complet avec features et target.

        Returns:
            Self (pattern fluent).
        """
        # 1. Split stratifié
        self._split_data(df)

        # 2. Encoder les catégorielles
        self._encode_categoricals()

        # 3. Préparer les features
        self._prepare_features()

        # 4. Entraîner les 3 modèles
        self._fit_logistic_regression()
        self._fit_random_forest()
        self._fit_xgboost()

        return self

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prédit les PD avec chaque modèle pour de nouvelles données.

        Args:
            df: DataFrame avec les mêmes features que l'entraînement.

        Returns:
            Dictionnaire {model_name: array de PD prédites}.
        """
        df_encoded = self._apply_encoding(df)
        predictions: Dict[str, np.ndarray] = {}

        for name, result in self.results.items():
            if name == "LR_WoE":
                df_woe = self.woe_binner.transform(df_encoded, self._get_numeric_cols())
                X = df_woe[self._woe_features].values
            else:
                X = df_encoded[self._raw_features].values

            predictions[name] = result.model.predict_proba(X)[:, 1]

        return predictions

    def get_comparison_table(self) -> pd.DataFrame:
        """Génère un tableau comparatif des 3 modèles.

        Returns:
            DataFrame avec métriques train/test pour chaque modèle.
        """
        records = []
        for name, result in self.results.items():
            records.append({
                "model": name,
                "auc_train": result.metrics_train["auc"],
                "auc_test": result.metrics_test["auc"],
                "gini_train": result.metrics_train["gini"],
                "gini_test": result.metrics_test["gini"],
                "ks_train": result.metrics_train["ks"],
                "ks_test": result.metrics_test["ks"],
                "psi": result.metrics_test["psi"],
                "overfit_gap": round(
                    result.metrics_train["auc"] - result.metrics_test["auc"], 4
                ),
            })
        return pd.DataFrame(records).sort_values("auc_test", ascending=False)

    def get_feature_importance_table(self) -> pd.DataFrame:
        """Retourne l'importance des features pour chaque modèle.

        Returns:
            DataFrame avec l'importance relative par feature et par modèle.
        """
        records = []
        for name, result in self.results.items():
            if result.feature_importance is not None:
                for feat, imp in result.feature_importance.items():
                    records.append({
                        "model": name,
                        "feature": feat,
                        "importance": round(imp, 4),
                    })
        return pd.DataFrame(records)

    # ──────────────────────────────────────────
    # MÉTHODES PRIVÉES
    # ──────────────────────────────────────────

    def _split_data(self, df: pd.DataFrame) -> None:
        """Split stratifié train/test.

        Args:
            df: DataFrame complet.
        """
        test_size = TEST_RATIO / (TEST_RATIO + (1 - TRAIN_RATIO - TEST_RATIO) + TRAIN_RATIO)
        # On fait un simple train/test split (70/30 pour avoir assez de données test)
        train_df, test_df = train_test_split(
            df,
            test_size=1 - TRAIN_RATIO,
            random_state=self.seed,
            stratify=df[TARGET],
        )
        self.X_train = train_df.drop(columns=[TARGET, "pd_latent", "client_id"], errors="ignore").reset_index(drop=True)
        self.X_test = test_df.drop(columns=[TARGET, "pd_latent", "client_id"], errors="ignore").reset_index(drop=True)
        self.y_train = train_df[TARGET].values
        self.y_test = test_df[TARGET].values

    def _encode_categoricals(self) -> None:
        """Encode les variables catégorielles avec LabelEncoder."""
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in self.X_train.columns]

        for col in cat_cols:
            le = LabelEncoder()
            self.X_train[col] = le.fit_transform(self.X_train[col].astype(str))
            self.X_test[col] = self.X_test[col].map(
                lambda x, _le=le: (
                    _le.transform([str(x)])[0]
                    if str(x) in _le.classes_
                    else -1
                )
            )
            self.label_encoders[col] = le

    def _get_numeric_cols(self) -> List[str]:
        """Retourne les colonnes numériques présentes dans X_train.

        Returns:
            Liste des noms de colonnes numériques.
        """
        return [c for c in NUMERICAL_FEATURES if c in self.X_train.columns]

    def _prepare_features(self) -> None:
        """Prépare les features WoE et brutes.

        - Impute les NaN par la médiane pour les features brutes
        - Applique le WoE binning pour la LR
        """
        numeric_cols = self._get_numeric_cols()
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in self.X_train.columns]

        # Imputer les NaN par la médiane (train)
        for col in numeric_cols:
            median_val = self.X_train[col].median()
            self.X_train[col] = self.X_train[col].fillna(median_val)
            self.X_test[col] = self.X_test[col].fillna(median_val)

        # WoE binning sur le train
        train_with_target = self.X_train.copy()
        train_with_target[TARGET] = self.y_train
        self.X_train = self.woe_binner.fit_transform(
            train_with_target, numeric_cols, TARGET,
        ).drop(columns=[TARGET])
        self.X_test = self.woe_binner.transform(self.X_test, numeric_cols)

        # Features WoE pour LR
        self._woe_features = [f"{c}_woe" for c in numeric_cols]

        # Features brutes (numériques + catégorielles encodées) pour RF/XGB
        self._raw_features = numeric_cols + cat_cols

    def _fit_logistic_regression(self) -> None:
        """Entraîne une Logistic Regression sur les features WoE avec calibration isotonic."""
        X_train_woe = self.X_train[self._woe_features].values
        X_test_woe = self.X_test[self._woe_features].values

        # Modèle de base
        base_lr = LogisticRegression(
            C=PD_CONFIG.lr_C,
            max_iter=PD_CONFIG.lr_max_iter,
            random_state=self.seed,
            solver="lbfgs",
        )

        # Calibration isotonic via cross-validation interne
        calibrated_lr = CalibratedClassifierCV(
            estimator=base_lr,
            method=PD_CONFIG.calibration_method,
            cv=5,
        )
        calibrated_lr.fit(X_train_woe, self.y_train)

        # Prédictions
        y_pred_train = calibrated_lr.predict_proba(X_train_woe)[:, 1]
        y_pred_test = calibrated_lr.predict_proba(X_test_woe)[:, 1]

        # Feature importance via les coefficients du modèle de base (re-fit)
        base_lr.fit(X_train_woe, self.y_train)
        coefs = np.abs(base_lr.coef_[0])
        coefs_norm = coefs / coefs.sum() if coefs.sum() > 0 else coefs
        feat_imp = dict(zip(self._woe_features, coefs_norm))

        self.results["LR_WoE"] = PDModelResult(
            name="LR_WoE",
            model=calibrated_lr,
            metrics_train=ModelMetrics.compute_all(self.y_train, y_pred_train),
            metrics_test=ModelMetrics.compute_all(
                self.y_test, y_pred_test, y_score_ref=y_pred_train,
            ),
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            feature_names=self._woe_features,
            feature_importance=feat_imp,
        )

    def _fit_random_forest(self) -> None:
        """Entraîne un Random Forest sur les features brutes."""
        X_train_raw = self.X_train[self._raw_features].values
        X_test_raw = self.X_test[self._raw_features].values

        rf = RandomForestClassifier(
            n_estimators=PD_CONFIG.rf_n_estimators,
            max_depth=PD_CONFIG.rf_max_depth,
            min_samples_leaf=PD_CONFIG.rf_min_samples_leaf,
            random_state=self.seed,
            n_jobs=-1,
        )
        rf.fit(X_train_raw, self.y_train)

        y_pred_train = rf.predict_proba(X_train_raw)[:, 1]
        y_pred_test = rf.predict_proba(X_test_raw)[:, 1]

        feat_imp = dict(zip(self._raw_features, rf.feature_importances_))

        self.results["RandomForest"] = PDModelResult(
            name="RandomForest",
            model=rf,
            metrics_train=ModelMetrics.compute_all(self.y_train, y_pred_train),
            metrics_test=ModelMetrics.compute_all(
                self.y_test, y_pred_test, y_score_ref=y_pred_train,
            ),
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            feature_names=self._raw_features,
            feature_importance=feat_imp,
        )

    def _fit_xgboost(self) -> None:
        """Entraîne un XGBoost sur les features brutes."""
        X_train_raw = self.X_train[self._raw_features].values
        X_test_raw = self.X_test[self._raw_features].values

        xgb = XGBClassifier(
            n_estimators=PD_CONFIG.xgb_n_estimators,
            max_depth=PD_CONFIG.xgb_max_depth,
            learning_rate=PD_CONFIG.xgb_learning_rate,
            subsample=PD_CONFIG.xgb_subsample,
            random_state=self.seed,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )
        xgb.fit(X_train_raw, self.y_train)

        y_pred_train = xgb.predict_proba(X_train_raw)[:, 1]
        y_pred_test = xgb.predict_proba(X_test_raw)[:, 1]

        feat_imp = dict(zip(self._raw_features, xgb.feature_importances_))

        self.results["XGBoost"] = PDModelResult(
            name="XGBoost",
            model=xgb,
            metrics_train=ModelMetrics.compute_all(self.y_train, y_pred_train),
            metrics_test=ModelMetrics.compute_all(
                self.y_test, y_pred_test, y_score_ref=y_pred_train,
            ),
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            feature_names=self._raw_features,
            feature_importance=feat_imp,
        )

    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique l'encodage catégoriel à un nouveau DataFrame.

        Args:
            df: DataFrame brut avec les mêmes colonnes.

        Returns:
            DataFrame avec catégorielles encodées.
        """
        df_out = df.copy()
        for col, le in self.label_encoders.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].map(
                    lambda x, _le=le: (
                        _le.transform([str(x)])[0]
                        if str(x) in _le.classes_
                        else -1
                    )
                )
        return df_out


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset

    print("=" * 65)
    print("IFRS 9 COCKPIT — Phase 2 : PD Models Benchmark")
    print("=" * 65)

    # Générer les données
    print("\n[1/4] Génération du dataset...")
    df_clients, _ = generate_dataset()
    print(f"       {len(df_clients):,} clients | DR = {df_clients[TARGET].mean():.2%}")

    # Entraîner les modèles
    print("\n[2/4] Entraînement des 3 modèles PD...")
    suite = PDModelSuite()
    suite.fit(df_clients)

    # WoE / IV
    print("\n[3/4] Information Value (WoE Binning) :")
    iv_table = suite.woe_binner.get_iv_table()
    print(iv_table.to_string(index=False))

    # Comparaison
    print("\n[4/4] Benchmark comparatif :")
    comparison = suite.get_comparison_table()
    print(comparison.to_string(index=False))

    # Détail par modèle
    for name, result in suite.results.items():
        print(f"\n--- {name} ---")
        print(f"  Train : AUC={result.metrics_train['auc']:.4f}  "
              f"Gini={result.metrics_train['gini']:.4f}  "
              f"KS={result.metrics_train['ks']:.4f}")
        print(f"  Test  : AUC={result.metrics_test['auc']:.4f}  "
              f"Gini={result.metrics_test['gini']:.4f}  "
              f"KS={result.metrics_test['ks']:.4f}  "
              f"PSI={result.metrics_test['psi']:.4f}")

    print("\n" + "=" * 65)
    print("Phase 2 validée.")
