"""Modeles de Probabilite de Defaut (PD) pour le Cockpit IFRS 9.

Implemente 3 algorithmes avec calibration isotonique sur chacun :
    1. Logistic Regression sur features WoE + calibration isotonique
    2. Random Forest + calibration isotonique
    3. XGBoost + calibration isotonique

La classe PDModelSuite encapsule le pipeline complet :
    data split -> WoE binning -> fit 3 modeles -> calibration -> metriques -> comparaison.

L'analyste peut selectionner le modele actif (FR6) via select_model().
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
)
from ifrs9_cockpit.models.woe import WoEBinner
from ifrs9_cockpit.analytics.metrics import ModelMetrics


@dataclass
class PDModelResult:
    """Resultat d'un modele PD individuel.

    Attributes:
        name: Nom du modele (LR_WoE, RandomForest, XGBoost).
        model: Objet modele entraine (CalibratedClassifierCV).
        metrics_train: Metriques sur le jeu d'entrainement.
        metrics_test: Metriques sur le jeu de test.
        y_pred_train: Probabilites predites (train).
        y_pred_test: Probabilites predites (test).
        feature_names: Noms des features utilisees.
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
    """Suite de modeles PD avec calibration isotonique et selection de modele actif.

    Orchestre le pipeline complet de modelisation PD :
        1. Split train/test stratifie
        2. Encodage des categorielles + WoE binning
        3. Entrainement de 3 algorithmes avec calibration isotonique
        4. Calcul des metriques (AUC, Gini, KS, PSI)
        5. Selection du modele actif (defaut : LR_WoE)

    Attributes:
        seed: Graine aleatoire.
        woe_binner: Instance WoEBinner fittee.
        label_encoders: Encodeurs pour les variables categorielles.
        results: Dictionnaire {model_name: PDModelResult}.
        active_model_name: Nom du modele actif pour predict_active().
    """

    AVAILABLE_MODELS: Tuple[str, ...] = ("LR_WoE", "RandomForest", "XGBoost")

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        """Initialise la suite de modeles.

        Args:
            seed: Graine pour reproductibilite.
        """
        self.seed = seed
        self.woe_binner = WoEBinner()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.results: Dict[str, PDModelResult] = {}
        self.active_model_name: str = "LR_WoE"

        # Donnees (remplies par fit)
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        # Features WoE pour la LR
        self._woe_features: List[str] = []
        # Features brutes pour RF/XGB
        self._raw_features: List[str] = []

    def fit(self, df: pd.DataFrame) -> PDModelSuite:
        """Pipeline principal : split, encode, train, calibrate, evaluate.

        Args:
            df: DataFrame credit complet avec features et target.

        Returns:
            Self (pattern fluent).
        """
        self._split_data(df)
        self._encode_categoricals()
        self._prepare_features()

        self._fit_logistic_regression()
        self._fit_random_forest()
        self._fit_xgboost()

        return self

    def select_model(self, name: str) -> None:
        """Selectionne le modele actif pour les predictions en aval (FR6).

        Args:
            name: Nom du modele ('LR_WoE', 'RandomForest', 'XGBoost').

        Raises:
            ValueError: Si le modele n'existe pas.
        """
        if name not in self.results:
            raise ValueError(
                f"Modele '{name}' inconnu. Disponibles : {list(self.results.keys())}"
            )
        self.active_model_name = name

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predit les PD avec chaque modele pour de nouvelles donnees.

        Args:
            df: DataFrame avec les memes features que l'entrainement.

        Returns:
            Dictionnaire {model_name: array de PD predites}.
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

    def predict_active(self, df: pd.DataFrame) -> np.ndarray:
        """Predit les PD avec le modele actif uniquement.

        Args:
            df: DataFrame avec les memes features que l'entrainement.

        Returns:
            Array de PD predites par le modele actif.
        """
        all_preds = self.predict(df)
        return all_preds[self.active_model_name]

    def get_comparison_table(self) -> pd.DataFrame:
        """Genere un tableau comparatif des 3 modeles.

        Returns:
            DataFrame avec metriques train/test pour chaque modele.
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
        """Retourne l'importance des features pour chaque modele.

        Returns:
            DataFrame avec l'importance relative par feature et par modele.
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
    # METHODES PRIVEES
    # ──────────────────────────────────────────

    def _split_data(self, df: pd.DataFrame) -> None:
        """Split stratifie train/test.

        Args:
            df: DataFrame credit complet.
        """
        train_df, test_df = train_test_split(
            df,
            test_size=1 - TRAIN_RATIO,
            random_state=self.seed,
            stratify=df[TARGET],
        )

        # Colonnes a exclure du feature set
        drop_cols = [TARGET, "pd_latent", "pd_origination", "enterprise_id"]
        self.X_train = train_df.drop(
            columns=[c for c in drop_cols if c in train_df.columns]
        ).reset_index(drop=True)
        self.X_test = test_df.drop(
            columns=[c for c in drop_cols if c in test_df.columns]
        ).reset_index(drop=True)
        self.y_train = train_df[TARGET].values
        self.y_test = test_df[TARGET].values

    def _encode_categoricals(self) -> None:
        """Encode les variables categorielles avec LabelEncoder."""
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
        """Retourne les colonnes numeriques presentes dans X_train."""
        return [c for c in NUMERICAL_FEATURES if c in self.X_train.columns]

    def _prepare_features(self) -> None:
        """Prepare les features WoE et brutes.

        - Impute les NaN par la mediane pour les features brutes
        - Applique le WoE binning pour la LR
        """
        numeric_cols = self._get_numeric_cols()
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in self.X_train.columns]

        # Imputer les NaN par la mediane (train)
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

        # Features brutes (numeriques + categorielles encodees) pour RF/XGB
        self._raw_features = numeric_cols + cat_cols

    def _calibrate_model(
        self,
        base_model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[Any, np.ndarray, np.ndarray]:
        """Calibre un modele via CalibratedClassifierCV isotonique.

        Args:
            base_model: Modele sklearn-compatible a calibrer.
            X_train: Features d'entrainement.
            X_test: Features de test.

        Returns:
            Tuple (modele_calibre, y_pred_train, y_pred_test).
        """
        calibrated = CalibratedClassifierCV(
            estimator=base_model,
            method=PD_CONFIG.calibration_method,
            cv=5,
        )
        calibrated.fit(X_train, self.y_train)

        y_pred_train = calibrated.predict_proba(X_train)[:, 1]
        y_pred_test = calibrated.predict_proba(X_test)[:, 1]

        return calibrated, y_pred_train, y_pred_test

    def _compute_metrics(
        self,
        y_pred_train: np.ndarray,
        y_pred_test: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calcule les metriques train et test."""
        metrics_train = ModelMetrics.compute_all(self.y_train, y_pred_train)
        metrics_test = ModelMetrics.compute_all(
            self.y_test, y_pred_test, y_score_ref=y_pred_train,
        )
        return metrics_train, metrics_test

    def _fit_logistic_regression(self) -> None:
        """Entraine une Logistic Regression sur les features WoE + calibration isotonique."""
        X_train_woe = self.X_train[self._woe_features].values
        X_test_woe = self.X_test[self._woe_features].values

        base_lr = LogisticRegression(
            C=PD_CONFIG.lr_C,
            max_iter=PD_CONFIG.lr_max_iter,
            random_state=self.seed,
            solver="lbfgs",
            class_weight="balanced",
        )

        calibrated, y_pred_train, y_pred_test = self._calibrate_model(
            base_lr, X_train_woe, X_test_woe,
        )

        # Feature importance via les coefficients du modele de base (re-fit)
        base_lr.fit(X_train_woe, self.y_train)
        coefs = np.abs(base_lr.coef_[0])
        coefs_norm = coefs / coefs.sum() if coefs.sum() > 0 else coefs
        feat_imp = dict(zip(self._woe_features, coefs_norm))

        metrics_train, metrics_test = self._compute_metrics(y_pred_train, y_pred_test)

        self.results["LR_WoE"] = PDModelResult(
            name="LR_WoE",
            model=calibrated,
            metrics_train=metrics_train,
            metrics_test=metrics_test,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            feature_names=self._woe_features,
            feature_importance=feat_imp,
        )

    def _fit_random_forest(self) -> None:
        """Entraine un Random Forest + calibration isotonique."""
        X_train_raw = self.X_train[self._raw_features].values
        X_test_raw = self.X_test[self._raw_features].values

        base_rf = RandomForestClassifier(
            n_estimators=PD_CONFIG.rf_n_estimators,
            max_depth=PD_CONFIG.rf_max_depth,
            min_samples_leaf=PD_CONFIG.rf_min_samples_leaf,
            random_state=self.seed,
            n_jobs=-1,
            class_weight="balanced",
        )

        calibrated, y_pred_train, y_pred_test = self._calibrate_model(
            base_rf, X_train_raw, X_test_raw,
        )

        # Feature importance depuis le modele de base (re-fit)
        base_rf.fit(X_train_raw, self.y_train)
        feat_imp = dict(zip(self._raw_features, base_rf.feature_importances_))

        metrics_train, metrics_test = self._compute_metrics(y_pred_train, y_pred_test)

        self.results["RandomForest"] = PDModelResult(
            name="RandomForest",
            model=calibrated,
            metrics_train=metrics_train,
            metrics_test=metrics_test,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            feature_names=self._raw_features,
            feature_importance=feat_imp,
        )

    def _fit_xgboost(self) -> None:
        """Entraine un XGBoost + calibration isotonique.

        Parametres fixes depuis PD_CONFIG (config-driven, NFR13).
        """
        X_train_raw = self.X_train[self._raw_features].values
        X_test_raw = self.X_test[self._raw_features].values

        # Ratio de desequilibre pour scale_pos_weight
        n_neg = (self.y_train == 0).sum()
        n_pos = max((self.y_train == 1).sum(), 1)
        spw = n_neg / n_pos

        base_xgb = XGBClassifier(
            n_estimators=PD_CONFIG.xgb_n_estimators,
            max_depth=PD_CONFIG.xgb_max_depth,
            learning_rate=PD_CONFIG.xgb_learning_rate,
            subsample=PD_CONFIG.xgb_subsample,
            random_state=self.seed,
            eval_metric="logloss",
            verbosity=0,
            scale_pos_weight=spw,
        )

        calibrated, y_pred_train, y_pred_test = self._calibrate_model(
            base_xgb, X_train_raw, X_test_raw,
        )

        # Feature importance depuis le modele de base (re-fit)
        base_xgb.fit(X_train_raw, self.y_train)
        feat_imp = dict(zip(self._raw_features, base_xgb.feature_importances_))

        metrics_train, metrics_test = self._compute_metrics(y_pred_train, y_pred_test)

        self.results["XGBoost"] = PDModelResult(
            name="XGBoost",
            model=calibrated,
            metrics_train=metrics_train,
            metrics_test=metrics_test,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            feature_names=self._raw_features,
            feature_importance=feat_imp,
        )

    def _apply_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique l'encodage categoriel a un nouveau DataFrame.

        Args:
            df: DataFrame brut avec les memes colonnes.

        Returns:
            DataFrame avec categorielles encodees.
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
    print("IFRS 9 COCKPIT — PD Models Benchmark (3 modeles + calibration)")
    print("=" * 65)

    # Generer les donnees (nouveau API : 3 DataFrames)
    print("\n[1/4] Generation du dataset...")
    df_credit, df_pe, df_history = generate_dataset()
    print(f"       {len(df_credit):,} entreprises | DR = {df_credit[TARGET].mean():.2%}")

    # Entrainer les modeles
    print("\n[2/4] Entrainement des 3 modeles PD (avec calibration isotonique)...")
    suite = PDModelSuite()
    suite.fit(df_credit)

    # WoE / IV
    print("\n[3/4] Information Value (WoE Binning) :")
    iv_table = suite.woe_binner.get_iv_table()
    print(iv_table.to_string(index=False))

    # Comparaison
    print("\n[4/4] Benchmark comparatif :")
    comparison = suite.get_comparison_table()
    print(comparison.to_string(index=False))

    # Detail par modele
    for name, result in suite.results.items():
        print(f"\n--- {name} (calibre) ---")
        print(f"  Train : AUC={result.metrics_train['auc']:.4f}  "
              f"Gini={result.metrics_train['gini']:.4f}  "
              f"KS={result.metrics_train['ks']:.4f}")
        print(f"  Test  : AUC={result.metrics_test['auc']:.4f}  "
              f"Gini={result.metrics_test['gini']:.4f}  "
              f"KS={result.metrics_test['ks']:.4f}  "
              f"PSI={result.metrics_test['psi']:.4f}")

    # Verification : PD dans [0, 1]
    print(f"\n--- Validation PD calibrees ---")
    all_ok = True
    for name, result in suite.results.items():
        pd_min = min(result.y_pred_train.min(), result.y_pred_test.min())
        pd_max = max(result.y_pred_train.max(), result.y_pred_test.max())
        ok = pd_min >= 0.0 and pd_max <= 1.0
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:15s} PD in [{pd_min:.6f}, {pd_max:.6f}]")
        all_ok &= ok

    # Verification : AUC > 0.70
    print(f"\n--- Validation AUC > 0.70 ---")
    for name, result in suite.results.items():
        auc_test = result.metrics_test["auc"]
        ok = auc_test > 0.70
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:15s} AUC test = {auc_test:.4f}")
        all_ok &= ok

    # Selection de modele
    print(f"\n--- Selection de modele (FR6) ---")
    print(f"  Modele actif : {suite.active_model_name}")
    suite.select_model("XGBoost")
    print(f"  Apres select_model('XGBoost') : {suite.active_model_name}")

    print(f"\n{'=' * 65}")
    if all_ok:
        print("Tous les modeles PD valides.")
    else:
        print("ATTENTION : certaines validations ont echoue.")
    print(f"{'=' * 65}")
