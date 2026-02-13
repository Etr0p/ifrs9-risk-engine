"""Modeles de Probabilite de Defaut (PD) pour le Cockpit IFRS 9.

Implemente 3 algorithmes avec calibration isotonique sur chacun :
    1. Logistic Regression sur features WoE + calibration isotonique
    2. TabNet (deep learning tabulaire) + calibration isotonique
    3. XGBoost + calibration isotonique

3 familles genuinement differentes :
    - LR_WoE : lineaire (frontiere convexe, interpretable)
    - TabNet : deep learning tabulaire (attention sequentielle, Sparsemax, Ghost BN)
    - XGBoost : ensemble d'arbres (frontiere en escalier)

Scorecard Entreprise Robuste (LR_WoE) :
    - Monotonicite WoE imposee via PAV + granularite minimale post-PAV
    - Bin NaN explicite avec fusion WoE-proximity si sous-peuple
    - Selection IV >= seuil configurable
    - Filtrage VIF (multicolinearite) avant contrainte de signe
    - Contrainte beta < 0 (coherence economique)
    - Prior correction (biais d'echantillonnage)
    - Scaling scorecard (PDO/Score)

Cible modelisee : P(Default = 1 | X), i.e. la probabilite de defaut.
    Convention : WoE positif = tranche sure, beta < 0 = WoE eleve diminue P(defaut).
    Certains systemes modelisent P(Non-Default = 1), ce qui inverse les signes.
    Ici, nous modelisons explicitement P(Default = 1).

Architecture Train/Serve :
    - save() : serialise la suite entrainee (sans X_train pour economie memoire)
    - load() : charge une suite pre-entrainee depuis le disque
    - Le dashboard peut charger des modeles entraines offline sur 1M+ lignes

La classe PDModelSuite encapsule le pipeline complet :
    data split -> WoE binning -> fit 3 modeles -> calibration -> metriques -> comparaison.

L'analyste peut selectionner le modele actif (FR6) via select_model().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from statsmodels.stats.outliers_influence import variance_inflation_factor

from ifrs9_cockpit.config import (
    CATEGORICAL_FEATURES,
    CLIPPING_BOUNDS,
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
        name: Nom du modele (LR_WoE, TabNet, XGBoost).
        model: Objet modele entraine (CalibratedClassifierCV).
        metrics_train: Metriques sur le jeu d'entrainement.
        metrics_test: Metriques sur le jeu de test.
        y_pred_train: Probabilites predites (train).
        y_pred_test: Probabilites predites (test).
        feature_names: Noms des features utilisees.
        feature_importance: Importance des features (si disponible).
        scorecard_params: Parametres du scaling scorecard (LR_WoE uniquement).
    """

    name: str
    model: Any
    metrics_train: Dict[str, float]
    metrics_test: Dict[str, float]
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    feature_names: List[str]
    feature_importance: Optional[Dict[str, float]] = None
    scorecard_params: Optional[Dict[str, float]] = None


class _TabNetSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper sklearn-compatible pour TabNetClassifier.

    Necessaire car CalibratedClassifierCV appelle clone(model).fit(X, y)
    sans passer eval_set. Le wrapper gere le split validation en interne.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.3,
        lambda_sparse: float = 1e-3,
        lr: float = 0.02,
        batch_size: int = 8192,
        virtual_batch_size: int = 512,
        max_epochs: int = 200,
        patience: int = 20,
        seed: int = 42,
    ) -> None:
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.lr = lr
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self._model = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TabNetSklearnWrapper":
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Split interne 85/15 pour early stopping
        n = len(X)
        n_val = max(int(n * 0.15), 1)
        indices = np.random.RandomState(self.seed).permutation(n)
        idx_train, idx_val = indices[n_val:], indices[:n_val]

        # Adapter batch_size et virtual_batch_size au dataset
        # (TabNet echoue si batch > n_train)
        n_train = len(idx_train)
        effective_batch = min(self.batch_size, n_train)
        effective_vbs = min(self.virtual_batch_size, effective_batch)

        self._model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_params={"lr": self.lr},
            device_name="auto",
            seed=self.seed,
            verbose=0,
        )

        self._model.fit(
            X[idx_train], y[idx_train],
            eval_set=[(X[idx_val], y[idx_val])],
            eval_metric=["auc"],
            batch_size=effective_batch,
            virtual_batch_size=effective_vbs,
            max_epochs=self.max_epochs,
            patience=self.patience,
        )

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Importance des features via les masques d'attention TabNet (natif)."""
        return self._model.feature_importances_


class PDModelSuite:
    """Suite de modeles PD avec calibration isotonique et selection de modele actif.

    Orchestre le pipeline complet de modelisation PD :
        1. Split train/test stratifie
        2. Encodage des categorielles + WoE binning
        3. Entrainement de 3 algorithmes avec calibration isotonique
        4. Calcul des metriques (AUC, Gini, KS, PSI)
        5. Selection du modele actif (defaut : LR_WoE)

    Scorecard robuste (LR_WoE) :
        - Selection IV (iv_min_threshold)
        - Contrainte beta < 0 (iterative)
        - Prior correction (biais class_weight="balanced")
        - Scaling scorecard (PDO/Score)

    Attributes:
        seed: Graine aleatoire.
        woe_binner: Instance WoEBinner fittee.
        label_encoders: Encodeurs pour les variables categorielles.
        results: Dictionnaire {model_name: PDModelResult}.
        active_model_name: Nom du modele actif pour predict_active().
    """

    AVAILABLE_MODELS: Tuple[str, ...] = ("LR_WoE", "TabNet", "XGBoost")

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
        # Features brutes pour MLP/XGB
        self._raw_features: List[str] = []
        # StandardScaler pour le MLP (sensible aux echelles)
        self._scaler: Optional[StandardScaler] = None
        # Medianes pour imputation des NaN dans predict()
        self._medians: Dict[str, float] = {}

        # Tracabilite scorecard
        self._excluded_features: List[str] = []
        self._vif_dropped: List[str] = []
        self._sign_dropped: List[str] = []
        # LR de base non-calibree (pour prior correction et scorecard)
        self._base_lr: Optional[LogisticRegression] = None

    def fit(
        self,
        df: pd.DataFrame,
        models: tuple[str, ...] | None = None,
    ) -> PDModelSuite:
        """Pipeline principal : split, encode, train, calibrate, evaluate.

        Args:
            df: DataFrame credit complet avec features et target.
            models: Tuple de noms de modeles a entrainer (defaut: tous).
                Ex: ("LR_WoE", "XGBoost") pour skip TabNet.

        Returns:
            Self (pattern fluent).
        """
        if models is None:
            models = self.AVAILABLE_MODELS

        self._split_data(df)
        self._encode_categoricals()
        self._prepare_features()

        if "LR_WoE" in models:
            self._fit_logistic_regression()
        if "TabNet" in models:
            self._fit_tabnet()
        if "XGBoost" in models:
            self._fit_xgboost()

        return self

    def select_model(self, name: str) -> None:
        """Selectionne le modele actif pour les predictions en aval (FR6).

        Args:
            name: Nom du modele ('LR_WoE', 'TabNet', 'XGBoost').

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
            elif name == "TabNet" and self._scaler is not None:
                X = self._scaler.transform(df_encoded[self._raw_features].values)
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

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Predit les scores scorecard pour le modele LR_WoE.

        Score = Offset + Factor × ln(p / (1-p)) avec p = PD calibree.
        Convention : score eleve = bon dossier (PD faible).

        Args:
            df: DataFrame avec les memes features que l'entrainement.

        Returns:
            Array de scores. Retourne un array vide si LR_WoE absent.
        """
        if "LR_WoE" not in self.results:
            return np.array([])

        result = self.results["LR_WoE"]
        if result.scorecard_params is None:
            return np.array([])

        # Obtenir les PD calibrees
        preds = self.predict(df)
        pd_calibrated = preds["LR_WoE"]

        return self._pd_to_score(pd_calibrated, result.scorecard_params)

    def save(self, path: str) -> None:
        """Serialise la suite entrainee (modeles + artefacts, sans X_train).

        X_train est supprime pour economiser la memoire (inutile en inference).
        X_test, y_test, y_pred_test sont conserves (metriques Tab 1, ROC, SHAP).

        Args:
            path: Chemin du fichier joblib de sortie.
        """
        import joblib
        x_train_backup = self.X_train
        self.X_train = None
        joblib.dump(self, path)
        self.X_train = x_train_backup

    @classmethod
    def load(cls, path: str) -> "PDModelSuite":
        """Charge une suite pre-entrainee depuis le disque.

        Args:
            path: Chemin du fichier joblib.

        Returns:
            PDModelSuite pre-entrainee.
        """
        import joblib
        return joblib.load(path)

    def _pd_to_score(
        self, pd_values: np.ndarray, params: Dict[str, float],
    ) -> np.ndarray:
        """Convertit des PD en scores via le scaling lineaire.

        Args:
            pd_values: Array de PD dans [0, 1].
            params: Parametres scorecard (factor, offset).

        Returns:
            Array de scores.
        """
        factor = params["factor"]
        offset = params["offset"]

        # Clipper les PD a un plancher realiste (Bale IRB floor = 0.03%)
        # Evite les scores extremes quand la calibration isotonique produit PD=0
        p = np.clip(pd_values, 3e-4, 1.0 - 3e-4)
        logit_p = np.log(p / (1.0 - p))

        # Score = Offset + Factor × logit (note: factor < 0 car PDO > 0)
        # Convention: score haut = bon dossier, donc Factor = -PDO/ln(2)
        return offset + factor * logit_p

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

    def _clip_and_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clippe les outliers et calcule les features engineered.

        Args:
            df: DataFrame avec les features brutes.

        Returns:
            DataFrame avec outliers clippes et features engineered ajoutees.
        """
        df = df.copy()

        # Clipping outliers
        for col, (lo, hi) in CLIPPING_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)

        # Feature engineering
        if "loan_amount" in df.columns and "revenue" in df.columns:
            df["loan_to_revenue"] = df["loan_amount"] / df["revenue"].clip(lower=1.0)
        if "collateral" in df.columns and "loan_amount" in df.columns:
            df["collateral_coverage"] = df["collateral"] / df["loan_amount"].clip(lower=1.0)

        return df

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

        # Clipping + feature engineering
        self.X_train = self._clip_and_engineer(self.X_train)
        self.X_test = self._clip_and_engineer(self.X_test)

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
        if self.X_train is not None:
            return [c for c in NUMERICAL_FEATURES if c in self.X_train.columns]
        # Apres load() X_train est None — utiliser _raw_features (persiste)
        if hasattr(self, "_raw_features") and self._raw_features:
            return [c for c in NUMERICAL_FEATURES if c in self._raw_features]
        return list(NUMERICAL_FEATURES)

    def _prepare_features(self) -> None:
        """Prepare les features WoE et brutes.

        - Impute les NaN par la mediane pour les features brutes (MLP/XGB)
        - Applique le WoE binning pour la LR (NaN geres via bin NaN explicite)
        - Filtre les features par IV >= iv_min_threshold
        """
        numeric_cols = self._get_numeric_cols()
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in self.X_train.columns]

        # WoE binning AVANT imputation mediane (pour conserver les NaN dans le bin NaN)
        train_with_target = self.X_train.copy()
        train_with_target[TARGET] = self.y_train
        self.X_train = self.woe_binner.fit_transform(
            train_with_target, numeric_cols, TARGET,
        ).drop(columns=[TARGET])
        self.X_test = self.woe_binner.transform(self.X_test, numeric_cols)

        # Imputer les NaN par la mediane pour les features brutes (MLP/XGB)
        for col in numeric_cols:
            median_val = self.X_train[col].median()
            self._medians[col] = float(median_val) if not np.isnan(median_val) else 0.0
            self.X_train[col] = self.X_train[col].fillna(self._medians[col])
            self.X_test[col] = self.X_test[col].fillna(self._medians[col])

        # --- Etape 3 : Selection IV ---
        all_woe_features = [f"{c}_woe" for c in numeric_cols]
        self._excluded_features = []
        self._woe_features = []

        for feat_woe in all_woe_features:
            base_feat = feat_woe.replace("_woe", "")
            iv_val = self.woe_binner.iv_.get(base_feat, 0.0)
            if iv_val >= PD_CONFIG.iv_min_threshold:
                self._woe_features.append(feat_woe)
            else:
                self._excluded_features.append(feat_woe)

        # Features brutes (numeriques + categorielles encodees) pour MLP/XGB
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
            cv=3,
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
        """Entraine une Logistic Regression sur les features WoE.

        Pipeline scorecard robuste :
            1. Fit initial avec class_weight="balanced"
            2. Contrainte beta < 0 (iterative)
            3. Prior correction (biais d'echantillonnage)
            4. Calibration isotonique
            5. Scaling scorecard (PDO/Score)
        """
        current_features = list(self._woe_features)
        self._vif_dropped = []
        self._sign_dropped = []

        # --- Etape 3b : Filtrage VIF (multicolinearite) ---
        # Un beta > 0 inattendu est souvent le symptome d'une multicolinearite
        # severe. Le filtrage VIF stabilise les signes des beta sans rejeter
        # arbitrairement des variables predictives (Anderson, 2007).
        vif_threshold = PD_CONFIG.vif_max_threshold
        if len(current_features) >= 2:
            keep_going = True
            while keep_going and len(current_features) >= 2:
                X_vif = self.X_train[current_features].values
                vifs = []
                for i in range(X_vif.shape[1]):
                    try:
                        vif_val = variance_inflation_factor(X_vif, i)
                    except (np.linalg.LinAlgError, ZeroDivisionError):
                        vif_val = float("inf")
                    vifs.append(vif_val)

                max_vif_idx = int(np.argmax(vifs))
                if vifs[max_vif_idx] > vif_threshold:
                    dropped = current_features.pop(max_vif_idx)
                    self._vif_dropped.append(dropped)
                else:
                    keep_going = False

        self._woe_features = current_features

        # --- Etape 4 : Contrainte de signe beta < 0 ---
        # WoE = ln(Sains/Defauts) => WoE eleve = bon dossier
        # beta < 0 => WoE monte => P(defaut) baisse (coherent)
        base_lr = None
        for _iteration in range(len(current_features)):
            if not current_features:
                break

            X_train_woe = self.X_train[current_features].values

            base_lr = LogisticRegression(
                C=PD_CONFIG.lr_C,
                max_iter=PD_CONFIG.lr_max_iter,
                random_state=self.seed,
                solver="lbfgs",
                class_weight="balanced",
            )
            base_lr.fit(X_train_woe, self.y_train)

            # Verifier les signes
            positive_mask = base_lr.coef_[0] > 0
            if not positive_mask.any():
                break

            # Supprimer les features avec beta > 0
            to_drop = [
                f for f, pos in zip(current_features, positive_mask) if pos
            ]
            self._sign_dropped.extend(to_drop)
            current_features = [f for f in current_features if f not in to_drop]

        # Mettre a jour les features retenues
        self._woe_features = current_features

        if not current_features or base_lr is None:
            # Fallback : garder toutes les features sans contrainte
            self._woe_features = [
                f"{c}_woe" for c in self._get_numeric_cols()
                if self.woe_binner.iv_.get(c, 0.0) >= PD_CONFIG.iv_min_threshold
            ]
            current_features = self._woe_features
            X_train_woe = self.X_train[current_features].values
            base_lr = LogisticRegression(
                C=PD_CONFIG.lr_C,
                max_iter=PD_CONFIG.lr_max_iter,
                random_state=self.seed,
                solver="lbfgs",
                class_weight="balanced",
            )
            base_lr.fit(X_train_woe, self.y_train)

        X_train_woe = self.X_train[current_features].values
        X_test_woe = self.X_test[current_features].values

        # --- Etape 6 : Prior correction (formule generalisee) ---
        # Formule : β₀_corr = β₀ + log(π_pop/(1-π_pop)) - log(π_train/(1-π_train))
        # Ref: King & Zeng (2001), "Logistic Regression in Rare Events Data"
        #
        # Avec class_weight="balanced", π_train effectif = 0.5 (les poids
        # equilibrent les classes). IMPORTANT : π_train doit etre exactement 0.5
        # pour que le second terme log(0.5/0.5) = 0 s'annule. Si on change la
        # strategie de sampling (ex. SMOTE, undersampling a un ratio != 50/50),
        # il faut mettre a jour pi_train en consequence.
        pi_pop = float(self.y_train.mean())  # prevalence reelle dans le portfolio
        pi_train = 0.5  # prevalence effective (balanced weighting → exactement 0.5)
        if 0 < pi_pop < 1 and 0 < pi_train < 1:
            correction = (
                np.log(pi_pop / (1 - pi_pop))
                - np.log(pi_train / (1 - pi_train))
            )
            base_lr.intercept_ = base_lr.intercept_ + correction

        # Stocker le LR de base (pre-calibration) pour le scoring
        self._base_lr = base_lr

        # Calibration isotonique
        calibrated, y_pred_train, y_pred_test = self._calibrate_model(
            base_lr, X_train_woe, X_test_woe,
        )

        # --- Etape 5 : Scaling scorecard ---
        pdo = PD_CONFIG.pdo
        target_score = PD_CONFIG.target_score
        target_odds = PD_CONFIG.target_odds

        factor = -pdo / np.log(2)
        offset = target_score - factor * np.log(target_odds)

        scorecard_params = {
            "factor": float(factor),
            "offset": float(offset),
            "pdo": float(pdo),
            "target_score": float(target_score),
            "target_odds": float(target_odds),
        }

        # Feature importance via les coefficients du modele de base
        coefs = np.abs(base_lr.coef_[0])
        coefs_norm = coefs / coefs.sum() if coefs.sum() > 0 else coefs
        feat_imp = dict(zip(current_features, coefs_norm))

        metrics_train, metrics_test = self._compute_metrics(y_pred_train, y_pred_test)

        self.results["LR_WoE"] = PDModelResult(
            name="LR_WoE",
            model=calibrated,
            metrics_train=metrics_train,
            metrics_test=metrics_test,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            feature_names=current_features,
            feature_importance=feat_imp,
            scorecard_params=scorecard_params,
        )

    def _fit_tabnet(self) -> None:
        """Entraine un TabNet (deep learning tabulaire) + calibration isotonique.

        TabNet utilise l'attention sequentielle avec Sparsemax pour selectionner
        les features pertinentes a chaque etape de decision. Ghost Batch
        Normalization assure la stabilite. Le StandardScaler est conserve
        pour coherence avec le pipeline predict().

        Feature importance native via les masques d'attention (pas d'approximation).
        """
        import torch

        X_train_raw = self.X_train[self._raw_features].values
        X_test_raw = self.X_test[self._raw_features].values

        # StandardScaler (TabNet est scale-invariant via Ghost BN,
        # mais le scaler garantit coherence du pipeline predict)
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train_raw)
        X_test_scaled = self._scaler.transform(X_test_raw)

        # Reproductibilite
        torch.manual_seed(self.seed)

        base_tabnet = _TabNetSklearnWrapper(
            n_d=PD_CONFIG.tabnet_n_d,
            n_a=PD_CONFIG.tabnet_n_a,
            n_steps=PD_CONFIG.tabnet_n_steps,
            gamma=PD_CONFIG.tabnet_gamma,
            lambda_sparse=PD_CONFIG.tabnet_lambda_sparse,
            lr=PD_CONFIG.tabnet_lr,
            batch_size=PD_CONFIG.tabnet_batch_size,
            virtual_batch_size=PD_CONFIG.tabnet_virtual_batch_size,
            max_epochs=PD_CONFIG.tabnet_max_epochs,
            patience=PD_CONFIG.tabnet_patience,
            seed=self.seed,
        )

        calibrated, y_pred_train, y_pred_test = self._calibrate_model(
            base_tabnet, X_train_scaled, X_test_scaled,
        )

        # Feature importance native via masques d'attention TabNet
        base_tabnet.fit(X_train_scaled, self.y_train)
        feat_imp_raw = base_tabnet.feature_importances_
        feat_imp_norm = feat_imp_raw / feat_imp_raw.sum() if feat_imp_raw.sum() > 0 else feat_imp_raw
        feat_imp = dict(zip(self._raw_features, feat_imp_norm))

        metrics_train, metrics_test = self._compute_metrics(y_pred_train, y_pred_test)

        self.results["TabNet"] = PDModelResult(
            name="TabNet",
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

        # Auto-detect GPU (CUDA) pour XGBoost
        try:
            import torch
            _xgb_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _xgb_device = "cpu"

        base_xgb = XGBClassifier(
            n_estimators=PD_CONFIG.xgb_n_estimators,
            max_depth=PD_CONFIG.xgb_max_depth,
            learning_rate=PD_CONFIG.xgb_learning_rate,
            subsample=PD_CONFIG.xgb_subsample,
            random_state=self.seed,
            eval_metric="logloss",
            verbosity=0,
            scale_pos_weight=spw,
            device=_xgb_device,
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
        """Applique l'encodage categoriel et l'imputation a un nouveau DataFrame.

        Args:
            df: DataFrame brut avec les memes colonnes.

        Returns:
            DataFrame avec categorielles encodees et NaN imputes.
        """
        df_out = df.copy()

        # Clipping + feature engineering (meme traitement que le train)
        df_out = self._clip_and_engineer(df_out)

        for col, le in self.label_encoders.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].map(
                    lambda x, _le=le: (
                        _le.transform([str(x)])[0]
                        if str(x) in _le.classes_
                        else -1
                    )
                )
        # Imputer les NaN avec les medianes du train
        for col, median_val in self._medians.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(median_val)
        return df_out


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset

    print("=" * 65)
    print("IFRS 9 COCKPIT — PD Models Benchmark (LR_WoE / TabNet / XGBoost)")
    print("=" * 65)

    # Generer les donnees (nouveau API : 3 DataFrames)
    print("\n[1/5] Generation du dataset...")
    df_credit, df_pe, df_history = generate_dataset()
    print(f"       {len(df_credit):,} entreprises | DR = {df_credit[TARGET].mean():.2%}")

    # Entrainer les modeles
    print("\n[2/5] Entrainement des 3 modeles PD (avec calibration isotonique)...")
    suite = PDModelSuite()
    suite.fit(df_credit)

    # WoE / IV
    print("\n[3/5] Information Value (WoE Binning) :")
    iv_table = suite.woe_binner.get_iv_table()
    print(iv_table.to_string(index=False))

    # IV filtering report
    print(f"\n  Features retenues (IV >= {PD_CONFIG.iv_min_threshold}) : "
          f"{len(suite._woe_features)}")
    if suite._excluded_features:
        print(f"  Features exclues (IV < seuil) : {suite._excluded_features}")

    # VIF filtering report
    if suite._vif_dropped:
        print(f"  Features supprimees (VIF > {PD_CONFIG.vif_max_threshold}) : {suite._vif_dropped}")

    # Sign constraint report
    if suite._sign_dropped:
        print(f"  Features supprimees (beta > 0) : {suite._sign_dropped}")
    print(f"  Features LR finales : {suite._woe_features}")

    # Beta coefficients
    if suite._base_lr is not None:
        print(f"\n  Coefficients LR (tous <= 0) :")
        for feat, coef in zip(suite._woe_features, suite._base_lr.coef_[0]):
            sign_ok = "OK" if coef <= 0 else "VIOLATION"
            print(f"    {feat:30s} beta = {coef:+.4f}  [{sign_ok}]")
        print(f"    {'intercept':30s} = {suite._base_lr.intercept_[0]:+.4f} (prior-corrected)")

    # Comparaison
    print(f"\n[4/5] Benchmark comparatif :")
    comparison = suite.get_comparison_table()
    print(comparison.to_string(index=False))

    # Scorecard distribution
    print(f"\n[5/5] Scorecard distribution :")
    lr_result = suite.results.get("LR_WoE")
    if lr_result and lr_result.scorecard_params:
        params = lr_result.scorecard_params
        print(f"  PDO = {params['pdo']:.0f} | Target Score = {params['target_score']:.0f} "
              f"| Target Odds = {params['target_odds']:.0f}:1")
        print(f"  Factor = {params['factor']:.4f} | Offset = {params['offset']:.4f}")

        # Calculer scores sur train
        scores_train = suite._pd_to_score(lr_result.y_pred_train, params)
        scores_test = suite._pd_to_score(lr_result.y_pred_test, params)
        print(f"  Train: min={scores_train.min():.0f} mean={scores_train.mean():.0f} "
              f"max={scores_train.max():.0f}")
        print(f"  Test:  min={scores_test.min():.0f} mean={scores_test.mean():.0f} "
              f"max={scores_test.max():.0f}")

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

    # Verification : monotonicity
    print(f"\n--- Validation monotonicite WoE ---")
    for feat, direction in suite.woe_binner.directions_.items():
        woe_detail = suite.woe_binner.get_woe_detail(feat)
        # Exclure bin_nan pour la verification de monotonicite
        woe_vals = woe_detail[woe_detail["bin"] != "bin_nan"]["woe"].values
        if len(woe_vals) >= 2:
            if direction == "increasing":
                monotone = all(woe_vals[i] <= woe_vals[i + 1] for i in range(len(woe_vals) - 1))
            else:
                monotone = all(woe_vals[i] >= woe_vals[i + 1] for i in range(len(woe_vals) - 1))
            status = "PASS" if monotone else "FAIL"
            print(f"  [{status}] {feat:20s} direction={direction:10s} bins={len(woe_vals)}")
            all_ok &= monotone

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
