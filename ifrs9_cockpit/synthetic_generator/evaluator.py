"""
RobustEvaluator — Multi-model evaluation for synthetic financial data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import warnings

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from ifrs9_cockpit.synthetic_generator.constants import TARGET, LATENT_PD, NOISE_FEATURES

warnings.filterwarnings("ignore")


# ===========================================================================
# Evaluator
# ===========================================================================

class RobustEvaluator:
    """
    Evaluation multi-modele :
        - Split temporel (par vintage) OU StratifiedKFold
        - Baseline LogReg + RandomForest + XGBoost
        - Class weighting + OneHotEncoder
        - Metriques : AUC, Average Precision, Brier Score
        - Null-feature importance test
    """

    def __init__(self, df: pd.DataFrame, n_folds: int = 5, temporal_split: bool = True):
        self.df = df.copy()
        self.n_folds = n_folds
        self.temporal_split = temporal_split

        self._meta_cols = [TARGET, LATENT_PD, "vintage_quarter"]
        self._cat_cols = ["sector", "company_size"]
        self._num_cols = [
            c for c in df.columns
            if c not in self._meta_cols + self._cat_cols
            and df[c].dtype != "object"
        ]

    def _preprocessor(self):
        return ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), self._num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self._cat_cols),
        ])

    def _get_models(self, n_pos: int, n_neg: int) -> dict:
        models = {
            "LogReg (baseline)": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=15, class_weight="balanced",
                n_jobs=-1, random_state=42,
            ),
        }
        if _HAS_XGB:
            models["XGBoost"] = XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                scale_pos_weight=min(n_neg / max(n_pos, 1), 10),
                eval_metric="logloss", random_state=42, verbosity=0,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            )
        return models

    def evaluate(self):
        df = self.df
        X = df.drop(columns=[c for c in self._meta_cols if c in df.columns])
        y = df[TARGET]
        n_pos, n_neg = (y == 1).sum(), (y == 0).sum()

        if self.temporal_split and "vintage_quarter" in df.columns:
            vintages = sorted(df["vintage_quarter"].unique())
            split_pt = int(0.70 * len(vintages))
            train_set = set(vintages[:split_pt])
            train_mask = df["vintage_quarter"].isin(train_set)
            folds = [(X[train_mask], y[train_mask], X[~train_mask], y[~train_mask])]
            print(
                f"\n>>> SPLIT TEMPOREL : train {vintages[0]}..{vintages[split_pt-1]}"
                f" | test {vintages[split_pt]}..{vintages[-1]}"
            )
        else:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            folds = [
                (X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te])
                for tr, te in skf.split(X, y)
            ]
            print(f"\n>>> {self.n_folds}-FOLD STRATIFIED CV")

        models = self._get_models(n_pos, n_neg)
        results = {name: {"auc": [], "ap": [], "brier": []} for name in models}

        for X_tr, y_tr, X_te, y_te in folds:
            for name, model in models.items():
                pipe = Pipeline([("prep", self._preprocessor()), ("clf", model)])
                pipe.fit(X_tr, y_tr)
                probs = pipe.predict_proba(X_te)[:, 1]
                results[name]["auc"].append(roc_auc_score(y_te, probs))
                results[name]["ap"].append(average_precision_score(y_te, probs))
                results[name]["brier"].append(brier_score_loss(y_te, probs))

        multi = len(folds) > 1
        print(f"\n{'Model':<20} {'AUC':>12} {'Avg Prec':>12} {'Brier':>12}")
        print("-" * 58)
        for name in models:
            r = results[name]
            if multi:
                print(
                    f"{name:<20} "
                    f"{np.mean(r['auc']):.4f}+/-{np.std(r['auc']):.3f} "
                    f"{np.mean(r['ap']):.4f}+/-{np.std(r['ap']):.3f} "
                    f"{np.mean(r['brier']):.4f}+/-{np.std(r['brier']):.3f}"
                )
            else:
                print(
                    f"{name:<20} "
                    f"{r['auc'][0]:.4f}        "
                    f"{r['ap'][0]:.4f}        "
                    f"{r['brier'][0]:.4f}"
                )

        self._noise_feature_test(X, y)

    def _noise_feature_test(self, X: pd.DataFrame, y: pd.Series):
        # max_depth adapte a la taille : profondeur suffisante pour
        # capturer les interactions 2-3 way mais pas trop pour eviter
        # de surestimer l'importance du bruit
        n = len(X)
        depth = 8 if n < 100_000 else 10 if n < 500_000 else 12
        pipe = Pipeline([
            ("prep", self._preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=depth, class_weight="balanced",
                n_jobs=-1, random_state=42,
            )),
        ])
        pipe.fit(X, y)

        num_names = list(self._num_cols)
        cat_names = list(
            pipe.named_steps["prep"].transformers_[1][1].get_feature_names_out()
        )
        all_names = num_names + cat_names
        importances = pipe.named_steps["clf"].feature_importances_
        sorted_feats = sorted(zip(all_names, importances), key=lambda x: x[1], reverse=True)

        noise_set = set(NOISE_FEATURES)

        print(f"\n>>> FEATURE IMPORTANCE (Top 10 + bruit)")
        print(f"{'Feature':<35} {'Importance':>12} {'Type':>8}")
        print("-" * 58)

        shown_noise = set()
        for rank, (name, imp) in enumerate(sorted_feats):
            if rank < 10:
                tag = "BRUIT" if name in noise_set else "signal"
                print(f"  {name:<33} {imp:.4f}         {tag}")
                if name in noise_set:
                    shown_noise.add(name)

        remaining_noise = noise_set - shown_noise
        if remaining_noise:
            print("  ...")
            for rank, (name, imp) in enumerate(sorted_feats):
                if name in remaining_noise:
                    print(f"  {name:<33} {imp:.4f}         BRUIT  (rang {rank+1}/{len(sorted_feats)})")

        noise_imp = sum(imp for name, imp in sorted_feats if name in noise_set)
        max_noise = max((imp for name, imp in sorted_feats if name in noise_set), default=0)
        n_signal_below = sum(
            1 for name, imp in sorted_feats
            if name not in noise_set and imp < max_noise
        )

        print(f"\n  Importance totale signal : {1 - noise_imp:.4f}")
        print(f"  Importance totale bruit  : {noise_imp:.4f}")
        print(f"  Max importance bruit     : {max_noise:.4f}")
        print(f"  Features signal sous le max bruit : {n_signal_below}")

        if noise_imp < 0.05:
            print("  [OK] Bruit correctement marginalise (<5% importance totale)")
        elif noise_imp < 0.10:
            print("  [WARN] Bruit un peu eleve (5-10%)")
        else:
            print("  [FAIL] Bruit significatif (>10%) -- overfitting")
