"""Diagnostic v4.5 updated — analyse DGP 50/30/20."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from ifrs9_cockpit.synthetic_generator_v4 import AdvancedFinancialGenerator

gen = AdvancedFinancialGenerator(
    n_rows=50_000, seed=42, base_default_rate=0.05, noise_sigma=7.0,
    missing_rate=0.0,
)
df = gen.generate()

TARGET = "target_default"
print("=== DIAGNOSTIC v4.5 (50K, no MNAR) ===")
print(f"DR: {df[TARGET].mean():.2%}\n")

# 1. Revenue par secteur
print("--- Revenue par secteur ---")
for s in gen.SECTORS:
    m = df["sector"] == s
    r = df.loc[m, "revenue"] / 1e6
    print(f"  {s:12s} med={r.median():7.1f}M  std={r.std():7.1f}M  p10={r.quantile(.1):6.1f}M  p90={r.quantile(.9):6.1f}M")

# 2. DPD par secteur
print("\n--- DPD par secteur ---")
for s in gen.SECTORS:
    m = df["sector"] == s
    d = df.loc[m, "days_past_due"]
    pz = (d == 0).mean()
    mnz = d[d > 0].mean() if (d > 0).any() else 0
    print(f"  {s:12s} zero={pz:.0%}  mean_nz={mnz:.0f}j")

# 3. Vintage / stress identifiability
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_v = le.fit_transform(df["vintage_quarter"].values)
Xm = df[["gdp_growth", "unemployment_rate", "interest_rate_10y"]].values
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
acc = cross_val_score(rf, Xm, y_v, cv=3, scoring="accuracy").mean()

vs = sorted(df["vintage_quarter"].unique())
sl = {vs[i] for i in gen.stress_quarters if i < len(vs)}
ys = df["vintage_quarter"].isin(sl).astype(int).values
auc_s = cross_val_score(rf, Xm, ys, cv=3, scoring="roc_auc").mean()
print(f"\n--- Vintage identifiability ---")
print(f"  Vintage accuracy: {acc:.1%} (random={1/24:.1%})")
print(f"  Stress AUC:       {auc_s:.3f}")

# 4. Feature importance RF
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

meta = [TARGET, "pd_latent", "vintage_quarter"]
cat = ["sector", "company_size"]
num = [c for c in df.columns if c not in meta + cat and df[c].dtype != "object"]
X = df.drop(columns=[c for c in meta if c in df.columns])
y = df[TARGET]

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
])
pipe = Pipeline([("prep", prep), ("clf", RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42))])
pipe.fit(X, y)
fn = num + list(pipe.named_steps["prep"].transformers_[1][1].get_feature_names_out())
imps = pipe.named_steps["clf"].feature_importances_
sfi = sorted(zip(fn, imps), key=lambda x: -x[1])

from ifrs9_cockpit.synthetic_generator_v4 import NOISE_FEATURES
noise_set = set(NOISE_FEATURES)
print("\n--- Feature Importance RF (top 20) ---")
for i, (name, imp) in enumerate(sfi[:20]):
    tag = "BRUIT" if name in noise_set else ""
    print(f"  {i+1:2d}. {name:30s} {imp:.4f}  {tag}")

noise_imp = sum(imp for n, imp in sfi if n in noise_set)
print(f"\n  Bruit total: {noise_imp:.4f}")

# 5. AUC 3-fold
auc_scores = cross_val_score(pipe, X, y, cv=3, scoring="roc_auc")
print(f"  AUC RF 3-fold: {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")

# 6. Top correlations
print("\n--- |corr| avec target (top 15) ---")
nc = df[num + [TARGET]].corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
for f, c in nc.head(15).items():
    print(f"  {f:30s} {c:.4f}")

# 7. company_size vs revenue
print("\n--- company_size ---")
for sz in ["PME", "ETI", "GE"]:
    m = df["company_size"] == sz
    print(f"  {sz}: n={m.sum():,}  rev_med={df.loc[m,'revenue'].median()/1e6:.1f}M")

# 8. DR par secteur
print("\n--- DR par secteur ---")
for s in gen.SECTORS:
    m = df["sector"] == s
    print(f"  {s:12s} DR={df.loc[m, TARGET].mean():.2%}")

# 9. Macro correlation avec target
print("\n--- Macro x Target ---")
for feat in ["gdp_growth", "unemployment_rate", "interest_rate_10y"]:
    c = df[[feat, TARGET]].corr().iloc[0, 1]
    print(f"  {feat:25s} corr={c:.4f}")

print("\n=== FIN ===")
