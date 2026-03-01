"""Telechargement et pre-processing des donnees Lending Club.

Script standalone execute une seule fois (ou en CI) :
    python ifrs9_cockpit/data/fetch_lending_club.py

Source : Figshare (miroir libre, pas de credentials)
    https://figshare.com/articles/dataset/Lending_Club/22121477

Ce dataset Figshare est une version pre-nettoyee du dataset Lending Club
(~332k prets, 27 colonnes). Les types sont deja numeriques (int_rate=float,
emp_length=float, loan_status=0/1 binaire).

Pipeline :
    1. Telecharge les CSV train+test depuis Figshare
    2. Selectionne les colonnes pertinentes
    3. Mappe vers le schema cockpit
    4. Nettoie (dropna, types, bornes)
    5. Echantillon stratifie 50k (par grade x purpose x term)
    6. Sauvegarde consumer_credit.parquet (~1-3 MB)
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------

# Figshare direct download URLs (libre, pas de credentials)
_TRAIN_URL = "https://ndownloader.figshare.com/files/39316160"
_TEST_URL = "https://ndownloader.figshare.com/files/39495787"

# Colonnes a conserver du CSV brut (Figshare variant)
_KEEP_COLS = [
    "id", "loan_amnt", "term", "int_rate", "sub_grade",
    "annual_inc", "dti", "fico_range_low", "fico_range_high",
    "emp_length", "home_ownership", "purpose", "loan_status",
    "revol_util", "revol_bal", "open_acc", "pub_rec",
    "pub_rec_bankruptcies", "mort_acc", "total_acc",
    "installment", "issue_d", "addr_state",
    "verification_status", "application_type",
]

# Mapping purpose Lending Club -> schema cockpit
_PURPOSE_MAP = {
    "debt_consolidation": "personal",
    "credit_card": "revolving",
    "home_improvement": "home_improvement",
    "major_purchase": "personal",
    "car": "auto",
    "medical": "personal",
    "small_business": "personal",
    "moving": "personal",
    "vacation": "personal",
    "house": "home_improvement",
    "wedding": "personal",
    "renewable_energy": "home_improvement",
    "educational": "personal",
}

# Taille echantillons stratifies (train/test disjoints)
_TRAIN_SIZE = 50_000
_TEST_SIZE = 15_000

# Output (2 parquets disjoints pour eviter le data leakage)
_TRAIN_OUTPUT_PATH = Path(__file__).parent / "consumer_credit_train.parquet"
_TEST_OUTPUT_PATH = Path(__file__).parent / "consumer_credit_test.parquet"
# Legacy path (backward compat — symlink vers train)
_LEGACY_OUTPUT_PATH = Path(__file__).parent / "consumer_credit.parquet"


# -----------------------------------------------
# TELECHARGEMENT
# -----------------------------------------------

def _download_csv(url: str, label: str) -> pd.DataFrame:
    """Telecharge un CSV depuis une URL."""
    import urllib.request

    print(f"  Telechargement {label}... ", end="", flush=True)
    response = urllib.request.urlopen(url)
    data = response.read()
    print(f"{len(data) / 1e6:.1f} MB")

    df = pd.read_csv(
        io.BytesIO(data),
        low_memory=False,
        na_values=["", "NA", "n/a"],
    )
    return df


def _download_all() -> pd.DataFrame:
    """Telecharge et concatene train + test."""
    print("[1/7] Telechargement des donnees Lending Club depuis Figshare...")
    df_train = _download_csv(_TRAIN_URL, "train")
    df_test = _download_csv(_TEST_URL, "test")
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"  Total brut : {len(df):,} lignes, {len(df.columns)} colonnes")
    return df


# -----------------------------------------------
# PRE-PROCESSING
# -----------------------------------------------

def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Selectionne les colonnes pertinentes."""
    print("[2/7] Selection des colonnes...")
    available = [c for c in _KEEP_COLS if c in df.columns]
    missing = set(_KEEP_COLS) - set(available)
    if missing:
        print(f"  Colonnes absentes (ignorees) : {missing}")
    return df[available].copy()


def _clean_types(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les types (adapte au variant Figshare pre-nettoye)."""
    print("[3/7] Nettoyage des types...")

    # term: " 60 months" -> 5.0 years, or already numeric
    if "term" in df.columns:
        term_vals = df["term"]
        if term_vals.dtype == object:
            term_vals = (
                term_vals.astype(str).str.strip()
                .str.replace(" months", "", regex=False)
                .apply(pd.to_numeric, errors="coerce")
                / 12.0
            )
        else:
            # Already numeric: convert months to years
            term_vals = pd.to_numeric(term_vals, errors="coerce")
            # If values are 36/60 (months), convert; if 3/5 (years), keep
            if term_vals.median() > 10:
                term_vals = term_vals / 12.0
        df["term"] = term_vals

    # int_rate: already float in Figshare (e.g. 16.99), convert to decimal
    if "int_rate" in df.columns:
        int_vals = pd.to_numeric(df["int_rate"], errors="coerce")
        # If values > 1 => percentage, convert to decimal
        if int_vals.median() > 1.0:
            int_vals = int_vals / 100.0
        df["int_rate"] = int_vals

    # revol_util: already float (e.g. 86.8 = 86.8%), convert to decimal
    if "revol_util" in df.columns:
        ru_vals = pd.to_numeric(df["revol_util"], errors="coerce")
        if ru_vals.median() > 1.0:
            ru_vals = ru_vals / 100.0
        df["revol_util"] = ru_vals

    # emp_length: already float in Figshare (e.g. 2.0, 10.0)
    if "emp_length" in df.columns:
        df["emp_length"] = pd.to_numeric(df["emp_length"], errors="coerce")

    # credit_score = average of fico range
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["credit_score"] = (
            (pd.to_numeric(df["fico_range_low"], errors="coerce")
             + pd.to_numeric(df["fico_range_high"], errors="coerce"))
            / 2.0
        ).round(0)

    # grade: extract from sub_grade if grade not present
    if "sub_grade" in df.columns and "grade" not in df.columns:
        df["grade"] = df["sub_grade"].astype(str).str[0]  # "D1" -> "D"

    # issue_d: parse date
    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="mixed", errors="coerce")

    # Ensure numeric
    for col in ["loan_amnt", "annual_inc", "dti", "revol_bal",
                "open_acc", "pub_rec", "total_acc", "mort_acc",
                "installment", "pub_rec_bankruptcies"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _map_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Mappe les colonnes vers le schema cockpit."""
    print("[4/7] Mapping vers le schema cockpit...")

    # loan_status: Figshare variant uses binary int with INVERTED encoding:
    #   1 = Fully Paid (good loan), 0 = Charged Off (default)
    # Original LC rate ~17% defaults, so 1 is the majority class.
    if "loan_status" in df.columns:
        ls = pd.to_numeric(df["loan_status"], errors="coerce").fillna(0)
        if ls.max() <= 1 and ls.dtype in (np.int64, np.float64, int, float):
            # Binary: 1 = Fully Paid (good), 0 = Charged Off (default)
            df["default_flag"] = (1 - ls).astype(int)
        else:
            # String statuses (other variants)
            status = df["loan_status"].astype(str).str.strip()
            df["default_flag"] = status.isin(
                {"Charged Off", "Default"}
            ).astype(int)

    # payment_status: derive from default_flag
    df["payment_status"] = np.where(
        df["default_flag"] == 1, "default", "current",
    )

    # purpose mapping
    if "purpose" in df.columns:
        df["loan_purpose"] = df["purpose"].map(_PURPOSE_MAP).fillna("other")

    # home_ownership: standardize
    if "home_ownership" in df.columns:
        ho = df["home_ownership"].astype(str).str.upper().str.strip()
        ho = ho.replace({"ANY": "OTHER", "NONE": "OTHER"})
        df["home_ownership"] = ho

    # LGD observed: no recovery data in Figshare variant
    # Use EBA-calibrated prior based on grade
    _GRADE_LGD = {
        "A": 0.55, "B": 0.60, "C": 0.63, "D": 0.67,
        "E": 0.72, "F": 0.78, "G": 0.82,
    }
    if "grade" in df.columns:
        df["lgd_observed"] = df["grade"].map(_GRADE_LGD).fillna(0.65)
        # For defaults: slightly higher LGD
        default_mask = df["default_flag"] == 1
        df.loc[default_mask, "lgd_observed"] = np.minimum(
            df.loc[default_mask, "lgd_observed"] + 0.05, 0.95,
        )
    else:
        df["lgd_observed"] = 0.65

    # credit_grade from sub_grade
    if "sub_grade" in df.columns:
        df["credit_grade"] = df["sub_grade"]

    # Rename vers schema cockpit
    rename_map = {
        "loan_amnt": "loan_amount",
        "annual_inc": "borrower_income",
        "emp_length": "employment_length",
        "revol_util": "utilization_rate",
        "revol_bal": "revolving_balance",
        "open_acc": "nb_active_credits",
        "pub_rec": "public_records",
        "issue_d": "origination_date",
        "addr_state": "region",
        "int_rate": "interest_rate",
        "term": "remaining_tenor",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    return df


def _drop_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les lignes avec donnees critiques manquantes."""
    print("[5/7] Filtrage et suppression des NaN critiques...")
    n_before = len(df)

    critical_cols = ["loan_amount", "credit_score", "default_flag", "borrower_income"]
    available_critical = [c for c in critical_cols if c in df.columns]
    df = df.dropna(subset=available_critical)

    # Bornes realistes
    if "loan_amount" in df.columns:
        df = df[(df["loan_amount"] > 0) & (df["loan_amount"] < 100_000)]
    if "borrower_income" in df.columns:
        df = df[(df["borrower_income"] > 0) & (df["borrower_income"] < 1_000_000)]
    if "credit_score" in df.columns:
        df = df[(df["credit_score"] >= 300) & (df["credit_score"] <= 850)]
    if "dti" in df.columns:
        df = df[(df["dti"] >= 0) & (df["dti"] <= 100)]

    print(f"  {n_before:,} -> {len(df):,} lignes ({n_before - len(df):,} supprimees)")
    return df.reset_index(drop=True)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Echantillon stratifie par grade x purpose x term."""
    print(f"[6/7] Echantillon stratifie de {n:,} prets...")

    if len(df) <= n:
        print(f"  Dataset ({len(df):,}) <= echantillon ({n:,}), pas de sampling")
        return df

    # Stratification columns
    strat_cols = []
    for col in ["grade", "loan_purpose", "remaining_tenor"]:
        if col in df.columns:
            strat_cols.append(col)

    if not strat_cols:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    # Create strata key
    df = df.copy()
    df["_strata"] = df[strat_cols].astype(str).agg("_".join, axis=1)

    # Proportional sampling per stratum
    strata_counts = df["_strata"].value_counts()
    total = len(df)
    rng = np.random.default_rng(seed)

    samples = []
    remaining = n
    strata_list = list(strata_counts.items())

    for i, (stratum, count) in enumerate(strata_list):
        if i == len(strata_list) - 1:
            n_sample = remaining
        else:
            n_sample = max(1, round(n * count / total))
            n_sample = min(n_sample, remaining)

        stratum_df = df[df["_strata"] == stratum]
        if len(stratum_df) <= n_sample:
            samples.append(stratum_df)
            remaining -= len(stratum_df)
        else:
            idx = rng.choice(len(stratum_df), size=n_sample, replace=False)
            samples.append(stratum_df.iloc[idx])
            remaining -= n_sample

        if remaining <= 0:
            break

    result = pd.concat(samples, ignore_index=True)
    result.drop(columns=["_strata"], inplace=True)

    print(f"  Echantillon final : {len(result):,} lignes")
    return result


def _prepare_output(df: pd.DataFrame) -> pd.DataFrame:
    """Selectionne les colonnes finales et optimise les types."""
    final_cols = [
        "loan_amount", "remaining_tenor", "interest_rate",
        "grade", "credit_grade", "borrower_income", "dti",
        "credit_score", "employment_length", "home_ownership",
        "loan_purpose", "default_flag", "payment_status",
        "utilization_rate", "revolving_balance",
        "nb_active_credits", "public_records",
        "lgd_observed", "origination_date", "region",
    ]
    available = [c for c in final_cols if c in df.columns]
    df_out = df[available].copy()

    # Assign consumer_id
    df_out.insert(0, "consumer_id", np.arange(len(df_out)))

    # Optimize types for parquet
    for col in ["default_flag"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(np.int8)
    for col in ["nb_active_credits", "public_records"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(0).astype(np.int16)
    if "credit_score" in df_out.columns:
        df_out["credit_score"] = df_out["credit_score"].astype(np.int16)

    return df_out


def _save_parquet(df_out: pd.DataFrame, output_path: Path, label: str) -> None:
    """Sauvegarde un DataFrame en parquet et affiche les stats."""
    df_out.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    size_mb = output_path.stat().st_size / 1e6
    print(f"  {label}: {len(df_out):,} lignes, {len(df_out.columns)} colonnes, {size_mb:.1f} MB")

    # Stats
    print(f"    Taux de defaut : {df_out['default_flag'].mean():.2%}")
    if "credit_score" in df_out.columns:
        print(f"    Credit score : {df_out['credit_score'].mean():.0f} "
              f"(med={df_out['credit_score'].median():.0f})")
    if "grade" in df_out.columns:
        print(f"    Grades : {df_out['grade'].value_counts().sort_index().to_dict()}")


# -----------------------------------------------
# MAIN
# -----------------------------------------------

def main() -> None:
    """Pipeline complet : download -> clean -> split -> save 2 parquets."""
    print("=" * 60)
    print("Lending Club -> consumer_credit_{train,test}.parquet")
    print("=" * 60)

    df = _download_all()
    df = _select_columns(df)
    df = _clean_types(df)
    df = _map_schema(df)
    df = _drop_and_filter(df)

    # Stratified train/test split AVANT sampling (disjoints)
    total_sample = _TRAIN_SIZE + _TEST_SIZE
    df_sampled = _stratified_sample(df, n=total_sample, seed=42)

    from sklearn.model_selection import train_test_split as _tts
    strat_col = df_sampled["grade"] if "grade" in df_sampled.columns else None
    df_train, df_test = _tts(
        df_sampled,
        train_size=_TRAIN_SIZE,
        test_size=_TEST_SIZE,
        random_state=42,
        stratify=strat_col,
    )

    print(f"\n[7/7] Sauvegarde des 2 parquets disjoints...")
    df_train_out = _prepare_output(df_train.reset_index(drop=True))
    df_test_out = _prepare_output(df_test.reset_index(drop=True))

    _save_parquet(df_train_out, _TRAIN_OUTPUT_PATH, "Train")
    _save_parquet(df_test_out, _TEST_OUTPUT_PATH, "Test")

    # Legacy symlink (backward compat) — copie train vers l'ancien path
    import shutil
    shutil.copy2(_TRAIN_OUTPUT_PATH, _LEGACY_OUTPUT_PATH)
    print(f"  Legacy: {_LEGACY_OUTPUT_PATH.name} (copie de train)")

    print("\nDone!")


if __name__ == "__main__":
    main()
