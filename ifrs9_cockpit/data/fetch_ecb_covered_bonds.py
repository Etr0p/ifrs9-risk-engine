"""Telechargement et pre-processing des donnees ECB Eligible Assets — Covered Bonds.

Script standalone execute une seule fois (ou en CI) :
    python ifrs9_cockpit/data/fetch_ecb_covered_bonds.py

Source : ECB Eligible Assets Database (gratuit, quotidien)
    https://www.ecb.europa.eu/paym/coll/assets/html/list-MID.en.html

Ce dataset contient tous les actifs eligibles comme collateral aux operations
ECB. On filtre les covered bonds (Pfandbriefe, obligations foncieres, cedulas)
et on mappe vers le schema cockpit.

Pipeline :
    1. Telecharge le CSV compresse BCE (~5-10 MB)
    2. Filtre les covered bonds (POTENTIALLY_OWN_USABLE_COVERED_BOND)
    3. Selectionne les colonnes pertinentes
    4. Mappe vers le schema cockpit
    5. Derive rating implicite depuis haircut/categorie
    6. Nettoie (dropna maturite, bornes coupon, maturites residuelles >0)
    7. Sauvegarde covered_bonds.parquet (~1-3 MB)
"""

from __future__ import annotations

import gzip
import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------

# ECB Eligible Assets Database — CSV compresse gzip
# URL pattern : ea_csv_YYMMDD.csv.gz (date du jour)
_ECB_BASE_URL = "https://www.ecb.europa.eu/paym/coll/assets/html/dla/ea_MID"

# Colonnes ECB a conserver
_KEEP_COLS = [
    "ISIN_CODE", "TYPE", "HAIRCUT_CATEGORY", "ISSUER_NAME",
    "ISSUER_RESIDENCE", "ISSUER_GROUP", "COUPON_RATE",
    "COUPON_DEFINITION", "ISSUANCE_DATE", "MATURITY_DATE",
    "DENOMINATION", "HAIRCUT", "REFERENCE_MARKET",
    "POTENTIALLY_OWN_USABLE_COVERED_BOND",
]

# Rating implicite depuis haircut_category ECB
_RATING_FROM_HAIRCUT_CAT = {
    "I": "AAA",
    "II": "AA",
    "III": "A",
    "IV": "BBB",
    "V": "BB",
}

# Mapping haircut ranges → rating implicite (fallback)
_RATING_FROM_HAIRCUT_PCT = [
    (2.0, "AAA"),
    (5.0, "AA"),
    (8.0, "A"),
    (12.0, "BBB"),
    (100.0, "BB"),
]

# Taille echantillon max (pour limiter la taille du parquet)
_MAX_POSITIONS = 5000

# Output
_OUTPUT_PATH = Path(__file__).parent / "covered_bonds.parquet"


# -----------------------------------------------
# TELECHARGEMENT
# -----------------------------------------------

def _build_url(date_str: str | None = None) -> str:
    """Construit l'URL de telechargement ECB pour une date donnee."""
    if date_str is None:
        dt = datetime.now()
    else:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    yy = dt.strftime("%y")
    mm = dt.strftime("%m")
    dd = dt.strftime("%d")
    return f"{_ECB_BASE_URL}/ea_csv_{yy}{mm}{dd}.csv.gz"


def _download_csv(max_retries: int = 5) -> pd.DataFrame:
    """Telecharge le CSV compresse ECB, retry sur dates precedentes."""
    import urllib.request
    import urllib.error

    dt = datetime.now()
    for attempt in range(max_retries):
        url = _build_url(dt.strftime("%Y-%m-%d"))
        print(f"  Tentative {attempt + 1}/{max_retries}: {url}")
        try:
            response = urllib.request.urlopen(url, timeout=60)
            data = response.read()
            print(f"  Telecharge : {len(data) / 1e6:.1f} MB")
            # Decompress gzip
            decompressed = gzip.decompress(data)
            df = pd.read_csv(
                io.BytesIO(decompressed),
                low_memory=False,
                na_values=["", "NA", "n/a", "-"],
                encoding="utf-8",
                sep=",",
            )
            # Try semicolon separator if only 1 column
            if len(df.columns) <= 2:
                df = pd.read_csv(
                    io.BytesIO(decompressed),
                    low_memory=False,
                    na_values=["", "NA", "n/a", "-"],
                    encoding="utf-8",
                    sep=";",
                )
            return df
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"  Erreur: {e}")
            # Try previous day
            from datetime import timedelta
            dt = dt - timedelta(days=1)

    raise RuntimeError(
        f"Impossible de telecharger les donnees ECB apres {max_retries} tentatives. "
        "Verifiez votre connexion Internet."
    )


# -----------------------------------------------
# PRE-PROCESSING
# -----------------------------------------------

def _filter_covered_bonds(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre les covered bonds depuis les eligible assets."""
    print("[2/7] Filtrage des covered bonds...")
    n_before = len(df)

    # Normalize column names (ECB sometimes uses mixed case)
    df.columns = [c.strip().upper() for c in df.columns]

    # Primary filter: POTENTIALLY_OWN_USABLE_COVERED_BOND == "Y"
    mask = pd.Series(False, index=df.index)
    if "POTENTIALLY_OWN_USABLE_COVERED_BOND" in df.columns:
        mask = df["POTENTIALLY_OWN_USABLE_COVERED_BOND"].astype(str).str.strip().str.upper() == "Y"

    # Secondary filter: TYPE contains "covered" (case insensitive)
    if "TYPE" in df.columns:
        type_mask = df["TYPE"].astype(str).str.lower().str.contains("covered", na=False)
        mask = mask | type_mask

    # Tertiary filter: HAIRCUT_CATEGORY indicates covered bond
    if "HAIRCUT_CATEGORY" in df.columns:
        cat_mask = df["HAIRCUT_CATEGORY"].astype(str).str.lower().str.contains("covered", na=False)
        mask = mask | cat_mask

    df_cb = df[mask].copy()
    print(f"  {n_before:,} actifs -> {len(df_cb):,} covered bonds")

    if len(df_cb) == 0:
        raise ValueError("Aucun covered bond trouve dans les donnees ECB.")

    return df_cb


def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Selectionne les colonnes pertinentes."""
    print("[3/7] Selection des colonnes...")
    # Normalize to uppercase for matching
    available = [c for c in [col.upper() for col in _KEEP_COLS] if c in df.columns]
    missing = set(col.upper() for col in _KEEP_COLS) - set(available)
    if missing:
        print(f"  Colonnes absentes (ignorees) : {missing}")
    return df[available].copy()


def _derive_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Derive le rating implicite depuis la categorie de haircut."""
    print("[5/7] Derivation du rating implicite...")

    ratings = pd.Series("A", index=df.index)  # default

    # From haircut category (Roman numeral or digit)
    if "HAIRCUT_CATEGORY" in df.columns:
        cat = df["HAIRCUT_CATEGORY"].astype(str).str.strip()
        # Extract Roman numeral or digit
        for key, rating in _RATING_FROM_HAIRCUT_CAT.items():
            mask = cat.str.contains(key, na=False, regex=False)
            ratings[mask] = rating

    # Fallback: from haircut percentage
    if "HAIRCUT" in df.columns:
        haircut = pd.to_numeric(df["HAIRCUT"], errors="coerce").fillna(5.0)
        for threshold, rating in _RATING_FROM_HAIRCUT_PCT:
            mask = (haircut <= threshold) & (ratings == "A")  # only override default
            ratings[mask] = rating

    df["implied_rating"] = ratings
    print(f"  Ratings: {ratings.value_counts().to_dict()}")
    return df


def _map_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Mappe les colonnes ECB vers le schema cockpit."""
    print("[4/7] Mapping vers le schema cockpit...")

    result = pd.DataFrame()

    # bond_id (ISIN)
    if "ISIN_CODE" in df.columns:
        result["bond_id"] = df["ISIN_CODE"].astype(str).str.strip()
    else:
        result["bond_id"] = [f"CB_{i:06d}" for i in range(len(df))]

    # issuer_name
    if "ISSUER_NAME" in df.columns:
        result["issuer_name"] = df["ISSUER_NAME"].astype(str).str.strip()
    else:
        result["issuer_name"] = "Unknown"

    # issuer_country
    if "ISSUER_RESIDENCE" in df.columns:
        result["issuer_country"] = df["ISSUER_RESIDENCE"].astype(str).str.strip().str.upper()
    else:
        result["issuer_country"] = "DE"

    # issuer_group
    if "ISSUER_GROUP" in df.columns:
        result["issuer_group"] = df["ISSUER_GROUP"].astype(str).str.strip()
    else:
        result["issuer_group"] = result["issuer_name"]

    # coupon_type
    if "COUPON_DEFINITION" in df.columns:
        cd = df["COUPON_DEFINITION"].astype(str).str.strip().str.lower()
        result["coupon_type"] = np.where(cd.str.contains("fix", na=False), "fixed", "floating")
    else:
        result["coupon_type"] = "fixed"

    # coupon_rate_pct
    if "COUPON_RATE" in df.columns:
        result["coupon_rate_pct"] = pd.to_numeric(df["COUPON_RATE"], errors="coerce").fillna(2.5)
    else:
        result["coupon_rate_pct"] = 2.5

    # denomination (currency)
    if "DENOMINATION" in df.columns:
        result["denomination"] = df["DENOMINATION"].astype(str).str.strip().str.upper()
    else:
        result["denomination"] = "EUR"

    # issuance_date, maturity_date
    for col_src, col_dst in [("ISSUANCE_DATE", "issuance_date"), ("MATURITY_DATE", "maturity_date")]:
        if col_src in df.columns:
            result[col_dst] = pd.to_datetime(df[col_src], format="mixed", errors="coerce")
        else:
            result[col_dst] = pd.NaT

    # tenor_years (calculated)
    now = pd.Timestamp.now()
    if "maturity_date" in result.columns:
        result["tenor_years"] = (
            (result["maturity_date"] - now).dt.days / 365.25
        ).round(2)
    else:
        result["tenor_years"] = 5.0

    # haircut_category
    if "HAIRCUT_CATEGORY" in df.columns:
        result["haircut_category"] = df["HAIRCUT_CATEGORY"].astype(str).str.strip()
    else:
        result["haircut_category"] = "II"

    # haircut_pct
    if "HAIRCUT" in df.columns:
        result["haircut_pct"] = pd.to_numeric(df["HAIRCUT"], errors="coerce").fillna(3.0)
    else:
        result["haircut_pct"] = 3.0

    return result


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les donnees (dropna, bornes, maturites residuelles >0)."""
    print("[6/7] Nettoyage...")
    n_before = len(df)

    # Drop missing maturity
    df = df.dropna(subset=["maturity_date"])

    # Filter positive tenor (residual maturity > 0)
    df = df[df["tenor_years"] > 0]

    # Clip coupon to reasonable range
    df["coupon_rate_pct"] = df["coupon_rate_pct"].clip(0, 15)

    # Clip tenor
    df["tenor_years"] = df["tenor_years"].clip(0.1, 30)

    # Clip haircut
    df["haircut_pct"] = df["haircut_pct"].clip(0, 50)

    print(f"  {n_before:,} -> {len(df):,} lignes ({n_before - len(df):,} supprimees)")
    return df.reset_index(drop=True)


def _finalize_and_save(df: pd.DataFrame, output_path: Path) -> None:
    """Selectionne les colonnes finales et sauvegarde en parquet."""
    print(f"[7/7] Sauvegarde vers {output_path}...")

    final_cols = [
        "bond_id", "issuer_name", "issuer_country", "issuer_group",
        "coupon_type", "coupon_rate_pct", "denomination",
        "issuance_date", "maturity_date", "tenor_years",
        "haircut_category", "haircut_pct", "implied_rating",
    ]
    available = [c for c in final_cols if c in df.columns]
    df_out = df[available].copy()

    # Limit size
    if len(df_out) > _MAX_POSITIONS:
        df_out = df_out.sample(n=_MAX_POSITIONS, random_state=42).reset_index(drop=True)

    df_out.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Sauvegarde OK : {len(df_out):,} lignes, {len(df_out.columns)} colonnes, {size_mb:.1f} MB")

    # Stats
    print("\n--- Statistiques ---")
    if "implied_rating" in df_out.columns:
        print(f"  Ratings : {df_out['implied_rating'].value_counts().to_dict()}")
    if "issuer_country" in df_out.columns:
        top5 = df_out["issuer_country"].value_counts().head(5)
        print(f"  Top 5 pays : {top5.to_dict()}")
    if "coupon_type" in df_out.columns:
        print(f"  Coupon type : {df_out['coupon_type'].value_counts().to_dict()}")
    if "tenor_years" in df_out.columns:
        print(f"  Tenor moyen : {df_out['tenor_years'].mean():.1f} ans")
    if "denomination" in df_out.columns:
        print(f"  Devises : {df_out['denomination'].value_counts().head(3).to_dict()}")


# -----------------------------------------------
# MAIN
# -----------------------------------------------

def main() -> None:
    """Pipeline complet : download -> filter -> map -> clean -> save."""
    print("=" * 60)
    print("ECB Eligible Assets -> covered_bonds.parquet")
    print("=" * 60)

    print("[1/7] Telechargement des donnees ECB...")
    df = _download_csv()
    print(f"  Total brut : {len(df):,} lignes, {len(df.columns)} colonnes")

    df = _filter_covered_bonds(df)
    df_mapped = _map_schema(df)
    df_mapped = _derive_rating(df)
    # Re-map after deriving rating (need to carry implied_rating)
    df_mapped = _map_schema(df)
    df_mapped["implied_rating"] = _derive_rating(df)["implied_rating"]
    df_mapped = _clean(df_mapped)
    _finalize_and_save(df_mapped, _OUTPUT_PATH)

    print("\nDone!")


if __name__ == "__main__":
    main()
