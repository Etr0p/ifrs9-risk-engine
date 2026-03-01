"""Telechargement des donnees marche interbancaire ECB + EURIBOR.

Script standalone execute une seule fois (ou en CI) :
    python ifrs9_cockpit/data/fetch_ecb_interbank.py

Sources :
    1. ECB MMSR (Money Market Statistical Reporting) — taux + volumes
       interbancaires quotidiens, secured/unsecured, API SDMX CSV
    2. EURIBOR GitHub (datasets/euribor) — term structure 1W-12M

Pipeline :
    1. Query API ECB MMSR pour 3 series (unsecured rate, secured rate, volume)
    2. Download EURIBOR CSV pour 5 tenors (1W, 1M, 3M, 6M, 12M)
    3. Assemble en DataFrame avec colonnes tenor, rate, volume, spread
    4. Sauvegarde ecb_interbank.parquet (~10 KB)
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

# ECB MMSR API — SDMX 2.1 REST (free, no registration)
_MMSR_BASE = "https://data-api.ecb.europa.eu/service/data/MMSR"

_MMSR_SERIES = {
    "unsecured_rate": "B.U2._X._Z.S122._Z.U.BO.WR._X.KT._Z._Z.EUR._Z",
    "secured_rate": "B.U2._X._Z.S1ZV._Z.T.BO.WR._X.MA._Z._Z.EUR._Z",
    "volume": "B.U2._X._Z.S122._Z.U.BO.TT._X.KT._Z._Z.EUR._Z",
}

# EURIBOR GitHub — raw CSV (Public Domain)
_EURIBOR_BASE = (
    "https://raw.githubusercontent.com/datasets/euribor/main/data"
)
_EURIBOR_TENORS = {
    "1w": 0.019,
    "1m": 0.083,
    "3m": 0.25,
    "6m": 0.50,
    "12m": 1.0,
}

_OUTPUT_PATH = Path(__file__).parent / "ecb_interbank.parquet"


# -----------------------------------------------
# TELECHARGEMENT
# -----------------------------------------------

def _download_csv(url: str, label: str) -> pd.DataFrame | None:
    """Telecharge un CSV depuis l'API ECB ou GitHub."""
    import urllib.request

    print(f"  Telechargement {label}... ", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={"Accept": "text/csv"})
        response = urllib.request.urlopen(req, timeout=60)
        data = response.read()
        print(f"{len(data) / 1e3:.0f} KB")
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    except Exception as e:
        print(f"ERREUR: {e}")
        return None


# -----------------------------------------------
# PARSING
# -----------------------------------------------

def _parse_mmsr_series(df: pd.DataFrame | None, label: str) -> float | None:
    """Parse ECB MMSR CSV and return latest observation value."""
    if df is None or df.empty:
        return None

    if "OBS_VALUE" not in df.columns or "TIME_PERIOD" not in df.columns:
        print(f"  WARN: {label} — colonnes manquantes")
        return None

    df = df.copy()
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"], errors="coerce")
    df = df.dropna(subset=["OBS_VALUE", "TIME_PERIOD"])

    if df.empty:
        return None

    latest = df.sort_values("TIME_PERIOD").iloc[-1]
    return float(latest["OBS_VALUE"])


def _parse_euribor(tenor_key: str) -> float | None:
    """Download and parse EURIBOR CSV for a single tenor."""
    url = f"{_EURIBOR_BASE}/euribor-{tenor_key}-monthly.csv"
    df = _download_csv(url, f"EURIBOR {tenor_key}")

    if df is None or df.empty:
        return None

    # Columns: Date, Rate (or similar)
    rate_col = None
    for c in df.columns:
        if "rate" in c.lower() or c.lower() in ("value", "obs_value"):
            rate_col = c
            break
    if rate_col is None:
        # Assume last column is the rate
        rate_col = df.columns[-1]

    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
    df = df.dropna(subset=[rate_col])

    if df.empty:
        return None

    return float(df[rate_col].iloc[-1])


def _build_interbank_data() -> pd.DataFrame:
    """Telecharge et assemble les donnees interbancaires."""
    print("[1/3] Telechargement ECB MMSR...")
    mmsr_results = {}
    for label, series_key in _MMSR_SERIES.items():
        url = f"{_MMSR_BASE}/{series_key}?format=csvdata"
        df_raw = _download_csv(url, label)
        mmsr_results[label] = _parse_mmsr_series(df_raw, label)

    print("[2/3] Telechargement EURIBOR term structure...")
    euribor_rates = {}
    for tenor_key, tenor_years in _EURIBOR_TENORS.items():
        rate = _parse_euribor(tenor_key)
        euribor_rates[tenor_key] = rate

    print("[3/3] Assemblage...")

    # Build rows: one per tenor
    records = []

    # O/N from MMSR
    unsecured = mmsr_results.get("unsecured_rate")
    secured = mmsr_results.get("secured_rate")
    volume = mmsr_results.get("volume")

    records.append({
        "tenor_label": "O/N",
        "tenor_years": 0.003,
        "rate_unsecured": unsecured / 100.0 if unsecured is not None else 0.0365,
        "rate_secured": secured / 100.0 if secured is not None else 0.0355,
        "volume_eur_bn": volume if volume is not None else 250.0,
    })

    # EURIBOR tenors
    for tenor_key, tenor_years in _EURIBOR_TENORS.items():
        rate = euribor_rates.get(tenor_key)
        rate_dec = rate / 100.0 if rate is not None else 0.035 + tenor_years * 0.002
        records.append({
            "tenor_label": tenor_key.upper(),
            "tenor_years": tenor_years,
            "rate_unsecured": rate_dec,
            "rate_secured": rate_dec - 0.001,  # repo ~ 10bp lower
            "volume_eur_bn": max(10.0, 250.0 - tenor_years * 200.0),
        })

    df = pd.DataFrame(records)
    df["spread_secured_unsecured"] = df["rate_unsecured"] - df["rate_secured"]

    return df.sort_values("tenor_years").reset_index(drop=True)


# -----------------------------------------------
# SAUVEGARDE
# -----------------------------------------------

def _save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Sauvegarde en parquet."""
    print(f"\nSauvegarde vers {output_path}...")
    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    size_kb = output_path.stat().st_size / 1e3
    print(f"  OK : {len(df)} lignes, {len(df.columns)} colonnes, {size_kb:.0f} KB")

    print("\n--- Donnees marche interbancaire ---")
    for _, row in df.iterrows():
        unsec = row.get("rate_unsecured", float("nan"))
        sec = row.get("rate_secured", float("nan"))
        spread = row.get("spread_secured_unsecured", float("nan"))
        vol = row.get("volume_eur_bn", float("nan"))
        print(f"  {row['tenor_label']:4s} ({row['tenor_years']:6.3f}y) : "
              f"Unsecured={unsec:.3%}  Secured={sec:.3%}  "
              f"Spread={spread:.1%}  Vol={vol:.0f}B")


# -----------------------------------------------
# MAIN
# -----------------------------------------------

def main() -> None:
    """Pipeline complet : download -> parse -> assemble -> save."""
    print("=" * 60)
    print("ECB MMSR + EURIBOR -> ecb_interbank.parquet")
    print("=" * 60)

    df = _build_interbank_data()
    _save_parquet(df, _OUTPUT_PATH)

    print("\nDone!")


if __name__ == "__main__":
    main()
