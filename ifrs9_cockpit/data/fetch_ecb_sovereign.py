"""Telechargement des courbes de taux souveraines ECB.

Script standalone execute une seule fois (ou en CI) :
    python ifrs9_cockpit/data/fetch_ecb_sovereign.py

Source : ECB Statistical Data Warehouse, API SDMX 2.1 REST
    (gratuit, sans inscription, sans limite de requetes).

Contenu : courbes spot zone euro (AAA + all-govt), 7 maturites
    (3M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y).

Pipeline :
    1. Query API ECB (CSV format)
    2. Parse et pivot par maturite
    3. Derniere observation disponible
    4. Sauvegarde ecb_sovereign_yields.parquet
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

# ECB SDW SDMX API — Zero-coupon yield curves
# B.U2.EUR.4F.G_N_A.SV_C_YM = AAA-rated government bonds (euro area)
# B.U2.EUR.4F.G_N_C.SV_C_YM = All government bonds (euro area)
_AAA_URL = (
    "https://data-api.ecb.europa.eu/service/data/"
    "YC/B.U2.EUR.4F.G_N_A.SV_C_YM.?format=csvdata"
)
_ALL_GOVT_URL = (
    "https://data-api.ecb.europa.eu/service/data/"
    "YC/B.U2.EUR.4F.G_N_C.SV_C_YM.?format=csvdata"
)

# Maturites cibles (en mois pour match ECB)
_TARGET_TENORS = {
    "3M": 0.25,
    "1Y": 1.0,
    "2Y": 2.0,
    "5Y": 5.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}

# ECB DATA_SUFFIX codes for tenors
_TENOR_SUFFIXES = {
    "SR_0.25": "3M",
    "SR_1": "1Y",
    "SR_2": "2Y",
    "SR_5": "5Y",
    "SR_10": "10Y",
    "SR_20": "20Y",
    "SR_30": "30Y",
}

_OUTPUT_PATH = Path(__file__).parent / "ecb_sovereign_yields.parquet"


# -----------------------------------------------
# TELECHARGEMENT
# -----------------------------------------------

def _download_csv(url: str, label: str) -> pd.DataFrame:
    """Telecharge un CSV depuis l'API ECB."""
    import urllib.request

    print(f"  Telechargement {label}... ", end="", flush=True)
    req = urllib.request.Request(url, headers={"Accept": "text/csv"})
    response = urllib.request.urlopen(req, timeout=60)
    data = response.read()
    print(f"{len(data) / 1e3:.0f} KB")

    df = pd.read_csv(io.BytesIO(data), low_memory=False)
    return df


# -----------------------------------------------
# PARSING
# -----------------------------------------------

def _parse_ecb_csv(df: pd.DataFrame, curve_label: str) -> pd.DataFrame:
    """Parse ECB SDMX CSV into tenor × date format."""
    print(f"  Parsing {curve_label}...")

    # ECB CSV has columns: KEY, FREQ, REF_AREA, ..., DATA_SUFFIX, TIME_PERIOD, OBS_VALUE
    if "OBS_VALUE" not in df.columns or "TIME_PERIOD" not in df.columns:
        raise ValueError(f"Expected OBS_VALUE and TIME_PERIOD columns, got: {list(df.columns)[:10]}")

    # DATA_SUFFIX contains the maturity code (e.g., SR_10 = 10Y spot rate)
    suffix_col = None
    for candidate in ["DATA_SUFFIX", "MATURITY", "data_suffix"]:
        if candidate in df.columns:
            suffix_col = candidate
            break

    if suffix_col is None:
        # Try to extract from KEY
        if "KEY" in df.columns:
            # KEY format: YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10
            df["_tenor_code"] = df["KEY"].str.split(".").str[-1]
            suffix_col = "_tenor_code"
        else:
            raise ValueError("Cannot find tenor column in ECB data")

    # Filter to target tenors
    df_filtered = df[df[suffix_col].isin(_TENOR_SUFFIXES.keys())].copy()
    df_filtered["tenor_label"] = df_filtered[suffix_col].map(_TENOR_SUFFIXES)
    df_filtered["OBS_VALUE"] = pd.to_numeric(df_filtered["OBS_VALUE"], errors="coerce")
    df_filtered["TIME_PERIOD"] = pd.to_datetime(df_filtered["TIME_PERIOD"], errors="coerce")

    # Get latest observation per tenor
    df_latest = (
        df_filtered
        .sort_values("TIME_PERIOD")
        .groupby("tenor_label")
        .last()
        .reset_index()
    )

    result = pd.DataFrame({
        "tenor_label": df_latest["tenor_label"],
        "tenor_years": df_latest["tenor_label"].map(_TARGET_TENORS),
        f"yield_{curve_label}": df_latest["OBS_VALUE"].values / 100.0,  # % -> decimal
        "observation_date": df_latest["TIME_PERIOD"],
    })

    return result


def _build_yield_curves() -> pd.DataFrame:
    """Telecharge et assemble les deux courbes ECB."""
    print("[1/3] Telechargement courbes ECB...")
    df_aaa_raw = _download_csv(_AAA_URL, "AAA curve")
    df_all_raw = _download_csv(_ALL_GOVT_URL, "All-govt curve")

    print("[2/3] Parsing...")
    df_aaa = _parse_ecb_csv(df_aaa_raw, "aaa")
    df_all = _parse_ecb_csv(df_all_raw, "all_govt")

    # Merge on tenor
    df = pd.merge(
        df_aaa[["tenor_label", "tenor_years", "yield_aaa", "observation_date"]],
        df_all[["tenor_label", "yield_all_govt"]],
        on="tenor_label",
        how="outer",
    )

    # Spread = all_govt - AAA (captures average credit premium)
    if "yield_aaa" in df.columns and "yield_all_govt" in df.columns:
        df["spread_govt_aaa"] = df["yield_all_govt"] - df["yield_aaa"]

    return df.sort_values("tenor_years").reset_index(drop=True)


# -----------------------------------------------
# SAUVEGARDE
# -----------------------------------------------

def _save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Sauvegarde en parquet."""
    print(f"[3/3] Sauvegarde vers {output_path}...")

    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    size_kb = output_path.stat().st_size / 1e3
    print(f"  OK : {len(df)} lignes, {len(df.columns)} colonnes, {size_kb:.0f} KB")

    print("\n--- Courbes de taux ECB ---")
    for _, row in df.iterrows():
        aaa = row.get("yield_aaa", float("nan"))
        allg = row.get("yield_all_govt", float("nan"))
        spread = row.get("spread_govt_aaa", float("nan"))
        print(f"  {row['tenor_label']:4s} ({row['tenor_years']:5.2f}y) : "
              f"AAA={aaa:.3%}  AllGovt={allg:.3%}  Spread={spread:.1%}")


# -----------------------------------------------
# MAIN
# -----------------------------------------------

def main() -> None:
    """Pipeline complet : download -> parse -> merge -> save."""
    print("=" * 60)
    print("ECB Yield Curves -> ecb_sovereign_yields.parquet")
    print("=" * 60)

    df = _build_yield_curves()
    _save_parquet(df, _OUTPUT_PATH)

    print("\nDone!")


if __name__ == "__main__":
    main()
