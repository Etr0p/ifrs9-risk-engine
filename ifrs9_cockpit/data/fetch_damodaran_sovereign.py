"""Telechargement et pre-processing des donnees risque souverain Damodaran.

Script standalone execute une seule fois (ou en CI) :
    python ifrs9_cockpit/data/fetch_damodaran_sovereign.py

Source : Aswath Damodaran (NYU Stern), XLSX gratuit, mise a jour annuelle.
    https://pages.stern.nyu.edu/~adamodar/pc/datasets/ctryprem.xlsx

Contenu : 143+ pays avec rating Moody's, CDS spreads, default spreads,
    equity risk premium.

Pipeline :
    1. Telecharge le XLSX depuis NYU Stern
    2. Parse la feuille "ERPs by country"
    3. Filtre sur 9 pays zone euro
    4. Ajoute les donnees fiscales Eurostat (hardcodees, mise a jour annuelle)
    5. Sauvegarde damodaran_sovereign.parquet (~5 KB)
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

_XLSX_URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/ctryprem.xlsx"
_SHEET_NAME = "ERPs by country"

# 9 pays zone euro a extraire
_EUROZONE_COUNTRIES = {
    "Germany", "France", "Italy", "Spain", "Belgium",
    "Netherlands", "Austria", "Portugal", "Ireland",
}

# Mapping noms Damodaran -> code interne
_COUNTRY_CODE_MAP = {
    "Germany": "DE", "France": "FR", "Italy": "IT", "Spain": "ES",
    "Belgium": "BE", "Netherlands": "NL", "Austria": "AT",
    "Portugal": "PT", "Ireland": "IE",
}

# Donnees fiscales Eurostat 2024 (hardcodees, updates annuellement)
_FISCAL_DATA = {
    "DE": {"debt_to_gdp": 0.64, "fiscal_balance": -0.015, "recovery_rate_hist": 0.95},
    "FR": {"debt_to_gdp": 1.12, "fiscal_balance": -0.055, "recovery_rate_hist": 0.90},
    "IT": {"debt_to_gdp": 1.37, "fiscal_balance": -0.045, "recovery_rate_hist": 0.65},
    "ES": {"debt_to_gdp": 1.07, "fiscal_balance": -0.035, "recovery_rate_hist": 0.75},
    "BE": {"debt_to_gdp": 1.05, "fiscal_balance": -0.042, "recovery_rate_hist": 0.88},
    "NL": {"debt_to_gdp": 0.47, "fiscal_balance": 0.005, "recovery_rate_hist": 0.95},
    "AT": {"debt_to_gdp": 0.77, "fiscal_balance": -0.030, "recovery_rate_hist": 0.90},
    "PT": {"debt_to_gdp": 0.99, "fiscal_balance": -0.010, "recovery_rate_hist": 0.70},
    "IE": {"debt_to_gdp": 0.44, "fiscal_balance": 0.015, "recovery_rate_hist": 0.80},
}

_OUTPUT_PATH = Path(__file__).parent / "damodaran_sovereign.parquet"


# -----------------------------------------------
# TELECHARGEMENT
# -----------------------------------------------

def _download_xlsx(url: str) -> bytes:
    """Telecharge un XLSX depuis une URL."""
    import urllib.request

    print(f"  Telechargement {url}... ", end="", flush=True)
    response = urllib.request.urlopen(url, timeout=30)
    data = response.read()
    print(f"{len(data) / 1e6:.1f} MB")
    return data


# -----------------------------------------------
# PARSING
# -----------------------------------------------

def _parse_erp_sheet(data: bytes) -> pd.DataFrame:
    """Parse la feuille ERPs by country du XLSX Damodaran."""
    print("[2/5] Parsing feuille ERPs by country...")

    # Damodaran XLSX has headers at row 6-8 typically; try multiple approaches
    df = pd.read_excel(
        io.BytesIO(data),
        sheet_name=_SHEET_NAME,
        header=None,
    )

    # Find the header row (contains "Country" or "country")
    header_row = None
    for i in range(min(20, len(df))):
        row_vals = df.iloc[i].astype(str).str.lower().tolist()
        if any("country" in v for v in row_vals):
            header_row = i
            break

    if header_row is None:
        raise ValueError("Cannot find header row with 'Country' in Damodaran XLSX")

    # Set headers and filter
    df.columns = df.iloc[header_row].astype(str).str.strip()
    df = df.iloc[header_row + 1:].reset_index(drop=True)

    # Standardize column names
    col_map = {}
    for col in df.columns:
        cl = str(col).lower().strip()
        if "country" in cl:
            col_map[col] = "country"
        elif "moody" in cl and "rating" in cl:
            col_map[col] = "moody_rating"
        elif "adj" in cl and "default" in cl and "spread" in cl:
            col_map[col] = "default_spread_bp"
        elif "default" in cl and "spread" in cl:
            col_map[col] = "default_spread_bp"
        elif "cds" in cl:
            col_map[col] = "cds_spread_bp"
        elif "equity" in cl and "risk" in cl and "premium" in cl:
            col_map[col] = "equity_risk_premium"
        elif "total" in cl and "equity" in cl and "risk" in cl:
            col_map[col] = "total_erp"

    df.rename(columns=col_map, inplace=True)
    return df


def _filter_eurozone(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre sur les 9 pays zone euro."""
    print("[3/5] Filtrage zone euro (9 pays)...")

    if "country" not in df.columns:
        raise ValueError("Column 'country' not found in parsed data")

    df["country"] = df["country"].astype(str).str.strip()
    mask = df["country"].isin(_EUROZONE_COUNTRIES)
    df_ez = df[mask].copy()

    if len(df_ez) == 0:
        raise ValueError(
            f"No eurozone countries found. Available: "
            f"{df['country'].unique()[:20]}"
        )

    print(f"  {len(df_ez)} pays trouves sur {len(_EUROZONE_COUNTRIES)} attendus")

    # Add country code
    df_ez["country_code"] = df_ez["country"].map(_COUNTRY_CODE_MAP)

    # Convert numeric columns
    for col in ["default_spread_bp", "cds_spread_bp", "equity_risk_premium"]:
        if col in df_ez.columns:
            df_ez[col] = pd.to_numeric(df_ez[col], errors="coerce")

    # CDS might be in percentage (e.g., 1.10 = 110bp), convert
    if "cds_spread_bp" in df_ez.columns:
        median_cds = df_ez["cds_spread_bp"].median()
        if median_cds < 10:  # likely in percentage points
            df_ez["cds_spread_bp"] = df_ez["cds_spread_bp"] * 100

    # Default spread might also be in percentage
    if "default_spread_bp" in df_ez.columns:
        median_ds = df_ez["default_spread_bp"].median()
        if median_ds < 10:  # likely in percentage points
            df_ez["default_spread_bp"] = df_ez["default_spread_bp"] * 100

    return df_ez


def _add_fiscal_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les donnees fiscales Eurostat."""
    print("[4/5] Ajout donnees fiscales Eurostat...")

    for col, default in [("debt_to_gdp", 0.80), ("fiscal_balance", -0.03),
                          ("recovery_rate_hist", 0.80)]:
        df[col] = df["country_code"].map(
            lambda cc, c=col, d=default: _FISCAL_DATA.get(cc, {}).get(c, d)
        )

    return df


def _save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Sauvegarde en parquet."""
    print(f"[5/5] Sauvegarde vers {output_path}...")

    final_cols = [
        "country", "country_code", "moody_rating",
        "default_spread_bp", "cds_spread_bp", "equity_risk_premium",
        "debt_to_gdp", "fiscal_balance", "recovery_rate_hist",
    ]
    available = [c for c in final_cols if c in df.columns]
    df_out = df[available].copy()

    df_out.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    size_kb = output_path.stat().st_size / 1e3
    print(f"  OK : {len(df_out)} lignes, {len(df_out.columns)} colonnes, {size_kb:.0f} KB")

    print("\n--- Donnees souveraines ---")
    for _, row in df_out.iterrows():
        cds = row.get("cds_spread_bp", "N/A")
        ds = row.get("default_spread_bp", "N/A")
        print(f"  {row['country']:15s} ({row.get('moody_rating', '?'):4s}) "
              f"CDS={cds:>6} bp  DefaultSpread={ds:>6} bp  "
              f"Debt/GDP={row.get('debt_to_gdp', 0):.0%}")


# -----------------------------------------------
# MAIN
# -----------------------------------------------

def main() -> None:
    """Pipeline complet : download -> parse -> filter -> enrich -> save."""
    print("=" * 60)
    print("Damodaran Sovereign Risk -> damodaran_sovereign.parquet")
    print("=" * 60)

    print("[1/5] Telechargement XLSX Damodaran...")
    data = _download_xlsx(_XLSX_URL)
    df = _parse_erp_sheet(data)
    df = _filter_eurozone(df)
    df = _add_fiscal_data(df)
    _save_parquet(df, _OUTPUT_PATH)

    print("\nDone!")


if __name__ == "__main__":
    main()
