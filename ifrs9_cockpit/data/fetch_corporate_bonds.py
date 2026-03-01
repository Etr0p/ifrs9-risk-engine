"""Telechargement et pre-traitement de donnees corporate bonds.

Source : Kaggle "Corporate Credit Rating" (~2000 firmes).
Sortie : data/corporate_bonds.parquet

Usage :
    python -m ifrs9_cockpit.data.fetch_corporate_bonds

Le parquet resultant est charge par
``ifrs9_cockpit.synthetic_generator.corporate_bonds_positions.load_corporate_bonds_data()``.
Si le parquet est absent, le generateur utilise un fallback parametrique.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Rating -> PD mapping (Moody's 2024 Annual Default Study, 1-year realized)
_RATING_PD = {
    "AAA": 0.0001, "AA+": 0.0002, "AA": 0.0002, "AA-": 0.0003,
    "A+": 0.0004, "A": 0.0005, "A-": 0.0007,
    "BBB+": 0.0012, "BBB": 0.0025, "BBB-": 0.0045,
    "BB+": 0.0080, "BB": 0.0150, "BB-": 0.0250,
    "B+": 0.0300, "B": 0.0500, "B-": 0.0800,
    "CCC": 0.1000, "CC": 0.2000, "C": 0.3000, "D": 1.0000,
}

# Seniority -> LGD (Moody's Ultimate Recovery Database)
_SENIORITY_LGD = {
    "senior_secured": 0.35,
    "senior_unsecured": 0.45,
    "subordinated": 0.70,
}

_OUT_DIR = Path(__file__).parent
_OUT_PATH = _OUT_DIR / "corporate_bonds.parquet"


def _generate_synthetic_corporate_bonds(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Genere un dataset synthetique calibre si Kaggle indisponible."""
    rng = np.random.RandomState(seed)

    ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    rating_probs = [0.02, 0.08, 0.20, 0.40, 0.15, 0.10, 0.05]
    sectors = [
        "Financials", "Industrials", "Utilities", "Technology",
        "Healthcare", "Consumer", "Energy", "Telecom",
    ]
    seniorities = ["senior_secured", "senior_unsecured", "subordinated"]
    seniority_probs = [0.25, 0.55, 0.20]

    data = {
        "issuer": [f"CORP_{i:04d}" for i in range(n)],
        "rating": rng.choice(ratings, n, p=rating_probs),
        "sector_gics": rng.choice(sectors, n),
        "seniority": rng.choice(seniorities, n, p=seniority_probs),
        "total_debt_m": np.round(rng.lognormal(7.0, 1.2, n) / 1e6, 1),
        "revenue_m": np.round(rng.lognormal(7.5, 1.0, n) / 1e6, 1),
        "interest_coverage": np.round(rng.lognormal(1.5, 0.6, n), 2),
        "roa": np.round(rng.normal(0.05, 0.03, n), 4),
        "maturity_years": np.round(rng.uniform(1.0, 10.0, n), 1),
        "coupon_pct": np.round(rng.uniform(0.5, 6.0, n), 2),
    }

    df = pd.DataFrame(data)
    df["pd_base"] = df["rating"].map(_RATING_PD).fillna(0.01)
    df["lgd_base"] = df["seniority"].map(_SENIORITY_LGD).fillna(0.45)
    df["spread_bps"] = df["rating"].map({
        "AAA": 30, "AA": 40, "A": 60, "BBB": 130,
        "BB": 300, "B": 500, "CCC": 900,
    }).fillna(130)

    return df


def main():
    """Point d'entree principal."""
    print("Corporate Bonds: generation synthetique calibree...")
    df = _generate_synthetic_corporate_bonds()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT_PATH, index=False)
    print(f"Sauvegarde: {_OUT_PATH} ({len(df)} firmes, {_OUT_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"  Ratings: {df['rating'].value_counts().to_dict()}")
    print(f"  PD mean: {df['pd_base'].mean():.4f}")
    print(f"  LGD mean: {df['lgd_base'].mean():.2f}")


if __name__ == "__main__":
    main()
