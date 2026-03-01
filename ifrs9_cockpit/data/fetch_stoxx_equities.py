"""Telechargement de donnees equities EURO STOXX 50 via yfinance.

Source : yfinance API (gratuit, ~50 tickers EURO STOXX 50).
Sortie : data/stoxx_equities.parquet

Usage :
    python -m ifrs9_cockpit.data.fetch_stoxx_equities

Le parquet resultant est charge par
``ifrs9_cockpit.synthetic_generator.equity_positions.load_equity_data()``.
Si le parquet est absent, le generateur utilise un fallback EU_LARGE_CAPS.

Note : yfinance peut etre lent ou indisponible. En cas d'echec, le generateur
fonctionne sans parquet via les constantes hardcodees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_OUT_DIR = Path(__file__).parent
_OUT_PATH = _OUT_DIR / "stoxx_equities.parquet"

# EURO STOXX 50 tickers (subset — les plus liquides)
_TICKERS = [
    "SAP.DE", "ASML.AS", "MC.PA", "SIE.DE", "TTE.PA",
    "AIR.PA", "SAN.PA", "BNP.PA", "DTE.DE", "ALV.DE",
    "OR.PA", "AI.PA", "IBE.MC", "INGA.AS", "BAS.DE",
    "ABI.BR", "ENEL.MI", "ISP.MI", "MUV2.DE", "PHIA.AS",
    "DG.PA", "ADS.DE", "KER.PA", "NOKIA.HE", "BBVA.MC",
    "ENI.MI", "BMW.DE", "VOW3.DE", "SU.PA", "BN.PA",
]

# GICS sector mapping
_TICKER_SECTOR = {
    "SAP.DE": "Technology", "ASML.AS": "Technology",
    "MC.PA": "Consumer", "SIE.DE": "Industrials",
    "TTE.PA": "Energy", "AIR.PA": "Industrials",
    "SAN.PA": "Healthcare", "BNP.PA": "Financials",
    "DTE.DE": "Telecom", "ALV.DE": "Financials",
    "OR.PA": "Consumer", "AI.PA": "Industrials",
    "IBE.MC": "Utilities", "INGA.AS": "Financials",
    "BAS.DE": "Industrials", "ABI.BR": "Consumer",
    "ENEL.MI": "Utilities", "ISP.MI": "Financials",
    "MUV2.DE": "Financials", "PHIA.AS": "Healthcare",
    "DG.PA": "Consumer", "ADS.DE": "Consumer",
    "KER.PA": "Consumer", "NOKIA.HE": "Technology",
    "BBVA.MC": "Financials", "ENI.MI": "Energy",
    "BMW.DE": "Consumer", "VOW3.DE": "Consumer",
    "SU.PA": "Industrials", "BN.PA": "Consumer",
}


def _fetch_yfinance() -> pd.DataFrame:
    """Telecharge via yfinance (necessite le package installe)."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance non installe. Fallback synthetique.")
        return _generate_synthetic_equities()

    records = []
    for ticker in _TICKERS:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or "marketCap" not in info:
                continue

            hist = stock.history(period="1y")
            if len(hist) < 50:
                continue

            returns = hist["Close"].pct_change().dropna()
            vol_252 = float(returns.std() * np.sqrt(252))

            records.append({
                "ticker": ticker,
                "name": info.get("shortName", ticker),
                "sector_gics": _TICKER_SECTOR.get(ticker, "Other"),
                "market_cap": info.get("marketCap", 0),
                "total_debt": info.get("totalDebt", 0) or 0,
                "beta": info.get("beta", 1.0) or 1.0,
                "volatility_252d": round(vol_252, 4),
                "dividend_yield": info.get("dividendYield", 0) or 0,
                "pe_ratio": info.get("trailingPE", 0) or 0,
            })
        except Exception as e:
            print(f"  Skip {ticker}: {e}")

    if not records:
        print("Aucune donnee yfinance. Fallback synthetique.")
        return _generate_synthetic_equities()

    return pd.DataFrame(records)


def _generate_synthetic_equities(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Genere des equities synthetiques calibrees si yfinance indisponible."""
    rng = np.random.RandomState(seed)

    sectors = ["Technology", "Financials", "Industrials", "Utilities",
               "Healthcare", "Consumer", "Energy", "Telecom"]
    sector_beta = {
        "Technology": 1.3, "Financials": 1.2, "Industrials": 1.1,
        "Utilities": 0.6, "Healthcare": 0.8, "Consumer": 1.0,
        "Energy": 1.1, "Telecom": 0.7,
    }

    sector_arr = rng.choice(sectors, n)
    data = {
        "ticker": [f"EU_{i:03d}" for i in range(n)],
        "name": [f"EuroStock_{i:03d}" for i in range(n)],
        "sector_gics": sector_arr,
        "market_cap": np.round(rng.lognormal(24.0, 1.0, n), 0),  # ~50B median
        "total_debt": np.round(rng.lognormal(22.5, 1.2, n), 0),  # ~10B median
        "beta": np.round(np.array([sector_beta[s] for s in sector_arr])
                         + rng.normal(0, 0.15, n), 2),
        "volatility_252d": np.round(rng.uniform(0.15, 0.35, n), 4),
        "dividend_yield": np.round(rng.uniform(0.01, 0.05, n), 4),
        "pe_ratio": np.round(rng.uniform(8, 30, n), 1),
    }

    return pd.DataFrame(data)


def main():
    """Point d'entree principal."""
    print("STOXX Equities: tentative yfinance...")
    df = _fetch_yfinance()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT_PATH, index=False)
    print(f"Sauvegarde: {_OUT_PATH} ({len(df)} tickers, {_OUT_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"  Sectors: {df['sector_gics'].value_counts().to_dict()}")
    if "beta" in df.columns:
        print(f"  Beta mean: {df['beta'].mean():.2f}")
        print(f"  Vol mean: {df['volatility_252d'].mean():.2%}")


if __name__ == "__main__":
    main()
