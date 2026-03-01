"""Telechargement optionnel de donnees BIS OTC Derivatives Statistics.

Source : BIS Data Portal (CSV).
Sortie : data/bis_derivatives.parquet (optionnel)

Usage :
    python -m ifrs9_cockpit.data.fetch_bis_derivatives

Le generateur derivatives_cva_positions.py fonctionne SANS ce parquet
(constantes BIS hardcodees). Ce script est optionnel pour calibration avancee.

Constantes codees (BIS OTC Derivatives Statistics, end-2024) :
    - Notionnel total : ~600T USD
    - Mix : IRS 75%, FX 15%, CDS 5%, Equity 3%, Commodity 2%
    - Gross Market Value : ~18T USD (~3% du notionnel)
    - Gross Credit Exposure (post-netting) : ~2.7T USD
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_OUT_DIR = Path(__file__).parent
_OUT_PATH = _OUT_DIR / "bis_derivatives.parquet"


def _generate_bis_calibration(seed: int = 42) -> pd.DataFrame:
    """Genere des statistiques BIS synthetiques pour calibration."""
    rng = np.random.RandomState(seed)

    desks = ["IRS", "FX", "CDS", "Equity_Deriv", "Commodity"]
    notional_mix = [0.75, 0.15, 0.05, 0.03, 0.02]
    gmv_ratio = [0.025, 0.030, 0.050, 0.060, 0.040]
    netting_reduction = [0.85, 0.80, 0.75, 0.70, 0.65]

    total_notional_usd = 600e12  # 600T USD

    records = []
    for desk, mix, gmv_r, net_r in zip(desks, notional_mix, gmv_ratio, netting_reduction):
        notional = total_notional_usd * mix
        gmv = notional * gmv_r
        gce = gmv * (1 - net_r)
        records.append({
            "desk": desk,
            "notional_usd": notional,
            "gross_market_value_usd": gmv,
            "netting_reduction": net_r,
            "gross_credit_exposure_usd": gce,
            "notional_share": mix,
        })

    return pd.DataFrame(records)


def main():
    """Point d'entree principal."""
    print("BIS Derivatives: generation calibration synthetique...")
    df = _generate_bis_calibration()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT_PATH, index=False)
    print(f"Sauvegarde: {_OUT_PATH} ({len(df)} desks)")
    for _, row in df.iterrows():
        print(f"  {row['desk']}: notional={row['notional_usd']/1e12:.0f}T, "
              f"GCE={row['gross_credit_exposure_usd']/1e12:.1f}T")


if __name__ == "__main__":
    main()
