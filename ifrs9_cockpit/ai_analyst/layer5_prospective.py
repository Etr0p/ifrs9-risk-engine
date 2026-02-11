"""Couche 5 — Prospective : trajectoires O-U, risk appetite, early warning (FR30).

Projette 3 trajectoires (favorable/centrale/adverse) via processus
d'Ornstein-Uhlenbeck (mean-reversion) a T+3/6/9/12 mois,
applique le cadre d'appetit au risque (feux tricolores) et calcule
les indicateurs d'alerte precoce composites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from ifrs9_cockpit.config import (
    SECTORS,
    SCENARIO_BASE,
    RISK_APPETITE_CONFIG,
    MACRO_MEAN_REVERSION,
)
from ifrs9_cockpit.ai_analyst.types import RegimeClassification


_HORIZONS = [3, 6, 9, 12]  # Mois

# Facteurs epsilon par scenario pour le processus O-U (deterministes).
# Favorable = choc negatif (amelioration), Central = neutre, Adverse = deterioration.
_SCENARIO_EPSILONS = {
    "Favorable": -1.5,
    "Central": 0.0,
    "Adverse": 1.5,
}

# Variables macro projetees
_MACRO_VARS = [
    "unemployment_rate",
    "gdp_growth",
    "interest_rate",
    "hpi_growth",
    "inflation_rate",
]


def project_trajectories(
    macro_params: Dict[str, float],
    regime: Optional[RegimeClassification] = None,
    n_months: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Projette les trajectoires macro via Ornstein-Uhlenbeck (mean-reversion) (FR30).

    Processus O-U discret :
        x(t+1) = x(t) + kappa * (theta - x(t)) * dt + sigma * sqrt(dt) * epsilon

    3 scenarios utilisent epsilon = {-1.5, 0, +1.5} (choc deterministe).
    Les parametres kappa, theta, sigma sont importes de MACRO_MEAN_REVERSION.

    Retourne les valeurs aux horizons T+3, T+6, T+9, T+12 pour compatibilite
    avec le format existant (3 trajectoires x 4 horizons = 12 lignes).

    Args:
        macro_params: Variables macro actuelles.
        regime: Classification de regime (ajuste les projections via drift additionnel).
        n_months: Horizon maximal de projection (defaut 12).
        seed: Graine aleatoire (reservee pour future extension stochastique).

    Returns:
        DataFrame avec colonnes : trajectory, horizon_months, + 5 variables macro.
    """
    # Note : seed est accepte pour compatibilite ascendante mais non utilise car
    # les 3 trajectoires utilisent des epsilon deterministes {-1.5, 0, +1.5}.
    _ = seed  # Suppress unused warning
    dt = 1.0 / 12  # Pas mensuel (fraction d'annee)
    records = []

    for traj_name, epsilon_sign in _SCENARIO_EPSILONS.items():
        # Simuler la trajectoire complete mois par mois pour chaque variable
        # puis extraire les horizons cibles.
        trajectories_by_var: Dict[str, List[float]] = {}

        for var in _MACRO_VARS:
            mr_params = MACRO_MEAN_REVERSION[var]
            kappa = mr_params["kappa"]
            theta = mr_params["theta"]
            sigma = mr_params["sigma"]
            x = macro_params.get(var, theta)

            monthly_values = []
            for month in range(1, n_months + 1):
                # O-U discret : x(t+1) = x(t) + kappa*(theta-x(t))*dt + sigma*sqrt(dt)*epsilon
                x = x + kappa * (theta - x) * dt + sigma * np.sqrt(dt) * epsilon_sign
                monthly_values.append(x)

            trajectories_by_var[var] = monthly_values

        # Extraire les 4 horizons cibles
        for horizon in _HORIZONS:
            if horizon > n_months:
                continue
            row = {"trajectory": traj_name, "horizon_months": horizon}

            for var in _MACRO_VARS:
                projected = trajectories_by_var[var][horizon - 1]  # 0-indexed

                # Ajustement regime (passe 2) : drift additionnel
                if regime is not None:
                    time_scale = horizon / 12.0
                    regime_adj = _regime_drift(var, regime.detected_regime, time_scale)
                    projected += regime_adj

                row[var] = round(projected, 2)

            records.append(row)

    return pd.DataFrame(records)


def compute_risk_appetite(
    result_credit: pd.DataFrame,
    result_pe: pd.DataFrame,
) -> pd.DataFrame:
    """Calcule les feux tricolores du risk appetite par cellule (FR30).

    10 cellules (5 secteurs x 2 canaux) x 3 metriques (ECL/EAD, RAROC proxy, HHI).

    Args:
        result_credit: Resultat ECLCalculator.
        result_pe: Resultat PECalculator.

    Returns:
        DataFrame avec colonnes sector, canal, metric, value, signal.
    """
    cfg = RISK_APPETITE_CONFIG
    records = []

    for sector in SECTORS:
        name = sector.name

        # -- Credit --
        mask_c = result_credit["sector"].values == name
        df_c = result_credit.loc[mask_c]

        if len(df_c) > 0:
            ead = df_c["ead"].sum()
            ecl = df_c["ecl_weighted"].sum()
            ecl_ead = ecl / max(ead, 1)

            signal_ecl = (
                "vert" if ecl_ead < cfg.ecl_ead_green
                else "ambre" if ecl_ead < cfg.ecl_ead_amber
                else "rouge"
            )

            records.append({
                "sector": name, "canal": "Credit",
                "metric": "ECL/EAD", "value": round(ecl_ead, 4),
                "signal": signal_ecl,
            })

        # -- PE --
        mask_p = result_pe["sector"].values == name
        df_p = result_pe.loc[mask_p]

        if len(df_p) > 0:
            drawdown = df_p["nav_drawdown"].mean()

            signal_dd = (
                "vert" if drawdown < cfg.nav_drawdown_green
                else "ambre" if drawdown < cfg.nav_drawdown_amber
                else "rouge"
            )

            records.append({
                "sector": name, "canal": "PE",
                "metric": "NAV_Drawdown", "value": round(drawdown, 4),
                "signal": signal_dd,
            })

    return pd.DataFrame(records)


def compute_early_warning(
    result_credit: pd.DataFrame,
    result_pe: pd.DataFrame,
    macro_params: Dict[str, float],
) -> pd.DataFrame:
    """Calcule les indicateurs d'alerte precoce composites (FR30).

    Combine les signaux de deterioration credit et PE en un score
    d'alerte par secteur.

    Args:
        result_credit: Resultat ECLCalculator.
        result_pe: Resultat PECalculator.
        macro_params: Variables macro actuelles.

    Returns:
        DataFrame avec colonnes sector, ew_score, ew_signal, components.
    """
    base = SCENARIO_BASE
    records = []

    # Stress macro global
    macro_stress = sum(
        abs(macro_params.get(v, getattr(base, v)) - getattr(base, v))
        for v in ["unemployment_rate", "gdp_growth", "interest_rate",
                   "hpi_growth", "inflation_rate"]
    )

    for sector in SECTORS:
        name = sector.name

        # Credit deterioration
        mask_c = result_credit["sector"].values == name
        df_c = result_credit.loc[mask_c]
        pd_mean = df_c["pd_12m"].mean() if len(df_c) > 0 else 0
        s2_pct = (df_c["stage"] == 2).mean() if len(df_c) > 0 else 0
        s3_pct = (df_c["stage"] == 3).mean() if len(df_c) > 0 else 0

        # PE deterioration
        mask_p = result_pe["sector"].values == name
        df_p = result_pe.loc[mask_p]
        drawdown = df_p["nav_drawdown"].mean() if len(df_p) > 0 else 0
        distress_pct = (df_p["risk_category"] == "Distressed").mean() if len(df_p) > 0 else 0

        # M4. Poids calibres par jugement d'expert (approche AHP simplifiee) :
        # - PD_mean (0.25) : indicateur le plus predictif du defaut
        # - Stage2_pct (0.15) : signal avance IFRS 9
        # - Stage3_pct (0.15) : indicateur retarde de pertes realisees
        # - PE_drawdown (0.20) : volatilite du portefeuille PE
        # - Distress_pct (0.10) : complementaire du drawdown
        # - Macro_stress (0.15) : forward-looking
        # Somme = 1.0 (contrainte normalisante)
        ew_score = (
            0.25 * min(pd_mean / 0.20, 1)       # PD normalise a 20%
            + 0.15 * min(s2_pct / 0.20, 1)       # Stage 2 normalise a 20%
            + 0.15 * min(s3_pct / 0.10, 1)       # Stage 3 normalise a 10%
            + 0.20 * min(drawdown / 0.30, 1)      # Drawdown normalise a 30%
            + 0.10 * min(distress_pct / 0.30, 1)  # Distress normalise a 30%
            + 0.15 * min(macro_stress / 10.0, 1)   # Macro normalise a 10pp
        )

        signal = (
            "vert" if ew_score < 0.25
            else "ambre" if ew_score < 0.50
            else "rouge"
        )

        records.append({
            "sector": name,
            "ew_score": round(ew_score, 4),
            "ew_signal": signal,
            "pd_mean": round(pd_mean, 4),
            "drawdown": round(drawdown, 4),
            "macro_stress": round(macro_stress, 2),
        })

    return pd.DataFrame(records)


def _regime_drift(var: str, regime: str, time_scale: float) -> float:
    """Derive additionnelle par regime (passe 2)."""
    drifts = {
        "Crise financiere": {"unemployment_rate": 0.5, "hpi_growth": -1.0, "gdp_growth": -0.5},
        "Stagflation": {"inflation_rate": 0.5, "interest_rate": 0.3, "gdp_growth": -0.3},
        "Rupture techno": {"unemployment_rate": 0.3, "gdp_growth": -0.2},
        "Resserrement": {"interest_rate": 0.5, "hpi_growth": -0.5},
        "Reprise": {"gdp_growth": 0.3, "unemployment_rate": -0.3},
    }
    return drifts.get(regime, {}).get(var, 0.0) * time_scale


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset
    from ifrs9_cockpit.models.pd_model import PDModelSuite
    from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
    from ifrs9_cockpit.engine.pe_calculator import PECalculator
    from ifrs9_cockpit.models.lgd_model import LGDModel
    from ifrs9_cockpit.models.ead_model import EADModel

    print("=" * 70)
    print("IFRS 9 COCKPIT — AI Analyst : Couches 4-5 (Regime & Prospective O-U)")
    print("=" * 70)

    # Pipeline
    print("\n[1/4] Pipelines credit/PE...")
    df_credit, df_pe, _ = generate_dataset()
    pd_suite = PDModelSuite()
    pd_suite.fit(df_credit)
    pd_current = pd_suite.predict_active(df_credit)
    ecl_calc = ECLCalculator(lgd_model=LGDModel(), ead_model=EADModel())
    result_credit = ecl_calc.calculate(df_credit, pd_current, pd_current * 0.8)
    result_pe = PECalculator().calculate(df_pe)

    macro_params = {
        "unemployment_rate": SCENARIO_BASE.unemployment_rate,
        "gdp_growth": SCENARIO_BASE.gdp_growth,
        "interest_rate": SCENARIO_BASE.interest_rate,
        "hpi_growth": SCENARIO_BASE.hpi_growth,
        "inflation_rate": SCENARIO_BASE.inflation_rate,
    }

    # Couche 4
    from ifrs9_cockpit.ai_analyst.layer4_regime import classify_regime
    print("\n[2/4] Couche 4 — Regime...")
    regime = classify_regime(macro_params)
    print(f"  Regime detecte : {regime.detected_regime}")
    for name, prob in regime.probabilities.items():
        print(f"    {name:20s} : {prob:.4f}")

    # Couche 5 — Trajectoires (Ornstein-Uhlenbeck)
    print("\n[3/4] Couche 5 — Trajectoires O-U T+3/6/9/12...")
    traj = project_trajectories(macro_params, regime)
    print(traj.to_string(index=False))

    # Risk appetite
    print("\n[4/4] Risk appetite (feux tricolores)...")
    ra = compute_risk_appetite(result_credit, result_pe)
    print(ra.to_string(index=False))

    # Early warning
    print("\n--- Early Warning ---")
    ew = compute_early_warning(result_credit, result_pe, macro_params)
    print(ew.to_string(index=False))

    # Validations
    print("\n--- Validations ---")
    all_ok = True

    ok = len(regime.probabilities) == 5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] 5 regimes ({len(regime.probabilities)})")
    all_ok &= ok

    ok = len(traj) == 12  # 3 trajectoires x 4 horizons
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] 12 projections (3x4) ({len(traj)})")
    all_ok &= ok

    ok = len(ra) == 10  # 5 secteurs x 2 canaux
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Risk appetite : 10 cellules ({len(ra)})")
    all_ok &= ok

    ok = len(ew) == 5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Early warning : 5 secteurs ({len(ew)})")
    all_ok &= ok

    ok = all(0 <= v <= 1 for v in ew["ew_score"])
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] EW scores dans [0,1] "
          f"(min={ew['ew_score'].min():.4f}, max={ew['ew_score'].max():.4f})")
    all_ok &= ok

    print(f"\n{'=' * 70}")
    if all_ok:
        print("Couches 4-5 (Regime & Prospective O-U) validees.")
    else:
        print("ATTENTION : certaines validations ont echoue.")
    print(f"{'=' * 70}")
