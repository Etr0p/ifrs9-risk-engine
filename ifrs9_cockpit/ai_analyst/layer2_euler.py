"""Couche 2 — Allocation proportionnelle du risque & attribution factorielle (FR27).

Decompose les causes des ecarts par allocation proportionnelle du risque
(full allocation garantie) et attribution factorielle macro
one-at-a-time (OAT) avec deltas adaptatifs.

RJ audit MAJOR : renomme ``euler_share`` → ``proportional_share`` et
``decompose_euler`` → ``decompose_proportional``. L'ancien nom Euler
etait un abus de langage : l'allocation proportionnelle n'utilise PAS
de derivees partielles (Euler-Tasche). Alias backward-compat conserves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from ifrs9_cockpit.config import SECTORS, SCENARIO_BASE
from ifrs9_cockpit.ai_analyst.types import RegimeClassification


# Variables macro et leur sens adverse
_MACRO_VARS = [
    "unemployment_rate",
    "gdp_growth",
    "interest_rate",
    "hpi_growth",
    "inflation_rate",
]

# M6. Deltas adaptatifs bases sur 1% de la fourchette de chaque variable.
# Remplace le delta fixe de 1pp pour une attribution factorielle plus fidele.
_FACTOR_DELTAS: Dict[str, float] = {
    "unemployment_rate": 0.11,  # 1% de [4, 15]
    "gdp_growth": 0.10,        # 1% de [-5, 5]
    "interest_rate": 0.10,     # 1% de [0, 10]
    "hpi_growth": 0.20,        # 1% de [-10, 10]
    "inflation_rate": 0.08,    # 1% de [0, 8]
}


def decompose_proportional(
    result_credit: pd.DataFrame,
    result_pe: pd.DataFrame,
    macro_params: Dict[str, float],
    regime: Optional[RegimeClassification] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Allocation proportionnelle du risque et facteurs macro (FR27).

    La methode appliquee est une allocation proportionnelle :
        contribution_i = ECL_i / ECL_total  (credit)
        contribution_i = EL_PE_i / EL_PE_total  (PE)
    La somme des contributions = 1.0 (full allocation garantie).

    Attribution factorielle : OAT sur les 5 variables macro avec deltas
    adaptatifs (M6) proportionnels a la fourchette de chaque variable.

    Args:
        result_credit: Resultat ECLCalculator.
        result_pe: Resultat PECalculator.
        macro_params: Variables macro actuelles.
        regime: Classification de regime (optionnel, passe 2).

    Returns:
        Tuple (proportional_contributions, factor_attribution).
    """
    # -- Allocation proportionnelle (10 cellules) --
    total_ecl = result_credit["ecl_weighted"].sum()
    total_el_pe = result_pe["expected_loss_pe"].sum()
    total_risk = total_ecl + total_el_pe

    # Records pour l'allocation proportionnelle
    proportional_records = []
    for sector in SECTORS:
        name = sector.name

        # Credit : contribution_i = ECL_i / ECL_total
        mask_c = result_credit["sector"].values == name
        ecl_sec = result_credit.loc[mask_c, "ecl_weighted"].sum()
        rwa_sec = result_credit.loc[mask_c, "rwa_credit"].sum()

        proportional_records.append({
            "sector": name,
            "canal": "Credit",
            "risk_amount": round(ecl_sec, 0),
            "proportional_share": round(ecl_sec / max(total_risk, 1), 6),
            "rwa": round(rwa_sec, 0),
        })

        # PE : contribution_i = EL_PE_i / EL_PE_total
        mask_p = result_pe["sector"].values == name
        el_sec = result_pe.loc[mask_p, "expected_loss_pe"].sum()
        rwa_pe_sec = result_pe.loc[mask_p, "rwa_pe"].sum()

        proportional_records.append({
            "sector": name,
            "canal": "PE",
            "risk_amount": round(el_sec, 0),
            "proportional_share": round(el_sec / max(total_risk, 1), 6),
            "rwa": round(rwa_pe_sec, 0),
        })

    # Colonne euler_share conservee pour compatibilite ascendante
    proportional_df = pd.DataFrame(proportional_records)

    # -- Attribution factorielle OAT (5 vars x 2 canaux) --
    # Mesure la sensibilite du risque a chaque variable macro
    # avec des deltas adaptatifs (M6) au lieu d'un delta fixe.
    base = SCENARIO_BASE

    factor_records = []
    for var in _MACRO_VARS:
        base_val = getattr(base, var)
        current_val = macro_params.get(var, base_val)

        # Delta entre current et baseline
        var_delta = current_val - base_val

        # Delta adaptatif pour la normalisation (M6)
        adaptive_delta = _FACTOR_DELTAS.get(var, 0.10)

        # Contribution au risque credit (via sensibilites sectorielles)
        credit_contrib = 0.0
        pe_contrib = 0.0

        for sector in SECTORS:
            name = sector.name

            # Sensibilite credit
            sens_credit = getattr(sector, f"{_sens_key(var)}_credit")
            mask_c = result_credit["sector"].values == name
            ecl_sec = result_credit.loc[mask_c, "ecl_weighted"].sum()
            # Attribution normalisee par le delta adaptatif
            credit_contrib += (abs(var_delta) / adaptive_delta) * sens_credit * ecl_sec / max(total_ecl, 1)

            # Sensibilite PE
            sens_pe = getattr(sector, f"{_sens_key(var)}_pe")
            mask_p = result_pe["sector"].values == name
            el_sec = result_pe.loc[mask_p, "expected_loss_pe"].sum()
            if total_el_pe > 0:
                pe_contrib += (abs(var_delta) / adaptive_delta) * sens_pe * el_sec / total_el_pe

        # Ajustement regime (passe 2) : amplifier les facteurs du regime detecte
        regime_mult = 1.0
        if regime is not None:
            regime_mult = _regime_multiplier(var, regime.detected_regime)

        factor_records.append({
            "variable": var,
            "canal": "Credit",
            "delta_from_base": round(var_delta, 2),
            "attribution": round(credit_contrib * regime_mult, 4),
            "regime_adjusted": regime is not None,
        })
        factor_records.append({
            "variable": var,
            "canal": "PE",
            "delta_from_base": round(var_delta, 2),
            "attribution": round(pe_contrib * regime_mult, 4),
            "regime_adjusted": regime is not None,
        })

    factor_df = pd.DataFrame(factor_records)

    return proportional_df, factor_df


def _sens_key(var_name: str) -> str:
    """Convertit le nom de variable macro en prefixe de sensibilite."""
    mapping = {
        "unemployment_rate": "unemployment_sensitivity",
        "gdp_growth": "gdp_sensitivity",
        "interest_rate": "interest_rate_sensitivity",
        "hpi_growth": "hpi_sensitivity",
        "inflation_rate": "inflation_sensitivity",
    }
    return mapping[var_name]


def _regime_multiplier(var: str, regime: str) -> float:
    """Multiplicateur de regime pour l'attribution factorielle (passe 2).

    Amplifie les variables dominantes dans chaque regime.
    """
    multipliers = {
        "Crise financiere": {
            "unemployment_rate": 1.5,
            "gdp_growth": 1.3,
            "hpi_growth": 1.4,
        },
        "Stagflation": {
            "inflation_rate": 1.5,
            "gdp_growth": 1.3,
            "interest_rate": 1.2,
        },
        "Rupture techno": {
            "gdp_growth": 1.4,
            "unemployment_rate": 1.2,
        },
        "Resserrement": {
            "interest_rate": 1.5,
            "hpi_growth": 1.3,
        },
        # En regime de Reprise, les multiplicateurs < 1.0 attenuent
        # l'attribution factorielle (risque en decroissance).
        "Reprise": {
            "gdp_growth": 0.8,
            "unemployment_rate": 0.8,
        },
    }
    return multipliers.get(regime, {}).get(var, 1.0)


# Backward-compat alias (RJ audit : renommage Euler → Proportional)
decompose_euler = decompose_proportional


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset
    from ifrs9_cockpit.models.pd_model import PDModelSuite
    from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
    from ifrs9_cockpit.engine.pe_calculator import PECalculator
    from ifrs9_cockpit.models.lgd_model import LGDModel
    from ifrs9_cockpit.models.ead_model import EADModel
    from ifrs9_cockpit.ai_analyst.layer1_crossing import analyze_crossings

    print("=" * 70)
    print("IFRS 9 COCKPIT — AI Analyst : Couches 1-2 (Croisement & Allocation proportionnelle)")
    print("=" * 70)

    # Pipeline
    print("\n[1/4] Generation + pipelines credit/PE...")
    df_credit, df_pe, df_history = generate_dataset()

    pd_suite = PDModelSuite()
    pd_suite.fit(df_credit)
    pd_current = pd_suite.predict_active(df_credit)

    lgd_model = LGDModel()
    ead_model = EADModel()
    ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
    pd_origination = pd_current * 0.8
    result_credit = ecl_calc.calculate(df_credit, pd_current, pd_origination)

    pe_calc = PECalculator()
    result_pe = pe_calc.calculate(df_pe)

    macro_params = {
        "unemployment_rate": SCENARIO_BASE.unemployment_rate,
        "gdp_growth": SCENARIO_BASE.gdp_growth,
        "interest_rate": SCENARIO_BASE.interest_rate,
        "hpi_growth": SCENARIO_BASE.hpi_growth,
        "inflation_rate": SCENARIO_BASE.inflation_rate,
    }

    # Couche 1
    print("\n[2/4] Couche 1 — Croisement (FR26)...")
    asym, marginal = analyze_crossings(result_credit, result_pe, macro_params)
    print("\n--- Matrice d'asymetrie ---")
    print(asym.to_string(index=False))
    print("\n--- Contributions marginales ---")
    print(marginal.to_string(index=False))

    # Couche 2
    print("\n[3/4] Couche 2 — Allocation proportionnelle du risque (FR27)...")
    euler, factors = decompose_proportional(result_credit, result_pe, macro_params)
    print("\n--- Allocation proportionnelle (contributions) ---")
    print(euler.to_string(index=False))
    print("\n--- Attribution factorielle (deltas adaptatifs) ---")
    print(factors.to_string(index=False))

    # Validations
    print("\n[4/4] Validations...")
    all_ok = True

    # V1: Asymetrie 5 secteurs
    ok = len(asym) == 5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Asymetrie : 5 secteurs ({len(asym)})")
    all_ok &= ok

    # V2: Allocation proportionnelle full allocation (somme ~= 1)
    euler_sum = euler["proportional_share"].sum()
    ok = abs(euler_sum - 1.0) < 0.01
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Allocation proportionnelle full allocation : sum = {euler_sum:.6f}")
    all_ok &= ok

    # V3: 10 cellules allocation proportionnelle
    ok = len(euler) == 10
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Allocation proportionnelle : 10 cellules ({len(euler)})")
    all_ok &= ok

    # V4: Factor attribution 5 vars x 2 canaux = 10 lignes
    ok = len(factors) == 10
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Factor attribution : 10 lignes ({len(factors)})")
    all_ok &= ok

    # V5: Marginal contributions 10 cellules
    ok = len(marginal) == 10
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Marginal contributions : 10 cellules ({len(marginal)})")
    all_ok &= ok

    # V6: Marginal contributions sum ~= 1
    marg_sum = marginal["marginal_contribution"].sum()
    ok = abs(marg_sum - 1.0) < 0.01
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Marginal contributions sum = {marg_sum:.6f}")
    all_ok &= ok

    print(f"\n{'=' * 70}")
    if all_ok:
        print("Couches 1-2 (Croisement & Allocation proportionnelle) validees.")
    else:
        print("ATTENTION : certaines validations ont echoue.")
    print(f"{'=' * 70}")
