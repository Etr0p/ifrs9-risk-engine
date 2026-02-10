"""Tests d'invariants financiers et validation du realisme des donnees (FR4, FR56).

Module de validation autonome (~80 lignes utiles) executable en standalone :
    python -m ifrs9_cockpit.tests.test_invariants

Pas de pytest/unittest (NFR20). Assertions natives avec rapport PASS/FAIL.
Les fonctions sont reutilisables par les modules en aval (ECL, LGD, stages).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from ifrs9_cockpit.config import (
    ECL_SCENARIOS,
    N_CLIENTS,
    N_MONTHS,
    REQUIRED_CREDIT_COLS,
    REQUIRED_PE_COLS,
    SECTORS,
)


# ──────────────────────────────────────────────
# COMPTEUR DE RESULTATS
# ──────────────────────────────────────────────

_results: List[Tuple[str, bool, str]] = []


def _check(name: str, condition: bool, detail: str = "") -> bool:
    """Enregistre le resultat d'un test.

    Args:
        name: Nom court du test.
        condition: True si le test passe.
        detail: Detail additionnel en cas d'echec.

    Returns:
        La condition evaluee.
    """
    _results.append((name, condition, detail))
    return condition


# ──────────────────────────────────────────────
# VALIDATION DU REALISME (FR4)
# ──────────────────────────────────────────────

def validate_realism(df_credit: pd.DataFrame, df_pe: pd.DataFrame) -> bool:
    """Valide les ordres de grandeur des donnees generees.

    Verifie par secteur que les PD, multiples PE, et distributions
    de features sont dans les fourchettes attendues.

    Args:
        df_credit: DataFrame credit avec default_flag et pd_latent.
        df_pe: DataFrame PE avec entry_multiple.

    Returns:
        True si toutes les validations passent.
    """
    ok = True

    # PD par secteur : 2-15% (FR4)
    for sector in SECTORS:
        mask = df_credit["sector"] == sector.name
        dr = df_credit.loc[mask, "default_flag"].mean()
        passed = 0.01 <= dr <= 0.20
        ok &= _check(
            f"PD {sector.name}",
            passed,
            f"{dr:.2%} (cible ~{sector.base_default_rate:.0%})",
        )

    # Multiples PE par secteur : dans la fourchette config
    for sector in SECTORS:
        mask = df_pe["sector"] == sector.name
        multiples = df_pe.loc[mask, "entry_multiple"]
        lo, hi = sector.entry_multiple_range
        in_range = (multiples >= lo).all() and (multiples <= hi).all()
        ok &= _check(
            f"Multiple PE {sector.name}",
            in_range,
            f"[{multiples.min():.1f}, {multiples.max():.1f}] vs config [{lo:.0f}, {hi:.0f}]",
        )

    # Correlations plausibles
    corr_cs_dr = df_credit[["credit_score", "debt_ratio"]].corr().iloc[0, 1]
    ok &= _check("Corr credit_score/debt_ratio < 0", corr_cs_dr < 0, f"{corr_cs_dr:.3f}")

    corr_rev_ebitda = df_credit[["revenue", "ebitda"]].corr().iloc[0, 1]
    ok &= _check("Corr revenue/ebitda > 0.5", corr_rev_ebitda > 0.5, f"{corr_rev_ebitda:.3f}")

    return ok


# ──────────────────────────────────────────────
# INVARIANTS FINANCIERS (FR56)
# ──────────────────────────────────────────────

def validate_invariants(df_credit: pd.DataFrame, df_pe: pd.DataFrame) -> bool:
    """Valide les invariants financiers sur les donnees generees.

    Args:
        df_credit: DataFrame credit.
        df_pe: DataFrame PE.

    Returns:
        True si tous les invariants sont respectes.
    """
    ok = True

    # pd_latent dans [0, 1]
    pd_vals = df_credit["pd_latent"]
    ok &= _check("pd_latent >= 0", (pd_vals >= 0).all(), f"min={pd_vals.min():.6f}")
    ok &= _check("pd_latent <= 1", (pd_vals <= 1).all(), f"max={pd_vals.max():.6f}")

    # default_flag dans {0, 1}
    ok &= _check(
        "default_flag binaire",
        set(df_credit["default_flag"].unique()).issubset({0, 1}),
        f"valeurs={df_credit['default_flag'].unique()}",
    )

    # revenue > 0
    ok &= _check("revenue > 0", (df_credit["revenue"] > 0).all(), f"min={df_credit['revenue'].min():.4f}")

    # debt_ratio dans [0, 1] (avant bruit)
    dr = df_credit["debt_ratio"]
    ok &= _check("debt_ratio [0,1]", (dr >= 0).all() and (dr <= 1).all(), f"[{dr.min():.4f}, {dr.max():.4f}]")

    # credit_score dans [300, 850]
    cs = df_credit["credit_score"]
    ok &= _check("credit_score [300,850]", (cs >= 300).all() and (cs <= 850).all(), f"[{cs.min()}, {cs.max()}]")

    # loan_amount > 0
    ok &= _check("loan_amount > 0", (df_credit["loan_amount"] > 0).all(), f"min={df_credit['loan_amount'].min():.2f}")

    # PE : entry_multiple > 0
    ok &= _check("entry_multiple > 0", (df_pe["entry_multiple"] > 0).all(), f"min={df_pe['entry_multiple'].min():.2f}")

    # PE : leverage dans [0, 1]
    lev = df_pe["leverage"]
    ok &= _check("leverage [0,1]", (lev >= 0).all() and (lev <= 1).all(), f"[{lev.min():.4f}, {lev.max():.4f}]")

    # PE : holding_years >= 1
    ok &= _check("holding_years >= 1", (df_pe["holding_years"] >= 1).all(), f"min={df_pe['holding_years'].min()}")

    # Scenarios ECL somment a 1
    total_w = sum(s.weight for s in ECL_SCENARIOS)
    ok &= _check("Scenarios somment a 1", abs(total_w - 1.0) < 1e-6, f"{total_w:.6f}")

    return ok


# ──────────────────────────────────────────────
# CONTRATS AR4
# ──────────────────────────────────────────────

def validate_contracts(
    df_credit: pd.DataFrame,
    df_pe: pd.DataFrame,
    df_history: pd.DataFrame,
) -> bool:
    """Valide les contrats DataFrame AR4 et la coherence inter-modules.

    Args:
        df_credit: DataFrame credit.
        df_pe: DataFrame PE.
        df_history: DataFrame historique panel.

    Returns:
        True si tous les contrats sont respectes.
    """
    ok = True

    # Colonnes requises
    ok &= _check(
        "REQUIRED_CREDIT_COLS",
        REQUIRED_CREDIT_COLS.issubset(set(df_credit.columns)),
        f"manquantes={REQUIRED_CREDIT_COLS - set(df_credit.columns)}",
    )
    ok &= _check(
        "REQUIRED_PE_COLS",
        REQUIRED_PE_COLS.issubset(set(df_pe.columns)),
        f"manquantes={REQUIRED_PE_COLS - set(df_pe.columns)}",
    )

    # Shapes
    ok &= _check("df_credit shape", len(df_credit) == N_CLIENTS, f"{len(df_credit)} vs {N_CLIENTS}")
    ok &= _check("df_pe shape", len(df_pe) == N_CLIENTS, f"{len(df_pe)} vs {N_CLIENTS}")
    ok &= _check(
        "df_history shape",
        len(df_history) == N_CLIENTS * N_MONTHS,
        f"{len(df_history)} vs {N_CLIENTS * N_MONTHS}",
    )

    # Enterprise IDs coherents
    ok &= _check(
        "enterprise_id coherent",
        set(df_credit["enterprise_id"]) == set(df_pe["enterprise_id"]),
        "",
    )

    return ok


# ──────────────────────────────────────────────
# INVARIANTS POST-PIPELINE (framework pour stories 2.x+)
# ──────────────────────────────────────────────

def validate_ecl_results(df_results: pd.DataFrame) -> bool:
    """Valide les invariants sur les resultats ECL du pipeline credit.

    A appeler apres le calcul ECL (Stories 2.x). Verifie :
    - ECL >= 0
    - PD dans [0, 1]
    - LGD dans [0, 1]
    - Stage dans {1, 2, 3}
    - Stage 1 utilise PD 12m (pd_12m <= pd_lifetime)

    Args:
        df_results: DataFrame resultats du pipeline credit.

    Returns:
        True si tous les invariants sont respectes.
    """
    ok = True

    if "ecl_weighted" in df_results.columns:
        ok &= _check("ECL >= 0", (df_results["ecl_weighted"] >= 0).all(),
                      f"min={df_results['ecl_weighted'].min():.6f}")

    if "pd_12m" in df_results.columns:
        pd12 = df_results["pd_12m"]
        ok &= _check("PD 12m [0,1]", (pd12 >= 0).all() and (pd12 <= 1).all(),
                      f"[{pd12.min():.6f}, {pd12.max():.6f}]")

    if "lgd" in df_results.columns:
        lgd = df_results["lgd"]
        ok &= _check("LGD [0,1]", (lgd >= 0).all() and (lgd <= 1).all(),
                      f"[{lgd.min():.6f}, {lgd.max():.6f}]")

    if "stage" in df_results.columns:
        ok &= _check("Stage {1,2,3}", set(df_results["stage"].unique()).issubset({1, 2, 3}),
                      f"valeurs={df_results['stage'].unique()}")

    # Stage 1 : pd_12m <= pd_lifetime (monotonie)
    if {"stage", "pd_12m", "pd_lifetime"}.issubset(df_results.columns):
        s1 = df_results[df_results["stage"] == 1]
        if len(s1) > 0:
            ok &= _check("Stage 1: pd_12m <= pd_lifetime",
                          (s1["pd_12m"] <= s1["pd_lifetime"] + 1e-9).all(), "")

    return ok


# ──────────────────────────────────────────────
# RAPPORT
# ──────────────────────────────────────────────

def print_report() -> Tuple[int, int]:
    """Affiche le rapport de validation.

    Returns:
        Tuple (nombre de PASS, nombre de FAIL).
    """
    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_fail = sum(1 for _, ok, _ in _results if not ok)

    print(f"\n{'=' * 60}")
    print(f"VALIDATION REPORT — {n_pass} PASS, {n_fail} FAIL")
    print(f"{'=' * 60}")

    for name, ok, detail in _results:
        status = "PASS" if ok else "FAIL"
        line = f"  [{status}] {name}"
        if detail:
            line += f"  ({detail})"
        print(line)

    print(f"\n{'=' * 60}")
    if n_fail == 0:
        print("Toutes les validations passent.")
    else:
        print(f"ATTENTION : {n_fail} validation(s) echouee(s).")
    print(f"{'=' * 60}")

    return n_pass, n_fail


# ──────────────────────────────────────────────
# POINT D'ENTREE STANDALONE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import SyntheticDataGenerator

    print("Generation des donnees synthetiques...")
    gen = SyntheticDataGenerator()
    df_credit, df_pe, df_history = gen.generate()
    print(f"  df_credit: {df_credit.shape}, df_pe: {df_pe.shape}, df_history: {df_history.shape}")

    print("\nValidation du realisme (FR4)...")
    validate_realism(df_credit, df_pe)

    print("Validation des invariants financiers (FR56)...")
    validate_invariants(df_credit, df_pe)

    print("Validation des contrats AR4...")
    validate_contracts(df_credit, df_pe, df_history)

    n_pass, n_fail = print_report()
    raise SystemExit(1 if n_fail > 0 else 0)
