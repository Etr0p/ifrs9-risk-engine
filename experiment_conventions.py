#!/usr/bin/env python
"""Experiment: comparison of 4 sign conventions in macro_to_z().

The Z-score transforms 5 macro variables into a single systematic factor.
For 3 variables the sign is unambiguous:
    - GDP:          higher = favorable   (z -= w*(x-mu)/sigma)
    - Unemployment: higher = adverse     (z += w*(x-mu)/sigma)
    - HPI:          higher = favorable   (z -= w*(x-mu)/sigma)

For 2 variables the sign is AMBIGUOUS:
    - Interest Rate: Merton (favorable) vs Debt Service (adverse)
    - Inflation:     Debt Erosion (favorable) vs Cost-Push (adverse)

This gives 4 conventions (2x2):
    A: IR favorable, Inflation favorable  (original code)
    B: IR adverse,   Inflation favorable  (current code)
    C: IR favorable, Inflation adverse
    D: IR adverse,   Inflation adverse

For each convention x 11 scenarios, the script runs the FULL pipeline and
scores each convention on 5 coherence criteria.

Usage:
    python experiment_conventions.py
"""

import sys
import os
import time
import warnings

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import polars as pl
from scipy.stats import spearmanr

# ── Project imports ──────────────────────────────────────────────────────
from ifrs9_cockpit.config import (
    PREDEFINED_SCENARIOS,
    SCENARIO_BASE,
)
from ifrs9_cockpit.synthetic_generator import generate_dataset
from ifrs9_cockpit.models.pd_model import PDModelSuite
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel
from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl
from ifrs9_cockpit.engine.comparator import PortfolioComparator
import ifrs9_cockpit.engine.balance_sheet_ecl as bse_module


# ======================================================================
# 1. PARAMETRIC macro_to_z — accepts a convention parameter
# ======================================================================

# Convention definitions:
#   "adverse" variables use:   z += w * (x - mu) / sigma
#   "favorable" variables use: z -= w * (x - mu) / sigma
#
# Unambiguous:
#   GDP            -> always favorable
#   Unemployment   -> always adverse
#   HPI            -> always favorable
#
# Ambiguous (varies by convention):
#   Interest Rate: A/C = favorable, B/D = adverse
#   Inflation:     A/B = favorable, C/D = adverse

CONVENTION_SIGNS = {
    "A": {"interest_rate": "favorable", "inflation_rate": "favorable"},
    "B": {"interest_rate": "adverse",   "inflation_rate": "favorable"},
    "C": {"interest_rate": "favorable", "inflation_rate": "adverse"},
    "D": {"interest_rate": "adverse",   "inflation_rate": "adverse"},
}

# Fixed signs for unambiguous variables
FIXED_ADVERSE = {"unemployment_rate"}
FIXED_FAVORABLE = {"gdp_growth", "hpi_growth"}


def macro_to_z_parametric(
    macro_params: dict,
    sensitivities: dict,
    convention: str,
) -> float:
    """Transform 5 macro variables into systematic factor Z using the
    specified sign convention.

    Z > 0 means stress (adverse); Z < 0 means relief (favorable).

    Args:
        macro_params: Dict of current macro values.
        sensitivities: Dict of asset-class sensitivities.
        convention: One of "A", "B", "C", "D".

    Returns:
        Systematic factor Z (standard normal scale).
    """
    _key_map = {
        "gdp_pct": ("gdp_growth", SCENARIO_BASE.gdp_growth, 1.8),
        "unemployment_rate": ("unemployment_rate", SCENARIO_BASE.unemployment_rate, 1.5),
        "interest_rate": ("interest_rate", SCENARIO_BASE.interest_rate, 1.0),
        "hpi_pct": ("hpi_growth", SCENARIO_BASE.hpi_growth, 3.0),
        "inflation_pct": ("inflation_rate", SCENARIO_BASE.inflation_rate, 1.2),
    }
    _direct_map = {
        "gdp_growth": (SCENARIO_BASE.gdp_growth, 1.8),
        "unemployment_rate": (SCENARIO_BASE.unemployment_rate, 1.5),
        "interest_rate": (SCENARIO_BASE.interest_rate, 1.0),
        "hpi_growth": (SCENARIO_BASE.hpi_growth, 3.0),
        "inflation_rate": (SCENARIO_BASE.inflation_rate, 1.2),
    }

    conv = CONVENTION_SIGNS[convention]

    z = 0.0
    for var_name, (mu, sigma) in _direct_map.items():
        # Resolve macro value (try direct name, then slider key)
        x = macro_params.get(var_name)
        if x is None:
            for slider_key, (mapped_var, mapped_mu, mapped_sigma) in _key_map.items():
                if mapped_var == var_name and slider_key in macro_params:
                    x = macro_params[slider_key]
                    break
        if x is None:
            continue

        w = sensitivities.get(var_name, 0.0)

        # Determine sign direction
        if var_name in FIXED_ADVERSE:
            # Always adverse: higher = worse
            z += w * (x - mu) / sigma
        elif var_name in FIXED_FAVORABLE:
            # Always favorable: higher = better
            z -= w * (x - mu) / sigma
        else:
            # Ambiguous variable: determined by convention
            direction = conv.get(var_name, "favorable")
            if direction == "adverse":
                z += w * (x - mu) / sigma
            else:
                z -= w * (x - mu) / sigma

    return z


# ======================================================================
# 2. SCENARIO CONVERSION & CLASSIFICATION
# ======================================================================

def convert_scenario(s: dict) -> dict:
    """Convert a PREDEFINED_SCENARIOS entry to macro_params dict."""
    bp = s.get("unemployment_bipolar", 0)
    return {
        "gdp_growth": s["gdp_pct"],
        "unemployment_rate": SCENARIO_BASE.unemployment_rate + abs(bp),
        "interest_rate": SCENARIO_BASE.interest_rate + s["interest_rate_bp"] / 100.0,
        "hpi_growth": s["hpi_pct"],
        "inflation_rate": s["inflation_pct"],
    }


# Scenario classification
ADVERSE_SCENARIOS = {
    "Crise financiere (GFC)",
    "Crise souveraine (2012)",
    "Stagflation",
    "Choc pandemique (COVID)",
}
FAVORABLE_SCENARIOS = {
    "Reprise",
    "Hypercroissance",
    "Boom immobilier",
}
NEUTRAL_SCENARIOS = {
    "Central",
    "Rupture techno",
    "Trappe a liquidite",
    "Transition climatique brutale",
}

# Historical RAROC ordering (1 = best, 11 = worst)
HISTORICAL_RANKING = {
    "Hypercroissance": 1,
    "Boom immobilier": 2,
    "Reprise": 3,
    "Central": 4,
    "Rupture techno": 5,
    "Transition climatique brutale": 6,
    "Trappe a liquidite": 7,
    "Choc pandemique (COVID)": 8,
    "Crise souveraine (2012)": 9,
    "Stagflation": 10,
    "Crise financiere (GFC)": 11,
}


# ======================================================================
# 3. PIPELINE RUNNER
# ======================================================================

def run_pipeline_for_convention(
    convention: str,
    bundle,
    pd_suite,
    lgd_model,
    ead_model,
    pd_current,
    pd_origination,
) -> dict:
    """Run the full pipeline for one convention across all 11 scenarios.

    Returns:
        Dict mapping scenario_name -> {
            "raroc_total": float (portfolio RAROC),
            "raroc_by_class": dict[str, float],
        }
    """
    # Save original macro_to_z so we can restore it
    original_macro_to_z = bse_module.macro_to_z

    # Monkey-patch macro_to_z with the parametric version for this convention
    bse_module.macro_to_z = lambda macro, sens: macro_to_z_parametric(
        macro, sens, convention
    )

    results = {}
    try:
        for scenario_name, scenario_dict in PREDEFINED_SCENARIOS.items():
            macro = convert_scenario(scenario_dict)

            # ECL pipeline (Level 1: corporate loans)
            ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
            res_credit = ecl_calc.calculate(
                bundle.df_credit,
                pd_current,
                pd_origination,
                unemployment_override=macro["unemployment_rate"],
                gdp_override=macro["gdp_growth"],
                interest_rate_override=macro["interest_rate"],
                hpi_override=macro["hpi_growth"],
                inflation_override=macro["inflation_rate"],
            )

            # PE pipeline
            pe_calc = PECalculator()
            # Determine if unemployment is crisis type (bipolar < 0 means econ crisis)
            bp = scenario_dict.get("unemployment_bipolar", 0)
            is_crisis = bp < 0
            res_pe = pe_calc.calculate(
                bundle.df_pe,
                unemployment_override=macro["unemployment_rate"],
                gdp_override=macro["gdp_growth"],
                interest_rate_override=macro["interest_rate"],
                hpi_override=macro["hpi_growth"],
                inflation_override=macro["inflation_rate"],
                unemployment_crisis=is_crisis,
            )

            # Balance sheet ECL (Level 2/3 classes — uses monkey-patched macro_to_z)
            bs_ecl = compute_balance_sheet_ecl(bundle.df_balance_sheet, macro)

            # Portfolio comparator -> RAROC multiclass
            comp = PortfolioComparator(
                res_credit, res_pe,
                df_balance_sheet_ecl=bs_ecl,
                macro_params=macro,
            )
            df_raroc = comp.compute_raroc_multiclass()

            # Extract portfolio-level RAROC (Total row)
            total_row = df_raroc.filter(pl.col("asset_class") == "Total")
            if len(total_row) > 0:
                raroc_total = float(total_row["raroc"][0])
            else:
                raroc_total = 0.0

            # Extract per-class RAROC
            raroc_by_class = {}
            for row in df_raroc.iter_rows(named=True):
                if row["asset_class"] != "Total":
                    raroc_by_class[row["asset_class"]] = row["raroc"]

            results[scenario_name] = {
                "raroc_total": raroc_total,
                "raroc_by_class": raroc_by_class,
            }
    finally:
        # Restore original macro_to_z
        bse_module.macro_to_z = original_macro_to_z

    return results


# ======================================================================
# 4. COHERENCE SCORING
# ======================================================================

def score_convention(results: dict) -> dict:
    """Score a convention on 5 coherence criteria.

    Args:
        results: Dict scenario_name -> {"raroc_total", "raroc_by_class"}.

    Returns:
        Dict with criterion scores and details.
    """
    # Extract portfolio RAROC for Central
    raroc_central = results["Central"]["raroc_total"]

    # ── C1: Monotonie adverse ──
    # All adverse scenarios should have RAROC < Central
    c1_details = {}
    c1_pass = 0
    c1_total = len(ADVERSE_SCENARIOS)
    for name in ADVERSE_SCENARIOS:
        r = results[name]["raroc_total"]
        ok = r < raroc_central
        c1_details[name] = {"raroc": r, "pass": ok}
        if ok:
            c1_pass += 1
    c1_score = c1_pass / max(c1_total, 1)

    # ── C2: Monotonie favorable ──
    # All favorable scenarios should have RAROC > Central
    c2_details = {}
    c2_pass = 0
    c2_total = len(FAVORABLE_SCENARIOS)
    for name in FAVORABLE_SCENARIOS:
        r = results[name]["raroc_total"]
        ok = r > raroc_central
        c2_details[name] = {"raroc": r, "pass": ok}
        if ok:
            c2_pass += 1
    c2_score = c2_pass / max(c2_total, 1)

    # ── C3: Signe correct ──
    # No favorable scenario with negative RAROC,
    # No adverse scenario with RAROC > Central + 5pp
    c3_pass = 0
    c3_total = len(FAVORABLE_SCENARIOS) + len(ADVERSE_SCENARIOS)
    c3_details = {}
    for name in FAVORABLE_SCENARIOS:
        r = results[name]["raroc_total"]
        ok = r >= 0
        c3_details[name] = {"raroc": r, "test": "raroc >= 0", "pass": ok}
        if ok:
            c3_pass += 1
    for name in ADVERSE_SCENARIOS:
        r = results[name]["raroc_total"]
        ok = r <= raroc_central + 0.05
        c3_details[name] = {
            "raroc": r,
            "test": f"raroc <= {raroc_central + 0.05:.4f}",
            "pass": ok,
        }
        if ok:
            c3_pass += 1
    c3_score = c3_pass / max(c3_total, 1)

    # ── C4: Coherence historique (Spearman rank correlation) ──
    # Rank scenarios by model RAROC (descending → rank 1 = highest RAROC)
    scenario_names = list(HISTORICAL_RANKING.keys())
    model_rarocs = [results[name]["raroc_total"] for name in scenario_names]
    historical_ranks = [HISTORICAL_RANKING[name] for name in scenario_names]

    # Rank model RAROC (highest RAROC = rank 1)
    # np.argsort on -array gives indices for descending sort
    sorted_idx = np.argsort(-np.array(model_rarocs))
    model_ranks = np.zeros(len(model_rarocs), dtype=int)
    for rank_val, idx in enumerate(sorted_idx, 1):
        model_ranks[idx] = rank_val

    rho, p_value = spearmanr(model_ranks, historical_ranks)
    # Score: rho in [-1, 1], map to [0, 1] via (rho + 1) / 2
    c4_score = max(0.0, (rho + 1) / 2)
    c4_details = {
        "spearman_rho": rho,
        "p_value": p_value,
        "model_ranks": dict(zip(scenario_names, model_ranks.tolist())),
        "historical_ranks": HISTORICAL_RANKING,
    }

    # ── C5: Stabilite inter-classes ──
    # For each adverse scenario, count how many of the 14 classes have
    # RAROC < their Central RAROC. Majority = stable.
    # Similarly for favorable: RAROC > Central.
    c5_agree_count = 0
    c5_total_checks = 0
    c5_details = {}

    central_by_class = results["Central"]["raroc_by_class"]

    for name in ADVERSE_SCENARIOS:
        class_rarocs = results[name]["raroc_by_class"]
        agree = 0
        total_classes = 0
        for cls, r in class_rarocs.items():
            if cls in central_by_class:
                total_classes += 1
                if r < central_by_class[cls]:
                    agree += 1
        majority = agree > total_classes / 2 if total_classes > 0 else False
        c5_details[name] = {
            "agree": agree,
            "total": total_classes,
            "majority": majority,
        }
        if majority:
            c5_agree_count += 1
        c5_total_checks += 1

    for name in FAVORABLE_SCENARIOS:
        class_rarocs = results[name]["raroc_by_class"]
        agree = 0
        total_classes = 0
        for cls, r in class_rarocs.items():
            if cls in central_by_class:
                total_classes += 1
                if r > central_by_class[cls]:
                    agree += 1
        majority = agree > total_classes / 2 if total_classes > 0 else False
        c5_details[name] = {
            "agree": agree,
            "total": total_classes,
            "majority": majority,
        }
        if majority:
            c5_agree_count += 1
        c5_total_checks += 1

    c5_score = c5_agree_count / max(c5_total_checks, 1)

    # ── Composite score ──
    # Equal weight (20% each)
    composite = (c1_score + c2_score + c3_score + c4_score + c5_score) / 5.0

    return {
        "C1_monotonie_adverse": {"score": c1_score, "details": c1_details},
        "C2_monotonie_favorable": {"score": c2_score, "details": c2_details},
        "C3_signe_correct": {"score": c3_score, "details": c3_details},
        "C4_coherence_historique": {"score": c4_score, "details": c4_details},
        "C5_stabilite_interclasses": {"score": c5_score, "details": c5_details},
        "composite": composite,
    }


# ======================================================================
# 5. OUTPUT FORMATTING
# ======================================================================

SEPARATOR = "=" * 88

def fmt_pct(v: float) -> str:
    """Format a float as percentage."""
    return f"{v * 100:+.2f}%"


def fmt_score(v: float) -> str:
    """Format a score in [0,1] as percentage."""
    return f"{v * 100:.0f}%"


def print_raroc_table(all_results: dict) -> None:
    """Print a table of portfolio RAROC per convention x scenario."""
    conventions = sorted(all_results.keys())
    scenarios = list(PREDEFINED_SCENARIOS.keys())

    # Header
    hdr = f"{'Scenario':<35s}"
    for conv in conventions:
        hdr += f" {'Conv ' + conv:>12s}"
    print(hdr)
    print("-" * len(hdr))

    for sc in scenarios:
        # Determine category tag
        if sc in ADVERSE_SCENARIOS:
            tag = " [ADV]"
        elif sc in FAVORABLE_SCENARIOS:
            tag = " [FAV]"
        else:
            tag = " [NEU]"
        line = f"{sc:<29s}{tag}"
        for conv in conventions:
            r = all_results[conv][sc]["raroc_total"]
            line += f"  {fmt_pct(r):>10s}"
        print(line)


def print_score_summary(all_scores: dict) -> None:
    """Print a summary table of scores per convention."""
    conventions = sorted(all_scores.keys())
    criteria = [
        "C1_monotonie_adverse",
        "C2_monotonie_favorable",
        "C3_signe_correct",
        "C4_coherence_historique",
        "C5_stabilite_interclasses",
    ]
    labels = {
        "C1_monotonie_adverse": "C1 Monotonie adverse",
        "C2_monotonie_favorable": "C2 Monotonie favorable",
        "C3_signe_correct": "C3 Signe correct",
        "C4_coherence_historique": "C4 Coherence historique",
        "C5_stabilite_interclasses": "C5 Stabilite inter-classes",
    }

    hdr = f"{'Criterion':<30s}"
    for conv in conventions:
        hdr += f" {'Conv ' + conv:>12s}"
    print(hdr)
    print("-" * len(hdr))

    for crit in criteria:
        line = f"{labels[crit]:<30s}"
        for conv in conventions:
            s = all_scores[conv][crit]["score"]
            line += f"  {fmt_score(s):>10s}"
        print(line)

    # Composite
    print("-" * len(hdr))
    line = f"{'COMPOSITE (equal weight)':<30s}"
    for conv in conventions:
        s = all_scores[conv]["composite"]
        line += f"  {fmt_score(s):>10s}"
    print(line)


def print_spearman_details(all_scores: dict) -> None:
    """Print Spearman rank correlation details for each convention."""
    conventions = sorted(all_scores.keys())

    for conv in conventions:
        details = all_scores[conv]["C4_coherence_historique"]["details"]
        rho = details["spearman_rho"]
        p = details["p_value"]
        print(f"\n  Convention {conv}: Spearman rho = {rho:+.4f} (p = {p:.4f})")

        # Show ranking comparison
        model_ranks = details["model_ranks"]
        hist_ranks = details["historical_ranks"]
        scenarios = sorted(hist_ranks.keys(), key=lambda s: hist_ranks[s])

        print(f"  {'Scenario':<35s} {'Hist':>5s} {'Model':>6s} {'Delta':>6s}")
        for sc in scenarios:
            h = hist_ranks[sc]
            m = model_ranks.get(sc, "?")
            delta = m - h if isinstance(m, int) else "?"
            print(f"  {sc:<35s} {h:>5d} {m:>6d} {delta:>+6d}")


def print_class_stability_details(all_scores: dict) -> None:
    """Print C5 inter-class stability details."""
    conventions = sorted(all_scores.keys())
    for conv in conventions:
        details = all_scores[conv]["C5_stabilite_interclasses"]["details"]
        print(f"\n  Convention {conv}:")
        for sc, info in details.items():
            tag = "PASS" if info["majority"] else "FAIL"
            print(f"    {sc:<35s}  {info['agree']:>2d}/{info['total']:>2d} agree  [{tag}]")


# ======================================================================
# 6. MAIN
# ======================================================================

def main():
    t_start = time.time()
    print(SEPARATOR)
    print("  EXPERIMENT: Sign Convention Comparison in macro_to_z()")
    print("  4 conventions x 11 scenarios = 44 full pipeline runs")
    print(SEPARATOR)

    # ── Step 1: Generate data (ONCE) ──
    print("\n[1/4] Generating dataset...")
    t0 = time.time()
    bundle = generate_dataset(n_clients=30_000, seed=123)
    print(f"       Done in {time.time() - t0:.1f}s")
    print(f"       Credit: {len(bundle.df_credit):,} rows")
    print(f"       PE:     {len(bundle.df_pe):,} rows")
    print(f"       BS:     {len(bundle.df_balance_sheet):,} asset classes")

    # ── Step 2: Fit models (ONCE) ──
    print("\n[2/4] Fitting PD/LGD/EAD models...")
    t0 = time.time()

    pd_suite = PDModelSuite(seed=123)
    pd_suite.fit(bundle.df_credit)
    pd_current = pd_suite.predict_active(bundle.df_credit)
    pd_origination = bundle.df_credit["pd_origination"].to_numpy()

    lgd_model = LGDModel()
    lgd_model.fit(bundle.df_credit)

    ead_model = EADModel()
    ead_model.fit(bundle.df_credit)

    print(f"       Done in {time.time() - t0:.1f}s")
    print(f"       PD mean (active model): {pd_current.mean():.4f}")
    print(f"       PD orig mean:           {pd_origination.mean():.4f}")

    # ── Step 3: Run pipelines for all 4 conventions ──
    print("\n[3/4] Running 44 pipeline evaluations...")
    all_results = {}
    all_scores = {}

    for conv in ["A", "B", "C", "D"]:
        t0 = time.time()
        print(f"\n       Convention {conv} "
              f"(IR={CONVENTION_SIGNS[conv]['interest_rate']}, "
              f"Infl={CONVENTION_SIGNS[conv]['inflation_rate']})...")

        results = run_pipeline_for_convention(
            convention=conv,
            bundle=bundle,
            pd_suite=pd_suite,
            lgd_model=lgd_model,
            ead_model=ead_model,
            pd_current=pd_current,
            pd_origination=pd_origination,
        )
        all_results[conv] = results

        # Score immediately
        scores = score_convention(results)
        all_scores[conv] = scores

        elapsed = time.time() - t0
        print(f"       Done in {elapsed:.1f}s  |  "
              f"Composite = {fmt_score(scores['composite'])}")

    # ── Step 4: Output results ──
    print(f"\n\n{'=' * 88}")
    print("  RESULTS")
    print(f"{'=' * 88}")

    # 4a. RAROC table
    print(f"\n{'_' * 88}")
    print("  Table 1: Portfolio RAROC by Convention x Scenario")
    print(f"{'_' * 88}\n")
    print_raroc_table(all_results)

    # 4b. Score summary
    print(f"\n{'_' * 88}")
    print("  Table 2: Coherence Scores by Convention")
    print(f"{'_' * 88}\n")
    print_score_summary(all_scores)

    # 4c. Spearman details
    print(f"\n{'_' * 88}")
    print("  Table 3: Spearman Rank Correlation (C4) — Historical vs Model")
    print(f"{'_' * 88}")
    print_spearman_details(all_scores)

    # 4d. Inter-class stability details
    print(f"\n{'_' * 88}")
    print("  Table 4: Inter-Class Stability (C5) — Majority Agreement")
    print(f"{'_' * 88}")
    print_class_stability_details(all_scores)

    # 4e. Detailed C1/C2 per convention
    print(f"\n{'_' * 88}")
    print("  Table 5: Monotonicity Details (C1 adverse, C2 favorable)")
    print(f"{'_' * 88}")
    for conv in ["A", "B", "C", "D"]:
        s = all_scores[conv]
        central_r = all_results[conv]["Central"]["raroc_total"]
        print(f"\n  Convention {conv} (Central RAROC = {fmt_pct(central_r)}):")

        print(f"    C1 Adverse (should be < Central):")
        for sc, info in s["C1_monotonie_adverse"]["details"].items():
            tag = "PASS" if info["pass"] else "FAIL"
            print(f"      {sc:<35s}  RAROC = {fmt_pct(info['raroc'])}  [{tag}]")

        print(f"    C2 Favorable (should be > Central):")
        for sc, info in s["C2_monotonie_favorable"]["details"].items():
            tag = "PASS" if info["pass"] else "FAIL"
            print(f"      {sc:<35s}  RAROC = {fmt_pct(info['raroc'])}  [{tag}]")

    # ── WINNER ──
    print(f"\n\n{'#' * 88}")
    print("  WINNER DECLARATION")
    print(f"{'#' * 88}\n")

    # Find the best convention
    best_conv = max(all_scores.keys(), key=lambda c: all_scores[c]["composite"])
    best_score = all_scores[best_conv]["composite"]

    # Summary line for each convention
    for conv in ["A", "B", "C", "D"]:
        s = all_scores[conv]["composite"]
        ir_sign = CONVENTION_SIGNS[conv]["interest_rate"]
        infl_sign = CONVENTION_SIGNS[conv]["inflation_rate"]
        marker = "  <<<  WINNER" if conv == best_conv else ""
        print(f"  Convention {conv} (IR={ir_sign:>9s}, Infl={infl_sign:>9s}): "
              f"Composite = {fmt_score(s)}{marker}")

    print(f"\n  --> Convention {best_conv} is the most coherent sign convention")
    print(f"      with a composite score of {fmt_score(best_score)}.")

    # Interpretation
    ir = CONVENTION_SIGNS[best_conv]["interest_rate"]
    infl = CONVENTION_SIGNS[best_conv]["inflation_rate"]
    print(f"\n  Interpretation:")
    if ir == "adverse":
        print(f"    - Interest Rate: ADVERSE (debt service burden dominates Merton drift)")
    else:
        print(f"    - Interest Rate: FAVORABLE (Merton risk-free drift dominates debt service)")
    if infl == "adverse":
        print(f"    - Inflation: ADVERSE (cost-push effect dominates debt erosion)")
    else:
        print(f"    - Inflation: FAVORABLE (debt erosion effect dominates cost-push)")

    # Check if current code (Convention B) is the winner
    if best_conv == "B":
        print(f"\n  This confirms the current implementation in balance_sheet_ecl.py")
        print(f"  (Convention B: IR adverse, Inflation favorable).")
    else:
        current_score = all_scores["B"]["composite"]
        print(f"\n  NOTE: The current implementation uses Convention B "
              f"(composite = {fmt_score(current_score)}).")
        print(f"  Convention {best_conv} scores higher. Consider updating macro_to_z().")

    total_elapsed = time.time() - t_start
    print(f"\n  Total elapsed time: {total_elapsed:.1f}s")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
