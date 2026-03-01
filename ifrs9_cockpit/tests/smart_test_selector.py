"""Plugin pytest pour selection intelligente de tests.

Detecte les fichiers modifies via ``git diff`` et ne selectionne que les
classes/fichiers de tests pertinents grace a une table de dependances.

Usage::

    # Lancer uniquement les tests impactes par les changements git
    pytest --smart

    # Forcer tous les tests (ignorer le filtre)
    pytest --all

    # Voir quels tests seraient selectionnes sans les lancer
    pytest --smart --collect-only

Le mapping MODULE_TO_TESTS definit, pour chaque module source, la liste
des patterns de tests a lancer.  Chaque pattern peut etre :
  - ``"test_file.py"``            → tout le fichier
  - ``"test_file.py::TestClass"`` → une seule classe de test

Les clefs terminant par ``/`` font un prefix-match (packages).

Si un module modifie n'est pas dans le mapping, les tests invariants
sont lances par securite.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Set

import pytest


# ──────────────────────────────────────────────
# DEPENDENCY MAP : source module → test patterns
# ──────────────────────────────────────────────

_TESTS_DIR = "ifrs9_cockpit/tests"

MODULE_TO_TESTS = {
    # ── Config package ──
    "ifrs9_cockpit/config/sectors.py": [
        "test_config.py::TestSectorConfig",
        "test_config.py::TestStandalone",
        "test_comparator.py::TestMulticlassAllocation",
        "test_comparator.py::TestRAROCMulticlass",
        "test_comparator.py::TestRAROCMulticlassReasonable",
        "test_multi_asset.py::TestAssetClassProfile",
        "test_multi_asset.py::TestRegulatoryNorms",
    ],
    "ifrs9_cockpit/config/scenarios.py": [
        "test_config.py::TestMacroScenarios",
        "test_config.py::TestStandalone",
        "test_config.py::TestValidation",
        "test_cro_analytics.py",
    ],
    "ifrs9_cockpit/config/models.py": [
        "test_config.py::TestDataFrameContracts",
        "test_config.py::TestFeaturesAndDashboard",
        "test_config.py::TestDataContract",
        "test_config.py::TestMathRigorStructures",
        "test_staging_ecl.py::TestStaging",
    ],
    "ifrs9_cockpit/config/basel.py": [
        "test_config.py::TestBaselConfig",
        "test_config.py::TestRiskAppetiteConfig",
        "test_config.py::TestPEClassification",
        "test_comparator.py::TestOptimizeAllocation",
        "test_comparator.py::TestMulticlassAllocation",
        "test_multi_asset.py::TestRegulatoryNorms",
    ],
    "ifrs9_cockpit/config/dashboard.py": [
        "test_config.py::TestFeaturesAndDashboard",
        "test_dashboard.py",
    ],
    "ifrs9_cockpit/config/rules.py": [
        "test_config.py::TestMacroIncoherenceRules",
        "test_config.py::TestValidation",
    ],
    "ifrs9_cockpit/config/__init__.py": [
        "test_config.py",
    ],

    # ── Engine Comparator package ──
    "ifrs9_cockpit/engine/comparator/metrics.py": [
        "test_comparator.py::TestAdvancedCreditMetrics",
        "test_comparator.py::TestHHICrossCell",
        "test_comparator.py::TestRAROCEVA",
        "test_comparator.py::TestRAROCEVACache",
        "test_comparator.py::TestResilienceScores",
        "test_comparator.py::TestAsymmetryMatrix",
        "test_comparator.py::TestHHINameLevel",
        "test_comparator.py::TestCorrelationPenalty",
        "test_comparator.py::TestGreenAssetRatio",
        "test_comparator.py::TestTexasRatioRenamed",
        "test_comparator.py::TestRAROCMulticlass",
        "test_comparator.py::TestRAROCMulticlassReasonable",
    ],
    "ifrs9_cockpit/engine/comparator/optimizer.py": [
        "test_comparator.py::TestOptimizeAllocation",
        "test_comparator.py::TestSoftmaxWeights",
        "test_comparator.py::TestMulticlassAllocation",
        "test_comparator.py::TestPhase1Phase2",
        "test_comparator.py::TestStressIntensity",
        "test_comparator.py::TestAsymmetricIlliquidity",
        "test_comparator.py::TestAsymmetricVolMultiplier",
        "test_comparator.py::TestAsymmetricPeBand",
        "test_comparator.py::TestBLConfidence",
        "test_comparator.py::TestCorrelation10x10",
    ],
    "ifrs9_cockpit/engine/comparator/sensitivity.py": [
        "test_comparator.py::TestCRR3Sensitivity",
        "test_comparator.py::TestTippingPoints",
    ],
    "ifrs9_cockpit/engine/comparator/crr3.py": [
        "test_comparator.py::TestCRR3RiskWeight",
        "test_comparator.py::TestStandalone",
    ],
    "ifrs9_cockpit/engine/comparator/__init__.py": [
        "test_comparator.py::TestPortfolioComparatorInit",
    ],

    # ── Engine modules ──
    "ifrs9_cockpit/engine/balance_sheet_ecl.py": [
        "test_multi_asset.py::TestVasicekECL",
        "test_multi_asset.py::TestBalanceSheetECL",
        "test_multi_asset.py::TestMacroToZ",
        "test_multi_asset.py::TestClimateRisk",
        "test_multi_asset.py::TestRegulatoryNorms",
        "test_comparator.py::TestRAROCMulticlass",
        "test_comparator.py::TestMulticlassAllocation",
    ],
    "ifrs9_cockpit/engine/staging.py": [
        "test_staging_ecl.py::TestStaging",
        "test_staging_ecl.py::TestStandalone",
        "test_invariants.py",
    ],
    "ifrs9_cockpit/engine/ecl_calculator.py": [
        "test_staging_ecl.py::TestECLCalculator",
        "test_invariants.py",
    ],
    "ifrs9_cockpit/engine/pe_calculator.py": [
        "test_pe.py",
        "test_comparator.py::TestRAROCEVA",
    ],
    "ifrs9_cockpit/engine/contagion.py": [
        "test_multi_asset.py::TestContagion",
        "test_multi_asset.py::TestClimateRisk",
    ],

    # ── Models ──
    "ifrs9_cockpit/models/pd_model.py": ["test_pd_model.py"],
    "ifrs9_cockpit/models/lgd_model.py": ["test_lgd_ead.py::TestLGDModel"],
    "ifrs9_cockpit/models/ead_model.py": ["test_lgd_ead.py::TestEADModel"],
    "ifrs9_cockpit/models/pe_model.py": [
        "test_pe.py::TestPEModelNAVBaseline",
        "test_pe.py::TestPEModelIPEVMethods",
        "test_pe.py::TestPEModelMultipleCompression",
        "test_pe.py::TestPEModelOverrideZero",
        "test_pe.py::TestPEModelScenarios",
        "test_pe.py::TestPEModelSummary",
        "test_pe.py::TestPEModelStandalone",
    ],
    "ifrs9_cockpit/models/woe.py": [
        "test_pd_model.py::TestPDModelTraining",
        "test_pd_model.py::TestExpertReviewImprovements",
    ],

    # ── Data generation ──
    "ifrs9_cockpit/data/generator.py": ["test_generator.py", "test_invariants.py"],
    "ifrs9_cockpit/synthetic_generator/": [
        "test_multi_asset.py::TestDGPBalanceSheet",
        "test_generator.py",
        "test_mortgage_positions.py",
        "test_consumer_positions.py",
        "test_sovereign_positions.py",
        "test_covered_bonds_positions.py",
        "test_interbank_positions.py",
        "test_securitisation_positions.py",
        "test_trade_finance_positions.py",
        "test_project_finance_positions.py",
    ],
    # Per-file position generators (more precise than prefix match)
    "ifrs9_cockpit/synthetic_generator/mortgage_positions.py": ["test_mortgage_positions.py"],
    "ifrs9_cockpit/synthetic_generator/consumer_positions.py": ["test_consumer_positions.py"],
    "ifrs9_cockpit/synthetic_generator/sovereign_positions.py": ["test_sovereign_positions.py"],
    "ifrs9_cockpit/synthetic_generator/covered_bonds_positions.py": ["test_covered_bonds_positions.py"],
    "ifrs9_cockpit/synthetic_generator/interbank_positions.py": ["test_interbank_positions.py"],
    "ifrs9_cockpit/synthetic_generator/securitisation_positions.py": ["test_securitisation_positions.py"],
    "ifrs9_cockpit/synthetic_generator/trade_finance_positions.py": ["test_trade_finance_positions.py"],
    "ifrs9_cockpit/synthetic_generator/project_finance_positions.py": ["test_project_finance_positions.py"],
    "ifrs9_cockpit/synthetic_generator/balance_sheet_ecl.py": [
        "test_multi_asset.py::TestBalanceSheetECL",
        "test_multi_asset.py::TestVasicekECL",
    ],
    "ifrs9_cockpit/synthetic_generator_v4.py": [
        "test_multi_asset.py::TestDGPBalanceSheet",
        "test_generator.py",
    ],

    # ── AI Analyst ──
    "ifrs9_cockpit/ai_analyst/layer1_crossing.py": ["test_cro_analytics.py::TestLayer1Crossing"],
    "ifrs9_cockpit/ai_analyst/layer2_euler.py": ["test_cro_analytics.py::TestLayer2Proportional"],
    "ifrs9_cockpit/ai_analyst/layer3_rst.py": [
        "test_cro_analytics.py::TestLayer3RST",
        "test_cro_analytics.py::TestAdversarialRST",
    ],
    "ifrs9_cockpit/ai_analyst/layer4_regime.py": ["test_cro_analytics.py::TestLayer4Regime"],
    "ifrs9_cockpit/ai_analyst/layer5_prospective.py": ["test_cro_analytics.py::TestLayer5Prospective"],
    "ifrs9_cockpit/ai_analyst/orchestrator.py": ["test_cro_analytics.py::TestOrchestrator"],
    "ifrs9_cockpit/ai_analyst/types.py": ["test_cro_analytics.py::TestTypes"],

    # ── Causal ──
    "ifrs9_cockpit/causal/engine.py": ["test_causal.py"],

    # ── Virtual CRO ──
    "ifrs9_cockpit/virtual_cro/agents.py": [
        "test_virtual_cro.py::TestMacroAgent",
        "test_virtual_cro.py::TestQuantAgent",
        "test_virtual_cro.py::TestPEAgent",
        "test_virtual_cro.py::TestContrarianAgent",
    ],
    "ifrs9_cockpit/virtual_cro/fusion.py": ["test_virtual_cro.py::TestDSFusion"],
    "ifrs9_cockpit/virtual_cro/qbaf.py": ["test_virtual_cro.py::TestNeSyQBAF"],
    "ifrs9_cockpit/virtual_cro/pma.py": ["test_virtual_cro.py::TestPMAEngine"],
    "ifrs9_cockpit/virtual_cro/engine.py": [
        "test_virtual_cro.py::TestVirtualCROEngine",
        "test_virtual_cro.py::TestVCROCharts",
    ],

    # ── Governance ──
    "ifrs9_cockpit/engine/conformal.py": ["test_governance.py::TestConformalPredictor"],
    "ifrs9_cockpit/engine/sobol_analysis.py": ["test_governance.py::TestSobolAnalysis"],
    "ifrs9_cockpit/engine/vrp.py": ["test_governance.py::TestVRP"],
    "ifrs9_cockpit/engine/rmt.py": ["test_governance.py::TestRMT"],
    "ifrs9_cockpit/engine/signatures.py": ["test_governance.py::TestSignatures"],
    "ifrs9_cockpit/engine/tda.py": ["test_governance.py::TestTDA"],
    "ifrs9_cockpit/engine/compliance_gates.py": ["test_governance.py::TestComplianceGates"],
    "ifrs9_cockpit/engine/hmm_regime.py": [
        "test_cro_analytics.py::TestLayer4Regime",
        "test_governance.py::TestGovernanceCharts",
    ],
    "ifrs9_cockpit/engine/gflownet.py": [
        "test_cro_analytics.py::TestLayer4Regime",
    ],

    # ── Dashboard (package prefix-match) ──
    "ifrs9_cockpit/dashboard/charts/": ["test_dashboard.py::TestChartsModule"],
    "ifrs9_cockpit/dashboard/callbacks/": [
        "test_dashboard.py::TestStory62PipelineOrchestration",
        "test_dashboard.py::TestStory65ShapAndExport",
        "test_dashboard.py::TestStory66RstDrillDown",
    ],
    "ifrs9_cockpit/dashboard/components/": ["test_dashboard.py::TestComponentsModule"],
    "ifrs9_cockpit/dashboard/layout.py": [
        "test_dashboard.py::TestStory61SidebarSlidersScenarios",
        "test_dashboard.py::TestStory64DesignSystemSteelBlue",
        "test_dashboard.py::TestStory64Tabs",
    ],
    "ifrs9_cockpit/dashboard/ids.py": ["test_dashboard.py"],
    "ifrs9_cockpit/dashboard/components_dash.py": ["test_dashboard.py::TestComponentsModule"],
    "ifrs9_cockpit/dashboard/styles.py": ["test_dashboard.py::TestStylesModule"],

    # ── Utils ──
    "ifrs9_cockpit/utils/helpers.py": [
        "test_config.py::TestStandalone",
        "test_staging_ecl.py::TestStandalone",
    ],

    # ── Training ──
    "ifrs9_cockpit/training/train.py": ["test_training.py"],

    # ── Conftest / test infra ──
    "ifrs9_cockpit/tests/conftest.py": ["__ALL__"],
}


def _get_changed_files() -> List[str]:
    """Retourne les fichiers modifies (staged + unstaged + untracked)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        tracked = result.stdout.strip().split("\n") if result.stdout.strip() else []

        result2 = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True, text=True, timeout=10,
        )
        unstaged = result2.stdout.strip().split("\n") if result2.stdout.strip() else []

        all_files = list(set(tracked + unstaged))
        return [f.replace("\\", "/") for f in all_files if f]

    except Exception:
        return []


def _resolve_tests(changed_files: List[str]) -> Set[str]:
    """Resout les fichiers modifies en patterns de tests a lancer.

    Returns:
        Set de patterns ``"test_file.py"`` ou ``"test_file.py::TestClass"``.
        Set vide = lancer tous les tests (__ALL__ ou conftest modifie).
    """
    test_patterns: Set[str] = set()
    has_unmapped = False

    for changed in changed_files:
        changed_norm = changed.replace("\\", "/")

        # Si c'est deja un fichier de test, l'inclure directement
        if changed_norm.startswith(f"{_TESTS_DIR}/test_"):
            test_patterns.add(changed_norm.split("/")[-1])
            continue

        # Chercher dans le mapping
        matched = False
        for module_pattern, tests in MODULE_TO_TESTS.items():
            if module_pattern.endswith("/"):
                # Package prefix match
                if module_pattern in changed_norm:
                    matched = True
                    if "__ALL__" in tests:
                        return set()
                    test_patterns.update(tests)
                    break
            else:
                # Exact file match
                if changed_norm.endswith(module_pattern) or changed_norm == module_pattern:
                    matched = True
                    if "__ALL__" in tests:
                        return set()
                    test_patterns.update(tests)
                    break

        if not matched:
            if "ifrs9_cockpit/" in changed_norm and changed_norm.endswith(".py"):
                has_unmapped = True

    # Si fichiers Python non mappes, ajouter les tests invariants par securite
    if has_unmapped:
        test_patterns.add("test_invariants.py")

    return test_patterns


# ──────────────────────────────────────────────
# PYTEST PLUGIN HOOKS
# ──────────────────────────────────────────────

def pytest_addoption(parser):
    """Ajoute les options --smart, --all et --slow a pytest."""
    parser.addoption(
        "--smart",
        action="store_true",
        default=False,
        help="Selectionner uniquement les tests impactes par les changements git.",
    )
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="Forcer tous les tests (ignore --smart).",
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Inclure les tests marques @pytest.mark.slow (standalones longs).",
    )


def pytest_configure(config):
    """Enregistre le marqueur 'slow'."""
    config.addinivalue_line(
        "markers",
        "slow: tests longs (standalone __main__, reproductibilite). "
        "Exclus par defaut, inclus avec --slow ou --all.",
    )


def pytest_collection_modifyitems(session, config, items):
    """Filtre les tests : smart-select par git diff + exclusion @slow."""
    run_all = config.getoption("--all", default=False)
    run_slow = config.getoption("--slow", default=False)
    run_smart = config.getoption("--smart", default=False)

    # --- Phase 1 : exclure @pytest.mark.slow sauf --slow ---
    if not run_slow:
        slow_marker = pytest.mark.slow
        kept = []
        slow_deselected = []
        for item in items:
            if item.get_closest_marker("slow"):
                slow_deselected.append(item)
            else:
                kept.append(item)
        if slow_deselected:
            config.hook.pytest_deselected(items=slow_deselected)
            items[:] = kept
            n_slow = len(slow_deselected)
            print(f"\n[smart-test] {n_slow} tests @slow exclus "
                  f"(utiliser --slow ou --all pour les inclure)")

    # --- Phase 2 : smart-select par git diff ---
    if run_all:
        return

    if not run_smart:
        return

    changed = _get_changed_files()
    if not changed:
        return

    test_patterns = _resolve_tests(changed)
    if not test_patterns:
        return

    selected = []
    deselected = []
    for item in items:
        item_file = str(item.fspath).replace("\\", "/").split("/")[-1]
        item_class = item.cls.__name__ if item.cls else None

        match = False
        for pattern in test_patterns:
            if "::" in pattern:
                pat_file, pat_class = pattern.split("::", 1)
                if item_file == pat_file and item_class == pat_class:
                    match = True
                    break
            else:
                if item_file == pattern:
                    match = True
                    break

        if match:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected

    n_total = len(selected) + len(deselected)
    n_changed = len(changed)
    n_patterns = len(test_patterns)
    n_files = len({p.split("::")[0] for p in test_patterns})
    n_classes = sum(1 for p in test_patterns if "::" in p)
    print(f"\n[smart-test] {n_changed} fichiers modifies -> "
          f"{n_patterns} patterns ({n_files} fichiers, {n_classes} classes) "
          f"-> {len(selected)}/{n_total} tests")
    print(f"[smart-test] Patterns: {', '.join(sorted(test_patterns))}")
