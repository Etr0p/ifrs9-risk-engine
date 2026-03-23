"""Tests unitaires pour ifrs9_cockpit/config.py (Story 1-1).

Couvre les 10 tâches de la story :
  T1: SectorConfig 5 secteurs avec double canal
  T2: BaselConfig CRR3
  T3: RiskAppetiteConfig seuils tricolores
  T4: 5 scénarios macro prédéfinis
  T5: Règles d'incohérence macro
  T6: Contrats DataFrame
  T7: Features entreprise et DashboardConfig Steel Blue
  T8: Seuils distress PE et secondary discount
  T9: Validation au chargement (ValueError)
  T10: Validation standalone
"""

from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from ifrs9_cockpit.config import (
    ALLOWED_LOAN_TYPES,
    ALLOWED_SECTORS,
    BASEL_CONFIG,
    CHART_COLORS,
    CLIPPING_BOUNDS,
    CREDIT_CATEGORICAL_FEATURES,
    CREDIT_NUMERICAL_FEATURES,
    CRO_CONFIG,
    DASHBOARD_CONFIG,
    ENGINEERED_FEATURES,
    EAD_CONFIG,
    ECL_SCENARIOS,
    IFRS9_CONFIG,
    LGD_CONFIG,
    LOGIT_AMPLITUDE,
    MACRO_COVARIANCE,
    MACRO_HISTORY_BASELINE,
    MACRO_INCOHERENCE_RULES,
    MACRO_MEAN_REVERSION,
    MACRO_VARIABLES_ORDER,
    N_CLIENTS,
    N_MONTHS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    PD_CONFIG,
    PE_CATEGORICAL_FEATURES,
    PE_CLASSIFICATION_CONFIG,
    PE_DISTRESS_LOGIT_SCALE,
    PE_NUMERICAL_FEATURES,
    PREDEFINED_SCENARIOS,
    RANDOM_SEED,
    REQUIRED_CREDIT_COLS,
    REQUIRED_CREDIT_RESULT_COLS,
    REQUIRED_PE_COLS,
    REQUIRED_PE_RESULT_COLS,
    RISK_APPETITE_CONFIG,
    SCENARIO_ADVERSE,
    SCENARIO_BASE,
    SCENARIO_FAVORABLE,
    SECTORS,
    SECTOR_NAMES,
    SEGMENTS,
    SICR_CONFIG,
    STAGE_COLORS,
    PE_CATEGORY_COLORS,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    BaselConfig,
    DashboardConfig,
    MacroIncoherenceRule,
    MacroScenario,
    PEClassificationConfig,
    RiskAppetiteConfig,
    SectorConfig,
    SICRConfig,
    validate_config,
)


# ============================================================
# T1 — SectorConfig : 5 secteurs, double canal, PE params
# ============================================================

class TestSectorConfig:
    """Tests Task 1 — SectorConfig et SECTORS."""

    def test_5_sectors_defined(self):
        assert len(SECTORS) == 5

    def test_sector_names(self):
        expected = {"Technologie", "Industrie", "Sante", "Immobilier", "Services"}
        assert {s.name for s in SECTORS} == expected

    def test_proportions_sum_to_1(self):
        total = sum(s.proportion for s in SECTORS)
        assert abs(total - 1.0) < 1e-6

    def test_each_sector_has_dual_sensitivities(self):
        """Chaque secteur a 10 sensibilités macro (5 × 2 canaux)."""
        for s in SECTORS:
            # Canal crédit
            assert isinstance(s.unemployment_sensitivity_credit, float)
            assert isinstance(s.gdp_sensitivity_credit, float)
            assert isinstance(s.interest_rate_sensitivity_credit, float)
            assert isinstance(s.hpi_sensitivity_credit, float)
            assert isinstance(s.inflation_sensitivity_credit, float)
            # Canal PE
            assert isinstance(s.unemployment_sensitivity_pe, float)
            assert isinstance(s.gdp_sensitivity_pe, float)
            assert isinstance(s.interest_rate_sensitivity_pe, float)
            assert isinstance(s.hpi_sensitivity_pe, float)
            assert isinstance(s.inflation_sensitivity_pe, float)

    def test_technologie_vulnerability_pocket(self):
        """Technologie: unemployment_credit=2.5, unemployment_pe=-1.5 (asymétrie)."""
        tech = next(s for s in SECTORS if s.name == "Technologie")
        assert tech.unemployment_sensitivity_credit == 2.5
        assert tech.unemployment_sensitivity_pe == -1.5
        assert tech.base_default_rate == 0.025

    def test_pe_valuation_params(self):
        """Chaque secteur a valuation_method, entry_multiple_range, exit_multiple_base."""
        for s in SECTORS:
            assert s.valuation_method in {"EV/Revenue", "EV/EBITDA", "Cap_rate/NOI"}
            lo, hi = s.entry_multiple_range
            assert lo < hi
            assert s.exit_multiple_base > 0

    def test_rho_lgd_cycle_per_sector(self):
        """Chaque secteur a un rho_lgd_cycle (EBA GL/2019/03)."""
        expected = {
            "Technologie": 0.25,
            "Industrie": 0.30,
            "Sante": 0.10,
            "Immobilier": 0.35,
            "Services": 0.20,
        }
        for s in SECTORS:
            assert s.rho_lgd_cycle == expected[s.name], f"{s.name}: {s.rho_lgd_cycle}"

    def test_frozen_dataclass(self):
        """SectorConfig est frozen — immutable."""
        with pytest.raises(FrozenInstanceError):
            SECTORS[0].proportion = 0.99  # type: ignore[misc]

    def test_sector_names_tuple(self):
        assert SECTOR_NAMES == tuple(s.name for s in SECTORS)

    def test_segments_alias(self):
        """SEGMENTS est un alias de SECTORS pour compatibilité."""
        assert SEGMENTS is SECTORS


# ============================================================
# T2 — BaselConfig CRR3
# ============================================================

class TestBaselConfig:
    """Tests Task 2 — BaselConfig."""

    def test_cet1_target(self):
        assert BASEL_CONFIG.cet1_target == 0.13

    def test_rwa_budget(self):
        assert BASEL_CONFIG.rwa_budget == 3_690_000_000_000.0

    def test_rw_pe_options_crr3(self):
        assert BASEL_CONFIG.rw_pe_options == (190, 250, 400)

    def test_rw_pe_default_in_options(self):
        assert BASEL_CONFIG.rw_pe_default in BASEL_CONFIG.rw_pe_options

    def test_pe_max_allocation(self):
        assert BASEL_CONFIG.pe_max_allocation == 0.15

    def test_hhi_max(self):
        assert BASEL_CONFIG.hhi_max == 2500

    def test_cir_and_tax(self):
        """RAROC complet : CIR et taux d'imposition."""
        assert BASEL_CONFIG.cir == 0.45
        assert BASEL_CONFIG.tax_rate == 0.25

    def test_liquidity_premium(self):
        assert BASEL_CONFIG.liquidity_premium_bps == 50

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            BASEL_CONFIG.cet1_target = 0.15  # type: ignore[misc]


# ============================================================
# T3 — RiskAppetiteConfig seuils tricolores
# ============================================================

class TestRiskAppetiteConfig:
    """Tests Task 3 — RiskAppetiteConfig."""

    def test_ecl_ead_thresholds(self):
        assert RISK_APPETITE_CONFIG.ecl_ead_green == 0.020
        assert RISK_APPETITE_CONFIG.ecl_ead_amber == 0.040

    def test_raroc_thresholds(self):
        assert RISK_APPETITE_CONFIG.raroc_green == 0.04
        assert RISK_APPETITE_CONFIG.raroc_amber == 0.02

    def test_hhi_thresholds(self):
        assert RISK_APPETITE_CONFIG.hhi_green == 1500
        assert RISK_APPETITE_CONFIG.hhi_amber == 2500

    def test_nav_drawdown_thresholds(self):
        assert RISK_APPETITE_CONFIG.nav_drawdown_green == 0.10
        assert RISK_APPETITE_CONFIG.nav_drawdown_amber == 0.25

    def test_green_below_amber(self):
        """Vert < ambre pour les métriques croissantes (ECL, HHI, drawdown)."""
        assert RISK_APPETITE_CONFIG.ecl_ead_green < RISK_APPETITE_CONFIG.ecl_ead_amber
        assert RISK_APPETITE_CONFIG.hhi_green < RISK_APPETITE_CONFIG.hhi_amber
        assert RISK_APPETITE_CONFIG.nav_drawdown_green < RISK_APPETITE_CONFIG.nav_drawdown_amber

    def test_raroc_green_above_amber(self):
        """RAROC : vert > ambre (métrique décroissante = pire)."""
        assert RISK_APPETITE_CONFIG.raroc_green > RISK_APPETITE_CONFIG.raroc_amber


# ============================================================
# T4 — 5 scénarios macro prédéfinis
# ============================================================

class TestMacroScenarios:
    """Tests Task 4 — Scénarios macro."""

    def test_3_ecl_scenarios(self):
        assert len(ECL_SCENARIOS) == 3

    def test_ecl_weights_sum_to_1(self):
        total = sum(s.weight for s in ECL_SCENARIOS)
        assert abs(total - 1.0) < 1e-6

    def test_ecl_weights_50_25_25(self):
        assert SCENARIO_BASE.weight == 0.50
        assert SCENARIO_ADVERSE.weight == 0.25
        assert SCENARIO_FAVORABLE.weight == 0.25

    def test_predefined_scenarios_count(self):
        expected = {
            "Central", "Crise financiere (GFC)", "Crise souveraine (2012)",
            "Stagflation", "Choc pandemique (COVID)", "Rupture techno", "Reprise",
            "Hypercroissance", "Boom immobilier", "Trappe a liquidite",
            "Normalisation monetaire (Volcker)",
            "Transition climatique brutale",
        }
        assert set(PREDEFINED_SCENARIOS.keys()) == expected

    def test_predefined_scenario_keys(self):
        """Chaque scenario predefini a au moins les 5 cles slider de base."""
        required_keys = {"interest_rate_bp", "unemployment_bipolar", "gdp_pct", "hpi_pct", "inflation_pct"}
        optional_keys = {"carbon_price_shock", "physical_severity"}
        for name, params in PREDEFINED_SCENARIOS.items():
            assert required_keys <= set(params.keys()), f"Scenario {name}: cles manquantes"
            extra = set(params.keys()) - required_keys - optional_keys
            assert not extra, f"Scenario {name}: cles inattendues {extra}"

    def test_scenarios_alias(self):
        """ECL_SCENARIOS et SCENARIOS sont le même objet."""
        from ifrs9_cockpit.config import SCENARIOS
        assert SCENARIOS is ECL_SCENARIOS

    def test_macro_history_baseline_60_months(self):
        """RJ audit: 60 mois minimum pour estimation covariance 5x5."""
        for var, values in MACRO_HISTORY_BASELINE.items():
            assert len(values) == 60, f"{var}: {len(values)} mois"


# ============================================================
# T5 — Règles d'incohérence macro
# ============================================================

class TestMacroIncoherenceRules:
    """Tests Task 5 — MACRO_INCOHERENCE_RULES."""

    def test_4_rules_defined(self):
        assert len(MACRO_INCOHERENCE_RULES) == 4

    def test_rule_structure(self):
        for rule in MACRO_INCOHERENCE_RULES:
            assert isinstance(rule, MacroIncoherenceRule)
            assert rule.name
            assert rule.description
            assert len(rule.conditions) >= 2

    def test_rule_names(self):
        names = {r.name for r in MACRO_INCOHERENCE_RULES}
        assert "gdp_unemployment_crisis" in names
        assert "gdp_unemployment_tech" in names
        assert "deflation_rates" in names
        assert "hpi_gdp" in names


# ============================================================
# T6 — Contrats DataFrame
# ============================================================

class TestDataFrameContracts:
    """Tests Task 6 — REQUIRED_*_COLS."""

    def test_credit_cols_has_enterprise_id(self):
        assert "enterprise_id" in REQUIRED_CREDIT_COLS

    def test_credit_cols_has_sector(self):
        assert "sector" in REQUIRED_CREDIT_COLS

    def test_credit_cols_has_default_flag(self):
        assert "default_flag" in REQUIRED_CREDIT_COLS

    def test_pe_cols_has_valuation_method(self):
        assert "valuation_method" in REQUIRED_PE_COLS

    def test_pe_cols_has_entry_multiple(self):
        assert "entry_multiple" in REQUIRED_PE_COLS

    def test_credit_result_cols(self):
        assert "ecl_weighted" in REQUIRED_CREDIT_RESULT_COLS
        assert "stage" in REQUIRED_CREDIT_RESULT_COLS

    def test_pe_result_cols(self):
        assert "nav" in REQUIRED_PE_RESULT_COLS
        assert "risk_category" in REQUIRED_PE_RESULT_COLS

    def test_frozensets(self):
        assert isinstance(REQUIRED_CREDIT_COLS, frozenset)
        assert isinstance(REQUIRED_PE_COLS, frozenset)


# ============================================================
# T7 — Features entreprise et DashboardConfig
# ============================================================

class TestFeaturesAndDashboard:
    """Tests Task 7 — Features et DashboardConfig."""

    def test_credit_numerical_features(self):
        base_expected = {"revenue", "ebitda", "debt_ratio", "credit_score",
                    "dpd", "collateral", "loan_amount", "utilization_rate",
                    "loan_to_revenue", "collateral_coverage",
                    "supplier_hhi", "customer_count",
                    "esg_score", "bank_relationship_years",
                    "ebitda_margin", "interest_coverage_ratio",
                    "cf_volatility", "current_ratio",
                    "working_capital_ratio", "net_debt_to_ebitda",
                    "nb_incidents_12m", "account_age_months"}
        from ifrs9_cockpit.config import CORPORATE_INTERACTION_FEATURES
        expected = base_expected | set(CORPORATE_INTERACTION_FEATURES)
        assert set(CREDIT_NUMERICAL_FEATURES) == expected

    def test_credit_categorical_features(self):
        assert "sector" in CREDIT_CATEGORICAL_FEATURES
        assert "loan_type" in CREDIT_CATEGORICAL_FEATURES
        assert "company_size" in CREDIT_CATEGORICAL_FEATURES

    def test_pe_numerical_features(self):
        assert "entry_multiple" in PE_NUMERICAL_FEATURES
        assert "leverage" in PE_NUMERICAL_FEATURES

    def test_pe_categorical_features(self):
        assert "sector" in PE_CATEGORICAL_FEATURES
        assert "valuation_method" in PE_CATEGORICAL_FEATURES

    def test_numerical_features_alias(self):
        assert NUMERICAL_FEATURES is CREDIT_NUMERICAL_FEATURES

    def test_categorical_features_alias(self):
        assert CATEGORICAL_FEATURES is CREDIT_CATEGORICAL_FEATURES

    def test_steel_blue_palette(self):
        assert DASHBOARD_CONFIG.theme_primary == "#3B82F6"

    def test_dark_background(self):
        assert DASHBOARD_CONFIG.theme_bg_dark == "#0C1222"

    def test_slider_interest_rate_range(self):
        lo, hi, step = DASHBOARD_CONFIG.stress_interest_rate_range
        assert lo == -400.0
        assert hi == 650.0
        assert step == 25.0

    def test_slider_unemployment_bipolar(self):
        lo, hi, step = DASHBOARD_CONFIG.stress_unemployment_range
        assert lo == -7.0
        assert hi == 7.0

    def test_slider_gdp_range(self):
        lo, hi, _ = DASHBOARD_CONFIG.stress_gdp_range
        assert lo == -8.0
        assert hi == 8.0

    def test_slider_hpi_range(self):
        lo, hi, _ = DASHBOARD_CONFIG.stress_hpi_range
        assert lo == -30.0
        assert hi == 20.0

    def test_slider_inflation_range(self):
        lo, hi, _ = DASHBOARD_CONFIG.stress_inflation_range
        assert lo == -2.0
        assert hi == 8.0


# ============================================================
# T8 — Seuils distress PE et secondary discount
# ============================================================

class TestPEClassification:
    """Tests Task 8 — PEClassificationConfig."""

    def test_performing_threshold(self):
        assert PE_CLASSIFICATION_CONFIG.distress_threshold_performing == 0.10

    def test_watchlist_threshold(self):
        assert PE_CLASSIFICATION_CONFIG.distress_threshold_watchlist == 0.30

    def test_secondary_discount(self):
        assert PE_CLASSIFICATION_CONFIG.secondary_discount == 0.10

    def test_performing_below_watchlist(self):
        assert (PE_CLASSIFICATION_CONFIG.distress_threshold_performing
                < PE_CLASSIFICATION_CONFIG.distress_threshold_watchlist)

    def test_lgd_equity(self):
        assert PE_CLASSIFICATION_CONFIG.lgd_equity == 0.60


# ============================================================
# T9 — Validation au chargement (ValueError)
# ============================================================

class TestValidation:
    """Tests Task 9 — validate_config() et ValueError."""

    def test_validate_config_passes(self):
        """La config par défaut est valide."""
        validate_config()  # ne doit pas lever d'exception

    def test_validate_called_at_import(self):
        """validate_config() est appelé à l'import (le module charge sans erreur)."""
        import ifrs9_cockpit.config  # noqa: F401 — aucune exception

    def test_seed(self):
        assert RANDOM_SEED == 123

    def test_dataset_params_preserved(self):
        """Les paramètres dataset existants sont conservés."""
        assert N_CLIENTS == 30_000
        assert N_MONTHS == 12
        assert TRAIN_RATIO == 0.7
        assert VALIDATION_RATIO == 0.15
        assert TEST_RATIO == 0.15

    def test_invalid_proportions_raises(self):
        """Proportions != 1 lève ValueError."""
        import ifrs9_cockpit.config as cfg
        original = cfg.SECTORS[:]
        try:
            # Remplacer dernier secteur par un avec proportion cassée
            bad = SectorConfig(
                name="Services", proportion=0.99,
                base_default_rate=0.04,
                unemployment_sensitivity_credit=1.2, gdp_sensitivity_credit=1.0,
                interest_rate_sensitivity_credit=0.8, hpi_sensitivity_credit=0.5,
                inflation_sensitivity_credit=1.8,
                unemployment_sensitivity_pe=1.0, gdp_sensitivity_pe=1.2,
                interest_rate_sensitivity_pe=1.0, hpi_sensitivity_pe=0.3,
                inflation_sensitivity_pe=2.0,
                valuation_method="EV/EBITDA",
                entry_multiple_range=(6.0, 10.0), exit_multiple_base=8.0,
                rho_lgd_cycle=0.20,
                green_share=0.25,
                revenue_range_m=(3.0, 150.0), ebitda_margin_range=(0.08, 0.18),
            )
            cfg.SECTORS[-1] = bad
            with pytest.raises(ValueError, match="proportions"):
                validate_config()
        finally:
            cfg.SECTORS[:] = original

    def test_invalid_ecl_weights_raises(self):
        """Poids ECL != 1 lève ValueError."""
        import ifrs9_cockpit.config as cfg
        original = cfg.ECL_SCENARIOS[:]
        try:
            bad = MacroScenario(
                name="Base", weight=0.99,
                gdp_growth=1.2, unemployment_rate=7.5, interest_rate=3.5,
                hpi_growth=2.0, inflation_rate=2.5,
                gdp_shock=0.0, unemployment_shock=0.0,
                interest_rate_shock=0.0, hpi_shock=0.0, inflation_shock=0.0,
            )
            cfg.ECL_SCENARIOS[0] = bad
            with pytest.raises(ValueError, match="ponderations"):
                validate_config()
        finally:
            cfg.ECL_SCENARIOS[:] = original

    def test_invalid_default_rate_raises(self):
        """Taux de défaut hors bornes lève ValueError."""
        import ifrs9_cockpit.config as cfg
        original = cfg.SECTORS[:]
        try:
            bad = SectorConfig(
                name="Services", proportion=0.15,
                base_default_rate=1.5,  # > 1 invalide
                unemployment_sensitivity_credit=1.2, gdp_sensitivity_credit=1.0,
                interest_rate_sensitivity_credit=0.8, hpi_sensitivity_credit=0.5,
                inflation_sensitivity_credit=1.8,
                unemployment_sensitivity_pe=1.0, gdp_sensitivity_pe=1.2,
                interest_rate_sensitivity_pe=1.0, hpi_sensitivity_pe=0.3,
                inflation_sensitivity_pe=2.0,
                valuation_method="EV/EBITDA",
                entry_multiple_range=(6.0, 10.0), exit_multiple_base=8.0,
                rho_lgd_cycle=0.20,
                green_share=0.25,
                revenue_range_m=(3.0, 150.0), ebitda_margin_range=(0.08, 0.18),
            )
            cfg.SECTORS[-1] = bad
            with pytest.raises(ValueError, match="defaut"):
                validate_config()
        finally:
            cfg.SECTORS[:] = original


# ============================================================
# T10 — Validation standalone
# ============================================================

class TestStandalone:
    """Tests Task 10 — Bloc __main__."""

    def test_standalone_runs_successfully(self):
        """python -m ifrs9_cockpit.config retourne 0."""
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "ifrs9_cockpit.config"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_standalone_output_contains_key_info(self):
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "ifrs9_cockpit.config"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
        )
        assert "5 Secteurs" in result.stdout
        assert "Configuration valide" in result.stdout
        assert "Basel III" in result.stdout
        assert "Risk Appetite" in result.stdout


# ============================================================
# Corrections rigueur mathématique — structures additionnelles
# ============================================================

class TestMathRigorStructures:
    """Tests pour les structures ajoutées par l'audit de rigueur."""

    def test_sicr_config_weights_sum_to_1(self):
        total = (SICR_CONFIG.w_pd_ratio + SICR_CONFIG.w_pd_delta
                 + SICR_CONFIG.w_dpd + SICR_CONFIG.w_macro)
        assert abs(total - 1.0) < 1e-6

    def test_sicr_config_values(self):
        assert SICR_CONFIG.w_pd_ratio == 0.25
        assert SICR_CONFIG.w_pd_delta == 0.25
        assert SICR_CONFIG.w_dpd == 0.20
        assert SICR_CONFIG.w_macro == 0.30
        assert SICR_CONFIG.threshold == 1.65

    def test_macro_mean_reversion_5_variables(self):
        assert len(MACRO_MEAN_REVERSION) == 5
        for var, params in MACRO_MEAN_REVERSION.items():
            assert "kappa" in params
            assert "sigma" in params
            assert "theta" in params
            assert params["kappa"] > 0

    def test_macro_covariance_5x5(self):
        assert len(MACRO_COVARIANCE) == 5
        for row in MACRO_COVARIANCE:
            assert len(row) == 5

    def test_macro_variables_order(self):
        assert len(MACRO_VARIABLES_ORDER) == 5
        assert "unemployment_rate" in MACRO_VARIABLES_ORDER
        assert "gdp_growth" in MACRO_VARIABLES_ORDER

    def test_logit_amplitude(self):
        """RJ audit: LOGIT_AMPLITUDE calibre dynamiquement, ~4.94 pour PD 6%→18%."""
        assert 4.5 < LOGIT_AMPLITUDE < 5.5  # calibre depuis _calibrate_logit_amplitude()

    def test_pe_distress_logit_scale(self):
        assert PE_DISTRESS_LOGIT_SCALE == 3.0

    def test_chart_colors(self):
        assert len(CHART_COLORS) >= 6
        assert CHART_COLORS[0] == "#3B82F6"  # Steel Blue primary

    def test_stage_colors(self):
        assert set(STAGE_COLORS.keys()) == {1, 2, 3}

    def test_pe_category_colors(self):
        assert set(PE_CATEGORY_COLORS.keys()) == {"Performing", "Watchlist", "Distressed"}


# ============================================================
# Contrat de donnees (Enums, Clipping, Features Engineered)
# ============================================================

class TestDataContract:
    """Tests pour ALLOWED_SECTORS, ALLOWED_LOAN_TYPES, CLIPPING_BOUNDS, ENGINEERED_FEATURES."""

    def test_allowed_sectors_matches_config(self):
        expected = {"Technologie", "Industrie", "Sante", "Immobilier", "Services"}
        assert ALLOWED_SECTORS == expected

    def test_allowed_sectors_is_frozenset(self):
        assert isinstance(ALLOWED_SECTORS, frozenset)

    def test_allowed_loan_types(self):
        assert ALLOWED_LOAN_TYPES == frozenset({"Revolving", "Term"})

    def test_allowed_loan_types_is_frozenset(self):
        assert isinstance(ALLOWED_LOAN_TYPES, frozenset)

    def test_clipping_bounds_keys(self):
        assert set(CLIPPING_BOUNDS.keys()) == {"debt_ratio", "credit_score", "utilization_rate"}

    def test_clipping_bounds_debt_ratio(self):
        lo, hi = CLIPPING_BOUNDS["debt_ratio"]
        assert lo == 0.0
        assert hi == 1.5

    def test_clipping_bounds_credit_score(self):
        lo, hi = CLIPPING_BOUNDS["credit_score"]
        assert lo == 300.0
        assert hi == 850.0

    def test_clipping_bounds_utilization_rate(self):
        lo, hi = CLIPPING_BOUNDS["utilization_rate"]
        assert lo == 0.0
        assert hi == 1.2

    def test_engineered_features(self):
        from ifrs9_cockpit.config import CORPORATE_INTERACTION_FEATURES
        expected = ["loan_to_revenue", "collateral_coverage"] + CORPORATE_INTERACTION_FEATURES
        assert ENGINEERED_FEATURES == expected

    def test_engineered_features_in_numerical(self):
        for feat in ENGINEERED_FEATURES:
            assert feat in CREDIT_NUMERICAL_FEATURES, f"{feat} absent de CREDIT_NUMERICAL_FEATURES"
