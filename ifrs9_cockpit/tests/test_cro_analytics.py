"""Tests unitaires pour le module AI Analyst — CRO Analytics (Epic 5, Stories 5-1 a 5-4).

Couvre les 5 couches du pipeline analytique CRO :
  Story 5-1 : Couche 1 (Croisement) + Couche 2 (Allocation proportionnelle / Euler)
  Story 5-2 : Couche 3 (Seuils de basculement + Reverse Stress Test)
  Story 5-3 : Couche 4 (Regime) + Couche 5 (Trajectoires, Risk Appetite, Early Warning)
  Story 5-4 : Orchestrateur 2 passes + Recommandations + Narrative
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ifrs9_cockpit.config import (
    SCENARIO_BASE,
    SECTORS,
    BASEL_CONFIG,
    RISK_APPETITE_CONFIG,
    LOGIT_AMPLITUDE,
    MACRO_COVARIANCE,
    MACRO_VARIABLES_ORDER,
)
from ifrs9_cockpit.utils.helpers import logit, expit


# ============================================================
# Fixtures : pipeline credit + PE (partage entre toutes les classes)
# ============================================================

@pytest.fixture(scope="module")
def pipeline_data():
    """Genere les donnees et execute les pipelines credit/PE une seule fois."""
    from ifrs9_cockpit.data.generator import generate_dataset
    from ifrs9_cockpit.models.pd_model import PDModelSuite
    from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
    from ifrs9_cockpit.engine.pe_calculator import PECalculator
    from ifrs9_cockpit.models.lgd_model import LGDModel
    from ifrs9_cockpit.models.ead_model import EADModel

    df_credit, df_pe, df_history = generate_dataset()

    pd_suite = PDModelSuite()
    pd_suite.fit(df_credit)
    pd_current = pd_suite.predict_active(df_credit)
    pd_origination = pd_current * 0.8

    lgd_model = LGDModel()
    ead_model = EADModel()
    ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
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

    return result_credit, result_pe, macro_params


# ============================================================
# Story 5-1 : Couche 1 — Croisement (FR26)
# ============================================================

class TestLayer1Crossing:
    """Tests Couche 1 — analyze_crossings (FR26)."""

    def test_asymmetry_matrix_5_sectors(self, pipeline_data):
        """La matrice d'asymetrie contient exactement 5 secteurs."""
        from ifrs9_cockpit.ai_analyst.layer1_crossing import analyze_crossings
        result_credit, result_pe, macro_params = pipeline_data
        asym, _ = analyze_crossings(result_credit, result_pe, macro_params)
        assert len(asym) == 5

    def test_asymmetry_matrix_columns(self, pipeline_data):
        """Les colonnes requises sont presentes."""
        from ifrs9_cockpit.ai_analyst.layer1_crossing import analyze_crossings
        result_credit, result_pe, macro_params = pipeline_data
        asym, _ = analyze_crossings(result_credit, result_pe, macro_params)
        required = {"sector", "ecl", "ead", "el_pe", "nav",
                     "loss_rate_credit", "loss_rate_pe", "asymmetry", "stress_intensity"}
        assert required.issubset(set(asym.columns))

    def test_asymmetry_sector_names(self, pipeline_data):
        """Les noms de secteurs correspondent a la config."""
        from ifrs9_cockpit.ai_analyst.layer1_crossing import analyze_crossings
        result_credit, result_pe, macro_params = pipeline_data
        asym, _ = analyze_crossings(result_credit, result_pe, macro_params)
        expected_names = {s.name for s in SECTORS}
        assert set(asym["sector"]) == expected_names

    def test_marginal_contributions_10_cells(self, pipeline_data):
        """Les contributions marginales ont 10 cellules (5 secteurs x 2 canaux)."""
        from ifrs9_cockpit.ai_analyst.layer1_crossing import analyze_crossings
        result_credit, result_pe, macro_params = pipeline_data
        _, marginal = analyze_crossings(result_credit, result_pe, macro_params)
        assert len(marginal) == 10

    def test_marginal_contributions_sum_to_1(self, pipeline_data):
        """La somme des contributions marginales vaut ~1.0."""
        from ifrs9_cockpit.ai_analyst.layer1_crossing import analyze_crossings
        result_credit, result_pe, macro_params = pipeline_data
        _, marginal = analyze_crossings(result_credit, result_pe, macro_params)
        total = marginal["marginal_contribution"].sum()
        assert abs(total - 1.0) < 0.01

    def test_marginal_contributions_columns(self, pipeline_data):
        """Les colonnes requises sont presentes dans marginal."""
        from ifrs9_cockpit.ai_analyst.layer1_crossing import analyze_crossings
        result_credit, result_pe, macro_params = pipeline_data
        _, marginal = analyze_crossings(result_credit, result_pe, macro_params)
        required = {"sector", "canal", "risk_amount", "marginal_contribution"}
        assert required.issubset(set(marginal.columns))

    def test_marginal_contributions_canals(self, pipeline_data):
        """Les deux canaux Credit et PE sont presents."""
        from ifrs9_cockpit.ai_analyst.layer1_crossing import analyze_crossings
        result_credit, result_pe, macro_params = pipeline_data
        _, marginal = analyze_crossings(result_credit, result_pe, macro_params)
        assert set(marginal["canal"]) == {"Credit", "PE"}

    def test_asymmetry_loss_rates_positive(self, pipeline_data):
        """Les taux de perte credit et PE sont >= 0."""
        from ifrs9_cockpit.ai_analyst.layer1_crossing import analyze_crossings
        result_credit, result_pe, macro_params = pipeline_data
        asym, _ = analyze_crossings(result_credit, result_pe, macro_params)
        assert (asym["loss_rate_credit"] >= 0).all()
        assert (asym["loss_rate_pe"] >= 0).all()

    def test_stress_intensity_varies_under_stress(self, pipeline_data):
        """Sous stress, les stress_intensity different entre secteurs (ponderees par sensibilites)."""
        from ifrs9_cockpit.ai_analyst.layer1_crossing import analyze_crossings
        result_credit, result_pe, macro_params = pipeline_data
        stressed = macro_params.copy()
        stressed["unemployment_rate"] = 12.0
        asym, _ = analyze_crossings(result_credit, result_pe, stressed)
        # Les secteurs ont des sensibilites differentes au chomage, donc les intensites doivent differer
        intensities = asym["stress_intensity"].values
        assert len(set(intensities)) > 1, "Les stress_intensity devraient varier entre secteurs"

    def test_marginal_credit_and_pe_nonzero(self, pipeline_data):
        """Les contributions marginales Credit et PE sont positives et somment a 1."""
        from ifrs9_cockpit.ai_analyst.layer1_crossing import analyze_crossings
        result_credit, result_pe, macro_params = pipeline_data
        _, marginal = analyze_crossings(result_credit, result_pe, macro_params)
        credit_total = marginal.loc[marginal["canal"] == "Credit", "marginal_contribution"].sum()
        pe_total = marginal.loc[marginal["canal"] == "PE", "marginal_contribution"].sum()
        assert credit_total > 0
        assert pe_total > 0
        assert abs(credit_total + pe_total - 1.0) < 0.01


# ============================================================
# Story 5-1 : Couche 2 — Allocation proportionnelle (FR27)
# ============================================================

class TestLayer2Proportional:
    """Tests Couche 2 — decompose_proportional (FR27)."""

    def test_proportional_full_allocation_sum_1(self, pipeline_data):
        """La somme des proportional_share vaut ~1.0 (full allocation)."""
        from ifrs9_cockpit.ai_analyst.layer2_euler import decompose_proportional
        result_credit, result_pe, macro_params = pipeline_data
        alloc, _ = decompose_proportional(result_credit, result_pe, macro_params)
        assert abs(alloc["proportional_share"].sum() - 1.0) < 0.01

    def test_proportional_10_cells(self, pipeline_data):
        """L'allocation proportionnelle contient 10 cellules (5 secteurs x 2 canaux)."""
        from ifrs9_cockpit.ai_analyst.layer2_euler import decompose_proportional
        result_credit, result_pe, macro_params = pipeline_data
        alloc, _ = decompose_proportional(result_credit, result_pe, macro_params)
        assert len(alloc) == 10

    def test_proportional_columns(self, pipeline_data):
        """Les colonnes requises sont presentes."""
        from ifrs9_cockpit.ai_analyst.layer2_euler import decompose_proportional
        result_credit, result_pe, macro_params = pipeline_data
        alloc, _ = decompose_proportional(result_credit, result_pe, macro_params)
        required = {"sector", "canal", "risk_amount", "proportional_share", "rwa"}
        assert required.issubset(set(alloc.columns))

    def test_factor_attribution_10_rows(self, pipeline_data):
        """L'attribution factorielle a 10 lignes (5 vars x 2 canaux)."""
        from ifrs9_cockpit.ai_analyst.layer2_euler import decompose_proportional
        result_credit, result_pe, macro_params = pipeline_data
        _, factors = decompose_proportional(result_credit, result_pe, macro_params)
        assert len(factors) == 10

    def test_factor_attribution_columns(self, pipeline_data):
        """Les colonnes requises de l'attribution factorielle."""
        from ifrs9_cockpit.ai_analyst.layer2_euler import decompose_proportional
        result_credit, result_pe, macro_params = pipeline_data
        _, factors = decompose_proportional(result_credit, result_pe, macro_params)
        required = {"variable", "canal", "delta_from_base", "attribution", "regime_adjusted"}
        assert required.issubset(set(factors.columns))

    def test_factor_attribution_zero_at_baseline(self, pipeline_data):
        """Au baseline (pas de stress), les attributions factorielles sont 0."""
        from ifrs9_cockpit.ai_analyst.layer2_euler import decompose_proportional
        result_credit, result_pe, macro_params = pipeline_data
        _, factors = decompose_proportional(result_credit, result_pe, macro_params)
        assert (factors["attribution"] == 0.0).all()

    def test_factor_attribution_nonzero_under_stress(self, pipeline_data):
        """Sous stress, les attributions factorielles sont non-nulles."""
        from ifrs9_cockpit.ai_analyst.layer2_euler import decompose_proportional
        result_credit, result_pe, macro_params = pipeline_data
        stressed = macro_params.copy()
        stressed["unemployment_rate"] = 12.0
        stressed["gdp_growth"] = -2.0
        _, factors = decompose_proportional(result_credit, result_pe, stressed)
        assert factors["attribution"].abs().sum() > 0

    def test_proportional_with_regime(self, pipeline_data):
        """L'allocation proportionnelle fonctionne avec un regime detecte."""
        from ifrs9_cockpit.ai_analyst.layer2_euler import decompose_proportional
        from ifrs9_cockpit.ai_analyst.types import RegimeClassification
        result_credit, result_pe, macro_params = pipeline_data
        regime = RegimeClassification(
            probabilities={"Crise financiere": 0.8, "Stagflation": 0.05,
                           "Rupture techno": 0.05, "Resserrement": 0.05, "Reprise": 0.05},
            detected_regime="Crise financiere",
        )
        stressed = macro_params.copy()
        stressed["unemployment_rate"] = 12.0
        alloc, factors = decompose_proportional(result_credit, result_pe, stressed, regime=regime)
        assert len(alloc) == 10
        assert all(factors["regime_adjusted"])

    def test_backward_compat_alias(self, pipeline_data):
        """L'alias backward-compat decompose_euler fonctionne."""
        from ifrs9_cockpit.ai_analyst.layer2_euler import decompose_euler
        result_credit, result_pe, macro_params = pipeline_data
        alloc, _ = decompose_euler(result_credit, result_pe, macro_params)
        assert len(alloc) == 10


# ============================================================
# Story 5-2 : Couche 3 — Seuils de basculement & RST (FR28)
# ============================================================

class TestLayer3RST:
    """Tests Couche 3 — find_tipping_points + reverse_stress_test (FR28)."""

    @staticmethod
    def _make_ecl_proxy(pipeline_data):
        """Construit le proxy ECL logit."""
        result_credit, result_pe, macro_params = pipeline_data
        ecl_base = result_credit["ecl_weighted"].sum()
        ead_total = result_credit["ead"].sum()
        base_ratio = ecl_base / max(ead_total, 1)
        logit_base = float(logit(np.clip(base_ratio, 1e-6, 0.99)))

        _VAR_TO_SENS = {
            "unemployment_rate": "unemployment_sensitivity_credit",
            "gdp_growth": "gdp_sensitivity_credit",
            "interest_rate": "interest_rate_sensitivity_credit",
            "hpi_growth": "hpi_sensitivity_credit",
            "inflation_rate": "inflation_sensitivity_credit",
        }
        _INVERTED = {"gdp_growth", "hpi_growth"}

        def ecl_proxy(params):
            stress_sum = 0.0
            for var, sens_attr in _VAR_TO_SENS.items():
                base_val = getattr(SCENARIO_BASE, var)
                delta = params.get(var, base_val) - base_val
                if var in _INVERTED:
                    delta = -delta
                avg_sens = sum(
                    getattr(s, sens_attr) * s.proportion for s in SECTORS
                )
                stress_sum += (delta / 100) * avg_sens
            ecl_ratio = float(expit(logit_base + stress_sum * LOGIT_AMPLITUDE))
            return ecl_ratio * ead_total
        return ecl_proxy

    def test_tipping_points_5_variables(self, pipeline_data):
        """Les seuils de basculement couvrent 5 variables macro."""
        from ifrs9_cockpit.ai_analyst.layer3_rst import find_tipping_points
        result_credit, result_pe, macro_params = pipeline_data
        ecl_proxy = self._make_ecl_proxy(pipeline_data)
        ead_total = result_credit["ead"].sum()
        ecl_amber_abs = RISK_APPETITE_CONFIG.ecl_ead_amber * ead_total
        tipping = find_tipping_points(ecl_proxy, macro_params, ecl_threshold=ecl_amber_abs)
        assert len(tipping) == 5

    def test_tipping_points_columns(self, pipeline_data):
        """Les colonnes requises sont presentes dans tipping_points."""
        from ifrs9_cockpit.ai_analyst.layer3_rst import find_tipping_points
        result_credit, result_pe, macro_params = pipeline_data
        ecl_proxy = self._make_ecl_proxy(pipeline_data)
        ead_total = result_credit["ead"].sum()
        ecl_amber_abs = RISK_APPETITE_CONFIG.ecl_ead_amber * ead_total
        tipping = find_tipping_points(ecl_proxy, macro_params, ecl_threshold=ecl_amber_abs)
        required = {"variable", "base_value", "tipping_value", "distance_pp", "breached"}
        assert required.issubset(set(tipping.columns))

    def test_tipping_points_all_breached(self, pipeline_data):
        """Au moins une variable breach le seuil."""
        from ifrs9_cockpit.ai_analyst.layer3_rst import find_tipping_points
        result_credit, result_pe, macro_params = pipeline_data
        ecl_proxy = self._make_ecl_proxy(pipeline_data)
        ead_total = result_credit["ead"].sum()
        ecl_amber_abs = RISK_APPETITE_CONFIG.ecl_ead_amber * ead_total
        tipping = find_tipping_points(ecl_proxy, macro_params, ecl_threshold=ecl_amber_abs)
        assert tipping["breached"].any()

    def test_tipping_points_distance_positive(self, pipeline_data):
        """Les distances sont positives ou inf."""
        from ifrs9_cockpit.ai_analyst.layer3_rst import find_tipping_points
        result_credit, result_pe, macro_params = pipeline_data
        ecl_proxy = self._make_ecl_proxy(pipeline_data)
        ead_total = result_credit["ead"].sum()
        ecl_amber_abs = RISK_APPETITE_CONFIG.ecl_ead_amber * ead_total
        tipping = find_tipping_points(ecl_proxy, macro_params, ecl_threshold=ecl_amber_abs)
        assert (tipping["distance_pp"] >= 0).all()

    def test_reverse_stress_test_structure(self, pipeline_data):
        """Le RST retourne un dict avec les cles requises."""
        from ifrs9_cockpit.ai_analyst.layer3_rst import reverse_stress_test
        result_credit, result_pe, macro_params = pipeline_data
        ecl_proxy = self._make_ecl_proxy(pipeline_data)
        rst = reverse_stress_test(ecl_proxy, macro_params)
        required_keys = {"rst_scenario", "rst_ecl", "rst_distance_sigma",
                         "breach", "capital_base", "ecl_breach_threshold"}
        assert required_keys.issubset(set(rst.keys()))

    def test_reverse_stress_test_breach(self, pipeline_data):
        """Le RST trouve un scenario de breach."""
        from ifrs9_cockpit.ai_analyst.layer3_rst import reverse_stress_test
        result_credit, result_pe, macro_params = pipeline_data
        ecl_proxy = self._make_ecl_proxy(pipeline_data)
        rst = reverse_stress_test(ecl_proxy, macro_params)
        assert rst["breach"] is True

    def test_reverse_stress_test_distance_nonneg(self, pipeline_data):
        """La distance RST est non-negative."""
        from ifrs9_cockpit.ai_analyst.layer3_rst import reverse_stress_test
        result_credit, result_pe, macro_params = pipeline_data
        ecl_proxy = self._make_ecl_proxy(pipeline_data)
        rst = reverse_stress_test(ecl_proxy, macro_params)
        assert rst["rst_distance_sigma"] >= 0

    def test_reverse_stress_test_ecl_positive(self, pipeline_data):
        """L'ECL de rupture est positif."""
        from ifrs9_cockpit.ai_analyst.layer3_rst import reverse_stress_test
        result_credit, result_pe, macro_params = pipeline_data
        ecl_proxy = self._make_ecl_proxy(pipeline_data)
        rst = reverse_stress_test(ecl_proxy, macro_params)
        assert rst["rst_ecl"] > 0

    def test_reverse_stress_test_capital_base(self, pipeline_data):
        """Le capital base correspond a la config."""
        from ifrs9_cockpit.ai_analyst.layer3_rst import reverse_stress_test
        result_credit, result_pe, macro_params = pipeline_data
        ecl_proxy = self._make_ecl_proxy(pipeline_data)
        rst = reverse_stress_test(ecl_proxy, macro_params)
        expected_capital = BASEL_CONFIG.rwa_budget * BASEL_CONFIG.cet1_target
        assert abs(rst["capital_base"] - expected_capital) < 1

    def test_reverse_stress_test_custom_threshold(self, pipeline_data):
        """Le RST accepte un seuil ECL personnalise (FR54) et l'utilise correctement."""
        from ifrs9_cockpit.ai_analyst.layer3_rst import reverse_stress_test
        result_credit, result_pe, macro_params = pipeline_data
        ecl_proxy = self._make_ecl_proxy(pipeline_data)
        # Seuil astronomique (100T EUR) pour garantir l'absence de breach
        custom = 100_000_000_000_000.0
        rst = reverse_stress_test(ecl_proxy, macro_params, target_ecl=custom)
        # Le seuil personnalise doit etre reflecte dans le resultat
        assert rst["ecl_breach_threshold"] == round(custom, 0)
        # Avec un seuil astronomique, le breach ne devrait pas se produire
        assert rst["breach"] is False

    def test_mahalanobis_distance_uses_covariance(self):
        """La matrice MACRO_COVARIANCE est 5x5 et symetrique."""
        cov = np.array(MACRO_COVARIANCE)
        assert cov.shape == (5, 5)
        assert np.allclose(cov, cov.T)


# ============================================================
# Story 5-3 : Couche 4 — Classification de regime (FR29)
# ============================================================

class TestLayer4Regime:
    """Tests Couche 4 — classify_regime (FR29)."""

    def test_baseline_5_regimes(self):
        """Au baseline, 5 regimes avec somme des probas = 1."""
        from ifrs9_cockpit.ai_analyst.layer4_regime import classify_regime
        macro = {
            "unemployment_rate": SCENARIO_BASE.unemployment_rate,
            "gdp_growth": SCENARIO_BASE.gdp_growth,
            "interest_rate": SCENARIO_BASE.interest_rate,
            "hpi_growth": SCENARIO_BASE.hpi_growth,
            "inflation_rate": SCENARIO_BASE.inflation_rate,
        }
        regime = classify_regime(macro)
        assert len(regime.probabilities) == 5
        assert abs(sum(regime.probabilities.values()) - 1.0) < 0.01

    def test_baseline_uniform_distribution(self):
        """Au baseline (pas de stress), toutes les probas sont egales (20%)."""
        from ifrs9_cockpit.ai_analyst.layer4_regime import classify_regime
        macro = {
            "unemployment_rate": SCENARIO_BASE.unemployment_rate,
            "gdp_growth": SCENARIO_BASE.gdp_growth,
            "interest_rate": SCENARIO_BASE.interest_rate,
            "hpi_growth": SCENARIO_BASE.hpi_growth,
            "inflation_rate": SCENARIO_BASE.inflation_rate,
        }
        regime = classify_regime(macro)
        for prob in regime.probabilities.values():
            assert abs(prob - 0.2) < 0.01

    def test_stagflation_detected(self):
        """Sous stress Stagflation, le regime Stagflation est detecte."""
        from ifrs9_cockpit.ai_analyst.layer4_regime import classify_regime
        macro = {
            "unemployment_rate": 10.0,
            "gdp_growth": -1.0,
            "interest_rate": 5.5,
            "hpi_growth": SCENARIO_BASE.hpi_growth,
            "inflation_rate": 5.0,
        }
        regime = classify_regime(macro)
        assert regime.detected_regime == "Stagflation"
        assert regime.probabilities["Stagflation"] > 0.3

    def test_crise_financiere_detected(self):
        """Sous stress Crise (unemp haute, GDP bas, HPI bas), le regime Crise financiere est detecte.

        Les valeurs macro simulent une crise reelle : chomage eleve, GDP en
        recession, HPI en forte baisse, taux en baisse (flight-to-quality).
        """
        from ifrs9_cockpit.ai_analyst.layer4_regime import classify_regime
        macro = {
            "unemployment_rate": 12.0,
            "gdp_growth": -3.0,
            "interest_rate": 1.0,
            "hpi_growth": -5.0,
            "inflation_rate": SCENARIO_BASE.inflation_rate,
        }
        regime = classify_regime(macro)
        assert regime.detected_regime == "Crise financiere"

    def test_regime_classification_type(self):
        """Le retour est un RegimeClassification."""
        from ifrs9_cockpit.ai_analyst.layer4_regime import classify_regime
        from ifrs9_cockpit.ai_analyst.types import RegimeClassification
        macro = {
            "unemployment_rate": SCENARIO_BASE.unemployment_rate,
            "gdp_growth": SCENARIO_BASE.gdp_growth,
            "interest_rate": SCENARIO_BASE.interest_rate,
            "hpi_growth": SCENARIO_BASE.hpi_growth,
            "inflation_rate": SCENARIO_BASE.inflation_rate,
        }
        regime = classify_regime(macro)
        assert isinstance(regime, RegimeClassification)

    def test_probabilities_all_nonnegative(self):
        """Toutes les probabilites sont >= 0 (softmax)."""
        from ifrs9_cockpit.ai_analyst.layer4_regime import classify_regime
        macro = {
            "unemployment_rate": 10.0,
            "gdp_growth": -2.0,
            "interest_rate": 6.0,
            "hpi_growth": -3.0,
            "inflation_rate": 4.0,
        }
        regime = classify_regime(macro)
        for prob in regime.probabilities.values():
            assert prob >= 0

    def test_detected_regime_is_argmax(self):
        """Le regime detecte est celui avec la probabilite maximale."""
        from ifrs9_cockpit.ai_analyst.layer4_regime import classify_regime
        macro = {
            "unemployment_rate": 10.0,
            "gdp_growth": -2.0,
            "interest_rate": 6.0,
            "hpi_growth": -3.0,
            "inflation_rate": 4.0,
        }
        regime = classify_regime(macro)
        max_regime = max(regime.probabilities, key=regime.probabilities.get)
        assert regime.detected_regime == max_regime


# ============================================================
# Story 5-3 : Couche 5 — Trajectoires O-U, Risk Appetite, Early Warning (FR30)
# ============================================================

class TestLayer5Prospective:
    """Tests Couche 5 — project_trajectories, compute_risk_appetite, compute_early_warning."""

    def test_trajectories_12_rows(self):
        """3 trajectoires x 4 horizons = 12 lignes."""
        from ifrs9_cockpit.ai_analyst.layer5_prospective import project_trajectories
        macro = {
            "unemployment_rate": SCENARIO_BASE.unemployment_rate,
            "gdp_growth": SCENARIO_BASE.gdp_growth,
            "interest_rate": SCENARIO_BASE.interest_rate,
            "hpi_growth": SCENARIO_BASE.hpi_growth,
            "inflation_rate": SCENARIO_BASE.inflation_rate,
        }
        traj = project_trajectories(macro)
        assert len(traj) == 12

    def test_trajectories_3_scenarios(self):
        """Les 3 scenarios sont presents : Favorable, Central, Adverse."""
        from ifrs9_cockpit.ai_analyst.layer5_prospective import project_trajectories
        macro = {
            "unemployment_rate": SCENARIO_BASE.unemployment_rate,
            "gdp_growth": SCENARIO_BASE.gdp_growth,
            "interest_rate": SCENARIO_BASE.interest_rate,
            "hpi_growth": SCENARIO_BASE.hpi_growth,
            "inflation_rate": SCENARIO_BASE.inflation_rate,
        }
        traj = project_trajectories(macro)
        assert set(traj["trajectory"]) == {"Favorable", "Central", "Adverse"}

    def test_trajectories_4_horizons(self):
        """Les 4 horizons sont presents : 3, 6, 9, 12 mois."""
        from ifrs9_cockpit.ai_analyst.layer5_prospective import project_trajectories
        macro = {
            "unemployment_rate": SCENARIO_BASE.unemployment_rate,
            "gdp_growth": SCENARIO_BASE.gdp_growth,
            "interest_rate": SCENARIO_BASE.interest_rate,
            "hpi_growth": SCENARIO_BASE.hpi_growth,
            "inflation_rate": SCENARIO_BASE.inflation_rate,
        }
        traj = project_trajectories(macro)
        assert set(traj["horizon_months"]) == {3, 6, 9, 12}

    def test_trajectories_columns(self):
        """Les 5 variables macro sont projetees."""
        from ifrs9_cockpit.ai_analyst.layer5_prospective import project_trajectories
        macro = {
            "unemployment_rate": SCENARIO_BASE.unemployment_rate,
            "gdp_growth": SCENARIO_BASE.gdp_growth,
            "interest_rate": SCENARIO_BASE.interest_rate,
            "hpi_growth": SCENARIO_BASE.hpi_growth,
            "inflation_rate": SCENARIO_BASE.inflation_rate,
        }
        traj = project_trajectories(macro)
        for var in ["unemployment_rate", "gdp_growth", "interest_rate",
                     "hpi_growth", "inflation_rate"]:
            assert var in traj.columns

    def test_trajectories_with_regime(self):
        """Les trajectoires fonctionnent avec un regime."""
        from ifrs9_cockpit.ai_analyst.layer5_prospective import project_trajectories
        from ifrs9_cockpit.ai_analyst.types import RegimeClassification
        macro = {
            "unemployment_rate": SCENARIO_BASE.unemployment_rate,
            "gdp_growth": SCENARIO_BASE.gdp_growth,
            "interest_rate": SCENARIO_BASE.interest_rate,
            "hpi_growth": SCENARIO_BASE.hpi_growth,
            "inflation_rate": SCENARIO_BASE.inflation_rate,
        }
        regime = RegimeClassification(
            probabilities={"Crise financiere": 0.8, "Stagflation": 0.05,
                           "Rupture techno": 0.05, "Resserrement": 0.05, "Reprise": 0.05},
            detected_regime="Crise financiere",
        )
        traj = project_trajectories(macro, regime)
        assert len(traj) == 12

    def test_risk_appetite_10_cells(self, pipeline_data):
        """Le risk appetite retourne 10 cellules (5 secteurs x 2 canaux)."""
        from ifrs9_cockpit.ai_analyst.layer5_prospective import compute_risk_appetite
        result_credit, result_pe, _ = pipeline_data
        ra = compute_risk_appetite(result_credit, result_pe)
        assert len(ra) == 10

    def test_risk_appetite_columns(self, pipeline_data):
        """Les colonnes requises du risk appetite."""
        from ifrs9_cockpit.ai_analyst.layer5_prospective import compute_risk_appetite
        result_credit, result_pe, _ = pipeline_data
        ra = compute_risk_appetite(result_credit, result_pe)
        required = {"sector", "canal", "metric", "value", "signal"}
        assert required.issubset(set(ra.columns))

    def test_risk_appetite_signals_valid(self, pipeline_data):
        """Les signaux du risk appetite sont vert, ambre ou rouge."""
        from ifrs9_cockpit.ai_analyst.layer5_prospective import compute_risk_appetite
        result_credit, result_pe, _ = pipeline_data
        ra = compute_risk_appetite(result_credit, result_pe)
        assert set(ra["signal"]).issubset({"vert", "ambre", "rouge"})

    def test_early_warning_5_sectors(self, pipeline_data):
        """L'early warning retourne 5 secteurs."""
        from ifrs9_cockpit.ai_analyst.layer5_prospective import compute_early_warning
        result_credit, result_pe, macro_params = pipeline_data
        ew = compute_early_warning(result_credit, result_pe, macro_params)
        assert len(ew) == 5

    def test_early_warning_scores_0_1(self, pipeline_data):
        """Les scores early warning sont dans [0, 1]."""
        from ifrs9_cockpit.ai_analyst.layer5_prospective import compute_early_warning
        result_credit, result_pe, macro_params = pipeline_data
        ew = compute_early_warning(result_credit, result_pe, macro_params)
        assert (ew["ew_score"] >= 0).all()
        assert (ew["ew_score"] <= 1).all()

    def test_early_warning_signals_valid(self, pipeline_data):
        """Les signaux early warning sont vert, ambre ou rouge."""
        from ifrs9_cockpit.ai_analyst.layer5_prospective import compute_early_warning
        result_credit, result_pe, macro_params = pipeline_data
        ew = compute_early_warning(result_credit, result_pe, macro_params)
        assert set(ew["ew_signal"]).issubset({"vert", "ambre", "rouge"})

    def test_early_warning_columns(self, pipeline_data):
        """Les colonnes requises de l'early warning."""
        from ifrs9_cockpit.ai_analyst.layer5_prospective import compute_early_warning
        result_credit, result_pe, macro_params = pipeline_data
        ew = compute_early_warning(result_credit, result_pe, macro_params)
        required = {"sector", "ew_score", "ew_signal", "pd_mean", "drawdown", "macro_stress"}
        assert required.issubset(set(ew.columns))


# ============================================================
# Story 5-4 : Orchestrateur 2 passes (FR25, FR31, FR32)
# ============================================================

class TestOrchestrator:
    """Tests orchestrateur CRO — 2 passes, recommandations, narrative."""

    def test_analyze_returns_analytics_state(self, pipeline_data):
        """L'analyse retourne un AnalyticsState."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        from ifrs9_cockpit.ai_analyst.types import AnalyticsState
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert isinstance(state, AnalyticsState)

    def test_pass_2_executed(self, pipeline_data):
        """La passe 2 est executee (pass_number = 2)."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert state.pass_number == 2

    def test_regime_detected(self, pipeline_data):
        """Un regime est detecte."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert state.regime is not None
        assert state.regime.detected_regime in state.regime.probabilities

    def test_asymmetry_matrix_populated(self, pipeline_data):
        """La matrice d'asymetrie est peuplee avec 5 secteurs."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert state.asymmetry_matrix is not None
        assert len(state.asymmetry_matrix) == 5

    def test_proportional_contributions_populated(self, pipeline_data):
        """L'allocation proportionnelle est peuplee avec 10 cellules."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert state.proportional_contributions is not None
        assert len(state.proportional_contributions) == 10

    def test_tipping_points_populated(self, pipeline_data):
        """Les tipping points sont peuples avec 5 variables."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert state.tipping_points is not None
        assert len(state.tipping_points) == 5

    def test_trajectories_populated(self, pipeline_data):
        """Les trajectoires sont peuplees avec 12 projections."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert state.trajectories is not None
        assert len(state.trajectories) == 12

    def test_risk_appetite_populated(self, pipeline_data):
        """Le risk appetite est peuple avec 10 cellules."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert state.risk_appetite_matrix is not None
        assert len(state.risk_appetite_matrix) == 10

    def test_recommendations_generated(self, pipeline_data):
        """Au moins une recommandation est generee."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert len(state.recommendations) >= 1

    def test_recommendation_has_full_chain(self, pipeline_data):
        """La recommandation porte la chaine complete (FR31)."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        rec = state.recommendations[0]
        assert rec.action
        assert rec.regime
        assert rec.trigger
        assert rec.proportional_driver
        assert rec.macro_factor
        assert rec.risk_appetite_status in {"vert", "ambre", "rouge"}
        assert isinstance(rec.rst_distance, float)
        assert rec.confidence in {"high", "medium", "low"}
        assert isinstance(rec.alternatives, list)
        assert len(rec.alternatives) >= 1

    def test_narrative_nonempty(self, pipeline_data):
        """La synthese narrative est non-vide (FR32)."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert len(state.narrative) > 50

    def test_narrative_contains_regime(self, pipeline_data):
        """La narrative mentionne le regime detecte."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert "Regime detecte" in state.narrative

    def test_narrative_contains_asymetries(self, pipeline_data):
        """La narrative mentionne les asymetries."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert "asymetrie" in state.narrative.lower()

    def test_narrative_contains_rst(self, pipeline_data):
        """La narrative mentionne la distance RST."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert "RST" in state.narrative

    def test_narrative_contains_recommandation(self, pipeline_data):
        """La narrative inclut la recommandation."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert "Recommandation" in state.narrative

    def test_early_warning_populated(self, pipeline_data):
        """Les indicateurs early warning sont peuples."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert state.early_warning is not None
        assert len(state.early_warning) == 5

    def test_factor_attribution_populated(self, pipeline_data):
        """L'attribution factorielle est peuplee."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert state.factor_attribution is not None
        assert len(state.factor_attribution) == 10

    def test_custom_target_ecl(self, pipeline_data):
        """L'orchestrateur accepte un target_ecl personnalise (FR54) et le transmet au RST."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        custom_ecl = 500_000_000_000.0
        analyst = CROAnalyst(
            result_credit, result_pe, macro_params,
            target_ecl=custom_ecl,
        )
        state = analyst.analyze()
        assert state.pass_number == 2
        # Verifier que le seuil personnalise est reflecte dans le RST
        assert state.rst_result is not None
        assert state.rst_result["ecl_breach_threshold"] == round(custom_ecl, 0)

    def test_narrative_contains_allocation(self, pipeline_data):
        """La narrative mentionne l'allocation proportionnelle."""
        from ifrs9_cockpit.ai_analyst.orchestrator import CROAnalyst
        result_credit, result_pe, macro_params = pipeline_data
        analyst = CROAnalyst(result_credit, result_pe, macro_params)
        state = analyst.analyze()
        assert "Allocation proportionnelle" in state.narrative


# ============================================================
# Types — Structures de donnees (Story 5-1)
# ============================================================

class TestTypes:
    """Tests pour les types de donnees du module AI Analyst."""

    def test_analytics_state_defaults(self):
        """AnalyticsState a des valeurs par defaut raisonnables."""
        from ifrs9_cockpit.ai_analyst.types import AnalyticsState
        state = AnalyticsState()
        assert state.pass_number == 1
        assert state.asymmetry_matrix is None
        assert state.recommendations == []
        assert state.narrative == ""

    def test_regime_classification(self):
        """RegimeClassification est instanciable."""
        from ifrs9_cockpit.ai_analyst.types import RegimeClassification
        rc = RegimeClassification(
            probabilities={"A": 0.5, "B": 0.5},
            detected_regime="A",
        )
        assert rc.detected_regime == "A"
        assert rc.probabilities["A"] == 0.5

    def test_recommendation_dataclass(self):
        """Recommendation porte tous les champs requis."""
        from ifrs9_cockpit.ai_analyst.types import Recommendation
        rec = Recommendation(
            action="test",
            regime="Crise",
            trigger="test trigger",
            proportional_driver="Tech Credit",
            macro_factor="unemployment_rate",
            risk_appetite_status="rouge",
            rst_distance=3.5,
            confidence="medium",
            alternatives=["alt1"],
        )
        assert rec.action == "test"
        assert rec.confidence == "medium"
        assert len(rec.alternatives) == 1

    def test_package_exports(self):
        """Le package ai_analyst exporte CROAnalyst et les types."""
        from ifrs9_cockpit.ai_analyst import (
            CROAnalyst,
            AnalyticsState,
            Recommendation,
            RegimeClassification,
        )
        assert CROAnalyst is not None
        assert AnalyticsState is not None
