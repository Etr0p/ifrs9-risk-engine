"""Tests unitaires pour le moteur PE IFRS 13 (Stories 3-1, 3-2, 3-3).

Story 3-1: Valoriser participations PE via NAV fondamentale et methodes IPEV
Story 3-2: Calculer metriques de performance PE sous stress
Story 3-3: Estimer distress, classifier et chiffrer cout de sortie

Couvre :
  - PEModel : NAV fondamentale, methodes IPEV, compression multiples, stress canaux
  - PECalculator : MOIC, IRR, DPI, RVPI, TVPI, sensibilites, drawdown
  - PECalculator : P(distress), classification, expected loss, exit cost, RWA
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ifrs9_cockpit.config import (
    PE_CLASSIFICATION_CONFIG,
    RANDOM_SEED,
    REQUIRED_PE_RESULT_COLS,
    SCENARIO_BASE,
    SECTORS,
)
from ifrs9_cockpit.data.generator import generate_dataset
from ifrs9_cockpit.engine.pe_calculator import PECalculator
from ifrs9_cockpit.models.pe_model import PEModel


# ============================================================
# Fixtures partageees (scope=module pour ne generer qu'une fois)
# ============================================================

@pytest.fixture(scope="module")
def dataset():
    """Genere le dataset complet une seule fois."""
    df_credit, df_pe, df_history = generate_dataset()
    return df_credit, df_pe, df_history


@pytest.fixture(scope="module")
def df_pe(dataset):
    """DataFrame PE brut."""
    return dataset[1]


@pytest.fixture(scope="module")
def pe_model():
    """Instance PEModel reproductible."""
    return PEModel(seed=RANDOM_SEED)


@pytest.fixture(scope="module")
def nav_baseline(pe_model, df_pe):
    """NAV et multiples baseline."""
    nav, mult = pe_model.calculate_nav(df_pe)
    return nav, mult


@pytest.fixture(scope="module")
def pe_calculator():
    """Instance PECalculator."""
    return PECalculator()


@pytest.fixture(scope="module")
def pe_result(pe_calculator, df_pe):
    """Resultat complet du calculateur PE (baseline)."""
    return pe_calculator.calculate(df_pe)


@pytest.fixture(scope="module")
def scenario_results(pe_calculator, df_pe):
    """Resultats PE sous les 3 scenarios ECL."""
    return pe_calculator.calculate_scenario_metrics(df_pe)


# ============================================================
# Story 3-1: Valoriser participations PE via NAV IPEV
# ============================================================

class TestPEModelNAVBaseline:
    """Tests Story 3-1 — NAV fondamentale et methodes IPEV."""

    def test_nav_all_positive(self, nav_baseline):
        """NAV >= 0 pour toute position baseline."""
        nav, _ = nav_baseline
        assert (nav >= 0).all(), f"NAV min={nav.min():.4f}"

    def test_nav_min_strictly_positive(self, nav_baseline):
        """NAV minimale strictement > 0 (pas de position a zero)."""
        nav, _ = nav_baseline
        assert nav.min() > 0, f"NAV min={nav.min():.6f}"

    def test_nav_shape_matches_input(self, nav_baseline, df_pe):
        """La NAV a le meme nombre d'elements que le DataFrame PE."""
        nav, _ = nav_baseline
        assert len(nav) == len(df_pe)

    def test_exit_multiples_shape(self, nav_baseline, df_pe):
        """Les multiples de sortie ont la bonne dimension."""
        _, mult = nav_baseline
        assert len(mult) == len(df_pe)

    def test_exit_multiples_positive(self, nav_baseline):
        """Multiples de sortie > 0."""
        _, mult = nav_baseline
        assert (mult > 0).all(), f"mult min={mult.min():.4f}"


class TestPEModelIPEVMethods:
    """Tests methodes IPEV par secteur (FR13)."""

    def test_technologie_uses_ev_revenue(self, df_pe):
        """Technologie utilise EV/Revenue."""
        tech = df_pe[df_pe["sector"] == "Technologie"]
        assert (tech["valuation_method"] == "EV/Revenue").all()

    def test_industrie_uses_ev_ebitda(self, df_pe):
        """Industrie utilise EV/EBITDA."""
        ind = df_pe[df_pe["sector"] == "Industrie"]
        assert (ind["valuation_method"] == "EV/EBITDA").all()

    def test_sante_uses_ev_ebitda(self, df_pe):
        """Sante utilise EV/EBITDA."""
        sante = df_pe[df_pe["sector"] == "Sante"]
        assert (sante["valuation_method"] == "EV/EBITDA").all()

    def test_services_uses_ev_ebitda(self, df_pe):
        """Services utilise EV/EBITDA."""
        services = df_pe[df_pe["sector"] == "Services"]
        assert (services["valuation_method"] == "EV/EBITDA").all()

    def test_immobilier_uses_cap_rate_noi(self, df_pe):
        """Immobilier utilise Cap_rate/NOI."""
        immo = df_pe[df_pe["sector"] == "Immobilier"]
        assert (immo["valuation_method"] == "Cap_rate/NOI").all()

    def test_all_valuation_methods_config_driven(self, df_pe):
        """Les methodes de valorisation correspondent aux config des secteurs."""
        for sector in SECTORS:
            mask = df_pe["sector"] == sector.name
            actual = df_pe.loc[mask, "valuation_method"].unique()
            assert len(actual) == 1
            assert actual[0] == sector.valuation_method


class TestPEModelMultipleCompression:
    """Tests compression des multiples (FR15)."""

    def test_multiples_in_ipev_range(self, nav_baseline, df_pe):
        """Multiples de sortie dans les fourchettes IPEV elargies."""
        _, mult = nav_baseline
        for sector in SECTORS:
            mask = df_pe["sector"].values == sector.name
            if mask.sum() == 0:
                continue
            low, high = sector.entry_multiple_range
            # Fourchette elargie : [low*0.5, high*1.5] (cf. pe_model.py)
            mult_sec = mult[mask]
            assert mult_sec.min() >= low * 0.5 - 1e-6, \
                f"{sector.name}: min mult={mult_sec.min():.2f}, floor={low*0.5:.2f}"
            assert mult_sec.max() <= high * 1.5 + 1e-6, \
                f"{sector.name}: max mult={mult_sec.max():.2f}, cap={high*1.5:.2f}"

    def test_monotonicity_rate_up_multiple_down(self, df_pe):
        """Monotone : hausse taux -> baisse multiple de sortie."""
        m1 = PEModel(seed=RANDOM_SEED)
        _, mult_low = m1.calculate_nav(df_pe, interest_rate_override=2.0)
        m2 = PEModel(seed=RANDOM_SEED)
        _, mult_high = m2.calculate_nav(df_pe, interest_rate_override=6.0)
        assert mult_low.mean() > mult_high.mean(), \
            f"mult(IR=2%)={mult_low.mean():.2f} should > mult(IR=6%)={mult_high.mean():.2f}"


class TestPEModelOverrideZero:
    """Tests gestion correcte de override=0.0."""

    def test_override_zero_not_ignored(self, df_pe):
        """Un override a 0.0 ne doit pas etre traite comme None (bug `or`)."""
        m1 = PEModel(seed=RANDOM_SEED)
        nav_zero, _ = m1.calculate_nav(df_pe, gdp_override=0.0)
        m2 = PEModel(seed=RANDOM_SEED)
        nav_none, _ = m2.calculate_nav(df_pe, gdp_override=None)
        # gdp_override=0.0 est different de gdp baseline (1.2%), donc les NAV diffèrent
        assert not np.allclose(nav_zero, nav_none, atol=0.01), \
            "override=0.0 est traite comme None (bug falsy)"

    def test_override_zero_interest_rate(self, df_pe):
        """Override interest_rate=0.0 doit produire un resultat different de baseline."""
        m1 = PEModel(seed=RANDOM_SEED)
        nav_zero, _ = m1.calculate_nav(df_pe, interest_rate_override=0.0)
        m2 = PEModel(seed=RANDOM_SEED)
        nav_base, _ = m2.calculate_nav(df_pe)
        # interest_rate baseline = 3.5, donc override=0.0 change le resultat
        assert not np.allclose(nav_zero, nav_base, atol=0.01), \
            "override interest_rate=0.0 ignore"


class TestPEModelScenarios:
    """Tests NAV sous 3 scenarios."""

    def test_nav_scenario_ordering(self, pe_model, df_pe):
        """Favorable > Base > Adverse en NAV totale."""
        navs = pe_model.calculate_nav_scenarios(df_pe)
        assert navs["Favorable"].sum() > navs["Base"].sum() > navs["Adverse"].sum()

    def test_nav_scenarios_all_positive(self, pe_model, df_pe):
        """NAV >= 0 sous tous les scenarios."""
        navs = pe_model.calculate_nav_scenarios(df_pe)
        for name, nav in navs.items():
            assert (nav >= 0).all(), f"Scenario {name}: NAV min={nav.min():.4f}"

    def test_nav_scenarios_3_keys(self, pe_model, df_pe):
        """3 scenarios : Base, Adverse, Favorable."""
        navs = pe_model.calculate_nav_scenarios(df_pe)
        assert set(navs.keys()) == {"Base", "Adverse", "Favorable"}


class TestPEModelSummary:
    """Tests resume NAV par secteur."""

    def test_summary_has_all_sectors(self, pe_model, df_pe, nav_baseline):
        """Le resume couvre les 5 secteurs."""
        nav, _ = nav_baseline
        summary = pe_model.get_nav_summary(df_pe, nav)
        assert set(summary["sector"].unique()) == {s.name for s in SECTORS}

    def test_summary_nav_total_matches(self, pe_model, df_pe, nav_baseline):
        """La NAV totale du resume correspond a la somme des NAV."""
        nav, _ = nav_baseline
        summary = pe_model.get_nav_summary(df_pe, nav)
        assert abs(summary["nav_total"].sum() - nav.sum()) < 1.0


class TestPEModelStandalone:
    """Test bloc __main__."""

    def test_pe_model_standalone_runs(self):
        """python -m ifrs9_cockpit.models.pe_model retourne 0."""
        result = subprocess.run(
            [sys.executable, "-m", "ifrs9_cockpit.models.pe_model"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_pe_model_standalone_output(self):
        """La sortie contient 'PE Model valide'."""
        result = subprocess.run(
            [sys.executable, "-m", "ifrs9_cockpit.models.pe_model"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "PE Model valide" in result.stdout


# ============================================================
# Story 3-2: Calculer metriques de performance PE sous stress
# ============================================================

class TestPECalculatorPerformanceMetrics:
    """Tests Story 3-2 — MOIC, IRR, DPI, RVPI, TVPI."""

    def test_moic_positive(self, pe_result):
        """MOIC >= 0 pour toutes les positions."""
        assert (pe_result["moic"] >= 0).all()

    def test_irr_bounded(self, pe_result):
        """IRR dans une plage raisonnable [-1, +10]."""
        assert (pe_result["irr"] >= -1.0).all()
        assert (pe_result["irr"] <= 10.0).all()

    def test_dpi_is_zero(self, pe_result):
        """DPI = 0 (pas de distributions intermediaires)."""
        assert (pe_result["dpi"] == 0).all()

    def test_rvpi_equals_moic(self, pe_result):
        """RVPI = MOIC (pas de distributions)."""
        assert np.allclose(pe_result["rvpi"].values, pe_result["moic"].values, atol=1e-4)

    def test_tvpi_equals_dpi_plus_rvpi(self, pe_result):
        """TVPI = DPI + RVPI (identite fondamentale)."""
        tvpi = pe_result["tvpi"].values
        dpi = pe_result["dpi"].values
        rvpi = pe_result["rvpi"].values
        assert np.allclose(tvpi, dpi + rvpi, atol=1e-3)

    def test_capital_invested_positive(self, pe_result):
        """Capital investi > 0 pour toute position."""
        assert (pe_result["capital_invested"] > 0).all()

    def test_nav_column_present(self, pe_result):
        """La colonne nav est presente dans le resultat."""
        assert "nav" in pe_result.columns

    def test_exit_multiple_present(self, pe_result):
        """La colonne exit_multiple est presente."""
        assert "exit_multiple" in pe_result.columns


class TestPECalculatorScenarioMetrics:
    """Tests metriques sous 3 scenarios."""

    def test_irr_ordering_across_scenarios(self, scenario_results):
        """IRR adverse < IRR base < IRR favorable."""
        irr_adv = scenario_results["Adverse"]["irr"].mean()
        irr_base = scenario_results["Base"]["irr"].mean()
        irr_fav = scenario_results["Favorable"]["irr"].mean()
        assert irr_adv < irr_base < irr_fav, \
            f"IRR: Adverse={irr_adv:.4f}, Base={irr_base:.4f}, Favorable={irr_fav:.4f}"

    def test_3_scenarios_present(self, scenario_results):
        """3 scenarios : Base, Adverse, Favorable."""
        assert set(scenario_results.keys()) == {"Base", "Adverse", "Favorable"}

    def test_scenario_results_have_all_columns(self, scenario_results):
        """Chaque scenario a toutes les colonnes requises."""
        for name, res in scenario_results.items():
            missing = REQUIRED_PE_RESULT_COLS - set(res.columns)
            assert len(missing) == 0, f"Scenario {name}: colonnes manquantes {missing}"


class TestPECalculatorSensitivities:
    """Tests sensibilites factorielles."""

    def test_sensitivities_5x5_non_zero(self, pe_calculator, df_pe):
        """Matrice de sensibilites 5x5 : chaque colonne a au moins une valeur non-nulle."""
        sens = pe_calculator.compute_factorial_sensitivities(df_pe, delta=1.0)
        assert sens.shape == (5, 5), f"Shape: {sens.shape}"
        for col in sens.columns:
            assert (sens[col].abs() > 0).any(), f"Colonne {col} entierement nulle"

    def test_sensitivities_sectors_match(self, pe_calculator, df_pe):
        """Les secteurs dans les sensibilites correspondent aux secteurs config."""
        sens = pe_calculator.compute_factorial_sensitivities(df_pe)
        assert set(sens.index) == {s.name for s in SECTORS}


class TestPECalculatorDrawdown:
    """Tests NAV drawdown."""

    def test_drawdown_non_negative(self, pe_result):
        """Drawdown >= 0 (par definition)."""
        assert (pe_result["nav_drawdown"] >= 0).all()

    def test_drawdown_bounded_by_one(self, pe_result):
        """Drawdown <= 1 (max = 100% de perte)."""
        assert (pe_result["nav_drawdown"] <= 1.0).all()

    def test_delta_nav_present(self, pe_result):
        """Colonne delta_nav presente."""
        assert "delta_nav" in pe_result.columns


class TestPECalculatorPerformanceSummary:
    """Tests resume de performance par secteur."""

    def test_summary_has_all_sectors(self, pe_calculator, pe_result):
        """Le resume couvre les 5 secteurs."""
        summary = pe_calculator.get_performance_summary(pe_result)
        assert set(summary["sector"].unique()) == {s.name for s in SECTORS}

    def test_summary_has_expected_columns(self, pe_calculator, pe_result):
        """Le resume a les colonnes attendues."""
        summary = pe_calculator.get_performance_summary(pe_result)
        expected_cols = {"sector", "count", "nav_total", "nav_mean",
                         "moic_mean", "irr_mean", "tvpi_mean",
                         "drawdown_mean", "capital_total"}
        assert expected_cols.issubset(set(summary.columns))


# ============================================================
# Story 3-3: Estimer distress, classifier, cout de sortie
# ============================================================

class TestPECalculatorDistress:
    """Tests Story 3-3 — Probabilite de distress (FR16)."""

    def test_distress_prob_in_0_1(self, pe_result):
        """P(distress) dans [0, 1] pour toute position."""
        dp = pe_result["distress_prob"]
        assert (dp >= 0).all(), f"P(distress) min={dp.min():.6f}"
        assert (dp <= 1).all(), f"P(distress) max={dp.max():.6f}"

    def test_distress_prob_baseline_below_100pct(self, pe_result):
        """P(distress) baseline < 100% (pas de certitude de defaut en base)."""
        assert pe_result["distress_prob"].max() < 1.0

    def test_distress_prob_baseline_above_zero(self, pe_result):
        """P(distress) baseline > 0% (au moins un risque residuel)."""
        assert pe_result["distress_prob"].min() > 0.0

    def test_expected_loss_pe_non_negative(self, pe_result):
        """Expected Loss PE >= 0 pour toute position."""
        assert (pe_result["expected_loss_pe"] >= 0).all()

    def test_expected_loss_formula(self, pe_result):
        """EL = P(distress) x LGD_equity x NAV."""
        lgd_eq = PE_CLASSIFICATION_CONFIG.lgd_equity
        el_calc = pe_result["distress_prob"] * lgd_eq * pe_result["nav"]
        assert np.allclose(
            pe_result["expected_loss_pe"].values,
            el_calc.values,
            atol=0.1,
        )


class TestPECalculatorClassification:
    """Tests classification PE (FR17)."""

    def test_risk_categories_valid(self, pe_result):
        """risk_category in {Performing, Watchlist, Distressed}."""
        valid = {"Performing", "Watchlist", "Distressed"}
        actual = set(pe_result["risk_category"].unique())
        assert actual.issubset(valid), f"Categories invalides: {actual - valid}"

    def test_performing_below_threshold(self, pe_result):
        """Performing <-> P(distress) < distress_threshold_performing."""
        perf = pe_result[pe_result["risk_category"] == "Performing"]
        if len(perf) > 0:
            threshold = PE_CLASSIFICATION_CONFIG.distress_threshold_performing
            assert (perf["distress_prob"] < threshold).all(), \
                f"Performing max P(distress)={perf['distress_prob'].max():.4f}, threshold={threshold}"

    def test_distressed_above_threshold(self, pe_result):
        """Distressed <-> P(distress) >= distress_threshold_watchlist."""
        distressed = pe_result[pe_result["risk_category"] == "Distressed"]
        if len(distressed) > 0:
            threshold = PE_CLASSIFICATION_CONFIG.distress_threshold_watchlist
            assert (distressed["distress_prob"] >= threshold).all(), \
                f"Distressed min P(distress)={distressed['distress_prob'].min():.4f}, threshold={threshold}"

    def test_watchlist_between_thresholds(self, pe_result):
        """Watchlist <-> P(distress) in [performing, watchlist[."""
        wl = pe_result[pe_result["risk_category"] == "Watchlist"]
        if len(wl) > 0:
            t_perf = PE_CLASSIFICATION_CONFIG.distress_threshold_performing
            t_watch = PE_CLASSIFICATION_CONFIG.distress_threshold_watchlist
            assert (wl["distress_prob"] >= t_perf).all()
            assert (wl["distress_prob"] < t_watch).all()

    def test_adverse_more_distressed_than_base(self, scenario_results):
        """Stress adverse -> plus de positions Distressed qu'en base."""
        n_base = (scenario_results["Base"]["risk_category"] == "Distressed").sum()
        n_adv = (scenario_results["Adverse"]["risk_category"] == "Distressed").sum()
        assert n_adv >= n_base, \
            f"Adverse Distressed ({n_adv}) should >= Base ({n_base})"


class TestPECalculatorExitCost:
    """Tests cout de sortie / rebalancement (FR18)."""

    def test_exit_cost_positive(self, pe_result):
        """Exit cost > 0 pour toute position."""
        assert (pe_result["exit_cost"] > 0).all()

    def test_exit_cost_formula(self, pe_result):
        """Exit cost = NAV x (1 - secondary_discount)."""
        sd = PE_CLASSIFICATION_CONFIG.secondary_discount
        expected = pe_result["nav"] * (1 - sd)
        assert np.allclose(
            pe_result["exit_cost"].values,
            expected.values,
            atol=0.1,
        )

    def test_exit_cost_less_than_nav(self, pe_result):
        """Exit cost < NAV (la decote reduit la valeur de sortie)."""
        assert (pe_result["exit_cost"] <= pe_result["nav"] + 0.01).all()


class TestPECalculatorRWA:
    """Tests RWA PE CRR3 (H8)."""

    def test_rwa_pe_positive(self, pe_result):
        """RWA PE > 0 pour toute position."""
        assert (pe_result["rwa_pe"] > 0).all()

    def test_rwa_pe_reasonable_range(self, pe_result):
        """RWA PE = NAV x RW/100, avec RW in {190, 250, 400}."""
        # RWA doit etre entre 1.90 x NAV et 4.00 x NAV
        ratio = pe_result["rwa_pe"] / pe_result["nav"]
        assert (ratio >= 1.89).all(), f"ratio min={ratio.min():.2f}"
        assert (ratio <= 4.01).all(), f"ratio max={ratio.max():.2f}"


class TestPECalculatorResultColumns:
    """Tests colonnes du resultat."""

    def test_required_pe_result_cols_present(self, pe_result):
        """Toutes les colonnes requises sont presentes."""
        missing = REQUIRED_PE_RESULT_COLS - set(pe_result.columns)
        assert len(missing) == 0, f"Colonnes manquantes: {missing}"

    def test_additional_metric_cols_present(self, pe_result):
        """Colonnes additionnelles de metriques presentes."""
        expected = {"moic", "irr", "dpi", "rvpi", "tvpi",
                    "capital_invested", "distress_prob", "exit_cost"}
        missing = expected - set(pe_result.columns)
        assert len(missing) == 0, f"Colonnes manquantes: {missing}"


class TestPECalculatorCircularImport:
    """Tests import circulaire pe_calculator -> comparator."""

    def test_compute_crr3_rw_importable(self):
        """compute_crr3_rw doit etre importable depuis comparator sans erreur."""
        from ifrs9_cockpit.engine.comparator import compute_crr3_rw
        assert callable(compute_crr3_rw)

    def test_pe_calculator_crr3_rw_used(self, pe_result):
        """Le RWA PE utilise bien compute_crr3_rw (RW in {190, 250, 400})."""
        ratio = pe_result["rwa_pe"] / pe_result["nav"]
        # Chaque ratio doit etre proche de l'un des 3 RW / 100
        # (tolerance pour l'arrondi des colonnes nav et rwa_pe)
        valid_ratios = np.array([1.90, 2.50, 4.00])
        for r in ratio:
            distances = np.abs(valid_ratios - r)
            assert distances.min() < 0.05, \
                f"Ratio RWA/NAV={r:.4f} n'est pas proche de {valid_ratios}"


class TestPECalculatorStandalone:
    """Test bloc __main__."""

    def test_pe_calculator_standalone_runs(self):
        """python -m ifrs9_cockpit.engine.pe_calculator retourne 0."""
        result = subprocess.run(
            [sys.executable, "-m", "ifrs9_cockpit.engine.pe_calculator"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_pe_calculator_standalone_output(self):
        """La sortie contient les validations PE Calculator."""
        result = subprocess.run(
            [sys.executable, "-m", "ifrs9_cockpit.engine.pe_calculator"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "PE Calculator" in result.stdout
        assert "valide" in result.stdout
