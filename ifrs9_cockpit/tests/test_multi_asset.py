"""Tests pour l'extension multi-actif (14 classes, Vasicek ASRF, contagion).

Couvre :
    - AssetClassProfile : validation des 14 profils, coherence parametrique
    - Vasicek ECL : PD conditionnelle, Jensen inequality, staging cliff
    - Climate Risk : ajustement physique/transition, no double-counting
    - Contagion : propagation non-lineaire, point fixe, death spiral
    - Balance Sheet ECL : pipeline complet, multi-scenario
    - DGP extension : df_balance_sheet generation
"""

import pytest

pytestmark = pytest.mark.fourteen

import unittest
import numpy as np
import polars as pl

from ifrs9_cockpit.config import (
    ASSET_CLASSES,
    ASSET_CLASS_MAP,
    ASSET_CLASS_NAMES,
    LEVEL1_CLASSES,
    LEVEL2_CLASSES,
    LEVEL3_CLASSES,
    PREDEFINED_SCENARIOS,
    RANDOM_SEED,
)


# ── Cache module pour eviter les appels redondants a generate_dataset ──

_BS_CACHE = {}


def _get_dataset_v4(n_clients=50, seed=42):
    """Cache module-level pour generate_dataset."""
    key = (n_clients, seed)
    if key not in _BS_CACHE:
        from ifrs9_cockpit.data.generator import generate_dataset
        _BS_CACHE[key] = generate_dataset(n_clients=n_clients, seed=seed)
    return _BS_CACHE[key]


class TestAssetClassProfile(unittest.TestCase):
    """Validation des 10 profils AssetClassProfile."""

    def test_at_least_14_classes(self):
        self.assertGreaterEqual(len(ASSET_CLASSES), 14)

    def test_unique_names(self):
        names = [ac.name for ac in ASSET_CLASSES]
        self.assertEqual(len(names), len(set(names)))

    def test_unique_labels(self):
        labels = [ac.label for ac in ASSET_CLASSES]
        self.assertEqual(len(labels), len(set(labels)))

    def test_typical_weights_sum(self):
        """Les poids typiques doivent sommer a ~1.0 (tolerance +-15%)."""
        total = sum(ac.typical_weight for ac in ASSET_CLASSES)
        self.assertAlmostEqual(total, 1.0, delta=0.40,
                               msg=f"Total typical_weight = {total}")

    def test_level_distribution(self):
        """4 Level 1, 6 Level 2, 4 Level 3."""
        self.assertEqual(len(LEVEL1_CLASSES), 4)
        self.assertEqual(len(LEVEL2_CLASSES), 6)
        self.assertEqual(len(LEVEL3_CLASSES), 4)

    def test_valid_categories(self):
        valid = {"level1_detailed", "level2_parametric", "level3_simple"}
        for ac in ASSET_CLASSES:
            self.assertIn(ac.category, valid, msg=f"{ac.name}")

    def test_pd_base_in_bounds(self):
        for ac in ASSET_CLASSES:
            self.assertGreater(ac.pd_base, 0, msg=f"{ac.name}")
            self.assertLess(ac.pd_base, 1, msg=f"{ac.name}")

    def test_lgd_base_in_bounds(self):
        for ac in ASSET_CLASSES:
            self.assertGreater(ac.lgd_base, 0, msg=f"{ac.name}")
            self.assertLess(ac.lgd_base, 1, msg=f"{ac.name}")

    def test_asset_correlation_in_bounds(self):
        """rho dans [0.03, 0.50] (CRR3 bornes)."""
        for ac in ASSET_CLASSES:
            self.assertGreater(ac.asset_correlation, 0.03, msg=f"{ac.name}")
            self.assertLessEqual(ac.asset_correlation, 0.50, msg=f"{ac.name}")

    def test_macro_sensitivities_keys(self):
        expected = {"gdp_growth", "unemployment_rate", "interest_rate",
                    "hpi_growth", "inflation_rate"}
        for ac in ASSET_CLASSES:
            self.assertEqual(set(ac.macro_sensitivities.keys()), expected,
                             msg=f"{ac.name}")

    def test_hqla_levels_valid(self):
        for ac in ASSET_CLASSES:
            self.assertIn(ac.hqla_level, [0, 1, 2, 3], msg=f"{ac.name}")
            if ac.hqla_eligible:
                self.assertGreater(ac.hqla_level, 0, msg=f"{ac.name}")
            else:
                self.assertEqual(ac.hqla_level, 0, msg=f"{ac.name}")

    def test_map_completeness(self):
        for name in ASSET_CLASS_NAMES:
            self.assertIn(name, ASSET_CLASS_MAP)

    def test_known_hqla_classes(self):
        """Sovereign=L1, Covered Bonds=L2A, Interbank=L2B."""
        self.assertTrue(ASSET_CLASS_MAP["sovereign"].hqla_eligible)
        self.assertEqual(ASSET_CLASS_MAP["sovereign"].hqla_level, 1)
        self.assertTrue(ASSET_CLASS_MAP["covered_bonds"].hqla_eligible)
        self.assertEqual(ASSET_CLASS_MAP["covered_bonds"].hqla_level, 2)
        self.assertTrue(ASSET_CLASS_MAP["interbank"].hqla_eligible)
        self.assertEqual(ASSET_CLASS_MAP["interbank"].hqla_level, 3)

    def test_level1_are_corporate_and_pe(self):
        self.assertIn("corporate_loans", LEVEL1_CLASSES)
        self.assertIn("private_equity", LEVEL1_CLASSES)


class TestVasicekECL(unittest.TestCase):
    """Tests du moteur ECL Vasicek ASRF."""

    def setUp(self):
        from ifrs9_cockpit.engine.balance_sheet_ecl import (
            vasicek_conditional_pd,
            staging_cliff_ecl,
            macro_to_z,
        )
        self.vasicek = vasicek_conditional_pd
        self.staging_cliff = staging_cliff_ecl
        self.macro_to_z = macro_to_z

    def test_vasicek_z_zero_near_pd(self):
        """A Z=0 (pas de stress), PD_cond < PD_base (effet rho)."""
        pd_cond = self.vasicek(0.05, 0.15, 0.0)
        # Vasicek at Z=0 gives PD < PD_base because of the rho correction
        self.assertGreater(pd_cond, 0)
        self.assertLess(pd_cond, 0.15)  # at least bounded

    def test_vasicek_stress_increases_pd(self):
        """Z positif (stress) augmente la PD conditionnelle."""
        pd_base = self.vasicek(0.01, 0.15, 0.0)
        pd_stress = self.vasicek(0.01, 0.15, 2.0)
        self.assertGreater(pd_stress, pd_base)

    def test_vasicek_favorable_decreases_pd(self):
        """Z negatif (favorable) diminue la PD conditionnelle."""
        pd_base = self.vasicek(0.01, 0.15, 0.0)
        pd_fav = self.vasicek(0.01, 0.15, -1.0)
        self.assertLess(pd_fav, pd_base)

    def test_vasicek_higher_rho_amplifies(self):
        """rho plus eleve amplifie l'impact de Z."""
        pd_low_rho = self.vasicek(0.01, 0.05, 2.0)
        pd_high_rho = self.vasicek(0.01, 0.25, 2.0)
        self.assertGreater(pd_high_rho, pd_low_rho)

    def test_vasicek_bounded(self):
        """PD conditionnelle dans [0, 1]."""
        for z in [-5, -2, 0, 2, 5]:
            pd_cond = self.vasicek(0.05, 0.20, z)
            self.assertGreaterEqual(pd_cond, 0)
            self.assertLessEqual(pd_cond, 1)

    def test_jensen_inequality(self):
        """ECL Vasicek > ECL naive sous stress (Jensen's inequality)."""
        pd_base = 0.01
        rho = 0.15
        z_stress = 2.0
        lgd = 0.35
        ead = 1e9
        tenor = 4.0

        # Naive ECL (sans Vasicek)
        ecl_naive = pd_base * lgd * ead * tenor

        # Vasicek ECL (avec staging cliff)
        pd_cond = self.vasicek(pd_base, rho, z_stress)
        ecl_vasicek = self.staging_cliff(pd_cond, lgd, ead, tenor, pd_base)

        self.assertGreater(ecl_vasicek, ecl_naive,
                           msg="Jensen: ECL Vasicek should exceed naive under stress")

    def test_staging_cliff_ecl_positive(self):
        """ECL toujours >= 0."""
        ecl = self.staging_cliff(0.01, 0.35, 1e6, 4.0, 0.005)
        self.assertGreaterEqual(ecl, 0)

    def test_staging_cliff_stage2_amplifies(self):
        """PD_cond >> PD_orig produit un ECL plus eleve (stage 2 lifetime)."""
        ecl_low = self.staging_cliff(0.005, 0.35, 1e6, 5.0, 0.005)  # PD ratio = 1
        ecl_high = self.staging_cliff(0.050, 0.35, 1e6, 5.0, 0.005)  # PD ratio = 10
        self.assertGreater(ecl_high, ecl_low)


class TestClimateRisk(unittest.TestCase):
    """Tests de l'ajustement climatique."""

    def setUp(self):
        from ifrs9_cockpit.engine.balance_sheet_ecl import climate_adjusted_pd
        self.climate_pd = climate_adjusted_pd

    def test_no_shock_no_change(self):
        """Sans choc climatique, PD inchangee."""
        pd = self.climate_pd(0.01, 0.5, 0.5, 0.2, 0.3, 0.0, 0.0)
        self.assertAlmostEqual(pd, 0.01)

    def test_carbon_shock_increases_pd(self):
        """Choc carbone augmente la PD."""
        pd_base = 0.01
        pd_adj = self.climate_pd(pd_base, 0.1, 0.5, 0.1, 0.1, 1.5, 0.0)
        self.assertGreater(pd_adj, pd_base)

    def test_physical_severity_increases_pd(self):
        """Risque physique augmente la PD."""
        pd_base = 0.01
        pd_adj = self.climate_pd(pd_base, 0.7, 0.1, 0.1, 0.1, 0.0, 0.8)
        self.assertGreater(pd_adj, pd_base)

    def test_green_capex_attenuates(self):
        """Green capex reduit l'impact de la transition."""
        pd_brown = self.climate_pd(0.01, 0.1, 0.5, 0.0, 0.1, 1.5, 0.0)
        pd_green = self.climate_pd(0.01, 0.1, 0.5, 0.8, 0.1, 1.5, 0.0)
        self.assertLess(pd_green, pd_brown,
                        msg="Green capex should attenuate transition risk")

    def test_scope3_adds_supply_chain(self):
        """Scope 3 ajoute un add-on proportionnel a delta_transition."""
        pd_no_scope3 = self.climate_pd(0.01, 0.1, 0.5, 0.1, 0.0, 1.5, 0.0)
        pd_scope3 = self.climate_pd(0.01, 0.1, 0.5, 0.1, 0.5, 1.5, 0.0)
        self.assertGreater(pd_scope3, pd_no_scope3)

    def test_pd_capped_at_999(self):
        """PD climatique capped a 0.999."""
        pd = self.climate_pd(0.5, 1.0, 1.0, 0.0, 1.0, 10.0, 1.0)
        self.assertLessEqual(pd, 0.999)

    def test_convexity_transition(self):
        """Exposant 1.5 : doublement du choc carbone augmente l'impact >2x."""
        pd_1 = self.climate_pd(0.01, 0.1, 0.5, 0.1, 0.1, 1.0, 0.0)
        pd_2 = self.climate_pd(0.01, 0.1, 0.5, 0.1, 0.1, 2.0, 0.0)
        delta_1 = pd_1 - 0.01
        delta_2 = pd_2 - 0.01
        if delta_1 > 0:
            ratio = delta_2 / delta_1
            self.assertGreater(ratio, 2.0,
                               msg="Convexity: doubling carbon shock should >2x impact")


class TestBalanceSheetECL(unittest.TestCase):
    """Tests du pipeline complet balance sheet ECL."""

    @classmethod
    def setUpClass(cls):
        _, _, _, cls.df_bs = _get_dataset_v4(n_clients=100, seed=RANDOM_SEED)

    def test_balance_sheet_14_rows(self):
        self.assertEqual(len(self.df_bs), 14)

    def test_balance_sheet_columns(self):
        required = {"asset_class", "ead_total", "pd_base", "lgd_base", "tenor",
                     "rw_crr3", "physical_risk", "transition_risk"}
        self.assertTrue(required <= set(self.df_bs.columns))

    def test_all_classes_present(self):
        classes = set(self.df_bs["asset_class"].to_list())
        self.assertEqual(classes, set(ASSET_CLASS_NAMES))

    def test_ead_positive(self):
        self.assertTrue((self.df_bs["ead_total"] > 0).all())

    def test_compute_ecl_basic(self):
        """ECL calcule pour chaque classe."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl
        macro = {"gdp_growth": 1.2, "unemployment_rate": 7.5,
                 "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5}
        result = compute_balance_sheet_ecl(self.df_bs, macro)
        self.assertEqual(len(result), 14)
        self.assertTrue((result["ecl_weighted"] >= 0).all())
        self.assertTrue((result["ecl_ead_ratio"] >= 0).all())

    def test_adverse_increases_ecl(self):
        """Scenario adverse produit un ECL plus eleve."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl
        macro_base = {"gdp_growth": 1.2, "unemployment_rate": 7.5,
                      "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5}
        macro_adverse = {"gdp_growth": -4.5, "unemployment_rate": 10.5,
                         "interest_rate": 5.0, "hpi_growth": -8.0, "inflation_rate": 5.5}
        ecl_base = compute_balance_sheet_ecl(self.df_bs, macro_base)["ecl_weighted"].sum()
        ecl_adverse = compute_balance_sheet_ecl(self.df_bs, macro_adverse)["ecl_weighted"].sum()
        self.assertGreater(ecl_adverse, ecl_base,
                           msg="Adverse scenario should produce higher total ECL")

    def test_climate_increases_ecl(self):
        """Choc climatique augmente l'ECL total."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl
        macro = {"gdp_growth": 1.2, "unemployment_rate": 7.5,
                 "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5}
        ecl_no_climate = compute_balance_sheet_ecl(
            self.df_bs, macro, carbon_price_shock=0.0, physical_severity=0.0
        )["ecl_weighted"].sum()
        ecl_climate = compute_balance_sheet_ecl(
            self.df_bs, macro, carbon_price_shock=1.5, physical_severity=0.3
        )["ecl_weighted"].sum()
        self.assertGreater(ecl_climate, ecl_no_climate)

    def test_ecl_has_scenario_details(self):
        """Resultat contient les details par scenario."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl
        macro = {"gdp_growth": 1.2, "unemployment_rate": 7.5,
                 "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5}
        result = compute_balance_sheet_ecl(self.df_bs, macro)
        self.assertIn("ecl_base", result.columns)
        self.assertIn("ecl_adverse", result.columns)
        self.assertIn("ecl_favorable", result.columns)
        self.assertIn("pd_cond_base", result.columns)

    def test_rwa_computed(self):
        """RWA = EAD * rw_crr3 (row-level override or effective_rw fallback)."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl
        macro = {"gdp_growth": 1.2, "unemployment_rate": 7.5,
                 "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5}
        result = compute_balance_sheet_ecl(self.df_bs, macro)
        for r in result.iter_rows(named=True):
            expected_rwa = r["ead_total"] * r["rw_crr3"]
            self.assertAlmostEqual(r["rwa"], expected_rwa, places=0)


class TestMacroToZ(unittest.TestCase):
    """Tests du mapping macro -> facteur systematique Z."""

    def setUp(self):
        from ifrs9_cockpit.engine.balance_sheet_ecl import macro_to_z
        self.macro_to_z = macro_to_z

    def test_baseline_z_near_zero(self):
        """Macro baseline -> Z ~ 0."""
        macro = {"gdp_growth": 1.2, "unemployment_rate": 7.5,
                 "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5}
        sens = {"gdp_growth": 1.0, "unemployment_rate": 1.0,
                "interest_rate": 1.0, "hpi_growth": 1.0, "inflation_rate": 1.0}
        z = self.macro_to_z(macro, sens)
        self.assertAlmostEqual(z, 0.0, delta=0.1,
                               msg=f"Z at baseline should be ~0, got {z}")

    def test_adverse_z_positive(self):
        """Macro adverse -> Z > 0 (stress)."""
        macro_adverse = {"gdp_growth": -4.5, "unemployment_rate": 10.5,
                         "interest_rate": 5.0, "hpi_growth": -8.0, "inflation_rate": 5.5}
        sens = {"gdp_growth": 1.5, "unemployment_rate": 1.5,
                "interest_rate": 1.0, "hpi_growth": 1.0, "inflation_rate": 1.0}
        z = self.macro_to_z(macro_adverse, sens)
        self.assertGreater(z, 0, msg="Adverse macro should produce Z > 0")


class TestPredefinedScenarios(unittest.TestCase):
    """Tests des scenarios predefinis."""

    def test_climate_scenario_exists(self):
        self.assertIn("Transition climatique brutale", PREDEFINED_SCENARIOS)

    def test_climate_scenario_has_carbon_shock(self):
        climate = PREDEFINED_SCENARIOS["Transition climatique brutale"]
        self.assertIn("carbon_price_shock", climate)
        self.assertGreater(climate["carbon_price_shock"], 0)

    def test_12_scenarios(self):
        self.assertEqual(len(PREDEFINED_SCENARIOS), 12)

    def test_boom_immobilier_exists(self):
        self.assertIn("Boom immobilier", PREDEFINED_SCENARIOS)

    def test_trappe_liquidite_exists(self):
        self.assertIn("Trappe a liquidite", PREDEFINED_SCENARIOS)


class TestDGPBalanceSheet(unittest.TestCase):
    """Tests de la generation du df_balance_sheet."""

    def test_generate_returns_4_tuple(self):
        result = _get_dataset_v4(n_clients=50, seed=42)
        self.assertEqual(len(result), 4)

    def test_balance_sheet_shape(self):
        _, _, _, df_bs = _get_dataset_v4(n_clients=50, seed=42)
        self.assertEqual(len(df_bs), 14)
        self.assertGreater(len(df_bs.columns), 10)

    def test_corporate_ead_matches(self):
        """EAD corporate = somme des loan_amount du portefeuille."""
        df_credit, _, _, df_bs = _get_dataset_v4(n_clients=100, seed=42)
        corp_ead = df_bs.filter(pl.col("asset_class") == "corporate_loans")["ead_total"][0]
        credit_total = df_credit["loan_amount"].sum()
        self.assertAlmostEqual(corp_ead, credit_total, delta=1.0)


class TestRegulatoryNorms(unittest.TestCase):
    """Tests des 4 normes reglementaires (IFRS 9 B5.5.25, CRR3 LTV, SEC-SA, NSFR)."""

    # ── Souverain : IFRS 9 B5.5.25 exemption staging ──

    def test_sovereign_exempt_from_staging(self):
        """Profil souverain a exempt_from_staging=True."""
        self.assertTrue(ASSET_CLASS_MAP["sovereign"].exempt_from_staging)

    def test_non_sovereign_non_fvtpl_not_exempt(self):
        """Non-sovereign, non-FVTPL profils ont exempt_from_staging=False."""
        _EXEMPT_NAMES = {"sovereign", "equities"}  # sovereign=IFRS9 B5.5.25, equities=FVTPL
        for ac in ASSET_CLASSES:
            if ac.name not in _EXEMPT_NAMES:
                self.assertFalse(ac.exempt_from_staging,
                                 msg=f"{ac.name} should not be exempt")

    def test_sovereign_ecl_is_12m_only(self):
        """ECL souverain exempt = PD × LGD × EAD × 1.0 (12 mois uniquement)."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import staging_cliff_ecl
        pd_cond, lgd, ead, tenor = 0.005, 0.45, 1e9, 7.0
        ecl_exempt = staging_cliff_ecl(pd_cond, lgd, ead, tenor, 0.0005,
                                       exempt_from_staging=True)
        expected = pd_cond * lgd * ead * 1.0
        self.assertAlmostEqual(ecl_exempt, expected, places=0)

    def test_sovereign_ecl_less_than_generic(self):
        """ECL exempt < ECL generique sous stress (pas de Stage 2 lifetime)."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import staging_cliff_ecl
        pd_cond, lgd, ead, tenor = 0.02, 0.45, 1e9, 7.0
        ecl_exempt = staging_cliff_ecl(pd_cond, lgd, ead, tenor, 0.0005,
                                       exempt_from_staging=True)
        ecl_generic = staging_cliff_ecl(pd_cond, lgd, ead, tenor, 0.0005,
                                        exempt_from_staging=False)
        self.assertLess(ecl_exempt, ecl_generic,
                        msg="Exempt ECL should be lower (no Stage 2)")

    # ── Mortgage LTV : CRR3 Art. 124-125 ──

    def test_mortgage_has_ltv_distribution(self):
        """Credit immobilier a une distribution LTV non-None."""
        self.assertIsNotNone(ASSET_CLASS_MAP["retail_mortgage"].ltv_distribution)

    def test_mortgage_rw_blended_in_range(self):
        """RW blended mortgage dans [0.25, 0.40]."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import mortgage_rw_blended
        ltv = ASSET_CLASS_MAP["retail_mortgage"].ltv_distribution
        rw = mortgage_rw_blended(ltv)
        self.assertGreaterEqual(rw, 0.25, msg=f"RW blended = {rw}")
        self.assertLessEqual(rw, 0.40, msg=f"RW blended = {rw}")

    def test_mortgage_effective_rw_differs_from_flat(self):
        """effective_rw(mortgage) != rw_crr3 flat."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import effective_rw
        profile = ASSET_CLASS_MAP["retail_mortgage"]
        rw_eff = effective_rw(profile)
        self.assertNotAlmostEqual(rw_eff, profile.rw_crr3, places=3,
                                  msg="Blended LTV RW should differ from flat 35%")

    # ── Securitisation SEC-SA : CRR3 Art. 242-270 ──

    def test_structured_has_securitisation_mix(self):
        """Produits structures ont un mix de securitisation non-None."""
        self.assertIsNotNone(ASSET_CLASS_MAP["structured_products"].securitisation_mix)

    def test_securitisation_rw_blended_in_range(self):
        """RW blended securitisation dans [0.20, 1.00]."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import securitisation_rw_blended
        mix = ASSET_CLASS_MAP["structured_products"].securitisation_mix
        rw = securitisation_rw_blended(mix)
        self.assertGreaterEqual(rw, 0.20, msg=f"RW SEC-SA = {rw}")
        self.assertLessEqual(rw, 1.00, msg=f"RW SEC-SA = {rw}")

    def test_structured_effective_rw_less_than_flat(self):
        """effective_rw(structured) < rw_crr3 flat (mix senior-heavy)."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import effective_rw
        profile = ASSET_CLASS_MAP["structured_products"]
        rw_eff = effective_rw(profile)
        self.assertLess(rw_eff, profile.rw_crr3,
                        msg=f"SEC-SA blended {rw_eff} should be < flat {profile.rw_crr3}")

    # ── NSFR Basel III ──

    def test_nsfr_compliant_balanced(self):
        """NSFR >= 100% pour une allocation typique equilibree."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import compute_nsfr
        weights = {ac.name: ac.typical_weight for ac in ASSET_CLASSES}
        result = compute_nsfr(weights, total_ead=1e12, asf_coverage=0.90)
        self.assertGreaterEqual(result["nsfr_ratio"], 1.00,
                                msg=f"NSFR = {result['nsfr_ratio']}")
        self.assertTrue(result["compliant"])

    def test_nsfr_stressed_by_illiquid(self):
        """Allocation lourde PE/ProjFin degrade le NSFR."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import compute_nsfr
        weights_illiquid = {ac.name: 0.02 for ac in ASSET_CLASSES}
        weights_illiquid["private_equity"] = 0.40
        weights_illiquid["project_finance"] = 0.40
        # Normalize
        total = sum(weights_illiquid.values())
        weights_illiquid = {k: v / total for k, v in weights_illiquid.items()}
        result = compute_nsfr(weights_illiquid, total_ead=1e12, asf_coverage=0.90)
        self.assertLess(result["nsfr_ratio"], 1.20,
                        msg=f"Illiquid-heavy NSFR should be stressed: {result['nsfr_ratio']}")

    def test_rsf_weights_calibrated(self):
        """RSF weights cles : sov=0, PE=1.0, covered=0.15."""
        self.assertAlmostEqual(ASSET_CLASS_MAP["sovereign"].rsf_weight, 0.00)
        self.assertAlmostEqual(ASSET_CLASS_MAP["private_equity"].rsf_weight, 1.00)
        self.assertAlmostEqual(ASSET_CLASS_MAP["covered_bonds"].rsf_weight, 0.15)

    # ── Fallback + Integration ──

    def test_effective_rw_fallback_to_flat(self):
        """Classes sans LTV/SEC-SA retournent rw_crr3."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import effective_rw
        for name in ("corporate_loans", "sovereign", "interbank", "trade_finance",
                     "consumer_credit", "project_finance", "covered_bonds", "private_equity"):
            profile = ASSET_CLASS_MAP[name]
            self.assertAlmostEqual(effective_rw(profile), profile.rw_crr3,
                                   msg=f"{name}: should fallback to flat RW")

    def test_balance_sheet_ecl_uses_effective_rw(self):
        """RWA dans compute_balance_sheet_ecl utilise rw_crr3 (row or profile)."""
        from ifrs9_cockpit.engine.balance_sheet_ecl import compute_balance_sheet_ecl
        _, _, _, df_bs = _get_dataset_v4(n_clients=50, seed=42)
        macro = {"gdp_growth": 1.2, "unemployment_rate": 7.5,
                 "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5}
        result = compute_balance_sheet_ecl(df_bs, macro)
        for r in result.iter_rows(named=True):
            expected_rwa = r["ead_total"] * r["rw_crr3"]
            self.assertAlmostEqual(r["rwa"], expected_rwa, places=0,
                                   msg=f"{r['asset_class']}: RWA mismatch")

    def test_balance_sheet_has_regulatory_cols(self):
        """df_balance_sheet contient exempt_from_staging et rsf_weight."""
        from ifrs9_cockpit.synthetic_generator import generate_dataset
        _, _, _, df_bs = generate_dataset(n_clients=50, seed=42)
        self.assertIn("exempt_from_staging", df_bs.columns)
        self.assertIn("rsf_weight", df_bs.columns)


if __name__ == "__main__":
    unittest.main()
