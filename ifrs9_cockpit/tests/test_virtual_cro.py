"""Tests exhaustifs pour le Virtual CRO NeSy MAS (Etape 4).

14 classes de tests couvrant les 4 couches :
    - Agents neuronaux (MacroAgent, QuantAgent, PEAgent, ContrarianAgent)
    - Fusion Dempster-Shafer (Jousselme, bBPA, 3 regimes)
    - NeSy QBAF (QE model, arguments symboliques, falsification)
    - PMA Engine (calcul, justification, conditions de validite)
    - VirtualCROEngine (integration bout-en-bout)
"""

import numpy as np
import polars as pl
import pytest

from ifrs9_cockpit.config import SCENARIO_BASE, SECTORS


# ──────────────────────────────────────────────
# Test Agents
# ──────────────────────────────────────────────

class TestMacroAgent:
    """Tests pour l'agent Macro."""

    def test_predict_returns_belief_mass(self):
        from ifrs9_cockpit.virtual_cro.agents import MacroAgent
        agent = MacroAgent(seed=42)
        macro = {
            "unemployment_rate": 7.5,
            "gdp_growth": 1.2,
            "interest_rate": 3.5,
            "hpi_growth": 2.0,
            "inflation_rate": 2.5,
        }
        belief = agent.predict(macro)
        assert belief.agent_name == "macro"
        assert len(belief.frame) == 4
        assert "uncertainty" in belief.masses
        assert abs(sum(belief.masses.values()) - 1.0) < 1e-6

    def test_confidence_bounded(self):
        from ifrs9_cockpit.virtual_cro.agents import MacroAgent
        agent = MacroAgent(seed=42)
        macro = {"unemployment_rate": 12.0, "gdp_growth": -5.0,
                 "interest_rate": 5.0, "hpi_growth": -10.0, "inflation_rate": 8.0}
        belief = agent.predict(macro)
        assert 0.3 <= belief.confidence <= 0.95

    def test_masses_non_negative(self):
        from ifrs9_cockpit.virtual_cro.agents import MacroAgent
        agent = MacroAgent(seed=42)
        macro = {"unemployment_rate": 5.0, "gdp_growth": 3.0,
                 "interest_rate": 1.0, "hpi_growth": 5.0, "inflation_rate": 1.5}
        belief = agent.predict(macro)
        for v in belief.masses.values():
            assert v >= 0.0

    def test_reproducible_with_seed(self):
        from ifrs9_cockpit.virtual_cro.agents import MacroAgent
        macro = {"unemployment_rate": 8.0, "gdp_growth": 0.5,
                 "interest_rate": 4.0, "hpi_growth": -2.0, "inflation_rate": 4.0}
        a1 = MacroAgent(seed=42).predict(macro)
        a2 = MacroAgent(seed=42).predict(macro)
        for h in a1.masses:
            assert abs(a1.masses[h] - a2.masses[h]) < 1e-10


class TestQuantAgent:
    """Tests pour l'agent Quant."""

    def _make_result_credit(self, n=100, seed=42):
        rng = np.random.RandomState(seed)
        return pl.DataFrame({
            "pd_12m": rng.uniform(0.01, 0.20, n),
            "stage": rng.choice([1, 2, 3], n, p=[0.7, 0.2, 0.1]),
            "ecl_weighted": rng.uniform(100, 10000, n),
            "ead": rng.uniform(10000, 100000, n),
        })

    def test_predict_returns_belief(self):
        from ifrs9_cockpit.virtual_cro.agents import QuantAgent
        agent = QuantAgent(seed=43)
        result = self._make_result_credit()
        belief = agent.predict(result)
        assert belief.agent_name == "quant"
        assert abs(sum(belief.masses.values()) - 1.0) < 1e-6

    def test_frame_has_3_classes(self):
        from ifrs9_cockpit.virtual_cro.agents import QuantAgent
        agent = QuantAgent(seed=43)
        belief = agent.predict(self._make_result_credit())
        assert len(belief.frame) == 3
        assert "amelioration" in belief.frame
        assert "stable" in belief.frame
        assert "degradation" in belief.frame


class TestPEAgent:
    """Tests pour l'agent PE."""

    def _make_result_pe(self, n=50, seed=42):
        rng = np.random.RandomState(seed)
        nav = rng.uniform(50e6, 200e6, n)
        delta = rng.uniform(-20e6, 10e6, n)
        cats = rng.choice(["Performing", "Watchlist", "Distressed"], n, p=[0.6, 0.3, 0.1])
        return pl.DataFrame({
            "nav": nav, "delta_nav": delta, "risk_category": cats,
            "leverage": rng.uniform(1.0, 4.0, n),
            "vintage": rng.choice([2018, 2019, 2020, 2021, 2022], n),
        })

    def test_predict_returns_belief(self):
        from ifrs9_cockpit.virtual_cro.agents import PEAgent
        agent = PEAgent(seed=44)
        belief = agent.predict(self._make_result_pe())
        assert belief.agent_name == "pe"
        assert abs(sum(belief.masses.values()) - 1.0) < 1e-6

    def test_frame_has_3_classes(self):
        from ifrs9_cockpit.virtual_cro.agents import PEAgent
        agent = PEAgent(seed=44)
        belief = agent.predict(self._make_result_pe())
        assert "sain" in belief.frame
        assert "distress" in belief.frame


class TestContrarianAgent:
    """Tests pour l'agent Contrarian."""

    def test_predict_from_other_agents(self):
        from ifrs9_cockpit.virtual_cro.agents import (
            MacroAgent, QuantAgent, PEAgent, ContrarianAgent, BeliefMass,
        )
        macro = MacroAgent(seed=42).predict({
            "unemployment_rate": 8.0, "gdp_growth": 0.5,
            "interest_rate": 4.0, "hpi_growth": -2.0, "inflation_rate": 4.0,
        })
        quant = BeliefMass(
            frame=["amelioration", "stable", "degradation"],
            masses={"amelioration": 0.2, "stable": 0.3, "degradation": 0.3, "uncertainty": 0.2},
            agent_name="quant", confidence=0.7,
        )
        pe = BeliefMass(
            frame=["sain", "watchlist", "distress"],
            masses={"sain": 0.4, "watchlist": 0.3, "distress": 0.1, "uncertainty": 0.2},
            agent_name="pe", confidence=0.8,
        )
        contrarian = ContrarianAgent(seed=99).predict(macro, quant, pe)
        assert contrarian.agent_name == "contrarian"
        assert "consensus_correct" in contrarian.masses
        assert "consensus_wrong" in contrarian.masses
        assert abs(sum(contrarian.masses.values()) - 1.0) < 1e-6

    def test_entropy_zero_for_uniform(self):
        from ifrs9_cockpit.virtual_cro.agents import ContrarianAgent
        # Single mass = 0 entropy
        masses = {"A": 1.0}
        e = ContrarianAgent._entropy(masses)
        assert abs(e) < 1e-6

    def test_entropy_positive_for_mixed(self):
        from ifrs9_cockpit.virtual_cro.agents import ContrarianAgent
        masses = {"A": 0.5, "B": 0.5}
        e = ContrarianAgent._entropy(masses)
        assert e > 0


class TestMLPBase:
    """Tests pour le MLP de base."""

    def test_forward_produces_probability(self):
        from ifrs9_cockpit.virtual_cro.agents import _MLPBase
        mlp = _MLPBase(input_dim=5, hidden1=8, hidden2=4, output_dim=3, seed=42)
        x = np.array([0.5, 0.3, 0.7, 0.1, 0.9])
        probs = mlp.forward(x)
        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert np.all(probs >= 0)

    def test_get_set_params(self):
        from ifrs9_cockpit.virtual_cro.agents import _MLPBase
        mlp = _MLPBase(input_dim=3, hidden1=4, hidden2=2, output_dim=2, seed=42)
        params = mlp.get_params()
        assert "W1" in params
        assert "b3" in params
        mlp2 = _MLPBase(input_dim=3, hidden1=4, hidden2=2, output_dim=2, seed=99)
        mlp2.set_params(params)
        x = np.array([0.5, 0.3, 0.7])
        np.testing.assert_array_almost_equal(mlp.forward(x), mlp2.forward(x))


# ──────────────────────────────────────────────
# Test Fusion
# ──────────────────────────────────────────────

class TestDSFusion:
    """Tests pour la fusion Dempster-Shafer."""

    def _make_belief(self, name, masses, confidence=0.8):
        from ifrs9_cockpit.virtual_cro.agents import BeliefMass
        frame = [k for k in masses if k != "uncertainty"]
        return BeliefMass(frame=frame, masses=masses, agent_name=name, confidence=confidence)

    def test_jousselme_distance_identical(self):
        from ifrs9_cockpit.virtual_cro.fusion import DSFusion
        m = {"A": 0.4, "B": 0.3, "uncertainty": 0.3}
        assert DSFusion.jousselme_distance(m, m) < 1e-10

    def test_jousselme_distance_symmetric(self):
        from ifrs9_cockpit.virtual_cro.fusion import DSFusion
        m1 = {"A": 0.6, "B": 0.2, "uncertainty": 0.2}
        m2 = {"A": 0.2, "B": 0.5, "uncertainty": 0.3}
        d12 = DSFusion.jousselme_distance(m1, m2)
        d21 = DSFusion.jousselme_distance(m2, m1)
        assert abs(d12 - d21) < 1e-10

    def test_jousselme_distance_range(self):
        from ifrs9_cockpit.virtual_cro.fusion import DSFusion
        m1 = {"A": 1.0}
        m2 = {"B": 1.0}
        d = DSFusion.jousselme_distance(m1, m2)
        assert 0 < d <= 1.0

    def test_fuse_single_belief(self):
        from ifrs9_cockpit.virtual_cro.fusion import DSFusion
        b = self._make_belief("agent1", {"A": 0.6, "B": 0.2, "uncertainty": 0.2})
        result = DSFusion().fuse([b])
        assert result.rule_used == "single"
        assert result.conflict_level == 0.0

    def test_fuse_empty(self):
        from ifrs9_cockpit.virtual_cro.fusion import DSFusion
        result = DSFusion().fuse([])
        assert result.fused_masses == {"uncertainty": 1.0}

    def test_dempster_agree(self):
        from ifrs9_cockpit.virtual_cro.fusion import DSFusion
        b1 = self._make_belief("a1", {"A": 0.7, "B": 0.1, "uncertainty": 0.2})
        b2 = self._make_belief("a2", {"A": 0.6, "B": 0.2, "uncertainty": 0.2})
        result = DSFusion(macro_severity=0.0).fuse([b1, b2])
        assert result.rule_used == "dempster"
        assert result.fused_masses.get("A", 0) > result.fused_masses.get("B", 0)

    def test_high_conflict_triggers_yager(self):
        from ifrs9_cockpit.virtual_cro.fusion import DSFusion
        b1 = self._make_belief("a1", {"A": 0.95, "uncertainty": 0.05}, confidence=0.95)
        b2 = self._make_belief("a2", {"B": 0.95, "uncertainty": 0.05}, confidence=0.95)
        result = DSFusion(macro_severity=0.0).fuse([b1, b2])
        # High conflict should trigger yager or pcr
        assert result.rule_used in ("yager", "pcr")

    def test_dynamic_threshold_decreases_in_crisis(self):
        from ifrs9_cockpit.virtual_cro.fusion import DSFusion
        tau_normal = DSFusion(macro_severity=0.0).dynamic_threshold
        tau_crisis = DSFusion(macro_severity=1.0).dynamic_threshold
        assert tau_crisis < tau_normal

    def test_fused_masses_sum_to_one(self):
        from ifrs9_cockpit.virtual_cro.fusion import DSFusion
        b1 = self._make_belief("a1", {"A": 0.5, "B": 0.3, "uncertainty": 0.2})
        b2 = self._make_belief("a2", {"A": 0.4, "B": 0.4, "uncertainty": 0.2})
        b3 = self._make_belief("a3", {"A": 0.3, "B": 0.5, "uncertainty": 0.2})
        result = DSFusion().fuse([b1, b2, b3])
        total = sum(result.fused_masses.values())
        assert abs(total - 1.0) < 1e-4

    def test_bbpa_discounting(self):
        from ifrs9_cockpit.virtual_cro.fusion import DSFusion
        b = self._make_belief("a1", {"A": 0.8, "uncertainty": 0.2}, confidence=0.5)
        discounted = DSFusion._bbpa_discount(b)
        # After discounting with conf=0.5, mass(A) should be 0.8 * 0.5 = 0.4
        assert abs(discounted.masses["A"] - 0.4) < 1e-6
        assert abs(sum(discounted.masses.values()) - 1.0) < 1e-6


# ──────────────────────────────────────────────
# Test NeSy QBAF
# ──────────────────────────────────────────────

class TestNeSyQBAF:
    """Tests pour le NeSy QBAF."""

    def _make_fusion_result(self, favorable=0.5, defavorable=0.2):
        from ifrs9_cockpit.virtual_cro.fusion import FusionResult
        from ifrs9_cockpit.virtual_cro.agents import BeliefMass
        return FusionResult(
            fused_masses={
                "favorable": favorable,
                "neutre": 0.1,
                "defavorable": defavorable,
                "uncertainty": 1.0 - favorable - defavorable - 0.1,
            },
            conflict_level=0.2,
            rule_used="dempster",
            pairwise_distances={},
            dynamic_threshold=0.45,
            agent_beliefs=[],
        )

    def test_evaluate_returns_result(self):
        from ifrs9_cockpit.virtual_cro.qbaf import NeSyQBAF
        qbaf = NeSyQBAF()
        qbaf.build_graph(
            fusion_result=self._make_fusion_result(),
            contrarian_wrong_prob=0.1,
            stage3_pct=0.02,
            sicr_pct=0.15,
            rst_distance=4.0,
            ecl_ead_ratio=0.03,
        )
        result = qbaf.evaluate()
        assert result.recommendation in ["maintenir", "surveiller", "reduire", "escalader"]
        assert -1 <= result.recommendation_strength <= 1
        assert result.convergence_iterations > 0

    def test_symbolic_override_forces_escalade(self):
        from ifrs9_cockpit.virtual_cro.qbaf import NeSyQBAF
        qbaf = NeSyQBAF()
        qbaf.build_graph(
            fusion_result=self._make_fusion_result(favorable=0.8, defavorable=0.05),
            contrarian_wrong_prob=0.05,
            stage3_pct=0.10,  # > 5% threshold -> symbolic override
            sicr_pct=0.30,
            rst_distance=4.0,
            ecl_ead_ratio=0.03,
        )
        result = qbaf.evaluate()
        # IFRS 9 hard constraint: Stage 3 > 5% forces escalade
        assert result.recommendation == "escalader"
        assert "ifrs9_hard_stage3" in result.symbolic_overrides

    def test_symbolic_arguments_are_fixed(self):
        from ifrs9_cockpit.virtual_cro.qbaf import NeSyQBAF
        qbaf = NeSyQBAF()
        qbaf.build_graph(
            fusion_result=self._make_fusion_result(),
            contrarian_wrong_prob=0.1,
            stage3_pct=0.08,
            sicr_pct=0.30,
            rst_distance=1.5,
            ecl_ead_ratio=0.05,
        )
        result = qbaf.evaluate()
        # Symbolic args should keep their base_strength unchanged
        for name, arg in qbaf.arguments.items():
            if arg.is_symbolic:
                assert abs(result.strengths[name] - arg.base_strength) < 1e-10

    def test_convergence_within_max_iter(self):
        from ifrs9_cockpit.virtual_cro.qbaf import NeSyQBAF
        qbaf = NeSyQBAF()
        qbaf.build_graph(
            fusion_result=self._make_fusion_result(),
            contrarian_wrong_prob=0.1,
            stage3_pct=0.02,
            sicr_pct=0.15,
            rst_distance=4.0,
            ecl_ead_ratio=0.03,
        )
        result = qbaf.evaluate()
        assert result.convergence_iterations <= qbaf.MAX_ITER

    def test_falsification_matrix_computed(self):
        from ifrs9_cockpit.virtual_cro.qbaf import NeSyQBAF
        qbaf = NeSyQBAF()
        qbaf.build_graph(
            fusion_result=self._make_fusion_result(),
            contrarian_wrong_prob=0.2,
            stage3_pct=0.02,
            sicr_pct=0.15,
            rst_distance=4.0,
            ecl_ead_ratio=0.03,
        )
        result = qbaf.evaluate()
        assert len(result.falsification_matrix) > 0
        for arg_name, deltas in result.falsification_matrix.items():
            assert "maintenir" in deltas
            assert "escalader" in deltas

    def test_rst_proximity_activates_symbolic(self):
        from ifrs9_cockpit.virtual_cro.qbaf import NeSyQBAF
        qbaf = NeSyQBAF()
        qbaf.build_graph(
            fusion_result=self._make_fusion_result(favorable=0.3, defavorable=0.4),
            contrarian_wrong_prob=0.1,
            stage3_pct=0.02,
            sicr_pct=0.15,
            rst_distance=1.2,  # < 2 sigma -> symbolic
            ecl_ead_ratio=0.05,
        )
        assert qbaf.arguments["rst_proximity"].is_symbolic is True

    def test_strengths_bounded(self):
        from ifrs9_cockpit.virtual_cro.qbaf import NeSyQBAF
        qbaf = NeSyQBAF()
        qbaf.build_graph(
            fusion_result=self._make_fusion_result(),
            contrarian_wrong_prob=0.5,
            stage3_pct=0.15,
            sicr_pct=0.40,
            rst_distance=0.5,
            ecl_ead_ratio=0.08,
        )
        result = qbaf.evaluate()
        for name, strength in result.strengths.items():
            assert -1.0 <= strength <= 1.0, f"{name} out of bounds: {strength}"


# ──────────────────────────────────────────────
# Test PMA
# ──────────────────────────────────────────────

class TestPMAEngine:
    """Tests pour le moteur PMA."""

    def _make_qbaf_result(self, recommendation="surveiller", strength=0.5):
        from ifrs9_cockpit.virtual_cro.qbaf import QBAFResult
        return QBAFResult(
            strengths={"maintenir": 0.2, "surveiller": strength, "reduire": 0.1, "escalader": 0.0},
            recommendation=recommendation,
            recommendation_strength=strength,
            symbolic_overrides=[],
            convergence_iterations=10,
        )

    def test_compute_returns_result(self):
        from ifrs9_cockpit.virtual_cro.pma import PMAEngine
        engine = PMAEngine()
        result = engine.compute(
            ecl_legal=1_000_000,
            qbaf_result=self._make_qbaf_result(),
            fusion_belief=0.7,
            fusion_conflict=0.2,
            macro_severity=0.3,
        )
        assert result.ecl_legal == 1_000_000
        assert result.ecl_committee > 0
        assert isinstance(result.pma_amount, float)
        assert isinstance(result.justification, dict)

    def test_maintenir_no_pma(self):
        from ifrs9_cockpit.virtual_cro.pma import PMAEngine
        engine = PMAEngine()
        result = engine.compute(
            ecl_legal=1_000_000,
            qbaf_result=self._make_qbaf_result("maintenir", 0.8),
            fusion_belief=0.8,
            fusion_conflict=0.1,
        )
        assert abs(result.pma_amount) < 1  # Essentially zero
        assert result.direction == "neutral"

    def test_escalader_increases_ecl(self):
        from ifrs9_cockpit.virtual_cro.pma import PMAEngine
        engine = PMAEngine()
        result = engine.compute(
            ecl_legal=1_000_000,
            qbaf_result=self._make_qbaf_result("escalader", 0.9),
            fusion_belief=0.8,
            fusion_conflict=0.1,
        )
        assert result.pma_amount > 0
        assert result.ecl_committee > result.ecl_legal
        assert result.direction == "increase"

    def test_low_confidence_rejects_pma(self):
        from ifrs9_cockpit.virtual_cro.pma import PMAEngine
        engine = PMAEngine()
        result = engine.compute(
            ecl_legal=1_000_000,
            qbaf_result=self._make_qbaf_result("reduire", 0.1),
            fusion_belief=0.2,  # Very low belief
            fusion_conflict=0.8,  # High conflict
        )
        assert result.is_valid is False
        assert len(result.rejection_reasons) > 0

    def test_justification_has_all_sections(self):
        from ifrs9_cockpit.virtual_cro.pma import PMAEngine
        engine = PMAEngine()
        result = engine.compute(
            ecl_legal=1_000_000,
            qbaf_result=self._make_qbaf_result(),
            fusion_belief=0.7,
            fusion_conflict=0.2,
        )
        assert "resume" in result.justification
        assert "consensus" in result.justification
        assert "reglementaire" in result.justification
        assert "sensibilite" in result.justification
        assert "contexte_macro" in result.justification

    def test_pma_ratio_bounded(self):
        from ifrs9_cockpit.virtual_cro.pma import PMAEngine
        engine = PMAEngine()
        result = engine.compute(
            ecl_legal=1_000_000,
            qbaf_result=self._make_qbaf_result("escalader", 1.0),
            fusion_belief=0.9,
            fusion_conflict=0.05,
            macro_severity=1.0,
        )
        # PMA ratio should be within tolerance or marked invalid
        if result.is_valid:
            assert abs(result.pma_ratio) <= 0.25 + 0.01  # small tolerance


# ──────────────────────────────────────────────
# Test Engine (integration)
# ──────────────────────────────────────────────

class TestVirtualCROEngine:
    """Tests d'integration bout-en-bout du VirtualCROEngine."""

    def _make_data(self, n_credit=200, n_pe=30, seed=42):
        rng = np.random.RandomState(seed)
        sectors = ["Technologie", "Industrie", "Sante", "Immobilier", "Services"]
        df_credit = pl.DataFrame({
            "enterprise_id": list(range(n_credit)),
            "sector": rng.choice(sectors, n_credit).tolist(),
            "pd_12m": rng.uniform(0.01, 0.15, n_credit),
            "pd_lifetime": rng.uniform(0.03, 0.30, n_credit),
            "lgd": rng.uniform(0.20, 0.60, n_credit),
            "ead": rng.uniform(100000, 5000000, n_credit),
            "ecl_weighted": rng.uniform(500, 50000, n_credit),
            "stage": rng.choice([1, 2, 3], n_credit, p=[0.7, 0.2, 0.1]).tolist(),
            "rwa_credit": rng.uniform(100000, 5000000, n_credit),
            "segment": rng.choice(sectors, n_credit).tolist(),
        })
        df_pe = pl.DataFrame({
            "enterprise_id": list(range(n_pe)),
            "sector": rng.choice(sectors, n_pe).tolist(),
            "nav": rng.uniform(50e6, 200e6, n_pe),
            "delta_nav": rng.uniform(-20e6, 10e6, n_pe),
            "expected_loss_pe": rng.uniform(0.5e6, 10e6, n_pe),
            "risk_category": rng.choice(
                ["Performing", "Watchlist", "Distressed"], n_pe, p=[0.6, 0.3, 0.1],
            ).tolist(),
            "rwa_pe": rng.uniform(100e6, 500e6, n_pe),
            "leverage": rng.uniform(1.0, 4.0, n_pe),
            "vintage": rng.choice([2018, 2019, 2020, 2021, 2022], n_pe).tolist(),
        })
        macro = {
            "unemployment_rate": 7.5,
            "gdp_growth": 1.2,
            "interest_rate": 3.5,
            "hpi_growth": 2.0,
            "inflation_rate": 2.5,
        }
        return df_credit, df_pe, macro

    def test_run_returns_result(self):
        from ifrs9_cockpit.virtual_cro import VirtualCROEngine
        df_credit, df_pe, macro = self._make_data()
        engine = VirtualCROEngine(seed=42)
        result = engine.run(df_credit, df_pe, macro, rst_distance=4.0)
        assert result.recommendation in ["maintenir", "surveiller", "reduire", "escalader"]
        assert result.summary != ""
        assert result.macro_severity >= 0

    def test_agent_beliefs_present(self):
        from ifrs9_cockpit.virtual_cro import VirtualCROEngine
        df_credit, df_pe, macro = self._make_data()
        result = VirtualCROEngine(seed=42).run(df_credit, df_pe, macro)
        assert "macro" in result.agent_beliefs
        assert "quant" in result.agent_beliefs
        assert "pe" in result.agent_beliefs
        assert "contrarian" in result.agent_beliefs

    def test_fusion_result_present(self):
        from ifrs9_cockpit.virtual_cro import VirtualCROEngine
        df_credit, df_pe, macro = self._make_data()
        result = VirtualCROEngine(seed=42).run(df_credit, df_pe, macro)
        assert result.fusion_result is not None
        assert result.fusion_result.rule_used in ("dempster", "pcr", "yager", "single")

    def test_pma_result_present(self):
        from ifrs9_cockpit.virtual_cro import VirtualCROEngine
        df_credit, df_pe, macro = self._make_data()
        result = VirtualCROEngine(seed=42).run(df_credit, df_pe, macro)
        assert result.pma_result is not None
        assert result.pma_result.ecl_legal > 0

    def test_crisis_scenario_increases_severity(self):
        from ifrs9_cockpit.virtual_cro import VirtualCROEngine
        df_credit, df_pe, _ = self._make_data()
        normal = {"unemployment_rate": 7.5, "gdp_growth": 1.2,
                  "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5}
        crisis = {"unemployment_rate": 12.0, "gdp_growth": -4.0,
                  "interest_rate": 5.0, "hpi_growth": -10.0, "inflation_rate": 7.0}
        engine = VirtualCROEngine(seed=42)
        r_normal = engine.run(df_credit, df_pe, normal)
        r_crisis = engine.run(df_credit, df_pe, crisis)
        assert r_crisis.macro_severity > r_normal.macro_severity

    def test_reproducible_with_seed(self):
        from ifrs9_cockpit.virtual_cro import VirtualCROEngine
        df_credit, df_pe, macro = self._make_data()
        r1 = VirtualCROEngine(seed=42).run(df_credit, df_pe, macro)
        r2 = VirtualCROEngine(seed=42).run(df_credit, df_pe, macro)
        assert r1.recommendation == r2.recommendation
        assert abs(r1.recommendation_strength - r2.recommendation_strength) < 1e-10

    def test_compute_macro_severity(self):
        from ifrs9_cockpit.virtual_cro.engine import VirtualCROEngine
        # Normal conditions -> low severity
        normal = {"unemployment_rate": 7.5, "gdp_growth": 1.2,
                  "interest_rate": 3.5, "hpi_growth": 2.0, "inflation_rate": 2.5}
        sev = VirtualCROEngine._compute_macro_severity(normal)
        assert 0.0 <= sev <= 0.2

        # Crisis -> high severity
        crisis = {"unemployment_rate": 12.0, "gdp_growth": -6.0,
                  "interest_rate": 5.0, "hpi_growth": -10.0, "inflation_rate": 8.0}
        sev_crisis = VirtualCROEngine._compute_macro_severity(crisis)
        assert sev_crisis > 0.4

    def test_unify_frames(self):
        from ifrs9_cockpit.virtual_cro.engine import VirtualCROEngine
        from ifrs9_cockpit.virtual_cro.agents import BeliefMass

        beliefs = [
            BeliefMass(
                frame=["expansion", "normal", "recession", "crise"],
                masses={"expansion": 0.3, "normal": 0.3, "recession": 0.2,
                        "crise": 0.1, "uncertainty": 0.1},
                agent_name="macro", confidence=0.8,
            ),
        ]
        unified = VirtualCROEngine._unify_frames(beliefs)
        assert len(unified) == 1
        u = unified[0]
        assert "favorable" in u.masses
        assert "neutre" in u.masses
        assert "defavorable" in u.masses
        assert abs(sum(u.masses.values()) - 1.0) < 1e-6
        assert abs(u.masses["favorable"] - 0.3) < 1e-6  # expansion -> favorable
        assert abs(u.masses["neutre"] - 0.3) < 1e-6     # normal -> neutre
        assert abs(u.masses["defavorable"] - 0.3) < 1e-6  # recession + crise


# ──────────────────────────────────────────────
# Test Charts (smoke tests)
# ──────────────────────────────────────────────

class TestVCROCharts:
    """Smoke tests pour les 4 charts VCRO."""

    def test_plot_agent_agreement(self):
        from ifrs9_cockpit.dashboard.charts import plot_vcro_agent_agreement
        from ifrs9_cockpit.virtual_cro.agents import BeliefMass
        beliefs = {
            "macro": BeliefMass(["a"], {"a": 0.5, "uncertainty": 0.5}, "macro", 0.7),
            "quant": BeliefMass(["a"], {"a": 0.6, "uncertainty": 0.4}, "quant", 0.8),
            "pe": BeliefMass(["a"], {"a": 0.4, "uncertainty": 0.6}, "pe", 0.6),
            "contrarian": BeliefMass(["a"], {"a": 0.3, "uncertainty": 0.7}, "contrarian", 0.5),
        }
        fig = plot_vcro_agent_agreement(beliefs)
        assert fig is not None

    def test_plot_belief_distribution(self):
        from ifrs9_cockpit.dashboard.charts import plot_vcro_belief_distribution
        fig = plot_vcro_belief_distribution(
            {"favorable": 0.4, "neutre": 0.2, "defavorable": 0.2, "uncertainty": 0.2},
            "dempster", 0.15,
        )
        assert fig is not None

    def test_plot_qbaf_strengths(self):
        from ifrs9_cockpit.dashboard.charts import plot_vcro_qbaf_strengths
        fig = plot_vcro_qbaf_strengths(
            {"maintenir": 0.5, "surveiller": 0.3, "reduire": -0.1, "escalader": -0.2,
             "fusion_favorable": 0.4, "ifrs9_hard_stage3": 0.0},
            [],
        )
        assert fig is not None

    def test_plot_pma_comparison(self):
        from ifrs9_cockpit.dashboard.charts import plot_vcro_pma_comparison
        fig = plot_vcro_pma_comparison(1_000_000, 1_050_000, 50_000, True)
        assert fig is not None
