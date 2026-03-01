"""Mixin sensibilite CRR3, allocation regime-switching et seuils de basculement."""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Dict, Optional, Tuple

from ifrs9_cockpit.config import (
    BASEL_CONFIG,
    SECTORS,
    SCENARIO_BASE,
    SECTOR_NAMES,
    RISK_APPETITE_CONFIG,
    _EXPERT_CORR,
)
from ifrs9_cockpit.engine.comparator.crr3 import compute_crr3_rw


class SensitivityMixin:
    """Mixin fournissant la sensibilite CRR3, l'allocation regime et les tipping points."""

    # ──────────────────────────────────────────────
    # SENSIBILITE CRR3 (FR23)
    # ──────────────────────────────────────────────

    def compute_crr3_sensitivity(self) -> pl.DataFrame:
        """Analyse de sensibilite sous les 3 risk weights PE CRR3 (FR23).

        Boucle sur BASEL_CONFIG.rw_pe_options : 190%, 250%, 400%.
        Calcule CET1 ratio et headroom pour chaque RW.
        Ajoute aussi le RW composite CRR3 (H8) par position.

        Returns:
            DataFrame avec colonnes rw_pe, rwa_pe, rwa_total, cet1_ratio, headroom.
        """
        rwa_credit_total = self.result_credit["rwa_credit"].sum()
        rwa_pe_current = self.result_pe["rwa_pe"].sum()
        coc = BASEL_CONFIG.cet1_target
        # Capital = fonds propres reels = (RWA_credit + RWA_PE_courant) x CET1_target
        # La sensibilite montre comment le ratio change si le RW PE est reclasse.
        actual_capital = (rwa_credit_total + rwa_pe_current) * coc
        records = []

        # H8 : RWA PE via score CRR3 composite (position par position)
        crr3_rw = compute_crr3_rw(self.result_pe)
        rwa_pe_composite = float(np.sum(
            self.result_pe["nav"].to_numpy() * crr3_rw / 100.0
        ))

        for rw_pe in BASEL_CONFIG.rw_pe_options:
            # Scenario fixe (stress test uniforme)
            nav_total = self.result_pe["nav"].sum()
            rwa_pe = nav_total * rw_pe / 100.0
            rwa_total = rwa_credit_total + rwa_pe
            cet1 = actual_capital / max(rwa_total, 1)
            headroom = round(cet1 - coc, 4)

            records.append({
                "rw_pe": rw_pe,
                "rwa_pe": round(rwa_pe, 0),
                "rwa_credit": round(rwa_credit_total, 0),
                "rwa_total": round(rwa_total, 0),
                "cet1_ratio": round(cet1, 4),
                "headroom": headroom,
                "feasible": headroom >= 0,
            })

        # Ligne supplementaire : RW composite CRR3 (H8)
        rwa_total_composite = rwa_credit_total + rwa_pe_composite
        cet1_composite = actual_capital / max(rwa_total_composite, 1)
        headroom_composite = round(cet1_composite - coc, 4)
        rw_moyen = int(np.round(np.mean(crr3_rw)))

        records.append({
            "rw_pe": rw_moyen,
            "rwa_pe": round(rwa_pe_composite, 0),
            "rwa_credit": round(rwa_credit_total, 0),
            "rwa_total": round(rwa_total_composite, 0),
            "cet1_ratio": round(cet1_composite, 4),
            "headroom": headroom_composite,
            "feasible": headroom_composite >= 0,
        })

        return pl.DataFrame(records)

    # ──────────────────────────────────────────────
    # REGIME-SWITCHING ALLOCATION (HMM + GFlowNet)
    # ──────────────────────────────────────────────

    def compute_regime_allocation(
        self,
        regime_result: object,
        gflownet_result: object = None,
    ) -> Dict[str, object]:
        """Allocation conditionnee par le regime HMM detecte.

        Adapte l'allocation credit/PE selon le regime macroeconomique :
            - Contraction : PE plafonne a 5%, surponderation secteurs defensifs
            - Recovery : PE a 15%, allocation equilibree
            - Expansion : PE jusqu'a 30%, identification alpha + capital relief

        Args:
            regime_result: HMMResult avec regime, scenario_weights,
                tau_multiplier, cvar_alpha, omega_scaling.
            gflownet_result: GFlowNetResult optionnel (scenarios generes).

        Returns:
            Dict avec allocation, signaux par secteur, metriques regime.
        """
        regime = getattr(regime_result, "regime", "recovery")
        regime_id = getattr(regime_result, "regime_id", 1)
        scenario_weights = getattr(regime_result, "scenario_weights", {})
        tau_mult = getattr(regime_result, "tau_multiplier", 1.0)
        probabilities = getattr(regime_result, "probabilities", [0.0, 1.0, 0.0])

        raroc_df = self.compute_raroc_eva()
        coc = BASEL_CONFIG.cet1_target

        # ── PE ceiling par regime ──
        pe_ceiling = {"contraction": 0.05, "recovery": 0.15, "expansion": 0.30}
        pe_max = pe_ceiling.get(regime, 0.15)
        pe_max = min(pe_max, BASEL_CONFIG.pe_max_allocation)

        # ── RAROC totaux ──
        total_credit = raroc_df.filter(
            (pl.col("canal") == "Credit") & (pl.col("sector") == "Total")
        )
        total_pe = raroc_df.filter(
            (pl.col("canal") == "PE") & (pl.col("sector") == "Total")
        )
        raroc_c = total_credit["raroc"].to_numpy()[0] if len(total_credit) > 0 else 0.0
        raroc_p = total_pe["raroc"].to_numpy()[0] if len(total_pe) > 0 else 0.0

        # ── PE allocation regime-conditionne ──
        if regime == "contraction":
            pe_alloc = pe_max  # Minimiser PE
        elif regime == "expansion" and raroc_p > raroc_c:
            pe_alloc = pe_max  # Maximiser PE si performant
        else:
            # Recovery : proportionnel au RAROC, borne
            ratio = max(0, raroc_p) / max(max(0, raroc_c) + max(0, raroc_p), 1e-6)
            pe_alloc = min(ratio, pe_max)
            pe_alloc = max(pe_alloc, 0.05)

        credit_alloc = 1.0 - pe_alloc

        # ── Marginal RORAC par secteur (signal) ──
        sector_signals = []
        for sector in SECTORS:
            name = sector.name
            rc_df = raroc_df.filter(
                (pl.col("sector") == name) & (pl.col("canal") == "Credit")
            )
            rp_df = raroc_df.filter(
                (pl.col("sector") == name) & (pl.col("canal") == "PE")
            )
            raroc_credit = rc_df["raroc"].to_numpy()[0] if len(rc_df) > 0 else 0.0
            raroc_pe = rp_df["raroc"].to_numpy()[0] if len(rp_df) > 0 else 0.0

            # Cyclicality : gdp_sensitivity as proxy
            cyclicality = (
                sector.gdp_sensitivity_credit + sector.gdp_sensitivity_pe
            ) / 2.0

            # Signal regime-dependant
            if regime == "contraction":
                # Favoriser defensifs (faible cyclicite)
                signal = "surponderer" if cyclicality < 1.0 else "sous-ponderer"
                action = "reduire_PE" if raroc_pe < 0 else "maintenir"
            elif regime == "expansion":
                # Favoriser les secteurs cycliques performants
                if raroc_pe > raroc_credit and raroc_pe > coc:
                    signal = "surponderer"
                    action = "augmenter_PE"
                elif raroc_pe > 0:
                    signal = "maintenir"
                    action = "maintenir"
                else:
                    signal = "sous-ponderer"
                    action = "reduire_PE"
            else:
                # Recovery : neutre
                signal = "maintenir"
                action = "surveiller"

            # Capital relief : candidat titrisation si RORAC marginal < seuil
            capital_relief = raroc_credit < coc and regime == "expansion"

            sector_signals.append({
                "sector": name,
                "raroc_credit": round(raroc_credit, 4),
                "raroc_pe": round(raroc_pe, 4),
                "cyclicality": round(cyclicality, 2),
                "signal": signal,
                "action": action,
                "capital_relief_candidate": capital_relief,
            })

        # ── GFlowNet best scenario integration ──
        gflownet_info = None
        if gflownet_result is not None:
            mode = getattr(gflownet_result, "mode", "unknown")
            best_scenario = getattr(gflownet_result, "best_scenario", {})
            best_reward = getattr(gflownet_result, "best_reward", 0.0)
            diversity = getattr(gflownet_result, "diversity", 0.0)
            n_scenarios = len(getattr(gflownet_result, "scenarios", []))
            gflownet_info = {
                "mode": mode,
                "best_scenario": best_scenario,
                "best_reward": round(best_reward, 4),
                "diversity": round(diversity, 3),
                "n_scenarios": n_scenarios,
            }

        # ── Metriques post-optimisation ──
        rwa_credit_total = self.result_credit["rwa_credit"].sum()
        rwa_pe_total = self.result_pe["rwa_pe"].sum()
        rwa_total = rwa_credit_total + rwa_pe_total
        actual_capital = rwa_total * coc
        cet1_ratio = actual_capital / max(rwa_total, 1)

        return {
            "regime": regime,
            "regime_id": regime_id,
            "regime_probabilities": list(probabilities),
            "tau_multiplier": round(tau_mult, 2),
            "scenario_weights": scenario_weights,
            "pe_ceiling": round(pe_max, 2),
            "credit_allocation": round(credit_alloc, 2),
            "pe_allocation": round(pe_alloc, 2),
            "raroc_credit": round(raroc_c, 4),
            "raroc_pe": round(raroc_p, 4),
            "sector_signals": sector_signals,
            "n_capital_relief": sum(
                1 for s in sector_signals if s["capital_relief_candidate"]
            ),
            "cet1_ratio": round(cet1_ratio, 4),
            "feasible": cet1_ratio >= coc,
            "gflownet": gflownet_info,
        }

    # ──────────────────────────────────────────────
    # SEUILS DE BASCULEMENT (FR24)
    # ──────────────────────────────────────────────

    def find_tipping_points(self) -> pl.DataFrame:
        """Identifie les seuils de basculement par secteur (FR24).

        Pour chaque secteur, determine si PE est preferable au credit
        en termes de RAROC, et le delta de basculement.

        Returns:
            DataFrame avec colonnes sector, raroc_credit, raroc_pe,
            preferred_canal, delta_to_switch.
        """
        raroc_df = self.compute_raroc_eva()
        records = []

        for sector in SECTORS:
            name = sector.name
            rc_df = raroc_df.filter(
                (pl.col("sector") == name) & (pl.col("canal") == "Credit")
            )
            rp_df = raroc_df.filter(
                (pl.col("sector") == name) & (pl.col("canal") == "PE")
            )
            raroc_c = rc_df["raroc"].to_numpy()[0] if len(rc_df) > 0 else 0.0
            raroc_p = rp_df["raroc"].to_numpy()[0] if len(rp_df) > 0 else 0.0

            preferred = "PE" if raroc_p > raroc_c else "Credit"
            delta = abs(raroc_p - raroc_c)

            records.append({
                "sector": name,
                "raroc_credit": raroc_c,
                "raroc_pe": raroc_p,
                "preferred_canal": preferred,
                "delta_to_switch": round(delta, 4),
            })

        return pl.DataFrame(records)
