"""Orchestrateur CRO — Pipeline analytique 2 passes (FR25, FR31, FR32).

Orchestre les 5 couches en 2 passes :
    Passe 1 : exploration (regime inconnu)
    Detection regime (Couche 4)
    Passe 2 : parametres conditionnes au regime detecte

Produit les recommandations explicables (FR31) et la synthese narrative (FR32).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import polars as pl

import numpy as np

from ifrs9_cockpit.config import SCENARIO_BASE, SECTORS, BASEL_CONFIG, LOGIT_AMPLITUDE, RISK_APPETITE_CONFIG
from ifrs9_cockpit.utils.helpers import logit, expit
from ifrs9_cockpit.ai_analyst.types import (
    AnalyticsState,
    Recommendation,
    RegimeClassification,
)
from ifrs9_cockpit.ai_analyst.layer1_crossing import analyze_crossings
from ifrs9_cockpit.ai_analyst.layer2_euler import decompose_proportional
from ifrs9_cockpit.ai_analyst.layer3_rst import (
    find_tipping_points,
    reverse_stress_test,
    adversarial_reverse_stress_test,
)
from ifrs9_cockpit.ai_analyst.layer4_regime import classify_regime
from ifrs9_cockpit.ai_analyst.layer5_prospective import (
    project_trajectories,
    compute_risk_appetite,
    compute_early_warning,
)


class CROAnalyst:
    """Orchestrateur du pipeline analytique CRO (5 couches, 2 passes).

    Attributes:
        result_credit: DataFrame resultat credit (ECLCalculator).
        result_pe: DataFrame resultat PE (PECalculator).
        macro_params: Variables macro actuelles.
    """

    def __init__(
        self,
        result_credit: pl.DataFrame,
        result_pe: pl.DataFrame,
        macro_params: Dict[str, float],
        target_ecl: Optional[float] = None,
    ) -> None:
        """Initialise l'analyste CRO.

        Args:
            result_credit: Resultat ECLCalculator.calculate().
            result_pe: Resultat PECalculator.calculate().
            macro_params: Dict des 5 variables macro.
            target_ecl: Seuil ECL personnalise pour RST (FR54). None = default.
        """
        self.result_credit = result_credit
        self.result_pe = result_pe
        self.macro_params = macro_params
        self.target_ecl = target_ecl

    def analyze(self) -> AnalyticsState:
        """Execute le pipeline complet 2 passes (FR25).

        Returns:
            AnalyticsState avec tous les resultats des 5 couches.
        """
        # -- Passe 1 : Exploration (regime inconnu) --
        state_p1 = self._run_layers(regime=None, pass_number=1)

        # -- Detection regime (Couche 4 passe 1) --
        regime = state_p1.regime

        # -- Passe 2 : Conditionnee au regime --
        state_p2 = self._run_layers(regime=regime, pass_number=2)

        # -- Synthese : recommandations + narrative --
        state_p2.recommendations = self._generate_recommendations(state_p2)
        state_p2.narrative = self._generate_narrative(state_p2)

        return state_p2

    def _run_layers(
        self,
        regime: Optional[RegimeClassification],
        pass_number: int,
    ) -> AnalyticsState:
        """Execute les 5 couches.

        Args:
            regime: Classification de regime (None = passe 1).
            pass_number: Numero de passe (1 ou 2).

        Returns:
            AnalyticsState partiel.
        """
        state = AnalyticsState(pass_number=pass_number)

        # Couche 1 — Croisement
        state.asymmetry_matrix, state.marginal_contributions = analyze_crossings(
            self.result_credit, self.result_pe, self.macro_params,
        )

        # Couche 2 — Allocation proportionnelle
        state.proportional_contributions, state.factor_attribution = decompose_proportional(
            self.result_credit, self.result_pe, self.macro_params, regime=regime,
        )

        # Couche 3 — RST (proxy ECL logit pour performance)
        ecl_base = self.result_credit["ecl_weighted"].sum()
        ead_total = self.result_credit["ead"].sum()
        base_ratio = ecl_base / max(ead_total, 1)
        logit_base = float(logit(np.clip(base_ratio, 1e-6, 0.99)))

        # Map variable -> attribut SectorConfig pour les sensibilites credit
        _VAR_TO_SENS = {
            "unemployment_rate": "unemployment_sensitivity_credit",
            "gdp_growth": "gdp_sensitivity_credit",
            "interest_rate": "interest_rate_sensitivity_credit",
            "hpi_growth": "hpi_sensitivity_credit",
            "inflation_rate": "inflation_sensitivity_credit",
        }
        # Variables inversees (baisse = adverse)
        _INVERTED = {"gdp_growth", "hpi_growth"}

        def ecl_proxy(params: Dict[str, float]) -> float:
            """Proxy ECL logit coherent avec ECLCalculator.

            Retourne le montant ECL absolu (EUR) pour comparaison
            avec le seuil de rupture RST (capital x 10%).
            """
            stress_sum = 0.0
            for var, sens_attr in _VAR_TO_SENS.items():
                base_val = getattr(SCENARIO_BASE, var)
                delta = params.get(var, base_val) - base_val
                if var in _INVERTED:
                    delta = -delta  # Baisse PIB/HPI = adverse
                # Sensibilite moyenne ponderee par proportion sectorielle
                avg_sens = sum(
                    getattr(s, sens_attr) * s.proportion for s in SECTORS
                )
                stress_sum += (delta / 100) * avg_sens
            ecl_ratio = float(expit(logit_base + stress_sum * LOGIT_AMPLITUDE))
            return ecl_ratio * ead_total  # Montant absolu EUR

        # Seuil tipping = max(ambre absolu, baseline ECL × 1.25)
        # Garantit que le seuil est au-dessus du baseline pour detecter un vrai choc.
        ecl_amber_abs = RISK_APPETITE_CONFIG.ecl_ead_amber * ead_total
        ecl_tipping_threshold = max(ecl_amber_abs, ecl_base * 1.25)
        state.tipping_points = find_tipping_points(
            ecl_proxy, self.macro_params, regime, ecl_threshold=ecl_tipping_threshold,
        )
        # RST retourne la distance de Mahalanobis (H6) depuis layer3_rst
        # Capital reel = RWA_credit × CET1_target (coherent avec comparator.py)
        _actual_capital = self.result_credit["rwa_credit"].sum() * BASEL_CONFIG.cet1_target
        rst = adversarial_reverse_stress_test(
            ecl_proxy, self.macro_params, regime,
            target_ecl=self.target_ecl, capital_base=_actual_capital,
        )
        state.rst_result = rst
        state.rst_distance = rst["rst_distance_sigma"]
        state.pareto_front = rst.get("pareto_front")

        # Couche 4 — Regime
        state.regime = classify_regime(self.macro_params)

        # Couche 5 — Prospective
        state.trajectories = project_trajectories(self.macro_params, regime)
        state.risk_appetite_matrix = compute_risk_appetite(
            self.result_credit, self.result_pe,
        )
        state.early_warning = compute_early_warning(
            self.result_credit, self.result_pe, self.macro_params,
        )

        return state

    def _generate_recommendations(self, state: AnalyticsState) -> List[Recommendation]:
        """Genere les recommandations explicables (FR31).

        Chaine de raisonnement : regime -> declencheur -> allocation proportionnelle ->
        facteur_macro -> risk_appetite -> distance_rst -> confiance.
        """
        recs = []
        regime = state.regime

        if regime is None:
            return recs

        # Top allocation proportionnelle driver
        alloc = state.proportional_contributions
        if alloc is not None and len(alloc) > 0:
            top_proportional = alloc.sort("proportional_share", descending=True).row(0, named=True)
            proportional_driver = f"{top_proportional['sector']} {top_proportional['canal']}"
        else:
            proportional_driver = "N/A"

        # Top factor
        factors = state.factor_attribution
        if factors is not None and len(factors) > 0:
            top_factor = factors.sort("attribution", descending=True).row(0, named=True)
            macro_factor = top_factor["variable"]
        else:
            macro_factor = "N/A"

        # Risk appetite predominant
        ra = state.risk_appetite_matrix
        rouge_count = 0
        if ra is not None and len(ra) > 0:
            rouge_count = int((ra["signal"] == "rouge").sum())
            ra_status = "rouge" if rouge_count > 3 else "ambre" if rouge_count > 1 else "vert"
        else:
            ra_status = "N/A"

        # RST distance (Mahalanobis)
        rst_dist = state.rst_distance

        # Confiance
        if rst_dist < 2.0:
            confidence = "low"
        elif rst_dist < 5.0:
            confidence = "medium"
        else:
            confidence = "high"

        # Recommandation principale
        if ra_status == "rouge":
            action = "Reduire l'exposition aux secteurs en rouge et renforcer les provisions"
            alternatives = [
                "Augmenter les reserves de capital CET1",
                "Activer le hedging sectoriel",
                "Reporter les nouvelles originations",
            ]
        elif ra_status == "ambre":
            action = "Surveiller les indicateurs d'alerte et preparer un plan de contingence"
            alternatives = [
                "Resserrer les criteres d'origination",
                "Diversifier le portefeuille",
            ]
        else:
            action = "Maintenir l'allocation actuelle avec monitoring standard"
            alternatives = ["Optimiser la diversification sectorielle"]

        recs.append(Recommendation(
            action=action,
            regime=regime.detected_regime,
            trigger=f"Risk appetite {ra_status} ({rouge_count} secteurs rouge)"
                if ra is not None else "N/A",
            proportional_driver=proportional_driver,
            macro_factor=macro_factor,
            risk_appetite_status=ra_status,
            rst_distance=rst_dist,
            confidence=confidence,
            alternatives=alternatives,
        ))

        return recs

    def _generate_narrative(self, state: AnalyticsState) -> Dict:
        """Genere la synthese narrative CRO structuree (FR32).

        Retourne un Dict avec 7 sections pour rendu HTML riche :
            - diagnostic: analyse du regime et du positionnement
            - concentration: risques de concentration identifies
            - facteur: facteur macro dominant et impact
            - resilience: evaluation de la solidite du portefeuille
            - action: recommandation principale
            - confiance: niveau de confiance et justification
            - alternatives: actions alternatives

        Returns:
            Dict[str, str] avec les 7 sections narratives.
        """
        regime = state.regime
        rec = state.recommendations[0] if state.recommendations else None
        rst_dist = state.rst_distance
        ra = state.risk_appetite_matrix

        # ── RST escalation logic ──
        ra_status = rec.risk_appetite_status if rec else "vert"
        effective_status = ra_status
        escalation_text = ""
        if rst_dist < 2.0 and ra_status == "ambre":
            effective_status = "rouge"
            escalation_text = (
                " L'escalade de ambre vers rouge est declenchee par la proximite "
                f"du point de rupture RST ({rst_dist:.1f} sigma < 2 sigma)."
            )
        elif rst_dist < 2.0 and ra_status == "vert":
            effective_status = "ambre"
            escalation_text = (
                " Attention : malgre un Risk Appetite vert, la distance RST de "
                f"{rst_dist:.1f} sigma signale une vulnerabilite proche du seuil de rupture."
            )

        # 1. Diagnostic
        diagnostic_parts = []
        if regime:
            prob = regime.probabilities.get(regime.detected_regime, 0)
            diagnostic_parts.append(
                f"Le portefeuille evolue dans un regime de type {regime.detected_regime} "
                f"(probabilite {prob:.0%})."
            )
        if rst_dist < 2.0:
            diagnostic_parts.append(
                f"Le Reverse Stress Test identifie un point de rupture a seulement "
                f"{rst_dist:.1f} sigma, signalant une faible marge de manoeuvre."
            )
        elif rst_dist < 5.0:
            diagnostic_parts.append(
                f"La distance de Mahalanobis au point de rupture est de {rst_dist:.1f} sigma, "
                f"indiquant une resilience moderee."
            )
        else:
            diagnostic_parts.append(
                f"La distance RST de {rst_dist:.1f} sigma confirme une marge de securite "
                f"confortable vis-a-vis du seuil de rupture."
            )
        if escalation_text:
            diagnostic_parts.append(escalation_text)
        diagnostic = " ".join(diagnostic_parts)

        # 2. Concentration
        asym = state.asymmetry_matrix
        concentration_parts = []
        if asym is not None and len(asym) > 0:
            top3 = (
                asym.with_columns(pl.col("asymmetry").abs().alias("_abs_asym"))
                .sort("_abs_asym", descending=True)
                .head(3)
            )
            sectors_list = ", ".join(
                f"{row['sector']} ({row['asymmetry']:+.4f})"
                for row in top3.iter_rows(named=True)
            )
            concentration_parts.append(
                f"Les secteurs a plus forte asymetrie credit/PE sont : {sectors_list}."
            )
            max_asym = top3["asymmetry"].abs().max()
            if max_asym > 0.03:
                concentration_parts.append(
                    "Cette asymetrie elevee suggere un risque de concentration sectorielle "
                    "necessitant une surveillance renforcee."
                )
        concentration = " ".join(concentration_parts) if concentration_parts else (
            "Aucun desequilibre sectoriel significatif identifie."
        )

        # 3. Facteur macro dominant
        factors = state.factor_attribution
        if factors is not None and len(factors) > 0:
            top_f = factors.sort("attribution", descending=True).row(0, named=True)
            facteur = (
                f"Le facteur macro dominant est {top_f['variable']} "
                f"(attribution {top_f['attribution']:.1%} de la variance ECL). "
                f"Les decisions de politique monetaire et les evolutions de ce parametre "
                f"devront etre suivies de pres."
            )
        else:
            facteur = "Aucune attribution factorielle significative identifiee."

        # 4. Resilience
        alloc = state.proportional_contributions
        if alloc is not None and len(alloc) > 0:
            top_cell = alloc.sort("proportional_share", descending=True).row(0, named=True)
            resilience = (
                f"L'allocation proportionnelle revele que {top_cell['sector']} "
                f"{top_cell['canal']} concentre {top_cell['proportional_share']:.1%} "
                f"du risque total du portefeuille. "
            )
            if top_cell['proportional_share'] > 0.30:
                resilience += (
                    "Cette concentration elevee reduit la diversification effective "
                    "et augmente la sensibilite aux chocs sectoriels."
                )
            else:
                resilience += (
                    "La repartition du risque reste suffisamment diversifiee "
                    "pour absorber des chocs sectoriels moderes."
                )
        else:
            resilience = "Evaluation de la resilience non disponible."

        # 5. Action
        if rec:
            if effective_status != ra_status:
                action = (
                    f"[ESCALADE {ra_status.upper()} → {effective_status.upper()}] "
                    f"{rec.action}"
                )
            else:
                action = rec.action
        else:
            action = "Maintenir l'allocation actuelle avec monitoring standard."

        # 6. Confiance
        if rec:
            conf_label = {"high": "elevee", "medium": "moderee", "low": "faible"}.get(
                rec.confidence, rec.confidence
            )
            confiance = (
                f"Niveau de confiance : {conf_label}. "
                f"Regime {rec.regime}, declencheur : {rec.trigger}. "
                f"Facteur macro dominant : {rec.macro_factor}."
            )
        else:
            confiance = "Evaluation de confiance non disponible."

        # 7. Alternatives
        if rec and rec.alternatives:
            alternatives = " | ".join(rec.alternatives)
        else:
            alternatives = "Aucune alternative identifiee."

        return {
            "diagnostic": diagnostic,
            "concentration": concentration,
            "facteur": facteur,
            "resilience": resilience,
            "action": action,
            "confiance": confiance,
            "alternatives": alternatives,
        }


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset
    from ifrs9_cockpit.models.pd_model import PDModelSuite
    from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
    from ifrs9_cockpit.engine.pe_calculator import PECalculator
    from ifrs9_cockpit.models.lgd_model import LGDModel
    from ifrs9_cockpit.models.ead_model import EADModel

    print("=" * 70)
    print("IFRS 9 COCKPIT — AI Analyst : Orchestrateur 2 Passes")
    print("=" * 70)

    # Pipeline
    print("\n[1/3] Pipelines credit/PE...")
    df_credit, df_pe, _, _ = generate_dataset()
    pd_suite = PDModelSuite()
    pd_suite.fit(df_credit)
    pd_current = pd_suite.predict_active(df_credit)
    ecl_calc = ECLCalculator(lgd_model=LGDModel(), ead_model=EADModel())
    result_credit = ecl_calc.calculate(df_credit, pd_current, pd_current * 0.8)
    result_pe = PECalculator().calculate(df_pe)

    macro_params = {
        "unemployment_rate": SCENARIO_BASE.unemployment_rate,
        "gdp_growth": SCENARIO_BASE.gdp_growth,
        "interest_rate": SCENARIO_BASE.interest_rate,
        "hpi_growth": SCENARIO_BASE.hpi_growth,
        "inflation_rate": SCENARIO_BASE.inflation_rate,
    }

    # Analyse 2 passes
    print("\n[2/3] Analyse CRO (2 passes)...")
    analyst = CROAnalyst(result_credit, result_pe, macro_params)
    state = analyst.analyze()

    print(f"\n  Passe      : {state.pass_number}")
    print(f"  Regime     : {state.regime.detected_regime if state.regime else 'N/A'}")
    print(f"  RST dist   : {state.rst_distance:.1f} sigma (Mahalanobis)")
    print(f"  Recs       : {len(state.recommendations)}")

    # Synthese narrative
    print("\n[3/3] Synthese narrative (FR32) :")
    print("-" * 50)
    print(state.narrative)
    print("-" * 50)

    # Validations
    print("\n--- Validations ---")
    all_ok = True

    ok = state.pass_number == 2
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Passe 2 executee (pass_number = {state.pass_number})")
    all_ok &= ok

    ok = state.regime is not None
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Regime detecte")
    all_ok &= ok

    ok = state.asymmetry_matrix is not None and len(state.asymmetry_matrix) == 5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Asymetry matrix 5 secteurs")
    all_ok &= ok

    ok = state.euler_contributions is not None and len(state.euler_contributions) == 10
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Allocation proportionnelle 10 cellules")
    all_ok &= ok

    ok = state.tipping_points is not None and len(state.tipping_points) == 5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Tipping points 5 variables")
    all_ok &= ok

    ok = state.trajectories is not None and len(state.trajectories) == 12
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Trajectoires 12 projections")
    all_ok &= ok

    ok = state.risk_appetite_matrix is not None and len(state.risk_appetite_matrix) == 10
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Risk appetite 10 cellules")
    all_ok &= ok

    ok = len(state.recommendations) >= 1
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Recommandations generees ({len(state.recommendations)})")
    all_ok &= ok

    ok = len(state.narrative) > 50
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Narrative non-vide ({len(state.narrative)} chars)")
    all_ok &= ok

    print(f"\n{'=' * 70}")
    if all_ok:
        print("Orchestrateur 2 passes valide.")
    else:
        print("ATTENTION : certaines validations ont echoue.")
    print(f"{'=' * 70}")
