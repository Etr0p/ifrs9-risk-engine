"""Calculateur ECL (Expected Credit Loss) IFRS 9.

Orchestre le calcul complet de l'ECL en combinant :
    - PD (Probability of Default) par stage et horizon
    - LGD (Loss Given Default) TTC ou Downturn
    - EAD (Exposure At Default) base ou stressée
    - Discount Factor (actualisation à l'EIR)
    - Pondération multi-scénarios (Base 50% + Adverse 25% + Favorable 25%)

Formule ECL par exposition :
    ECL = PD × LGD × EAD × DF

Où l'horizon PD dépend du stage :
    - Stage 1 : PD 12 mois
    - Stage 2/3 : PD lifetime (cumulative sur l'horizon)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.config import (
    IFRS9_CONFIG,
    SCENARIOS,
    SCENARIO_BASE,
    MacroScenario,
    RANDOM_SEED,
    SECTORS,
    BASEL_CONFIG,
    LOGIT_AMPLITUDE,
)
from ifrs9_cockpit.utils.helpers import logit, expit
from ifrs9_cockpit.engine.staging import StagingEngine
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel


# Borne superieure PD pour eviter log(0) dans les calculs hazard rate et Merton
_PD_CLIP_MAX: float = 0.9999


class ECLCalculator:
    """Calculateur ECL multi-scénarios IFRS 9.

    Pipeline :
        1. Calcul des PD par scénario (Base + Adverse + Favorable)
        2. Affectation aux stages via StagingEngine
        3. Calcul LGD (TTC pour Base/Favorable, Downturn pour Adverse)
        4. Calcul EAD (base ou stressée selon scénario)
        5. Actualisation + pondération des scénarios (50/25/25)

    Attributes:
        staging_engine: Moteur de staging.
        lgd_model: Modèle LGD calibré.
        ead_model: Modèle EAD calibré.
        discount_rate: Taux d'actualisation annuel.
        lifetime_years: Horizon lifetime pour Stage 2/3.
        scenarios: Liste des scénarios macroéconomiques.
    """

    def __init__(
        self,
        lgd_model: LGDModel,
        ead_model: EADModel,
        staging_engine: Optional[StagingEngine] = None,
        discount_rate: float = IFRS9_CONFIG.discount_rate,
        lifetime_years: int = IFRS9_CONFIG.lifetime_horizon_years,
        scenarios: Optional[List[MacroScenario]] = None,
    ) -> None:
        """Initialise le calculateur ECL.

        Args:
            lgd_model: Modèle LGD calibré.
            ead_model: Modèle EAD calibré.
            staging_engine: Moteur de staging (créé par défaut si None).
            discount_rate: Taux d'actualisation (EIR proxy).
            lifetime_years: Horizon pour le calcul lifetime.
            scenarios: Scénarios macro (Base + Adverse + Favorable par défaut).
        """
        self.staging_engine = staging_engine or StagingEngine()
        self.lgd_model = lgd_model
        self.ead_model = ead_model
        self.discount_rate = discount_rate
        self.lifetime_years = lifetime_years
        self.scenarios = scenarios or list(SCENARIOS)

    def calculate(
        self,
        df: pd.DataFrame,
        pd_current: np.ndarray,
        pd_origination: np.ndarray,
        unemployment_override: Optional[float] = None,
        gdp_override: Optional[float] = None,
        interest_rate_override: Optional[float] = None,
        hpi_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
    ) -> pd.DataFrame:
        """Calcule l'ECL pour chaque exposition avec pondération multi-scénarios.

        Args:
            df: DataFrame clients avec toutes les features nécessaires.
            pd_current: PD courante (Point-In-Time).
            pd_origination: PD à l'origination.
            unemployment_override: Override du taux de chômage (stress test).
            gdp_override: Override de la croissance PIB (stress test).
            interest_rate_override: Override du taux directeur BCE (stress test).
            hpi_override: Override de la variation prix immobiliers (stress test).
            inflation_override: Override de l'inflation IPC (stress test).

        Returns:
            DataFrame avec colonnes ajoutées : stage, pd_12m, pd_lifetime,
            lgd, ead, discount_factor, ecl_by_scenario, ecl_weighted.
        """
        n = len(df)
        dpd = self._get_dpd(df)
        default_flag = df["default_flag"].values if "default_flag" in df.columns else np.zeros(n)

        # Calculer la PD stressée de référence (scénario Base avec overrides)
        # pour le staging et l'affichage des KPI
        has_override = any(v is not None for v in [
            unemployment_override, gdp_override, interest_rate_override,
            hpi_override, inflation_override,
        ])
        if has_override:
            pd_stressed_ref = self._adjust_pd_for_scenario(
                pd_current, df, self.scenarios[0],
                unemployment_override=unemployment_override,
                gdp_override=gdp_override,
                interest_rate_override=interest_rate_override,
                hpi_override=hpi_override,
                inflation_override=inflation_override,
            )
        else:
            pd_stressed_ref = pd_current.copy()

        # Staging sur les PD stressées (réagit aux sliders)
        stages = self.staging_engine.assign_stages(
            pd_stressed_ref, pd_origination, dpd, default_flag,
        )

        # Calcul ECL par scénario
        ecl_scenarios: Dict[str, np.ndarray] = {}
        scenario_details: Dict[str, Dict[str, np.ndarray]] = {}

        for scenario in self.scenarios:
            # Ajuster les PD selon le scénario macro
            pd_adjusted = self._adjust_pd_for_scenario(
                pd_current, df, scenario,
                unemployment_override=unemployment_override,
                gdp_override=gdp_override,
                interest_rate_override=interest_rate_override,
                hpi_override=hpi_override,
                inflation_override=inflation_override,
            )

            # PD par horizon selon le stage
            pd_12m = pd_adjusted.copy()
            pd_lifetime = self._compute_lifetime_pd(pd_adjusted, stages)

            # PD effective : 12m pour Stage 1, lifetime pour Stage 2/3
            pd_effective = np.where(stages == 1, pd_12m, pd_lifetime)
            # Stage 3 : PD = 1.0 (défaut avéré)
            pd_effective = np.where(stages == 3, 1.0, pd_effective)

            # LGD selon le type de scénario (HPI impacte la valeur du collatéral)
            is_adverse = scenario.name == "Adverse"
            scenario_hpi = hpi_override if hpi_override is not None else scenario.hpi_growth
            lgd = self.lgd_model.predict(df, downturn=is_adverse, hpi_override=scenario_hpi)

            # EAD selon le stress
            ead = self.ead_model.predict(df, stressed=is_adverse)

            # Discount factor PD-marginal weighted (C2)
            df_factor = self._compute_discount_factor(stages, pd_12m)

            # ECL = PD × LGD × EAD × DF
            ecl = pd_effective * lgd * ead * df_factor

            ecl_scenarios[scenario.name] = ecl
            scenario_details[scenario.name] = {
                "pd_effective": pd_effective,
                "lgd": lgd,
                "ead": ead,
                "discount_factor": df_factor,
            }

        # Pondération des scénarios
        ecl_weighted = np.zeros(n)
        for scenario in self.scenarios:
            ecl_weighted += scenario.weight * ecl_scenarios[scenario.name]

        # Construire le résultat
        result = df.copy()
        result["stage"] = stages
        result["pd_12m"] = pd_stressed_ref
        result["pd_lifetime"] = self._compute_lifetime_pd(pd_stressed_ref, stages)

        # Utiliser les détails du scénario Base pour les colonnes principales
        base_details = scenario_details.get("Base", scenario_details[self.scenarios[0].name])
        result["lgd"] = base_details["lgd"]
        result["ead"] = base_details["ead"]
        result["discount_factor"] = base_details["discount_factor"]

        for scenario_name, ecl in ecl_scenarios.items():
            result[f"ecl_{scenario_name.lower()}"] = np.round(ecl, 2)

        result["ecl_weighted"] = np.round(ecl_weighted, 2)

        # RWA credit (SA corporate) = EAD x risk_weight
        result["rwa_credit"] = np.round(
            base_details["ead"] * BASEL_CONFIG.rw_credit, 2
        )

        # Merton credit spread (H4)
        result["credit_spread"] = self._compute_credit_spread(
            pd_stressed_ref, base_details["lgd"],
        )

        return result

    def compute_ecl_summary(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """Résumé de l'ECL par stage et secteur.

        Args:
            result_df: DataFrame résultat de calculate().

        Returns:
            DataFrame agrégé avec ECL total, moyen, coverage ratio.
        """
        summary = (
            result_df.groupby(["stage", "sector"])
            .agg(
                count=("ecl_weighted", "size"),
                ecl_total=("ecl_weighted", "sum"),
                ecl_mean=("ecl_weighted", "mean"),
                ead_total=("ead", "sum"),
                pd_mean=("pd_12m", "mean"),
            )
            .reset_index()
        )
        summary["coverage_ratio"] = np.where(
            summary["ead_total"] > 0,
            summary["ecl_total"] / summary["ead_total"],
            0,
        )
        return summary.round(4)

    def compute_waterfall(
        self,
        ecl_t0: np.ndarray,
        ecl_t1: np.ndarray,
        stages_t0: np.ndarray,
        stages_t1: np.ndarray,
        segments: np.ndarray,
    ) -> pd.DataFrame:
        """Construit le waterfall de variation ECL entre deux dates.

        Décompose la variation ECL en :
            - New business
            - Stage migration
            - PD/LGD/EAD changes
            - Derecognition

        Args:
            ecl_t0: ECL à la date initiale.
            ecl_t1: ECL à la date finale.
            stages_t0: Stages à t0.
            stages_t1: Stages à t1.
            segments: Segments clients.

        Returns:
            DataFrame avec la décomposition du waterfall.
        """
        records = []

        # Total
        total_t0 = ecl_t0.sum()
        total_t1 = ecl_t1.sum()
        delta = total_t1 - total_t0

        # Migration impact (upgrade/downgrade)
        upgraded = (stages_t1 < stages_t0)
        downgraded = (stages_t1 > stages_t0)
        stable = (stages_t1 == stages_t0)

        records.append({
            "component": "ECL Ouverture",
            "amount": round(total_t0, 2),
        })
        records.append({
            "component": "Downgrades (migration)",
            "amount": round((ecl_t1[downgraded] - ecl_t0[downgraded]).sum(), 2),
        })
        records.append({
            "component": "Upgrades (migration)",
            "amount": round((ecl_t1[upgraded] - ecl_t0[upgraded]).sum(), 2),
        })
        records.append({
            "component": "Paramètres (PD/LGD/EAD)",
            "amount": round((ecl_t1[stable] - ecl_t0[stable]).sum(), 2),
        })
        records.append({
            "component": "ECL Clôture",
            "amount": round(total_t1, 2),
        })
        records.append({
            "component": "Variation nette",
            "amount": round(delta, 2),
        })

        return pd.DataFrame(records)

    # ──────────────────────────────────────────
    # MÉTHODES PRIVÉES
    # ──────────────────────────────────────────

    def _get_dpd(self, df: pd.DataFrame) -> np.ndarray:
        """Extrait les DPD du DataFrame (ou zéro si absent).

        Args:
            df: DataFrame clients.

        Returns:
            Array de DPD.
        """
        if "dpd" in df.columns:
            return df["dpd"].values.astype(int)
        return np.zeros(len(df), dtype=int)

    def _adjust_pd_for_scenario(
        self,
        pd_base: np.ndarray,
        df: pd.DataFrame,
        scenario: MacroScenario,
        unemployment_override: Optional[float] = None,
        gdp_override: Optional[float] = None,
        interest_rate_override: Optional[float] = None,
        hpi_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
    ) -> np.ndarray:
        """Ajuste les PD selon le scénario macroéconomique (logit-space).

        Formule Merton-Vasicek :
            logit(PD_stressed) = logit(PD_base) + LOGIT_AMPLITUDE × Σ(shock_i × sens_i)

        Les chocs sont des deltas normalises (positif = adverse) calcules
        depuis les valeurs absolues des variables macro vs baseline.
        La transformation logit garantit PD ∈ ]0, 1[ sans clip.

        5 canaux de transmission : chômage, PIB, taux directeur,
        prix immobiliers, inflation.

        Args:
            pd_base: PD de base (modèle).
            df: DataFrame clients (pour les secteurs).
            scenario: Scénario macroéconomique.
            unemployment_override: Override chômage pour stress test interactif.
            gdp_override: Override PIB pour stress test interactif.
            interest_rate_override: Override taux directeur BCE.
            hpi_override: Override variation prix immobiliers.
            inflation_override: Override inflation IPC.

        Returns:
            PD ajustées pour le scénario, bornées dans ]0, 1[ par expit.
        """
        # Transformer en espace logit (log-odds)
        logit_pd = logit(pd_base)

        base = SCENARIO_BASE

        # Déterminer les valeurs macro effectives
        has_override = any(v is not None for v in [
            unemployment_override, gdp_override, interest_rate_override,
            hpi_override, inflation_override,
        ])
        if has_override:
            # Mode stress test interactif : valeurs absolues ou baseline
            unemp_val = unemployment_override if unemployment_override is not None else base.unemployment_rate
            gdp_val = gdp_override if gdp_override is not None else base.gdp_growth
            ir_val = interest_rate_override if interest_rate_override is not None else base.interest_rate
            hpi_val = hpi_override if hpi_override is not None else base.hpi_growth
            infl_val = inflation_override if inflation_override is not None else base.inflation_rate
        else:
            # Mode scénario prédéfini : valeurs absolues du scénario
            unemp_val = scenario.unemployment_rate
            gdp_val = scenario.gdp_growth
            ir_val = scenario.interest_rate
            hpi_val = scenario.hpi_growth
            infl_val = scenario.inflation_rate

        # Deltas normalisés (positif = adverse, symétrique favorable/adverse)
        # Plus de max(0, ...) : les scénarios favorables réduisent la PD
        unemp_shock = (unemp_val - base.unemployment_rate) / 100
        gdp_shock = (base.gdp_growth - gdp_val) / 100         # Inversé : baisse PIB = adverse
        ir_shock = (ir_val - base.interest_rate) / 100
        hpi_shock = (base.hpi_growth - hpi_val) / 100          # Inversé : baisse HPI = adverse
        infl_shock = (infl_val - base.inflation_rate) / 100

        # Appliquer les chocs par secteur dans l'espace logit
        for sec in SECTORS:
            mask = df["sector"].values == sec.name
            sector_shock = (
                unemp_shock * sec.unemployment_sensitivity_credit
                + gdp_shock * sec.gdp_sensitivity_credit
                + ir_shock * sec.interest_rate_sensitivity_credit
                + hpi_shock * sec.hpi_sensitivity_credit
                + infl_shock * sec.inflation_sensitivity_credit
            )
            # Shift additif en logit-space (expit garantit PD ∈ ]0, 1[)
            logit_pd[mask] += sector_shock * LOGIT_AMPLITUDE

        return expit(logit_pd)

    def _compute_lifetime_pd(
        self,
        pd_12m: np.ndarray,
        stages: np.ndarray,
    ) -> np.ndarray:
        """Convertit la PD 12 mois en PD lifetime via hazard rate (C1).

        Formule exacte (hazard rate constant) :
            h = -log(1 - PD_12m)
            PD_lifetime = 1 - exp(-h * T)

        Ou T = 1 pour Stage 1, lifetime_horizon_years pour Stage 2/3.
        Plus precise que l'approximation (1 - PD)^T pour PD elevees.

        Args:
            pd_12m: PD 12 mois.
            stages: Stages (l'horizon depend du stage).

        Returns:
            PD lifetime.
        """
        # Horizon residuel par stage
        t_residuel = np.where(
            stages == 1,
            1.0,  # Stage 1 : 12 mois
            float(self.lifetime_years),  # Stage 2/3 : horizon lifetime
        )

        # Hazard rate (C1)
        h = -np.log(1 - np.clip(pd_12m, 0, _PD_CLIP_MAX))
        pd_lifetime = 1 - np.exp(-h * t_residuel)
        return np.clip(pd_lifetime, 0, 1)

    def _compute_discount_factor(
        self,
        stages: np.ndarray,
        pd_12m: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Calcule le facteur d'actualisation PD-marginal weighted (C2).

        Stage 1 : actualisation sur 1 an = 1/(1+r)
        Stage 2/3 : DF pondere par la PD marginale a chaque annee :
            h = -log(1 - PD_12m)
            S(t) = exp(-h * t)
            PD_marginal(t) = S(t-1) - S(t)
            DF(t) = 1 / (1+r)^t
            DF_weighted = sum(PD_marginal(t) * DF(t)) / sum(PD_marginal(t))

        Ce DF pondere concentre l'actualisation sur les annees ou le
        defaut est le plus probable, conformement a IFRS 9 IE24.

        Args:
            stages: Array de stages.
            pd_12m: PD 12 mois (pour le calcul PD-marginal). Si None,
                retombe sur la moyenne arithmetique classique.

        Returns:
            Facteurs d'actualisation.
        """
        r = self.discount_rate
        horizon = self.lifetime_years

        # Stage 1 : DF sur 1 an
        df_1y = 1 / (1 + r)

        if pd_12m is None:
            # Fallback : moyenne arithmetique classique
            years = np.arange(1, horizon + 1)
            df_lifetime = np.mean(1 / (1 + r) ** years)
            return np.where(stages == 1, df_1y, df_lifetime)

        # Stage 2/3 : DF PD-marginal weighted (C2)
        n = len(stages)
        df_weighted = np.full(n, df_1y)  # Initialiser au DF 1 an

        # Masque des expositions lifetime (Stage 2/3)
        lifetime_mask = stages != 1
        if not lifetime_mask.any():
            return df_weighted

        # Hazard rate vectorise
        h = -np.log(1 - np.clip(pd_12m[lifetime_mask], 0, _PD_CLIP_MAX))

        # Survival, PD marginale, DF par annee
        # S(t) = exp(-h * t) pour t = 0, 1, ..., horizon
        # PD_marginal(t) = S(t-1) - S(t) pour t = 1, ..., horizon
        # DF(t) = 1 / (1+r)^t
        numerator = np.zeros(lifetime_mask.sum())
        denominator = np.zeros(lifetime_mask.sum())
        for t in range(1, horizon + 1):
            s_prev = np.exp(-h * (t - 1))
            s_curr = np.exp(-h * t)
            pm = s_prev - s_curr  # PD marginale annee t
            df_t = 1 / (1 + r) ** t
            numerator += pm * df_t
            denominator += pm

        # Eviter division par zero (PD ~ 0 => denominator ~ 0)
        df_weighted[lifetime_mask] = numerator / np.maximum(denominator, 1e-10)

        return df_weighted

    def _compute_credit_spread(
        self,
        pd_12m: np.ndarray,
        lgd: np.ndarray,
        t: float = 1.0,
    ) -> np.ndarray:
        """Calcule le credit spread Merton par position (H4).

        Formule :
            spread = -log(1 - PD_12m * LGD) / T + liquidity_premium

        Ou liquidity_premium = BASEL_CONFIG.liquidity_premium_bps / 10000.
        Le spread est borne dans [50bps, 2000bps].

        Args:
            pd_12m: PD 12 mois.
            lgd: LGD par position.
            t: Horizon en annees (defaut = 1 an).

        Returns:
            Array de credit spreads (en decimale, ex: 0.0050 = 50bps).
        """
        liquidity_premium = BASEL_CONFIG.liquidity_premium_bps / 10_000
        # Borner le produit PD * LGD pour eviter log(0)
        expected_loss = np.clip(pd_12m * lgd, 0, _PD_CLIP_MAX)
        spread = -np.log(1 - expected_loss) / t + liquidity_premium
        return np.clip(spread, 0.0050, 0.2000)


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset
    from ifrs9_cockpit.models.pd_model import PDModelSuite
    from ifrs9_cockpit.utils.helpers import format_euro, format_pct

    print("=" * 65)
    print("IFRS 9 COCKPIT — Phase 3 : ECL Calculation")
    print("=" * 65)

    # 1. Data
    print("\n[1/5] Génération des données...")
    df_credit, df_pe, df_history = generate_dataset()
    print(f"       {len(df_credit):,} entreprises | DR = {df_credit['default_flag'].mean():.2%}")

    # 2. PD Models
    print("\n[2/5] Entraînement des modèles PD...")
    pd_suite = PDModelSuite()
    pd_suite.fit(df_credit)
    best_model = "LR_WoE"
    pd_predictions = pd_suite.predict(df_credit)
    pd_current = pd_predictions[best_model]
    # Utiliser pd_origination reelle du generateur (H2)
    pd_origination = df_credit["pd_origination"].values
    print(f"       PD moyenne ({best_model}) : {pd_current.mean():.4f}")
    print(f"       PD origination moyenne    : {pd_origination.mean():.4f}")

    # 3. LGD
    print("\n[3/5] Calibration du modèle LGD...")
    lgd_model = LGDModel()
    lgd_model.fit(df_credit)
    lgd_summary = lgd_model.get_summary(df_credit)
    print(lgd_summary.to_string(index=False))

    # 4. EAD
    print("\n[4/5] Calibration du modèle EAD...")
    ead_model = EADModel()
    ead_model.fit(df_credit)
    ead_summary = ead_model.get_summary(df_credit)
    print(ead_summary.to_string(index=False))

    # 5. ECL
    print("\n[5/5] Calcul ECL multi-scénarios...")
    ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
    result = ecl_calc.calculate(df_credit, pd_current, pd_origination)

    # Résultats
    print("\n--- Distribution des Stages ---")
    staging = StagingEngine()
    stages = result["stage"].values
    ead_values = result["ead"].values
    stage_summary = staging.get_stage_summary(stages, ead_values)
    print(stage_summary.to_string(index=False))

    print(f"\n--- ECL Total ---")
    print(f"  ECL Base       : {format_euro(result['ecl_base'].sum())}")
    print(f"  ECL Adverse    : {format_euro(result['ecl_adverse'].sum())}")
    print(f"  ECL Favorable  : {format_euro(result['ecl_favorable'].sum())}")
    print(f"  ECL Pondéré    : {format_euro(result['ecl_weighted'].sum())}")
    print(f"  EAD Total      : {format_euro(result['ead'].sum())}")
    print(f"  Coverage       : {format_pct(result['ecl_weighted'].sum() / result['ead'].sum())}")

    print("\n--- ECL par Secteur ---")
    for sec in df_credit["sector"].unique():
        mask = result["sector"] == sec
        ecl_sec = result.loc[mask, "ecl_weighted"].sum()
        ead_sec = result.loc[mask, "ead"].sum()
        cov = ecl_sec / ead_sec if ead_sec > 0 else 0
        print(f"  {sec:20s} : ECL = {format_euro(ecl_sec):>12s}  |  Coverage = {format_pct(cov)}")

    print("\n--- Credit Spread Merton (H4) ---")
    spread = result["credit_spread"]
    print(f"  Spread moyen   : {spread.mean() * 10_000:.0f} bps")
    print(f"  Spread median  : {spread.median() * 10_000:.0f} bps")
    print(f"  Spread min     : {spread.min() * 10_000:.0f} bps")
    print(f"  Spread max     : {spread.max() * 10_000:.0f} bps")

    print("\n" + "=" * 65)
    print("Phase 3 validée.")
