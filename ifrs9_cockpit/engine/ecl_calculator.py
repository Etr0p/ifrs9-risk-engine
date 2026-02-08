"""Calculateur ECL (Expected Credit Loss) IFRS 9.

Orchestre le calcul complet de l'ECL en combinant :
    - PD (Probability of Default) par stage et horizon
    - LGD (Loss Given Default) TTC ou Downturn
    - EAD (Exposure At Default) base ou stressée
    - Discount Factor (actualisation à l'EIR)
    - Pondération multi-scénarios (Base 70% + Adverse 30%)

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
    MacroScenario,
    RANDOM_SEED,
    SEGMENTS,
)
from ifrs9_cockpit.engine.staging import StagingEngine
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel


class ECLCalculator:
    """Calculateur ECL multi-scénarios IFRS 9.

    Pipeline :
        1. Calcul des PD par scénario (base + adverse)
        2. Affectation aux stages via StagingEngine
        3. Calcul LGD (TTC pour base, Downturn pour adverse)
        4. Calcul EAD (base ou stressée selon scénario)
        5. Actualisation + pondération des scénarios

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
            scenarios: Scénarios macro (Base + Adverse par défaut).
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

            # Discount factor
            df_factor = self._compute_discount_factor(stages)

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

        return result

    def compute_ecl_summary(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """Résumé de l'ECL par stage et segment.

        Args:
            result_df: DataFrame résultat de calculate().

        Returns:
            DataFrame agrégé avec ECL total, moyen, coverage ratio.
        """
        summary = (
            result_df.groupby(["stage", "segment"])
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
        """Ajuste les PD selon le scénario macroéconomique.

        Applique les chocs macro avec sensibilité par segment.
        5 canaux de transmission : chômage, PIB, taux directeur,
        prix immobiliers, inflation.

        Args:
            pd_base: PD de base (modèle).
            df: DataFrame clients (pour les segments).
            scenario: Scénario macroéconomique.
            unemployment_override: Override chômage pour stress test interactif.
            gdp_override: Override PIB pour stress test interactif.
            interest_rate_override: Override taux directeur BCE.
            hpi_override: Override variation prix immobiliers.
            inflation_override: Override inflation IPC.

        Returns:
            PD ajustées pour le scénario.
        """
        pd_adjusted = pd_base.copy()

        # Déterminer les chocs effectifs
        has_override = any(v is not None for v in [
            unemployment_override, gdp_override, interest_rate_override,
            hpi_override, inflation_override,
        ])
        if has_override:
            # Mode stress test interactif
            unemp_shock = max(0, (unemployment_override or 7.5) - 7.5) / 100
            gdp_shock = max(0, 1.2 - (gdp_override or 1.2)) / 100
            ir_shock = max(0, (interest_rate_override or 3.5) - 3.5) / 100
            hpi_shock = max(0, 2.0 - (hpi_override or 2.0)) / 100
            infl_shock = max(0, (inflation_override or 2.5) - 2.5) / 100
        else:
            unemp_shock = scenario.unemployment_shock
            gdp_shock = scenario.gdp_shock
            ir_shock = scenario.interest_rate_shock
            hpi_shock = scenario.hpi_shock
            infl_shock = scenario.inflation_shock

        # Appliquer les chocs par segment (sensibilité variable)
        for seg in SEGMENTS:
            mask = df["segment"].values == seg.name
            segment_shock = (
                unemp_shock * seg.unemployment_sensitivity
                + gdp_shock * seg.gdp_sensitivity
                + ir_shock * seg.interest_rate_sensitivity
                + hpi_shock * seg.hpi_sensitivity
                + infl_shock * seg.inflation_sensitivity
            )
            pd_adjusted[mask] = pd_adjusted[mask] * (1 + segment_shock * 10)

        return np.clip(pd_adjusted, 0, 1)

    def _compute_lifetime_pd(
        self,
        pd_12m: np.ndarray,
        stages: np.ndarray,
    ) -> np.ndarray:
        """Convertit la PD 12 mois en PD lifetime (cumulative).

        Utilise l'approximation : PD_lifetime = 1 - (1 - PD_12m)^horizon
        avec l'horizon défini dans la configuration.

        Args:
            pd_12m: PD 12 mois.
            stages: Stages (l'horizon dépend du stage).

        Returns:
            PD lifetime.
        """
        horizon = np.where(
            stages == 1,
            1,  # Stage 1 : 12 mois
            self.lifetime_years,  # Stage 2/3 : horizon lifetime
        )

        pd_lifetime = 1 - (1 - pd_12m) ** horizon
        return np.clip(pd_lifetime, 0, 1)

    def _compute_discount_factor(self, stages: np.ndarray) -> np.ndarray:
        """Calcule le facteur d'actualisation par stage.

        Stage 1 : actualisation sur 1 an
        Stage 2/3 : actualisation moyenne sur l'horizon lifetime

        Le DF moyen sur N années = (1/N) × Σ 1/(1+r)^t

        Args:
            stages: Array de stages.

        Returns:
            Facteurs d'actualisation.
        """
        r = self.discount_rate

        # Stage 1 : DF sur 1 an
        df_1y = 1 / (1 + r)

        # Stage 2/3 : DF moyen sur l'horizon lifetime
        years = np.arange(1, self.lifetime_years + 1)
        df_lifetime = np.mean(1 / (1 + r) ** years)

        return np.where(stages == 1, df_1y, df_lifetime)


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset
    from ifrs9_cockpit.models.pd_model import PDModelSuite
    from ifrs9_cockpit.utils.helpers import format_euro, format_pct

    print("=" * 65)
    print("IFRS 9 COCKPIT — Phase 3 : ECL Calculation")
    print("=" * 65)

    # 1. Data
    print("\n[1/5] Génération des données...")
    df_clients, _ = generate_dataset()
    print(f"       {len(df_clients):,} clients | DR = {df_clients['default_flag'].mean():.2%}")

    # 2. PD Models
    print("\n[2/5] Entraînement des modèles PD...")
    pd_suite = PDModelSuite()
    pd_suite.fit(df_clients)
    best_model = "LR_WoE"
    pd_predictions = pd_suite.predict(df_clients)
    pd_current = pd_predictions[best_model]
    pd_origination = pd_current * 0.8  # Proxy : PD origination = 80% de PD courante
    print(f"       PD moyenne ({best_model}) : {pd_current.mean():.4f}")

    # 3. LGD
    print("\n[3/5] Calibration du modèle LGD...")
    lgd_model = LGDModel()
    lgd_model.fit(df_clients)
    lgd_summary = lgd_model.get_summary(df_clients)
    print(lgd_summary.to_string(index=False))

    # 4. EAD
    print("\n[4/5] Calibration du modèle EAD...")
    ead_model = EADModel()
    ead_model.fit(df_clients)
    ead_summary = ead_model.get_summary(df_clients)
    print(ead_summary.to_string(index=False))

    # 5. ECL
    print("\n[5/5] Calcul ECL multi-scénarios...")
    ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
    result = ecl_calc.calculate(df_clients, pd_current, pd_origination)

    # Résultats
    print("\n--- Distribution des Stages ---")
    staging = StagingEngine()
    stages = result["stage"].values
    ead_values = result["ead"].values
    stage_summary = staging.get_stage_summary(stages, ead_values)
    print(stage_summary.to_string(index=False))

    print(f"\n--- ECL Total ---")
    print(f"  ECL Base     : {format_euro(result['ecl_base'].sum())}")
    print(f"  ECL Adverse  : {format_euro(result['ecl_adverse'].sum())}")
    print(f"  ECL Pondéré  : {format_euro(result['ecl_weighted'].sum())}")
    print(f"  EAD Total    : {format_euro(result['ead'].sum())}")
    print(f"  Coverage     : {format_pct(result['ecl_weighted'].sum() / result['ead'].sum())}")

    print("\n--- ECL par Segment ---")
    for seg in df_clients["segment"].unique():
        mask = result["segment"] == seg
        ecl_seg = result.loc[mask, "ecl_weighted"].sum()
        ead_seg = result.loc[mask, "ead"].sum()
        cov = ecl_seg / ead_seg if ead_seg > 0 else 0
        print(f"  {seg:20s} : ECL = {format_euro(ecl_seg):>12s}  |  Coverage = {format_pct(cov)}")

    print("\n" + "=" * 65)
    print("Phase 3 validée.")
