"""Calculateur ECL (Expected Credit Loss) IFRS 9.

Orchestre le calcul complet de l'ECL en combinant :
    - PD (Probability of Default) par stage et horizon
    - LGD (Loss Given Default) TTC ou Downturn
    - EAD (Exposure At Default) base ou stressee
    - Discount Factor (actualisation a l'EIR)
    - Ponderation multi-scenarios (Base 50% + Adverse 25% + Favorable 25%)

Formule ECL par exposition :
    ECL = PD x LGD x EAD x DF

Ou l'horizon PD depend du stage :
    - Stage 1 : PD 12 mois
    - Stage 2/3 : PD lifetime (cumulative sur l'horizon)
"""

from __future__ import annotations

import numpy as np
import polars as pl
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
from ifrs9_cockpit.utils.frame_compat import to_pandas, ensure_numpy
from ifrs9_cockpit.engine.staging import StagingEngine
from ifrs9_cockpit.models.lgd_model import LGDModel
from ifrs9_cockpit.models.ead_model import EADModel


def compute_corporate_rw(df: pl.DataFrame) -> np.ndarray:
    """CRR3 SA Art. 122 + SME support factor Art. 501.
    - Credit score -> rating synthetique -> RW de base
    - Si CA < 50M EUR -> facteur PME 0.7619
    - Defaut -> 150%
    """
    n = len(df)
    if "credit_score" not in df.columns:
        return np.full(n, 1.00)
    cs = df["credit_score"].to_numpy().astype(float)
    rw = np.full(n, 1.00)          # Unrated = 100%
    rw[cs >= 750] = 0.20           # AAA/AA equivalent
    rw[(cs >= 680) & (cs < 750)] = 0.50  # A equivalent
    rw[(cs >= 600) & (cs < 680)] = 0.75  # BBB equivalent
    rw[(cs >= 500) & (cs < 600)] = 1.00  # BB equivalent
    rw[cs < 500] = 1.50            # below BB equivalent
    # SME support factor (Art. 501) : CA < 50M EUR
    if "revenue" in df.columns:
        revenue = df["revenue"].to_numpy().astype(float)
        sme_mask = revenue < 50_000_000
        rw[sme_mask] *= 0.7619
    # Defaut : 150%
    if "default_flag" in df.columns:
        rw[df["default_flag"].to_numpy() == 1] = 1.50
    return rw


# Borne superieure PD pour eviter log(0) dans les calculs hazard rate et Merton
_PD_CLIP_MAX: float = 0.9999


class ECLCalculator:
    """Calculateur ECL multi-scenarios IFRS 9.

    Pipeline :
        1. Calcul des PD par scenario (Base + Adverse + Favorable)
        2. Affectation aux stages via StagingEngine
        3. Calcul LGD (TTC pour Base/Favorable, Downturn pour Adverse)
        4. Calcul EAD (base ou stressee selon scenario)
        5. Actualisation + ponderation des scenarios (50/25/25)

    Attributes:
        staging_engine: Moteur de staging.
        lgd_model: Modele LGD calibre.
        ead_model: Modele EAD calibre.
        discount_rate: Taux d'actualisation annuel.
        lifetime_years: Horizon lifetime pour Stage 2/3.
        scenarios: Liste des scenarios macroeconomiques.
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
            lgd_model: Modele LGD calibre.
            ead_model: Modele EAD calibre.
            staging_engine: Moteur de staging (cree par defaut si None).
            discount_rate: Taux d'actualisation (EIR proxy).
            lifetime_years: Horizon pour le calcul lifetime.
            scenarios: Scenarios macro (Base + Adverse + Favorable par defaut).
        """
        self.staging_engine = staging_engine or StagingEngine()
        self.lgd_model = lgd_model
        self.ead_model = ead_model
        self.discount_rate = discount_rate
        self.lifetime_years = lifetime_years
        self.scenarios = scenarios or list(SCENARIOS)

    def calculate(
        self,
        df: pl.DataFrame,
        pd_current: np.ndarray,
        pd_origination: np.ndarray,
        unemployment_override: Optional[float] = None,
        gdp_override: Optional[float] = None,
        interest_rate_override: Optional[float] = None,
        hpi_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
    ) -> pl.DataFrame:
        """Calcule l'ECL pour chaque exposition avec ponderation multi-scenarios.

        Args:
            df: DataFrame clients avec toutes les features necessaires.
            pd_current: PD courante (Point-In-Time).
            pd_origination: PD a l'origination.
            unemployment_override: Override du taux de chomage (stress test).
            gdp_override: Override de la croissance PIB (stress test).
            interest_rate_override: Override du taux directeur BCE (stress test).
            hpi_override: Override de la variation prix immobiliers (stress test).
            inflation_override: Override de l'inflation IPC (stress test).

        Returns:
            DataFrame avec colonnes ajoutees : stage, pd_12m, pd_lifetime,
            lgd, ead, discount_factor, ecl_by_scenario, ecl_weighted.
        """
        n = len(df)
        dpd = self._get_dpd(df)
        default_flag = df["default_flag"].to_numpy() if "default_flag" in df.columns else np.zeros(n)

        # Calculer la PD stressee de reference (scenario Base avec overrides)
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

        # Construire le dict macro pour le SICR forward-looking (IFRS 9 B5.5.17)
        base = SCENARIO_BASE
        if has_override:
            _macro_for_sicr = {
                "unemployment_rate": unemployment_override if unemployment_override is not None else base.unemployment_rate,
                "gdp_growth": gdp_override if gdp_override is not None else base.gdp_growth,
                "interest_rate": interest_rate_override if interest_rate_override is not None else base.interest_rate,
                "hpi_growth": hpi_override if hpi_override is not None else base.hpi_growth,
                "inflation_rate": inflation_override if inflation_override is not None else base.inflation_rate,
            }
        else:
            _macro_for_sicr = None

        # Staging sur les PD stressees (reagit aux sliders + macro Z-score)
        stages = self.staging_engine.assign_stages(
            pd_stressed_ref, pd_origination, dpd, default_flag,
            macro_params=_macro_for_sicr,
        )

        # Calcul ECL par scenario
        ecl_scenarios: Dict[str, np.ndarray] = {}
        scenario_details: Dict[str, Dict[str, np.ndarray]] = {}

        for scenario in self.scenarios:
            # Ajuster les PD selon le scenario macro
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
            # Stage 3 : PD = 1.0 (defaut avere)
            pd_effective = np.where(stages == 3, 1.0, pd_effective)

            # LGD selon le type de scenario (HPI impacte la valeur du collateral)
            is_adverse = scenario.name == "Adverse"
            scenario_hpi = hpi_override if hpi_override is not None else scenario.hpi_growth
            lgd = self.lgd_model.predict(df, downturn=is_adverse, hpi_override=scenario_hpi)

            # EAD selon le stress
            ead = self.ead_model.predict(df, stressed=is_adverse)

            # Discount factor PD-marginal weighted (C2)
            df_factor = self._compute_discount_factor(stages, pd_12m)

            # ECL = PD x LGD x EAD x DF
            ecl = pd_effective * lgd * ead * df_factor

            ecl_scenarios[scenario.name] = ecl
            scenario_details[scenario.name] = {
                "pd_effective": pd_effective,
                "lgd": lgd,
                "ead": ead,
                "discount_factor": df_factor,
            }

        # Ponderation des scenarios
        ecl_weighted = np.zeros(n)
        for scenario in self.scenarios:
            ecl_weighted += scenario.weight * ecl_scenarios[scenario.name]

        # Construire le resultat
        base_details = scenario_details.get("Base", scenario_details[self.scenarios[0].name])

        new_cols = [
            pl.Series("stage", stages),
            pl.Series("pd_12m", pd_stressed_ref),
            pl.Series("pd_lifetime", self._compute_lifetime_pd(pd_stressed_ref, stages)),
            pl.Series("lgd", base_details["lgd"]),
            pl.Series("ead", base_details["ead"]),
            pl.Series("discount_factor", base_details["discount_factor"]),
        ]
        for scenario_name, ecl in ecl_scenarios.items():
            new_cols.append(pl.Series(f"ecl_{scenario_name.lower()}", np.round(ecl, 2)))
        # Position-level RW (CRR3 SA Art. 122 + SME Art. 501)
        rw_position = compute_corporate_rw(df)

        new_cols.extend([
            pl.Series("ecl_weighted", np.round(ecl_weighted, 2)),
            pl.Series("rw_crr3", rw_position),
            pl.Series("rwa_credit", np.round(base_details["ead"] * rw_position, 2)),
            pl.Series("credit_spread", self._compute_credit_spread(
                pd_stressed_ref, base_details["lgd"],
            )),
        ])

        result = df.clone()
        result = result.with_columns(new_cols)

        return result

    def compute_ecl_summary(self, result_df: pl.DataFrame) -> pl.DataFrame:
        """Resume de l'ECL par stage et secteur.

        Args:
            result_df: DataFrame resultat de calculate().

        Returns:
            DataFrame agrege avec ECL total, moyen, coverage ratio.
        """
        summary = (
            result_df.group_by(["stage", "sector"])
            .agg(
                pl.col("ecl_weighted").count().alias("count"),
                pl.col("ecl_weighted").sum().alias("ecl_total"),
                pl.col("ecl_weighted").mean().alias("ecl_mean"),
                pl.col("ead").sum().alias("ead_total"),
                pl.col("pd_12m").mean().alias("pd_mean"),
            )
        )
        summary = summary.with_columns(
            pl.when(pl.col("ead_total") > 0)
            .then(pl.col("ecl_total") / pl.col("ead_total"))
            .otherwise(0.0)
            .alias("coverage_ratio")
        )
        # Round numerical columns
        numerical_cols = ["ecl_total", "ecl_mean", "ead_total", "pd_mean", "coverage_ratio"]
        summary = summary.with_columns([pl.col(c).round(4) for c in numerical_cols])
        return summary

    def compute_waterfall(
        self,
        ecl_t0: np.ndarray,
        ecl_t1: np.ndarray,
        stages_t0: np.ndarray,
        stages_t1: np.ndarray,
        segments: np.ndarray,
    ) -> pl.DataFrame:
        """Construit le waterfall de variation ECL entre deux dates.

        Decompose la variation ECL en :
            - New business
            - Stage migration
            - PD/LGD/EAD changes
            - Derecognition

        Args:
            ecl_t0: ECL a la date initiale.
            ecl_t1: ECL a la date finale.
            stages_t0: Stages a t0.
            stages_t1: Stages a t1.
            segments: Segments clients.

        Returns:
            DataFrame avec la decomposition du waterfall.
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
            "component": "Parametres (PD/LGD/EAD)",
            "amount": round((ecl_t1[stable] - ecl_t0[stable]).sum(), 2),
        })
        records.append({
            "component": "ECL Cloture",
            "amount": round(total_t1, 2),
        })
        records.append({
            "component": "Variation nette",
            "amount": round(delta, 2),
        })

        return pl.DataFrame(records)

    # ------------------------------------------
    # METHODES PRIVEES
    # ------------------------------------------

    def _get_dpd(self, df: pl.DataFrame) -> np.ndarray:
        """Extrait les DPD du DataFrame (ou zero si absent).

        Args:
            df: DataFrame clients.

        Returns:
            Array de DPD.
        """
        if "dpd" in df.columns:
            return df["dpd"].to_numpy().astype(int)
        return np.zeros(len(df), dtype=int)

    def _adjust_pd_for_scenario(
        self,
        pd_base: np.ndarray,
        df: pl.DataFrame,
        scenario: MacroScenario,
        unemployment_override: Optional[float] = None,
        gdp_override: Optional[float] = None,
        interest_rate_override: Optional[float] = None,
        hpi_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
    ) -> np.ndarray:
        """Ajuste les PD selon le scenario macroeconomique (logit-space).

        Formule Merton-Vasicek :
            logit(PD_stressed) = logit(PD_base) + LOGIT_AMPLITUDE x Sum(shock_i x sens_i)

        Les chocs sont des deltas normalises (positif = adverse) calcules
        depuis les valeurs absolues des variables macro vs baseline.
        La transformation logit garantit PD dans ]0, 1[ sans clip.

        5 canaux de transmission : chomage, PIB, taux directeur,
        prix immobiliers, inflation.

        Args:
            pd_base: PD de base (modele).
            df: DataFrame clients (pour les secteurs).
            scenario: Scenario macroeconomique.
            unemployment_override: Override chomage pour stress test interactif.
            gdp_override: Override PIB pour stress test interactif.
            interest_rate_override: Override taux directeur BCE.
            hpi_override: Override variation prix immobiliers.
            inflation_override: Override inflation IPC.

        Returns:
            PD ajustees pour le scenario, bornees dans ]0, 1[ par expit.
        """
        # Transformer en espace logit (log-odds)
        logit_pd = logit(pd_base)

        base = SCENARIO_BASE

        # Determiner les valeurs macro effectives
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
            # Mode scenario predefini : valeurs absolues du scenario
            unemp_val = scenario.unemployment_rate
            gdp_val = scenario.gdp_growth
            ir_val = scenario.interest_rate
            hpi_val = scenario.hpi_growth
            infl_val = scenario.inflation_rate

        # Deltas normalises (positif = adverse, symetrique favorable/adverse)
        # Plus de max(0, ...) : les scenarios favorables reduisent la PD
        unemp_shock = (unemp_val - base.unemployment_rate) / 100
        gdp_shock = (base.gdp_growth - gdp_val) / 100         # Inverse : baisse PIB = adverse
        ir_shock = (ir_val - base.interest_rate) / 100
        hpi_shock = (base.hpi_growth - hpi_val) / 100          # Inverse : baisse HPI = adverse
        infl_shock = (infl_val - base.inflation_rate) / 100

        # Appliquer les chocs par secteur dans l'espace logit
        sectors = df["sector"].to_numpy()
        for sec in SECTORS:
            mask = sectors == sec.name
            sector_shock = (
                unemp_shock * sec.unemployment_sensitivity_credit
                + gdp_shock * sec.gdp_sensitivity_credit
                + ir_shock * sec.interest_rate_sensitivity_credit
                + hpi_shock * sec.hpi_sensitivity_credit
                + infl_shock * sec.inflation_sensitivity_credit
            )
            # Shift additif en logit-space (expit garantit PD dans ]0, 1[)
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
    print("IFRS 9 COCKPIT -- Phase 3 : ECL Calculation")
    print("=" * 65)

    # 1. Data
    print("\n[1/5] Generation des donnees...")
    df_credit, df_pe, df_history, _ = generate_dataset()
    print(f"       {len(df_credit):,} entreprises | DR = {df_credit['default_flag'].mean():.2%}")

    # 2. PD Models
    print("\n[2/5] Entrainement des modeles PD...")
    pd_suite = PDModelSuite()
    pd_suite.fit(df_credit)
    best_model = "LR_WoE"
    pd_predictions = pd_suite.predict(df_credit)
    pd_current = pd_predictions[best_model]
    # Utiliser pd_origination reelle du generateur (H2)
    pd_origination = df_credit["pd_origination"].to_numpy()
    print(f"       PD moyenne ({best_model}) : {pd_current.mean():.4f}")
    print(f"       PD origination moyenne    : {pd_origination.mean():.4f}")

    # 3. LGD
    print("\n[3/5] Calibration du modele LGD...")
    lgd_model = LGDModel()
    lgd_model.fit(df_credit)
    lgd_summary = lgd_model.get_summary(df_credit)
    print(to_pandas(lgd_summary).to_string(index=False))

    # 4. EAD
    print("\n[4/5] Calibration du modele EAD...")
    ead_model = EADModel()
    ead_model.fit(df_credit)
    ead_summary = ead_model.get_summary(df_credit)
    print(to_pandas(ead_summary).to_string(index=False))

    # 5. ECL
    print("\n[5/5] Calcul ECL multi-scenarios...")
    ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
    result = ecl_calc.calculate(df_credit, pd_current, pd_origination)

    # Resultats
    print("\n--- Distribution des Stages ---")
    staging = StagingEngine()
    stages = result["stage"].to_numpy()
    ead_values = result["ead"].to_numpy()
    stage_summary = staging.get_stage_summary(stages, ead_values)
    print(to_pandas(stage_summary).to_string(index=False))

    print(f"\n--- ECL Total ---")
    print(f"  ECL Base       : {format_euro(result['ecl_base'].sum())}")
    print(f"  ECL Adverse    : {format_euro(result['ecl_adverse'].sum())}")
    print(f"  ECL Favorable  : {format_euro(result['ecl_favorable'].sum())}")
    print(f"  ECL Pondere    : {format_euro(result['ecl_weighted'].sum())}")
    print(f"  EAD Total      : {format_euro(result['ead'].sum())}")
    print(f"  Coverage       : {format_pct(result['ecl_weighted'].sum() / result['ead'].sum())}")

    print("\n--- ECL par Secteur ---")
    for sec in df_credit["sector"].unique().to_list():
        ecl_sec = result.filter(pl.col("sector") == sec)["ecl_weighted"].sum()
        ead_sec = result.filter(pl.col("sector") == sec)["ead"].sum()
        cov = ecl_sec / ead_sec if ead_sec > 0 else 0
        print(f"  {sec:20s} : ECL = {format_euro(ecl_sec):>12s}  |  Coverage = {format_pct(cov)}")

    print("\n--- Credit Spread Merton (H4) ---")
    spread = result["credit_spread"]
    print(f"  Spread moyen   : {spread.mean() * 10_000:.0f} bps")
    print(f"  Spread median  : {spread.median() * 10_000:.0f} bps")
    print(f"  Spread min     : {spread.min() * 10_000:.0f} bps")
    print(f"  Spread max     : {spread.max() * 10_000:.0f} bps")

    print("\n" + "=" * 65)
    print("Phase 3 validee.")
