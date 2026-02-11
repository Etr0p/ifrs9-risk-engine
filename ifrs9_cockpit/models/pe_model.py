"""Modele de valorisation PE (Private Equity) IFRS 13.

Calcule la NAV fondamentale via methodes IPEV par secteur,
avec 3 canaux de stress distincts :
    - Canal EBITDA : stress sur le metric (revenue ou EBITDA)
    - Canal Multiples : compression des multiples de sortie
    - Canal Leverage : hausse du levier effectif

Formule NAV par position :
    NAV = Metric_stresse x Multiple_compresse x (1 - Leverage_stresse)

Methodes IPEV par secteur (config-driven) :
    - Technologie : EV/Revenue (metric = revenue)
    - Industrie, Sante, Services : EV/EBITDA (metric = ebitda)
    - Immobilier : Cap rate/NOI (metric = ebitda, proxy NOI)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from ifrs9_cockpit.config import (
    RANDOM_SEED,
    SCENARIOS,
    SCENARIO_BASE,
    SECTORS,
    SectorConfig,
)


class PEModel:
    """Modele de valorisation PE IFRS 13.

    Calcule la NAV fondamentale avec methodes IPEV differenciees
    par secteur et 3 canaux de stress macro.

    Attributes:
        seed: Graine aleatoire.
        rng: Generateur numpy.
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        """Initialise le modele PE.

        Args:
            seed: Graine pour reproductibilite.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def calculate_nav(
        self,
        df_pe: pd.DataFrame,
        unemployment_override: Optional[float] = None,
        gdp_override: Optional[float] = None,
        interest_rate_override: Optional[float] = None,
        hpi_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule la NAV pour chaque position PE.

        NAV = Metric_stresse x Multiple_compresse x (1 - Leverage_stresse)

        Args:
            df_pe: DataFrame PE avec colonnes sector, revenue, ebitda,
                entry_multiple, leverage, valuation_method.
            unemployment_override: Taux de chomage (%).
            gdp_override: Croissance PIB (%).
            interest_rate_override: Taux directeur BCE (%).
            hpi_override: Variation prix immobiliers (%).
            inflation_override: Inflation IPC (%).

        Returns:
            Tuple (nav_array, exit_multiples_array) en M EUR.
        """
        n = len(df_pe)
        nav = np.zeros(n)
        exit_multiples = np.zeros(n)

        # Calculer les deltas macro (signe : positif = adverse)
        deltas = self._compute_macro_deltas(
            unemployment_override, gdp_override,
            interest_rate_override, hpi_override,
            inflation_override,
        )

        for sector in SECTORS:
            mask = df_pe["sector"].values == sector.name
            if mask.sum() == 0:
                continue

            # Canal 1 : Metrique IPEV (revenue ou ebitda)
            metric = self._get_valuation_metric(df_pe, mask, sector)
            metric_stressed = self._stress_metric(metric, sector, deltas)

            # Canal 2 : Multiple de sortie compresse
            compressed_mult = self._compress_exit_multiple(
                sector.exit_multiple_base, sector, deltas,
            )
            exit_multiples[mask] = compressed_mult

            # Canal 3 : Leverage stresse
            leverage = df_pe.loc[mask, "leverage"].values.astype(float)
            leverage_stressed = self._stress_leverage(leverage, sector, deltas)

            # NAV = Metric x Multiple x (1 - Leverage)
            nav[mask] = metric_stressed * compressed_mult * (1 - leverage_stressed)

        # Dispersion realiste (+/- 3%)
        noise = self.rng.normal(1.0, 0.03, n)
        nav *= noise

        return np.maximum(nav, 0), exit_multiples

    def calculate_nav_scenarios(
        self,
        df_pe: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Calcule la NAV sous les 3 scenarios ECL.

        Args:
            df_pe: DataFrame PE.

        Returns:
            Dict {scenario_name: nav_array}.
        """
        results = {}
        for scenario in SCENARIOS:
            nav, _ = self.calculate_nav(
                df_pe,
                unemployment_override=scenario.unemployment_rate,
                gdp_override=scenario.gdp_growth,
                interest_rate_override=scenario.interest_rate,
                hpi_override=scenario.hpi_growth,
                inflation_override=scenario.inflation_rate,
            )
            results[scenario.name] = nav
        return results

    def get_nav_summary(
        self,
        df_pe: pd.DataFrame,
        nav: np.ndarray,
    ) -> pd.DataFrame:
        """Resume de la NAV par secteur et methode de valorisation.

        Args:
            df_pe: DataFrame PE.
            nav: Array de NAV calculees.

        Returns:
            DataFrame recapitulatif.
        """
        summary_df = df_pe[["sector", "valuation_method"]].copy()
        summary_df["nav"] = nav
        summary_df["entry_multiple"] = df_pe["entry_multiple"].values

        return (
            summary_df.groupby(["sector", "valuation_method"])
            .agg(
                count=("nav", "size"),
                nav_mean=("nav", "mean"),
                nav_total=("nav", "sum"),
                nav_std=("nav", "std"),
                avg_entry_mult=("entry_multiple", "mean"),
            )
            .round(2)
            .reset_index()
        )

    # ──────────────────────────────────────────
    # METHODES PRIVEES
    # ──────────────────────────────────────────

    def _compute_macro_deltas(
        self,
        unemployment: Optional[float],
        gdp: Optional[float],
        interest_rate: Optional[float],
        hpi: Optional[float],
        inflation: Optional[float],
    ) -> Dict[str, float]:
        """Calcule les deltas macro par rapport au scenario de base.

        Convention : delta positif = choc adverse.

        Args:
            unemployment: Taux de chomage (%).
            gdp: Croissance PIB (%).
            interest_rate: Taux directeur BCE (%).
            hpi: Variation prix immobiliers (%).
            inflation: Inflation IPC (%).

        Returns:
            Dict des deltas normalises.
        """
        base = SCENARIO_BASE
        # Utiliser `is not None` au lieu de `or` pour gerer correctement override=0.0
        unemp = unemployment if unemployment is not None else base.unemployment_rate
        gdp_val = gdp if gdp is not None else base.gdp_growth
        ir_val = interest_rate if interest_rate is not None else base.interest_rate
        hpi_val = hpi if hpi is not None else base.hpi_growth
        infl_val = inflation if inflation is not None else base.inflation_rate
        return {
            "unemployment": (unemp - base.unemployment_rate) / 100,
            "gdp": (base.gdp_growth - gdp_val) / 100,
            "interest_rate": (ir_val - base.interest_rate) / 100,
            "hpi": (base.hpi_growth - hpi_val) / 100,
            "inflation": (infl_val - base.inflation_rate) / 100,
        }

    def _get_valuation_metric(
        self,
        df_pe: pd.DataFrame,
        mask: np.ndarray,
        sector: SectorConfig,
    ) -> np.ndarray:
        """Retourne la metrique IPEV appropriee pour le secteur.

        Args:
            df_pe: DataFrame PE.
            mask: Masque booleeen du secteur.
            sector: Configuration du secteur.

        Returns:
            Array de metriques (revenue ou ebitda en M EUR).
        """
        if sector.valuation_method == "EV/Revenue":
            return df_pe.loc[mask, "revenue"].values.astype(float)
        # EV/EBITDA et Cap_rate/NOI utilisent EBITDA (proxy NOI pour immobilier)
        return df_pe.loc[mask, "ebitda"].values.astype(float)

    def _stress_metric(
        self,
        metric: np.ndarray,
        sector: SectorConfig,
        deltas: Dict[str, float],
    ) -> np.ndarray:
        """Stresse la metrique IPEV (canal EBITDA).

        Baisse PIB et hausse chomage reduisent l'EBITDA/revenue.
        Utilise exp(-stress) pour un facteur multiplicatif toujours positif
        et symetrique en log-space (pas de floor artificiel).

        Args:
            metric: Metrique de base (M EUR).
            sector: Configuration du secteur.
            deltas: Deltas macro normalises.

        Returns:
            Metrique stressee.
        """
        stress = (
            deltas["gdp"] * sector.gdp_sensitivity_pe
            + deltas["unemployment"] * sector.unemployment_sensitivity_pe
            + deltas["inflation"] * sector.inflation_sensitivity_pe
        )
        # exp(-stress × 3.0) : toujours > 0 par construction (pas de floor artificiel)
        factor = np.exp(-stress * 3.0)
        return metric * factor

    def _compress_exit_multiple(
        self,
        base_multiple: float,
        sector: SectorConfig,
        deltas: Dict[str, float],
    ) -> float:
        """Comprime le multiple de sortie (canal Multiples).

        Monotone : hausse taux -> baisse multiple, baisse PIB -> baisse multiple.
        Utilise exp(-compression) pour un facteur multiplicatif symetrique.

        Args:
            base_multiple: Multiple de sortie baseline.
            sector: Configuration du secteur.
            deltas: Deltas macro normalises.

        Returns:
            Multiple compresse (scalaire).
        """
        compression = (
            deltas["interest_rate"] * sector.interest_rate_sensitivity_pe
            + deltas["gdp"] * sector.gdp_sensitivity_pe
            + deltas["hpi"] * sector.hpi_sensitivity_pe
        )
        # exp(-compression × 5.0) : toujours > 0 par construction (pas de floor artificiel)
        factor = float(np.exp(-compression * 5.0))
        compressed = base_multiple * factor

        # Borner dans la fourchette IPEV du secteur
        low, high = sector.entry_multiple_range
        return np.clip(compressed, low * 0.5, high * 1.5)

    def _stress_leverage(
        self,
        leverage: np.ndarray,
        sector: SectorConfig,
        deltas: Dict[str, float],
    ) -> np.ndarray:
        """Stresse le leverage effectif (canal Leverage).

        Hausse des taux -> hausse des charges de dette -> leverage effectif augmente.

        Args:
            leverage: Leverage de base.
            sector: Configuration du secteur.
            deltas: Deltas macro normalises.

        Returns:
            Leverage stresse, borne a [0, 0.95].
        """
        ir_impact = (
            deltas["interest_rate"] * sector.interest_rate_sensitivity_pe
            * 2.0  # Echelle leverage : 2× la sensibilite taux
        )
        leverage_stressed = leverage + ir_impact
        return np.clip(leverage_stressed, 0, 0.95)


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset
    from ifrs9_cockpit.utils.helpers import format_euro

    print("=" * 65)
    print("IFRS 9 COCKPIT — PE Model : Valorisation NAV IPEV")
    print("=" * 65)

    # 1. Data
    print("\n[1/3] Generation des donnees...")
    df_credit, df_pe, df_history = generate_dataset()
    print(f"       {len(df_pe):,} positions PE")

    # 2. NAV Baseline
    print("\n[2/3] Calcul NAV baseline...")
    pe_model = PEModel()
    nav_base, mult_base = pe_model.calculate_nav(df_pe)
    print(f"       NAV totale : {nav_base.sum():,.0f} M EUR")
    print(f"       NAV moyenne : {nav_base.mean():,.1f} M EUR")

    summary = pe_model.get_nav_summary(df_pe, nav_base)
    print("\n--- NAV par secteur ---")
    print(summary.to_string(index=False))

    # 3. Scenarios
    print("\n[3/3] NAV sous 3 scenarios...")
    nav_scenarios = pe_model.calculate_nav_scenarios(df_pe)
    for name, nav in nav_scenarios.items():
        print(f"  {name:12s} : NAV totale = {nav.sum():>12,.0f} M EUR | moy = {nav.mean():>8,.1f} M EUR")

    # Validation
    print("\n--- Validations ---")
    all_ok = True

    # NAV > 0
    ok = (nav_base >= 0).all()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] NAV >= 0 (min={nav_base.min():.2f})")
    all_ok &= ok

    # Ordering: Favorable > Base > Adverse
    nav_adv = nav_scenarios.get("Adverse", nav_base)
    nav_fav = nav_scenarios.get("Favorable", nav_base)
    ok_order = nav_fav.sum() > nav_base.sum() > nav_adv.sum()
    status = "PASS" if ok_order else "FAIL"
    print(f"  [{status}] NAV ordering: Favorable ({nav_fav.sum():,.0f}) > Base ({nav_base.sum():,.0f}) > Adverse ({nav_adv.sum():,.0f})")
    all_ok &= ok_order

    # Multiples in range
    for sector in SECTORS:
        mask = df_pe["sector"].values == sector.name
        if mask.sum() == 0:
            continue
        low, high = sector.entry_multiple_range
        mult_sec = mult_base[mask]
        ok_mult = mult_sec.min() >= low * 0.5 and mult_sec.max() <= high * 1.5
        status = "PASS" if ok_mult else "FAIL"
        print(f"  [{status}] {sector.name:15s} multiple = {mult_sec.mean():.2f}x "
              f"(fourchette [{low*0.5:.1f}, {high*1.5:.1f}])")
        all_ok &= ok_mult

    # Monotonicity: rate up -> multiple down
    pe_model2 = PEModel()
    _, mult_low_rate = pe_model2.calculate_nav(df_pe, interest_rate_override=2.0)
    pe_model3 = PEModel()
    _, mult_high_rate = pe_model3.calculate_nav(df_pe, interest_rate_override=6.0)
    ok_mono = mult_low_rate.mean() > mult_high_rate.mean()
    status = "PASS" if ok_mono else "FAIL"
    print(f"  [{status}] Monotonicity: mult(IR=2%) = {mult_low_rate.mean():.2f}x > mult(IR=6%) = {mult_high_rate.mean():.2f}x")
    all_ok &= ok_mono

    print(f"\n{'=' * 65}")
    if all_ok:
        print("PE Model valide.")
    else:
        print("ATTENTION : certaines validations ont echoue.")
    print(f"{'=' * 65}")
