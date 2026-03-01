"""Modele de valorisation PE (Private Equity) IFRS 13.

Calcule la NAV fondamentale via methodes IPEV par secteur,
avec 3 canaux de stress distincts :
    - Canal EBITDA : stress sur le metric (revenue ou EBITDA)
    - Canal Multiples : compression des multiples de sortie
    - Canal Leverage : hausse du levier effectif

Formule NAV par position :
    NAV = Metric_stresse x Multiple_compresse x (1 - Leverage_stresse)

Methodes IPEV par secteur (config-driven, IPEV Valuation Guidelines 2025) :
    - Technologie : EV/Revenue (metric = revenue, standard SaaS/Tech)
    - Industrie, Sante, Services : EV/EBITDA (metric = ebitda)
    - Immobilier : Cap rate/NOI (metric = ebitda × (1 - NOI_OPEX_RATIO),
      conversion EBITDA→NOI via ratio charges d'exploitation)

Bruit de dispersion :
    Modele a facteur sectoriel pour capturer la correlation intra-sectorielle.
    ε_i = 1 + sqrt(ρ)·σ·Z_secteur + sqrt(1-ρ)·σ·Z_idio
    Positions du meme secteur et vintage partagent un facteur commun.
    Ref: Preqin (2023) correlation intra-fonds.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Dict, Optional, Tuple

from ifrs9_cockpit.config import (
    RANDOM_SEED,
    SCENARIOS,
    SCENARIO_BASE,
    SECTORS,
    SectorConfig,
    NOI_OPEX_RATIO,
    PE_NOISE_INTRA_SECTOR_CORR,
)
from ifrs9_cockpit.utils.frame_compat import to_pandas, to_polars


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
        df_pe,
        unemployment_override: Optional[float] = None,
        gdp_override: Optional[float] = None,
        interest_rate_override: Optional[float] = None,
        hpi_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
        unemployment_crisis: bool = False,
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
            unemployment_crisis: Si True, le chomage est de nature "crise
                economique" et penalise tous les secteurs PE y compris ceux
                a sensibilite negative (ex. Technologie). Si False (defaut),
                le chomage est de nature "rupture techno" et les sensibilites
                negatives beneficient au secteur.

        Returns:
            Tuple (nav_array, exit_multiples_array) en M EUR.
        """
        df_pe = to_pandas(df_pe)
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
            metric_stressed = self._stress_metric(
                metric, sector, deltas, unemployment_crisis,
            )

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

        # Dispersion realiste (+/- 3%) avec correlation intra-sectorielle.
        # Modele a facteur : ε_i = 1 + sqrt(ρ)·σ·Z_secteur + sqrt(1-ρ)·σ·Z_idio
        # Preserves la variance marginale σ² = 0.03² tout en creant une
        # correlation ρ entre positions du meme secteur (Preqin 2023).
        # RNG reinitialise (seed+7) pour determinisme cross-scenario.
        noise_rng = np.random.default_rng(self.seed + 7)
        sigma = 0.03
        rho = PE_NOISE_INTRA_SECTOR_CORR
        sqrt_rho = np.sqrt(rho)
        sqrt_1mrho = np.sqrt(1 - rho)
        # Facteurs sectoriels (1 par secteur, meme pour toutes positions du secteur)
        sector_factors = {s.name: noise_rng.normal(0, 1) for s in SECTORS}
        # Facteurs idiosyncratiques (1 par position)
        idio = noise_rng.normal(0, 1, n)
        noise = np.ones(n)
        for sector in SECTORS:
            mask = df_pe["sector"].values == sector.name
            if mask.sum() == 0:
                continue
            z_sector = sector_factors[sector.name]
            noise[mask] = 1 + sigma * (sqrt_rho * z_sector + sqrt_1mrho * idio[mask])
        nav *= noise

        # Floor a 1 EUR (1e-6 M EUR) : une position PE a toujours
        # une valeur residuelle > 0 (option sur actif net)
        return np.maximum(nav, 1e-6), exit_multiples

    def calculate_nav_scenarios(
        self,
        df_pe,
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
        df_pe,
        nav: np.ndarray,
    ) -> pl.DataFrame:
        """Resume de la NAV par secteur et methode de valorisation.

        Args:
            df_pe: DataFrame PE.
            nav: Array de NAV calculees.

        Returns:
            DataFrame recapitulatif.
        """
        df_pe = to_pandas(df_pe)
        import pandas as pd
        summary_df = df_pe[["sector", "valuation_method"]].copy()
        summary_df["nav"] = nav
        summary_df["entry_multiple"] = df_pe["entry_multiple"].values

        result_pd = (
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
        return pl.from_pandas(result_pd)

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
        df_pe,
        mask: np.ndarray,
        sector: SectorConfig,
    ) -> np.ndarray:
        """Retourne la metrique IPEV appropriee pour le secteur.

        Args:
            df_pe: DataFrame PE.
            mask: Masque booleeen du secteur.
            sector: Configuration du secteur.

        Returns:
            Array de metriques (revenue, ebitda ou NOI en M EUR).
        """
        if sector.valuation_method == "EV/Revenue":
            return df_pe.loc[mask, "revenue"].values.astype(float)
        ebitda = df_pe.loc[mask, "ebitda"].values.astype(float)
        if sector.valuation_method == "Cap_rate/NOI":
            # Conversion EBITDA → NOI : deduit les charges d'exploitation (OPEX)
            # que l'EBITDA ne capture pas (frais de gestion, maintenance, assurance).
            # Source : CBRE/JLL benchmark, OPEX commercial RE = 12-18%.
            return ebitda * (1 - NOI_OPEX_RATIO)
        return ebitda

    def _stress_metric(
        self,
        metric: np.ndarray,
        sector: SectorConfig,
        deltas: Dict[str, float],
        unemployment_crisis: bool = False,
    ) -> np.ndarray:
        """Stresse la metrique IPEV (canal EBITDA).

        Baisse PIB et hausse chomage reduisent l'EBITDA/revenue.
        Utilise exp(-stress) pour un facteur multiplicatif toujours positif
        et symetrique en log-space (pas de floor artificiel).

        Args:
            metric: Metrique de base (M EUR).
            sector: Configuration du secteur.
            deltas: Deltas macro normalises.
            unemployment_crisis: Si True, utilise abs(sensitivity) pour
                chomage (crise eco = toujours adverse).

        Returns:
            Metrique stressee.
        """
        unemp_sens = (
            abs(sector.unemployment_sensitivity_pe)
            if unemployment_crisis
            else sector.unemployment_sensitivity_pe
        )
        stress = (
            deltas["gdp"] * sector.gdp_sensitivity_pe
            + deltas["unemployment"] * unemp_sens
            + deltas["inflation"] * sector.inflation_sensitivity_pe
        )
        # exp(-stress × 1.5) : toujours > 0 par construction (pas de floor artificiel)
        # Scale 1.5 calibre pour qu'un choc PIB de -3pp reduise l'EBITDA de ~5-8%
        # (benchmark Preqin 2023, mouvements trimestriels medians).
        factor = np.exp(-stress * 1.5)
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
        # Recession dampening: attenuate rate-cut benefit in adverse conditions
        gdp_shock = deltas.get("gdp", 0.0)
        ir_shock = deltas.get("interest_rate", 0.0)
        unemp_shock = deltas.get("unemployment", 0.0)

        # Cancel 85% of IR benefit when economy is deteriorating
        adverse_signal = max(gdp_shock, unemp_shock)
        if adverse_signal > 0 and ir_shock < 0:
            recession_depth = min(adverse_signal / 0.02, 1.0)
            ir_benefit = ir_shock * sector.interest_rate_sensitivity_pe
            compression -= ir_benefit * recession_depth * 0.85

        # Growth offsetting: attenuate rate-hike compression when GDP is growing
        if gdp_shock < 0 and ir_shock > 0:
            growth_depth = min(abs(gdp_shock) / 0.03, 1.0)
            ir_compression = ir_shock * sector.interest_rate_sensitivity_pe
            compression -= ir_compression * growth_depth * 0.50
        # exp(-compression × 2.0) : toujours > 0 par construction (pas de floor artificiel)
        # Scale 2.0 calibre pour qu'un choc taux de +100bp comprime les multiples
        # de ~3-5% (benchmark EBA 2023, sensibilite PE mid-market).
        factor = float(np.exp(-compression * 2.0))
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
            * 0.5  # Echelle leverage : 0.5× la sensibilite taux
            # +100bp → ~+0.5-1pp leverage (coherent avec LBO mid-market)
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
    df_credit, df_pe, df_history, _ = generate_dataset()
    print(f"       {len(df_pe):,} positions PE")

    # 2. NAV Baseline
    print("\n[2/3] Calcul NAV baseline...")
    pe_model = PEModel()
    nav_base, mult_base = pe_model.calculate_nav(df_pe)
    print(f"       NAV totale : {nav_base.sum():,.0f} M EUR")
    print(f"       NAV moyenne : {nav_base.mean():,.1f} M EUR")

    summary = pe_model.get_nav_summary(df_pe, nav_base)
    print("\n--- NAV par secteur ---")
    print(summary.to_pandas().to_string(index=False))

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
    sectors_arr = df_pe["sector"].to_numpy() if hasattr(df_pe, "to_numpy") else df_pe["sector"].values
    for sector in SECTORS:
        mask = sectors_arr == sector.name
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
