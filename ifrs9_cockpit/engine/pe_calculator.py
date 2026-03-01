"""Calculateur PE (Private Equity) IFRS 13.

Orchestre le calcul complet des metriques de performance PE :
    - MOIC (Multiple on Invested Capital)
    - IRR approx. (Internal Rate of Return, simplifiee sans J-curve)
    - DPI (Distributed to Paid-In)
    - RVPI (Residual Value to Paid-In)
    - TVPI (Total Value to Paid-In)

Et les analyses de risque :
    - Sensibilites factorielles dNAV/d(macro) par secteur (matrice 5x5)
    - NAV drawdown par position et agrege

Formules simplifiees (pas de cash flows intermediaires) :
    Capital_investi = Metric x Entry_Multiple x (1 - Leverage)
    MOIC = NAV / Capital_investi
    IRR_approx = MOIC^(1/holding_years) - 1
    DPI = 0 (pas de distributions intermediaires)
    RVPI = MOIC
    TVPI = DPI + RVPI = MOIC

Limites du modele :
    L1. IRR simplifiee : pas de cash flows intermediaires (appels de fonds,
        distributions partielles). L'IRR est surestimee pour vintages recents
        et ne capture pas la J-curve. Ne pas comparer aux IRR GP publiees.
    L2. Sensibilites statiques : les sensibilites PE par secteur sont des
        moyennes de cycle (expert judgment). Pas de modulation par vintage
        ni par position dans le cycle economique.
    L3. Bruit intra-sectoriel : le modele a facteur capture la correlation
        systematique (macro) et intra-sectorielle, mais pas la correlation
        idiosyncratique entre positions du meme GP ou fonds.
    L4. Pas de backtesting : les parametres sont calibres sur jugement expert
        et benchmarks industriels, pas sur historique de portefeuille reel.
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
    PE_CLASSIFICATION_CONFIG,
    REQUIRED_PE_RESULT_COLS,
    BASEL_CONFIG,
    PE_DISTRESS_LOGIT_SCALE,
)
from ifrs9_cockpit.utils.helpers import logit, expit
from ifrs9_cockpit.utils.frame_compat import to_pandas, to_polars, ensure_numpy
from ifrs9_cockpit.models.pe_model import PEModel


# Noms des 5 variables macro pour les sensibilites
_MACRO_VARS = [
    "unemployment_rate",
    "gdp_growth",
    "interest_rate",
    "hpi_growth",
    "inflation_rate",
]


class PECalculator:
    """Calculateur de metriques PE IFRS 13.

    Pipeline :
        1. Calcul NAV sous 3 scenarios via PEModel
        2. Calcul du capital investi (entry)
        3. Metriques de performance (MOIC, IRR, DPI, RVPI, TVPI)
        4. Sensibilites factorielles dNAV/d(macro)
        5. NAV drawdown et delta_nav

    Attributes:
        pe_model: Modele de valorisation PE.
    """

    def __init__(self, pe_model: Optional[PEModel] = None) -> None:
        """Initialise le calculateur PE.

        Args:
            pe_model: Modele PE calibre (cree par defaut si None).
        """
        self.pe_model = pe_model or PEModel()

    def calculate(
        self,
        df_pe: pl.DataFrame,
        unemployment_override: Optional[float] = None,
        gdp_override: Optional[float] = None,
        interest_rate_override: Optional[float] = None,
        hpi_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
        unemployment_crisis: bool = False,
    ) -> pl.DataFrame:
        """Calcule les metriques PE pour chaque position.

        Args:
            df_pe: DataFrame PE avec colonnes requises.
            unemployment_override: Override chomage pour stress test.
            gdp_override: Override PIB pour stress test.
            interest_rate_override: Override taux BCE pour stress test.
            hpi_override: Override HPI pour stress test.
            inflation_override: Override inflation pour stress test.
            unemployment_crisis: Si True, le chomage est de nature "crise
                economique" et les sensibilites PE negatives (ex. Tech)
                sont prises en abs() pour penaliser tous les secteurs.

        Returns:
            DataFrame avec colonnes ajoutees : nav, capital_invested,
            moic, irr, dpi, rvpi, tvpi, delta_nav, nav_drawdown.
        """
        # Accept both pandas and polars (incremental migration)
        df_pe = to_polars(df_pe)
        n = len(df_pe)

        # PEModel still expects pandas — convert at boundary
        df_pe_pd = to_pandas(df_pe)

        # NAV baseline (scenario base avec overrides si fournis)
        nav_base, mult_base = self.pe_model.calculate_nav(
            df_pe_pd,
            unemployment_override=unemployment_override,
            gdp_override=gdp_override,
            interest_rate_override=interest_rate_override,
            hpi_override=hpi_override,
            inflation_override=inflation_override,
            unemployment_crisis=unemployment_crisis,
        )

        # NAV reference (sans stress) pour calculer delta_nav
        nav_ref, _ = self.pe_model.calculate_nav(df_pe_pd)

        # Capital investi a l'entree
        capital = self._compute_capital_invested(df_pe)

        # Metriques de performance
        moic = np.where(capital > 0, nav_base / capital, 0)
        holding = df_pe["holding_years"].to_numpy().astype(float)
        irr = np.where(
            (moic > 0) & (holding > 0),
            np.power(moic, 1.0 / holding) - 1,
            0,
        )
        # Realistic IRR bounds: 35% = top-decile PE vintage (Cambridge Associates)
        irr = np.clip(irr, -0.50, 0.35)
        dpi = np.zeros(n)  # Pas de distributions intermediaires
        rvpi = moic.copy()
        tvpi = dpi + rvpi

        # Delta NAV et drawdown
        delta_nav = nav_base - nav_ref
        nav_drawdown = np.where(
            nav_ref > 0,
            np.maximum(0, (nav_ref - nav_base) / nav_ref),
            0,
        )

        # Probabilite de distress conditionnelle (FR16)
        distress_prob = self._calculate_distress_prob(
            df_pe, moic,
            unemployment_override=unemployment_override,
            gdp_override=gdp_override,
            interest_rate_override=interest_rate_override,
            hpi_override=hpi_override,
            inflation_override=inflation_override,
            unemployment_crisis=unemployment_crisis,
        )

        # Classification PE (FR17)
        risk_category = self._classify_pe(distress_prob)

        # Expected Loss PE (FR16) = P(distress) x LGD_equity x NAV
        lgd_eq = PE_CLASSIFICATION_CONFIG.lgd_equity
        expected_loss_pe = distress_prob * lgd_eq * nav_base

        # Cout de sortie (FR18) avec DLOM ajuste par vintage.
        # Fonds jeunes (holding < threshold) sont moins liquides -> decote plus elevee.
        # DLOM_eff = base x (1 + factor x max(0, threshold - holding) / threshold)
        # Ref: AICPA Practice Aid (2013), Pratt & Grabowski (2014).
        cfg_pe = PE_CLASSIFICATION_CONFIG
        holding = df_pe["holding_years"].to_numpy().astype(float)
        vintage_adj = np.maximum(0, cfg_pe.dlom_vintage_threshold - holding) / cfg_pe.dlom_vintage_threshold
        effective_discount = cfg_pe.secondary_discount * (1 + cfg_pe.dlom_vintage_factor * vintage_adj)
        exit_cost = nav_base * (1 - effective_discount)

        # H8 : RWA PE via score CRR3 composite (Art. 133) — position par position
        # Remplace le RW fixe par un score gradue (190/250/400)
        from ifrs9_cockpit.engine.comparator import compute_crr3_rw
        # On construit un DataFrame temporaire (pandas) avec les colonnes necessaires
        # compute_crr3_rw still expects pandas
        _tmp_pe = df_pe.clone()
        _tmp_pe = _tmp_pe.with_columns([
            pl.Series("nav", nav_base),
            pl.Series("moic", moic),
            pl.Series("distress_prob", distress_prob),
            pl.Series("capital_invested", capital),
        ])
        crr3_rw = compute_crr3_rw(to_pandas(_tmp_pe))
        rwa_pe = nav_base * crr3_rw / 100.0

        # Construire le resultat
        result = df_pe.clone()
        new_cols = [
            pl.Series("nav", np.round(nav_base, 2)),
            pl.Series("exit_multiple", mult_base),
            pl.Series("capital_invested", np.round(capital, 2)),
            pl.Series("moic", np.round(moic, 4)),
            pl.Series("irr", np.round(irr, 4)),
            pl.Series("dpi", dpi),
            pl.Series("rvpi", np.round(rvpi, 4)),
            pl.Series("tvpi", np.round(tvpi, 4)),
            pl.Series("delta_nav", np.round(delta_nav, 2)),
            pl.Series("nav_drawdown", np.round(nav_drawdown, 4)),
            pl.Series("distress_prob", np.round(distress_prob, 4)),
            pl.Series("risk_category", risk_category),
            pl.Series("expected_loss_pe", np.round(expected_loss_pe, 2)),
            pl.Series("exit_cost", np.round(exit_cost, 2)),
            pl.Series("rwa_pe", np.round(rwa_pe, 2)),
        ]
        result = result.with_columns(new_cols)

        return result

    def calculate_scenario_metrics(
        self,
        df_pe: pl.DataFrame,
    ) -> Dict[str, pl.DataFrame]:
        """Calcule les metriques PE sous les 3 scenarios.

        Args:
            df_pe: DataFrame PE.

        Returns:
            Dict {scenario_name: result_DataFrame}.
        """
        # Accept both pandas and polars (incremental migration)
        df_pe = to_polars(df_pe)
        results = {}
        for scenario in SCENARIOS:
            result = self.calculate(
                df_pe,
                unemployment_override=scenario.unemployment_rate,
                gdp_override=scenario.gdp_growth,
                interest_rate_override=scenario.interest_rate,
                hpi_override=scenario.hpi_growth,
                inflation_override=scenario.inflation_rate,
            )
            results[scenario.name] = result
        return results

    def compute_factorial_sensitivities(
        self,
        df_pe: pl.DataFrame,
        delta: float = 1.0,
    ) -> pl.DataFrame:
        """Calcule la matrice de sensibilites factorielles dNAV/d(macro).

        Perturbe chaque variable macro de +delta et mesure l'impact
        sur la NAV agregee par secteur.

        Args:
            df_pe: DataFrame PE.
            delta: Amplitude de la perturbation (en unite de la variable).

        Returns:
            DataFrame 5x5 (secteurs x variables macro) avec dNAV en %.
        """
        # Accept both pandas and polars (incremental migration)
        df_pe = to_polars(df_pe)
        base = SCENARIO_BASE

        # PEModel expects pandas
        df_pe_pd = to_pandas(df_pe)

        # NAV reference
        nav_ref, _ = self.pe_model.calculate_nav(df_pe_pd)
        sectors_arr = df_pe["sector"].to_numpy()
        nav_ref_by_sector = {}
        for sector in SECTORS:
            mask = sectors_arr == sector.name
            nav_ref_by_sector[sector.name] = nav_ref[mask].sum()

        # Perturbation par variable
        overrides_base = {
            "unemployment_override": base.unemployment_rate,
            "gdp_override": base.gdp_growth,
            "interest_rate_override": base.interest_rate,
            "hpi_override": base.hpi_growth,
            "inflation_override": base.inflation_rate,
        }

        param_map = {
            "unemployment_rate": "unemployment_override",
            "gdp_growth": "gdp_override",
            "interest_rate": "interest_rate_override",
            "hpi_growth": "hpi_override",
            "inflation_rate": "inflation_override",
        }

        records = []
        for sector in SECTORS:
            mask = sectors_arr == sector.name
            row = {"sector": sector.name}

            for var_name in _MACRO_VARS:
                # Perturber de +delta (direction adverse)
                overrides = overrides_base.copy()
                param_key = param_map[var_name]

                if var_name in ("gdp_growth", "hpi_growth"):
                    # Pour PIB et HPI, adverse = baisse
                    overrides[param_key] = overrides_base[param_key] - delta
                else:
                    # Pour chomage, taux, inflation, adverse = hausse
                    overrides[param_key] = overrides_base[param_key] + delta

                nav_perturbed, _ = self.pe_model.calculate_nav(df_pe_pd, **overrides)
                nav_perturbed_sector = nav_perturbed[mask].sum()

                # Sensibilite en % de NAV
                if nav_ref_by_sector[sector.name] > 0:
                    sensitivity = (
                        (nav_perturbed_sector - nav_ref_by_sector[sector.name])
                        / nav_ref_by_sector[sector.name]
                    ) * 100
                else:
                    sensitivity = 0.0

                row[var_name] = round(sensitivity, 2)

            records.append(row)

        return pl.DataFrame(records)

    def get_performance_summary(
        self,
        result_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Resume des metriques PE par secteur.

        Args:
            result_df: DataFrame resultat de calculate().

        Returns:
            DataFrame recapitulatif.
        """
        # Accept both pandas and polars (incremental migration)
        result_df = to_polars(result_df)
        return (
            result_df.group_by("sector")
            .agg(
                pl.col("nav").count().alias("count"),
                pl.col("nav").sum().alias("nav_total"),
                pl.col("nav").mean().alias("nav_mean"),
                pl.col("moic").mean().alias("moic_mean"),
                pl.col("irr").mean().alias("irr_mean"),
                pl.col("tvpi").mean().alias("tvpi_mean"),
                pl.col("nav_drawdown").mean().alias("drawdown_mean"),
                pl.col("capital_invested").sum().alias("capital_total"),
            )
        )

    # ------------------------------------------
    # METHODES PRIVEES
    # ------------------------------------------

    def _calculate_distress_prob(
        self,
        df_pe: pl.DataFrame,
        moic: np.ndarray,
        unemployment_override: Optional[float] = None,
        gdp_override: Optional[float] = None,
        interest_rate_override: Optional[float] = None,
        hpi_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
        unemployment_crisis: bool = False,
    ) -> np.ndarray:
        """Calcule la probabilite de distress conditionnelle (FR16, logit-space).

        logit(P_distress) = logit(base) + moic_logit_adj + macro_logit_adj
            - base = 0.05 (taux de defaut implicite PE)
            - moic_logit_adj : -2 x log(MOIC) pour MOIC < 1 (penalite logarithmique)
            - macro_logit_adj : stress macro x PE_DISTRESS_LOGIT_SCALE

        P(distress) = expit(logit_sum), automatiquement dans ]0, 1[.

        Args:
            df_pe: DataFrame PE.
            moic: MOIC par position.
            unemployment_override: Override chomage.
            gdp_override: Override PIB.
            interest_rate_override: Override taux BCE.
            hpi_override: Override HPI.
            inflation_override: Override inflation.
            unemployment_crisis: Si True, utilise abs(sensitivity) pour
                chomage PE (crise eco = toujours adverse).

        Returns:
            Array de P(distress) par position, dans ]0, 1[.
        """
        n = len(df_pe)
        base = SCENARIO_BASE

        # Probabilite de base en espace logit
        logit_distress = np.full(n, logit(0.05))

        # Ajustement MOIC en logit-space (logarithmique)
        # MOIC < 1 -> penalite proportionnelle a -log(MOIC)
        # MOIC=0.5 -> +1.4 logit, MOIC=0.1 -> +4.6 logit, MOIC>=1 -> 0
        moic_logit_adj = np.where(
            moic < 1.0,
            -2.0 * np.log(np.maximum(moic, 0.01)),
            0.0,
        )
        logit_distress += moic_logit_adj

        # Ajustement macro via sensibilites PE par secteur
        unemp = unemployment_override if unemployment_override is not None else base.unemployment_rate
        gdp = gdp_override if gdp_override is not None else base.gdp_growth
        ir = interest_rate_override if interest_rate_override is not None else base.interest_rate
        hpi = hpi_override if hpi_override is not None else base.hpi_growth
        infl = inflation_override if inflation_override is not None else base.inflation_rate

        # Deltas macro normalises (positif = adverse, /100 pour coherence logit)
        d_unemp = (unemp - base.unemployment_rate) / 100
        d_gdp = (base.gdp_growth - gdp) / 100         # Inverse : baisse PIB = adverse
        d_ir = (ir - base.interest_rate) / 100
        d_hpi = (base.hpi_growth - hpi) / 100          # Inverse : baisse HPI = adverse
        d_infl = (infl - base.inflation_rate) / 100

        sectors_arr = df_pe["sector"].to_numpy()
        for sector in SECTORS:
            mask = sectors_arr == sector.name
            if mask.sum() == 0:
                continue

            # Choc macro composite via sensibilites PE (logit-space)
            # En mode crise, abs(sensitivity) pour chomage : tout secteur souffre.
            unemp_sens = (
                abs(sector.unemployment_sensitivity_pe)
                if unemployment_crisis
                else sector.unemployment_sensitivity_pe
            )
            macro_adj = (
                d_unemp * unemp_sens
                + d_gdp * sector.gdp_sensitivity_pe
                + d_ir * sector.interest_rate_sensitivity_pe
                + d_hpi * sector.hpi_sensitivity_pe
                + d_infl * sector.inflation_sensitivity_pe
            )
            logit_distress[mask] += macro_adj * PE_DISTRESS_LOGIT_SCALE

        # Retour en espace probabilite (expit borne naturellement dans ]0, 1[)
        return expit(logit_distress)

    def _classify_pe(self, distress_prob: np.ndarray) -> np.ndarray:
        """Classifie les positions PE selon P(distress) (FR17).

        Categories :
            - Performing : P(distress) < distress_threshold_performing
            - Watchlist  : P(distress) < distress_threshold_watchlist
            - Distressed : P(distress) >= distress_threshold_watchlist

        Args:
            distress_prob: Array de P(distress) par position.

        Returns:
            Array de categories (str).
        """
        cfg = PE_CLASSIFICATION_CONFIG
        n = len(distress_prob)
        categories = np.full(n, "Performing", dtype=object)

        watchlist_mask = distress_prob >= cfg.distress_threshold_performing
        distressed_mask = distress_prob >= cfg.distress_threshold_watchlist

        categories[watchlist_mask] = "Watchlist"
        categories[distressed_mask] = "Distressed"

        return categories

    def _compute_capital_invested(self, df_pe: pl.DataFrame) -> np.ndarray:
        """Calcule le capital investi a l'entree (equity portion du LBO).

        Capital = Metric x Entry_Multiple x (1 - Leverage)
        Pour Cap_rate/NOI (Immobilier) : Metric = EBITDA x (1 - NOI_OPEX_RATIO).

        Args:
            df_pe: DataFrame PE.

        Returns:
            Array de capital investi en M EUR.
        """
        from ifrs9_cockpit.config import NOI_OPEX_RATIO

        n = len(df_pe)
        capital = np.zeros(n)

        # Pre-extract numpy arrays for masked indexing
        sectors_arr = df_pe["sector"].to_numpy()
        revenue_arr = df_pe["revenue"].to_numpy().astype(float)
        ebitda_arr = df_pe["ebitda"].to_numpy().astype(float)
        entry_mult_arr = df_pe["entry_multiple"].to_numpy().astype(float)
        leverage_arr = df_pe["leverage"].to_numpy().astype(float)

        for sector in SECTORS:
            mask = sectors_arr == sector.name
            if mask.sum() == 0:
                continue

            # Metrique d'entree selon la methode IPEV
            if sector.valuation_method == "EV/Revenue":
                metric = revenue_arr[mask]
            else:
                metric = ebitda_arr[mask]
                if sector.valuation_method == "Cap_rate/NOI":
                    metric = metric * (1 - NOI_OPEX_RATIO)

            entry_mult = entry_mult_arr[mask]
            leverage = leverage_arr[mask]

            capital[mask] = metric * entry_mult * (1 - leverage)

        # Floor a 1 EUR (1e-6 M EUR) — tout investissement PE a un capital > 0
        return np.maximum(capital, 1e-6)


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset
    from ifrs9_cockpit.utils.helpers import format_pct

    print("=" * 65)
    print("IFRS 9 COCKPIT — PE Calculator : Distress, Classification & Cout de Sortie")
    print("=" * 65)

    # 1. Data
    print("\n[1/5] Generation des donnees...")
    df_credit, df_pe_raw, df_history, _ = generate_dataset()
    # Ensure Polars (generate_dataset may return either)
    from ifrs9_cockpit.utils.frame_compat import to_polars
    df_pe = to_polars(df_pe_raw)
    print(f"       {len(df_pe):,} positions PE")

    # 2. Metriques baseline (inclut distress, classification, EL, exit_cost, rwa)
    print("\n[2/5] Calcul metriques baseline + distress/classification...")
    pe_calc = PECalculator()
    result = pe_calc.calculate(df_pe)

    # Resume classification
    print("\n--- Classification PE (baseline) ---")
    vc = result["risk_category"].value_counts()
    vc_dict = dict(zip(vc["risk_category"].to_list(), vc["count"].to_list()))
    for cat in ["Performing", "Watchlist", "Distressed"]:
        cnt = vc_dict.get(cat, 0)
        pct = cnt / len(result) * 100
        print(f"  {cat:12s} : {cnt:>5,} positions ({pct:5.1f}%)")

    # Resume distress / EL
    print(f"\n--- Distress & Expected Loss (baseline) ---")
    print(f"  P(distress) : mean = {result['distress_prob'].mean():.4f}, "
          f"max = {result['distress_prob'].max():.4f}")
    print(f"  Expected Loss PE totale : {result['expected_loss_pe'].sum():>12,.0f} M EUR")
    print(f"  Exit cost total         : {result['exit_cost'].sum():>12,.0f} M EUR")
    print(f"  RWA PE total            : {result['rwa_pe'].sum():>12,.0f} M EUR")

    # 3. Scenarios
    print("\n[3/5] Metriques sous 3 scenarios...")
    scenario_results = pe_calc.calculate_scenario_metrics(df_pe)
    for name, res in scenario_results.items():
        irr_mean = res["irr"].mean()
        n_distressed = (res["risk_category"] == "Distressed").sum()
        el_total = res["expected_loss_pe"].sum()
        print(f"  {name:12s} : IRR = {format_pct(irr_mean)} | "
              f"Distressed = {n_distressed:>4,} | "
              f"EL_PE = {el_total:>10,.0f} M EUR")

    # 4. Sensibilites factorielles
    print("\n[4/5] Sensibilites factorielles dNAV/d(macro) (%, +1pp adverse) :")
    sensitivities = pe_calc.compute_factorial_sensitivities(df_pe, delta=1.0)
    print(sensitivities.to_pandas().set_index("sector").to_string())

    # 5. Performance par secteur
    print("\n[5/5] Performance par secteur ---")
    summary = pe_calc.get_performance_summary(result)
    print(to_pandas(summary).to_string(index=False))

    # -- Validations --
    print("\n--- Validations ---")
    all_ok = True

    # V1: risk_category in {Performing, Watchlist, Distressed}
    valid_cats = {"Performing", "Watchlist", "Distressed"}
    actual_cats = set(result["risk_category"].unique().to_list())
    ok = actual_cats.issubset(valid_cats)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] risk_category in {{Performing, Watchlist, Distressed}} "
          f"(actual: {actual_cats})")
    all_ok &= ok

    # V2: expected_loss_pe >= 0
    ok = (result["expected_loss_pe"] >= 0).all()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] expected_loss_pe >= 0 (min = {result['expected_loss_pe'].min():.2f})")
    all_ok &= ok

    # V3: distress_prob dans [0, 1]
    ok = (result["distress_prob"] >= 0).all() and (result["distress_prob"] <= 1).all()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] distress_prob in [0,1] "
          f"(min={result['distress_prob'].min():.4f}, max={result['distress_prob'].max():.4f})")
    all_ok &= ok

    # V4: Stress extreme -> plus de Distressed vs baseline
    n_distressed_base = (scenario_results["Base"]["risk_category"] == "Distressed").sum()
    n_distressed_adv = (scenario_results["Adverse"]["risk_category"] == "Distressed").sum()
    ok = n_distressed_adv >= n_distressed_base
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Adverse Distressed ({n_distressed_adv}) >= "
          f"Base Distressed ({n_distressed_base})")
    all_ok &= ok

    # V5: REQUIRED_PE_RESULT_COLS presentes
    missing_cols = REQUIRED_PE_RESULT_COLS - set(result.columns)
    ok = len(missing_cols) == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] REQUIRED_PE_RESULT_COLS presentes "
          f"(manquantes: {missing_cols if missing_cols else 'aucune'})")
    all_ok &= ok

    # V6: IRR ordering
    irr_base = scenario_results["Base"]["irr"].mean()
    irr_adv = scenario_results["Adverse"]["irr"].mean()
    irr_fav = scenario_results["Favorable"]["irr"].mean()
    ok = irr_adv < irr_base < irr_fav
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] IRR ordering: Adverse ({format_pct(irr_adv)}) < "
          f"Base ({format_pct(irr_base)}) < Favorable ({format_pct(irr_fav)})")
    all_ok &= ok

    # V7: TVPI = DPI + RVPI
    tvpi = result["tvpi"].to_numpy()
    dpi = result["dpi"].to_numpy()
    rvpi = result["rvpi"].to_numpy()
    ok = np.allclose(tvpi, dpi + rvpi, atol=0.001)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] TVPI = DPI + RVPI (max diff = {np.max(np.abs(tvpi - dpi - rvpi)):.6f})")
    all_ok &= ok

    # V8: Sensibilites non-nulles
    sens_pd = sensitivities.to_pandas().set_index("sector")
    ok = (sens_pd.abs() > 0).any().all()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Sensibilites 5x5 non-nulles")
    all_ok &= ok

    # V9: NAV >= 0
    ok = (result["nav"] >= 0).all()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] NAV >= 0 (min = {result['nav'].min():.2f})")
    all_ok &= ok

    # V10: exit_cost > 0 pour toutes les positions
    ok = (result["exit_cost"] > 0).all()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] exit_cost > 0 (min = {result['exit_cost'].min():.2f})")
    all_ok &= ok

    # V11: rwa_pe > 0
    ok = (result["rwa_pe"] > 0).all()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] rwa_pe > 0 (min = {result['rwa_pe'].min():.2f})")
    all_ok &= ok

    print(f"\n{'=' * 65}")
    if all_ok:
        print("PE Calculator (Distress + Classification + Cout de Sortie) valide.")
    else:
        print("ATTENTION : certaines validations ont echoue.")
    print(f"{'=' * 65}")
