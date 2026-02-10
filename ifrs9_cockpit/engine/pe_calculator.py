"""Calculateur PE (Private Equity) IFRS 13.

Orchestre le calcul complet des metriques de performance PE :
    - MOIC (Multiple on Invested Capital)
    - IRR (Internal Rate of Return)
    - DPI (Distributed to Paid-In)
    - RVPI (Residual Value to Paid-In)
    - TVPI (Total Value to Paid-In)

Et les analyses de risque :
    - Sensibilites factorielles dNAV/d(macro) par secteur (matrice 5x5)
    - NAV drawdown par position et agrege

Formules simplifiees (pas de cash flows intermediaires) :
    Capital_investi = Metric x Entry_Multiple x (1 - Leverage)
    MOIC = NAV / Capital_investi
    IRR = MOIC^(1/holding_years) - 1
    DPI = 0 (pas de distributions intermediaires)
    RVPI = MOIC
    TVPI = DPI + RVPI = MOIC
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
    PE_CLASSIFICATION_CONFIG,
    REQUIRED_PE_RESULT_COLS,
    BASEL_CONFIG,
    PE_DISTRESS_LOGIT_SCALE,
)
from ifrs9_cockpit.utils.helpers import logit, expit
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
        df_pe: pd.DataFrame,
        unemployment_override: Optional[float] = None,
        gdp_override: Optional[float] = None,
        interest_rate_override: Optional[float] = None,
        hpi_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
    ) -> pd.DataFrame:
        """Calcule les metriques PE pour chaque position.

        Args:
            df_pe: DataFrame PE avec colonnes requises.
            unemployment_override: Override chomage pour stress test.
            gdp_override: Override PIB pour stress test.
            interest_rate_override: Override taux BCE pour stress test.
            hpi_override: Override HPI pour stress test.
            inflation_override: Override inflation pour stress test.

        Returns:
            DataFrame avec colonnes ajoutees : nav, capital_invested,
            moic, irr, dpi, rvpi, tvpi, delta_nav, nav_drawdown.
        """
        n = len(df_pe)

        # NAV baseline (scenario base avec overrides si fournis)
        nav_base, mult_base = self.pe_model.calculate_nav(
            df_pe,
            unemployment_override=unemployment_override,
            gdp_override=gdp_override,
            interest_rate_override=interest_rate_override,
            hpi_override=hpi_override,
            inflation_override=inflation_override,
        )

        # NAV reference (sans stress) pour calculer delta_nav
        nav_ref, _ = self.pe_model.calculate_nav(df_pe)

        # Capital investi a l'entree
        capital = self._compute_capital_invested(df_pe)

        # Metriques de performance
        moic = np.where(capital > 0, nav_base / capital, 0)
        holding = df_pe["holding_years"].values.astype(float)
        irr = np.where(
            (moic > 0) & (holding > 0),
            np.power(moic, 1.0 / holding) - 1,
            0,
        )
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
        )

        # Classification PE (FR17)
        risk_category = self._classify_pe(distress_prob)

        # Expected Loss PE (FR16) = P(distress) x LGD_equity x NAV
        lgd_eq = PE_CLASSIFICATION_CONFIG.lgd_equity
        expected_loss_pe = distress_prob * lgd_eq * nav_base

        # Cout de sortie (FR18) = NAV x (1 - secondary_discount)
        sec_disc = PE_CLASSIFICATION_CONFIG.secondary_discount
        exit_cost = nav_base * (1 - sec_disc)

        # H8 : RWA PE via score CRR3 composite (Art. 133) — position par position
        # Remplace le RW fixe par un score gradue (190/250/400)
        from ifrs9_cockpit.engine.comparator import compute_crr3_rw
        # On construit un DataFrame temporaire avec les colonnes necessaires
        _tmp_pe = df_pe.copy()
        _tmp_pe["nav"] = nav_base
        _tmp_pe["moic"] = moic
        _tmp_pe["distress_prob"] = distress_prob
        _tmp_pe["capital_invested"] = capital
        crr3_rw = compute_crr3_rw(_tmp_pe)
        rwa_pe = nav_base * crr3_rw / 100.0

        # Construire le resultat
        result = df_pe.copy()
        result["nav"] = np.round(nav_base, 2)
        result["exit_multiple"] = mult_base
        result["capital_invested"] = np.round(capital, 2)
        result["moic"] = np.round(moic, 4)
        result["irr"] = np.round(irr, 4)
        result["dpi"] = dpi
        result["rvpi"] = np.round(rvpi, 4)
        result["tvpi"] = np.round(tvpi, 4)
        result["delta_nav"] = np.round(delta_nav, 2)
        result["nav_drawdown"] = np.round(nav_drawdown, 4)
        result["distress_prob"] = np.round(distress_prob, 4)
        result["risk_category"] = risk_category
        result["expected_loss_pe"] = np.round(expected_loss_pe, 2)
        result["exit_cost"] = np.round(exit_cost, 2)
        result["rwa_pe"] = np.round(rwa_pe, 2)

        return result

    def calculate_scenario_metrics(
        self,
        df_pe: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Calcule les metriques PE sous les 3 scenarios.

        Args:
            df_pe: DataFrame PE.

        Returns:
            Dict {scenario_name: result_DataFrame}.
        """
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
        df_pe: pd.DataFrame,
        delta: float = 1.0,
    ) -> pd.DataFrame:
        """Calcule la matrice de sensibilites factorielles dNAV/d(macro).

        Perturbe chaque variable macro de +delta et mesure l'impact
        sur la NAV agregee par secteur.

        Args:
            df_pe: DataFrame PE.
            delta: Amplitude de la perturbation (en unite de la variable).

        Returns:
            DataFrame 5x5 (secteurs x variables macro) avec dNAV en %.
        """
        base = SCENARIO_BASE

        # NAV reference
        nav_ref, _ = self.pe_model.calculate_nav(df_pe)
        nav_ref_by_sector = {}
        for sector in SECTORS:
            mask = df_pe["sector"].values == sector.name
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
            mask = df_pe["sector"].values == sector.name
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

                nav_perturbed, _ = self.pe_model.calculate_nav(df_pe, **overrides)
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

        return pd.DataFrame(records).set_index("sector")

    def get_performance_summary(
        self,
        result_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Resume des metriques PE par secteur.

        Args:
            result_df: DataFrame resultat de calculate().

        Returns:
            DataFrame recapitulatif.
        """
        return (
            result_df.groupby("sector")
            .agg(
                count=("nav", "size"),
                nav_total=("nav", "sum"),
                nav_mean=("nav", "mean"),
                moic_mean=("moic", "mean"),
                irr_mean=("irr", "mean"),
                tvpi_mean=("tvpi", "mean"),
                drawdown_mean=("nav_drawdown", "mean"),
                capital_total=("capital_invested", "sum"),
            )
            .round(4)
            .reset_index()
        )

    # ──────────────────────────────────────────
    # METHODES PRIVEES
    # ──────────────────────────────────────────

    def _calculate_distress_prob(
        self,
        df_pe: pd.DataFrame,
        moic: np.ndarray,
        unemployment_override: Optional[float] = None,
        gdp_override: Optional[float] = None,
        interest_rate_override: Optional[float] = None,
        hpi_override: Optional[float] = None,
        inflation_override: Optional[float] = None,
    ) -> np.ndarray:
        """Calcule la probabilite de distress conditionnelle (FR16, logit-space).

        logit(P_distress) = logit(base) + moic_logit_adj + macro_logit_adj
            - base = 0.05 (taux de defaut implicite PE)
            - moic_logit_adj : -2 × log(MOIC) pour MOIC < 1 (penalite logarithmique)
            - macro_logit_adj : stress macro × PE_DISTRESS_LOGIT_SCALE

        P(distress) = expit(logit_sum), automatiquement dans ]0, 1[.

        Args:
            df_pe: DataFrame PE.
            moic: MOIC par position.
            unemployment_override: Override chomage.
            gdp_override: Override PIB.
            interest_rate_override: Override taux BCE.
            hpi_override: Override HPI.
            inflation_override: Override inflation.

        Returns:
            Array de P(distress) par position, dans ]0, 1[.
        """
        n = len(df_pe)
        base = SCENARIO_BASE

        # Probabilite de base en espace logit
        logit_distress = np.full(n, logit(0.05))

        # Ajustement MOIC en logit-space (logarithmique)
        # MOIC < 1 → penalite proportionnelle a -log(MOIC)
        # MOIC=0.5 → +1.4 logit, MOIC=0.1 → +4.6 logit, MOIC>=1 → 0
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

        for sector in SECTORS:
            mask = df_pe["sector"].values == sector.name
            if mask.sum() == 0:
                continue

            # Choc macro composite via sensibilites PE (logit-space)
            macro_adj = (
                d_unemp * sector.unemployment_sensitivity_pe
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

    def _compute_capital_invested(self, df_pe: pd.DataFrame) -> np.ndarray:
        """Calcule le capital investi a l'entree (equity portion du LBO).

        Capital = Metric x Entry_Multiple x (1 - Leverage)

        Args:
            df_pe: DataFrame PE.

        Returns:
            Array de capital investi en M EUR.
        """
        n = len(df_pe)
        capital = np.zeros(n)

        for sector in SECTORS:
            mask = df_pe["sector"].values == sector.name
            if mask.sum() == 0:
                continue

            # Metrique d'entree selon la methode IPEV
            if sector.valuation_method == "EV/Revenue":
                metric = df_pe.loc[mask, "revenue"].values.astype(float)
            else:
                metric = df_pe.loc[mask, "ebitda"].values.astype(float)

            entry_mult = df_pe.loc[mask, "entry_multiple"].values.astype(float)
            leverage = df_pe.loc[mask, "leverage"].values.astype(float)

            capital[mask] = metric * entry_mult * (1 - leverage)

        return np.maximum(capital, 0)


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset
    from ifrs9_cockpit.utils.helpers import format_pct

    print("=" * 65)
    print("IFRS 9 COCKPIT — PE Calculator : Distress, Classification & Cout de Sortie")
    print("=" * 65)

    # 1. Data
    print("\n[1/5] Generation des donnees...")
    df_credit, df_pe, df_history = generate_dataset()
    print(f"       {len(df_pe):,} positions PE")

    # 2. Metriques baseline (inclut distress, classification, EL, exit_cost, rwa)
    print("\n[2/5] Calcul metriques baseline + distress/classification...")
    pe_calc = PECalculator()
    result = pe_calc.calculate(df_pe)

    # Resume classification
    print("\n--- Classification PE (baseline) ---")
    cat_counts = result["risk_category"].value_counts()
    for cat in ["Performing", "Watchlist", "Distressed"]:
        cnt = cat_counts.get(cat, 0)
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
    print(sensitivities.to_string())

    # 5. Performance par secteur
    print("\n[5/5] Performance par secteur ---")
    summary = pe_calc.get_performance_summary(result)
    print(summary.to_string(index=False))

    # ── Validations ──
    print("\n--- Validations ---")
    all_ok = True

    # V1: risk_category in {Performing, Watchlist, Distressed}
    valid_cats = {"Performing", "Watchlist", "Distressed"}
    actual_cats = set(result["risk_category"].unique())
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
    tvpi = result["tvpi"].values
    dpi = result["dpi"].values
    rvpi = result["rvpi"].values
    ok = np.allclose(tvpi, dpi + rvpi, atol=0.001)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] TVPI = DPI + RVPI (max diff = {np.max(np.abs(tvpi - dpi - rvpi)):.6f})")
    all_ok &= ok

    # V8: Sensibilites non-nulles
    ok = (sensitivities.abs() > 0).any().all()
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
