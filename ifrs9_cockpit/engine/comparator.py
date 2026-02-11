"""Comparateur Credit / PE et metriques avancees.

Module central de l'Epic 4 : compare les portefeuilles credit et PE,
calcule les metriques avancees (FR42), la concentration cross-cell (FR41),
et les denominateurs communs RAROC/EVA (FR44).

Pipeline :
    1. Metriques credit avancees (NPL ratio, cost of risk, Texas ratio, etc.)
    2. HHI cross-cell (10 cellules : 5 secteurs x 2 canaux)
    3. RAROC / EVA par cellule (secteur x canal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from ifrs9_cockpit.config import (
    BASEL_CONFIG,
    IFRS9_CONFIG,
    SECTORS,
    SCENARIOS,
    SCENARIO_BASE,
    REQUIRED_CREDIT_RESULT_COLS,
    REQUIRED_PE_RESULT_COLS,
    RISK_APPETITE_CONFIG,
)

# Capital de base (fonds propres) pour les ratios prudentiels
_CAPITAL_BASE: float = BASEL_CONFIG.rwa_budget * BASEL_CONFIG.cet1_target


def compute_crr3_rw(result_pe: pd.DataFrame) -> np.ndarray:
    """Score CRR3 composite (Art. 133) pour Risk Weight PE.

    Combine 4 dimensions de risque pour attribuer un RW gradue :
        - Performance financiere (MOIC)
        - Qualite du portefeuille (P(distress))
        - Risque de levier
        - Environnement macro

    Score = 0.30 x financial_perf + 0.25 x portfolio_quality
          + 0.25 x leverage_risk + 0.20 x macro_env

    Classification CRR3 :
        - score < 0.25 → 190% (IRB diversifie)
        - score < 0.50 → 250% (equity general)
        - score >= 0.50 → 400% (speculatif)

    Args:
        result_pe: DataFrame PE avec colonnes moic, nav, p_distress/distress_prob,
                   leverage, risk_category.

    Returns:
        Array de Risk Weights (190, 250 ou 400) par position.
    """
    n = len(result_pe)

    # Performance financiere (MOIC-based)
    moic = result_pe.get("moic", pd.Series(np.ones(n))).values.astype(float)
    nav_initial = result_pe.get(
        "nav_initial",
        result_pe.get("capital_invested", pd.Series(result_pe["nav"].values))
    ).values.astype(float)
    effective_moic = np.where(
        moic > 0,
        moic,
        result_pe["nav"].values / np.maximum(nav_initial, 1.0),
    )
    financial_perf = np.where(
        effective_moic < 1.5,
        np.clip(1.0 - effective_moic, 0, 1),
        0.0,
    )

    # Qualite du portefeuille (P(distress))
    portfolio_quality = result_pe.get(
        "p_distress",
        result_pe.get("distress_prob", pd.Series(np.zeros(n)))
    ).values.astype(float)

    # Risque de levier (normalise par 0.95)
    leverage = result_pe.get("leverage", pd.Series(np.full(n, 0.5))).values.astype(float)
    leverage_risk = np.clip(leverage / 0.95, 0, 1)

    # Environnement macro (reutilise P(distress) comme proxy)
    macro_env = portfolio_quality.copy()

    score = (
        0.30 * financial_perf
        + 0.25 * portfolio_quality
        + 0.25 * leverage_risk
        + 0.20 * macro_env
    )

    rw = np.where(score < 0.25, 190, np.where(score < 0.50, 250, 400))
    return rw


class PortfolioComparator:
    """Comparateur de portefeuilles credit et PE.

    Orchestre le calcul des metriques avancees, de la concentration
    et des denominateurs communs (RAROC, EVA) pour la comparaison
    cross-canal.

    Attributes:
        result_credit: DataFrame resultat du pipeline credit (ECLCalculator).
        result_pe: DataFrame resultat du pipeline PE (PECalculator).
    """

    def __init__(
        self,
        result_credit: pd.DataFrame,
        result_pe: pd.DataFrame,
    ) -> None:
        """Initialise le comparateur.

        Args:
            result_credit: Resultat ECLCalculator.calculate().
            result_pe: Resultat PECalculator.calculate().

        Raises:
            AssertionError: Si les colonnes requises sont absentes.
        """
        missing_credit = REQUIRED_CREDIT_RESULT_COLS - set(result_credit.columns)
        assert not missing_credit, f"Colonnes credit manquantes: {missing_credit}"

        missing_pe = REQUIRED_PE_RESULT_COLS - set(result_pe.columns)
        assert not missing_pe, f"Colonnes PE manquantes: {missing_pe}"

        self.result_credit = result_credit
        self.result_pe = result_pe
        self._raroc_eva_cache: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────
    # METRIQUES CREDIT AVANCEES (FR42)
    # ──────────────────────────────────────────────

    def compute_advanced_credit_metrics(self) -> pd.DataFrame:
        """Calcule les metriques credit avancees par secteur (FR42).

        Metriques :
            - npl_ratio : EAD Stage 3 / EAD total
            - cost_of_risk_bps : ECL / (EAD x T_moyen) x 10 000 (M8 annualise)
            - coverage_ratio : ECL Stage 3 / EAD Stage 3 (M10 unifie)
            - texas_ratio : EAD Stage 3 / (ECL Stage 3 + capital_share) (C3 corrige)
            - pd_mean : PD 12m moyenne
            - stage2_pct : Part Stage 2 en EAD
            - rwa_density : RWA / EAD

        Returns:
            DataFrame avec metriques par secteur + ligne Total.
        """
        df = self.result_credit
        records = []

        for sector in SECTORS:
            mask = df["sector"].values == sector.name
            df_sec = df.loc[mask]

            if len(df_sec) == 0:
                continue

            ead_total = df_sec["ead"].sum()
            ecl_total = df_sec["ecl_weighted"].sum()
            rwa_total = df_sec["rwa_credit"].sum()

            ead_s3 = df_sec.loc[df_sec["stage"] == 3, "ead"].sum()
            ead_s2 = df_sec.loc[df_sec["stage"] == 2, "ead"].sum()
            ecl_s3 = df_sec.loc[df_sec["stage"] == 3, "ecl_weighted"].sum()

            # Part du capital proportionnelle au secteur
            capital_share = _CAPITAL_BASE * (ead_total / max(df["ead"].sum(), 1))

            # M8 : Cost of Risk annualise (divise par maturite moyenne residuelle)
            t_moyen = IFRS9_CONFIG.lifetime_horizon_years

            records.append({
                "sector": sector.name,
                "canal": "Credit",
                "count": len(df_sec),
                "ead_total": round(ead_total, 0),
                "ecl_total": round(ecl_total, 0),
                "npl_ratio": round(ead_s3 / max(ead_total, 1), 4),
                "cost_of_risk_bps": round(
                    ecl_total / max(ead_total * t_moyen, 1) * 10_000, 1
                ),
                # M10 : Coverage = provisions Stage 3 / exposition Stage 3
                "coverage_ratio": round(ecl_s3 / max(ead_s3, 1), 4) if ead_s3 > 0 else 0.0,
                # C3 : Texas = EAD_S3 / (ECL_S3 + capital_share) — provisions NPL uniquement
                "texas_ratio": round(ead_s3 / max(ecl_s3 + capital_share, 1), 4),
                "pd_mean": round(df_sec["pd_12m"].mean(), 4),
                "stage2_pct": round(ead_s2 / max(ead_total, 1), 4),
                "rwa_density": round(rwa_total / max(ead_total, 1), 4),
            })

        result = pd.DataFrame(records)

        # Ligne Total
        ead_all = df["ead"].sum()
        ecl_all = df["ecl_weighted"].sum()
        rwa_all = df["rwa_credit"].sum()
        ead_s3_all = df.loc[df["stage"] == 3, "ead"].sum()
        ead_s2_all = df.loc[df["stage"] == 2, "ead"].sum()
        ecl_s3_all = df.loc[df["stage"] == 3, "ecl_weighted"].sum()
        t_moyen_all = IFRS9_CONFIG.lifetime_horizon_years

        total_row = pd.DataFrame([{
            "sector": "Total",
            "canal": "Credit",
            "count": len(df),
            "ead_total": round(ead_all, 0),
            "ecl_total": round(ecl_all, 0),
            "npl_ratio": round(ead_s3_all / max(ead_all, 1), 4),
            "cost_of_risk_bps": round(
                ecl_all / max(ead_all * t_moyen_all, 1) * 10_000, 1
            ),
            # M10 : Coverage = provisions Stage 3 / exposition Stage 3
            "coverage_ratio": round(ecl_s3_all / max(ead_s3_all, 1), 4) if ead_s3_all > 0 else 0.0,
            # C3 : Texas = EAD_S3 / (ECL_S3 + capital) — provisions NPL uniquement
            "texas_ratio": round(ead_s3_all / max(ecl_s3_all + _CAPITAL_BASE, 1), 4),
            "pd_mean": round(df["pd_12m"].mean(), 4),
            "stage2_pct": round(ead_s2_all / max(ead_all, 1), 4),
            "rwa_density": round(rwa_all / max(ead_all, 1), 4),
        }])

        return pd.concat([result, total_row], ignore_index=True)

    # ──────────────────────────────────────────────
    # HHI CROSS-CELL (FR41)
    # ──────────────────────────────────────────────

    def compute_hhi_crosscell(self) -> Dict[str, object]:
        """Calcule le HHI par secteur et cross-cell (FR41).

        Cross-cell = 10 cellules (5 secteurs x 2 canaux : Credit + PE).
        Part de chaque cellule = exposure / exposure totale.

        Returns:
            Dict avec :
                - hhi_credit : HHI du portefeuille credit (5 secteurs)
                - hhi_pe : HHI du portefeuille PE (5 secteurs)
                - hhi_crosscell : HHI des 10 cellules secteur x canal
                - shares : DataFrame des parts par cellule
        """
        df_c = self.result_credit
        df_p = self.result_pe

        ead_total = df_c["ead"].sum()
        nav_total = df_p["nav"].sum()
        exposure_total = ead_total + nav_total

        shares_data = []

        for sector in SECTORS:
            # Credit
            mask_c = df_c["sector"].values == sector.name
            ead_sec = df_c.loc[mask_c, "ead"].sum()
            share_c = ead_sec / max(exposure_total, 1)
            shares_data.append({
                "sector": sector.name,
                "canal": "Credit",
                "exposure": ead_sec,
                "share": share_c,
            })

            # PE
            mask_p = df_p["sector"].values == sector.name
            nav_sec = df_p.loc[mask_p, "nav"].sum()
            share_p = nav_sec / max(exposure_total, 1)
            shares_data.append({
                "sector": sector.name,
                "canal": "PE",
                "exposure": nav_sec,
                "share": share_p,
            })

        shares_df = pd.DataFrame(shares_data)

        # M9 : HHI full precision — arrondir uniquement a l'affichage
        # HHI credit (5 cellules)
        credit_shares = shares_df.loc[shares_df["canal"] == "Credit", "exposure"].values
        credit_total = credit_shares.sum()
        hhi_credit = float(np.sum((credit_shares / max(credit_total, 1)) ** 2) * 10_000)

        # HHI PE (5 cellules)
        pe_shares = shares_df.loc[shares_df["canal"] == "PE", "exposure"].values
        pe_total = pe_shares.sum()
        hhi_pe = float(np.sum((pe_shares / max(pe_total, 1)) ** 2) * 10_000)

        # HHI cross-cell (10 cellules)
        all_shares = shares_df["share"].values
        hhi_crosscell = float(np.sum(all_shares ** 2) * 10_000)

        return {
            "hhi_credit": hhi_credit,
            "hhi_pe": hhi_pe,
            "hhi_crosscell": hhi_crosscell,
            "shares": shares_df,
        }

    # ──────────────────────────────────────────────
    # RAROC / EVA PAR CELLULE (FR44)
    # ──────────────────────────────────────────────

    def compute_raroc_eva(self) -> pd.DataFrame:
        """Calcule RAROC et EVA par cellule secteur x canal (FR44).

        Resultat mis en cache apres le premier appel (les donnees ne changent
        pas au sein d'une instance).

        H5 : Formule RAROC complete avec CIR et impots.

        Credit :
            credit_spread = -ln(1 - PD*LGD) + liquidity_premium (Merton)
            NII = sum(EAD_i x credit_spread_i)
            Revenue_net = NII x (1 - CIR)
            Profit_net = (Revenue_net - ECL) x (1 - tax_rate)
            RAROC = Profit_net / (RWA x CET1_target)

        PE :
            Revenue_net = (NAV x IRR) x (1 - CIR)
            Profit_net = (Revenue_net - EL_PE) x (1 - tax_rate)
            RAROC = Profit_net / (RWA_PE x CET1_target)

        EVA = (RAROC - cost_of_capital) x Capital_allocated

        Returns:
            DataFrame avec RAROC et EVA par cellule (10 lignes + 2 totaux).
        """
        if self._raroc_eva_cache is not None:
            return self._raroc_eva_cache
        df_c = self.result_credit
        df_p = self.result_pe
        coc = BASEL_CONFIG.cet1_target

        records = []

        for sector in SECTORS:
            # ── Credit ──
            mask_c = df_c["sector"].values == sector.name
            df_sec_c = df_c.loc[mask_c]

            if len(df_sec_c) > 0:
                ead = df_sec_c["ead"].sum()
                ecl = df_sec_c["ecl_weighted"].sum()
                rwa = df_sec_c["rwa_credit"].sum()

                # H5 : RAROC complet avec CIR et impots
                # Credit spread Merton : -ln(1 - PD*LGD) + prime de liquidite
                pd_arr = df_sec_c["pd_12m"].values.astype(float)
                lgd_arr = df_sec_c["lgd"].values.astype(float)
                ead_arr = df_sec_c["ead"].values.astype(float)
                liq_premium = BASEL_CONFIG.liquidity_premium_bps / 10_000
                cs_arr = -np.log(np.maximum(1 - pd_arr * lgd_arr, 1e-10)) / 1.0 + liq_premium
                cs_arr = np.clip(cs_arr, 0.0050, 0.2000)
                nii = float(np.sum(ead_arr * cs_arr))
                revenue_net = nii * (1 - BASEL_CONFIG.cir)
                profit_net = (revenue_net - ecl) * (1 - BASEL_CONFIG.tax_rate)
                capital_c = rwa * coc
                raroc_c = profit_net / max(capital_c, 1)
                eva_c = (raroc_c - coc) * capital_c

                records.append({
                    "sector": sector.name,
                    "canal": "Credit",
                    "exposure": round(ead, 0),
                    "revenue": round(nii, 0),
                    "loss": round(ecl, 0),
                    "rwa": round(rwa, 0),
                    "capital": round(capital_c, 0),
                    "raroc": round(raroc_c, 4),
                    "eva": round(eva_c, 0),
                })

            # ── PE ──
            mask_p = df_p["sector"].values == sector.name
            df_sec_p = df_p.loc[mask_p]

            if len(df_sec_p) > 0:
                nav = df_sec_p["nav"].sum()
                el_pe = df_sec_p["expected_loss_pe"].sum()
                rwa_pe = df_sec_p["rwa_pe"].sum()
                # Revenu PE = NAV x IRR moyen (rendement annuel)
                irr_mean = df_sec_p["irr"].mean()
                rev_pe = nav * irr_mean
                # H5 : RAROC complet avec CIR et impots
                revenue_net_pe = rev_pe * (1 - BASEL_CONFIG.cir)
                profit_net_pe = (revenue_net_pe - el_pe) * (1 - BASEL_CONFIG.tax_rate)
                capital_p = rwa_pe * coc
                raroc_p = profit_net_pe / max(capital_p, 1)
                eva_p = (raroc_p - coc) * capital_p

                records.append({
                    "sector": sector.name,
                    "canal": "PE",
                    "exposure": round(nav, 0),
                    "revenue": round(rev_pe, 0),
                    "loss": round(el_pe, 0),
                    "rwa": round(rwa_pe, 0),
                    "capital": round(capital_p, 0),
                    "raroc": round(raroc_p, 4),
                    "eva": round(eva_p, 0),
                })

        result = pd.DataFrame(records)

        # Totaux par canal
        for canal in ["Credit", "PE"]:
            sub = result.loc[result["canal"] == canal]
            if len(sub) == 0:
                continue
            exp = sub["exposure"].sum()
            rev = sub["revenue"].sum()
            loss = sub["loss"].sum()
            rwa = sub["rwa"].sum()
            cap = sub["capital"].sum()
            # H5 : RAROC complet avec CIR et impots
            rev_net = rev * (1 - BASEL_CONFIG.cir)
            profit = (rev_net - loss) * (1 - BASEL_CONFIG.tax_rate)
            raroc = profit / max(cap, 1)
            eva = (raroc - coc) * cap

            result = pd.concat([result, pd.DataFrame([{
                "sector": "Total",
                "canal": canal,
                "exposure": round(exp, 0),
                "revenue": round(rev, 0),
                "loss": round(loss, 0),
                "rwa": round(rwa, 0),
                "capital": round(cap, 0),
                "raroc": round(raroc, 4),
                "eva": round(eva, 0),
            }])], ignore_index=True)

        self._raroc_eva_cache = result
        return result

    # ──────────────────────────────────────────────
    # SCORE DE RESILIENCE (FR19)
    # ──────────────────────────────────────────────

    def _resilience_score_credit(self) -> pd.DataFrame:
        """Calcule le score de resilience credit par secteur (FR19).

        Score = (1 - NPL_ratio) x (1 - CoR_norm) x (1 - Stage2_pct)
        Normalise dans [0, 1].

        Returns:
            DataFrame avec colonnes sector, resilience_credit.
        """
        df = self.result_credit
        records = []

        # Cost of risk max pour normalisation
        cors = []
        for sector in SECTORS:
            mask = df["sector"].values == sector.name
            ead = df.loc[mask, "ead"].sum()
            ecl = df.loc[mask, "ecl_weighted"].sum()
            cors.append(ecl / max(ead, 1))
        max_cor = max(cors) if cors else 1.0

        for sector in SECTORS:
            mask = df["sector"].values == sector.name
            df_sec = df.loc[mask]
            if len(df_sec) == 0:
                continue

            ead = df_sec["ead"].sum()
            ead_s3 = df_sec.loc[df_sec["stage"] == 3, "ead"].sum()
            ead_s2 = df_sec.loc[df_sec["stage"] == 2, "ead"].sum()
            ecl = df_sec["ecl_weighted"].sum()

            npl_r = ead_s3 / max(ead, 1)
            cor_norm = (ecl / max(ead, 1)) / max(max_cor, 1e-6)
            s2_pct = ead_s2 / max(ead, 1)

            score = (1 - npl_r) * (1 - cor_norm) * (1 - s2_pct)
            records.append({"sector": sector.name, "resilience_credit": round(score, 4)})

        return pd.DataFrame(records)

    def _resilience_score_pe(self) -> pd.DataFrame:
        """Calcule le score de resilience PE par secteur (FR19).

        Score = (1 - drawdown_mean) x (1 - distress_pct) x MOIC_norm
        Normalise dans [0, 1].

        Returns:
            DataFrame avec colonnes sector, resilience_pe.
        """
        df = self.result_pe
        records = []

        # MOIC max pour normalisation
        moic_max = df["moic"].max() if "moic" in df.columns else 1.0

        for sector in SECTORS:
            mask = df["sector"].values == sector.name
            df_sec = df.loc[mask]
            if len(df_sec) == 0:
                continue

            drawdown = df_sec["nav_drawdown"].mean()
            distress_pct = (df_sec["risk_category"] == "Distressed").mean()
            moic_norm = df_sec["moic"].mean() / max(moic_max, 1e-6)

            score = (1 - drawdown) * (1 - distress_pct) * moic_norm
            records.append({"sector": sector.name, "resilience_pe": round(score, 4)})

        return pd.DataFrame(records)

    # ──────────────────────────────────────────────
    # MATRICE D'ASYMETRIE (FR20)
    # ──────────────────────────────────────────────

    def build_asymmetry_matrix(self) -> pd.DataFrame:
        """Construit la matrice d'asymetrie credit vs PE par secteur (FR20).

        Colonnes :
            - ecl_sector : ECL credit du secteur
            - el_pe_sector : Expected Loss PE du secteur
            - loss_ratio : EL_PE / ECL (ratio d'asymetrie)
            - rwa_credit / rwa_pe / rwa_ratio
            - raroc_credit / raroc_pe / raroc_delta
            - resilience_credit / resilience_pe

        Returns:
            DataFrame 5 lignes (secteurs) avec metriques comparatives.
        """
        df_c = self.result_credit
        df_p = self.result_pe

        # Scores de resilience
        resil_credit = self._resilience_score_credit().set_index("sector")
        resil_pe = self._resilience_score_pe().set_index("sector")

        # RAROC par cellule (reutilise compute_raroc_eva)
        raroc_df = self.compute_raroc_eva()

        records = []
        for sector in SECTORS:
            name = sector.name

            # Credit
            mask_c = df_c["sector"].values == name
            ecl_sec = df_c.loc[mask_c, "ecl_weighted"].sum()
            ead_sec = df_c.loc[mask_c, "ead"].sum()
            rwa_c = df_c.loc[mask_c, "rwa_credit"].sum()

            # PE
            mask_p = df_p["sector"].values == name
            el_pe_sec = df_p.loc[mask_p, "expected_loss_pe"].sum()
            nav_sec = df_p.loc[mask_p, "nav"].sum()
            rwa_p = df_p.loc[mask_p, "rwa_pe"].sum()

            # Ratios
            loss_ratio = el_pe_sec / max(ecl_sec, 1)
            rwa_ratio = rwa_p / max(rwa_c, 1)

            # RAROC
            rc = raroc_df.loc[
                (raroc_df["sector"] == name) & (raroc_df["canal"] == "Credit"), "raroc"
            ]
            rp = raroc_df.loc[
                (raroc_df["sector"] == name) & (raroc_df["canal"] == "PE"), "raroc"
            ]
            raroc_c = rc.values[0] if len(rc) > 0 else 0.0
            raroc_p = rp.values[0] if len(rp) > 0 else 0.0

            records.append({
                "sector": name,
                "ecl_credit": round(ecl_sec, 0),
                "el_pe": round(el_pe_sec, 0),
                "loss_ratio": round(loss_ratio, 4),
                "ead_credit": round(ead_sec, 0),
                "nav_pe": round(nav_sec, 0),
                "rwa_credit": round(rwa_c, 0),
                "rwa_pe": round(rwa_p, 0),
                "rwa_ratio": round(rwa_ratio, 6),
                "raroc_credit": raroc_c,
                "raroc_pe": raroc_p,
                "raroc_delta": round(raroc_p - raroc_c, 4),
                "resilience_credit": resil_credit.loc[name, "resilience_credit"]
                    if name in resil_credit.index else 0.0,
                "resilience_pe": resil_pe.loc[name, "resilience_pe"]
                    if name in resil_pe.index else 0.0,
            })

        return pd.DataFrame(records)

    # ──────────────────────────────────────────────
    # OPTIMISEUR 3 NIVEAUX (FR22)
    # ──────────────────────────────────────────────

    def optimize_allocation(self) -> Dict[str, object]:
        """Optimise l'allocation credit/PE en 3 niveaux (FR22).

        Niveau 1 : rotation sectorielle credit (poids par RAROC relatif)
        Niveau 2 : rotation sectorielle PE (poids par RAROC relatif)
        Niveau 3 : reallocation inter-canal (maximize RAROC global)

        Contraintes :
            - PE allocation <= BASEL_CONFIG.pe_max_allocation
            - Poids secteurs dans [0.05, 0.40]
            - Min 3 secteurs > 5%

        Returns:
            Dict avec allocation optimale et metriques.
        """
        raroc_df = self.compute_raroc_eva()

        # ── Niveau 1 : Poids secteurs credit ──
        credit_cells = raroc_df.loc[
            (raroc_df["canal"] == "Credit") & (raroc_df["sector"] != "Total")
        ].copy()
        w_credit = self._optimize_sector_weights(credit_cells)

        # ── Niveau 2 : Poids secteurs PE ──
        pe_cells = raroc_df.loc[
            (raroc_df["canal"] == "PE") & (raroc_df["sector"] != "Total")
        ].copy()
        w_pe = self._optimize_sector_weights(pe_cells)

        # ── Niveau 3 : Split credit/PE ──
        total_credit = raroc_df.loc[
            (raroc_df["canal"] == "Credit") & (raroc_df["sector"] == "Total")
        ]
        total_pe = raroc_df.loc[
            (raroc_df["canal"] == "PE") & (raroc_df["sector"] == "Total")
        ]

        raroc_c = total_credit["raroc"].values[0] if len(total_credit) > 0 else 0.0
        raroc_p = total_pe["raroc"].values[0] if len(total_pe) > 0 else 0.0

        # Heuristique : part PE proportionnelle au RAROC relatif, plafonnee
        if raroc_c <= 0 and raroc_p <= 0:
            # Les deux negatifs : minimiser PE (plus risque)
            pe_alloc = 0.05
        elif raroc_p > raroc_c:
            pe_alloc = min(0.30, BASEL_CONFIG.pe_max_allocation)
        else:
            pe_alloc = 0.10

        credit_alloc = 1.0 - pe_alloc

        # ── Metriques post-optimisation ──
        rwa_credit_total = self.result_credit["rwa_credit"].sum()
        rwa_pe_total = self.result_pe["rwa_pe"].sum()
        rwa_weighted = credit_alloc * rwa_credit_total + pe_alloc * rwa_pe_total
        cet1_ratio = _CAPITAL_BASE / max(rwa_weighted, 1)

        return {
            "credit_allocation": round(credit_alloc, 2),
            "pe_allocation": round(pe_alloc, 2),
            "sector_weights_credit": w_credit,
            "sector_weights_pe": w_pe,
            "raroc_credit": round(raroc_c, 4),
            "raroc_pe": round(raroc_p, 4),
            "rwa_weighted": round(rwa_weighted, 0),
            "cet1_ratio": round(cet1_ratio, 4),
            "cet1_headroom": round(cet1_ratio - BASEL_CONFIG.cet1_target, 4),
        }

    def _optimize_sector_weights(
        self,
        cells: pd.DataFrame,
    ) -> Dict[str, float]:
        """Optimise les poids sectoriels par RAROC relatif.

        Heuristique : softmax sur RAROC, clip dans [0.05, 0.40],
        renormalise a somme 1.

        Args:
            cells: DataFrame avec colonnes sector, raroc.

        Returns:
            Dict {sector_name: weight}.
        """
        sectors = cells["sector"].values
        rarocs = cells["raroc"].values.astype(float)

        # M7 : Softmax avec temperature adaptative
        # scale = 1/var(raroc), clip dans [2, 50] pour stabilite numerique
        scale = np.clip(1.0 / max(np.var(rarocs), 1e-6), 2.0, 50.0)
        # Stabilite numerique : soustraire le max pour eviter overflow/underflow
        shifted = rarocs * scale - np.max(rarocs * scale)
        exp_vals = np.exp(shifted)
        raw_weights = exp_vals / exp_vals.sum()

        # Clip dans [0.05, 0.40]
        clipped = np.clip(raw_weights, 0.05, 0.40)
        clipped = clipped / clipped.sum()  # Renormaliser

        return {s: round(w, 4) for s, w in zip(sectors, clipped)}

    # ──────────────────────────────────────────────
    # SENSIBILITE CRR3 (FR23)
    # ──────────────────────────────────────────────

    def compute_crr3_sensitivity(self) -> pd.DataFrame:
        """Analyse de sensibilite sous les 3 risk weights PE CRR3 (FR23).

        Boucle sur BASEL_CONFIG.rw_pe_options : 190%, 250%, 400%.
        Calcule CET1 ratio et headroom pour chaque RW.
        Ajoute aussi le RW composite CRR3 (H8) par position.

        Returns:
            DataFrame avec colonnes rw_pe, rwa_pe, rwa_total, cet1_ratio, headroom.
        """
        rwa_credit_total = self.result_credit["rwa_credit"].sum()
        records = []

        # H8 : RWA PE via score CRR3 composite (position par position)
        crr3_rw = compute_crr3_rw(self.result_pe)
        rwa_pe_composite = float(np.sum(
            self.result_pe["nav"].values * crr3_rw / 100.0
        ))

        for rw_pe in BASEL_CONFIG.rw_pe_options:
            # Scenario fixe (stress test uniforme)
            nav_total = self.result_pe["nav"].sum()
            rwa_pe = nav_total * rw_pe / 100.0
            rwa_total = rwa_credit_total + rwa_pe
            cet1 = _CAPITAL_BASE / max(rwa_total, 1)
            headroom = cet1 - BASEL_CONFIG.cet1_target

            records.append({
                "rw_pe": rw_pe,
                "rwa_pe": round(rwa_pe, 0),
                "rwa_credit": round(rwa_credit_total, 0),
                "rwa_total": round(rwa_total, 0),
                "cet1_ratio": round(cet1, 4),
                "headroom": round(headroom, 4),
                "feasible": headroom >= 0,
            })

        # Ligne supplementaire : RW composite CRR3 (H8)
        rwa_total_composite = rwa_credit_total + rwa_pe_composite
        cet1_composite = _CAPITAL_BASE / max(rwa_total_composite, 1)
        headroom_composite = cet1_composite - BASEL_CONFIG.cet1_target
        rw_moyen = int(np.round(np.mean(crr3_rw)))

        records.append({
            "rw_pe": rw_moyen,
            "rwa_pe": round(rwa_pe_composite, 0),
            "rwa_credit": round(rwa_credit_total, 0),
            "rwa_total": round(rwa_total_composite, 0),
            "cet1_ratio": round(cet1_composite, 4),
            "headroom": round(headroom_composite, 4),
            "feasible": headroom_composite >= 0,
        })

        return pd.DataFrame(records)

    # ──────────────────────────────────────────────
    # SEUILS DE BASCULEMENT (FR24)
    # ──────────────────────────────────────────────

    def find_tipping_points(self) -> pd.DataFrame:
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
            rc = raroc_df.loc[
                (raroc_df["sector"] == name) & (raroc_df["canal"] == "Credit"), "raroc"
            ]
            rp = raroc_df.loc[
                (raroc_df["sector"] == name) & (raroc_df["canal"] == "PE"), "raroc"
            ]
            raroc_c = rc.values[0] if len(rc) > 0 else 0.0
            raroc_p = rp.values[0] if len(rp) > 0 else 0.0

            preferred = "PE" if raroc_p > raroc_c else "Credit"
            delta = abs(raroc_p - raroc_c)

            records.append({
                "sector": name,
                "raroc_credit": raroc_c,
                "raroc_pe": raroc_p,
                "preferred_canal": preferred,
                "delta_to_switch": round(delta, 4),
            })

        return pd.DataFrame(records)


if __name__ == "__main__":
    from ifrs9_cockpit.data.generator import generate_dataset
    from ifrs9_cockpit.models.pd_model import PDModelSuite
    from ifrs9_cockpit.engine.ecl_calculator import ECLCalculator
    from ifrs9_cockpit.engine.pe_calculator import PECalculator
    from ifrs9_cockpit.models.lgd_model import LGDModel
    from ifrs9_cockpit.models.ead_model import EADModel
    from ifrs9_cockpit.utils.helpers import format_pct, format_euro

    print("=" * 70)
    print("IFRS 9 COCKPIT — Comparateur : Optimisation & CRR3")
    print("=" * 70)

    # 1. Pipeline credit
    print("\n[1/5] Pipeline credit...")
    df_credit, df_pe, df_history = generate_dataset()
    print(f"       {len(df_credit):,} credits, {len(df_pe):,} PE")

    pd_suite = PDModelSuite()
    pd_suite.fit(df_credit)
    pd_current = pd_suite.predict_active(df_credit)

    lgd_model = LGDModel()
    ead_model = EADModel()
    ecl_calc = ECLCalculator(lgd_model=lgd_model, ead_model=ead_model)
    pd_origination = pd_current * 0.8  # Proxy : PD origination = 80% de PD courante
    result_credit = ecl_calc.calculate(df_credit, pd_current, pd_origination)

    # 2. Pipeline PE
    print("[2/5] Pipeline PE...")
    pe_calc = PECalculator()
    result_pe = pe_calc.calculate(df_pe)

    # 3. Metriques avancees
    print("\n[3/5] Metriques credit avancees (FR42)...")
    comparator = PortfolioComparator(result_credit, result_pe)
    adv_metrics = comparator.compute_advanced_credit_metrics()
    print(adv_metrics.to_string(index=False))

    # 4. HHI cross-cell
    print("\n[4/5] HHI cross-cell (FR41)...")
    hhi_result = comparator.compute_hhi_crosscell()
    print(f"  HHI Credit     : {hhi_result['hhi_credit']:.2f}")
    print(f"  HHI PE         : {hhi_result['hhi_pe']:.2f}")
    print(f"  HHI Cross-cell : {hhi_result['hhi_crosscell']:.2f}")
    print("\n  --- Parts par cellule ---")
    print(hhi_result["shares"].to_string(index=False))

    # 5. RAROC / EVA
    print("\n[5/5] RAROC / EVA par cellule (FR44)...")
    raroc_eva = comparator.compute_raroc_eva()
    print(raroc_eva.to_string(index=False))

    # 6. Matrice d'asymetrie (FR20)
    print("\n[6/7] Matrice d'asymetrie credit vs PE (FR20)...")
    asymmetry = comparator.build_asymmetry_matrix()
    cols_display = ["sector", "ecl_credit", "el_pe", "loss_ratio",
                    "raroc_credit", "raroc_pe", "raroc_delta",
                    "resilience_credit", "resilience_pe"]
    print(asymmetry[cols_display].to_string(index=False))

    # 7. Scores de resilience (FR19)
    print("\n[7/7] Scores de resilience (FR19)...")
    resil_c = comparator._resilience_score_credit()
    resil_p = comparator._resilience_score_pe()
    print("  Credit :")
    for _, r in resil_c.iterrows():
        print(f"    {r['sector']:12s} : {r['resilience_credit']:.4f}")
    print("  PE :")
    for _, r in resil_p.iterrows():
        print(f"    {r['sector']:12s} : {r['resilience_pe']:.4f}")

    # 8. Optimisation 3 niveaux (FR22)
    print("\n[8/10] Optimisation allocation (FR22)...")
    optim = comparator.optimize_allocation()
    print(f"  Credit : {optim['credit_allocation']:.0%} | PE : {optim['pe_allocation']:.0%}")
    print(f"  RAROC Credit = {optim['raroc_credit']:.4f} | RAROC PE = {optim['raroc_pe']:.4f}")
    print(f"  Poids secteurs credit : {optim['sector_weights_credit']}")
    print(f"  Poids secteurs PE     : {optim['sector_weights_pe']}")
    print(f"  CET1 ratio post-optim = {optim['cet1_ratio']:.4f} "
          f"(headroom = {optim['cet1_headroom']:+.4f})")

    # 9. Sensibilite CRR3 (FR23)
    print("\n[9/10] Sensibilite CRR3 (FR23)...")
    crr3 = comparator.compute_crr3_sensitivity()
    print(crr3.to_string(index=False))

    # 10. Seuils de basculement (FR24)
    print("\n[10/10] Seuils de basculement (FR24)...")
    tipping = comparator.find_tipping_points()
    print(tipping.to_string(index=False))

    # ── Validations ──
    print("\n--- Validations ---")
    all_ok = True

    # V1: rwa_credit present
    ok = "rwa_credit" in result_credit.columns
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] rwa_credit present dans result_credit")
    all_ok &= ok

    # V2: REQUIRED_CREDIT_RESULT_COLS
    missing = REQUIRED_CREDIT_RESULT_COLS - set(result_credit.columns)
    ok = len(missing) == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] REQUIRED_CREDIT_RESULT_COLS (manquantes: {missing if missing else 'aucune'})")
    all_ok &= ok

    # V3: HHI dans [0, 10000]
    ok = 0 <= hhi_result["hhi_crosscell"] <= 10_000
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] HHI cross-cell dans [0, 10000] ({hhi_result['hhi_crosscell']:.2f})")
    all_ok &= ok

    # V4: NPL ratio dans [0, 1]
    npl = adv_metrics.loc[adv_metrics["sector"] == "Total", "npl_ratio"].values[0]
    ok = 0 <= npl <= 1
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] NPL ratio dans [0, 1] ({npl:.4f})")
    all_ok &= ok

    # V5: RAROC calcule par cellule (10 cellules + 2 totaux)
    ok = len(raroc_eva) == 12
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] RAROC/EVA : 12 lignes (10 cellules + 2 totaux) ({len(raroc_eva)})")
    all_ok &= ok

    # V6: Cost of risk > 0
    cor = adv_metrics.loc[adv_metrics["sector"] == "Total", "cost_of_risk_bps"].values[0]
    ok = cor > 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Cost of risk > 0 ({cor:.1f} bps)")
    all_ok &= ok

    # V7: Matrice d'asymetrie 5 secteurs
    ok = len(asymmetry) == 5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Matrice asymetrie : 5 secteurs ({len(asymmetry)})")
    all_ok &= ok

    # V8: Resilience scores dans [0, 1]
    resil_vals = list(resil_c["resilience_credit"].values) + list(resil_p["resilience_pe"].values)
    ok = all(0 <= v <= 1 for v in resil_vals)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Resilience scores dans [0, 1] "
          f"(min={min(resil_vals):.4f}, max={max(resil_vals):.4f})")
    all_ok &= ok

    # V9: loss_ratio >= 0
    ok = (asymmetry["loss_ratio"] >= 0).all()
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] loss_ratio >= 0 (min={asymmetry['loss_ratio'].min():.4f})")
    all_ok &= ok

    # V10: PE allocation <= max
    ok = optim["pe_allocation"] <= BASEL_CONFIG.pe_max_allocation
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] PE allocation ({optim['pe_allocation']:.0%}) "
          f"<= {BASEL_CONFIG.pe_max_allocation:.0%}")
    all_ok &= ok

    # V11: Somme poids credit = 1
    sum_wc = sum(optim["sector_weights_credit"].values())
    ok = abs(sum_wc - 1.0) < 0.01
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Somme poids credit = {sum_wc:.4f}")
    all_ok &= ok

    # V12: Somme poids PE = 1
    sum_wp = sum(optim["sector_weights_pe"].values())
    ok = abs(sum_wp - 1.0) < 0.01
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Somme poids PE = {sum_wp:.4f}")
    all_ok &= ok

    # V13: CRR3 sensitivity 3 RW fixes + 1 composite CRR3 = 4 lignes
    ok = len(crr3) == 4
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] CRR3 sensitivity : 3 RW fixes + 1 composite ({len(crr3)})")
    all_ok &= ok

    # V14: Tipping points 5 secteurs
    ok = len(tipping) == 5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Tipping points : 5 secteurs ({len(tipping)})")
    all_ok &= ok

    # INFO: HHI credit
    hhi_c = hhi_result["hhi_credit"]
    breach = " (BREACH)" if hhi_c > BASEL_CONFIG.hhi_max else ""
    print(f"  [INFO] HHI credit = {hhi_c:.2f} (seuil optimiseur = {BASEL_CONFIG.hhi_max}){breach}")

    print(f"\n{'=' * 70}")
    if all_ok:
        print("Comparateur (Optimisation & CRR3) valide.")
    else:
        print("ATTENTION : certaines validations ont echoue.")
    print(f"{'=' * 70}")
