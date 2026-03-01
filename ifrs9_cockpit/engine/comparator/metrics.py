"""Mixin metriques credit avancees, concentration, RAROC, resilience, asymetrie."""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Dict, Optional, Tuple

from ifrs9_cockpit.config import (
    ASSET_CLASSES,
    ASSET_CLASS_MAP,
    BASEL_CONFIG,
    IFRS9_CONFIG,
    LGD_CONFIG,
    SECTORS,
    SCENARIOS,
    SCENARIO_BASE,
    RISK_APPETITE_CONFIG,
    SECTOR_NAMES,
)
from ifrs9_cockpit.engine.balance_sheet_ecl import effective_rw


class MetricsMixin:
    """Mixin fournissant les metriques credit, HHI, RAROC, resilience et asymetrie."""

    # ──────────────────────────────────────────────
    # METRIQUES CREDIT AVANCEES (FR42)
    # ──────────────────────────────────────────────

    def compute_advanced_credit_metrics(self) -> pl.DataFrame:
        """Calcule les metriques credit avancees par secteur (FR42).

        Metriques :
            - npl_ratio : EAD Stage 3 / EAD total
            - cost_of_risk_bps : ECL / (EAD x T_moyen) x 10 000 (M8 annualise)
            - coverage_ratio : ECL Stage 3 / EAD Stage 3 (M10 unifie)
            - texas_ratio_synth : EAD Stage 3 / (ECL Stage 3 + capital_share) (C3, synth.)
            - pd_mean : PD 12m moyenne
            - stage2_pct : Part Stage 2 en EAD
            - rwa_density : RWA / EAD

        Returns:
            DataFrame avec metriques par secteur + ligne Total.
        """
        df = self.result_credit
        coc = BASEL_CONFIG.cet1_target
        records = []

        # Pre-compute portfolio-level totals for capital share calculation
        portfolio_rwa_total = df["rwa_credit"].sum()
        portfolio_ead_total = df["ead"].sum()

        for sector in SECTORS:
            df_sec = df.filter(pl.col("sector") == sector.name)

            if len(df_sec) == 0:
                continue

            ead_total = df_sec["ead"].sum()
            ecl_total = df_sec["ecl_weighted"].sum()
            rwa_total = df_sec["rwa_credit"].sum()

            ead_s3 = df_sec.filter(pl.col("stage") == 3)["ead"].sum()
            ead_s2 = df_sec.filter(pl.col("stage") == 2)["ead"].sum()
            ecl_s3 = df_sec.filter(pl.col("stage") == 3)["ecl_weighted"].sum()

            # Part du capital proportionnelle au secteur (base = RWA reel)
            portfolio_capital = portfolio_rwa_total * coc
            capital_share = portfolio_capital * (ead_total / max(portfolio_ead_total, 1))

            # M8 : Cost of Risk annualise — horizon EAD-pondere par stage
            # Stage 1 = 1 an, Stage 2/3 = lifetime_horizon_years
            ead_s1 = df_sec.filter(pl.col("stage") == 1)["ead"].sum()
            t_moyen = (
                (ead_s1 * 1.0 + (ead_total - ead_s1) * IFRS9_CONFIG.lifetime_horizon_years)
                / max(ead_total, 1)
            )

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
                # C3 : Texas Synthetique = EAD_S3 / (ECL_S3 + capital reglementaire)
                # Denominateur non-standard : capital alloue (RWA x CET1), pas fonds propres tangibles.
                "texas_ratio_synth": round(ead_s3 / max(ecl_s3 + capital_share, 1), 4),
                "pd_mean": round(df_sec["pd_12m"].mean(), 4),
                "stage2_pct": round(ead_s2 / max(ead_total, 1), 4),
                "rwa_density": round(rwa_total / max(ead_total, 1), 4),
            })

        result = pl.DataFrame(records)

        # Ligne Total
        ead_all = df["ead"].sum()
        ecl_all = df["ecl_weighted"].sum()
        rwa_all = df["rwa_credit"].sum()
        ead_s3_all = df.filter(pl.col("stage") == 3)["ead"].sum()
        ead_s2_all = df.filter(pl.col("stage") == 2)["ead"].sum()
        ead_s1_all = df.filter(pl.col("stage") == 1)["ead"].sum()
        ecl_s3_all = df.filter(pl.col("stage") == 3)["ecl_weighted"].sum()
        t_moyen_all = (
            (ead_s1_all * 1.0 + (ead_all - ead_s1_all) * IFRS9_CONFIG.lifetime_horizon_years)
            / max(ead_all, 1)
        )

        total_row = pl.DataFrame([{
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
            # C3 : Texas Synthetique = EAD_S3 / (ECL_S3 + capital reglementaire)
            "texas_ratio_synth": round(ead_s3_all / max(ecl_s3_all + rwa_all * coc, 1), 4),
            "pd_mean": round(df["pd_12m"].mean(), 4),
            "stage2_pct": round(ead_s2_all / max(ead_all, 1), 4),
            "rwa_density": round(rwa_all / max(ead_all, 1), 4),
        }])

        return pl.concat([result, total_row])

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
            ead_sec = df_c.filter(pl.col("sector") == sector.name)["ead"].sum()
            share_c = ead_sec / max(exposure_total, 1)
            shares_data.append({
                "sector": sector.name,
                "canal": "Credit",
                "exposure": ead_sec,
                "share": share_c,
            })

            # PE
            nav_sec = df_p.filter(pl.col("sector") == sector.name)["nav"].sum()
            share_p = nav_sec / max(exposure_total, 1)
            shares_data.append({
                "sector": sector.name,
                "canal": "PE",
                "exposure": nav_sec,
                "share": share_p,
            })

        shares_df = pl.DataFrame(shares_data)

        # M9 : HHI full precision — arrondir uniquement a l'affichage
        # HHI credit (5 cellules)
        credit_shares = shares_df.filter(pl.col("canal") == "Credit")["exposure"].to_numpy()
        credit_total = credit_shares.sum()
        hhi_credit = float(np.sum((credit_shares / max(credit_total, 1)) ** 2) * 10_000)

        # HHI PE (5 cellules)
        pe_shares = shares_df.filter(pl.col("canal") == "PE")["exposure"].to_numpy()
        pe_total = pe_shares.sum()
        hhi_pe = float(np.sum((pe_shares / max(pe_total, 1)) ** 2) * 10_000)

        # HHI cross-cell (10 cellules)
        all_shares = shares_df["share"].to_numpy()
        hhi_crosscell = float(np.sum(all_shares ** 2) * 10_000)

        return {
            "hhi_credit": hhi_credit,
            "hhi_pe": hhi_pe,
            "hhi_crosscell": hhi_crosscell,
            "hhi_name_credit": self._compute_hhi_name_level("credit"),
            "hhi_name_pe": self._compute_hhi_name_level("pe"),
            "shares": shares_df,
        }

    def _compute_hhi_name_level(self, canal: str) -> float:
        """Calcule le HHI par contrepartie (Name Concentration, ICAAP Pilier 2).

        HHI_name = sum((EAD_i / EAD_total)^2) * 10 000
        Un HHI_name > 50 (echelle 10k) signale une concentration idiosyncratique.

        Args:
            canal: "credit" ou "pe".

        Returns:
            HHI name level en echelle 10 000.
        """
        if canal == "credit":
            df = self.result_credit
            exposure_col = "ead"
        else:
            df = self.result_pe
            exposure_col = "nav"

        exposures = (
            df.group_by("enterprise_id")
            .agg(pl.col(exposure_col).sum().alias(exposure_col))
        )[exposure_col].to_numpy()
        total = exposures.sum()
        if total <= 0:
            return 0.0
        shares = exposures / total
        return float(np.sum(shares ** 2) * 10_000)

    # ──────────────────────────────────────────────
    # GREEN ASSET RATIO (ESG placeholder)
    # ──────────────────────────────────────────────

    def compute_green_asset_ratio(self) -> Dict[str, object]:
        """Calcule le Green Asset Ratio declaratif par secteur et aggregate.

        GAR = sum(EAD_i * green_share_i) / sum(EAD_i) pour le credit.
        Pour le PE : sum(NAV_i * green_share_i) / sum(NAV_i).

        Les green_share par secteur sont declaratifs (SectorConfig.green_share),
        non audites. Placeholder pour conformite reglementaire BCE 2024.

        Returns:
            Dict avec gar_credit, gar_pe, gar_total, details DataFrame.
        """
        sector_map = {s.name: s.green_share for s in SECTORS}

        # Credit
        df_c = self.result_credit.clone()
        # Map sector to green_share
        green_shares_c = [sector_map.get(s, 0.0) for s in df_c["sector"].to_list()]
        df_c = df_c.with_columns(pl.Series("green_share", green_shares_c))
        ead_total = df_c["ead"].sum()
        gar_credit = float((df_c["ead"].to_numpy() * df_c["green_share"].to_numpy()).sum() / max(ead_total, 1))

        # PE
        df_p = self.result_pe.clone()
        green_shares_p = [sector_map.get(s, 0.0) for s in df_p["sector"].to_list()]
        df_p = df_p.with_columns(pl.Series("green_share", green_shares_p))
        nav_total = df_p["nav"].sum()
        gar_pe = float((df_p["nav"].to_numpy() * df_p["green_share"].to_numpy()).sum() / max(nav_total, 1))

        # Total pondere
        total_exposure = ead_total + nav_total
        gar_total = (gar_credit * ead_total + gar_pe * nav_total) / max(total_exposure, 1)

        # Detail par secteur
        details = []
        for s in SECTORS:
            details.append({
                "sector": s.name,
                "green_share": s.green_share,
                "ead_sector": df_c.filter(pl.col("sector") == s.name)["ead"].sum(),
                "nav_sector": df_p.filter(pl.col("sector") == s.name)["nav"].sum(),
            })

        return {
            "gar_credit": round(gar_credit, 4),
            "gar_pe": round(gar_pe, 4),
            "gar_total": round(gar_total, 4),
            "details": pl.DataFrame(details),
        }

    # ──────────────────────────────────────────────
    # RAROC / EVA PAR CELLULE (FR44)
    # ──────────────────────────────────────────────

    def compute_raroc_eva(self) -> pl.DataFrame:
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
            df_sec_c = df_c.filter(pl.col("sector") == sector.name)

            if len(df_sec_c) > 0:
                ead = df_sec_c["ead"].sum()
                rwa = df_sec_c["rwa_credit"].sum()

                # H5 : RAROC complet avec CIR et impots
                # NII contractuel : spread fixe a l'origination (ne bouge pas avec le stress).
                # La banque a facture un spread base sur le risque initial du secteur.
                # En stress, seul l'EL (PD courante x LGD) change -> RAROC bouge.
                pd_arr = df_sec_c["pd_12m"].to_numpy().astype(float)
                lgd_arr = df_sec_c["lgd"].to_numpy().astype(float)
                ead_arr = df_sec_c["ead"].to_numpy().astype(float)
                liq_premium = BASEL_CONFIG.liquidity_premium_bps / 10_000
                comm_margin = BASEL_CONFIG.commercial_margin_bps / 10_000
                # Spread Merton a l'origination (PD_base x LGD_TTC du secteur)
                pd_base = sector.base_default_rate
                lgd_ttc = LGD_CONFIG.lgd_ttc_mean
                cs_origination = -np.log(max(1 - pd_base * lgd_ttc, 1e-10)) + liq_premium + comm_margin
                cs_origination = np.clip(cs_origination, 0.0050, 0.2000)
                nii = float(ead * cs_origination)
                # Perte annuelle attendue a la PD courante (stressee)
                # C'est ici que le stress impacte : PD -> EL -> profit -> RAROC
                annual_el = float(np.sum(pd_arr * lgd_arr * ead_arr))
                revenue_net = nii * (1 - BASEL_CONFIG.cir)
                profit_net = (revenue_net - annual_el) * (1 - BASEL_CONFIG.tax_rate)
                capital_c = rwa * coc
                raroc_c = profit_net / max(capital_c, 1)
                eva_c = (raroc_c - coc) * capital_c

                records.append({
                    "sector": sector.name,
                    "canal": "Credit",
                    "exposure": round(ead, 0),
                    "revenue": round(nii, 0),
                    "loss": round(annual_el, 0),
                    "rwa": round(rwa, 0),
                    "capital": round(capital_c, 0),
                    "raroc": round(raroc_c, 4),
                    "eva": round(eva_c, 0),
                })

            # ── PE ──
            df_sec_p = df_p.filter(pl.col("sector") == sector.name)

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

        result = pl.DataFrame(records)

        # Totaux par canal
        for canal in ["Credit", "PE"]:
            sub = result.filter(pl.col("canal") == canal)
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

            result = pl.concat([result, pl.DataFrame([{
                "sector": "Total",
                "canal": canal,
                "exposure": round(exp, 0),
                "revenue": round(rev, 0),
                "loss": round(loss, 0),
                "rwa": round(rwa, 0),
                "capital": round(cap, 0),
                "raroc": round(raroc, 4),
                "eva": round(eva, 0),
            }])])

        self._raroc_eva_cache = result
        return result

    # ──────────────────────────────────────────────
    # SCORE DE RESILIENCE (FR19)
    # ──────────────────────────────────────────────

    def _resilience_score_credit(self) -> pl.DataFrame:
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
            df_sec = df.filter(pl.col("sector") == sector.name)
            ead = df_sec["ead"].sum()
            ecl = df_sec["ecl_weighted"].sum()
            cors.append(ecl / max(ead, 1))
        max_cor = max(cors) if cors else 1.0

        for sector in SECTORS:
            df_sec = df.filter(pl.col("sector") == sector.name)
            if len(df_sec) == 0:
                continue

            ead = df_sec["ead"].sum()
            ead_s3 = df_sec.filter(pl.col("stage") == 3)["ead"].sum()
            ead_s2 = df_sec.filter(pl.col("stage") == 2)["ead"].sum()
            ecl = df_sec["ecl_weighted"].sum()

            npl_r = ead_s3 / max(ead, 1)
            cor_norm = (ecl / max(ead, 1)) / max(max_cor, 1e-6)
            s2_pct = ead_s2 / max(ead, 1)

            score = (1 - npl_r) * (1 - cor_norm) * (1 - s2_pct)
            records.append({"sector": sector.name, "resilience_credit": round(score, 4)})

        return pl.DataFrame(records)

    def _resilience_score_pe(self) -> pl.DataFrame:
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
            df_sec = df.filter(pl.col("sector") == sector.name)
            if len(df_sec) == 0:
                continue

            drawdown = df_sec["nav_drawdown"].mean()
            distress_count = df_sec.filter(pl.col("risk_category") == "Distressed").height
            distress_pct = distress_count / max(len(df_sec), 1)
            moic_norm = df_sec["moic"].mean() / max(moic_max, 1e-6)

            score = (1 - drawdown) * (1 - distress_pct) * moic_norm
            records.append({"sector": sector.name, "resilience_pe": round(score, 4)})

        return pl.DataFrame(records)

    # ──────────────────────────────────────────────
    # MATRICE D'ASYMETRIE (FR20)
    # ──────────────────────────────────────────────

    # ──────────────────────────────────────────────
    # RAROC MULTI-ACTIF (10 classes)
    # ──────────────────────────────────────────────

    def compute_raroc_multiclass(self) -> pl.DataFrame:
        """Calcule RAROC et EVA pour les 10 classes d'actifs.

        Level 1 (corporate_loans, private_equity) : agrege depuis les donnees
        position par position (result_credit, result_pe).
        Level 2/3 (8 classes) : parametrique depuis df_balance_sheet_ecl.

        Formule RAROC identique a compute_raroc_eva() :
            Revenue = EAD x spread (Merton + liquidity + commercial margin)
            Profit = (Revenue x (1 - CIR) - Loss) x (1 - tax)
            RAROC = Profit / Capital   (Capital = RWA x CET1_target)

        Returns:
            DataFrame avec 10 lignes + 1 Total, colonnes :
            asset_class, label, exposure, revenue, loss, rwa, capital, raroc, eva.
        """
        if self._raroc_multiclass_cache is not None:
            return self._raroc_multiclass_cache

        df_c = self.result_credit
        df_p = self.result_pe
        coc = BASEL_CONFIG.cet1_target
        liq_premium = BASEL_CONFIG.liquidity_premium_bps / 10_000
        comm_margin = BASEL_CONFIG.commercial_margin_bps / 10_000
        # Basel III leverage ratio floor: even RW=0 classes (sovereign)
        # must consume economic capital. Prevents infinite RAROC.
        lev_floor = BASEL_CONFIG.leverage_max  # 3.3%

        # ── Funding cost & inflation adjustments (EBA IRRBB / SREP) ──
        # When rates rise, fixed-rate assets don't reprice but funding does.
        # Combined sensitivity = duration/D_REF × net_pass_through:
        #   - (1 - alm_hedge_ratio) ≈ 0.70 : fraction non-hedgee (EBA 2024)
        #   - deposit_beta ≈ 0.45 : repricing des depots (ECB IMIR 2023)
        #   - Net: 0.70 × 0.45 ≈ 0.30
        # Calibration: EBA 2023 adverse → NII impact ~-12% a -15%.
        # Inflation: 60% des OPEX indexes (staff 50% + immo 10%).
        _D_REF = 10.0
        _NET_PASS_THROUGH = 0.30
        if getattr(self, '_macro_params', None) is not None:
            _delta_ir = (self._macro_params.get("interest_rate",
                         SCENARIO_BASE.interest_rate)
                         - SCENARIO_BASE.interest_rate) / 100.0
            _delta_infl = max(0.0,
                         (self._macro_params.get("inflation_rate",
                          SCENARIO_BASE.inflation_rate)
                          - SCENARIO_BASE.inflation_rate) / 100.0)
        else:
            _delta_ir = 0.0
            _delta_infl = 0.0

        # Class-specific commercial margins (bps -> decimal).
        # Market instruments (sovereign, covered bonds, interbank) carry
        # minimal mark-up vs. originated loans (corporate, consumer).
        # Sources: ECB IMIR/MIR, EBA FINREP, Bloomberg indices.
        _CLASS_MARGIN_BPS = {
            "sovereign": 10,           # near risk-free, minimal margin
            "covered_bonds": 40,       # market instrument, double-recours
            "interbank": 15,           # O/N-3M, operational margin only
            "trade_finance": 90,       # short-tenor, LC fees
            "retail_mortgage": 130,    # originated, moderate NIM
            "structured_products": 150,  # market instrument, complexity premium
            "project_finance": 220,    # originated, arrangement fees
            "consumer_credit": 350,    # originated, high NIM (unsecured)
            "equities": 0,             # FVTPL: equity risk premium, not spread
            "corporate_bonds": 80,     # market, spread + margin
            "repos_sft": 5,            # repo rate spread, minimal
            "derivatives_cva": 30,     # net carry after CVA charge
        }

        records = []

        for ac in ASSET_CLASSES:
            if ac.name == "corporate_loans":
                # ── Aggregate from position-level credit data ──
                ead = df_c["ead"].sum()
                rwa = df_c["rwa_credit"].sum()
                ecl = df_c["ecl_weighted"].sum()
                # Spread at origination (mean across sectors)
                pd_base_avg = df_c["pd_12m"].mean()
                lgd_ttc = LGD_CONFIG.lgd_ttc_mean
                cs = -np.log(max(1 - pd_base_avg * lgd_ttc, 1e-10)) + liq_premium + comm_margin
                cs = np.clip(cs, 0.0050, 0.2000)
                nii = float(ead * cs)
                # Funding cost: duration-based ALM mismatch
                _fb = min(ac.duration / _D_REF, 1.0) * _NET_PASS_THROUGH
                nii = nii - float(ead * _fb * _delta_ir)
                # Annual EL (stressed PD x LGD x EAD)
                annual_el = float(np.sum(
                    df_c["pd_12m"].to_numpy().astype(float)
                    * df_c["lgd"].to_numpy().astype(float)
                    * df_c["ead"].to_numpy().astype(float)
                ))
                cir_eff = getattr(ac, 'cir_class', BASEL_CONFIG.cir)
                cir_eff = cir_eff * (1.0 + 0.60 * _delta_infl)
                revenue_net = nii * (1 - cir_eff)
                profit = (revenue_net - annual_el) * (1 - BASEL_CONFIG.tax_rate)
                # Capital = max(regulatory, leverage ratio floor)
                # Basel III leverage ratio backstop: even RW=0 assets consume capital
                capital = max(rwa * coc, ead * lev_floor)
                raroc = profit / max(capital, 1)
                eva = (raroc - coc) * capital

                records.append({
                    "asset_class": ac.name,
                    "label": ac.label,
                    "exposure": round(ead, 0),
                    "revenue": round(nii, 0),
                    "loss": round(annual_el, 0),
                    "rwa": round(rwa, 0),
                    "capital": round(capital, 0),
                    "raroc": round(raroc, 4),
                    "eva": round(eva, 0),
                    "profit_rate": round(profit / max(ead, 1), 6),
                })

            elif ac.name == "private_equity":
                # ── Aggregate from position-level PE data ──
                nav = df_p["nav"].sum()
                el_pe = df_p["expected_loss_pe"].sum()
                rwa_pe = df_p["rwa_pe"].sum()
                irr_mean = df_p["irr"].mean()
                rev_pe = nav * irr_mean
                cir_eff = getattr(ac, 'cir_class', BASEL_CONFIG.cir)
                revenue_net = rev_pe * (1 - cir_eff)
                profit = (revenue_net - el_pe) * (1 - BASEL_CONFIG.tax_rate)
                capital = max(rwa_pe * coc, nav * lev_floor)
                raroc = profit / max(capital, 1)
                eva = (raroc - coc) * capital

                records.append({
                    "asset_class": ac.name,
                    "label": ac.label,
                    "exposure": round(nav, 0),
                    "revenue": round(rev_pe, 0),
                    "loss": round(el_pe, 0),
                    "rwa": round(rwa_pe, 0),
                    "capital": round(capital, 0),
                    "raroc": round(raroc, 4),
                    "eva": round(eva, 0),
                    "profit_rate": round(profit / max(nav, 1), 6),
                })

            else:
                # ── Level 2/3 parametric from balance_sheet_ecl ──
                # Dispatch by accounting treatment (FVTPL, FVOCI, amortised_cost)

                # --- FVTPL branch (equities): no ECL, MTM P&L ---
                if getattr(ac, "accounting_treatment", "amortised_cost") == "fvtpl":
                    # Revenue = EAD x equity_risk_premium (Damodaran 2024 EU: ~5.5%)
                    _EQUITY_RISK_PREMIUM = 0.055
                    if self.df_balance_sheet_ecl is not None:
                        row_df = self.df_balance_sheet_ecl.filter(
                            pl.col("asset_class") == ac.name
                        )
                        if len(row_df) > 0:
                            row = row_df.row(0, named=True)
                            ead = float(row["ead_total"])
                            rwa = float(row.get("rwa", ead * effective_rw(ac)))
                            annual_el = float(row.get("mtm_loss", ead * 0.10))
                        else:
                            ead = BASEL_CONFIG.rwa_budget * ac.typical_weight
                            rwa = ead * effective_rw(ac)
                            annual_el = ead * 0.10  # fallback 10% haircut
                    else:
                        ead = BASEL_CONFIG.rwa_budget * ac.typical_weight
                        rwa = ead * effective_rw(ac)
                        annual_el = ead * 0.10

                    nii = float(ead * _EQUITY_RISK_PREMIUM)
                    cir_eff = getattr(ac, 'cir_class', BASEL_CONFIG.cir)
                    revenue_net = nii * (1 - cir_eff)
                    profit = (revenue_net - annual_el) * (1 - BASEL_CONFIG.tax_rate)
                    capital = max(rwa * coc, ead * lev_floor)
                    raroc = profit / max(capital, 1)
                    eva = (raroc - coc) * capital

                    records.append({
                        "asset_class": ac.name,
                        "label": ac.label,
                        "exposure": round(ead, 0),
                        "revenue": round(nii, 0),
                        "loss": round(annual_el, 0),
                        "rwa": round(rwa, 0),
                        "capital": round(capital, 0),
                        "raroc": round(raroc, 4),
                        "eva": round(eva, 0),
                        "profit_rate": round(profit / max(ead, 1), 6),
                    })
                    continue

                # --- Standard ECL-based classes (amortised_cost, fvoci) ---
                # RAROC uses Base-scenario annual EL (12-month P&L flow),
                # NOT the IFRS 9 lifetime ECL provision (balance-sheet stock).
                pd_for_spread = ac.pd_base
                lgd_for_spread = ac.lgd_base
                if self.df_balance_sheet_ecl is None:
                    # Fallback: synthetic from AssetClassProfile
                    ead = BASEL_CONFIG.rwa_budget * ac.typical_weight
                    annual_el = ead * ac.pd_base * ac.lgd_base
                    rwa = ead * effective_rw(ac)
                else:
                    row_df = self.df_balance_sheet_ecl.filter(
                        pl.col("asset_class") == ac.name
                    )
                    if len(row_df) == 0:
                        ead = BASEL_CONFIG.rwa_budget * ac.typical_weight
                        annual_el = ead * ac.pd_base * ac.lgd_base
                        rwa = ead * effective_rw(ac)
                    else:
                        row = row_df.row(0, named=True)
                        ead = float(row["ead_total"])
                        rwa = float(row.get("rwa", ead * effective_rw(ac)))
                        # Loss: PD conditionnelle stressee (Vasicek under current macro)
                        # pd_cond_base captures scenario stress; falls back to pd_base
                        pd_cond = float(row.get("pd_cond_base", row.get("pd_base", ac.pd_base)))
                        lgd_row = float(row.get("lgd_base", ac.lgd_base))
                        lgd_row = max(lgd_row, ac.input_floor_lgd)
                        annual_el = pd_cond * lgd_row * ead
                        # Revenue: spread contractuel (PD a l'origination, statique)
                        pd_for_spread = float(row.get("pd_base", ac.pd_base))
                        lgd_for_spread = lgd_row

                # Merton spread from portfolio-level PD/LGD + class-specific margin
                pd_eff = max(pd_for_spread, ac.input_floor_pd)
                lgd_eff = max(lgd_for_spread, ac.input_floor_lgd)
                class_margin = _CLASS_MARGIN_BPS.get(ac.name, comm_margin * 10_000) / 10_000
                cs = -np.log(max(1 - pd_eff * lgd_eff, 1e-10)) + liq_premium + class_margin
                cs = np.clip(cs, 0.0020, 0.2000)
                nii = float(ead * cs)
                # Funding cost: duration-based ALM mismatch
                _fb = min(ac.duration / _D_REF, 1.0) * _NET_PASS_THROUGH
                nii = nii - float(ead * _fb * _delta_ir)

                cir_eff = getattr(ac, 'cir_class', BASEL_CONFIG.cir)
                cir_eff = cir_eff * (1.0 + 0.60 * _delta_infl)
                revenue_net = nii * (1 - cir_eff)
                profit = (revenue_net - annual_el) * (1 - BASEL_CONFIG.tax_rate)
                capital = max(rwa * coc, ead * lev_floor)
                raroc = profit / max(capital, 1)
                eva = (raroc - coc) * capital

                records.append({
                    "asset_class": ac.name,
                    "label": ac.label,
                    "exposure": round(ead, 0),
                    "revenue": round(nii, 0),
                    "loss": round(annual_el, 0),
                    "rwa": round(rwa, 0),
                    "capital": round(capital, 0),
                    "raroc": round(raroc, 4),
                    "eva": round(eva, 0),
                    "profit_rate": round(profit / max(ead, 1), 6),
                })

        result = pl.DataFrame(records)

        # Total row
        exp_all = result["exposure"].sum()
        rev_all = result["revenue"].sum()
        loss_all = result["loss"].sum()
        rwa_all = result["rwa"].sum()
        cap_all = max(result["capital"].sum(), exp_all * lev_floor)
        _cir_total = BASEL_CONFIG.cir * (1.0 + 0.60 * _delta_infl)
        rev_net_all = rev_all * (1 - _cir_total)
        profit_all = (rev_net_all - loss_all) * (1 - BASEL_CONFIG.tax_rate)
        raroc_all = profit_all / max(cap_all, 1)
        eva_all = (raroc_all - coc) * cap_all

        result = pl.concat([result, pl.DataFrame([{
            "asset_class": "Total",
            "label": "Total",
            "exposure": round(exp_all, 0),
            "revenue": round(rev_all, 0),
            "loss": round(loss_all, 0),
            "rwa": round(rwa_all, 0),
            "capital": round(cap_all, 0),
            "raroc": round(raroc_all, 4),
            "eva": round(eva_all, 0),
            "profit_rate": round(profit_all / max(exp_all, 1), 6),
        }])])

        self._raroc_multiclass_cache = result
        return result

    def build_asymmetry_matrix(self) -> pl.DataFrame:
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
        resil_credit_df = self._resilience_score_credit()
        resil_pe_df = self._resilience_score_pe()

        # Build lookup dicts for resilience scores
        resil_credit_map = {}
        for row in resil_credit_df.iter_rows(named=True):
            resil_credit_map[row["sector"]] = row["resilience_credit"]
        resil_pe_map = {}
        for row in resil_pe_df.iter_rows(named=True):
            resil_pe_map[row["sector"]] = row["resilience_pe"]

        # RAROC par cellule (reutilise compute_raroc_eva)
        raroc_df = self.compute_raroc_eva()

        records = []
        for sector in SECTORS:
            name = sector.name

            # Credit
            df_sec_c = df_c.filter(pl.col("sector") == name)
            ecl_sec = df_sec_c["ecl_weighted"].sum()
            ead_sec = df_sec_c["ead"].sum()
            rwa_c = df_sec_c["rwa_credit"].sum()

            # PE
            df_sec_p = df_p.filter(pl.col("sector") == name)
            el_pe_sec = df_sec_p["expected_loss_pe"].sum()
            nav_sec = df_sec_p["nav"].sum()
            rwa_p = df_sec_p["rwa_pe"].sum()

            # Ratios
            loss_ratio = el_pe_sec / max(ecl_sec, 1)
            rwa_ratio = rwa_p / max(rwa_c, 1)

            # RAROC
            rc_df = raroc_df.filter(
                (pl.col("sector") == name) & (pl.col("canal") == "Credit")
            )
            rp_df = raroc_df.filter(
                (pl.col("sector") == name) & (pl.col("canal") == "PE")
            )
            raroc_c = rc_df["raroc"].to_numpy()[0] if len(rc_df) > 0 else 0.0
            raroc_p = rp_df["raroc"].to_numpy()[0] if len(rp_df) > 0 else 0.0

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
                "resilience_credit": resil_credit_map.get(name, 0.0),
                "resilience_pe": resil_pe_map.get(name, 0.0),
            })

        return pl.DataFrame(records)
