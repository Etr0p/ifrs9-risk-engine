"""Generateur de donnees synthetiques pour le Cockpit IFRS 9.

Produit un portefeuille dual de 10 000 entreprises reparties sur 5 secteurs
avec des features specifiques credit (IFRS 9) et PE (IFRS 13), plus un
historique panel mensuel (12 mois) incluant les 5 variables macro.

Sortie : (df_credit, df_pe, df_history) — trois DataFrames conformes aux
contrats AR4 definis dans config.py.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from ifrs9_cockpit.config import (
    MACRO_HISTORY_BASELINE,
    N_CLIENTS,
    N_MONTHS,
    RANDOM_SEED,
    REQUIRED_CREDIT_COLS,
    REQUIRED_PE_COLS,
    SCENARIO_BASE,
    SECTORS,
    SECTOR_NAMES,
    SectorConfig,
    TARGET,
)


# Moyennes et ecarts-types credit score par secteur (calibration generateur)
_CREDIT_SCORE_MEANS: Dict[str, int] = {
    "Technologie": 620, "Industrie": 660, "Sante": 700,
    "Immobilier": 650, "Services": 670,
}
_CREDIT_SCORE_STDS: Dict[str, int] = {
    "Technologie": 80, "Industrie": 60, "Sante": 50,
    "Immobilier": 65, "Services": 55,
}


class SyntheticDataGenerator:
    """Genere un portefeuille dual (credit + PE) d'entreprises synthetiques.

    Le generateur cree 10 000 profils d'entreprises repartis sur 5 secteurs
    avec des features communes (revenue, EBITDA, debt_ratio) et des features
    specifiques a chaque canal (credit et PE). Le defaut est modelise via un
    modele logistique latent utilisant les sensibilites macro par secteur.

    Attributes:
        n_clients: Nombre total d'entreprises a generer.
        n_months: Nombre de mois d'historique.
        seed: Graine pour la reproductibilite.
        rng: Generateur aleatoire numpy isole.
    """

    def __init__(
        self,
        n_clients: int = N_CLIENTS,
        n_months: int = N_MONTHS,
        seed: int = RANDOM_SEED,
    ) -> None:
        """Initialise le generateur.

        Args:
            n_clients: Nombre d'entreprises a generer.
            n_months: Nombre de mois d'historique.
            seed: Graine aleatoire pour reproductibilite.
        """
        self.n_clients = n_clients
        self.n_months = n_months
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Pipeline principal de generation.

        Returns:
            Tuple contenant :
                - df_credit : DataFrame credit (10K lignes, features credit + default_flag).
                - df_pe : DataFrame PE (10K lignes, features PE).
                - df_history : DataFrame panel (120K lignes, enterprise_id x month).
        """
        # 1. Assigner les secteurs et generer les features communes
        sectors, sector_configs = self._assign_sectors()
        common = self._generate_common_features(sectors, sector_configs)

        # 2. Generer les features specifiques credit
        df_credit = self._generate_credit_features(common, sector_configs)

        # 3. Generer les features specifiques PE
        df_pe = self._generate_pe_features(common, sector_configs)

        # 4. Calculer le default_flag (canal credit)
        df_credit = self._compute_default_flag(df_credit, sector_configs)

        # 5. Generer l'historique panel 12 mois
        df_history = self._generate_monthly_history(df_credit)

        # 6. Ajouter du bruit realiste (propager outliers revenue vers df_pe)
        df_credit, df_pe = self._add_realistic_noise(df_credit, df_pe)

        # 7. Valider les contrats AR4
        self._validate_contracts(df_credit, df_pe, df_history)

        return df_credit, df_pe, df_history

    # ──────────────────────────────────────────────
    # SECTEURS
    # ──────────────────────────────────────────────

    def _assign_sectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Assigne un secteur a chaque entreprise selon les proportions config.

        Returns:
            Tuple contenant :
                - sectors : Array de noms de secteurs (n_clients,).
                - sector_configs : Array de SectorConfig correspondants (n_clients,).
        """
        sector_names = [s.name for s in SECTORS]
        proportions = [s.proportion for s in SECTORS]
        sectors = self.rng.choice(
            sector_names,
            size=self.n_clients,
            p=proportions,
        )
        # Pre-calculer la config de chaque entreprise pour acces vectorise
        config_map = {s.name: s for s in SECTORS}
        sector_configs = np.array([config_map[name] for name in sectors])
        return sectors, sector_configs

    # ──────────────────────────────────────────────
    # FEATURES COMMUNES
    # ──────────────────────────────────────────────

    def _generate_common_features(
        self,
        sectors: np.ndarray,
        sector_configs: np.ndarray,
    ) -> pd.DataFrame:
        """Genere les features partagees entre credit et PE.

        Features communes : enterprise_id, sector, revenue, ebitda, debt_ratio.
        Ces valeurs sont identiques dans df_credit et df_pe.

        Args:
            sectors: Array des noms de secteurs assignes.
            sector_configs: Array des SectorConfig correspondants.

        Returns:
            DataFrame avec les features communes.
        """
        n = self.n_clients

        # Revenue (millions EUR) — log-uniforme dans la fourchette du secteur
        revenue = np.empty(n)
        ebitda = np.empty(n)
        debt_ratio = np.empty(n)

        for sector in SECTORS:
            mask = sectors == sector.name
            count = mask.sum()
            if count == 0:
                continue

            # Revenue : log-uniforme pour distribution realiste (plus de PME que de grandes)
            log_min = np.log(sector.revenue_range_m[0])
            log_max = np.log(sector.revenue_range_m[1])
            revenue[mask] = np.exp(self.rng.uniform(log_min, log_max, count))

            # EBITDA = revenue * marge EBITDA (uniforme dans la fourchette)
            margin = self.rng.uniform(
                sector.ebitda_margin_range[0],
                sector.ebitda_margin_range[1],
                count,
            )
            ebitda[mask] = revenue[mask] * margin

            # Debt ratio — distribution Beta (mode autour de 0.35, differenciee)
            # Immobilier plus leve, Sante moins
            if sector.name == "Immobilier":
                debt_ratio[mask] = np.clip(self.rng.beta(4.0, 3.0, count), 0.05, 0.95)
            elif sector.name == "Sante":
                debt_ratio[mask] = np.clip(self.rng.beta(2.0, 5.0, count), 0.05, 0.95)
            elif sector.name == "Technologie":
                debt_ratio[mask] = np.clip(self.rng.beta(2.5, 4.0, count), 0.05, 0.95)
            else:
                debt_ratio[mask] = np.clip(self.rng.beta(3.0, 4.0, count), 0.05, 0.95)

        return pd.DataFrame({
            "enterprise_id": np.arange(n),
            "sector": sectors,
            "revenue": np.round(revenue, 4),
            "ebitda": np.round(ebitda, 4),
            "debt_ratio": np.round(debt_ratio, 4),
        })

    # ──────────────────────────────────────────────
    # FEATURES CREDIT
    # ──────────────────────────────────────────────

    def _generate_credit_features(
        self,
        common: pd.DataFrame,
        sector_configs: np.ndarray,
    ) -> pd.DataFrame:
        """Genere les features specifiques au canal credit.

        Produit des distributions correlees : un credit_score bas implique
        un debt_ratio plus eleve, plus de DPD, et un collateral plus faible.

        Args:
            common: DataFrame avec features communes.
            sector_configs: Array des SectorConfig par entreprise.

        Returns:
            DataFrame credit complet (communes + specifiques).
        """
        n = self.n_clients
        sectors = common["sector"].values
        df = common.copy()

        # Credit score — normale tronquee, differenciee par secteur
        credit_score = np.empty(n)
        for sector_name in SECTOR_NAMES:
            mask = sectors == sector_name
            count = mask.sum()
            if count == 0:
                continue
            raw = self.rng.normal(_CREDIT_SCORE_MEANS[sector_name],
                                 _CREDIT_SCORE_STDS[sector_name], count)
            credit_score[mask] = np.clip(raw, 300, 850).astype(int)

        df["credit_score"] = credit_score.astype(int)

        # Correler debt_ratio au credit_score (ajustement post-generation)
        # Plus le score est bas, plus le debt_ratio augmente
        credit_factor = (credit_score - 650) / 200  # [-1.75, +1.0]
        df["debt_ratio"] = np.clip(
            df["debt_ratio"].values - credit_factor * 0.08,
            0.05, 0.95,
        ).round(4)

        # DPD (Days Past Due) — inverse du credit score
        dpd_probs = {
            "Technologie": 0.15, "Industrie": 0.10, "Sante": 0.05,
            "Immobilier": 0.12, "Services": 0.08,
        }
        has_dpd = np.zeros(n, dtype=bool)
        dpd_values = np.zeros(n, dtype=int)

        for sector_name in SECTOR_NAMES:
            mask = sectors == sector_name
            count = mask.sum()
            if count == 0:
                continue
            # Probabilite de DPD > 0, modulee par le credit_score
            base_prob = dpd_probs[sector_name]
            score_mod = np.clip((650 - credit_score[mask]) / 500, -0.1, 0.2)
            p_dpd = np.clip(base_prob + score_mod, 0.01, 0.5)
            has_dpd[mask] = self.rng.random(count) < p_dpd

        # Valeurs de DPD pour ceux qui en ont
        n_with_dpd = has_dpd.sum()
        if n_with_dpd > 0:
            dpd_choices = np.array([15, 30, 60, 90, 120])
            dpd_probs_dist = np.array([0.40, 0.30, 0.15, 0.10, 0.05])
            dpd_values[has_dpd] = self.rng.choice(
                dpd_choices, size=n_with_dpd, p=dpd_probs_dist,
            )

        df["dpd"] = dpd_values

        # Loan type : Revolving vs Term
        revolving_probs = {
            "Technologie": 0.40, "Industrie": 0.25, "Sante": 0.20,
            "Immobilier": 0.15, "Services": 0.30,
        }
        loan_type = np.full(n, "Term", dtype=object)
        for sector_name in SECTOR_NAMES:
            mask = sectors == sector_name
            count = mask.sum()
            if count == 0:
                continue
            is_revolving = self.rng.random(count) < revolving_probs[sector_name]
            loan_type[mask] = np.where(is_revolving, "Revolving", "Term")

        df["loan_type"] = loan_type

        # Loan amount — correle au revenue, differencies par loan_type
        revenue_arr = df["revenue"].values  # En millions EUR
        multiplier = np.where(
            loan_type == "Revolving",
            self.rng.uniform(0.3, 1.5, n),
            self.rng.uniform(1.0, 4.0, n),
        )
        # Loan amount en EUR (revenue est en millions)
        df["loan_amount"] = np.round(revenue_arr * multiplier * 1_000_000, 2)

        # Collateral — correle au loan_amount, fort pour Immobilier
        collateral_ratios = {
            "Technologie": (0.15, 0.50), "Industrie": (0.30, 0.70),
            "Sante": (0.25, 0.60), "Immobilier": (0.60, 1.00),
            "Services": (0.20, 0.55),
        }
        collateral = np.empty(n)
        for sector_name in SECTOR_NAMES:
            mask = sectors == sector_name
            count = mask.sum()
            if count == 0:
                continue
            lo, hi = collateral_ratios[sector_name]
            ratio = self.rng.uniform(lo, hi, count)
            collateral[mask] = df["loan_amount"].values[mask] * ratio

        df["collateral"] = np.round(collateral, 2)

        # Utilization rate — distribution Beta, differenciee
        util_rate = np.where(
            loan_type == "Revolving",
            self.rng.beta(2.5, 3.0, n),
            self.rng.beta(7.0, 2.0, n),
        )
        # Ajustement par fragilite sectorielle
        fragile_mask = np.isin(sectors, ["Technologie"])
        util_rate = np.where(fragile_mask, np.clip(util_rate + 0.10, 0, 1), util_rate)
        df["utilization_rate"] = np.round(util_rate, 4)

        # Features engineered
        df["loan_to_revenue"] = np.round(
            df["loan_amount"].values / np.clip(df["revenue"].values, 1.0, None), 4,
        )
        df["collateral_coverage"] = np.round(
            df["collateral"].values / np.clip(df["loan_amount"].values, 1.0, None), 4,
        )

        return df

    # ──────────────────────────────────────────────
    # FEATURES PE
    # ──────────────────────────────────────────────

    def _generate_pe_features(
        self,
        common: pd.DataFrame,
        sector_configs: np.ndarray,
    ) -> pd.DataFrame:
        """Genere les features specifiques au canal PE.

        Args:
            common: DataFrame avec features communes.
            sector_configs: Array des SectorConfig par entreprise.

        Returns:
            DataFrame PE complet (communes + specifiques PE).
        """
        n = self.n_clients
        sectors = common["sector"].values
        df = common[["enterprise_id", "sector", "revenue", "ebitda"]].copy()

        # Entry multiple — uniforme dans la fourchette IPEV du secteur
        entry_multiple = np.empty(n)
        for sector in SECTORS:
            mask = sectors == sector.name
            count = mask.sum()
            if count == 0:
                continue
            entry_multiple[mask] = self.rng.uniform(
                sector.entry_multiple_range[0],
                sector.entry_multiple_range[1],
                count,
            )

        df["entry_multiple"] = np.round(entry_multiple, 2)

        # Leverage — differenciee par secteur
        leverage_params = {
            "Technologie": (0.10, 0.40),
            "Industrie": (0.30, 0.60),
            "Sante": (0.20, 0.50),
            "Immobilier": (0.50, 0.70),
            "Services": (0.20, 0.50),
        }
        leverage = np.empty(n)
        for sector_name in SECTOR_NAMES:
            mask = sectors == sector_name
            count = mask.sum()
            if count == 0:
                continue
            lo, hi = leverage_params[sector_name]
            leverage[mask] = self.rng.uniform(lo, hi, count)

        df["leverage"] = np.round(leverage, 4)

        # Vintage — annee d'investissement (2018-2025)
        df["vintage"] = self.rng.integers(2018, 2026, size=n)

        # Holding years = 2026 - vintage
        df["holding_years"] = 2026 - df["vintage"].values

        # Valuation method — depuis la config du secteur
        config_map = {s.name: s.valuation_method for s in SECTORS}
        df["valuation_method"] = pd.Series(sectors).map(config_map).values

        return df

    # ──────────────────────────────────────────────
    # DEFAULT FLAG
    # ──────────────────────────────────────────────

    def _compute_default_flag(
        self,
        df: pd.DataFrame,
        sector_configs: np.ndarray,
    ) -> pd.DataFrame:
        """Calcule le flag de defaut via un modele logistique latent non-lineaire.

        Le modele combine features entreprise + sensibilites macro sectorielles
        avec 5 effets non-lineaires qui differencient les familles d'algorithmes :
            1. Seuil debt_ratio : acceleration convexe au-dela de 0.60
            2. Interaction debt_ratio × taux : les firmes levierisees souffrent
               davantage en hausse de taux
            3. Bimodalite Tech : startups = haut risque OU succes (pas lineaire)
            4. Courbe en U du vintage : prets tres recents et tres anciens
               defaillent davantage
            5. Queues epaisses : evenements extremes rares sur revenue

        Args:
            df: DataFrame credit avec features entreprise.
            sector_configs: Array des SectorConfig par entreprise.

        Returns:
            DataFrame enrichi avec 'default_flag' et 'pd_latent'.
        """
        n = len(df)

        # Conditions macro du dernier mois observe
        current_unemployment = MACRO_HISTORY_BASELINE["unemployment_rate"][-1]
        current_gdp = MACRO_HISTORY_BASELINE["gdp_growth"][-1]
        current_interest_rate = MACRO_HISTORY_BASELINE["interest_rate"][-1]
        current_hpi = MACRO_HISTORY_BASELINE["hpi_growth"][-1]
        current_inflation = MACRO_HISTORY_BASELINE["inflation_rate"][-1]

        # Intercept par secteur : logit(base_default_rate) calibre directement
        # la PD cible. Les features ajoutent de la dispersion intra-secteur.
        sectors = df["sector"].values
        z = np.empty(n)

        for sector in SECTORS:
            mask = sectors == sector.name
            # Intercept = logit(base_default_rate)
            z[mask] = np.log(
                sector.base_default_rate / (1.0 - sector.base_default_rate)
            )

        # ── Contribution lineaire du credit score (centree par secteur) ──
        credit_score_vals = df["credit_score"].values
        cs_centered = np.empty(n)
        for sector_name, cs_mean in _CREDIT_SCORE_MEANS.items():
            mask = sectors == sector_name
            cs_centered[mask] = (cs_mean - credit_score_vals[mask]) / 100
        z += cs_centered * 1.2

        # ── Contribution lineaire du debt ratio ──
        debt_ratio = df["debt_ratio"].values
        z += (debt_ratio - 0.42) * 1.5

        # ── (1) SEUIL DEBT RATIO : acceleration convexe au-dela de 0.60 ──
        # Les entreprises a levier > 60% subissent un risque quadratique
        # supplementaire. Cet effet n'est capturable que par des modeles
        # non-lineaires (arbres, reseaux de neurones).
        excess_debt = np.maximum(0.0, debt_ratio - 0.60)
        z += excess_debt ** 2 * 8.0  # +0.32 logit a 80%, +1.28 a 100%

        # ── Contribution lineaire du DPD ──
        z += (df["dpd"].values - 3.5) / 90.0 * 1.5

        # ── Contribution lineaire de l'utilization rate ──
        util_rate = df["utilization_rate"].values
        z += (util_rate - 0.58) * 0.6

        # ── Contribution lineaire du revenue (taille = stabilite) ──
        log_rev = np.log1p(df["revenue"].values)
        z -= np.clip((log_rev - 3.5) / 3.0, -0.3, 0.3) * 0.3

        # ── (2) INTERACTION debt_ratio × taux d'interet ──
        # Les entreprises tres endettees souffrent plus quand les taux montent.
        # Effet d'interaction que seuls les modeles non-lineaires captent.
        rate_delta = (current_interest_rate - SCENARIO_BASE.interest_rate) / 100
        z += debt_ratio * rate_delta * 3.0  # interaction croisee

        # ── (3) BIMODALITE TECH ──
        # Les startups tech ont un profil bimodal : celles a faible revenue
        # et fort levier sont tres risquees (cash-burn), tandis que celles a
        # fort revenue sont resilientes (economies d'echelle). Cet effet
        # cree une distribution non-separable lineairement.
        tech_mask = sectors == "Technologie"
        tech_small = tech_mask & (df["revenue"].values < 20)  # PME tech < 20M
        tech_large = tech_mask & (df["revenue"].values >= 80)  # Large tech >= 80M
        z[tech_small] += 0.5   # PME tech : +5pp PD environ
        z[tech_large] -= 0.4   # Large tech : -4pp PD (resilientes)

        # ── (4) COURBE EN U DU DPD ──
        # Les prets avec DPD intermediaire (30-60j) sont en zone de vigilance
        # mais peuvent se regulariser. Les DPD > 90j ont un risque qui
        # explose de maniere non-lineaire (passage en defaut technique).
        dpd_vals = df["dpd"].values
        z += np.where(dpd_vals > 60, (dpd_vals - 60) / 30.0 * 0.8, 0.0)

        # ── (5) QUEUES EPAISSES : choc idiosyncratique ──
        # 3% des entreprises subissent un choc idiosyncratique non
        # explicable par les features observees (fraude, perte client majeur,
        # changement reglementaire). Cet effet ajoute du bruit heteroscedastique.
        shock_mask = self.rng.random(n) < 0.03
        z[shock_mask] += self.rng.normal(1.5, 0.5, shock_mask.sum())

        # Compensation du biais : les effets non-lineaires ajoutent un
        # biais positif moyen. On compense pour maintenir les taux de defaut
        # proches des cibles sectorielles.
        z -= 0.55

        # Contribution macro x sensibilite secteur (canal credit)
        # References = SCENARIO_BASE (contribution = 0 au scenario central)
        ref_unemployment = SCENARIO_BASE.unemployment_rate
        ref_gdp = SCENARIO_BASE.gdp_growth
        ref_interest_rate = SCENARIO_BASE.interest_rate
        ref_hpi = SCENARIO_BASE.hpi_growth
        ref_inflation = SCENARIO_BASE.inflation_rate

        for sector in SECTORS:
            mask = sectors == sector.name

            unemployment_effect = (
                (current_unemployment - ref_unemployment) / 100
                * sector.unemployment_sensitivity_credit
            )
            gdp_effect = (
                (ref_gdp - current_gdp) / 100
                * sector.gdp_sensitivity_credit
            )
            interest_rate_effect = (
                (current_interest_rate - ref_interest_rate) / 100
                * sector.interest_rate_sensitivity_credit
            )
            hpi_effect = (
                (ref_hpi - current_hpi) / 100
                * sector.hpi_sensitivity_credit
            )
            inflation_effect = (
                (current_inflation - ref_inflation) / 100
                * sector.inflation_sensitivity_credit
            )

            z[mask] += (
                unemployment_effect + gdp_effect
                + interest_rate_effect + hpi_effect + inflation_effect
            )

        # Probabilite de defaut via logistique
        pd_latent = 1.0 / (1.0 + np.exp(-z))

        # Tirage Bernoulli
        df[TARGET] = (self.rng.random(n) < pd_latent).astype(int)

        # Stocker la PD latente pour validation
        df["pd_latent"] = np.round(pd_latent, 6)

        # pd_origination = PD a l'origination (latente du modele generatif)
        # Utilisee par ECLCalculator pour le SICR au lieu du proxy 0.80 * pd_current
        df["pd_origination"] = np.round(pd_latent, 6)

        return df

    # ──────────────────────────────────────────────
    # HISTORIQUE PANEL
    # ──────────────────────────────────────────────

    def _generate_monthly_history(self, df_credit: pd.DataFrame) -> pd.DataFrame:
        """Genere un panel mensuel (enterprise_id x month) vectorise.

        Chaque mois contient les variables macro, le solde courant et le DPD.
        La generation est entierement vectorisee (pas de boucle Python sur
        les 120K lignes).

        Args:
            df_credit: DataFrame credit avec features entreprise.

        Returns:
            DataFrame panel (n_clients x n_months lignes).
        """
        n = self.n_clients
        m = self.n_months
        total = n * m

        # Indices vectorises
        enterprise_ids = np.repeat(np.arange(n), m)
        months = np.tile(np.arange(1, m + 1), n)

        # Macro variables — tuile pour chaque entreprise ([-m:] car MACRO_HISTORY peut etre > m)
        macro_gdp = np.tile(MACRO_HISTORY_BASELINE["gdp_growth"][-m:], n)
        macro_unemp = np.tile(MACRO_HISTORY_BASELINE["unemployment_rate"][-m:], n)
        macro_interest = np.tile(MACRO_HISTORY_BASELINE["interest_rate"][-m:], n)
        macro_hpi = np.tile(MACRO_HISTORY_BASELINE["hpi_growth"][-m:], n)
        macro_inflation = np.tile(MACRO_HISTORY_BASELINE["inflation_rate"][-m:], n)

        # Balances : random walk autour du loan_amount
        loan_amounts = np.repeat(df_credit["loan_amount"].values, m)
        balance_noise = self.rng.normal(0, 0.05, total)
        balances = np.maximum(0, loan_amounts * (0.85 + balance_noise))

        # DPD mensuel — correle au credit score
        credit_scores = np.repeat(df_credit["credit_score"].values, m)
        dpd_probs = np.clip((650 - credit_scores) / 1000, 0, 0.3)
        has_dpd = self.rng.random(total) < dpd_probs

        dpd_choices = np.array([15, 30, 60, 90, 120])
        dpd_choice_probs = np.array([0.40, 0.30, 0.15, 0.10, 0.05])
        dpd_values = np.where(
            has_dpd,
            self.rng.choice(dpd_choices, size=total, p=dpd_choice_probs),
            0,
        )

        return pd.DataFrame({
            "enterprise_id": enterprise_ids,
            "month": months,
            "balance": np.round(balances, 2),
            "dpd": dpd_values.astype(int),
            "gdp_growth": macro_gdp,
            "unemployment_rate": macro_unemp,
            "interest_rate": macro_interest,
            "hpi_growth": macro_hpi,
            "inflation_rate": macro_inflation,
        })

    # ──────────────────────────────────────────────
    # BRUIT REALISTE
    # ──────────────────────────────────────────────

    def _add_realistic_noise(
        self,
        df_credit: pd.DataFrame,
        df_pe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ajoute du bruit realiste pour simuler la qualite de donnees.

        Introduit ~2% de valeurs manquantes dans les features non-critiques
        et quelques outliers sur revenue pour tester la robustesse.
        Les outliers revenue sont propages a df_pe pour coherence.

        Args:
            df_credit: DataFrame credit complet.
            df_pe: DataFrame PE complet.

        Returns:
            Tuple (df_credit, df_pe) avec bruit ajoute.
        """
        n = len(df_credit)

        # Valeurs manquantes sur collateral (~2%)
        missing_mask = self.rng.random(n) < 0.02
        df_credit.loc[missing_mask, "collateral"] = np.nan

        # Valeurs manquantes sur utilization_rate (~1.5%)
        missing_mask2 = self.rng.random(n) < 0.015
        df_credit.loc[missing_mask2, "utilization_rate"] = np.nan

        # Quelques outliers sur revenue (top 0.5%) — propages aux deux DataFrames
        outlier_mask = self.rng.random(n) < 0.005
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            multiplier = self.rng.uniform(3, 5, n_outliers)
            df_credit.loc[outlier_mask, "revenue"] = (
                df_credit.loc[outlier_mask, "revenue"] * multiplier
            )
            df_pe.loc[outlier_mask, "revenue"] = (
                df_pe.loc[outlier_mask, "revenue"] * multiplier
            )

        return df_credit, df_pe

    # ──────────────────────────────────────────────
    # VALIDATION CONTRATS AR4
    # ──────────────────────────────────────────────

    def _validate_contracts(
        self,
        df_credit: pd.DataFrame,
        df_pe: pd.DataFrame,
        df_history: pd.DataFrame,
    ) -> None:
        """Valide les contrats DataFrame AR4.

        Verifie que les colonnes obligatoires sont presentes et que les
        shapes correspondent aux attentes.

        Args:
            df_credit: DataFrame credit.
            df_pe: DataFrame PE.
            df_history: DataFrame historique panel.

        Raises:
            ValueError: Si un contrat est viole.
        """
        # Colonnes credit
        credit_cols = set(df_credit.columns)
        missing_credit = REQUIRED_CREDIT_COLS - credit_cols
        if missing_credit:
            raise ValueError(
                f"Colonnes credit manquantes : {missing_credit}"
            )

        # Colonnes PE
        pe_cols = set(df_pe.columns)
        missing_pe = REQUIRED_PE_COLS - pe_cols
        if missing_pe:
            raise ValueError(
                f"Colonnes PE manquantes : {missing_pe}"
            )

        # Shapes
        if len(df_credit) != self.n_clients:
            raise ValueError(
                f"df_credit : {len(df_credit)} lignes, attendu {self.n_clients}"
            )
        if len(df_pe) != self.n_clients:
            raise ValueError(
                f"df_pe : {len(df_pe)} lignes, attendu {self.n_clients}"
            )
        expected_history = self.n_clients * self.n_months
        if len(df_history) != expected_history:
            raise ValueError(
                f"df_history : {len(df_history)} lignes, attendu {expected_history}"
            )

        # Enterprise IDs coherents
        credit_ids = set(df_credit["enterprise_id"])
        pe_ids = set(df_pe["enterprise_id"])
        if credit_ids != pe_ids:
            raise ValueError(
                "Enterprise IDs incoherents entre df_credit et df_pe"
            )

    # ──────────────────────────────────────────────
    # STATISTIQUES
    # ──────────────────────────────────────────────

    def get_summary_stats(
        self,
        df_credit: pd.DataFrame,
        df_pe: pd.DataFrame,
    ) -> Dict[str, object]:
        """Calcule les statistiques descriptives du dataset genere.

        Args:
            df_credit: DataFrame credit.
            df_pe: DataFrame PE.

        Returns:
            Dictionnaire de statistiques cles.
        """
        total = len(df_credit)
        defaults = df_credit[TARGET].sum()
        return {
            "n_enterprises": total,
            "n_defaults": int(defaults),
            "default_rate": round(defaults / total, 4),
            "sector_distribution": (
                df_credit["sector"]
                .value_counts(normalize=True)
                .round(4)
                .to_dict()
            ),
            "sector_default_rates": (
                df_credit.groupby("sector")[TARGET]
                .mean()
                .round(4)
                .to_dict()
            ),
            "mean_credit_score": round(df_credit["credit_score"].mean(), 1),
            "mean_debt_ratio": round(df_credit["debt_ratio"].mean(), 4),
            "mean_revenue_m": round(df_credit["revenue"].mean(), 2),
            "pe_multiples_by_sector": (
                df_pe.groupby("sector")["entry_multiple"]
                .agg(["min", "mean", "max"])
                .round(2)
                .to_dict("index")
            ),
            "revolving_pct": round(
                (df_credit["loan_type"] == "Revolving").mean(), 4
            ),
            "missing_pct": round(
                df_credit.isnull().mean().mean(), 4
            ),
        }


def generate_dataset(
    n_clients: int = N_CLIENTS,
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Genere le dataset via le synthetic_generator (bridge).

    Delegue a synthetic_generator.generate_dataset() qui produit un
    4-tuple (df_credit, df_pe, df_history, df_balance_sheet) compatible
    avec le cockpit.

    Args:
        n_clients: Nombre d'entreprises.
        seed: Graine aleatoire (123 pour le portfolio, 42 pour l'entrainement).

    Returns:
        Tuple (df_credit, df_pe, df_history, df_balance_sheet).
    """
    from ifrs9_cockpit.synthetic_generator import (
        generate_dataset as _sg_generate,
    )

    return _sg_generate(n_clients=n_clients, seed=seed)


if __name__ == "__main__":
    print("=" * 60)
    print("IFRS 9 COCKPIT — Synthetic Data Generation Report")
    print("=" * 60)

    # Generation
    gen = SyntheticDataGenerator()
    df_credit, df_pe, df_history = gen.generate()
    stats = gen.get_summary_stats(df_credit, df_pe)

    print(f"\nEntreprises generees : {stats['n_enterprises']:,}")
    print(f"Defauts              : {stats['n_defaults']:,}")
    print(f"Taux de defaut       : {stats['default_rate']:.2%}")
    print(f"Score credit moyen   : {stats['mean_credit_score']}")
    print(f"Debt ratio moyen     : {stats['mean_debt_ratio']:.2%}")
    print(f"Revenue moyen        : {stats['mean_revenue_m']:.2f} M EUR")
    print(f"Part revolving       : {stats['revolving_pct']:.1%}")
    print(f"Missing values       : {stats['missing_pct']:.2%}")

    print(f"\n--- Shapes ---")
    print(f"  df_credit  : {df_credit.shape}")
    print(f"  df_pe      : {df_pe.shape}")
    print(f"  df_history : {df_history.shape}")

    print(f"\n--- Repartition sectorielle ---")
    for sector in SECTORS:
        obs_pct = stats["sector_distribution"].get(sector.name, 0)
        obs_dr = stats["sector_default_rates"].get(sector.name, 0)
        pe_stats = stats["pe_multiples_by_sector"].get(sector.name, {})
        pe_min = pe_stats.get("min", 0)
        pe_mean = pe_stats.get("mean", 0)
        pe_max = pe_stats.get("max", 0)
        print(
            f"  {sector.name:15s} | obs={obs_pct:5.1%} (cfg={sector.proportion:.0%}) "
            f"| DR={obs_dr:5.2%} (cible~{sector.base_default_rate:.0%}) "
            f"| Multiple PE {pe_min:.1f}-{pe_mean:.1f}-{pe_max:.1f}x "
            f"(cfg={sector.entry_multiple_range[0]:.0f}-{sector.entry_multiple_range[1]:.0f}x)"
        )

    print(f"\n--- Correlations credit (credit_score, debt_ratio, revenue, ebitda) ---")
    corr_cols = ["credit_score", "debt_ratio", "revenue", "ebitda"]
    corr_matrix = df_credit[corr_cols].corr().round(3)
    print(corr_matrix.to_string())

    print(f"\n--- Contrats AR4 ---")
    credit_ok = REQUIRED_CREDIT_COLS.issubset(set(df_credit.columns))
    pe_ok = REQUIRED_PE_COLS.issubset(set(df_pe.columns))
    print(f"  REQUIRED_CREDIT_COLS : {'PASS' if credit_ok else 'FAIL'}")
    print(f"  REQUIRED_PE_COLS     : {'PASS' if pe_ok else 'FAIL'}")

    print(f"\n--- Test reproductibilite ---")
    gen2 = SyntheticDataGenerator()
    df_credit2, df_pe2, df_history2 = gen2.generate()
    hash1 = hashlib.md5(
        pd.util.hash_pandas_object(df_credit).values.tobytes()
    ).hexdigest()
    hash2 = hashlib.md5(
        pd.util.hash_pandas_object(df_credit2).values.tobytes()
    ).hexdigest()
    repro_ok = hash1 == hash2
    print(f"  Run 1 hash : {hash1[:16]}...")
    print(f"  Run 2 hash : {hash2[:16]}...")
    print(f"  Reproductibilite : {'PASS' if repro_ok else 'FAIL'}")

    print(f"\n--- Apercu df_credit ---")
    print(df_credit.head(3).to_string())
    print(f"\nColonnes : {list(df_credit.columns)}")

    print(f"\n--- Apercu df_pe ---")
    print(df_pe.head(3).to_string())
    print(f"\nColonnes : {list(df_pe.columns)}")

    print(f"\n--- Apercu df_history ---")
    print(df_history.head(5).to_string())

    if credit_ok and pe_ok and repro_ok:
        print("\nGeneration valide.")
    else:
        print("\nATTENTION : des validations ont echoue.")
