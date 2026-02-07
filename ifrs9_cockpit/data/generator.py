"""Générateur de données synthétiques pour le Cockpit IFRS 9.

Produit un dataset de 10 000 clients avec historique mensuel,
variables macro et segments intentionnellement fragiles
(jeunes actifs très sensibles au chômage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.config import (
    CATEGORICAL_FEATURES,
    MACRO_HISTORY_BASELINE,
    N_CLIENTS,
    N_MONTHS,
    NUMERICAL_FEATURES,
    RANDOM_SEED,
    SEGMENTS,
    SegmentConfig,
    TARGET,
)


class SyntheticDataGenerator:
    """Génère un jeu de données bancaires synthétiques réalistes.

    Le générateur crée des profils clients corrélés avec des variables
    macroéconomiques et introduit des segments fragiles dont le taux
    de défaut réagit fortement aux conditions économiques.

    Attributes:
        n_clients: Nombre total de clients à générer.
        n_months: Nombre de mois d'historique.
        seed: Graine pour la reproductibilité.
        rng: Générateur aléatoire numpy.
    """

    def __init__(
        self,
        n_clients: int = N_CLIENTS,
        n_months: int = N_MONTHS,
        seed: int = RANDOM_SEED,
    ) -> None:
        """Initialise le générateur.

        Args:
            n_clients: Nombre de clients à générer.
            n_months: Nombre de mois d'historique.
            seed: Graine aléatoire pour reproductibilité.
        """
        self.n_clients = n_clients
        self.n_months = n_months
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Pipeline principal de génération.

        Returns:
            Tuple contenant :
                - df_clients : DataFrame avec un snapshot par client
                  (features + target), prêt pour la modélisation.
                - df_history : DataFrame panel avec historique mensuel
                  (client_id × month), incluant les variables macro.
        """
        # 1. Assigner les segments
        segments = self._assign_segments()

        # 2. Générer les features statiques par client
        df_clients = self._generate_client_features(segments)

        # 3. Générer l'historique mensuel avec macro
        df_history = self._generate_monthly_history(df_clients)

        # 4. Calculer le flag de défaut (snapshot final)
        df_clients = self._compute_default_flag(df_clients)

        # 5. Générer les données de crédit (montants, types)
        df_clients = self._generate_credit_data(df_clients)

        # 6. Ajouter le bruit réaliste
        df_clients = self._add_realistic_noise(df_clients)

        return df_clients, df_history

    def _assign_segments(self) -> np.ndarray:
        """Assigne un segment à chaque client selon les proportions configurées.

        Returns:
            Array de noms de segments, un par client.
        """
        segment_names = [s.name for s in SEGMENTS]
        proportions = [s.proportion for s in SEGMENTS]
        return self.rng.choice(
            segment_names,
            size=self.n_clients,
            p=proportions,
        )

    def _get_segment_config(self, name: str) -> SegmentConfig:
        """Récupère la configuration d'un segment par son nom.

        Args:
            name: Nom du segment.

        Returns:
            Configuration du segment.
        """
        return next(s for s in SEGMENTS if s.name == name)

    def _generate_client_features(self, segments: np.ndarray) -> pd.DataFrame:
        """Génère les features de chaque client en fonction de son segment.

        Produit des distributions réalistes et corrélées : un score
        de crédit bas implique un revenu plus faible et un ratio
        d'endettement plus élevé.

        Args:
            segments: Array des segments assignés.

        Returns:
            DataFrame avec les features client.
        """
        n = self.n_clients
        records: List[Dict] = []

        for i in range(n):
            seg = self._get_segment_config(segments[i])

            # Âge dans la tranche du segment
            age = self.rng.integers(seg.age_range[0], seg.age_range[1] + 1)

            # Score de crédit (distribution normale tronquée autour de la moyenne segment)
            credit_score = int(np.clip(
                self.rng.normal(seg.avg_credit_score, 60),
                300, 850,
            ))

            # Revenu corrélé au score de crédit
            income_base = self.rng.uniform(*seg.income_range)
            credit_factor = (credit_score - 500) / 350  # [-0.57, 1.0]
            income = max(12_000, income_base * (0.7 + 0.3 * credit_factor))

            # Durée d'emploi (corrélée à l'âge)
            max_employment = max(0, age - 20)
            employment_duration = self.rng.integers(0, max(1, max_employment + 1))

            # Type d'emploi
            if employment_duration > 5:
                emp_probs = [0.70, 0.15, 0.10, 0.05]
            elif employment_duration > 2:
                emp_probs = [0.40, 0.30, 0.20, 0.10]
            else:
                emp_probs = [0.15, 0.25, 0.35, 0.25]
            employment_type = self.rng.choice(
                ["CDI", "CDD", "Interim", "Independant"],
                p=emp_probs,
            )

            # Ratio d'endettement (inversement corrélé au score)
            debt_ratio_base = max(0.05, 0.60 - credit_factor * 0.25)
            debt_ratio = np.clip(
                self.rng.normal(debt_ratio_base, 0.08),
                0.02, 0.95,
            )

            # Nombre de lignes de crédit
            nb_credit_lines = max(1, int(self.rng.poisson(2.5 + credit_factor)))

            # Historique de retard
            if credit_score < 600:
                nb_past_due = int(self.rng.poisson(1.5))
            elif credit_score < 700:
                nb_past_due = int(self.rng.poisson(0.5))
            else:
                nb_past_due = int(self.rng.poisson(0.15))

            # Mois depuis la dernière délinquance
            if nb_past_due > 0:
                months_since_delinq = self.rng.integers(1, 25)
            else:
                months_since_delinq = 99  # Jamais en retard → valeur sentinelle

            records.append({
                "client_id": i,
                "segment": seg.name,
                "age": age,
                "income": round(income, 2),
                "credit_score": credit_score,
                "employment_duration": employment_duration,
                "employment_type": employment_type,
                "debt_ratio": round(debt_ratio, 4),
                "nb_credit_lines": nb_credit_lines,
                "nb_past_due_30d": nb_past_due,
                "months_since_last_delinquency": months_since_delinq,
            })

        return pd.DataFrame(records)

    def _generate_credit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Génère les données de crédit : montant, type de prêt, utilisation.

        Args:
            df: DataFrame client avec features existantes.

        Returns:
            DataFrame enrichi avec les données de crédit.
        """
        n = len(df)

        # Type de prêt : revolving vs terme
        loan_type_probs = np.where(
            df["segment"].isin(["Jeunes_Actifs", "Primo_Accedants"]),
            0.45,  # Plus de revolving chez les jeunes
            0.25,
        )
        df["loan_type"] = np.where(
            self.rng.random(n) < loan_type_probs,
            "Revolving",
            "Term",
        )

        # Montant du prêt (corrélé au revenu)
        income_arr = df["income"].values
        multiplier = np.where(
            df["loan_type"] == "Revolving",
            self.rng.uniform(0.3, 1.5, n),
            self.rng.uniform(2.0, 8.0, n),
        )
        df["loan_amount"] = np.round(income_arr * multiplier, 2)

        # Taux d'utilisation (pour revolving principalement)
        base_util = np.where(
            df["loan_type"] == "Revolving",
            self.rng.beta(2.5, 3.0, n),     # Distribution réaliste
            self.rng.beta(8.0, 2.0, n),     # Prêts terme : haute utilisation
        )
        # Les clients fragiles utilisent plus leurs lignes
        fragile_mask = df["segment"].isin(["Jeunes_Actifs", "Primo_Accedants"])
        base_util = np.where(fragile_mask, np.clip(base_util + 0.15, 0, 1), base_util)
        df["utilization_rate"] = np.round(base_util, 4)

        return df

    def _compute_default_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule le flag de défaut basé sur un modèle latent réaliste.

        Le modèle combine les caractéristiques client avec les
        sensibilités macroéconomiques par segment. Les jeunes actifs
        ont une sensibilité au chômage 2.5x, créant des poches de
        risque intentionnelles pour les stress tests.

        Args:
            df: DataFrame avec les features client.

        Returns:
            DataFrame avec la colonne 'default_flag' ajoutée.
        """
        n = len(df)

        # Conditions macro du dernier mois observé
        current_unemployment = MACRO_HISTORY_BASELINE["unemployment_rate"][-1]
        current_gdp = MACRO_HISTORY_BASELINE["gdp_growth"][-1]

        # Intercept négatif pour calibrer le taux de défaut global (~5-7%)
        z = np.full(n, -4.2)

        # Contribution du score de crédit (le driver principal)
        credit_z = (650 - df["credit_score"].values) / 100
        z += credit_z * 0.8

        # Contribution du ratio d'endettement
        z += (df["debt_ratio"].values - 0.35) * 1.5

        # Contribution de l'historique de retard
        z += df["nb_past_due_30d"].values * 0.6

        # Contribution de la stabilité d'emploi
        emp_map = {"CDI": -0.2, "CDD": 0.3, "Interim": 0.6, "Independant": 0.2}
        emp_contrib = df["employment_type"].map(emp_map).values.astype(float)
        z += emp_contrib

        # Contribution de la durée d'emploi (effet protecteur)
        emp_dur = df["employment_duration"].values.copy()
        emp_dur = np.where(np.isnan(emp_dur), 3.0, emp_dur)
        z -= np.clip(emp_dur / 20, 0, 1) * 0.5

        # Contribution macro × sensibilité segment
        for seg in SEGMENTS:
            mask = df["segment"].values == seg.name
            unemployment_effect = (
                (current_unemployment - 7.0)
                * seg.unemployment_sensitivity
                * 0.15
            )
            gdp_effect = (
                (1.5 - current_gdp)
                * seg.gdp_sensitivity
                * 0.10
            )
            z[mask] += unemployment_effect + gdp_effect

            # Ajuster le niveau de base par segment
            z[mask] += np.log(seg.base_default_rate / 0.05) * 0.5

        # Probabilité de défaut via logistique
        pd_latent = 1.0 / (1.0 + np.exp(-z))

        # Tirage Bernoulli
        df[TARGET] = (self.rng.random(n) < pd_latent).astype(int)

        # Stocker la PD latente pour validation
        df["pd_latent"] = np.round(pd_latent, 6)

        return df

    def _generate_monthly_history(self, df_clients: pd.DataFrame) -> pd.DataFrame:
        """Génère un panel mensuel (client_id × month) avec trajectoire.

        Chaque mois contient :
            - Les variables macroéconomiques
            - Le solde courant du client (avec tendance)
            - Les jours de retard (DPD)

        Args:
            df_clients: DataFrame des features client.

        Returns:
            DataFrame panel (n_clients × n_months lignes).
        """
        records: List[Dict] = []
        macro_gdp = MACRO_HISTORY_BASELINE["gdp_growth"]
        macro_unemp = MACRO_HISTORY_BASELINE["unemployment_rate"]

        # Pré-calcul des trajectoires de solde pour tous les clients
        n = len(df_clients)

        for month in range(self.n_months):
            # Solde : random walk autour du montant initial
            balance_noise = self.rng.normal(0, 0.05, n)
            balances = df_clients["income"].values * (0.3 + balance_noise)

            # DPD (Days Past Due) — corrélé au score de crédit
            dpd_probs = np.clip(
                (650 - df_clients["credit_score"].values) / 1000,
                0, 0.3,
            )
            has_dpd = self.rng.random(n) < dpd_probs
            dpd_values = np.where(
                has_dpd,
                self.rng.choice([15, 30, 60, 90, 120], n, p=[0.4, 0.3, 0.15, 0.10, 0.05]),
                0,
            )

            for i in range(n):
                records.append({
                    "client_id": i,
                    "month": month + 1,
                    "balance": round(max(0, balances[i]), 2),
                    "dpd": int(dpd_values[i]),
                    "gdp_growth": macro_gdp[month],
                    "unemployment_rate": macro_unemp[month],
                })

        return pd.DataFrame(records)

    def _add_realistic_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute du bruit réaliste pour simuler la qualité de données bancaires.

        Introduit ~2% de valeurs manquantes dans les features non-critiques
        et quelques outliers pour tester la robustesse des modèles.

        Args:
            df: DataFrame client complet.

        Returns:
            DataFrame avec bruit ajouté.
        """
        n = len(df)

        # Valeurs manquantes sur employment_duration (~3%)
        missing_mask = self.rng.random(n) < 0.03
        df.loc[missing_mask, "employment_duration"] = np.nan

        # Valeurs manquantes sur months_since_last_delinquency (~2%)
        missing_mask2 = self.rng.random(n) < 0.02
        df.loc[missing_mask2, "months_since_last_delinquency"] = np.nan

        # Quelques outliers sur income (top 0.5%)
        outlier_mask = self.rng.random(n) < 0.005
        df.loc[outlier_mask, "income"] = df.loc[outlier_mask, "income"] * self.rng.uniform(3, 5, outlier_mask.sum())

        return df

    def get_summary_stats(self, df_clients: pd.DataFrame) -> Dict[str, object]:
        """Calcule les statistiques descriptives du dataset généré.

        Args:
            df_clients: DataFrame client.

        Returns:
            Dictionnaire de statistiques clés.
        """
        total = len(df_clients)
        defaults = df_clients[TARGET].sum()
        return {
            "n_clients": total,
            "n_defaults": int(defaults),
            "default_rate": round(defaults / total, 4),
            "segment_distribution": (
                df_clients["segment"]
                .value_counts(normalize=True)
                .round(4)
                .to_dict()
            ),
            "segment_default_rates": (
                df_clients.groupby("segment")[TARGET]
                .mean()
                .round(4)
                .to_dict()
            ),
            "mean_credit_score": round(df_clients["credit_score"].mean(), 1),
            "mean_debt_ratio": round(df_clients["debt_ratio"].mean(), 4),
            "revolving_pct": round(
                (df_clients["loan_type"] == "Revolving").mean(), 4
            ),
            "missing_pct": round(
                df_clients.isnull().mean().mean(), 4
            ),
        }


def generate_dataset(
    n_clients: int = N_CLIENTS,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fonction utilitaire pour générer le dataset en une ligne.

    Args:
        n_clients: Nombre de clients.
        seed: Graine aléatoire.

    Returns:
        Tuple (df_clients, df_history).
    """
    generator = SyntheticDataGenerator(n_clients=n_clients, seed=seed)
    return generator.generate()


if __name__ == "__main__":
    # Exécution standalone pour validation
    gen = SyntheticDataGenerator()
    df_clients, df_history = gen.generate()
    stats = gen.get_summary_stats(df_clients)

    print("=" * 60)
    print("IFRS 9 COCKPIT — Synthetic Data Generation Report")
    print("=" * 60)
    print(f"\nClients générés    : {stats['n_clients']:,}")
    print(f"Défauts            : {stats['n_defaults']:,}")
    print(f"Taux de défaut     : {stats['default_rate']:.2%}")
    print(f"Score crédit moyen : {stats['mean_credit_score']}")
    print(f"Debt ratio moyen   : {stats['mean_debt_ratio']:.2%}")
    print(f"Part revolving     : {stats['revolving_pct']:.1%}")
    print(f"Missing values     : {stats['missing_pct']:.2%}")
    print(f"\nHistorique         : {len(df_history):,} lignes "
          f"({stats['n_clients']:,} clients × {gen.n_months} mois)")

    print("\n--- Distribution par segment ---")
    for seg, pct in stats["segment_distribution"].items():
        dr = stats["segment_default_rates"].get(seg, 0)
        print(f"  {seg:20s} : {pct:6.1%} du portfolio | DR = {dr:.2%}")

    print("\n--- Aperçu df_clients ---")
    print(df_clients.head(3).to_string())
    print(f"\nShape : {df_clients.shape}")
    print(f"Columns : {list(df_clients.columns)}")

    print("\n--- Aperçu df_history ---")
    print(df_history.head(5).to_string())
    print(f"\nShape : {df_history.shape}")
