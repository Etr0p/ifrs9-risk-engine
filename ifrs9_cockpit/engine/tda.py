"""Analyse Topologique des Données (TDA) — Indice de fragilité sectorielle.

Détecte l'effondrement de la diversification sectorielle via l'homologie
persistante du complexe de Vietoris-Rips construit sur la matrice de
corrélation inter-sectorielle.

Principe :
    1. Construire la matrice de distance d_ij = 1 - |ρ_ij| entre secteurs.
    2. Filtration de Vietoris-Rips : augmenter ε de 0 à 1.
       - ε petit → graphe déconnecté (secteurs indépendants = diversifié).
       - ε grand → graphe complet (tout corrélé = perte de diversification).
    3. Homologie persistante :
       - H0 (composantes connexes) : suit la fusion des clusters.
       - H1 (trous/cycles) : détecte les structures de corrélation non-triviales.
    4. Indice de fragilité = 1 - (longueur moyenne des barres H1) / max_possible.
       - Fragilité haute → les "trous" disparaissent vite → perte de diversification.
       - Fragilité basse → structure riche → diversification préservée.

Confiné aux agrégats sectoriels (5×5 ou 10×10) — O(N³) interdit pour 30K clients.

Références :
    - Edelsbrunner & Harer (2010): Computational Topology
    - Topaz, Ziegelmeier, Halverson (2015): TDA of Financial Time Series
    - Gidea & Katz (2018): Topological Data Analysis of Financial Crises
    - Kramár et al. (2013): Persistence of Force Networks in Compressed Granular Media
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PersistenceBar:
    """Barre de persistance (naissance, mort) dans la filtration.

    Attributes:
        birth: Seuil ε de naissance de la feature topologique.
        death: Seuil ε de mort (disparition).
        dimension: Dimension homologique (0=composante, 1=trou).
        persistence: Durée de vie = death - birth.
    """
    birth: float
    death: float
    dimension: int

    @property
    def persistence(self) -> float:
        return self.death - self.birth


@dataclass(frozen=True)
class TDAResult:
    """Résultat de l'analyse topologique.

    Attributes:
        fragility_index: Indice de fragilité topologique [0, 1].
            0 = diversification parfaite, 1 = corrélation totale.
        bars_h0: Barres H0 (composantes connexes).
        bars_h1: Barres H1 (trous/cycles).
        n_components_at_threshold: Nombre de composantes à ε = 0.5.
        max_persistence_h1: Plus longue barre H1 (feature la plus stable).
        mean_correlation: Corrélation moyenne absolue.
        distance_matrix: Matrice de distances utilisée.
        sector_names: Noms des secteurs.
    """
    fragility_index: float
    bars_h0: List[PersistenceBar]
    bars_h1: List[PersistenceBar]
    n_components_at_threshold: int
    max_persistence_h1: float
    mean_correlation: float
    distance_matrix: np.ndarray
    sector_names: List[str]


def _correlation_to_distance(corr: np.ndarray) -> np.ndarray:
    """Convertit une matrice de corrélation en matrice de distance.

    d(i,j) = 1 - |ρ(i,j)|
    Métrique ultra-métrique : d=0 si parfaitement corrélés (positif ou négatif),
    d=1 si indépendants.

    Args:
        corr: Matrice de corrélation (N, N).

    Returns:
        Matrice de distances (N, N).
    """
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    return dist


def _vietoris_rips_h0(dist: np.ndarray) -> List[PersistenceBar]:
    """Calcule H0 (composantes connexes) via filtration de Vietoris-Rips.

    Équivalent au dendrogramme de single-linkage clustering.

    Algorithme :
        1. Trier toutes les arêtes par poids croissant.
        2. Union-Find : fusionner les composantes quand l'arête apparaît.
        3. Chaque fusion = mort d'une composante H0.

    Args:
        dist: Matrice de distances (N, N).

    Returns:
        Liste de barres H0.
    """
    n = dist.shape[0]

    # Extraire les arêtes (i < j) triées par distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist[i, j], i, j))
    edges.sort()

    # Union-Find
    parent = list(range(n))
    birth = [0.0] * n  # Toutes les composantes naissent à ε=0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    bars = []
    for d_val, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            # Fusionner : la composante la plus jeune meurt
            parent[rj] = ri
            bars.append(PersistenceBar(birth=0.0, death=d_val, dimension=0))

    # La dernière composante ne meurt jamais (persiste à l'infini → clamp à 1.0)
    bars.append(PersistenceBar(birth=0.0, death=1.0, dimension=0))

    return bars


def _vietoris_rips_h1(dist: np.ndarray) -> List[PersistenceBar]:
    """Calcule H1 (trous/cycles) via filtration de Vietoris-Rips simplifiée.

    Pour N ≤ 10, on peut énumérer tous les triangles et détecter
    les cycles qui se forment puis se remplissent.

    Algorithme simplifié (correct pour petites matrices) :
        1. Trier les arêtes par poids croissant.
        2. Pour chaque nouveau triangle (3 arêtes toutes présentes),
           vérifier si un cycle H1 existait et est maintenant rempli.
        3. Un cycle naît quand 2 arêtes d'un triangle sont présentes
           mais pas la 3ème, et meurt quand la 3ème apparaît.

    Args:
        dist: Matrice de distances (N, N).

    Returns:
        Liste de barres H1 (peut être vide pour N petit).
    """
    n = dist.shape[0]
    if n < 3:
        return []

    # Toutes les arêtes triées
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist[i, j], i, j))
    edges.sort()

    # Union-Find pour H0
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
            return True  # Fusion (arbre)
        return False  # Cycle

    bars_h1 = []

    for d_val, i, j in edges:
        if not union(i, j):
            # Cette arête crée un cycle (pas de fusion → H1 naît)
            # Chercher quand ce cycle sera rempli par un triangle
            # Approximation : le cycle meurt à la prochaine arête de triangle
            death = d_val  # Simplification : cycle immédiatement rempli

            # Chercher le triangle minimal contenant (i, j)
            min_third = 1.0
            for k in range(n):
                if k != i and k != j:
                    d_ik = dist[i, k]
                    d_jk = dist[j, k]
                    max_edge = max(d_val, d_ik, d_jk)
                    if max_edge < min_third:
                        min_third = max_edge

            if min_third > d_val:
                bars_h1.append(PersistenceBar(
                    birth=d_val,
                    death=min_third,
                    dimension=1,
                ))

    return bars_h1


def compute_fragility_index(
    correlation_matrix: np.ndarray,
    sector_names: Optional[List[str]] = None,
) -> TDAResult:
    """Calcule l'indice de fragilité topologique depuis la corrélation sectorielle.

    Args:
        correlation_matrix: Matrice de corrélation (N, N) entre secteurs.
        sector_names: Noms des secteurs. Si None, utilise S0, S1, ...

    Returns:
        TDAResult avec indice de fragilité et diagramme de persistance.
    """
    corr = np.asarray(correlation_matrix, dtype=float)
    n = corr.shape[0]

    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"Matrice doit etre carree, recu {corr.shape}")

    if sector_names is None:
        sector_names = [f"S{i}" for i in range(n)]

    # Forcer symétrie et diagonale = 1
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    # Distance
    dist = _correlation_to_distance(corr)

    # Homologie persistante
    bars_h0 = _vietoris_rips_h0(dist)
    bars_h1 = _vietoris_rips_h1(dist)

    # Indice de fragilité
    # Basé sur la corrélation moyenne + absence de trous H1
    mean_corr = np.mean(np.abs(corr[np.triu_indices(n, k=1)]))

    if bars_h1:
        max_pers_h1 = max(b.persistence for b in bars_h1)
        mean_pers_h1 = np.mean([b.persistence for b in bars_h1])
        # Fragilité = corrélation élevée + trous de courte durée
        h1_component = 1.0 - min(mean_pers_h1 / 0.5, 1.0)  # Normaliser par 0.5
    else:
        max_pers_h1 = 0.0
        h1_component = 1.0  # Pas de trous = corrélation homogène = fragile

    # Fragilité composite : 60% corrélation + 40% topologie
    fragility = 0.6 * mean_corr + 0.4 * h1_component
    fragility = float(np.clip(fragility, 0, 1))

    # Nombre de composantes à ε = 0.5
    n_comp = sum(1 for b in bars_h0 if b.death > 0.5)

    return TDAResult(
        fragility_index=round(fragility, 4),
        bars_h0=bars_h0,
        bars_h1=bars_h1,
        n_components_at_threshold=n_comp,
        max_persistence_h1=round(max_pers_h1, 4),
        mean_correlation=round(mean_corr, 4),
        distance_matrix=dist,
        sector_names=sector_names,
    )


def compute_macro_fragility(
    macro_history: Dict[str, List[float]],
    window: int = 24,
) -> TDAResult:
    """Calcule la fragilité topologique à partir de l'historique macro.

    Estime la corrélation empirique sur une fenêtre glissante puis
    applique l'analyse TDA.

    Args:
        macro_history: Dict {var_name: [v_1, ..., v_T]}.
        window: Fenêtre pour estimation de corrélation (mois).

    Returns:
        TDAResult avec indice de fragilité.
    """
    _MACRO_VARS = [
        "unemployment_rate",
        "gdp_growth",
        "interest_rate",
        "hpi_growth",
        "inflation_rate",
    ]

    var_names = [v for v in _MACRO_VARS if v in macro_history]
    arrays = [np.array(macro_history[v]) for v in var_names]
    T = min(len(a) for a in arrays)

    # Prendre les derniers `window` mois
    start = max(0, T - window)
    data = np.column_stack([a[start:T] for a in arrays])

    # Corrélation empirique
    corr = np.corrcoef(data.T)

    # Labels courts
    _SHORT = {
        "unemployment_rate": "Chomage",
        "gdp_growth": "PIB",
        "interest_rate": "Taux",
        "hpi_growth": "HPI",
        "inflation_rate": "Inflation",
    }
    labels = [_SHORT.get(v, v) for v in var_names]

    return compute_fragility_index(corr, sector_names=labels)
