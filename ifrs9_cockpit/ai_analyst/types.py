"""Types de donnees pour le module AI Analyst.

Structures auto-documentees permettant au dashboard de lire des champs
types au lieu de naviguer un dictionnaire opaque.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class RegimeClassification:
    """Resultat de la classification de regime macro (Couche 4).

    Attributes:
        probabilities: Probabilites softmax sur les 5 regimes canoniques.
        detected_regime: Regime dominant (probabilite max).
    """

    probabilities: Dict[str, float]
    detected_regime: str


@dataclass
class Recommendation:
    """Recommandation explicable du CRO virtuel (FR31).

    Porte la chaine complete de raisonnement :
        regime -> declencheur -> euler -> facteur_macro -> risk_appetite ->
        distance_rst -> confiance -> alternatives.

    Attributes:
        action: Action recommandee.
        regime: Regime detecte.
        trigger: Declencheur principal.
        euler_driver: Cellule Euler dominante.
        macro_factor: Variable macro dominante.
        risk_appetite_status: Statut feux tricolores (vert/ambre/rouge).
        rst_distance: Distance au scenario de rupture (sigma).
        confidence: Niveau de confiance (high/medium/low).
        alternatives: Actions alternatives possibles.
    """

    action: str
    regime: str
    trigger: str
    euler_driver: str
    macro_factor: str
    risk_appetite_status: str
    rst_distance: float
    confidence: str
    alternatives: List[str] = field(default_factory=list)


@dataclass
class AnalyticsState:
    """Etat complet du pipeline analytique CRO (5 couches, 2 passes).

    Structure auto-documentee consommee par le dashboard.

    Attributes:
        asymmetry_matrix: Matrice d'asymetrie 5 secteurs (Couche 1).
        marginal_contributions: Contributions marginales par cellule (Couche 1).
        euler_contributions: Decomposition Euler 10 cellules (Couche 2).
        factor_attribution: Attribution factorielle 5 vars x 2 canaux (Couche 2).
        tipping_points: Seuils de basculement par secteur (Couche 3).
        rst_result: Scenario de rupture minimal (Couche 3).
        rst_distance: Distance RST en sigma (Couche 3).
        regime: Classification de regime (Couche 4).
        trajectories: Projections T+3/6/9/12 (Couche 5).
        risk_appetite_matrix: Feux tricolores par cellule (Couche 5).
        early_warning: Indicateurs d'alerte precoce (Couche 5).
        recommendations: Liste de recommandations (Synthese).
        narrative: Synthese narrative CRO (Synthese).
        pass_number: Numero de passe (1 ou 2).
    """

    # Couche 1 — Croisement
    asymmetry_matrix: Optional[pd.DataFrame] = None
    marginal_contributions: Optional[pd.DataFrame] = None

    # Couche 2 — Decomposition Euler
    euler_contributions: Optional[pd.DataFrame] = None
    factor_attribution: Optional[pd.DataFrame] = None

    # Couche 3 — Seuils & RST
    tipping_points: Optional[pd.DataFrame] = None
    rst_result: Optional[Dict[str, float]] = None
    rst_distance: float = 0.0

    # Couche 4 — Regime
    regime: Optional[RegimeClassification] = None

    # Couche 5 — Prospective
    trajectories: Optional[pd.DataFrame] = None
    risk_appetite_matrix: Optional[pd.DataFrame] = None
    early_warning: Optional[pd.DataFrame] = None

    # Synthese
    recommendations: List[Recommendation] = field(default_factory=list)
    narrative: str = ""
    pass_number: int = 1
