"""Configuration — scenarios macro, trajectoires, covariance."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


# ──────────────────────────────────────────────
# VARIABLES MACROECONOMIQUES
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class MacroScenario:
    """Parametres d'un scenario macroeconomique pour le calcul ECL.

    Les 3 scenarios ECL (Base, Adverse, Favorable) sont ponderes pour
    produire un ECL pondere conforme IFRS 9 par. 5.5.17.

    Attributes:
        name: Nom du scenario.
        weight: Ponderation dans le calcul ECL (somme des 3 = 1.0).
        gdp_growth: Croissance du PIB annualisee (%).
        unemployment_rate: Taux de chomage (%).
        interest_rate: Taux directeur BCE (%).
        hpi_growth: Variation annuelle des prix immobiliers (%).
        inflation_rate: Inflation annuelle IPC (%).
        gdp_shock: Choc PIB applique au defaut.
        unemployment_shock: Choc chomage applique au defaut.
        interest_rate_shock: Choc taux directeur applique au defaut.
        hpi_shock: Choc prix immobiliers applique a la LGD.
        inflation_shock: Choc inflation applique au defaut.
    """

    name: str
    weight: float
    gdp_growth: float
    unemployment_rate: float
    interest_rate: float
    hpi_growth: float
    inflation_rate: float
    gdp_shock: float
    unemployment_shock: float
    interest_rate_shock: float
    hpi_shock: float
    inflation_shock: float
    # Champs climatiques (optionnels, default 0.0 = pas de choc climatique)
    carbon_price_shock: float = 0.0    # choc normalise prix carbone (1.0 = doublement)
    physical_severity: float = 0.0     # severite risque physique [0-1]


# --- Scenarios ECL (ponderes 50/25/25) ---

SCENARIO_BASE = MacroScenario(
    name="Base",
    weight=0.50,
    gdp_growth=1.2,
    unemployment_rate=7.5,
    interest_rate=3.5,
    hpi_growth=2.0,
    inflation_rate=2.5,
    gdp_shock=0.0,
    unemployment_shock=0.0,
    interest_rate_shock=0.0,
    hpi_shock=0.0,
    inflation_shock=0.0,
)

SCENARIO_ADVERSE = MacroScenario(
    name="Adverse",
    weight=0.25,
    gdp_growth=-1.5,
    unemployment_rate=10.5,
    interest_rate=5.0,
    hpi_growth=-8.0,
    inflation_rate=5.5,
    gdp_shock=0.03,
    unemployment_shock=0.05,
    interest_rate_shock=0.02,
    hpi_shock=0.04,
    inflation_shock=0.02,
)

SCENARIO_FAVORABLE = MacroScenario(
    name="Favorable",
    weight=0.25,
    gdp_growth=2.5,
    unemployment_rate=5.5,
    interest_rate=2.0,
    hpi_growth=5.0,
    inflation_rate=1.5,
    gdp_shock=-0.01,
    unemployment_shock=-0.02,
    interest_rate_shock=-0.01,
    hpi_shock=-0.02,
    inflation_shock=-0.01,
)

ECL_SCENARIOS: List[MacroScenario] = [SCENARIO_BASE, SCENARIO_ADVERSE, SCENARIO_FAVORABLE]

# Alias de compatibilite — les modules existants importent SCENARIOS
SCENARIOS: List[MacroScenario] = ECL_SCENARIOS

# --- Scenarios predefinis pour le dropdown dashboard (FR34) ---
# Valeurs des 5 sliders : interest_rate_bp, unemployment_bipolar, gdp_pct, hpi_pct, inflation_pct
# Le dropdown pre-remplit les sliders ; l'analyste peut ensuite ajuster a la main.

# Convention bipolaire chomage :
#   signe = nature (+  rupture techno, - = crise eco)
#   |valeur| = amplitude de hausse du chomage en pp vs base
#   Les deux extremes augmentent le taux de chomage : base + |slider|.
#
# Calibration : chaque scenario est ancre sur un evenement historique reel
# de la zone euro, avec donnees Eurostat / BCE / BRI.
PREDEFINED_SCENARIOS: Dict[str, Dict[str, float]] = {
    "Central": {
        # Baseline BCE 2024 : croissance tendancielle, inflation cible, taux neutre.
        # interest_rate_bp = 0 : Central = SCENARIO_BASE exact (pas de delta taux).
        "interest_rate_bp": 0.0,
        "unemployment_bipolar": 0.0,
        "gdp_pct": 1.2,
        "hpi_pct": 2.0,
        "inflation_pct": 2.5,
    },
    "Crise financiere (GFC)": {
        # Lehman / subprimes 2008-09 : PIB zone euro -4.5%, chomage 7.6→9.6%,
        # BCE baisse taux de 4.25% a 1.0% (-325bp mais baseline=3.5% → -250bp),
        # HPI zone euro -3%, deflation (HICP 0.3%).
        # Sources : Eurostat, BCE SDW, World Bank.
        "interest_rate_bp": -250.0,
        "unemployment_bipolar": -2.0,   # crise eco : +2pp chomage
        "gdp_pct": -4.5,
        "hpi_pct": -3.0,
        "inflation_pct": 0.3,
    },
    "Crise souveraine (2012)": {
        # Crise dette souveraine PIIGS 2011-12 : double-dip, PIB -0.9%,
        # chomage pic 12% (vs 7.5% base → +4pp), BCE baisse a 0.75%,
        # HPI -2.5%, inflation ancree 2.5%. Recession longue et diffuse.
        # Sources : Eurostat, BCE, Wikipedia European debt crisis.
        "interest_rate_bp": -275.0,
        "unemployment_bipolar": -4.0,   # crise eco : +4pp chomage (pic historique)
        "gdp_pct": -0.9,
        "hpi_pct": -2.5,
        "inflation_pct": 2.5,
    },
    "Stagflation": {
        # Choc petrolier 1974-75 (transpose en zone euro moderne) :
        # PIB -1%, chomage +3pp, banque centrale forcee de monter les taux
        # malgre la recession (lutte anti-inflation), HPI -5% reel,
        # inflation 8% (plafond credible en regime de ciblage moderne,
        # vs 13.2% historique 1974). Borne sup realisee : HICP 8.4% en 2022.
        # Sources : BCE Monthly Bulletin, OCDE, Eurostat.
        "interest_rate_bp": 250.0,
        "unemployment_bipolar": -3.0,   # crise eco : +3pp chomage
        "gdp_pct": -1.0,
        "hpi_pct": -5.0,
        "inflation_pct": 8.0,
    },
    "Choc pandemique (COVID)": {
        # COVID-19 T2 2020 annualise : PIB zone euro -6.1% (pire que GFC),
        # chomage officiel +0.5pp seulement (masque par SURE/kurzarbeit),
        # BCE a 0% (PEPP 1 850 Md EUR), HPI paradoxalement +5% (teletravail,
        # taux zero, stimulus fiscal), quasi-deflation HICP 0.3%.
        # Sources : Eurostat, BCE, World Bank.
        "interest_rate_bp": -350.0,
        "unemployment_bipolar": -0.5,   # crise eco : +0.5pp (masque par dispositifs)
        "gdp_pct": -6.0,
        "hpi_pct": 5.0,
        "inflation_pct": 0.3,
    },
    "Rupture techno": {
        # Dot-com bust 2001-03 + analogie IA : PIB zone euro +0.9% (ralentissement
        # sans recession), chomage +1.5pp concentre sur secteur tech,
        # BCE baisse taux a 2% (-100bp), immobilier non affecte (+3%),
        # inflation stable 2.2%. PE tech beneficie (destruction creatrice).
        # Sources : Eurostat, BCE, OCDE.
        "interest_rate_bp": -100.0,
        "unemployment_bipolar": 1.5,    # rupture techno : +1.5pp, PE tech beneficie
        "gdp_pct": 0.9,
        "hpi_pct": 3.0,
        "inflation_pct": 2.2,
    },
    "Reprise": {
        # Expansion zone euro 2017-18 : PIB +2.5%, chomage en baisse (pas de hausse),
        # BCE ultra-accommodante a 0% (QE toujours actif), HPI +4.5%,
        # inflation 1.6% sous-cible — le « Goldilocks » europeen.
        # Sources : Eurostat, FMI REO Europe 2017, BCE.
        "interest_rate_bp": -200.0,
        "unemployment_bipolar": 0.0,    # reprise : pas de hausse de chomage
        "gdp_pct": 2.5,
        "hpi_pct": 4.5,
        "inflation_pct": 1.6,
    },
    "Hypercroissance": {
        # Trente Glorieuses 1950-73 (transpose en zone euro moderne) :
        # PIB +5% (France +5.8%, Allemagne +6.0%, moyenne CEE ~5%),
        # plein emploi (chomage 1-3%, pas de hausse vs baseline),
        # taux d'interet eleves 8% (rendements obligataires 7-9%, taux directeurs
        # Bundesbank ~5-7%, marche exige prime sur la croissance),
        # immobilier +10% (urbanisation massive, baby-boom, investissement logement),
        # inflation moderee 4% (« repression financiere » typique de l'epoque).
        # Scenario ideal pour comparer rentabilite PE vs credit en regime de forte
        # croissance : PE beneficie du levier de croissance, credit du volume.
        # Sources : INSEE, Bundesbank, Maddison Project Database, OCDE.
        "interest_rate_bp": 450.0,
        "unemployment_bipolar": 0.0,    # plein emploi, pas de hausse du chomage
        "gdp_pct": 5.0,
        "hpi_pct": 10.0,
        "inflation_pct": 4.0,
    },
    "Boom immobilier": {
        # Expansion immobiliere Europe 2015-19 / post-COVID 2021-22.
        # HPI forte hausse, taux moderes, inflation cible, economie stable.
        # Scenario de surchauffe immobiliere sans recession.
        # Sources : Eurostat HPI, BCE, OCDE.
        "interest_rate_bp": 0.0,
        "unemployment_bipolar": 0.0,
        "gdp_pct": 1.8,
        "hpi_pct": 8.0,
        "inflation_pct": 2.5,
    },
    "Trappe a liquidite": {
        # Japon 1995-2015 / zone euro 2014-16. Taux zero, deflation,
        # stagnation economique persistante. HPI en recul, PIB nul.
        # Sources : BoJ, BCE, FMI Japan Article IV.
        "interest_rate_bp": -350.0,
        "unemployment_bipolar": -2.5,   # crise eco : +2.5pp chomage
        "gdp_pct": 0.0,
        "hpi_pct": -3.0,
        "inflation_pct": -0.5,
    },
    "Transition climatique brutale": {
        # NGFS Sudden Wake-Up Call : doublement brutal du prix carbone,
        # recession de transition, chomage sectoriel, HPI affecte (DPE),
        # inflation energetique. Premier scenario avec choc climatique.
        # Sources : NGFS Short-Term Scenarios (May 2025), CLIMACRED.
        "interest_rate_bp": 100.0,
        "unemployment_bipolar": -1.5,   # crise eco : +1.5pp chomage sectoriel
        "gdp_pct": -0.8,
        "hpi_pct": -4.0,
        "inflation_pct": 4.5,
        "carbon_price_shock": 1.5,      # prix carbone ×2.5 (normalise : 1.0 = doublement)
        "physical_severity": 0.3,       # risque physique modere
    },
}

# Volatilites annualisees historiques par variable macro (en points de pourcentage).
# Sources : BCE Statistical Data Warehouse, Eurostat, consensus Bloomberg (2010-2024).
_MACRO_VOLATILITIES: Dict[str, float] = {
    "unemployment_rate": 1.5,   # ecart-type annuel zone euro (~1-2 pp)
    "gdp_growth":       1.8,   # ecart-type annuel croissance PIB UE
    "interest_rate":    1.0,   # ecart-type taux directeur BCE
    "hpi_growth":       3.0,   # ecart-type prix immobilier (plus volatile)
    "inflation_rate":   1.2,   # ecart-type HICP zone euro
}

# Historique macro mensuel (baseline) — 60 mois (5 ans).
# Utilise UNIQUEMENT par le generateur de donnees (generator.py) pour
# fournir des niveaux macro courants. N'est PAS utilise pour estimer
# la matrice de covariance (celle-ci est construite via Expert Judgment).
def _generate_macro_history(n_months: int = 60, seed: int = 42) -> Dict[str, List[float]]:
    """Genere un historique macro synthetique pour le generateur de donnees."""
    import numpy as _np
    rng = _np.random.RandomState(seed)
    _params = {
        "gdp_growth": (1.2, 0.4, 0.7),
        "unemployment_rate": (7.5, 0.3, 0.9),
        "interest_rate": (3.25, 0.25, 0.95),
        "hpi_growth": (2.2, 0.5, 0.8),
        "inflation_rate": (2.6, 0.3, 0.85),
    }
    history: Dict[str, List[float]] = {}
    for var, (mu, sigma, phi) in _params.items():
        series = [mu]
        for _ in range(n_months - 1):
            x_next = mu + phi * (series[-1] - mu) + sigma * rng.randn()
            series.append(round(x_next, 4))
        history[var] = series
    return history


MACRO_HISTORY_BASELINE: Dict[str, List[float]] = _generate_macro_history(60)

# Equilibre structurel long-terme (theta O-U).
# RJ audit MEDIUM : theta etait = SCENARIO_BASE, creant une circularite
# (le scenario de base etant une conjoncture, pas un equilibre structurel).
# Les valeurs ci-dessous representent l'equilibre de long terme (10+ ans)
# independant de la conjoncture actuelle (SCENARIO_BASE).
# Sources : consensus OCDE/FMI long-terme, potentiel de croissance UE.
MACRO_STRUCTURAL_EQUILIBRIUM: Dict[str, float] = {
    "unemployment_rate": 6.5,    # NAIRU zone euro (estimation structurelle)
    "gdp_growth": 1.5,          # Croissance potentielle UE long-terme
    "interest_rate": 2.5,       # Taux neutre r* (Laubach-Williams)
    "hpi_growth": 2.0,          # Inflation + productivite immobiliere
    "inflation_rate": 2.0,      # Cible BCE
}

# Mean-reversion Ornstein-Uhlenbeck par variable (H10).
# kappa = vitesse de retour vers theta, sigma = volatilite annualisee.
# theta = equilibre structurel long-terme (NAIRU, r*, cible BCE).
# x(t+1) = x(t) + kappa×(theta-x(t))×dt + sigma×sqrt(dt)×epsilon
MACRO_MEAN_REVERSION: Dict[str, Dict[str, float]] = {
    "unemployment_rate": {"kappa": 0.5, "sigma": 0.8, "theta": MACRO_STRUCTURAL_EQUILIBRIUM["unemployment_rate"]},
    "gdp_growth": {"kappa": 1.0, "sigma": 1.2, "theta": MACRO_STRUCTURAL_EQUILIBRIUM["gdp_growth"]},
    "interest_rate": {"kappa": 0.3, "sigma": 0.5, "theta": MACRO_STRUCTURAL_EQUILIBRIUM["interest_rate"]},
    "hpi_growth": {"kappa": 0.7, "sigma": 2.0, "theta": MACRO_STRUCTURAL_EQUILIBRIUM["hpi_growth"]},
    "inflation_rate": {"kappa": 0.8, "sigma": 0.6, "theta": MACRO_STRUCTURAL_EQUILIBRIUM["inflation_rate"]},
}

# Matrice de covariance macro 5×5 (H6, RST Mahalanobis).
# RJ audit v2 : construite a partir d'une matrice de correlation de consensus
# (Expert Judgment / litterature macro-finance) et des volatilites historiques.
# Pas d'estimation sur donnees synthetiques (circularite : les AR(1) independants
# produisent une covariance diagonale, rendant la re-estimation inutile).
# Formule : Σ_ij = ρ_ij × σ_i × σ_j
# Regularisation Ledoit-Wolf (2004) appliquee pour garantir la definie-positivite.
# Ordre : unemployment_rate, gdp_growth, interest_rate, hpi_growth, inflation_rate

# Matrice de correlation de consensus (Expert Judgment).
# Sources : FMI WEO correlations (2000-2024), BCE research papers, Okun's law,
# Phillips curve, Taylor rule relationships.
#                unemp   gdp     ir      hpi     infl
_EXPERT_CORR = [
    [ 1.00, -0.70,  0.15, -0.40,  0.20],  # unemployment
    [-0.70,  1.00, -0.10,  0.50, -0.15],  # gdp_growth
    [ 0.15, -0.10,  1.00, -0.25,  0.60],  # interest_rate
    [-0.40,  0.50, -0.25,  1.00, -0.10],  # hpi_growth
    [ 0.20, -0.15,  0.60, -0.10,  1.00],  # inflation_rate
]

def _build_macro_covariance() -> "list[list[float]]":
    """Construit Σ = diag(σ) × R_expert × diag(σ), regularisee Ledoit-Wolf."""
    import numpy as _np
    from sklearn.covariance import LedoitWolf
    _vars = ["unemployment_rate", "gdp_growth", "interest_rate", "hpi_growth", "inflation_rate"]
    sigma = _np.array([_MACRO_VOLATILITIES[v] for v in _vars])
    R = _np.array(_EXPERT_CORR)
    # Σ_ij = σ_i × ρ_ij × σ_j
    S = _np.outer(sigma, sigma) * R
    # Regularisation Ledoit-Wolf pour garantir definie-positivite numerique
    lw = LedoitWolf(assume_centered=True)
    lw.location_ = _np.zeros(len(_vars))
    lw.covariance_ = S
    # Shrinkage vers cible diagonale
    target = _np.diag(_np.diag(S))
    alpha = 0.1  # retractation legere (matrice source deja bien conditionnee)
    S_shrunk = (1 - alpha) * S + alpha * target
    return S_shrunk.tolist()


MACRO_COVARIANCE: list[list[float]] = _build_macro_covariance()
MACRO_VARIABLES_ORDER: Tuple[str, ...] = (
    "unemployment_rate", "gdp_growth", "interest_rate", "hpi_growth", "inflation_rate",
)
