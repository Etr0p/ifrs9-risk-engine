"""Configuration centrale du Cockpit IFRS 9.

Regroupe tous les hyperparametres, constantes metier, seuils et
parametres de scenarios utilises a travers le projet.

Source unique de verite — tous les modules importent depuis ce fichier.
Aucun seuil ou hyperparametre ne doit etre hardcode ailleurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# REPRODUCTIBILITE
# ──────────────────────────────────────────────
RANDOM_SEED: int = 42

# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
N_CLIENTS: int = 30_000
N_MONTHS: int = 12
TRAIN_RATIO: float = 0.7
VALIDATION_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# ──────────────────────────────────────────────
# SECTEURS ENTREPRISE (5 secteurs, double canal)
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class SectorConfig:
    """Configuration d'un secteur entreprise avec sensibilites duales.

    Chaque secteur possede 10 sensibilites macro (5 variables x 2 canaux)
    representant des multiplicateurs d'impact sur le risque.

    Canal credit : impact sur la PD et la LGD (depreciation IFRS 9).
    Canal PE : impact sur la NAV et les multiples (juste valeur IFRS 13).

    Convention de signe des sensibilites :
        > 0 : choc adverse augmente le risque du canal.
        < 0 : choc adverse diminue le risque (effet asymetrique).
        |valeur| > 1 : amplifie par rapport a la reference.

    Attributes:
        name: Nom du secteur (Technologie, Industrie, Sante, Immobilier, Services).
        proportion: Part dans le portefeuille (somme des 5 secteurs = 1.0).
        base_default_rate: Taux de defaut de base du secteur.
        unemployment_sensitivity_credit: Sensibilite du canal credit au chomage.
        gdp_sensitivity_credit: Sensibilite du canal credit au PIB.
        interest_rate_sensitivity_credit: Sensibilite du canal credit au taux BCE.
        hpi_sensitivity_credit: Sensibilite du canal credit aux prix immobiliers.
        inflation_sensitivity_credit: Sensibilite du canal credit a l'inflation.
        unemployment_sensitivity_pe: Sensibilite du canal PE au chomage.
        gdp_sensitivity_pe: Sensibilite du canal PE au PIB.
        interest_rate_sensitivity_pe: Sensibilite du canal PE au taux BCE.
        hpi_sensitivity_pe: Sensibilite du canal PE aux prix immobiliers.
        inflation_sensitivity_pe: Sensibilite du canal PE a l'inflation.
        valuation_method: Methode IPEV de valorisation PE du secteur.
        entry_multiple_range: Fourchette (min, max) des multiples d'entree.
        exit_multiple_base: Multiple de sortie baseline.
        revenue_range_m: Fourchette (min, max) du chiffre d'affaires en millions EUR.
        ebitda_margin_range: Fourchette (min, max) de la marge EBITDA en ratio [0,1].
        rho_lgd_cycle: Correlation LGD-cycle pour le calcul LGD Downturn (EBA GL/2019/03).
            LGD_DT = LGD_TTC × (1 + rho_lgd_cycle × |Z_stress|).
    """

    name: str
    proportion: float
    base_default_rate: float

    # 5 sensibilites macro — canal CREDIT
    unemployment_sensitivity_credit: float
    gdp_sensitivity_credit: float
    interest_rate_sensitivity_credit: float
    hpi_sensitivity_credit: float
    inflation_sensitivity_credit: float

    # 5 sensibilites macro — canal PE
    unemployment_sensitivity_pe: float
    gdp_sensitivity_pe: float
    interest_rate_sensitivity_pe: float
    hpi_sensitivity_pe: float
    inflation_sensitivity_pe: float

    # PE — Valorisation IPEV
    valuation_method: str
    entry_multiple_range: Tuple[float, float]
    exit_multiple_base: float

    # Correlation LGD-cycle (EBA GL/2019/03) — pour LGD Downturn
    rho_lgd_cycle: float

    # ESG — Green Asset Ratio declaratif (placeholder GAR, BCE 2024)
    # Part estimee des actifs alignes taxonomie EU par secteur.
    # Source : consensus sectoriel simplifie (declaratif, non audite).
    green_share: float

    # Fourchettes de generation pour le portefeuille synthetique
    revenue_range_m: Tuple[float, float]
    ebitda_margin_range: Tuple[float, float]


SECTORS: List[SectorConfig] = [
    # ── Technologie (25%) ──
    # Poche de vulnerabilite intentionnelle : sensibilite credit tres elevee
    # au chomage (2.5x), mais PE tech beneficie de la rupture technologique
    # (sens PE chomage = -1.5, signe negatif = effet inverse).
    # Methode IPEV : EV/Revenue (standard SaaS/Tech, IPEV 2025).
    SectorConfig(
        name="Technologie",
        proportion=0.25,
        base_default_rate=0.06,
        # Credit : fragile (startups, cash-burn, levier)
        unemployment_sensitivity_credit=2.5,
        gdp_sensitivity_credit=1.8,
        interest_rate_sensitivity_credit=1.2,
        hpi_sensitivity_credit=0.5,
        inflation_sensitivity_credit=1.5,
        # PE : multiples tres sensibles aux taux (WACC), mais chomage tech = benefique
        unemployment_sensitivity_pe=-1.5,
        gdp_sensitivity_pe=1.0,
        interest_rate_sensitivity_pe=2.5,
        hpi_sensitivity_pe=0.3,
        inflation_sensitivity_pe=0.8,
        valuation_method="EV/Revenue",
        entry_multiple_range=(4.0, 8.0),
        # Exit > entry : creation de valeur PE (croissance CA + expansion multiples)
        # MOIC cible ~1.6x sur 4-5 ans = IRR ~12% (median Preqin 2023 tech buyout)
        exit_multiple_base=9.0,
        green_share=0.40,
        rho_lgd_cycle=0.25,
        revenue_range_m=(5.0, 200.0),
        ebitda_margin_range=(0.05, 0.30),
    ),
    # ── Industrie (25%) ──
    # Cyclique, tres sensible au PIB. Credit et PE souffrent ensemble
    # en recession classique. Methode IPEV : EV/EBITDA mid-market.
    SectorConfig(
        name="Industrie",
        proportion=0.25,
        base_default_rate=0.05,
        # Credit : cyclique, levier operationnel eleve
        unemployment_sensitivity_credit=1.5,
        gdp_sensitivity_credit=2.0,
        interest_rate_sensitivity_credit=1.0,
        hpi_sensitivity_credit=0.8,
        inflation_sensitivity_credit=1.2,
        # PE : EBITDA et multiples sensibles au cycle
        unemployment_sensitivity_pe=1.0,
        gdp_sensitivity_pe=2.5,
        interest_rate_sensitivity_pe=1.5,
        hpi_sensitivity_pe=0.5,
        inflation_sensitivity_pe=1.0,
        valuation_method="EV/EBITDA",
        entry_multiple_range=(4.0, 8.0),
        # MOIC cible ~1.5x (median Preqin 2023 industrial buyout)
        exit_multiple_base=8.5,
        green_share=0.20,
        rho_lgd_cycle=0.30,
        revenue_range_m=(10.0, 500.0),
        ebitda_margin_range=(0.08, 0.20),
    ),
    # ── Sante (15%) ──
    # Defensif : faibles sensibilites macro (demande inelastique).
    # Multiples eleves (10-15x) refletant le premium defensif.
    SectorConfig(
        name="Sante",
        proportion=0.15,
        base_default_rate=0.03,
        # Credit : tres resilient
        unemployment_sensitivity_credit=0.5,
        gdp_sensitivity_credit=0.4,
        interest_rate_sensitivity_credit=0.6,
        hpi_sensitivity_credit=0.3,
        inflation_sensitivity_credit=0.8,
        # PE : multiples stables, risque principalement reglementaire (hors macro)
        unemployment_sensitivity_pe=0.3,
        gdp_sensitivity_pe=0.5,
        interest_rate_sensitivity_pe=0.5,
        hpi_sensitivity_pe=0.2,
        inflation_sensitivity_pe=0.5,
        valuation_method="EV/EBITDA",
        entry_multiple_range=(10.0, 15.0),
        # MOIC cible ~1.4x (healthcare = premium defensif, expansion moderee)
        exit_multiple_base=17.0,
        green_share=0.30,
        rho_lgd_cycle=0.10,
        revenue_range_m=(5.0, 300.0),
        ebitda_margin_range=(0.10, 0.25),
    ),
    # ── Immobilier (20%) ──
    # Tres sensible aux taux (LTV, charges) et au HPI (collateral).
    # Credit : LGD directement impactee par le HPI via le canal collateral.
    # PE : cap rate s'expand quand les taux montent -> compression NAV.
    SectorConfig(
        name="Immobilier",
        proportion=0.20,
        base_default_rate=0.05,
        # Credit : LTV eleve, charges de taux, collateral immobilier
        unemployment_sensitivity_credit=1.0,
        gdp_sensitivity_credit=1.2,
        interest_rate_sensitivity_credit=2.0,
        hpi_sensitivity_credit=2.5,
        inflation_sensitivity_credit=1.0,
        # PE : cap rate = f(taux), NAV = NOI / cap rate
        unemployment_sensitivity_pe=0.8,
        gdp_sensitivity_pe=1.0,
        interest_rate_sensitivity_pe=2.0,
        hpi_sensitivity_pe=3.0,
        inflation_sensitivity_pe=0.5,
        valuation_method="Cap_rate/NOI",
        entry_multiple_range=(4.0, 7.0),
        # MOIC cible ~1.5x (value-add real estate, Preqin 2023)
        exit_multiple_base=8.0,
        green_share=0.15,
        rho_lgd_cycle=0.35,
        revenue_range_m=(2.0, 100.0),
        ebitda_margin_range=(0.40, 0.70),
    ),
    # ── Services (15%) ──
    # Sensibilite moderee au cycle, mais tres expose a l'inflation
    # (masse salariale = cout principal -> compression marges).
    SectorConfig(
        name="Services",
        proportion=0.15,
        base_default_rate=0.04,
        # Credit : charges salariales, sensible a l'inflation
        unemployment_sensitivity_credit=1.2,
        gdp_sensitivity_credit=1.0,
        interest_rate_sensitivity_credit=0.8,
        hpi_sensitivity_credit=0.5,
        inflation_sensitivity_credit=1.8,
        # PE : marge comprimee par l'inflation salariale
        unemployment_sensitivity_pe=1.0,
        gdp_sensitivity_pe=1.2,
        interest_rate_sensitivity_pe=1.0,
        hpi_sensitivity_pe=0.3,
        inflation_sensitivity_pe=2.0,
        valuation_method="EV/EBITDA",
        entry_multiple_range=(6.0, 10.0),
        # MOIC cible ~1.4x (services = expansion moderee)
        exit_multiple_base=11.0,
        green_share=0.25,
        rho_lgd_cycle=0.20,
        revenue_range_m=(3.0, 150.0),
        ebitda_margin_range=(0.08, 0.18),
    ),
]

SECTOR_NAMES: Tuple[str, ...] = tuple(s.name for s in SECTORS)

# Alias de compatibilite — les modules analytics importent SEGMENTS
SEGMENTS: List[SectorConfig] = SECTORS

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
        "interest_rate_bp": 50.0,
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

# ──────────────────────────────────────────────
# MODELES PD
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class PDModelConfig:
    """Configuration des modeles de Probabilite de Defaut.

    3 familles genuinement differentes :
        - LR_WoE : lineaire (frontiere convexe, interpretable)
        - TabNet : deep learning tabulaire (attention sequentielle, Sparsemax)
        - XGBoost : ensemble d'arbres (frontiere en escalier)

    Attributes:
        n_woe_bins: Nombre de bins pour le WoE binning.
        lr_C: Regularisation Logistic Regression.
        lr_max_iter: Iterations max LR.
        tabnet_n_d: Largeur couche decision TabNet (64 = ~120k params).
        tabnet_n_a: Largeur couche attention TabNet (64 = Gold Standard).
        tabnet_n_steps: Etapes d'attention sequentielle.
        tabnet_gamma: Coefficient relaxation reutilisation features.
        tabnet_lambda_sparse: Penalite sparsite.
        tabnet_lr: Learning rate Adam.
        tabnet_batch_size: Taille batch (8192 pour 1M lignes).
        tabnet_virtual_batch_size: Ghost BN virtual batch.
        tabnet_max_epochs: Epoques max (200 avec patience 20).
        tabnet_patience: Early stopping patience.
        xgb_n_estimators: Nombre d'arbres XGBoost.
        xgb_max_depth: Profondeur max XGB.
        xgb_learning_rate: Learning rate XGB.
        xgb_subsample: Sous-echantillonnage XGB.
        calibration_method: Methode de calibration ('isotonic' ou 'sigmoid').
        iv_min_threshold: Seuil IV minimum pour selection des features (0.02 = non predictif).
        vif_max_threshold: Seuil VIF maximum pour filtrage multicolinearite (5.0 = standard).
            VIF > 5 indique une multicolinearite forte. Applique avant la contrainte beta < 0.
        woe_epsilon: Lissage Laplace pour WoE (evite ln(0) si bin vide).
        min_events_per_bin: Nombre minimum de defauts par bin WoE (robustesse statistique).
        pdo: Points to Double the Odds pour le scaling scorecard.
        target_score: Score cible au point d'ancrage (odds = target_odds).
        target_odds: Ratio de cotes au target_score.
    """

    n_woe_bins: int = 10
    lr_C: float = 1.0
    lr_max_iter: int = 1000
    # --- TabNet Gold Standard (1M lignes / ~120k params) ---
    tabnet_n_d: int = 64
    tabnet_n_a: int = 64
    tabnet_n_steps: int = 5
    tabnet_gamma: float = 1.3
    tabnet_lambda_sparse: float = 1e-3
    tabnet_lr: float = 0.02
    tabnet_batch_size: int = 8192
    tabnet_virtual_batch_size: int = 512
    tabnet_max_epochs: int = 200
    tabnet_patience: int = 20
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    calibration_method: str = "isotonic"
    iv_min_threshold: float = 0.02
    vif_max_threshold: float = 5.0
    woe_epsilon: float = 0.5
    min_events_per_bin: int = 20
    pdo: int = 20
    target_score: int = 600
    target_odds: float = 50.0


PD_CONFIG = PDModelConfig()

# ──────────────────────────────────────────────
# MODELE LGD
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class LGDConfig:
    """Configuration du modele Loss Given Default.

    Attributes:
        lgd_ttc_mean: LGD Through-The-Cycle moyenne.
        lgd_ttc_std: Ecart-type LGD TTC.
        downturn_add_on: Add-on pour LGD Downturn.
        recovery_rate_floor: Plancher du taux de recouvrement.
        cure_rate: Taux de cure (sortie de defaut).
    """

    lgd_ttc_mean: float = 0.35
    lgd_ttc_std: float = 0.15
    downturn_add_on: float = 0.10
    recovery_rate_floor: float = 0.05
    cure_rate: float = 0.15


LGD_CONFIG = LGDConfig()

# ──────────────────────────────────────────────
# MODELE EAD
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class EADConfig:
    """Configuration du modele Exposure At Default.

    Attributes:
        ccf_revolving: Credit Conversion Factor pour lignes revolving.
        ccf_term_loan: CCF pour prets a terme.
        utilization_draw_stress: Stress sur le tirage en cas de defaut.
    """

    ccf_revolving: float = 0.75
    ccf_term_loan: float = 1.0
    utilization_draw_stress: float = 0.20


EAD_CONFIG = EADConfig()

# ──────────────────────────────────────────────
# MOTEUR IFRS 9
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class IFRS9Config:
    """Configuration du moteur IFRS 9.

    Attributes:
        sicr_threshold_multiplier: Multiplicateur PD pour declenchement SICR.
        stage3_dpd_threshold: Jours de retard pour passage en Stage 3.
        stage3_pd_threshold: PD seuil pour passage en Stage 3.
        discount_rate: Taux d'actualisation annuel (EIR proxy).
        lifetime_horizon_years: Horizon lifetime pour Stage 2/3 (annees).
    """

    sicr_threshold_multiplier: float = 2.0
    stage3_dpd_threshold: int = 90
    stage3_pd_threshold: float = 0.30
    discount_rate: float = 0.045
    lifetime_horizon_years: int = 3


IFRS9_CONFIG = IFRS9Config()


@dataclass(frozen=True)
class SICRConfig:
    """Configuration SICR multi-facteurs (IFRS 9 §B5.5.17).

    Score composite pour le declenchement Stage 2 :
        SICR_score = w_pd_ratio × (PD_current / PD_origination - 1)
                   + w_pd_delta × max(0, PD_current - PD_origination)
                   + w_dpd × (DPD / 30)
                   + w_macro × macro_z_score
        Stage 2 si SICR_score > threshold.

    Attributes:
        w_pd_ratio: Poids du ratio PD relatif (facteur le plus predictif).
        w_pd_delta: Poids du delta PD absolu.
        w_dpd: Poids du retard de paiement normalise.
        w_macro: Poids du Z-score macro (forward-looking).
        threshold: Seuil de declenchement SICR (calibre pour ~10% Stage 2).
    """

    w_pd_ratio: float = 0.30
    w_pd_delta: float = 0.30
    w_dpd: float = 0.25
    w_macro: float = 0.15
    threshold: float = 0.70


SICR_CONFIG = SICRConfig()

# ──────────────────────────────────────────────
# CALIBRATION LOGIT (MERTON-VASICEK)
# ──────────────────────────────────────────────

# Amplitude de transmission macro → logit(PD).
#
# Fondement theorique : dans le modele Vasicek a facteur unique,
# PD_stressed = Phi(Phi^-1(PD) + sqrt(rho) * z_macro).
# Pour la correlation d'actif corporate rho ≈ 0.20 (Bale II mid-point),
# un stress 2-sigma donne un shift logit ≈ 0.89.
#
# RJ audit MEDIUM : Λ=6.0 etait un nombre magique. Desormais calibre via
# _calibrate_logit_amplitude() a partir de 2 points d'ancrage EBA :
#   PD_base=0.06, PD_adverse=0.18 (×3), composite_shock=0.25.
# Formule : Λ = (logit(PD_adv) - logit(PD_base)) / composite_shock.
def _calibrate_logit_amplitude(
    pd_base: float = 0.06,
    pd_adverse: float = 0.18,
    composite_shock: float = 0.25,
) -> float:
    """Calibre Λ depuis 2 points d'ancrage EBA (stress test adverse).

    Args:
        pd_base: PD corporate baseline (defaut EBA mid-point).
        pd_adverse: PD corporate sous scenario adverse (×3 EBA).
        composite_shock: Choc macro composite moyen en scenario adverse.

    Returns:
        Λ tel que logit(PD_base) + Λ × composite_shock ≈ logit(PD_adverse).
    """
    import math
    logit_base = math.log(pd_base / (1 - pd_base))
    logit_adv = math.log(pd_adverse / (1 - pd_adverse))
    return (logit_adv - logit_base) / composite_shock


LOGIT_AMPLITUDE: float = _calibrate_logit_amplitude()  # ≈ 4.94, calibre sur PD 6%→18% (×3 EBA)

# Echelle logit pour P(distress) PE.
# Plus faible que credit car les PE sont en equity (junior tranche)
# et les sensibilites PE sont deja plus elevees dans SectorConfig.
PE_DISTRESS_LOGIT_SCALE: float = 3.0

# Ratio charges d'exploitation pour convertir EBITDA en NOI (Net Operating Income)
# pour la methode Cap_rate/NOI en immobilier.
# L'EBITDA surestime le NOI car il inclut frais de gestion et charges non-operationnelles.
# Source : benchmarks CBRE/JLL (2023), OPEX commercial RE = 12-18% du revenu brut.
NOI_OPEX_RATIO: float = 0.15

# Correlation intra-sectorielle pour le bruit de dispersion NAV.
# Modele a facteur : ε_i = 1 + sqrt(ρ)·σ·Z_secteur + sqrt(1-ρ)·σ·Z_idio
# Positions du meme secteur partagent un facteur commun (meme vintage, meme GP).
# Source : Preqin (2023), correlation intra-fonds vintage estimee 0.3-0.5.
PE_NOISE_INTRA_SECTOR_CORR: float = 0.40

# ──────────────────────────────────────────────
# CONTRAINTES BALE III / CRR3
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class BaselConfig:
    """Contraintes prudentielles Bale III et grille CRR3.

    Le risk weight PE n'est pas un 400% fixe : la CRR3 (jan. 2025)
    prevoit une grille graduee (190% IRB diversifie, 250% general equity,
    400% speculatif). L'optimiseur teste les 3 classifications.

    Ref: BCBS d424 (2017), CRR3 Art. 133 (SA equity RW), Art. 155 (IRB equity).

    Attributes:
        cet1_target: Ratio CET1 cible.
        rwa_budget: Budget RWA en EUR.
        leverage_max: Ratio de levier max Bale III.
        rw_credit: Risk Weight moyen credit corporate (SA).
        rw_pe_default: Risk Weight PE par defaut (general equity CRR3).
        rw_pe_options: Tuple des 3 classifications CRR3 testees.
        pe_max_allocation: Allocation maximale en PE (fraction du portefeuille).
        hhi_max: HHI maximum acceptable (seuil de concentration).
        min_sectors_above_5pct: Nombre minimum de secteurs > 5% (diversification).
        cir: Cost/Income Ratio pour le calcul RAROC complet.
        tax_rate: Taux d'imposition effectif pour le profit net RAROC.
        liquidity_premium_bps: Prime de liquidite en bps ajoutee au spread Merton.
        commercial_margin_bps: Marge commerciale bancaire en bps ajoutee au NII.
            En pratique, les banques facturent 1.5-2.5x le spread risk-neutral.
            150 bps est un proxy mid-market corporate (EBA 2023 benchmarks).
    """

    cet1_target: float = 0.13
    # Budget RWA calibre au portefeuille synthetique.
    # ~10K entreprises × ~2M EUR EAD moyen × RW 1.0 ≈ 20 Md EUR RWA total.
    # Budget 10 Md = 50% d'utilisation (marge d'optimisation).
    rwa_budget: float = 10_000_000_000.0
    leverage_max: float = 0.033
    rw_credit: float = 1.0
    rw_pe_default: int = 250
    rw_pe_options: Tuple[int, ...] = (190, 250, 400)
    pe_max_allocation: float = 0.40
    hhi_max: int = 2500
    min_sectors_above_5pct: int = 3
    # RAROC complet (H5) : CIR et impots
    cir: float = 0.45
    tax_rate: float = 0.25
    # Spread Merton (H4) : prime de liquidite
    liquidity_premium_bps: int = 50
    # Marge commerciale bancaire (NII = spread Merton + liquidite + marge)
    commercial_margin_bps: int = 150
    # Penalite de correlation pour l'optimiseur Softmax (RJ audit v3).
    # Score_i = RAROC_i - lambda * sum(w_j * rho_ij).
    # lambda > 0 penalise les secteurs correles aux autres.
    lambda_correlation: float = 0.5


BASEL_CONFIG = BaselConfig()

# ──────────────────────────────────────────────
# RISK APPETITE — SEUILS TRICOLORES
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class RiskAppetiteConfig:
    """Seuils de l'appetit au risque (cadre FSB 2013, feux tricolores).

    Convention : vert = acceptable, ambre = vigilance, rouge = breach.
    Les seuils definissent la frontiere vert/ambre et ambre/rouge.

    Attributes:
        ecl_ead_green: Seuil ECL/EAD vert (< seuil = vert).
        ecl_ead_amber: Seuil ECL/EAD ambre (< seuil = ambre, >= = rouge).
        raroc_green: Seuil RAROC vert (> seuil = vert).
        raroc_amber: Seuil RAROC ambre (> seuil = ambre, <= = rouge).
        hhi_green: Seuil HHI vert (< seuil = diversifie).
        hhi_amber: Seuil HHI ambre (< seuil = modere, >= = concentre).
        nav_drawdown_green: Seuil NAV drawdown PE vert.
        nav_drawdown_amber: Seuil NAV drawdown PE ambre.
    """

    ecl_ead_green: float = 0.020
    ecl_ead_amber: float = 0.040
    raroc_green: float = 0.04
    raroc_amber: float = 0.02
    hhi_green: int = 1500
    hhi_amber: int = 2500
    nav_drawdown_green: float = 0.10
    nav_drawdown_amber: float = 0.25
    # HHI Name Level (concentration par contrepartie, ICAAP Pilier 2)
    # Echelle 10 000. Seuil 50 = alerte concentration idiosyncratique.
    hhi_name_green: int = 30
    hhi_name_amber: int = 50


RISK_APPETITE_CONFIG = RiskAppetiteConfig()

# ──────────────────────────────────────────────
# CLASSIFICATION PE (Performing/Watchlist/Distressed)
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class PEClassificationConfig:
    """Seuils de classification des participations PE.

    Calibration sources :
        - distress thresholds : quartiles historiques de defaut PE.
          Preqin (2022) : ~8% des fonds en distress (Q3), ~25% en watchlist.
          Seuils 10%/30% sont conservateurs vs. benchmarks industriels.
        - lgd_equity : Moody's Recovery & LGD study (2023), equity tranche
          recovery rate ~40% en moyenne -> LGD = 60%.
        - secondary_discount : decote marche secondaire PE, Jefferies/Lazard
          (2023) : median 8-12% pour buyouts mid-market.
        - dlom_vintage_factor : DLOM (Discount for Lack of Marketability)
          ajuste par la maturite. Fonds jeunes = moins liquides.
          Lit. AICPA (2013), Pratt & Grabowski (2014) : DLOM 15-30% PE.
          Le facteur 0.50 donne une fourchette 10-15% (conservateur).

    Attributes:
        distress_threshold_performing: P(distress) max pour rester Performing.
        distress_threshold_watchlist: P(distress) max pour Watchlist (au-dela = Distressed).
        secondary_discount: Decote de marche secondaire de base (~10% buyout).
        lgd_equity: LGD sur les investissements en equity (perte en cas de distress).
        dlom_vintage_factor: Facteur multiplicatif pour l'ajustement DLOM vintage.
            DLOM_effectif = secondary_discount × (1 + factor × max(0, threshold - holding) / threshold)
        dlom_vintage_threshold: Seuil de maturite (annees) au-dela duquel pas de surcharge DLOM.
    """

    distress_threshold_performing: float = 0.10
    distress_threshold_watchlist: float = 0.30
    secondary_discount: float = 0.10
    lgd_equity: float = 0.60
    dlom_vintage_factor: float = 0.50
    dlom_vintage_threshold: float = 5.0


PE_CLASSIFICATION_CONFIG = PEClassificationConfig()

# ──────────────────────────────────────────────
# VIRTUAL CRO — SEUILS D'ALERTE
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class CROAlertConfig:
    """Seuils d'alerte pour le module Virtual CRO.

    Attributes:
        ecl_variation_alert: Variation ECL declenchant une alerte.
        stage2_warning_pct: Part Stage 2 declenchant un warning.
        psi_drift_threshold: PSI seuil pour alerte de drift.
        concentration_threshold: Seuil de concentration sectorielle.
    """

    ecl_variation_alert: float = 0.15
    stage2_warning_pct: float = 0.20
    psi_drift_threshold: float = 0.15
    concentration_threshold: float = 0.30


CRO_CONFIG = CROAlertConfig()

# ──────────────────────────────────────────────
# DASHBOARD
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class DashboardConfig:
    """Configuration du dashboard Streamlit.

    Palette Steel Blue sur fond sombre, conforme au design system
    institutional defini dans dashboard-ui-ux-research-2026-02-10.md.

    Slider ranges conformes au PRD FR33.

    Attributes:
        page_title: Titre de la page.
        page_icon: Icone de la page.
        layout: Layout Streamlit ('wide' ou 'centered').
        theme_primary: Couleur primaire hex (Steel Blue).
        theme_secondary: Couleur secondaire hex (Steel Blue 800).
        theme_accent: Couleur d'accent hex (Emerald).
        theme_bg_dark: Fond sombre hex.
        theme_bg_card: Fond de carte hex.
        theme_text: Couleur texte principal hex.
        theme_text_muted: Couleur texte secondaire hex.
        stress_interest_rate_range: Range slider taux BCE (min_bp, max_bp, step_bp).
        stress_unemployment_range: Range slider chomage bipolaire (min, max, step).
        stress_gdp_range: Range slider PIB (min_pct, max_pct, step_pct).
        stress_hpi_range: Range slider HPI (min_pct, max_pct, step_pct).
        stress_inflation_range: Range slider inflation (min_pct, max_pct, step_pct).
    """

    page_title: str = "IFRS 9 Risk Cockpit"
    page_icon: str = "$"
    layout: str = "wide"
    # Palette Steel Blue sur fond sombre — WCAG AAA (7:1+ sur #0C1222)
    theme_primary: str = "#3B82F6"
    theme_primary_text: str = "#60A5FA"  # Blue 400 pour texte (8.2:1)
    theme_secondary: str = "#1E40AF"
    theme_accent: str = "#34D399"  # Emerald 400 (9.6:1)
    theme_bg_dark: str = "#0C1222"
    theme_bg_card: str = "#1E293B"
    theme_text: str = "#F8FAFC"
    theme_text_muted: str = "#A1B2C8"  # Slate clair (7.2:1)
    # Couleurs semantiques — WCAG AAA
    color_success: str = "#34D399"  # Emerald 400 (9.6:1)
    color_warning: str = "#FBBF24"  # Amber 400 (11.2:1)
    color_danger: str = "#F87171"  # Red 400 (7.5:1)
    color_info: str = "#22D3EE"  # Cyan 400 (10.1:1)
    # Slider ranges (FR33) — taux en bp, chomage bipolaire, reste en pct
    stress_interest_rate_range: Tuple[float, float, float] = (-400.0, 650.0, 25.0)
    stress_unemployment_range: Tuple[float, float, float] = (-7.0, 7.0, 1.0)
    stress_gdp_range: Tuple[float, float, float] = (-8.0, 8.0, 0.5)
    stress_hpi_range: Tuple[float, float, float] = (-30.0, 20.0, 1.0)
    stress_inflation_range: Tuple[float, float, float] = (-2.0, 8.0, 0.5)


DASHBOARD_CONFIG = DashboardConfig()

# Palette de donnees pour les graphiques Plotly (6 couleurs, dans cet ordre)
CHART_COLORS: Tuple[str, ...] = (
    "#3B82F6",  # Steel Blue (primary)
    "#34D399",  # Emerald 400 (success) — WCAG AAA
    "#FBBF24",  # Amber 400 (warning) — WCAG AAA
    "#F87171",  # Red 400 (danger) — WCAG AAA
    "#22D3EE",  # Cyan 400 (info) — WCAG AAA
    "#8B5CF6",  # Purple (accent)
)

# Palette CVD-safe IBM (daltoniens) — pour séries de données multi-catégories
CVD_SAFE_COLORS: Tuple[str, ...] = (
    "#648FFF",  # Blue
    "#785EF0",  # Purple
    "#DC267F",  # Magenta
    "#FE6100",  # Orange
    "#FFB000",  # Gold
)

STAGE_COLORS: Dict[int, str] = {
    1: "#34D399",  # Emerald 400 — WCAG AAA
    2: "#FBBF24",  # Amber 400 — WCAG AAA
    3: "#F87171",  # Red 400 — WCAG AAA
}

PE_CATEGORY_COLORS: Dict[str, str] = {
    "Performing": "#34D399",
    "Watchlist": "#FBBF24",
    "Distressed": "#F87171",
}

# ──────────────────────────────────────────────
# FEATURES (listes de reference pour les modeles)
# ──────────────────────────────────────────────

# Features credit — utilisees par pd_model, lgd_model, ead_model
CREDIT_NUMERICAL_FEATURES: List[str] = [
    "revenue",
    "ebitda",
    "debt_ratio",
    "credit_score",
    "dpd",
    "collateral",
    "loan_amount",
    "utilization_rate",
    "loan_to_revenue",        # engineered
    "collateral_coverage",    # engineered
]

CREDIT_CATEGORICAL_FEATURES: List[str] = [
    "sector",
    "loan_type",
]

# Features PE — utilisees par pe_model, pe_calculator
PE_NUMERICAL_FEATURES: List[str] = [
    "revenue",
    "ebitda",
    "entry_multiple",
    "leverage",
    "vintage",
    "holding_years",
]

PE_CATEGORICAL_FEATURES: List[str] = [
    "sector",
    "valuation_method",
]

TARGET: str = "default_flag"

# Aliases de compatibilite avec le code existant (pd_model.py importe ces noms)
NUMERICAL_FEATURES: List[str] = CREDIT_NUMERICAL_FEATURES
CATEGORICAL_FEATURES: List[str] = CREDIT_CATEGORICAL_FEATURES

# ──────────────────────────────────────────────
# CONTRATS DATAFRAME (AR4)
# ──────────────────────────────────────────────

REQUIRED_CREDIT_COLS: frozenset[str] = frozenset({
    "enterprise_id", "sector", "revenue", "ebitda", "debt_ratio",
    "credit_score", "dpd", "collateral", "loan_amount", "utilization_rate",
    "default_flag", "pd_origination",
})

# ──────────────────────────────────────────────
# CONTRAT DE DONNEES (Enums, Clipping, Features Engineered)
# ──────────────────────────────────────────────

# Domaines de valeurs categorielles (Enums strictes)
ALLOWED_SECTORS: frozenset[str] = frozenset(s.name for s in SECTORS)
ALLOWED_LOAN_TYPES: frozenset[str] = frozenset({"Revolving", "Term"})

# Bornes de clipping outliers (appliquees avant entrainement)
CLIPPING_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "debt_ratio": (0.0, 1.5),
    "credit_score": (300.0, 850.0),
    "utilization_rate": (0.0, 1.2),
}

# Features engineered calculees a la volee
ENGINEERED_FEATURES: List[str] = ["loan_to_revenue", "collateral_coverage"]

REQUIRED_PE_COLS: frozenset[str] = frozenset({
    "enterprise_id", "sector", "revenue", "ebitda",
    "entry_multiple", "leverage", "vintage", "holding_years",
    "valuation_method",
})

# Colonnes requises en sortie du pipeline credit (pour le comparateur)
REQUIRED_CREDIT_RESULT_COLS: frozenset[str] = frozenset({
    "enterprise_id", "sector", "pd_12m", "pd_lifetime",
    "lgd", "ead", "ecl_weighted", "stage", "rwa_credit",
})

# Colonnes requises en sortie du pipeline PE (pour le comparateur)
REQUIRED_PE_RESULT_COLS: frozenset[str] = frozenset({
    "enterprise_id", "sector", "nav", "delta_nav",
    "expected_loss_pe", "risk_category", "rwa_pe",
})

# ──────────────────────────────────────────────
# REGLES D'INCOHERENCE MACRO (FR36)
# ──────────────────────────────────────────────
# Chaque regle = (nom, description, dict de conditions).
# Les conditions utilisent les cles des sliders.
# L'evaluation est faite dans le dashboard, pas ici.

@dataclass(frozen=True)
class MacroIncoherenceRule:
    """Regle de detection d'incoherence entre variables macro.

    Attributes:
        name: Identifiant court de la regle.
        description: Message affiche a l'utilisateur.
        conditions: Dict de conditions (cle slider -> seuil).
            Format : {"variable_op": seuil} ou op est "gt" ou "lt".
    """

    name: str
    description: str
    conditions: Tuple[Tuple[str, str, float], ...]


MACRO_INCOHERENCE_RULES: Tuple[MacroIncoherenceRule, ...] = (
    MacroIncoherenceRule(
        name="gdp_unemployment_crisis",
        description="PIB > +3% et chomage crise > +3pp simultanement",
        conditions=(
            ("gdp_pct", "gt", 3.0),
            ("unemployment_bipolar", "lt", -3.0),
        ),
    ),
    MacroIncoherenceRule(
        name="gdp_unemployment_tech",
        description="PIB > +3% et chomage techno > +3pp simultanement",
        conditions=(
            ("gdp_pct", "gt", 3.0),
            ("unemployment_bipolar", "gt", 3.0),
        ),
    ),
    MacroIncoherenceRule(
        name="deflation_rates",
        description="Inflation < 0% et taux > +200bp",
        conditions=(
            ("inflation_pct", "lt", 0.0),
            ("interest_rate_bp", "gt", 200.0),
        ),
    ),
    MacroIncoherenceRule(
        name="hpi_gdp",
        description="HPI > +10% et PIB < -2%",
        conditions=(
            ("hpi_pct", "gt", 10.0),
            ("gdp_pct", "lt", -2.0),
        ),
    ),
)

# ──────────────────────────────────────────────
# VALIDATION DE LA CONFIGURATION (FR57)
# ──────────────────────────────────────────────

def validate_config() -> None:
    """Valide la coherence de la configuration au chargement.

    Verifie les invariants de configuration :
    - Proportions des secteurs somment a 1.
    - Ponderations des scenarios ECL somment a 1.
    - RW PE par defaut dans les options CRR3.
    - Taux de defaut de base dans les bornes [0, 1].
    - Proportions dans [0, 1].
    - Fourchettes de multiples coherentes (min < max).

    Raises:
        ValueError: Si une contrainte de configuration est violee.
    """

    # Proportions des secteurs
    total_proportion = sum(s.proportion for s in SECTORS)
    if abs(total_proportion - 1.0) > 1e-6:
        raise ValueError(
            f"Les proportions des secteurs ne somment pas a 1 : {total_proportion:.6f}"
        )

    # Ponderations des scenarios ECL
    total_weight = sum(s.weight for s in ECL_SCENARIOS)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Les ponderations des scenarios ECL ne somment pas a 1 : {total_weight:.6f}"
        )

    # RW PE par defaut dans les options
    if BASEL_CONFIG.rw_pe_default not in BASEL_CONFIG.rw_pe_options:
        raise ValueError(
            f"RW PE par defaut ({BASEL_CONFIG.rw_pe_default}) "
            f"absent des options CRR3 {BASEL_CONFIG.rw_pe_options}"
        )

    # Validation par secteur
    for sector in SECTORS:
        if not 0.0 < sector.proportion <= 1.0:
            raise ValueError(
                f"Proportion du secteur {sector.name} hors bornes : {sector.proportion}"
            )

        if not 0.0 < sector.base_default_rate < 1.0:
            raise ValueError(
                f"Taux de defaut du secteur {sector.name} hors bornes : "
                f"{sector.base_default_rate}"
            )

        if sector.entry_multiple_range[0] >= sector.entry_multiple_range[1]:
            raise ValueError(
                f"Fourchette de multiples incoherente pour {sector.name} : "
                f"{sector.entry_multiple_range}"
            )

        if sector.ebitda_margin_range[0] >= sector.ebitda_margin_range[1]:
            raise ValueError(
                f"Fourchette de marge EBITDA incoherente pour {sector.name} : "
                f"{sector.ebitda_margin_range}"
            )

    # Poids SICR somment a 1
    sicr_total = (
        SICR_CONFIG.w_pd_ratio + SICR_CONFIG.w_pd_delta
        + SICR_CONFIG.w_dpd + SICR_CONFIG.w_macro
    )
    if abs(sicr_total - 1.0) > 1e-6:
        raise ValueError(
            f"Les poids SICR ne somment pas a 1 : {sicr_total:.6f}"
        )

    # 5 noms de secteurs attendus
    expected_names = {"Technologie", "Industrie", "Sante", "Immobilier", "Services"}
    actual_names = {s.name for s in SECTORS}
    if actual_names != expected_names:
        raise ValueError(
            f"Secteurs attendus {expected_names}, trouves {actual_names}"
        )

    # Scenarios predefinis minimum attendus
    expected_scenarios = {
        "Central", "Stagflation", "Rupture techno", "Reprise"
    }
    actual_scenarios = set(PREDEFINED_SCENARIOS.keys())
    if not expected_scenarios.issubset(actual_scenarios):
        raise ValueError(
            f"Scenarios predefinis manquants {expected_scenarios - actual_scenarios}, "
            f"trouves {actual_scenarios}"
        )


# Validation automatique a l'import
validate_config()


# ──────────────────────────────────────────────
# POINT D'ENTREE STANDALONE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("IFRS 9 Risk Cockpit — Configuration")
    print("=" * 60)

    print(f"\nSeed : {RANDOM_SEED}")
    print(f"Clients : {N_CLIENTS:,}")
    print(f"Mois historique : {N_MONTHS}")

    print(f"\n--- {len(SECTORS)} Secteurs ---")
    for s in SECTORS:
        print(
            f"  {s.name:15s} | prop={s.proportion:.0%} | PD base={s.base_default_rate:.0%} "
            f"| IPEV={s.valuation_method:15s} | multiple={s.exit_multiple_base:.1f}x"
        )
    total = sum(s.proportion for s in SECTORS)
    print(f"  Total proportions : {total:.4f}")

    print(f"\n--- {len(ECL_SCENARIOS)} Scenarios ECL ---")
    for sc in ECL_SCENARIOS:
        print(f"  {sc.name:12s} | poids={sc.weight:.0%}")
    total_w = sum(sc.weight for sc in ECL_SCENARIOS)
    print(f"  Total poids : {total_w:.4f}")

    print(f"\n--- {len(PREDEFINED_SCENARIOS)} Scenarios predefinis (dropdown) ---")
    for name, params in PREDEFINED_SCENARIOS.items():
        print(f"  {name:20s} | {params}")

    print(f"\n--- Basel III / CRR3 ---")
    print(f"  CET1 cible : {BASEL_CONFIG.cet1_target:.1%}")
    print(f"  RW PE : {BASEL_CONFIG.rw_pe_options} (defaut={BASEL_CONFIG.rw_pe_default}%)")
    print(f"  PE max allocation : {BASEL_CONFIG.pe_max_allocation:.0%}")
    print(f"  HHI max : {BASEL_CONFIG.hhi_max}")

    print(f"\n--- Risk Appetite ---")
    print(f"  ECL/EAD : vert < {RISK_APPETITE_CONFIG.ecl_ead_green:.1%} "
          f"| ambre < {RISK_APPETITE_CONFIG.ecl_ead_amber:.1%} | rouge")
    print(f"  RAROC   : vert > {RISK_APPETITE_CONFIG.raroc_green:.1%} "
          f"| ambre > {RISK_APPETITE_CONFIG.raroc_amber:.1%} | rouge")
    print(f"  HHI     : vert < {RISK_APPETITE_CONFIG.hhi_green} "
          f"| ambre < {RISK_APPETITE_CONFIG.hhi_amber} | rouge")

    print(f"\n--- Contrats DataFrame ---")
    print(f"  Credit cols : {len(REQUIRED_CREDIT_COLS)} colonnes")
    print(f"  PE cols     : {len(REQUIRED_PE_COLS)} colonnes")
    print(f"  Credit result : {len(REQUIRED_CREDIT_RESULT_COLS)} colonnes")
    print(f"  PE result     : {len(REQUIRED_PE_RESULT_COLS)} colonnes")

    print(f"\n--- Regles d'incoherence macro : {len(MACRO_INCOHERENCE_RULES)} ---")
    for rule in MACRO_INCOHERENCE_RULES:
        print(f"  {rule.name:25s} | {rule.description}")

    print(f"\n--- Classification PE ---")
    print(f"  Performing : P(distress) < {PE_CLASSIFICATION_CONFIG.distress_threshold_performing:.0%}")
    print(f"  Watchlist  : P(distress) < {PE_CLASSIFICATION_CONFIG.distress_threshold_watchlist:.0%}")
    print(f"  Distressed : P(distress) >= {PE_CLASSIFICATION_CONFIG.distress_threshold_watchlist:.0%}")
    print(f"  Secondary discount : {PE_CLASSIFICATION_CONFIG.secondary_discount:.0%}")

    print(f"\n--- Palette ---")
    print(f"  Primary  : {DASHBOARD_CONFIG.theme_primary}")
    print(f"  Bg dark  : {DASHBOARD_CONFIG.theme_bg_dark}")

    print("\nConfiguration valide.")
