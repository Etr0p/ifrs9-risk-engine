"""Configuration — secteurs entreprise et classes d'actifs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# REPRODUCTIBILITE
# ──────────────────────────────────────────────
RANDOM_SEED: int = 123

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
        base_default_rate=0.025,
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
        base_default_rate=0.020,
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
        base_default_rate=0.012,
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
        base_default_rate=0.022,
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
        base_default_rate=0.018,
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
# CLASSES D'ACTIFS (10 classes, 3 niveaux de profondeur)
# ──────────────────────────────────────────────
# Extension multi-actif : chaque classe est definie par un profil
# parametrique integrant risque de credit (Vasicek ASRF), risque
# climatique (physique/transition, EBA/GL/2025/01), contagion
# (Eisenberg-Noe), et contraintes regulatoires (CRR3, LCR/NSFR).

@dataclass(frozen=True)
class AssetClassProfile:
    """Profil d'une classe d'actifs pour le moteur multi-actif.

    3 niveaux de profondeur :
        - level1_detailed : position par position (Corporate Loans, PE)
        - level2_parametric : parametrique enrichi (Souverain, Immo Retail, ProjFin)
        - level3_simple : parametrique simple (Covered Bonds, Conso, Trade, Interbank, Titrisation)

    Attributes:
        name: Identifiant interne (e.g. "sovereign", "retail_mortgage").
        label: Libelle affichage (e.g. "Obligations Souveraines").
        category: Niveau de profondeur ("level1_detailed" | "level2_parametric" | "level3_simple").
        typical_weight: Poids typique dans un bilan bancaire europeen (somme ~1.0).
        pd_base: PD de base TTC (Through-The-Cycle).
        pd_std: Dispersion intra-classe de la PD.
        lgd_base: LGD de base.
        tenor: Maturite moyenne (annees).
        asset_correlation: Correlation d'actif Vasicek rho (CRR3 Art. 153/154).
        rw_crr3: Risk Weight standardise CRR3.
        output_floor_binding: Si l'output floor CRR3 contraint cette classe.
        input_floor_pd: PD floor reglementaire CRR3.
        input_floor_lgd: LGD floor reglementaire CRR3.
        physical_risk: Vulnerabilite au risque physique climatique [0-1].
        transition_risk: Vulnerabilite au risque de transition carbone [0-1].
        green_capex_ratio: Part d'investissement vert (attenuateur transition) [0-1].
        scope3_exposure: Exposition chaine d'approvisionnement [0-1].
        absorption_buffer: Capacite d'absorption de choc avant contagion [0-1].
        macro_sensitivities: Dict des sensibilites aux 5 variables macro.
        hqla_eligible: Eligibilite HQLA pour le LCR.
        hqla_level: Niveau HQLA (1=0% haircut, 2=15%, 3=50%, 0=non eligible).
    """

    name: str
    label: str
    category: str  # "level1_detailed" | "level2_parametric" | "level3_simple"
    typical_weight: float

    # --- Risque de credit (Vasicek ASRF) ---
    pd_base: float
    pd_std: float
    lgd_base: float
    tenor: float
    asset_correlation: float  # rho CRR3 Art. 153/154

    # --- Reglementaire CRR3 ---
    rw_crr3: float
    output_floor_binding: bool
    input_floor_pd: float
    input_floor_lgd: float

    # --- Risque climatique (EBA/GL/2025/01, NGFS) ---
    physical_risk: float
    transition_risk: float
    green_capex_ratio: float
    scope3_exposure: float

    # --- Contagion (Eisenberg-Noe) ---
    absorption_buffer: float
    macro_sensitivities: Dict[str, float]

    # --- Liquidite (LCR/NSFR) ---
    hqla_eligible: bool
    hqla_level: int  # 1, 2, 3, ou 0 (non eligible)

    # --- Structure de couts par metier (EBA FINREP 2023, McKinsey 2024) ---
    cir_class: float     # Cost/Income Ratio specifique a la classe d'actifs
    # --- Capacite de marche EU (market impact logarithmique, Kyle 1985) ---
    market_capacity_eur: float = 1_000e9  # taille du marche EU en euros

    # --- Normes reglementaires complementaires ---
    exempt_from_staging: bool = False  # IFRS 9 B5.5.25 (souverain = pas de Stage 2)
    ltv_distribution: Optional[Dict[float, float]] = None  # CRR3 Art. 124-125 LTV schedule
    securitisation_mix: Optional[Dict[str, float]] = None   # CRR3 Art. 242-270 SEC-SA tranching
    rsf_weight: float = 0.50  # NSFR Basel III — Required Stable Funding weight

    # --- Traitement comptable (FVTPL / FVOCI / amortised_cost) ---
    accounting_treatment: str = "amortised_cost"
    # "amortised_cost" : ECL standard (defaut, 10 classes existantes)
    # "fvoci" : ECL + OCI mark-to-market (corporate_bonds)
    # "fvtpl" : pas d'ECL, MTM P&L loss (equities)

    # --- IRRBB (Basel III Avr 2016, EBA/GL/2022/14) ---
    duration: float = 3.0  # Modified duration effective (annees)

    # --- Override de volatilite marche (pour l'optimizer BL-CVaR) ---
    market_vol_override: Optional[float] = None
    # Si present, utilise directement dans le calcul de vol au lieu de
    # la formule pd_std * lgd_base * 10. Pour les classes de marche :
    # equities ~20%, corporate_bonds ~6%, repos ~0.5%, derives ~10%.


# --- 14 profils de classes d'actifs calibres ---
# Sources : CRR3, EBA, BCE, Preqin, Moody's, S&P, BRI
ASSET_CLASSES: List[AssetClassProfile] = [
    # ── 1. Prets Corporate (Level 1 — position par position, existant) ──
    AssetClassProfile(
        name="corporate_loans",
        label="Prets Corporate",
        category="level1_detailed",
        typical_weight=0.22,
        pd_base=0.045,  # moyenne ponderee des 5 secteurs
        pd_std=0.025,
        lgd_base=0.35,
        tenor=4.0,
        asset_correlation=0.18,  # CRR3 Art. 153, PD moyen ~4.5%
        rw_crr3=1.00,  # SA = 100% ; IRB calcule dynamiquement
        output_floor_binding=True,
        input_floor_pd=0.0003,  # CRR3 : 3bp corporate
        input_floor_lgd=0.25,   # CRR3 : 25% unsecured
        physical_risk=0.15,
        transition_risk=0.50,   # moyenne ponderee secteurs
        green_capex_ratio=0.20,
        scope3_exposure=0.25,
        absorption_buffer=0.40,
        macro_sensitivities={
            "gdp_growth": 1.5, "unemployment_rate": 1.5,
            "interest_rate": 1.2, "hpi_growth": 0.8, "inflation_rate": 1.2,
        },
        hqla_eligible=False,
        hqla_level=0,
        cir_class=0.40,
        market_capacity_eur=5_000e9,  # ECB MFI balance sheet
        rsf_weight=0.50,
        duration=3.0,  # Tenor moyen 4Y, prepaiement → ~3Y effectif
        market_vol_override=0.035,  # iTraxx Main 5Y ~3.5% (Markit 2024)
    ),
    # ── 2. Private Equity (Level 1 — position par position, existant) ──
    AssetClassProfile(
        name="private_equity",
        label="Private Equity",
        category="level1_detailed",
        typical_weight=0.04,
        pd_base=0.06,
        pd_std=0.035,
        lgd_base=0.45,
        tenor=5.0,
        asset_correlation=0.24,  # CRR3 Art. 153, traite comme equity
        rw_crr3=2.50,  # CRR3 Art. 133 : 250% (defaut), 190/400 selon risque
        output_floor_binding=True,
        input_floor_pd=0.0003,
        input_floor_lgd=0.25,
        physical_risk=0.10,
        transition_risk=0.30,
        green_capex_ratio=0.25,
        scope3_exposure=0.15,
        absorption_buffer=0.30,
        macro_sensitivities={
            "gdp_growth": 1.8, "unemployment_rate": 0.5,
            "interest_rate": 1.0, "hpi_growth": 0.5, "inflation_rate": 1.0,
            # IR 1.0 (not 2.0): PE uses fixed-rate LBO debt (3-5yr), can time exits,
            # less directly exposed than variable-rate mortgages (Kaplan & Strömberg 2009)
        },
        hqla_eligible=False,
        hqla_level=0,
        cir_class=0.30,  # LP bancaire: mgmt fees ~15% + admin ~5% + carry ~10%
        market_capacity_eur=300e9,  # Preqin EU
        rsf_weight=1.00,  # NSFR : PE illiquide, RSF 100%
        duration=5.0,  # Illiquide, sensible au taux d'actualisation
        market_vol_override=0.18,  # Cambridge PE Index ~18% (Preqin 2024)
    ),
    # ── 3. Obligations Souveraines (Level 2 — parametrique enrichi) ──
    # Rating moyen AA/A (zone euro core), PD tres basse, HQLA Level 1.
    # Contagion : source majeure (nexus souverain-bancaire).
    AssetClassProfile(
        name="sovereign",
        label="Obligations Souveraines",
        category="level2_parametric",
        typical_weight=0.14,
        pd_base=0.0005,  # AA/A zone euro ~5bp TTC
        pd_std=0.006,  # was 0.002 — reflects peripheral vol (BTP ~8% in 2011-12)
        lgd_base=0.45,
        tenor=7.0,
        asset_correlation=0.06,  # Souverain : rho empirique 0.03-0.08
        # (Longstaff et al. 2011, IMF GFSR 2012). CRR3 Art. 153 (0.24)
        # concerne corporates/institutions, PAS le souverain (0% RW std).
        rw_crr3=0.0,     # AAA/AA domestique = 0% (CRR3 Art. 114)
        output_floor_binding=False,  # 0% RW : floor non contraignant
        input_floor_pd=0.0003,
        input_floor_lgd=0.45,  # souverain : pas de collateral specifique
        physical_risk=0.15,
        transition_risk=0.20,
        green_capex_ratio=0.0,   # pas d'investissement propre
        scope3_exposure=0.10,
        absorption_buffer=0.30,  # reserve fiscale
        macro_sensitivities={
            # Souverain zone euro core (AA/A) : flight-to-quality en crise.
            # En GFC, les spreads souverains core ont BAISSE (Bunds -100bp,
            # OATs -80bp) — zero perte credit sur DE/FR/NL/FI/AT. Meme IT/ES
            # n'ont pas fait defaut. Sensibilites tres reduites car :
            # - Souverainete monetaire (BCE OMT/PEPP backstop)
            # - Defauts souverains EU zone euro : 0 en 75 ans (hors Grece PSI)
            # - Correlation empirique : Longstaff 2011 rho~0.05 (vs 0.24 corp)
            # Source : ECB SDW, BIS QR 2014, Reinhart & Rogoff (2009).
            "gdp_growth": 0.15, "unemployment_rate": 0.10,
            "interest_rate": 0.10, "hpi_growth": 0.05, "inflation_rate": 0.20,
        },
        hqla_eligible=True,
        hqla_level=1,  # Level 1 : 0% haircut LCR
        cir_class=0.08,
        market_capacity_eur=11_000e9,  # ECB SDW govt debt
        exempt_from_staging=True,  # IFRS 9 B5.5.25 : souverain AAA/AA exempt de Stage 2
        rsf_weight=0.00,  # NSFR : HQLA L1, RSF 0%
        duration=6.0,  # Benchmark 7-10Y EU
        market_vol_override=0.04,  # EU Govt 10Y yield vol ~4% (ECB SDW)
    ),
    # ── 4. Credit Immobilier Retail (Level 2 — parametrique enrichi) ──
    # Tres sensible au HPI et aux taux. LTV = driver principal.
    # Risque physique eleve (inondations, DPE/EPC).
    AssetClassProfile(
        name="retail_mortgage",
        label="Credit Immobilier",
        category="level2_parametric",
        typical_weight=0.15,
        pd_base=0.012,
        pd_std=0.008,
        lgd_base=0.15,  # faible grace au collateral immobilier
        tenor=20.0,
        asset_correlation=0.15,  # CRR3 Art. 154(2)(a) : fixe 0.15
        rw_crr3=0.35,    # CRR3 Art. 125 : 20-70% selon LTV
        output_floor_binding=True,
        input_floor_pd=0.0003,
        input_floor_lgd=0.05,  # CRR3 : 5% residentiel garanti
        physical_risk=0.70,    # inondations, DPE, canicule
        transition_risk=0.30,  # renovation energetique obligatoire
        green_capex_ratio=0.10,
        scope3_exposure=0.05,
        absorption_buffer=0.35,
        macro_sensitivities={
            "gdp_growth": 0.8, "unemployment_rate": 1.2,
            "interest_rate": 2.0, "hpi_growth": 2.5, "inflation_rate": 0.5,
        },
        hqla_eligible=False,
        hqla_level=0,
        cir_class=0.42,
        market_capacity_eur=5_000e9,  # ECB residential mortgage stock
        ltv_distribution={  # CRR3 Art. 124-125 : distribution LTV residentiel FR/EU
            0.50: 0.15, 0.60: 0.25, 0.70: 0.25,
            0.80: 0.20, 0.90: 0.10, 1.00: 0.05,
        },
        rsf_weight=0.35,  # NSFR : prets immobiliers < 35 ans
        duration=4.0,  # Tenor 20Y, CPR ramene a ~4Y effectif
        market_vol_override=0.02,  # RMBS AAA spread vol ~2% (Barclays 2024)
    ),
    # ── 5. Financement de Projet (Level 2 — parametrique enrichi) ──
    # Cash-flow isole, risque de construction, slotting CRR3.
    # Peut etre "vert" (PF energies renouvelables : green_capex eleve).
    AssetClassProfile(
        name="project_finance",
        label="Financement de Projet",
        category="level2_parametric",
        typical_weight=0.04,
        pd_base=0.018,
        pd_std=0.012,
        lgd_base=0.35,
        tenor=12.0,
        asset_correlation=0.20,  # intermediaire corporate/retail
        rw_crr3=1.30,    # CRR3 Art. 153(5) : slotting 130% (strong)
        output_floor_binding=True,
        input_floor_pd=0.0003,
        input_floor_lgd=0.25,
        physical_risk=0.25,    # risque de construction, localisation
        transition_risk=0.60,  # forte exposition energetique
        green_capex_ratio=0.50,  # mix : ~50% PF renouvelables
        scope3_exposure=0.15,
        absorption_buffer=0.25,
        macro_sensitivities={
            "gdp_growth": 1.0, "unemployment_rate": 0.5,
            # IR sensitivity = 0 : pour les SPV leverages, la baisse des taux
            # reduit le service de dette (favorable) mais signale une faiblesse
            # macro (adverse). Les deux effets se compensent. Le risque macro
            # est deja capture par GDP et unemployment. (Blanc et al. 2014)
            "interest_rate": 0.0, "hpi_growth": 0.3, "inflation_rate": 0.8,
        },
        hqla_eligible=False,
        hqla_level=0,
        cir_class=0.55,
        market_capacity_eur=500e9,  # Dealogic, Refinitiv
        rsf_weight=0.85,  # NSFR : long tenor, illiquide
        duration=7.0,  # Long tenor, pas d'option de prepaiement
        market_vol_override=0.05,  # Infra debt spread vol ~5% (Preqin 2024)
    ),
    # ── 6. Obligations Securisees / Covered Bonds (Level 2 — parametric) ──
    # Double recours (emetteur + cover pool), PD/LGD tres faibles.
    # HQLA Level 2A. Contagion depuis immobilier (cover pool = mortgages).
    AssetClassProfile(
        name="covered_bonds",
        label="Obligations Securisees",
        category="level2_parametric",
        typical_weight=0.07,
        pd_base=0.0001,
        pd_std=0.0005,
        lgd_base=0.10,
        tenor=5.0,
        asset_correlation=0.15,  # similaire retail mortgage (cover pool)
        rw_crr3=0.10,    # CRR3 Art. 129 : 10%
        output_floor_binding=False,
        input_floor_pd=0.0003,
        input_floor_lgd=0.05,   # garanti par cover pool
        physical_risk=0.20,     # via cover pool immobilier
        transition_risk=0.10,
        green_capex_ratio=0.05,
        scope3_exposure=0.05,
        absorption_buffer=0.50,  # double recours = haute absorption
        macro_sensitivities={
            "gdp_growth": 0.3, "unemployment_rate": 0.3,
            "interest_rate": 0.8, "hpi_growth": 1.0, "inflation_rate": 0.3,
        },
        hqla_eligible=True,
        hqla_level=2,  # Level 2A : 15% haircut LCR
        cir_class=0.15,
        market_capacity_eur=2_500e9,  # ECBC Factbook
        rsf_weight=0.15,  # NSFR : HQLA L2A, RSF 15%
        duration=5.0,  # Benchmark Pfandbrief 5Y
        market_vol_override=0.01,  # Pfandbrief spread vol ~1% (vdp 2024)
    ),
    # ── 7. Credit Consommation (Level 3 — simple) ──
    # PD elevee, LGD elevee (pas de collateral), correlation faible.
    # Contagion depuis corporate (chomage → menages).
    AssetClassProfile(
        name="consumer_credit",
        label="Credit Consommation",
        category="level3_simple",
        typical_weight=0.05,
        pd_base=0.035,
        pd_std=0.020,
        lgd_base=0.65,
        tenor=3.0,
        asset_correlation=0.04,  # CRR3 Art. 154(2)(b) : revolving 0.04
        rw_crr3=0.75,    # CRR3 Art. 123 : 75%
        output_floor_binding=True,
        input_floor_pd=0.0005,  # CRR3 : 5bp retail
        input_floor_lgd=0.25,
        physical_risk=0.05,
        transition_risk=0.10,
        green_capex_ratio=0.0,
        scope3_exposure=0.05,
        absorption_buffer=0.30,
        macro_sensitivities={
            "gdp_growth": 1.0, "unemployment_rate": 2.0,
            "interest_rate": 1.0, "hpi_growth": 0.3, "inflation_rate": 1.5,
        },
        hqla_eligible=False,
        hqla_level=0,
        cir_class=0.58,
        market_capacity_eur=750e9,  # ECB EU consumer credit outstanding ~700-750B
        rsf_weight=0.50,
        duration=2.0,  # Tenor court (~3Y)
        market_vol_override=0.04,  # Consumer ABS spread vol ~4% (JPM 2024)
    ),
    # ── 8. Trade Finance (Level 3 — simple) ──
    # Tenor court (6 mois), CCF eleve, PD faible (auto-liquidating).
    # Contagion depuis interbancaire (lignes de confirmation LC).
    AssetClassProfile(
        name="trade_finance",
        label="Trade Finance",
        category="level3_simple",
        typical_weight=0.04,
        pd_base=0.008,
        pd_std=0.005,
        lgd_base=0.30,
        tenor=0.5,
        asset_correlation=0.20,
        rw_crr3=0.20,    # CRR3 Art. 111 : 20% (court terme)
        output_floor_binding=False,
        input_floor_pd=0.0003,
        input_floor_lgd=0.25,
        physical_risk=0.15,
        transition_risk=0.40,  # supply chain exposure
        green_capex_ratio=0.10,
        scope3_exposure=0.50,  # Scope 3 dominant (supply chain)
        absorption_buffer=0.20,
        macro_sensitivities={
            # ICC Trade Register 2024 : default rate < 0.06% meme en GFC.
            # Auto-liquidatif, tenor < 6m, adosse a des flux reels de marchandises.
            # Quasi-insensible aux crises macro (BCBS205 : waiver du floor maturite).
            "gdp_growth": 0.3, "unemployment_rate": 0.1,
            "interest_rate": 0.1, "hpi_growth": 0.0, "inflation_rate": 0.2,
        },
        hqla_eligible=False,
        hqla_level=0,
        cir_class=0.30,
        market_capacity_eur=1_000e9,  # ICC Trade Register
        rsf_weight=0.10,  # NSFR : auto-liquidating, tenor < 1 an
        duration=0.5,  # Auto-liquidating, < 6 mois
        market_vol_override=0.015,  # Short-term trade vol ~1.5% (ICC 2024)
    ),
    # ── 9. Interbancaire (Level 3 — simple) ──
    # PD tres faible, risque systemique eleve, HQLA Level 2B.
    # Contagion : cible majeure depuis souverain (rating correlation).
    AssetClassProfile(
        name="interbank",
        label="Interbancaire",
        category="level3_simple",
        typical_weight=0.03,  # EBA: unsecured interbank 2-5% of total assets (post-TLTRO)
        pd_base=0.0005,
        pd_std=0.002,
        lgd_base=0.45,
        tenor=0.25,  # tres court terme (O/N, 1W, 3M)
        asset_correlation=0.24,  # CRR3 Art. 153 : traite comme corporate
        rw_crr3=0.20,    # CRR3 Art. 120 : 20% (banque notee A+)
        output_floor_binding=False,
        input_floor_pd=0.0003,
        input_floor_lgd=0.45,
        physical_risk=0.05,
        transition_risk=0.10,
        green_capex_ratio=0.0,
        scope3_exposure=0.05,
        absorption_buffer=0.30,
        macro_sensitivities={
            # Interbancaire : tenor ultra-court (O/N-3M), BRRD bail-in protege,
            # risque de contagion systemique mais PD faible. IR sensitivity reduite
            # car les depots interbancaires ne sont pas des obligations a taux fixe.
            "gdp_growth": 0.3, "unemployment_rate": 0.2,
            "interest_rate": 0.3, "hpi_growth": 0.1, "inflation_rate": 0.3,
        },
        hqla_eligible=True,
        hqla_level=3,  # Level 2B : 50% haircut LCR
        cir_class=0.12,
        market_capacity_eur=500e9,  # ECB unsecured interbank outstanding post-TLTRO ~400-600B
        rsf_weight=0.00,  # NSFR : O/N-3M, RSF 0%
        duration=0.25,  # O/N a 3M
        market_vol_override=0.01,  # EURIBOR unsecured vol ~1% (ECB 2024)
    ),
    # ── 10. Titrisation / Produits Structures (Level 3 — simple) ──
    # Correlation eleve (portefeuille granulaire → correlation implicite),
    # risque de tranching, opacite. CRR3 : look-through obligatoire.
    AssetClassProfile(
        name="structured_products",
        label="Titrisation",
        category="level3_simple",
        typical_weight=0.02,  # poids faible post-GFC
        pd_base=0.010,
        pd_std=0.020,  # cliff effect amplification (BIS QR 2014)
        lgd_base=0.40,
        tenor=5.0,
        asset_correlation=0.30,  # correlation implicite elevee
        rw_crr3=1.00,    # CRR3 : 100% (senior, STS, look-through)
        output_floor_binding=True,
        input_floor_pd=0.0003,
        input_floor_lgd=0.25,
        physical_risk=0.10,
        transition_risk=0.20,
        green_capex_ratio=0.05,
        scope3_exposure=0.10,
        absorption_buffer=0.25,
        macro_sensitivities={
            # Titrisation post-GFC : senior STS dominante (60%+ du portefeuille).
            # IR sensitivity reduite (1.5→0.8) car les structures a taux variable
            # et la granularite du pool amortissent l'impact taux direct.
            # Sensible surtout au GDP (prepayment) et HPI (RMBS/CMBS sous-jacent).
            "gdp_growth": 0.7, "unemployment_rate": 0.5,
            "interest_rate": 0.8, "hpi_growth": 0.8, "inflation_rate": 0.3,
        },
        hqla_eligible=False,
        hqla_level=0,
        cir_class=0.50,
        market_capacity_eur=1_500e9,  # EU ABS/RMBS/CMBS
        securitisation_mix={  # CRR3 Art. 242-270 SEC-SA tranching mix
            "senior_sts": 0.60, "senior_non_sts": 0.25, "mezzanine": 0.15,
        },
        rsf_weight=0.85,  # NSFR : illiquide, complexe
        duration=4.0,  # WAL tranche
        market_vol_override=0.06,  # ABS spread vol ~6% (BIS QR 2024)
    ),
    # ── 11. Actions Cotees / Equities (Level 1 — yfinance real data) ──
    # FVTPL : pas d'ECL (IFRS 9 §4.1.4), impact direct en P&L via MTM.
    # Merton PD, beta × index haircut, CRR3 Art. 133 listed equity = 100%.
    AssetClassProfile(
        name="equities",
        label="Actions Cotees",
        category="level1_detailed",
        typical_weight=0.03,
        pd_base=0.020,    # Merton-implied, recalibre TTC
        pd_std=0.015,
        lgd_base=0.85,    # Equity = first loss, recovery ~15%
        tenor=0.0,        # No maturity (perpetual)
        asset_correlation=0.24,  # CRR3 equity
        rw_crr3=1.00,     # CRR3 Art. 133 listed equity
        output_floor_binding=True,
        input_floor_pd=0.0003,
        input_floor_lgd=0.25,
        physical_risk=0.15,
        transition_risk=0.45,
        green_capex_ratio=0.20,
        scope3_exposure=0.20,
        absorption_buffer=0.10,  # Equities = first to absorb losses
        macro_sensitivities={
            "gdp_growth": 2.0, "unemployment_rate": 1.5,
            "interest_rate": 1.5, "hpi_growth": 0.5, "inflation_rate": 1.0,
        },
        hqla_eligible=False,
        hqla_level=0,
        cir_class=0.35,
        market_capacity_eur=2_000e9,  # EU bank equity holdings
        accounting_treatment="fvtpl",
        exempt_from_staging=True,  # FVTPL → pas de staging
        rsf_weight=0.85,  # NSFR: equities illiquide
        duration=0.0,  # FVTPL, pas de cash flows contractuels
        market_vol_override=0.20,  # VSTOXX ~20%
    ),
    # ── 12. Obligations Corporate (Level 1 — Kaggle real data) ──
    # FVOCI : ECL standard + OCI mark-to-market (IFRS 9 §5.7.10).
    # Rating-based PD/LGD, spread duration, CRR3 ECRA blend.
    AssetClassProfile(
        name="corporate_bonds",
        label="Obligations Corporate",
        category="level1_detailed",
        typical_weight=0.06,
        pd_base=0.012,    # Blend IG/HY (70/30)
        pd_std=0.010,
        lgd_base=0.45,    # Senior unsecured average
        tenor=5.0,
        asset_correlation=0.06,  # Empirique IG/HY blend (De Servigny & Renault 2002)
        rw_crr3=0.65,     # ECRA blend (20-150%)
        output_floor_binding=True,
        input_floor_pd=0.0003,
        input_floor_lgd=0.25,
        physical_risk=0.15,
        transition_risk=0.40,
        green_capex_ratio=0.15,
        scope3_exposure=0.25,
        absorption_buffer=0.35,
        macro_sensitivities={
            # IR sensitivity reduite : taux bas = refinancement moins cher,
            # spreads comprimes, prix obligataires hauts → FAVORABLE pour le
            # credit corporate. La valeur 1.2 produisait Z=+2.4 en Reprise
            # (taux BCE 0%) → RAROC -10% en expansion, absurde economiquement.
            # Source : Gilchrist & Zakrajsek (2012) EBP, Merton (1974).
            "gdp_growth": 1.2, "unemployment_rate": 1.0,
            "interest_rate": 0.3, "hpi_growth": 0.3, "inflation_rate": 0.8,
        },
        hqla_eligible=True,
        hqla_level=2,  # L2A pour IG
        cir_class=0.20,
        market_capacity_eur=3_000e9,  # EU IG+HY outstanding
        accounting_treatment="fvoci",
        rsf_weight=0.50,
        duration=4.0,  # Benchmark IG 5Y
        market_vol_override=0.06,  # Spread vol ~6%
    ),
    # ── 13. Repos / SFT (Level 2 — ICMA-calibrated) ──
    # Collateralise, haircut spiral Brunnermeier-Pedersen (2009).
    # CRR3 Art. 222-224 CRM. Tenor ultra-court.
    AssetClassProfile(
        name="repos_sft",
        label="Repos / SFT",
        category="level2_parametric",
        typical_weight=0.04,
        pd_base=0.0005,   # Counterparty PD (same as interbank)
        pd_std=0.002,
        lgd_base=0.10,    # Collateralise → faible LGD
        tenor=0.08,       # ~1 mois moyen
        asset_correlation=0.24,
        rw_crr3=0.10,     # Post-CRM ~10%
        output_floor_binding=False,
        input_floor_pd=0.0003,
        input_floor_lgd=0.05,
        physical_risk=0.02,
        transition_risk=0.05,
        green_capex_ratio=0.0,
        scope3_exposure=0.02,
        absorption_buffer=0.50,  # Collateral = haute absorption
        macro_sensitivities={
            "gdp_growth": 0.2, "unemployment_rate": 0.1,
            "interest_rate": 0.2, "hpi_growth": 0.1, "inflation_rate": 0.1,
        },
        hqla_eligible=False,
        hqla_level=0,
        cir_class=0.08,
        market_capacity_eur=8_000e9,  # ICMA European repo survey
        rsf_weight=0.00,  # NSFR: matched repos RSF 0%
        duration=0.1,  # ~1 mois moyen
        market_vol_override=0.005,  # Near-zero normal vol
    ),
    # ── 14. Derives / CVA (Level 2 — BIS-calibrated) ──
    # SA-CCR (CRR3 Art. 274-280), SA-CVA (Art. 382-386).
    # 3 desks IRS/FX/CDS, netting, CSA, wrong-way risk.
    AssetClassProfile(
        name="derivatives_cva",
        label="Derives / CVA",
        category="level2_parametric",
        typical_weight=0.04,
        pd_base=0.003,    # Counterparty PD (ISDA netting)
        pd_std=0.005,
        lgd_base=0.35,    # CSA/netting reduces
        tenor=3.0,        # Maturite moyenne
        asset_correlation=0.20,
        rw_crr3=0.60,     # SA-CVA blend
        output_floor_binding=True,
        input_floor_pd=0.0003,
        input_floor_lgd=0.25,
        physical_risk=0.05,
        transition_risk=0.15,
        green_capex_ratio=0.0,
        scope3_exposure=0.10,
        absorption_buffer=0.30,
        macro_sensitivities={
            "gdp_growth": 0.8, "unemployment_rate": 0.5,
            # IR sensitivity = 0 : les desks derives sont generalement hedges
            # contre le risque directionnel de taux. La PD contrepartie est
            # capturee par GDP/unemployment, pas par la direction des taux.
            # (Gregory 2020, Counterparty Credit Risk)
            "interest_rate": 0.0, "hpi_growth": 0.2, "inflation_rate": 0.5,
        },
        hqla_eligible=False,
        hqla_level=0,
        cir_class=0.45,
        market_capacity_eur=1_000e9,  # BIS OTC derivatives (EAD basis)
        rsf_weight=0.50,
        duration=3.0,  # DV01 effectif du book de CVA hedge
        market_vol_override=0.10,  # MTM swing ~10%
    ),
]

ASSET_CLASS_NAMES: Tuple[str, ...] = tuple(ac.name for ac in ASSET_CLASSES)
ASSET_CLASS_LABELS: Tuple[str, ...] = tuple(ac.label for ac in ASSET_CLASSES)
ASSET_CLASS_MAP: Dict[str, AssetClassProfile] = {ac.name: ac for ac in ASSET_CLASSES}

# Regroupements par niveau de profondeur
LEVEL1_CLASSES: Tuple[str, ...] = tuple(
    ac.name for ac in ASSET_CLASSES if ac.category == "level1_detailed"
)
LEVEL2_CLASSES: Tuple[str, ...] = tuple(
    ac.name for ac in ASSET_CLASSES if ac.category == "level2_parametric"
)
LEVEL3_CLASSES: Tuple[str, ...] = tuple(
    ac.name for ac in ASSET_CLASSES if ac.category == "level3_simple"
)
