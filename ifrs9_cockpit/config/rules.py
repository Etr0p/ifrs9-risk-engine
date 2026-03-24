"""Configuration — regles d'incoherence macro et validation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# REGLES D'INCOHERENCE MACRO (FR36)
# ──────────────────────────────────────────────
# Chaque regle = (nom, description, dict de conditions).
# Les conditions utilisent les cles des sliders.
# L'evaluation est faite dans le pipeline, pas ici.

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
    from ifrs9_cockpit.config import (
        SECTORS, ECL_SCENARIOS, BASEL_CONFIG, SICR_CONFIG,
        PREDEFINED_SCENARIOS, ASSET_CLASSES,
    )

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

    # Validation des classes d'actifs (>= 10 classes)
    if len(ASSET_CLASSES) < 10:
        raise ValueError(f"Au moins 10 classes d'actifs attendues, {len(ASSET_CLASSES)} trouvees")

    for ac in ASSET_CLASSES:
        if ac.category not in ("level1_detailed", "level2_parametric", "level3_simple"):
            raise ValueError(f"Categorie invalide pour {ac.name}: {ac.category}")
        if not 0.0 < ac.pd_base < 1.0:
            raise ValueError(f"pd_base hors bornes pour {ac.name}: {ac.pd_base}")
        if not 0.0 < ac.lgd_base < 1.0:
            raise ValueError(f"lgd_base hors bornes pour {ac.name}: {ac.lgd_base}")
        if not 0.0 < ac.asset_correlation <= 0.5:
            raise ValueError(f"asset_correlation hors bornes pour {ac.name}: {ac.asset_correlation}")
        expected_macro_keys = {"gdp_growth", "unemployment_rate", "interest_rate", "hpi_growth", "inflation_rate"}
        if set(ac.macro_sensitivities.keys()) != expected_macro_keys:
            raise ValueError(f"macro_sensitivities invalides pour {ac.name}")
