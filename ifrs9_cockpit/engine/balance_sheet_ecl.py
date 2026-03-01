"""Moteur ECL Vasicek ASRF pour les classes d'actifs Level 2/3.

Calcule l'ECL parametrique pour les 8 classes non-Level 1 en utilisant
le modele Vasicek ASRF (CRR3 Art. 153/154) avec :
    - PD conditionnelle Vasicek (Jensen-correct)
    - Staging cliff (S1→S2 via seuil SICR)
    - Ajustement climatique idiosyncrasique (avant Vasicek, pas de double-comptage)
    - Ponderation multi-scenarios (Base 50% + Adverse 25% + Favorable 25%)

References :
    - Vasicek (1991) : Loan portfolio value
    - CRR3 Art. 153/154 : formule IRB asset correlation
    - BIS WP 1274 (Jul 2025) : risque physique dans ASRF
    - EBA/GL/2025/01 : ESG dans l'evaluation du risque de credit
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm

from ifrs9_cockpit.config import (
    ASSET_CLASSES,
    ASSET_CLASS_MAP,
    AssetClassProfile,
    BASEL_CONFIG,
    SCENARIO_BASE,
    SCENARIOS,
)


def vasicek_conditional_pd(
    pd_base: float,
    rho: float,
    z: float,
) -> float:
    """PD conditionnelle Vasicek ASRF (CRR3 Art. 153).

    PD_cond(Z) = Phi((Phi^-1(PD) + sqrt(rho) * Z) / sqrt(1 - rho))

    Args:
        pd_base: PD inconditionnelle TTC.
        rho: Correlation d'actif (CRR3).
        z: Facteur systematique (normal standard).

    Returns:
        PD conditionnelle au facteur systematique Z.
    """
    pd_clipped = np.clip(pd_base, 1e-8, 1.0 - 1e-8)
    numerator = norm.ppf(pd_clipped) + np.sqrt(rho) * z
    denominator = np.sqrt(1.0 - rho)
    return float(norm.cdf(numerator / denominator))


def climate_adjusted_pd(
    pd_base: float,
    physical_risk: float,
    transition_risk: float,
    green_capex_ratio: float,
    scope3_exposure: float,
    carbon_price_shock: float = 0.0,
    physical_severity: float = 0.0,
) -> float:
    """Ajustement climatique idiosyncrasique de la PD.

    Applique AVANT le transform Vasicek (pas de double-comptage macro).

    Transition : delta_t = r_transition * (carbon_shock)^1.5 * (1 - 0.6 * green_capex)
    Physique : delta_p = r_physical * physical_severity
    Combine : delta = max(delta_t, delta_p) + 0.3 * scope3 * delta_t

    Args:
        pd_base: PD de base TTC.
        physical_risk: Vulnerabilite physique [0-1].
        transition_risk: Vulnerabilite transition [0-1].
        green_capex_ratio: Part investissement vert [0-1].
        scope3_exposure: Exposition supply chain [0-1].
        carbon_price_shock: Choc prix carbone normalise (1.0 = doublement).
        physical_severity: Severite risque physique [0-1].

    Returns:
        PD ajustee climatiquement (toujours avant Vasicek).
    """
    if carbon_price_shock <= 0.0 and physical_severity <= 0.0:
        return pd_base

    # Canal transition (convexite : exposant 1.5)
    delta_transition = (
        transition_risk
        * (max(0.0, carbon_price_shock) ** 1.5)
        * (1.0 - 0.6 * green_capex_ratio)
    )

    # Canal physique (lineaire)
    delta_physical = physical_risk * max(0.0, physical_severity)

    # Combinaison : max (canaux alternatifs) + add-on Scope 3
    delta_climate = max(delta_transition, delta_physical) + 0.3 * scope3_exposure * delta_transition

    pd_adjusted = pd_base * (1.0 + delta_climate)
    return min(pd_adjusted, 0.999)


def macro_to_z(
    macro_params: Dict[str, float],
    sensitivities: Dict[str, float],
) -> float:
    """Transforme les 5 variables macro en facteur systematique Z.

    Z = sum_i(w_i * (x_i - mu_i) / sigma_i)

    Les mu et sigma sont les valeurs de reference du SCENARIO_BASE.

    Args:
        macro_params: Dict des valeurs macro courantes.
        sensitivities: Dict des sensibilites de la classe d'actifs.

    Returns:
        Facteur systematique Z (normal standard).
    """
    # Mapping slider keys → macro variable names
    _key_map = {
        "gdp_pct": ("gdp_growth", SCENARIO_BASE.gdp_growth, 1.8),
        "unemployment_rate": ("unemployment_rate", SCENARIO_BASE.unemployment_rate, 1.5),
        "interest_rate": ("interest_rate", SCENARIO_BASE.interest_rate, 1.0),
        "hpi_pct": ("hpi_growth", SCENARIO_BASE.hpi_growth, 3.0),
        "inflation_pct": ("inflation_rate", SCENARIO_BASE.inflation_rate, 1.2),
    }
    # Direct macro variable names
    _direct_map = {
        "gdp_growth": (SCENARIO_BASE.gdp_growth, 1.8),
        "unemployment_rate": (SCENARIO_BASE.unemployment_rate, 1.5),
        "interest_rate": (SCENARIO_BASE.interest_rate, 1.0),
        "hpi_growth": (SCENARIO_BASE.hpi_growth, 3.0),
        "inflation_rate": (SCENARIO_BASE.inflation_rate, 1.2),
    }

    z = 0.0
    for var_name, (mu, sigma) in _direct_map.items():
        # Try direct name first, then slider key
        x = macro_params.get(var_name)
        if x is None:
            # Try slider key mapping
            for slider_key, (mapped_var, mapped_mu, mapped_sigma) in _key_map.items():
                if mapped_var == var_name and slider_key in macro_params:
                    x = macro_params[slider_key]
                    break
        if x is None:
            continue

        w = sensitivities.get(var_name, 0.0)

        # Signe : Z > 0 = stress = adverse
        # unemployment: hausse = adverse → z += w * (x-mu)/sigma
        # interest_rate: hausse = adverse (debt service burden) → z += w * (x-mu)/sigma
        # GDP, HPI, inflation: hausse = favorable → z -= w * (x-mu)/sigma
        #
        # IR rationale: the primary channel is debt service burden (Duffie et al. 2007).
        # Higher rates → higher debt service costs → higher default probability.
        # Rate cuts → lower debt service → fewer defaults (favorable).
        # This dominates the Merton risk-free drift effect empirically.
        if var_name in ("unemployment_rate", "interest_rate"):
            z += w * (x - mu) / sigma
        else:
            z -= w * (x - mu) / sigma

    return z


# ──────────────────────────────────────────────
# NORMES REGLEMENTAIRES (CRR3, IFRS 9 B5.5.25, NSFR)
# ──────────────────────────────────────────────

# CRR3 Art. 124-125 : schedule RW residentiel par tranche LTV
_LTV_RW_SCHEDULE: Dict[float, float] = {
    0.50: 0.20, 0.60: 0.25, 0.70: 0.30,
    0.80: 0.35, 0.90: 0.50, 1.00: 0.70,
}

# CRR3 Art. 242-270 : SEC-SA RW par tranche de securitisation
_SEC_SA_RW: Dict[str, float] = {
    "senior_sts": 0.20, "senior_non_sts": 0.60,
    "mezzanine": 1.00, "junior": 2.50,
}


def mortgage_rw_blended(ltv_distribution: Dict[float, float]) -> float:
    """RW pondere par distribution LTV (CRR3 Art. 124-125).

    Args:
        ltv_distribution: Dict {ltv_bucket: proportion}.
            Ex: {0.50: 0.15, 0.60: 0.25, ...}

    Returns:
        RW pondere (ex: ~0.33 pour portefeuille FR/EU representatif).
    """
    rw_total = 0.0
    weight_total = 0.0
    for ltv, proportion in ltv_distribution.items():
        rw = _LTV_RW_SCHEDULE.get(ltv, 0.70)  # fallback to highest RW
        rw_total += rw * proportion
        weight_total += proportion
    return rw_total / max(weight_total, 1e-10)


def securitisation_rw_blended(mix: Dict[str, float]) -> float:
    """RW pondere par mix de tranches SEC-SA (CRR3 Art. 242-270).

    Args:
        mix: Dict {tranche_name: proportion}.
            Ex: {"senior_sts": 0.60, "senior_non_sts": 0.25, "mezzanine": 0.15}

    Returns:
        RW pondere (ex: ~0.42 pour mix type post-GFC).
    """
    rw_total = 0.0
    weight_total = 0.0
    for tranche, proportion in mix.items():
        rw = _SEC_SA_RW.get(tranche, 1.00)  # fallback to mezzanine RW
        rw_total += rw * proportion
        weight_total += proportion
    return rw_total / max(weight_total, 1e-10)


def effective_rw(profile: AssetClassProfile) -> float:
    """RW effectif tenant compte des granularites reglementaires.

    Dispatch :
        - retail_mortgage → blended LTV (CRR3 Art. 124-125)
        - structured_products → blended SEC-SA (CRR3 Art. 242-270)
        - autres → profile.rw_crr3 (flat)

    Args:
        profile: AssetClassProfile.

    Returns:
        RW effectif.
    """
    if profile.ltv_distribution is not None:
        return mortgage_rw_blended(profile.ltv_distribution)
    if profile.securitisation_mix is not None:
        return securitisation_rw_blended(profile.securitisation_mix)
    return profile.rw_crr3


def compute_nsfr(
    allocation_weights: Dict[str, float],
    total_ead: float,
    asf_coverage: float = 0.90,
) -> Dict[str, object]:
    """Calcule le Net Stable Funding Ratio (NSFR Basel III).

    NSFR = ASF / RSF >= 100%.
    ASF = total_ead × asf_coverage (depots stables + fonds propres).
    RSF = sum(w_i × ead_i × rsf_weight_i).

    Args:
        allocation_weights: Dict {class_name: weight}.
        total_ead: EAD total du bilan.
        asf_coverage: Part stable des financements (depots + CET1).

    Returns:
        Dict avec nsfr_ratio, asf, rsf, compliant.
    """
    asf = total_ead * asf_coverage
    rsf = 0.0
    for class_name, weight in allocation_weights.items():
        profile = ASSET_CLASS_MAP.get(class_name)
        if profile is not None:
            rsf += weight * total_ead * profile.rsf_weight
    nsfr_ratio = asf / max(rsf, 1e-10)
    return {
        "nsfr_ratio": round(nsfr_ratio, 4),
        "asf": round(asf, 0),
        "rsf": round(rsf, 0),
        "compliant": nsfr_ratio >= 1.00,
    }


def compute_lcr(
    allocation_weights: Dict[str, float],
    total_ead: float,
    tnco_fraction: float = 0.15,
) -> Dict[str, object]:
    """Calcule le Liquidity Coverage Ratio (LCR Basel III).

    LCR = HQLA / TNCO >= 100%.
    HQLA = sum(w_i * EAD * (1 - haircut_i)) pour classes HQLA eligible.
    Haircut: L1=0%, L2A=15%, L2B=50%. Cap L2 a 40% du HQLA total.
    TNCO = total_ead * tnco_fraction (simplifie).

    Args:
        allocation_weights: Dict {class_name: weight}.
        total_ead: EAD total du bilan.
        tnco_fraction: Part des sorties nettes de tresorerie.

    Returns:
        Dict avec lcr_ratio, hqla_total, tnco, hqla_by_level.
    """
    _HAIRCUT = {"L1": 0.0, "L2A": 0.15, "L2B": 0.50}
    # Map hqla_level int to string
    _LEVEL_MAP = {1: "L1", 2: "L2A", 3: "L2B"}
    hqla_by_level = {"L1": 0.0, "L2A": 0.0, "L2B": 0.0}
    for name, weight in allocation_weights.items():
        ac = ASSET_CLASS_MAP.get(name)
        if ac is not None and ac.hqla_eligible:
            level = _LEVEL_MAP.get(ac.hqla_level, "L2B")
            hqla_by_level[level] += weight * total_ead * (1 - _HAIRCUT[level])
    # Cap L2 a 40% du total HQLA (Basel III Art. 12)
    hqla_l1 = hqla_by_level["L1"]
    hqla_l2_raw = hqla_by_level["L2A"] + hqla_by_level["L2B"]
    hqla_l2 = min(hqla_l2_raw, 0.40 / 0.60 * hqla_l1) if hqla_l1 > 0 else hqla_l2_raw
    hqla_total = hqla_l1 + hqla_l2
    tnco = total_ead * tnco_fraction
    lcr = hqla_total / max(tnco, 1e-6)
    return {
        "lcr_ratio": round(lcr, 4),
        "hqla_total": round(hqla_total, 0),
        "tnco": round(tnco, 0),
        "hqla_by_level": hqla_by_level,
    }


def compute_irrbb_eve(
    allocation_weights: Dict[str, float],
    total_ead: float,
    cet1_capital: float,
    rate_shock: float = 0.02,
    eve_limit: float = 0.15,
    alm_hedge_ratio: Optional[float] = None,
) -> Dict[str, object]:
    """IRRBB EVE (Basel III Avr 2016, EBA/GL/2022/14).

    EVE shock = total_ead * effective_duration * rate_shock
    effective_duration = weighted_duration * (1 - alm_hedge_ratio)
    Compliant si EVE_shock <= eve_limit * cet1_capital.

    The alm_hedge_ratio reflects the bank's ALM hedging programme
    (interest rate swaps, futures). EBA 2024 data shows large EU banks
    hedge 30-50% of their duration exposure. Default from BASEL_CONFIG (0.30).

    Args:
        allocation_weights: Dict {class_name: weight}.
        total_ead: EAD total du bilan.
        cet1_capital: Capital CET1 (EUR).
        rate_shock: Choc de taux parallele (+200bp standard).
        eve_limit: Limite EVE en % du Tier 1 (15% standard).
        alm_hedge_ratio: Fraction de la duration couverte par ALM (0.30 = 30%).

    Returns:
        Dict avec eve_shock, eve_limit_eur, weighted_duration, compliant, eve_ratio.
    """
    if alm_hedge_ratio is None:
        alm_hedge_ratio = BASEL_CONFIG.alm_hedge_ratio

    weighted_duration = 0.0
    for class_name, weight in allocation_weights.items():
        profile = ASSET_CLASS_MAP.get(class_name)
        if profile is not None:
            weighted_duration += weight * profile.duration

    effective_duration = weighted_duration * (1.0 - alm_hedge_ratio)
    eve_shock = total_ead * effective_duration * rate_shock
    eve_limit_eur = eve_limit * cet1_capital
    eve_ratio = eve_shock / max(eve_limit_eur, 1e-10)
    compliant = bool(eve_shock <= eve_limit_eur)

    return {
        "eve_shock": round(eve_shock, 0),
        "eve_limit_eur": round(eve_limit_eur, 0),
        "weighted_duration": round(weighted_duration, 4),
        "effective_duration": round(effective_duration, 4),
        "compliant": compliant,
        "eve_ratio": round(eve_ratio, 4),
    }


def staging_cliff_ecl(
    pd_cond: float,
    lgd: float,
    ead: float,
    tenor: float,
    pd_orig: float,
    sicr_threshold: float = 3.0,
    exempt_from_staging: bool = False,
) -> float:
    """ECL avec effet de cliff staging (S1 12m vs S2 lifetime).

    La proportion du portefeuille en Stage 2 est estimee a partir de
    la PD conditionnelle vs le seuil SICR (PD_cond / PD_orig > threshold).

    Si exempt_from_staging=True (IFRS 9 B5.5.25, ex: souverain AAA/AA),
    retourne uniquement l'ECL 12 mois (pas de Stage 2).

    ECL = EAD * LGD * [(1-p_S2) * PD_cond * 1y + p_S2 * PD_cond * tenor]

    Args:
        pd_cond: PD conditionnelle (post-Vasicek).
        lgd: Loss Given Default.
        ead: Exposure At Default.
        tenor: Maturite moyenne (annees).
        pd_orig: PD a l'origination (pour SICR).
        sicr_threshold: Multiplicateur SICR (defaut 3x).
        exempt_from_staging: Si True, ECL 12m uniquement (IFRS 9 B5.5.25).

    Returns:
        ECL en valeur absolue.
    """
    # IFRS 9 B5.5.25 : low credit risk exemption (sovereign AAA/AA)
    if exempt_from_staging:
        return max(0.0, pd_cond * lgd * ead * 1.0)

    # Proportion en Stage 2 : soft transition
    if pd_orig > 0:
        pd_ratio = pd_cond / pd_orig
        # Logistic smooth transition around threshold
        p_stage2 = 1.0 / (1.0 + np.exp(-5.0 * (pd_ratio / sicr_threshold - 1.0)))
    else:
        p_stage2 = 0.0

    # Stage 1 : PD 12 mois (approximation : PD_cond ~ PD_1y pour faibles PD)
    ecl_s1 = pd_cond * lgd * ead * 1.0

    # Stage 2 : PD lifetime = 1 - (1-PD_cond)^tenor
    pd_lifetime = 1.0 - (1.0 - pd_cond) ** tenor
    ecl_s2 = pd_lifetime * lgd * ead

    ecl = (1.0 - p_stage2) * ecl_s1 + p_stage2 * ecl_s2
    return max(0.0, ecl)


def _compute_fvtpl_mtm_loss(
    row: dict,
    macro_params: Dict[str, float],
    profile: AssetClassProfile,
) -> float:
    """Compute mark-to-market P&L for FVTPL instruments (equities).

    VSTOXX-based model (Bekaert & Hoerova 2014, Whaley 2009):
    The VSTOXX ("fear index") is translated into monetary information:
    - VSTOXX high (peur) → marche baisse → perte (valeur positive)
    - VSTOXX bas (confiance) → marche monte → gain (valeur negative)
    - VSTOXX = 18 (normal) → neutre → zero

    Calibration targets (historical):
        GFC 2008:   VSTOXX ~75, EURO STOXX 50 -45%  → perte ~32-41%
        COVID 2020: VSTOXX ~80, EURO STOXX 50 -38%   → perte ~41%
        Sov 2011:   VSTOXX ~45, EURO STOXX 50 -25%   → perte ~18%
        Normal:     VSTOXX ~18, EURO STOXX 50 ~+8%   → neutre
        Bull 2017:  VSTOXX ~12, EURO STOXX 50 +12%   → gain ~6%

    Args:
        row: Balance sheet row dict.
        macro_params: Current macro parameters.
        profile: AssetClassProfile.

    Returns:
        MTM P&L: positive = loss, negative = gain.
        Capped at [-30%, +90%] of EAD.
    """
    ead = row["ead_total"]
    base_gdp = SCENARIO_BASE.gdp_growth
    base_unemp = SCENARIO_BASE.unemployment_rate

    gdp = macro_params.get("gdp_growth", macro_params.get("gdp_pct", base_gdp))
    unemp = macro_params.get("unemployment_rate", base_unemp)

    # ── Step 1: Macro → stress intensity ──
    # GDP: primary equity driver (Bloom 2009, Baker-Bloom-Davis 2016)
    # Unemployment: secondary signal (labor market → earnings)
    delta_gdp = -(gdp - base_gdp) / 1.8    # normalized, positive = adverse
    delta_unemp = (unemp - base_unemp) / 1.5
    stress = np.clip(0.60 * delta_gdp + 0.40 * delta_unemp, -2.0, 4.0)

    # ── Step 2: VSTOXX estimation ──
    # Asymmetric: vol spikes fast in stress, compresses slowly in calm
    # Floor at 12% (VSTOXX never below ~11% historically)
    _VSTOXX_BASE = 18.0   # Long-run VSTOXX median (ECB SDW 2005-2024)
    _VSTOXX_CAP = 80.0    # GFC/COVID peak
    _KAPPA = 0.45          # Amplification (calibrated: stress=2.9 → VSTOXX~66)

    if stress > 0:
        vstoxx = _VSTOXX_BASE * np.exp(_KAPPA * stress)
    else:
        vstoxx = max(12.0, _VSTOXX_BASE * np.exp(0.20 * stress))
    vstoxx = min(vstoxx, _VSTOXX_CAP)

    # ── Step 3: VSTOXX → MTM impact (exponentiel) ──
    # Loi exponentielle : l'indice de peur traduit en info monetaire.
    # Les krachs s'accelerent (panic, margin calls, cascades — Bouchaud
    # & Potters 2003, Gabaix 2012) et les rallyes aussi (momentum, FOMO).
    # Formulation: impact = A × (exp(k × |dev|) - 1) / (exp(k × max_dev) - 1)
    _MAX_LOSS = 0.41    # GFC: VSTOXX=80 → haircut 41% (EBA 2023)
    _MAX_GAIN = 0.06    # Bull: VSTOXX=12 → appreciation 6% (au-dela de l'ERP)
    _K_LOSS = 0.035     # Convexite pertes (moderee a VSTOXX=30, forte a 70+)
    _K_GAIN = 0.20      # Convexite gains (VSTOXX range [12,18] = 6 pts)

    deviation = vstoxx - _VSTOXX_BASE
    if deviation >= 0:
        max_dev = _VSTOXX_CAP - _VSTOXX_BASE        # 62
        norm = np.exp(_K_LOSS * max_dev) - 1.0       # ~7.76
        mtm_impact = _MAX_LOSS * (np.exp(_K_LOSS * deviation) - 1.0) / norm
    else:
        max_dev_gain = _VSTOXX_BASE - 12.0           # 6 (floor historique)
        norm = np.exp(_K_GAIN * max_dev_gain) - 1.0  # ~2.32
        mtm_impact = -_MAX_GAIN * (np.exp(_K_GAIN * abs(deviation)) - 1.0) / norm

    # ── Step 4: portfolio beta ──
    beta = profile.macro_sensitivities.get("gdp_growth", 1.5) / 1.5

    mtm_pnl = ead * mtm_impact * beta
    # Cap: max loss 90%, max gain 30% of EAD
    return float(np.clip(mtm_pnl, -0.30 * ead, 0.90 * ead))


def compute_balance_sheet_ecl(
    df_balance_sheet,
    macro_params: Dict[str, float],
    carbon_price_shock: float = 0.0,
    physical_severity: float = 0.0,
):
    """Calcule l'ECL pour les 10 classes d'actifs (Level 1/2/3).

    Pipeline :
        1. Pour chaque classe : climate-adjusted PD → Vasicek PD_cond → staging cliff ECL
        2. Multi-scenario : Base (50%) + Adverse (25%) + Favorable (25%)
        3. Contagion appliquee en aval (dans ContagionEngine)

    Args:
        df_balance_sheet: DataFrame (10 rows, cols: asset_class, ead_total, ...).
            Accepts both Polars and Pandas DataFrames.
        macro_params: Dict des valeurs macro courantes.
        carbon_price_shock: Choc carbone normalise.
        physical_severity: Severite physique.

    Returns:
        pl.DataFrame enrichi avec colonnes ECL.
    """
    import polars as pl
    from ifrs9_cockpit.utils.frame_compat import to_polars

    df_bs = to_polars(df_balance_sheet)
    results = []

    for row in df_bs.iter_rows(named=True):
        ac_name = row["asset_class"]
        profile = ASSET_CLASS_MAP[ac_name]
        ead = row["ead_total"]

        # ── FVTPL dispatch: no ECL, compute MTM loss instead ──
        if getattr(profile, "accounting_treatment", "amortised_cost") == "fvtpl":
            mtm_loss = _compute_fvtpl_mtm_loss(row, macro_params, profile)
            row_pd = row.get("pd_base", profile.pd_base)
            row_lgd = row.get("lgd_base", profile.lgd_base)
            results.append({
                "asset_class": ac_name,
                "label": profile.label,
                "category": profile.category,
                "ead_total": ead,
                "pd_base": row_pd,
                "pd_climate": row_pd,  # no climate adj for FVTPL
                "lgd_base": row_lgd,
                "tenor": profile.tenor,
                "rho": profile.asset_correlation,
                "rw_crr3": row.get("rw_crr3", effective_rw(profile)),
                "ecl_weighted": 0.0,  # FVTPL: no ECL
                "mtm_loss": mtm_loss,
                "ecl_ead_ratio": 0.0,
                "rwa": ead * row.get("rw_crr3", effective_rw(profile)),
            })
            continue

        # Scenarios ECL (Base, Adverse, Favorable)
        ecl_weighted = 0.0
        scenario_details = {}

        for scenario in SCENARIOS:
            # Macro params pour ce scenario
            sc_macro = {
                "gdp_growth": scenario.gdp_growth,
                "unemployment_rate": scenario.unemployment_rate,
                "interest_rate": scenario.interest_rate,
                "hpi_growth": scenario.hpi_growth,
                "inflation_rate": scenario.inflation_rate,
            }

            # Merge avec overrides utilisateur (pour Base seulement, les scenarios
            # Adverse/Favorable sont relatifs)
            if scenario.name == "Base":
                for k, v in macro_params.items():
                    if k in sc_macro:
                        sc_macro[k] = v

            # 1. Climate-adjusted PD (use row-level if available, fallback to profile)
            row_pd = row.get("pd_base", profile.pd_base)
            pd_climate = climate_adjusted_pd(
                pd_base=max(row_pd, profile.input_floor_pd),
                physical_risk=profile.physical_risk,
                transition_risk=profile.transition_risk,
                green_capex_ratio=profile.green_capex_ratio,
                scope3_exposure=profile.scope3_exposure,
                carbon_price_shock=carbon_price_shock,
                physical_severity=physical_severity,
            )

            # 2. Macro → Z
            z = macro_to_z(sc_macro, profile.macro_sensitivities)
            # Adverse/Favorable : add scenario shocks to Z
            if scenario.name == "Adverse":
                z += 1.0  # +1 sigma adverse
            elif scenario.name == "Favorable":
                z -= 0.5  # -0.5 sigma favorable
            # Cap Z to ±4 sigma (99.997th percentile)
            z = np.clip(z, -4.0, 4.0)

            # 3. Vasicek conditional PD
            pd_cond = vasicek_conditional_pd(pd_climate, profile.asset_correlation, z)

            # 4. LGD (use row-level if available, fallback to profile)
            row_lgd = row.get("lgd_base", profile.lgd_base)
            lgd = max(row_lgd, profile.input_floor_lgd)
            if scenario.name == "Adverse":
                lgd = min(lgd * 1.15, 0.95)  # +15% LGD downturn

            # 5. Staging cliff ECL (IFRS 9 B5.5.25 exemption for sovereign)
            ecl_scenario = staging_cliff_ecl(
                pd_cond=pd_cond,
                lgd=lgd,
                ead=ead,
                tenor=profile.tenor,
                pd_orig=row_pd,
                exempt_from_staging=profile.exempt_from_staging,
            )

            ecl_weighted += scenario.weight * ecl_scenario
            scenario_details[f"ecl_{scenario.name.lower()}"] = ecl_scenario
            scenario_details[f"pd_cond_{scenario.name.lower()}"] = pd_cond

        results.append({
            "asset_class": ac_name,
            "label": profile.label,
            "category": profile.category,
            "ead_total": ead,
            "pd_base": row_pd,
            "pd_climate": pd_climate,
            "lgd_base": row_lgd,
            "tenor": profile.tenor,
            "rho": profile.asset_correlation,
            "rw_crr3": row.get("rw_crr3", effective_rw(profile)),
            "ecl_weighted": ecl_weighted,
            "ecl_ead_ratio": ecl_weighted / ead if ead > 0 else 0.0,
            "rwa": ead * row.get("rw_crr3", effective_rw(profile)),
            **scenario_details,
        })

    return pl.DataFrame(results)
