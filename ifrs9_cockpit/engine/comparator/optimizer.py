"""Mixin BL-CVaR asymetrique et optimisation d'allocation N classes.

Architecture endogene (v3 — market impact logarithmique) :
    Phase 1 : BL-CVaR pur (zero floor/cap, zero penalite artificielle) :
        - CIR par classe (structure de couts)
        - Compression de spread logarithmique (Kyle 1985, Almgren-Chriss 2001)
        - Ledoit-Wolf shrinkage (covariance stabilisee)
        La diversification emerge naturellement du market impact :
        quand la banque consomme une part significative du marche,
        le spread d'achat se compresse → le RAROC effectif baisse.
    Phase 2 : Normes reglementaires (ajustement sequentiel minimal) :
        - CET1 >= 13%
        - LCR >= 100%
        - NSFR >= 100%
"""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple

from ifrs9_cockpit.config import (
    ASSET_CLASSES,
    ASSET_CLASS_MAP,
    ASSET_CLASS_NAMES,
    BASEL_CONFIG,
    SECTORS,
    SCENARIO_BASE,
    _EXPERT_CORR,
    SECTOR_NAMES,
    RISK_APPETITE_CONFIG,
)
from ifrs9_cockpit.engine.balance_sheet_ecl import effective_rw, compute_nsfr, compute_lcr, compute_irrbb_eve




class OptimizerMixin:
    """Mixin fournissant les equations asymetriques BL-CVaR et l'optimisation."""

    # ──────────────────────────────────────────────
    # ASYMMETRIC BL-CVaR EQUATIONS
    # ──────────────────────────────────────────────

    @staticmethod
    def _stress_intensity(r_c_spot: float, r_p_spot: float, r_neutral: Optional[float] = None) -> float:
        """CRO-pessimist stress: max of credit and PE channels.

        s_i = -r_i / r_neutral. Take the more adverse (higher s).
        r_neutral auto-calibrated: mean of positive RAROC channels.
        """
        if r_neutral is None:
            positives = [r for r in (r_c_spot, r_p_spot) if r > 0]
            r_neutral = float(np.mean(positives)) if positives else 0.07
        s_credit = -r_c_spot / max(r_neutral, 1e-10)
        s_pe = -r_p_spot / max(r_neutral, 1e-10)
        return max(s_credit, s_pe)

    @staticmethod
    def _asymmetric_illiquidity(s: float, base: float = 0.005, scale: float = 0.025, gamma: float = 1.5) -> float:
        """Convex illiquidity premium.

        illiq(s) = base + scale * max(0, s)^gamma.
        Flat in expansion (~base), spikes non-linearly in crisis.
        Ref: Ang, Papanikolaou & Westerfield (2014).
        """
        return base + scale * max(0.0, s) ** gamma

    @staticmethod
    def _asymmetric_vol_multiplier(s: float, alpha_down: float = 0.40, alpha_up: float = 0.10) -> float:
        """GJR-GARCH vol multiplier.

        mult(s) = 1 + alpha_down * max(0, s) - alpha_up * max(0, -s).
        4:1 asymmetry: vol increases faster in downturns.
        Ref: Glosten, Jagannathan & Runkle (1993).
        """
        return 1.0 + alpha_down * max(0.0, s) - alpha_up * max(0.0, -s)

    @staticmethod
    def _asymmetric_pe_band(s: float, pe_calm: float = 0.15, pe_stress: float = 0.05, pe_min: float = 0.005, k: float = 3.0) -> Tuple[float, float]:
        """Sigmoid band compression for PE allocation bounds.

        pe_max(s) = pe_stress + (pe_calm - pe_stress) * (1 - sigmoid(k*s)).
        Returns (pe_min, pe_max_s).
        Ref: Ang & Bekaert (2004).
        """
        sigmoid_val = 1.0 / (1.0 + np.exp(-k * s))
        pe_max_s = pe_stress + (pe_calm - pe_stress) * (1.0 - sigmoid_val)
        return (pe_min, float(np.clip(pe_max_s, pe_min, pe_calm)))

    @staticmethod
    def _bl_confidence(s: float, k_credit: float = 1.5, k_pe: float = 2.5) -> Tuple[float, float]:
        """BL confidence sigmoid -- stress-dependent.

        c(s) = 0.30 + 0.65 * sigmoid(-k*s), bounded [0.30, 0.95].
        PE confidence drops 1.7x faster than credit in crisis.
        Ref: Ang & Bekaert (2002).

        Returns:
            (conf_credit, conf_pe).
        """
        def _conf(k: float) -> float:
            sig = 1.0 / (1.0 + np.exp(k * s))  # sigmoid(-k*s)
            return float(np.clip(0.30 + 0.65 * sig, 0.30, 0.95))
        return (_conf(k_credit), _conf(k_pe))

    # ──────────────────────────────────────────────
    # MATRICE DE CORRELATION NxN (MACRO SENSITIVITIES)
    # ──────────────────────────────────────────────

    @staticmethod
    def _ledoit_wolf_shrink(corr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Ledoit-Wolf single-factor shrinkage pour stabiliser la correlation NxN.

        Target F = constant-correlation matrix (average off-diagonal).
        Reduces estimation error by 20-50% (Ledoit & Wolf 2004).

        Args:
            corr: Raw correlation matrix (N x N).

        Returns:
            (shrunk_corr, lambda_lw): Shrunk matrix and shrinkage intensity.
        """
        n = len(corr)
        # Target: constant-correlation matrix
        avg_corr = (corr.sum() - n) / max(n * (n - 1), 1)
        F = np.full_like(corr, avg_corr)
        np.fill_diagonal(F, 1.0)
        # Shrinkage intensity (Oracle Approximating formula, simplified)
        delta = corr - F
        num = np.sum(delta ** 2)
        denom = max(num, 1e-10)  # self-consistent: lambda = num / denom
        # Use Frobenius norm ratio as proxy
        lambda_lw = np.clip(num / (num + np.sum((corr - np.eye(n)) ** 2) + 1e-10), 0.05, 0.50)
        shrunk = (1 - lambda_lw) * corr + lambda_lw * F
        # Ensure still a valid correlation matrix
        np.fill_diagonal(shrunk, 1.0)
        shrunk = (shrunk + shrunk.T) / 2.0
        return shrunk, float(lambda_lw)

    @staticmethod
    def _build_corr_matrix() -> Tuple[np.ndarray, float]:
        """Construit la matrice de correlation NxN entre classes d'actifs.

        Methode : similarite cosinus des vecteurs macro_sensitivities
        (5 dimensions : gdp, unemployment, interest, hpi, inflation).

        Regularisation :
            - Diagonale forcee a 1.0
            - Clip dans [-0.90, 0.95] pour eviter la singularite
            - Ledoit-Wolf shrinkage (stabilisation)
            - Nearest PSD si necessaire (Higham)

        Returns:
            (corr, lambda_lw): Matrice NxN symetrique definie positive
            et intensite du shrinkage.
        """
        n = len(ASSET_CLASSES)
        macro_keys = ["gdp_growth", "unemployment_rate", "interest_rate",
                       "hpi_growth", "inflation_rate"]

        # Build sensitivity matrix (N x 5)
        S = np.zeros((n, len(macro_keys)))
        for i, ac in enumerate(ASSET_CLASSES):
            for j, key in enumerate(macro_keys):
                S[i, j] = ac.macro_sensitivities.get(key, 0.0)

        # Cosine similarity
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        S_norm = S / norms
        corr = S_norm @ S_norm.T

        # Flight-to-quality: sovereign inversely correlated with risky assets
        # BIS WP 2014, ECB FS Review 2012: sovereign-credit correlation flips
        # negative during crisis (doom loop / safe haven)
        class_names = [ac.name for ac in ASSET_CLASSES]
        _FTQ_ADJUSTMENT = {
            ("sovereign", "corporate_loans"): -0.40,
            ("sovereign", "consumer_credit"): -0.30,
            ("sovereign", "private_equity"): -0.35,
            ("sovereign", "project_finance"): -0.25,
            # New classes: flight-to-quality vs equities, corp bonds
            ("sovereign", "equities"): -0.35,
            ("sovereign", "corporate_bonds"): -0.30,
            # Cross-asset reinforcement
            ("equities", "corporate_loans"): +0.10,
            ("derivatives_cva", "corporate_loans"): +0.10,
        }
        for (a, b), adj in _FTQ_ADJUSTMENT.items():
            i_a = class_names.index(a) if a in class_names else -1
            i_b = class_names.index(b) if b in class_names else -1
            if i_a >= 0 and i_b >= 0:
                corr[i_a, i_b] += adj
                corr[i_b, i_a] += adj

        # Regularize
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -0.90, 0.95)
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)

        # Ledoit-Wolf shrinkage (stabilise la covariance)
        corr, lambda_lw = OptimizerMixin._ledoit_wolf_shrink(corr)

        # Ensure PSD (nearest correlation matrix, simple eigenvalue clipping)
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-6)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Re-normalize to correlation matrix
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)

        return corr, lambda_lw

    # Backward-compat alias (returns just the matrix for old callers)
    @staticmethod
    def _build_corr_10x10() -> np.ndarray:
        corr, _ = OptimizerMixin._build_corr_matrix()
        return corr

    # ──────────────────────────────────────────────
    # SPREAD COMPRESSION (Kyle 1985, Almgren-Chriss 2001)
    # ──────────────────────────────────────────────

    @staticmethod
    def _spread_compression(w: np.ndarray, class_names: list,
                            total_ead: float, mu_base: np.ndarray) -> np.ndarray:
        """Compression de spread par market impact logarithmique.

        Modele Kyle (1985) / Almgren-Chriss (2001) adapte au portefeuille :
        quand la banque consomme une part significative du marche,
        le spread d'achat se compresse → le RAROC effectif baisse.

        compression_i = -ln(1 - share_i)  avec share_i = (w_i * total_ead) / market_capacity_i
        mu_effective_i = mu_base_i / (1 + compression_i)

        Proprietes :
            - A 1% du marche → compression ~0.01 → quasi-zero
            - A 30% → compression ~0.36 → marge baisse de 26%
            - A 50% → compression ~0.69 → marge baisse de 41%
            - A 80% → compression ~1.61 → marge baisse de 62%
            - A 100% → compression → infini → impossible

        Zero parametre au-dela de la taille du marche (donnee reelle).

        Args:
            w: Poids d'allocation (N,).
            class_names: Noms des classes d'actifs.
            total_ead: Exposition totale du portefeuille (EUR).
            mu_base: RAROC de base (N,).

        Returns:
            mu_effective: RAROC apres compression de spread (N,).
        """
        mu_eff = np.copy(mu_base)
        for i, name in enumerate(class_names):
            ac = ASSET_CLASS_MAP[name]
            market_cap = ac.market_capacity_eur
            share = (w[i] * total_ead) / market_cap
            share = np.clip(share, 0.0, 0.95)  # cap numerique a 95%
            compression = -np.log(1.0 - share)
            mu_eff[i] = mu_base[i] / (1.0 + compression)
        return mu_eff

    # ──────────────────────────────────────────────
    # LCR / NSFR REGULATORY ADJUSTMENTS (Phase 2)
    # ──────────────────────────────────────────────

    @staticmethod
    def _compute_lcr_ratio(w: np.ndarray, class_names: list, total_exposure: float) -> float:
        """Compute LCR ratio inline."""
        alloc = {name: float(w[i]) for i, name in enumerate(class_names)}
        return compute_lcr(alloc, total_exposure)["lcr_ratio"]

    @staticmethod
    def _adaptive_step(deficit: float) -> float:
        """Pas proportionnel au deficit. Zero parametre arbitraire."""
        return np.clip(abs(deficit) * 0.10, 0.002, 0.02)

    @staticmethod
    def _enforce_market_caps(w: np.ndarray, class_names: list, n: int) -> np.ndarray:
        """Enforce per-class market capacity cap. Redistribute excess proportionally.

        Applied post Phase 2 to prevent regulatory adjustments (NSFR, IRRBB)
        from pushing any class beyond its market capacity share.

        Args:
            w: Allocation weights (N,).
            class_names: Asset class names.
            n: Number of classes.

        Returns:
            Capped and renormalized weights.
        """
        market_caps = np.array([ASSET_CLASS_MAP[name].market_capacity_eur for name in class_names])
        max_w = market_caps / market_caps.sum()
        for _ in range(10):
            capped = False
            for i in range(n):
                if w[i] > max_w[i]:
                    excess = w[i] - max_w[i]
                    w[i] = max_w[i]
                    uncapped = [j for j in range(n) if j != i and w[j] < max_w[j]]
                    if uncapped:
                        unc_total = sum(w[j] for j in uncapped)
                        for j in uncapped:
                            w[j] += excess * (w[j] / max(unc_total, 1e-10))
                    capped = True
            if not capped:
                break
            w = np.maximum(w, 0)
            w /= w.sum()
        return w

    @staticmethod
    def _adjust_for_lcr(w: np.ndarray, class_names: list, total_exposure: float, target: float = 1.0) -> np.ndarray:
        """Ajustement minimal : +HQLA L1, -non-HQLA jusqu'a LCR >= target.

        Convergence adaptive: pas proportionnel au deficit LCR.
        """
        w = w.copy()
        # L1 HQLA (0% haircut) — most effective for LCR
        l1_idx = [i for i, name in enumerate(class_names)
                  if ASSET_CLASS_MAP[name].hqla_eligible and ASSET_CLASS_MAP[name].hqla_level == 1]
        non_hqla_idx = [i for i, name in enumerate(class_names) if not ASSET_CLASS_MAP[name].hqla_eligible]
        for _ in range(100):
            lcr = OptimizerMixin._compute_lcr_ratio(w, class_names, total_exposure)
            if lcr >= target:
                break
            transfer = OptimizerMixin._adaptive_step(target - lcr)
            non_hqla_total = sum(w[j] for j in non_hqla_idx)
            if non_hqla_total < transfer:
                break
            # Prioritize L1 HQLA (sovereign) for maximum LCR impact
            target_idx = l1_idx if l1_idx else [i for i, name in enumerate(class_names) if ASSET_CLASS_MAP[name].hqla_eligible]
            target_total = sum(w[j] for j in target_idx)
            if target_total < 1e-6:
                target_total = 1e-6
            for j in non_hqla_idx:
                w[j] -= transfer * (w[j] / non_hqla_total)
            for j in target_idx:
                w[j] += transfer * (w[j] / target_total)
        w = np.maximum(w, 0)
        w /= w.sum()
        return w

    @staticmethod
    def _adjust_for_nsfr(w: np.ndarray, class_names: list, total_exposure: float, target: float = 1.0) -> np.ndarray:
        """Ajustement minimal : reduire classes a haut RSF, augmenter low-RSF.

        Convergence adaptive: pas proportionnel au deficit NSFR.
        """
        w = w.copy()
        for _ in range(100):
            alloc = {name: float(w[i]) for i, name in enumerate(class_names)}
            nsfr = compute_nsfr(alloc, total_exposure)["nsfr_ratio"]
            if nsfr >= target:
                break
            transfer = OptimizerMixin._adaptive_step(target - nsfr)
            # Trier par RSF decroissant, transferer vers low-RSF
            rsf_weights = [(i, ASSET_CLASS_MAP[name].rsf_weight) for i, name in enumerate(class_names)]
            rsf_weights.sort(key=lambda x: -x[1])
            high_rsf = [i for i, rsf in rsf_weights[:4] if w[i] > 0.005]
            low_rsf = [i for i, rsf in rsf_weights[-4:]]
            if not high_rsf or not low_rsf:
                break
            for i in high_rsf:
                w[i] -= transfer / len(high_rsf)
            # Distribute proportionally to remaining capacity (cap - current)
            # so classes with more headroom absorb more transfer.
            market_caps_arr = np.array([ASSET_CLASS_MAP[class_names[j]].market_capacity_eur
                                        for j in low_rsf])
            max_w_arr = market_caps_arr / np.array(
                [ASSET_CLASS_MAP[name].market_capacity_eur for name in class_names]).sum()
            remaining = np.maximum(max_w_arr - np.array([w[j] for j in low_rsf]), 0)
            rem_total = remaining.sum()
            if rem_total > 1e-10:
                for k, j in enumerate(low_rsf):
                    w[j] += transfer * (remaining[k] / rem_total)
            else:
                per_low = transfer / len(low_rsf)
                for j in low_rsf:
                    w[j] += per_low
        w = np.maximum(w, 0)
        w /= w.sum()
        return w

    # ──────────────────────────────────────────────
    # IRRBB ADJUSTMENT (Phase 2, after NSFR)
    # ──────────────────────────────────────────────

    @staticmethod
    def _adjust_for_irrbb(
        w: np.ndarray,
        class_names: list,
        total_exposure: float,
        cet1_capital: float,
    ) -> np.ndarray:
        """Ajustement minimal : transferer des classes haute-duration vers basse-duration.

        Meme pattern que _adjust_for_lcr / _adjust_for_nsfr.
        Si IRRBB EVE non-compliant, transfere 0.5% par iteration
        des classes haute-duration (>2Y) vers basse-duration (<=2Y).
        Preserve HQLA L1 (sovereign) to avoid LCR conflict — reduce non-HQLA
        high-duration first, sovereign only as last resort.
        Max 80 iterations.
        """
        w = w.copy()
        # High-duration non-HQLA-L1 first (avoid sovereign/LCR conflict)
        high_dur_non_l1 = [i for i, name in enumerate(class_names)
                           if ASSET_CLASS_MAP[name].duration > 2.0
                           and not (ASSET_CLASS_MAP[name].hqla_eligible
                                    and ASSET_CLASS_MAP[name].hqla_level == 1)]
        high_dur_l1 = [i for i, name in enumerate(class_names)
                       if ASSET_CLASS_MAP[name].duration > 2.0
                       and ASSET_CLASS_MAP[name].hqla_eligible
                       and ASSET_CLASS_MAP[name].hqla_level == 1]
        # Market capacity caps for recipient filtering
        market_caps = np.array([ASSET_CLASS_MAP[name].market_capacity_eur for name in class_names])
        max_w = market_caps / market_caps.sum()
        # All low-duration classes (excl. duration=0 FVTPL like equities)
        all_low_dur = [i for i, name in enumerate(class_names)
                       if 0 < ASSET_CLASS_MAP[name].duration <= 2.0]
        if not all_low_dur:
            return w

        for _phase, high_idx in enumerate([high_dur_non_l1, high_dur_l1]):
            if not high_idx:
                continue
            for _ in range(100):
                alloc = {name: float(w[i]) for i, name in enumerate(class_names)}
                irrbb = compute_irrbb_eve(alloc, total_exposure, cet1_capital)
                if irrbb["compliant"]:
                    break
                transfer = OptimizerMixin._adaptive_step(irrbb["eve_ratio"] - 1.0)
                high_total = sum(w[j] for j in high_idx)
                if high_total < transfer:
                    break
                # Dynamic cap-aware recipient selection: prefer classes below cap
                low_dur_idx = [i for i in all_low_dur if w[i] < max_w[i] - 0.001]
                if not low_dur_idx:
                    # All at cap: fall back to all low-dur (regulatory > advisory cap)
                    low_dur_idx = all_low_dur
                for j in high_idx:
                    w[j] -= transfer * (w[j] / high_total)
                # Distribute proportionally to remaining capacity (cap - current)
                market_caps_arr = np.array([ASSET_CLASS_MAP[class_names[j]].market_capacity_eur
                                            for j in low_dur_idx])
                max_w_arr = market_caps_arr / np.array(
                    [ASSET_CLASS_MAP[name].market_capacity_eur for name in class_names]).sum()
                remaining = np.maximum(max_w_arr - np.array([w[j] for j in low_dur_idx]), 0)
                rem_total = remaining.sum()
                if rem_total > 1e-10:
                    for k, j in enumerate(low_dur_idx):
                        w[j] += transfer * (remaining[k] / rem_total)
                else:
                    per_low = transfer / len(low_dur_idx)
                    for j in low_dur_idx:
                        w[j] += per_low
        w = np.maximum(w, 0)
        w /= w.sum()
        return w

    # ──────────────────────────────────────────────
    # OPTIMISEUR BL-CVaR N CLASSES — ALLOCATION ENDOGENE
    # ──────────────────────────────────────────────

    def optimize_allocation(self, macro_params: Optional[Dict[str, float]] = None,
                            cvar_alpha: float = 0.95) -> Dict[str, object]:
        """Optimise l'allocation sur N classes d'actifs via BL-CVaR endogene.

        Architecture Phase 1 / Phase 2 :
            Phase 1 : BL-CVaR pur (zero floor/cap, zero penalite artificielle).
                      Forces economiques endogenes :
                      - CIR par classe (dans mu_raroc via compute_raroc_multiclass)
                      - Compression de spread logarithmique (market impact)
                      - Ledoit-Wolf shrinkage (covariance stabilisee)
                      Recherche 1D sur parametre de risque alpha in [0, 1].
                      alpha=0 : poids typiques (equilibre de marche)
                      alpha=1 : poids RAROC-optimal (agressif)
            Phase 2 : Normes reglementaires (ajustement sequentiel minimal) :
                      CET1 >= 13%, LCR >= 100%, NSFR >= 100%.

        Objectif (Rockafellar-Uryasev) :
            max[ mu_eff(w) - kappa * CVaR_0(w) ]
            ou mu_eff_i = mu_base_i / (1 + compression_i)
            et compression_i = -ln(1 - share_i)

        Args:
            macro_params: Dict macro optionnel.

        Returns:
            Dict avec allocation optimale N classes et metriques BL-CVaR.
            Backward-compatible keys : credit_allocation, pe_allocation,
            sector_weights_credit, sector_weights_pe.
        """
        raroc_mc = self.compute_raroc_multiclass()
        raroc_2ch = self.compute_raroc_eva()  # for backward-compat sector weights
        coc = BASEL_CONFIG.cet1_target

        # ── RAROC and profit_rate vectors for N classes ──
        classes_df = raroc_mc.filter(pl.col("asset_class") != "Total")
        n = len(classes_df)
        class_names = classes_df["asset_class"].to_list()
        mu_raroc = classes_df["raroc"].to_numpy().astype(float).copy()
        mu_profit = classes_df["profit_rate"].to_numpy().astype(float).copy()

        # ── Spot returns for stress intensity (uses RAROC, not profit_rate) ──
        idx_corporate = class_names.index("corporate_loans")
        idx_pe = class_names.index("private_equity")
        r_c_spot = float(mu_raroc[idx_corporate])
        r_p_spot = float(mu_raroc[idx_pe])

        # Use PE IRR - loss rate as spot PE return
        irr_pe = self.result_pe["irr"].mean() if len(self.result_pe) > 0 else 0.08
        el_pe = self.result_pe["expected_loss_pe"].sum()
        nav_pe = self.result_pe["nav"].sum()
        r_p_spot = irr_pe - el_pe / max(nav_pe, 1)

        # Auto-calibrated r_neutral from mu_raroc
        positives = mu_raroc[mu_raroc > 0]
        r_neutral = float(np.mean(positives)) if len(positives) > 0 else 0.07
        stress = self._stress_intensity(r_c_spot, r_p_spot, r_neutral)

        # ── Asymmetric equations SUPPRESSED (CVaR gradient + CET1 Phase 2 suffisent) ──
        # Backward-compat: neutral values in output
        illiq_premium = 0.0    # Supprime: CVaR gradient penalise deja les actifs a queue lourde
        vol_mult = 1.0         # Supprime: kappa_base auto-calibre s'adapte au regime
        pe_band = (0.005, 0.20)  # Supprime: CET1 Phase 2 empeche les surexpositions
        conf_credit, conf_pe = 0.50, 0.50  # Supprime: non utilise dans l'objectif

        capital_arr = classes_df["capital"].to_numpy().astype(float)
        ead_arr = classes_df["exposure"].to_numpy().astype(float)

        # ── Volatility per class: market_vol_override for all 14 classes ──
        vol_return = np.zeros(n)
        for i, name in enumerate(class_names):
            ac = ASSET_CLASS_MAP[name]
            vol_return[i] = ac.market_vol_override

        # ── Convert to profit_rate-space: vol_profit = vol_return × (capital/EAD) ──
        # Aligns Sigma with mu_profit so BL-CVaR objective is in consistent units.
        cap_ead = capital_arr / np.maximum(ead_arr, 1)
        vol = vol_return * cap_ead

        # ── Covariance matrix NxN with Ledoit-Wolf shrinkage ──
        corr_14, lambda_lw = self._build_corr_matrix()
        Sigma = np.outer(vol, vol) * corr_14
        Sigma = (Sigma + Sigma.T) / 2

        # ── RMT denoising (Marchenko-Pastur) ──
        from ifrs9_cockpit.engine.rmt import denoise_covariance
        from ifrs9_cockpit.config import MACRO_HISTORY_BASELINE
        _n_obs = len(next(iter(MACRO_HISTORY_BASELINE.values())))
        rmt_result = denoise_covariance(Sigma, n_observations=_n_obs)
        Sigma = rmt_result.covariance_clean
        Sigma = (Sigma + Sigma.T) / 2

        eigvals = np.linalg.eigvalsh(Sigma)
        if eigvals.min() < 1e-8:
            Sigma += np.eye(n) * (1e-6 - min(0, eigvals.min()))

        # ── Base weights: typical_weight (equilibre de marche) ──
        w_base = np.array([ASSET_CLASS_MAP[name].typical_weight for name in class_names])
        w_base /= w_base.sum()

        # ── Total EAD (needed for spread compression) ──
        exposure_per_class = classes_df["exposure"].to_numpy().astype(float)
        total_ead = exposure_per_class.sum()

        # Also compute RAROC-based softmax for backward-compat metrics
        w_raroc_raw = self._softmax_10(mu_raroc, corr_14)
        mu_raroc_compressed = self._spread_compression(w_raroc_raw, class_names, total_ead, mu_raroc)
        w_raroc = self._softmax_10(mu_raroc_compressed, corr_14)

        # ── Phase 1 : BL-CVaR pur (profit_rate objective, market impact logarithmique) ──
        # Monte Carlo CVaR setup
        rng = np.random.default_rng(42)
        n_scenarios = 5000
        alpha_cvar = cvar_alpha
        L_chol = np.linalg.cholesky(Sigma)
        Z = rng.standard_normal((n_scenarios, n))
        scenarios_0 = Z @ L_chol.T  # zero-mean
        cutoff = max(int(n_scenarios * (1.0 - alpha_cvar)), 1)

        # Auto-calibrated kappa: balanced at w_base (obj=0 at typical weights).
        # kappa = mu_p_base / CVaR_base ensures allocations better than w_base
        # have positive objective. Scale-invariant: works for both RAROC and profit_rate.
        mu_eff_base = self._spread_compression(w_base, class_names, total_ead, mu_profit)
        mu_p_base = float(w_base @ mu_eff_base)
        port_dev_base = scenarios_0 @ w_base
        cvar_base = -float(np.mean(np.sort(port_dev_base)[:cutoff]))
        kappa_base = mu_p_base / max(cvar_base, 1e-10)
        kappa_eff = kappa_base  # Direct: no vol_mult correction

        # ── CVaR-gradient optimal direction (Rockafellar-Uryasev 2002) ──
        # Score_i = d(obj)/d(w_i) = mu_eff_i - kappa * d(CVaR)/d(w_i)
        # where d(CVaR)/d(w_i) = -E[epsilon_i | portfolio in 5% tail]
        # By construction, softmax(scores) produces a direction where the
        # BL-CVaR objective improves, guaranteeing alpha > 0 in grid search.
        tail_threshold = np.sort(port_dev_base)[cutoff - 1]
        tail_mask = port_dev_base <= tail_threshold
        cvar_gradient = -scenarios_0[tail_mask].mean(axis=0)
        bl_cvar_scores = mu_eff_base - kappa_base * cvar_gradient
        bl_scores_norm = bl_cvar_scores / max(np.max(np.abs(bl_cvar_scores)), 1e-10)
        w_profit_raw = self._softmax_10(bl_scores_norm, corr_14)
        # Second pass with spread compression at w_profit_raw
        mu_comp = self._spread_compression(w_profit_raw, class_names, total_ead, mu_profit)
        bl_scores_comp = mu_comp - kappa_base * cvar_gradient
        bl_comp_norm = bl_scores_comp / max(np.max(np.abs(bl_scores_comp)), 1e-10)
        w_profit_opt = self._softmax_10(bl_comp_norm, corr_14)

        best_obj = -1e10
        best_alpha = 0.5
        best_w = w_base.copy()

        for alpha_trial in np.linspace(0.0, 1.0, 201):
            # Interpolate between typical (conservative) and profit-rate-optimal (aggressive)
            w_trial = (1.0 - alpha_trial) * w_base + alpha_trial * w_profit_opt
            w_trial = np.maximum(w_trial, 0)      # no short selling
            w_trial /= w_trial.sum()               # simple normalization

            # Objective: mu_eff(w) - kappa * CVaR_0(w)  (pur BL-CVaR, zero penalite)
            mu_eff = self._spread_compression(w_trial, class_names, total_ead, mu_profit)
            mu_p = float(w_trial @ mu_eff)
            port_dev = scenarios_0 @ w_trial
            port_sorted = np.sort(port_dev)
            cvar_0 = -float(np.mean(port_sorted[:cutoff]))
            obj = mu_p - kappa_eff * cvar_0

            if obj > best_obj:
                best_obj = obj
                best_alpha = alpha_trial
                best_w = w_trial.copy()

        # Save Phase 1 weights before regulatory adjustments
        phase1_weights = best_w.copy()
        self._phase1_weights_debug = phase1_weights.copy()
        self._best_alpha_debug = best_alpha
        self._w_profit_opt_debug = w_profit_opt.copy()
        self._w_base_debug = w_base.copy()
        self._class_names_debug = class_names

        # ── Phase 2 : Normes reglementaires (ajustement sequentiel minimal) ──
        # Compute full-balance-sheet RWA
        rwa_per_class = classes_df["rwa"].to_numpy().astype(float)
        total_exposure = total_ead  # already computed above

        # ECL-adjusted CET1 capital
        cet1_capital_base = BASEL_CONFIG.rwa_budget * coc
        ecl_total = self.result_credit["ecl_weighted"].sum()
        el_pe_total = self.result_pe["expected_loss_pe"].sum()
        # Add parametric ECL from balance sheet classes
        ecl_bs = 0.0
        if self.df_balance_sheet_ecl is not None:
            bs_classes = self.df_balance_sheet_ecl.filter(
                ~pl.col("asset_class").is_in(["corporate_loans", "private_equity"])
            )
            ecl_bs = float(bs_classes["ecl_weighted"].sum()) if len(bs_classes) > 0 else 0.0
        cet1_capital = cet1_capital_base - ecl_total - el_pe_total - ecl_bs

        # RWA weighted by allocation
        w_current = exposure_per_class / max(total_exposure, 1)

        def _compute_rwa(w_alloc):
            rwa = 0.0
            for i in range(n):
                if w_current[i] > 1e-10:
                    rwa += w_alloc[i] * (rwa_per_class[i] / w_current[i])
                else:
                    ac = ASSET_CLASS_MAP[class_names[i]]
                    rwa += w_alloc[i] * total_exposure * effective_rw(ac)
            return rwa

        rwa_weighted = _compute_rwa(best_w)
        headroom = cet1_capital - coc * rwa_weighted

        # 2a. CET1 feasibility (binary search between w_base and best_w)
        cet1_delta = 0.0
        if headroom < 0:
            w_before_cet1 = best_w.copy()
            lo, hi = 0.0, 1.0
            for _ in range(25):
                mid = (lo + hi) / 2
                w_test = (1.0 - mid) * w_base + mid * best_w
                w_test = np.maximum(w_test, 0)
                w_test /= w_test.sum()
                rwa_test = _compute_rwa(w_test)
                if cet1_capital - coc * rwa_test >= 0:
                    lo = mid
                else:
                    hi = mid
            best_w = (1.0 - lo) * w_base + lo * best_w
            best_w = np.maximum(best_w, 0)
            best_w /= best_w.sum()
            rwa_weighted = _compute_rwa(best_w)
            headroom = cet1_capital - coc * rwa_weighted
            cet1_delta = float(np.sum(np.abs(best_w - w_before_cet1)))

        # 2b. LCR
        lcr_delta = 0.0
        lcr_ratio = self._compute_lcr_ratio(best_w, class_names, total_exposure)
        if lcr_ratio < BASEL_CONFIG.lcr_target:
            w_before_lcr = best_w.copy()
            best_w = self._adjust_for_lcr(best_w, class_names, total_exposure, BASEL_CONFIG.lcr_target)
            lcr_ratio = self._compute_lcr_ratio(best_w, class_names, total_exposure)
            lcr_delta = float(np.sum(np.abs(best_w - w_before_lcr)))
            # Recompute RWA after LCR adjustment
            rwa_weighted = _compute_rwa(best_w)
            headroom = cet1_capital - coc * rwa_weighted

        # 2c. NSFR
        nsfr_delta = 0.0
        nsfr_result = compute_nsfr(
            allocation_weights={name: float(best_w[i]) for i, name in enumerate(class_names)},
            total_ead=total_exposure,
            asf_coverage=BASEL_CONFIG.asf_deposit_coverage,
        )
        if nsfr_result["nsfr_ratio"] < 1.0:
            w_before_nsfr = best_w.copy()
            best_w = self._adjust_for_nsfr(best_w, class_names, total_exposure)
            nsfr_result = compute_nsfr(
                allocation_weights={name: float(best_w[i]) for i, name in enumerate(class_names)},
                total_ead=total_exposure,
                asf_coverage=BASEL_CONFIG.asf_deposit_coverage,
            )
            nsfr_delta = float(np.sum(np.abs(best_w - w_before_nsfr)))
            # Recompute RWA after NSFR adjustment
            rwa_weighted = _compute_rwa(best_w)
            headroom = cet1_capital - coc * rwa_weighted

        # 2d. IRRBB EVE + LCR joint iteration
        # Sovereign is both high-HQLA (LCR needs it) and high-duration (IRRBB limits it).
        # Iterate IRRBB → LCR until both converge (max 5 outer rounds).
        w_before_irrbb_lcr = best_w.copy()
        for _joint_iter in range(10):
            irrbb_result = compute_irrbb_eve(
                allocation_weights={name: float(best_w[i]) for i, name in enumerate(class_names)},
                total_ead=total_exposure,
                cet1_capital=cet1_capital,
            )
            lcr_ratio = self._compute_lcr_ratio(best_w, class_names, total_exposure)
            irrbb_ok = irrbb_result["compliant"]
            lcr_ok = lcr_ratio >= BASEL_CONFIG.lcr_target
            if irrbb_ok and lcr_ok:
                break
            if not irrbb_ok:
                best_w = self._adjust_for_irrbb(best_w, class_names, total_exposure, cet1_capital)
            lcr_ratio = self._compute_lcr_ratio(best_w, class_names, total_exposure)
            if lcr_ratio < BASEL_CONFIG.lcr_target:
                best_w = self._adjust_for_lcr(best_w, class_names, total_exposure, BASEL_CONFIG.lcr_target)
        # Final state
        irrbb_result = compute_irrbb_eve(
            allocation_weights={name: float(best_w[i]) for i, name in enumerate(class_names)},
            total_ead=total_exposure,
            cet1_capital=cet1_capital,
        )
        lcr_ratio = self._compute_lcr_ratio(best_w, class_names, total_exposure)
        irrbb_delta = float(np.sum(np.abs(best_w - w_before_irrbb_lcr)))

        # 2e. Market capacity cap (post regulatory adjustments)
        # Phase 2 (NSFR, IRRBB) may push low-RSF/low-duration classes beyond
        # their market capacity share. Cap then restore regulatory compliance.
        # Regulatory norms (LCR/NSFR/IRRBB) take precedence over market caps.
        best_w = self._enforce_market_caps(best_w, class_names, n)

        # Re-verify and restore regulatory compliance after capping
        lcr_ratio_post = self._compute_lcr_ratio(best_w, class_names, total_exposure)
        if lcr_ratio_post < BASEL_CONFIG.lcr_target:
            best_w = self._adjust_for_lcr(best_w, class_names, total_exposure, BASEL_CONFIG.lcr_target)
        nsfr_post = compute_nsfr(
            {name: float(best_w[i]) for i, name in enumerate(class_names)},
            total_exposure,
        )["nsfr_ratio"]
        if nsfr_post < 1.0:
            best_w = self._adjust_for_nsfr(best_w, class_names, total_exposure)
        irrbb_post = compute_irrbb_eve(
            {name: float(best_w[i]) for i, name in enumerate(class_names)},
            total_exposure, cet1_capital,
        )
        if not irrbb_post["compliant"]:
            best_w = self._adjust_for_irrbb(best_w, class_names, total_exposure, cet1_capital)
            # LCR may need re-fixing after IRRBB
            lcr_ratio_post = self._compute_lcr_ratio(best_w, class_names, total_exposure)
            if lcr_ratio_post < BASEL_CONFIG.lcr_target:
                best_w = self._adjust_for_lcr(best_w, class_names, total_exposure, BASEL_CONFIG.lcr_target)

        # Final state after all adjustments
        irrbb_result = compute_irrbb_eve(
            allocation_weights={name: float(best_w[i]) for i, name in enumerate(class_names)},
            total_ead=total_exposure,
            cet1_capital=cet1_capital,
        )
        lcr_ratio = self._compute_lcr_ratio(best_w, class_names, total_exposure)
        nsfr_result = compute_nsfr(
            allocation_weights={name: float(best_w[i]) for i, name in enumerate(class_names)},
            total_ead=total_exposure,
            asf_coverage=BASEL_CONFIG.asf_deposit_coverage,
        )
        rwa_weighted = _compute_rwa(best_w)
        headroom = cet1_capital - coc * rwa_weighted

        cet1_ratio = cet1_capital / max(rwa_weighted, 1)
        headroom_eur = headroom

        # ── Build class_weights dict ──
        class_weights = {name: round(float(best_w[i]), 4) for i, name in enumerate(class_names)}

        # ── CVaR of optimal portfolio ──
        port_opt_dev = scenarios_0 @ best_w
        port_opt_sorted = np.sort(port_opt_dev)
        cvar_final = -float(np.mean(port_opt_sorted[:cutoff]))

        # ── Backward-compatible Credit/PE split ──
        pe_alloc = float(best_w[class_names.index("private_equity")])
        credit_alloc = round(1.0 - pe_alloc, 4)

        # Sector weights (within Credit and PE channels, unchanged)
        credit_cells = raroc_2ch.filter(
            (pl.col("canal") == "Credit") & (pl.col("sector") != "Total")
        )
        pe_cells = raroc_2ch.filter(
            (pl.col("canal") == "PE") & (pl.col("sector") != "Total")
        )
        w_credit = self._optimize_sector_weights(credit_cells) if len(credit_cells) > 0 else {}
        w_pe = self._optimize_sector_weights(pe_cells) if len(pe_cells) > 0 else {}

        # ── Portfolio-level metrics (with market impact) ──
        mu_eff_profit_final = self._spread_compression(best_w, class_names, total_ead, mu_profit)
        profit_rate_portfolio = float(best_w @ mu_eff_profit_final)
        # Backward compat: RAROC portfolio too
        mu_eff_raroc_final = self._spread_compression(best_w, class_names, total_ead, mu_raroc)
        raroc_portfolio = float(best_w @ mu_eff_raroc_final)
        port_vol = float(np.sqrt(best_w @ Sigma @ best_w))
        sharpe = raroc_portfolio / max(port_vol, 1e-9)
        leverage_ratio = cet1_ratio * 0.4

        # RAROC credit/pe for backward compat
        total_credit_row = raroc_2ch.filter(
            (pl.col("canal") == "Credit") & (pl.col("sector") == "Total")
        )
        total_pe_row = raroc_2ch.filter(
            (pl.col("canal") == "PE") & (pl.col("sector") == "Total")
        )
        raroc_c = float(total_credit_row["raroc"].to_numpy()[0]) if len(total_credit_row) > 0 else 0.0
        raroc_p = float(total_pe_row["raroc"].to_numpy()[0]) if len(total_pe_row) > 0 else 0.0

        # PE free (what Phase 1 would give without constraints)
        pe_free = round(float(w_raroc[class_names.index("private_equity")]), 2)

        return {
            # ── N-class allocation ──
            "class_weights": class_weights,
            "n_classes": n,
            "constraint_mode": "endogenous",
            "objective": "profit_rate",
            # ── Backward-compatible Credit/PE split ──
            "credit_allocation": round(credit_alloc, 4),
            "pe_allocation": round(pe_alloc, 4),
            "pe_free": pe_free,
            "pe_band": list(pe_band),
            "sector_weights_credit": w_credit,
            "sector_weights_pe": w_pe,
            "raroc_credit": round(raroc_c, 4),
            "raroc_pe": round(raroc_p, 4),
            # ── Portfolio metrics ──
            "rwa_weighted": round(rwa_weighted, 0),
            "cet1_ratio": round(cet1_ratio, 4),
            "cet1_headroom": round(cet1_ratio - coc, 4),
            "headroom_m": round(headroom_eur / 1e6, 1),
            "feasible": headroom_eur >= 0,
            "method": f"BL-CVaR-{n}C",
            "cvar_95": round(cvar_final, 4),
            "cvar_alpha": round(alpha_cvar, 2),
            "kappa": round(kappa_base, 4),
            "kappa_pe_eff": round(kappa_eff, 4),
            "n_scenarios": n_scenarios,
            "mu_raroc_credit": round(raroc_c, 4),
            "mu_raroc_pe": round(raroc_p, 4),
            "sharpe": round(sharpe, 2),
            "portfolio_vol": round(port_vol, 4),
            "profit_rate_portfolio": round(profit_rate_portfolio, 6),
            "raroc_portfolio": round(raroc_portfolio, 4),
            "sigma_credit": round(float(vol[idx_corporate]), 4),
            "sigma_pe": round(float(vol[idx_pe]), 4),
            "correlation": round(float(corr_14[idx_corporate, idx_pe]), 2),
            "k_e": round(coc, 4),
            "leverage_ratio": round(leverage_ratio, 4),
            "stress_intensity": round(stress, 2),
            "illiquidity_premium": round(illiq_premium, 4),
            "vol_multiplier": round(vol_mult, 2),
            "bl_confidence": (round(conf_credit, 2), round(conf_pe, 2)),
            # ── Endogenous allocation details ──
            "typical_weights": {name: ASSET_CLASS_MAP[name].typical_weight for name in class_names},
            "phase1_weights": {name: round(float(phase1_weights[i]), 4) for i, name in enumerate(class_names)},
            "regulatory_adjustments": {
                "cet1_delta": round(cet1_delta, 4),
                "lcr_delta": round(lcr_delta, 4),
                "nsfr_delta": round(nsfr_delta, 4),
                "irrbb_delta": round(irrbb_delta, 4),
            },
            "covariance_shrinkage_lambda": round(lambda_lw, 4),
            "rmt_n_signal": rmt_result.n_signal,
            "rmt_n_noise": rmt_result.n_noise,
            "rmt_noise_fraction": round(rmt_result.noise_fraction, 4),
            "spread_compression": {
                name: round(float(-np.log(1.0 - np.clip(
                    best_w[i] * total_ead / ASSET_CLASS_MAP[name].market_capacity_eur,
                    0.0, 0.95))), 4)
                for i, name in enumerate(class_names)
            },
            "risk_alpha": round(best_alpha, 3),
            "corr_10x10": corr_14.tolist(),
            # ── LCR ──
            "lcr_ratio": round(lcr_ratio, 4),
            "lcr_compliant": lcr_ratio >= BASEL_CONFIG.lcr_target,
            # ── NSFR ──
            "nsfr_ratio": nsfr_result["nsfr_ratio"],
            "nsfr_compliant": nsfr_result["compliant"],
            # ── IRRBB ──
            "irrbb_eve_ratio": irrbb_result["eve_ratio"],
            "irrbb_weighted_duration": irrbb_result["weighted_duration"],
            "irrbb_compliant": irrbb_result["compliant"],
        }

    @staticmethod
    def _softmax_10(mu: np.ndarray, corr: np.ndarray, lam: Optional[float] = None) -> np.ndarray:
        """Softmax RAROC-optimal avec penalite de correlation pour N classes.

        Score_i = RAROC_i - lam * sum(w_j * rho_ij) pour j!=i
        Deux passes : Softmax naif -> scores penalises -> Softmax final.
        lam auto-calibre: mean(|off-diag correlations|).

        Args:
            mu: RAROC vector (N,).
            corr: Correlation matrix (NxN).
            lam: Penalite de correlation (auto si None).

        Returns:
            Array de poids normalises (N,).
        """
        n = len(mu)
        # Auto-calibrate lam from off-diagonal correlations
        if lam is None:
            off_diag = corr[np.triu_indices(n, k=1)]
            lam = float(np.mean(np.abs(off_diag)))
        # Pass 1: naive Softmax
        w0 = OptimizerMixin._softmax_weights_10(mu, n)
        # Pass 2: correlation-penalized scores
        scores = np.zeros(n)
        for i in range(n):
            penalty = sum(w0[j] * corr[i, j] for j in range(n) if j != i)
            scores[i] = mu[i] - lam * penalty
        return OptimizerMixin._softmax_weights_10(scores, n)

    @staticmethod
    def _softmax_weights_10(values: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        """Softmax libre pour N classes (Phase 1 = aucune contrainte).

        Phase 1 est une optimisation LIBRE : seul le softmax et un floor
        numerique (1/n^2) s'appliquent. Les caps market_capacity sont
        appliques en Phase 2e via _enforce_market_caps().

        La temperature est auto-calibree sur l'ecart-type des scores pour
        produire un ratio ~20x entre la meilleure et la pire classe :
        T = 3 / max(std, 0.01). Quand les scores sont tres comprimes
        (std ~0.04), T monte a ~75 et produit une distribution tres
        differenciee. Quand ils sont etales (std ~0.35), T ~8.6 et la
        distribution reste raisonnable.

        Args:
            values: Array de scores (BL-CVaR ou correlation-penalises).
            n: Number of classes (for min_weight auto-calibration).

        Returns:
            Array de poids normalises — libre, sans cap.
        """
        if n is None:
            n = len(values)
        min_weight = 1.0 / (n * n)  # auto: n=14 -> 0.0051

        # Temperature auto-calibree pour un ratio max best/worst = 20x.
        # T = ln(max_ratio) / range(values). Borne le ratio independamment
        # de la distribution des scores, evitant la concentration extreme
        # quand un score est un outlier (ex. GFC : PE score >> autres).
        _MAX_RATIO = 20.0
        score_range = float(np.max(values) - np.min(values))
        if score_range < 1e-8:
            return np.ones(n) / n  # scores identiques -> equiponderation
        temperature = np.log(_MAX_RATIO) / score_range

        shifted = values * temperature - np.max(values * temperature)
        exp_vals = np.exp(shifted)
        w = exp_vals / exp_vals.sum()

        # Enforce minimum weight for numerical stability (no cap — Phase 1 libre)
        w = np.maximum(w, min_weight)
        w = w / w.sum()
        return w

    def _optimize_sector_weights(
        self,
        cells: pl.DataFrame,
    ) -> Dict[str, float]:
        """Optimise les poids sectoriels par RAROC penalise des correlations.

        RJ audit v3 : le Softmax naif ignore les correlations inter-secteurs.
        Score_i = RAROC_i - lambda * sum(w_j * rho_ij) penalise les secteurs
        fortement correles aux autres, favorisant la diversification.

        Algorithme iteratif (2 passes) :
            Passe 1 : poids Softmax naif sur RAROC.
            Passe 2 : score penalise = RAROC - lambda * sum(w_j * rho_ij),
                       puis Softmax sur scores penalises.

        Args:
            cells: DataFrame avec colonnes sector, raroc.

        Returns:
            Dict {sector_name: weight}.
        """
        sectors = cells["sector"].to_list()
        rarocs = cells["raroc"].to_numpy().astype(float)
        n = len(sectors)

        # Matrice de correlation entre les secteurs presents
        # _EXPERT_CORR est 5x5 dans l'ordre SECTOR_NAMES
        sector_indices = [list(SECTOR_NAMES).index(s) for s in sectors]
        corr_matrix = np.array(_EXPERT_CORR)
        # Sous-matrice pour les secteurs presents
        rho = corr_matrix[np.ix_(sector_indices, sector_indices)]

        # Auto-calibre: mean(|off-diag correlations|) (zero parametre arbitraire)
        off_diag_sector = rho[np.triu_indices(n, k=1)]
        lam = float(np.mean(np.abs(off_diag_sector)))

        # Passe 1 : Softmax naif pour obtenir les poids initiaux
        w = self._softmax_weights(rarocs)

        # Passe 2 : Score penalise par la correlation
        penalized_scores = np.zeros(n)
        for i in range(n):
            corr_penalty = sum(w[j] * rho[i, j] for j in range(n) if j != i)
            penalized_scores[i] = rarocs[i] - lam * corr_penalty

        # Softmax sur scores penalises
        final_weights = self._softmax_weights(penalized_scores)

        return {s: round(w, 4) for s, w in zip(sectors, final_weights)}

    @staticmethod
    def _softmax_weights(values: np.ndarray) -> np.ndarray:
        """Softmax adaptative avec floor naturel 1/n et renormalisation.

        Scale = 1/var(values) sans clip. Floor = 1/n (diversification minimale).

        Args:
            values: Array de scores (RAROC ou scores penalises).

        Returns:
            Array de poids normalises.
        """
        n = len(values)
        scale = 1.0 / max(np.var(values), 1e-8)
        shifted = values * scale - np.max(values * scale)
        exp_vals = np.exp(shifted)
        raw_weights = exp_vals / exp_vals.sum()
        # Floor naturel: 1/n (pas de parametre arbitraire)
        floored = np.maximum(raw_weights, 1.0 / n)
        return floored / floored.sum()
