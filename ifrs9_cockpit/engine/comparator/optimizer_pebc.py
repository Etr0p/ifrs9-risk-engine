"""Mixin BL-CVaR 10 cellules (5 secteurs x 2 canaux) — portage pe-bc.

Architecture identique a l'optimiseur 14 classes (optimizer.py) mais
operant sur les 10 cellules du modele pe-bc :
    [Tech_Credit, Ind_Credit, San_Credit, Immo_Credit, Svc_Credit,
     Tech_PE, Ind_PE, San_PE, Immo_PE, Svc_PE]

Phase 1 : BL-CVaR pur (CVaR gradient + spread compression logarithmique)
Phase 2 : Normes reglementaires (CET1, LCR, NSFR, IRRBB) via ajustements
          sequentiels minimaux.

Les volatilites et capacites de marche proviennent des attributs
dual-channel de SectorConfig (market_vol_credit/pe, market_capacity_credit/pe_eur).
"""

from __future__ import annotations

import numpy as np
import polars as pl
from typing import Dict, Optional, Tuple

from scipy.linalg import cholesky

from ifrs9_cockpit.config import (
    BASEL_CONFIG,
    SECTORS,
    SECTOR_NAMES,
    RANDOM_SEED,
)
from ifrs9_cockpit.engine.comparator.regulatory_pebc import (
    compute_lcr_10,
    compute_nsfr_10,
    compute_irrbb_eve_10,
)

# Noms des 10 cellules dans l'ordre canonique :
# [Tech_C, Ind_C, San_C, Immo_C, Svc_C, Tech_PE, Ind_PE, San_PE, Immo_PE, Svc_PE]
_CELL_NAMES_PEBC: Tuple[str, ...] = tuple(
    f"{s.name}_{'Credit' if c == 0 else 'PE'}"
    for c in (0, 1) for s in SECTORS
)

_N_CELLS = len(SECTORS) * 2  # 10


class PebcOptimizerMixin:
    """Mixin fournissant l'optimiseur BL-CVaR pe-bc 10 cellules."""

    # ──────────────────────────────────────────────
    # MATRICE DE CORRELATION 10x10 (COSINE SIMILARITY MACRO)
    # ──────────────────────────────────────────────

    @staticmethod
    def _build_corr_matrix_pebc() -> Tuple[np.ndarray, float]:
        """Construit la matrice de correlation 10x10 par cosine similarity.

        Les 10 cellules sont les 5 secteurs x 2 canaux (credit puis PE).
        Chaque cellule a un vecteur de 5 sensibilites macro.
        La correlation est la cosine similarity entre les vecteurs.

        Regularisation :
            - Flight-to-quality (Sante defensif en crise)
            - Clip [-0.90, 0.95]
            - Ledoit-Wolf shrinkage (stabilisation)
            - PSD enforcement (eigenvalue clipping)

        Returns:
            (corr_10x10, lambda_lw) : matrice de correlation et shrinkage.
        """
        n = _N_CELLS

        # Matrice S (10 x 5) des sensibilites macro
        S = np.zeros((n, 5))
        for i, sector in enumerate(SECTORS):
            # Credit (lignes 0-4)
            S[i, :] = [
                sector.unemployment_sensitivity_credit,
                sector.gdp_sensitivity_credit,
                sector.interest_rate_sensitivity_credit,
                sector.hpi_sensitivity_credit,
                sector.inflation_sensitivity_credit,
            ]
            # PE (lignes 5-9)
            S[i + 5, :] = [
                sector.unemployment_sensitivity_pe,
                sector.gdp_sensitivity_pe,
                sector.interest_rate_sensitivity_pe,
                sector.hpi_sensitivity_pe,
                sector.inflation_sensitivity_pe,
            ]

        # Normalisation par ligne (L2)
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        S_norm = S / norms

        # Cosine similarity
        corr = S_norm @ S_norm.T
        np.fill_diagonal(corr, 1.0)

        # Clip [-0.90, 0.95], symetrise
        corr = np.clip(corr, -0.90, 0.95)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

        # Flight-to-quality : sante defensif en crise
        cell_names_list = list(_CELL_NAMES_PEBC)
        _FTQ_10 = {
            ("Sante_Credit", "Immobilier_Credit"): -0.15,
            ("Sante_Credit", "Industrie_Credit"): -0.10,
            ("Sante_PE", "Immobilier_PE"): -0.15,
            ("Technologie_Credit", "Technologie_PE"): +0.10,
            ("Immobilier_Credit", "Immobilier_PE"): +0.10,
        }
        for (a, b), adj in _FTQ_10.items():
            i_a = cell_names_list.index(a) if a in cell_names_list else -1
            i_b = cell_names_list.index(b) if b in cell_names_list else -1
            if i_a >= 0 and i_b >= 0:
                corr[i_a, i_b] += adj
                corr[i_b, i_a] += adj

        # Re-clip apres FTQ et re-symetrise
        corr = np.clip(corr, -0.90, 0.95)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

        # Ledoit-Wolf shrinkage vers matrice a correlation constante
        off_diag = corr[np.triu_indices(n, k=1)]
        rho_bar = float(np.mean(off_diag))
        F = np.full((n, n), rho_bar)
        np.fill_diagonal(F, 1.0)

        # Lambda auto-calibre (Ledoit-Wolf estimateur simplifie)
        d2 = np.sum((corr - F) ** 2) / (n * n)
        lambda_lw = float(np.clip(d2 / max(d2 + 1e-6, 1e-10), 0.05, 0.50))

        shrunk = (1 - lambda_lw) * corr + lambda_lw * F

        # PSD enforcement : eigenvalues clip a 1e-6, reconstruction
        eigvals, eigvecs = np.linalg.eigh(shrunk)
        eigvals = np.maximum(eigvals, 1e-6)
        shrunk = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Renormalise diagonale a 1
        d_inv = 1.0 / np.sqrt(np.diag(shrunk))
        shrunk = shrunk * np.outer(d_inv, d_inv)
        np.fill_diagonal(shrunk, 1.0)

        return shrunk, lambda_lw

    # ──────────────────────────────────────────────
    # SPREAD COMPRESSION (Kyle 1985, Almgren-Chriss 2001)
    # ──────────────────────────────────────────────

    @staticmethod
    def _spread_compression_pebc(
        w: np.ndarray,
        total_ead: float,
        mu_base: np.ndarray,
    ) -> np.ndarray:
        """Market impact log-compression (Kyle 1985) pour 10 cellules.

        share_i = (w_i * total_ead) / market_capacity_i
        compression_i = -ln(1 - min(share_i, 0.95))
        mu_eff_i = mu_base_i / (1 + compression_i)

        Args:
            w: Array de poids (10 cellules).
            total_ead: Exposition totale du portefeuille.
            mu_base: Rendements de base (10 cellules).

        Returns:
            mu_eff: Rendements effectifs apres compression.
        """
        capacities = np.array([
            s.market_capacity_credit_eur for s in SECTORS
        ] + [
            s.market_capacity_pe_eur for s in SECTORS
        ])

        share = (w * total_ead) / np.maximum(capacities, 1.0)
        compression = -np.log(1 - np.minimum(share, 0.95))
        mu_eff = mu_base / (1 + compression)
        return mu_eff

    # ──────────────────────────────────────────────
    # CVaR MONTE CARLO
    # ──────────────────────────────────────────────

    @staticmethod
    def _compute_cvar_pebc(
        w: np.ndarray,
        Sigma: np.ndarray,
        n_scenarios: int = 5000,
        alpha: float = 0.95,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """CVaR Monte Carlo par simulation de scenarios.

        Args:
            w: Vecteur de poids (10 cellules).
            Sigma: Matrice de covariance 10x10.
            n_scenarios: Nombre de scenarios Monte Carlo.
            alpha: Niveau de confiance CVaR (0.95 = tail 5%).

        Returns:
            (cvar, scenarios, tail_mask) : CVaR, matrice de scenarios,
            masque des scenarios dans la queue.
        """
        rng = np.random.default_rng(RANDOM_SEED)
        L = cholesky(Sigma, lower=True)
        Z = rng.standard_normal((n_scenarios, len(w)))
        scenarios = Z @ L.T  # n_scenarios x 10

        portfolio_losses = scenarios @ w
        cutoff = max(int(n_scenarios * (1 - alpha)), 1)
        sorted_losses = np.sort(portfolio_losses)
        cvar = float(-np.mean(sorted_losses[:cutoff]))

        # Masque des scenarios dans la queue (pour gradient)
        threshold = sorted_losses[cutoff - 1]
        tail_mask = portfolio_losses <= threshold

        return cvar, scenarios, tail_mask

    # ──────────────────────────────────────────────
    # SOFTMAX (TEMPERATURE AUTO-CALIBREE)
    # ──────────────────────────────────────────────

    @staticmethod
    def _softmax_weights_pebc(values: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        """Softmax avec temperature auto-calibree et floor 1/n^2.

        Temperature = ln(20) / score_range : le ratio max/min des poids
        bruts est calibre a 20x, sans parametre arbitraire.
        Floor = 1/n^2 (n=10 -> 0.01) : diversification minimale.

        Args:
            values: Array de scores BL-CVaR.
            n: Nombre de cellules (defaut: len(values)).

        Returns:
            Array de poids normalises.
        """
        if n is None:
            n = len(values)
        score_range = float(np.max(values) - np.min(values))
        if score_range < 1e-8:
            return np.ones(n) / n
        temperature = np.log(20) / max(score_range, 1e-10)
        shifted = values * temperature - np.max(values * temperature)
        exp_vals = np.exp(shifted)
        w = exp_vals / exp_vals.sum()
        min_weight = 1.0 / (n * n)  # n=10 -> 0.01
        w = np.maximum(w, min_weight)
        return w / w.sum()

    @staticmethod
    def _softmax_pebc(mu: np.ndarray, corr: np.ndarray) -> np.ndarray:
        """Softmax BL-CVaR a 2 passes avec lambda auto-calibre.

        Passe 1 : poids initiaux via softmax sur mu.
        Passe 2 : scores penalises = mu_i - lambda * sum(w_j * corr_ij),
                   lambda = mean(|off-diag corr|).

        Args:
            mu: Vecteur de scores (10 cellules).
            corr: Matrice de correlation 10x10.

        Returns:
            Array de poids optimises.
        """
        n = len(mu)
        w0 = PebcOptimizerMixin._softmax_weights_pebc(mu, n)

        # Auto-calibre lambda = mean(|off-diag|)
        off_diag = corr[np.triu_indices(n, k=1)]
        lam = float(np.mean(np.abs(off_diag)))

        scores = np.zeros(n)
        for i in range(n):
            penalty = sum(w0[j] * corr[i, j] for j in range(n) if j != i)
            scores[i] = mu[i] - lam * penalty

        return PebcOptimizerMixin._softmax_weights_pebc(scores, n)

    # ──────────────────────────────────────────────
    # STRESS DIAGNOSTICS
    # ──────────────────────────────────────────────

    @staticmethod
    def _stress_intensity_pebc(
        r_c_spot: float, r_p_spot: float, r_neutral: Optional[float] = None,
    ) -> float:
        """Intensite du stress — CRO-pessimiste (pire des 2 canaux).

        Args:
            r_c_spot: RAROC credit spot.
            r_p_spot: RAROC PE spot.
            r_neutral: RAROC neutre (defaut = CET1 target).

        Returns:
            s >= 0 en crise, < 0 en expansion.
        """
        if r_neutral is None:
            r_neutral = BASEL_CONFIG.cet1_target
        r_neutral = max(abs(r_neutral), 1e-10)
        return max(-r_c_spot / r_neutral, -r_p_spot / r_neutral)

    @staticmethod
    def _asymmetric_illiquidity_pebc(
        s: float, base: float = 0.005, scale: float = 0.025, gamma: float = 1.5,
    ) -> float:
        """Prime d'illiquidite asymetrique (Ang et al. 2014)."""
        return base + scale * max(0.0, s) ** gamma

    @staticmethod
    def _asymmetric_vol_multiplier_pebc(
        s: float, alpha_down: float = 0.40, alpha_up: float = 0.10,
    ) -> float:
        """Multiplicateur de volatilite asymetrique (GJR-GARCH 4:1)."""
        return 1.0 + alpha_down * max(0.0, s) - alpha_up * max(0.0, -s)

    @staticmethod
    def _bl_confidence_pebc(
        s: float, k_credit: float = 1.5, k_pe: float = 2.5,
    ) -> Tuple[float, float]:
        """Confiance BL asymetrique credit/PE (Ang & Bekaert 2002)."""
        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + np.exp(-x))
        tau_c = 0.30 + 0.65 * _sigmoid(-k_credit * s)
        tau_p = 0.30 + 0.65 * _sigmoid(-k_pe * s)
        return float(tau_c), float(tau_p)

    @staticmethod
    def _asymmetric_pe_band_pebc(
        s: float, pe_calm: float = 0.15, pe_stress: float = 0.05,
    ) -> Tuple[float, float]:
        """Bande PE sigmoid stress-dependante."""
        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + np.exp(-x))
        pe_max = pe_stress + (pe_calm - pe_stress) * _sigmoid(-2.0 * s)
        pe_min = 0.01
        return float(pe_min), float(pe_max)

    # ──────────────────────────────────────────────
    # AJUSTEMENTS REGULATOIRES (Phase 2)
    # ──────────────────────────────────────────────

    @staticmethod
    def _adaptive_step_pebc(deficit: float) -> float:
        """Pas adaptatif pour ajustements regulatoires."""
        return float(np.clip(abs(deficit) * 0.10, 0.002, 0.02))

    @staticmethod
    def _adjust_for_lcr_pebc(w: np.ndarray, total_ead: float) -> np.ndarray:
        """Ajuste les poids pour respecter la contrainte LCR.

        Dans pe-bc, aucune cellule n'est HQLA. L'ajustement reduit les
        cellules les plus illiquides (PE) et transfere vers les cellules
        credit les plus liquides (duration courte).
        """
        w = w.copy()
        rsf = np.array(
            [s.rsf_weight_credit for s in SECTORS]
            + [s.rsf_weight_pe for s in SECTORS]
        )
        for _ in range(20):
            lcr = compute_lcr_10(w, _CELL_NAMES_PEBC, SECTORS, total_ead)
            if lcr["lcr_ratio"] >= BASEL_CONFIG.lcr_target:
                break
            deficit = BASEL_CONFIG.lcr_target - lcr["lcr_ratio"]
            step = PebcOptimizerMixin._adaptive_step_pebc(deficit)
            # Transfer from highest RSF to lowest RSF
            i_max = int(np.argmax(rsf * w))
            i_min = int(np.argmin(rsf + (1.0 - w) * 10))
            transfer = min(step, w[i_max] * 0.5)
            w[i_max] -= transfer
            w[i_min] += transfer
            w = w / w.sum()
        return w

    @staticmethod
    def _adjust_for_nsfr_pebc(w: np.ndarray, total_ead: float) -> np.ndarray:
        """Ajuste les poids pour respecter la contrainte NSFR."""
        w = w.copy()
        rsf = np.array(
            [s.rsf_weight_credit for s in SECTORS]
            + [s.rsf_weight_pe for s in SECTORS]
        )
        for _ in range(20):
            nsfr = compute_nsfr_10(w, _CELL_NAMES_PEBC, SECTORS, total_ead)
            if nsfr["nsfr_ratio"] >= BASEL_CONFIG.nsfr_target:
                break
            deficit = BASEL_CONFIG.nsfr_target - nsfr["nsfr_ratio"]
            step = PebcOptimizerMixin._adaptive_step_pebc(deficit)
            effective_rsf = rsf * w
            i_max = int(np.argmax(effective_rsf))
            i_min = int(np.argmin(rsf + (1.0 - w) * 10))
            transfer = min(step, w[i_max] * 0.5)
            w[i_max] -= transfer
            w[i_min] += transfer
            w = w / w.sum()
        return w

    @staticmethod
    def _adjust_for_irrbb_pebc(
        w: np.ndarray, total_ead: float, cet1_capital: float,
    ) -> np.ndarray:
        """Ajuste les poids pour respecter la contrainte IRRBB EVE."""
        w = w.copy()
        dur = np.array(
            [s.duration_credit for s in SECTORS]
            + [s.duration_pe for s in SECTORS]
        )
        for _ in range(20):
            irrbb = compute_irrbb_eve_10(
                w, _CELL_NAMES_PEBC, SECTORS, total_ead, cet1_capital,
            )
            if irrbb["compliant"]:
                break
            deficit = irrbb["eve_ratio"] - 1.0
            step = PebcOptimizerMixin._adaptive_step_pebc(deficit)
            effective_dur = dur * w
            i_max = int(np.argmax(effective_dur))
            i_min = int(np.argmin(dur + (1.0 - w) * 100))
            transfer = min(step, w[i_max] * 0.5)
            w[i_max] -= transfer
            w[i_min] += transfer
            w = w / w.sum()
        return w

    @staticmethod
    def _enforce_market_caps_pebc(w: np.ndarray) -> np.ndarray:
        """Cap par market_capacity, redistribution proportionnelle."""
        capacities = np.array([
            s.market_capacity_credit_eur for s in SECTORS
        ] + [
            s.market_capacity_pe_eur for s in SECTORS
        ])
        total_cap = capacities.sum()
        max_share = capacities / total_cap
        clipped = np.minimum(w, max_share)
        excess = w.sum() - clipped.sum()
        if excess > 1e-10:
            room = max_share - clipped
            room_total = room.sum()
            if room_total > 1e-10:
                clipped += excess * room / room_total
        return clipped / clipped.sum()

    # ──────────────────────────────────────────────
    # OPTIMISEUR BL-CVaR 10 CELLULES — POINT D'ENTREE
    # ──────────────────────────────────────────────

    def optimize_allocation_pebc(
        self,
        macro_params: Optional[Dict[str, float]] = None,
        cvar_alpha: float = 0.95,
    ) -> Dict[str, object]:
        """Optimise l'allocation BL-CVaR sur 10 cellules (5 secteurs x 2 canaux).

        Phase 1 : BL-CVaR (CVaR gradient + spread compression)
            1. Extraire profit_rate par cellule depuis compute_raroc_eva()
            2. Construire corr_10x10 par cosine similarity macro
            3. Sigma = diag(vol) @ corr @ diag(vol)
            4. Monte Carlo CVaR 5000 scenarios
            5. Auto-calibrer kappa = mu_portfolio / CVaR(w_base)
            6. Scores BL-CVaR : score_i = mu_eff_i - kappa * dCVaR/dw_i
            7. Grid search alpha sur [0, 1] pour mixing w_base/w_opt

        Phase 2 : Contraintes reglementaires (CET1, LCR, NSFR, IRRBB)

        HMM regime conditioning (Hamilton 1989) :
            If macro_params provided and cvar_alpha not explicitly overridden,
            detect_regime() adjusts cvar_alpha dynamically.

        Args:
            macro_params: Dict macro optionnel pour conditionnement HMM.
            cvar_alpha: Niveau de confiance CVaR (defaut 0.95).

        Returns:
            Dict backward-compatible + metriques CVaR.
            ``method`` = ``"BL-CVaR-10C"``.
        """
        # ── HMM regime conditioning ──
        _hmm_regime = None
        if macro_params is not None and cvar_alpha == 0.95:
            try:
                from ifrs9_cockpit.engine.hmm_regime import detect_regime
                _hmm_result = detect_regime(macro_params)
                cvar_alpha = _hmm_result.cvar_alpha
                _hmm_regime = _hmm_result.regime
            except Exception:
                pass
        # compute_raroc_eva() is provided by MetricsMixin (already on PortfolioComparator)
        raroc_df: pl.DataFrame = self.compute_raroc_eva()  # type: ignore[attr-defined]
        coc = BASEL_CONFIG.cet1_target
        n = _N_CELLS

        # ── Extraction des profit_rates par cellule ──
        mu = np.zeros(n)
        for i, sector in enumerate(SECTORS):
            rc = raroc_df.filter(
                (pl.col("sector") == sector.name) & (pl.col("canal") == "Credit")
            )
            mu[i] = float(rc["profit_rate"][0]) if len(rc) > 0 else 0.0
            rp = raroc_df.filter(
                (pl.col("sector") == sector.name) & (pl.col("canal") == "PE")
            )
            mu[i + 5] = float(rp["profit_rate"][0]) if len(rp) > 0 else 0.0

        # ── Volatilites par cellule (capital-adjusted) ──
        vol_return = np.array([
            s.market_vol_credit for s in SECTORS
        ] + [
            s.market_vol_pe for s in SECTORS
        ])
        cap_ead = np.zeros(n)
        result_credit: pl.DataFrame = self.result_credit  # type: ignore[attr-defined]
        result_pe: pl.DataFrame = self.result_pe  # type: ignore[attr-defined]
        for i, sector in enumerate(SECTORS):
            cr = result_credit.filter(pl.col("sector") == sector.name)
            rwa_c = float(cr["rwa_credit"].sum()) if len(cr) > 0 else 0.0
            ead_c = float(cr["ead"].sum()) if len(cr) > 0 else 0.0
            cap_ead[i] = (rwa_c * coc) / max(ead_c, 1.0)
            pr = result_pe.filter(pl.col("sector") == sector.name)
            rwa_p = float(pr["rwa_pe"].sum()) if len(pr) > 0 else 0.0
            nav_p = float(pr["nav"].sum()) if len(pr) > 0 else 0.0
            cap_ead[i + 5] = (rwa_p * coc) / max(nav_p, 1.0)
        vol = vol_return * cap_ead

        # ── Poids de base (proportionnels a l'exposition) ──
        exposures = np.zeros(n)
        for i, sector in enumerate(SECTORS):
            cr = result_credit.filter(pl.col("sector") == sector.name)
            exposures[i] = float(cr["ead"].sum()) if len(cr) > 0 else 0.0
            pr = result_pe.filter(pl.col("sector") == sector.name)
            exposures[i + 5] = float(pr["nav"].sum()) if len(pr) > 0 else 0.0
        total_ead = float(exposures.sum())
        w_base = exposures / max(total_ead, 1.0)

        # ── Correlation 10x10 ──
        corr_10, lambda_lw = self._build_corr_matrix_pebc()

        # ── Matrice de covariance ──
        Sigma = np.outer(vol, vol) * corr_10

        # ── Phase 1 : BL-CVaR ──
        N_MC = 5000

        # Spread compression sur w_base
        mu_eff = self._spread_compression_pebc(w_base, total_ead, mu)

        # CVaR sur w_base
        cvar_base, scenarios, tail_mask = self._compute_cvar_pebc(
            w_base, Sigma, N_MC, alpha=cvar_alpha,
        )

        # Kappa auto-calibre
        mu_portfolio = float(w_base @ mu_eff)
        kappa = abs(mu_portfolio) / max(cvar_base, 1e-10)

        # Gradient CVaR : moyenne des scenarios dans la queue
        grad_cvar = -np.mean(scenarios[tail_mask], axis=0)

        # Scores BL-CVaR
        scores = mu_eff - kappa * grad_cvar

        # Passe 1 : softmax sur scores
        w_raw = self._softmax_pebc(scores, corr_10)

        # Passe 2 : compression sur w_raw, recalcul scores
        mu_eff_2 = self._spread_compression_pebc(w_raw, total_ead, mu)
        scores_2 = mu_eff_2 - kappa * grad_cvar
        w_opt = self._softmax_pebc(scores_2, corr_10)

        # Grid search alpha : w_trial = (1-alpha)*w_base + alpha*w_opt
        best_alpha = 0.0
        best_obj = float("-inf")
        for alpha_trial in np.linspace(0, 1, 201):
            w_trial = (1 - alpha_trial) * w_base + alpha_trial * w_opt
            cvar_trial, _, _ = self._compute_cvar_pebc(w_trial, Sigma, N_MC, alpha=cvar_alpha)
            mu_trial = self._spread_compression_pebc(w_trial, total_ead, mu)
            obj = float(w_trial @ mu_trial) - kappa * cvar_trial
            if obj > best_obj:
                best_obj = obj
                best_alpha = alpha_trial

        best_w = (1 - best_alpha) * w_base + best_alpha * w_opt
        best_w = best_w / best_w.sum()

        # ── Phase 2 : Contraintes reglementaires ──
        rwa_credit_total = float(result_credit["rwa_credit"].sum())
        rwa_pe_total = float(result_pe["rwa_pe"].sum())
        rwa_total = rwa_credit_total + rwa_pe_total
        actual_capital = rwa_total * coc

        # Poids credit vs PE
        w_credit_sum = float(best_w[:5].sum())
        w_pe_sum = float(best_w[5:].sum())

        rwa_weighted = w_credit_sum * rwa_credit_total + w_pe_sum * rwa_pe_total
        cet1_ratio = actual_capital / max(rwa_weighted, 1)
        headroom_eur = actual_capital - coc * rwa_weighted

        # 2a. Si CET1 insuffisant : binary search entre w_base et best_w
        regulatory_adj = 0.0
        if headroom_eur < 0:
            lo, hi = 0.0, 1.0
            for _ in range(25):
                mid = (lo + hi) / 2
                w_test = (1 - mid) * w_base + mid * best_w
                w_test = w_test / w_test.sum()
                wc_test = w_test[:5].sum()
                wp_test = w_test[5:].sum()
                rwa_w_test = wc_test * rwa_credit_total + wp_test * rwa_pe_total
                cet1_test = actual_capital / max(rwa_w_test, 1)
                if cet1_test >= coc:
                    lo = mid
                else:
                    hi = mid
            best_w = (1 - lo) * w_base + lo * best_w
            best_w = best_w / best_w.sum()
            w_credit_sum = float(best_w[:5].sum())
            w_pe_sum = float(best_w[5:].sum())
            rwa_weighted = w_credit_sum * rwa_credit_total + w_pe_sum * rwa_pe_total
            cet1_ratio = actual_capital / max(rwa_weighted, 1)
            headroom_eur = actual_capital - coc * rwa_weighted
            regulatory_adj = lo - best_alpha

        # 2b. LCR
        lcr_delta = 0.0
        lcr = compute_lcr_10(best_w, _CELL_NAMES_PEBC, SECTORS, total_ead)
        if lcr["lcr_ratio"] < BASEL_CONFIG.lcr_target:
            w_before_lcr = best_w.copy()
            best_w = self._adjust_for_lcr_pebc(best_w, total_ead)
            lcr_delta = float(np.sum(np.abs(best_w - w_before_lcr)))

        # 2c. NSFR
        nsfr_delta = 0.0
        nsfr = compute_nsfr_10(best_w, _CELL_NAMES_PEBC, SECTORS, total_ead)
        if nsfr["nsfr_ratio"] < BASEL_CONFIG.nsfr_target:
            w_before_nsfr = best_w.copy()
            best_w = self._adjust_for_nsfr_pebc(best_w, total_ead)
            nsfr_delta = float(np.sum(np.abs(best_w - w_before_nsfr)))

        # 2d. IRRBB + LCR joint (10 iterations)
        cet1_capital = actual_capital
        w_before_irrbb_lcr = best_w.copy()
        for _ in range(10):
            irrbb = compute_irrbb_eve_10(
                best_w, _CELL_NAMES_PEBC, SECTORS, total_ead, cet1_capital,
            )
            lcr_check = compute_lcr_10(best_w, _CELL_NAMES_PEBC, SECTORS, total_ead)
            if irrbb["compliant"] and lcr_check["lcr_ratio"] >= BASEL_CONFIG.lcr_target:
                break
            if not irrbb["compliant"]:
                best_w = self._adjust_for_irrbb_pebc(best_w, total_ead, cet1_capital)
            if lcr_check["lcr_ratio"] < BASEL_CONFIG.lcr_target:
                best_w = self._adjust_for_lcr_pebc(best_w, total_ead)
        irrbb_delta = float(np.sum(np.abs(best_w - w_before_irrbb_lcr)))

        # 2e. Market capacity caps
        best_w = self._enforce_market_caps_pebc(best_w)

        # Re-verify and restore regulatory compliance after capping
        lcr_post = compute_lcr_10(best_w, _CELL_NAMES_PEBC, SECTORS, total_ead)
        if lcr_post["lcr_ratio"] < BASEL_CONFIG.lcr_target:
            best_w = self._adjust_for_lcr_pebc(best_w, total_ead)
        nsfr_post = compute_nsfr_10(best_w, _CELL_NAMES_PEBC, SECTORS, total_ead)
        if nsfr_post["nsfr_ratio"] < BASEL_CONFIG.nsfr_target:
            best_w = self._adjust_for_nsfr_pebc(best_w, total_ead)
        irrbb_post = compute_irrbb_eve_10(
            best_w, _CELL_NAMES_PEBC, SECTORS, total_ead, cet1_capital,
        )
        if not irrbb_post["compliant"]:
            best_w = self._adjust_for_irrbb_pebc(best_w, total_ead, cet1_capital)
            # LCR may need re-fixing after IRRBB
            lcr_post2 = compute_lcr_10(best_w, _CELL_NAMES_PEBC, SECTORS, total_ead)
            if lcr_post2["lcr_ratio"] < BASEL_CONFIG.lcr_target:
                best_w = self._adjust_for_lcr_pebc(best_w, total_ead)

        # Final state after all adjustments
        w_credit_sum = float(best_w[:5].sum())
        w_pe_sum = float(best_w[5:].sum())
        rwa_weighted = w_credit_sum * rwa_credit_total + w_pe_sum * rwa_pe_total
        cet1_ratio = actual_capital / max(rwa_weighted, 1)
        headroom_eur = actual_capital - coc * rwa_weighted
        lcr_final = compute_lcr_10(best_w, _CELL_NAMES_PEBC, SECTORS, total_ead)
        nsfr_final = compute_nsfr_10(best_w, _CELL_NAMES_PEBC, SECTORS, total_ead)
        irrbb_final = compute_irrbb_eve_10(
            best_w, _CELL_NAMES_PEBC, SECTORS, total_ead, cet1_capital,
        )

        # ── Extraction des RAROC totaux ──
        total_credit = raroc_df.filter(
            (pl.col("canal") == "Credit") & (pl.col("sector") == "Total")
        )
        total_pe = raroc_df.filter(
            (pl.col("canal") == "PE") & (pl.col("sector") == "Total")
        )
        raroc_c = float(total_credit["raroc"][0]) if len(total_credit) > 0 else 0.0
        raroc_p = float(total_pe["raroc"][0]) if len(total_pe) > 0 else 0.0

        # ── Stress diagnostics ──
        stress = self._stress_intensity_pebc(raroc_c, raroc_p)
        illiq = self._asymmetric_illiquidity_pebc(stress)
        vol_mult = self._asymmetric_vol_multiplier_pebc(stress)
        conf_c, conf_p = self._bl_confidence_pebc(stress)
        pe_min_s, pe_max_s = self._asymmetric_pe_band_pebc(stress)

        # ── Poids sectoriels (normalises intra-canal) ──
        sector_weights_credit = {}
        for i, sector in enumerate(SECTORS):
            sector_weights_credit[sector.name] = round(
                best_w[i] / max(w_credit_sum, 1e-10), 4,
            )
        sector_weights_pe = {}
        for i, sector in enumerate(SECTORS):
            sector_weights_pe[sector.name] = round(
                best_w[i + 5] / max(w_pe_sum, 1e-10), 4,
            )

        # ── CVaR final ──
        cvar_final, _, _ = self._compute_cvar_pebc(best_w, Sigma, N_MC, alpha=cvar_alpha)

        # ── Spread compression final ──
        spread_comp = {}
        capacities = np.array([
            s.market_capacity_credit_eur for s in SECTORS
        ] + [
            s.market_capacity_pe_eur for s in SECTORS
        ])
        share = (best_w * total_ead) / np.maximum(capacities, 1.0)
        compression = -np.log(1 - np.minimum(share, 0.95))
        for i, cname in enumerate(_CELL_NAMES_PEBC):
            spread_comp[cname] = round(float(compression[i]), 6)

        return {
            # Backward-compatible
            "credit_allocation": round(float(w_credit_sum), 4),
            "pe_allocation": round(float(w_pe_sum), 4),
            "sector_weights_credit": sector_weights_credit,
            "sector_weights_pe": sector_weights_pe,
            "raroc_credit": round(raroc_c, 4),
            "raroc_pe": round(raroc_p, 4),
            "rwa_weighted": round(rwa_weighted, 0),
            "cet1_ratio": round(cet1_ratio, 4),
            "cet1_headroom": round(cet1_ratio - coc, 4),
            "headroom_m": round(headroom_eur / 1e6, 1),
            "feasible": headroom_eur >= 0,
            # BL-CVaR specific
            "method": "BL-CVaR-10C",
            "n_classes": _N_CELLS,
            "class_weights": {
                cname: round(float(best_w[i]), 6)
                for i, cname in enumerate(_CELL_NAMES_PEBC)
            },
            "cvar_95": round(cvar_final, 6),
            "cvar_alpha": round(cvar_alpha, 2),
            "hmm_regime": _hmm_regime,
            "kappa": round(float(kappa), 4),
            "n_scenarios": N_MC,
            "risk_alpha": round(float(best_alpha), 4),
            "portfolio_vol": round(float(np.sqrt(best_w @ Sigma @ best_w)), 6),
            "profit_rate_portfolio": round(float(best_w @ mu_eff), 6),
            "spread_compression": spread_comp,
            "covariance_shrinkage_lambda": round(lambda_lw, 4),
            "corr_10x10": corr_10.tolist(),
            "phase1_weights": {
                cname: round(float(w_opt[i]), 6)
                for i, cname in enumerate(_CELL_NAMES_PEBC)
            },
            # Stress diagnostics
            "stress_intensity": round(stress, 4),
            "illiquidity_premium": round(illiq, 6),
            "vol_multiplier": round(vol_mult, 4),
            "bl_confidence": (round(conf_c, 4), round(conf_p, 4)),
            "pe_band": (round(pe_min_s, 4), round(pe_max_s, 4)),
            # Regulatory
            "lcr_ratio": round(lcr_final["lcr_ratio"], 4),
            "lcr_compliant": lcr_final["lcr_ratio"] >= BASEL_CONFIG.lcr_target,
            "nsfr_ratio": round(nsfr_final["nsfr_ratio"], 4),
            "nsfr_compliant": nsfr_final["nsfr_ratio"] >= BASEL_CONFIG.nsfr_target,
            "irrbb_eve_ratio": round(irrbb_final["eve_ratio"], 4),
            "irrbb_compliant": irrbb_final["compliant"],
            "regulatory_adjustments": {
                "cet1_delta": round(regulatory_adj, 4),
                "lcr_delta": round(lcr_delta, 4),
                "nsfr_delta": round(nsfr_delta, 4),
                "irrbb_delta": round(irrbb_delta, 4),
            },
        }
