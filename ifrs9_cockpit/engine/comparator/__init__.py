"""Comparateur Credit / PE et metriques avancees.

Module central de l'Epic 4 : compare les portefeuilles credit et PE,
calcule les metriques avancees (FR42), la concentration cross-cell (FR41),
et les denominateurs communs RAROC/EVA (FR44).

Pipeline :
    1. Metriques credit avancees (NPL ratio, cost of risk, Texas ratio, etc.)
    2. HHI cross-cell (10 cellules : 5 secteurs x 2 canaux)
    3. RAROC / EVA par cellule (secteur x canal)
"""

from __future__ import annotations

import polars as pl
from typing import Dict, Optional, Tuple

from ifrs9_cockpit.config import (
    REQUIRED_CREDIT_RESULT_COLS,
    REQUIRED_PE_RESULT_COLS,
)
from ifrs9_cockpit.engine.comparator.crr3 import compute_crr3_rw
from ifrs9_cockpit.engine.comparator.metrics import MetricsMixin
from ifrs9_cockpit.engine.comparator.optimizer import OptimizerMixin
from ifrs9_cockpit.engine.comparator.sensitivity import SensitivityMixin

# Capital CET1 = BASEL_CONFIG.rwa_budget x cet1_target = 3.69T x 13% = 479.7 Md EUR.
# rwa_budget represente le RWA total de la banque (pas du portefeuille).


class PortfolioComparator(MetricsMixin, OptimizerMixin, SensitivityMixin):
    """Comparateur de portefeuilles credit et PE.

    Orchestre le calcul des metriques avancees, de la concentration
    et des denominateurs communs (RAROC, EVA) pour la comparaison
    cross-canal et multi-actif (10 classes).

    Attributes:
        result_credit: DataFrame resultat du pipeline credit (ECLCalculator).
        result_pe: DataFrame resultat du pipeline PE (PECalculator).
        df_balance_sheet_ecl: DataFrame ECL des 10 classes (balance_sheet_ecl).
    """

    def __init__(
        self,
        result_credit: pl.DataFrame,
        result_pe: pl.DataFrame,
        df_balance_sheet_ecl: Optional[pl.DataFrame] = None,
        macro_params: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialise le comparateur.

        Args:
            result_credit: Resultat ECLCalculator.calculate().
            result_pe: Resultat PECalculator.calculate().
            df_balance_sheet_ecl: Resultat compute_balance_sheet_ecl() (10 rows).
            macro_params: Parametres macro du scenario courant (pour cout de funding).

        Raises:
            AssertionError: Si les colonnes requises sont absentes.
        """
        missing_credit = REQUIRED_CREDIT_RESULT_COLS - set(result_credit.columns)
        assert not missing_credit, f"Colonnes credit manquantes: {missing_credit}"

        missing_pe = REQUIRED_PE_RESULT_COLS - set(result_pe.columns)
        assert not missing_pe, f"Colonnes PE manquantes: {missing_pe}"

        self.result_credit = result_credit
        self.result_pe = result_pe
        self.df_balance_sheet_ecl = df_balance_sheet_ecl
        self._macro_params = macro_params
        self._raroc_eva_cache: Optional[pl.DataFrame] = None
        self._raroc_multiclass_cache: Optional[pl.DataFrame] = None


__all__ = ["compute_crr3_rw", "PortfolioComparator"]
