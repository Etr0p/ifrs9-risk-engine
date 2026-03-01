"""DatasetBundle — conteneur type pour les 4 DataFrames + positions.

Remplace le pattern df_balance_sheet.attrs["*_positions"] qui n'existe
pas en Polars. Le __iter__ assure la retro-compatibilite avec le
tuple-unpacking existant:

    df_credit, df_pe, df_history, df_balance_sheet = generate_dataset()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


@dataclass
class DatasetBundle:
    """Bundle contenant les 4 DataFrames principaux et les positions par classe d'actif."""

    df_credit: Any
    df_pe: Any
    df_history: Any
    df_balance_sheet: Any

    # Positions par classe d'actif (remplace les 12 .attrs)
    mortgage_positions: Optional[Any] = field(default=None, repr=False)
    consumer_positions: Optional[Any] = field(default=None, repr=False)
    trade_positions: Optional[Any] = field(default=None, repr=False)
    project_positions: Optional[Any] = field(default=None, repr=False)
    securitisation_positions: Optional[Any] = field(default=None, repr=False)
    sovereign_positions: Optional[Any] = field(default=None, repr=False)
    covered_bonds_positions: Optional[Any] = field(default=None, repr=False)
    interbank_positions: Optional[Any] = field(default=None, repr=False)
    equity_positions: Optional[Any] = field(default=None, repr=False)
    corporate_bonds_positions: Optional[Any] = field(default=None, repr=False)
    repos_sft_positions: Optional[Any] = field(default=None, repr=False)
    derivatives_cva_positions: Optional[Any] = field(default=None, repr=False)

    def __iter__(self) -> Iterator[Any]:
        """Retro-compat: tuple unpacking fonctionne toujours.

        Usage:
            df_credit, df_pe, df_history, df_balance_sheet = bundle
        """
        yield self.df_credit
        yield self.df_pe
        yield self.df_history
        yield self.df_balance_sheet

    def __len__(self) -> int:
        """Retourne 4 pour la retro-compat avec len(tuple)."""
        return 4

    def __getitem__(self, idx: int) -> Any:
        """Acces par index pour la retro-compat.

        Usage:
            bundle[0]  # df_credit
            bundle[3]  # df_balance_sheet
        """
        return (self.df_credit, self.df_pe, self.df_history, self.df_balance_sheet)[idx]
