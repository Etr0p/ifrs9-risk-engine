"""Couche de compatibilite Pandas <-> Polars.

Fournit des fonctions de conversion bidirectionnelles pour la migration
incrementale. Chaque module peut accepter l'un ou l'autre format et
convertir au besoin.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl


def to_pandas(df: Any) -> pd.DataFrame:
    """Convertit en pd.DataFrame si necessaire.

    Identite si deja pd.DataFrame, conversion si pl.DataFrame.
    """
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pl.LazyFrame):
        return df.collect().to_pandas()
    return df


def to_polars(df: Any) -> pl.DataFrame:
    """Convertit en pl.DataFrame si necessaire.

    Identite si deja pl.DataFrame, conversion si pd.DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    return df


def ensure_numpy(col: Any, dtype: Any = None) -> np.ndarray:
    """Extrait un numpy array depuis Polars Series, Pandas Series, ou numpy.

    Gere la conversion null -> NaN pour les frontieres numpy.

    Args:
        col: Polars Series, Pandas Series, ou numpy array.
        dtype: dtype numpy cible (optionnel).

    Returns:
        np.ndarray avec le dtype demande.
    """
    if isinstance(col, pl.Series):
        arr = col.to_numpy(allow_copy=True)
    elif isinstance(col, pd.Series):
        arr = col.values
    elif isinstance(col, np.ndarray):
        arr = col
    else:
        arr = np.asarray(col)
    if dtype is not None:
        return arr.astype(dtype)
    return arr
