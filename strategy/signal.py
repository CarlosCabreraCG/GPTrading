"""
Representación de señales de trading.

Una señal es una intención: LONG, SHORT o FLAT.
La señal NO contiene información de precio ni de ejecución.
Solo dice "quiero estar en esta dirección para la próxima vela".
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class Signal(IntEnum):
    """
    Dirección de la posición deseada.

    Usamos IntEnum para que se pueda usar directamente en operaciones
    vectorizadas de numpy/pandas.
    """
    FLAT = 0
    LONG = 1
    SHORT = -1


def validate_signal_series(signals: pd.Series) -> None:
    """
    Verifica que una Series contiene solo valores válidos de Signal.
    Lanza ValueError si hay algo inesperado.
    """
    if not isinstance(signals, pd.Series):
        raise TypeError(f"Se esperaba pd.Series, recibido {type(signals)}")

    unique = set(signals.dropna().unique())
    valid = {int(s) for s in Signal}
    invalid = unique - valid
    if invalid:
        raise ValueError(
            f"Señales inválidas encontradas: {invalid}. "
            f"Valores permitidos: {valid}"
        )

    if signals.isna().any():
        raise ValueError(
            f"La serie de señales contiene {signals.isna().sum()} NaN. "
            "Todas las señales deben ser FLAT/LONG/SHORT explícitos."
        )


def signals_from_array(arr: np.ndarray, index: pd.Index) -> pd.Series:
    """Construye una Series de señales a partir de un array de enteros."""
    s = pd.Series(arr, index=index, name="signal")
    validate_signal_series(s)
    return s