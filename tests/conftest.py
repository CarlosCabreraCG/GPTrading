"""
Fixtures y fábricas de datos sintéticos compartidas por los tests.

Diseño data-driven:
- `make_ohlc` genera OHLC sintético (random walk) reproducible.
- `add_required_columns` añade una serie sintética por cada columna que el
  registro de features requiere y que no es OHLC (p. ej. `dxy_close`).
  Las columnas salen de `required_columns()`, no de una lista a mano: añadir
  una feature cross-asset nueva no requiere tocar este archivo.
- `make_full_ohlc` = OHLC + todas las columnas requeridas por el registro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.registry import OHLC_COLUMNS, required_columns


# ---------------------------------------------------------------------------
# Fábricas
# ---------------------------------------------------------------------------

def make_ohlc(
    n: int = 500,
    seed: int = 42,
    start_price: float = 1.10,
    drift: float = 0.00005,
    vol: float = 0.001,
    wick: float = 0.0005,
    freq: str = "15min",
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """
    OHLC sintético: random walk log-normal con OHLC coherente.

    Orden de llamadas al RNG fijo (normal, uniform, uniform) para que los datos
    sean reproducibles con la misma seed.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")

    returns = rng.normal(loc=drift, scale=vol, size=n)
    close = start_price * np.exp(np.cumsum(returns))

    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, wick, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, wick, size=n))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )


def make_linear_ohlc(
    n: int,
    start_price: float,
    end_price: float,
    half_range: float = 0.0001,
    freq: str = "15min",
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """OHLC determinista con close lineal entre start_price y end_price."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = np.linspace(start_price, end_price, n)
    return pd.DataFrame(
        {"open": close, "high": close + half_range,
         "low": close - half_range, "close": close},
        index=idx,
    )


def extra_required_columns() -> list[str]:
    """Columnas requeridas por el registro que no son OHLC, ordenadas."""
    return sorted(required_columns() - OHLC_COLUMNS)


def add_required_columns(df: pd.DataFrame, seed: int = 1000) -> pd.DataFrame:
    """
    Devuelve una copia de df con una serie sintética (random walk positivo)
    por cada columna requerida por el registro que df no tenga.

    Cada columna usa una seed derivada distinta para que no sean idénticas.
    """
    out = df.copy()
    n = len(out)
    for i, col in enumerate(extra_required_columns()):
        if col in out.columns:
            continue
        rng = np.random.default_rng(seed + i)
        returns = rng.normal(loc=0.0, scale=0.001, size=n)
        out[col] = 100.0 * np.exp(np.cumsum(returns))
    return out


def make_full_ohlc(**kwargs) -> pd.DataFrame:
    """OHLC + todas las columnas que requiere el registro de features."""
    return add_required_columns(make_ohlc(**kwargs))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlc_only() -> pd.DataFrame:
    """500 velas de 15m, solo OHLC (sin columnas cross-asset)."""
    return make_ohlc()


@pytest.fixture
def ohlc_full() -> pd.DataFrame:
    """500 velas de 15m, OHLC + todas las columnas requeridas por el registro."""
    return make_full_ohlc()
