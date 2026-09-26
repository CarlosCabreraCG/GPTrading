"""
Indicadores técnicos como funciones puras sobre pandas Series.

Reglas:
- Todas las funciones son causales: el valor en t solo depende de datos hasta t.
- No usan shift(-n). No usan center=True en rolling.
- Devuelven Series con el mismo índice que la entrada.
- Los parámetros (n) se pasan explícitamente; no hay defaults ocultos.

Ninguna función usa volumen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Medias
# ---------------------------------------------------------------------------

def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


# ---------------------------------------------------------------------------
# Volatilidad
# ---------------------------------------------------------------------------

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    True Range. Usa el close anterior, por lo tanto es causal.
    TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """
    Average True Range. Wilder's smoothing (EMA con alpha=1/n).
    min_periods=n para que los primeros valores sean NaN, no prematuros.
    """
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rolling_std(series: pd.Series, n: int) -> pd.Series:
    """Desviación estándar móvil."""
    return series.rolling(window=n, min_periods=n).std()


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, n: int) -> pd.Series:
    """
    Relative Strength Index (Wilder).
    Devuelve valores en [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))

    # Si avg_loss == 0 y avg_gain > 0, RSI = 100 (caso límite)
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    # Si ambos son 0, RSI = 50 (mercado sin movimiento)
    out = out.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)

    return out


def roc(close: pd.Series, n: int) -> pd.Series:
    """Rate of Change: (close_t / close_{t-n}) - 1."""
    return close.pct_change(periods=n)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """
    Average Directional Index (Wilder).
    Devuelve valores en [0, 100] aproximadamente.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    atr_n = atr(high, low, close, n)

    plus_di = 100.0 * (
        plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_n
    )
    minus_di = 100.0 * (
        minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr_n
    )

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_out = dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    return adx_out


# ---------------------------------------------------------------------------
# Breakout / Bandas
# ---------------------------------------------------------------------------

def bollinger(
    close: pd.Series, n: int, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bandas de Bollinger.
    Devuelve (lower, mid, upper).
    """
    mid = sma(close, n)
    std = rolling_std(close, n)
    upper = mid + k * std
    lower = mid - k * std
    return lower, mid, upper


def rolling_max_prev(series: pd.Series, n: int) -> pd.Series:
    """
    Máximo de las últimas n velas EXCLUYENDO la actual.
    Equivalente a rolling(n).max().shift(1), pero explícito.
    """
    return series.rolling(window=n, min_periods=n).max().shift(1)


def rolling_min_prev(series: pd.Series, n: int) -> pd.Series:
    """Mínimo de las últimas n velas EXCLUYENDO la actual."""
    return series.rolling(window=n, min_periods=n).min().shift(1)