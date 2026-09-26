"""
Registro de features permitidas.

Cada feature tiene:
- nombre estable (será el terminal que GP verá)
- función que la calcula
- parámetros por defecto
- descripción

GP NO puede inventar features. Solo puede usar las que están aquí.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from features import indicators as ind


# ---------------------------------------------------------------------------
# Definición de una feature
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    func: Callable[[pd.DataFrame], pd.Series]
    description: str


# ---------------------------------------------------------------------------
# Funciones de cálculo (reciben el DataFrame OHLC limpio)
# ---------------------------------------------------------------------------

def _return(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change(periods=n)


def _log_return(close: pd.Series, n: int) -> pd.Series:
    return np.log(close / close.shift(n))


def _atr_norm(df: pd.DataFrame, n: int) -> pd.Series:
    a = ind.atr(df["high"], df["low"], df["close"], n)
    return a / df["close"]


def _rolling_std_returns(close: pd.Series, n: int) -> pd.Series:
    return close.pct_change().rolling(window=n, min_periods=n).std()


def _hl_range_atr(df: pd.DataFrame, n: int) -> pd.Series:
    a = ind.atr(df["high"], df["low"], df["close"], n)
    return (df["high"] - df["low"]) / a


def _dist_sma_atr(df: pd.DataFrame, n: int, atr_n: int) -> pd.Series:
    s = ind.sma(df["close"], n)
    a = ind.atr(df["high"], df["low"], df["close"], atr_n)
    return (df["close"] - s) / a


def _dist_ema_atr(df: pd.DataFrame, n: int, atr_n: int) -> pd.Series:
    e = ind.ema(df["close"], n)
    a = ind.atr(df["high"], df["low"], df["close"], atr_n)
    return (df["close"] - e) / a


def _close_over_sma(close: pd.Series, n: int) -> pd.Series:
    return close / ind.sma(close, n) - 1.0


def _close_over_ema(close: pd.Series, n: int) -> pd.Series:
    return close / ind.ema(close, n) - 1.0


def _rsi(close: pd.Series, n: int) -> pd.Series:
    return ind.rsi(close, n)


def _roc(close: pd.Series, n: int) -> pd.Series:
    return ind.roc(close, n)


def _adx(df: pd.DataFrame, n: int) -> pd.Series:
    return ind.adx(df["high"], df["low"], df["close"], n)


def _dist_prev_high_atr(df: pd.DataFrame, n: int, atr_n: int) -> pd.Series:
    prev_high = ind.rolling_max_prev(df["high"], n)
    a = ind.atr(df["high"], df["low"], df["close"], atr_n)
    return (df["close"] - prev_high) / a


def _dist_prev_low_atr(df: pd.DataFrame, n: int, atr_n: int) -> pd.Series:
    prev_low = ind.rolling_min_prev(df["low"], n)
    a = ind.atr(df["high"], df["low"], df["close"], atr_n)
    return (df["close"] - prev_low) / a


def _bb_position(df: pd.DataFrame, n: int, k: float) -> pd.Series:
    lower, mid, upper = ind.bollinger(df["close"], n, k)
    width = (upper - lower).replace(0.0, np.nan)
    return (df["close"] - mid) / width


def _bb_width(df: pd.DataFrame, n: int, k: float) -> pd.Series:
    lower, mid, upper = ind.bollinger(df["close"], n, k)
    return (upper - lower) / mid


# ---------------------------------------------------------------------------
# Catálogo de features
# ---------------------------------------------------------------------------

FEATURES: dict[str, FeatureSpec] = {
    # --- Retornos ---
    "return_1":  FeatureSpec("return_1",  lambda df: _return(df["close"], 1),  "Retorno simple 1 vela"),
    "return_5":  FeatureSpec("return_5",  lambda df: _return(df["close"], 5),  "Retorno simple 5 velas"),
    "return_20": FeatureSpec("return_20", lambda df: _return(df["close"], 20), "Retorno simple 20 velas"),
    "log_return_1":  FeatureSpec("log_return_1",  lambda df: _log_return(df["close"], 1),  "Log-retorno 1 vela"),
    "log_return_5":  FeatureSpec("log_return_5",  lambda df: _log_return(df["close"], 5),  "Log-retorno 5 velas"),
    "log_return_20": FeatureSpec("log_return_20", lambda df: _log_return(df["close"], 20), "Log-retorno 20 velas"),

    # --- Volatilidad ---
    "atr_14_norm":      FeatureSpec("atr_14_norm",      lambda df: _atr_norm(df, 14),            "ATR(14) / close"),
    "rolling_std_20":   FeatureSpec("rolling_std_20",   lambda df: _rolling_std_returns(df["close"], 20), "std de retornos, ventana 20"),
    "hl_range_atr_14":  FeatureSpec("hl_range_atr_14",  lambda df: _hl_range_atr(df, 14),        "(high-low) / ATR(14)"),

    # --- Tendencia ---
    "dist_sma_20_atr":  FeatureSpec("dist_sma_20_atr",  lambda df: _dist_sma_atr(df, 20, 14),    "(close - SMA20) / ATR(14)"),
    "dist_ema_20_atr":  FeatureSpec("dist_ema_20_atr",  lambda df: _dist_ema_atr(df, 20, 14),    "(close - EMA20) / ATR(14)"),
    "close_over_sma_20": FeatureSpec("close_over_sma_20", lambda df: _close_over_sma(df["close"], 20), "close/SMA20 - 1"),
    "close_over_ema_20": FeatureSpec("close_over_ema_20", lambda df: _close_over_ema(df["close"], 20), "close/EMA20 - 1"),

    # --- Momentum ---
    "rsi_14": FeatureSpec("rsi_14", lambda df: _rsi(df["close"], 14), "RSI(14)"),
    "roc_10": FeatureSpec("roc_10", lambda df: _roc(df["close"], 10), "Rate of Change 10"),
    "adx_14": FeatureSpec("adx_14", lambda df: _adx(df, 14),          "ADX(14)"),

    # --- Breakout ---
    "dist_prev_high_20": FeatureSpec("dist_prev_high_20", lambda df: _dist_prev_high_atr(df, 20, 14), "(close - max(high,20).shift(1)) / ATR(14)"),
    "dist_prev_low_20":  FeatureSpec("dist_prev_low_20",  lambda df: _dist_prev_low_atr(df, 20, 14),  "(close - min(low,20).shift(1)) / ATR(14)"),
    "bb_position_20":    FeatureSpec("bb_position_20",    lambda df: _bb_position(df, 20, 2.0),       "(close - BB_mid) / (BB_up - BB_low)"),
    "bb_width_20":       FeatureSpec("bb_width_20",       lambda df: _bb_width(df, 20, 2.0),          "(BB_up - BB_low) / BB_mid"),
}


def feature_names() -> list[str]:
    """Lista ordenada de nombres de features disponibles."""
    return list(FEATURES.keys())