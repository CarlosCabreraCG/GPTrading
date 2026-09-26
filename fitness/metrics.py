"""
Métricas de rendimiento para el fitness.

Todas las funciones operan sobre una BacktestResult o sobre sus componentes
directos (equity curve, lista de trades). No dependen del GP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.results import BacktestResult, compute_metrics


def annualization_factor(index) -> float:
    """Factor de anualización. Acepta DatetimeIndex o cualquier índice."""
    if not isinstance(index, pd.DatetimeIndex):
        # Fallback: asumir 15m (900s) como frecuencia por defecto
        return float((365.25 * 24 * 3600) / 900.0)
    if len(index) < 3:
        return 1.0
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    freq = float(deltas.median())
    if freq <= 0:
        return 1.0
    return float((365.25 * 24 * 3600) / freq)


def sharpe(equity: pd.Series) -> float:
    """Sharpe anualizado (rf=0)."""
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().fillna(0.0)
    std = returns.std()
    if std <= 0:
        return 0.0
    ann = annualization_factor(equity.index)
    return float((returns.mean() / std) * np.sqrt(ann))


def sortino(equity: pd.Series) -> float:
    """Sortino anualizado (solo downside deviation)."""
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().fillna(0.0)
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std() <= 0:
        return 0.0
    ann = annualization_factor(equity.index)
    return float((returns.mean() / downside.std()) * np.sqrt(ann))


def max_drawdown(equity: pd.Series) -> float:
    """Max drawdown como fracción (negativo)."""
    if len(equity) < 2:
        return 0.0
    rolling_max = equity.cummax()
    dd = (equity / rolling_max) - 1.0
    return float(dd.min())


def calmar(equity: pd.Series) -> float:
    """Calmar = total_return / |max_drawdown|."""
    if len(equity) < 2:
        return 0.0
    total_ret = float((equity.iloc[-1] / equity.iloc[0]) - 1.0)
    mdd = max_drawdown(equity)
    if mdd >= 0:
        return 0.0
    return total_ret / abs(mdd)


def profit_factor(trades: list) -> float:
    """Suma de PnLs positivos / |suma de PnLs negativos|."""
    if not trades:
        return 0.0
    pnls = np.array([t.net_pnl for t in trades])
    gross_profit = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    gross_loss = float(-pnls[pnls < 0].sum()) if (pnls < 0).any() else 0.0
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def expectancy(trades: list) -> float:
    """PnL medio por operación."""
    if not trades:
        return 0.0
    return float(np.mean([t.net_pnl for t in trades]))


def turnover(position: pd.Series) -> int:
    """Número de cambios de posición."""
    if len(position) < 2:
        return 0
    return int((position.diff().fillna(0) != 0).sum())


def compute_all(result: BacktestResult) -> dict[str, float]:
    """Atajo: devuelve el diccionario completo de métricas."""
    return compute_metrics(result)