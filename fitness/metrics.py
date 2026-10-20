"""
Métricas de rendimiento para el fitness.

Todas las funciones operan sobre una BacktestResult o sobre sus componentes
directos (equity curve, lista de trades). No dependen del GP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtest.results import BacktestResult

def annualization_factor(index, symbol: str) -> float:
    """
    Factor de anualización basado en barras reales por año del símbolo.
    """
    from config.settings import bars_per_year

    if len(index) < 3:
        return 1.0

    if not isinstance(index, pd.DatetimeIndex):
        # Fallback conservador: asumir 15m
        bar_seconds = 900.0
    else:
        deltas = index.to_series().diff().dropna().dt.total_seconds()
        bar_seconds = float(deltas.median())
        if bar_seconds <= 0:
            return 1.0

    return bars_per_year(symbol, bar_seconds)


def sharpe(equity: pd.Series, symbol: str) -> float:
    """Sharpe anualizado (rf=0)."""
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().fillna(0.0)
    std = returns.std()
    if std <= 0:
        return 0.0
    ann = annualization_factor(equity.index, symbol)
    return float((returns.mean() / std) * np.sqrt(ann))


def sortino(equity: pd.Series, symbol: str) -> float:
    """Sortino anualizado (solo downside deviation)."""
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().fillna(0.0)
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std() <= 0:
        return 0.0
    ann = annualization_factor(equity.index, symbol)
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

def compute_metrics(result, symbol: str) -> dict[str, float]:
    """Métricas completas de un BacktestResult."""
    equity = result.equity
    trades = result.trades

    gross_total = sum(t.gross_pnl for t in trades)
    cost_total = sum(t.total_cost for t in trades)

    return {
        "sharpe": sharpe(equity, symbol),
        "sortino": sortino(equity, symbol),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity),
        "profit_factor": profit_factor(trades),
        "expectancy": expectancy(trades),
        "total_return": float((equity.iloc[-1] / equity.iloc[0]) - 1.0),
        "n_trades": len(trades),
        "turnover": turnover(result.position),
        "final_equity": float(equity.iloc[-1]),
        "total_gross_pnl": float(gross_total),
        "total_cost": float(cost_total),
    }