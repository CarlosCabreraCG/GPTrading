"""
Resultados del backtest.

Contiene:
- BacktestResult: equity curve, trades, métricas.
- compute_metrics: función que calcula todas las métricas a partir de equity y trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.execution import TradeRecord


@dataclass
class BacktestResult:
    """
    Resultado completo de un backtest.

    equity: Serie indexada por timestamp, valor de la cartera al close de cada vela.
    position: Serie con la dirección de la posición al close de cada vela.
    trades: lista de TradeRecord (operaciones cerradas).
    initial_capital: capital inicial.
    """
    equity: pd.Series
    position: pd.Series
    trades: list[TradeRecord] = field(default_factory=list)
    initial_capital: float = 10_000.0

    @property
    def total_return(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.equity.iloc[-1] / self.initial_capital) - 1.0

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def metrics(self) -> dict[str, float]:
        """Calcula y devuelve todas las métricas relevantes."""
        return compute_metrics(self)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _equity_returns(equity: pd.Series) -> pd.Series:
    """Retornos por vela de la curva de equity."""
    return equity.pct_change().fillna(0.0)


def _annualization_factor(index: pd.DatetimeIndex) -> float:
    """
    Factor de anualización basado en la frecuencia modal del índice.

    Para 15m: 4 velas/hora * 24h * 365d = 35040 velas/año (forex ~24/5,
    pero usamos 24/7 como aproximación conservadora).
    """
    if len(index) < 3:
        return 1.0
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    freq_seconds = float(deltas.median())
    if freq_seconds <= 0:
        return 1.0
    periods_per_year = (365.25 * 24 * 3600) / freq_seconds
    return float(periods_per_year)


def compute_metrics(result: BacktestResult) -> dict[str, float]:
    """Calcula el conjunto completo de métricas."""
    equity = result.equity
    returns = _equity_returns(equity)
    ann_factor = _annualization_factor(equity.index)

    # Sharpe (asumiendo rf=0)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(ann_factor)
    else:
        sharpe = 0.0

    # Sortino (solo downside deviation)
    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * np.sqrt(ann_factor)
    else:
        sortino = 0.0

    # Max Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max) - 1.0
    max_dd = float(drawdown.min())

    # Calmar
    total_ret = result.total_return
    calmar = total_ret / abs(max_dd) if max_dd < 0 else 0.0

    # Profit Factor y Expectancy (sobre trades cerrados)
    if result.trades:
        pnls = np.array([t.net_pnl for t in result.trades])
        gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0.0
        gross_loss = -pnls[pnls < 0].sum() if (pnls < 0).any() else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        expectancy = float(pnls.mean())
    else:
        profit_factor = 0.0
        expectancy = 0.0

    # Turnover: número de transiciones de posición
    pos = result.position
    turnover = int((pos.diff().fillna(0) != 0).sum())

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "profit_factor": float(profit_factor),
        "expectancy": expectancy,
        "total_return": float(total_ret),
        "n_trades": result.n_trades,
        "turnover": turnover,
        "final_equity": float(equity.iloc[-1]),
    }