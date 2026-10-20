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
from fitness.metrics import compute_metrics

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

    def metrics(self, symbol: str) -> dict[str, float]:
        return compute_metrics(self, symbol)