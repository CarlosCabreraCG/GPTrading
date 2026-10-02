"""
Función de fitness final.

Combina:
- Sharpe sobre el slice de datos
- Penalizaciones (complejidad, turnover, inactividad, estabilidad)
- Restricción de mínimo de trades

La función de fitness se construye con `make_fitness_fn(...)`, que devuelve
un callable apto para `evolve(...)`.

IMPORTANTE: el fitness se evalúa sobre un slice (IS, OOS o fold). El slice
se pasa como argumento al construir la función, no en cada llamada, para
que el callable sea simple y compatible con `evolve`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from config.settings import DEFAULT_COSTS, CostModel
from fitness.complexity import ComplexityWeights, complexity_penalty
from fitness.metrics import sharpe, sortino, turnover
from gp.evaluator import EvalContext
from gp.individual import Individual


@dataclass(frozen=True)
class FitnessConfig:
    """Parámetros del fitness."""
    # Backtest
    symbol: str = "EURUSD" 
    initial_capital: float = 10_000.0
    costs: CostModel | None = None 
    size: float = 1.0

    # Restricciones
    min_trades: int = 5
    max_turnover_ratio: float = 0.5   # transiciones / n_velas
    max_flat_ratio: float = 0.98      # si está flat >98% del tiempo, penalizar

    # Pesos de penalización
    complexity: ComplexityWeights = field(default_factory=ComplexityWeights)
    turnover_penalty_weight: float = 0.02
    inactivity_penalty_weight: float = 0.5

    # Métrica principal
    primary_metric: str = "sharpe"    # "sharpe" o "sortino"

    def __post_init__(self):
        if self.costs is None:
            from config.settings import get_costs
            object.__setattr__(self, "costs", get_costs(self.symbol))
            
# ---------------------------------------------------------------------------
# Construcción del fitness
# ---------------------------------------------------------------------------

def make_fitness_fn(
    ohlc: pd.DataFrame,
    ctx: EvalContext,
    config: FitnessConfig,
):
    """
    Devuelve una función fitness_fn(individual) -> float.

    El ohlc y el ctx están capturados en el closure. Esto permite que el
    callable tenga la firma simple que `evolve` espera.
    """
    backtest_config = BacktestConfig(
        symbol=config.symbol,
        initial_capital=config.initial_capital,
        costs=config.costs,
        size=config.size,
    )

    n_rows = len(ohlc)

    def fitness_fn(ind: Individual) -> float:
        try:
            signals_array = ind.evaluate(ctx)
        except Exception:
            return float("-inf")

        # Convertir a Series con el índice del ohlc
        signals = pd.Series(signals_array, index=ohlc.index, name="signal")

        # Restricción: inactividad total
        n_active = int((signals != 0).sum())
        if n_active == 0:
            return float("-inf")

        flat_ratio = 1.0 - (n_active / n_rows)
        if flat_ratio > config.max_flat_ratio:
            return float("-inf")

        # Backtest
        try:
            result = run_backtest(ohlc, signals, backtest_config)
        except Exception:
            return float("-inf")

        # Mínimo de trades
        if result.n_trades < config.min_trades:
            return float("-inf")

        # Métrica principal
        if config.primary_metric == "sharpe":
            perf = sharpe(result.equity)
        elif config.primary_metric == "sortino":
            perf = sortino(result.equity)
        else:
            raise ValueError(f"Métrica no soportada: {config.primary_metric}")

        # Turnover
        to = turnover(result.position)
        turnover_ratio = to / n_rows
        if turnover_ratio > config.max_turnover_ratio:
            return float("-inf")

        # Penalizaciones
        p_complexity = complexity_penalty(ind, config.complexity)
        p_turnover = config.turnover_penalty_weight * turnover_ratio
        p_inactivity = config.inactivity_penalty_weight * flat_ratio

        fitness = perf - p_complexity - p_turnover - p_inactivity
        return float(fitness)

    return fitness_fn