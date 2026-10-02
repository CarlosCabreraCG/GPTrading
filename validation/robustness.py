"""
Pruebas de robustez.

Evalúan un individuo YA ENTRENADO sobre datos distintos a los de entrenamiento.
No re-optimizan. Si el individuo funciona sin reajustar, hay evidencia de
generalización estructural.

Pruebas:
- Cross-pair: entrenar en EURUSD, evaluar en GBPUSD, USDJPY, AUDUSD.
- Cross-timeframe: entrenar en 15m, evaluar en H1 y H4.

IMPORTANTE: cross-timeframe sobre el mismo período NO es muestra
independiente. Solo es evidencia de robustez estructural, no de OOS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from config.settings import DEFAULT_COSTS
from features.engine import compute_features, drop_warmup_rows
from fitness.metrics import sharpe
from fitness.objective import FitnessConfig
from gp.evaluator import EvalContext
from gp.individual import Individual


@dataclass
class RobustnessResult:
    """Resultado de una prueba de robustez."""
    label: str
    n_bars: int
    n_trades: int
    total_return: float
    sharpe: float
    max_dd: float
    signals_distribution: dict[int, int]


def _build_context(ohlc: pd.DataFrame) -> tuple[EvalContext, pd.DataFrame]:
    feats = compute_features(ohlc)
    df_aligned, feats_aligned = drop_warmup_rows(ohlc, feats)
    ctx = EvalContext(
        features={name: feats_aligned[name].to_numpy() for name in feats_aligned.columns},
        n=len(feats_aligned),
    )
    return ctx, df_aligned


def evaluate_on_other_data(
    individual: Individual,
    ohlc: pd.DataFrame,
    label: str,
    symbol: str,
    initial_capital: float = 10_000.0,
    fitness_config: FitnessConfig | None = None,
) -> RobustnessResult:
    """
    Evalúa un individuo sobre un DataFrame OHLC cualquiera.

    No entrena. Solo evalúa la señal y corre el backtest.
    """
    if fitness_config is None:
        fitness_config = FitnessConfig(symbol=symbol)
    if fitness_config.symbol != symbol:
        raise ValueError(
            f"symbol={symbol} no coincide con fitness_config.symbol="
            f"{fitness_config.symbol}"
        )
    ctx, df_aligned = _build_context(ohlc)

    # Evaluar señal
    signals_array = individual.evaluate(ctx)
    signals = pd.Series(signals_array, index=df_aligned.index, name="signal")

    # Distribución de señales
    unique, counts = np.unique(signals_array, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique, counts)}

    # Backtest
    bt_config = BacktestConfig(
        symbol=symbol,
        initial_capital=initial_capital,
        costs=fitness_config.costs,
        size=fitness_config.size,
    )
    result = run_backtest(df_aligned, signals, bt_config)

    total_ret = float((result.equity.iloc[-1] / initial_capital) - 1.0)
    sh = sharpe(result.equity)
    dd = float((result.equity / result.equity.cummax() - 1.0).min())

    return RobustnessResult(
        label=label,
        n_bars=len(df_aligned),
        n_trades=result.n_trades,
        total_return=total_ret,
        sharpe=sh,
        max_dd=dd,
        signals_distribution=dist,
    )


def cross_pair_test(
    individual: Individual,
    pairs_data: dict[str, pd.DataFrame],
    symbol_map: dict[str, str], 
    initial_capital: float = 10_000.0,
) -> list[RobustnessResult]:
    """
    Evalúa un individuo sobre múltiples pares.

    pairs_data: dict {pair_name: ohlc_df}.
    Devuelve una lista de RobustnessResult, uno por par.
    """
    results = []
    for pair_name, ohlc in pairs_data.items():
        if pair_name not in symbol_map:
            raise KeyError(
                f"Falta symbol para '{pair_name}' en symbol_map. "
                f"Disponibles: {sorted(symbol_map.keys())}"
            )
        sym = symbol_map[pair_name]
        try:
            r = evaluate_on_other_data(
                individual, ohlc, label=pair_name,
                symbol=sym,
                initial_capital=initial_capital,
            )
            results.append(r)
        except Exception as e:
            results.append(
                RobustnessResult(
                    label=f"{pair_name} (ERROR: {e})",
                    n_bars=0, n_trades=0,
                    total_return=0.0, sharpe=0.0, max_dd=0.0,
                    signals_distribution={},
                )
            )
    return results


def cross_timeframe_test(
    individual: Individual,
    tf_data: dict[str, pd.DataFrame],
    symbol_map: dict[str, str],
    initial_capital: float = 10_000.0,
) -> list[RobustnessResult]:
    """
    Evalúa un individuo sobre múltiples temporalidades.

    tf_data: dict {timeframe_label: ohlc_df}.
    """
    results = []
    for label, ohlc in tf_data.items():
        if label not in symbol_map:
            raise KeyError(
                f"Falta symbol para '{label}' en symbol_map. "
                f"Disponibles: {sorted(symbol_map.keys())}"
            )
        sym = symbol_map[label]
        try:
            r = evaluate_on_other_data(
                individual, ohlc, label=label,
                symbol=sym,
                initial_capital=initial_capital,
            )
            results.append(r)
        except Exception as e:
            results.append(
                RobustnessResult(
                    label=f"{label} (ERROR: {e})",
                    n_bars=0, n_trades=0,
                    total_return=0.0, sharpe=0.0, max_dd=0.0,
                    signals_distribution={},
                )
            )
    return results


def summarize_robustness(results: list[RobustnessResult]) -> dict:
    """Resumen agregado de una batería de pruebas."""
    if not results:
        return {"n_tests": 0}

    sharpes = [r.sharpe for r in results if np.isfinite(r.sharpe)]
    returns = [r.total_return for r in results]

    return {
        "n_tests": len(results),
        "n_positive_sharpe": int(sum(1 for s in sharpes if s > 0)),
        "pct_positive_sharpe": float(np.mean([s > 0 for s in sharpes])) if sharpes else 0.0,
        "mean_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
        "median_sharpe": float(np.median(sharpes)) if sharpes else 0.0,
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "worst_return": float(np.min(returns)),
        "best_return": float(np.max(returns)),
    }