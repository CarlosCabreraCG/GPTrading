"""
Walk-Forward Optimization.

Para cada fold:
1. Calcular features sobre IS y OOS por separado (evitando look-ahead
   entre bloques).
2. Construir el EvalContext de IS.
3. Entrenar GP en IS con el fitness.
4. Evaluar el mejor individuo de IS en OOS.
5. Registrar equity OOS.

Al final:
- Equity stitched: concatenación de las equity OOS, cada una empezando
  con el capital final de la anterior.
- Métricas agregadas por fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from fitness.metrics import sharpe
from fitness.objective import FitnessConfig, make_fitness_fn
from features.engine import compute_features, drop_warmup_rows
from gp.evaluator import EvalContext
from gp.evolution import EvolutionConfig, EvolutionResult, evolve
from validation.folds import Fold, WFConfig, generate_folds


@dataclass
class FoldResult:
    """Resultado de un fold individual."""
    fold: Fold
    best_individual: object      # Individual (evitamos import circular)
    best_fitness_is: float
    is_result: object            # BacktestResult
    oos_result: object           # BacktestResult
    oos_sharpe: float
    oos_return: float
    oos_n_trades: int
    oos_max_dd: float


@dataclass
class WalkForwardResult:
    """Resultado completo del WFO."""
    symbol: str
    fold_results: list[FoldResult] = field(default_factory=list)
    equity_stitched: pd.Series | None = None
    metrics_aggregated: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(ohlc: pd.DataFrame) -> tuple[EvalContext, pd.DataFrame]:
    """
    Calcula features y devuelve (EvalContext, df_aligned).

    El EvalContext contiene los arrays de features alineados con df_aligned.
    """
    feats = compute_features(ohlc)
    df_aligned, feats_aligned = drop_warmup_rows(ohlc, feats)
    ctx = EvalContext(
        features={
            name: feats_aligned[name].to_numpy()
            for name in feats_aligned.columns
        },
        n=len(feats_aligned),
    )
    return ctx, df_aligned


def _oos_sharpe(equity: pd.Series) -> float:
    return sharpe(equity)


# ---------------------------------------------------------------------------
# WFO
# ---------------------------------------------------------------------------

def run_walk_forward(
    ohlc: pd.DataFrame,
    wf_config: WFConfig,
    evolution_config: EvolutionConfig,
    fitness_config: FitnessConfig,
    symbol: str,
    initial_capital: float = 10_000.0,
    verbose: bool = False,
) -> WalkForwardResult:
    """
    Ejecuta WFO completo.

    ohlc: DataFrame completo, indexado por DatetimeIndex.
    wf_config: configuración de folds.
    evolution_config: configuración del GP.
    fitness_config: configuración del fitness.
    initial_capital: capital inicial del primer fold.
    verbose: si True, imprime progreso por fold.
    """
    folds = generate_folds(ohlc, wf_config)
    if verbose:
        print(f"Generados {len(folds)} folds")
    if fitness_config.symbol != symbol:
        raise ValueError(
            f"symbol={symbol} no coincide con fitness_config.symbol="
            f"{fitness_config.symbol}"
        )
    backtest_config = BacktestConfig(
        symbol=symbol,
        initial_capital=initial_capital,
        costs=fitness_config.costs,
        size=fitness_config.size,
    )

    fold_results: list[FoldResult] = []
    current_capital = initial_capital

    for fold in folds:
        if verbose:
            print(f"\n{fold}")

        # 1. Features IS y OOS por separado (evita look-ahead entre bloques)
        ctx_is, df_is = _build_context(fold.is_df)
        ctx_oos, df_oos = _build_context(fold.oos_df)

        if len(df_is) < wf_config.min_is_size or len(df_oos) < wf_config.min_oos_size:
            if verbose:
                print(f"  Fold descartado por tamaño tras warmup")
            continue

        # 2. Fitness sobre IS
        fitness_fn = make_fitness_fn(df_is, ctx_is, fitness_config)

        # 3. Evolución en IS
        evo_result: EvolutionResult = evolve(
            fitness_fn=fitness_fn,
            config=evolution_config,
            grammar=__import__("gp.grammar", fromlist=["DEFAULT_GRAMMAR"]).DEFAULT_GRAMMAR,
        )

        if verbose:
            print(f"  Best IS fitness: {evo_result.best_fitness:.4f}, "
                  f"size={evo_result.best_individual.size}")

        # 4. Evaluar el mejor en IS (para tener su equity IS)
        signals_is = pd.Series(
            evo_result.best_individual.evaluate(ctx_is),
            index=df_is.index,
            name="signal",
        )
        is_result = run_backtest(
            df_is, signals_is,backtest_config
        )

        # 5. Evaluar el MISMO individuo en OOS
        signals_oos = pd.Series(
            evo_result.best_individual.evaluate(ctx_oos),
            index=df_oos.index,
            name="signal",
        )
        oos_result = run_backtest(
            df_oos, signals_oos,backtest_config
        )

        oos_sharpe = _oos_sharpe(oos_result.equity)
        oos_return = float(
            (oos_result.equity.iloc[-1] / current_capital) - 1.0
        )
        oos_dd = float(
            (oos_result.equity / oos_result.equity.cummax() - 1.0).min()
        )

        fold_results.append(
            FoldResult(
                fold=fold,
                best_individual=evo_result.best_individual,
                best_fitness_is=evo_result.best_fitness,
                is_result=is_result,
                oos_result=oos_result,
                oos_sharpe=oos_sharpe,
                oos_return=oos_return,
                oos_n_trades=oos_result.n_trades,
                oos_max_dd=oos_dd,
            )
        )

        if verbose:
            print(f"  OOS: return={oos_return:+.4%}, sharpe={oos_sharpe:.3f}, "
                  f"trades={oos_result.n_trades}, maxDD={oos_dd:.2%}")

        # Actualizar capital para el siguiente fold
        current_capital = float(oos_result.equity.iloc[-1])

    # Equity stitched
    if fold_results:
        equity_pieces = []
        for i, fr in enumerate(fold_results):
            eq = fr.oos_result.equity.copy()
            if i > 0:
                # Reescalar para que empiece donde terminó el fold anterior
                prev_final = fold_results[i - 1].oos_result.equity.iloc[-1]
                first = eq.iloc[0]
                if first != 0:
                    eq = eq * (prev_final / first)
            equity_pieces.append(eq)
        equity_stitched = pd.concat(equity_pieces)
    else:
        equity_stitched = None

    # Métricas agregadas
    metrics_aggregated = _aggregate_metrics(fold_results, equity_stitched)

    return WalkForwardResult(
        symbol=symbol,
        fold_results=fold_results,
        equity_stitched=equity_stitched,
        metrics_aggregated=metrics_aggregated,
    )


def _aggregate_metrics(
    fold_results: list[FoldResult],
    equity_stitched: pd.Series | None,
) -> dict:
    """Calcula métricas agregadas del WFO."""
    if not fold_results:
        return {"n_folds": 0}

    oos_returns = [fr.oos_return for fr in fold_results]
    oos_sharpes = [fr.oos_sharpe for fr in fold_results]
    oos_trades = [fr.oos_n_trades for fr in fold_results]
    oos_dds = [fr.oos_max_dd for fr in fold_results]

    out = {
        "n_folds": len(fold_results),
        "pct_positive_folds": float(np.mean([r > 0 for r in oos_returns])),
        "mean_fold_return": float(np.mean(oos_returns)),
        "median_fold_return": float(np.median(oos_returns)),
        "worst_fold_return": float(np.min(oos_returns)),
        "best_fold_return": float(np.max(oos_returns)),
        "std_fold_return": float(np.std(oos_returns)),
        "mean_fold_sharpe": float(np.mean(oos_sharpes)),
        "median_fold_sharpe": float(np.median(oos_sharpes)),
        "worst_fold_sharpe": float(np.min(oos_sharpes)),
        "best_fold_sharpe": float(np.max(oos_sharpes)),
        "total_trades": int(np.sum(oos_trades)),
        "mean_trades_per_fold": float(np.mean(oos_trades)),
        "worst_fold_dd": float(np.min(oos_dds)),
        "mean_fold_dd": float(np.mean(oos_dds)),
    }

    if equity_stitched is not None and len(equity_stitched) > 1:
        out["stitched_total_return"] = float(
            (equity_stitched.iloc[-1] / equity_stitched.iloc[0]) - 1.0
        )
        out["stitched_sharpe"] = sharpe(equity_stitched)
        out["stitched_max_dd"] = float(
            (equity_stitched / equity_stitched.cummax() - 1.0).min()
        )

    return out