"""
Tests de WFO.

Adaptados a la nueva WFConfig basada en tamaños absolutos en velas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fitness.objective import FitnessConfig
from gp.evolution import EvolutionConfig
from validation.folds import Fold, WFConfig, generate_folds
from validation.walk_forward import (
    WalkForwardResult,
    _aggregate_metrics,
    run_walk_forward,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_ohlc() -> pd.DataFrame:
    """OHLC sintético de 2000 velas de 15m, sin gaps."""
    n = 2000
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(42)
    returns = rng.normal(0.00001, 0.001, n)
    close = 1.10 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0003, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0003, n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests de folds
# ---------------------------------------------------------------------------

def test_generate_folds_basic(synthetic_ohlc):
    config = WFConfig(
        is_size=500, oos_size=200,
        min_is_size=400, min_oos_size=150,
    )
    folds = generate_folds(synthetic_ohlc, config)
    assert len(folds) > 0
    for f in folds:
        assert f.n_is == 500
        assert f.n_oos == 200


def test_folds_are_temporally_ordered(synthetic_ohlc):
    config = WFConfig(is_size=500, oos_size=200,
                     min_is_size=400, min_oos_size=150)
    folds = generate_folds(synthetic_ohlc, config)
    for f in folds:
        assert f.is_end < f.oos_start
        assert (f.oos_df.index.to_series().diff().dropna() > pd.Timedelta(0)).all()


def test_folds_no_oos_overlap(synthetic_ohlc):
    """Con step=oos_size, los OOS no deben solaparse."""
    config = WFConfig(
        is_size=500, oos_size=200, step_size=200,
        min_is_size=400, min_oos_size=150,
    )
    folds = generate_folds(synthetic_ohlc, config)
    for i in range(len(folds) - 1):
        assert folds[i].oos_end < folds[i + 1].oos_start


def test_folds_step_equals_oos_by_default(synthetic_ohlc):
    config = WFConfig(is_size=500, oos_size=200,
                     min_is_size=400, min_oos_size=150)
    folds = generate_folds(synthetic_ohlc, config)
    # Sin step explícito, el step debe ser oos_size
    for i in range(len(folds) - 1):
        assert folds[i + 1].oos_start > folds[i].oos_end


def test_folds_rejects_zero_sizes():
    with pytest.raises(ValueError, match="is_size y oos_size"):
        WFConfig(is_size=0, oos_size=200)
    with pytest.raises(ValueError, match="is_size y oos_size"):
        WFConfig(is_size=500, oos_size=0)


def test_folds_rejects_too_small_dataset():
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {"open": np.ones(n), "high": np.ones(n),
         "low": np.ones(n), "close": np.ones(n)},
        index=idx,
    )
    config = WFConfig(is_size=500, oos_size=200)
    with pytest.raises(ValueError, match="demasiado pequeño"):
        generate_folds(df, config)


def test_folds_deterministic(synthetic_ohlc):
    config = WFConfig(is_size=500, oos_size=200,
                     min_is_size=400, min_oos_size=150)
    f1 = generate_folds(synthetic_ohlc, config)
    f2 = generate_folds(synthetic_ohlc, config)
    assert len(f1) == len(f2)
    for a, b in zip(f1, f2):
        assert a.is_start == b.is_start
        assert a.oos_end == b.oos_end


def test_folds_respects_max_timestamp(synthetic_ohlc):
    """La salvaguarda contra el final test debe truncar el dataset."""
    max_ts = synthetic_ohlc.index[1500]
    config = WFConfig(
        is_size=500, oos_size=200,
        min_is_size=400, min_oos_size=150,
        max_timestamp=max_ts,
    )
    folds = generate_folds(synthetic_ohlc, config)
    for f in folds:
        assert f.oos_end <= max_ts


def test_folds_max_timestamp_no_folds_raises(synthetic_ohlc):
    max_ts = synthetic_ohlc.index[100]  # demasiado pronto
    config = WFConfig(
        is_size=500, oos_size=200,
        max_timestamp=max_ts,
    )
    with pytest.raises(ValueError, match="demasiado pequeño|Ninguna fila"):
        generate_folds(synthetic_ohlc, config)


def test_folds_count_expected(synthetic_ohlc):
    """
    Con n=2000, is=500, oos=200, step=200:
    starts válidos: 0, 200, 400, 600, 800, 1000, 1200.
    Total: 7 folds.
    """
    config = WFConfig(is_size=500, oos_size=200,
                     min_is_size=400, min_oos_size=150)
    folds = generate_folds(synthetic_ohlc, config)
    assert len(folds) == 7


# ---------------------------------------------------------------------------
# Tests de WFO completo
# ---------------------------------------------------------------------------

def test_wfo_runs_on_synthetic(synthetic_ohlc):
    wf_config = WFConfig(
        is_size=500, oos_size=200,
        min_is_size=400, min_oos_size=150,
    )
    evo_config = EvolutionConfig(
        population_size=20,
        n_generations=2,
        tournament_size=3,
        n_elites=1,
        crossover_rate=0.8,
        mutation_rate=0.2,
        seed=42,
    )
    fitness_config = FitnessConfig(symbol="EURUSD",min_trades=2, max_turnover_ratio=1.0)

    result = run_walk_forward(
        synthetic_ohlc, wf_config, evo_config, fitness_config,
        symbol="EURUSD",
        initial_capital=10_000.0,
    )

    assert isinstance(result, WalkForwardResult)
    assert len(result.fold_results) > 0
    assert result.equity_stitched is not None
    assert len(result.equity_stitched) > 0
    assert result.metrics_aggregated["n_folds"] == len(result.fold_results)


def test_wfo_equity_stitched_is_continuous(synthetic_ohlc):
    wf_config = WFConfig(
        is_size=500, oos_size=200,
        min_is_size=400, min_oos_size=150,
    )
    evo_config = EvolutionConfig(population_size=10, n_generations=2, seed=1)
    fitness_config = FitnessConfig(symbol="EURUSD",min_trades=2, max_turnover_ratio=1.0)

    result = run_walk_forward(
        synthetic_ohlc, wf_config, evo_config, fitness_config,symbol="EURUSD"
    )

    eq = result.equity_stitched
    returns = eq.pct_change().dropna()
    assert (returns.abs() < 0.05).all(), "Equity stitched con saltos bruscos"


def test_wfo_aggregated_metrics_keys(synthetic_ohlc):
    wf_config = WFConfig(
        is_size=500, oos_size=200,
        min_is_size=400, min_oos_size=150,
    )
    evo_config = EvolutionConfig(population_size=10, n_generations=2, seed=1)
    fitness_config = FitnessConfig(symbol="EURUSD",min_trades=2, max_turnover_ratio=1.0)

    result = run_walk_forward(
        synthetic_ohlc, wf_config, evo_config, fitness_config,symbol="EURUSD"
    )
    m = result.metrics_aggregated
    for key in [
        "n_folds", "pct_positive_folds", "mean_fold_return",
        "median_fold_return", "worst_fold_return", "best_fold_return",
        "mean_fold_sharpe", "median_fold_sharpe", "total_trades",
        "stitched_total_return", "stitched_sharpe", "stitched_max_dd",
    ]:
        assert key in m, f"Falta métrica agregada: {key}"


def test_wfo_deterministic(synthetic_ohlc):
    wf_config = WFConfig(
        is_size=500, oos_size=200,
        min_is_size=400, min_oos_size=150,
    )
    evo_config = EvolutionConfig(population_size=10, n_generations=2, seed=99)
    fitness_config = FitnessConfig(symbol="EURUSD",min_trades=2, max_turnover_ratio=1.0)

    r1 = run_walk_forward(synthetic_ohlc, wf_config, evo_config, fitness_config,symbol="EURUSD")
    r2 = run_walk_forward(synthetic_ohlc, wf_config, evo_config, fitness_config,symbol="EURUSD")

    assert r1.metrics_aggregated["n_folds"] == r2.metrics_aggregated["n_folds"]
    assert r1.metrics_aggregated["mean_fold_return"] == pytest.approx(
        r2.metrics_aggregated["mean_fold_return"]
    )


# ---------------------------------------------------------------------------
# Tests de _aggregate_metrics
# ---------------------------------------------------------------------------

def test_aggregate_metrics_empty():
    m = _aggregate_metrics([], None)
    assert m["n_folds"] == 0


def test_aggregate_metrics_pct_positive():
    class FakeFold:
        oos_return = 0.01
        oos_sharpe = 1.0
        oos_n_trades = 5
        oos_max_dd = -0.02

    frs = [FakeFold(), FakeFold(), FakeFold()]
    m = _aggregate_metrics(frs, None)
    assert m["n_folds"] == 3
    assert m["pct_positive_folds"] == 1.0
    assert m["mean_fold_return"] == pytest.approx(0.01)

def test_wfo_uses_different_seed_per_fold(synthetic_ohlc, monkeypatch):
    """
    Verifica que cada fold usa una semilla distinta.
    Se intercepta evolve para capturar las semillas usadas.
    """
    from validation import walk_forward as wf_module
    captured_seeds = []

    original_evolve = wf_module.evolve
    def spy_evolve(fitness_fn, config, grammar=None):
        captured_seeds.append(config.seed)
        return original_evolve(fitness_fn, config, grammar)

    monkeypatch.setattr(wf_module, "evolve", spy_evolve)

    wf_config = WFConfig(
        is_size=500, oos_size=200,
        min_is_size=400, min_oos_size=150,
    )
    evo_config = EvolutionConfig(population_size=5, n_generations=1, seed=1000)
    fitness_config = FitnessConfig(symbol="EURUSD", min_trades=2, max_turnover_ratio=1.0)

    run_walk_forward(
        synthetic_ohlc, wf_config, evo_config, fitness_config,
        symbol="EURUSD",
    )

    # Debe haber una semilla por fold, todas distintas
    assert len(captured_seeds) == len(set(captured_seeds))
    # La primera debe ser 1000 + 0 = 1000
    assert captured_seeds[0] == 1000
    # La segunda debe ser 1000 + 1 = 1001
    assert captured_seeds[1] == 1001