"""Tests de robustness, final_test y tracker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.tracker import ExperimentTracker
from gp.evaluator import TreeNode
from gp.individual import Individual
from gp.nodes import POSITION_LONG
from data.splitter import SplitConfig, split_development_final
from validation.final_test import FinalTestConfig, run_final_test, split_final_test
from validation.robustness import (
    cross_pair_test,
    evaluate_on_other_data,
    summarize_robustness,
)
from tests.conftest import make_linear_ohlc


@pytest.fixture
def ohlc_1() -> pd.DataFrame:
    return make_linear_ohlc(500, 1.10, 1.11)


@pytest.fixture
def ohlc_2() -> pd.DataFrame:
    return make_linear_ohlc(500, 1.20, 1.19)


@pytest.fixture
def long_individual() -> Individual:
    tree = TreeNode(node=POSITION_LONG, children=[], feature_name=None)
    return Individual(root=tree)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_evaluate_on_other_data(long_individual, ohlc_1):
    r = evaluate_on_other_data(long_individual, ohlc_1, label="test", symbol="EURUSD")
    assert r.n_bars > 0
    assert r.label == "test"
    assert 1 in r.signals_distribution


def test_cross_pair_test(long_individual, ohlc_1, ohlc_2):
    results = cross_pair_test(long_individual, {"A": ohlc_1, "B": ohlc_2}, {"A": "EURUSD", "B": "XAUUSD"})
    assert len(results) == 2
    assert results[0].label == "A"
    assert results[1].label == "B"


def test_summarize_robustness( long_individual, ohlc_1, ohlc_2):
    results = cross_pair_test(long_individual, {"A": ohlc_1, "B": ohlc_2}, {"A": "EURUSD", "B": "XAUUSD"})
    s = summarize_robustness(results)
    assert s["n_tests"] == 2
    assert "mean_sharpe" in s


# ---------------------------------------------------------------------------
# Final test
# ---------------------------------------------------------------------------

def test_split_final_test(ohlc_1):
    config = FinalTestConfig(ratio=0.2, lock_file=None)
    dev, final = split_final_test(ohlc_1, config)
    assert len(dev) + len(final) == len(ohlc_1)
    assert len(final) == 100
    assert dev.index[-1] < final.index[0]

def test_run_final_test(long_individual, ohlc_1, tmp_path):
    config = FinalTestConfig(ratio=0.2, lock_file=tmp_path / "lock.txt")
    dev, final = split_final_test(ohlc_1, config)
    result = run_final_test(long_individual, final, config, symbol="EURUSD")
    assert result.n_bars > 0
    assert config.lock_file.exists()


def test_run_final_test_twice_raises(long_individual, ohlc_1, tmp_path):
    config = FinalTestConfig(ratio=0.2, lock_file=tmp_path / "lock.txt")
    dev, final = split_final_test(ohlc_1, config)
    run_final_test(long_individual, final, config, symbol="EURUSD",)
    with pytest.raises(RuntimeError, match="ya fue ejecutado"):
        run_final_test(long_individual, final, config, symbol="EURUSD",)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

def test_tracker_register_and_load(tmp_path):
    path = tmp_path / "tracker.jsonl"
    tracker = ExperimentTracker(path, symbol="EURUSD")
    assert tracker.n_experiments == 0

    tracker.register(
        symbol="EURUSD",
        n_trials=1000,
        config={"a": 1, "b": 2},
        is_metrics={"sharpe": 1.0},
        oos_metrics={"sharpe": 0.5},
    )
    assert tracker.n_experiments == 1
    assert path.exists()

    # Recargar
    tracker2 = ExperimentTracker(path, symbol="EURUSD")
    assert tracker2.n_experiments == 1


def test_tracker_deflated_sharpe(tmp_path):
    path = tmp_path / "tracker.jsonl"
    tracker = ExperimentTracker(path, symbol="EURUSD")
    for i in range(100):
        tracker.register(
            symbol="EURUSD",
            n_trials=1000,
            config={"seed": i},
            is_metrics={},
            oos_metrics={},
        )
    dsr = tracker.deflated_sharpe_ratio(observed_sharpe=2.0, n_observations=1000)
    assert 0.0 <= dsr <= 1.0

def test_split_config_rejects_zero_ratio():
    with pytest.raises(ValueError, match="final_test_ratio"):
        SplitConfig(final_test_ratio=0.0)


def test_split_config_rejects_negative_ratio():
    with pytest.raises(ValueError, match="final_test_ratio"):
        SplitConfig(final_test_ratio=-0.1)


def test_split_config_rejects_too_large_ratio():
    with pytest.raises(ValueError, match="final_test_ratio"):
        SplitConfig(final_test_ratio=0.6)


def _flat_ohlc(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": np.ones(n), "high": np.ones(n),
         "low": np.ones(n), "close": np.ones(n)},
        index=idx,
    )


def test_split_rejects_small_final_test():
    # 4990 filas con ratio 0.10 -> 499 velas, < 500 (mínimo). Debe fallar.
    config = SplitConfig(final_test_ratio=0.10, save_final_test=False)
    with pytest.raises(ValueError, match="Final test demasiado pequeño.*mínimo 500"):
        split_development_final(_flat_ohlc(4990), config)


def test_split_accepts_minimum_final_test():
    # 5000 filas con ratio 0.10 -> 500 velas, justo el mínimo. Debe aceptarse.
    config = SplitConfig(final_test_ratio=0.10, save_final_test=False)
    development, final_test = split_development_final(_flat_ohlc(5000), config)
    assert len(final_test) == 500
    assert len(development) == 4500


def test_split_by_fixed_start_date_is_stable():
    """La frontera por fecha no cambia aunque el dataset crezca por el final."""
    df = _flat_ohlc(6000)
    start = df.index[5000]
    config = SplitConfig(final_test_start=start, save_final_test=False)
    dev, final = split_development_final(df, config)
    assert len(dev) == 5000 and final.index[0] == start
    # Con más datos al final, el development no cambia
    dev2, _ = split_development_final(_flat_ohlc(7000), config)
    assert len(dev2) == 5000 and dev2.index[-1] == dev.index[-1]


def test_split_fixed_start_rejects_small_final_test():
    df = _flat_ohlc(6000)
    config = SplitConfig(final_test_start=df.index[5600], save_final_test=False)
    with pytest.raises(ValueError, match="Final test demasiado pequeño"):
        split_development_final(df, config)


def test_tracker_n_trials(tmp_path):
    from experiments.tracker import ExperimentTracker
    tracker = ExperimentTracker(tmp_path / "t.jsonl", symbol="EURUSD")
    tracker.register(
        symbol="EURUSD", n_trials=1000,
        config={"a": 1}, is_metrics={}, oos_metrics={},
    )
    assert tracker.records[0].n_trials == 1000


def test_sharpe_ci_returns_finite_for_large_sample():
    from validation.walk_forward import _sharpe_ci
    import pandas as pd, numpy as np
    idx = pd.date_range("2024-01-01", periods=1000, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    eq = pd.Series(10000 * np.exp(np.cumsum(rng.normal(0.0001, 0.001, 1000))), index=idx)
    sr, se, lo, hi = _sharpe_ci(eq, "EURUSD")
    assert np.isfinite(se) and np.isfinite(lo) and np.isfinite(hi)
    assert lo < sr < hi
