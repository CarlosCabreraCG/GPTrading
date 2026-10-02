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
from validation.final_test import FinalTestConfig, run_final_test, split_final_test
from validation.robustness import (
    cross_pair_test,
    evaluate_on_other_data,
    summarize_robustness,
)


@pytest.fixture
def ohlc_1() -> pd.DataFrame:
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = np.linspace(1.10, 1.11, n)
    return pd.DataFrame(
        {"open": close, "high": close + 0.0001,
         "low": close - 0.0001, "close": close},
        index=idx,
    )


@pytest.fixture
def ohlc_2() -> pd.DataFrame:
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = np.linspace(1.20, 1.19, n)
    return pd.DataFrame(
        {"open": close, "high": close + 0.0001,
         "low": close - 0.0001, "close": close},
        index=idx,
    )


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


def test_split_final_test_zero_ratio(ohlc_1):
    config = FinalTestConfig(ratio=0.0, lock_file=None)
    dev, final = split_final_test(ohlc_1, config)
    assert len(dev) == len(ohlc_1)
    assert len(final) == 0


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
            config={"seed": i},
            is_metrics={},
            oos_metrics={},
        )
    dsr = tracker.deflated_sharpe_ratio(observed_sharpe=2.0, n_observations=1000)
    assert 0.0 <= dsr <= 1.0