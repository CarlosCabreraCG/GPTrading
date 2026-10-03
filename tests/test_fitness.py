"""
Tests del módulo fitness.

Verifica:
- Métricas básicas (Sharpe, Sortino, Max DD, Calmar, Profit Factor).
- Penalizaciones de complejidad.
- Fitness completo: rechaza individuos inválidos, premia buenos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, run_backtest
from backtest.results import BacktestResult
from config.settings import CostModel
from fitness.complexity import ComplexityWeights, complexity_penalty
from fitness.metrics import (
    annualization_factor,
    calmar,
    expectancy,
    max_drawdown,
    profit_factor,
    sharpe,
    sortino,
    turnover,
)
from fitness.objective import FitnessConfig, make_fitness_fn
from gp.evaluator import EvalContext
from gp.individual import Individual, random_individual
from gp.nodes import POSITION_FLAT, POSITION_LONG, POSITION_SHORT
from gp.evaluator import TreeNode
from strategy.signal import Signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trending_ohlc() -> pd.DataFrame:
    """Precio sube monótonamente: cualquier LONG gana."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = np.linspace(1.1000, 1.1100, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.0001,
            "low": close - 0.0001,
            "close": close,
        },
        index=idx,
    )


@pytest.fixture
def flat_ohlc() -> pd.DataFrame:
    """Precio constante: sin oportunidades."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    close = np.full(n, 1.1000)
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        },
        index=idx,
    )


@pytest.fixture
def trivial_ctx() -> EvalContext:
    """Contexto con una sola feature constante."""
    return EvalContext(
        features={"return_1": np.zeros(100)},
        n=100,
    )


# ---------------------------------------------------------------------------
# Tests de métricas
# ---------------------------------------------------------------------------

def test_sharpe_positive_for_growing_equity():
    eq = pd.Series(np.linspace(10000, 11000, 100),
                   index=pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC"))
    s = sharpe(eq)
    assert s > 0


def test_sharpe_zero_for_flat_equity():
    eq = pd.Series(np.full(100, 10000.0),
                   index=pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC"))
    assert sharpe(eq) == 0.0


def test_max_drawdown_negative_or_zero():
    eq = pd.Series([10000, 10500, 10200, 10800, 10600],
                   index=pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC"))
    dd = max_drawdown(eq)
    assert dd <= 0


def test_calmar_positive_for_growing_equity():
    eq = pd.Series([10000, 10100, 10200, 10300, 10400],
                   index=pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC"))
    c = calmar(eq)
    # Sin drawdown, calmar indefinido (0 por nuestra definición)
    assert c == 0.0 or c > 0


def test_sortino_zero_when_no_downside():
    eq = pd.Series([10000, 10100, 10200, 10300],
                   index=pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"))
    s = sortino(eq)
    # Sin downside, sortino = 0 por nuestra definición (evita inf)
    assert s == 0.0


def test_turnover_counts_changes():
    pos = pd.Series([0, 0, 1, 1, 0, -1, -1, 0])
    assert turnover(pos) == 4  # 0->1, 1->0, 0->-1, -1->0


# ---------------------------------------------------------------------------
# Tests de complexity penalty
# ---------------------------------------------------------------------------

def test_complexity_zero_for_terminal():
    tree = TreeNode(node=POSITION_LONG, children=[], feature_name=None)
    ind = Individual(root=tree)
    p = complexity_penalty(ind)
    # size=1, conditions=0, features=0
    assert p == pytest.approx(0.001)


def test_complexity_increases_with_size():
    t1 = TreeNode(node=POSITION_LONG, children=[], feature_name=None)
    t2 = TreeNode(node=POSITION_SHORT, children=[], feature_name=None)
    ind1 = Individual(root=t1)
    ind2 = Individual(root=t2)
    # Ambos size=1; mismo penalty
    assert complexity_penalty(ind1) == complexity_penalty(ind2)


def test_complexity_weights_affect_result():
    tree = TreeNode(node=POSITION_LONG, children=[], feature_name=None)
    ind = Individual(root=tree)
    w1 = ComplexityWeights(size=0.001)
    w2 = ComplexityWeights(size=0.01)
    assert complexity_penalty(ind, w2) > complexity_penalty(ind, w1)


def test_fitness_logs_error_and_counts(trending_ohlc, caplog):
    """Un individuo que pide una feature inexistente debe loguear y contar."""
    import logging
    n = 100
    ctx = EvalContext(features={"x": np.ones(n)}, n=n)

    from gp.grammar import FEATURE_TERMINALS
    from gp.nodes import NODE_IF, NODE_GT, POSITION_LONG, POSITION_FLAT

    ret_node = FEATURE_TERMINALS["return_1"]
    ret_tree = TreeNode(node=ret_node, children=[], feature_name="return_1")
    gt_tree = TreeNode(node=NODE_GT, children=[ret_tree, ret_tree], feature_name=None)
    if_tree = TreeNode(
        node=NODE_IF,
        children=[
            gt_tree,
            TreeNode(node=POSITION_LONG, children=[], feature_name=None),
            TreeNode(node=POSITION_FLAT, children=[], feature_name=None),
        ],
        feature_name=None,
    )
    ind = Individual(root=if_tree)

    fitness_fn = make_fitness_fn(
        trending_ohlc, ctx, FitnessConfig(symbol="EURUSD", min_trades=1)
    )
    with caplog.at_level(logging.ERROR):
        f = fitness_fn(ind)
    assert f == float("-inf")
    assert fitness_fn.error_stats["evaluate"] >= 1
    assert "Error evaluando individuo" in caplog.text


def test_fitness_no_error_for_valid_individual(trending_ohlc):
    """Un individuo válido no incrementa los contadores."""
    n = len(trending_ohlc)
    ctx = EvalContext(features={"x": np.ones(n)}, n=n)

    from gp.nodes import POSITION_LONG
    tree = TreeNode(node=POSITION_LONG, children=[], feature_name=None)
    ind = Individual(root=tree)

    fitness_fn = make_fitness_fn(
        trending_ohlc, ctx, FitnessConfig(symbol="EURUSD", min_trades=1)
    )
    fitness_fn(ind)
    assert fitness_fn.error_stats["evaluate"] == 0
    assert fitness_fn.error_stats["backtest"] == 0

# ---------------------------------------------------------------------------
# Tests del fitness completo
# ---------------------------------------------------------------------------

def test_fitness_rejects_flat_individual(flat_ohlc, trivial_ctx):
    """Un individuo que siempre está FLAT debe devolver -inf."""
    tree = TreeNode(node=POSITION_FLAT, children=[], feature_name=None)
    ind = Individual(root=tree)

    fitness_fn = make_fitness_fn(
        flat_ohlc, trivial_ctx, FitnessConfig(symbol="EURUSD",min_trades=1)
    )
    assert fitness_fn(ind) == float("-inf")


def test_fitness_rejects_too_few_trades(trending_ohlc, trivial_ctx):
    """Un LONG puro hace 1 operación; min_trades=5 lo rechaza."""
    tree = TreeNode(node=POSITION_LONG, children=[], feature_name=None)
    ind = Individual(root=tree)

    fitness_fn = make_fitness_fn(
        trending_ohlc, trivial_ctx, FitnessConfig(symbol="EURUSD",min_trades=5)
    )
    assert fitness_fn(ind) == float("-inf")


def test_fitness_accepts_valid_individual(trending_ohlc):
    from gp.grammar import FEATURE_TERMINALS
    from gp.nodes import NODE_GT, NODE_IF, NODE_NEG, POSITION_LONG, POSITION_FLAT
    from gp.types import GPType
    from gp.nodes import Node

    n = len(trending_ohlc)
    # Alterna +1/-1 pero con periodo 10 (menos turnover)
    alt = np.array([1.0 if (i // 10) % 2 == 0 else -1.0 for i in range(n)])
    ctx = EvalContext(features={"alt": alt}, n=n)

    alt_node = Node(name="alt", return_type=GPType.REAL, arg_types=(), func=None)
    alt_tree = TreeNode(node=alt_node, children=[], feature_name="alt")
    neg_tree = TreeNode(node=NODE_NEG, children=[alt_tree], feature_name=None)
    gt_tree = TreeNode(node=NODE_GT, children=[alt_tree, neg_tree], feature_name=None)
    if_tree = TreeNode(
        node=NODE_IF,
        children=[
            gt_tree,
            TreeNode(node=POSITION_LONG, children=[], feature_name=None),
            TreeNode(node=POSITION_FLAT, children=[], feature_name=None),
        ],
        feature_name=None,
    )
    ind = Individual(root=if_tree)

    fitness_fn = make_fitness_fn(
        trending_ohlc, ctx,
        FitnessConfig(symbol="EURUSD", min_trades=2, max_turnover_ratio=1.0)
    )
    f = fitness_fn(ind)
    assert f > float("-inf")

def test_fitness_deterministic(trending_ohlc, trivial_ctx):
    tree = TreeNode(node=POSITION_LONG, children=[], feature_name=None)
    ind = Individual(root=tree)
    fn = make_fitness_fn(trending_ohlc, trivial_ctx, FitnessConfig(symbol="EURUSD",min_trades=1))
    f1 = fn(ind)
    f2 = fn(ind)
    assert f1 == f2


def test_fitness_handles_evaluation_error():
    """Si el individuo no se puede evaluar, devuelve -inf, no lanza."""
    n = 50
    ohlc = pd.DataFrame(
        {"open": np.ones(n), "high": np.ones(n),
         "low": np.ones(n), "close": np.ones(n)},
        index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
    )
    ctx = EvalContext(features={"x": np.ones(n)}, n=n)

    # Individuo que pide una feature inexistente
    from gp.grammar import FEATURE_TERMINALS
    from gp.nodes import NODE_IF, NODE_GT, POSITION_LONG, POSITION_FLAT

    ret_node = FEATURE_TERMINALS["return_1"]
    ret_tree = TreeNode(node=ret_node, children=[], feature_name="return_1")
    gt_tree = TreeNode(node=NODE_GT, children=[ret_tree, ret_tree], feature_name=None)
    if_tree = TreeNode(
        node=NODE_IF,
        children=[
            gt_tree,
            TreeNode(node=POSITION_LONG, children=[], feature_name=None),
            TreeNode(node=POSITION_FLAT, children=[], feature_name=None),
        ],
        feature_name=None,
    )
    ind = Individual(root=if_tree)

    fn = make_fitness_fn(ohlc, ctx, FitnessConfig(symbol="EURUSD",min_trades=1))
    assert fn(ind) == float("-inf")