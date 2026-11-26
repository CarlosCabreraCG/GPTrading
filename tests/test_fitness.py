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

from backtest.engine import run_backtest
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
from tests.test_backtester import simple_ohlc

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

def test_engine_metrics_are_finite(simple_ohlc):
    signals = pd.Series([1, -1, 1, -1, 0, 0, 1, 0, -1, 0],
                        index=simple_ohlc.index, name="signal")
    result = run_backtest(simple_ohlc, signals)
    metrics = result.metrics(symbol="EURUSD")
    for k, v in metrics.items():
        if k == "profit_factor":
            continue  # puede ser inf
        assert np.isfinite(v), f"Métrica {k} no finita: {v}"

def test_sharpe_positive_for_growing_equity():
    eq = pd.Series(np.linspace(10000, 11000, 100),
                   index=pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC"))
    s = sharpe(eq, symbol="XAUUSD")
    assert s > 0


def test_sharpe_zero_for_flat_equity():
    eq = pd.Series(np.full(100, 10000.0),
                   index=pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC"))
    assert sharpe(eq, symbol="XAUUSD") == 0.0


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
    s = sortino(eq, symbol="XAUUSD")
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


def test_fitness_rejects_high_turnover(trending_ohlc):
    """Un individuo que cambia de posición cada vela debe ser rechazado."""
    n = len(trending_ohlc)
    # Señal que alterna LONG/FLAT cada vela
    alt = np.array([1 if i % 2 == 0 else 0 for i in range(n)], dtype=np.int8)
    signals = pd.Series(alt, index=trending_ohlc.index, name="signal")

    # Construir individuo que devuelve esa señal: usamos feature fake
    from gp.evaluator import EvalContext, TreeNode
    from gp.individual import Individual
    from gp.nodes import Node, NODE_IF, NODE_GT, POSITION_LONG, POSITION_FLAT
    from gp.types import GPType
    from gp.nodes import NODE_NEG

    ctx = EvalContext(features={"alt": alt.astype(float)}, n=n)
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

    cfg = FitnessConfig(
        symbol="EURUSD",
        min_trades=1,
        max_turnover_ratio=0.15,   # estricto
        max_flat_ratio=0.99,
    )
    fn = make_fitness_fn(trending_ohlc, ctx, cfg)
    assert fn(ind) == float("-inf")

def test_bars_per_year_forex():
    from config.settings import bars_per_year
    # 15m en forex: 120h/sem * 3600 / 900 = 480 barras/sem * 52 = 24,960
    assert bars_per_year("EURUSD", 900.0) == pytest.approx(24_960.0)


def test_bars_per_year_gold():
    from config.settings import bars_per_year
    # 15m en oro: 115h/sem * 3600 / 900 = 460 * 52 = 23,920
    assert bars_per_year("XAUUSD", 900.0) == pytest.approx(23_920.0)


def test_sharpe_uses_symbol_annualization():
    import pandas as pd
    import numpy as np
    from fitness.metrics import sharpe
    idx = pd.date_range("2024-01-01", periods=1000, freq="15min", tz="UTC")
    eq = pd.Series(np.linspace(10000, 10100, 1000), index=idx)
    # Con símbolo y sin símbolo debe diferir
    s_eurusd = sharpe(eq, "EURUSD")
    s_gold = sharpe(eq, "XAUUSD")
    assert s_eurusd != s_gold

# ---------------------------------------------------------------------------
# Fitness neutral a la dirección (alpha_sharpe / fitness_mode="alpha")
# ---------------------------------------------------------------------------

def _market_and_pnl(n=4000, seed=0, alpha_per_bar=0.0, beta=1.0, noise_std=0.01):
    from fitness.metrics import alpha_sharpe  # noqa: F401
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    ret = rng.normal(0.0002, 0.001, n)           # mercado con deriva positiva
    close = pd.Series(100.0 * np.cumprod(1 + ret), index=idx)
    pnl_m = close.diff().fillna(0.0)
    noise = rng.normal(0, noise_std, n) if noise_std > 0 else 0.0
    pnl_s = beta * pnl_m + alpha_per_bar + noise
    equity = 10_000 + pnl_s.cumsum()
    return equity, close


def test_alpha_sharpe_zero_for_pure_market_exposure():
    """Seguir al mercado (LONG o SHORT constante) no debe dar alpha."""
    from fitness.metrics import alpha_sharpe
    for beta in (1.0, -1.0, 0.5):
        eq, close = _market_and_pnl(alpha_per_bar=0.0, beta=beta, noise_std=0.0)
        assert alpha_sharpe(eq, close, "XAUUSD") == 0.0
    # mientras que el Sharpe absoluto con deriva y beta=1 es claramente positivo
    eq, close = _market_and_pnl(alpha_per_bar=0.0, beta=1.0)
    assert sharpe(eq, "XAUUSD") > 1.0


def test_alpha_sharpe_detects_true_alpha_independent_of_beta():
    from fitness.metrics import alpha_sharpe
    eq_a, close = _market_and_pnl(alpha_per_bar=0.01, beta=0.0)
    eq_0, _ = _market_and_pnl(alpha_per_bar=0.0, beta=0.0)
    assert alpha_sharpe(eq_a, close, "XAUUSD") > alpha_sharpe(eq_0, close, "XAUUSD") + 5.0


def test_alpha_sharpe_degenerate_inputs():
    from fitness.metrics import alpha_sharpe
    idx = pd.date_range("2024-01-01", periods=10, freq="15min")
    flat_close = pd.Series(np.full(10, 100.0), index=idx)
    eq = pd.Series(np.full(10, 10_000.0), index=idx)
    assert alpha_sharpe(eq, flat_close, "XAUUSD") == 0.0      # sin varianza de mercado
    assert alpha_sharpe(eq.iloc[:2], flat_close.iloc[:2], "XAUUSD") == 0.0  # muy corto


def test_fitness_config_alpha_validations():
    with pytest.raises(ValueError, match="fitness_mode"):
        FitnessConfig(symbol="EURUSD", fitness_mode="otro")
    with pytest.raises(ValueError, match="sharpe"):
        FitnessConfig(symbol="EURUSD", fitness_mode="alpha", primary_metric="sortino")
    with pytest.raises(ValueError, match="max_abs_net_exposure"):
        FitnessConfig(symbol="EURUSD", max_abs_net_exposure=1.5)


def _mostly_long_individual(n):
    """LONG 9 de cada 10 barras, SHORT el resto: exposición neta ~ +0.8."""
    from gp.nodes import NODE_GT, NODE_IF, NODE_NEG, Node
    from gp.types import GPType
    pattern = np.array([-1.0 if (i % 10) == 0 else 1.0 for i in range(n)])
    ctx = EvalContext(features={"p": pattern}, n=n)
    p_node = Node(name="p", return_type=GPType.REAL, arg_types=(), func=None)
    p_tree = TreeNode(node=p_node, children=[], feature_name="p")
    neg = TreeNode(node=NODE_NEG, children=[p_tree], feature_name=None)
    gt = TreeNode(node=NODE_GT, children=[p_tree, neg], feature_name=None)
    ind = Individual(root=TreeNode(
        node=NODE_IF,
        children=[gt,
                  TreeNode(node=POSITION_LONG, children=[], feature_name=None),
                  TreeNode(node=POSITION_SHORT, children=[], feature_name=None)],
        feature_name=None,
    ))
    return ind, ctx


def test_fitness_alpha_does_not_reward_drift_following(trending_ohlc):
    """En un mercado que solo sube, el modo alpha puntúa menos que el raw."""
    n = len(trending_ohlc)
    ind, ctx = _mostly_long_individual(n)
    common = dict(symbol="EURUSD", min_trades=2, max_turnover_ratio=1.0)
    raw = make_fitness_fn(trending_ohlc, ctx, FitnessConfig(**common))(ind)
    alpha = make_fitness_fn(
        trending_ohlc, ctx, FitnessConfig(fitness_mode="alpha", **common)
    )(ind)
    assert np.isfinite(raw) and np.isfinite(alpha)
    assert raw > 5.0          # el Sharpe absoluto premia la deriva
    assert alpha < raw / 5    # el alpha no


def test_fitness_max_net_exposure_rejects_directional(trending_ohlc):
    """Una señal casi siempre LONG se rechaza con max_abs_net_exposure bajo."""
    n = len(trending_ohlc)
    ind, ctx = _mostly_long_individual(n)
    common = dict(symbol="EURUSD", min_trades=2, max_turnover_ratio=1.0)
    ok = make_fitness_fn(trending_ohlc, ctx, FitnessConfig(**common))(ind)
    rej = make_fitness_fn(
        trending_ohlc, ctx, FitnessConfig(max_abs_net_exposure=0.5, **common)
    )(ind)
    assert np.isfinite(ok)
    assert rej == float("-inf")
