"""
Tests del backtester.

Fase 2a: tests de signal, position, costs.
Fase 2b: tests de execution, engine, results (se añadirán después).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.costs import cost_of_transition
from backtest.position import Position
from config.settings import CostModel
from strategy.signal import Signal, signals_from_array, validate_signal_series


# ---------------------------------------------------------------------------
# Tests de Signal
# ---------------------------------------------------------------------------

def test_signal_values():
    assert Signal.FLAT == 0
    assert Signal.LONG == 1
    assert Signal.SHORT == -1


def test_validate_signal_series_ok():
    s = pd.Series([0, 1, -1, 0, 1], name="signal")
    validate_signal_series(s)  # no debe lanzar


def test_validate_signal_series_rejects_invalid():
    s = pd.Series([0, 1, 2, 0], name="signal")
    with pytest.raises(ValueError, match="inválidas"):
        validate_signal_series(s)


def test_validate_signal_series_rejects_nan():
    s = pd.Series([0, 1, np.nan, 0], name="signal")
    with pytest.raises(ValueError, match="NaN"):
        validate_signal_series(s)


def test_signals_from_array():
    idx = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    arr = np.array([0, 1, 1, -1, 0])
    s = signals_from_array(arr, idx)
    assert (s.index == idx).all()
    assert s.tolist() == [0, 1, 1, -1, 0]


# ---------------------------------------------------------------------------
# Tests de Position
# ---------------------------------------------------------------------------

def test_position_starts_flat():
    p = Position()
    assert p.is_flat
    assert p.entry_price is None
    assert p.unrealized_pnl(1.1000) == 0.0


def test_position_open_long():
    p = Position()
    p.open(Signal.LONG, price=1.1000, time=pd.Timestamp("2024-01-01", tz="UTC"))
    assert p.is_long
    assert p.entry_price == 1.1000
    # PnL con precio subiendo
    assert p.unrealized_pnl(1.1010) == pytest.approx(0.0010)
    # PnL con precio bajando
    assert p.unrealized_pnl(1.0990) == pytest.approx(-0.0010)


def test_position_open_short():
    p = Position()
    p.open(Signal.SHORT, price=1.1000, time=pd.Timestamp("2024-01-01", tz="UTC"))
    assert p.is_short
    # PnL con precio bajando (short gana)
    assert p.unrealized_pnl(1.0990) == pytest.approx(0.0010)
    # PnL con precio subiendo (short pierde)
    assert p.unrealized_pnl(1.1010) == pytest.approx(-0.0010)


def test_position_close():
    p = Position()
    p.open(Signal.LONG, price=1.1000, time=pd.Timestamp("2024-01-01", tz="UTC"))
    p.close()
    assert p.is_flat
    assert p.entry_price is None
    assert p.unrealized_pnl(1.2000) == 0.0


def test_position_cannot_open_when_not_flat():
    p = Position()
    p.open(Signal.LONG, price=1.1000, time=pd.Timestamp("2024-01-01", tz="UTC"))
    with pytest.raises(RuntimeError):
        p.open(Signal.SHORT, price=1.1000, time=pd.Timestamp("2024-01-01", tz="UTC"))


def test_position_cannot_open_flat():
    p = Position()
    with pytest.raises(ValueError):
        p.open(Signal.FLAT, price=1.1000, time=pd.Timestamp("2024-01-01", tz="UTC"))


# ---------------------------------------------------------------------------
# Tests de Costs
# ---------------------------------------------------------------------------

@pytest.fixture
def costs() -> CostModel:
    return CostModel(
        spread=0.00010,       # 1 pip
        slippage=0.00003,     # 0.3 pip
        commission_per_trade=0.0,
        swap_per_night=0.0,
    )


def test_cost_flat_to_flat_is_zero(costs):
    c = cost_of_transition(Signal.FLAT, Signal.FLAT, costs)
    assert c.total == 0.0


def test_cost_no_change_is_zero(costs):
    assert cost_of_transition(Signal.LONG, Signal.LONG, costs).total == 0.0
    assert cost_of_transition(Signal.SHORT, Signal.SHORT, costs).total == 0.0


def test_cost_flat_to_long(costs):
    c = cost_of_transition(Signal.FLAT, Signal.LONG, costs)
    expected_open = costs.spread / 2 + costs.slippage
    assert c.close_cost == 0.0
    assert c.open_cost == pytest.approx(expected_open)
    assert c.total == pytest.approx(expected_open)


def test_cost_long_to_flat(costs):
    c = cost_of_transition(Signal.LONG, Signal.FLAT, costs)
    expected_close = costs.spread / 2 + costs.slippage
    assert c.open_cost == 0.0
    assert c.close_cost == pytest.approx(expected_close)
    assert c.total == pytest.approx(expected_close)


def test_cost_long_to_short(costs):
    c = cost_of_transition(Signal.LONG, Signal.SHORT, costs)
    one_side = costs.spread / 2 + costs.slippage
    assert c.close_cost == pytest.approx(one_side)
    assert c.open_cost == pytest.approx(one_side)
    assert c.total == pytest.approx(2 * one_side)


def test_cost_with_commission(costs):
    costs_comm = CostModel(
        spread=0.00010,
        slippage=0.00003,
        commission_per_trade=0.00002,
        swap_per_night=0.0,
    )
    c = cost_of_transition(Signal.FLAT, Signal.LONG, costs_comm)
    one_side = costs_comm.spread / 2 + costs_comm.slippage
    assert c.total == pytest.approx(one_side + costs_comm.commission_per_trade)


def test_cost_size_scales_linearly(costs):
    c1 = cost_of_transition(Signal.FLAT, Signal.LONG, costs, size=1.0)
    c2 = cost_of_transition(Signal.FLAT, Signal.LONG, costs, size=2.0)
    assert c2.total == pytest.approx(2 * c1.total)


# ---------------------------------------------------------------------------
# Tests de execution
# ---------------------------------------------------------------------------

from backtest.engine import BacktestConfig, run_backtest
from backtest.execution import PortfolioState, apply_transition
from backtest.position import Position
from backtest.results import compute_metrics


@pytest.fixture
def simple_ohlc() -> pd.DataFrame:
    """
    OHLC sintético determinista para tests de ejecución.
    10 velas, precio conocido en cada una.
    """
    idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open":  [1.1000, 1.1010, 1.1020, 1.1030, 1.1040,
                      1.1050, 1.1060, 1.1070, 1.1080, 1.1090],
            "high":  [1.1005, 1.1015, 1.1025, 1.1035, 1.1045,
                      1.1055, 1.1065, 1.1075, 1.1085, 1.1095],
            "low":   [1.0995, 1.1005, 1.1015, 1.1025, 1.1035,
                      1.1045, 1.1055, 1.1065, 1.1075, 1.1085],
            "close": [1.1005, 1.1015, 1.1025, 1.1035, 1.1045,
                      1.1055, 1.1065, 1.1075, 1.1085, 1.1095],
        },
        index=idx,
    )
    return df


def test_execution_no_transition():
    """FLAT -> FLAT no genera trade ni coste."""
    state = PortfolioState(cash=10_000.0, position=Position())
    trade, breakdown = apply_transition(
        state, Signal.FLAT, 1.1000, pd.Timestamp("2024-01-01", tz="UTC")
    )
    assert trade is None
    assert breakdown.total == 0.0
    assert state.cash == 10_000.0


def test_execution_flat_to_long():
    """FLAT -> LONG abre posición y descuenta coste de apertura."""
    state = PortfolioState(cash=10_000.0, position=Position())
    trade, breakdown = apply_transition(
        state, Signal.LONG, 1.1000, pd.Timestamp("2024-01-01", tz="UTC")
    )
    assert trade is None  # no se cerró nada
    assert state.position.is_long
    assert state.position.entry_price == 1.1000
    # Coste = spread/2 + slippage = 0.00005 + 0.00003 = 0.00008
    assert breakdown.total == pytest.approx(0.00008)
    assert state.cash == pytest.approx(10_000.0 - 0.00008)


def test_execution_long_to_flat_realizes_pnl():
    """LONG -> FLAT cierra posición, realiza PnL, genera TradeRecord."""
    state = PortfolioState(cash=10_000.0, position=Position())
    # Abrir LONG a 1.1000
    apply_transition(
        state, Signal.LONG, 1.1000, pd.Timestamp("2024-01-01", tz="UTC")
    )
    # Cerrar a 1.1010 (ganancia de 10 pips)
    trade, breakdown = apply_transition(
        state, Signal.FLAT, 1.1010, pd.Timestamp("2024-01-01 00:15", tz="UTC")
    )
    assert trade is not None
    assert trade.direction == Signal.LONG
    assert trade.entry_price == 1.1000
    assert trade.exit_price == 1.1010
    assert trade.gross_pnl == pytest.approx(0.0010)
    assert state.position.is_flat


def test_execution_long_to_short_pays_double_cost():
    """LONG -> SHORT cierra y abre: paga coste de cierre + apertura."""
    state = PortfolioState(cash=10_000.0, position=Position())
    apply_transition(state, Signal.LONG, 1.1000, pd.Timestamp("2024-01-01", tz="UTC"))
    cash_before = state.cash
    trade, breakdown = apply_transition(
        state, Signal.SHORT, 1.1010, pd.Timestamp("2024-01-01 00:15", tz="UTC")
    )
    # Coste total de la transición = 2 * (spread/2 + slippage) = 2 * 0.00008
    assert breakdown.total == pytest.approx(0.00016)
    assert state.cash == pytest.approx(cash_before - 0.00016)
    assert state.position.is_short


# ---------------------------------------------------------------------------
# Tests de engine (NO LOOK-AHEAD) — los más importantes
# ---------------------------------------------------------------------------

def test_engine_signal_executed_next_open(simple_ohlc):
    """
    Test crítico: una señal LONG en t=0 debe ejecutarse al OPEN de t=1,
    no al close de t=0.
    """
    # Señal: LONG solo en t=0, luego FLAT
    signals = pd.Series([1] + [0] * 9, index=simple_ohlc.index, name="signal")
    result = run_backtest(
        simple_ohlc, signals, BacktestConfig(symbol="EURUSD", initial_capital=10_000.0)
    )
    # La posición al close de t=0 debe ser FLAT (aún no se ejecutó)
    assert result.position.iloc[0] == int(Signal.FLAT)
    # La posición al close de t=1 debe ser LONG (se ejecutó al open de t=1)
    assert result.position.iloc[1] == int(Signal.LONG)


def test_engine_no_lookahead_equity(simple_ohlc):
    """
    Test crítico de look-ahead: si truncamos el DataFrame después de t=k,
    el equity calculado hasta t=k debe ser idéntico al del run completo.
    Si el engine mirara datos futuros, esto fallaría.
    """
    signals_full = pd.Series([1] * 10, index=simple_ohlc.index, name="signal")
    result_full = run_backtest(simple_ohlc, signals_full)

    # Truncar en t=5
    k = 5
    ohlc_trunc = simple_ohlc.iloc[: k + 1]
    signals_trunc = signals_full.iloc[: k + 1]
    result_trunc = run_backtest(ohlc_trunc, signals_trunc)

    # El equity en t=0..k debe ser idéntico
    pd.testing.assert_series_equal(
        result_full.equity.iloc[: k + 1],
        result_trunc.equity,
        check_names=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_engine_position_series_matches_signals_shifted(simple_ohlc):
    """
    La posición al close de t debe ser igual a la señal de t-1
    (porque la señal de t-1 se ejecuta al open de t, y al close de t
    ya está reflejada).
    """
    signals = pd.Series([0, 1, 1, -1, -1, 0, 1, 0, 0, -1],
                        index=simple_ohlc.index, name="signal")
    result = run_backtest(simple_ohlc, signals)

    # position[t] == signal[t-1] para t >= 1
    for t in range(1, len(signals)):
        assert result.position.iloc[t] == signals.iloc[t - 1], (
            f"t={t}: position={result.position.iloc[t]}, "
            f"signal[t-1]={signals.iloc[t-1]}"
        )


def test_engine_zero_signals_no_trades(simple_ohlc):
    """Todas las señales FLAT: sin trades, equity constante."""
    signals = pd.Series([0] * 10, index=simple_ohlc.index, name="signal")
    result = run_backtest(simple_ohlc, signals, BacktestConfig(symbol="EURUSD", initial_capital=10_000.0))

    assert result.n_trades == 0
    assert (result.equity == 10_000.0).all()
    assert (result.position == 0).all()


def test_engine_costs_reduce_equity(simple_ohlc):
    """
    Una estrategia que abre y cierra sin ganancia bruta debe perder
    exactamente el coste total.
    """
    # LONG en t=0, FLAT en t=1
    signals = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        index=simple_ohlc.index, name="signal")
    result = run_backtest(simple_ohlc, signals, BacktestConfig(symbol="EURUSD", initial_capital=10_000.0))

    # Abre LONG al open de t=1 (1.1010), cierra al open de t=2 (1.1020)
    # Gross PnL = 1.1020 - 1.1010 = 0.0010
    # Coste total = apertura (0.00008) + cierre (0.00008) = 0.00016
    # Net = 0.0010 - 0.00016 = 0.00084
    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.gross_pnl == pytest.approx(0.0010)
    expected_equity = 10_000.0 + 0.0010 - 0.00016
    # El equity final debe reflejar el net
    assert result.equity.iloc[-1] == pytest.approx(expected_equity, abs=1e-9)

def test_net_pnl_equals_gross_minus_total_cost(simple_ohlc):
    signals = pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        index=simple_ohlc.index, name="signal")
    result = run_backtest(simple_ohlc, signals,
                          BacktestConfig(symbol="EURUSD", initial_capital=10_000.0))
    for t in result.trades:
        assert t.total_cost == pytest.approx(t.open_cost + t.close_cost)
        assert t.net_pnl == pytest.approx(t.gross_pnl - t.total_cost)

def test_equity_final_equals_capital_plus_net(simple_ohlc):
    signals = pd.Series([1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        index=simple_ohlc.index, name="signal")
    result = run_backtest(simple_ohlc, signals,
                          BacktestConfig(symbol="EURUSD", initial_capital=10_000.0))
    total_net = sum(t.net_pnl for t in result.trades)
    expected_equity = 10_000.0 + total_net
    assert result.equity.iloc[-1] == pytest.approx(expected_equity, abs=1e-9)

def test_engine_metrics_are_finite(simple_ohlc):
    signals = pd.Series([1, -1, 1, -1, 0, 0, 1, 0, -1, 0],
                        index=simple_ohlc.index, name="signal")
    result = run_backtest(simple_ohlc, signals)
    metrics = result.metrics()
    for k, v in metrics.items():
        if k == "profit_factor":
            continue  # puede ser inf
        assert np.isfinite(v), f"Métrica {k} no finita: {v}"


def test_engine_rejects_mismatched_lengths(simple_ohlc):
    signals = pd.Series([0] * 5, index=simple_ohlc.index[:5], name="signal")
    with pytest.raises(ValueError, match="longitud"):
        run_backtest(simple_ohlc, signals)


def test_engine_rejects_invalid_signals(simple_ohlc):
    signals = pd.Series([0, 1, 2, 0, 0, 0, 0, 0, 0, 0],
                        index=simple_ohlc.index, name="signal")
    with pytest.raises(ValueError, match="inválidas"):
        run_backtest(simple_ohlc, signals)