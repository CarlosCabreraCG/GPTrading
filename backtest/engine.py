"""
Motor de backtest.

Lógica temporal (crítica, leer con cuidado):

    Para cada vela t en [0, N-1]:
        - Si t == 0:
            * No hay señal pendiente.
            * Registrar equity inicial al close de t=0.
            * Guardar signal[0] como pendiente para ejecutar en t=1.
        - Si t > 0:
            * EJECUTAR la señal pendiente de t-1 al OPEN de t.
            * MARCAR A MERCADO al CLOSE de t.
            * REGISTRAR equity al close de t.
            * GUARDAR signal[t] como pendiente para ejecutar en t+1.

    La señal en t nunca usa datos de t+1. La orden de t-1 se ejecuta al open
    de t (que es pasado cuando procesamos t). No hay look-ahead.

Costes: se aplican al cambio de posición, descontados del cash.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.execution import PortfolioState, TradeRecord, apply_transition
from backtest.position import Position
from backtest.results import BacktestResult
from config.settings import DEFAULT_COSTS, CostModel
from strategy.signal import Signal, validate_signal_series


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    costs: CostModel = DEFAULT_COSTS
    size: float = 1.0


def run_backtest(
    ohlc: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Ejecuta el backtest.

    Parámetros:
    - ohlc: DataFrame con columnas open, high, low, close. Indexado por datetime.
    - signals: Series de Signal (0/1/-1), misma longitud e índice que ohlc.
    - config: configuración (capital inicial, costes, tamaño).

    Devuelve BacktestResult.
    """
    if config is None:
        config = BacktestConfig()

    # Validaciones
    if len(ohlc) != len(signals):
        raise ValueError(
            f"ohlc y signals tienen distinta longitud: {len(ohlc)} vs {len(signals)}"
        )
    if not (ohlc.index == signals.index).all():
        raise ValueError("ohlc y signals deben compartir el mismo índice.")

    required = {"open", "high", "low", "close"}
    if not required.issubset(ohlc.columns):
        raise ValueError(f"Faltan columnas en ohlc: {required - set(ohlc.columns)}")

    validate_signal_series(signals)

    # Estado inicial
    state = PortfolioState(
        cash=config.initial_capital,
        position=Position(),
    )

    n = len(ohlc)
    equity_arr = np.full(n, np.nan)
    position_arr = np.zeros(n, dtype=int)
    trades: list[TradeRecord] = []

    pending_signal: Signal | None = None

    open_prices = ohlc["open"].to_numpy()
    close_prices = ohlc["close"].to_numpy()
    timestamps = ohlc.index

    for t in range(n):
        if t == 0:
            # No hay nada pendiente. Equity inicial al close de t=0.
            equity_arr[t] = config.initial_capital
            position_arr[t] = int(state.position.direction)
            pending_signal = Signal(int(signals.iloc[t]))
            continue

        # 1. Ejecutar la señal pendiente (generada en t-1) al OPEN de t
        if pending_signal is not None:
            trade, _ = apply_transition(
                state=state,
                target_direction=pending_signal,
                execution_price=float(open_prices[t]),
                execution_time=timestamps[t],
                costs=config.costs,
                size=config.size,
            )
            if trade is not None:
                trades.append(trade)

        # 2. Marcar a mercado al CLOSE de t
        #    (el equity se calcula con cash + PnL no realizado)
        equity_arr[t] = state.equity(float(close_prices[t]))
        position_arr[t] = int(state.position.direction)

        # 3. Guardar la señal de t para ejecutar en t+1
        pending_signal = Signal(int(signals.iloc[t]))

    equity_series = pd.Series(equity_arr, index=ohlc.index, name="equity")
    position_series = pd.Series(position_arr, index=ohlc.index, name="position")

    return BacktestResult(
        equity=equity_series,
        position=position_series,
        trades=trades,
        initial_capital=config.initial_capital,
    )


@dataclass
class BarLog:
    """Estado completo de la cartera en una vela."""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    signal: int              # señal generada en esta vela (para la siguiente)
    pending_signal: int | None  # señal pendiente de ejecutar (la de t-1)
    position_before: int     # dirección antes de ejecutar la orden
    position_after: int      # dirección después de ejecutar
    execution_price: float | None  # precio de ejecución (open de esta vela)
    trade_opened: bool
    trade_closed: bool
    trade_direction: int | None    # dirección del trade cerrado (si hubo cierre)
    trade_entry_price: float | None
    trade_exit_price: float | None
    trade_gross_pnl: float | None
    trade_cost: float | None
    cash_before: float
    cash_after: float
    unrealized_pnl: float
    equity: float


def run_backtest_with_log(
    ohlc: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig | None = None,
) -> tuple[BacktestResult, list[BarLog]]:
    """
    Igual que run_backtest, pero devuelve además un log detallado por vela.
    Útil para auditar el comportamiento del backtester trade por trade.
    """
    if config is None:
        config = BacktestConfig()

    if len(ohlc) != len(signals):
        raise ValueError("ohlc y signals con distinta longitud")
    if not (ohlc.index == signals.index).all():
        raise ValueError("índices distintos")

    from strategy.signal import validate_signal_series
    validate_signal_series(signals)

    state = PortfolioState(cash=config.initial_capital, position=Position())
    n = len(ohlc)
    equity_arr = np.full(n, np.nan)
    position_arr = np.zeros(n, dtype=int)
    trades: list[TradeRecord] = []
    logs: list[BarLog] = []

    pending_signal: Signal | None = None

    open_prices = ohlc["open"].to_numpy()
    high_prices = ohlc["high"].to_numpy()
    low_prices = ohlc["low"].to_numpy()
    close_prices = ohlc["close"].to_numpy()
    timestamps = ohlc.index

    for t in range(n):
        cash_before = state.cash
        position_before = int(state.position.direction)
        execution_price: float | None = None
        trade_opened = False
        trade_closed = False
        trade_dir: int | None = None
        trade_entry: float | None = None
        trade_exit: float | None = None
        trade_gross: float | None = None
        trade_cost: float | None = None

        if t == 0:
            equity_arr[t] = config.initial_capital
            position_arr[t] = int(state.position.direction)
            pending_signal = Signal(int(signals.iloc[t]))
            logs.append(BarLog(
                timestamp=timestamps[t],
                open=float(open_prices[t]), high=float(high_prices[t]),
                low=float(low_prices[t]), close=float(close_prices[t]),
                signal=int(signals.iloc[t]),
                pending_signal=None,
                position_before=position_before,
                position_after=position_before,
                execution_price=None,
                trade_opened=False, trade_closed=False,
                trade_direction=None, trade_entry_price=None,
                trade_exit_price=None, trade_gross_pnl=None, trade_cost=None,
                cash_before=cash_before, cash_after=state.cash,
                unrealized_pnl=0.0,
                equity=config.initial_capital,
            ))
            continue

        # Ejecutar señal pendiente al open de t
        if pending_signal is not None:
            execution_price = float(open_prices[t])
            trade, _ = apply_transition(
                state=state,
                target_direction=pending_signal,
                execution_price=execution_price,
                execution_time=timestamps[t],
                costs=config.costs,
                size=config.size,
            )
            if trade is not None:
                trades.append(trade)
                trade_closed = True
                trade_dir = int(trade.direction)
                trade_entry = trade.entry_price
                trade_exit = trade.exit_price
                trade_gross = trade.gross_pnl
                trade_cost = trade.cost
            if position_before != int(pending_signal):
                trade_opened = int(pending_signal) != 0

        equity_arr[t] = state.equity(float(close_prices[t]))
        position_arr[t] = int(state.position.direction)

        logs.append(BarLog(
            timestamp=timestamps[t],
            open=float(open_prices[t]), high=float(high_prices[t]),
            low=float(low_prices[t]), close=float(close_prices[t]),
            signal=int(signals.iloc[t]),
            pending_signal=int(pending_signal) if pending_signal is not None else None,
            position_before=position_before,
            position_after=int(state.position.direction),
            execution_price=execution_price,
            trade_opened=trade_opened,
            trade_closed=trade_closed,
            trade_direction=trade_dir,
            trade_entry_price=trade_entry,
            trade_exit_price=trade_exit,
            trade_gross_pnl=trade_gross,
            trade_cost=trade_cost,
            cash_before=cash_before,
            cash_after=state.cash,
            unrealized_pnl=state.position.unrealized_pnl(float(close_prices[t])),
            equity=float(equity_arr[t]),
        ))

        pending_signal = Signal(int(signals.iloc[t]))

    equity_series = pd.Series(equity_arr, index=ohlc.index, name="equity")
    position_series = pd.Series(position_arr, index=ohlc.index, name="position")

    result = BacktestResult(
        equity=equity_series,
        position=position_series,
        trades=trades,
        initial_capital=config.initial_capital,
    )
    return result, logs


def logs_to_dataframe(logs: list[BarLog]) -> pd.DataFrame:
    """Convierte el log a DataFrame para inspección."""
    return pd.DataFrame([l.__dict__ for l in logs]).set_index("timestamp")