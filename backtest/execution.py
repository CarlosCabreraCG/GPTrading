"""
Lógica de ejecución de órdenes.

Este módulo contiene la ÚNICA función que modifica el estado de la cartera
(cash + posición) cuando se ejecuta una orden. Todo lo demás es cálculo.

Regla: la ejecución ocurre al open de la vela t+1, con la señal generada
al close de la vela t. Esta función se llama con el open_price de t+1.

Los costes se descuentan del cash en el momento de la transición.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from backtest.costs import CostBreakdown, cost_of_transition
from backtest.position import Position
from config.settings import DEFAULT_COSTS, CostModel
from strategy.signal import Signal


@dataclass
class TradeRecord:
    """Registro de una transición de posición (una 'operación' completa o parcial)."""
    entry_time: datetime
    exit_time: datetime | None
    direction: Signal
    entry_price: float
    exit_price: float | None
    size: float
    gross_pnl: float
    open_cost: float
    close_cost: float
    total_cost: float
    net_pnl: float


@dataclass
class PortfolioState:
    """
    Estado completo de la cartera en un momento dado.

    cash: efectivo disponible. Empieza en initial_capital.
    position: posición actual (LONG/SHORT/FLAT).
    realized_pnl: PnL acumulado de operaciones cerradas.
    total_costs: costes acumulados pagados.
    """
    cash: float
    position: Position
    realized_pnl: float = 0.0
    total_costs: float = 0.0

    def equity(self, current_price: float) -> float:
        """Valor total de la cartera al precio actual."""
        return self.cash + self.position.unrealized_pnl(current_price)


def apply_transition(
    state: PortfolioState,
    target_direction: Signal,
    execution_price: float,
    execution_time: datetime,
    costs: CostModel = DEFAULT_COSTS,
    size: float = 1.0,
) -> tuple[TradeRecord | None, CostBreakdown]:
    """
    Aplica una transición de posición al estado de la cartera.

    Pasos:
    1. Calcular coste de la transición (from -> to).
    2. Si había posición, cerrarla: realizar PnL, registrar TradeRecord.
    3. Si la nueva dirección no es FLAT, abrir posición al execution_price.
    4. Descontar costes del cash.

    Devuelve (TradeRecord si se cerró una operación, CostBreakdown aplicado).
    """
    from_direction = state.position.direction

    if from_direction == target_direction:
        # No hay transición
        return None, CostBreakdown(0.0, 0.0, 0.0)

    # 1. Coste
    breakdown = cost_of_transition(from_direction, target_direction, costs, size)

    # 2. Cerrar posición existente (si la hay)
    trade_record: TradeRecord | None = None
    if from_direction != Signal.FLAT:
        entry_price = state.position.entry_price
        entry_time = state.position.entry_time
        entry_cost = state.position.entry_cost
        if entry_price is None or entry_time is None:
            raise RuntimeError("Posición no FLAT sin entry_price/entry_time")

        if from_direction == Signal.LONG:
            gross = size * (execution_price - entry_price)
        else:  # SHORT
            gross = size * (entry_price - execution_price)

        # Coste total de esta operación = coste de apertura (pagado antes) +
        # coste de cierre (mitad del breakdown actual). Pero por simplicidad
        # contable, atribuimos a esta operación solo la parte de cierre.
        # El coste de apertura ya se descontó del cash en su momento.
        close_cost = breakdown.close_cost + breakdown.commission / 2.0
        total_cost = entry_cost + close_cost
        net = gross - total_cost

        trade_record = TradeRecord(
            entry_time=entry_time,
            exit_time=execution_time,
            direction=from_direction,
            entry_price=entry_price,
            exit_price=execution_price,
            size=size,
            gross_pnl=gross,
            open_cost=entry_cost,
            close_cost=close_cost,
            total_cost=total_cost,
            net_pnl=net,
        )

        state.realized_pnl += net
        state.cash += gross
        state.position.close()

    # 3. Abrir nueva posición (si no es FLAT)
    if target_direction != Signal.FLAT:
        one_side_cost = size * (costs.spread / 2 + costs.slippage)
        open_cost_for_new = one_side_cost + size * costs.commission_per_trade / 2.0
        state.position.open(target_direction, execution_price, execution_time,
                            entry_cost=open_cost_for_new)

    # 4. Descontar costes totales del cash
    state.cash -= breakdown.total
    state.total_costs += breakdown.total

    return trade_record, breakdown