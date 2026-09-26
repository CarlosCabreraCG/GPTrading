"""
Modelo de costes aplicado a cambios de posición.

Los costes se expresan en unidades de precio (no en pips, no en %).
Para EURUSD, 1 pip = 0.0001.

Política de costes:
- Cada vez que se ABRE una posición, se paga: spread/2 + slippage
- Cada vez que se CIERRA una posición, se paga: spread/2 + slippage
- Un cambio de dirección (LONG → SHORT) implica cerrar + abrir:
    coste = 2 * (spread/2 + slippage)
- La comisión se aplica una vez por operación (apertura o cierre).
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import DEFAULT_COSTS, CostModel
from strategy.signal import Signal


@dataclass(frozen=True)
class CostBreakdown:
    """Desglose del coste de un cambio de posición."""
    close_cost: float
    open_cost: float
    commission: float

    @property
    def total(self) -> float:
        return self.close_cost + self.open_cost + self.commission


def cost_of_transition(
    from_direction: Signal,
    to_direction: Signal,
    costs: CostModel = DEFAULT_COSTS,
    size: float = 1.0,
) -> CostBreakdown:
    """
    Calcula el coste de pasar de una dirección a otra.

    Casos:
    - FLAT → FLAT: 0
    - FLAT → LONG/SHORT: apertura
    - LONG/SHORT → FLAT: cierre
    - LONG → SHORT o SHORT → LONG: cierre + apertura
    - LONG → LONG o SHORT → SHORT: 0 (no hay cambio)
    """
    if from_direction == to_direction:
        return CostBreakdown(0.0, 0.0, 0.0)

    half_spread = costs.spread / 2.0
    slip = costs.slippage

    close_cost = 0.0
    open_cost = 0.0
    commission = 0.0

    # Cerrar posición existente
    if from_direction != Signal.FLAT:
        close_cost = size * (half_spread + slip)
        commission += size * costs.commission_per_trade

    # Abrir posición nueva
    if to_direction != Signal.FLAT:
        open_cost = size * (half_spread + slip)
        commission += size * costs.commission_per_trade

    return CostBreakdown(close_cost, open_cost, commission)