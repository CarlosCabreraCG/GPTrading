"""
Estado de la posición a lo largo del backtest.

Una posición tiene:
- direction: LONG, SHORT o FLAT
- entry_price: precio al que se abrió (None si FLAT)
- entry_time: timestamp de apertura (None si FLAT)
- size: unidades (1.0 por defecto)

El módulo también define la lógica de "cambio de posición" que el engine
usará para decidir si hay que ejecutar una orden.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from strategy.signal import Signal


@dataclass
class Position:
    """Estado mutable de la posición actual."""
    direction: Signal = Signal.FLAT
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    entry_cost: float = 0.0
    size: float = 1.0

    @property
    def is_flat(self) -> bool:
        return self.direction == Signal.FLAT

    @property
    def is_long(self) -> bool:
        return self.direction == Signal.LONG

    @property
    def is_short(self) -> bool:
        return self.direction == Signal.SHORT

    def open(self, direction, price, time, entry_cost: float = 0.0) -> None:
        """Abre una posición nueva. Asume que estaba FLAT."""
        if direction == Signal.FLAT:
            raise ValueError("No se puede abrir una posición FLAT.")
        if not self.is_flat:
            raise RuntimeError(
                f"No se puede abrir posición {direction.name} "
                f"estando en {self.direction.name}."
            )
        self.direction = direction
        self.entry_price = float(price)
        self.entry_time = time
        self.entry_cost = entry_cost
        
    def close(self) -> None:
        """Cierra la posición actual y vuelve a FLAT."""
        self.direction = Signal.FLAT
        self.entry_price = None
        self.entry_time = None
        self.entry_cost = 0.0

    def unrealized_pnl(self, current_price: float) -> float:
        """
        PnL no realizado de la posición actual, dado un precio de mercado.

        Para LONG:  size * (current_price - entry_price)
        Para SHORT: size * (entry_price - current_price)
        Para FLAT:  0
        """
        if self.is_flat or self.entry_price is None:
            return 0.0
        if self.is_long:
            return self.size * (current_price - self.entry_price)
        else:  # SHORT
            return self.size * (self.entry_price - current_price)