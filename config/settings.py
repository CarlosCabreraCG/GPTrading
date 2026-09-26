"""
Configuración global del proyecto trading_gp.

Todos los parámetros ajustables viven aquí. Ningún módulo debe hardcodear
valores que puedan necesitar cambiarse (rutas, costes, límites numéricos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
import os

try:
    from dotenv import load_dotenv
except ImportError:  # Allows scripts to show a clear dependency error later.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# ---------------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------------
# Trading OHLC
DICT_TEMP = {
    "1m": "7d",    # 1 minuto: máx 7 días
    "5m": "60d",   # 5 minutos: máx 60 días
    "15m": "60d",  # 15 minutos: máx 60 días
    "1h": "730d",  # 1 hora: máx 2 años (aprox)
    "1d": "max"    # 1 día: histórico completo
}

SYMBOL   = os.getenv("TRADING_SYMBOL", "EURUSD=X")
INTERVAL = os.getenv("TRADING_INTERVAL", "15m")
if INTERVAL not in DICT_TEMP:
    valid_intervals = ", ".join(DICT_TEMP)
    raise ValueError(f"TRADING_INTERVAL invalido: {INTERVAL}. Usa uno de: {valid_intervals}")

SYMBOL_FILE = f"{SYMBOL}_{INTERVAL}".replace("-", "_").replace("=", "_")
PERIOD   = os.getenv("TRADING_PERIOD", DICT_TEMP[INTERVAL])

# Archivo se guarda tipo "EUR_USD_X_15m_60d.csv"
DF_FILE = f"{SYMBOL_FILE }_{PERIOD}.csv".replace("-", "_").replace("=", "_")

# Rutas
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data_files"
EXPERIMENTS_DIR: Final[Path] = PROJECT_ROOT / "experiments_output"

# CSV por defecto para desarrollo (60 días, 15m)
DEFAULT_CSV: Final[Path] = DATA_DIR / DF_FILE 

# ---------------------------------------------------------------------------
# Protección numérica para operadores GP
# ---------------------------------------------------------------------------

#: Valores por debajo de este umbral en valor absoluto se tratan como cero
#: en protected_division e inverse.
EPSILON: Final[float] = 1e-9

#: Clamp superior/inferior para cualquier operación que pueda explotar.
MAX_ABS_VALUE: Final[float] = 1e6


# ---------------------------------------------------------------------------
# Costes de transacción (placeholder, ajustar con datos reales del broker)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """
    Costes expresados en unidades de precio (no en pips, no en %).

    Para EURUSD con precio ~1.14:
        1 pip = 0.0001
    """
    spread: float = 0.00010        # 1.0 pip
    slippage: float = 0.00003      # 0.3 pip
    commission_per_trade: float = 0.0  # retail forex típico
    swap_per_night: float = 0.0    # ignorado en v1

    @property
    def round_trip_cost(self) -> float:
        """Coste total aproximado de abrir y cerrar una posición."""
        return 2 * (self.spread / 2 + self.slippage) + self.commission_per_trade


DEFAULT_COSTS: Final[CostModel] = CostModel()


# ---------------------------------------------------------------------------
# Límites de complejidad del árbol GP
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPLimits:
    """Límites duros del árbol. Se aplican durante la inicialización y la evolución."""
    max_depth: int = 6
    max_nodes: int = 40
    max_conditions: int = 8
    max_indicators: int = 10
    max_nesting: int = 3


DEFAULT_GP_LIMITS: Final[GPLimits] = GPLimits()


# ---------------------------------------------------------------------------
# Splits temporales (se usarán en Fase 4, pero los definimos ya)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitConfig:
    """Proporciones para Walk-Forward. Se afinarán en Fase 4."""
    is_ratio: float = 0.70   # 70% in-sample
    oos_ratio: float = 0.30  # 30% out-of-sample
    #: Último bloque reservado como FINAL TEST, intocable durante desarrollo.
    final_test_ratio: float = 0.0  # 0.0 en v1 con 60 días; >0 cuando haya más datos


DEFAULT_SPLIT: Final[SplitConfig] = SplitConfig()


# ---------------------------------------------------------------------------
# Configuración de validación de datos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataValidation:
    """Reglas que el cleaner aplicará al CSV."""
    required_columns: tuple[str, ...] = ("datetime", "open", "high", "low", "close")
    #: Máximo gap permitido entre velas consecutivas como múltiplo del timeframe.
    #: Un gap de 3x el timeframe se tolera (fin de semana, feriado).
    max_gap_multiplier: float = 3.0
    #: Si True, el cleaner lanza excepción si detecta gaps mayores.
    raise_on_large_gap: bool = False
    #: Si True, se eliminan filas con NaN en OHLC.
    drop_ohlc_nans: bool = True


DEFAULT_DATA_VALIDATION: Final[DataValidation] = DataValidation()