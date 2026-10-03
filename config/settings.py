"""
Configuración global del proyecto trading_gp.

Todos los parámetros ajustables viven aquí. Ningún módulo debe hardcodear
valores que puedan necesitarse cambiar (rutas, costes, límites numéricos).

Los costes por símbolo se leen de un archivo .env en la raíz del proyecto.
Si el .env no existe, se usan valores por defecto razonables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data_files"
EXPERIMENTS_DIR: Final[Path] = PROJECT_ROOT / "experiments_output"

DEFAULT_CSV: Final[Path] = DATA_DIR / "EURUSD_X_15m_5y.csv"

# ---------------------------------------------------------------------------
# Protección numérica para operadores GP
# ---------------------------------------------------------------------------

EPSILON: Final[float] = 1e-9
MAX_ABS_VALUE: Final[float] = 1e6

# ---------------------------------------------------------------------------
# Símbolos soportados
# ---------------------------------------------------------------------------

SUPPORTED_SYMBOLS: Final[tuple[str, ...]] = (
    "EURUSD",
    "GBPUSD",
    "XAUUSD",
    "XAGUSD",
    "WTI",
    "BRENT",
    "USA500",
    "USATECH",
)

# ---------------------------------------------------------------------------
# Lectura de .env
# ---------------------------------------------------------------------------

def _load_env_file(path: Path) -> dict[str, str]:
    """
    Lee un archivo .env sin dependencias externas.

    Formato: KEY=value, una por línea. Líneas vacías y # comentarios se ignoran.
    """
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


_ENV_PATH = PROJECT_ROOT / ".env"
_ENV = _load_env_file(_ENV_PATH)

# Añadir también las variables de entorno del sistema (prioridad sobre .env)
for k in list(os.environ.keys()):
    if k.endswith(("_SPREAD", "_SLIPPAGE", "_COMMISSION")):
        _ENV[k] = os.environ[k]


# ---------------------------------------------------------------------------
# CostModel
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """
    Costes expresados en unidades de precio del instrumento.

    Para EURUSD con precio ~1.14: 1 pip = 0.0001.
    Para XAUUSD con precio ~2000: 1 pip típico = 0.10 USD (depende del broker).
    Para índices: 1 punto = 1.0.
    """
    spread: float
    slippage: float
    commission_per_trade: float = 0.0
    swap_per_night: float = 0.0

    @property
    def round_trip_cost(self) -> float:
        """Coste total aproximado de abrir y cerrar una posición."""
        return 2 * (self.spread / 2 + self.slippage) + self.commission_per_trade


# ---------------------------------------------------------------------------
# Valores por defecto por símbolo (usados si no hay .env)
# ---------------------------------------------------------------------------

_DEFAULT_COSTS: dict[str, CostModel] = {
    "XAUUSD": CostModel(spread=0.30, slippage=0.10, commission_per_trade=0.0),
    "XAGUSD": CostModel(spread=0.03, slippage=0.01, commission_per_trade=0.0),
    "EURUSD": CostModel(spread=0.00010, slippage=0.00003, commission_per_trade=0.0),
    "GBPUSD": CostModel(spread=0.00012, slippage=0.00004, commission_per_trade=0.0), 
    "WTI":    CostModel(spread=0.03, slippage=0.01, commission_per_trade=0.0),
    "BRENT":  CostModel(spread=0.03, slippage=0.01, commission_per_trade=0.0),
    "USA500": CostModel(spread=0.50, slippage=0.20, commission_per_trade=0.0),
    "USATECH": CostModel(spread=2.00, slippage=0.80, commission_per_trade=0.0),
}


def _build_costs_from_env() -> dict[str, CostModel]:
    """
    Construye COSTS_BY_SYMBOL combinando defaults con .env.

    Para cada símbolo:
    - spread = _ENV["<SYM>_SPREAD"] si existe, si no el default.
    - slippage = _ENV["<SYM>_SLIPPAGE"] si existe, si no el default.
    - commission = _ENV["<SYM>_COMMISSION"] si existe, si no 0.0.
    """
    out: dict[str, CostModel] = {}
    for symbol in SUPPORTED_SYMBOLS:
        default = _DEFAULT_COSTS[symbol]

        spread_key = f"{symbol}_SPREAD"
        slip_key = f"{symbol}_SLIPPAGE"
        comm_key = f"{symbol}_COMMISSION"

        spread = float(_ENV[spread_key]) if spread_key in _ENV else default.spread
        slip = float(_ENV[slip_key]) if slip_key in _ENV else default.slippage
        comm = float(_ENV[comm_key]) if comm_key in _ENV else default.commission_per_trade

        out[symbol] = CostModel(
            spread=spread,
            slippage=slip,
            commission_per_trade=comm,
            swap_per_night=0.0,
        )
    return out


COSTS_BY_SYMBOL: Final[dict[str, CostModel]] = _build_costs_from_env()


def get_costs(symbol: str) -> CostModel:
    """
    Devuelve el CostModel para un símbolo.

    Lanza KeyError si el símbolo no está registrado.
    """
    if symbol not in COSTS_BY_SYMBOL:
        raise KeyError(
            f"Símbolo '{symbol}' no registrado en COSTS_BY_SYMBOL. "
            f"Soportados: {sorted(COSTS_BY_SYMBOL.keys())}. "
            f"Añádelo a config/settings.py y a .env si es necesario."
        )
    return COSTS_BY_SYMBOL[symbol]


def infer_symbol_from_path(csv_path: str | Path) -> str:
    """
    Infiere el símbolo desde el nombre del archivo.

    Ejemplo: 'XAUUSD_X_15m_5y.csv' -> 'XAUUSD'.

    Lanza ValueError si no se puede inferir o si el símbolo no está soportado.
    """
    path = Path(csv_path)
    stem = path.stem  # sin extensión
    # El símbolo es el primer bloque antes de '_'
    parts = stem.split("_")
    if not parts:
        raise ValueError(f"No se puede inferir símbolo de: {csv_path}")

    candidate = parts[0].upper()
    if candidate not in SUPPORTED_SYMBOLS:
        raise ValueError(
            f"Símbolo inferido '{candidate}' no está en SUPPORTED_SYMBOLS. "
            f"Soportados: {SUPPORTED_SYMBOLS}. "
            f"Usa --symbol para forzar uno distinto."
        )
    return candidate


# ---------------------------------------------------------------------------
# Deprecado: DEFAULT_COSTS
# ---------------------------------------------------------------------------

#: Deprecado. Usar get_costs(symbol) en su lugar.
#: Se mantiene solo por retrocompatibilidad con tests antiguos.
DEFAULT_COSTS: Final[CostModel] = COSTS_BY_SYMBOL["EURUSD"]


# ---------------------------------------------------------------------------
# Límites de complejidad del árbol GP
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPLimits:
    max_depth: int = 6
    max_nodes: int = 40
    max_conditions: int = 8
    max_indicators: int = 10
    max_nesting: int = 3


DEFAULT_GP_LIMITS: Final[GPLimits] = GPLimits()


# ---------------------------------------------------------------------------
# Splits temporales
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitConfig:
    is_ratio: float = 0.70
    oos_ratio: float = 0.30
    final_test_ratio: float = 0.10  # antes era 0.0; ahora obligatorio 10%


DEFAULT_SPLIT: Final[SplitConfig] = SplitConfig()


# ---------------------------------------------------------------------------
# Validación de datos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataValidation:
    required_columns: tuple[str, ...] = ("datetime", "open", "high", "low", "close")
    max_gap_multiplier: float = 3.0
    raise_on_large_gap: bool = False
    drop_ohlc_nans: bool = True


DEFAULT_DATA_VALIDATION: Final[DataValidation] = DataValidation()