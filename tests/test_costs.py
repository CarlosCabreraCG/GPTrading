"""
Tests de costes por símbolo.

Data-driven: los tests por símbolo se parametrizan sobre SUPPORTED_SYMBOLS.
Añadir un instrumento en config/settings.py genera sus tests automáticamente;
si falta su coste o sus horas de trading, el test correspondiente falla.

Verifica:
- Cada símbolo soportado tiene CostModel válido y TRADING_HOURS_PER_WEEK.
- infer_symbol_from_path reconoce cada símbolo soportado.
- BacktestConfig resuelve costes automáticamente desde symbol.
- get_costs / BacktestConfig fallan para símbolos no registrados.
"""

from __future__ import annotations

import pytest

from backtest.engine import BacktestConfig
from config.settings import (
    COSTS_BY_SYMBOL,
    SUPPORTED_SYMBOLS,
    TRADING_HOURS_PER_WEEK,
    CostModel,
    bars_per_year,
    get_costs,
    infer_symbol_from_path,
)


# ---------------------------------------------------------------------------
# Consistencia del catálogo de símbolos
# ---------------------------------------------------------------------------

def test_supported_symbols_not_empty_and_unique():
    assert len(SUPPORTED_SYMBOLS) > 0
    assert len(set(SUPPORTED_SYMBOLS)) == len(SUPPORTED_SYMBOLS)


def test_supported_symbols_are_uppercase():
    for sym in SUPPORTED_SYMBOLS:
        assert sym == sym.upper(), f"{sym} debe ir en mayúsculas"


def test_costs_by_symbol_matches_supported():
    assert set(COSTS_BY_SYMBOL) == set(SUPPORTED_SYMBOLS)


# ---------------------------------------------------------------------------
# Tests por símbolo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sym", SUPPORTED_SYMBOLS)
def test_symbol_has_valid_costs(sym):
    c = get_costs(sym)
    assert isinstance(c, CostModel)
    assert c.spread > 0, f"{sym}: spread debe ser > 0"
    assert c.slippage >= 0, f"{sym}: slippage debe ser >= 0"
    assert c.commission_per_trade >= 0, f"{sym}: comisión debe ser >= 0"


@pytest.mark.parametrize("sym", SUPPORTED_SYMBOLS)
def test_symbol_has_trading_hours(sym):
    assert sym in TRADING_HOURS_PER_WEEK, f"Falta TRADING_HOURS_PER_WEEK[{sym}]"
    assert 0 < TRADING_HOURS_PER_WEEK[sym] <= 168


@pytest.mark.parametrize("sym", SUPPORTED_SYMBOLS)
def test_symbol_bars_per_year_positive(sym):
    assert bars_per_year(sym, bar_seconds=15 * 60) > 0


@pytest.mark.parametrize("sym", SUPPORTED_SYMBOLS)
def test_infer_symbol_from_path_for_each_symbol(sym):
    assert infer_symbol_from_path(f"{sym.lower()}_15m_5y.csv") == sym
    assert infer_symbol_from_path(f"/path/to/{sym}_X_15m_5y.csv") == sym


@pytest.mark.parametrize("sym", SUPPORTED_SYMBOLS)
def test_backtest_config_resolves_costs_for_each_symbol(sym):
    config = BacktestConfig(symbol=sym)
    assert config.costs == get_costs(sym)


# ---------------------------------------------------------------------------
# Valores concretos y errores
# ---------------------------------------------------------------------------

def test_get_costs_returns_correct_for_xauusd():
    c = get_costs("XAUUSD")
    assert c.spread == 0.30
    assert c.slippage == 0.10


def test_get_costs_returns_correct_for_eurusd():
    c = get_costs("EURUSD")
    assert c.spread == 0.00010
    assert c.slippage == 0.00003


def test_get_costs_raises_for_unknown():
    with pytest.raises(KeyError, match="no registrado"):
        get_costs("UNKNOWN")


def test_bars_per_year_raises_for_unknown():
    with pytest.raises(KeyError):
        bars_per_year("UNKNOWN", bar_seconds=900)


def test_infer_symbol_from_path_invalid():
    with pytest.raises(ValueError, match="no está en SUPPORTED_SYMBOLS"):
        infer_symbol_from_path("UNKNOWN_15m_5y.csv")


def test_backtest_config_raises_for_unknown_symbol():
    with pytest.raises(KeyError, match="no registrado"):
        BacktestConfig(symbol="UNKNOWN")


def test_backtest_config_accepts_explicit_costs():
    custom = CostModel(spread=0.99, slippage=0.01)
    config = BacktestConfig(symbol="XAUUSD", costs=custom)
    assert config.costs.spread == 0.99


def test_backtest_config_requires_symbol():
    with pytest.raises(TypeError):
        BacktestConfig()
