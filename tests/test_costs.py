"""
Tests de costes por símbolo.

Verifica:
- COSTS_BY_SYMBOL contiene los 7 símbolos esperados.
- get_costs devuelve el CostModel correcto.
- get_costs lanza KeyError para símbolos no registrados.
- infer_symbol_from_path funciona con nombres válidos e inválidos.
- BacktestConfig resuelve costes automáticamente desde symbol.
- BacktestConfig falla si symbol no está registrado.
"""

from __future__ import annotations

import pytest

from backtest.engine import BacktestConfig
from config.settings import (
    COSTS_BY_SYMBOL,
    SUPPORTED_SYMBOLS,
    CostModel,
    get_costs,
    infer_symbol_from_path,
)


def test_supported_symbols_has_expected():
    expected = {"EURUSD", "XAUUSD", "XAGUSD", "WTI", "BRENT", "USA500", "USATECH"}
    assert set(SUPPORTED_SYMBOLS) == expected


def test_costs_by_symbol_has_all_supported():
    for sym in SUPPORTED_SYMBOLS:
        assert sym in COSTS_BY_SYMBOL, f"Falta {sym}"
        assert isinstance(COSTS_BY_SYMBOL[sym], CostModel)


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


def test_infer_symbol_from_path_valid():
    assert infer_symbol_from_path("XAUUSD_X_15m_5y.csv") == "XAUUSD"
    assert infer_symbol_from_path("EURUSD_X_15m_5y.csv") == "EURUSD"
    assert infer_symbol_from_path("/path/to/WTI_X_15m_5y.csv") == "WTI"


def test_infer_symbol_from_path_invalid():
    with pytest.raises(ValueError, match="no está en SUPPORTED_SYMBOLS"):
        infer_symbol_from_path("UNKNOWN_X_15m_5y.csv")


def test_backtest_config_resolves_costs_from_symbol():
    config = BacktestConfig(symbol="XAUUSD")
    assert config.costs is not None
    assert config.costs.spread == 0.30


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


def test_supported_symbols_has_expected():
    expected = {"EURUSD", "GBPUSD", "XAUUSD", "XAGUSD", "WTI", "BRENT", "USA500", "USATECH"}
    assert set(SUPPORTED_SYMBOLS) == expected