"""Coherencia de la configuración de scripts/run_all_wfo.py (sin ejecutar WFO)."""

from __future__ import annotations

import pytest

from config.settings import SUPPORTED_SYMBOLS, infer_symbol_from_path
from data.normalize import canonical_filename, parse_filename, resolve_symbol
from scripts.run_all_wfo import EXPERIMENTS


@pytest.mark.parametrize("exp", EXPERIMENTS, ids=[e["name"] for e in EXPERIMENTS])
def test_experiment_csv_has_canonical_name(exp):
    parsed = parse_filename(exp["csv"])
    symbol = resolve_symbol(parsed.raw_symbol)
    assert infer_symbol_from_path(exp["csv"]) == symbol in SUPPORTED_SYMBOLS
    assert exp["csv"].endswith(canonical_filename(symbol, parsed.timeframe, parsed.period))


@pytest.mark.parametrize("exp", EXPERIMENTS, ids=[e["name"] for e in EXPERIMENTS])
def test_non_15m_experiments_set_bars_per_month(exp):
    """run_wfo asume 2100 velas/mes (15m); cualquier otro timeframe debe indicarlo."""
    tf = parse_filename(exp["csv"]).timeframe
    extra = exp.get("extra_args", [])
    if tf == "15m":
        assert "--bars-per-month" not in extra
    else:
        assert "--bars-per-month" in extra
