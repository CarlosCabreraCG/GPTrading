"""Tests de data.normalize sobre CSV sintéticos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config.settings import infer_symbol_from_path
from data import normalize as nz
from data.cleaner import load_and_clean
from data.csv_tools import check_rows, read_head


def _write(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


ROWS_DMY = [
    "01/01/2024 00:00,1.0,1.2,0.9,1.1",
    "01/01/2024 00:15,1.1,1.3,1.0,1.2",
    "01/01/2024 00:30,1.2,1.3,1.1,1.25",
]


@pytest.mark.parametrize("token,expected", [
    ("15m", "15m"), ("m15", "15m"), ("M15", "15m"), ("H4", "4h"),
    ("4h", "4h"), ("1d", "1d"), ("5y", None), ("abc", None),
])
def test_normalize_timeframe(token, expected):
    assert nz.normalize_timeframe(token) == expected


@pytest.mark.parametrize("name,expected", [
    ("lightcmdusd_15m_5y.csv", nz.ParsedName("lightcmdusd", "15m", "5y")),
    ("dxy_m15_5y.csv", nz.ParsedName("dxy", "15m", "5y")),
    ("xauusd-H4-5y.csv", nz.ParsedName("xauusd", "4h", "5y")),
    ("eurusd_15m_60d.csv", nz.ParsedName("eurusd", "15m", "60d")),
    ("xagusd.csv", nz.ParsedName("xagusd", None, None)),
])
def test_parse_filename(name, expected):
    assert nz.parse_filename(name) == expected


@pytest.mark.parametrize("raw,sym", [
    ("lightcmdusd", "WTI"), ("brentcmdusd", "BRENT"), ("usa500idxusd", "USA500"),
    ("usatechidxusd", "USATECH"), ("dollaridxusd", "DXY"), ("XAUUSD", "XAUUSD"),
    ("dxy", "DXY"),
])
def test_resolve_symbol(raw, sym):
    assert nz.resolve_symbol(raw) == sym


def test_resolve_symbol_unknown():
    with pytest.raises(ValueError, match="No se reconoce"):
        nz.resolve_symbol("dyx")


@pytest.mark.parametrize("alias", sorted(nz.SOURCE_SYMBOL_ALIASES))
def test_canonical_name_is_inferable(alias):
    """Todo alias de la fuente produce un nombre que infer_symbol_from_path acepta."""
    name = nz.canonical_filename(nz.resolve_symbol(alias), "15m", "5y")
    assert infer_symbol_from_path(name) == nz.SOURCE_SYMBOL_ALIASES[alias]


def test_rename_and_replace(tmp_path):
    src = _write(tmp_path / "lightcmdusd_15m_5y.csv", "datetime,open,high,low,close", ROWS_DMY)
    res = nz.normalize_csv(src, replace=True)
    assert res.target == tmp_path / "wti_15m_5y.csv"
    assert res.source_removed and not src.exists()
    df, _ = load_and_clean(res.target)
    assert len(df) == 3
    assert df["close"].tolist() == [1.1, 1.2, 1.25]


def test_without_replace_keeps_source(tmp_path):
    src = _write(tmp_path / "brentcmdusd_15m_5y.csv", "datetime,open,high,low,close", ROWS_DMY)
    res = nz.normalize_csv(src)
    assert src.exists() and res.target.exists()


def test_reorders_columns_and_drops_extra(tmp_path):
    # Mismo orden que eurusd_15m_60d.csv + una columna extra
    rows = ["01/01/2024 00:00,1.1,1.2,0.9,1.0,10", "01/01/2024 00:15,1.2,1.3,1.0,1.1,20"]
    src = _write(tmp_path / "eurusd_15m_60d.csv", "Datetime,Close,High,Low,Open,Volume", rows)
    res = nz.normalize_csv(src)
    assert res.target == src and not res.renamed
    assert res.dropped_columns == ("Volume",)
    head = read_head(src)
    assert list(head.columns) == ["datetime", "open", "high", "low", "close"]
    assert head["open"].tolist() == ["1.0", "1.1"]
    assert all(c.ok for c in check_rows(head))


@pytest.mark.parametrize("rows,fmt", [
    (["2024-01-01 00:00:00,1,1,1,1", "2024-01-01 00:15:00,1,1,1,1"], "%Y-%m-%d %H:%M:%S"),
    (["01.01.2024 00:00:00.000,1,1,1,1", "01.01.2024 00:15:00.000,1,1,1,1"], "%d.%m.%Y %H:%M:%S.%f"),
    (["2024-01-01T00:00:00+01:00,1,1,1,1", "2024-01-01T00:15:00+01:00,1,1,1,1"], "ISO8601"),
])
def test_converts_datetime_formats(tmp_path, rows, fmt):
    src = _write(tmp_path / "xauusd_15m_1d.csv", "datetime,open,high,low,close", rows)
    res = nz.normalize_csv(src)
    assert res.input_datetime_format == fmt
    first = read_head(src, 1)["datetime"].iloc[0]
    # ISO con +01:00 se convierte a UTC
    assert first == ("31/12/2023 23:00" if fmt == "ISO8601" else "01/01/2024 00:00")


def test_day_month_not_guessed_per_row(tmp_path):
    """dd/mm se aplica a todas las filas: 02/01 es 2 de enero, no 1 de febrero."""
    rows = ["02/01/2024 00:00,1,1,1,1", "13/01/2024 00:00,1,1,1,1"]
    src = _write(tmp_path / "xauusd_1d.csv", "datetime,open,high,low,close", rows)
    nz.normalize_csv(src)
    df, _ = load_and_clean(src)
    assert df.index[0] == pd.Timestamp("2024-01-02")


def test_unparseable_datetime_raises(tmp_path):
    src = _write(tmp_path / "xauusd_15m.csv", "datetime,open,high,low,close", ["mañana,1,1,1,1"])
    with pytest.raises(ValueError, match="formato de fecha"):
        nz.normalize_csv(src)


def test_timeframe_inferred_from_data(tmp_path):
    src = _write(tmp_path / "xauusd.csv", "datetime,open,high,low,close", ROWS_DMY)
    res = nz.normalize_csv(src, dry_run=True)
    assert res.target.name == "xauusd_15m.csv"
    assert not res.written and not (tmp_path / "xauusd_15m.csv").exists()


def test_explicit_symbol(tmp_path):
    src = _write(tmp_path / "dyx_15m_5y.csv", "datetime,open,high,low,close", ROWS_DMY)
    res = nz.normalize_csv(src, symbol="DXY", replace=True)
    assert res.target.name == "dxy_15m_5y.csv"


def test_refuses_overwrite(tmp_path):
    _write(tmp_path / "wti_15m_5y.csv", "datetime,open,high,low,close", ROWS_DMY)
    src = _write(tmp_path / "lightcmdusd_15m_5y.csv", "datetime,open,high,low,close", ROWS_DMY)
    with pytest.raises(FileExistsError):
        nz.normalize_csv(src, replace=True)
    assert src.exists()


def test_missing_column_raises(tmp_path):
    src = _write(tmp_path / "xauusd_15m.csv", "datetime,open,high,close", ["01/01/2024 00:00,1,1,1"])
    with pytest.raises(ValueError, match="Faltan columnas"):
        nz.normalize_csv(src)


def test_does_not_sort_or_dedupe(tmp_path):
    """El normalizador no limpia: orden y duplicados quedan para el cleaner."""
    rows = [ROWS_DMY[1], ROWS_DMY[0], ROWS_DMY[0]]
    src = _write(tmp_path / "xauusd_15m.csv", "datetime,open,high,low,close", rows)
    res = nz.normalize_csv(src)
    assert res.n_rows == 3
    assert read_head(src)["datetime"].tolist() == [
        "01/01/2024 00:15", "01/01/2024 00:00", "01/01/2024 00:00"]
