"""Tests de data.csv_tools sobre CSV sintéticos (nunca sobre data_files/)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data import csv_tools as ct
from tests.conftest import make_ohlc


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    out = df.copy()
    out.index = out.index.tz_localize(None).strftime("%d/%m/%Y %H:%M")
    out.index.name = "datetime"
    out.to_csv(path)
    return path


@pytest.fixture
def good_csv(tmp_path) -> Path:
    return _write_csv(make_ohlc(n=300), tmp_path / "xauusd_15m_5y.csv")


@pytest.fixture
def bad_csv(tmp_path) -> Path:
    lines = [
        "datetime,open,high,low,close",
        "01/01/2024 00:00,1.0,1.2,0.9,1.1",
        "2024-01-01 00:15,1.1,1.2,1.0,1.1",    # formato de fecha incorrecto
        "01/01/2024 00:30,1.1,1.0,1.0,1.15",   # high < close
        "01/01/2024 00:30,1.1,1.2,1.0,abc",    # duplicado + close no numérico
        "01/01/2024 00:15,1.1,1.2,-1.0,1.1",   # fuera de orden + low negativo
    ]
    p = tmp_path / "bad.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_read_head_and_tail(good_csv):
    full = pd.read_csv(good_csv, dtype=str)
    head = ct.read_head(good_csv, 5)
    tail = ct.read_tail(good_csv, 5, block_size=64)  # bloque pequeño: varias lecturas
    pd.testing.assert_frame_equal(head, full.head(5).reset_index(drop=True))
    pd.testing.assert_frame_equal(tail, full.tail(5).reset_index(drop=True))


def test_read_tail_larger_than_file(bad_csv):
    tail = ct.read_tail(bad_csv, 100)
    assert len(tail) == 5
    assert list(tail.columns) == list(ct.REQUIRED_COLUMNS)


def test_read_tail_without_trailing_newline(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("datetime,open,high,low,close\n01/01/2024 00:00,1,1,1,1\n01/01/2024 00:15,2,2,2,2",
                 encoding="utf-8")
    assert ct.read_tail(p, 1)["open"].tolist() == ["2"]
    assert ct.count_rows(p) == 2


def test_count_rows(good_csv):
    assert ct.count_rows(good_csv) == 300


def test_check_rows_good(good_csv):
    checks = ct.check_rows(ct.read_head(good_csv, 5))
    assert len(checks) == 5
    assert all(c.ok for c in checks)
    assert [c.row for c in checks] == [1, 2, 3, 4, 5]


def test_check_rows_detects_issues(bad_csv):
    checks = ct.check_rows(ct.read_head(bad_csv, 5))
    assert checks[0].ok
    assert any("no cumple" in i for i in checks[1].issues)
    assert "high < max(open, close)" in checks[2].issues
    assert any("no numérico" in i for i in checks[3].issues)
    assert any("no creciente" in i for i in checks[3].issues)
    assert any("no positivo" in i for i in checks[4].issues)
    assert any("no creciente" in i for i in checks[4].issues)


def test_check_rows_missing_column(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("datetime,open,high,close\n01/01/2024 00:00,1,1,1\n", encoding="utf-8")
    (check,) = ct.check_rows(ct.read_head(p))
    assert "falta columna 'low'" in check.issues


def test_csv_info(good_csv):
    info = ct.csv_info(good_csv)
    assert info.n_rows == 300
    assert info.columns == list(ct.REQUIRED_COLUMNS)
    assert info.first_timestamp == "01/01/2024 00:00"
    assert info.missing_columns == [] and info.extra_columns == []


def test_scan_good(good_csv):
    rep = ct.scan_csv(good_csv, chunksize=64)  # varios chunks
    assert rep.ok
    assert rep.n_rows == 300
    assert rep.inferred_frequency_seconds == 900.0
    assert rep.n_gaps == 0


def test_scan_bad(bad_csv):
    rep = ct.scan_csv(bad_csv, chunksize=2)
    assert not rep.ok
    assert rep.n_rows == 5
    assert rep.n_bad_datetime == 1
    assert rep.nan_counts["close"] == 1
    assert rep.n_non_positive == 1
    assert rep.n_ohlc_incoherent == 1  # high < close (low negativo cuenta como <= 0)
    assert rep.examples["OHLC incoherente"] == [3]
    assert rep.n_duplicate_timestamps == 1  # 00:15 de la fila 2 es inválido
    assert rep.n_out_of_order == 1
    assert rep.examples["datetime inválido"] == [2]


def test_scan_detects_gaps(tmp_path):
    df = make_ohlc(n=100)
    df = df.drop(df.index[40:50])
    rep = ct.scan_csv(_write_csv(df, tmp_path / "g.csv"))
    assert rep.ok  # un gap no es un error de integridad
    assert rep.n_gaps == 1
    assert rep.max_gap_seconds == 11 * 900


def test_scan_missing_columns_raises(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("datetime,close\n01/01/2024 00:00,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="faltan columnas"):
        ct.scan_csv(p)


def test_overlap(tmp_path):
    a = _write_csv(make_ohlc(n=100, start="2024-01-01"), tmp_path / "a.csv")
    b = _write_csv(make_ohlc(n=100, start="2024-01-01 12:00"), tmp_path / "b.csv")
    rep = ct.overlap([a, b])
    assert rep.common_start == pd.Timestamp("2024-01-01 12:00")
    assert rep.common_end == pd.Timestamp("2024-01-01 00:00") + pd.Timedelta(minutes=15 * 99)


def test_overlap_disjoint(tmp_path):
    a = _write_csv(make_ohlc(n=10, start="2024-01-01"), tmp_path / "a.csv")
    b = _write_csv(make_ohlc(n=10, start="2024-02-01"), tmp_path / "b.csv")
    assert not ct.overlap([a, b]).has_overlap


def test_cli_exit_codes(good_csv, bad_csv, capsys):
    assert ct.main(["head", str(good_csv)]) == 0
    assert ct.main(["head", str(bad_csv)]) == 1
    assert ct.main(["tail", str(good_csv), "-n", "3"]) == 0
    assert ct.main(["scan", str(bad_csv)]) == 1
    assert ct.main(["info", str(good_csv)]) == 0
    out = capsys.readouterr().out
    assert "filas OK" in out and "CON PROBLEMAS" in out
