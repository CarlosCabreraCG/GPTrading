"""Tests de data.converter (resample OHLC) sobre datos sintéticos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.cleaner import load_and_clean
from data.converter import main, resample_csv, resample_ohlc, timeframe_seconds
from tests.conftest import make_ohlc


def _naive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.tz_localize(None)
    out.index.name = "datetime"
    return out


def _write(df: pd.DataFrame, path: Path) -> Path:
    out = _naive(df)
    out.index = out.index.strftime("%d/%m/%Y %H:%M")
    out.to_csv(path)
    return path


@pytest.mark.parametrize("tf,sec", [("15m", 900), ("m15", 900), ("4h", 14400), ("H4", 14400), ("1d", 86400)])
def test_timeframe_seconds(tf, sec):
    assert timeframe_seconds(tf) == sec


def test_resample_aggregation_and_labels():
    df = _naive(make_ohlc(n=32))  # 8h de 15m desde 00:00
    out = resample_ohlc(df, "4h")
    assert list(out.index) == [pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-01-01 04:00")]
    first = df.iloc[:16]
    assert out.iloc[0]["open"] == first["open"].iloc[0]
    assert out.iloc[0]["high"] == first["high"].max()
    assert out.iloc[0]["low"] == first["low"].min()
    assert out.iloc[0]["close"] == first["close"].iloc[-1]


def test_resample_is_causal():
    """La vela etiquetada t solo usa datos de [t, t+4h): truncar después no la cambia."""
    df = _naive(make_ohlc(n=96))
    full = resample_ohlc(df, "4h")
    for i, t in enumerate(full.index):
        end = t + pd.Timedelta(hours=4) - pd.Timedelta(minutes=15)
        partial = resample_ohlc(df.loc[:end], "4h")
        pd.testing.assert_series_equal(partial.loc[t], full.loc[t])


def test_resample_drops_empty_buckets():
    df = _naive(make_ohlc(n=96))
    df = df[(df.index.hour < 8) | (df.index.hour >= 16)]  # sin sesión 08:00-16:00
    out = resample_ohlc(df, "4h")
    assert [t.hour for t in out.index] == [0, 4, 16, 20]
    assert out.notna().all().all()


def test_resample_requires_clean_index():
    df = _naive(make_ohlc(n=10))
    with pytest.raises(ValueError):
        resample_ohlc(pd.concat([df, df.iloc[[0]]]), "4h")
    with pytest.raises(TypeError):
        resample_ohlc(df.reset_index(), "4h")


def test_resample_csv_writes_canonical_file(tmp_path):
    src = _write(make_ohlc(n=96 * 3), tmp_path / "dxy_15m_5y.csv")
    res = resample_csv(src, "4h")
    assert res.target == tmp_path / "dxy_4h_5y.csv"
    assert res.n_rows_input == 288 and res.n_rows_output == 18
    df, rep = load_and_clean(res.target)
    assert rep.inferred_frequency_seconds == 14400
    assert list(df.columns) == ["open", "high", "low", "close"]


def test_resample_csv_rejects_upsampling_and_non_multiples(tmp_path):
    src = _write(make_ohlc(n=50, freq="1h"), tmp_path / "dxy_1h_5y.csv")
    with pytest.raises(ValueError, match="mayor"):
        resample_csv(src, "15m")
    with pytest.raises(ValueError, match="mayor"):
        resample_csv(src, "1h")
    src2 = _write(make_ohlc(n=50, freq="4h"), tmp_path / "xauusd_4h_5y.csv")
    with pytest.raises(ValueError, match="múltiplo"):
        resample_csv(src2, "6h")


def test_resample_csv_refuses_overwrite(tmp_path):
    src = _write(make_ohlc(n=96), tmp_path / "dxy_15m_5y.csv")
    resample_csv(src, "4h")
    with pytest.raises(FileExistsError):
        resample_csv(src, "4h")
    resample_csv(src, "4h", overwrite=True)


def test_cli(tmp_path, capsys):
    src = _write(make_ohlc(n=96), tmp_path / "dxy_15m_5y.csv")
    assert main([str(src), "--timeframe", "4h"]) == 0
    assert "dxy_4h_5y.csv" in capsys.readouterr().out
    assert main([str(src), "--timeframe", "4h"]) == 1  # ya existe
