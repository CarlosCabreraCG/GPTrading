"""Tests de data.merge_cross sobre CSV sintéticos."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from config.settings import infer_symbol_from_path
from data.cleaner import load_and_clean
from data.merge_cross import (
    cross_column,
    main,
    merge_cross,
    merge_cross_to_csv,
    merged_filename,
)
from features.engine import compute_features
from features.registry import CROSS_ASSETS
from tests.conftest import make_ohlc


def _write(df: pd.DataFrame, path: Path) -> Path:
    out = df.copy()
    out.index = out.index.tz_localize(None).strftime("%d/%m/%Y %H:%M")
    out.index.name = "datetime"
    out.to_csv(path)
    return path


@pytest.fixture
def files(tmp_path) -> dict[str, Path]:
    """Base XAUUSD + XAGUSD (empieza más tarde) + DXY (le faltan velas)."""
    base = make_ohlc(n=200, seed=1, start_price=2000.0)
    xag = make_ohlc(n=200, seed=2, start_price=25.0, start="2024-01-01 02:00")
    dxy = make_ohlc(n=200, seed=3, start_price=100.0)
    dxy = dxy.drop(dxy.index[50:60])
    return {
        "base": _write(base, tmp_path / "xauusd_15m_5y.csv"),
        "xag": _write(xag, tmp_path / "xagusd_15m_5y.csv"),
        "dxy": _write(dxy, tmp_path / "dxy_15m_5y.csv"),
    }


def test_cross_column_and_filename():
    assert cross_column("XAGUSD") == "xagusd_close"
    name = merged_filename("data_files/xauusd_15m_5y.csv", ["XAGUSD", "DXY"])
    assert name == "xauusd_15m_5y_x_xagusd_dxy.csv"
    assert infer_symbol_from_path(name) == "XAUUSD"


def test_merge_columns_and_inner_alignment(files):
    df, rep = merge_cross(files["base"], [files["xag"], files["dxy"]])
    assert list(df.columns) == ["open", "high", "low", "close", "xagusd_close", "dxy_close"]
    # XAG empieza 8 velas después; DXY pierde 10 velas dentro del rango común
    assert len(df) == 200 - 8 - 10
    assert df.index[0] == pd.Timestamp("2024-01-01 02:00")
    assert df.notna().all().all()
    assert rep.n_rows_base == 200 and rep.n_rows_output == len(df)
    assert [s.n_rows_after for s in rep.steps] == [192, 182]


def test_values_aligned_by_timestamp(files):
    df, _ = merge_cross(files["base"], [files["xag"]])
    xag, _ = load_and_clean(files["xag"])
    base, _ = load_and_clean(files["base"])
    pd.testing.assert_series_equal(
        df["xagusd_close"], xag["close"].loc[df.index], check_names=False)
    pd.testing.assert_frame_equal(df[["open", "high", "low", "close"]], base.loc[df.index])


def test_duplicates_in_cross_do_not_multiply_rows(tmp_path, files):
    raw = pd.read_csv(files["xag"])
    pd.concat([raw, raw.iloc[[10, 20]]]).to_csv(tmp_path / "xagusd_dup.csv", index=False)
    df, _ = merge_cross(files["base"], [tmp_path / "xagusd_dup.csv"])
    assert df.index.is_unique
    assert len(df) == 192


def test_frequency_mismatch_raises(tmp_path, files):
    h1 = _write(make_ohlc(n=50, freq="1h"), tmp_path / "dxy_1h_5y.csv")
    with pytest.raises(ValueError, match="Frecuencia distinta"):
        merge_cross(files["base"], [h1])


def test_repeated_symbol_raises(files):
    with pytest.raises(ValueError, match="repetido"):
        merge_cross(files["base"], [files["xag"], files["xag"]])
    with pytest.raises(ValueError, match="repetido"):
        merge_cross(files["base"], [files["base"]])


def test_non_canonical_name_raises(tmp_path, files):
    bad = _write(make_ohlc(n=50), tmp_path / "lightcmdusd_15m_5y.csv")
    with pytest.raises(ValueError, match="SUPPORTED_SYMBOLS"):
        merge_cross(files["base"], [bad])


def test_no_overlap_raises(tmp_path, files):
    late = _write(make_ohlc(n=50, start="2025-01-01"), tmp_path / "dxy_15m_5y.csv")
    with pytest.raises(ValueError, match="no deja filas"):
        merge_cross(files["base"], [late])


def test_warns_when_symbol_has_no_features(tmp_path, files, caplog):
    assert "brent" not in CROSS_ASSETS
    brent = _write(make_ohlc(n=200, seed=9), tmp_path / "brent_15m_5y.csv")
    with caplog.at_level(logging.WARNING, logger="data.merge_cross"):
        _, rep = merge_cross(files["base"], [brent])
    assert not rep.steps[0].has_features
    assert "CROSS_ASSETS" in caplog.text


def test_roundtrip_csv_generates_cross_features(files):
    """El CSV generado se carga con load_and_clean y activa las features cross."""
    path, rep = merge_cross_to_csv(files["base"], [files["xag"], files["dxy"]])
    assert path.name == "xauusd_15m_5y_x_xagusd_dxy.csv"
    df, _ = load_and_clean(path)
    assert len(df) == rep.n_rows_output
    feats = compute_features(df)
    for sym in ("xagusd", "dxy"):
        assert f"{sym}_ret_1" in feats.columns
        assert f"spread_ret_{sym}" in feats.columns


def test_refuses_overwrite(files):
    merge_cross_to_csv(files["base"], [files["xag"]])
    with pytest.raises(FileExistsError):
        merge_cross_to_csv(files["base"], [files["xag"]])
    merge_cross_to_csv(files["base"], [files["xag"]], overwrite=True)


def test_cli(files, tmp_path, capsys):
    out = tmp_path / "out" / "merged.csv"
    assert main(["--base", str(files["base"]), "--cross", str(files["xag"]),
                 str(files["dxy"]), "--output", str(out)]) == 0
    assert out.exists()
    assert "xagusd_close" in capsys.readouterr().out
