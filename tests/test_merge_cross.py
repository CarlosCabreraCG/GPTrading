"""Tests de data.merge_cross sobre CSV sintéticos."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from config.settings import infer_symbol_from_path
from data.cleaner import load_and_clean
from data.merge_cross import (
    CoverageLimits,
    MergeCoverageError,
    compute_coverage,
    cross_column,
    main,
    merge_cross,
    merge_cross_to_csv,
    merged_filename,
)
from features.engine import compute_features
from tests.conftest import make_ohlc


def _write(df: pd.DataFrame, path: Path) -> Path:
    out = df.copy()
    out.index = out.index.tz_localize(None).strftime("%d/%m/%Y %H:%M")
    out.index.name = "datetime"
    out.to_csv(path)
    return path


@pytest.fixture
def files(tmp_path) -> dict[str, Path]:
    """Base XAUUSD + XAGUSD (empieza 8 velas más tarde) + DXY (5 huecos interiores)."""
    base = make_ohlc(n=200, seed=1, start_price=2000.0)
    xag = make_ohlc(n=200, seed=2, start_price=25.0, start="2024-01-01 02:00")
    dxy = make_ohlc(n=200, seed=3, start_price=100.0)
    dxy = dxy.drop(dxy.index[50:55])
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
    # XAG empieza 8 velas después; DXY pierde 5 velas dentro del rango común
    assert len(df) == 200 - 8 - 5
    assert df.index[0] == pd.Timestamp("2024-01-01 02:00")
    assert df.notna().all().all()
    assert rep.n_rows_base == 200 and rep.n_rows_output == len(df)
    assert [s.cumulative.n_output for s in rep.steps] == [192, 187]
    assert rep.ok and not rep.forced
    cov = rep.coverage
    assert cov.n_edge_dropped == 8 and cov.n_interior_dropped == 5
    assert cov.n_broken == 1  # un único hueco (5 velas seguidas)
    assert cov.interior_drop_pct == pytest.approx(100 * 5 / 192)


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


# ---------------------------------------------------------------------------
# Cobertura
# ---------------------------------------------------------------------------

def _session_cross(path: Path, n: int = 96 * 10) -> Path:
    """Cross que no cotiza de 21:00 a 24:00 (como DXY / BRENT)."""
    df = make_ohlc(n=n, seed=5, start_price=100.0)
    return _write(df[df.index.hour < 21], path)


@pytest.fixture
def long_base(tmp_path) -> Path:
    return _write(make_ohlc(n=96 * 10, seed=1, start_price=2000.0),
                  tmp_path / "xauusd_15m_5y.csv")


def test_compute_coverage_unit():
    base = pd.date_range("2024-01-01", periods=10, freq="15min")
    merged = base[[2, 3, 5, 6, 7]]  # borde: 0,1 y 8,9 fuera; interior: falta 4
    cov = compute_coverage(base, merged, base[2], base[7])
    assert cov.n_edge_dropped == 4
    assert cov.n_interior_dropped == 1
    assert cov.n_broken == 1
    assert cov.total_drop_pct == pytest.approx(50.0)
    assert cov.interior_drop_pct == pytest.approx(100 / 6)
    assert cov.broken_pct == pytest.approx(25.0)


def test_session_gap_cross_is_rejected(tmp_path, long_base):
    dxy = _session_cross(tmp_path / "dxy_15m_5y.csv")
    with pytest.raises(MergeCoverageError, match="No se puede trabajar") as exc:
        merge_cross(long_base, [dxy])
    msg = str(exc.value)
    assert "XAUUSD + DXY" in msg and "--force" in msg


def test_culprit_is_identified_among_several(tmp_path, long_base):
    ok = _write(make_ohlc(n=96 * 10, seed=2, start_price=25.0), tmp_path / "xagusd_15m_5y.csv")
    dxy = _session_cross(tmp_path / "dxy_15m_5y.csv")
    _, rep = merge_cross(long_base, [ok, dxy], check=False)
    assert not rep.ok
    assert rep.culprits() == ["DXY"]
    assert rep.steps[0].pairwise.violations(rep.limits) == []
    # 9 noches x 12 velas interiores; las 3h finales del último día son borde
    assert rep.coverage.n_edge_dropped == 12
    assert rep.coverage.interior_drop_pct == pytest.approx(100 * 9 * 12 / (960 - 12))


def test_force_allows_merge(tmp_path, long_base, caplog):
    dxy = _session_cross(tmp_path / "dxy_15m_5y.csv")
    with caplog.at_level(logging.WARNING, logger="data.merge_cross"):
        df, rep = merge_cross(long_base, [dxy], force=True)
    assert rep.forced and not rep.ok
    assert len(df) == rep.n_rows_output > 0
    assert "FORZADO" in caplog.text
    path, rep2 = merge_cross_to_csv(long_base, [dxy], force=True)
    assert path.exists() and rep2.forced


def test_rejected_merge_writes_nothing(tmp_path, long_base):
    dxy = _session_cross(tmp_path / "dxy_15m_5y.csv")
    with pytest.raises(MergeCoverageError):
        merge_cross_to_csv(long_base, [dxy])
    assert not (tmp_path / "xauusd_15m_5y_x_dxy.csv").exists()


def test_custom_limits(tmp_path, long_base):
    dxy = _session_cross(tmp_path / "dxy_15m_5y.csv")
    lax = CoverageLimits(max_interior_drop_pct=20, max_broken_pct=5, max_total_drop_pct=20)
    _, rep = merge_cross(long_base, [dxy], limits=lax)
    assert rep.ok


def test_edge_trim_only_counts_towards_total(tmp_path, long_base):
    """Un cross que empieza mucho más tarde no crea huecos, pero acorta el histórico."""
    late = make_ohlc(n=96 * 10, seed=3, start_price=100.0).iloc[300:]
    late = _write(late, tmp_path / "wti_15m_5y.csv")
    with pytest.raises(MergeCoverageError, match="pérdida total"):
        merge_cross(long_base, [late])
    _, rep = merge_cross(long_base, [late], check=False)
    assert rep.coverage.interior_drop_pct == 0 and rep.coverage.n_broken == 0
    assert rep.coverage.edge_drop_pct == pytest.approx(100 * 300 / 960)


def test_cli_coverage_modes(tmp_path, long_base, capsys):
    dxy = _session_cross(tmp_path / "dxy_15m_5y.csv")
    base_args = ["--base", str(long_base), "--cross", str(dxy)]
    assert main(base_args) == 2
    assert "RECHAZADO" in capsys.readouterr().out
    assert main(base_args + ["--dry-run"]) == 2
    assert not (tmp_path / "xauusd_15m_5y_x_dxy.csv").exists()
    assert main(base_args + ["--force"]) == 0
    assert "FORZADO" in capsys.readouterr().out
    assert (tmp_path / "xauusd_15m_5y_x_dxy.csv").exists()


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
