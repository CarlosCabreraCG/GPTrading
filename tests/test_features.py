"""
Tests de features.

El test más importante es el de causalidad: una feature calculada hasta t
NO debe cambiar si añadimos datos después de t.

Data-driven: los tests por feature se parametrizan sobre
`available_features(columnas de make_full_ohlc())`, así que cualquier feature
nueva del registro (OHLC, cross-asset o de contexto) se testea sin tocar
este archivo.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from features import indicators as ind
from features.engine import compute_features, drop_warmup_rows, resolve_features
from features.registry import (
    CROSS_ASSETS,
    FEATURES,
    available_features,
    feature_names,
    ohlc_feature_names,
)
from tests.conftest import extra_required_columns, make_full_ohlc

#: Features calculables con el fixture completo (se usan para parametrizar).
FULL_FEATURES = available_features(make_full_ohlc().columns)


# ---------------------------------------------------------------------------
# Cobertura del fixture
# ---------------------------------------------------------------------------

def test_full_fixture_covers_all_registered_features():
    """El fixture completo debe permitir calcular TODAS las features del registro."""
    assert FULL_FEATURES == feature_names()


# ---------------------------------------------------------------------------
# Tests de causalidad (NO look-ahead), una instancia por feature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", FULL_FEATURES)
def test_feature_is_causal(ohlc_full, name):
    """
    Calcular la feature en la serie completa vs en una truncada debe dar
    exactamente el mismo resultado en el tramo común.

    Si una feature usara datos futuros, los valores del tramo común cambiarían.
    """
    df = ohlc_full
    split = 300  # truncamos aquí

    full_vals = compute_features(df, names=[name])[name].iloc[:split]
    trunc_vals = compute_features(df.iloc[:split], names=[name])[name]

    # NaN en el mismo sitio, valores iguales donde no hay NaN
    pd.testing.assert_series_equal(
        full_vals,
        trunc_vals,
        check_names=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
        obj=f"Feature '{name}' no es causal (look-ahead bias)",
    )


def test_atr_causal(ohlc_only):
    df = ohlc_only
    split = 200
    a_full = ind.atr(df["high"], df["low"], df["close"], 14)
    a_trunc = ind.atr(
        df["high"].iloc[:split], df["low"].iloc[:split], df["close"].iloc[:split], 14
    )
    pd.testing.assert_series_equal(
        a_full.iloc[:split], a_trunc, check_names=False, rtol=1e-12
    )


def test_rsi_causal(ohlc_only):
    df = ohlc_only
    split = 200
    r_full = ind.rsi(df["close"], 14)
    r_trunc = ind.rsi(df["close"].iloc[:split], 14)
    pd.testing.assert_series_equal(
        r_full.iloc[:split], r_trunc, check_names=False, rtol=1e-12
    )


def test_adx_causal(ohlc_only):
    df = ohlc_only
    split = 200
    adx_full = ind.adx(df["high"], df["low"], df["close"], 14)
    adx_trunc = ind.adx(
        df["high"].iloc[:split], df["low"].iloc[:split], df["close"].iloc[:split], 14
    )
    pd.testing.assert_series_equal(
        adx_full.iloc[:split], adx_trunc, check_names=False, rtol=1e-10, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Tests de correctitud
# ---------------------------------------------------------------------------

def test_sma_matches_manual():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ind.sma(s, 3)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(3.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_rsi_bounds(ohlc_only):
    r = ind.rsi(ohlc_only["close"], 14).dropna()
    assert (r >= 0).all()
    assert (r <= 100).all()


def test_adx_bounds(ohlc_only):
    a = ind.adx(
        ohlc_only["high"], ohlc_only["low"], ohlc_only["close"], 14
    ).dropna()
    assert (a >= 0).all()
    assert (a <= 100).all()


def test_bb_position_within_reasonable_range(ohlc_only):
    """bb_position no está limitada a [-1, 1] por construcción,
    pero en datos sintéticos razonables debería estar en un rango acotado."""
    feats = compute_features(ohlc_only)
    bb = feats["bb_position_20"].dropna()
    # Sin outliers extremos
    assert bb.abs().max() < 5.0


@pytest.mark.parametrize("name", FULL_FEATURES)
def test_feature_is_not_constant(ohlc_full, name):
    """Ninguna feature debe ser constante; si lo es, GP no puede usarla."""
    vals = compute_features(ohlc_full, names=[name])[name].dropna()
    assert vals.nunique() > 10, f"Feature '{name}' casi constante"


@pytest.mark.parametrize("name", FULL_FEATURES)
def test_feature_has_no_infinities(ohlc_full, name):
    """Ninguna feature debe contener ±inf; solo valores finitos o NaN."""
    col = compute_features(ohlc_full, names=[name])[name]
    finite_or_nan = np.isfinite(col) | col.isna()
    assert finite_or_nan.all(), f"Feature '{name}' contiene infinitos"


def test_features_aligned_with_input(ohlc_full):
    feats = compute_features(ohlc_full)
    assert len(feats) == len(ohlc_full)
    assert (feats.index == ohlc_full.index).all()


def test_drop_warmup_rows(ohlc_full):
    feats = compute_features(ohlc_full)
    df_aligned, feats_aligned = drop_warmup_rows(ohlc_full, feats)
    assert len(df_aligned) == len(feats_aligned)
    assert feats_aligned.notna().all().all()
    # Debe haber eliminado al menos ~20 filas por warmup
    assert len(df_aligned) < len(ohlc_full)


# ---------------------------------------------------------------------------
# Dependencias de columnas (requires) y disponibilidad
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", feature_names())
def test_available_features_respects_requires(name):
    """Una feature está disponible con sus `requires` y deja de estarlo si falta una."""
    req = FEATURES[name].requires
    assert name in available_features(req)
    for col in req:
        assert name not in available_features(req - {col})


@pytest.mark.parametrize("sym", CROSS_ASSETS)
def test_cross_asset_generates_features(sym):
    col = f"{sym}_close"
    for name in (f"{sym}_ret_1", f"spread_ret_{sym}"):
        assert name in FEATURES, f"Falta {name} para {sym}"
        assert col in FEATURES[name].requires


def test_ohlc_only_excludes_cross_without_error(ohlc_only, caplog):
    """Sin columnas cross-asset no hay KeyError: se excluyen con WARNING."""
    with caplog.at_level(logging.WARNING, logger="features.engine"):
        feats = compute_features(ohlc_only)
    assert list(feats.columns) == ohlc_feature_names()
    if extra_required_columns():
        assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_full_data_emits_no_warning(ohlc_full, caplog):
    with caplog.at_level(logging.WARNING, logger="features.engine"):
        compute_features(ohlc_full)
    assert not caplog.records


def test_strict_raises_when_columns_missing(ohlc_only):
    if not extra_required_columns():
        pytest.skip("El registro no tiene features que requieran columnas extra")
    with pytest.raises(ValueError, match="columnas ausentes"):
        compute_features(ohlc_only, strict=True)
    with pytest.raises(ValueError, match="columnas ausentes"):
        resolve_features(ohlc_only.columns, strict=True)


def test_explicit_unavailable_feature_raises(ohlc_only):
    missing = [n for n in feature_names() if n not in ohlc_feature_names()]
    if not missing:
        pytest.skip("Todas las features son OHLC")
    with pytest.raises(ValueError, match="columnas ausentes"):
        compute_features(ohlc_only, names=[missing[0]])


def test_explicit_unknown_feature_raises(ohlc_only):
    with pytest.raises(ValueError, match="no registradas"):
        compute_features(ohlc_only, names=["no_existe"])
