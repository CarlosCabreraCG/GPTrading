"""
Tests de features.

El test más importante es el de causalidad: una feature calculada hasta t
NO debe cambiar si añadimos datos después de t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import indicators as ind
from features.engine import compute_features, drop_warmup_rows
from features.registry import FEATURES, feature_names


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_ohlc() -> pd.DataFrame:
    """
    Genera un DataFrame OHLC sintético con propiedades conocidas.
    Precio sigue un random walk con tendencia suave.
    """
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")

    # Random walk con drift pequeño
    returns = rng.normal(loc=0.00005, scale=0.001, size=n)
    close = 1.10 * np.exp(np.cumsum(returns))

    # Construir OHLC coherente
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0005, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0005, size=n))

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=idx,
    )
    return df


# ---------------------------------------------------------------------------
# Tests de causalidad (NO look-ahead)
# ---------------------------------------------------------------------------

def test_all_features_are_causal(synthetic_ohlc):
    """
    Para cada feature: calcularla en la serie completa vs en una truncada
    debe dar exactamente el mismo resultado en el tramo común.

    Si una feature usara datos futuros, los valores del tramo común cambiarían.
    """
    df = synthetic_ohlc
    split = 300  # truncamos aquí

    df_truncated = df.iloc[:split]

    features_full = compute_features(df)
    features_trunc = compute_features(df_truncated)

    for name in feature_names():
        full_vals = features_full[name].iloc[:split]
        trunc_vals = features_trunc[name]

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


def test_atr_causal(synthetic_ohlc):
    df = synthetic_ohlc
    split = 200
    a_full = ind.atr(df["high"], df["low"], df["close"], 14)
    a_trunc = ind.atr(
        df["high"].iloc[:split], df["low"].iloc[:split], df["close"].iloc[:split], 14
    )
    pd.testing.assert_series_equal(
        a_full.iloc[:split], a_trunc, check_names=False, rtol=1e-12
    )


def test_rsi_causal(synthetic_ohlc):
    df = synthetic_ohlc
    split = 200
    r_full = ind.rsi(df["close"], 14)
    r_trunc = ind.rsi(df["close"].iloc[:split], 14)
    pd.testing.assert_series_equal(
        r_full.iloc[:split], r_trunc, check_names=False, rtol=1e-12
    )


def test_adx_causal(synthetic_ohlc):
    df = synthetic_ohlc
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


def test_rsi_bounds(synthetic_ohlc):
    r = ind.rsi(synthetic_ohlc["close"], 14).dropna()
    assert (r >= 0).all()
    assert (r <= 100).all()


def test_adx_bounds(synthetic_ohlc):
    a = ind.adx(
        synthetic_ohlc["high"], synthetic_ohlc["low"], synthetic_ohlc["close"], 14
    ).dropna()
    assert (a >= 0).all()
    assert (a <= 100).all()


def test_bb_position_within_reasonable_range(synthetic_ohlc):
    """bb_position no está limitada a [-1, 1] por construcción,
    pero en datos sintéticos razonables debería estar en un rango acotado."""
    feats = compute_features(synthetic_ohlc)
    bb = feats["bb_position_20"].dropna()
    # Sin outliers extremos
    assert bb.abs().max() < 5.0


def test_no_feature_is_constant(synthetic_ohlc):
    """Ninguna feature debe ser constante; si lo es, GP no puede usarla."""
    feats = compute_features(synthetic_ohlc).dropna()
    for name in feats.columns:
        assert feats[name].nunique() > 10, f"Feature '{name}' casi constante"


def test_no_infinities(synthetic_ohlc):
    """Ninguna feature debe contener ±inf; solo valores finitos o NaN."""
    feats = compute_features(synthetic_ohlc)
    for name in feats.columns:
        col = feats[name]
        finite_or_nan = np.isfinite(col) | col.isna()
        assert finite_or_nan.all(), f"Feature '{name}' contiene infinitos"


def test_features_aligned_with_input(synthetic_ohlc):
    feats = compute_features(synthetic_ohlc)
    assert len(feats) == len(synthetic_ohlc)
    assert (feats.index == synthetic_ohlc.index).all()


def test_drop_warmup_rows(synthetic_ohlc):
    feats = compute_features(synthetic_ohlc)
    df_aligned, feats_aligned = drop_warmup_rows(synthetic_ohlc, feats)
    assert len(df_aligned) == len(feats_aligned)
    assert feats_aligned.notna().all().all()
    # Debe haber eliminado al menos ~20 filas por warmup
    assert len(df_aligned) < len(synthetic_ohlc)