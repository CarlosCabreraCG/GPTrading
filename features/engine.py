"""
Motor de cálculo de features.

Dado un DataFrame OHLC limpio, calcula todas las features del registry
y devuelve un DataFrame alineado por índice (mismo índice que la entrada).

Reglas:
- No modifica el DataFrame de entrada.
- No elimina filas. Las primeras filas tendrán NaN por min_periods.
- Todas las features son causales.
"""

from __future__ import annotations

import pandas as pd

from features.registry import FEATURES


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todas las features del registry.

    Devuelve un DataFrame con una columna por feature, indexado igual que df.
    """
    out = {}
    for name, spec in FEATURES.items():
        out[name] = spec.func(df)

    features = pd.DataFrame(out, index=df.index)
    return features


def drop_warmup_rows(
    df: pd.DataFrame,
    features: pd.DataFrame,
    min_valid_features: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Elimina las filas iniciales donde las features aún tienen NaN por warmup.

    Estrategia: eliminamos las filas donde CUALQUIER feature es NaN.
    Con features de ventana 20, esto elimina las primeras ~20-30 velas.

    min_valid_features: si se especifica, solo exige que al menos ese número
    de features estén no-NaN. Por defecto exige todas.

    Devuelve (df_alineado, features_alineadas).
    """
    if min_valid_features is None:
        valid_mask = features.notna().all(axis=1)
    else:
        valid_mask = features.notna().sum(axis=1) >= min_valid_features

    return df.loc[valid_mask].copy(), features.loc[valid_mask].copy()