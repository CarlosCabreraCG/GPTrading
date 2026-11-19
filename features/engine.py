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

import logging
from typing import Iterable

import pandas as pd

from features.registry import FEATURES, available_features

logger = logging.getLogger(__name__)


def _missing_detail(names: Iterable[str], columns: set[str]) -> dict[str, list[str]]:
    return {n: sorted(FEATURES[n].requires - columns) for n in names}


def resolve_features(columns: Iterable[str], strict: bool = False) -> list[str]:
    """
    Devuelve las features registradas calculables con `columns`.

    Las features excluidas por falta de columnas se registran con WARNING.
    strict: si True, lanza ValueError en lugar de excluirlas.
    """
    cols = set(columns)
    available = available_features(cols)
    missing = [n for n in FEATURES if n not in available]
    if missing:
        detail = _missing_detail(missing, cols)
        if strict:
            raise ValueError(f"Features no calculables por columnas ausentes: {detail}")
        logger.warning("Features excluidas por columnas ausentes: %s", detail)
    return available


def compute_features(
    df: pd.DataFrame,
    names: Iterable[str] | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Calcula features del registry.

    names: features a calcular. Si es None, todas las disponibles para las
        columnas de df (ver `resolve_features`, con su WARNING / strict).
        Si se pasa explícitamente, una feature no registrada o no calculable
        lanza ValueError.

    Devuelve un DataFrame con una columna por feature, indexado igual que df.
    """
    if names is None:
        selected = resolve_features(df.columns, strict=strict)
    else:
        selected = list(names)
        unknown = [n for n in selected if n not in FEATURES]
        if unknown:
            raise ValueError(f"Features no registradas: {unknown}")
        available = set(available_features(df.columns))
        not_available = [n for n in selected if n not in available]
        if not_available:
            detail = _missing_detail(not_available, set(df.columns))
            raise ValueError(
                f"Features pedidas no calculables por columnas ausentes: {detail}"
            )

    out = {}
    for name in selected:
        out[name] = FEATURES[name].func(df)

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