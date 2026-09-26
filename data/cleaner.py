"""
Limpieza y validación de datos. Esta es la única capa que modifica el DataFrame
crudo. Devuelve un DataFrame limpio + un DataQualityReport.

Reglas:
- Ordenar por datetime ascendente.
- Eliminar duplicados exactos de timestamp (conservar el primero).
- Opcionalmente eliminar filas con NaN en OHLC.
- Detectar gaps temporales.
- NO rellenar gaps automáticamente (eso puede introducir look-ahead).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import DEFAULT_DATA_VALIDATION, DataValidation
from data.models import DataQualityReport


def _infer_frequency_seconds(ts: pd.Series) -> float | None:
    """Infiere la frecuencia modal (en segundos) entre timestamps consecutivos."""
    if len(ts) < 3:
        return None
    deltas = ts.diff().dropna().dt.total_seconds()
    # Moda robusta: mediana de los deltas (evita que un gap grande la distorsione)
    return float(deltas.median())


def _detect_gaps(
    ts: pd.Series,
    freq_seconds: float | None,
    max_gap_multiplier: float,
) -> tuple[int, float | None]:
    """
    Devuelve (n_gaps, max_gap_seconds).
    Un gap se cuenta cuando el delta entre timestamps consecutivos supera
    freq_seconds * max_gap_multiplier.
    """
    if freq_seconds is None or len(ts) < 2:
        return 0, None

    deltas = ts.diff().dropna().dt.total_seconds()
    threshold = freq_seconds * max_gap_multiplier
    gaps = deltas[deltas > threshold]

    n_gaps = int(len(gaps))
    max_gap = float(deltas.max()) if len(deltas) > 0 else None
    return n_gaps, max_gap


def clean(
    df: pd.DataFrame,
    validation: DataValidation = DEFAULT_DATA_VALIDATION,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Limpia el DataFrame crudo.

    Devuelve (df_limpio, reporte). El df limpio está ordenado, sin duplicados
    de timestamp, y opcionalmente sin NaNs en OHLC.
    """
    n_input = len(df)
    df = df.copy()

    # 1. Verificar columnas requeridas
    missing = [c for c in validation.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    # 2. Ordenar por timestamp
    df = df.sort_values("datetime", kind="mergesort").reset_index(drop=True)

    # 3. Eliminar duplicados de timestamp (conservar primero)
    before = len(df)
    df = df.drop_duplicates(subset="datetime", keep="first").reset_index(drop=True)
    n_duplicates = before - len(df)

    # 4. Eliminar NaNs en OHLC si se pide
    n_nans_removed = 0
    if validation.drop_ohlc_nans:
        ohlc_cols = ["open", "high", "low", "close"]
        before = len(df)
        df = df.dropna(subset=ohlc_cols).reset_index(drop=True)
        n_nans_removed = before - len(df)

    # 5. Inferir frecuencia y detectar gaps
    freq_seconds = _infer_frequency_seconds(df["datetime"])
    n_gaps, max_gap = _detect_gaps(
        df["datetime"], freq_seconds, validation.max_gap_multiplier
    )

    if validation.raise_on_large_gap and n_gaps > 0:
        raise ValueError(
            f"Se detectaron {n_gaps} gaps mayores a "
            f"{validation.max_gap_multiplier}x la frecuencia modal."
        )

    report = DataQualityReport(
        n_rows_input=n_input,
        n_rows_output=len(df),
        n_duplicates_removed=n_duplicates,
        n_nans_removed=n_nans_removed,
        n_gaps_detected=n_gaps,
        max_gap_seconds=max_gap,
        first_timestamp=df["datetime"].iloc[0] if len(df) > 0 else None,
        last_timestamp=df["datetime"].iloc[-1] if len(df) > 0 else None,
        inferred_frequency_seconds=freq_seconds,
    )
    df = df.set_index("datetime", drop=True)
    return df, report


def load_and_clean(
    path: Path | str | None = None,
    validation: DataValidation = DEFAULT_DATA_VALIDATION,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Atajo: cargar + limpiar en un solo paso."""
    from data.loader import load_raw_csv

    raw = load_raw_csv(path)
    return clean(raw, validation)