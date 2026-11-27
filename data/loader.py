"""
Carga del CSV crudo. No limpia, no valida, no transforma.
Solo lee y devuelve un DataFrame con tipos correctos.

Este repositorio no descarga datos: los CSV llegan ya descargados a
`data_files/` desde fuentes externas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import CSV_DATETIME_FORMAT, DEFAULT_CSV


def check_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica columnas mínimas y convierte tipos.

    - `datetime` se parsea con CSV_DATETIME_FORMAT (dd/mm/YYYY HH:MM).
    - open/high/low/close se convierten a numérico (valores inválidos -> NaN).

    Columnas extra (p. ej. `xagusd_close`) se conservan tal cual.
    """
    if "datetime" not in df.columns:
        raise ValueError(
            f"El CSV no tiene columna 'datetime'. Columnas: {list(df.columns)}"
        )
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        format=CSV_DATETIME_FORMAT,
        utc=False,
    )

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_raw_csv(path: Path | str | None = None) -> pd.DataFrame:
    """
    Carga un CSV de velas (cualquier símbolo soportado).

    Devuelve un DataFrame con columnas:
        datetime (datetime64[ns], naive), open, high, low, close
        (+ columnas extra del CSV, p. ej. `{sym}_close` cross-asset)

    No ordena, no elimina duplicados, no rellena NaNs. Eso es tarea del cleaner.
    """
    csv_path = Path(path) if path is not None else DEFAULT_CSV

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    return check_format(df)
