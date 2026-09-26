"""
Carga del CSV crudo. No limpia, no valida, no transforma.
Solo lee y devuelve un DataFrame con tipos correctos.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import SYMBOL, INTERVAL, DICT_TEMP, DEFAULT_CSV

def check_format(df):
    if "datetime" not in df.columns:
        raise ValueError(
            f"El CSV no tiene columna 'datetime'. Columnas: {list(df.columns)}"
        )
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        format="%d/%m/%Y %H:%M",
        utc=False,
    )

    # Asegurar tipos numéricos en OHLCV
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def download_yf():

    data = yf.download(tickers=SYMBOL, period=DICT_TEMP[INTERVAL], interval=INTERVAL)

    data = data.reset_index()

    data['Datetime'] = pd.to_datetime(data['Datetime']).dt.tz_localize(None)
    data['Datetime'] = data['Datetime'].dt.strftime('%d/%m/%Y %H:%M')

    df_formatted = data[['Datetime', 'Open', 'High', 'Low', 'Close']].copy()
    df_formatted.columns = ['datetime', 'open', 'high', 'low', 'close']

    return df_formatted

def load_raw_csv(path: Path | str | None = None) -> pd.DataFrame:
    """
    Carga el CSV de velas EURUSD.

    Devuelve un DataFrame con columnas:
        datetime (datetime64[ns, tz]), open, high, low, close

    No ordena, no elimina duplicados, no rellena NaNs. Eso es tarea del cleaner.
    """
    csv_path = Path(path) if path is not None else DEFAULT_CSV

    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    return check_format(df)

def download_load(save: bool = False, path: Path | str | None = None):
    df = download_yf()
    df = check_format(df)
    
    if save: df.to_csv(DEFAULT_CSV, index=False)
    return df
