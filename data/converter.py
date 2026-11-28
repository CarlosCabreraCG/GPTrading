"""
Conversión de timeframe (resample OHLC a una frecuencia mayor).

Uso típico: generar un timeframe que la fuente no trae (p. ej. DXY 4h a
partir de DXY 15m) para poder fusionarlo con `data.merge_cross`.

Convención de vela (la misma que la fuente de datos; verificado: XAUUSD,
BRENT y EURUSD 15m -> 4h coinciden al 100% con sus H4 descargados):
- Buckets alineados a medianoche (00:00, 04:00, 08:00, ... para 4h).
- Etiqueta = inicio del bucket, intervalo cerrado a la izquierda
  [t, t + tf). La vela etiquetada t solo contiene datos de ese intervalo,
  así que con señal en t y ejecución al open de t+1 no hay look-ahead.
- open = primero, high = máx, low = mín, close = último.
- Buckets sin datos (fines de semana, horas sin sesión) se eliminan; no se
  rellena nada.

Solo convierte OHLC: para datasets cross-asset, resamplear cada instrumento
por separado y después fusionar con `data.merge_cross`.

CLI:
    python -m data.converter data_files/dxy_15m_5y.csv --timeframe 4h
    # -> data_files/dxy_4h_5y.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from config.settings import infer_symbol_from_path
from data.cleaner import clean
from data.loader import load_raw_csv
from data.normalize import canonical_filename, normalize_timeframe, parse_filename, write_csv

OHLC_AGG: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last"}

_UNIT_SECONDS: dict[str, int] = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
_PANDAS_UNIT: dict[str, str] = {"m": "min", "h": "h", "d": "D"}


def timeframe_seconds(timeframe: str) -> int:
    """'15m' -> 900, '4h' -> 14400."""
    tf = normalize_timeframe(timeframe)
    if tf is None:
        raise ValueError(f"Timeframe no válido: {timeframe}")
    return int(tf[:-1]) * _UNIT_SECONDS[tf[-1]]


def _pandas_rule(timeframe: str) -> str:
    tf = normalize_timeframe(timeframe)
    if tf is None:
        raise ValueError(f"Timeframe no válido: {timeframe}")
    unit = tf[-1]
    if unit == "w":
        raise ValueError("Resample semanal no soportado (alineación de semana ambigua)")
    return f"{int(tf[:-1])}{_PANDAS_UNIT[unit]}"


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resamplea un DataFrame OHLC indexado por datetime (salida del cleaner)
    a `timeframe`. Devuelve solo open/high/low/close, sin buckets vacíos.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("El DataFrame debe estar indexado por datetime (usar cleaner.clean)")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("El índice debe estar ordenado y sin duplicados (usar cleaner.clean)")
    out = (
        df[list(OHLC_AGG)]
        .resample(_pandas_rule(timeframe), label="left", closed="left")
        .agg(OHLC_AGG)
        .dropna(how="all")
    )
    out.index.name = "datetime"
    return out


@dataclass(frozen=True)
class ResampleResult:
    source: Path
    target: Path
    symbol: str
    source_timeframe_seconds: float | None
    target_timeframe: str
    n_rows_input: int
    n_rows_output: int

    def summary(self) -> str:
        return (
            f"{self.source.name} ({self.n_rows_input} velas, {self.source_timeframe_seconds}s) "
            f"-> {self.target.name} ({self.n_rows_output} velas, {self.target_timeframe})"
        )


def resample_csv(
    source: Path | str,
    timeframe: str,
    dest_dir: Path | str | None = None,
    overwrite: bool = False,
) -> ResampleResult:
    """
    Lee un CSV canónico, lo limpia, lo resamplea y lo escribe con nombre
    canónico `{symbol}_{timeframe}_{period}.csv` junto al origen (o en dest_dir).
    """
    source = Path(source)
    symbol = infer_symbol_from_path(source)
    tf = normalize_timeframe(timeframe)
    if tf is None:
        raise ValueError(f"Timeframe no válido: {timeframe}")

    df, report = clean(load_raw_csv(source))
    src_freq = report.inferred_frequency_seconds
    target_seconds = timeframe_seconds(tf)
    if src_freq is not None:
        if target_seconds <= src_freq:
            raise ValueError(
                f"El timeframe destino ({tf}) debe ser mayor que el de origen ({src_freq:.0f}s)"
            )
        if target_seconds % src_freq != 0:
            raise ValueError(
                f"{tf} no es múltiplo del timeframe de origen ({src_freq:.0f}s)"
            )

    out = resample_ohlc(df, tf)
    parsed = parse_filename(source)
    target = (Path(dest_dir) if dest_dir is not None else source.parent) / canonical_filename(
        symbol, tf, parsed.period
    )
    if target.exists() and not overwrite:
        raise FileExistsError(f"El destino ya existe: {target} (usa --overwrite)")
    write_csv(out.reset_index(), target)
    return ResampleResult(source, target, symbol, src_freq, tf, len(df), len(out))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resample OHLC a un timeframe mayor.")
    parser.add_argument("csvs", nargs="+", help="CSV(s) canónicos de origen.")
    parser.add_argument("--timeframe", required=True, help="Timeframe destino, p. ej. 4h, 1h, 1d.")
    parser.add_argument("--dest-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    rc = 0
    for p in args.csvs:
        try:
            print(resample_csv(p, args.timeframe, args.dest_dir, args.overwrite).summary())
        except (ValueError, FileExistsError, FileNotFoundError, TypeError) as e:
            print(f"{p}: ERROR {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
