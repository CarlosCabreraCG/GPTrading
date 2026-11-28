"""
Normalización de CSV externos al formato del repositorio.

Los CSV llegan de otro repositorio (descarga) con nombres y formatos que no
siempre encajan aquí. Este módulo los lleva al formato canónico:

- Nombre:   `{symbol}_{timeframe}_{period}.csv` en minúsculas, con `symbol`
            en SUPPORTED_SYMBOLS (p. ej. `lightcmdusd_15m_5y.csv` -> `wti_15m_5y.csv`).
            Así `infer_symbol_from_path` lo reconoce sin cambios.
- Columnas: `datetime,open,high,low,close`, en ese orden.
- Fecha:    CSV_DATETIME_FORMAT (dd/mm/YYYY HH:MM), sin zona horaria.

NO ordena, NO elimina duplicados, NO rellena gaps: eso es tarea del cleaner.
Solo cambia nombre, orden de columnas y formato de fecha.

CLI:
    python -m data.normalize data_files/*.csv --dry-run
    python -m data.normalize data_files/lightcmdusd_15m_5y.csv --replace
    python -m data.normalize data_files/dyx_15m_5y.csv --symbol DXY --replace
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from config.settings import CSV_DATETIME_FORMAT, SUPPORTED_SYMBOLS

logger = logging.getLogger(__name__)

#: Nombres de instrumento usados por la fuente de descarga (Dukascopy)
#: -> símbolo canónico del repositorio. Claves en minúsculas.
SOURCE_SYMBOL_ALIASES: dict[str, str] = {
    "lightcmdusd": "WTI",
    "brentcmdusd": "BRENT",
    "usa500idxusd": "USA500",
    "usatechidxusd": "USATECH",
    "dollaridxusd": "DXY",
}

#: Alias de nombres de columna -> nombre canónico.
COLUMN_ALIASES: dict[str, str] = {
    "datetime": "datetime", "date": "datetime", "time": "datetime",
    "timestamp": "datetime", "gmt time": "datetime", "local time": "datetime",
    "open": "open", "high": "high", "low": "low", "close": "close",
}

CANONICAL_COLUMNS: tuple[str, ...] = ("datetime", "open", "high", "low", "close")

#: Formatos de fecha aceptados en la entrada, por orden de preferencia.
#: Se elige el primero que parsea TODAS las fechas no vacías (sin adivinar
#: día/mes fila a fila, para no mezclar convenciones).
INPUT_DATETIME_FORMATS: tuple[str, ...] = (
    CSV_DATETIME_FORMAT,
    "%d/%m/%Y %H:%M:%S",
    "%d.%m.%Y %H:%M:%S.%f",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "ISO8601",
)

_TF_RE = re.compile(r"^(?:(\d+)([mhdw])|([mhdw])(\d+))$")
_PERIOD_RE = re.compile(r"^\d+(?:d|w|mo|m|y)$")


# ---------------------------------------------------------------------------
# Nombres de archivo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedName:
    raw_symbol: str
    timeframe: str | None
    period: str | None


def normalize_timeframe(token: str) -> str | None:
    """'15m', 'm15', 'M15' -> '15m'; 'H4' -> '4h'. None si no es un timeframe."""
    m = _TF_RE.match(token.lower())
    if not m:
        return None
    if m.group(1):
        return f"{int(m.group(1))}{m.group(2)}"
    return f"{int(m.group(4))}{m.group(3)}"


def parse_filename(path: Path | str) -> ParsedName:
    """
    Separa `{symbol}_{timeframe}_{period}` del nombre de archivo.

    El símbolo es el primer bloque; el timeframe es el primer bloque posterior
    que parezca un timeframe; el periodo, el siguiente que parezca un periodo.
    """
    parts = [p for p in re.split(r"[_\-\s]+", Path(path).stem) if p]
    if not parts:
        raise ValueError(f"Nombre de archivo vacío: {path}")
    tf: str | None = None
    period: str | None = None
    for tok in parts[1:]:
        if tf is None and normalize_timeframe(tok):
            tf = normalize_timeframe(tok)
        elif tf is not None and period is None and _PERIOD_RE.match(tok.lower()):
            period = tok.lower()
    return ParsedName(raw_symbol=parts[0], timeframe=tf, period=period)


def resolve_symbol(raw: str) -> str:
    """Símbolo canónico (SUPPORTED_SYMBOLS) a partir del nombre de la fuente."""
    key = raw.lower()
    sym = SOURCE_SYMBOL_ALIASES.get(key, key.upper())
    if sym not in SUPPORTED_SYMBOLS:
        raise ValueError(
            f"No se reconoce el símbolo '{raw}'. Soportados: {SUPPORTED_SYMBOLS}. "
            f"Pásalo explícito (--symbol) o añádelo a SOURCE_SYMBOL_ALIASES."
        )
    return sym


def canonical_filename(symbol: str, timeframe: str, period: str | None = None) -> str:
    """`{symbol}_{timeframe}_{period}.csv` en minúsculas."""
    if symbol.upper() not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Símbolo '{symbol}' no está en SUPPORTED_SYMBOLS")
    parts = [symbol.lower(), timeframe.lower()] + ([period.lower()] if period else [])
    return "_".join(parts) + ".csv"


def _timeframe_from_seconds(seconds: float) -> str:
    s = int(round(seconds))
    for unit, size in (("w", 604800), ("d", 86400), ("h", 3600), ("m", 60)):
        if s % size == 0:
            return f"{s // size}{unit}"
    raise ValueError(f"No se puede expresar la frecuencia {seconds}s como timeframe")


# ---------------------------------------------------------------------------
# Contenido
# ---------------------------------------------------------------------------

def parse_datetimes(values: pd.Series) -> tuple[pd.Series, str]:
    """
    Parsea fechas probando INPUT_DATETIME_FORMATS. Devuelve (serie, formato).

    Fechas con zona horaria se convierten a UTC y se dejan sin tz.
    """
    s = values.astype(str).str.strip()
    non_empty = s[s != ""]
    for fmt in INPUT_DATETIME_FORMATS:
        try:
            parsed = pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
        if parsed.loc[non_empty.index].isna().any():
            continue
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
        return parsed, fmt
    raise ValueError(
        f"Ningún formato de fecha conocido parsea todas las filas. "
        f"Ejemplos: {non_empty.head(3).tolist()}"
    )


def normalize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """
    Devuelve (df canónico, columnas descartadas, formato de fecha detectado).

    Renombra columnas por alias, reordena a CANONICAL_COLUMNS, convierte a
    numérico y parsea las fechas. Columnas extra se descartan (y se informan).
    """
    rename: dict[str, str] = {}
    for col in df.columns:
        canon = COLUMN_ALIASES.get(str(col).strip().lower())
        if canon is not None:
            if canon in rename.values():
                raise ValueError(f"Columna '{canon}' duplicada (alias de '{col}')")
            rename[col] = canon
    out = df.rename(columns=rename)
    missing = [c for c in CANONICAL_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Faltan columnas {missing}. Columnas: {list(df.columns)}")
    dropped = [c for c in df.columns if c not in rename]

    out = out[list(CANONICAL_COLUMNS)].copy()
    out["datetime"], fmt = parse_datetimes(out["datetime"])
    for c in CANONICAL_COLUMNS[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out, dropped, fmt


def write_csv(df: pd.DataFrame, path: Path | str) -> None:
    """
    Escribe un DataFrame con columna `datetime` en el formato del repositorio.
    Escritura atómica (archivo temporal + replace).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, date_format=CSV_DATETIME_FORMAT)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Archivo completo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizeResult:
    source: Path
    target: Path
    symbol: str
    timeframe: str
    period: str | None
    n_rows: int
    input_datetime_format: str
    dropped_columns: tuple[str, ...]
    renamed: bool
    source_removed: bool
    written: bool

    def summary(self) -> str:
        action = "escrito" if self.written else "dry-run"
        extra = f", descartadas {list(self.dropped_columns)}" if self.dropped_columns else ""
        rm = " (origen eliminado)" if self.source_removed else ""
        return (
            f"{self.source.name} -> {self.target.name} [{action}{rm}] "
            f"{self.symbol} {self.timeframe} {self.period or ''} | {self.n_rows} filas | "
            f"fecha entrada '{self.input_datetime_format}'{extra}"
        )


def normalize_csv(
    source: Path | str,
    symbol: str | None = None,
    timeframe: str | None = None,
    period: str | None = None,
    dest_dir: Path | str | None = None,
    replace: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> NormalizeResult:
    """
    Normaliza un CSV externo al formato y nombre canónicos.

    symbol / timeframe / period: si no se pasan se infieren del nombre del
        archivo (timeframe, en su defecto, de la frecuencia de los datos).
    dest_dir: carpeta destino (por defecto la del origen).
    replace: si el nombre cambia, elimina el origen tras escribir y verificar.
    overwrite: permite sobrescribir un destino distinto del origen ya existente.
    dry_run: no escribe nada; solo devuelve qué haría.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    parsed = parse_filename(source)
    sym = symbol.upper() if symbol else resolve_symbol(parsed.raw_symbol)
    if sym not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Símbolo '{sym}' no está en SUPPORTED_SYMBOLS")

    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    df, dropped, fmt = normalize_frame(raw)

    tf = normalize_timeframe(timeframe) if timeframe else parsed.timeframe
    if tf is None:
        deltas = df["datetime"].sort_values().diff().dropna().dt.total_seconds()
        if deltas.empty:
            raise ValueError(f"No se puede inferir el timeframe de {source}")
        tf = _timeframe_from_seconds(float(deltas.median()))
    per = period.lower() if period else parsed.period

    target = Path(dest_dir) if dest_dir is not None else source.parent
    target = target / canonical_filename(sym, tf, per)
    same_file = target.resolve() == source.resolve()
    if target.exists() and not same_file and not overwrite:
        raise FileExistsError(f"El destino ya existe: {target} (usa overwrite=True)")

    removed = False
    if not dry_run:
        write_csv(df, target)
        written_rows = sum(1 for _ in target.open("rb")) - 1
        if written_rows != len(df):
            raise RuntimeError(f"Verificación fallida: {written_rows} != {len(df)} filas")
        if replace and not same_file:
            source.unlink()
            removed = True
    if dropped:
        logger.warning("%s: columnas descartadas %s", source.name, dropped)

    return NormalizeResult(
        source=source, target=target, symbol=sym, timeframe=tf, period=per,
        n_rows=len(df), input_datetime_format=fmt, dropped_columns=tuple(dropped),
        renamed=not same_file, source_removed=removed, written=not dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normaliza CSV externos al formato del repo.")
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--symbol", help="Símbolo explícito (solo con un archivo).")
    parser.add_argument("--timeframe", help="Timeframe explícito, p. ej. 15m, 4h.")
    parser.add_argument("--period", help="Periodo explícito, p. ej. 5y, 60d.")
    parser.add_argument("--dest-dir", default=None)
    parser.add_argument("--replace", action="store_true",
                        help="Eliminar el archivo origen si el nombre cambia.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.symbol and len(args.csvs) > 1:
        parser.error("--symbol solo puede usarse con un único archivo")

    rc = 0
    for p in args.csvs:
        try:
            res = normalize_csv(
                p, symbol=args.symbol, timeframe=args.timeframe, period=args.period,
                dest_dir=args.dest_dir, replace=args.replace,
                overwrite=args.overwrite, dry_run=args.dry_run,
            )
            print(res.summary())
        except (ValueError, FileExistsError, FileNotFoundError) as e:
            print(f"{p}: ERROR {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
