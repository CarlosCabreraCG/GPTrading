"""
Herramientas para inspeccionar CSV de velas sin cargarlos enteros en pantalla
(ni, cuando es posible, en memoria).

Comandos (CLI):
    python -m data.csv_tools head    <csv> [-n 5]   # primeras filas + chequeo por fila
    python -m data.csv_tools tail    <csv> [-n 5]   # últimas filas (lee desde el final)
    python -m data.csv_tools info    <csv>          # tamaño, columnas, filas, rango
    python -m data.csv_tools scan    <csv>          # integridad completa por bloques
    python -m data.csv_tools overlap <csv> <csv>... # rango temporal común

Solo lectura: ninguna función modifica el archivo.
"""

from __future__ import annotations

import argparse
import io
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from config.settings import CSV_DATETIME_FORMAT
from data.cleaner import _detect_gaps

#: Columnas mínimas de un CSV de velas del proyecto, en orden canónico.
REQUIRED_COLUMNS: tuple[str, ...] = ("datetime", "open", "high", "low", "close")
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

#: Tamaño de bloque para lecturas por chunks.
DEFAULT_CHUNKSIZE: int = 200_000


# ---------------------------------------------------------------------------
# Lectura parcial
# ---------------------------------------------------------------------------

def read_header(path: Path | str) -> list[str]:
    """Nombres de columna (primera línea), sin leer el resto del archivo."""
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        line = f.readline()
    return [c.strip() for c in line.strip().split(",")] if line else []


def read_head(path: Path | str, n: int = 5) -> pd.DataFrame:
    """Primeras `n` filas como texto (sin conversión de tipos)."""
    return pd.read_csv(path, nrows=n, dtype=str, keep_default_na=False)


def read_tail(path: Path | str, n: int = 5, block_size: int = 8192) -> pd.DataFrame:
    """
    Últimas `n` filas como texto. Lee bloques desde el final del archivo,
    por lo que el coste no depende del tamaño del CSV.
    """
    path = Path(path)
    header = read_header(path)
    with path.open("rb") as f:
        f.seek(0, io.SEEK_END)
        end = f.tell()
        pos = end
        data = b""
        # n filas + posible salto final + línea parcial => n + 2 saltos bastan
        while pos > 0 and data.count(b"\n") <= n + 1:
            step = min(block_size, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
    lines = [ln for ln in data.decode("utf-8").splitlines() if ln.strip()]
    if pos == 0 and lines:
        lines = lines[1:]  # quitar la cabecera si llegamos al inicio
    lines = lines[-n:] if n > 0 else []
    if not lines:
        return pd.DataFrame(columns=header, dtype=str)
    return pd.read_csv(
        io.StringIO("\n".join(lines)),
        header=None, names=header, dtype=str, keep_default_na=False,
    )


def count_rows(path: Path | str, block_size: int = 1 << 20) -> int:
    """Número de filas de datos (sin cabecera), contando saltos de línea en binario."""
    n_newlines = 0
    last = b""
    with Path(path).open("rb") as f:
        while chunk := f.read(block_size):
            n_newlines += chunk.count(b"\n")
            last = chunk
    if not last:
        return 0
    lines = n_newlines + (0 if last.endswith(b"\n") else 1)
    return max(lines - 1, 0)


# ---------------------------------------------------------------------------
# Chequeo por fila (head / tail)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RowCheck:
    """Resultado del chequeo de una fila."""
    row: int
    values: dict[str, str]
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def _parse_float(value: str) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _parse_dt(value: str) -> datetime | None:
    try:
        return datetime.strptime(value.strip(), CSV_DATETIME_FORMAT)
    except (AttributeError, ValueError):
        return None


def check_rows(df: pd.DataFrame, first_row: int = 1) -> list[RowCheck]:
    """
    Chequea fila a fila un DataFrame leído como texto (read_head / read_tail).

    Por fila: datetime con CSV_DATETIME_FORMAT, precios numéricos y > 0,
    high >= max(open, close), low <= min(open, close), datetime estrictamente
    creciente respecto a la fila anterior.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    out: list[RowCheck] = []
    prev_dt: datetime | None = None
    for i, (_, rec) in enumerate(df.iterrows()):
        values = {c: str(rec[c]) for c in df.columns}
        issues: list[str] = [f"falta columna '{c}'" for c in missing]

        dt = _parse_dt(values["datetime"]) if "datetime" in values else None
        if "datetime" in values and dt is None:
            issues.append(f"datetime '{values['datetime']}' no cumple {CSV_DATETIME_FORMAT}")
        if dt is not None and prev_dt is not None and dt <= prev_dt:
            issues.append(f"datetime no creciente (anterior {prev_dt:%d/%m/%Y %H:%M})")
        prev_dt = dt if dt is not None else prev_dt

        px: dict[str, float] = {}
        for c in PRICE_COLUMNS:
            if c not in values:
                continue
            x = _parse_float(values[c])
            if x is None:
                issues.append(f"{c} '{values[c]}' no numérico")
            elif x <= 0:
                issues.append(f"{c}={x} no positivo")
            else:
                px[c] = x
        if len(px) == 4:
            if px["high"] < max(px["open"], px["close"]):
                issues.append("high < max(open, close)")
            if px["low"] > min(px["open"], px["close"]):
                issues.append("low > min(open, close)")
            if px["high"] < px["low"]:
                issues.append("high < low")

        out.append(RowCheck(row=first_row + i, values=values, issues=tuple(issues)))
    return out


def format_row_checks(checks: Sequence[RowCheck], columns: Sequence[str]) -> str:
    """Tabla legible de filas + estado."""
    cols = list(columns)
    widths = {c: max(len(c), *(len(ch.values.get(c, "")) for ch in checks)) if checks else len(c)
              for c in cols}
    head = "  ".join(["fila".rjust(6)] + [c.ljust(widths[c]) for c in cols] + ["estado"])
    lines = [head, "-" * len(head)]
    for ch in checks:
        status = "OK" if ch.ok else "; ".join(ch.issues)
        lines.append("  ".join(
            [str(ch.row).rjust(6)] + [ch.values.get(c, "").ljust(widths[c]) for c in cols] + [status]
        ))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Info rápida
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CsvInfo:
    path: Path
    size_bytes: int
    columns: list[str]
    n_rows: int
    first_timestamp: str | None
    last_timestamp: str | None
    missing_columns: list[str]
    extra_columns: list[str]

    def summary(self) -> str:
        return "\n".join([
            f"CSV: {self.path}",
            f"  tamaño:          {self.size_bytes / 1e6:.2f} MB",
            f"  columnas:        {self.columns}",
            f"  faltan:          {self.missing_columns or '-'}",
            f"  extra:           {self.extra_columns or '-'}",
            f"  filas:           {self.n_rows}",
            f"  primer ts:       {self.first_timestamp}",
            f"  último ts:       {self.last_timestamp}",
        ])


def csv_info(path: Path | str) -> CsvInfo:
    """Metadatos del CSV leyendo solo cabecera, primera y última fila."""
    path = Path(path)
    columns = read_header(path)
    head = read_head(path, 1)
    tail = read_tail(path, 1)
    first = head["datetime"].iloc[0] if "datetime" in head and len(head) else None
    last = tail["datetime"].iloc[0] if "datetime" in tail and len(tail) else None
    return CsvInfo(
        path=path,
        size_bytes=path.stat().st_size,
        columns=columns,
        n_rows=count_rows(path),
        first_timestamp=first,
        last_timestamp=last,
        missing_columns=[c for c in REQUIRED_COLUMNS if c not in columns],
        extra_columns=[c for c in columns if c not in REQUIRED_COLUMNS],
    )


# ---------------------------------------------------------------------------
# Scan completo por bloques
# ---------------------------------------------------------------------------

@dataclass
class ScanReport:
    """Resumen de integridad de todo el archivo (sin volcar datos)."""
    path: Path
    n_rows: int = 0
    nan_counts: dict[str, int] = field(default_factory=dict)
    n_bad_datetime: int = 0
    n_non_positive: int = 0
    n_ohlc_incoherent: int = 0
    n_duplicate_timestamps: int = 0
    n_out_of_order: int = 0
    inferred_frequency_seconds: float | None = None
    n_gaps: int = 0
    max_gap_seconds: float | None = None
    first_timestamp: pd.Timestamp | None = None
    last_timestamp: pd.Timestamp | None = None
    #: Ejemplos (nº de fila de datos, 1-based) de cada tipo de problema.
    examples: dict[str, list[int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (
            not any(self.nan_counts.values())
            and self.n_bad_datetime == 0
            and self.n_non_positive == 0
            and self.n_ohlc_incoherent == 0
            and self.n_duplicate_timestamps == 0
            and self.n_out_of_order == 0
        )

    def summary(self) -> str:
        nans = {k: v for k, v in self.nan_counts.items() if v} or "-"
        lines = [
            f"Scan: {self.path}",
            f"  filas:              {self.n_rows}",
            f"  rango:              {self.first_timestamp} .. {self.last_timestamp}",
            f"  NaNs:               {nans}",
            f"  datetime inválido:  {self.n_bad_datetime}",
            f"  precios <= 0:       {self.n_non_positive}",
            f"  OHLC incoherente:   {self.n_ohlc_incoherent}",
            f"  ts duplicados:      {self.n_duplicate_timestamps}",
            f"  fuera de orden:     {self.n_out_of_order}",
            f"  freq (s):           {self.inferred_frequency_seconds}",
            f"  gaps (>3x freq):    {self.n_gaps} (máx {self.max_gap_seconds} s)",
        ]
        for k, rows in self.examples.items():
            lines.append(f"  ej. {k}: filas {rows}")
        lines.append(f"  estado:             {'OK' if self.ok else 'CON PROBLEMAS'}")
        return "\n".join(lines)


def _add_examples(report: ScanReport, key: str, mask: np.ndarray, offset: int, k: int = 5) -> None:
    if not mask.any():
        return
    bucket = report.examples.setdefault(key, [])
    if len(bucket) >= k:
        return
    rows = (np.flatnonzero(mask) + offset + 1).tolist()
    bucket.extend(rows[: k - len(bucket)])


def scan_csv(
    path: Path | str,
    chunksize: int = DEFAULT_CHUNKSIZE,
    max_gap_multiplier: float = 3.0,
) -> ScanReport:
    """
    Recorre el CSV completo por bloques y devuelve un resumen de integridad.

    En memoria solo se acumulan los timestamps (int64), no el DataFrame.
    """
    path = Path(path)
    report = ScanReport(path=path)
    header = read_header(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{path}: faltan columnas requeridas {missing}")

    report.nan_counts = {c: 0 for c in header if c != "datetime"}
    ts_chunks: list[np.ndarray] = []
    offset = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype={"datetime": str}):
        n = len(chunk)
        dt = pd.to_datetime(chunk["datetime"], format=CSV_DATETIME_FORMAT, errors="coerce")
        bad_dt = dt.isna().to_numpy()
        report.n_bad_datetime += int(bad_dt.sum())
        _add_examples(report, "datetime inválido", bad_dt, offset)

        for c in report.nan_counts:
            col = pd.to_numeric(chunk[c], errors="coerce")
            report.nan_counts[c] += int(col.isna().sum())

        o, h, l, c_ = (pd.to_numeric(chunk[c], errors="coerce").to_numpy(dtype=float)
                       for c in PRICE_COLUMNS)
        with np.errstate(invalid="ignore"):
            non_pos = (o <= 0) | (h <= 0) | (l <= 0) | (c_ <= 0)
            incoh = (h < np.maximum(o, c_)) | (l > np.minimum(o, c_)) | (h < l)
        report.n_non_positive += int(non_pos.sum())
        report.n_ohlc_incoherent += int(incoh.sum())
        _add_examples(report, "precio <= 0", non_pos, offset)
        _add_examples(report, "OHLC incoherente", incoh, offset)

        ts_chunks.append(dt.to_numpy(dtype="datetime64[ns]"))
        offset += n

    report.n_rows = offset
    if not ts_chunks:
        return report

    ts_all = np.concatenate(ts_chunks)
    valid = ~np.isnat(ts_all)
    ts = ts_all[valid]
    if len(ts) == 0:
        return report

    # Orden y duplicados sobre la secuencia tal cual está en el archivo
    diffs = np.diff(ts.astype("int64"))
    out_of_order = diffs < 0
    report.n_out_of_order = int(out_of_order.sum())
    valid_rows = np.flatnonzero(valid)
    if out_of_order.any():
        report.examples["fuera de orden"] = (valid_rows[1:][out_of_order][:5] + 1).tolist()
    dup_mask = pd.Series(ts).duplicated().to_numpy()
    report.n_duplicate_timestamps = int(dup_mask.sum())
    if dup_mask.any():
        report.examples["ts duplicado"] = (valid_rows[dup_mask][:5] + 1).tolist()

    # Frecuencia y gaps sobre la serie ordenada y única (como haría el cleaner)
    s = pd.Series(np.unique(ts))
    report.first_timestamp = s.iloc[0]
    report.last_timestamp = s.iloc[-1]
    if len(s) >= 3:
        freq = float(s.diff().dropna().dt.total_seconds().median())
        report.inferred_frequency_seconds = freq
        report.n_gaps, report.max_gap_seconds = _detect_gaps(s, freq, max_gap_multiplier)
    return report


# ---------------------------------------------------------------------------
# Solapamiento temporal entre archivos
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OverlapReport:
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
    common_start: pd.Timestamp | None
    common_end: pd.Timestamp | None

    @property
    def has_overlap(self) -> bool:
        return self.common_start is not None

    def summary(self) -> str:
        lines = ["Rangos:"]
        for name, (a, b) in self.ranges.items():
            lines.append(f"  {name:<32} {a} .. {b}")
        if self.has_overlap:
            lines.append(f"Rango común: {self.common_start} .. {self.common_end}")
        else:
            lines.append("Rango común: (sin solapamiento)")
        return "\n".join(lines)


def overlap(paths: Iterable[Path | str]) -> OverlapReport:
    """
    Rango temporal común de varios CSV usando solo primera y última fila
    (asume archivos ordenados; verificar con scan_csv si hay dudas).
    """
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for p in paths:
        info = csv_info(p)
        a = pd.to_datetime(info.first_timestamp, format=CSV_DATETIME_FORMAT)
        b = pd.to_datetime(info.last_timestamp, format=CSV_DATETIME_FORMAT)
        ranges[Path(p).name] = (a, b)
    if not ranges:
        return OverlapReport(ranges, None, None)
    start = max(a for a, _ in ranges.values())
    end = min(b for _, b in ranges.values())
    if start > end:
        return OverlapReport(ranges, None, None)
    return OverlapReport(ranges, start, end)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspección de CSV de velas.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("head", "tail"):
        sp = sub.add_parser(name)
        sp.add_argument("csv")
        sp.add_argument("-n", type=int, default=5)
    sub.add_parser("info").add_argument("csv")
    sp = sub.add_parser("scan")
    sp.add_argument("csv")
    sp.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    sub.add_parser("overlap").add_argument("csvs", nargs="+")
    args = parser.parse_args(argv)

    if args.cmd in ("head", "tail"):
        if args.cmd == "head":
            df, first = read_head(args.csv, args.n), 1
        else:
            df = read_tail(args.csv, args.n)
            first = count_rows(args.csv) - len(df) + 1
        checks = check_rows(df, first_row=first)
        print(format_row_checks(checks, list(df.columns)))
        n_bad = sum(not c.ok for c in checks)
        print(f"\n{len(checks) - n_bad}/{len(checks)} filas OK")
        return 0 if n_bad == 0 else 1
    if args.cmd == "info":
        print(csv_info(args.csv).summary())
        return 0
    if args.cmd == "scan":
        rep = scan_csv(args.csv, chunksize=args.chunksize)
        print(rep.summary())
        return 0 if rep.ok else 1
    rep = overlap(args.csvs)
    print(rep.summary())
    return 0 if rep.has_overlap else 1


if __name__ == "__main__":
    raise SystemExit(main())
