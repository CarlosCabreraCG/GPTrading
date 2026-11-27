"""
Merge cross-asset: añade el close de otros instrumentos a un instrumento base.

Objetivo: entrenar GP sobre un instrumento (base) considerando el efecto de
otros (cross). El resultado es el OHLC del base + una columna `{sym}_close`
por cada instrumento cruzado, que es la convención que espera
`features.registry` (ver CROSS_ASSETS).

Reglas:
- Entradas en formato canónico (usar `data.normalize` antes si hace falta).
  El símbolo se infiere del nombre con `infer_symbol_from_path`.
- Cada entrada pasa por `data.cleaner.clean` (orden, duplicados, NaNs) antes
  de unir, para que un timestamp duplicado no multiplique filas.
- Unión INNER por timestamp exacto: solo se conservan velas presentes en
  todos los instrumentos. No se rellena nada (ni ffill ni bfill): una vela
  cross que no existe no se inventa.
- Todos los instrumentos deben tener la misma frecuencia.

Nombre de salida: `{base_stem}_x_{sym1}_{sym2}.csv`
(p. ej. `xauusd_15m_5y_x_xagusd_dxy.csv`); empieza por el símbolo base, así
`run_wfo` infiere el símbolo correcto (costes del base).

CLI:
    python -m data.merge_cross --base data_files/xauusd_15m_5y.csv \\
        --cross data_files/xagusd_15m_5y.csv data_files/dxy_15m_5y.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from config.settings import CSV_DATETIME_FORMAT, infer_symbol_from_path
from data.cleaner import clean
from data.loader import load_raw_csv
from features.registry import CROSS_ASSETS

logger = logging.getLogger(__name__)

OHLC: tuple[str, ...] = ("open", "high", "low", "close")


def cross_column(symbol: str) -> str:
    """Nombre de la columna cross para un símbolo: `{sym}_close` en minúsculas."""
    return f"{symbol.lower()}_close"


def merged_filename(base_path: Path | str, cross_symbols: Sequence[str]) -> str:
    """`{base_stem}_x_{sym1}_{sym2}.csv`."""
    syms = "_".join(s.lower() for s in cross_symbols)
    return f"{Path(base_path).stem}_x_{syms}.csv"


@dataclass(frozen=True)
class CrossStep:
    symbol: str
    column: str
    n_rows_input: int
    n_rows_after: int
    has_features: bool


@dataclass(frozen=True)
class MergeReport:
    base_symbol: str
    n_rows_base: int
    steps: tuple[CrossStep, ...]
    n_rows_output: int
    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None
    frequency_seconds: float | None

    @property
    def n_rows_dropped(self) -> int:
        return self.n_rows_base - self.n_rows_output

    def summary(self) -> str:
        lines = [
            f"Merge cross-asset (base {self.base_symbol})",
            f"  filas base (limpio):  {self.n_rows_base}",
        ]
        for s in self.steps:
            note = "" if s.has_features else "  [SIN features en registry]"
            lines.append(
                f"  + {s.column:<16} filas {s.n_rows_input:>7} -> tras unir {s.n_rows_after:>7}{note}"
            )
        pct = 100.0 * self.n_rows_dropped / self.n_rows_base if self.n_rows_base else 0.0
        lines += [
            f"  filas salida:         {self.n_rows_output} "
            f"(descartadas {self.n_rows_dropped}, {pct:.1f}%)",
            f"  rango:                {self.first_timestamp} .. {self.last_timestamp}",
            f"  freq (s):             {self.frequency_seconds}",
        ]
        return "\n".join(lines)


def _load_clean(path: Path | str) -> tuple[pd.DataFrame, float | None]:
    df, report = clean(load_raw_csv(path))
    return df, report.inferred_frequency_seconds


def merge_cross(
    base_csv: Path | str,
    cross_csvs: Sequence[Path | str],
) -> tuple[pd.DataFrame, MergeReport]:
    """
    Une el OHLC de `base_csv` con el close de cada CSV de `cross_csvs`.

    Devuelve (df, report). `df` tiene índice `datetime` (como el cleaner) y
    columnas open, high, low, close, {sym}_close...
    """
    if not cross_csvs:
        raise ValueError("Se necesita al menos un instrumento cross")

    base_symbol = infer_symbol_from_path(base_csv)
    base, base_freq = _load_clean(base_csv)
    merged = base[list(OHLC)].copy()
    n_base = len(merged)

    seen = {base_symbol}
    steps: list[CrossStep] = []
    for path in cross_csvs:
        sym = infer_symbol_from_path(path)
        if sym in seen:
            raise ValueError(f"Símbolo repetido en el merge: {sym}")
        seen.add(sym)

        cross, freq = _load_clean(path)
        if base_freq is not None and freq is not None and freq != base_freq:
            raise ValueError(
                f"Frecuencia distinta: base {base_symbol} {base_freq}s vs {sym} {freq}s"
            )
        col = cross_column(sym)
        merged = merged.join(cross[["close"]].rename(columns={"close": col}), how="inner")

        has_features = sym.lower() in CROSS_ASSETS
        if not has_features:
            logger.warning(
                "%s no está en CROSS_ASSETS: la columna %s no generará features "
                "hasta añadirlo en features/registry.py", sym, col,
            )
        steps.append(CrossStep(sym, col, len(cross), len(merged), has_features))

    if merged.empty:
        raise ValueError("El merge no deja filas: los instrumentos no comparten timestamps")

    report = MergeReport(
        base_symbol=base_symbol,
        n_rows_base=n_base,
        steps=tuple(steps),
        n_rows_output=len(merged),
        first_timestamp=merged.index[0],
        last_timestamp=merged.index[-1],
        frequency_seconds=base_freq,
    )
    return merged, report


def write_merged(df: pd.DataFrame, path: Path | str) -> Path:
    """Escribe el resultado de `merge_cross` en el formato CSV del repositorio."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index()
    out.to_csv(path, index=False, date_format=CSV_DATETIME_FORMAT)
    return path


def merge_cross_to_csv(
    base_csv: Path | str,
    cross_csvs: Sequence[Path | str],
    output: Path | str | None = None,
    overwrite: bool = False,
) -> tuple[Path, MergeReport]:
    """merge_cross + write_merged. Por defecto escribe junto al base."""
    df, report = merge_cross(base_csv, cross_csvs)
    if output is None:
        symbols = [s.symbol for s in report.steps]
        output = Path(base_csv).parent / merged_filename(base_csv, symbols)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"El destino ya existe: {output} (usa --overwrite)")
    return write_merged(df, output), report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Añade el close de otros instrumentos (cross) a un instrumento base."
    )
    parser.add_argument("--base", required=True, help="CSV del instrumento a entrenar.")
    parser.add_argument("--cross", required=True, nargs="+", help="CSV(s) cross.")
    parser.add_argument("--output", default=None,
                        help="Ruta de salida. Por defecto {base}_x_{syms}.csv junto al base.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    try:
        path, report = merge_cross_to_csv(args.base, args.cross, args.output, args.overwrite)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        parser.error(str(e))
    print(report.summary())
    print(f"\nArchivo generado: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
