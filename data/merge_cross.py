"""
Merge cross-asset: añade el close de otros instrumentos a un instrumento base.

Objetivo: entrenar GP sobre un instrumento (base) considerando el efecto de
otros (cross). El resultado es el OHLC del base + una columna `{sym}_close`
por cada instrumento cruzado, que es la convención que espera
`features.registry` (ver CROSS_ASSETS: todo símbolo soportado genera features).

Reglas:
- Entradas en formato canónico (usar `data.normalize` antes si hace falta).
  El símbolo se infiere del nombre con `infer_symbol_from_path`.
- Cada entrada pasa por `data.cleaner.clean` (orden, duplicados, NaNs) antes
  de unir, para que un timestamp duplicado no multiplique filas.
- Unión INNER por timestamp exacto: solo se conservan velas presentes en
  todos los instrumentos. No se rellena nada (ni ffill ni bfill): una vela
  cross que no existe no se inventa.
- Todos los instrumentos deben tener la misma frecuencia.

Control de cobertura (ver `Coverage` y `CoverageLimits`)
--------------------------------------------------------
El inner join elimina velas del base. Se distinguen tres medidas, respecto
al base limpio:

- edge_drop:     velas fuera del rango temporal común (un cross empieza más
                 tarde / acaba antes). Solo acorta el histórico.
- interior_drop: velas DENTRO del rango común que faltan en algún cross.
                 Son huecos: suelen ser horas de sesión que el cross no
                 cotiza (DXY y BRENT no cotizan ~21:00-24:00 ni domingos),
                 así que eliminan sistemáticamente las mismas horas del base.
- broken:        % de velas del resultado cuya vela anterior ya no es la
                 anterior real del base. En ellas return_1, ATR, etc. y el
                 PnL abarcan velas inexistentes.

Calibración (datos 15m y 4h, 5 años, todos los pares base+cross): los pares
"compatibles" quedan en interior_drop <= 2.1% y broken <= 0.3%; los pares
con sesión distinta (DXY, BRENT frente a 24h) saltan a 17-25% y 1.2-3.8%.
No hay casos intermedios. Límites por defecto: interior 5%, broken 1%,
total 15% (este último sobre todo por recortes de borde). Si se superan, el
merge se rechaza ("no se puede trabajar con estos instrumentos") salvo
`force=True`, que lo permite con WARNING.

Nombre de salida: `{base_stem}_x_{sym1}_{sym2}.csv`
(p. ej. `xauusd_15m_5y_x_xagusd_dxy.csv`); empieza por el símbolo base, así
`run_wfo` infiere el símbolo correcto (costes del base).

CLI:
    python -m data.merge_cross --base data_files/xauusd_15m_5y.csv \\
        --cross data_files/xagusd_15m_5y.csv data_files/wti_15m_5y.csv
    python -m data.merge_cross --base ... --cross ... --dry-run   # solo informe
    python -m data.merge_cross --base ... --cross ... --force     # ignora límites
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from config.settings import CSV_DATETIME_FORMAT, infer_symbol_from_path
from data.cleaner import clean
from data.loader import load_raw_csv

logger = logging.getLogger(__name__)

OHLC: tuple[str, ...] = ("open", "high", "low", "close")


# ---------------------------------------------------------------------------
# Nombres
# ---------------------------------------------------------------------------

def cross_column(symbol: str) -> str:
    """Nombre de la columna cross para un símbolo: `{sym}_close` en minúsculas."""
    return f"{symbol.lower()}_close"


def merged_filename(base_path: Path | str, cross_symbols: Sequence[str]) -> str:
    """`{base_stem}_x_{sym1}_{sym2}.csv`."""
    syms = "_".join(s.lower() for s in cross_symbols)
    return f"{Path(base_path).stem}_x_{syms}.csv"


# ---------------------------------------------------------------------------
# Cobertura
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageLimits:
    """Pérdida máxima tolerada (en %, respecto al base limpio)."""
    max_interior_drop_pct: float = 5.0
    max_broken_pct: float = 1.0
    max_total_drop_pct: float = 15.0


DEFAULT_COVERAGE_LIMITS = CoverageLimits()


@dataclass(frozen=True)
class Coverage:
    """Cuánto del base sobrevive al merge (ver docstring del módulo)."""
    n_base: int
    n_output: int
    n_edge_dropped: int
    n_interior_dropped: int
    n_broken: int

    @property
    def total_drop_pct(self) -> float:
        return 100.0 * (self.n_base - self.n_output) / self.n_base if self.n_base else 0.0

    @property
    def edge_drop_pct(self) -> float:
        return 100.0 * self.n_edge_dropped / self.n_base if self.n_base else 0.0

    @property
    def interior_drop_pct(self) -> float:
        n_common = self.n_base - self.n_edge_dropped
        return 100.0 * self.n_interior_dropped / n_common if n_common else 0.0

    @property
    def broken_pct(self) -> float:
        return 100.0 * self.n_broken / (self.n_output - 1) if self.n_output > 1 else 0.0

    def violations(self, limits: CoverageLimits) -> list[str]:
        out = []
        if self.interior_drop_pct > limits.max_interior_drop_pct:
            out.append(f"huecos interiores {self.interior_drop_pct:.1f}% "
                       f"> {limits.max_interior_drop_pct}%")
        if self.broken_pct > limits.max_broken_pct:
            out.append(f"continuidad rota {self.broken_pct:.1f}% > {limits.max_broken_pct}%")
        if self.total_drop_pct > limits.max_total_drop_pct:
            out.append(f"pérdida total {self.total_drop_pct:.1f}% > {limits.max_total_drop_pct}%")
        return out

    def short(self) -> str:
        return (f"total {self.total_drop_pct:5.1f}% | borde {self.edge_drop_pct:5.1f}% | "
                f"interior {self.interior_drop_pct:5.1f}% | rota {self.broken_pct:4.1f}%")


def compute_coverage(
    base_index: pd.DatetimeIndex,
    merged_index: pd.DatetimeIndex,
    common_start: pd.Timestamp,
    common_end: pd.Timestamp,
) -> Coverage:
    """
    Cobertura de `merged_index` (subconjunto de `base_index`, ambos ordenados).

    common_start / common_end: rango temporal común de todas las entradas.
    """
    in_common = (base_index >= common_start) & (base_index <= common_end)
    n_common = int(in_common.sum())
    n_edge = len(base_index) - n_common

    # Continuidad: posición de cada vela del merge dentro del base. Si dos
    # velas consecutivas del merge no son consecutivas en el base, hay hueco.
    pos = base_index.get_indexer(merged_index)
    n_broken = int((np.diff(pos) != 1).sum()) if len(pos) > 1 else 0

    return Coverage(
        n_base=len(base_index),
        n_output=len(merged_index),
        n_edge_dropped=n_edge,
        n_interior_dropped=n_common - len(merged_index),
        n_broken=n_broken,
    )


class MergeCoverageError(ValueError):
    """El merge pierde demasiadas velas del base para ser utilizable."""


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossStep:
    symbol: str
    column: str
    n_rows_input: int
    #: Cobertura del base frente a ESTE cross solo (identifica al culpable).
    pairwise: Coverage
    #: Cobertura acumulada tras añadir este cross.
    cumulative: Coverage


@dataclass(frozen=True)
class MergeReport:
    base_symbol: str
    n_rows_base: int
    steps: tuple[CrossStep, ...]
    coverage: Coverage
    limits: CoverageLimits
    violations: tuple[str, ...]
    forced: bool
    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None
    frequency_seconds: float | None

    @property
    def n_rows_output(self) -> int:
        return self.coverage.n_output

    @property
    def n_rows_dropped(self) -> int:
        return self.coverage.n_base - self.coverage.n_output

    @property
    def ok(self) -> bool:
        return not self.violations

    def culprits(self) -> list[str]:
        """Cross que por sí solos (frente al base) ya violan los límites."""
        return [s.symbol for s in self.steps if s.pairwise.violations(self.limits)]

    def summary(self) -> str:
        lines = [
            f"Merge cross-asset (base {self.base_symbol}, {self.n_rows_base} velas limpias)",
            f"  {'cross':<16} {'filas':>7}  cobertura frente al base solo / acumulada",
        ]
        for s in self.steps:
            flag = "  <-- excede límites" if s.pairwise.violations(self.limits) else ""
            lines.append(f"  + {s.column:<14} {s.n_rows_input:>7}  solo:  {s.pairwise.short()}{flag}")
            lines.append(f"  {'':<16} {'':>7}  acum.: {s.cumulative.short()}")
        lines += [
            f"  filas salida:  {self.n_rows_output} (descartadas {self.n_rows_dropped})",
            f"  rango:         {self.first_timestamp} .. {self.last_timestamp}",
            f"  freq (s):      {self.frequency_seconds}",
            f"  límites:       interior {self.limits.max_interior_drop_pct}% | "
            f"rota {self.limits.max_broken_pct}% | total {self.limits.max_total_drop_pct}%",
        ]
        if self.ok:
            lines.append("  veredicto:     OK")
        else:
            verdict = "FORZADO (force=True)" if self.forced else "RECHAZADO"
            lines.append(f"  veredicto:     {verdict}: {'; '.join(self.violations)}")
            if self.culprits():
                lines.append(f"  causa:         {self.base_symbol} + {', '.join(self.culprits())}")
        return "\n".join(lines)


def _load_clean(path: Path | str) -> tuple[pd.DataFrame, float | None]:
    df, report = clean(load_raw_csv(path))
    return df, report.inferred_frequency_seconds


def _coverage_against(
    base_idx: pd.DatetimeIndex, indexes: Sequence[pd.DatetimeIndex]
) -> Coverage:
    merged = base_idx
    for idx in indexes:
        merged = merged.intersection(idx)
    start = max([base_idx[0], *(i[0] for i in indexes)])
    end = min([base_idx[-1], *(i[-1] for i in indexes)])
    return compute_coverage(base_idx, merged, start, end)


def merge_cross(
    base_csv: Path | str,
    cross_csvs: Sequence[Path | str],
    limits: CoverageLimits = DEFAULT_COVERAGE_LIMITS,
    force: bool = False,
    check: bool = True,
) -> tuple[pd.DataFrame, MergeReport]:
    """
    Une el OHLC de `base_csv` con el close de cada CSV de `cross_csvs`.

    Devuelve (df, report). `df` tiene índice `datetime` (como el cleaner) y
    columnas open, high, low, close, {sym}_close...

    limits: pérdida máxima tolerada. Si se supera lanza MergeCoverageError
        salvo force=True (entonces solo WARNING y report.forced=True).
    check: si False no lanza nunca (útil para solo inspeccionar el informe).
    """
    if not cross_csvs:
        raise ValueError("Se necesita al menos un instrumento cross")

    base_symbol = infer_symbol_from_path(base_csv)
    base, base_freq = _load_clean(base_csv)
    if base.empty:
        raise ValueError(f"El base {base_csv} no tiene filas tras limpiar")
    merged = base[list(OHLC)].copy()

    seen = {base_symbol}
    steps: list[CrossStep] = []
    cross_indexes: list[pd.DatetimeIndex] = []
    for path in cross_csvs:
        sym = infer_symbol_from_path(path)
        if sym in seen:
            raise ValueError(f"Símbolo repetido en el merge: {sym}")
        seen.add(sym)

        cross, freq = _load_clean(path)
        if cross.empty:
            raise ValueError(f"El cross {path} no tiene filas tras limpiar")
        if base_freq is not None and freq is not None and freq != base_freq:
            raise ValueError(
                f"Frecuencia distinta: base {base_symbol} {base_freq}s vs {sym} {freq}s"
            )
        col = cross_column(sym)
        merged = merged.join(cross[["close"]].rename(columns={"close": col}), how="inner")
        cross_indexes.append(cross.index)
        steps.append(CrossStep(
            symbol=sym,
            column=col,
            n_rows_input=len(cross),
            pairwise=_coverage_against(base.index, [cross.index]),
            cumulative=_coverage_against(base.index, cross_indexes),
        ))

    if merged.empty:
        raise MergeCoverageError(
            "El merge no deja filas: los instrumentos no comparten timestamps"
        )

    coverage = steps[-1].cumulative
    violations = tuple(coverage.violations(limits))
    report = MergeReport(
        base_symbol=base_symbol,
        n_rows_base=len(base),
        steps=tuple(steps),
        coverage=coverage,
        limits=limits,
        violations=violations,
        forced=bool(violations) and force,
        first_timestamp=merged.index[0],
        last_timestamp=merged.index[-1],
        frequency_seconds=base_freq,
    )
    if violations and check:
        if not force:
            raise MergeCoverageError(
                "No se puede trabajar con este merge:\n" + report.summary()
                + "\nUsa force=True (--force) para generarlo igualmente."
            )
        logger.warning("Merge FORZADO pese a: %s", "; ".join(violations))
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
    limits: CoverageLimits = DEFAULT_COVERAGE_LIMITS,
    force: bool = False,
) -> tuple[Path, MergeReport]:
    """merge_cross + write_merged. Por defecto escribe junto al base."""
    df, report = merge_cross(base_csv, cross_csvs, limits=limits, force=force)
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo mostrar el informe de cobertura, sin escribir.")
    parser.add_argument("--force", action="store_true",
                        help="Generar el merge aunque supere los límites de cobertura.")
    d = DEFAULT_COVERAGE_LIMITS
    parser.add_argument("--max-interior-drop", type=float, default=d.max_interior_drop_pct)
    parser.add_argument("--max-broken", type=float, default=d.max_broken_pct)
    parser.add_argument("--max-total-drop", type=float, default=d.max_total_drop_pct)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

    limits = CoverageLimits(args.max_interior_drop, args.max_broken, args.max_total_drop)
    try:
        if args.dry_run:
            _, report = merge_cross(args.base, args.cross, limits=limits, check=False)
            print(report.summary())
            return 0 if report.ok else 2
        path, report = merge_cross_to_csv(
            args.base, args.cross, args.output, args.overwrite, limits, args.force
        )
    except MergeCoverageError as e:
        print(e)
        return 2
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        parser.error(str(e))
    print(report.summary())
    print(f"\nArchivo generado: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
