"""
Splitter temporal.

Divide el dataset en:
- development: bloque para WFO y optimización de hiperparámetros.
- final_test: bloque intocable, reservado al final del histórico.

El final test NO se usa en ningún momento del desarrollo. Solo se
ejecuta UNA VEZ, sobre la estrategia congelada, al final del proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    """Configuración del split temporal."""
    final_test_ratio: float = 0.10
    #: Si True, guarda el final_test en un CSV separado.
    save_final_test: bool = True
    #: Ruta donde guardar el final_test si save_final_test=True.
    final_test_path: Path | None = None


def split_development_final(
    df: pd.DataFrame,
    config: SplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el DataFrame en (development, final_test).

    El final_test es el ÚLTIMO bloque cronológico del dataset.
    El development es todo lo anterior.

    Requisitos:
    - df debe estar ordenado cronológicamente ascendente.
    - df debe tener al menos 1000 filas para que el split tenga sentido.
    """
    if len(df) < 1000:
        raise ValueError(
            f"Dataset demasiado pequeño para split: {len(df)} filas"
        )
    if not (0.0 < config.final_test_ratio < 1.0):
        raise ValueError(
            f"final_test_ratio debe estar en (0, 1), "
            f"recibido {config.final_test_ratio}"
        )

    # Verificar orden cronológico
    if not df.index.is_monotonic_increasing:
        raise ValueError("El DataFrame debe estar ordenado cronológicamente.")

    n = len(df)
    n_final = int(round(n * config.final_test_ratio))
    n_dev = n - n_final

    development = df.iloc[:n_dev].copy()
    final_test = df.iloc[n_dev:].copy()

    if config.save_final_test and config.final_test_path is not None:
        path = Path(config.final_test_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        final_test.to_csv(path)
        print(f"Final test guardado en: {path} ({len(final_test)} filas)")
        print(f"  Desde: {final_test.index[0]}")
        print(f"  Hasta: {final_test.index[-1]}")

    return development, final_test


def describe_split(
    development: pd.DataFrame,
    final_test: pd.DataFrame,
) -> str:
    """Devuelve un resumen legible del split."""
    lines = [
        "Split temporal:",
        "----------------",
        f"  Development: {len(development):>8} velas "
        f"[{development.index[0]} .. {development.index[-1]}]",
        f"  Final test:  {len(final_test):>8} velas "
        f"[{final_test.index[0]} .. {final_test.index[-1]}]",
        f"  Total:       {len(development) + len(final_test):>8} velas",
    ]
    return "\n".join(lines)