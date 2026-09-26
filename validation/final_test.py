"""
Bloque final intocable.

Con 60 días de datos y final_test_ratio=0.0, este módulo no se activa.
Con datasets más grandes, se reserva el último bloque del histórico
y NUNCA se toca durante el desarrollo.

Regla estricta: si un investigador mira el final test, el final test
deja de ser válido. Solo se ejecuta UNA VEZ, al final, sobre la
estrategia ya congelada.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gp.individual import Individual
from gp.serialization import load_individual
from validation.robustness import RobustnessResult, evaluate_on_other_data


@dataclass
class FinalTestConfig:
    """
    Configuración del bloque final.

    ratio: fracción del dataset reservada como final test.
    lock_file: archivo donde se registra si el final test ya fue ejecutado.
    """
    ratio: float = 0.10
    lock_file: Path | None = None


def split_final_test(
    df: pd.DataFrame,
    config: FinalTestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el DataFrame en (development, final_test).

    development: 1 - ratio del dataset.
    final_test: ratio del dataset, al final.

    El final_test NO debe usarse durante el desarrollo.
    """
    if config.ratio <= 0:
        return df.copy(), df.iloc[0:0].copy()  # dev vacío

    n = len(df)
    n_final = int(round(n * config.ratio))
    n_dev = n - n_final

    if n_final < 1:
        raise ValueError(
            f"final_test_ratio={config.ratio} produce 0 filas con n={n}"
        )

    dev = df.iloc[:n_dev].copy()
    final = df.iloc[n_dev:].copy()
    return dev, final


def run_final_test(
    individual: Individual,
    final_test_df: pd.DataFrame,
    config: FinalTestConfig,
    initial_capital: float = 10_000.0,
) -> RobustnessResult:
    """
    Ejecuta el final test.

    Si config.lock_file existe, lanza RuntimeError. Esto evita ejecutar
    el final test más de una vez, lo cual invalidaría su interpretación.
    """
    if config.lock_file is not None and config.lock_file.exists():
        raise RuntimeError(
            f"El final test ya fue ejecutado. Lock file: {config.lock_file}. "
            "Si quieres re-ejecutarlo, borra el lock manualmente y asume "
            "que el resultado pierde validez estadística."
        )

    result = evaluate_on_other_data(
        individual=individual,
        ohlc=final_test_df,
        label="FINAL_TEST",
        initial_capital=initial_capital,
    )

    if config.lock_file is not None:
        config.lock_file.parent.mkdir(parents=True, exist_ok=True)
        config.lock_file.write_text(
            f"Final test ejecutado.\n"
            f"n_bars={result.n_bars}\n"
            f"return={result.total_return:.6f}\n"
            f"sharpe={result.sharpe:.4f}\n"
        )

    return result