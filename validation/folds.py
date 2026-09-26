"""
Generación de folds para Walk-Forward Optimization.

Un fold es una tupla (IS, OOS):
- IS: DataFrame con el bloque in-sample.
- OOS: DataFrame con el bloque out-of-sample inmediatamente siguiente.

Modo rolling: IS tiene tamaño fijo, avanza step por step.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    """Un fold de WFO."""
    index: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    is_df: pd.DataFrame
    oos_df: pd.DataFrame

    @property
    def n_is(self) -> int:
        return len(self.is_df)

    @property
    def n_oos(self) -> int:
        return len(self.oos_df)

    def __repr__(self) -> str:
        return (
            f"Fold({self.index}: "
            f"IS=[{self.is_start} .. {self.is_end}] ({self.n_is} velas), "
            f"OOS=[{self.oos_start} .. {self.oos_end}] ({self.n_oos} velas))"
        )

@dataclass(frozen=True)
class WFConfig:
    """
    Configuración del WFO.

    is_size: número de velas en IS.
    oos_size: número de velas en OOS.
    step_size: cuánto avanza el inicio del bloque entre folds.
               Si None, usa oos_size (sin solapamiento).
    min_is_size: mínimo aceptable de velas en IS tras filtrar.
    min_oos_size: mínimo aceptable de velas en OOS tras filtrar.
    max_timestamp: salvaguarda contra el final test.
    """
    is_size: int
    oos_size: int
    step_size: int | None = None
    min_is_size: int | None = None   # por defecto, 90% de is_size
    min_oos_size: int | None = None  # por defecto, 90% de oos_size
    max_timestamp: pd.Timestamp | None = None

    def __post_init__(self):
        if self.is_size <= 0 or self.oos_size <= 0:
            raise ValueError("is_size y oos_size deben ser > 0")
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError("step_size debe ser > 0")
        # defaults
        object.__setattr__(
            self, "min_is_size",
            self.min_is_size if self.min_is_size is not None else int(self.is_size * 0.9),
        )
        object.__setattr__(
            self, "min_oos_size",
            self.min_oos_size if self.min_oos_size is not None else int(self.oos_size * 0.9),
        )

    @property
    def effective_step(self) -> int:
        return self.step_size if self.step_size is not None else self.oos_size

    @property
    def block_size(self) -> int:
        return self.is_size + self.oos_size


def generate_folds(df: pd.DataFrame, config: WFConfig) -> list[Fold]:
    if config.max_timestamp is not None:
        df = df[df.index <= config.max_timestamp]
        if len(df) == 0:
            raise ValueError(f"Ninguna fila cumple index <= {config.max_timestamp}")

    if len(df) < config.block_size:
        raise ValueError(
            f"Dataset demasiado pequeño: {len(df)} filas, "
            f"se requiere al menos block_size={config.block_size}"
        )

    n = len(df)
    folds: list[Fold] = []
    fold_idx = 0
    start = 0

    while start + config.block_size <= n:
        is_df = df.iloc[start : start + config.is_size].copy()
        oos_df = df.iloc[start + config.is_size : start + config.block_size].copy()

        if len(is_df) >= config.min_is_size and len(oos_df) >= config.min_oos_size:
            folds.append(
                Fold(
                    index=fold_idx,
                    is_start=is_df.index[0],
                    is_end=is_df.index[-1],
                    oos_start=oos_df.index[0],
                    oos_end=oos_df.index[-1],
                    is_df=is_df,
                    oos_df=oos_df,
                )
            )
            fold_idx += 1

        start += config.effective_step

    if not folds:
        raise ValueError("No se generó ningún fold válido. Revisa la configuración.")

    return folds