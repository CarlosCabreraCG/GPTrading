"""
Modelos de datos. Dataclasses inmutables para representar OHLCV y metadatos.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class Bar:
    """Una vela individual."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float



@dataclass(frozen=True)
class DataQualityReport:
    """
    Resultado del proceso de limpieza. Se devuelve junto con el DataFrame
    limpio para que el llamador pueda inspeccionar qué se hizo.
    """
    n_rows_input: int
    n_rows_output: int
    n_duplicates_removed: int
    n_nans_removed: int
    n_gaps_detected: int
    max_gap_seconds: Optional[float]
    first_timestamp: Optional[datetime]
    last_timestamp: Optional[datetime]
    inferred_frequency_seconds: Optional[float]

    def summary(self) -> str:
        lines = [
            "DataQualityReport",
            "-----------------",
            f"  input rows:      {self.n_rows_input}",
            f"  output rows:     {self.n_rows_output}",
            f"  duplicates:      {self.n_duplicates_removed}",
            f"  NaNs removed:    {self.n_nans_removed}",
            f"  gaps detected:   {self.n_gaps_detected}",
            f"  max gap (s):     {self.max_gap_seconds}",
            f"  first ts:        {self.first_timestamp}",
            f"  last ts:         {self.last_timestamp}",
            f"  freq (s):        {self.inferred_frequency_seconds}",
        ]
        return "\n".join(lines)