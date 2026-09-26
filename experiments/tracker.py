"""
Registro de experimentos.

Cada ejecución del pipeline se registra con un hash único, parámetros,
y resultados. Esto permite:
- Reproducibilidad.
- Auditoría de cuántos experimentos se han probado.
- Cálculo de Deflated Sharpe Ratio sobre el conjunto completo.

El problema del multiple testing: si pruebas 1000 estrategias y eliges la
mejor, su Sharpe no es estadísticamente significativo aunque parezca alto.
El tracker registra cuántos experimentos has hecho para poder ajustar.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ExperimentRecord:
    """Un experimento registrado."""
    experiment_id: str
    timestamp: str
    config: dict
    is_metrics: dict
    oos_metrics: dict
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _hash_config(config: dict) -> str:
    """Hash determinista de un dict de configuración."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


@dataclass
class ExperimentTracker:
    """Tracker de experimentos, con persistencia a JSONL."""
    storage_path: Path
    records: list[ExperimentRecord] = field(default_factory=list)

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if self.storage_path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        with self.storage_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.records.append(ExperimentRecord(**d))

    def register(
        self,
        config: dict,
        is_metrics: dict,
        oos_metrics: dict,
        notes: str = "",
    ) -> ExperimentRecord:
        """Registra un experimento y lo persiste."""
        exp_id = _hash_config(config)
        ts = datetime.now(timezone.utc).isoformat()

        record = ExperimentRecord(
            experiment_id=exp_id,
            timestamp=ts,
            config=config,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            notes=notes,
        )
        self.records.append(record)

        with self.storage_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), default=str) + "\n")

        return record

    @property
    def n_experiments(self) -> int:
        return len(self.records)

    def deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

        Ajusta el Sharpe observado por el número de experimentos realizados.

        observed_sharpe: Sharpe del mejor experimento.
        n_observations: número de velas del OOS.
        skewness, kurtosis: de la distribución de retornos OOS.
        """
        from scipy.stats import norm

        n_trials = max(self.n_experiments, 1)

        # Sharpe esperado bajo H0 con n_trials independientes
        # E[max(Sharpe)] ≈ sqrt(2 * log(n_trials)) para muestras grandes
        # Ajuste por varianza del estimador
        euler_mascheroni = 0.5772156649
        e_max = np.sqrt(2 * np.log(n_trials)) - (
            np.log(np.log(n_trials)) + np.log(4 * np.pi)
        ) / (2 * np.sqrt(2 * np.log(n_trials)))

        # Varianza del estimador de Sharpe (Lo, 2002)
        var_sr = (
            1
            - skewness * observed_sharpe
            + ((kurtosis - 1) / 4) * observed_sharpe ** 2
        ) / max(n_observations - 1, 1)

        if var_sr <= 0:
            return 0.0

        # DSR = P(SR_observado > E[max(SR)] bajo H0)
        z = (observed_sharpe - e_max) / np.sqrt(var_sr)
        dsr = float(norm.cdf(z))
        return dsr