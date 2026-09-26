"""
Persistencia de resultados de experimentos.

Guarda:
- Individuos serializados (JSON).
- Curvas de equity (CSV).
- Métricas agregadas (JSON).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gp.individual import Individual
from gp.serialization import save_individual
from validation.walk_forward import WalkForwardResult


def save_experiment(
    base_dir: Path,
    experiment_id: str,
    individual: Individual,
    wfo_result: WalkForwardResult,
    extra_metadata: dict | None = None,
) -> Path:
    """
    Guarda un experimento completo en disco.

    Estructura:
    base_dir/
      experiment_id/
        individual.json
        equity_stitched.csv
        metrics.json
        metadata.json
    """
    base_dir = Path(base_dir)
    exp_dir = base_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Individuo
    save_individual(individual, exp_dir / "individual.json")

    # Equity stitched
    if wfo_result.equity_stitched is not None:
        wfo_result.equity_stitched.to_csv(exp_dir / "equity_stitched.csv")

    # Métricas
    with (exp_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(wfo_result.metrics_aggregated, f, indent=2, default=str)

    # Metadata
    meta = {
        "experiment_id": experiment_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "n_folds": len(wfo_result.fold_results),
    }
    if extra_metadata:
        meta.update(extra_metadata)
    with (exp_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    return exp_dir


def list_experiments(base_dir: Path) -> list[str]:
    """Lista los IDs de experimentos guardados."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []
    return sorted(
        d.name for d in base_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()
    )


def load_experiment_metrics(base_dir: Path, experiment_id: str) -> dict:
    """Carga las métricas de un experimento."""
    path = Path(base_dir) / experiment_id / "metrics.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)