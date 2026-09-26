"""
Script: corre GP sobre un dataset y devuelve el mejor individuo.

Uso:
    python -m scripts.run_gp --csv data_files/EURUSD_X_15m_60d.csv \
        --population 50 --generations 15 --seed 42 \
        --output experiments_output/gp_run_001
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data.cleaner import load_and_clean
from features.engine import compute_features, drop_warmup_rows
from fitness.objective import FitnessConfig, make_fitness_fn
from gp.evaluator import EvalContext
from gp.evolution import EvolutionConfig, evolve
from gp.serialization import save_individual


def main():
    parser = argparse.ArgumentParser(description="Corre GP sobre un CSV de velas.")
    parser.add_argument("--csv", required=True, help="Ruta al CSV.")
    parser.add_argument("--output", required=True, help="Directorio de salida.")
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--tournament", type=int, default=5)
    parser.add_argument("--elites", type=int, default=2)
    parser.add_argument("--crossover", type=float, default=0.8)
    parser.add_argument("--mutation", type=float, default=0.2)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Cargar y limpiar
    df, report = load_and_clean(args.csv)
    print(report.summary())

    # Features
    feats = compute_features(df)
    df_aligned, feats_aligned = drop_warmup_rows(df, feats)

    ctx = EvalContext(
        features={name: feats_aligned[name].to_numpy() for name in feats_aligned.columns},
        n=len(feats_aligned),
    )

    # Fitness
    fitness_config = FitnessConfig(min_trades=args.min_trades)
    fitness_fn = make_fitness_fn(df_aligned, ctx, fitness_config)

    # Evolución
    evo_config = EvolutionConfig(
        population_size=args.population,
        n_generations=args.generations,
        tournament_size=args.tournament,
        n_elites=args.elites,
        crossover_rate=args.crossover,
        mutation_rate=args.mutation,
        seed=args.seed,
    )

    t0 = time.time()
    result = evolve(fitness_fn, evo_config)
    elapsed = time.time() - t0

    # Guardar
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_individual(result.best_individual, out_dir / "best_individual.json")

    summary = {
        "best_fitness": result.best_fitness,
        "best_size": result.best_individual.size,
        "best_depth": result.best_individual.depth,
        "best_conditions": result.best_individual.n_conditions(),
        "best_features": sorted(result.best_individual._collect_features()),
        "elapsed_seconds": elapsed,
        "history_best": result.history_best,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nBest fitness: {result.best_fitness:.4f}")
    print(f"Best size: {result.best_individual.size}")
    print(f"Best depth: {result.best_individual.depth}")
    print(f"Best features: {sorted(result.best_individual._collect_features())}")
    print(f"Tiempo: {elapsed:.2f}s")
    print(f"Guardado en: {out_dir}")


if __name__ == "__main__":
    main()