import numpy as np
import time
from data.cleaner import load_and_clean
from features.engine import compute_features, drop_warmup_rows
from fitness.objective import FitnessConfig
from gp.evolution import EvolutionConfig
from validation.folds import WFConfig, generate_folds
from validation.walk_forward import run_walk_forward

df, _ = load_and_clean("data_files/EURUSD_X_15m_60d.csv")

# Con 5648 velas, ¿cuántos folds caben?
wf_config = WFConfig(
    is_ratio=0.7, oos_ratio=0.3,
    min_is_size=500, min_oos_size=200,
)

folds = generate_folds(df, wf_config)
print(f"Folds generados: {len(folds)}")
for f in folds:
    print(f"  {f}")

if len(folds) > 0:
    evo_config = EvolutionConfig(
        population_size=30,
        n_generations=8,
        tournament_size=4,
        n_elites=2,
        crossover_rate=0.8,
        mutation_rate=0.2,
        seed=42,
    )
    fitness_config = FitnessConfig(
        min_trades=5,
        max_turnover_ratio=0.5,
        primary_metric="sharpe",
    )

    t0 = time.time()
    result = run_walk_forward(
        df, wf_config, evo_config, fitness_config,
        initial_capital=10_000.0,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\nWFO completado en {elapsed:.2f}s")
    print(f"\nMétricas agregadas:")
    for k, v in result.metrics_aggregated.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")