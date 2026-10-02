import numpy as np
import time
from data.cleaner import load_and_clean
from features.engine import compute_features, drop_warmup_rows
from gp.evaluator import EvalContext
from gp.evolution import EvolutionConfig, evolve
from fitness.objective import FitnessConfig, make_fitness_fn

df, _ = load_and_clean("data_files/EURUSD_X_15m_60d.csv")
feats = compute_features(df)
df_aligned, feats_aligned = drop_warmup_rows(df, feats)

ctx = EvalContext(
    features={name: feats_aligned[name].to_numpy() for name in feats_aligned.columns},
    n=len(feats_aligned),
)

fitness_config = FitnessConfig(
    min_trades=10,
    max_turnover_ratio=0.3,
    primary_metric="sharpe",
)
fitness_fn = make_fitness_fn(df_aligned, ctx, fitness_config)

config = EvolutionConfig(
    population_size=50,
    n_generations=15,
    tournament_size=5,
    n_elites=2,
    crossover_rate=0.8,
    mutation_rate=0.2,
    seed=42,
)

t0 = time.time()
result = evolve(fitness_fn, config)
elapsed = time.time() - t0

print(f"Evolución completada en {elapsed:.2f}s")
print(f"Best fitness: {result.best_fitness:.4f}")
print(f"Best size: {result.best_individual.size}")
print(f"Best depth: {result.best_individual.depth}")
print(f"Best conditions: {result.best_individual.n_conditions()}")
print(f"Best features: {result.best_individual._collect_features()}")
print(f"\nHistoria best: {[f'{x:.4f}' for x in result.history_best]}")
print(f"Historia mean: {[f'{x:.4f}' for x in result.history_mean]}")

sig = result.best_individual.evaluate(ctx)
unique, counts = np.unique(sig, return_counts=True)
print(f"\nDistribución de señales: {dict(zip(unique.tolist(), counts.tolist()))}")