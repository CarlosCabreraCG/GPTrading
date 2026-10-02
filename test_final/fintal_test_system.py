# Script rápido de sanity check
from data.cleaner import load_and_clean
from data.splitter import SplitConfig, split_development_final
from validation.folds import WFConfig, generate_folds
from fitness.objective import FitnessConfig, make_fitness_fn
from gp.evolution import EvolutionConfig, evolve
from features.engine import compute_features, drop_warmup_rows
from gp.evaluator import EvalContext

df, _ = load_and_clean("data_files/EURUSD_X_15m_5y.csv")
dev, _ = split_development_final(df, SplitConfig(final_test_ratio=0.10, save_final_test=False))

wf_config = WFConfig(is_size=12*2100, oos_size=3*2100, max_timestamp=dev.index[-1])
folds = generate_folds(dev, wf_config)
fold = folds[0]

feats = compute_features(fold.is_df)
df_is, feats_is = drop_warmup_rows(fold.is_df, feats)
ctx = EvalContext(
    features={n: feats_is[n].to_numpy() for n in feats_is.columns},
    n=len(feats_is),
)

fitness_fn = make_fitness_fn(df_is, ctx, FitnessConfig(min_trades=5))
evo_config = EvolutionConfig(population_size=30, n_generations=1, seed=42)
result = evolve(fitness_fn, evo_config)

print(f"Best fitness gen 0: {result.best_fitness}")
finite = [f for f in result.history_best if f > float("-inf")]
print(f"Generaciones con best finito: {len(finite)}/1")