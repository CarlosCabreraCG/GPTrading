"""Diagnóstico read-only del GP (ver docs/ANALISIS_GP.md).
Uso (desde la raíz): python -W ignore docs/diag_gp.py "$(pwd)"
Tarda ~1-2 min. No modifica nada."""
import sys, collections, time
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd

from data.cleaner import load_and_clean
from data.splitter import SplitConfig, split_development_final
from features.engine import compute_features, resolve_features
from gp.grammar import build_grammar
from gp.individual import random_individual
from gp.evaluator import EvalContext
from gp.types import GPType
from config.settings import DEFAULT_GP_LIMITS
from fitness.objective import FitnessConfig, make_fitness_fn
from fitness.metrics import sharpe
from validation.walk_forward import _sharpe_ci, _buy_and_hold_equity, _build_context_and_split
from validation.folds import WFConfig, generate_folds

df, _ = load_and_clean(sys.argv[1] + "/data_files/xauusd_15m_5y.csv")
dev, _ft = split_development_final(df, SplitConfig(final_test_ratio=0.10, save_final_test=False, final_test_path=None))
print("dev rows", len(dev), dev.index[0], dev.index[-1])

names = resolve_features(dev.columns)
grammar = build_grammar(names)
print("n features", len(names))

# ---------- 1. Escalas de features
feats = compute_features(dev, names=names).dropna()
print("\n== Escala de features (std / p1 / p99) ==")
for n in names:
    s = feats[n]
    print(f"{n:22s} std={s.std():10.5f} p1={s.quantile(.01):10.5f} p99={s.quantile(.99):10.5f} inf={int(np.isinf(s).sum())}")

# ---------- 2. Estadística de árboles aleatorios
rng = np.random.default_rng(0)
N = 3000
roots = collections.Counter(); sizes = []; depths = []; ncond = []
for _ in range(N):
    ind = random_individual(grammar, DEFAULT_GP_LIMITS, rng)
    roots[ind.root.node.name] += 1
    sizes.append(ind.size); depths.append(ind.depth); ncond.append(ind.n_conditions())
print("\n== Árboles aleatorios (N=3000) ==")
print("raíz:", {k: round(v / N, 3) for k, v in roots.items()})
print("size mean/median/max", np.mean(sizes), np.median(sizes), max(sizes))
print("depth mean", np.mean(depths), " n_conditions mean", np.mean(ncond))
print("frac con 0 condiciones (señal constante):", np.mean(np.array(ncond) == 0))

# ---------- 3. Fitness en un fold real
cfg = WFConfig(is_size=12 * 2100, oos_size=3 * 2100, max_timestamp=dev.index[-1])
folds = generate_folds(dev, cfg)
fold = folds[5]
ctx_is, df_is, ctx_oos, df_oos = _build_context_and_split(fold.is_df, fold.oos_df, names)
fcfg = FitnessConfig(symbol="XAUUSD")
ffn = make_fitness_fn(df_is, ctx_is, fcfg)
rng = np.random.default_rng(7)
fits = []; sigsets = set(); ninf_reason = collections.Counter()
t0 = time.time()
inds = [random_individual(grammar, DEFAULT_GP_LIMITS, rng) for _ in range(120)]
for ind in inds:
    s = ind.evaluate(ctx_is)
    vals, cnt = np.unique(s, return_counts=True)
    sigsets.add(hash(s.tobytes()))
    f = ffn(ind); fits.append(f)
    if not np.isfinite(f):
        act = (s != 0).mean(); turn = (np.diff(s) != 0).mean()
        if act == 0: ninf_reason["flat_total"] += 1
        elif len(vals) == 1: ninf_reason["constante_no_flat"] += 1
        elif turn > 0.15: ninf_reason["turnover>0.15"] += 1
        elif (1 - act) > 0.9: ninf_reason["flat>0.9"] += 1
        else: ninf_reason["min_trades/otros"] += 1
print(f"\n== Fitness de 120 individuos aleatorios en fold 5 (IS) [{time.time()-t0:.1f}s] ==")
fa = np.array(fits)
print("frac -inf:", np.mean(~np.isfinite(fa)), dict(ninf_reason))
print("señales distintas:", len(sigsets), "/ 120")
fin = fa[np.isfinite(fa)]
print("fitness finitos: n=", len(fin), "mean", fin.mean() if len(fin) else None, "max", fin.max() if len(fin) else None)

# ---------- 4. Bug B&H sharpe
bh = _buy_and_hold_equity(dev.iloc[: 17 * 6300], "XAUUSD")
from config.settings import bars_per_year
N17 = 17 * 6300
print("\n== B&H Sharpe ==")
print("bh_sharpe calculado por el código:", sharpe(bh, "XAUUSD"))
print("sqrt(bars_per_year/N) (solo depende de N):", np.sqrt(bars_per_year("XAUUSD", 900) / len(bh)))
dev2 = dev.copy(); dev2["close"] = dev2["close"].iloc[::-1].values  # otro camino de precios, mismo signo de retorno
dev2["open"] = dev2["open"].iloc[0]
bh2 = _buy_and_hold_equity(dev2.iloc[: 17 * 6300], "XAUUSD")
print("bh_sharpe con camino de precios distinto, mismo signo:", sharpe(bh2, "XAUUSD"))

# ---------- 5. IC del Sharpe
rets = np.random.default_rng(1).normal(0.0, 0.0006, 107100)  # ruido puro
eq = pd.Series(10000 * np.cumprod(1 + rets), index=dev.index[:107100])
sr, se, lo, hi = _sharpe_ci(eq, "XAUUSD")
print("\n== IC Sharpe sobre ruido puro (mean=0, n=107100) ==")
print(f"sharpe={sr:.3f} se_codigo={se:.4f} IC=[{lo:.3f},{hi:.3f}]")
ann = bars_per_year("XAUUSD", 900)
print("se teórico anualizado ~", np.sqrt(ann / 107100), " -> IC95 ~ ±", 1.96 * np.sqrt(ann / 107100))
# frecuencia de 'significativo' en 200 ruidos
sig = 0
for i in range(200):
    r = np.random.default_rng(100 + i).normal(0.0, 0.0006, 107100)
    e = pd.Series(10000 * np.cumprod(1 + r), index=dev.index[:107100])
    _, _, l, h = _sharpe_ci(e, "XAUUSD")
    sig += int(l > 0 or h < 0)
print("frac de ruido puro declarado 'significativo':", sig / 200)
