"""
Diagnóstico de UN fold de WFO (no es un entrenamiento completo).

Corre el GP sobre el IS de un fold, instrumentando cada evaluación de fitness
sin tocar `evolve`, y registra:
- Dinámica por generación: % de fitness -inf (y su motivo), señales distintas,
  clones genotípicos, mejor/media, tamaño medio.
- Comportamiento de la gramática: uso de nodos/features, condiciones GT/LT
  degeneradas (casi siempre verdaderas o falsas), pares de escalas.
- Reglas de cruce con close (baseline manual) para decidir si hace falta una
  métrica/feature adicional.

NO toca el final test (usa solo el development) ni modifica código del GP.
Uso:
    python -m scripts.diag_fold --csv data_files/xauusd_15m_5y.csv \
        --fold 5 --population 30 --generations 8 --seed 42 \
        --output experiments_output/diag_fold/p30g8
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from config.settings import infer_symbol_from_path
from data.cleaner import load_and_clean
from data.splitter import SplitConfig, split_development_final
from experiments.tracker import ExperimentTracker
from features.engine import resolve_features
from fitness.metrics import sharpe, turnover
from fitness.objective import FitnessConfig, make_fitness_fn
from gp.evaluator import TreeNode, evaluate
from gp.evolution import EvolutionConfig, evolve
from gp.grammar import build_grammar
from validation.folds import WFConfig, generate_folds
from validation.walk_forward import _build_context_and_split


def tree_str(t: TreeNode) -> str:
    """Representación textual canónica de un árbol (para detectar clones)."""
    if t.is_feature_terminal() and t.feature_name:
        return t.feature_name
    if not t.children:
        return t.node.name
    return f"{t.node.name}({','.join(tree_str(c) for c in t.children)})"


def walk(t: TreeNode):
    yield t
    for c in t.children:
        yield from walk(c)


def describe_operand(t: TreeNode) -> str:
    if t.is_feature_terminal() and t.feature_name:
        return t.feature_name
    return "expr"


def reason_for_inf(signals: pd.Series, ohlc, bt_cfg, fcfg: FitnessConfig) -> str:
    """Replica los cortes de fitness/objective.py para etiquetar un -inf."""
    n = len(signals)
    n_active = int((signals != 0).sum())
    if n_active == 0:
        return "flat_total"
    if 1.0 - n_active / n > fcfg.max_flat_ratio:
        return "flat>90%"
    if signals.nunique() == 1:
        return "constante_no_flat"
    res = run_backtest(ohlc, signals, bt_cfg)
    if res.n_trades < fcfg.min_trades:
        return "trades<min"
    if turnover(res.position) / n > fcfg.max_turnover_ratio:
        return "turnover"
    return "otro"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fold", type=int, default=5)
    ap.add_argument("--population", type=int, default=30)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-trades", type=int, default=20)
    ap.add_argument("--fitness-mode", choices=["raw", "alpha"], default="raw")
    ap.add_argument("--max-net-exposure", type=float, default=None)
    ap.add_argument("--max-turnover-ratio", type=float, default=0.15)
    ap.add_argument("--is-months", type=int, default=12)
    ap.add_argument("--oos-months", type=int, default=3)
    ap.add_argument("--bars-per-month", type=int, default=2100)
    ap.add_argument("--tracker", default="experiments_output/tracker.jsonl")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    symbol = infer_symbol_from_path(args.csv)

    df, _ = load_and_clean(args.csv)
    dev, _final = split_development_final(
        df, SplitConfig(final_test_ratio=0.10, save_final_test=False)
    )  # el final test se descarta aquí, no se usa
    names = resolve_features(dev.columns)
    grammar = build_grammar(names)

    wf = WFConfig(
        is_size=args.is_months * args.bars_per_month,
        oos_size=args.oos_months * args.bars_per_month,
        max_timestamp=dev.index[-1],
    )
    folds = generate_folds(dev, wf)
    fold = folds[args.fold]
    ctx_is, df_is, ctx_oos, df_oos = _build_context_and_split(
        fold.is_df, fold.oos_df, names
    )
    print(f"Fold {fold.index}: IS {len(df_is)} velas, OOS {len(df_oos)} velas")

    fcfg = FitnessConfig(
        symbol=symbol, min_trades=args.min_trades,
        max_turnover_ratio=args.max_turnover_ratio,
        fitness_mode=args.fitness_mode, max_abs_net_exposure=args.max_net_exposure,
    )
    bt_cfg = BacktestConfig(symbol=symbol, costs=fcfg.costs, size=fcfg.size)
    base_fn = make_fitness_fn(df_is, ctx_is, fcfg)

    # ---------------- instrumentación del fitness ----------------
    records: list[dict] = []
    pop_size = args.population

    def spy(ind):
        f = base_fn(ind)
        idx = len(records)
        sig = pd.Series(ind.evaluate(ctx_is), index=df_is.index)
        rec = {
            "gen": idx // pop_size,
            "fit": f,
            "size": ind.size,
            "n_cond": ind.n_conditions(),
            "tree": tree_str(ind.root),
            "sig_hash": hash(sig.to_numpy().tobytes()),
            "reason": None,
            "ind": ind,
        }
        if not np.isfinite(f):
            rec["reason"] = reason_for_inf(sig, df_is, bt_cfg, fcfg)
        records.append(rec)
        return f

    evo = EvolutionConfig(
        population_size=args.population,
        n_generations=args.generations,
        seed=args.seed + fold.index,  # misma derivación que el WFO
    )
    t0 = time.time()
    res = evolve(fitness_fn=spy, config=evo, grammar=grammar)
    elapsed = time.time() - t0
    print(f"Evolución: {elapsed:.0f}s, {len(records)} evaluaciones")

    # ---------------- dinámica por generación ----------------
    per_gen = []
    seen_trees: set[str] = set()
    for g in range(args.generations):
        rs = [r for r in records if r["gen"] == g]
        fin = [r for r in rs if np.isfinite(r["fit"])]
        trees = [r["tree"] for r in rs]
        new_trees = [t for t in set(trees) if t not in seen_trees]
        seen_trees.update(trees)
        per_gen.append({
            "gen": g,
            "n": len(rs),
            "pct_inf": 1 - len(fin) / len(rs),
            "reasons": dict(collections.Counter(r["reason"] for r in rs if r["reason"])),
            "distinct_trees": len(set(trees)),
            "distinct_signals": len({r["sig_hash"] for r in rs}),
            "distinct_signals_finite": len({r["sig_hash"] for r in fin}),
            "new_trees_vs_history": len(new_trees),
            "best": max((r["fit"] for r in rs), default=float("-inf")),
            "mean_finite": float(np.mean([r["fit"] for r in fin])) if fin else None,
            "mean_size": float(np.mean([r["size"] for r in rs])),
            "mean_n_cond": float(np.mean([r["n_cond"] for r in rs])),
            "frac_zero_cond": float(np.mean([r["n_cond"] == 0 for r in rs])),
        })

    # ---------------- comportamiento de la gramática ----------------
    node_use_all = collections.Counter()
    node_use_fin = collections.Counter()
    feat_use_fin = collections.Counter()
    feat_use_all = collections.Counter()
    cond_rows = []
    seen_cond_trees: set[str] = set()
    for r in records:
        if r["tree"] in seen_cond_trees:
            continue  # cada genotipo una sola vez
        seen_cond_trees.add(r["tree"])
        fin = np.isfinite(r["fit"])
        for n in walk(r["ind"].root):
            node_use_all[n.node.name] += 1
            if fin:
                node_use_fin[n.node.name] += 1
            if n.is_feature_terminal() and n.feature_name:
                feat_use_all[n.feature_name] += 1
                if fin:
                    feat_use_fin[n.feature_name] += 1
            if n.node.name in ("GT", "LT"):
                truth = np.asarray(evaluate(n, ctx_is), dtype=bool)
                ratio = float(truth.mean())
                cond_rows.append({
                    "op": n.node.name,
                    "left": describe_operand(n.children[0]),
                    "right": describe_operand(n.children[1]),
                    "true_ratio": ratio,
                    "degenerate": ratio < 0.02 or ratio > 0.98,
                    "finite_ind": fin,
                })
    n_cond = len(cond_rows)
    n_degen = sum(c["degenerate"] for c in cond_rows)
    pair_counter = collections.Counter(
        (c["left"], c["right"]) for c in cond_rows if c["left"] != "expr" and c["right"] != "expr"
    )
    degen_pairs = collections.Counter(
        (c["left"], c["right"]) for c in cond_rows
        if c["degenerate"] and c["left"] != "expr" and c["right"] != "expr"
    )
    same_feat = sum(1 for c in cond_rows if c["left"] == c["right"] and c["left"] != "expr")

    # ---------------- mejor individuo: IS y OOS ----------------
    best = res.best_individual
    sig_is = pd.Series(best.evaluate(ctx_is), index=df_is.index)
    sig_oos = pd.Series(best.evaluate(ctx_oos), index=df_oos.index)
    r_is = run_backtest(df_is, sig_is, bt_cfg)
    r_oos = run_backtest(df_oos, sig_oos, bt_cfg)
    best_info = {
        "tree": tree_str(best.root),
        "fitness_is": res.best_fitness,
        "sharpe_is": sharpe(r_is.equity, symbol),
        "sharpe_oos": sharpe(r_oos.equity, symbol),
        "trades_is": r_is.n_trades,
        "trades_oos": r_oos.n_trades,
        "found_in_gen": next(
            (r["gen"] for r in records if r["fit"] == res.best_fitness), None
        ),
    }

    # ---------------- reglas de cruce con close (baseline manual) ----------------
    # Hipótesis: ¿hace falta una feature adicional de "cruce" o basta GT/LT?
    cross_rules = {}

    def build_rule(kind, x, k, index):
        s = pd.Series(np.sign(x), index=index).fillna(0)
        if kind == "nivel":
            return s.astype(int)
        # cruce: la señal solo existe k velas tras cambiar de signo; si no, FLAT
        prev = s.shift(1).fillna(0)
        crossed = (s != prev) & (s != 0)
        age = (~crossed).astype(int).groupby(crossed.cumsum()).cumsum()
        return s.where(age < k, 0).astype(int)

    RULES = {
        "nivel_close_sma20 (long si close>SMA20, short si <)": ("nivel", "close_over_sma_20", 0),
        "nivel_close_ema20": ("nivel", "close_over_ema_20", 0),
        "cruce_close_sma20_hold4": ("cruce", "close_over_sma_20", 4),
        "cruce_close_sma20_hold16": ("cruce", "close_over_sma_20", 16),
        "cruce_close_ema20_hold16": ("cruce", "close_over_ema_20", 16),
    }
    for name, (kind, feat, k) in RULES.items():
        s_is = build_rule(kind, ctx_is.features[feat], k, df_is.index)
        s_oos = build_rule(kind, ctx_oos.features[feat], k, df_oos.index)
        r = run_backtest(df_is, s_is, bt_cfg)
        ro = run_backtest(df_oos, s_oos, bt_cfg)
        to = turnover(r.position) / len(s_is)
        cross_rules[name] = {
            "sharpe_is": sharpe(r.equity, symbol), "trades_is": r.n_trades,
            "turnover_ratio_is": to, "passes_turnover_cap": to <= fcfg.max_turnover_ratio,
            "sharpe_oos": sharpe(ro.equity, symbol), "trades_oos": ro.n_trades,
        }

    # features "pseudo-cero" (actúan como constante 0 al compararse)
    feats_is = pd.DataFrame({n: ctx_is.features[n] for n in names})
    pseudo_zero = {
        n: {"std": float(feats_is[n].std()), "mean": float(feats_is[n].mean()),
            "frac_pos": float((feats_is[n] > 0).mean())}
        for n in names
        if abs(feats_is[n].mean()) < 0.25 * feats_is[n].std() + 1e-12
    }
    # ¿qué fracción de los GT/LT entre dos features es "cruce con cero" de facto?
    zero_like = {"return_1", "log_return_1", "return_5", "log_return_5",
                 "return_20", "log_return_20", "roc_10"}
    cond_vs_zero = sum(
        1 for c in cond_rows
        if (c["right"] in zero_like and c["left"] not in ("expr",))
        or (c["left"] in zero_like and c["right"] not in ("expr",))
    )

    report = {
        "meta": {
            "csv": args.csv, "symbol": symbol, "fold": fold.index,
            "is_range": [str(df_is.index[0]), str(df_is.index[-1])],
            "oos_range": [str(df_oos.index[0]), str(df_oos.index[-1])],
            "population": args.population, "generations": args.generations,
            "seed": args.seed, "fold_seed": evo.seed,
            "fitness_mode": args.fitness_mode, "max_net_exposure": args.max_net_exposure,
            "max_turnover_ratio": args.max_turnover_ratio,
            "n_evals": len(records), "elapsed_s": elapsed,
            "n_features": len(names),
        },
        "per_gen": per_gen,
        "best": best_info,
        "grammar": {
            "node_use_all_unique_genotypes": dict(node_use_all.most_common()),
            "node_use_finite_genotypes": dict(node_use_fin.most_common()),
            "feature_use_finite": dict(feat_use_fin.most_common()),
            "feature_use_all": dict(feat_use_all.most_common()),
            "n_conditions_unique": n_cond,
            "n_conditions_degenerate": n_degen,
            "same_feature_both_sides": same_feat,
            "conditions_vs_zero_like_feature": cond_vs_zero,
            "top_pairs": [[list(k), v] for k, v in pair_counter.most_common(10)],
            "top_degenerate_pairs": [[list(k), v] for k, v in degen_pairs.most_common(10)],
        },
        "cross_rules": cross_rules,
        "pseudo_zero_features": pseudo_zero,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Informe: {out / 'report.json'}")

    ExperimentTracker(symbol=symbol, storage_path=Path(args.tracker)).register(
        symbol=symbol,
        n_trials=len(records),
        config={
            "kind": "diag_fold", "csv": args.csv, "fold": fold.index,
            "population": args.population, "generations": args.generations,
            "seed": args.seed,
            "fitness_mode": args.fitness_mode, "max_net_exposure": args.max_net_exposure,
            "max_turnover_ratio": args.max_turnover_ratio,
        },
        is_metrics={"sharpe_is": best_info["sharpe_is"], "fitness_is": res.best_fitness},
        oos_metrics={"sharpe_oos": best_info["sharpe_oos"], "trades_oos": best_info["trades_oos"]},
        notes="Diagnóstico de un solo fold (no es un WFO completo; no usar para conclusiones de edge).",
    )


if __name__ == "__main__":
    main()
