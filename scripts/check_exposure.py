"""
¿Los mejores individuos de scripts/diag_fold.py son estrategias direccionales
reales o casi "comprar y mantener"?

Para cada report.json de una matriz fold x semilla reconstruye el mejor individuo
(desde la cadena `best.tree`), y compara sobre IS y OOS: exposición neta, nº de
transiciones y Sharpe, contra una señal constante LONG (buy & hold con los mismos
costes, mark-to-market barra a barra) y contra su opuesta (SHORT).

Uso:
    python -m scripts.check_exposure --csv data_files/xauusd_15m_5y.csv \
        --dir experiments_output/diag_fold/matrix
Solo usa el development; no toca el final test.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, run_backtest
from config.settings import infer_symbol_from_path
from data.cleaner import load_and_clean
from data.splitter import SplitConfig, split_development_final
from features.engine import resolve_features
from fitness.metrics import sharpe
from fitness.objective import FitnessConfig
from gp.grammar import build_grammar
from gp.serialization import _build_node_index, dict_to_tree
from validation.folds import WFConfig, generate_folds
from validation.walk_forward import _build_context_and_split

_TOKEN = re.compile(r"[A-Za-z_0-9]+|[(),]")


def parse_tree(s: str) -> dict:
    """Parsea 'IF(GT(a,b),LONG,SHORT)' al dict de gp.serialization."""
    toks = _TOKEN.findall(s)
    pos = 0

    def node() -> dict:
        nonlocal pos
        name = toks[pos]
        pos += 1
        children = []
        if pos < len(toks) and toks[pos] == "(":
            pos += 1
            while True:
                children.append(node())
                if toks[pos] == ",":
                    pos += 1
                    continue
                pos += 1  # ')'
                break
        return {"name": name, "feature_name": None, "children": children, "_raw": name}

    return node()


def mark_features(d: dict, feature_names: set[str]) -> dict:
    """Los terminales de feature se serializan como name=feature, feature_name=feature."""
    if d["name"] in feature_names and not d["children"]:
        d["feature_name"] = d["name"]
    for c in d["children"]:
        mark_features(c, feature_names)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--is-months", type=int, default=12)
    ap.add_argument("--oos-months", type=int, default=3)
    ap.add_argument("--bars-per-month", type=int, default=2100)
    args = ap.parse_args()

    symbol = infer_symbol_from_path(args.csv)
    df, _ = load_and_clean(args.csv)
    dev, _ = split_development_final(df, SplitConfig(final_test_ratio=0.10, save_final_test=False))
    names = resolve_features(dev.columns)
    grammar = build_grammar(names)
    node_index = _build_node_index(grammar)
    folds = generate_folds(dev, WFConfig(
        is_size=args.is_months * args.bars_per_month,
        oos_size=args.oos_months * args.bars_per_month,
        max_timestamp=dev.index[-1],
    ))
    fcfg = FitnessConfig(symbol=symbol)
    bt = BacktestConfig(symbol=symbol, costs=fcfg.costs, size=fcfg.size)

    cache: dict[int, tuple] = {}
    print("fold seed | OOS: long% short% flat% netExp trans  shGP   shB&H  | IS: netExp  shGP   shB&H | OOS precio")
    for p in sorted(Path(args.dir).glob("*/report.json")):
        rep = json.loads(p.read_text(encoding="utf-8"))
        fi, seed = rep["meta"]["fold"], rep["meta"]["seed"]
        if fi not in cache:
            cache[fi] = _build_context_and_split(folds[fi].is_df, folds[fi].oos_df, names)
        ctx_is, df_is, ctx_oos, df_oos = cache[fi]

        d = mark_features(parse_tree(rep["best"]["tree"]), set(names))
        root = dict_to_tree(d, node_index)

        from gp.individual import Individual
        ind = Individual(root=root)

        def stats(df_, ctx):
            sig = pd.Series(ind.evaluate(ctx), index=df_.index).astype(int)
            res = run_backtest(df_, sig, bt)
            const = run_backtest(df_, pd.Series(1, index=df_.index), bt)
            return sig, sharpe(res.equity, symbol), sharpe(const.equity, symbol)

        sig_o, sh_o, bh_o = stats(df_oos, ctx_oos)
        sig_i, sh_i, bh_i = stats(df_is, ctx_is)
        trans_o = int((sig_o.diff().fillna(0) != 0).sum())
        move = df_oos["close"].iloc[-1] / df_oos["open"].iloc[0] - 1
        print(f"{fi:>4} {seed:>5} |      {(sig_o == 1).mean():5.0%} {(sig_o == -1).mean():5.0%} "
              f"{(sig_o == 0).mean():5.0%} {sig_o.mean():+6.2f} {trans_o:>5} {sh_o:+6.2f} {bh_o:+6.2f} "
              f"|     {sig_i.mean():+6.2f} {sh_i:+6.2f} {bh_i:+6.2f} | {move:+.1%}")


if __name__ == "__main__":
    main()
