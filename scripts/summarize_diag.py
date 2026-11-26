"""
Agrega los report.json generados por scripts/diag_fold.py (matriz fold x semilla).

Uso:
    python -m scripts.summarize_diag --dir experiments_output/diag_fold/matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.dir).glob("*/report.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        pg = r["per_gen"]
        g = r["grammar"]
        first_viable = next((x["gen"] for x in pg if x["pct_inf"] < 1.0), None)
        last = pg[-1]
        best_gen = r["best"]["found_in_gen"]
        rows.append({
            "fold": r["meta"]["fold"], "seed": r["meta"]["seed"],
            "inf_g0": pg[0]["pct_inf"], "first_viable": first_viable,
            "inf_mean": float(np.mean([x["pct_inf"] for x in pg])),
            "trees_peak": max(x["distinct_trees"] for x in pg),
            "trees_last": last["distinct_trees"],
            "sigfin_last": last["distinct_signals_finite"],
            "best_gen": best_gen,
            "fit_is": r["best"]["fitness_is"], "sharpe_is": r["best"]["sharpe_is"],
            "sharpe_oos": r["best"]["sharpe_oos"], "trades_is": r["best"]["trades_is"],
            "degen": g["n_conditions_degenerate"] / max(g["n_conditions_unique"], 1),
            "samefeat": g["same_feature_both_sides"] / max(g["n_conditions_unique"], 1),
            "zero_like": g["conditions_vs_zero_like_feature"] / max(g["n_conditions_unique"], 1),
        })

    hdr = ("fold seed  inf0 1stOK inf% treesPk treesEnd sigFin bestGen "
           "fitIS shIS shOOS trIS degen% same% zero%")
    print(hdr)
    for x in rows:
        print(f"{x['fold']:>4} {x['seed']:>5} {x['inf_g0']:.2f} {str(x['first_viable']):>5} "
              f"{x['inf_mean']:.2f} {x['trees_peak']:>7} {x['trees_last']:>7} {x['sigfin_last']:>6} "
              f"{str(x['best_gen']):>7} {x['fit_is']:+.2f} {x['sharpe_is']:+.2f} "
              f"{x['sharpe_oos']:+.2f} {x['trades_is']:>4} {x['degen']:.0%} {x['samefeat']:.0%} {x['zero_like']:.0%}")
    if rows:
        print("\nMedias:")
        for k in ("inf_g0", "inf_mean", "trees_peak", "trees_last", "sigfin_last",
                  "fit_is", "sharpe_is", "sharpe_oos", "degen", "samefeat", "zero_like"):
            print(f"  {k:12s} {np.mean([x[k] for x in rows]):+.3f}")
        # Sharpe IS vs OOS: degradación media
        d = [x["sharpe_is"] - x["sharpe_oos"] for x in rows]
        print(f"  degradación Sharpe IS-OOS media {np.mean(d):+.2f} "
              f"(n={len(rows)}, OOS>0 en {sum(x['sharpe_oos'] > 0 for x in rows)})")


if __name__ == "__main__":
    main()
