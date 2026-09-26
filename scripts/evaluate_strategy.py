"""
Script: evalúa un individuo guardado sobre otros datos (cross-pair, cross-TF).

Uso:
    python -m scripts.evaluate_strategy --individual experiments_output/gp_run_001/best_individual.json \
        --csvs data_files/GBPUSD_X_15m_60d.csv data_files/USDJPY_X_15m_60d.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.cleaner import load_and_clean
from gp.serialization import load_individual
from validation.robustness import (
    cross_pair_test,
    cross_timeframe_test,
    summarize_robustness,
)


def main():
    parser = argparse.ArgumentParser(description="Evalúa un individuo sobre otros datos.")
    parser.add_argument("--individual", required=True)
    parser.add_argument("--csvs", nargs="+", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    individual = load_individual(args.individual)

    datasets = {}
    for csv_path in args.csvs:
        label = Path(csv_path).stem
        df, _ = load_and_clean(csv_path)
        datasets[label] = df

    results = cross_pair_test(individual, datasets)
    summary = summarize_robustness(results)

    for r in results:
        print(f"\n{r.label}:")
        print(f"  n_bars={r.n_bars}, trades={r.n_trades}")
        print(f"  return={r.total_return:+.4%}, sharpe={r.sharpe:.3f}, maxDD={r.max_dd:.2%}")
        print(f"  signals={r.signals_distribution}")

    print(f"\nResumen: {json.dumps(summary, indent=2, default=str)}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "individual": args.individual,
                    "results": [r.__dict__ for r in results],
                    "summary": summary,
                },
                f, indent=2, default=str,
            )
        print(f"\nGuardado en: {out}")


if __name__ == "__main__":
    main()