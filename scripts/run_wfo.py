"""
Script: Walk-Forward Optimization sobre datos de 5 años.

Reserva el 10% final como final test intocable.
Corre WFO sobre el 90% restante.

Modos:
- --dry-run: solo genera y reporta folds, no corre GP.

Uso:
    python -m scripts.run_wfo --csv data_files/EURUSD_X_15m_5y.csv \
        --output experiments_output/wfo_5y_baseline \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data.cleaner import load_and_clean
from data.splitter import SplitConfig, describe_split, split_development_final
from experiments.tracker import ExperimentTracker
from fitness.objective import FitnessConfig
from gp.evolution import EvolutionConfig
from validation.folds import WFConfig, generate_folds
from validation.walk_forward import run_walk_forward
from config.settings import SUPPORTED_SYMBOLS, infer_symbol_from_path
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

def main():
    parser = argparse.ArgumentParser(description="WFO sobre 5 años de datos.")
    parser.add_argument("--symbol", default=None, help="Símbolo. Si no se pasa, se infiere del nombre del CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracker", default="experiments_output/tracker.jsonl")
    parser.add_argument("--final-test-ratio", type=float, default=0.10)
    parser.add_argument("--final-test-path", default=None)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--is-ratio", type=float, default=0.8)
    parser.add_argument("--min-is-size", type=int, default=20000)
    parser.add_argument("--min-oos-size", type=int, default=5000)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo generar folds y reportar, sin correr GP.")
    parser.add_argument("--is-months", type=int, default=12)
    parser.add_argument("--oos-months", type=int, default=3)
    args = parser.parse_args()

    if args.symbol is None:
        try:
            symbol = infer_symbol_from_path(args.csv)
        except ValueError as e:
            parser.error(str(e))
    else:
        symbol = args.symbol
        if symbol not in SUPPORTED_SYMBOLS:
            parser.error(f"--symbol {symbol} no está en {SUPPORTED_SYMBOLS}")

    print(f"Símbolo: {symbol}")
    # ------------------------------------------------------------------
    # 1. Cargar y limpiar
    # ------------------------------------------------------------------
    df, report = load_and_clean(args.csv)
    print(report.summary())
    print(f"\nTotal filas: {len(df)}")
    print(f"Rango: {df.index[0]} .. {df.index[-1]}")

    # ------------------------------------------------------------------
    # 2. Split development / final test
    # ------------------------------------------------------------------
    final_test_path = (
        Path(args.final_test_path)
        if args.final_test_path
        else Path(args.output) / "final_test.csv"
    )
    split_config = SplitConfig(
        final_test_ratio=args.final_test_ratio,
        save_final_test=True,
        final_test_path=final_test_path,
    )
    development, final_test = split_development_final(df, split_config)
    print()
    print(describe_split(development, final_test))

    # Guardar el timestamp límite del development (para salvaguarda)
    max_dev_ts = development.index[-1]

    # ------------------------------------------------------------------
    # 3. Generar folds
    # ------------------------------------------------------------------
    VELAS_POR_MES = 2100 # 2100 para 15m  |  128 por 4H

    wf_config = WFConfig(
        is_size=args.is_months * VELAS_POR_MES,
        oos_size=args.oos_months * VELAS_POR_MES,
        max_timestamp=max_dev_ts,
    )

    print("\nGenerando folds...")
    folds = generate_folds(development, wf_config)
    print(f"Folds generados: {len(folds)}\n")
    for f in folds:
        print(f"  {f}")

    if args.dry_run:
        print("\n--dry-run activado. No se corre GP.")
        return

    # ------------------------------------------------------------------
    # 4. Correr WFO
    # ------------------------------------------------------------------
    evo_config = EvolutionConfig(
        population_size=args.population,
        n_generations=args.generations,
        seed=args.seed,
    )
    fitness_config = FitnessConfig(
        symbol=symbol,
        min_trades=args.min_trades)

    t0 = time.time()
    result = run_walk_forward(
        development,
        wf_config,
        evo_config,
        fitness_config,
        initial_capital=10_000.0,
        symbol=symbol,
        verbose=True,
    )
    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # 5. Guardar resultados y registrar
    # ------------------------------------------------------------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.equity_stitched is not None:
        result.equity_stitched.to_csv(out_dir / "equity_stitched.csv")

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result.metrics_aggregated, f, indent=2, default=str)

    tracker = ExperimentTracker(Path(args.tracker))
    tracker.register(
        symbol=symbol,
        config={
            "csv": args.csv,
            "is_ratio": args.is_ratio,
            "population": args.population,
            "generations": args.generations,
            "min_trades": args.min_trades,
            "seed": args.seed,
        },
        is_metrics={},
        oos_metrics=result.metrics_aggregated,
        notes=f"WFO 5y, {len(result.fold_results)} folds",
    )

    print(f"\nWFO completado en {elapsed:.2f}s")
    print(f"Total experimentos registrados: {tracker.n_experiments}")
    print("\nMétricas agregadas:")
    for k, v in result.metrics_aggregated.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nErrores de evaluación: {result.metrics_aggregated.get('errors_evaluate', 0)}")
    print(f"Errores de backtest: {result.metrics_aggregated.get('errors_backtest', 0)}")

if __name__ == "__main__":
    main()