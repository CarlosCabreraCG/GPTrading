"""
Orquestador mínimo.

Ejecuta el pipeline completo con una configuración por defecto.

Uso:
    python main.py --csv data_files/EURUSD_X_15m_60d.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Pipeline completo trading_gp.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="experiments_output")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # Paso 1: WFO
    print("=" * 60)
    print("PASO 1: Walk-Forward Optimization")
    print("=" * 60)
    wfo_out = out_base / f"wfo_seed{args.seed}"
    subprocess.run(
        [
            sys.executable, "-m", "scripts.run_wfo",
            "--csv", args.csv,
            "--output", str(wfo_out),
            "--population", str(args.population),
            "--generations", str(args.generations),
            "--seed", str(args.seed),
        ],
        check=True,
    )

    print("\n" + "=" * 60)
    print("Pipeline completado.")
    print(f"Resultados en: {out_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()