"""
Corre los 4 WFOs del baseline en serie.

Los experimentos se ejecutan uno tras otro. Si uno falla, el script
aborta y no corre los siguientes.

Uso:
    python -m scripts.run_all_wfo
    python -m scripts.run_all_wfo --only gold
    python -m scripts.run_all_wfo --skip gold
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


# Configuración de cada experimento
EXPERIMENTS = [
    {
        "name": "gold",
        "csv": "data_files/XAUUSD_X_15m_5y.csv",
        "output": "experiments_output/wfo_gold_v2",
    },
    {
        "name": "eurusd_15m",
        "csv": "data_files/EURUSD_X_15m_5y.csv",
        "output": "experiments_output/wfo_eurusd_15m_v2",
    },
    {
        "name": "gbpusd_15m",
        "csv": "data_files/GBPUSD_X_15m_5y.csv",
        "output": "experiments_output/wfo_gbpusd_15m_v2",
    },
    {
        "name": "eurusd_h4",
        "csv": "data_files/EURUSD_X_H4_5y.csv",
        "output": "experiments_output/wfo_eurusd_h4_v2",
    },
]

# Parámetros comunes a todos
COMMON_ARGS = [
    "--is-months", "12",
    "--oos-months", "3",
    "--population", "30",
    "--generations", "8",
    "--min-trades", "5",
    "--seed", "42",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="Correr solo este experimento (por nombre).")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Saltar estos experimentos (por nombre).")
    args = parser.parse_args()

    selected = []
    for exp in EXPERIMENTS:
        if args.only and exp["name"] != args.only:
            continue
        if exp["name"] in args.skip:
            continue
        selected.append(exp)

    if not selected:
        print("No hay experimentos seleccionados.")
        sys.exit(1)

    print(f"Experimentos a correr: {[e['name'] for e in selected]}")
    print(f"Tiempo estimado total: ~{len(selected) * 60} minutos\n")

    start_total = time.time()
    results = []

    for i, exp in enumerate(selected, 1):
        print("=" * 70)
        print(f"[{i}/{len(selected)}] {exp['name']}")
        print(f"  CSV:    {exp['csv']}")
        print(f"  Output: {exp['output']}")
        print("=" * 70)

        csv_path = Path(exp["csv"])
        if not csv_path.exists():
            print(f"  ❌ CSV no encontrado: {csv_path}")
            print(f"  Abortando.")
            sys.exit(1)

        cmd = [
            sys.executable, "-m", "scripts.run_wfo",
            "--csv", exp["csv"],
            "--output", exp["output"],
            *COMMON_ARGS,
        ]

        t0 = time.time()
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0

        status = "OK" if result.returncode == 0 else f"FALLO (exit {result.returncode})"
        results.append((exp["name"], status, elapsed))

        if result.returncode != 0:
            print(f"\n❌ {exp['name']} falló. Abortando el resto.")
            break

        print(f"\n✅ {exp['name']} completado en {elapsed/60:.1f} min\n")

    total_elapsed = time.time() - start_total

    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    for name, status, elapsed in results:
        print(f"  {name:20s} {status:10s} {elapsed/60:6.1f} min")
    print(f"\nTiempo total: {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()