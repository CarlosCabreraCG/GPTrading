"""
Análisis post-WFO del experimento de oro.

Uso:
    python check_gold.py
"""

import json
from pathlib import Path

# Ajusta esta ruta a donde tengas el output del WFO de oro
OUTPUT_DIR = Path("experiments_output/wfo_42seed")

# Cargar métricas
with (OUTPUT_DIR / "metrics.json").open() as f:
    m = json.load(f)

print("=" * 60)
print("MÉTRICAS AGREGADAS (con decimales completos)")
print("=" * 60)
for k, v in m.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.10f}")
    else:
        print(f"  {k}: {v}")

# Si guardaste los fold_results, aquí podrías iterar sobre ellos.
# Pero el run_wfo.py actual solo guarda metrics.json, no fold_results.
# Para analizar fold por fold, hay que modificar run_wfo.py o correr
# el WFO desde un script Python que devuelva el WalkForwardResult.