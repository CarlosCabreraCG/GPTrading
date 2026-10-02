# trading_gp

Sistema de Genetic Programming para descubrir reglas de trading sobre forex.
Prioriza robustez y generalización sobre rendimiento histórico máximo.

## Estructura
trading_gp/
├── config/ Constantes globales (costes, límites GP, splits)
├── data/ Carga, limpieza, split temporal
├── features/ Indicadores y motor de features normalizadas
├── gp/ Genetic Programming tipado (nodos, gramática, evolución)
├── strategy/ Representación de señales
├── backtest/ Motor de backtest causal con costes
├── fitness/ Métricas, complejidad, función objetivo
├── validation/ Walk-Forward, folds, robustness, final test
├── experiments/ Tracker y persistencia de experimentos
├── scripts/ Scripts ejecutables desde CLI
├── tests/ Tests unitarios
└── main.py Orquestador
## Preparación del entorno

```bash
pip install -r requirements.txt    # pandas, numpy, pytest, scipy
```
Flujo típico de trabajo

1. Auditoría del backtester (opcional, pero recomendado antes de confiar en resultados)
```bash
python -m scripts.audit_backtester \
    --csv data_files/EURUSD_X_15m_5y.csv \
    --output experiments_output/audit \
    --n-bars 2000 \
    --period 100
```
Corre el backtester sobre una señal trivial (LONG 100 velas, FLAT 100 velas)
y verifica:

Ejecución al open de t+1 (no al close de la misma vela).

PnL bruto cuadra con entry/exit prices.

Costes aplicados correctamente.

Identidad contable: equity_final = capital + sum(gross) - sum(costes).

Genera bar_log.csv (estado por vela) y trades_audit.csv (auditoría por trade).

2. Walk-Forward Optimization completo
```bash
python -m scripts.run_wfo \
    --csv data_files/EURUSD_X_15m_5y.csv \
    --output experiments_output/wfo_5y_baseline \
    --is-months 12 \
    --oos-months 3 \
    --population 30 \
    --generations 8 \
    --min-trades 5 \
    --seed 42
```
Reserva el 10% final del dataset como final test intocable.

Genera folds rolling: IS = 12 meses, OOS = 3 meses, step = OOS (sin solapamiento).

Entrena GP en cada IS, evalúa el mejor en el OOS correspondiente.

Concatena equity OOS en una curva "stitched".

Registra el experimento en experiments_output/tracker.jsonl.

Tiempo estimado: ~15 min con --population 20 --generations 5, ~1 hora con los defaults.

3. Dry-run de WFO (solo ver folds, sin correr GP)
```bash
python -m scripts.run_wfo 
    --csv data_files/EURUSD_X_15m_5y.csv 
    --output experiments_output/wfo_dryrun 
    --is-months 12 
    --oos-months 3 
    --dry-run
```
Genera y reporta los folds, pero no ejecuta GP. Útil para verificar
que la configuración de WFO es coherente antes de gastar horas.

4. GP simple (sin WFO)
```bash
python -m scripts.run_gp 
    --csv data_files/EURUSD_X_15m_5y.csv 
    --output experiments_output/gp_run_001 
    --population 50 
    --generations 15 
    --seed 42
```
Corre GP sobre el dataset completo (sin separar IS/OOS). Útil para
exploración rápida. No usar para validación seria.

5. Evaluación cross-pair / cross-timeframe
```bash
python -m scripts.evaluate_strategy \
    --individual experiments_output/gp_run_001/best_individual.json \
    --csvs data_files/GBPUSD_X_15m_5y.csv data_files/USDJPY_X_15m_5y.csv
```
Evalúa un individuo ya entrenado sobre otros pares, sin reajustar.
Si funciona sin reoptimizar, hay evidencia de generalización estructural.

6. Pipeline completo (orquestador)
```bash
python main.py 
    --csv data_files/EURUSD_X_15m_5y.csv 
    --output-dir experiments_output 
    --population 30 
    --generations 8 
    --seed 42
```
Llama a run_wfo con la configuración por defecto.

Tests
```bash
python -m pytest tests/ -v -W error::RuntimeWarning
```
Con -W error::RuntimeWarning los warnings de numpy se convierten en errores.
Esto garantiza que no hay overflows silenciosos en los operadores protegidos.

Tests por fase:

test_data.py — carga, limpieza, features temporales.

test_features.py — causalidad (no look-ahead), correctitud de indicadores.

test_nodes.py — operadores protegidos, gramática tipada.

test_individual.py — construcción aleatoria, evaluación.

test_backtester.py — ejecución al open de t+1, contabilidad, costes.

test_fitness.py — métricas, complejidad, función objetivo.

test_wfo.py — generación de folds, WFO completo.

test_robustness.py — cross-pair, final test, tracker.

Parámetros clave
WFConfig (validation/folds.py)
is_size: velas en in-sample (ej. 12 * 2100 = 12 meses).

oos_size: velas en out-of-sample (ej. 3 * 2100 = 3 meses).

step_size: avance entre folds. Default = oos_size (sin solapamiento).

max_timestamp: salvaguarda contra el final test.

Constante útil: VELAS_POR_MES = 2100 (forex ~5 días/semana).

EvolutionConfig (gp/evolution.py)
population_size: individuos por generación.

n_generations: iteraciones evolutivas.

tournament_size: tamaño de torneo para selección.

n_elites: mejores preservados sin cambio.

crossover_rate, mutation_rate.

seed: determinismo.

FitnessConfig (fitness/objective.py)
min_trades: mínimo de operaciones para aceptar individuo.

max_turnover_ratio: máximo de transiciones / velas.

max_flat_ratio: máximo de tiempo en FLAT.

primary_metric: "sharpe" o "sortino".

Pesos de penalización (complejidad, turnover, inactividad).

Resultados y artefactos
Cada ejecución de WFO genera:

text
experiments_output/wfo_<name>/
├── final_test.csv          # Bloque final reservado (no usar hasta el final)
├── equity_stitched.csv     # Curva OOS concatenada
└── metrics.json            # Métricas agregadas

experiments_output/tracker.jsonl   # Registro de todos los experimentos
El tracker.jsonl permite calcular el Deflated Sharpe Ratio sobre
el conjunto completo de experimentos, ajustando por multiple testing.

Final test
El 10% final del dataset se reserva como final test intocable.
No se usa durante el desarrollo. Solo se ejecuta UNA VEZ, sobre la
estrategia congelada, al final del proyecto.

Si se ejecuta antes, el resultado pierde validez estadística.
El lock file en experiments_output/wfo_<name>/final_test.csv sirve
como recordatorio.

Notas sobre datos
Sin volumen (los CSVs de forex retail suelen tener volume=0).

El cleaner detecta gaps pero NO los rellena (evita look-ahead).

Los gaps de fin de semana son normales (~48h).

No redondear decimales: la precisión extra no hace daño.

Velas con rango grande por noticias son parte del mercado y no se filtran.

Decisión de diseño
El sistema optimiza generalización y estabilidad, no rendimiento
histórico máximo. Por eso:

GP tipado restringido a una gramática financiera.

Fitness con penalizaciones de complejidad, turnover e inactividad.

WFO con folds rolling y final test intocable.

Tracker para auditar multiple testing.

Tests que garantizan ausencia de look-ahead.

text
