# trading_gp

Sistema de Genetic Programming (GP) tipado para descubrir reglas de trading sobre forex,
metales, materias primas e índices (`EURUSD`, `GBPUSD`, `XAUUSD`, `XAGUSD`, `WTI`, `BRENT`,
`USA500`, `USATECH`). Prioriza robustez y generalización sobre rendimiento histórico máximo.

> **Estado:** el resultado empírico es negativo *provisional*. Una auditoría posterior
> encontró defectos en la evaluación y en la búsqueda GP. Ver
> [`docs/ANALISIS_GP.md`](docs/ANALISIS_GP.md) y [`CLAUDE.md`](CLAUDE.md).

## Estructura

```text
trading_gp/
├── config/       Constantes globales (costes, límites GP, splits)
├── data/         Carga, limpieza, split temporal
├── features/     Indicadores y motor de features normalizadas
├── gp/           Genetic Programming tipado (nodos, gramática, evolución)
├── strategy/     Representación de señales
├── backtest/     Motor de backtest causal con costes
├── fitness/      Métricas, complejidad, función objetivo
├── validation/   Walk-Forward, folds, robustness, final test
├── experiments/  Tracker y persistencia de experimentos
├── scripts/      Scripts ejecutables desde CLI
├── tests/        Tests unitarios (350)
├── test_final/   Scripts del final test (no ejecutar durante el desarrollo)
├── docs/         Auditoría técnica (ANALISIS_GP.md) y script de diagnóstico
└── main.py       Orquestador
```

## Preparación del entorno

Requiere Python 3.12+.

```bash
pip install -r requirements.txt    # pandas, numpy, scipy, pytest, yfinance
cp .env.example .env               # costes por símbolo; ajustar con valores del broker
```

Los CSV van en `data_files/` (ignorado por git). Convención de nombres:
`<simbolo>_<tf>_5y.csv` (p. ej. `xauusd_15m_5y.csv`); el símbolo se infiere del nombre
si no se pasa `--symbol`.

## Flujo de trabajo

### 1. Auditoría del backtester (recomendado antes de confiar en resultados)

```bash
python -m scripts.audit_backtester \
    --csv data_files/eurusd_15m_5y.csv \
    --output experiments_output/audit \
    --n-bars 2000 \
    --period 100
```

Corre el backtester sobre una señal trivial (LONG 100 velas, FLAT 100 velas) y verifica:

- Ejecución al open de t+1 (no al close de la misma vela).
- PnL bruto cuadra con entry/exit prices.
- Costes aplicados correctamente.
- Identidad contable: `equity_final = capital + sum(gross) - sum(costes)`.

Genera `bar_log.csv` (estado por vela) y `trades_audit.csv` (auditoría por trade).

### 2. Walk-Forward Optimization completo

```bash
python -m scripts.run_wfo \
    --csv data_files/eurusd_15m_5y.csv \
    --output experiments_output/wfo_5y_baseline \
    --is-months 12 \
    --oos-months 3 \
    --population 30 \
    --generations 8 \
    --min-trades 20 \
    --seed 42
```

- Reserva el 10% final del dataset como final test intocable (mínimo 500 velas).
- Genera folds rolling: IS = 12 meses, OOS = 3 meses, step = OOS (sin solapamiento).
- Entrena GP en cada IS y evalúa el mejor individuo en el OOS correspondiente.
- Concatena la equity OOS en una curva "stitched".
- Registra el experimento en `experiments_output/tracker.jsonl`.
- `--min-trades` vale 20 por defecto. El tiempo va de decenas de minutos a horas según
  `--population`/`--generations` (el backtester es un bucle Python).

> Validar cualquier resultado positivo con **al menos 3 seeds muy separadas**
> (p. ej. 42, 1042, 2042).

### 3. Dry-run de WFO (solo ver folds, sin correr GP)

```bash
python -m scripts.run_wfo \
    --csv data_files/eurusd_15m_5y.csv \
    --output experiments_output/wfo_dryrun \
    --is-months 12 \
    --oos-months 3 \
    --dry-run
```

Genera y reporta los folds sin ejecutar GP. Útil para verificar que la configuración es
coherente antes de gastar horas.

### 4. GP simple (sin WFO)

```bash
python -m scripts.run_gp \
    --csv data_files/eurusd_15m_5y.csv \
    --output experiments_output/gp_run_001 \
    --population 50 \
    --generations 15 \
    --seed 42
```

Corre GP sobre el dataset completo (sin separar IS/OOS). Solo para exploración rápida;
**no usar para validación seria**.

### 5. Evaluación cross-pair / cross-timeframe

```bash
python -m scripts.evaluate_strategy \
    --individual experiments_output/gp_run_001/best_individual.json \
    --csvs data_files/gbpusd_15m_5y.csv data_files/eurusd_15m_5y.csv
```

Evalúa un individuo ya entrenado sobre otros datos, sin reajustar. Si funciona sin
reoptimizar, hay evidencia de generalización estructural.

### 6. Baselines en serie

```bash
python -m scripts.run_all_wfo                 # todos (oro, EURUSD, GBPUSD, EURUSD H4)
python -m scripts.run_all_wfo --only gold     # solo uno
python -m scripts.run_all_wfo --skip gold     # todos menos oro
```

### 7. Timeframes distintos de 15m (`--bars-per-month`)

```bash
python -m scripts.run_wfo \
    --csv data_files/xauusd_4h_5y.csv \
    --output experiments_output/wfo_gold_h4 \
    --is-months 12 \
    --oos-months 3 \
    --bars-per-month 128
```

Por defecto `--bars-per-month` es 2100 (15m, forex ~5 días/semana); para H4 usar 128.

### 8. Pipeline completo (orquestador)

```bash
python main.py \
    --csv data_files/eurusd_15m_5y.csv \
    --output-dir experiments_output \
    --population 30 \
    --generations 8 \
    --seed 42
```

Llama a `run_wfo` con la configuración indicada.

## Tests

```bash
python -m pytest tests/ -v -W error::RuntimeWarning
```

Con `-W error::RuntimeWarning` los warnings de numpy se convierten en errores, lo que
garantiza que no hay overflows silenciosos en los operadores protegidos.

| Archivo | Cubre |
|---|---|
| `test_features.py` | causalidad (no look-ahead), correctitud de indicadores, carga y limpieza |
| `test_nodes.py` | operadores protegidos, gramática tipada |
| `test_individual.py` / `test_evolution.py` | construcción aleatoria, evaluación, operadores y evolución |
| `test_backtester.py` / `test_costs.py` | ejecución al open de t+1, contabilidad, costes por símbolo |
| `test_fitness.py` | métricas, complejidad, función objetivo |
| `test_wfo.py` | generación de folds, WFO completo |
| `test_robustness.py` | cross-pair, final test, tracker |

`tests/conftest.py` genera las columnas cross-asset necesarias para los fixtures.

## Parámetros clave

**`WFConfig`** (`validation/folds.py`)

- `is_size`: velas in-sample (p. ej. `12 * 2100` = 12 meses en 15m).
- `oos_size`: velas out-of-sample (p. ej. `3 * 2100`).
- `step_size`: avance entre folds; por defecto `oos_size` (sin solapamiento).
- `max_timestamp`: salvaguarda contra el final test.

**`EvolutionConfig`** (`gp/evolution.py`): `population_size`, `n_generations`,
`tournament_size`, `n_elites`, `crossover_rate`, `mutation_rate`, `seed`.

**`FitnessConfig`** (`fitness/objective.py`): `min_trades`, `max_turnover_ratio`
(transiciones / velas), `max_flat_ratio`, `primary_metric` (`"sharpe"` o `"sortino"`) y
pesos de penalización (complejidad, turnover, inactividad).

## Resultados y artefactos

```text
experiments_output/wfo_<name>/
├── final_test.csv          # Bloque final reservado (no usar hasta el final)
├── equity_stitched.csv     # Curva OOS concatenada
└── metrics.json            # Métricas agregadas
experiments_output/tracker.jsonl   # Registro de todos los experimentos
```

`tracker.jsonl` permite calcular el Deflated Sharpe Ratio sobre el conjunto de
experimentos, ajustando por multiple testing.

> Las métricas `bh_sharpe`, `excess_sharpe_vs_bh` y los IC95% del Sharpe tienen errores
> conocidos (ver `docs/ANALISIS_GP.md` §2). No las cites hasta que se corrijan.

## Final test

El 10% final del dataset se reserva como final test intocable y no se usa durante el
desarrollo. Solo se ejecuta **una vez**, sobre la estrategia congelada, al final del
proyecto. Si se ejecuta antes, el resultado pierde validez estadística.

El `final_test.csv` de cada `wfo_<name>/` es solo un volcado del bloque reservado, no un
lock. El lock real es el parámetro opcional `FinalTestConfig.lock_file`
(`validation/final_test.py`): si se indica, `run_final_test` lo crea al ejecutarse y se
niega a ejecutarse de nuevo mientras exista.

## Notas sobre datos

- Sin volumen (los CSV de forex retail suelen tener `volume=0`).
- El cleaner detecta gaps pero NO los rellena (evita look-ahead).
- Los gaps de fin de semana son normales (~48 h).
- No redondear decimales: la precisión extra no hace daño.
- Velas con rango grande por noticias son parte del mercado y no se filtran.

## Decisión de diseño

El sistema optimiza generalización y estabilidad, no rendimiento histórico máximo:

- GP tipado restringido a una gramática financiera.
- Fitness con penalizaciones de complejidad, turnover e inactividad.
- WFO con folds rolling y final test intocable.
- Tracker para auditar multiple testing.
- Tests que garantizan ausencia de look-ahead.
