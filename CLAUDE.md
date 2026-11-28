# CLAUDE.md — trading_gp

Guía de contexto para Claude que continúe este proyecto.
Asume que no tienes acceso al historial de conversación.

---

## TL;DR

Sistema de Genetic Programming (GP) tipado para descubrir reglas de trading
sobre forex, metales y materias primas. Pipeline completo, causal, sin look-ahead,
con WFO honesto, costes por símbolo y final test intocable.

> ⚠️ **Lee `docs/ANALISIS_GP.md` antes de fiarte de los números de este documento.**
> La auditoría detectó que el Sharpe de buy & hold es una constante, que el IC95% del
> Sharpe es ~44x demasiado estrecho y que el GP apenas explora (77% de árboles iniciales
> constantes). El "resultado negativo" de abajo está **pendiente de re-medir**.

**Resultado empírico (provisional): negativo.** GP con indicadores clásicos + features cross-asset
no encuentra edge robusto en forex ni oro, en 5 años de datos, con costes realistas.
El único resultado que batió a buy & hold (Sharpe +0.80, oro+plata+DXY, seed 42)
cayó a +0.25 con seed 123. No robusto a la aleatoriedad de GP.

**La infraestructura es sólida y reutilizable.** Los 350 tests pasan
(`pytest tests/ -q -W error::RuntimeWarning`).
`HANDOFF.md` está en `.gitignore` y **no existe en el repo**; si no lo tienes, la
referencia equivalente es `docs/ANALISIS_GP.md`.

---

## Reglas de trabajo

### Lo que NO debes hacer

1. **No tocar el final test** (10% del dataset, mínimo 500 velas). Es intocable
   durante el desarrollo. Solo se ejecuta UNA VEZ, al final, sobre la estrategia
   congelada.
2. **No romper las reglas de diseño** de la sección 3 del HANDOFF.md:
   - GP tipado con gramática restringida.
   - Features normalizadas (no precios absolutos).
   - Ejecución al open de t+1 (señal en t → orden en t+1).
   - Costes realistas por símbolo desde el inicio.
   - WFO con final test intocable.
   - Cualquier feature nueva debe tener test de causalidad.
   - Semilla derivada por fold (`seed + fold.index`).
3. **No confiar en un resultado positivo de una sola seed.** El proyecto demostró
   que un Sharpe de 0.80 con seed 42 puede caer a 0.25 con seed 123. Cualquier
   resultado positivo debe validarse con al menos 3 seeds distintas.
4. **No añadir features sin test de causalidad.** `test_all_features_are_causal`
   en `tests/test_features.py` verifica que una feature calculada sobre datos
   truncados hasta t dé el mismo valor que sobre datos completos. Si falla, hay
   look-ahead.
5. **No ignorar el Deflated Sharpe.** El tracker registra `n_trials` reales
   (population × generations × folds). Si has probado muchas configuraciones,
   el DSR ajusta el mejor Sharpe por multiple testing.

### Lo que SÍ debes hacer

1. **Leer `docs/ANALISIS_GP.md` (y HANDOFF.md si lo tienes en local) antes de tocar nada.** Tiene la arquitectura completa,
   las correcciones aplicadas (Tandas 1-3), y los resultados empíricos.
2. **Correr `pytest tests/ -v -W error::RuntimeWarning` antes y después de
   cualquier cambio.** Los warnings de numpy se convierten en errores para
   garantizar que no hay overflows silenciosos.
3. **Correr el audit del backtester** (`scripts/audit_backtester.py`) si vas a
   confiar en resultados. Verifica timing, PnL, costes, equity, trades.
4. **Registrar cada experimento en el tracker** (`experiments/tracker.py`).
5. **Validar cualquier resultado positivo con 3 seeds distintas** antes de
   declararlo como edge.
6. **Documentar los resultados negativos.** Son tan valiosos como los positivos.

---

## Estructura del proyecto

```
trading_gp/
├── config/settings.py         Constantes, COSTS_BY_SYMBOL, get_costs,
│                              bars_per_year, GPLimits, SplitConfig
├── data/                      loader, cleaner, splitter, models,
│                              csv_tools, normalize, merge_cross
├── features/                  indicators, registry, engine
├── gp/                        types, nodes, grammar, evaluator, individual,
│                              population, operators, evolution, serialization
├── strategy/signal.py         Signal (FLAT/LONG/SHORT)
├── backtest/                  engine, execution, position, costs, results
├── fitness/                   metrics, complexity, objective
├── validation/                folds, walk_forward, robustness, final_test
├── experiments/               tracker, storage
├── scripts/                   run_gp, run_wfo, run_all_wfo, audit_backtester,
│                              evaluate_strategy
├── tests/                     350 tests (conftest.py genera columnas cross-asset)
├── test_final/                Scripts del final test (NO ejecutar sin orden del usuario)
├── docs/                      ANALISIS_GP.md (auditoría) + diag_gp.py (diagnóstico)
├── main.py                    Orquestador
├── .env / .env.example        Costes por símbolo
├── HANDOFF.md                 Traspaso completo (gitignored, puede no existir)
└── CLAUDE.md                  Este documento
```

Símbolos soportados: `EURUSD`, `GBPUSD`, `XAUUSD`, `XAGUSD`, `WTI`, `BRENT`,
`USA500`, `USATECH`, `DXY`, `BTCUSD` (ver `SUPPORTED_SYMBOLS`).

---

## Datos (`data_files/`, ignorado por git)

Los CSV vienen de otro repositorio (descarga). Aquí no se descarga nada.
Formato canónico: `{symbol}_{tf}_{period}.csv` (p. ej. `wti_15m_5y.csv`),
columnas `datetime,open,high,low,close`, fecha `dd/mm/YYYY HH:MM`.
No leer los CSV enteros: usar las herramientas.

```bash
# Inspección sin cargar el archivo
python -m data.csv_tools head data_files/xauusd_15m_5y.csv      # 5 filas + chequeo
python -m data.csv_tools tail data_files/xauusd_15m_5y.csv -n 3
python -m data.csv_tools info data_files/xauusd_15m_5y.csv
python -m data.csv_tools scan data_files/xauusd_15m_5y.csv      # integridad completa
python -m data.csv_tools overlap data_files/xauusd_15m_5y.csv data_files/dxy_15m_5y.csv

# Corregir nombre/formato de un CSV nuevo (alias de la fuente en SOURCE_SYMBOL_ALIASES)
python -m data.normalize data_files/lightcmdusd_15m_5y.csv --replace
python -m data.normalize data_files/raro.csv --symbol DXY --replace

# Timeframe que la fuente no trae (p. ej. DXY 4h) a partir de 15m
python -m data.converter data_files/dxy_15m_5y.csv --timeframe 4h   # -> dxy_4h_5y.csv

# Dataset cross-asset: OHLC del base + {sym}_close de cada cross (inner join)
python -m data.merge_cross --base data_files/xauusd_15m_5y.csv     --cross data_files/xagusd_15m_5y.csv data_files/wti_15m_5y.csv
# -> data_files/xauusd_15m_5y_x_xagusd_wti.csv (run_wfo infiere XAUUSD)
python -m data.merge_cross --base ... --cross ... --dry-run   # solo informe
python -m data.merge_cross --base ... --cross ... --force     # ignora límites
```

Todo símbolo de `SUPPORTED_SYMBOLS` genera features cross (`CROSS_ASSETS`).
Mismo timeframe obligatorio en base y cross (15m con 15m, 4h con 4h).

Control de cobertura del merge (límites por defecto, calibrados con los
datos reales): huecos interiores <= 5%, continuidad rota <= 1%, pérdida
total <= 15%. Si se superan, el merge se rechaza indicando el cross
culpable; `--force` lo genera igualmente. DXY y BRENT no cotizan
21:00-24:00 ni domingos: frente a instrumentos 24h pierden 17-25% de las
velas y se rechazan (entre ellos dos, BRENT + DXY, son compatibles).
Los `xauusd_15m_5y_x_*dxy*.csv` de los experimentos anteriores solo se
pueden regenerar con `--force`.

## Comandos esenciales

### Tests

```bash
python -m pytest tests/ -v -W error::RuntimeWarning
```

### Audit del backtester

```bash
python -m scripts.audit_backtester \
    --csv data_files/xauusd_15m_5y.csv \
    --output experiments_output/audit \
    --n-bars 2000 --period 100
```

### Dry-run de WFO (ver folds sin correr GP)

```bash
python -m scripts.run_wfo \
    --csv data_files/xauusd_15m_5y.csv \
    --output experiments_output/wfo_gold_dryrun \
    --is-months 12 --oos-months 3 --dry-run
```

### WFO completo

```bash
python -m scripts.run_wfo \
    --csv data_files/xauusd_15m_5y.csv \
    --output experiments_output/wfo_gold_v5 \
    --is-months 12 --oos-months 3 \
    --population 30 --generations 8 \
    --min-trades 20 --max-turnover-ratio 0.15 \
    --seed 42
```

Para H4, añadir `--bars-per-month 128` (15m: default 2100).
Para validar con 3 seeds usa seeds muy separadas (p. ej. 42, 1042, 2042): la semilla
por fold es `seed + fold.index`, así que 42 y 43 comparten secuencias desplazadas.

### Correr los 4 WFOs baseline en serie

```bash
python -m scripts.run_all_wfo          # todos
python -m scripts.run_all_wfo --only gold
python -m scripts.run_all_wfo --skip gold
```

---

## Cómo añadir features nuevas

### Features del propio OHLC

Editar `features/registry.py`:

```python
def _mi_feature(df: pd.DataFrame, param: int) -> pd.Series:
    # Cálculo causal. Nunca shift(-n), nunca center=True.
    return (df["close"] - ind.sma(df["close"], param)) / ind.atr(
        df["high"], df["low"], df["close"], param
    )

FEATURES["mi_feature_20"] = FeatureSpec(
    name="mi_feature_20",
    func=lambda df: _mi_feature(df, 20),
    description="Descripción legible",
)
```

Automáticamente aparece en `feature_names()`, en `FEATURE_TERMINALS`,
GP puede usarla, y los tests de causalidad la verifican.

### Features cross-asset

Requieren que el DataFrame tenga columnas extra (ej. `dxy_close`, `xagusd_close`).
El merge se hace antes de `compute_features`:

```python
df = df.merge(
    df_dxy[["datetime", "close"]].rename(columns={"close": "dxy_close"}),
    on="datetime", how="inner",
)
```

Y en `features/registry.py`:

```python
FEATURES["dxy_ret_1"] = FeatureSpec(
    name="dxy_ret_1",
    func=lambda df: df["dxy_close"].pct_change(1),
    description="Retorno 1 vela del DXY",
)
```

**Fixtures de tests:** `tests/conftest.py` genera dinámicamente las columnas
cross-asset a partir de `required_columns()`, así que añadir features cross-asset ya no
rompe `test_features.py`, `test_wfo.py` ni `test_robustness.py`.

Los datos cross-asset (DXY, plata, WTI) y el script de merge **no están en el repo**
(`data_files/` ignorado; `check_cross.py`/`datastractor.py` borrados), por lo que esos
resultados no son reproducibles desde aquí. Verifica además que todas las fuentes usen
la misma convención de timestamp (apertura de vela) para evitar look-ahead de una barra.

## Resultados empíricos (post-correcciones) — ⚠️ columnas `IC95%` y `vs B&H` NO FIABLES

`bh_sharpe` es una constante (`sqrt(bars_per_year/N)`) y el IC95% usa una fórmula con
unidades mezcladas (`docs/ANALISIS_GP.md` §2.1–2.2). Solo el Sharpe stitched es
informativo, y aun así sin significancia estadística demostrada. Re-medir (Fase A del
plan) antes de citar estos valores.

| Instrumento | TF | Folds | Sharpe stitched | IC95% | vs B&H | Veredicto |
|---|---|---|---|---|---|---|
| EURUSD | 15m | 13 | negativo | — | — | Sin edge |
| EURUSD | H4 | 14 | negativo | — | — | Sin edge |
| XAUUSD | 15m | 17 | +0.11 | [0.09, 0.13] | -0.37 | Sin edge |
| XAUUSD | H4 | 17 | +0.10 | [0.05, 0.14] | -0.38 | Sin edge |
| XAUUSD + XAGUSD | 15m | 16 | +0.31 | [0.25, 0.37] | -0.18 | No bate B&H |
| XAUUSD + XAGUSD + DXY | 15m | 11 | +0.80 (seed 42) | [0.63, 0.97] | +0.21 | **No robusto** |
| XAUUSD + XAGUSD + DXY | 15m | 11 | +0.25 (seed 123) | [0.19, 0.31] | -0.34 | No robusto |
| XAUUSD + XAGUSD + DXY + WTI | 15m | 11 | +0.63 (seed 42) | [0.49, 0.77] | +0.04 | Diluye |

**No reintentar estas combinaciones sin cambiar algo estructural**
(features, gramática, fitness, período de datos, o instrumento) **o sin haber
corregido primero la evaluación y la búsqueda GP** (ver `docs/ANALISIS_GP.md` §5).

---

## Hipótesis pendientes (si alguien quiere continuar)

1. **Cambiar la gramática:** permitir score continuo en vez de IF anidados.
   Alto riesgo de overfitting.
2. **Features de contexto:** hora del día, día de la semana, volatilidad
   relativa. Baratas de añadir.
3. **Otros instrumentos:** índices (USA500, USATECH), Brent. El pipeline
   los soporta.
4. **Otras métricas de fitness:** Sortino, Calmar, Profit Factor como
   métrica primaria.
5. **Tanda 4 (GP):** ramped half-and-half, constantes efímeras, LOG/SQRT
   con NaN. La auditoría la reclasifica como **alto impacto** (ANALISIS_GP §1.1–1.2).
6. **Tanda 5 (limpieza):** vectorizar backtester, eliminar duplicación en
   `engine.py`, `__import__` en `walk_forward`. Vectorizar es **prerrequisito**
   de escalar la búsqueda GP (ANALISIS_GP §1.7).

---

## Convenciones de código

- Python 3.12+.
- Type hints en todas las firmas públicas.
- `from __future__ import annotations` al inicio de cada módulo.
- Dataclasses para configuraciones (`@dataclass(frozen=True)` cuando sea
  posible).
- Sin dependencias externas más allá de pandas, numpy, scipy, pytest.
- Tests con `pytest`, fixtures en `conftest.py` o en el propio archivo de test.
- Nombres en inglés para código, español para docstrings y comentarios.

---

## Notas para Claude

- **El proyecto tiene un "resultado negativo provisional".** No lo des por
  concluyente: la auditoría (`docs/ANALISIS_GP.md`) muestra que la evaluación y la
  búsqueda GP tenían defectos. No añadas features ni cambies parámetros sin una
  hipótesis clara; el orden recomendado es Fase A (evaluación) → B (búsqueda GP)
  → C (escala) → D (estructural).
- **Si el usuario pide continuar**, la única vía con potencial real es
  cambiar algo estructural: gramática (score continuo), features de
  contexto (hora, día), u otro instrumento (índices, Brent).
- **Cualquier resultado positivo debe validarse con 3 seeds distintas.**
  El proyecto demostró que un Sharpe de 0.80 puede caer a 0.25 con otra
  semilla. No confíes en una sola corrida.
- **El HANDOFF.md tiene todos los detalles.** Este CLAUDE.md es un resumen.
  Si necesitas más contexto, lee HANDOFF.md.
- **El final test está intocable.** No lo ejecutes hasta que el usuario
  decida congelar la estrategia y hacer la prueba final.

## Flujo de trabajo esperado

- Antes de modificar archivos, propón un plan y espera mi OK.
- Usa modo plan (Shift+Tab) para tareas que toquen >1 archivo.
- No hagas refactors no solicitados.
- Tras cada cambio, corre pytest y reporta el resultado.
- Haz commits atómicos por tarea; no mezcles features + tests + limpieza.
- Si una tarea es ambigua, pregunta antes de asumir.
- Cuando termines una tarea, sugiere `/clear` para la siguiente.

## Tarea activa (actualizar al cambiar de foco)

1. ~~Hacer escalable la creación de tests al añadir instrumentos (data-driven).~~
   **Hecho** (`conftest.py` + `required_columns()`, `test_costs` parametrizado).
2. Añadir features de contexto: hora del día, día de la semana, volatilidad relativa.
3. Evaluar Tandas 4 y 5: **sí son relevantes** (ver `docs/ANALISIS_GP.md` §4–5).
   Antes de la tarea 2 conviene cerrar la Fase A del plan (evaluación honesta).

Orden: 1 → 2 → 3. No empezar la siguiente sin cerrar la anterior.
