```markdown
# CLAUDE.md — trading_gp

Guía de contexto para Claude que continúe este proyecto.
Asume que no tienes acceso al historial de conversación.

---

## TL;DR

Sistema de Genetic Programming (GP) tipado para descubrir reglas de trading
sobre forex, metales y materias primas. Pipeline completo, causal, sin look-ahead,
con WFO honesto, costes por símbolo y final test intocable.

**Resultado empírico: negativo.** GP con indicadores clásicos + features cross-asset
no encuentra edge robusto en forex ni oro, en 5 años de datos, con costes realistas.
El único resultado que batió a buy & hold (Sharpe +0.80, oro+plata+DXY, seed 42)
cayó a +0.25 con seed 123. No robusto a la aleatoriedad de GP.

**La infraestructura es sólida y reutilizable.** aunque cuando se añadieron otros features de Cross, hubo tests que fallaron, antes de eso todos los test funcionaban..
El HANDOFF.md completo tiene todos los detalles.

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

1. **Leer HANDOFF.md antes de tocar nada.** Tiene la arquitectura completa,
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
├── data/                      loader, cleaner, splitter, models
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
├── tests/                     ~100 tests
├── main.py                    Orquestador
├── .env / .env.example        Costes por símbolo
├── HANDOFF.md                 Documento completo de traspaso
└── CLAUDE.md                  Este documento
```

Símbolos soportados: `EURUSD`, `GBPUSD`, `XAUUSD`, `XAGUSD`, `WTI`, `BRENT`,
`USA500`, `USATECH`.

---

## Comandos esenciales

### Tests

```bash
python -m pytest tests/ -v -W error::RuntimeWarning
```

### Audit del backtester

```bash
python -m scripts.audit_backtester \
    --csv data_files/XAUUSD_X_15m_5y.csv \
    --output experiments_output/audit \
    --n-bars 2000 --period 100
```

### Dry-run de WFO (ver folds sin correr GP)

```bash
python -m scripts.run_wfo \
    --csv data_files/XAUUSD_X_15m_5y.csv \
    --output experiments_output/wfo_gold_dryrun \
    --is-months 12 --oos-months 3 --dry-run
```

### WFO completo

```bash
python -m scripts.run_wfo \
    --csv data_files/XAUUSD_X_15m_5y.csv \
    --output experiments_output/wfo_gold_v5 \
    --is-months 12 --oos-months 3 \
    --population 30 --generations 8 \
    --min-trades 20 --max-turnover-ratio 0.15 \
    --seed 42
```

Para H4, añadir `--bars-per-month 128` (15m: default 2100).

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

**Problema conocido pendiente:** los fixtures de tests (`synthetic_ohlc`,
`ohlc_1`, etc.) no incluyen columnas cross-asset. Al añadir features
cross-asset, los tests de `test_features.py`, `test_wfo.py`,
`test_robustness.py` fallan con `KeyError`. Solución pendiente:
`conftest.py` con fixture dinámico que añada las columnas extra necesarias,
o separar `FEATURES` (OHLC) de `CROSS_ASSET_FEATURES` con un flag
`include_cross_asset` en `compute_features`.

## Resultados empíricos (post-correcciones)

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
(features, gramática, fitness, período de datos, o instrumento).

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
   con NaN. Bajo impacto esperado.
6. **Tanda 5 (limpieza):** vectorizar backtester, eliminar duplicación en
   `engine.py`, `__import__` en `walk_forward`. No urgente.

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

- **El proyecto está en un estado de "resultado negativo documentado".**
  No intentes "arreglarlo" añadiendo más features o cambiando parámetros
  sin una hipótesis clara. El espacio (features clásicas + gramática
  IF anidados + costes realistas) no contiene señal explotable en estos
  instrumentos.
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

1. Hacer escalable la creación de tests al añadir instrumentos (data-driven).
2. Añadir features de contexto: hora del día, día de la semana, volatilidad relativa.
3. Evaluar Tandas 4 y 5 del HANDOFF.md solo si son relevantes tras lo anterior.

Orden: 1 → 2 → 3. No empezar la siguiente sin cerrar la anterior.