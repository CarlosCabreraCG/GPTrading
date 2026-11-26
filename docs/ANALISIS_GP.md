# ANALISIS_GP.md — Auditoría técnica del GP y del pipeline de evaluación

> Para Claude (otra sesión) sin historial. Léelo ANTES de tocar nada, junto con CLAUDE.md.
> `HANDOFF.md` está en `.gitignore` y **no existe en el repo local**; este documento
> lo complementa en lo referente a gramática, búsqueda evolutiva y validez estadística.
>
> Fecha del análisis: 2026-11-25. Estado del repo: `main`, árbol limpio, 350 tests pasan
> (`pytest tests/ -q -W error::RuntimeWarning`). No se modificó código; solo se añadió `docs/`.

---

## 0. Resumen ejecutivo

El "resultado negativo" de CLAUDE.md **no es todavía concluyente**, por dos razones
independientes que se refuerzan:

1. **Las métricas de comparación están rotas** (sección 2): el Sharpe de buy & hold es una
   constante que solo depende del nº de velas, y el IC95 del Sharpe es ~44x demasiado
   estrecho (declara "significativo" el 99% del ruido puro). La tabla de resultados de
   CLAUDE.md (columnas `IC95%`, `vs B&H`, veredictos) **no es fiable**. Ni el "+0.80 vs
   +0.21" ni los "−0.37 vs B&H" significan lo que parecen.
2. **El GP casi no está buscando** (sección 1): 77% de la población inicial son árboles de
   señal constante, 87.5% tienen fitness `-inf`, y de 120 individuos solo hay 21 señales
   distintas. Además la gramática no tiene constantes, ni operadores temporales, ni
   normalización común de escalas. Con población 30 × 8 generaciones, el algoritmo es
   esencialmente una búsqueda aleatoria sobre unas pocas decenas de árboles viables.

Es decir: el experimento no demostró "no hay edge en este espacio", demostró "este GP
mal configurado no encuentra nada, y lo mide con una regla defectuosa".

**Orden recomendado (detalle en sección 5):** (A) arreglar evaluación + tests de
regresión → (B) re-medir baseline honesto → (C) arreglar la búsqueda GP (init, constantes,
escalas, fitness) → (D) vectorizar backtester y subir población/generaciones → (E) cambios
estructurales (operadores temporales, features de contexto, semántica de posición).

---

## 1. Gramática y GP: errores y carencias

Evidencia empírica reproducible: `python -W ignore docs/diag_gp.py "$(pwd)"`
(datos: `data_files/xauusd_15m_5y.csv`, 20 features OHLC, fold 5, límites por defecto).

### 1.1 Inicialización: la mayoría de árboles son constantes (CRÍTICO)

`gp/individual.py::_random_tree_recursive` elige el nodo **uniformemente** entre los
"elegibles" del tipo. Para `POSITION` hay 4 nodos (`IF, LONG, SHORT, FLAT`) → **75% de
raíces son terminales**. Lo mismo en las ramas `then/else`, y para `REAL` (20 terminales
vs 9 operadores) la mayoría de sub-árboles también son hojas.

Medido (N=3000 árboles aleatorios):

| Métrica | Valor |
|---|---|
| Raíz = FLAT / SHORT / LONG / IF | 26.5% / 25.8% / 24.7% / 23.0% |
| Tamaño medio / mediana | 4.3 / **1** nodo |
| Nº medio de condiciones (GT/LT) | 0.6 |
| **Señal constante (0 condiciones)** | **77%** |

Consecuencia (120 individuos en el IS del fold 5): **87.5% → fitness `-inf`**
(39 FLAT total, 63 constantes no-flat → `n_trades<min_trades`, 2 turnover, 1 otros);
**solo 21 señales distintas de 120**; 15 con fitness finito (media −2.7, máx +0.31).
Con `population=30` quedan ~4 individuos viables en la generación 0.

Además solo existe método *grow* (no hay *full*, ni ramped half-and-half; es la "Tanda 4"
de CLAUDE.md, valorada allí como "bajo impacto": **está infravalorada**).

**Arreglo propuesto:**
- Forzar raíz `IF` (o al menos `p(IF) ≥ 0.9`) y probabilidad de terminal dependiente de la profundidad.
- Ramped half-and-half (full + grow, profundidades 2..max_depth).
- Rechazar en `random_individual` los árboles de señal constante / con pocas transiciones
  (re-muestrear hasta tener `n_trades ≥ min_trades` en el IS, con tope de intentos).
- Deduplicar por hash de la señal (`s.tobytes()`) para exigir diversidad conductual.

### 1.2 Sin constantes efímeras (ERC) y con escalas heterogéneas (CRÍTICO)

`GT/LT` comparan `REAL` con `REAL`, y los únicos `REAL` terminales son features. No existe
ningún terminal constante. Una regla clásica como `RSI > 70` es **inexpresable**; hay que
comparar features entre sí aunque tengan unidades distintas. Escalas medidas:

| Feature | std |
|---|---|
| `return_1`, `atr_14_norm`, `rolling_std_20`, `bb_width_20` | ~5e-4 – 1e-3 |
| `close_over_sma_20`, `roc_10`, `return_20` | ~2e-3 – 4e-3 |
| `dist_*_atr`, `hl_range_atr_14`, `bb_position_20` | ~0.3 – 1.8 |
| `rsi_14`, `adx_14` | **~11–12 (rango 0–100)** |

`GT(rsi_14, return_1)` o `LT(atr_14_norm, adx_14)` son **casi constantes** (siempre
verdaderas o siempre falsas): condiciones degeneradas que no discriminan. La "tipificación"
solo distingue REAL/BOOLEAN/POSITION, no *unidades/escala*. CLAUDE.md afirma "Features
normalizadas", pero solo lo están en el sentido de no ser precios absolutos.

**Arreglo propuesto (cualquiera de los dos, o ambos):**
- Normalizar todas las features a una escala común **causal**: z-score con ventana rolling
  (p. ej. 500 barras, `min_periods` completo) o rango percentil rolling en [0,1]. Test de
  causalidad obligatorio (`test_all_features_are_causal`).
- Añadir terminal `ERC` (constante muestreada de U(−3,3) tras z-score) → requiere campo
  `value` en `TreeNode`, soporte en `evaluator`, `serialization`, `operators` (mutación de
  constantes con perturbación gaussiana) y `check_types`.

### 1.3 Sin memoria temporal (ALTO)

Cada feature tiene **una única ventana fija** (1, 5, 10, 14, 20) y los nodos no tienen
operadores temporales. Imposible expresar: cruce (`close` cruza `EMA`), cambio de una
feature, máximo/mínimo reciente de una feature, persistencia ("RSI>70 durante 3 velas"),
ni ajustar periodos. Las reglas son función del instante `t` únicamente.

**Arreglo propuesto:** nodos `LAG(REAL, k)` con `k` en {1,2,3,5,10} (solo `shift(+k)`,
con test de causalidad), `DELTA`, `MEAN_k`/`MAX_k`/`MIN_k` rolling, y `CROSS_UP/CROSS_DOWN`
(BOOLEAN) construidos sobre LAG. Alternativa más barata: añadir al registry versiones
multi-ventana (`rsi_7/14/28`, `ema 10/50/200`, etc.) y features de cambio (`d_rsi_3`).

### 1.4 Semántica de posición: sin estado, sin salidas (ALTO)

El árbol devuelve `LONG/SHORT/FLAT` en **cada** barra, sin histéresis, sin tiempo mínimo
de tenencia, sin stop/take-profit. En 15m esto genera alto turnover; el único control es
el corte duro `max_turnover_ratio=0.15` (→ `-inf`). En el WFO de oro ya se ve: ~144
trades/fold con Sharpe OOS medio por fold −0.45.

**Arreglo propuesto (hipótesis, no verificada):** (a) min-holding de N barras como
parámetro del backtest (no del árbol), (b) salidas ATR (SL/TP/time-stop) en el backtester,
(c) probar H1/H4 donde los costes pesan menos (el repo soporta `--bars-per-month 128`).

### 1.5 Fitness con precipicio `-inf` (ALTO)

`fitness/objective.py` devuelve `-inf` para: error, inactividad total, flat>90%,
`n_trades<min_trades`, `turnover>0.15`. No hay gradiente entre "casi válido" y "inválido";
el torneo entre `-inf` elige el primero. Con 87.5% de `-inf` en gen 0, la selección es
básicamente ruido.

**Arreglo propuesto:** penalizaciones graduadas (p. ej. `fitness = -K - violación` con
violación continua: déficit de trades, exceso de turnover) para que existan pendientes de
mejora; dejar `-inf` solo para errores de ejecución.

### 1.6 Sobreajuste dentro del IS: una sola trayectoria (ALTO)

El fitness es el Sharpe del IS completo (12 meses) de **un único camino**. El mejor de
240 evaluaciones/fold se selecciona por su Sharpe en ese camino; no hay holdout interno ni
medida de estabilidad. Penalizaciones de complejidad (0.001/nodo, 0.05/condición,
0.02/feature) son pequeñas frente al ruido del Sharpe.

**Arreglo propuesto:** fitness = media − λ·desv. (o mínimo) del Sharpe en K sub-bloques
del IS (p. ej. 4 trimestres), o validación interna anidada (embargo entre bloques).

### 1.7 Operadores evolutivos pobres (MEDIO)

- `crossover`: elige el *tipo* uniformemente (POSITION/BOOLEAN/REAL) y luego el nodo
  uniformemente → no hay sesgo Koza 90/10 hacia nodos internos; con árboles de ~4 nodos
  muchos cruces son no-ops (intercambian terminales iguales).
- `mutate`: `mutation_rate=0.1` por hijo, 1 solo tipo de mutación (reemplazo de subárbol).
  Genera el sub-árbol con `remaining_depth=limits.max_depth` (no la profundidad restante),
  así que con frecuencia viola límites y cae al siguiente tipo (3 intentos máximo y
  devuelve copia sin cambios). Sin point-mutation, sin hoist, sin perturbación de constantes.
- Sin caché de fitness: élites y duplicados se **re-evalúan** cada generación.
- Sin control de bloat más allá de límites duros; sin protección de diversidad.
- Tamaño de búsqueda ridículo: `population=30`, `generations=8` ⇒ 240 evaluaciones/fold
  (4 080 en 17 folds). GP normalmente necesita 10²–10³ individuos × 50–100 generaciones.
- Cuello de botella: `backtest/engine.py` es un bucle Python (~0.2 s por backtest de 25k
  barras, medido). 500×50×17 folds ≈ 425k evaluaciones ≈ 24 h. **La vectorización del
  backtester ("Tanda 5", marcada "no urgente") es prerrequisito de cualquier búsqueda seria.**

### 1.8 Features: cobertura limitada (MEDIO)

- Solo retorno, volatilidad, distancia a medias, RSI/ROC/ADX, breakout, Bollinger. Sin
  features de contexto (hora/sesión, día de semana, vol relativa a su historia, rango
  intradía vs. media) — es la "Tarea activa 2" de CLAUDE.md y encaja aquí.
- Cross-asset: solo `ret_1` y `spread_ret_1` (retorno de **1 barra**). Sin multi-horizonte,
  sin ratio oro/plata (z-score), sin correlación rolling oro–DXY, sin tipos reales/yields.
- El merge cross-asset (`how="inner"` en `datetime`) elimina barras donde el activo
  secundario no cotiza (DXY y metales tienen horarios distintos) y los datos de DXY/plata/
  WTI **no están en el repo** (`data_files/` ignorado) **ni hay script de merge** (los
  scripts `check_cross`/`datastractor` fueron borrados, commit `aac8229`). Los resultados
  cross-asset de CLAUDE.md **no son reproducibles desde el repo**. Verificar además que el
  timestamp de cada fuente tiene la misma convención (apertura vs cierre de vela) para
  descartar look-ahead de una barra.

### 1.9 Menores / limpieza

- `_max_if_nesting` duplicada en `gp/individual.py` y `gp/operators.py`;
  `import copy` dentro del bucle de `evolve`; `__import__` en `walk_forward` (ya listado).
- `scripts/run_wfo.py`: `--min-is-size` y `--min-oos-size` no se usan en ningún sitio e
  `--is-ratio` solo se vuelca a los metadatos (los folds se definen por `--is-months`/`--oos-months`).
- `data/loader.py::download_load(symbol=...)` ignora `symbol` (llama `download_yf()` con
  EURUSD por defecto) y `download_yf` añade `=X` (inválido para oro/índices en yfinance).
- Semilla por fold `seed+fold.index`: seeds 42 y 43 comparten secuencias desplazadas;
  para las "3 seeds" de validación usar seeds muy separadas (42, 1042, 2042).
- CLAUDE.md dice "~100 tests"; hay **350** y el "problema conocido pendiente" de fixtures
  cross-asset **ya está resuelto** (`tests/conftest.py` genera columnas desde
  `required_columns()`). Actualizar CLAUDE.md.

---

## 2. Errores en la evaluación (invalidan la tabla de resultados) — CRÍTICO

### 2.1 Sharpe de buy & hold es una constante (verificado)

`validation/walk_forward.py::_buy_and_hold_equity` crea una equity **plana** igual a
`capital` en todas las barras salvo la última (`capital + net_pnl`). Con un único retorno
no nulo al final, `sharpe()` devuelve `±sqrt(bars_per_year / N)` **independientemente de la
trayectoria de precios**. Comprobado:

- bh_sharpe oro (17 folds, N=107 100): código = **0.47259144**; `sqrt(23920/107100)` =
  **0.47259144** (idéntico, coincide con `experiments_output/tracker.jsonl`).
- Con otra trayectoria de precios del mismo signo, mismo resultado.

⇒ `bh_sharpe`, `excess_sharpe_vs_bh` y todos los "vs B&H" de CLAUDE.md son artefactos.
(Esto se introdujo en el commit "fix: buy and hold comparition solved".)

**Arreglo:** B&H con mark-to-market barra a barra: posición LONG 1 unidad desde el open
de la 1ª barra, equity = `capital + size*(close_t - entry) - coste_entrada`, coste de
salida al final. Reutilizar `run_backtest` con señal constante LONG sobre el OOS apilado
(coherencia de costes). Añadir test: B&H sobre serie creciente vs decreciente da Sharpe
de signo opuesto y magnitud distinta.

### 2.2 IC95% del Sharpe ~44x demasiado estrecho (verificado)

`_sharpe_ci` aplica la fórmula de Lo (2002), que es para SR **por periodo**, con el SR
**anualizado** y `n`=nº de barras de 15 min. Resultado: `se≈0.011` para oro cuando el
error estándar anualizado real es `≈ sqrt(bars_per_year/N) ≈ 0.47` (IC95 ≈ ±0.93, y mayor
con colas gordas y autocorrelación). Test con **ruido puro** (media 0, N=107 100):
`se_codigo=0.0033`, y **el 99% de 200 series de ruido se declaran
`stitched_sharpe_significant=True`**.

⇒ Todos los `IC95%` y `significant` de CLAUDE.md son inválidos. Con el error correcto, el
Sharpe 0.80 de oro+plata+DXY (≈2.75 años OOS) tendría IC ≈ [−0.1, 1.7]: **no distinguible de
0 ni siquiera con seed 42**. Lo mismo para el +0.31 y el +0.11.
`ExperimentTracker.deflated_sharpe_ratio` mezcla las mismas unidades (SR anualizado con
varianza por periodo) → también hay que revisarlo.

**Arreglo:** calcular SR por periodo (o sobre retornos diarios agregados) para la fórmula
de Lo, o mejor usar bootstrap por bloques (stationary/circular block bootstrap) sobre los
retornos de la equity stitched; reescalar a anual. Aplicar el mismo criterio al DSR.
Añadir test de ruido puro: la tasa de falsos "significativos" debe ser ≈5%.

### 2.3 `oos_return` por fold con base errónea

En `run_walk_forward` cada fold OOS arranca con `initial_capital` (10 000), pero se divide
`oos_result.equity.iloc[-1] / current_capital` con `current_capital` = equity final del
fold anterior. Para folds ≥1 el retorno mezcla bases → distorsiona `oos_return`,
`pct_positive_folds`, `mean/median/best/worst_fold_return`, `std_fold_return`. (El
Sharpe por fold y el stitched no se ven afectados.)
**Arreglo:** `eq[-1]/eq[0] - 1` por fold.

### 2.4 Tamaño fijo = 1 unidad

`FitnessConfig.size=1.0` ⇒ 1 oz de oro (exposición ≈ 0.1–0.25× capital) y 1 unidad de
EURUSD (≈ 0.0001× capital). El Sharpe es invariante a escala, pero `total_return`,
`max_drawdown`, comparación con B&H en rendimiento y la reescala multiplicativa del
stitched (PnL aditivo vs reescala multiplicativa) son engañosos. Mejor: unidades =
`capital*leverage/precio` fijadas por fold, o documentar explícitamente la convención.

### 2.5 Otros puntos de rigor

- **Forking paths:** el dev set se ha usado en muchas iteraciones (wfo v2…v6, combos de
  activos, 2 seeds). `n_trials` del tracker cuenta evaluaciones GP, no las decisiones
  manuales de configuración. Cualquier resultado positivo futuro debe tratarse con prudencia
  aunque el DSR parezca bueno; el final test sigue sin tocarse (no tocar).
- El WFO elige **un** individuo por fold sin ensamblar ni comprobar estabilidad
  (p. ej. top-k votando); el ruido de selección es alto.

---

## 2b. Debilidades adicionales (revisión posterior)

### 2b.1 El final test se define por porcentaje, no por fecha (ALTO)

`data/splitter.py::split_development_final` toma el último `final_test_ratio` (10%) del
CSV que se cargue. Si el CSV se redescarga o amplía, o si el `inner` merge cross-asset
cambia el nº de barras, la frontera se desplaza: datos usados para desarrollar pueden caer
dentro del "final test", y cada combinación de activos tiene un final test distinto. No hay
embargo entre el último OOS y el final test, y 10% de 5 años (~6 meses) es un único
régimen de mercado: poca potencia estadística.
**Arreglo:** fijar una fecha de corte constante (`--final-test-start`, ver `SplitConfig.final_test_start`)
y registrarla en el tracker; considerar un final test más largo y un embargo.

### 2b.2 `test_final/` obsoleto y confuso (MEDIO)

Los scripts de `test_final/` no son el final test real (ese es `validation/final_test.py`),
usan `WFConfig(is_ratio=..., oos_ratio=...)` (campos que ya no existen → `TypeError`) y un CSV
inexistente (`eurusd_15m_60d.csv`). Uno tiene typo (`fintal_test_system.py`). Eliminados.

### 2b.3 Costes optimistas (MEDIO)

- `CostModel.swap_per_night` no se usa en el backtester: posiciones que duran días no pagan carry.
- Spread constante: no se ensancha en rollover, noticias ni sesión asiática, justo donde un
  GP en 15m tiende a operar. Sesga a favor de la estrategia.
**Arreglo:** aplicar swap por noche mantenida y un multiplicador de spread por franja horaria
(parámetro del backtest, con test de contabilidad).

### 2b.4 Sin convención de zona horaria (MEDIO, bloquea features de contexto)

`data/loader.py` lee `datetime` naive (`%d/%m/%Y %H:%M`) sin zona ni DST. Si el dato está en
hora de broker (con DST) o en UTC no está documentado. Las features de hora/sesión/día de la
"Tarea activa 2" quedarían desplazadas 1 h la mitad del año.
**Arreglo:** decidir y documentar la convención (recomendado UTC), normalizar en el loader y
añadir un test.

---

## 3. Qué SÍ está bien (no lo rompas)

- Backtester: ejecución al open de t+1, costes por símbolo, cierre forzado al final del
  bloque, audit script. Contabilidad coherente (revisado `engine.py`/`execution.py`).
- Features/causalidad: indicadores con `min_periods`, `shift(+n)` únicamente; features
  calculadas sobre IS+OOS concatenados y recortadas (no hay look-ahead hacia el futuro
  del bloque; el warm-up del OOS usa datos IS pasados, correcto).
- Folds rolling sin solape de OOS, `max_timestamp` como salvaguarda del final test,
  semilla derivada por fold, tracker con `n_trials`, tests de causalidad.
- Operadores protegidos (div, sqrt, log) y manejo de excepciones en fitness con contador.

---

## 4. Estado de la "Tarea activa" de CLAUDE.md

1. *Tests data-driven por instrumento*: **hecho en lo esencial** (`conftest.py` +
   `required_columns()`, `test_costs` parametrizado). Se puede cerrar.
2. *Features de contexto*: pendiente; ver 1.8. Hacerlo **después** de 1.1–1.5 y sección 2,
   si no el GP no sabrá explotarlas.
3. *Tandas 4 y 5*: **sí son relevantes** (1.1 y 1.7); Tanda 5 (vectorizar backtester) es
   prerrequisito de escalar la búsqueda.

---

## 5. Plan recomendado (cada punto = tarea atómica / commit propio, con plan y OK previo)

**Fase A — Evaluación honesta (primero, no depende del GP)**
1. Corregir B&H (2.1) + test de regresión.
2. Corregir IC/DSR (2.2) + test de ruido puro (falsos positivos ≈5%).
3. Corregir `oos_return` (2.3) + test.
   (Antes de re-ejecutar: fijar la fecha de corte del final test, 2b.1.)
4. Re-ejecutar `scripts/run_wfo` oro 15m (seed 42/1042/2042) y **reemplazar la tabla de
   resultados de CLAUDE.md** con números válidos. Registrar en tracker.

**Fase B — Búsqueda GP útil**
5. Inicialización: raíz IF, ramped half-and-half, rechazo de constantes, dedupe (1.1).
6. Normalización causal de features + terminal ERC (1.2).
7. Fitness con penalizaciones graduadas (1.5) y robustez en sub-bloques del IS (1.6).
8. Métricas de salud del GP en cada run (log): % `-inf` gen0, nº de señales distintas,
   diversidad por generación. **Criterio de éxito de la fase:** `docs/diag_gp.py` debe dar
   <30% `-inf` en gen 0 y >80% de señales distintas.

**Fase C — Escala**
9. Vectorizar backtester (equivalencia exacta con el actual: test de comparación trade a
   trade contra `run_backtest`), caché de fitness por hash de señal, `multiprocessing`.
10. Subir a population ≥ 300, generations ≥ 40; actualizar el cálculo de `n_trials`.

**Fase D — Cambios estructurales (hipótesis a probar, una por experimento)**
11. Operadores temporales (LAG/CROSS/rolling) (1.3).
12. (Requiere 2b.4: convención de zona horaria) Features de contexto (hora, sesión, día, vol relativa) y cross-asset multi-horizonte.
13. Costes con swap y spread variable (2b.3). Semántica de posición: min-holding / salidas ATR; H1/H4 (1.4).
14. Solo si algo sobrevive a ≥3 seeds muy separadas **y** al IC correcto: proponer
    congelar y ejecutar el final test (decisión exclusiva del usuario).

**Reglas que se mantienen:** no tocar el final test; toda feature nueva con test de
causalidad; `pytest tests/ -v -W error::RuntimeWarning` antes y después; registrar
experimentos; documentar negativos; propuesta de plan y OK del usuario antes de editar.

---

## 6. Reproducibilidad de este análisis

- `docs/diag_gp.py`: reproduce todas las cifras de las secciones 1.1, 1.2, 2.1 y 2.2
  (`python -W ignore docs/diag_gp.py "$(pwd)"`, ~1–2 min, solo lectura).
- Cifras clave: 77% señales constantes; 87.5% `-inf` en gen 0; 21/120 señales distintas;
  bh_sharpe = 0.4726 = sqrt(bars_per_year/N); 99% del ruido puro "significativo".
- Archivos relevantes: `gp/individual.py` (init), `gp/grammar.py`, `gp/nodes.py`,
  `gp/operators.py`, `gp/evolution.py`, `fitness/objective.py`,
  `validation/walk_forward.py` (`_buy_and_hold_equity`, `_sharpe_ci`, `oos_return`),
  `experiments/tracker.py` (DSR), `features/registry.py`, `backtest/engine.py`.
