# EVAL_FOLD_XAUUSD.md — Comportamiento del GP en un solo fold (XAUUSD 15m)

> Diagnóstico, **no** un WFO completo. Un fold, una semilla: sirve para ver la *dinámica*
> del GP, no para concluir si hay edge. No toca el final test.
> Reproducir: `python -W ignore -m scripts.diag_fold --csv data_files/xauusd_15m_5y.csv --fold 5 --population 30 --generations 8 --seed 42 --output experiments_output/diag_fold/p30g8`
> (y `--population 100 --generations 20` para la segunda). Informes JSON en `experiments_output/diag_fold/` (ignorado por git).
> Ambas corridas registradas en `experiments_output/tracker.jsonl` (`kind: diag_fold`, n_trials 240 y 2000).

**Fold 5:** IS 2020-05-06 → 2021-06-01 (25 174 velas), OOS 2021-06-01 → 2021-09-03 (6 300).
20 features OHLC (las cross-asset no están disponibles en este CSV). Semilla del fold = 42 + 5.
Costes XAUUSD por defecto: spread 0.30 + slippage 0.10 → **0.50 USD por operación de 1 oz** (≈0.027%).

---

## 1. ¿Mueren demasiado rápido? Sí con la configuración del WFO (30×8); no por extinción, sino por falta de tiempo

### Corrida A — población 30, 8 generaciones (la del WFO baseline)

| Gen | % fitness `-inf` | Árboles distintos | Señales distintas (finitas) | Mejor | % sin condiciones |
|---|---|---|---|---|---|
| 0 | **100%** | 7 | 4 (0) | −inf | 87% |
| 1 | **100%** | 6 | 5 (0) | −inf | 83% |
| 2 | **100%** | 8 | 6 (0) | −inf | 70% |
| 3 | 97% | 12 | 7 (1) | −2.07 | 53% |
| 4 | 77% | 17 | 11 (4) | −2.07 | 43% |
| 5 | 40% | 14 | 9 (5) | −2.07 | 17% |
| 6 | 17% | 7 | 5 (2) | −2.07 | 7% |
| 7 | 13% | 6 | 5 (3) | −1.98 | 3% |

- **Las 3 primeras generaciones (37% de la búsqueda) no tienen ningún individuo viable**: no hay
  selección, es azar. El primer individuo con fitness finito aparece en la gen 3.
- La población empieza a ser viable en las gens 5–7, y **el run termina justo ahí**. El mejor se
  encuentra en la **última generación** (gen 7) y tiene **fitness −1.98, Sharpe IS −1.83**: el "ganador"
  pierde dinero incluso en el IS. Es un *flip* permanente `IF(NOT(GT(...)),LONG,SHORT)` (~1 500 operaciones
  en el IS, ≈1 cada 17 velas), que el tope `max_turnover_ratio=0.15` permite.
- Causas de `-inf` (acumulado): 61% señal constante (LONG/SHORT siempre), 27% FLAT total, 8% turnover, 3% FLAT>90%.
  Coincide con `ANALISIS_GP.md` §1.1 (inicialización degenerada).
- Su Sharpe OOS fue **+0.69**, es decir, ruido: un individuo con Sharpe IS −1.8 no puede "generalizar" a +0.69
  por mérito; no hay que leer nada en ese número.

### Corrida B — población 100, 20 generaciones

- Gen 0: 95% `-inf`; **a partir de la gen 3 ya solo ~20–30% `-inf`**. Con población 100 sí hay masa
  crítica de viables, pero sigue habiendo un **20–33% de `-inf` estable en todo el run** (señales constantes y turnover).
- El mejor fitness sube de −2.8 (gen 0) → +0.59 (gen 3) → +0.77 (gen 6) → **+0.85 (gen 10) y no mejora
  en las 10 generaciones siguientes** (estancamiento).
- **Pérdida de diversidad:** árboles distintos 79 (gen 4–5) → **34 (gen 19)**; señales distintas con fitness finito
  ~30 → **13–14**. Con población 100, solo ~14 comportamientos distintos sobreviven. Hay convergencia prematura y
  no hay caché de fitness ni mecanismo de diversidad (ANALISIS §1.7).
- Bloat controlado: tamaño medio sube a 22 (gen 4) y baja a ~10.
- Mejor individuo: `IF(AND(GT(return_20,roc_10),GT(rolling_std_20,atr_14_norm)),SHORT,LONG)`;
  Sharpe IS **+1.04** (103 operaciones), Sharpe OOS **−0.75**. **Corrección (ver §7):** no es un
  sobreajuste "clásico": esta regla está LONG el 98% del tiempo, y el oro cayó un 4% en ese OOS
  (buy & hold: Sharpe −1.09). Es sesgo direccional, no una regla que "deje de funcionar".

**Conclusión:** con 30×8 el GP "muere" antes de arrancar (no llega a ser viable hasta el final);
con 100×20 arranca pero se estanca y se homogeneiza en ~10 generaciones. Ambas configuraciones son
cortas para la gramática actual; la solución no es solo subir población, sino corregir inicialización y
mantener diversidad.

---

## 2. Comportamiento de la gramática

(Corrida B, genotipos únicos evaluados: 2 018 condiciones `GT/LT`.)

- **39% de las condiciones son degeneradas** (verdaderas <2% o >98% del tiempo en el IS): 780/2 018.
  La pareja más repetida es `GT/LT(rolling_std_20, atr_14_norm)` (**325** casos degenerados): dos medidas
  de volatilidad de escalas parecidas, una casi siempre mayor que la otra → condición casi constante que el
  GP usa como si fuese información, **incluida en el mejor individuo**. Otras: `return_5` vs `rsi_14` (escalas 0.002 vs 0–100),
  `dist_prev_high_20` vs `dist_ema_20_atr`.
- **Comparaciones triviales:** `GT(x, x)` (misma feature en ambos lados) aparece en **142 de 2 018 condiciones** (7%;
  p. ej. `dist_prev_high_20` vs sí misma: 87). Siempre es falso (o siempre verdadero con `GE`): pura constante.
- **Uso de nodos en genotipos viables:** dominan `GT` (1 267) > `AND` (673) > `IF` (605); `LT` solo 94 y `OR` 83.
  Los operadores aritméticos casi no se usan (MUL 56, NEG 33, ADD 23, SUB 7, DIV 3): el GP vive en reglas
  `AND` de comparaciones entre features.
- **Features favoritas de los viables:** `rolling_std_20` (394), `roc_10` (383), `atr_14_norm` (351),
  `dist_prev_high_20` (349), `return_20` (296). Son sobre todo de **volatilidad/retorno**; las
  distancias a medias (`close_over_sma_20`: 11, `dist_sma_20_atr`: 22) casi no se usan. Muy pocos individuos viables
  usan `close` vs media, que es justo la clase de regla "clásica".
- En la corrida A, con árboles de tamaño ~2–4 al principio, el 87% de los individuos de la gen 0 tenían **0 condiciones**.

---

## 3. Baselines manuales (¿qué pasa con reglas simples de cruce con close?)

Mismas condiciones del fold 5, 1 oz, costes por defecto. **IS** (25 174 velas):

| Regla | Operaciones | Bruto (USD) | Costes (USD) | Neto (USD) | Sharpe IS | Sharpe OOS | ¿Pasa tope turnover 0.15? |
|---|---|---|---|---|---|---|---|
| Nivel: LONG si close>SMA20, SHORT si no | 3 322 | −509 | 1 661 | −2 170 | −6.65 | −7.39 | sí (0.132) |
| Nivel con EMA20 | 3 594 | — | — | — | −6.73 | −7.25 | sí (0.143) |
| Cruce SMA20, mantener 4 velas | 3 322 | — | — | — | −8.56 | −8.82 | no (0.19) |
| Cruce SMA20, mantener 16 velas | 3 322 | — | — | — | −6.78 | −7.87 | no (0.152) |
| Cruce EMA20, mantener 16 velas | 3 594 | — | — | — | −7.59 | −9.94 | no (0.160) |
| Mejor GP corrida A (aprox.) | ~1 520 | −180 | 760 | −940 | −1.83 | +0.69 | sí |
| Señal aleatoria 50/50 cada vela | 12 665 | −16 | 6 333 | −6 349 | — | — | no |

**Lectura clave:** el **coste** es el factor dominante. Una regla de nivel pierde 1 661 USD solo en costes
(el bruto ya es −509). El `max_turnover_ratio = 0.15` permite hasta ~3 700 transiciones en el IS (una cada 7 velas),
que es un régimen donde casi ninguna señal sobrevive a 0.5 USD/operación. El tope es demasiado holgado para 15m;
la presión de selección debería empujar hacia pocas operaciones largas.

---

## 4. ¿Hace falta una métrica adicional para cruces con close?

**Respuesta corta: sí, pero no una "métrica de cruce" sola.** Hay tres carencias, de más a menos importante:

1. **No hay constante (ni un terminal 0).** Las features ya están normalizadas respecto a close
   (`close_over_sma_20 = close/SMA20 − 1`), por lo que "close cruza por encima de la SMA20" equivale a
   `close_over_sma_20 > 0`. Pero `GT(close_over_sma_20, 0)` **no es expresable**: el GP la sustituye por
   features cercanas a cero (`return_1`, `roc_10`, `log_return_*`, `return_20`...), lo que ocurre en
   **1 068 de 2 018 condiciones (53%)**. Esos "ceros" tienen std 0.001–0.005, parecida a la de `close_over_sma_20`
   (0.0026), así que el signo de la comparación es **ruido**, no el cruce real. Esto refuerza ANALISIS §1.2:
   añadir un terminal `ZERO` (o ERC) es la corrección más barata y de mayor impacto para estas reglas.
2. **Sin memoria, no hay "cruce" (evento), solo "estado".** `GT(f, 0)` describe estar por encima, no *haber cruzado*.
   Para eventos hacen falta (a) un nodo `LAG`/`CROSS_UP/CROSS_DOWN` (ANALISIS §1.3), o (b) la alternativa barata:
   **features precomputadas de cruce** como `bars_since_cross_sma_20` (con signo, acotada, p. ej. `clip(k, 0, 50)/50 * sign`).
   Recomendación: (b) primero, con test de causalidad; (a) si (b) funciona.
3. **Faltan referencias de otros horizontes.** Con una sola SMA/EMA (ventana 20) no se pueden expresar cruces
   entre medias (EMA rápida vs lenta) ni tendencias de fondo. Añadir `close_over_ema_50/200` y
   `ema_spread_20_100 = EMA20/EMA100 − 1`.

**Advertencia importante:** añadir estas features **no resolverá el problema de costes** (sección 3):
un cruce de SMA20 sin filtros hace 3 322 operaciones y pierde. Debe ir acompañado de (i) un tope de
turnover mucho más bajo para 15m (p. ej. ≤0.03–0.05) o un mínimo de tenencia, y (ii) mejor inicialización
y fitness (ANALISIS §1.1, §1.5). Toda feature nueva requiere test en `test_all_features_are_causal`.

---

## 5. Acciones recomendadas (derivadas de este diagnóstico)

1. **Mejorar la inicialización** y rechazar señales constantes en `random_individual` (reduce el 95–100% de `-inf` inicial).
2. **Prohibir/penalizar condiciones degeneradas:** `GT(x,x)` y comparaciones entre features con escalas incompatibles
   (normalización común + ERC/`ZERO`).
3. **Bajar `max_turnover_ratio`** para 15m (o añadir min-holding) y comprobar con `scripts/diag_fold.py` que
   las señales viables sobreviven a los costes.
4. **Diversidad:** caché de fitness y eliminación de clones; métrica de salud en cada run
   (`% -inf`, señales distintas, árboles distintos), que ya produce `diag_fold.py`.
5. **Features de cruce** (sección 4) *después* de 1–3, y siempre con test de causalidad.
6. Repetir este diagnóstico en otros folds y con 3 seeds muy separadas (42, 1042, 2042) antes de generalizar;
   aquí solo hay un fold y una semilla.

## 6. Limitaciones de este análisis

- Un solo fold (el 5) y una sola semilla (42): variabilidad no medida.
- Los baselines de cruce son reglas fijas de referencia, no una búsqueda exhaustiva.
- Los resultados "OOS" de los mejores individuos son de **un único bloque de 3 meses** y no tienen valor estadístico.
- No se usaron features cross-asset (no hay datos en el repo).
- Las cifras de la fila "Mejor GP corrida A (aprox.)" vienen de reconstruir manualmente la señal (difieren ~3% en nº de operaciones
  del informe JSON: 1 483 vs ≈1 520).

---

## 7. Matriz 4 folds × 3 semillas (100×20, sin cross-asset)

Folds 1, 5, 9, 13 (de 17), semillas 42, 1042, 2042 (12 corridas, 24 000 evaluaciones; registradas en el tracker con `kind: diag_fold`).
Reproducir: `scripts/diag_fold.py` por cada par, luego `python -m scripts.summarize_diag --dir experiments_output/diag_fold/matrix`
y `python -m scripts.check_exposure --csv data_files/xauusd_15m_5y.csv --dir experiments_output/diag_fold/matrix`.

### 7.1 Dinámica (consistente en las 12 corridas)

| Métrica | Valor |
|---|---|
| % `-inf` en gen 0 | 88–98% (media 94%) |
| Primer individuo viable | **gen 0 en todas** (con población 100 hay ~5% viables desde el inicio) |
| % `-inf` medio durante el run | 30–43% (media 35%) |
| Árboles distintos, pico → final | 75 de media → **30** |
| Señales distintas con fitness finito al final | 1–22 (media **9** de 100) |
| Mejor encontrado en gen | 2–19, mediana ~10: el estancamiento y la pérdida de diversidad se repiten |
| Condiciones degeneradas | 20–57% (media **40%**); comparaciones feature-vs-misma 2–12% |
| Condiciones contra una feature "pseudo-cero" | 21–60% (media **40%**) |

Con población 100 el GP no muere al inicio (a diferencia de 30×8), pero **colapsa en diversidad** (9 comportamientos
distintos de 100 al final). Pares degenerados más repetidos en las 12 corridas: `log_return_1` vs `return_1` (casi idénticas, 374),
`rolling_std_20` vs `atr_14_norm` (624 en ambos sentidos), `roc_10` vs `roc_10` (225), `hl_range_atr_14` vs `bb_position_20` (158).
Features más usadas por los viables: `bb_width_20`, `log_return_20`, `log_return_1`, `rolling_std_20`, `atr_14_norm`, `log_return_5`,
`close_over_ema_20`, `roc_10`: volatilidad y retornos, otra vez, no distancia a medias.

### 7.2 HALLAZGO PRINCIPAL: el GP converge a "comprar y mantener" (o "vender y mantener")

| Fold | Precio en OOS | Posición del mejor (OOS) | Sharpe GP (3 semillas) | Sharpe B&H LONG | Sharpe IS GP / B&H |
|---|---|---|---|---|---|
| 1 | +18.5% | 99–100% LONG | +3.94 / +4.22 / +4.07 | **+4.13** | +1.36…+1.56 / +1.40 |
| 5 | −4.0% | 96–98% LONG | −1.86 / −1.38 / −0.75 | −1.09 | +0.97…+1.33 / +0.70 |
| 9 | −11.1% | **100% SHORT** | +3.07 / +3.34 / +2.98 | −3.05 (LONG) | +0.24…+0.26 / −0.28 |
| 13 | +0.7% | 100% LONG | +0.22 / +0.17 / +0.04 | +0.28 | +0.36…+0.45 / +0.48 |

- **Los 12 mejores individuos son direccionales constantes** (≥96% del tiempo en un solo sentido).
  Sus pocas transiciones (4–60 en el OOS) apenas modifican el resultado: el Sharpe OOS es el de la tendencia del oro
  en ese bloque. En los folds con subida (1, 13) igualan al B&H y no lo superan (+4.07 vs +4.13; +0.04…+0.22 vs +0.28).
- El GP aprende **el signo de la deriva del IS** y lo mantiene: en el fold 9 el IS fue bajista (B&H −0.28), así que
  todas las semillas se hacen SHORT y en el OOS (−11%) "ganan" Sharpe +3. Eso es persistencia de tendencia del oro
  entre bloques consecutivos, **no habilidad de la regla**, y habría perdido igual de fuerte si el régimen hubiera girado.
- Por tanto, el Sharpe OOS medio de **+1.50 (positivo en 9 de 12)** de la matriz **no es evidencia de edge**: refleja
  que el oro tiene tendencia persistente trimestre a trimestre. Se anula con un benchmark adecuado (B&H o neutralidad direccional).
- Esto cuadra con la corrida A del fold 5 (Sharpe IS −1.8 con un flip permanente): cuando el IS es plano o ruidoso el GP flipa;
  cuando hay deriva, la acompaña.

### 7.3 Por qué ocurre (y qué cambiar)

El fitness es el **Sharpe absoluto** del IS, que premia cualquier sesgo direccional correcto, y la penalización de
inactividad/turnover favorece pocas operaciones. La estrategia más barata de optimizar es "estar siempre dentro en el sentido de la deriva".
**Recomendaciones (alta prioridad, antes de añadir más features):**
1. Evaluar el fitness como **exceso sobre B&H** (o un Sharpe de la parte neutral a la dirección: retornos de la estrategia menos
   `beta * retorno del mercado`) para que replicar la deriva no puntúe.
2. Alternativa/complemento: exigir exposición equilibrada (p. ej. `|exposición neta| ≤ 0.5` y mínimo de tiempo en cada sentido) o penalizarla.
3. Reportar siempre, junto a cada resultado, el B&H del **mismo bloque con los mismos costes** y la exposición neta del individuo
   (`scripts/check_exposure.py`).
4. Recordar que `_buy_and_hold_equity` del WFO está roto (ANALISIS §2.1): no usarlo como referencia hasta corregirlo.

Esta recomendación **se suma** a las de la sección 5 y debería ir en la Fase B del plan (fitness), antes de las features de cruce.

## 8. Respuesta a "¿necesito otros instrumentos / cross-asset?"

No. Todo lo anterior usa solo OHLC de XAUUSD (20 features). Las "reglas de cruce" son close contra sus propias medias.
Las features cross-asset (DXY, XAG, WTI) no se han usado en este diagnóstico (no hay datos en el repo).

---

## 9. Variantes de fitness sobre la misma matriz (4 folds × 3 semillas, 100×20)

Mismos folds (1, 5, 9, 13), semillas (42, 1042, 2042) y datos que §7. Implementación: rama `feat/fitness-alpha`
(`fitness_mode`, `max_abs_net_exposure`, `max_turnover_ratio`). Informes en `experiments_output/diag_fold/matrix_*`.

| Variante | `% -inf` medio | Exposición del mejor | Sharpe IS medio | Sharpe OOS medio (>0) | **Alpha OOS medio (>0)** |
|---|---|---|---|---|---|
| A. raw (Sharpe absoluto, turnover ≤0.15) | 35% | 11/12 casi constante (≥96%) | +0.80 | +1.50 (9/12) | **−1.17 (4/12)** |
| B. alpha (neutral a la dirección) | 32% | 11/12 casi constante | −0.39 | −0.83 (3/12) | **−1.67 (3/12)** |
| C. alpha + |exposición neta| ≤ 0.5 | 38% | 12/12 ~50/50, 400–700 transiciones en OOS | −2.53 | −3.57 (1/12) | **−3.72 (1/12)** |
| D. alpha + exposición ≤ 0.5 + turnover ≤ 0.03 | **100%** | no hay individuos válidos | n/a | n/a | n/a |

(Con un solo bloque OOS de 3 meses el error estándar del Sharpe anualizado es ~2.4 por celda: ninguna cifra individual es
concluyente; lo informativo son los patrones consistentes en las 12 celdas.)

### Lectura

1. **A → B:** el modo alpha elimina el "Sharpe OOS" inflado de A (+1.50 pasa a −0.83): confirma que aquel número venía de seguir
   la tendencia. Pero B **sigue convergiendo a posiciones casi constantes** (el alpha de una constante es ~0 − costes, y a
   partir de ahí el GP solo añade unas decenas de cambios que sobreajustan el IS: alpha IS hasta +1.6, OOS medio −1.7).
2. **C:** forzar equilibrio (|exposición| ≤ 0.5) empuja al GP a reglas ~50/50 que cambian de lado 400–700 veces por trimestre
   (1 600–2 800 en el IS). Con 0.5 USD por operación los costes las destruyen: **ningún individuo tiene fitness IS positivo**
   (mejor medio −2.76). No es "no hay edge" sino que lo único fácilmente alcanzable con esa restricción es ruido que paga costes.
3. **D:** al exigir además turnover ≤0.03 (≤ ~750 transiciones en el IS), **ningún individuo de la población (2 000 evaluaciones por
   corrida) cumple a la vez** mínimo de trades, exposición equilibrada y turnover bajo: el 100% es `-inf` en todas las generaciones,
   así que el "mejor" que reporta el script es el primer individuo de la población, sin significado.
   Es la evidencia directa del problema del **precipicio `-inf`** (ANALISIS §1.5): con restricciones duras y una población inicial
   casi aleatoria, la región factible está vacía en la práctica y no hay gradiente hacia ella.

### Conclusiones y siguiente paso

- El fitness neutral a la dirección es **necesario** (corrige un sesgo real), pero **no suficiente**.
- Para poder concluir algo sobre "edge" hace falta antes que el GP *pueda buscar* en la región útil (equilibrada y de baja rotación):
  **penalizaciones graduadas en lugar de `-inf`** (déficit de trades, exceso de turnover, exceso de exposición como violaciones continuas)
  e **inicialización que garantice individuos factibles** (ANALISIS §1.1, §1.5). Este es ahora el siguiente paso de mayor prioridad.
- Hasta entonces, ninguna de estas corridas permite afirmar ni descartar edge en XAUUSD 15m: A está contaminada por la deriva,
  B/C por costes y búsqueda, D por infactibilidad.
- Recomendación de uso inmediato: en `run_wfo`, usar `--fitness-mode alpha` para cualquier experimento nuevo y reportar siempre exposición neta y B&H
  del mismo bloque (`scripts/check_exposure.py`).
