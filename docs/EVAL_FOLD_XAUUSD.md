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
  Sharpe IS **+1.04** (103 operaciones), Sharpe OOS **−0.75**. Ejemplo típico de sobreajuste al IS
  (un único fold, no concluyente, pero coherente con ANALISIS §1.6).

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
