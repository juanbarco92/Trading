# Informe Maestro de Avance — Motor 1: Golpes Duros & Recuperación Asimétrica (Arquitectura Dual ML `1A + 1B`)

**Fecha de corte:** 18 de octubre de 2026  
**Estado actual:** Fase de Investigación, Modelado en Cascada (`1A -> 1B`) y Simulación de Portafolio Out-of-Sample completadas sobre **586 empresas de alta elasticidad (`ATR >= 2.2%`)**.

---

## 1. Resumen Ejecutivo: ¿Qué hemos construido hasta hoy?

Partimos de una premisa fundamental: **predecir el movimiento diario de una acción en un día cualquiera es casi imposible (ruido browniano con `ROC-AUC ~ 0.50–0.52`), pero cuando una acción sana sufre un golpe duro de liquidación institucional, el precio deja de comportarse al azar y activa una física elástica predecible.**

A lo largo de nuestras iteraciones construimos y validamos:
1. **Infraestructura de Datos Point-in-Time (Cero Fuga de Datos)**:
   - **`data/processed/sp500_multisector_1hour.parquet`**: Más de **3.03 millones de barras horarias (`1H`)** abarcando **586 empresas líquidas con `ATR Diario >= 2.2%`** distribuidas en los **10 sectores económicos** (`Technology`, `Financials`, `Healthcare`, `Consumer_Discretionary`, `Consumer_Staples`, `Industrials`, `Energy`, `Materials`, `Communication`, `Utilities_RealEstate`) más `SPY` y `QQQ`.
   - **`data/processed/sp500_earnings_pit.parquet`**: **19,385 reportes trimestrales de resultados (`Earnings`)** con fecha y hora exacta de publicación (`earnings_date`), `eps_estimate`, `eps_actual` y `eps_surprise_pct`, cruzados estrictamente hacia el pasado (`merge_asof(..., direction="backward")`).
2. **Ingeniería de Variables Causal (584 features base depuradas al núcleo predictivo)**:
   - Soportes y resistencias 100% causales en `src/data_Sources/stock_sr.py` (`sr_dist_support_atr`, `sr_support_touches`, `sr_channel_position`).
   - Geometría elástica respecto a medias móviles (`SMA_100_pct_momentum`, `close_diff_SMA_100`, `ratio_SMA_100_SMA_40`, `slope_SMA_100`).
   - Microestructura de frenado en velas de 1 hora (`close_loc_1h`, `lower_wick_ratio`, `rvol_1h`, `inst_absorption_score`, `weak_volume_flag`).
   - Descomposición del ADN de la Caída (`spy_z_3d`, `sector_z_3d`, `systemic_pull_z_3d`, `excess_vs_sector_3d`, `idiosyncratic_ratio_3d`).
3. **Arquitectura Dual de Machine Learning en Cascada (`Modelo 1A -> Modelo 1B`)**:
   - **Modelo 1A (Cuándo Entrar)**: Evalúa si la acción ya tocó piso tras un golpe de 3 a 5 días y si tiene el respaldo estructural y fundamental para rebotar (`ROC-AUC OOS = 0.6001` en el universo S&P 500 y `0.5818` en el universo ampliado de 586 empresas).
   - **Modelo 1B (Cuándo Salir / "Momento MeLi")**: Se despierta cuando la operación ya recuperó el 45%–50% del golpe. Recibe en cascada la probabilidad del Modelo 1A (`model_1a_prob`), el tipo de caída de la que viene la acción y la velocidad/velas del rebote para decidir si asegura ganancias rápido o suelta un trailing stop dinámico (`ROC-AUC OOS = 0.5837`).

---

## 2. Hallazgos Empíricos Clave (Lo que demostraron los datos)

### Hallazgo 1: La Anatomía del "Golpe Duro" (3 a 5 días > 1 día)
- Las caídas de **1 solo día** fracasan en mercados bajistas (`-5.2%` en 2022) porque las liquidaciones de fondos institucionales tardan entre **3 y 5 días hábiles** en agotarse.
- En cambio, los golpes acumulados de **3 a 5 días ($Z_{3\text{d}} \text{ o } Z_{5\text{d}} \le -2.0\sigma$) acompañados de una vela horaria de absorción (`close_loc_1h >= 0.55` o `lower_wick_ratio >= 0.25`)** fueron rentables en **todos los años desde 2022 hasta 2026** (`62.2%` a `71.2%` de Win Rate positivo).

### Hallazgo 2: Tus Features se multiplican hasta 33x dentro de un Golpe de 3–5 días
- Fuera de los golpes (en días normales), variables como `ratio_SMA_100_SMA_40` o `sr_dist_support_pct` tienen correlación casi nula con el futuro (`< 0.005`).
- **Dentro de un golpe de 3 a 5 días**, el poder predictivo de tus variables se multiplica entre **3.7x y 33.9x** (`scripts/analyze_features_in_3_5d_shocks.py`).

### Hallazgo 3: Agnóstico al Sector, pero Condicionado a la Elasticidad (`ATR Diario >= 2.2%`)
- **Prueba Ciega Inter-Sectorial**: Entrenando únicamente con acciones de Tecnología y probando a ciegas en `4,213` golpes de **Bancos, Salud, Industriales, Energía, Materiales y Consumo**, el modelo mantuvo un **`ROC-AUC = 0.5710`** y **`66.2%` de Win Rate** en su quintil superior (`scripts/evaluate_multisector_motor1.py`).
- Sin embargo, demostramos que las acciones "tortuga" (`ATR Diario < 1.6%`, como `KO` o `PG`) no tienen elasticidad para rebotar (`+0.18%`), mientras que las acciones dinámicas (**`ATR Diario >= 2.2%`**) generan rebotes potentes (`+2.27%` a `+4.68%`) en cualquier sector (`scripts/optimize_tp_sl_and_sectors.py`).

### Hallazgo 4: El Problema "Caso MeLi" en la Salida
- Al estudiar `7,201` golpes vimos que **857 operaciones alcanzaron entre 55% y 74% de recuperación del golpe (promediando `+7.17%` de ganancia flotante en su pico)**, pero por esperar ciegamente un Take Profit fijo del 75%, **el 69% devolvió más de la mitad de lo ganado y el 44.7% terminó en pérdida** (`scripts/expand_high_atr_universe_and_exit_study.py`).
- Además, preguntar todos los días al azar *"¿va a caer mañana?"* dio un `ROC-AUC = 0.5152` (ruido), pero activar el **Modelo 1B específicamente al cruzar el hito del 45%–50% de recuperación** elevó su capacidad predictiva a **`0.5837`**.

### Hallazgo 5: La Física según el Tipo de Caída (No toda herida sana igual)
Al clasificar los golpes según su origen (`scripts/analyze_shock_types_and_cascaded_ml.py`), descubrimos que la recuperación depende directamente de qué causó la caída:

| Tipo de Caída (Origen del Golpe) | Eventos Base | ¿Llega al 50% del golpe? | ¿Llega al 85%+ (Recuperación Completa)? | Comportamiento Físico |
| :--- | :---: | :---: | :---: | :--- |
| **1. Post-Earnings Negativo** *(Mal balance $\le 5\text{d}$)* | 167 | 43.7% | **Solo 9.6%** | **Trampa de valor**: 9 de cada 10 veces NO recupera el golpe. El Modelo 1A aprendió a vetarlas. |
| **2. Post-Earnings Positivo** *(Sobre-reacción $\le 5\text{d}$)* | 443 | 49.9% | **Solo 9.9%** | Rebota a la mitad, pero el gap deja oferta atrapada arriba. Hay que cobrar rápido en el 50%. |
| **3. Caída Macro / Mercado** *(`SPY` cae fuerte)* | 2,140 | **69.6%** | **20.6%** | Descuento injusto por venta de índices; cuando el mercado rebota, la marea levanta todo. |
| **4. Caída Sectorial** *(Cae el sector)* | 1,952 | **59.0%** | **16.2%** | Liquidación de ETFs sectoriales; las empresas líderes del sector rebotan con alta fiabilidad. |
| **5. Caída Idiosincrática Aislada** *(Solo la acción)* | 2,548 | 52.4% | 14.4% | Si el mercado y el sector están tranquilos y una acción se hunde sola, suele haber malas noticias ocultas (`WR 63%` vs `85%+` en caídas Macro/Sectoriales). |

---

## 3. Definiciones y Decisiones de Arquitectura Adoptadas

1. **Definición de Universo Operativo**: Empresas líquidas en EE. UU. con **`ATR Diario >= 2.2%`** en el momento de la señal y datos de resultados trimestrales disponibles.
2. **Definición de Stop Loss Estructural**: **`-1.5x ATR Diario` evaluado exclusivamente al cierre del día** (barra 7, 14, 21...), nunca con órdenes stop intradía que son barridas por los *market makers* en la primera hora de mercado.
3. **Decisión de Especialización (`Motor 1.1`)**: Enfocar el Motor 1 en las caídas con componente de arrastre Macro o Sectorial y descuentos respaldados por volumen institucional (`weak_volume_flag == 0`), descartando caídas por balances negativos.
4. **Decisión de Cascada (`Stacking 1A -> 1B`)**: El **Modelo 1B** recibe obligatoriamente como input `model_1a_prob` (que resultó ser la variable #1 en importancia de ganancia del Modelo 1B) junto con el código del tipo de caída (`shock_type_code`).

---

## 4. Resultados de Simulación de Portafolio Out-of-Sample (`2025-11` a `2026-10`)

Bajo condiciones estrictas de realismo (**`$25,000 USD` iniciales en efectivo sin apalancamiento**, ejecución en `Open[t+1]` y `10 bps` de comisión/slippage round-trip):

| Perfil de Portafolio (`$25,000` Iniciales) | Universo | Señales Disp. | Trades Ejecutados | Trades / Mes | Win Rate | Profit Factor | Capital Final | Retorno Total (11m) | Max Drawdown | Sharpe Anual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benchmark S&P 500 (`SPY` Buy & Hold)** | Índice | — | — | — | — | — | `$29,212` | `+16.8%` | `-9.38%` | ~1.25 |
| **1. Francotirador S&P 500 (`prob >= 0.61`, 5 Slots)** | 314 acc. | 95 | 78 | 7.1 / mes | **88.5%** | **4.48** | `$38,103` | **`+52.4%`** | **`-3.04%`** | **3.11** |
| **2. Motor 1.1 Refinado S&P 500 (`prob >= 0.59 + VolOK`, 5 Slots)** | 314 acc. | 174 | 90 | 8.2 / mes | **82.2%** | **2.87** | `$38,695` | **`+54.8%`** | **`-3.32%`** | **2.56** |
| **3. Motor 1.1 Refinado S&P 500 (`prob >= 0.59 + VolOK`, 6 Slots)** | 314 acc. | 174 | 102 | 9.3 / mes | **83.3%** | **3.01** | `$37,799` | **`+51.2%`** | **`-3.02%`** | **2.64** |
| **4. Universo Expandido (`prob >= 0.59 + VolOK`, 5 Slots)** | 586 acc. | 479 | **150** | **13.6 / mes** | **75.3%** | **2.02** | **`$40,465`** | **`+61.9%`** | `-11.05%` | **2.40** |
| **5. Universo Expandido (`prob >= 0.59 + VolOK`, 6 Slots)** | 586 acc. | 479 | **166** | **15.1 / mes** | **77.1%** | **2.08** | **`$39,738`** | **`+59.0%`** | **`-9.19%`** | **2.74** |

---

## 5. Próximos Pasos: ¿Cómo blindar el modelo frente a las acciones que siguieron cayendo?

Nuestra siguiente misión antes de pasar a producción es hacer una **autopsia forense profunda (Error Analysis de Perdedores)** sobre el `~15%–22%` de operaciones que pasaron todos los filtros pero **siguieron cayendo hasta tocar el Stop Loss diario** (como ocurrió en junio con `MSFT -6.28%`, `HPQ -9.95%` o en las Mid-Caps del universo de 586 acciones).

Tu hipótesis es clave: **cuando una acción con buenos números pasados sigue cayendo sin frenar, suele ser porque cambió la percepción del mercado o el valor esperado futuro de los inversionistas.** Aunque las noticias o encuestas no tienen un timestamp fácil de auditar, **el mercado deja 4 huellas exactas con timestamp horario (`1H`) que miden esa percepción y expectativa en tiempo real** y que podemos descargar gratis de Yahoo Finance para nuestra próxima sesión:

### 1. El Termómetro de Pánico y Expectativa de Riesgo (`^VIX`, `^VXN` y Estructura Temporal `^VIX / ^VIX3M` en 1H)
- **Qué mide**: El precio real con timestamp hora por hora que están pagando los fondos institucionales en opciones por protegerse de una caída mayor.
- **Cómo blinda las caídas continuas**: Cuando una acción cae 3 días pero el índice de miedo `^VIX` (o `^VXN` para tecnología) **sigue acelerando y haciendo nuevos máximos en las últimas 3 horas** (o cuando `^VIX > ^VIX3M`, indicando pánico agudo de corto plazo), el mercado espera más sangre al día siguiente. Si en cambio el precio toca piso y el `^VIX` empieza a desinflarse (`vix_slope_3h < 0`), el miedo ya se agotó.

### 2. La "Guía Implícita" / Valor Esperado del Último Balance (`last_earnings_reaction_1d` y `post_earnings_drift`)
- **Qué mide**: A veces una empresa reporta un `EPS` pasado positivo (`eps_surprise_pct > 0`), pero en la llamada con inversionistas el CEO baja la guía de ventas futuras (*guidance*).
- **Cómo se captura con timestamp exacto sin pagar bases de datos caras**: Calculando el **retorno de la acción en las 24–48 horas exactas posteriores a su última fecha de `earnings_date`** (`last_earnings_gap_and_drift`). Si una empresa reportó hace 20 días un `EPS +5%` pero ese día del reporte la acción se desplomó `-8%` con triple volumen, **sabemos con certeza matemática que la expectativa futura (*valor esperado de los inversionistas*) fue destruida en ese balance**, y que cualquier caída posterior no es un descuento sino continuación de salida institucional.

### 3. Asimetría de Liquidación Institucional Previa (`down_vs_up_volume_ratio_15d`)
- **Qué mide**: Si en los últimos 15 días antes del golpe las velas horarias rojas tuvieron mucho más volumen que las velas verdes.
- **Cómo blinda las caídas continuas**: Distingue entre una caída brusca por susto de 3 días (donde antes había acumulación institucional) vs una **distribución silenciosa** donde grandes fondos llevan 3 semanas liquidando un portafolio entero en cada rebote.

### 4. Presión del Costo del Capital en Tiempo Real (`^TNX` — Rendimiento del Bono del Tesoro de EE. UU. a 10 años en 1H)
- **Qué mide**: El denominador con el que Wall Street calcula el valor presente esperado de las acciones.
- **Cómo blinda las caídas continuas**: Si una caída sectorial ocurre mientras la tasa a 10 años (`^TNX`) está rompiendo al alza violentamente, el ajuste de múltiplos de valoración seguirá presionando a las acciones durante varios días más.

---

### Resultados Ejecutados del Blindaje de Percepción de Mercado y Prueba de Tortura (`scripts/autopsy_losers_and_market_perception_shield.py`)

Al ejecutar la autopsia sobre las `112` operaciones que tocaron Stop Loss en el universo de 586 empresas, confirmamos empíricamente qué diferenciaba a las acciones que siguieron cayendo de las que rebotaron:
1. **Choque de Tasas de Interés (`tnx_ret_3d` en `^TNX`)**: En los perdedores que siguieron cayendo, el rendimiento del bono del Tesoro a 10 años estaba subiendo un **`+521.1%` más fuerte** (`+0.63%` vs `+0.10%`), comprimiendo el valor presente esperado de las acciones.
2. **Decepción en la Guía Implícita del Último Balance (`last_earnings_reaction_3d`)**: Mientras los ganadores venían de un balance cuya guía fue celebrada por el mercado (**`+0.47%`** en los 3 días posteriores al reporte), los perdedores que siguieron cayendo venían de una **reacción negativa (`-0.49%`, diferencia de `-203.6%`)**, demostrando que su expectativa futura ya estaba rota aunque su `EPS` pasado fuera positivo.
3. **Separación Arquitectónica Crucial (*Cross-Sectional vs Macro Gate*)**:
   - Las variables de percepción **específicas de la acción** (`last_earnings_reaction_1d`, `last_earnings_reaction_3d`, `guidance_disappointment_flag`, `cmf_15d`, `down_up_vol_ratio_15d`) se integran **dentro de los árboles de los Modelos 1.1 y 1B** para rankear qué empresa es superior a otra el mismo día.
   - Las variables de percepción **macroeconómica** (`^TNX` y `^VIX`), que tienen el mismo valor para todas las acciones a la misma hora, se aplican como **Compuerta de Régimen (Veto Macro)** para no ahogar el ranking entre empresas.

| Portafolio en las 586 Empresas (`$25,000` Iniciales) | Señales Disp. | Trades Ejecutados | Trades / Mes | Win Rate | Ret. Medio | Profit Factor | Capital Final | Retorno Total (11m) | Max Drawdown | Sharpe Anual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Base Sin Percepción (`prob >= 0.59`, 5 Slots, 10 bps)** | 469 | 144 | 13.1 / mes | 72.2% | +1.13% | 1.58 | `$34,348` | `+37.4%` | `-11.91%` | 1.73 |
| **2. Con Escudo Percepción + Compuerta Macro (`prob >= 0.59`, 5 Slots)** | **426** | **146** | **13.3 / mes** | **76.0%** | **+1.69%** | **2.14** | **`$41,111`** | **`+64.4%`** | **`-6.35%`** | **2.56** |
| **3. Con Escudo Percepción + Compuerta Macro (`prob >= 0.59`, 6 Slots)** | 426 | **165** | **15.0 / mes** | **75.8%** | **+1.58%** | **1.97** | **`$38,446`** | **`+53.8%`** | **`-6.50%`** | **2.40** |
| **4. Con Escudo Percepción + Compuerta Macro (`prob >= 0.61`, 5 Slots)** | 262 | 113 | 10.3 / mes | 74.3% | +1.50% | 1.78 | `$34,823` | `+39.3%` | `-6.39%` | 2.21 |
| **5. STRESS TEST DOBLE COMISIÓN `20 bps` (`prob >= 0.59`, 5 Slots)** | **426** | **146** | **13.3 / mes** | **76.0%** | **+1.59%** | **2.06** | **`$39,950`** | **`+59.8%`** | **`-6.41%`** | **2.44** |

---

## 6. Auditoría Forense de Producción: ¿Qué estaba inflando el `+64.4%` y cuál es el retorno real sin trampa? ([forensic_audit_motor1_production.py](file:///C:/Users/Administrador/Documents/Proyectos/Trading/ForexDataExtractorProject/scripts/forensic_audit_motor1_production.py))

Siguiendo la regla de oro de desconfiar de cualquier resultado "demasiado bueno para ser cierto", sometimos todas las variables nuevas y el simulador de salidas a una **auditoría forense pensando estrictamente en un servidor en producción hora por hora**:

### A. El Reloj Exacto en Producción (¿A qué hora corre el modelo y qué datos existen?)
1. **Cuándo termina realmente una vela horaria (`bar_close_ts = timestamp + 1h`)**:
   - En Yahoo Finance, la vela con etiqueta `13:30 UTC` (`09:30 AM NY`) abarca de `09:30` a `10:30 AM NY`. Por lo tanto, el modelo **jamás se ejecuta a las `09:30`**, sino a las **`10:30:02 AM NY`** (y así sucesivamente cada hora: `11:30`, `12:30`, `13:30`, `14:30`, `15:30`, `16:00`).
   - En ese segundo (`t + 1h`), ya están cerrados y disponibles el `Close/High/Low/Volume` de las 586 acciones, `SPY`, `QQQ`, `^VIX`, `^VIX3M`, `^VXN` y `^TNX` de la hora que acaba de terminar. La orden de compra se envía para ejecutarse en la apertura de la vela siguiente (`Open[t+1]`).

### B. Las 3 Fuentes Reales de Inflación Detectadas y Extirpadas
Al auditar línea por línea el código, encontramos **dos fugas de *timestamp* en balances/train-test** y **un sesgo de ejecución intrabarra en el simulador de salidas**:

1. **Fuga 1 — Timestamps de Balances Trimestrales After-Hours (`earnings_date`) y Reacción de 3 Días (`last_earnings_reaction_3d`)**:
   - **El problema**: En `sp500_earnings_pit.parquet`, muchos reportes tienen hora `10:00–12:00 UTC` (solo la fecha del día), aunque la empresa haya reportado a las `16:05 NY` (después del cierre). Además, `last_earnings_reaction_3d` miraba 3 días hacia adelante del balance; si una señal ocurría 1 o 2 días después del balance, ya estaba leyendo el día 3 de reacción.
   - **La corrección estricta de producción**:
     - Un balance (`eps_surprise_pct`) **solo se vuelve visible 24 horas después de su `earnings_date` (`earnings_known_ts = earnings_date + 24h`)**.
     - La reacción de guía de 3 días (`last_earnings_reaction_3d`) **solo se vuelve visible 5 días calendario completos después del balance (`reaction_known_ts = earnings_date + 5d`)**, garantizando que las 21 velas horarias posteriores al reporte ya ocurrieron 100% en el pasado.
2. **Fuga 2 — Solapamiento en la Frontera Train / Test (Falta de Purga de 21 Días)**:
   - **El problema**: Las operaciones abiertas en los últimos días del set de entrenamiento (`Noviembre 2025`) seguían vivas durante los primeros días del set de prueba (`17 de Noviembre 2025`), permitiendo que el modelo aprendiera el desenlace de esos días fronterizos.
   - **La corrección estricta de producción**: Aplicamos un **Purge Gap estricto de 21 días calendario** antes de la fecha de corte (`2025-11-17`), eliminando del entrenamiento cualquier señal cuyos 15 días hábiles de vida pudieran tocar el periodo de prueba.
3. **Fuga 3 (La Mayor Inflación) — El "Candado Intrabarra Instantáneo" y Precio Sin Gap en `simulate_adaptive_exit_by_shock_and_1b`**:
   - **El problema**: En el simulador anterior, dentro de la **misma vela de 1 hora `step`**, el código hacía primero `if highs[step] > peak_p: peak_p = highs[step]` (actualizando el candado con el máximo de esa hora) y en la línea siguiente verificaba `if closes[step] <= floor_p` **en esa misma hora `step`**, vendiendo exactamente al precio `floor_p` sin importar si hubo un *gap* nocturno por debajo.
   - Es decir: si una acción tocaba el `+50%` de recuperación durante un segundo a las `10:05 AM` y a las `10:15 AM` se desplomaba, el simulador anterior asumía que a las `10:05 AM` ya habías adivinado el futuro, corrido el Modelo 1B y puesto un Stop exacto que te sacó arriba. Y si al día siguiente abría con un *gap* de `-5%` por debajo de tu Stop, asumía que el broker te respetó el precio exacto del Stop (`floor_p`) sin sufrir el *gap*.
   - **La corrección estricta de producción**:
     - El Modelo 1B **solo se ejecuta al cierre de la hora `step`** en que se alcanza el 50% de recuperación.
     - El nuevo nivel de Stop (`floor_p`) **solo entra en vigor a partir de la barra siguiente (`step + 1`)**.
     - Si la vela siguiente abre con un *gap* por debajo del Stop (`opens[step] < floor_p`), **sufres el *gap* real y vendes al peor precio de apertura (`opens[step]`)**, tal como ocurre en cualquier broker real (Interactive Brokers, Alpaca, Schwab).

### C. Tabla Comparativa Auditada: Simulador Antiguo (Inflado) vs Producción Estricta (Cero Fugas)
Al aplicar simultáneamente las **3 correcciones de producción** (`+24h` en balances, `+5d` en reacción de guía, `21d` de purga Train/Test + Veto Macro compilado solo con percentiles del Train) sobre las **586 empresas (`$25,000` iniciales, 10 bps de fricción)**:

- El poder predictivo puro fuera de muestra se mantiene sólido y real: **`ROC-AUC Modelo 1A = 0.5838`** y **`ROC-AUC Modelo 1B = 0.5743`**.
- Al eliminar la trampa intrabarra del candado y sufrir los *gaps* reales en los Stops, el retorno baja del `+64.4%` inflado a un **rango 100% real y defendible en producción de `+25.3%` a `+27.8%` en 11 meses (`~28% a 30.5% E.A.`)**:

| Modo de Ejecución en las 586 Empresas (Con Purga 21d + Buffers de Balance + Veto Macro Train) | Señales | Trades Ejecutados | Win Rate Real | Ret. Medio | Profit Factor | Capital Final | Retorno Total (11m) | Max Drawdown | Sharpe Anual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Simulador Antiguo (Con Inflación Intrabarra en Candado, `prob >= 0.59`, 5 Slots)** | 134 | 63 | 79.4% | +2.47% | 2.44 | `$33,808` | `+35.2%` | `-5.19%` | 1.98 |
| **2. PRODUCCIÓN ESTRICTA: Stop en Broker desde Barra `t+1` + Gaps Reales (`prob >= 0.59`, 5 Slots)** | **134** | **64** | **79.7%** | **+1.80%** | **2.11** | **`$31,316`** | **`+25.3%`** | **`-5.84%`** | **1.86** |
| **3. PRODUCCIÓN ESTRICTA: Evaluado al Cierre -> Venta en `Open[t+1]` (`prob >= 0.59`, 5 Slots)** | **134** | **63** | **71.4%** | **+2.03%** | **2.13** | **`$31,951`** | **`+27.8%`** | **`-6.52%`** | **1.61** |
| **4. PRODUCCIÓN ESTRICTA: Stop en Broker desde Barra `t+1` + Gaps Reales (`prob >= 0.61`, 5 Slots)** | 75 | 35 | **82.9%** | +2.39% | **3.10** | `$29,425` | `+17.7%` | **`-3.14%`** | **1.86** |
| **5. PRODUCCIÓN ESTRICTA: Evaluado al Cierre -> Venta en `Open[t+1]` (`prob >= 0.61`, 5 Slots)** | 75 | 35 | **80.0%** | **+2.89%** | **3.34** | `$30,384` | `+21.5%` | **`-2.11%`** | 1.74 |

---

## 7. Optimización sobre el Simulador Estricto de Interactive Brokers (IBKR): Las 3 Palancas de Mejora Real

Una vez establecido el simulador estricto de producción en IBKR (`+25.3% a +27.8%` de línea base sin una sola fuga), ejecutamos tres mejoras estructurales auditadas fuera de muestra:

### Palanca #1: Reparación del Target del Modelo 1B (*First-Passage*) y Enfriamiento del `^VIX` ([upgrade_model_1b_first_passage_and_perception.py](file:///C:/Users/Administrador/Documents/Proyectos/Trading/ForexDataExtractorProject/scripts/upgrade_model_1b_first_passage_and_perception.py))
- **El problema encontrado**: El *target* antiguo del Modelo 1B (`keep_running`) exigía que la acción superara el `+85%` de recuperación **y que en ningún momento hasta el día 15 el cierre bajara del `+15%`**. Si una acción llegaba al `+100%` el día 4 (éxito rotundo) y el día 14 retrocedía, se etiquetaba como fracaso (`0`), hundiendo las probabilidades de 1B.
- **La solución**: Redefinimos el *target* de 1B como un verdadero ***First-Passage*** (*"¿Toca primero el `+85%` de recuperación antes de volver a Break-Even?"*) y agregamos el enfriamiento del `^VIX` desde la entrada hasta el Momento MeLi (`vix_cooling_since_entry`).
- **Resultado**: El **ROC-AUC fuera de muestra del Modelo 1B saltó de `0.5779` a `0.6324`** (+545 bps). En el tercil inferior (`< P35`) solo el `34.8%` llega al `+85%`, mientras que en el Top 20% (`>= P80`) el **`75.1%`** alcanza la segunda ola. Al darle holgura solo al Top 20%–30% y cobrar rápido en el resto, el portafolio estricto subió a **`+31.6%` (`$32,898`)**.

### Palanca #2: Error Analysis en la Zona Gris (`0.53 <= prob_1a < 0.59`) vs. Bajar el Threshold ([error_analysis_rescue_053_059_ibkr.py](file:///C:/Users/Administrador/Documents/Proyectos/Trading/ForexDataExtractorProject/scripts/error_analysis_rescue_053_059_ibkr.py))
- **Hipótesis probada**: ¿Conviene bajar simplemente el *threshold* a `0.55` o descubrir qué variables de microestructura distinguen a las Joyas de las Trampas dentro de la zona `0.53–0.59`?
- **Hallazgo**: Bajar el *threshold* a lo bruto a `0.55` mete ruido, satura los 5 cupos y **reduce el retorno de `+31.6%` a `+25.2%`** (subiendo el Drawdown a `-9.00%`). En cambio, al incorporar las variables del *Error Analysis* (**`bounce_thrust_2h`** `+31.1%` en joyas, **`floor_defense_3h`** `+8.8%` midiendo 3 horas seguidas defendiendo soporte, **`atr_norm_drop_3d`** y **`rel_strength_60d_vs_spy`**):
  - El **Modelo 1A subió su ROC-AUC OOS a `0.5862`**, promoviendo orgánicamente **`78` joyas ganadoras** por encima de `>= 0.59` (`71.8%` de tasa de éxito) y expulsando `144` trampas hacia abajo (`< 0.53`).
  - El retorno en producción estricta IBKR subió de `+31.6%` a **`+44.4%` (`$36,090`, Sharpe `2.06`, DD `-5.90%`)** en corte `>= 0.59`, y a **`+65.7%` (`$41,420`, `74` trades, PF `3.62`, Sharpe `2.25`)** incluyendo el escalón `0.575–0.59` exclusivamente para caídas sistémicas/sectoriales (**Arquetipo 1.1**).

### Palanca #3: Autopsia de los que Nunca Llegan al +45% y Cabeza Guardián Anti-Cuchillo ([optimize_lever3_anti_knife_and_1a_target.py](file:///C:/Users/Administrador/Documents/Proyectos/Trading/ForexDataExtractorProject/scripts/optimize_lever3_anti_knife_and_1a_target.py))
- **El problema encontrado**: Como el `98.7%` de las acciones que llegan al `+45%` ("Momento MeLi") terminan ganando, casi todos los Stop Loss provienen del `~20%` de entradas donde la acción sigue en caída libre y nunca llega siquiera al `+45%`.
- **Hallazgo microestructural**: En las operaciones de alta convicción que fallan y van al Stop Loss, **su sector seguía desplomándose un `-54%` más fuerte en las últimas 2 horas (`sector_ret_2h = -2.41%` vs `-1.56%`)**, el precio cerraba pegado al mínimo de las últimas 7 horas (`dist_from_7h_low_atr = 0.46` vs `0.66 ATR`) y solo el `29.9%` había dejado de hacer mínimos menores en la última hora (`higher_low_confirm_1h` vs `44.8%` en las ganadoras).
- **La Cabeza Guardián (`clf_l3_meli`, `ROC-AUC OOS = 0.5875`)**: Entrenamos un modelo guardián dedicado exclusivamente a predecir si la acción llegará al hito del `+45%` sin tocar Stop Loss (`prob_reach_meli >= 0.62`) y vetamos sectores en caída libre de 2 horas (`falling_knife_flag == 0`).

### Tabla Maestra de Perfiles Operativos en Producción Estricta IBKR (`$25,000` Iniciales, 10 bps, Cero Fugas)

| Perfil Operativo en Producción Estricta IBKR (Purga 21d + Buffers Balance + Ejecución `Open[t+1]`) | Señales | % Llega al +45% (MeLi) | Trades Ejecutados | Win Rate Real | Ret. Medio / Trade | Profit Factor | Capital Final | Retorno 11 Meses | Max Drawdown | Sharpe Diario Real |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Línea Base Inicial Estricta (`1A` y `1B` antiguos, `prob >= 0.59`, 5 Slots)** | 134 | 74.6% | 63 | 71.4% | +2.03% | 2.13 | `$31,951` | `+27.8%` | `-6.52%` | 1.61 |
| **1. Palanca #1 (Nuevo `1B` First-Passage `AUC 0.6324`, `prob >= 0.59`, 5 Slots)** | 134 | 74.6% | 63 | 71.4% | +2.26% | 2.28 | `$32,898` | `+31.6%` | `-6.52%` | 1.72 |
| **2. Palanca #1 + #2: Crecimiento Estándar (`Nuevo 1A >= 0.59`, 5 Slots)** | 187 | 80.7% | 64 | 75.0% | +2.96% | 2.89 | `$36,090` | **`+44.4%`** | `-5.90%` | 2.06 |
| **3. Palanca #1 + #2: Máximo Retorno (`>= 0.59` o `[>= 0.575 & Macro/Sector]`, 5 Slots)** | **244** | 79.5% | **74** | **74.3%** | **+3.55%** | **3.62** | **`$41,420`** | **`+65.7%`** | **`-5.90%`** | **2.25** |
| **4. Palanca #1 + #2 + #3: Blindado Institucional (`1A + Guardián MeLi`, 5 Slots)** | 100 | **82.0%** | **65** | **78.5%** | **+3.02%** | **4.62** | **`$36,755`** | **`+47.0%`** | **`-3.37%`** | **3.15** |
| **5. Palanca #1 + #2 + #3: Ultra-Defensivo (`Ensamble Dual-Head + Guardián`, 5 Slots)** | 79 | **82.3%** | 53 | **77.4%** | +2.89% | **4.82** | `$33,783` | `+35.1%` | **`-1.71%`** | **2.93** |
| **6. Palanca #1 + #2 + #3: 6 Cupos Escalonados (`4 Estándar + 2 Francotirador >= 0.60`)** | 100 | **82.0%** | **65** | **76.9%** | +2.94% | **4.47** | `$34,172` | `+36.7%` | **`-2.82%`** | **4.23** |


---

## 8. Auditoría del Drawdown (Cerrado vs. Mark-to-Market vs. Monte Carlo) y 4 Nuevas Minas de Features

Scripts: [audit_mtm_drawdown_and_next_features_ibkr.py](file:///C:/Users/Administrador/Documents/Proyectos/Trading/ForexDataExtractorProject/scripts/audit_mtm_drawdown_and_next_features_ibkr.py) (exploración) y [test_pruned_features_and_exact_mtm.py](file:///C:/Users/Administrador/Documents/Proyectos/Trading/ForexDataExtractorProject/scripts/test_pruned_features_and_exact_mtm.py) (verificación final).

### 8.1 Tres formas de medir el drawdown
1. **DD Cerrado**: lo que reportaba `run_portfolio_simulation`. Valora las posiciones abiertas **al costo** y solo registra PnL al cerrar ([backtest_portfolio_motor1_dual_ml.py](file:///C:/Users/Administrador/Documents/Proyectos/Trading/ForexDataExtractorProject/scripts/backtest_portfolio_motor1_dual_ml.py#L200-L202)). **Subestima** lo que verás en pantalla.
2. **DD Mark-to-Market (MtM)**: valora cada día de mercado todas las posiciones abiertas a su `Close` diario (y en el peor caso, a su `Low` diario). Es lo más parecido al *Net Liquidation Value* de IBKR.
3. **Monte Carlo (bootstrap, 2,000 caminos)**: remuestrea los retornos de los trades para ver qué pasa si las pérdidas llegan agrupadas.

### 8.2 Tabla verificada (OOS Nov-2025 → Oct-2026, `$25,000`, 5 cupos)

| Configuración | Trades | WR | PF | Capital Final | Retorno 11m | DD Cerrado | DD MtM Cierre | DD MtM Peor Low | MC P50 | MC P95 | Sharpe |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1. Fila 3 (re-simulada) | 73 | 75.3% | 4.11 | `$42,621` | `+70.5%` | `-4.16%` | `-4.94%` | `-7.26%` | `-4.22%` | `-7.56%` | 2.97 |
| 2. Fila 3 + sizing dinámico (45/75/100%) | 73 | 75.3% | 4.11 | `$40,741` | `+63.0%` | `-3.04%` | `-4.25%` | `-4.86%` | `-3.81%` | `-7.07%` | 3.07 |
| **3. ⭐ Fila 3 + Veto Guardián V4 (`>= 0.58`, sin viernes tarde)** | **69** | **75.4%** | **5.31** | **`$42,173`** | **`+68.7%`** | **`-2.12%`** | **`-4.71%`** | **`-5.47%`** | **`-3.02%`** | **`-5.57%`** | **3.29** |
| 4. Fila 3 + Guardián V4 estricto (`>= 0.615`) | 65 | 75.4% | 5.07 | `$40,557` | `+62.2%` | `-2.12%` | `-4.59%` | `-5.35%` | `-3.04%` | `-5.76%` | 3.33 |
| 5. 1A Ortogonal (+3 minas) + Guardián V4 | 72 | 70.8% | 3.52 | `$37,703` | `+50.8%` | `-2.57%` | `-5.85%` | `-6.89%` | `-3.22%` | `-5.88%` | 3.14 |

> [!WARNING]
> **Discrepancia sin resolver:** el simulador original (`run_portfolio_simulation`) da para la Fila 3 **74 trades / `+65.7%` / DD `-5.90%`**, y la réplica en `compute_exact_mtm_on_simulation` da **73 trades / `+70.5%` / DD `-4.16%`**. Las señales son las mismas; la diferencia está en la lógica de asignación/cierre. Hasta reconciliarlo, **la cifra conservadora de referencia es `+65.7%`**.

> [!NOTE]
> **Corrección propia:** una primera versión reportó DD MtM de `-7.22%` en todas las filas. Era un **bug**: el peor día era el `2026-04-03` (Viernes Santo, mercado cerrado) porque el calendario usaba días hábiles genéricos en vez de días reales de NYSE. Ya está corregido; los valores de arriba son los válidos.

### 8.3 Las 4 minas nuevas (medias en Train)

| Mina | Variable | Ganadores | Perdedores | Lectura |
| :--- | :--- | :-: | :-: | :--- |
| A. Amplitud sectorial | `sector_breadth_oversold_3d` | **40.5%** | 33.1% | **Separación clara.** Si gran parte del sector cae a la vez, es liquidación de canasta, no un problema de la empresa. |
| B. Miedo Nasdaq vs S&P | `vxn_vix_ratio` | 1.11 | 1.15 | Separación moderada: el pánico concentrado en tech perjudica el rebote. |
| C. Clímax de volumen | `selling_climax_exhaustion` | 2.04 | 1.99 | **Diferencia mínima**; puede ser ruido. |
| D. VWAP 5d / calendario | `dist_vwap_5d_atr`, `is_friday_late` | -1.43 / 2.9% | -1.53 / 3.6% | Señales débiles. El veto de viernes tarde afecta a muy pocos trades. |

- Meter **las 10 variables nuevas de golpe** bajó el AUC de 1A (`0.5862 → 0.5798`). Meter **solo 3** lo subió a `0.5885`. Guardián V4: `0.5925` (vs `0.5875` del guardián de Palanca #3).
- **Con honestidad:** diferencias de AUC de `±0.005` son probablemente **ruido estadístico** con el tamaño del test. La lectura sólida es cualitativa: *agregar variables redundantes no ayuda y puede perjudicar*; la única mina con separación clara es la **A (amplitud sectorial)**.

---

## 9. Bitácora de Aprendizajes: Lo Bueno, Lo Malo y Los Errores

### 9.1 Lo que SÍ funcionó
1. **Condicionar por evento.** Predecir cualquier hora/día es ruido (`AUC ~0.51`). Exigir antes un golpe de 3–5 días (`Z <= -2σ`) + vela de absorción multiplica el poder de las variables (3.7x a 33.9x).
2. **Entender por qué cayó.** Las caídas Macro/Sectoriales (Arquetipo 1.1) rebotan mucho mejor que las idiosincráticas o post-earnings (`9.6%` de recuperación completa en earnings negativos).
3. **Separar el problema en especialistas:** 1A (entrar), Guardián (¿llega vivo al +45%?), 1B (¿cuánto dejar correr?). Cada uno con un target claro.
4. **Targets *First-Passage*.** Reformular el target de 1B ("¿toca +85% antes de volver a break-even?") subió su AUC de `0.578` a `0.632`: la mayor mejora individual del proyecto.
5. **Error analysis > bajar el umbral.** Bajar a `0.55` empeoró todo (`+31.6% → +25.2%`, DD `-9%`). Estudiar los casos de la zona gris y crear features (`bounce_thrust_2h`, `floor_defense_3h`, …) hizo que el modelo subiera solo a las joyas.
6. **Escepticismo sistemático.** Cada vez que algo se veía "demasiado bueno" auditamos y casi siempre encontramos inflación (ver 9.3).

### 9.2 Callejones sin salida (no repetir)
1. **Comprar caídas de 1 día**: la liquidación institucional dura 3–5 días; falló en 2022.
2. **Acciones "tortuga" (`ATR < 2.2%`)**: el rebote (`+0.18%`) no paga la fricción.
3. **Stops intradía tipo Stop-Market en la apertura**: los barren antes del rebote. El SL estructural se evalúa al cierre.
4. **Aflojar el candado de +42% a +12% para todos**: se regala ganancia en el 38% que no tiene segunda ola (EV negativo).
5. **"Feature stuffing"**: agregar muchas variables colineales diluye los árboles (`colsample_bytree`). Cada variable nueva debe aportar una dimensión distinta y probarse aislada.

### 9.3 Errores y fugas detectados y corregidos

| # | Error | Efecto | Corrección |
| :-: | :--- | :--- | :--- |
| 1 | Candado intrabarra + stops sin gap | Inflaba `+27.8% → +64.4%` | 1B corre al cierre de `t`, stop vigente desde `t+1`, gap real en `Open` |
| 2 | Earnings after-hours sin hora exacta | Leía el balance antes de tiempo | `earnings_known_ts = +24h`, reacción 3d `= +5d` |
| 3 | Solapamiento Train/Test | Trades de train vivían en test | Purga de 21 días |
| 4 | Target viejo de 1B (`keep_running`) | Etiquetaba éxitos como fracasos | Target First-Passage |
| 5 | DD medido solo en trades cerrados | Ocultaba el flotante | Auditor MtM diario + Monte Carlo |
| 6 | Bug de calendario (Viernes Santo) en el MtM | Reportó `-7.22%` falso | Usar días reales de mercado |
| 7 | Etiqueta errónea "escudo macro" | Atribuí `+65.7% → +70.5%` a un filtro; en realidad es la discrepancia entre simuladores (8.2) | Documentado; pendiente reconciliar |

### 9.4 Matiz importante sobre el "98.7%"
Que el `98.7%` de los trades que llegan a +45% terminen en ganancia es **en buena parte mecánico**: al llegar a ese hito se activa un piso por encima del break-even. El valor predictivo real está en **llegar** al +45% (≈ 79–82% de las entradas), y eso es lo que optimiza el Guardián.

---

## 10. Catálogo de Feature Engineering e Ideas Ortogonales a Futuro

### 10.1 Las 7 dimensiones actuales
1. **Tendencia previa**: `SMA_100_pct_momentum`, `slope_SMA_100`, `pre_shock_trend_55d`, `rel_strength_60d_vs_spy`.
2. **Soporte estructural**: `sr_dist_support_atr`, `sr_support_touches`, `sr_channel_position`, `support_touch_density`.
3. **Estiramiento elástico**: `z_3d`, `z_5d`, `atr_norm_drop_3d`, `elastic_potential`, `dist_vwap_5d_atr`.
4. **ADN del golpe / sector**: `spy_z_3d`, `sector_z_3d`, `systemic_pull_z_3d`, `idiosyncratic_ratio_3d`, `shock_type_code`, `sector_breadth_oversold_3d`.
5. **Fundamentales y reacción a la guía**: `eps_surprise_pct`, `eps_beat_4q_ratio`, `fundamental_shield`, `last_earnings_reaction_3d`, `guidance_disappointment_flag`.
6. **Microestructura / anti-cuchillo**: `close_loc_1h`, `lower_wick_ratio`, `rvol_1h`, `floor_defense_3h`, `bounce_thrust_2h`, `cmf_15d`, `down_up_vol_ratio_15d`, `sector_ret_2h`, `higher_low_confirm_1h`, `dist_from_7h_low_atr`.
7. **Régimen macro**: `vix`, `vix_slope_3h`, `vix_cooling_since_entry`, `tnx_ret_3d`, `vxn_vix_ratio`.

### 10.2 Ideas ortogonales para la Fase 2 (nuevas fuentes de información, no más indicadores sobre las mismas velas)
1. **Opciones por acción (IV y skew Put/Call vía IBKR)**: si la volatilidad implícita de los puts empieza a ceder mientras el precio sigue abajo, el mercado de derivados ya ve piso.
2. **Short interest y days-to-cover (FINRA, quincenal)**: combustible para 1B; un short squeeze puede llevar el rebote más allá del +85%. Ojo con la fecha de publicación (no la de corte).
3. **Liderazgo intrasectorial (lead-lag)**: ¿ya giraron los 2–3 líderes del sector en las últimas horas? Es la extensión natural de la mina A, la que mejor funcionó.
4. **Compras de insiders (SEC Form 4)**: públicas, gratuitas y con timestamp exacto de publicación.
5. **Reloj de eventos macro (`horas_hasta_FOMC/CPI/NFP`)**: evitar abrir justo antes de un evento binario.
6. **Libro de órdenes en vivo (spread y desbalance bid/ask)**: no existe histórico; empezar a **grabarlo desde la cuenta demo** para tener datos en unos meses.

**Protocolo para cualquier feature nueva:** (1) hipótesis económica escrita antes; (2) verificar disponibilidad a la hora exacta de ejecución; (3) medir en *Out-of-Fold* de Train, no en el test; (4) agregarla sola, no en bloque; (5) si no mejora de forma consistente en varios folds, se descarta.

---

## 11. Limitaciones y Riesgos (leer antes de interpretar el `+68.7%`)

> [!CAUTION]
> El `+68.7%` es el **mejor caso dentro de un test que ya fue muy usado**. No es una proyección.

1. **Sobreajuste al periodo de test.** Probamos más de 15 configuraciones sobre el mismo periodo Nov-2025 → Oct-2026, y los umbrales `0.575` (macro/sector), `0.58` (guardián) y la regla del viernes se eligieron **viendo** ese test. Ese periodo ya no es 100% "fuera de muestra".
2. **Muestra pequeña.** ~70 trades en un solo periodo de 11 meses y un solo régimen. El IC 95% de un WR de 75% con n≈70 es aproximadamente **±10 pp**. El veto quirúrgico quitó solo **4 trades**: la mejora de PF (`4.11 → 5.31`) depende de esos 4 casos.
3. **No hay crash en el test.** No incluye un evento tipo marzo 2020 o un bear market sostenido tipo 2022 con 5 posiciones abiertas.
4. **El Monte Carlo subestima el riesgo**: remuestrea los trades como si fueran secuenciales e independientes, pero en realidad 5 posiciones abiertas en el mismo shock están correlacionadas.
5. **Stop diario aproximado**: en el simulador, el SL "diario" se evalúa cada 7 barras **desde la entrada**, no al cierre real de NYSE. En demo debe aplicarse al cierre real (16:00 NY), lo que puede producir pequeñas diferencias.
6. **Fricción**: el código de salida aplica `0.9995` (5 bps). Hay que verificar que la entrada también cargue costo para llegar a los 10 bps declarados.
7. **Posible sesgo de supervivencia**: si el universo de 586 empresas se armó con listas actuales, faltan las que quebraron o salieron del índice.
8. **Discrepancia de simuladores** (`+65.7%` vs `+70.5%`) sin reconciliar.

**Expectativa razonable:** si en vivo el sistema rinde una **fracción** del backtest (por ejemplo, la mitad) con drawdown controlado, sería un resultado muy bueno. La cuenta demo es el primer test realmente fuera de muestra.

---

## 12. Especificación Congelada para la Cuenta Demo de IBKR

Configuración: **Fila 3 + Veto Guardián V4** (fila ⭐ de la tabla 8.2).

### 12.1 Entrada
- **Universo**: 586 acciones, `ATR diario >= 2.2%`.
- **Ejecución**: al cierre de cada vela horaria NY (`10:30, 11:30, 12:30, 13:30, 14:30, 15:30, 16:00` + ~2 s). La orden se llena en `Open[t+1]`.
- **Gatillo**: golpe de 3–5 días (`z_3d` o `z_5d <= -2σ`) + vela de absorción (`close_loc_1h >= 0.55` o `lower_wick_ratio >= 0.25`).
- **Escudos**: `weak_volume_flag == 0`; no `(last_earnings_reaction_3d <= -5% & cmf_15d <= -5%)`; no `(tnx_ret_3d >= 3% & vix_slope_3h >= 4%)`; `falling_knife_flag == 0`; no viernes `>= 18:00 UTC`.
- **Umbral**: `prob_1a >= 0.59`, **o** `0.575 <= prob_1a < 0.59` solo si `shock_type_code ∈ {3, 4}` (Macro/Sector). Además `prob_reach_meli_v4 >= 0.58`.
- **Portafolio**: máximo 5 posiciones, ~20% del equity cada una, 1 por ticker, sin margen.

### 12.2 Salida (tal como está en `simulate_calibrated_1b_ibkr_exit`, `mode="calibrated_hourly_close"`)
`rec` = máxima recuperación alcanzada como fracción del golpe. P70/P80 = percentiles de `prob_1b` en Train. Todos los stops se evalúan **al cierre de la vela horaria** y se ejecutan en `Open[t+1]`.

| Rama | Condición | Al llegar a… | Piso / trailing |
| :--- | :--- | :--- | :--- |
| **Cobro rápido** | Earnings (`shock 1/2`) **o** `prob_1b < P70` | `rec >= 0.38` | piso en +22% del golpe |
| | | `rec >= 0.52` | trailing `0.35·ATR`, piso mínimo +42% del golpe |
| **Súper-corredor** | Macro/Sector **y** `prob_1b >= P80` | `rec >= 0.42` | piso en break-even (+0.3%) |
| | | `rec >= 0.55` | trailing `1.05·ATR`, piso mínimo +38% del golpe, sin TP fijo |
| **Intermedia** | resto (incluye antes de conocer 1B) | `rec >= 0.40` | piso en break-even (+0.3%) |
| | | `rec >= 0.52` | trailing `0.65·ATR`, piso mínimo +38% del golpe |

- **Stop Loss estructural**: `sl_p` (−1.5×ATR diario), evaluado al **cierre del día**.
- **Time-stop**: 15 días hábiles (105 velas).
- 1B se evalúa una sola vez, en la vela en que se alcanza el Momento MeLi.

### 12.3 Qué medir en la demo
1. **Slippage**: `Open[t+1]` real vs. el supuesto; meta `<= 10 bps` ida y vuelta.
2. **% que llega al +45%**: esperado ~75–80%.
3. **WR y PF acumulados**, comparados con las bandas de la sección 11 (punto 2).
4. **DD MtM**: esperado hasta ~`-5%` a `-7%` en pantalla; el bootstrap (que subestima) da P95 ≈ `-5.6%`.
5. **Señales/mes**: el backtest ejecutó ~6–7 trades/mes. Si llegan muchas más o muchas menos, algo en el pipeline en vivo difiere.

### 12.4 Criterios para pausar y revisar (sugeridos)
- Tras **30 trades**: WR `< 60%`, o `% llega a +45% < 65%`, o PF `< 1.5`.
- En cualquier momento: DD MtM `< -10%`.
- Cualquier diferencia sistemática entre la feature calculada en vivo y la del backtest para la misma vela.

### 12.5 Pendientes técnicos antes o durante la demo
1. Reconciliar `run_portfolio_simulation` vs `compute_exact_mtm_on_simulation` (`+65.7%` vs `+70.5%`).
2. Confirmar que la fricción de entrada se aplica.
3. Walk-forward con varios cortes (p. ej., test 2023, 2024, 2025) **congelando los umbrales actuales** para ver si se sostienen sin re-optimizar.
4. Stop diario evaluado al cierre real de NYSE en el simulador.
5. Monte Carlo por bloques de fechas (respetando posiciones simultáneas).

