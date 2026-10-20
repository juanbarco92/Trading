# Reglas Mandatorias de Ingeniería y Modelado del Proyecto

## 1. REGLA FUNDAMENTAL DE VARIABLES Y FEATURES PREVIAS
> [!IMPORTANT]
> **Prohibición de Atajos o Simplificaciones Rudimentarias:**
> Al probar, calibrar o entrenar cualquier modelo (sea de ingreso, salida, clasificación o filtrado), es **ESTRICTAMENTE MANDATORIO** utilizar el conjunto robusto de variables estructurales y causales que se han venido hipotetizando, diseñando y creando en el proyecto.
> 
> En particular:
> - **Soporte y Resistencia:** Usar siempre el motor causal especializado de `src/utils/stock_sr.py` (`compute_stock_support_resistance_features`):
>   * `sr_dist_support_pct` / `sr_dist_support_atr`: Distancia real y normalizada al soporte.
>   * `sr_support_touches`: Número de testeos confirmados del nivel de soporte.
>   * `sr_channel_position`: Posición del precio dentro del canal soporte-resistencia.
>   * `sr_breakdown_sup`: Señal de quiebre de valle estructural confirmado.
>   * `sr_bars_since_sup`: Frescura o antigüedad de la zona de soporte.
> - **Microestructura y Volumen:** Features de penetración (`penetration_pct`), mechas (`upper_wick_ratio`, `lower_wick_ratio`), cuerpo marubozu (`marubozu_bear_strength`), aceleración de volumen y poder de quiebre (`rvol_break_power`).
> - **Filtros Agnósticos:** Tail risk anti-squeeze (`upside_tail_risk_20d`) y momentum relativo frente a benchmark.
> 
> **Queda terminantemente prohibido** sustituir estas características por simplificaciones 'perezosas' (como un `rolling min` o `rolling max` elemental de N barras) que ignoren la estructura causal validada del proyecto.

## 2. AISLAMIENTO ESTRICTO DE MOTORES Y ARQUETIPOS
- Cada motor (Motor 1: Longs, Motor 2: Shorts) debe evaluarse en aislamiento 100% puro.
- Dentro de Motor 2, cada Arquetipo (Breakdown Estructural, Rollover Top, PEAD) debe contar con su propia arquitectura de modelado y análisis independiente, sin mezclas opacas.

## 3. ESTÁNDAR INSTITUCIONAL WALK-FORWARD
- Cero lookahead bias: Toda feature, soporte o normalización debe ser estrictamente causal.
- Evaluación en Expanding Windows Walk-Forward out-of-sample (2022 a 2026).
- Gestión de portafolio realista: $25,000 USD, 5 slots de $5,000 USD, 10 bps de comisión y borrow de Interactive Brokers.
