---
trigger: always_on
description: Reglas obligatorias de ingeniería de features y evaluación de modelos
---

# Reglas Mandatorias de Ingeniería y Modelado del Proyecto

## 1. OBLIGATORIEDAD DE VARIABLES Y FEATURES ESTRUCTURALES DEL PROYECTO
Al probar, calibrar o entrenar cualquier modelo en este proyecto, es **ESTRICTAMENTE MANDATORIO** emplear el conjunto completo de features estructurales y causales desarrolladas en la base de código. 

Queda expresamente prohibido recurrir a aproximaciones rudimentarias o atajos simplistas (como mínimos móviles básicos).

### Features Mandatorias de Soporte y Resistencia (`src/utils/stock_sr.py`):
- `sr_dist_support_pct` / `sr_dist_support_atr`: Distancia al soporte en % y unidades de ATR.
- `sr_dist_resistance_pct` / `sr_dist_resistance_atr`: Distancia a resistencia en % y ATR.
- `sr_support_touches`: Número de testeos confirmados del nivel de soporte por valles causales.
- `sr_channel_position`: Posición del precio en el canal Soporte-Resistencia.
- `sr_breakdown_sup`: Señal de ruptura causal confirmada de valle de soporte.
- `sr_bars_since_sup`: Antigüedad / frescura del soporte.

### Features Mandatorias de Microestructura y Ruptura:
- `penetration_pct`: Profundidad real de perforación bajo el nivel estructural.
- `lower_wick_ratio` y `upper_wick_ratio`: Control de absorción compradora intradiaria.
- `rvol_break_power`: `rvol_1h * (1.0 - close_loc)`.
- `marubozu_bear_strength`: `body_ratio * (1.0 - upper_wick)`.
- `upside_tail_risk_20d`: Filtro agnóstico anti-squeeze.

## 2. ARQUITECTURA DE DOS MODELOS (INGRESO Y SALIDA) POR ARQUETIPO
- Cada arquetipo debe tener su modelo de percepción/ingreso (Modelo A) entrenado con features especializadas para su dinámica.
- El modelo de salida (Modelo B) gobierna la gestión de riesgo barra a barra con lógica de trailing stop, absorción y tiempo.
