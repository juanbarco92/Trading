# 📜 Directrices Universales de Desarrollo y Protocolo Cuantitativo
# Proyecto: ForexDataExtractorProject (Sistema Cuantitativo IBKR)

Este documento contiene las reglas de trabajo, diseño metodológico y protocolo de evaluación **OBLIGATORIOS E INVIOLABLES** para cualquier agente de IA o sesión de programación en este repositorio.

---

## 🔒 1. Principio Fundamental: Cero Fuga de Datos (*Walk-Forward Estricto*)

1. **Prohibición de Lookahead e In-Sample Leakage**:
   - Está terminantemente prohibido evaluar un período o año con un modelo que haya visto esos datos en su entrenamiento.
   - **Protocolo de Validación Temporal (Ventana Expansiva)**:
     - **Test 2022 (Crash)**: Modelo entrenado estrictamente con datos de **2020 y 2021**. Jamás debe haber visto 2022.
     - **Test 2023**: Modelo reentrenado con datos de **2020 a 2022**.
     - **Test 2024**: Modelo reentrenado con datos de **2020 a 2023**.
     - **Test 2025 - 2026**: Modelo reentrenado con datos de **2020 a 2024**.
2. **Buffer Causal en Fundamentales y Eventos**:
   - Todo balance trimestral (Earnings) debe respetar un buffer estricto de **+24 horas causal** para que la información esté efectivamente publicada y procesada antes de cualquier toma de decisión.

---

## 📊 2. Protocolo Estándar de Análisis de Resultados (Inmutable)

Cualquier prueba de estrategia, backtest o evaluación de impacto (sea de Motor 1, Motor 2 o Cartera Combinada) **DEBE generar SIEMPRE las 6 Secciones Obligatorias** utilizando el módulo central:
`from src.evaluation.standard_performance_reporter import StandardPerformanceReporter`

### Las 6 Secciones Obligatorias:

1. **Resumen Ejecutivo de Cartera**:
   - Capital Inicial ($25,000 USD base), Capital Final, Beneficio Neto Total ($).
   - Win Rate Global, Profit Factor, Esperanza Matemática ($ y % por trade).
   - Ratio Ganancia/Pérdida ($W/L$), Max Drawdown Cerrado y Max Drawdown Mark-to-Market (MtM vela a vela).
   - Duración media de trade (días y horas de mercado).

2. **Desglose Anual y Consistencia Multi-Ciclo (Año por Año)**:
   - Prohibido presentar solo métricas agregadas.
   - Tabla año por año (2020 a 2026) con: Trades, Win Rate, Profit Factor, PnL Neto ($), Retorno Anual % sobre capital de cuenta, y Max DD del año.

3. **Cobertura del Universo y Concentración de Alfa**:
   - Cantidad de activos únicos operados vs. universo total.
   - **Índice de Concentración de Ganancias**: % del beneficio total aportado por el Top 5 de acciones. Si supera el 65%, debe alertarse como riesgo de sobreconcentración.
   - Lista detallada de los **Top 5 activos más rentables** y los **Top 5 activos con mayores pérdidas**.

4. **Anatomía de Fallas y Razones de Salida**:
   - Desglose exhaustivo de **por qué se ganó o por qué se perdió**:
     - `% por Stop Loss duro` (fallo de hipótesis o volatilidad).
     - `% por Time-out / Max Holding` (inercia sin movimiento).
     - `% por Breakeven Trailing` (ganancia mínima asegurada).
     - `% por Target Hit` (meta completa alcanzada).
     - `% por Absorción táctica` (salida por giro del mercado).
   - Métricas de Win Rate y PnL medio por cada motivo de salida.

5. **Atribución y Aislamiento por Motor / Arquetipo**:
   - Se debe aislar el rendimiento específico de cada motor (Motor 1 Longs vs. Motor 2 Shorts) antes de presentar cifras consolidadas.
   - Desglose por arquetipo específico (PEAD vs. Breakdown vs. Absorción de Suelo).

6. **Diagnóstico Metodológico y Control de Calidad**:
   - Verificación de Drawdown (< 18%).
   - Verificación de Profit Factor (>= 1.50).
   - Verificación de consistencia (0 años negativos).

---

## 🚫 3. Prohibición de Salir a TWS / Demo sin Aprobación

- Ningún agente debe proponer, inducir o ejecutar conexiones operativas a TWS (ni en demo ni en real) hasta que:
  1. La totalidad del dataset del universo extendido esté descargado y validado en disco.
  2. El modelo pase el protocolo Walk-Forward ciego con las 6 secciones aprobadas.
  3. El **USUARIO dé explícitamente el GO por escrito**.

---

## 📁 4. Ubicación de Datos y Módulos Críticos

- **Datos IBKR Multi-año (2020 - 2026)**: `data_ibkr/{ticker}/1hour/{YYYYMM}.csv`
- **Módulo de Reporte Estándar**: `src/evaluation/standard_performance_reporter.py`
- **Modelos Entrenados**: `models/`
- **Features y Detección**: `src/features/`
- **Documentación y Diseños**: `docs/`
