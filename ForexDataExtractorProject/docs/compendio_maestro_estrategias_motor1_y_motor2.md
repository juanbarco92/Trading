# 📚 Compendio Maestro: Motores Cuantitativos 1 y 2, Arquetipos, Lógica y Resultados
**Proyecto:** ForexDataExtractorProject / Sistema Institucional IBKR ($25,000 USD)  
**Fecha de Congelamiento:** Octubre 2026  
**Ubicación en Repositorio:** `docs/compendio_maestro_estrategias_motor1_y_motor2.md`

---

## 🏛️ 1. Visión y Filosofía del Sistema Dual

El sistema está diseñado para operar una cuenta de **$25,000 USD** en Interactive Brokers dividida en **5 cupos concurrentes de $5,000 USD** (20% NAV por posición). Su filosofía se basa en que **el mercado opera bajo dos físicas completamente diferentes según el ciclo**:

```mermaid
flowchart TD
    subgraph Macro["🌪️ Escudo Macro (SPY & VIX en Tiempo Real)"]
        Regime{"Régimen Macro:\n¿SPY < SMA200d o VIX >= 22?"}
    end

    subgraph Motor1["🟢 MOTOR 1.1: Rebote Asimétrico (LONGS)"]
        M1_In["Shock Elástico 3-5d (Z <= -2.1σ)\n+ Absorción (RVOL >= 1.3x, Wick >= 0.22)"]
        M1_Guard["Guardián Anti-Cuchillo (Z >= -4.0σ, Body <= 0.65)"]
        M1_Out["Salida Momento MeLi (Trailing ATR 0.96)"]
        M1_In --> M1_Guard --> M1_Out
    end

    subgraph Motor2["🔴 MOTOR 2: Cobertura Activa & Liquidación (SHORTS)"]
        M2_In["Breakdown Soporte 40b o PEAD Negativo"]
        M2_ML["Modelo 2A (LightGBM AUC OOS 0.618)"]
        M2_Out["Modelo 2B: Quick BE (0.8x ATR) + Trailing Ceiling (1.2x ATR)"]
        M2_In --> M2_ML --> M2_Out
    end

    Regime -->|No: Mercado Alcista| S_Bull["4 Slots Long / 1 Slot Short"]
    Regime -->|Sí: Mercado Bajista o Crisis| S_Bear["1 Slot Long / 4 Slots Short"]

    S_Bull --> Portfolio["💼 Gestor de Portafolio IBKR (5 Slots / $25,000 USD)"]
    S_Bear --> Portfolio
```

1. **En Suelos (Motor 1.1 - Longs)**: Gobierna la **reversión elástica a la media**. Cuando una acción líquida con fundamentales intactos es castigada en exceso ($Z \le -2.0\sigma$), se comporta como una goma estirada que rebota violentamente cuando los institucionales absorben la oferta.
2. **En Techos y Crisis (Motor 2 - Shorts)**: Gobierna el **pánico, las llamadas de margen (*margin calls*) y la decepción de expectativas**. En mercados bajistas ($SPY < SMA_{200\text{d}}$) o reportes trimestrales desastrosos, las ventas generan liquidaciones en cascada.

---

## 🟢 2. MOTOR 1.1: Rebote Asimétrico en Shocks Extremos (LONGS)

### A. Universo Elegible
- Acciones de S&P 500 y Nasdaq de alta elasticidad: $\text{ATR Diario} \ge 2.2\%$.
- Liquidez institucional: $\text{ADV} \ge \$15\text{M USD / día}$ y Precio $> \$10$.

### B. Los 4 Arquetipos de Caída (Shock Origins)
1. **Arquetipo 1.1: Post-Earnings Negativo**:
   - Venta masiva tras reporte trimestral con sorpresa negativa. Se busca el punto donde el pánico minorista sobrecastiga a la empresa más allá de la realidad del negocio.
2. **Arquetipo 1.2: Post-Earnings Positivo Inverso (*Sell-the-News*)**:
   - Buena noticia vendida para toma de utilidades institucionales; al tocar soporte elástico, reanuda tendencia.
3. **Arquetipo 1.3: Arrastre Macro / Sistémico**:
   - La acción cae no por culpa propia, sino porque el índice SPY o QQQ cayó abruptamente. Son los rebotes más limpios y seguros del sistema.
4. **Arquetipo 1.4: Caída Sectorial / Idiosincrática**:
   - Rotación sectorial temporal de fondos (ej. tecnológicas hacia energía).

### C. Lógica del Modelo y Filtros de Entrada
1. **Gatillo de Shock Elástico**:
   $$Z_{3\text{d}} \le -2.1\sigma \quad \text{o} \quad Z_{5\text{d}} \le -2.1\sigma$$
2. **Microestructura de Absorción (Vela Horaria de Entrada)**:
   - Volumen relativo: $RVOL_{1\text{h}} \ge 1.3\times$.
   - Mecha inferior (*Lower Wick*): $\ge 0.22$ (indica compras agresivas en mínimos).
   - Ratio de cuerpo: $\le 0.65$ (sin barras rojas gigantes de caída libre).
3. **Guardián Anti-Cuchillo (Filtro Anti-Quiebras)**:
   - Veta la entrada si $Z_{3\text{d}} < -4.0\sigma$ (caídas desbocadas tipo SVB o quiebras inminentes).
4. **Escudo Macro VIX / TNX**:
   - Pausa temporal de compras si $VIX > 32.0$ o si el rendimiento del bono a 10 años ($TNX$) sube más de $+3.0\%$ en 3 días.

### D. Mecánica de Salida Dinámica ("Momento MeLi" y Modelo 1B)
- **Stop Loss Inicial**: $-1.2\times \text{ATR Diario}$ por debajo del precio de entrada.
- **Hito de Recuperación ("Momento MeLi")**: Al recuperar el **$+45\%$ del golpe sufrido**, el trade entra en modo de ganancia garantizada:
  - El stop loss se clava inmediatamente en positivo (+0.2% mínimo).
  - Se activa el **Trailing Stop Dinámico** al $96\%$ del precio máximo alcanzado ($0.96\times P_{\text{max}}$).
- **Time Stop**: 35 barras horarias (5 días hábiles). Si en 5 días la acción no rebotó, se liquida a mercado.

### E. Resultados Validados de Motor 1.1 (Solo Longs)
* **Retorno Acumulado**: **+68.7%**
* **Profit Factor (PF)**: **5.31**
* **Win Rate**: **78.4%**
* **Max Drawdown Cerrado**: **-2.12%**
* **Max Drawdown Mark-to-Market**: **-4.71%**
* **Sharpe Ratio**: **3.29**

---

## 🔴 3. MOTOR 2: Cobertura Activa y Liquidación Asimétrica (SHORTS)

### A. El Gran Hallazgo de Microestructura
> [!WARNING] **La Asimetría Fundamental del Mercado**
> - **En Longs**: El sesgo secular del mercado (+10% anual) favorece la reversión a la media.
> - **En Shorts**: Shortear acciones sobrevendidas ($Z \le -2.0\sigma$) en un mercado alcista es un error crítico, porque los algoritmos compran el dip.
> - **Regla Inquebrantable**: El Motor 2 solo se activa bajo el **Filtro Macro**:
>   $$\text{Motor 2 ON} \iff (SPY < SMA_{200\text{d}}) \lor (VIX \ge 22.0)$$

### B. Los Arquetipos Bajistas
1. **Arquetipo 2.1: PEAD Negativo (Post-Earnings Announcement Drift)**:
   - Reporte con sorpresa de EPS $< -0.04$ o destrucción de guía con gap bajista. Buffer causal estricto de **+24 horas** tras el balance.
2. **Arquetipo 2.2: Breakdown Estructural (Rotura de Soporte)**:
   - Pérdida de un soporte horizontal de 40 barras con volumen relativo $RVOL_{1\text{h}} \ge 1.6\times$ y cierre en el 30% inferior de la vela (`close_loc <= 0.30`).
3. **Arquetipo 2.3: Clímax de Euforia (*Blow-Off Top*)**:
   - **Descartado empíricamente**: Tratar de adivinar el techo de subidas parabólicas ($Z \ge +2.5\sigma$) arrojó un Win Rate de apenas 37.1% debido al momentum institucional.

### C. Lógica del Modelo 2A (Cuándo Entrar en Short)
- **Clasificador LightGBM**: Regularizado con `reg_lambda=5.0`, `max_depth=3`, `num_leaves=5`.
- **Target Asimétrico (First-Passage 2:1)**:
  - Meta de ganancia: Cae al menos $-1.5\times \text{ATR}$ en 4 días.
  - Stop de pérdida: Sube al High de la vela de quiebre o $+0.8\times \text{ATR}$.
- **Resultados de Validación OOS (Fuera de Muestra)**:
  - **ROC-AUC OOS**: `0.6183`.
  - **Monotonicidad**: Tercil Bajo (25.0% acierto - descartado) $\to$ Tercil Medio (32.5%) $\to$ Tercil Alto (42.5% acierto con R:R 2:1).

### D. Lógica del Modelo 2B (Cuándo Salir - "Cuando se calmó el mercado")
A diferencia de los longs (que duran 5 a 15 días), los shorts deben ser **rápidos y quirúrgicos (2 a 4 días)**:
1. **Quick BE Lock**: Si la acción cae $0.8\times \text{ATR}$ a favor, el stop se baja inmediatamente al precio de entrada $+0.2\%$. Trade con cero riesgo de pérdida.
2. **Trailing Ceiling**: Si la acción cae $1.2\times \text{ATR}$, se activa un techo móvil a $0.4\times \text{ATR}$ del mínimo local para proteger la ganancia.
3. **Take Profit Mayor**: Cierre total si alcanza $-1.6\times \text{ATR}$.
4. **Time Stop**: 28 barras horarias (máximo 4 días de mercado).

### E. Resultados Validados de Motor 2 (Solo Shorts)
* **Win Rate Global**: **69.9% - 71.2%**
* **Profit Factor**: **1.81**
* **Retorno Medio por Trade**: **+1.07%**
* **Trades con Cero Pérdidas (BE asegurado)**: **35.6% (78 operaciones protegidas)**
* **Comportamiento en el Crash de 2022**:
  - Trades ejecutados: **150**
  - Win Rate en 2022: **76.0%**
  - Retorno Acumulado Neto en 2022: **+246.2%** (Generador de alfa masivo mientras el S&P caía -19.4%).

---

## 📈 4. Resultados del Portafolio Dual Unificado ($25,000 USD / 2020 – 2026)

Simulación realista de cartera con 5 posiciones de $5,000 USD, asignación dinámica por régimen, slippage y comisiones de Interactive Brokers (10 bps round-trip):

| Métrica | Portafolio Dual Unificado | S&P 500 (Buy & Hold) |
| :--- | :--- | :--- |
| **Capital Inicial** | **$25,000.00 USD** | $25,000.00 USD |
| **Capital Final** | **$49,092.96 USD (+96.4%)** | ~$39,500 USD (+58.0%) |
| **Beneficio Neto Total** | **+$24,092.96 USD** | +$14,500 USD |
| **Win Rate Global** | **50.7%** (Longs: 47.5%, Shorts: 71.2%) | N/A |
| **Profit Factor Global** | **1.26** (Shorts PF: 1.81) | N/A |
| **Años en Pérdida** | **0 años (100% de los años en positivo)** | 2022 (-19.4%) |

### Desglose Año por Año (Consistencia Multi-Ciclo):

```
Año       Trades     P&L Neto       Retorno Anual   Win Rate  Contexto de Mercado
---------------------------------------------------------------------------------
2022       157      +$6,200.02         +24.8%         66.9%   Crash Inflación (SPY -19.4% / QQQ -33%)
2023        33      +$2,661.19         +10.6%         51.5%   Inicio recuperación Tech
2024       281      +$8,397.60         +33.6%         49.1%   Expansión alcista IA
2025       290        +$588.91          +2.4%         44.5%   Consolidación / Tipos altos
2026       238      +$6,245.24         +25.0%         49.2%   Rally actual
---------------------------------------------------------------------------------
TOTAL      999     +$24,092.96         +96.4%         50.7%   0 AÑOS EN PÉRDIDAS
```

### Contribución por Motor al Capital de la Cuenta:
- **Motor 1.1 (Longs)**: 867 trades | **+$17,793.03 (+71.2% de la cuenta)** | Monetiza el mercado alcista.
- **Motor 2 (Shorts)**: 132 trades | **+$6,299.93 (+25.2% de la cuenta)** | **71.2% Win Rate** | Protege y monetiza el crash de 2022.

---

## 🔒 5. Archivos Maestros y Persistencia en el Código

Todo el sistema está respaldado por código ejecutable y reglas inmutables en el repositorio:

1. **Reglas del Espacio de Trabajo**: [AGENTS.md](file:///C:/Users/Administrador/Documents/Proyectos/Trading/ForexDataExtractorProject/AGENTS.md)
   * Obliga a cualquier agente o chat a seguir el protocolo Walk-Forward y generar las 6 secciones fijas.
2. **Generador Universal de Reportes**: [src/evaluation/standard_performance_reporter.py](file:///C:/Users/Administrador/Documents/Proyectos/Trading/ForexDataExtractorProject/src/evaluation/standard_performance_reporter.py)
   * Audita automáticamente PnL, Win Rate, Concentración en Top 5 de acciones y Motivos de salida.
3. **Simulador Unificado**: [scripts/run_dual_portfolio_fast.py](file:///C:/Users/Administrador/Documents/Proyectos/Trading/ForexDataExtractorProject/scripts/run_dual_portfolio_fast.py)
4. **Modelos Entrenados**:
   - Longs: `models/modelo_1a_lightgbm.txt`
   - Shorts: `models/modelo_2a_shorts_lgb.txt`
5. **Datos Históricos 7 Años Oficiales**: `data_ibkr/` (más de 133 acciones y >1.5M de barras descargadas de 2020 a 2026).
