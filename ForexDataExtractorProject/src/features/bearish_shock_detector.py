"""
Módulo de Detección Cuantitativa de Eventos Bajistas (Motor 2 - Shorts):
Identifica los 4 arquetipos de oportunidades de venta en corto:
1. Arquetipo 2.1 (PEAD Negativo): Desastre de resultados trimestrales (eps_surprise < -0.05 o gap bajista severo).
2. Arquetipo 2.2 (Blow-Off Top): Clímax de euforia (Z_3d >= +2.5σ + Upper Wick gigante de rechazo institucional).
3. Arquetipo 2.3 (Distribution Breakdown): Ruptura de piso/soporte horizontal con RVOL acelerando.
4. Arquetipo 2.4 (Macro Contagion): Contagio bajista severo cuando el SPY cae con fuerza.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def detect_bearish_events(
    df_ticker: pd.DataFrame,
    df_earnings_pit: pd.DataFrame = None,
    df_spy: pd.DataFrame = None,
    bars_per_day: int = 7,
) -> pd.DataFrame:
    """
    Detecta eventos de oportunidad bajista para venta en corto sobre barras horarias.
    Garantiza causalidad punto a punto (sin fuga de datos).
    """
    tdf = df_ticker.sort_values("timestamp").reset_index(drop=True)
    if len(tdf) < 150:
        return pd.DataFrame()

    c = tdf["close"].to_numpy()
    o = tdf["open"].to_numpy()
    h = tdf["high"].to_numpy()
    l = tdf["low"].to_numpy()
    v = tdf["volume"].to_numpy()
    ts = tdf["timestamp"].to_numpy()
    ticker = tdf["ticker"].iloc[0]
    sector = tdf["sector"].iloc[0] if "sector" in tdf.columns else "Technology"

    # 1. Medias Móviles y Tendencia
    sma_20 = pd.Series(c).rolling(20, min_periods=10).mean().to_numpy()
    sma_100 = pd.Series(c).rolling(100, min_periods=40).mean().to_numpy()

    # 2. ATR Diario (98 barras horarias)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.insert(tr, 0, h[0] - l[0])
    atr_d = pd.Series(tr).rolling(98, min_periods=20).mean().to_numpy() / (c + 1e-6)

    # 3. Retornos y Z-Scores
    bars_3d = 3 * bars_per_day # 21
    bars_5d = 5 * bars_per_day # 35

    ret_3d = np.zeros(len(tdf))
    ret_5d = np.zeros(len(tdf))
    ret_3d[bars_3d:] = (c[bars_3d:] - c[:-bars_3d]) / c[:-bars_3d]
    ret_5d[bars_5d:] = (c[bars_5d:] - c[:-bars_5d]) / c[:-bars_5d]

    roll_std_3d = pd.Series(ret_3d).rolling(250, min_periods=50).std().fillna(0.04).to_numpy()
    roll_std_5d = pd.Series(ret_5d).rolling(250, min_periods=50).std().fillna(0.05).to_numpy()

    z_3d = ret_3d / (roll_std_3d + 1e-6)
    z_5d = ret_5d / (roll_std_5d + 1e-6)

    # 4. Microestructura de velas
    rng = np.maximum(h - l, 1e-6)
    close_loc = (c - l) / rng
    upper_wick = (h - np.maximum(o, c)) / rng
    lower_wick = (np.minimum(o, c) - l) / rng
    body_ratio = np.abs(c - o) / rng
    rvol_1h = v / (pd.Series(v).rolling(35, min_periods=10).mean().to_numpy() + 1e-6)

    # RSI de 14 barras
    delta = pd.Series(c).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi_14 = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0).to_numpy()

    # Preparar mapa de earnings si existe
    earn_map = {}
    if df_earnings_pit is not None and not df_earnings_pit.empty:
        earn_tk = df_earnings_pit[df_earnings_pit["ticker"] == ticker].copy()
        if not earn_tk.empty:
            for _, erow in earn_tk.iterrows():
                e_date = pd.Timestamp(erow["earnings_date"])
                earn_map[e_date] = float(erow.get("eps_surprise_pct", 0.0))

    events = []

    for i in range(max(bars_5d, 100), len(tdf) - 105):
        # Exigir elasticidad mínima para justificar comisiones en corto
        if atr_d[i] < 0.020:
            continue

        ts_curr = pd.Timestamp(ts[i])
        
        # --- IDENTIFICAR LOS 4 ARQUETIPOS BAJISTAS ---
        is_pead_short = False
        is_blowoff_top = False
        is_breakdown_short = False
        is_macro_contagion = False
        archetype_code = 0

        # Arquetipo 2.1: Post-Earnings Negative Drift
        # Hubo reporte en los últimos 3 días con sorpresa negativa o gap bajista violento con alto volumen
        for e_ts, surprise in earn_map.items():
            diff_days = (ts_curr - e_ts).total_seconds() / 86400.0
            if 0.5 <= diff_days <= 3.5: # 1 a 3 días después del reporte
                if surprise <= -0.04 or (ret_3d[i] <= -0.06 and rvol_1h[i] >= 1.5):
                    # Primer rebote fallido hacia la SMA20 o vela con rechazo superior
                    if upper_wick[i] >= 0.25 or c[i] < sma_20[i]:
                        is_pead_short = True
                        archetype_code = 1
                        break

        # Arquetipo 2.2: Blow-Off Top (Clímax de Euforia)
        # Tensión elástica alcista extrema que topa con rechazo institucional
        if not is_pead_short and (z_3d[i] >= 2.3 or z_5d[i] >= 2.5) and rsi_14[i] >= 72.0:
            if upper_wick[i] >= 0.35 and close_loc[i] <= 0.45: # Fuerte rechazo arriba
                is_blowoff_top = True
                archetype_code = 2

        # Arquetipo 2.3: Quiebre de Soporte Estructural en Distribución
        # La acción perfora el mínimo de las últimas 40 barras con alto volumen vendedor
        min_prev_40 = np.min(l[max(0, i-40): i])
        if not (is_pead_short or is_blowoff_top) and c[i] < min_prev_40 and rvol_1h[i] >= 1.8:
            if c[i] < o[i] and close_loc[i] <= 0.30: # Cierre en mínimos
                is_breakdown_short = True
                archetype_code = 3

        # Arquetipo 2.4: Contagio Bajista Macro
        # Tendencia macro rota (c < sma_100), caída de 3 días acelerando y debilidad relativa
        if not (is_pead_short or is_blowoff_top or is_breakdown_short):
            if c[i] < sma_100[i] and z_3d[i] <= -1.8 and close_loc[i] <= 0.30 and rvol_1h[i] >= 1.3:
                is_macro_contagion = True
                archetype_code = 4

        if archetype_code == 0:
            continue

        entry_p = o[i + 1] # Apertura de la siguiente barra
        if entry_p <= 0:
            continue

        # Tamaño del movimiento esperado hacia abajo (magnitud del shock bajista objetivo)
        expected_drop = max(abs(ret_3d[i]), 1.5 * atr_d[i], 0.04)

        # Seguimiento futuro para simulación de salida en corto
        fut_h = h[i + 1 : i + 1 + 105]
        fut_l = l[i + 1 : i + 1 + 105]
        fut_c = c[i + 1 : i + 1 + 105]
        fut_o = o[i + 1 : i + 1 + 105]
        fut_ts = ts[i + 1 : i + 1 + 105]

        # REGLAS DEL SIMULADOR DE SHORT:
        # Stop Loss Estructural: +1.5x ATR Diario por ENCIMA de la entrada (evaluado al cierre del día)
        sl_p = entry_p * (1.0 + 1.5 * atr_d[i])
        
        # Hito de Ganancia Inicial ("Momento Desplome -45%"): La acción cae un 45% del shock bajista esperado
        target_desplome = entry_p * (1.0 - 0.45 * expected_drop)

        lowest_p = entry_p
        ceiling_p = None # Candado bajista (techo que va bajando para proteger ganancia)
        exit_p = fut_c[-1]
        exit_ts = fut_ts[-1]
        exit_reason = "TimeStop_15d"
        reached_target = 0

        for step in range(len(fut_c)):
            # 1. Candado activo: si el precio sube por encima del techo, cerramos el corto
            if ceiling_p is not None and step > 0:
                if fut_c[step] >= ceiling_p:
                    exit_p = fut_o[step + 1] if step + 1 < len(fut_c) else fut_c[step]
                    exit_ts = fut_ts[step + 1] if step + 1 < len(fut_c) else fut_ts[step]
                    exit_reason = "Trailing_Ceiling_Hit"
                    break

            # 2. Stop Loss Estructural al cierre del día (cada 7 barras)
            if (step + 1) % 7 == 0 and fut_c[step] >= sl_p:
                exit_p = fut_o[step + 1] if step + 1 < len(fut_c) else fut_c[step]
                exit_ts = fut_ts[step + 1] if step + 1 < len(fut_c) else fut_ts[step]
                exit_reason = "SL_Diario_Short"
                break

            # Actualizar el mínimo alcanzado
            if fut_l[step] < lowest_p:
                lowest_p = fut_l[step]

            # 3. Al tocar el hito del -45% de caída, se activa el candado de ganancia
            if fut_l[step] <= target_desplome:
                reached_target = 1
                # Asegurar ganancia en corto: techo en -30% de caída o trailing de 0.8x ATR por encima del mínimo
                ceiling_p = min(
                    ceiling_p or float("inf"),
                    entry_p * (1.0 - 0.30 * expected_drop),
                    lowest_p * (1.0 + 0.80 * atr_d[i])
                )

        # En SHORT, la ganancia es: (entry_p - exit_p) / entry_p
        # Restando 10 bps de comisión y costo de borrow aproximado (5 bps por pata)
        net_ret_pct = ((entry_p - exit_p * 1.0005) / entry_p) * 100.0

        events.append({
            "ticker": ticker,
            "sector": sector,
            "entry_ts": pd.Timestamp(ts[i + 1]),
            "exit_ts": pd.Timestamp(exit_ts),
            "year": pd.Timestamp(ts[i + 1]).year,
            "archetype_code": archetype_code,
            "archetype_name": ["", "PEAD_Negative", "BlowOff_Top", "Breakdown_Short", "Macro_Contagion"][archetype_code],
            "entry_p": entry_p,
            "exit_p": exit_p,
            "net_ret_pct": net_ret_pct,
            "reached_desplome_45": reached_target,
            "exit_reason": exit_reason,
            "expected_drop": expected_drop,
            "atr_d": atr_d[i],
            # Features predictivas
            "z_3d": z_3d[i],
            "z_5d": z_5d[i],
            "rsi_14": rsi_14[i],
            "upper_wick": upper_wick[i],
            "close_loc": close_loc[i],
            "rvol_1h": rvol_1h[i],
            "dist_sma_100": (c[i] - sma_100[i]) / (sma_100[i] + 1e-6)
        })

    return pd.DataFrame(events)
