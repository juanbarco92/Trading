"""
stock_sr.py - Soporte y Resistencia Adaptado para Acciones (100% Causal, Zero Lookahead Bias).
Inspirado fielmente en la metodología de src/utils/support_resistance.py del usuario:
- Detección causal de picos en High y valles en Low en tiempo real (al cierre de cada barra t).
- Un pico ocurrido en t - distance se confirma estrictamente en la barra t (usando solo barras <= t).
- Ningún parámetro global (prominencia, ATR medio) se calcula sobre el dataset completo o futuro.
- Cero Lookahead Bias garantizado por construcción causal.
"""

import numpy as np
import pandas as pd


def compute_stock_support_resistance_features(
    df: pd.DataFrame,
    distance: int = 5,
    atr_window: int = 14,
    lookback_window: int = 250,  # Aprox 1 mes de trading en velas 1H
    cluster_tol_atr: float = 0.5,
) -> pd.DataFrame:
    """
    Calcula variables estructurales de Soporte y Resistencia para cada vela de forma 100% causal.
    """
    df_out = df.copy()
    n = len(df_out)
    
    high = df_out["high"].to_numpy(dtype=np.float64)
    low = df_out["low"].to_numpy(dtype=np.float64)
    close = df_out["close"].to_numpy(dtype=np.float64)
    
    # Calcular ATR puramente causal (rolling window pasado)
    if "atr14" in df_out.columns:
        atr = df_out["atr14"].to_numpy(dtype=np.float64)
    else:
        tr1 = high - low
        c_prev = np.roll(close, 1)
        c_prev[0] = close[0]
        tr2 = np.abs(high - c_prev)
        tr3 = np.abs(low - c_prev)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Rolling causal usando solo el pasado
        atr = pd.Series(tr).rolling(atr_window, min_periods=1).mean().to_numpy()

    # Arrays de características pre-alocados
    dist_support_pct = np.full(n, 5.0, dtype=np.float32)
    dist_resistance_pct = np.full(n, 5.0, dtype=np.float32)
    dist_support_atr = np.full(n, 3.0, dtype=np.float32)
    dist_resistance_atr = np.full(n, 3.0, dtype=np.float32)
    support_touches = np.zeros(n, dtype=np.float32)
    resistance_touches = np.zeros(n, dtype=np.float32)
    sr_channel_position = np.full(n, 0.5, dtype=np.float32)
    sr_breakout_res = np.zeros(n, dtype=np.float32)
    sr_breakdown_sup = np.zeros(n, dtype=np.float32)
    sr_bounce_sup = np.zeros(n, dtype=np.float32)
    sr_bounce_res = np.zeros(n, dtype=np.float32)
    sr_bars_since_sup = np.full(n, 999.0, dtype=np.float32)
    sr_bars_since_res = np.full(n, 999.0, dtype=np.float32)

    # Listas dinámicas para acumular picos y valles confirmados causales: (idx_pico, precio)
    confirmed_peaks = []
    confirmed_valleys = []

    for t in range(distance * 2, n):
        cur_close = close[t]
        cur_atr = max(atr[t], 1e-4)
        
        # 1. Comprobar si la barra (t - distance) se confirma como pico en la barra t
        cand_idx = t - distance
        cand_high = high[cand_idx]
        cand_low = low[cand_idx]
        
        # Ventana local pasada para confirmación: [cand_idx - distance, cand_idx + distance] == [t - 2*distance, t]
        window_high = high[t - 2 * distance : t + 1]
        window_low = low[t - 2 * distance : t + 1]
        
        # Es un pico si es el máximo local y tiene prominencia sobre el mínimo local relativo al ATR
        if cand_high == np.max(window_high) and (cand_high - np.min(window_low)) >= 0.25 * cur_atr:
            # Confirmado causalmente en t
            confirmed_peaks.append((cand_idx, cand_high))
            
        # Es un valle si es el mínimo local y tiene prominencia sobre el máximo local relativo al ATR
        if cand_low == np.min(window_low) and (np.max(window_high) - cand_low) >= 0.25 * cur_atr:
            confirmed_valleys.append((cand_idx, cand_low))
            
        # Filtrar picos y valles que estén dentro de la ventana de lookback
        t_min = max(0, t - lookback_window)
        valid_peaks = [p for p in confirmed_peaks if p[0] >= t_min]
        valid_valleys = [v for v in confirmed_valleys if v[0] >= t_min]
        
        # Resistencias: picos con precio por encima del precio actual
        res_candidates = [p for p in valid_peaks if p[1] > cur_close]
        # Soportes: valles con precio por debajo del precio actual
        sup_candidates = [v for v in valid_valleys if v[1] < cur_close]
        
        tol = cluster_tol_atr * cur_atr
        
        if res_candidates:
            # Resistencia más cercana
            nearest_res_idx, nearest_res_price = min(res_candidates, key=lambda x: x[1] - cur_close)
            d_res = nearest_res_price - cur_close
            dist_resistance_pct[t] = float(d_res / cur_close * 100.0)
            dist_resistance_atr[t] = float(d_res / cur_atr)
            sr_bars_since_res[t] = float(t - nearest_res_idx)
            # Conteo de toques de picos en la misma zona de resistencia
            resistance_touches[t] = float(sum(1 for p in valid_peaks if abs(p[1] - nearest_res_price) <= tol))
            if d_res <= 0.3 * cur_atr and close[t] < close[t-1]:
                sr_bounce_res[t] = 1.0
                
        if sup_candidates:
            # Soporte más cercano
            nearest_sup_idx, nearest_sup_price = min(sup_candidates, key=lambda x: cur_close - x[1])
            d_sup = cur_close - nearest_sup_price
            dist_support_pct[t] = float(d_sup / cur_close * 100.0)
            dist_support_atr[t] = float(d_sup / cur_atr)
            sr_bars_since_sup[t] = float(t - nearest_sup_idx)
            # Conteo de toques de valles en la misma zona de soporte
            support_touches[t] = float(sum(1 for v in valid_valleys if abs(v[1] - nearest_sup_price) <= tol))
            if d_sup <= 0.3 * cur_atr and close[t] > close[t-1]:
                sr_bounce_sup[t] = 1.0

        # Posición en canal
        tot_span = dist_support_atr[t] + dist_resistance_atr[t]
        if tot_span > 1e-4:
            sr_channel_position[t] = float(dist_support_atr[t] / tot_span)
            
        # Rupturas (Breakouts)
        if valid_peaks:
            recent_p = [p[1] for p in valid_peaks if p[0] >= t - 15]
            if recent_p:
                highest_recent_p = max(recent_p)
                if close[t] > highest_recent_p and close[t-1] <= highest_recent_p:
                    sr_breakout_res[t] = 1.0
                    
        if valid_valleys:
            recent_v = [v[1] for v in valid_valleys if v[0] >= t - 15]
            if recent_v:
                lowest_recent_v = min(recent_v)
                if close[t] < lowest_recent_v and close[t-1] >= lowest_recent_v:
                    sr_breakdown_sup[t] = 1.0

    # Asignar al DataFrame
    df_out["sr_dist_support_pct"] = dist_support_pct
    df_out["sr_dist_resistance_pct"] = dist_resistance_pct
    df_out["sr_dist_support_atr"] = dist_support_atr
    df_out["sr_dist_resistance_atr"] = dist_resistance_atr
    df_out["sr_support_touches"] = support_touches
    df_out["sr_resistance_touches"] = resistance_touches
    df_out["sr_channel_position"] = sr_channel_position
    df_out["sr_breakout_res"] = sr_breakout_res
    df_out["sr_breakdown_sup"] = sr_breakdown_sup
    df_out["sr_bounce_sup"] = sr_bounce_sup
    df_out["sr_bounce_res"] = sr_bounce_res
    df_out["sr_bars_since_sup"] = sr_bars_since_sup
    df_out["sr_bars_since_res"] = sr_bars_since_res
    
    return df_out
