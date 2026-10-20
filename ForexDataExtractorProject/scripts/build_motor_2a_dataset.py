"""
Pipeline de Extracción y Etiquetado del Modelo 2A (Cuándo entrar en Short):
- Universo: 586 acciones líquidas de alto beta (ATR >= 2.2%).
- Eventos: 
  1. PEAD Negativo (reporte trimestral con sorpresa < -0.04 o destrucción de guía + gap bajista).
  2. Ruptura de Soporte Estructural con RVOL alto.
- Target Puro (First-Passage Downward):
  * Target = 1: La acción cae al menos un -5.0% (o -1.2x ATR) en los siguientes 5 días hábiles (35 barras)
    ANTES de que el precio suba por encima del Stop Loss (+1.2x ATR).
  * Target = 0: Si toca primero el Stop Loss arriba (trampa de oso / rebote fallido).
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOURLY_PARQUET = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
EARNINGS_PARQUET = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"
OUT_DATASET = PROJECT_ROOT / "data" / "processed" / "motor_2a_dataset_shorts.parquet"

def build_motor_2a_dataset():
    print("=" * 80)
    print("EXTRAYENDO EVENTOS Y FEATURES PARA EL MODELO 2A (SHORTS)")
    print("=" * 80)
    
    t0 = time.time()
    df_raw = pd.read_parquet(HOURLY_PARQUET)
    earn_df = pd.read_parquet(EARNINGS_PARQUET)
    
    # Excluir índices de las señales individuales
    df_stocks = df_raw[~df_raw["ticker"].isin(["SPY", "QQQ"])].copy()
    df_spy = df_raw[df_raw["ticker"] == "SPY"].sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    spy_c = df_spy["close"]
    spy_sma_100 = spy_c.rolling(100, min_periods=40).mean()
    spy_ret_3d = spy_c.pct_change(21).fillna(0.0)
    
    # Mapeo de earnings
    earn_df["earnings_date"] = pd.to_datetime(earn_df["earnings_date"], utc=True)
    earn_map = {}
    for tk, gdf in earn_df.groupby("ticker"):
        earn_map[tk] = gdf.set_index("earnings_date")["eps_surprise_pct"].to_dict()
        
    print(f"Barras horarias: {len(df_stocks):,} en {df_stocks['ticker'].nunique()} activos.")
    print(f"Reportes trimestrales: {len(earn_df):,}")
    
    events = []
    event_id = 0
    
    for ticker, tdf in df_stocks.groupby("ticker"):
        tdf = tdf.sort_values("timestamp").reset_index(drop=True)
        if len(tdf) < 150:
            continue
            
        c = tdf["close"].to_numpy()
        o = tdf["open"].to_numpy()
        h = tdf["high"].to_numpy()
        l = tdf["low"].to_numpy()
        v = tdf["volume"].to_numpy()
        ts = tdf["timestamp"].to_numpy()
        sec = tdf["sector"].iloc[0] if "sector" in tdf.columns else "Technology"
        tk_earnings = earn_map.get(ticker, {})
        
        # 1. Medias Móviles
        sma_20 = pd.Series(c).rolling(20, min_periods=10).mean().to_numpy()
        sma_40 = pd.Series(c).rolling(40, min_periods=20).mean().to_numpy()
        sma_100 = pd.Series(c).rolling(100, min_periods=40).mean().to_numpy()
        
        # 2. ATR Diario (98 barras horarias)
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr_d = pd.Series(tr).rolling(98, min_periods=20).mean().to_numpy() / (c + 1e-6)
        
        # 3. Retornos y Z-score
        ret_3d = np.zeros(len(tdf))
        ret_5d = np.zeros(len(tdf))
        ret_3d[21:] = (c[21:] - c[:-21]) / (c[:-21] + 1e-6)
        ret_5d[35:] = (c[35:] - c[:-35]) / (c[:-35] + 1e-6)
        
        roll_std_3d = pd.Series(ret_3d).rolling(250, min_periods=50).std().fillna(0.04).to_numpy()
        z_3d = ret_3d / (roll_std_3d + 1e-6)
        
        # 4. Microestructura
        rng = np.maximum(h - l, 1e-6)
        close_loc = (c - l) / rng
        upper_wick = (h - np.maximum(o, c)) / rng
        lower_wick = (np.minimum(o, c) - l) / rng
        body_ratio = np.abs(c - o) / rng
        rvol_1h = v / (pd.Series(v).rolling(35, min_periods=10).mean().to_numpy() + 1e-6)
        
        # RSI 14
        delta = pd.Series(c).diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14, min_periods=14).mean() / (loss.rolling(14, min_periods=14).mean() + 1e-9)
        rsi_14 = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0).to_numpy()
        
        for i in range(100, len(tdf) - 45):
            if atr_d[i] < 0.020:
                continue
                
            cur_ts = pd.Timestamp(ts[i])
            
            # Chequear si hubo earnings reciente (1 a 4 días antes con buffer de +24h causal)
            is_pead_short = False
            last_surprise = 0.0
            for e_date, surprise in tk_earnings.items():
                diff_d = (cur_ts - e_date).total_seconds() / 86400.0
                if 1.0 <= diff_d <= 4.0: # Seguro, 24h después del balance
                    if surprise <= -0.03 or (ret_3d[i] <= -0.05 and rvol_1h[i] >= 1.4):
                        if upper_wick[i] >= 0.20 or c[i] < sma_20[i]:
                            is_pead_short = True
                            last_surprise = surprise
                            break
                            
            # Chequear ruptura de soporte estructural
            min_prev_40 = np.min(l[max(0, i-40): i])
            is_breakdown = False
            if not is_pead_short:
                if c[i] < min_prev_40 and rvol_1h[i] >= 1.7 and close_loc[i] <= 0.35:
                    is_breakdown = True
                    
            if not (is_pead_short or is_breakdown):
                continue
                
            archetype_code = 1 if is_pead_short else 2
            entry_p = o[i + 1]
            if entry_p <= 0:
                continue
                
            # TARGET FIRST-PASSAGE EN CORTO (5 días = 35 barras horarias):
            # Target = 1 si alcanza un target de caída de -1.2x ATR antes de subir al Stop (+1.2x ATR)
            fut_h = h[i + 1 : i + 1 + 35]
            fut_l = l[i + 1 : i + 1 + 35]
            fut_c = c[i + 1 : i + 1 + 35]
            
            target_drop_p = entry_p * (1.0 - 1.2 * atr_d[i])
            stop_loss_p = entry_p * (1.0 + 1.2 * atr_d[i])
            
            target_hit = 0
            for step in range(len(fut_c)):
                # ¿Toca primero el target de caída?
                if fut_l[step] <= target_drop_p:
                    target_hit = 1
                    break
                # ¿O toca primero el Stop Loss arriba?
                if fut_h[step] >= stop_loss_p:
                    target_hit = 0
                    break
                    
            # Contexto de SPY
            spy_under_sma = 1.0 if cur_ts in spy_c and spy_c[cur_ts] < spy_sma_100.get(cur_ts, 0.0) else 0.0
            spy_r3d_val = float(spy_ret_3d.get(cur_ts, 0.0))
            
            event_id += 1
            events.append({
                "event_id": event_id,
                "ticker": ticker,
                "sector": sec,
                "timestamp": cur_ts,
                "entry_ts": pd.Timestamp(ts[i + 1]),
                "year": cur_ts.year,
                "archetype_code": archetype_code,
                "archetype_name": "PEAD_Negative" if archetype_code == 1 else "Breakdown_Short",
                "entry_p": entry_p,
                "target_drop_p": target_drop_p,
                "stop_loss_p": stop_loss_p,
                "target_2a": target_hit,
                # Features Predictivas del Modelo 2A
                "atr_d": atr_d[i],
                "ret_3d": ret_3d[i],
                "ret_5d": ret_5d[i],
                "z_3d": z_3d[i],
                "rvol_1h": rvol_1h[i],
                "close_loc_1h": close_loc[i],
                "upper_wick_ratio": upper_wick[i],
                "lower_wick_ratio": lower_wick[i],
                "body_ratio": body_ratio[i],
                "rsi_14": rsi_14[i],
                "dist_sma_20": (c[i] - sma_20[i]) / (sma_20[i] + 1e-6),
                "dist_sma_40": (c[i] - sma_40[i]) / (sma_40[i] + 1e-6),
                "dist_sma_100": (c[i] - sma_100[i]) / (sma_100[i] + 1e-6),
                "eps_surprise_pct": last_surprise,
                "spy_under_sma100": spy_under_sma,
                "spy_ret_3d": spy_r3d_val
            })
            
    df_events = pd.DataFrame(events)
    print(f"\n[OK] Extracción completada en {time.time() - t0:.1f}s.")
    print(f"Total eventos bajistas extraídos: {len(df_events):,}")
    print(f"Tasa de éxito base (First-Passage Target=1): {df_events['target_2a'].mean()*100:.1f}%")
    print(f"Distribución por arquetipo:\n{df_events['archetype_name'].value_counts()}")
    
    df_events.to_parquet(OUT_DATASET, index=False)
    print(f"\nGuardado exitosamente en: {OUT_DATASET}")
    return df_events

if __name__ == "__main__":
    build_motor_2a_dataset()
