"""
Extracción y Etiquetado Causal de Motor 2 (Shorts & Cobertura Activa)
sobre el Dataset Maestro Extendido de Interactive Brokers (2020 - 2026).
Implementa TICK-M2-02 y TICK-M2-03.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HOURLY_PARQUET = PROJECT_ROOT / "data_ibkr" / "sp500_ibkr_extended_7years_1hour.parquet"
EARNINGS_PARQUET = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"
OUT_DATASET = PROJECT_ROOT / "data" / "processed" / "motor_2_dataset_7years_ibkr.parquet"

def extract_motor_2_dataset():
    print("=" * 85)
    print("EXTRACCIÓN DE EVENTOS Y FEATURES MOTOR 2 (IBKR 2020 - 2026) [TICK-M2-02 & TICK-M2-03]")
    print("=" * 85)
    t0 = time.time()
    
    if not HOURLY_PARQUET.exists():
        raise FileNotFoundError(f"No se encontró {HOURLY_PARQUET}")
        
    df_raw = pd.read_parquet(HOURLY_PARQUET)
    earn_df = pd.read_parquet(EARNINGS_PARQUET)
    
    # 1. SPY y Régimen Macro
    df_spy = df_raw[df_raw["ticker"] == "SPY"].sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    spy_c = df_spy["close"]
    spy_sma200_d = spy_c.rolling(1400, min_periods=400).mean() # SMA 200 diaria (200 * 7 barras)
    spy_ret_3d = spy_c.pct_change(21).fillna(0.0)
    spy_ret_5d = spy_c.pct_change(35).fillna(0.0)
    
    # 2. Mapeo Causal de Earnings PIT
    earn_df["earnings_date"] = pd.to_datetime(earn_df["earnings_date"], utc=True)
    earn_map = {}
    for tk, gdf in earn_df.groupby("ticker"):
        earn_map[tk] = gdf.set_index("earnings_date")["eps_surprise_pct"].to_dict()
        
    df_stocks = df_raw[~df_raw["ticker"].isin(["SPY", "QQQ"])].copy()
    print(f"Dataset horario cargado: {len(df_stocks):,} barras en {df_stocks['ticker'].nunique()} tickers.")
    print(f"Período: {df_stocks['timestamp'].min().date()} a {df_stocks['timestamp'].max().date()}")
    
    events = []
    event_id = 0
    
    for ticker, tdf in df_stocks.groupby("ticker"):
        tdf = tdf.sort_values("timestamp").reset_index(drop=True)
        if len(tdf) < 200:
            continue
            
        c = tdf["close"].to_numpy()
        o = tdf["open"].to_numpy()
        h = tdf["high"].to_numpy()
        l = tdf["low"].to_numpy()
        v = tdf["volume"].to_numpy()
        ts = tdf["timestamp"].to_numpy()
        sec = tdf["sector"].iloc[0] if "sector" in tdf.columns else "General"
        tk_earn = earn_map.get(ticker, {})
        
        # Medias Móviles y ATR diario (98 barras = 14 días a 7 barras/día)
        sma_20 = pd.Series(c).rolling(20, min_periods=10).mean().to_numpy()
        sma_40 = pd.Series(c).rolling(40, min_periods=20).mean().to_numpy()
        
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr_d = pd.Series(tr).rolling(98, min_periods=20).mean().to_numpy() / (c + 1e-6)
        
        # Retornos y Z-score
        ret_3d = np.zeros(len(tdf))
        ret_5d = np.zeros(len(tdf))
        ret_3d[21:] = (c[21:] - c[:-21]) / (c[:-21] + 1e-6)
        ret_5d[35:] = (c[35:] - c[:-35]) / (c[:-35] + 1e-6)
        roll_std_3d = pd.Series(ret_3d).rolling(250, min_periods=50).std().fillna(0.04).to_numpy()
        z_3d = ret_3d / (roll_std_3d + 1e-6)
        
        # Microestructura de velas
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
        
        for i in range(120, len(tdf) - 35):
            if atr_d[i] < 0.022: # Filtro de elasticidad mínima
                continue
                
            cur_ts = pd.Timestamp(ts[i])
            
            # FILTRO MACRO INSTITUCIONAL: SPY bajo SMA 200 diaria o SPY en shock correctivo (-2.5% en 5d)
            is_macro_bear = False
            if cur_ts in spy_c:
                spy_val = spy_c[cur_ts]
                spy_200 = spy_sma200_d.get(cur_ts, np.nan)
                spy_r5 = spy_ret_5d.get(cur_ts, 0.0)
                if (not np.isnan(spy_200) and spy_val < spy_200) or (spy_r5 <= -0.025):
                    is_macro_bear = True
                    
            if not is_macro_bear:
                continue
                
            # ARQUETIPO 1: PEAD NEGATIVO (+24h causal)
            is_pead_short = False
            last_surprise = 0.0
            for e_date, surprise in tk_earn.items():
                diff_d = (cur_ts - e_date).total_seconds() / 86400.0
                if 1.0 <= diff_d <= 4.0:
                    if surprise <= -0.04 or (ret_3d[i] <= -0.06 and rvol_1h[i] >= 1.5):
                        if upper_wick[i] >= 0.20 or c[i] < sma_20[i]:
                            is_pead_short = True
                            last_surprise = surprise
                            break
                            
            # ARQUETIPO 2: BREAKDOWN ESTRUCTURAL
            min_prev_40 = np.min(l[max(0, i-40): i])
            is_breakdown = False
            if not is_pead_short:
                if c[i] < min_prev_40 and rvol_1h[i] >= 1.6 and close_loc[i] <= 0.30:
                    is_breakdown = True
                    
            if not (is_pead_short or is_breakdown):
                continue
                
            archetype_code = 1 if is_pead_short else 2
            entry_p = o[i + 1]
            if entry_p <= 0:
                continue
                
            # REGLAS DE TARGET ASIMÉTRICO 2:1 (First-Passage)
            # Stop Loss en el High de la vela de quiebre o máx +0.8x ATR
            stop_loss_p = max(h[i], entry_p * (1.0 + 0.8 * atr_d[i]))
            target_drop_p = entry_p * (1.0 - 1.5 * atr_d[i])
            
            # Evaluación First-Passage a 28 barras (4 días hábiles)
            fut_h = h[i + 1 : i + 1 + 28]
            fut_l = l[i + 1 : i + 1 + 28]
            
            target_hit = 0
            for step in range(len(fut_h)):
                # ¿Toca primero la meta de caída (-1.5x ATR)?
                if fut_l[step] <= target_drop_p:
                    target_hit = 1
                    break
                # ¿O toca primero el Stop Loss arriba (+0.8x ATR)?
                if fut_h[step] >= stop_loss_p:
                    target_hit = 0
                    break
                    
            # Debilidad relativa vs SPY
            spy_r3 = float(spy_ret_3d.get(cur_ts, 0.0))
            rel_strength_3d = ret_3d[i] - spy_r3
            
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
                # Features Predictivas
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
                "eps_surprise_pct": last_surprise,
                "rel_strength_3d": rel_strength_3d,
                "spy_ret_3d": spy_r3
            })
            
    df_events = pd.DataFrame(events)
    print(f"\n[OK] Extracción completada en {time.time() - t0:.1f}s.")
    print(f"Total eventos bajistas calificados: {len(df_events):,}")
    print(f"Tasa de éxito base First-Passage (2:1): {df_events['target_2a'].mean()*100:.1f}%")
    print(f"\nDistribución por Arquetipo:\n{df_events['archetype_name'].value_counts()}")
    print(f"\nDistribución por Año:\n{df_events.groupby('year')['target_2a'].agg(['count', 'mean'])}")
    
    OUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df_events.to_parquet(OUT_DATASET, index=False)
    print(f"\n[OK] Guardado en: {OUT_DATASET}")
    return df_events

if __name__ == "__main__":
    extract_motor_2_dataset()
