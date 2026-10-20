"""
Auditoría Cuantitativa de Oportunidades No Mapeadas ("Money Left on the Table")
Motor 2 (Shorts & Cobertura Activa)
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_motor_2_walkforward_pipeline import (
    PARQUET_IBKR, EARNINGS_PARQUET, extract_all_motor_2_events,
    run_walk_forward_ml, simulate_dynamic_trading_2b, simulate_portfolio_slots
)

def run_opportunity_audit():
    print("=" * 90)
    print("AUDITORIA CUANTITATIVA: OPORTUNIDADES PERDIDAS Y DINERO SOBRE LA MESA (MOTOR 2)")
    print("=" * 90)
    
    df_raw = pd.read_parquet(PARQUET_IBKR)
    earn_df = pd.read_parquet(EARNINGS_PARQUET)
    
    # 1. Ejecutar pipeline actual para tener los trades que sí ejecutamos
    print("\n1. Extrayendo eventos actuales y simulando Walk-Forward...")
    df_events = extract_all_motor_2_events(df_raw, earn_df)
    oos_df, _ = run_walk_forward_ml(df_events)
    trades_df = simulate_dynamic_trading_2b(oos_df, df_raw)
    
    prob_cutoff = oos_df["prob_2a"].median()
    qualified = trades_df[trades_df["prob_2a"] >= prob_cutoff].copy().reset_index(drop=True)
    exec_df, p_stats = simulate_portfolio_slots(qualified, initial_capital=25000.0, max_slots=5)
    
    print(f"Trades ejecutados por Motor 2 actual: {len(exec_df)}")
    
    # 2. AUDITORÍA DEL MERCADO REAL: Detectar TODOS los shocks bajistas en 1-2 días (14 barras)
    print("\n2. Escaneando la totalidad del mercado IBKR (82 activos, 2020-2026)...")
    tickers = [t for t in df_raw["ticker"].unique() if t not in ["SPY", "QQQ"]]
    
    # Vamos a indexar los trades ejecutados para saber si un shock fue capturado
    # Clave: ticker + fecha/hora aproximada (+/- 2 barras)
    executed_keys = set()
    for _, tr in exec_df.iterrows():
        executed_keys.add((tr["ticker"], tr["entry_ts"]))
        
    shocks = []
    
    for ticker in tickers:
        tdf = df_raw[df_raw["ticker"] == ticker].sort_values("timestamp").reset_index(drop=True)
        if len(tdf) < 200:
            continue
            
        c = tdf["close"].to_numpy()
        o = tdf["open"].to_numpy()
        h = tdf["high"].to_numpy()
        l = tdf["low"].to_numpy()
        v = tdf["volume"].to_numpy()
        ts = tdf["timestamp"].to_numpy()
        
        # ATR diario
        tr_arr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr_arr = np.insert(tr_arr, 0, h[0] - l[0])
        atr_d = pd.Series(tr_arr).rolling(98, min_periods=20).mean().to_numpy()
        atr_pct = atr_d / (c + 1e-6)
        
        # RVOL
        rvol = v / (pd.Series(v).rolling(35, min_periods=10).mean().to_numpy() + 1e-6)
        
        # Escaneo de caídas en 14 barras (2 días)
        # Buscamos eventos independientes de caída >= 6% en 14 barras
        last_shock_bar = -999
        for i in range(50, len(tdf) - 15):
            # Solo evaluar periodo Walk-Forward (2022 en adelante)
            cur_year = pd.Timestamp(ts[i]).year
            if cur_year < 2022:
                continue
                
            # Caída máxima en las próximas 14 barras respecto al close actual
            fut_lows = l[i+1 : i+1+14]
            min_fut = np.min(fut_lows)
            drop_pct = (min_fut - c[i]) / c[i] # e.g. -0.08 (-8%)
            
            if drop_pct <= -0.06: # Caída >= 6% en 2 días
                # Para evitar contar 10 barras consecutivas del mismo crash, agrupamos por clusters (mínimo 10 barras de separación)
                if i - last_shock_bar < 7:
                    continue
                last_shock_bar = i
                
                # CARACTERIZACIÓN DEL ORIGEN DE LA CAÍDA (¿De dónde venía?)
                min_40 = np.min(l[i-40: i])
                min_20 = np.min(l[i-20: i])
                min_10 = np.min(l[i-10: i])
                max_20 = np.max(h[i-20: i])
                max_40 = np.max(h[i-40: i])
                
                broke_min_40 = c[i] < min_40
                broke_min_20 = c[i] < min_20
                broke_min_10 = c[i] < min_10
                
                # Distancia al máximo reciente de 20 barras
                dist_to_max20 = (c[i] - max_20) / max_20
                # Si estaba cerca de máximos (-3% a 0%), es una reversión / techo / blow-off top
                is_rollover_top = dist_to_max20 >= -0.04
                
                # Microestructura de la barra de inicio
                rng = max(h[i] - l[i], 1e-6)
                c_loc = (c[i] - l[i]) / rng
                rvol_i = rvol[i]
                
                # ¿Fue capturado por nuestro modelo?
                # Revisamos si hubo entrada en [i-1, i+2]
                was_captured = False
                for delta in range(-1, 3):
                    idx_check = i + delta
                    if 0 <= idx_check < len(ts):
                        if (ticker, ts[idx_check]) in executed_keys:
                            was_captured = True
                            break
                            
                shocks.append({
                    "ticker": ticker,
                    "timestamp": ts[i],
                    "year": cur_year,
                    "drop_pct": drop_pct,
                    "atr_pct": atr_pct[i],
                    "drop_in_atr": abs(drop_pct) / (atr_pct[i] + 1e-6),
                    "broke_min_40": broke_min_40,
                    "broke_min_20": broke_min_20,
                    "broke_min_10": broke_min_10,
                    "is_rollover_top": is_rollover_top,
                    "rvol_1h": rvol_i,
                    "close_loc": c_loc,
                    "was_captured": was_captured
                })
                
    shocks_df = pd.DataFrame(shocks)
    
    print("\n" + "=" * 90)
    print("2. TOTAL DE SHOCKS BAJISTAS REALES EN EL MERCADO (2022 - 2026):")
    print("=" * 90)
    print(f"Total shocks de >= 6% en 1-2 días: {len(shocks_df)}")
    print(f"  - De los cuales >= 8%:  {(shocks_df['drop_pct'] <= -0.08).sum()} ({(shocks_df['drop_pct'] <= -0.08).mean()*100:.1f}%)")
    print(f"  - De los cuales >= 10%: {(shocks_df['drop_pct'] <= -0.10).sum()} ({(shocks_df['drop_pct'] <= -0.10).mean()*100:.1f}%)")
    print(f"  - De los cuales >= 15%: {(shocks_df['drop_pct'] <= -0.15).sum()} ({(shocks_df['drop_pct'] <= -0.15).mean()*100:.1f}%)")
    
    print(f"\n¿Cuántos de estos shocks capturó Motor 2 con su trigger actual?")
    captured_count = shocks_df['was_captured'].sum()
    print(f"  >> Capturados: {captured_count} de {len(shocks_df)} ({captured_count/len(shocks_df)*100:.1f}%)")
    print(f"  >> NO CAPTURADOS (PERDIDOS): {len(shocks_df) - captured_count} ({ (1 - captured_count/len(shocks_df))*100:.1f}%)")
    
    print("\n" + "=" * 90)
    print("3. ORIGEN ANATÓMICO DE LAS CAÍDAS (¿POR QUÉ NO LAS CAPTURAMOS?):")
    print("=" * 90)
    uncaptured = shocks_df[~shocks_df['was_captured']]
    
    print(f"Sobre los {len(uncaptured)} shocks perdidos:")
    print(f"1. ¿Rompieron mínimo de 40 barras en la barra de inicio?:")
    print(f"   - SÍ rompieron min 40: {uncaptured['broke_min_40'].sum()} ({uncaptured['broke_min_40'].mean()*100:.1f}%)")
    print(f"   - NO rompieron min 40: {(~uncaptured['broke_min_40']).sum()} ({(~uncaptured['broke_min_40']).mean()*100:.1f}%)  <-- ¡AQUÍ ESTÁ EL CUELLO DE BOTELLA!")
    
    print(f"\n2. Para los que NO rompieron min 40, ¿dónde empezaron a caer?:")
    print(f"   - Rompieron soporte de 20 barras: {uncaptured['broke_min_20'].sum()} ({uncaptured['broke_min_20'].mean()*100:.1f}%)")
    print(f"   - Rompieron soporte de 10 barras: {uncaptured['broke_min_10'].sum()} ({uncaptured['broke_min_10'].mean()*100:.1f}%)")
    print(f"   - Empezaron cerca de máximos (Rollover / Techo / Falla de impulso): {uncaptured['is_rollover_top'].sum()} ({uncaptured['is_rollover_top'].mean()*100:.1f}%)")
    
    print(f"\n3. Comportamiento de Volumen en el inicio de la caída:")
    print(f"   - RVOL >= 1.6: {(uncaptured['rvol_1h'] >= 1.6).sum()} ({(uncaptured['rvol_1h'] >= 1.6).mean()*100:.1f}%)")
    print(f"   - RVOL < 1.6 (Empezó sin volumen y el volumen vino después de caer): {(uncaptured['rvol_1h'] < 1.6).sum()} ({(uncaptured['rvol_1h'] < 1.6).mean()*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # 4. DINERO SOBRE LA MESA EN TRADES EJECUTADOS (MFE - MAX FAVORABLE EXCURSION)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("4. AUDITORÍA DE MFE EN TRADES EJECUTADOS (¿CUÁNTO MÁS CAYÓ TRAS SALIR?):")
    print("=" * 90)
    
    # En exec_df, calcular cuánto cayó la acción durante las 14 barras posteriores a la entrada
    mfe_list = []
    target_hits = exec_df[exec_df["exit_reason"] == "target_hit"].copy()
    
    for _, tr in target_hits.iterrows():
        tk = tr["ticker"]
        en_t = tr["entry_ts"]
        en_p = tr["entry_p"]
        
        tdf = df_raw[df_raw["ticker"] == tk].sort_values("timestamp").reset_index(drop=True)
        idx = tdf[tdf["timestamp"] == en_t].index
        if len(idx) > 0:
            i = idx[0]
            fut_lows = tdf.loc[i : min(len(tdf)-1, i + 14), "low"].values
            max_drop = (np.min(fut_lows) - en_p) / en_p # e.g. -0.09 (-9%)
            # Nuestra ganancia capturada
            realized_gain = tr["net_ret"]
            # Exceso dejado sobre la mesa
            excess_drop = max(0.0, abs(max_drop) - realized_gain)
            mfe_list.append({
                "ticker": tk,
                "entry_time": en_t,
                "realized_gain_pct": realized_gain * 100,
                "max_drop_pct": abs(max_drop) * 100,
                "excess_pct": excess_drop * 100,
                "excess_usd": tr["slot_allocated"] * excess_drop
            })
            
    mfe_df = pd.DataFrame(mfe_list)
    print(f"Total trades ganadores por Target Hit auditados: {len(mfe_df)}")
    print(f"Ganancia media realizada por trade: {mfe_df['realized_gain_pct'].mean():.2f}%")
    print(f"Caída máxima potencial media (MFE): {mfe_df['max_drop_pct'].mean():.2f}%")
    print(f"Exceso medio dejado sobre la mesa: {mfe_df['excess_pct'].mean():.2f}% por trade")
    print(f"DINERO TOTAL DEJADO SOBRE LA MESA EN LOS TARGET HITS: ${mfe_df['excess_usd'].sum():+,.0f} USD")
    print(f"  - De los cuales cayeron más de un 6%: {(mfe_df['max_drop_pct'] >= 6.0).sum()} ({(mfe_df['max_drop_pct'] >= 6.0).mean()*100:.1f}%)")
    print(f"  - De los cuales cayeron más de un 10%: {(mfe_df['max_drop_pct'] >= 10.0).sum()} ({(mfe_df['max_drop_pct'] >= 10.0).mean()*100:.1f}%)")
    
    # Top 10 trades con mayor dinero dejado sobre la mesa
    print("\nTOP 10 TRADES CON MÁS DINERO DEJADO SOBRE LA MESA (CRASHES BRUTALES DESPERDICIADOS):")
    top_left = mfe_df.sort_values("excess_usd", ascending=False).head(10)
    for _, r in top_left.iterrows():
        print(f"  {r['ticker']} ({str(r['entry_time'])[:16]}): Ganamos {r['realized_gain_pct']:.1f}% pero cayó {r['max_drop_pct']:.1f}% (+${r['excess_usd']:,.0f} USD dejados)")
    print("=" * 90)

if __name__ == "__main__":
    run_opportunity_audit()
