"""
Simulador y Evaluador del Modelo 2B (Gestión Dinámica de Salida en Shorts):
- Monitorea barra a barra las condiciones de agotamiento del impulso vendedor:
  1. Quick BE: Cuando cae 0.9x ATR, asegurar Breakeven (0.0% riesgo).
  2. Salida por Absorción ("Se calmó el mercado"):
     - Vela martillo alcista con lower_wick >= 0.35 y cierre verde.
     - O RSI horario saliendo de sobreventa (> 42) con rebote.
  3. Trailing Ceiling: Techo dinámico a 0.6x ATR sobre el mínimo local.
  4. Time Stop: Máximo 4 días (28 barras horarias).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOURLY_PARQUET = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "motor_2a_dataset_refined.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_2a_shorts_lgb.txt"

def evaluate_motor_2b():
    print("=" * 80)
    print("EVALUACION DEL MODELO 2B: GESTION DINAMICA DE SALIDA EN SHORTS")
    print("=" * 80)
    
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    df_events = pd.read_parquet(DATASET_PATH)
    
    features = [
        "atr_d", "ret_3d", "ret_5d", "z_3d", "rvol_1h", "close_loc_1h",
        "upper_wick_ratio", "lower_wick_ratio", "body_ratio", "rsi_14",
        "dist_sma_20", "dist_sma_40", "eps_surprise_pct", "rel_strength_3d",
        "spy_ret_3d"
    ]
    
    # Predecir probabilidad con Modelo 2A
    df_events["prob_2a"] = booster.predict(df_events[features])
    
    # Filtrar solo trades de alta convicción (Probabilidad >= mediana)
    threshold = df_events["prob_2a"].median()
    qualified = df_events[df_events["prob_2a"] >= threshold].copy().reset_index(drop=True)
    print(f"Trades seleccionados por Modelo 2A (Prob >= {threshold:.3f}): {len(qualified)} eventos.")
    
    # Cargar datos horarios completos para simulación barra a barra
    df_raw = pd.read_parquet(HOURLY_PARQUET)
    stocks_dict = {}
    for tk, gdf in df_raw.groupby("ticker"):
        gdf = gdf.sort_values("timestamp").reset_index(drop=True)
        stocks_dict[tk] = gdf
        
    trades_static = []
    trades_model_2b = []
    
    for _, row in qualified.iterrows():
        tk = row["ticker"]
        if tk not in stocks_dict:
            continue
        stk_df = stocks_dict[tk]
        ts_entry = row["entry_ts"]
        
        matches = stk_df[stk_df["timestamp"] == ts_entry]
        if len(matches) == 0:
            continue
        idx_entry = matches.index[0]
        
        entry_p = row["entry_p"]
        atr = row["atr_d"]
        
        # 1. Salida Estática (Stop 0.8x ATR, Target 1.5x ATR, 28 barras)
        stop_p = max(row["stop_loss_p"], entry_p * (1.0 + 0.8 * atr))
        target_p = entry_p * (1.0 - 1.5 * atr)
        
        # Simular Estático
        fut_bars = stk_df.iloc[idx_entry : idx_entry + 28]
        ret_stat = 0.0
        reason_stat = "time"
        for _, bar in fut_bars.iterrows():
            if bar["high"] >= stop_p:
                ret_stat = (entry_p - stop_p) / entry_p
                reason_stat = "stop"
                break
            if bar["low"] <= target_p:
                ret_stat = (entry_p - target_p) / entry_p
                reason_stat = "target"
                break
        else:
            ret_stat = (entry_p - fut_bars.iloc[-1]["close"]) / entry_p
            
        trades_static.append({
            "ticker": tk, "year": row["year"], "ret": ret_stat, "reason": reason_stat
        })
        
        # 2. Salida Dinámica con MODELO 2B
        ret_2b = 0.0
        reason_2b = "time"
        dyn_stop = stop_p
        be_locked = False
        min_seen = entry_p
        
        for step, (_, bar) in enumerate(fut_bars.iterrows()):
            bh = bar["high"]
            bl = bar["low"]
            bc = bar["close"]
            bo = bar["open"]
            rng = max(bh - bl, 1e-6)
            lower_wick = (min(bo, bc) - bl) / rng
            
            min_seen = min(min_seen, bl)
            gain_now = (entry_p - bl) / entry_p
            
            # Condición A: Stop Loss alcanzado
            if bh >= dyn_stop:
                ret_2b = (entry_p - dyn_stop) / entry_p
                reason_2b = "stop_loss" if not be_locked else "be_stop"
                break
                
            # Condición B: Target mayor alcanzado
            if bl <= target_p:
                ret_2b = (entry_p - target_p) / entry_p
                reason_2b = "target_hit"
                break
                
            # Condición C: Quick BE Lock (si ganancia >= 0.8x ATR)
            if gain_now >= (0.8 * atr) and not be_locked:
                dyn_stop = entry_p * 0.999 # Asegura 0.1% de ganancia mínima
                be_locked = True
                
            # Condición D: Trailing Ceiling (si la caída fue profunda, asegurar ganancias)
            if gain_now >= (1.2 * atr):
                trailing_ceiling = min_seen + (0.5 * atr * entry_p)
                dyn_stop = min(dyn_stop, trailing_ceiling)
                
            # Condición E: Absorción de compradores ("El mercado se calmó / Rebotó")
            # Si ya tenemos ganancia y aparece una vela martillo con mecha inferior >= 0.38 y cierre alcista
            if gain_now >= (0.5 * atr) and lower_wick >= 0.38 and bc > bo:
                ret_2b = (entry_p - bc) / entry_p
                reason_2b = "absorption_calm"
                break
        else:
            ret_2b = (entry_p - fut_bars.iloc[-1]["close"]) / entry_p
            reason_2b = "time_max"
            
        trades_model_2b.append({
            "ticker": tk, "year": row["year"], "ret": ret_2b, "reason": reason_2b
        })
        
    df_stat = pd.DataFrame(trades_static)
    df_2b = pd.DataFrame(trades_model_2b)
    
    print("\n" + "=" * 50)
    print("COMPARATIVA DE RESULTADOS: ESTATICO vs MODELO 2B DINAMICO")
    print("=" * 50)
    
    def calc_metrics(df_sub, name):
        wr = (df_sub["ret"] > 0).mean() * 100
        avg_ret = df_sub["ret"].mean() * 100
        wins = df_sub[df_sub["ret"] > 0]["ret"].sum()
        losses = abs(df_sub[df_sub["ret"] <= 0]["ret"].sum())
        pf = wins / (losses + 1e-6)
        print(f"\n--- {name} ---")
        print(f"Total Trades: {len(df_sub)}")
        print(f"Win Rate: {wr:.1f}%")
        print(f"Retorno Promedio por Trade: {avg_ret:.2f}%")
        print(f"Profit Factor: {pf:.2f}")
        return pf
        
    calc_metrics(df_stat, "1. SALIDA ESTATICA (Stop/Target Fijo)")
    calc_metrics(df_2b, "2. MODELO 2B (Salida Dinámica por Absorción y Trailing)")
    
    print("\nDesglose de Salidas en Modelo 2B:")
    summary_2b = df_2b.groupby("reason")["ret"].agg(["count", lambda x: f"{(x>0).mean()*100:.1f}%", lambda x: f"{x.mean()*100:.2f}%"])
    summary_2b.columns = ["Trades", "WinRate", "AvgRet"]
    print(summary_2b)
    
    print("\nDesempeño del Modelo 2B por Año:")
    by_year = df_2b.groupby("year")["ret"].agg(["count", lambda x: f"{(x>0).mean()*100:.1f}%", lambda x: f"{x.mean()*100:.2f}%", lambda x: f"{x.sum()*100:.1f}%"])
    by_year.columns = ["Trades", "WinRate", "AvgRet", "RetAcumulado"]
    print(by_year)

if __name__ == "__main__":
    evaluate_motor_2b()
