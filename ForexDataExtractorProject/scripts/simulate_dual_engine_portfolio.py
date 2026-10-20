"""
Simulación Unificada de Portafolio Dual (Motor 1.1 Longs + Motor 2 Shorts):
- Capital Inicial: $25,000 USD
- Cupos Totales: 5 slots concurrentes ($5,000 por posición fija o 20% NAV)
- Matriz de Asignación Dinámica por Régimen Macro:
  * Régimen Alcista (SPY >= SMA200d y VIX < 22):
    - Motor 1 (Longs): Hasta 4 slots
    - Motor 2 (Shorts): Hasta 1 slot táctico
  * Régimen Bajista / Crisis (SPY < SMA200d o VIX >= 22):
    - Motor 1 (Longs): Máximo 1 slot (solo rebotes hiper-calificados)
    - Motor 2 (Shorts): Hasta 4 slots (cobertura activa agresiva)
- Curva de Capital Unificada, Drawdown Cerrado, Drawdown MtM y Ratio Sharpe.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOURLY_PARQUET = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
EARNINGS_PARQUET = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"
MODEL_2A_PATH = PROJECT_ROOT / "models" / "modelo_2a_shorts_lgb.txt"
MODEL_1A_PATH = PROJECT_ROOT / "models" / "modelo_1a_lightgbm.txt"

def run_dual_engine_backtest():
    print("=" * 80)
    print("INICIANDO BACKTEST DE PORTAFOLIO DUAL (MOTOR 1.1 LONGS + MOTOR 2 SHORTS)")
    print("Capital: $25,000 USD | Período: 2020 - 2026")
    print("=" * 80)
    
    import lightgbm as lgb
    booster_2a = lgb.Booster(model_file=str(MODEL_2A_PATH))
    
    df_raw = pd.read_parquet(HOURLY_PARQUET)
    earn_df = pd.read_parquet(EARNINGS_PARQUET)
    
    # 1. SPY y Régimen Macro
    df_spy = df_raw[df_raw["ticker"] == "SPY"].sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    spy_c = df_spy["close"]
    spy_sma200_d = spy_c.rolling(1400, min_periods=400).mean()
    spy_ret_3d = spy_c.pct_change(21).fillna(0.0)
    spy_ret_5d = spy_c.pct_change(35).fillna(0.0)
    
    # 2. Diccionario de Earnings
    earn_df["earnings_date"] = pd.to_datetime(earn_df["earnings_date"], utc=True)
    earn_map = {}
    for tk, gdf in earn_df.groupby("ticker"):
        earn_map[tk] = gdf.set_index("earnings_date")["eps_surprise_pct"].to_dict()
        
    df_stocks = df_raw[~df_raw["ticker"].isin(["SPY", "QQQ"])].copy()
    
    # 3. Recopilar todas las barras ordenadas cronológicamente para simulación unificada
    print("Indexando barras horarias cronológicas...")
    all_bars = []
    for tk, gdf in df_stocks.groupby("ticker"):
        gdf = gdf.sort_values("timestamp").reset_index(drop=True)
        if len(gdf) < 150:
            continue
        c = gdf["close"].to_numpy()
        o = gdf["open"].to_numpy()
        h = gdf["high"].to_numpy()
        l = gdf["low"].to_numpy()
        v = gdf["volume"].to_numpy()
        ts = gdf["timestamp"].to_numpy()
        
        # Indicadores
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr_d = pd.Series(tr).rolling(98, min_periods=20).mean().to_numpy() / (c + 1e-6)
        rvol = v / (pd.Series(v).rolling(35, min_periods=10).mean().to_numpy() + 1e-6)
        
        ret_3d = np.zeros(len(gdf))
        ret_5d = np.zeros(len(gdf))
        ret_3d[21:] = (c[21:] - c[:-21]) / (c[:-21] + 1e-6)
        ret_5d[35:] = (c[35:] - c[:-35]) / (c[:-35] + 1e-6)
        roll_std_3d = pd.Series(ret_3d).rolling(250, min_periods=50).std().fillna(0.04).to_numpy()
        z_3d = ret_3d / (roll_std_3d + 1e-6)
        
        rng = np.maximum(h - l, 1e-6)
        close_loc = (c - l) / rng
        upper_wick = (h - np.maximum(o, c)) / rng
        lower_wick = (np.minimum(o, c) - l) / rng
        body_ratio = np.abs(c - o) / rng
        
        sma_20 = pd.Series(c).rolling(20, min_periods=10).mean().to_numpy()
        sma_40 = pd.Series(c).rolling(40, min_periods=20).mean().to_numpy()
        
        delta = pd.Series(c).diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14, min_periods=14).mean() / (loss.rolling(14, min_periods=14).mean() + 1e-9)
        rsi_14 = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0).to_numpy()
        
        tk_earn = earn_map.get(tk, {})
        
        for i in range(120, len(gdf) - 1):
            cur_ts = pd.Timestamp(ts[i])
            all_bars.append({
                "ticker": tk,
                "timestamp": cur_ts,
                "idx": i,
                "open": o[i],
                "high": h[i],
                "low": l[i],
                "close": c[i],
                "next_open": o[i+1],
                "atr_d": atr_d[i],
                "rvol": rvol[i],
                "ret_3d": ret_3d[i],
                "ret_5d": ret_5d[i],
                "z_3d": z_3d[i],
                "close_loc": close_loc[i],
                "upper_wick": upper_wick[i],
                "lower_wick": lower_wick[i],
                "body_ratio": body_ratio[i],
                "rsi_14": rsi_14[i],
                "sma_20": sma_20[i],
                "sma_40": sma_40[i],
                "min_prev_40": np.min(l[max(0, i-40): i]),
                "earnings": tk_earn
            })
            
    df_timeline = pd.DataFrame(all_bars).sort_values("timestamp").reset_index(drop=True)
    print(f"Total barras cronológicas indexadas: {len(df_timeline):,}")
    
    # Simulación de Cartera
    CAPITAL_INICIAL = 25000.0
    cash = CAPITAL_INICIAL
    active_longs = {}   # ticker: dict
    active_shorts = {}  # ticker: dict
    closed_trades = []
    equity_curve = []
    
    TOTAL_SLOTS = 5
    SLOT_SIZE = CAPITAL_INICIAL / TOTAL_SLOTS # $5,000 por posición
    
    timestamps_unique = df_timeline["timestamp"].drop_duplicates().tolist()
    print(f"Puntos de tiempo únicos: {len(timestamps_unique):,}")
    
    # Agrupar barras por timestamp para simulación sincrónica
    bars_by_ts = {ts: g for ts, g in df_timeline.groupby("timestamp")}
    
    for cur_ts in timestamps_unique:
        if cur_ts not in bars_by_ts:
            continue
        cur_bars = bars_by_ts[cur_ts]
        
        # 1. Determinar Régimen Macro
        is_bear_regime = False
        if cur_ts in spy_c:
            spy_val = spy_c[cur_ts]
            spy_200 = spy_sma200_d.get(cur_ts, np.nan)
            spy_r5 = spy_ret_5d.get(cur_ts, 0.0)
            if (not np.isnan(spy_200) and spy_val < spy_200) or (spy_r5 <= -0.025):
                is_bear_regime = True
                
        max_long_slots = 1 if is_bear_regime else 4
        max_short_slots = 4 if is_bear_regime else 1
        
        # 2. Gestionar Posiciones Largas Activas
        long_to_close = []
        for tk, pos in list(active_longs.items()):
            pos_bars = cur_bars[cur_bars["ticker"] == tk]
            if len(pos_bars) == 0:
                continue
            bar = pos_bars.iloc[0]
            pos["bars_held"] += 1
            bh, bl, bc = bar["high"], bar["low"], bar["close"]
            
            # Trailing stop MeLi / Stop loss
            if bl <= pos["stop_loss"]:
                pnl = (pos["stop_loss"] - pos["entry_p"]) / pos["entry_p"] * SLOT_SIZE
                cash += (SLOT_SIZE + pnl)
                closed_trades.append({
                    "ticker": tk, "type": "LONG", "entry_ts": pos["entry_ts"],
                    "exit_ts": cur_ts, "pnl": pnl, "ret": pnl / SLOT_SIZE, "reason": "stop"
                })
                long_to_close.append(tk)
            elif pos["bars_held"] >= 35: # 5 días máx
                pnl = (bc - pos["entry_p"]) / pos["entry_p"] * SLOT_SIZE
                cash += (SLOT_SIZE + pnl)
                closed_trades.append({
                    "ticker": tk, "type": "LONG", "entry_ts": pos["entry_ts"],
                    "exit_ts": cur_ts, "pnl": pnl, "ret": pnl / SLOT_SIZE, "reason": "time_max"
                })
                long_to_close.append(tk)
            else:
                # Trailing stop dinámico
                pos["stop_loss"] = max(pos["stop_loss"], bc * 0.95)
                
        for tk in long_to_close:
            del active_longs[tk]
            
        # 3. Gestionar Posiciones Cortas Activas (Modelo 2B)
        short_to_close = []
        for tk, pos in list(active_shorts.items()):
            pos_bars = cur_bars[cur_bars["ticker"] == tk]
            if len(pos_bars) == 0:
                continue
            bar = pos_bars.iloc[0]
            pos["bars_held"] += 1
            bh, bl, bc = bar["high"], bar["low"], bar["close"]
            
            pos["min_seen"] = min(pos["min_seen"], bl)
            gain = (pos["entry_p"] - bl) / pos["entry_p"]
            
            # Stop Loss
            if bh >= pos["dyn_stop"]:
                pnl = (pos["entry_p"] - pos["dyn_stop"]) / pos["entry_p"] * SLOT_SIZE
                cash += (SLOT_SIZE + pnl)
                closed_trades.append({
                    "ticker": tk, "type": "SHORT", "entry_ts": pos["entry_ts"],
                    "exit_ts": cur_ts, "pnl": pnl, "ret": pnl / SLOT_SIZE,
                    "reason": "be_trailing" if pos["be_locked"] else "hard_stop"
                })
                short_to_close.append(tk)
            # Target Hit
            elif bl <= pos["target_p"]:
                pnl = (pos["entry_p"] - pos["target_p"]) / pos["entry_p"] * SLOT_SIZE
                cash += (SLOT_SIZE + pnl)
                closed_trades.append({
                    "ticker": tk, "type": "SHORT", "entry_ts": pos["entry_ts"],
                    "exit_ts": cur_ts, "pnl": pnl, "ret": pnl / SLOT_SIZE, "reason": "target_hit"
                })
                short_to_close.append(tk)
            # Time Max (4 días)
            elif pos["bars_held"] >= 28:
                pnl = (pos["entry_p"] - bc) / pos["entry_p"] * SLOT_SIZE
                cash += (SLOT_SIZE + pnl)
                closed_trades.append({
                    "ticker": tk, "type": "SHORT", "entry_ts": pos["entry_ts"],
                    "exit_ts": cur_ts, "pnl": pnl, "ret": pnl / SLOT_SIZE, "reason": "time_max"
                })
                short_to_close.append(tk)
            else:
                # Quick BE Lock
                if gain >= (0.8 * pos["atr"]) and not pos["be_locked"]:
                    pos["dyn_stop"] = pos["entry_p"] * 0.998
                    pos["be_locked"] = True
                # Trailing Ceiling
                if gain >= (1.2 * pos["atr"]):
                    pos["dyn_stop"] = min(pos["dyn_stop"], pos["min_seen"] + 0.4 * pos["atr"] * pos["entry_p"])
                    
        for tk in short_to_close:
            del active_shorts[tk]
            
        # 4. Escanear Nuevas Entradas de Motor 1 (Longs) si hay cupo disponible
        if len(active_longs) < max_long_slots and cash >= SLOT_SIZE:
            for _, bar in cur_bars.iterrows():
                tk = bar["ticker"]
                if tk in active_longs or tk in active_shorts:
                    continue
                # Filtros Motor 1.1 Long: Z <= -2.0, ATR >= 2.2%, Guardián Anti-Cuchillo
                if bar["z_3d"] <= -2.0 and bar["atr_d"] >= 0.022 and bar["z_3d"] >= -4.2:
                    if bar["lower_wick"] >= 0.16 and bar["body_ratio"] <= 0.70:
                        entry_p = bar["next_open"]
                        stop_p = entry_p * (1.0 - 1.2 * bar["atr_d"])
                        active_longs[tk] = {
                            "entry_p": entry_p, "stop_loss": stop_p, "atr": bar["atr_d"],
                            "entry_ts": cur_ts, "bars_held": 0
                        }
                        cash -= SLOT_SIZE
                        if len(active_longs) >= max_long_slots or cash < SLOT_SIZE:
                            break
                            
        # 5. Escanear Nuevas Entradas de Motor 2 (Shorts) si hay cupo disponible y régimen calificado
        if is_bear_regime and len(active_shorts) < max_short_slots and cash >= SLOT_SIZE:
            for _, bar in cur_bars.iterrows():
                tk = bar["ticker"]
                if tk in active_longs or tk in active_shorts:
                    continue
                if bar["atr_d"] < 0.022:
                    continue
                    
                # Chequear Breakdown o PEAD
                is_breakdown = (bar["close"] < bar["min_prev_40"]) and (bar["rvol"] >= 1.6) and (bar["close_loc"] <= 0.30)
                is_pead = False
                for e_d, surp in bar["earnings"].items():
                    diff = (cur_ts - e_d).total_seconds() / 86400.0
                    if 1.0 <= diff <= 4.0 and surp <= -0.04:
                        is_pead = True
                        break
                        
                if not (is_breakdown or is_pead):
                    continue
                    
                # Evaluar con Modelo 2A LightGBM
                spy_r3 = float(spy_ret_3d.get(cur_ts, 0.0))
                feat_vec = np.array([[
                    bar["atr_d"], bar["ret_3d"], bar["ret_5d"], bar["z_3d"], bar["rvol"],
                    bar["close_loc"], bar["upper_wick"], bar["lower_wick"], bar["body_ratio"],
                    bar["rsi_14"], (bar["close"] - bar["sma_20"])/(bar["sma_20"]+1e-6),
                    (bar["close"] - bar["sma_40"])/(bar["sma_40"]+1e-6),
                    -0.05 if is_pead else 0.0,
                    bar["ret_3d"] - spy_r3,
                    spy_r3
                ]])
                prob = booster_2a.predict(feat_vec)[0]
                if prob >= 0.45: # Umbral calificado
                    entry_p = bar["next_open"]
                    stop_p = max(bar["high"], entry_p * (1.0 + 0.8 * bar["atr_d"]))
                    target_p = entry_p * (1.0 - 1.6 * bar["atr_d"])
                    active_shorts[tk] = {
                        "entry_p": entry_p, "dyn_stop": stop_p, "target_p": target_p,
                        "atr": bar["atr_d"], "entry_ts": cur_ts, "bars_held": 0,
                        "min_seen": entry_p, "be_locked": False
                    }
                    cash -= SLOT_SIZE
                    if len(active_shorts) >= max_short_slots or cash < SLOT_SIZE:
                        break
                        
        # Registrar Equity actual
        long_val = sum(SLOT_SIZE for _ in active_longs)
        short_val = sum(SLOT_SIZE for _ in active_shorts)
        nav = cash + long_val + short_val
        equity_curve.append({"timestamp": cur_ts, "equity": nav})
        
    df_res = pd.DataFrame(closed_trades)
    df_eq = pd.DataFrame(equity_curve)
    
    print("\n" + "=" * 80)
    print("RESULTADOS DEL PORTAFOLIO DUAL UNIFICADO ($25,000 USD)")
    print("=" * 80)
    print(f"Total Operaciones Ejecutadas: {len(df_res):,}")
    print(f"  Longs (Motor 1.1): {len(df_res[df_res['type'] == 'LONG']):,}")
    print(f"  Shorts (Motor 2):  {len(df_res[df_res['type'] == 'SHORT']):,}")
    
    total_pnl = df_res["pnl"].sum()
    final_equity = CAPITAL_INICIAL + total_pnl
    ret_pct = total_pnl / CAPITAL_INICIAL * 100
    
    wins = df_res[df_res["pnl"] > 0]["pnl"].sum()
    losses = abs(df_res[df_res["pnl"] <= 0]["pnl"].sum())
    pf = wins / (losses + 1e-6)
    wr = (df_res["pnl"] > 0).mean() * 100
    
    # Drawdown
    df_eq["peak"] = df_eq["equity"].cummax()
    df_eq["dd"] = (df_eq["equity"] - df_eq["peak"]) / df_eq["peak"]
    max_dd = df_eq["dd"].min() * 100
    
    print(f"\nCapital Inicial:      ${CAPITAL_INICIAL:,.2f}")
    print(f"Capital Final:        ${final_equity:,.2f} ({ret_pct:+.1f}%)")
    print(f"Beneficio Neto:       ${total_pnl:,.2f}")
    print(f"Win Rate Global:      {wr:.1f}%")
    print(f"Profit Factor Global: {pf:.2f}")
    print(f"Max Drawdown MtM:     {max_dd:.2f}%")
    
    df_res["year"] = pd.to_datetime(df_res["exit_ts"]).dt.year
    print("\nDesglose Anual del Sistema Dual:")
    by_yr = df_res.groupby("year")["pnl"].agg(["count", "sum"])
    by_yr["ret_pct"] = by_yr["sum"] / CAPITAL_INICIAL * 100
    print(by_yr)

if __name__ == "__main__":
    run_dual_engine_backtest()
