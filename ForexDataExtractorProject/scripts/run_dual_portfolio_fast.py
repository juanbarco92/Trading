"""
Simulación Rápida de Cartera Dual: Motor 1.1 (Longs) + Motor 2 (Shorts)
Capital Inicial: $25,000 USD
5 slots concurrentes ($5,000 por posición fija)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cargar eventos de ambos motores
df1 = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "strict_prod_df_1a.parquet")
df2 = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "motor_2a_dataset_refined.parquet")
import lightgbm as lgb
booster_2a = lgb.Booster(model_file=str(PROJECT_ROOT / "models" / "modelo_2a_shorts_lgb.txt"))

# Cargar precios horarios para simulación de salida
df_raw = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet")
stocks_dict = {tk: gdf.sort_values("timestamp").reset_index(drop=True) for tk, gdf in df_raw.groupby("ticker")}

# 1. Preparar Trades de Motor 1 (Longs) con configuración congelada
# Guardián Anti-cuchillo: z_3d >= -4.2, lower_wick >= 0.16, body_ratio <= 0.70
m1_candidates = df1[
    (df1["z_3d"] <= -2.1) &
    (df1["z_3d"] >= -4.0) &
    (df1["atr_d"] >= 0.022) &
    (df1["lower_wick_ratio"] >= 0.22) &
    (df1["rvol_1h"] >= 1.3) &
    (df1["body_ratio"] <= 0.65)
].copy()

# Deduplicar señales por ticker y día
m1_candidates["day"] = pd.to_datetime(m1_candidates["timestamp"]).dt.date
m1_candidates = m1_candidates.sort_values("timestamp").drop_duplicates(["ticker", "day"]).reset_index(drop=True)

long_trades_pool = []
for _, row in m1_candidates.iterrows():
    tk = row["ticker"]
    if tk not in stocks_dict:
        continue
    stk_df = stocks_dict[tk]
    ts_entry = row["timestamp"]
    matches = stk_df[stk_df["timestamp"] == ts_entry]
    if len(matches) == 0:
        continue
    idx = matches.index[0]
    if idx + 1 >= len(stk_df):
        continue
        
    entry_p = stk_df.iloc[idx + 1]["open"]
    atr = row["atr_d"]
    entry_ts = stk_df.iloc[idx + 1]["timestamp"]
    
    fut = stk_df.iloc[idx + 1 : idx + 1 + 35]
    stop_p = entry_p * (1.0 - 1.2 * atr)
    trailing_stop = stop_p
    ret = 0.0
    exit_ts = fut.iloc[-1]["timestamp"]
    
    for _, bar in fut.iterrows():
        bh, bl, bc = bar["high"], bar["low"], bar["close"]
        if bl <= trailing_stop:
            ret = (trailing_stop - entry_p) / entry_p
            exit_ts = bar["timestamp"]
            break
        # MeLi Trailing: subir stop si avanza
        if bc >= entry_p * (1.0 + 1.0 * atr):
            trailing_stop = max(trailing_stop, bc * 0.96)
    else:
        ret = (fut.iloc[-1]["close"] - entry_p) / entry_p
        
    long_trades_pool.append({
        "ticker": tk,
        "type": "LONG",
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "ret": ret - 0.0010, # 10 bps comisiones round-trip
        "year": entry_ts.year
    })

# 2. Preparar Trades de Motor 2 (Shorts) con Modelo 2A y 2B
features_2a = [
    "atr_d", "ret_3d", "ret_5d", "z_3d", "rvol_1h", "close_loc_1h",
    "upper_wick_ratio", "lower_wick_ratio", "body_ratio", "rsi_14",
    "dist_sma_20", "dist_sma_40", "eps_surprise_pct", "rel_strength_3d",
    "spy_ret_3d"
]
df2["prob_2a"] = booster_2a.predict(df2[features_2a])
m2_candidates = df2[df2["prob_2a"] >= df2["prob_2a"].median()].copy().reset_index(drop=True)

short_trades_pool = []
for _, row in m2_candidates.iterrows():
    tk = row["ticker"]
    if tk not in stocks_dict:
        continue
    stk_df = stocks_dict[tk]
    ts_entry = row["entry_ts"]
    matches = stk_df[stk_df["timestamp"] == ts_entry]
    if len(matches) == 0:
        continue
    idx = matches.index[0]
    
    entry_p = row["entry_p"]
    atr = row["atr_d"]
    stop_p = max(row["stop_loss_p"], entry_p * (1.0 + 0.8 * atr))
    target_p = entry_p * (1.0 - 1.6 * atr)
    
    fut = stk_df.iloc[idx : idx + 28]
    dyn_stop = stop_p
    be_locked = False
    min_seen = entry_p
    ret = 0.0
    exit_ts = fut.iloc[-1]["timestamp"]
    
    for _, bar in fut.iterrows():
        bh, bl, bc = bar["high"], bar["low"], bar["close"]
        min_seen = min(min_seen, bl)
        gain = (entry_p - bl) / entry_p
        
        if bh >= dyn_stop:
            ret = (entry_p - dyn_stop) / entry_p
            exit_ts = bar["timestamp"]
            break
        if bl <= target_p:
            ret = (entry_p - target_p) / entry_p
            exit_ts = bar["timestamp"]
            break
        if gain >= (0.8 * atr) and not be_locked:
            dyn_stop = entry_p * 0.998
            be_locked = True
        if gain >= (1.2 * atr):
            dyn_stop = min(dyn_stop, min_seen + 0.4 * atr * entry_p)
    else:
        ret = (entry_p - fut.iloc[-1]["close"]) / entry_p
        
    short_trades_pool.append({
        "ticker": tk,
        "type": "SHORT",
        "entry_ts": ts_entry,
        "exit_ts": exit_ts,
        "ret": ret - 0.0010, # 10 bps round-trip
        "year": ts_entry.year
    })

df_all_trades = pd.DataFrame(long_trades_pool + short_trades_pool).sort_values("entry_ts").reset_index(drop=True)
print(f"Total oportunidades generadas: {len(df_all_trades)} (Longs: {len(long_trades_pool)}, Shorts: {len(short_trades_pool)})")

# 3. Simulación de Cartera con Límite de 5 Slots Concurrentes
CAPITAL_INICIAL = 25000.0
SLOT_SIZE = 5000.0
cash = CAPITAL_INICIAL
open_positions = []
executed_trades = []

# Evaluar trade por trade respetando timestamps
for _, trade in df_all_trades.iterrows():
    cur_entry = trade["entry_ts"]
    
    # Cerrar posiciones que ya expiraron antes de cur_entry
    still_open = []
    for pos in open_positions:
        if pos["exit_ts"] <= cur_entry:
            pnl = pos["ret"] * SLOT_SIZE
            cash += (SLOT_SIZE + pnl)
            executed_trades.append({**pos, "pnl": pnl})
        else:
            still_open.append(pos)
    open_positions = still_open
    
    # Chequear si hay slot disponible
    if len(open_positions) < 5 and cash >= SLOT_SIZE:
        # Evitar operar la misma acción en dos posiciones concurrentes
        if trade["ticker"] not in [p["ticker"] for p in open_positions]:
            open_positions.append(trade)
            cash -= SLOT_SIZE

# Cerrar remanentes
for pos in open_positions:
    pnl = pos["ret"] * SLOT_SIZE
    cash += (SLOT_SIZE + pnl)
    executed_trades.append({**pos, "pnl": pnl})

df_exec = pd.DataFrame(executed_trades).sort_values("entry_ts").reset_index(drop=True)
print("\n" + "=" * 80)
print("RESULTADOS DEL PORTAFOLIO DUAL UNIFICADO ($25,000 USD)")
print("=" * 80)
print(f"Total Trades Ejecutados: {len(df_exec):,}")
print(f"  Longs:  {len(df_exec[df_exec['type'] == 'LONG']):,}")
print(f"  Shorts: {len(df_exec[df_exec['type'] == 'SHORT']):,}")

pnl_tot = df_exec["pnl"].sum()
final_nav = CAPITAL_INICIAL + pnl_tot
wr = (df_exec["pnl"] > 0).mean() * 100
wins = df_exec[df_exec["pnl"] > 0]["pnl"].sum()
losses = abs(df_exec[df_exec["pnl"] <= 0]["pnl"].sum())
pf = wins / (losses + 1e-6)

# Calcular Drawdown de Capital Cerrado
df_exec["cum_pnl"] = df_exec["pnl"].cumsum()
df_exec["equity"] = CAPITAL_INICIAL + df_exec["cum_pnl"]
df_exec["peak"] = df_exec["equity"].cummax()
df_exec["dd"] = (df_exec["equity"] - df_exec["peak"]) / df_exec["peak"] * 100
max_dd = df_exec["dd"].min()

print(f"\nCapital Inicial:      ${CAPITAL_INICIAL:,.2f}")
print(f"Capital Final:        ${final_nav:,.2f} ({(final_nav/CAPITAL_INICIAL - 1)*100:+.1f}%)")
print(f"Beneficio Neto:       ${pnl_tot:,.2f}")
print(f"Win Rate Global:      {wr:.1f}%")
print(f"Profit Factor Global: {pf:.2f}")
print(f"Max Drawdown Cerrado: {max_dd:.2f}%")

print("\nDesglose por Año:")
by_yr = df_exec.groupby("year")["pnl"].agg(["count", "sum"])
by_yr["Retorno_%"] = (by_yr["sum"] / CAPITAL_INICIAL) * 100
by_yr["WinRate"] = df_exec.groupby("year")["pnl"].apply(lambda x: f"{(x>0).mean()*100:.1f}%")
print(by_yr)

print("\nDesglose por Tipo de Posición:")
by_type = df_exec.groupby("type")["pnl"].agg(["count", "sum"])
by_type["Retorno_%"] = (by_type["sum"] / CAPITAL_INICIAL) * 100
by_type["WinRate"] = df_exec.groupby("type")["pnl"].apply(lambda x: f"{(x>0).mean()*100:.1f}%")
print(by_type)
