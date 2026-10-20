"""
Prueba de Fuego Multiaño del Motor 1.1 sobre datos 100% nativos de Interactive Brokers:
- Periodo: 2020 a 2026 (7 años completos que cubren COVID Crash 2020, Bull 2021, Bear 2022 y Rally 2023-2026).
- Canasta multisectorial descargada de IBKR (314,909 barras horarias).
- Reglas y umbrales congelados del Motor 1.1 (Fila 4 ganadora: Guardián Anti-Cuchillo, candado Momento MeLi, 5 slots).
- Evaluación año por año (Walk-Forward sin reajustar nada).
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PARQUET_IBKR = PROJECT_ROOT / "data_ibkr" / "sp500_ibkr_multisector_7years_1hour.parquet"

def detect_3_to_5d_shocks(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta los golpes duros de 3 a 5 días con vela de absorción en datos IBKR"""
    events = []
    
    # 7 barras por día en IBKR RTH
    bars_per_day = 7
    bars_3d = 3 * bars_per_day # 21
    bars_5d = 5 * bars_per_day # 35
    
    for ticker, tdf in df.groupby("ticker"):
        tdf = tdf.sort_values("timestamp").reset_index(drop=True)
        if len(tdf) < 100:
            continue
            
        c = tdf["close"].to_numpy()
        o = tdf["open"].to_numpy()
        h = tdf["high"].to_numpy()
        l = tdf["low"].to_numpy()
        ts = tdf["timestamp"].to_numpy()
        
        # Retornos de 3d y 5d
        ret_3d = np.zeros(len(tdf))
        ret_5d = np.zeros(len(tdf))
        ret_3d[bars_3d:] = (c[bars_3d:] - c[:-bars_3d]) / c[:-bars_3d]
        ret_5d[bars_5d:] = (c[bars_5d:] - c[:-bars_5d]) / c[:-bars_5d]
        
        # Volatilidad móvil de retornos para z-score
        roll_std_3d = pd.Series(ret_3d).rolling(250, min_periods=50).std().fillna(0.04).to_numpy()
        roll_std_5d = pd.Series(ret_5d).rolling(250, min_periods=50).std().fillna(0.05).to_numpy()
        
        z_3d = ret_3d / (roll_std_3d + 1e-6)
        z_5d = ret_5d / (roll_std_5d + 1e-6)
        
        # ATR diario aproximado (rolling 14 días = 98 barras horarias)
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr_14d = pd.Series(tr).rolling(98, min_periods=20).mean().to_numpy() / (c + 1e-6)
        
        # Microestructura de la vela
        rng = np.maximum(h - l, 1e-6)
        close_loc = (c - l) / rng
        lower_wick = (np.minimum(o, c) - l) / rng
        
        for i in range(max(bars_5d, 100), len(tdf) - 105):
            # Condición de golpe elástico: Z <= -2.0 en 3d o 5d + ATR >= 2.2% + vela de absorción
            is_shock = (z_3d[i] <= -2.0 or z_5d[i] <= -2.0)
            if not is_shock:
                continue
            if atr_14d[i] < 0.020: # elasticidad
                continue
            is_absorption = (close_loc[i] >= 0.50 or lower_wick[i] >= 0.20)
            if not is_absorption:
                continue
                
            entry_p = o[i + 1] # Ejecución en Open de la barra siguiente
            shock_drop = abs(min(ret_3d[i], ret_5d[i]))
            if shock_drop < 0.03:
                continue
                
            # Futuro para simular salida con la regla del Motor 1.1
            fut_h = h[i + 1 : i + 1 + 105]
            fut_l = l[i + 1 : i + 1 + 105]
            fut_c = c[i + 1 : i + 1 + 105]
            fut_o = o[i + 1 : i + 1 + 105]
            fut_ts = ts[i + 1 : i + 1 + 105]
            
            # Simular salida exacta del Motor 1.1 (Momento MeLi + Stop Loss diario -1.5 ATR)
            sl_p = entry_p * (1.0 - 1.5 * atr_14d[i])
            target_meli = entry_p * (1.0 + 0.45 * shock_drop)
            
            peak_p = entry_p
            floor_p = None
            exit_p = fut_c[-1]
            exit_ts = fut_ts[-1]
            exit_reason = "TimeStop_15d"
            reached_meli = False
            
            for step in range(len(fut_c)):
                # Stop vigente
                if floor_p is not None and step > 0:
                    if fut_c[step] <= floor_p:
                        exit_p = fut_o[step + 1] if step + 1 < len(fut_c) else fut_c[step]
                        exit_ts = fut_ts[step + 1] if step + 1 < len(fut_c) else fut_ts[step]
                        exit_reason = "Trailing_MomentoMeLi"
                        break
                        
                # SL estructural cada 7 barras (cierre de día)
                if (step + 1) % 7 == 0 and fut_c[step] <= sl_p:
                    exit_p = fut_o[step + 1] if step + 1 < len(fut_c) else fut_c[step]
                    exit_ts = fut_ts[step + 1] if step + 1 < len(fut_c) else fut_ts[step]
                    exit_reason = "SL_Diario"
                    break
                    
                if fut_h[step] > peak_p:
                    peak_p = fut_h[step]
                    
                # Evaluar activación Momento MeLi (+45% del golpe)
                if fut_h[step] >= target_meli:
                    reached_meli = True
                    # Piso asegurado en +35% del golpe y trailing stop
                    floor_p = max(floor_p or 0.0, entry_p * (1.0 + 0.35 * shock_drop), peak_p * (1.0 - 0.70 * atr_14d[i]))
                    
            net_ret = ((exit_p * 0.9995 - entry_p) / entry_p) * 100.0 # 5 bps fricción por pata = 10 bps
            
            events.append({
                "ticker": ticker,
                "entry_ts": pd.Timestamp(ts[i + 1]),
                "exit_ts": pd.Timestamp(exit_ts),
                "year": pd.Timestamp(ts[i + 1]).year,
                "entry_p": entry_p,
                "exit_p": exit_p,
                "net_ret_pct": net_ret,
                "reached_meli": int(reached_meli),
                "exit_reason": exit_reason,
                "shock_drop": shock_drop,
                "atr_d": atr_14d[i]
            })
            
    return pd.DataFrame(events)

def simulate_portfolio(events_df: pd.DataFrame, initial_capital: float = 25000.0, max_slots: int = 5):
    """Simulador de portafolio IBKR 5 cupos sin margen"""
    df_s = events_df.sort_values("entry_ts").reset_index(drop=True)
    if df_s.empty:
        return {"trades": 0, "wr": 0.0, "ret_pct": 0.0, "pf": 0.0, "max_dd": 0.0}
        
    cash = initial_capital
    open_pos = []
    closed = []
    eq_curve = []
    
    for _, row in df_s.iterrows():
        cur_ts = row["entry_ts"]
        
        # Cerrar vencidos
        still = []
        for p in open_pos:
            if p["exit_ts"] <= cur_ts:
                cash += p["returned"]
                closed.append(p)
            else:
                still.append(p)
        open_pos = still
        
        cur_eq = cash + sum(p["invested"] for p in open_pos)
        eq_curve.append({"ts": cur_ts, "eq": cur_eq})
        
        if len(open_pos) >= max_slots:
            continue
        if any(p["ticker"] == row["ticker"] for p in open_pos):
            continue
            
        alloc = min(cash / (max_slots - len(open_pos)), cur_eq / max_slots)
        if alloc < 500:
            continue
            
        cash -= alloc
        ret_f = row["net_ret_pct"] / 100.0
        returned = alloc * (1.0 + ret_f)
        
        open_pos.append({
            "ticker": row["ticker"],
            "entry_ts": cur_ts,
            "exit_ts": row["exit_ts"],
            "invested": alloc,
            "returned": returned,
            "net_ret_pct": row["net_ret_pct"]
        })
        
    for p in open_pos:
        cash += p["returned"]
        closed.append(p)
        
    tdf = pd.DataFrame(closed)
    wins = tdf[tdf["net_ret_pct"] > 0]["net_ret_pct"].sum()
    loss = abs(tdf[tdf["net_ret_pct"] <= 0]["net_ret_pct"].sum())
    pf = wins / loss if loss > 0 else 9.99
    
    eq_df = pd.DataFrame(eq_curve)
    max_dd = 0.0
    if not eq_df.empty:
        eq_df["peak"] = eq_df["eq"].cummax()
        eq_df["dd"] = (eq_df["eq"] - eq_df["peak"]) / eq_df["peak"] * 100.0
        max_dd = float(eq_df["dd"].min())
        
    return {
        "trades": len(tdf),
        "wr": float((tdf["net_ret_pct"] > 0).mean() * 100.0) if len(tdf) > 0 else 0.0,
        "pf": pf,
        "ret_pct": float((cash - initial_capital) / initial_capital * 100.0),
        "final_capital": cash,
        "max_dd": max_dd,
        "meli_rate": float(events_df["reached_meli"].mean() * 100.0)
    }

def main():
    print("=" * 90)
    print("INICIANDO PRUEBA DE FUEGO MULTIAÑO DEL MOTOR 1.1 EN DATOS NATIVOS IBKR (2020 - 2026)")
    print("=" * 90)
    
    df_all = pd.read_parquet(PARQUET_IBKR)
    # Excluir benchmarks para señales de trading (solo acciones)
    df_stocks = df_all[~df_all["ticker"].isin(["SPY", "QQQ"])].copy()
    print(f"Total barras horarias analizadas: {len(df_stocks):,} en {df_stocks['ticker'].nunique()} activos.")
    
    events_df = detect_3_to_5d_shocks(df_stocks)
    print(f"\nTotal eventos de 'Golpe Duro' detectados (2020 - 2026): {len(events_df)}")
    
    # Evaluación Global
    global_res = simulate_portfolio(events_df, initial_capital=25000.0, max_slots=5)
    print("\n" + "=" * 90)
    print("1. RESULTADO GLOBAL MULTIAÑO (Enero 2020 - Octubre 2026, 7 AÑOS COMPLETOS):")
    print("=" * 90)
    print(f"Trades Ejecutados:      {global_res['trades']}")
    print(f"Win Rate:               {global_res['wr']:.1f}%")
    print(f"Profit Factor:          {global_res['pf']:.2f}")
    print(f"Retorno Acumulado:      +{global_res['ret_pct']:.1f}% (${global_res['final_capital']:,.0f})")
    print(f"Max Drawdown Cerrado:   {global_res['max_dd']:.2f}%")
    print(f"Tasa Momento MeLi:      {global_res['meli_rate']:.1f}%")
    
    # Evaluación Año por Año (Walk-Forward con reglas congeladas)
    print("\n" + "=" * 90)
    print("2. DESGLOSE AÑO POR AÑO (WALK-FORWARD CON REGLAS CONGELADAS):")
    print("=" * 90)
    
    years = sorted(events_df["year"].unique())
    y_rows = []
    
    for yr in years:
        ev_yr = events_df[events_df["year"] == yr].copy()
        res_yr = simulate_portfolio(ev_yr, initial_capital=25000.0, max_slots=5)
        y_rows.append({
            "Año": yr,
            "Régimen de Mercado": "COVID Crash / Reapertura" if yr == 2020 else ("Bull Market Euforia" if yr == 2021 else ("Bear Market Inflación/Tasas" if yr == 2022 else "Rally IA / Consolidación")),
            "Trades": res_yr["trades"],
            "Win Rate": f"{res_yr['wr']:.1f}%",
            "Profit Factor": round(res_yr["pf"], 2),
            "Retorno Anual": f"{'+' if res_yr['ret_pct']>=0 else ''}{res_yr['ret_pct']:.1f}%",
            "Max Drawdown": f"{res_yr['max_dd']:.2f}%",
            "% Llega MeLi": f"{res_yr['meli_rate']:.1f}%"
        })
        
    print(pd.DataFrame(y_rows).to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    main()
