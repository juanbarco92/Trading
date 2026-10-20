"""
Test de Validación y Diagnóstico Inicial del Motor 2 (Shorts):
- Aplica el detector de eventos bajistas sobre los 7 años de datos de Interactive Brokers (2020 a 2026).
- Cruza con el dataset de reportes de resultados PIT (19,385 earnings).
- Evalúa el Win Rate, Profit Factor y Retorno año por año en operaciones exclusivamente de VENTA EN CORTO (Shorts).
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.bearish_shock_detector import detect_bearish_events
from scripts.run_multiaño_fire_test_ibkr import simulate_portfolio

PARQUET_IBKR = PROJECT_ROOT / "data_ibkr" / "sp500_ibkr_multisector_7years_1hour.parquet"
EARNINGS_PARQUET = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"

def main():
    print("=" * 90)
    print("INICIANDO PRUEBA PILOTO DEL MOTOR 2 (SHORTS / VENTAS EN CORTO) EN DATOS IBKR (2020 - 2026)")
    print("=" * 90)

    # 1. Cargar datos de IBKR y Earnings
    df_all = pd.read_parquet(PARQUET_IBKR)
    df_stocks = df_all[~df_all["ticker"].isin(["SPY", "QQQ"])].copy()
    
    earn_df = pd.read_parquet(EARNINGS_PARQUET)
    print(f"Barras horarias cargadas: {len(df_stocks):,} en {df_stocks['ticker'].nunique()} activos.")
    print(f"Reportes trimestrales disponibles: {len(earn_df):,}")

    # 2. Detectar eventos bajistas
    print("\nDetectando eventos bajistas (PEAD negativo, Blow-off tops, Breakdown, Contagio)...")
    t0 = time.time()
    
    all_events = []
    for ticker, tdf in df_stocks.groupby("ticker"):
        ev_tk = detect_bearish_events(tdf, df_earnings_pit=earn_df)
        if not ev_tk.empty:
            all_events.append(ev_tk)
            
    if not all_events:
        print("[ERROR] No se detectaron eventos bajistas.")
        return
        
    events_df = pd.concat(all_events, ignore_index=True)
    events_df["reached_meli"] = events_df["reached_desplome_45"]
    print(f"[OK] Total eventos bajistas detectados en 7 años: {len(events_df):,} en {time.time() - t0:.1f}s")

    # 3. Distribución por Arquetipo
    print("\n" + "=" * 90)
    print("1. ANATOMÍA Y RENDIMIENTO POR ARQUETIPO BAJISTA (SHORTS):")
    print("=" * 90)
    arq_stats = []
    for code, gdf in events_df.groupby("archetype_name"):
        wr = (gdf["net_ret_pct"] > 0).mean() * 100.0
        wins = gdf[gdf["net_ret_pct"] > 0]["net_ret_pct"].sum()
        loss = abs(gdf[gdf["net_ret_pct"] <= 0]["net_ret_pct"].sum())
        pf = wins / loss if loss > 0 else 9.99
        arq_stats.append({
            "Arquetipo": code,
            "Eventos": len(gdf),
            "Win_Rate": f"{wr:.1f}%",
            "Ret_Promedio": f"{gdf['net_ret_pct'].mean():+.2f}%",
            "Profit_Factor": round(pf, 2),
            "%_Toca_Desplome_45": f"{gdf['reached_desplome_45'].mean()*100:.1f}%"
        })
    print(pd.DataFrame(arq_stats).to_string(index=False))

    # 4. Simulación de Portafolio Short ($25,000, 5 cupos)
    print("\n" + "=" * 90)
    print("2. RENDIMIENTO GLOBAL EN VENTA EN CORTO (SHORTS 2020 - 2026):")
    print("=" * 90)
    glob_res = simulate_portfolio(events_df, initial_capital=25000.0, max_slots=5)
    print(f"Trades Ejecutados en Corto: {glob_res['trades']}")
    print(f"Win Rate:                   {glob_res['wr']:.1f}%")
    print(f"Profit Factor:              {glob_res['pf']:.2f}")
    print(f"Retorno Acumulado en Short: +{glob_res['ret_pct']:.1f}% (${glob_res['final_capital']:,.0f})")
    print(f"Max Drawdown:               {glob_res['max_dd']:.2f}%")

    # 5. Desglose Año por Año (¿Cómo le fue en el Bear Market de 2022 vs Bull de 2021?)
    print("\n" + "=" * 90)
    print("3. DESGLOSE AÑO POR AÑO EN VENTA EN CORTO (SHORTS):")
    print("=" * 90)
    y_rows = []
    for yr in sorted(events_df["year"].unique()):
        ev_yr = events_df[events_df["year"] == yr]
        res_yr = simulate_portfolio(ev_yr, initial_capital=25000.0, max_slots=5)
        y_rows.append({
            "Año": yr,
            "Régimen": "COVID Crash" if yr == 2020 else ("Bull Market" if yr == 2021 else ("Bear Market" if yr == 2022 else "Rally IA")),
            "Trades_Short": res_yr["trades"],
            "Win_Rate": f"{res_yr['wr']:.1f}%",
            "Retorno_Anual_Short": f"{'+' if res_yr['ret_pct']>=0 else ''}{res_yr['ret_pct']:.1f}%",
            "Profit_Factor": round(res_yr["pf"], 2),
            "Max_Drawdown": f"{res_yr['max_dd']:.2f}%"
        })
    print(pd.DataFrame(y_rows).to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    main()
