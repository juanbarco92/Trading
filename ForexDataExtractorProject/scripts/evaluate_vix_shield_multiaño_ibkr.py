"""
Prueba del Escudo Macro (VIX y Tasas TNX) sobre los datos de 7 años de IBKR (2020 - 2026):
- ¿Qué pasa si apagamos las compras cuando el VIX está en pánico agudo (> 32) o acelerando fuerte?
- Compara el Drawdown y Retorno en 2020 (COVID) y 2022 (Bear Market) con vs sin Escudo Macro.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PARQUET_IBKR = PROJECT_ROOT / "data_ibkr" / "sp500_ibkr_multisector_7years_1hour.parquet"

from scripts.run_multiaño_fire_test_ibkr import detect_3_to_5d_shocks, simulate_portfolio

def main():
    print("=" * 90)
    print("APLICANDO ESCUDO MACRO (VIX) A LA PRUEBA MULTIAÑO DE 7 AÑOS IBKR (2020 - 2026)")
    print("=" * 90)
    
    df_all = pd.read_parquet(PARQUET_IBKR)
    
    # 1. Cargar serie diaria del VIX oficial (2020 a 2026)
    vix_df = pd.read_parquet(PROJECT_ROOT / "data_ibkr" / "vix_daily_2020_2026.parquet")
    # Aplanar nombres de columnas si vienen como tuplas de Yahoo
    vix_df.columns = [c[0] if isinstance(c, tuple) else str(c).replace("('", "").replace("', '')", "").replace("', '^VIX')", "").replace("'", "") for c in vix_df.columns]
    
    date_col = [c for c in vix_df.columns if "Date" in c][0]
    close_col = [c for c in vix_df.columns if "Close" in c][0]
    vix_df["date_key"] = pd.to_datetime(vix_df[date_col]).dt.tz_localize(None).dt.floor("D")
    vix_map = vix_df.set_index("date_key")[close_col].to_dict()
    
    # 2. Detectar golpes en las acciones
    df_stocks = df_all[~df_all["ticker"].isin(["SPY", "QQQ"])].copy()
    events_df = detect_3_to_5d_shocks(df_stocks)
    
    # Mapear VIX en el día exacto de la entrada
    events_df["entry_day"] = pd.to_datetime(events_df["entry_ts"]).dt.tz_localize(None).dt.floor("D")
    events_df["vix_level"] = events_df["entry_day"].map(vix_map).ffill().fillna(20.0)
    
    # 3. Filtrar eventos con el Escudo Macro:
    # No comprar cuando el mercado está en pánico extremo (VIX >= 32.0, como durante el colapso de marzo 2020)
    events_shielded = events_df[events_df["vix_level"] < 32.0].copy()
    
    print(f"Total eventos brutos: {len(events_df)}")
    print(f"Eventos aprobados tras Escudo Macro VIX: {len(events_shielded)} (vetados {len(events_df) - len(events_shielded)} cuchillos de pánico)")
    
    # Comparar Año por Año
    years = sorted(events_df["year"].unique())
    rows = []
    
    for yr in years:
        raw_yr = events_df[events_df["year"] == yr]
        shd_yr = events_shielded[events_shielded["year"] == yr]
        
        r_raw = simulate_portfolio(raw_yr, 25000.0, 5)
        r_shd = simulate_portfolio(shd_yr, 25000.0, 5)
        
        rows.append({
            "Año": yr,
            "Trades_Bruto": r_raw["trades"],
            "Retorno_Bruto": f"{'+' if r_raw['ret_pct']>=0 else ''}{r_raw['ret_pct']:.1f}%",
            "MaxDD_Bruto": f"{r_raw['max_dd']:.2f}%",
            "Trades_ConEscudo": r_shd["trades"],
            "Retorno_ConEscudo": f"{'+' if r_shd['ret_pct']>=0 else ''}{r_shd['ret_pct']:.1f}%",
            "MaxDD_ConEscudo": f"{r_shd['max_dd']:.2f}%",
            "PF_ConEscudo": round(r_shd["pf"], 2)
        })
        
    print("\n" + "=" * 115)
    print("COMPARATIVA AÑO POR AÑO: SISTEMA BRUTO VS SISTEMA CON ESCUDO MACRO VIX:")
    print("=" * 115)
    print(pd.DataFrame(rows).to_string(index=False))
    print("=" * 115)
    
    # Global con Escudo
    glob_shd = simulate_portfolio(events_shielded, 25000.0, 5)
    print("\nRESULTADO GLOBAL MULTIAÑO CON ESCUDO MACRO (2020 - 2026):")
    print(f"Trades: {glob_shd['trades']} | WR: {glob_shd['wr']:.1f}% | PF: {glob_shd['pf']:.2f}")
    print(f"Retorno Acumulado: +{glob_shd['ret_pct']:.1f}% (${glob_shd['final_capital']:,.0f})")
    print(f"Max Drawdown: {glob_shd['max_dd']:.2f}% (frente a -25.48% bruto)")

if __name__ == "__main__":
    main()
