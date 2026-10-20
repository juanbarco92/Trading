"""
annual_returns_breakdown.py - Desglose detallado de Rendimiento Efectivo Anual (EA),
mínimo, máximo y promedio por año (2022, 2023, 2024, 2025, 2026).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


def main():
    curves_path = Path("data/processed/backtest_equity_curves.parquet")
    raw_data_path = Path("data/processed/universe_1hour.parquet")

    curves = pd.read_parquet(curves_path)
    raw_df = pd.read_parquet(raw_data_path)

    # Añadir SPY Buy & Hold como comparador directo normalizado a $100,000
    spy_df = raw_df[raw_df["ticker"] == "SPY"].sort_values("timestamp")
    spy_series = spy_df.set_index("timestamp")["close"]
    # Reindexar al índice de las curvas
    spy_reindexed = spy_series.reindex(curves.index, method="ffill")
    spy_equity = (spy_reindexed / spy_reindexed.iloc[0]) * 100_000.0
    curves["SPY (Buy & Hold)"] = spy_equity

    # Resample a fin de año (Año calendario)
    # Tomar el último valor de cada año y el inicial
    yearly_equity = curves.resample("YE").last()
    
    # Agregar el inicio (cierre de 2021)
    first_row = pd.DataFrame([curves.iloc[0]], index=[pd.Timestamp("2021-12-31", tz="UTC")])
    full_yearly = pd.concat([first_row, yearly_equity])

    # Calcular retornos porcentuales por año
    yearly_returns = full_yearly.pct_change().dropna() * 100.0
    yearly_returns.index = [d.year for d in yearly_returns.index]

    print("=" * 90)
    print("DESGLOSE DE RENDIMIENTO EFECTIVO POR AÑO (2022 - 2026)")
    print("=" * 90)
    print(yearly_returns.round(2).to_string())

    print("\n" + "=" * 90)
    print("ESTADÍSTICAS ANUALES RESUMIDAS (PROMEDIO, MÍNIMO, MÁXIMO Y CAGR GLOBAL):")
    print("=" * 90)

    summary_rows = []
    duration_years = (curves.index[-1] - curves.index[0]).days / 365.25

    for col in curves.columns:
        rets = yearly_returns[col]
        initial_val = curves[col].iloc[0]
        final_val = curves[col].iloc[-1]
        cagr = (((final_val / initial_val) ** (1.0 / duration_years)) - 1.0) * 100.0
        
        summary_rows.append({
            "Estrategia": col,
            "Rendimiento EA (CAGR)": f"{cagr:+.2f}%",
            "Promedio Anual Aritmético": f"{rets.mean():+.2f}%",
            "Año Mínimo (Peor Año)": f"{rets.min():+.2f}% ({rets.idxmin()})",
            "Año Máximo (Mejor Año)": f"{rets.max():+.2f}% ({rets.idxmax()})",
            "Desviación Estándar Anual": f"{rets.std():.2f}%",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()
