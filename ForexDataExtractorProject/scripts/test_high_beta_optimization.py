"""
test_high_beta_optimization.py - Optimización de Sizing y Selección de Activos
Compara configuraciones estáticas (sin trailing asfixiante) con:
- Riesgo 1% ($1,000) vs 2% ($2,000)
- Todo el universo (21 activos) vs Líderes de Alto Beta
- Filtro Alpha de Liderazgo (alpha_5d > 0)
- Desglose detallado año por año (2022, 2023, 2024, 2025, 2026)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scripts.test_active_trailing_and_shorts import simulate_bar_by_bar_active

def main():
    data_path = Path("data/processed/labeled_universe_1hour.parquet")
    df = pd.read_parquet(data_path).dropna(subset=["target_quality_trade"]).reset_index(drop=True)

    # Señal base institucional: Breakout + Bull Market + RVOL > 1.2
    df["signal_dir"] = 0
    base_long = (
        (df["dist_to_res20_atr"] < 0.5) &
        (df["is_market_bullish"] == 1) &
        (df["rvol"] > 1.2)
    )
    df.loc[base_long, "signal_dir"] = 1

    # Líderes de Alto Beta
    high_beta_leaders = ["PLTR", "NVDA", "TSLA", "MELI", "AMD", "ARM", "COIN", "MSTR", "META", "SHOP", "SMCI"]

    experiments = [
        # (Nombre, Riesgo, MaxPos, SoloLeaders, ExigirAlphaPositivo)
        ("1. Universo Completo (21 activos) | Riesgo $1,000 (1%)", 1000.0, 5, False, False),
        ("2. Universo Completo (21 activos) | Riesgo $2,000 (2%)", 2000.0, 7, False, False),
        ("3. Universo Completo | Riesgo $2,000 + Filtro Alpha > 0", 2000.0, 7, False, True),
        ("4. Líderes Alto Beta (11 activos) | Riesgo $1,000 (1%)", 1000.0, 5, True, False),
        ("5. Líderes Alto Beta | Riesgo $2,000 (2%)", 2000.0, 7, True, False),
        ("6. LÍDERES ALTO BETA | Riesgo $2,000 + Filtro Alpha > 0", 2000.0, 7, True, True),
    ]

    results = []
    annual_tables = {}

    for name, risk, max_pos, only_lead, req_alpha in experiments:
        df_run = df.copy()
        if only_lead:
            df_run = df_run[df_run["ticker"].isin(high_beta_leaders + ["SPY"])].copy()
        if req_alpha:
            df_run.loc[df_run["alpha_5d"] <= 0, "signal_dir"] = 0

        eq_df, tr, met = simulate_bar_by_bar_active(
            df_run,
            initial_capital=100_000.0,
            risk_per_trade_usd=risk,
            max_open_positions=max_pos,
            use_breakeven=False,  # NO asfixiar trades ganadores
            use_trailing=False,   # Dejar correr el TP completo de 3x ATR
            friday_filter=True,
        )

        # Desglose anual
        s = eq_df.set_index("timestamp")["total_equity"].resample("YE").last()
        first = pd.Series([100_000.0], index=[pd.Timestamp("2021-12-31", tz="UTC")])
        fy = pd.concat([first, s])
        ann_rets = fy.pct_change().dropna() * 100.0
        ann_rets.index = ann_rets.index.year
        annual_tables[name] = ann_rets

        results.append({
            "Configuración": name,
            "Retorno Total": f"{met['retorno_total_pct']:+.2f}%",
            "CAGR %": f"{met['cagr_pct']:.2f}%",
            "Sharpe": met["sharpe_ratio"],
            "Max DD %": f"{met['max_drawdown_pct']:.2f}%",
            "Win Rate": f"{met['win_rate_pct']:.2f}%",
            "Profit Factor": met["profit_factor"],
            "Trades": met["total_trades"],
            "Ganancia USD": f"${met['ganancia_neta']:+,.0f}",
        })

    print("=" * 95)
    print("COMPARATIVA DE OPTIMIZACIÓN INSTITUCIONAL DE SWING TRADING (2022 - 2026)")
    print("=" * 95)
    print(pd.DataFrame(results).to_string(index=False))

    print("\n--- DESGLOSE DE RENTABILIDAD ANUAL (%) POR AÑO ---")
    ann_df = pd.DataFrame(annual_tables).round(2)
    print(ann_df.to_string())
    print("=" * 95)

if __name__ == "__main__":
    main()
