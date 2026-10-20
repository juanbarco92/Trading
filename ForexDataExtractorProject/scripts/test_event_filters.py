"""
test_event_filters.py - Análisis de eventos y filtros de entrada institucionales.
"""

import pandas as pd
import numpy as np


def main():
    df = pd.read_parquet("data/processed/labeled_universe_1hour.parquet").dropna(subset=["target_quality_trade"])
    print(f"Total velas analizadas: {len(df):,}")
    print(f"Tasa base global (random bar): {df['target_quality_trade'].mean() * 100:.2f}%\n")

    filters = [
        ("Base (Sin filtro / Todas las velas)", np.ones(len(df), dtype=bool)),
        ("is_bullish_absorption == 1", df["is_bullish_absorption"] == 1),
        ("rvol_surge == 1", df["rvol_surge"] == 1),
        ("is_vol_compression == 1", df["is_vol_compression"] == 1),
        ("is_above_vwap & is_market_bullish", (df["is_above_vwap"] == 1) & (df["is_market_bullish"] == 1)),
        ("alpha_5d > 0 (Fuerza relativa vs SPY)", df["alpha_5d"] > 0),
        ("rs_divergence_bullish == 1", df["rs_divergence_bullish"] == 1),
        ("dist_to_sup20_atr < 1.0 (Rebote cerca de soporte)", df["dist_to_sup20_atr"] < 1.0),
        ("dist_to_res20_atr < 0.5 (Breakout resistencia)", df["dist_to_res20_atr"] < 0.5),
        ("Breakout + Bull Market + RVOL > 1.2", (df["dist_to_res20_atr"] < 0.5) & (df["is_market_bullish"] == 1) & (df["rvol"] > 1.2)),
        ("Compresión Vol + Alpha 5D > 0 + Bull Market", (df["is_vol_compression"] == 1) & (df["alpha_5d"] > 0) & (df["is_market_bullish"] == 1)),
        ("Absorption + Above VWAP + Bull Market", (df["is_bullish_absorption"] == 1) & (df["is_above_vwap"] == 1) & (df["is_market_bullish"] == 1)),
        ("Soporte + Compresión + Alpha > 0", (df["dist_to_sup20_atr"] < 1.5) & (df["is_vol_compression"] == 1) & (df["alpha_5d"] > 0)),
    ]

    rows = []
    for name, mask in filters:
        sub = df[mask]
        n = len(sub)
        if n > 0:
            tp_rate = sub["target_quality_trade"].mean() * 100.0
            wr = (sub["trade_net_return"] > 0).mean() * 100.0
            ret = sub["trade_net_return"].mean()
            # Retorno de ganadores vs perdedores
            wins = sub.loc[sub["trade_net_return"] > 0, "trade_net_return"]
            losses = sub.loc[sub["trade_net_return"] <= 0, "trade_net_return"]
            pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999.0
            rows.append({
                "Filtro / Setup": name,
                "Señales": f"{n:,}",
                "Frecuencia %": f"{(n/len(df))*100:.1f}%",
                "TP Rate %": f"{tp_rate:.2f}%",
                "Win Rate %": f"{wr:.2f}%",
                "Retorno Medio Neto": f"{ret:+.2f}%",
                "Profit Factor": f"{pf:.2f}",
            })

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
