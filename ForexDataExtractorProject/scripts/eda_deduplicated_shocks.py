"""
eda_deduplicated_shocks.py - Análisis de Eventos Únicos (Sin Solapamiento) de Golpes Duros.

Garantiza que si una acción cae durante varias horas de un mismo episodio, se cuente como
UN SOLO EVENTO (cooldown mínimo de 35 barras = 5 días por ticker entre eventos),
evaluando la estabilidad año por año (2022 a 2026).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def deduplicate_events(df_events: pd.DataFrame, time_col: str, min_gap_hours: int = 120) -> pd.DataFrame:
    """Filtra eventos para un mismo ticker dejando al menos min_gap_hours (ej. 5 días calendario) entre señales."""
    kept = []
    for ticker, grp in df_events.sort_values(time_col).groupby("ticker"):
        last_time = None
        for idx, row in grp.iterrows():
            t = row[time_col]
            if last_time is None or (t - last_time).total_seconds() / 3600.0 >= min_gap_hours:
                kept.append(idx)
                last_time = t
    return df_events.loc[kept].sort_values(time_col).reset_index(drop=True)


def main():
    data_path = PROJECT_ROOT / "data" / "processed" / "universe_1hour.parquet"
    df = pd.read_parquet(data_path)
    time_col = "timestamp" if "timestamp" in df.columns else "datetime"
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(["ticker", time_col]).reset_index(drop=True)

    spy = df[df["ticker"] == "SPY"].sort_values(time_col).set_index(time_col)
    spy_ret_1d = spy["close"].pct_change(7)
    spy_ret_3d = spy["close"].pct_change(21)
    spy_ret_5d = spy["close"].pct_change(35)
    spy_open_next = spy["open"].shift(-1)
    spy_fut_10d = (spy["close"].shift(-70) - spy_open_next) / spy_open_next * 100.0
    spy_fut_15d = (spy["close"].shift(-105) - spy_open_next) / spy_open_next * 100.0

    stocks = df[~df["ticker"].isin(["SPY", "QQQ"])].copy().reset_index(drop=True)
    stocks["spy_ret_1d"] = stocks[time_col].map(spy_ret_1d)
    stocks["spy_ret_3d"] = stocks[time_col].map(spy_ret_3d)
    stocks["spy_ret_5d"] = stocks[time_col].map(spy_ret_5d)
    stocks["spy_fut_10d"] = stocks[time_col].map(spy_fut_10d)
    stocks["spy_fut_15d"] = stocks[time_col].map(spy_fut_15d)

    g = stocks.groupby("ticker", group_keys=False)
    stocks["ret_1d"] = g["close"].pct_change(7)
    stocks["ret_3d"] = g["close"].pct_change(21)
    stocks["ret_5d"] = g["close"].pct_change(35)

    stocks["vol_1d"] = g["ret_1d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["vol_3d"] = g["ret_3d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["vol_5d"] = g["ret_5d"].transform(lambda s: s.rolling(420, min_periods=70).std())

    stocks["z_1d"] = stocks["ret_1d"] / stocks["vol_1d"].replace(0, np.nan)
    stocks["z_3d"] = stocks["ret_3d"] / stocks["vol_3d"].replace(0, np.nan)
    stocks["z_5d"] = stocks["ret_5d"] / stocks["vol_5d"].replace(0, np.nan)

    # Volumen y vela 1H
    stocks["vol_sma50_1h"] = g["volume"].transform(lambda s: s.rolling(50, min_periods=10).mean())
    stocks["rvol_1h"] = (stocks["volume"] / stocks["vol_sma50_1h"].replace(0, np.nan)).clip(0, 15.0)
    rng_1h = (stocks["high"] - stocks["low"]).replace(0, np.nan)
    stocks["close_loc_1h"] = (stocks["close"] - stocks["low"]) / rng_1h
    stocks["lower_wick_1h"] = (np.minimum(stocks["open"], stocks["close"]) - stocks["low"]) / rng_1h

    # Desenlaces futuros (Open[t+1])
    open_next = g["open"].shift(-1)
    stocks["fut_ret_10d"] = (g["close"].shift(-70) - open_next) / open_next * 100.0
    stocks["fut_ret_15d"] = (g["close"].shift(-105) - open_next) / open_next * 100.0
    stocks["fut_alpha_10d"] = stocks["fut_ret_10d"] - stocks["spy_fut_10d"]
    stocks["fut_alpha_15d"] = stocks["fut_ret_15d"] - stocks["spy_fut_15d"]
    stocks["year"] = stocks[time_col].dt.year
    stocks["hour_utc"] = stocks[time_col].dt.hour

    valid = stocks.dropna(subset=["z_3d", "fut_ret_10d", "fut_alpha_10d"]).copy()

    # Probar 4 arquetipos de eventos deduplicados (mínimo 5 días entre eventos de la misma acción)
    archetypes = [
        (
            "Arquetipo 1: Golpe 1 Día (z_1d <= -2.5) + SPY Estable (> 0%) [Idiosincrático Puro]",
            (valid["z_1d"] <= -2.5) & (valid["spy_ret_1d"] > 0.0),
        ),
        (
            "Arquetipo 2: Golpe 1 Día (z_1d <= -2.5) + SPY Cayendo (<= -1.0%) [Arrastre Macro]",
            (valid["z_1d"] <= -2.5) & (valid["spy_ret_1d"] <= -0.01),
        ),
        (
            "Arquetipo 3: Liquidación 3-5 Días (z_3d o z_5d <= -2.5) + Vela Absorción/Giro 1H",
            ((valid["z_3d"] <= -2.5) | (valid["z_5d"] <= -2.5))
            & ((valid["close_loc_1h"] >= 0.60) | (valid["lower_wick_1h"] >= 0.30))
            & (valid["rvol_1h"] >= 1.3),
        ),
        (
            "Arquetipo 4: Capitulación Extrema (z_3d <= -3.0) + Tarde Institucional (16-20 UTC) + Giro 1H",
            (valid["z_3d"] <= -3.0)
            & (valid["hour_utc"].isin([16, 17, 18, 19, 20]))
            & (valid["close_loc_1h"] >= 0.55),
        ),
    ]

    print("=" * 100)
    print("COMPARATIVA DE ARQUETIPOS DE 'GOLPE DURO' (EVENTOS ÚNICOS DEDUPLICADOS, SIN SOLAPAMIENTO)")
    print("=" * 100)

    for name, mask in archetypes:
        ev = deduplicate_events(valid[mask], time_col, min_gap_hours=120)
        print(f"\n>>> {name}")
        print(f"    Total Eventos Únicos: {len(ev)} | Ret 10d: {ev['fut_ret_10d'].mean():+.2f}% | Alpha 10d: {ev['fut_alpha_10d'].mean():+.2f}% | WR 10d: {(ev['fut_ret_10d']>0).mean()*100:.1f}% | Alpha>0: {(ev['fut_alpha_10d']>0).mean()*100:.1f}%")
        print(f"    Ret 15d: {ev['fut_ret_15d'].mean():+.2f}% | Alpha 15d: {ev['fut_alpha_15d'].mean():+.2f}% | WR 15d: {(ev['fut_ret_15d']>0).mean()*100:.1f}% | Alpha15d>0: {(ev['fut_alpha_15d']>0).mean()*100:.1f}%")

        # Desglose anual
        yr_rows = []
        for yr, grp in ev.groupby("year"):
            yr_rows.append({
                "Año": yr,
                "N_Eventos": len(grp),
                "Ret_10d %": round(grp["fut_ret_10d"].mean(), 2),
                "Alpha_10d %": round(grp["fut_alpha_10d"].mean(), 2),
                "Ret_15d %": round(grp["fut_ret_15d"].mean(), 2),
                "Alpha_15d %": round(grp["fut_alpha_15d"].mean(), 2),
                "WR_15d %": round((grp["fut_ret_15d"] > 0).mean() * 100, 1),
                "Alpha15d>0 %": round((grp["fut_alpha_15d"] > 0).mean() * 100, 1),
            })
        print(pd.DataFrame(yr_rows).to_string(index=False))


if __name__ == "__main__":
    main()
