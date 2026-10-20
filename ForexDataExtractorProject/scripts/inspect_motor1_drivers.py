"""
inspect_motor1_drivers.py - Diagnóstico de qué aprendió exactamente el árbol en los 5,595 golpes
y validación estricta Out-of-Time + Out-of-Sector (sin compartir fechas).
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from scripts.evaluate_multisector_motor1 import compute_core_user_features_per_ticker, deduplicate_events


def main():
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    earn_path = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"

    df = pd.read_parquet(hourly_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    spy = df[df["ticker"] == "SPY"].sort_values("timestamp").copy()
    spy["hour_key"] = spy["timestamp"].dt.floor("h")
    spy = spy.drop_duplicates(subset=["hour_key"]).set_index("hour_key")

    spy_open_next = spy["open"].shift(-1)
    spy_fut_10d = (spy["close"].shift(-70) - spy_open_next) / spy_open_next * 100.0
    spy_fut_15d = (spy["close"].shift(-105) - spy_open_next) / spy_open_next * 100.0
    spy_ret_3d = spy["close"].pct_change(21)

    stocks = df[~df["ticker"].isin(["SPY", "QQQ"])].copy().reset_index(drop=True)
    stocks["hour_key"] = stocks["timestamp"].dt.floor("h")
    stocks["spy_fut_10d"] = stocks["hour_key"].map(spy_fut_10d).ffill()
    stocks["spy_fut_15d"] = stocks["hour_key"].map(spy_fut_15d).ffill()
    stocks["spy_ret_3d"] = stocks["hour_key"].map(spy_ret_3d).fillna(0.0)

    g = stocks.groupby("ticker", group_keys=False)
    stocks["ret_3d"] = g["close"].pct_change(21)
    stocks["ret_5d"] = g["close"].pct_change(35)
    stocks["ret_60d"] = g["close"].pct_change(420)
    stocks["vol_3d"] = g["ret_3d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["vol_5d"] = g["ret_5d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["z_3d"] = stocks["ret_3d"] / stocks["vol_3d"].replace(0, np.nan)
    stocks["z_5d"] = stocks["ret_5d"] / stocks["vol_5d"].replace(0, np.nan)

    open_next = g["open"].shift(-1)
    stocks["fut_ret_10d"] = (g["close"].shift(-70) - open_next) / open_next * 100.0
    stocks["fut_ret_15d"] = (g["close"].shift(-105) - open_next) / open_next * 100.0
    stocks["fut_alpha_10d"] = stocks["fut_ret_10d"] - stocks["spy_fut_10d"]
    stocks["fut_alpha_15d"] = stocks["fut_ret_15d"] - stocks["spy_fut_15d"]

    processed_groups = []
    for tk in stocks["ticker"].unique():
        sub = stocks[stocks["ticker"] == tk]
        sub_feat = compute_core_user_features_per_ticker(sub)
        shock_sub = sub_feat[(sub_feat["z_3d"] <= -2.0) | (sub_feat["z_5d"] <= -2.0)].copy()
        if not shock_sub.empty:
            processed_groups.append(shock_sub)

    shocks_all = pd.concat(processed_groups, ignore_index=True).dropna(subset=["fut_ret_10d", "fut_alpha_10d"]).reset_index(drop=True)

    earn_df = pd.read_parquet(earn_path)
    earn_df["earnings_date"] = pd.to_datetime(earn_df["earnings_date"], utc=True)
    earn_df = earn_df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
    earn_df["eps_beat_4q_ratio"] = earn_df.groupby("ticker")["eps_beat"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    earn_df["eps_surprise_4q_mean"] = earn_df.groupby("ticker")["eps_surprise_pct"].transform(lambda s: s.rolling(4, min_periods=1).mean().clip(-100, 200))

    shocks_all = shocks_all.sort_values("timestamp").reset_index(drop=True)
    earn_sorted = earn_df[["ticker", "earnings_date", "eps_surprise_pct", "eps_positive", "eps_beat", "eps_beat_4q_ratio", "eps_surprise_4q_mean"]].sort_values("earnings_date").reset_index(drop=True)

    shocks_all = pd.merge_asof(shocks_all, earn_sorted, left_on="timestamp", right_on="earnings_date", by="ticker", direction="backward")
    shocks_all["days_since_earnings"] = ((shocks_all["timestamp"] - shocks_all["earnings_date"]).dt.total_seconds() / 86400.0).clip(0, 120).fillna(45.0)
    shocks_all["eps_surprise_pct"] = shocks_all["eps_surprise_pct"].clip(-100, 200).fillna(0.0)
    shocks_all["eps_positive"] = shocks_all["eps_positive"].fillna(1.0)
    shocks_all["eps_beat"] = shocks_all["eps_beat"].fillna(0.5)
    shocks_all["eps_beat_4q_ratio"] = shocks_all["eps_beat_4q_ratio"].fillna(0.5)
    shocks_all["eps_surprise_4q_mean"] = shocks_all["eps_surprise_4q_mean"].fillna(0.0)

    shocks_dedup = deduplicate_events(shocks_all, "timestamp", min_gap_hours=48)
    shocks_dedup["target_recovery"] = ((shocks_dedup["fut_alpha_10d"] > 1.0) & (shocks_dedup["fut_ret_10d"] > 2.0)).astype(int)

    # 1. ¿Qué descubrió el modelo sobre days_since_earnings?
    print("=" * 95)
    print("1. EL DESCUBRIMIENTO DE 'days_since_earnings' (¿CUÁNDO DENTRO DEL TRIMESTRE OCURRE EL GOLPE?):")
    print("=" * 95)
    bins = [0, 3, 15, 45, 75, 125]
    labels = [
        "1. Día 0 a 3 Post-Earnings (Choque de Resultados)",
        "2. Día 4 a 15 Post-Earnings (Digestión Post-Reporte)",
        "3. Día 16 a 45 (Mitad de Trimestre Tranquila)",
        "4. Día 46 a 75 (Pre-Earnings / Rumores)",
        "5. Día 76+ (Justo antes del siguiente reporte)"
    ]
    shocks_dedup["earn_window"] = pd.cut(shocks_dedup["days_since_earnings"], bins=bins, labels=labels, include_lowest=True)
    ew = shocks_dedup.groupby("earn_window", observed=False).agg(
        eventos=("target_recovery", "count"),
        exito_fuerte_pct=("target_recovery", lambda s: round(s.mean() * 100, 1)),
        ret_10d_pct=("fut_ret_10d", lambda s: round(s.mean(), 2)),
        alpha_10d_pct=("fut_alpha_10d", lambda s: round(s.mean(), 2)),
        ret_15d_pct=("fut_ret_15d", lambda s: round(s.mean(), 2)),
        alpha_15d_pct=("fut_alpha_15d", lambda s: round(s.mean(), 2)),
        wr_15d_pct=("fut_ret_15d", lambda s: round((s > 0).mean() * 100, 1)),
    ).reset_index()
    print(ew.to_string(index=False))

    # 2. ¿Qué pasa en el Choque de Resultados (Día 0 a 5) según si batió o falló EPS + Absorción?
    print("\n" + "=" * 95)
    print("2. RADIOGRAFÍA DEL GOLPE EN EARNINGS (DÍA 0 A 5 TRAS PUBLICAR RESULTADOS):")
    print("=" * 95)
    pe = shocks_dedup[shocks_dedup["days_since_earnings"] <= 5.0].copy()
    pe_cases = [
        ("Falló EPS (Surprise < 0%) y cayó fuerte", pe["eps_surprise_pct"] < 0),
        ("Superó EPS (Surprise > 0%) pero cayó por pánico", pe["eps_surprise_pct"] > 0),
        ("Superó EPS Fuerte (> +8%) + Estiramiento SMA_100 (<-6%)", (pe["eps_surprise_pct"] > 8.0) & (pe["close_diff_SMA_100"] < -6.0)),
    ]
    pe_rows = []
    for name, m in pe_cases:
        sub = pe[m]
        if len(sub) > 0:
            pe_rows.append({
                "Caso Post-Earnings (<=5 días)": name,
                "Eventos": len(sub),
                "Ret_10d %": round(sub["fut_ret_10d"].mean(), 2),
                "Alpha_10d %": round(sub["fut_alpha_10d"].mean(), 2),
                "Ret_15d %": round(sub["fut_ret_15d"].mean(), 2),
                "Alpha_15d %": round(sub["fut_alpha_15d"].mean(), 2),
                "WR_15d %": round((sub["fut_ret_15d"] > 0).mean() * 100, 1),
            })
    print(pd.DataFrame(pe_rows).to_string(index=False))


if __name__ == "__main__":
    main()
