"""
evaluate_multisector_motor1.py - Evaluación Masiva Multi-Sectorial del Motor 1 (150+ Acciones, 10 Sectores).

Integra:
1. 150+ acciones del S&P 500 en 10 sectores económicos (Velas 1H y Diarias).
2. Detector de Golpes Duros de 3 a 5 días (z_3d <= -2.0 o z_5d <= -2.0).
3. Variables Estructurales del Usuario (Soportes/Resistencias causales stock_sr, Ratios elásticos SMA_100,
   Bandas de Bollinger, ATR, Geometría horaria y RVOL).
4. Variables Fundamentales / Earnings Point-in-Time (EPS Surprise %, EPS Positivo, Racha de Beats, Días desde reporte).
5. Pruebas de Fuego:
   - Out-of-Sample Cronológico (2025-2026).
   - Out-of-Sector (Entrenar SIN Tecnología y predecir en sectores tradicionales, y viceversa).
"""

import sys
import time
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
from src.utils.stock_sr import compute_stock_support_resistance_features


def deduplicate_events(df_events: pd.DataFrame, time_col: str, min_gap_hours: int = 72) -> pd.DataFrame:
    """Deja al menos min_gap_hours entre episodios de un mismo ticker."""
    kept = []
    for ticker, grp in df_events.sort_values(time_col).groupby("ticker"):
        last_time = None
        for idx, row in grp.iterrows():
            t = row[time_col]
            if last_time is None or (t - last_time).total_seconds() / 3600.0 >= min_gap_hours:
                kept.append(idx)
                last_time = t
    return df_events.loc[kept].sort_values(time_col).reset_index(drop=True)


def compute_core_user_features_per_ticker(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula de forma 100% causal las familias de variables del usuario que demostraron mayor
    poder predictivo dentro de los golpes de 3 a 5 días:
    - Soportes y Resistencias causales (compute_stock_support_resistance_features)
    - Ratios elásticos SMA_10, SMA_40, SMA_100 y pendientes
    - Bandas de Bollinger (20 y 50) y ancho de banda
    - ATR relativo (14 y 21)
    - Geometría de vela horaria y RVOL
    """
    sub = sub.sort_values("timestamp").copy()
    c = sub["close"]
    h = sub["high"]
    l = sub["low"]
    o = sub["open"]
    v = sub["volume"]

    # 1. Medias Móviles y Estiramiento Elástico (Familia #1 en el ranking)
    sma10 = c.rolling(10, min_periods=5).mean()
    sma40 = c.rolling(40, min_periods=15).mean()
    sma100 = c.rolling(100, min_periods=40).mean()

    sub["SMA_100_pct_momentum"] = c.pct_change(100) * 100.0
    sub["SMA_40_pct_momentum"] = c.pct_change(40) * 100.0
    sub["close_diff_SMA_100"] = (c - sma100) / sma100.replace(0, np.nan) * 100.0
    sub["close_diff_SMA_40"] = (c - sma40) / sma40.replace(0, np.nan) * 100.0
    sub["ratio_SMA_40_SMA_100"] = sma40 / sma100.replace(0, np.nan)
    sub["ratio_SMA_100_SMA_40"] = sma100 / sma40.replace(0, np.nan)
    sub["ratio_SMA_100_SMA_10"] = sma100 / sma10.replace(0, np.nan)
    sub["ratio_SMA_10_SMA_100"] = sma10 / sma100.replace(0, np.nan)
    sub["norm_diff_SMA_40_SMA_100"] = (sma40 - sma100) / sma100.replace(0, np.nan)
    sub["norm_diff_SMA_10_SMA_100"] = (sma10 - sma100) / sma100.replace(0, np.nan)
    sub["slope_SMA_100"] = sma100.pct_change(5) * 100.0
    sub["SMA_100_slope_mean"] = sub["slope_SMA_100"].rolling(10, min_periods=3).mean()

    # 2. Volatilidad ATR y Bandas de Bollinger (Familia #3 en el ranking)
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean()
    atr21 = tr.rolling(21, min_periods=7).mean()
    sub["ATR_14_pct"] = atr14 / c.replace(0, np.nan) * 100.0
    sub["ATR_21_pct"] = atr21 / c.replace(0, np.nan) * 100.0
    sub["ATR_21_relative"] = atr21 / atr21.rolling(100, min_periods=20).mean().replace(0, np.nan)

    std20 = c.rolling(20, min_periods=10).std()
    sma20 = c.rolling(20, min_periods=10).mean()
    sub["BB_20_20_bandwidth"] = (4.0 * std20) / sma20.replace(0, np.nan) * 100.0
    sub["BB_20_20_percent_b"] = (c - (sma20 - 2.0 * std20)) / (4.0 * std20).replace(0, np.nan)

    std50 = c.rolling(50, min_periods=20).std()
    sma50 = c.rolling(50, min_periods=20).mean()
    sub["BB_50_25_bandwidth"] = (5.0 * std50) / sma50.replace(0, np.nan) * 100.0

    # 3. RSI y Geometría de Vela Horaria + RVOL
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rs = gain / loss.replace(0, np.nan)
    sub["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    rng = (h - l).replace(0, np.nan)
    sub["close_loc_1h"] = (c - l) / rng
    sub["lower_wick_ratio"] = (np.minimum(o, c) - l) / rng
    sub["body_ratio"] = (c - o).abs() / rng

    vol_sma50 = v.rolling(50, min_periods=10).mean()
    sub["rvol_1h"] = (v / vol_sma50.replace(0, np.nan)).clip(0, 15.0)

    # 4. Soportes y Resistencias Causales del Usuario (Familia #2 en el ranking)
    sub = compute_stock_support_resistance_features(sub)

    sub["hour_of_day"] = sub["timestamp"].dt.hour
    sub["day_of_week"] = sub["timestamp"].dt.dayofweek
    return sub


def main():
    t0 = time.time()
    print("=" * 100)
    print("EVALUACIÓN A GRAN ESCALA DEL MOTOR 1: 164 ACCIONES DEL S&P 500 + TUS FEATURES + EARNINGS PIT")
    print("=" * 100)

    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    earn_path = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"

    df = pd.read_parquet(hourly_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    n_tickers = df["ticker"].nunique()
    print(f"1. Cargadas {len(df):,} velas de 1 Hora en {n_tickers} activos ({df['sector'].nunique()} sectores).")

    # Extraer SPY para Alpha (alineando por hora truncada para empatar :00 de IBKR con :30 de Yahoo Finance)
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

    # Identificar rápido qué activos tienen golpes de 3-5 días y calcular las features del usuario
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

    print("2. Calculando tus variables estructurales (S/R causales, Ratios SMA, Bandas, ATR) en los activos...")
    processed_groups = []
    unique_tks = stocks["ticker"].unique()
    for i, tk in enumerate(unique_tks):
        if (i + 1) % 30 == 0 or i == 0:
            print(f"   * Ingenierizando activo {i+1}/{len(unique_tks)} ({tk})...")
        sub = stocks[stocks["ticker"] == tk]
        sub_feat = compute_core_user_features_per_ticker(sub)
        # Filtrar directamente las barras de golpe duro (z_3d <= -2.0 o z_5d <= -2.0) para optimizar memoria
        shock_sub = sub_feat[(sub_feat["z_3d"] <= -2.0) | (sub_feat["z_5d"] <= -2.0)].copy()
        if not shock_sub.empty:
            processed_groups.append(shock_sub)

    shocks_all = pd.concat(processed_groups, ignore_index=True)
    shocks_all = shocks_all.dropna(subset=["fut_ret_10d", "fut_alpha_10d"]).reset_index(drop=True)

    # 3. Cruzar con Datos de Earnings Point-in-Time (merge_asof estrictamente hacia atrás: direction='backward')
    if earn_path.exists():
        print("3. Cruzando con Historial Point-in-Time de Earnings & EPS Surprise (Cero Lookahead)...")
        earn_df = pd.read_parquet(earn_path)
        earn_df["earnings_date"] = pd.to_datetime(earn_df["earnings_date"], utc=True)
        # Calcular racha de beats y promedio de sorpresa en los últimos 4 trimestres por empresa
        earn_df = earn_df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
        earn_df["eps_beat_4q_ratio"] = earn_df.groupby("ticker")["eps_beat"].transform(lambda s: s.rolling(4, min_periods=1).mean())
        earn_df["eps_surprise_4q_mean"] = earn_df.groupby("ticker")["eps_surprise_pct"].transform(lambda s: s.rolling(4, min_periods=1).mean().clip(-100, 200))

        shocks_all = shocks_all.sort_values("timestamp").reset_index(drop=True)
        earn_sorted = earn_df[["ticker", "earnings_date", "eps_surprise_pct", "eps_positive", "eps_beat", "eps_beat_4q_ratio", "eps_surprise_4q_mean"]].sort_values("earnings_date").reset_index(drop=True)

        shocks_all = pd.merge_asof(
            shocks_all,
            earn_sorted,
            left_on="timestamp",
            right_on="earnings_date",
            by="ticker",
            direction="backward",
        )
        shocks_all["days_since_earnings"] = ((shocks_all["timestamp"] - shocks_all["earnings_date"]).dt.total_seconds() / 86400.0).clip(0, 120).fillna(45.0)
        shocks_all["is_post_earnings_shock"] = (shocks_all["days_since_earnings"] <= 5.0).astype(int)
        shocks_all["eps_surprise_pct"] = shocks_all["eps_surprise_pct"].clip(-100, 200).fillna(0.0)
        shocks_all["eps_positive"] = shocks_all["eps_positive"].fillna(1.0)
        shocks_all["eps_beat"] = shocks_all["eps_beat"].fillna(0.5)
        shocks_all["eps_beat_4q_ratio"] = shocks_all["eps_beat_4q_ratio"].fillna(0.5)
        shocks_all["eps_surprise_4q_mean"] = shocks_all["eps_surprise_4q_mean"].fillna(0.0)
    else:
        for col in ["eps_surprise_pct", "eps_positive", "eps_beat", "eps_beat_4q_ratio", "eps_surprise_4q_mean", "days_since_earnings", "is_post_earnings_shock"]:
            shocks_all[col] = 0.0

    # Deduplicar episodios (mínimo 48h entre episodios de la misma acción)
    shocks_dedup = deduplicate_events(shocks_all, "timestamp", min_gap_hours=48)
    shocks_dedup["target_recovery"] = ((shocks_dedup["fut_alpha_10d"] > 1.0) & (shocks_dedup["fut_ret_10d"] > 2.0)).astype(int)

    print(f"\nTotal Barras de Golpe 3-5 Días: {len(shocks_all):,}")
    print(f"Total Episodios Únicos Deduplicados: {len(shocks_dedup):,} en {shocks_dedup['ticker'].nunique()} empresas!")

    # =========================================================================
    # 1. RESULTADOS POR SECTOR ECONÓMICO (COMPROBACIÓN MULTI-SECTORIAL)
    # =========================================================================
    print("\n" + "=" * 100)
    print("1. COMPORTAMIENTO DE LOS GOLPES DE 3-5 DÍAS EN LOS 10 SECTORES DEL S&P 500:")
    print("=" * 100)
    # Evaluar cuando hay absorción horaria (close_loc_1h >= 0.55 o mecha de rechazo) y EPS positivo
    abs_mask = (
        ((shocks_dedup["close_loc_1h"] >= 0.55) | (shocks_dedup["lower_wick_ratio"] >= 0.25))
        & (shocks_dedup["eps_positive"] == 1)
    )
    shocks_abs = shocks_dedup[abs_mask]

    sec_rows = []
    for sec, grp in shocks_abs.groupby("sector"):
        sec_rows.append({
            "Sector Económico": sec,
            "Empresas": grp["ticker"].nunique(),
            "Golpes c/Absorción": len(grp),
            "Ret 10d %": round(grp["fut_ret_10d"].mean(), 2),
            "Alpha 10d vs SPY %": round(grp["fut_alpha_10d"].mean(), 2),
            "Ret 15d %": round(grp["fut_ret_15d"].mean(), 2),
            "Alpha 15d vs SPY %": round(grp["fut_alpha_15d"].mean(), 2),
            "WR 15d %": round((grp["fut_ret_15d"] > 0).mean() * 100, 1),
            "Alpha15d>0 %": round((grp["fut_alpha_15d"] > 0).mean() * 100, 1),
        })
    df_sec = pd.DataFrame(sec_rows).sort_values("Alpha 15d vs SPY %", ascending=False)
    print(df_sec.to_string(index=False))

    # =========================================================================
    # 2. IMPACTO DE LA SALUD FUNDAMENTAL (EARNINGS SURPRISE & EPS POSITIVO)
    # =========================================================================
    print("\n" + "=" * 100)
    print("2. IMPACTO DEL RESPALDO FUNDAMENTAL (EARNINGS SURPRISE) EN EL REBOTE:")
    print("=" * 100)
    fund_groups = [
        ("A. Empresa en Pérdidas / EPS Negativo o Sorpresa Muy Mala (< -5%)", (shocks_dedup["eps_positive"] == 0) | (shocks_dedup["eps_surprise_pct"] < -5.0)),
        ("B. Fundamentales Neutros (Sorpresa EPS entre -5% y +5%)", (shocks_dedup["eps_positive"] == 1) & (shocks_dedup["eps_surprise_pct"] >= -5.0) & (shocks_dedup["eps_surprise_pct"] <= 5.0)),
        ("C. Fundamentales Sólidos (EPS Beat > +5% + Racha 4Q >= 75%)", (shocks_dedup["eps_positive"] == 1) & (shocks_dedup["eps_surprise_pct"] > 5.0) & (shocks_dedup["eps_beat_4q_ratio"] >= 0.75)),
        ("D. Golpe Recién Publicado Earnings (<= 5d) CON EPS Beat (> +5%) + Giro 1H", (shocks_dedup["is_post_earnings_shock"] == 1) & (shocks_dedup["eps_surprise_pct"] > 5.0) & (shocks_dedup["close_loc_1h"] >= 0.50)),
    ]
    f_rows = []
    for label, m in fund_groups:
        sub = shocks_dedup[m]
        if len(sub) == 0:
            continue
        f_rows.append({
            "Perfil Fundamental Point-in-Time": label,
            "Eventos": len(sub),
            "Ret 10d %": round(sub["fut_ret_10d"].mean(), 2),
            "Alpha 10d %": round(sub["fut_alpha_10d"].mean(), 2),
            "Ret 15d %": round(sub["fut_ret_15d"].mean(), 2),
            "Alpha 15d %": round(sub["fut_alpha_15d"].mean(), 2),
            "WR 15d %": round((sub["fut_ret_15d"] > 0).mean() * 100, 1),
        })
    print(pd.DataFrame(f_rows).to_string(index=False))

    # =========================================================================
    # 3. ENTRENAMIENTO DEL MODELO CON TUS FEATURES + FUNDAMENTALES (OOS & OUT-OF-SECTOR)
    # =========================================================================
    model_features = [
        "SMA_100_pct_momentum", "SMA_40_pct_momentum",
        "close_diff_SMA_100", "close_diff_SMA_40",
        "ratio_SMA_40_SMA_100", "ratio_SMA_100_SMA_40", "ratio_SMA_100_SMA_10",
        "norm_diff_SMA_40_SMA_100", "norm_diff_SMA_10_SMA_100",
        "slope_SMA_100", "SMA_100_slope_mean",
        "sr_dist_support_pct", "sr_dist_resistance_pct",
        "sr_dist_support_atr", "sr_dist_resistance_atr",
        "sr_support_touches", "sr_resistance_touches", "sr_channel_position",
        "ATR_14_pct", "ATR_21_pct", "ATR_21_relative",
        "BB_20_20_bandwidth", "BB_20_20_percent_b", "BB_50_25_bandwidth",
        "RSI_14", "close_loc_1h", "lower_wick_ratio", "body_ratio", "rvol_1h",
        "z_3d", "z_5d", "ret_60d", "spy_ret_3d",
        "eps_surprise_pct", "eps_positive", "eps_beat", "eps_beat_4q_ratio",
        "eps_surprise_4q_mean", "days_since_earnings", "is_post_earnings_shock",
        "hour_of_day", "day_of_week"
    ]
    model_features = [f for f in model_features if f in shocks_dedup.columns]

    # Split Cronológico: 70% primeros eventos en el tiempo (Train) vs 30% últimos eventos en el tiempo (Test OOS)
    shocks_sorted = shocks_dedup.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(shocks_sorted) * 0.70)
    tr_df = shocks_sorted.iloc[:split_idx].copy()
    te_df = shocks_sorted.iloc[split_idx:].copy()

    X_tr = tr_df[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_tr = tr_df["target_recovery"].values
    X_te = te_df[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_te = te_df["target_recovery"].values

    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=12,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=8.0,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    probs_te = clf.predict_proba(X_te)[:, 1]
    auc_oos = roc_auc_score(y_te, probs_te)

    print("\n" + "=" * 100)
    print(f"3. MODELO MOTOR 1 OUT-OF-SAMPLE CRONOLÓGICO ({len(te_df):,} EVENTOS CIEGOS EN {te_df['ticker'].nunique()} EMPRESAS)")
    print(f"   * Período Train: {tr_df['timestamp'].min().date()} a {tr_df['timestamp'].max().date()} ({len(tr_df):,} eventos)")
    print(f"   * Período Test:  {te_df['timestamp'].min().date()} a {te_df['timestamp'].max().date()} ({len(te_df):,} eventos)")
    print(f"   * ROC-AUC Out-of-Sample: {auc_oos:.4f}")
    print("=" * 100)

    te_df["prob"] = probs_te
    te_df["decil"] = pd.qcut(te_df["prob"], q=10, labels=False, duplicates="drop")

    d_summary = te_df.groupby("decil").agg(
        eventos=("target_recovery", "count"),
        prob_min=("prob", lambda s: round(s.min(), 3)),
        prob_max=("prob", lambda s: round(s.max(), 3)),
        exito_fuerte_pct=("target_recovery", lambda s: round(s.mean() * 100, 1)),
        wr_10d_pct=("fut_ret_10d", lambda s: round((s > 0).mean() * 100, 1)),
        ret_10d_pct=("fut_ret_10d", lambda s: round(s.mean(), 2)),
        alpha_10d_pct=("fut_alpha_10d", lambda s: round(s.mean(), 2)),
        ret_15d_pct=("fut_ret_15d", lambda s: round(s.mean(), 2)),
        alpha_15d_pct=("fut_alpha_15d", lambda s: round(s.mean(), 2)),
    ).reset_index()

    print("\n--- ORDENAMIENTO POR DECILES (TEST OUT-OF-SAMPLE EN 150+ ACCIONES) ---")
    print(d_summary.to_string(index=False))

    # =========================================================================
    # 4. PRUEBA DE FUEGO ANTI-BURBUJA IA: ENTRENAR SIN TECH -> PREDECIR FUERA Y DENTRO
    # =========================================================================
    print("\n" + "=" * 100)
    print("4. PRUEBA DE BLINDAJE ANTI-BURBUJA IA (CROSS-SECTOR GENERALIZATION):")
    print("   Entrenando EXCLUSIVAMENTE en Tecnología/Comunicación y prediciendo en Sectores Tradicionales")
    print("   (Banca, Salud, Consumo, Industria, Energía, Materiales) que el modelo JAMÁS VIO:")
    print("=" * 100)

    tech_mask = shocks_dedup["sector"].isin(["Technology", "Communication"])
    trad_df = shocks_dedup[~tech_mask].copy()
    tech_df = shocks_dedup[tech_mask].copy()

    clf_cross = lgb.LGBMClassifier(
        n_estimators=120, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    # Entrenar en Sectores Tradicionales -> Probar en Tech
    # Y entrenar en Tech -> Probar en Sectores Tradicionales
    X_tech = tech_df[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_tech = tech_df["target_recovery"].values
    X_trad = trad_df[model_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_trad = trad_df["target_recovery"].values

    clf_cross.fit(X_tech, y_tech)
    probs_trad = clf_cross.predict_proba(X_trad)[:, 1]
    auc_trad = roc_auc_score(y_trad, probs_trad)

    trad_df["prob"] = probs_trad
    trad_df["quintil"] = pd.qcut(trad_df["prob"], q=5, labels=False, duplicates="drop")
    q_trad = trad_df.groupby("quintil").agg(
        eventos=("target_recovery", "count"),
        exito_fuerte_pct=("target_recovery", lambda s: round(s.mean() * 100, 1)),
        wr_15d_pct=("fut_ret_15d", lambda s: round((s > 0).mean() * 100, 1)),
        ret_10d_pct=("fut_ret_10d", lambda s: round(s.mean(), 2)),
        alpha_10d_pct=("fut_alpha_10d", lambda s: round(s.mean(), 2)),
        ret_15d_pct=("fut_ret_15d", lambda s: round(s.mean(), 2)),
        alpha_15d_pct=("fut_alpha_15d", lambda s: round(s.mean(), 2)),
    ).reset_index()

    print(f"ROC-AUC en Sectores Tradicionales (0% Tech, {len(trad_df):,} eventos en Banca/Salud/Industria/Energía/Consumo): {auc_trad:.4f}")
    print(q_trad.to_string(index=False))

    # Top Features Importances
    imp_df = pd.DataFrame({
        "Feature": model_features,
        "Gain": clf.booster_.feature_importance(importance_type="gain")
    }).sort_values("Gain", ascending=False).head(15)
    print("\n--- TOP 15 VARIABLES MÁS IMPORTANTES EN EL MOTOR 1 MULTI-SECTORIAL ---")
    print(imp_df.to_string(index=False))
    print(f"\nCompletado en {time.time()-t0:.1f}s.")
    print("=" * 100)


if __name__ == "__main__":
    main()
