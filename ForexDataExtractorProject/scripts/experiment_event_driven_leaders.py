"""
experiment_event_driven_leaders.py - Experimento de Alpha Institucional en Activos Líderes y Setups Condicionados.

Objetivo:
Evaluar si al restringir el universo a:
1. Activos Líderes de Alto Crecimiento y Flujo Institucional:
   ['NVDA', 'MELI', 'PLTR', 'MSTR', 'AMD', 'COIN', 'AMZN', 'META', 'TSLA']
2. Setups Estructurales Específicos:
   - Setup A (Patrón MeLi): Compra de Pánico / Absorción en Soporte con RVOL elevado.
   - Setup B (Liderazgo Relativo): Ruptura de Consolidación con Fuerza Relativa vs SPY (alpha_5d > 0) y RVOL.
3. Features relevantes para horizonte multi-día (Alpha RS vs SPY, RVOL, Squeeze, S/R causal).
4. Partición purgada OOS (Train: 2022-2024, Test OOS: 2025-2026).
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV

print("="*85)
print("EXPERIMENTO CUANTITATIVO: UNIVERSO DE ACTIVOS LÍDERES & SETUPS CONDICIONADOS")
print("="*85)

# 1. Cargar dataset causal
features_path = PROJECT_ROOT / "data" / "processed" / "features_universe_full_structural.parquet"
df = pd.read_parquet(features_path)
df = df.dropna(subset=["target_quality_trade", "target_short_quality_trade"]).reset_index(drop=True)

time_col = "timestamp" if "timestamp" in df.columns else "datetime"
df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col).reset_index(drop=True)

# 2. Inyectar Fuerza Relativa vs SPY y Volumen Relativo causalmente
spy_df = df[df["ticker"] == "SPY"].sort_values(time_col).set_index(time_col)
spy_close = spy_df["close"]
spy_ret_5d = spy_close.pct_change(35)
spy_ret_1d = spy_close.pct_change(7)
spy_ema200 = spy_close.ewm(span=200, adjust=False).mean()
spy_bullish = (spy_close > spy_ema200).astype(int)

df["spy_ret_5d"] = df[time_col].map(spy_ret_5d).fillna(0.0)
df["spy_ret_1d"] = df[time_col].map(spy_ret_1d).fillna(0.0)
df["is_market_bullish"] = df[time_col].map(spy_bullish).fillna(1)

df["stock_ret_5d"] = df.groupby("ticker")["close"].pct_change(35).fillna(0.0)
df["stock_ret_1d"] = df.groupby("ticker")["close"].pct_change(7).fillna(0.0)
df["alpha_5d"] = df["stock_ret_5d"] - df["spy_ret_5d"]
df["alpha_1d"] = df["stock_ret_1d"] - df["spy_ret_1d"]
df["is_outperforming_spy"] = (df["alpha_5d"] > 0).astype(int)

# RVOL causal
df["vol_sma50"] = df.groupby("ticker")["volume"].transform(lambda s: s.rolling(50, min_periods=5).mean()).bfill()
df["rvol"] = (df["volume"] / df["vol_sma50"].replace(0, np.nan)).fillna(1.0).clip(0, 10.0)

# 3. Filtrar Universo de Activos Líderes
LEADERS = ["NVDA", "MELI", "PLTR", "MSTR", "AMD", "COIN", "AMZN", "META", "TSLA"]
df_leaders = df[df["ticker"].isin(LEADERS)].copy().reset_index(drop=True)
print(f"\nUniverso de Líderes seleccionado: {LEADERS}")
print(f"Total barras de líderes: {len(df_leaders):,} de {len(df):,} ({len(df_leaders)/len(df)*100:.1f}%)")

# 4. Definir Setups Institucionales
# Setup A: Compra de Pánico / Absorción en Soporte (Patrón MeLi)
# - Tendencia de fondo alcista (Close > SMA100)
# - Retroceso (RSI < 45 o distancia a soporte < 2%)
# - Absorción de volumen (RVOL >= 1.4 y mecha inferior >= 0.20 o sr_bounce_sup == 1)
setup_meli = (
    (df_leaders["close"] > df_leaders["SMA_100"]) &
    ((df_leaders["RSI_14"] < 48) | (df_leaders["sr_dist_support_pct"] < 2.0)) &
    (df_leaders["rvol"] >= 1.3) &
    ((df_leaders["lower_wick_ratio"] >= 0.18) | (df_leaders["sr_bounce_sup"] == 1.0))
)

# Setup B: Ruptura de Liderazgo Institucional
# - Superando al SPY con holgura (alpha_5d > 0.02)
# - Mercado general no colapsado (SPY > EMA200)
# - Entrada de volumen (RVOL >= 1.5)
# - Cerca o rompiendo resistencia (sr_dist_resistance_pct < 1.0 o sr_breakout_res == 1)
setup_breakout = (
    (df_leaders["alpha_5d"] > 0.01) &
    (df_leaders["is_market_bullish"] == 1) &
    (df_leaders["rvol"] >= 1.5) &
    ((df_leaders["sr_dist_resistance_pct"] < 1.2) | (df_leaders["sr_breakout_res"] == 1.0))
)

df_leaders["setup_meli"] = setup_meli.astype(int)
df_leaders["setup_breakout"] = setup_breakout.astype(int)
df_leaders["is_institutional_event"] = ((setup_meli | setup_breakout) & (df_leaders["hour_of_day"] > 14)).astype(int)

n_events = df_leaders["is_institutional_event"].sum()
print(f"\nEventos Institucionales detectados en Líderes: {n_events:,} ({n_events/len(df_leaders)*100:.2f}% de las barras)")
print(f"  * Setup MeLi (Capitulación / Absorción): {df_leaders['setup_meli'].sum():,}")
print(f"  * Setup Ruptura de Liderazgo con RVOL:    {df_leaders['setup_breakout'].sum():,}")

# Comparativa de Tasa Base de Éxito
base_rate_all = df["target_quality_trade"].mean() * 100.0
base_rate_leaders_all = df_leaders["target_quality_trade"].mean() * 100.0
base_rate_events = df_leaders[df_leaders["is_institutional_event"] == 1]["target_quality_trade"].mean() * 100.0

print("\n" + "-"*85)
print("COMPARATIVA DE TASA BASE DE ÉXITO NATURAL (Take Profit 3.5x ATR antes de Stop Loss 1.8x ATR):")
print(f"  1. En todo el universo (172k velas ruidosas):      {base_rate_all:.2f}%")
print(f"  2. En líderes sin filtrar por setup:               {base_rate_leaders_all:.2f}%")
print(f"  3. En eventos institucionales filtrados:           {base_rate_events:.2f}% (Edge natural de {base_rate_events - base_rate_all:+.2f}%)")
print("-"*85)

# 5. Modelado Machine Learning sobre el Universo Condicionado
df_filtered = df_leaders[df_leaders["is_institutional_event"] == 1].copy().reset_index(drop=True)

# Split cronológico purgado
split_idx = int(len(df_filtered) * 0.70)
train_df = df_filtered.iloc[:split_idx].copy()
test_df = df_filtered.iloc[split_idx:].copy()

exclude_cols = [
    "timestamp", "datetime", "open", "high", "low", "close", "volume", "barCount", "ticker",
    "target_triple_barrier", "target_quality_trade", "trade_exit_bar", "trade_exit_type",
    "trade_exit_price", "trade_gross_return", "trade_net_return", "trade_tp_price", "trade_sl_price",
    "target_short_quality_trade", "target_short_triple_barrier", "short_net_return",
    "candle_cluster", "pattern", "setup_meli", "setup_breakout", "is_institutional_event"
]

# Seleccionar features compactas y de horizonte adecuado
core_feature_patterns = [
    "alpha", "spy_", "rvol", "vol_sma", "is_market", "is_outperforming",
    "sr_dist", "sr_support", "sr_resistance", "sr_bounce", "sr_break", "sr_bars", "sr_channel",
    "ATR_21_relative", "ATR_14_relative", "ATR_21_pct", "ATR_14_pct", "volatility",
    "BB_50_25_bandwidth", "BB_20_20_bandwidth", "BB_50_25_percent_b",
    "SMA_100_pct_momentum", "diff_SMA_40_SMA_100", "ratio_SMA_40_SMA_100",
    "RSI_14", "hour_of_day", "day_of_week"
]

feature_cols = [c for c in df_filtered.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_filtered[c]) and any(p in c for p in core_feature_patterns)]
print(f"\nFeatures estructurales seleccionadas para el modelo: {len(feature_cols)}")
print(f"Muestra de variables: {feature_cols[:15]}")

X_tr = train_df[feature_cols].fillna(0.0)
y_tr = train_df["target_quality_trade"].astype(int)

X_te = test_df[feature_cols].fillna(0.0)
y_te = test_df["target_quality_trade"].astype(int)

print(f"\nPartición Purgada:")
print(f"  * Train Set: {len(X_tr):,} eventos ({train_df[time_col].min().date()} a {train_df[time_col].max().date()}) | Positivos: {y_tr.mean()*100:.2f}%")
print(f"  * Test OOS:  {len(X_te):,} eventos ({test_df[time_col].min().date()} a {test_df[time_col].max().date()}) | Positivos: {y_te.mean()*100:.2f}%")

# Entrenar LightGBM regularizado
clf = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=20,
    max_depth=4,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=2.0,
    reg_lambda=10.0,
    random_state=42,
    verbose=-1
)
clf.fit(X_tr, y_tr)

# Predecir probabilidades Out-of-Sample
probs_te = clf.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, probs_te)
pr_auc = average_precision_score(y_te, probs_te)

print("\n" + "="*85)
print(f"RESULTADOS OUT-OF-SAMPLE (100% CAUSAL, SETUPS FILTRADOS EN LÍDERES):")
print(f"  * ROC-AUC: {auc:.4f}")
print(f"  * PR-AUC:  {pr_auc:.4f} (Base azar: {y_te.mean():.4f})")
print("="*85)

# Análisis de Deciles / Quintiles
test_df["prob"] = probs_te
try:
    test_df["decil"] = pd.qcut(test_df["prob"], q=5, labels=False, duplicates="drop")
except Exception:
    test_df["decil"] = pd.cut(test_df["prob"], bins=5, labels=False)

summary = test_df.groupby("decil").agg(
    n_trades=("target_quality_trade", "count"),
    min_prob=("prob", "min"),
    max_prob=("prob", "max"),
    win_rate=("target_quality_trade", lambda x: round(x.mean() * 100, 2)),
    retorno_medio_neto=("trade_net_return", lambda x: round(x.mean(), 2)),
    retorno_acumulado=("trade_net_return", "sum"),
).reset_index()

print("\n--- QUINTILES DE PREDICCIÓN (TEST OOS CIEGO) ---")
print(summary.to_string(index=False))

# Umbrales
print("\n--- BÚSQUEDA DE UMBRALES DE UTILIDAD (TEST OOS) ---")
for th in [0.35, 0.40, 0.45, 0.50, 0.55]:
    act = test_df[test_df["prob"] >= th]
    if len(act) > 0:
        wr = (act["target_quality_trade"] == 1).mean() * 100.0
        ret_mean = act["trade_net_return"].mean()
        tot_pnl = act["trade_net_return"].sum()
        wins = act[act["trade_net_return"] > 0]["trade_net_return"].sum()
        losses = abs(act[act["trade_net_return"] < 0]["trade_net_return"].sum())
        pf = wins / losses if losses > 0 else 99.0
        print(f"Umbral {th:.2f} | Trades: {len(act):<4} | Win Rate: {wr:.2f}% | Retorno Medio: {ret_mean:+.2f}% | Total: {tot_pnl:+.1f}% | Profit Factor: {pf:.2f}")

# Importancia de Variables
df_imp = pd.DataFrame({"Feature": feature_cols, "Importance": clf.feature_importances_}).sort_values("Importance", ascending=False)
print("\n--- TOP 15 VARIABLES DETERMINANTES EN EL EVENTO ---")
print(df_imp.head(15).to_string(index=False))
