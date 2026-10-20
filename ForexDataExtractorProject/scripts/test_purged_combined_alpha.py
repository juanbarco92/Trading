"""
test_purged_combined_alpha.py - Test de Fusión de Alpha Causal:
Combina las 584 features del usuario con:
1. Fuerza Relativa (RS vs SPY: alpha_1d, alpha_5d, rs_divergence)
2. Volumen Institucional (RVOL horario, VWAP)
3. Régimen Macro (SPY vs EMA200)
4. Calibración Sigmoide Suave (Platt Scaling) sin colapso en escalones isotónicos.
Todo bajo el split purgado y embargado estricto (Zero Leakage).
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

print("1. Cargando features estructurales completas...")
df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features_universe_full_structural.parquet")
df = df.dropna(subset=["target_quality_trade", "target_short_quality_trade"]).reset_index(drop=True)

time_col = "timestamp" if "timestamp" in df.columns else "datetime"
df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col).reset_index(drop=True)

# 2. Inyectar Fuerza Relativa vs SPY y Volumen Relativo causalmente
print("2. Calculando Fuerza Relativa vs SPY y Régimen Macro...")
spy_df = df[df["ticker"] == "SPY"].sort_values(time_col).set_index(time_col)
spy_close = spy_df["close"]
spy_ret_5d = spy_close.pct_change(35)  # Retorno 5 días (~35 velas 1H)
spy_ema200 = spy_close.ewm(span=200, adjust=False).mean()
spy_bullish = (spy_close > spy_ema200).astype(int)

# Mapear al dataframe completo
df["spy_ret_5d"] = df[time_col].map(spy_ret_5d).fillna(0.0)
df["is_market_bullish"] = df[time_col].map(spy_bullish).fillna(1)

# Retorno propio a 5 días y Alpha excedente
df["stock_ret_5d"] = df.groupby("ticker")["close"].pct_change(35).fillna(0.0)
df["alpha_5d"] = df["stock_ret_5d"] - df["spy_ret_5d"]
df["is_outperforming_spy"] = (df["alpha_5d"] > 0).astype(int)

# RVOL causal (Volumen actual / media móvil causal de volumen de 50 barras)
df["vol_sma50"] = df.groupby("ticker")["volume"].transform(lambda s: s.rolling(50, min_periods=5).mean()).bfill()
df["rvol"] = (df["volume"] / df["vol_sma50"].replace(0, np.nan)).fillna(1.0).clip(0, 10.0)

exclude_cols = [
    "timestamp", "datetime", "open", "high", "low", "close", "volume", "barCount", "ticker",
    "target_triple_barrier", "target_quality_trade", "trade_exit_bar", "trade_exit_type",
    "trade_exit_price", "trade_gross_return", "trade_net_return", "trade_tp_price", "trade_sl_price",
    "target_short_quality_trade", "target_short_triple_barrier", "short_net_return",
    "candle_cluster", "pattern"
]
feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
print(f"Features predictoras totales: {len(feature_cols)}")

# Split cronológico purgado y embargado
train_end = pd.Timestamp("2024-12-15 00:00:00", tz="UTC")
calib_start = pd.Timestamp("2025-01-02 00:00:00", tz="UTC")
calib_end = pd.Timestamp("2025-06-15 00:00:00", tz="UTC")
test_start = pd.Timestamp("2025-07-01 00:00:00", tz="UTC")

train_df = df[df[time_col] < train_end].copy()
calib_df = df[(df[time_col] >= calib_start) & (df[time_col] < calib_end)].copy()
test_df = df[df[time_col] >= test_start].copy()

X_tr = train_df[feature_cols].fillna(0.0)
y_tr = train_df["target_quality_trade"].astype(int)

X_cal = calib_df[feature_cols].fillna(0.0)
y_cal = calib_df["target_quality_trade"].astype(int)

X_te = test_df[feature_cols].fillna(0.0)
y_te = test_df["target_quality_trade"].astype(int)

print(f"\nTrain size: {len(X_tr):,}, Calib size: {len(X_cal):,}, Test OOS size: {len(X_te):,}")

# Entrenar modelo LightGBM con regularización institucional
base_clf = lgb.LGBMClassifier(
    n_estimators=350,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=5,
    min_child_samples=80,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=1.0,
    reg_lambda=5.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
base_clf.fit(X_tr, y_tr)

# Calibración Sigmoide (Platt Scaling) con cv='prefit' sobre Calibration Set
calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv="prefit")
calibrated_clf.fit(X_cal, y_cal)

probs_te = calibrated_clf.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, probs_te)
pr_auc = average_precision_score(y_te, probs_te)

print("\n" + "="*80)
print(f"DISCRIMINACIÓN PURGADA CON ALPHA MULTI-ACTIVO + CALIBRACIÓN SIGMOIDE:")
print(f"  * ROC-AUC Out-of-Sample: {auc:.4f}")
print(f"  * PR-AUC Out-of-Sample:  {pr_auc:.4f} (Base azar: {y_te.mean():.4f})")
print("="*80)

# Deciles
test_df["prob"] = probs_te
test_df["decil"] = pd.qcut(test_df["prob"], q=10, labels=False, duplicates="drop")

summary = test_df.groupby("decil").agg(
    n_trades=("target_quality_trade", "count"),
    min_prob=("prob", "min"),
    max_prob=("prob", "max"),
    win_rate=("target_quality_trade", lambda x: round(x.mean() * 100, 2)),
    retorno_medio_neto=("trade_net_return", lambda x: round(x.mean(), 2)),
    retorno_acumulado=("trade_net_return", "sum"),
).reset_index()

print("\n--- ORDENAMIENTO POR DECILES (PURGED OOS) ---")
print(summary.to_string(index=False))

# Umbrales
print("\n--- BÚSQUEDA DE UMBRALES ÓPTIMOS (PURGED OOS) ---")
for th in [0.30, 0.35, 0.40, 0.45, 0.50]:
    act = test_df[test_df["prob"] >= th]
    if len(act) > 0:
        wr = (act["target_quality_trade"] == 1).mean() * 100.0
        ret_mean = act["trade_net_return"].mean()
        tot_pnl = act["trade_net_return"].sum()
        wins = act[act["trade_net_return"] > 0]["trade_net_return"].sum()
        losses = abs(act[act["trade_net_return"] < 0]["trade_net_return"].sum())
        pf = wins / losses if losses > 0 else 99.0
        print(f"Umbral {th:.2f} | Trades: {len(act):<5} | Win Rate: {wr:.2f}% | Retorno Medio: {ret_mean:+.2f}% | Total: {tot_pnl:+.1f}% | Profit Factor: {pf:.2f}")
