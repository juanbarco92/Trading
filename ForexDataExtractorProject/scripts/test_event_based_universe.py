"""
test_event_based_universe.py - Experimento Cuantitativo:
¿Qué pasa si en lugar de entrenar sobre 172,000 velas continuas de ruido,
entrenamos EXCLUSIVAMENTE sobre eventos específicos de alta convicción institucional?

Compara:
1. Universo Ruidoso Continuo (Todas las velas).
2. Universo Filtrado por Eventos:
   - Setup Dip-Buying / Rebote en Pánico (El patrón de MeLi):
     Tendencia alcista (Close > SMA100), retroceso a soporte (RSI < 45 o cerca de soporte),
     y vela con mecha de absorción / rechazo inferior.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV

print("Cargando features estructurales causales...")
df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features_universe_full_structural.parquet")
df = df.dropna(subset=["target_quality_trade", "target_short_quality_trade"]).reset_index(drop=True)

time_col = "timestamp" if "timestamp" in df.columns else "datetime"
df[time_col] = pd.to_datetime(df[time_col])
df = df.sort_values(time_col).reset_index(drop=True)

# 1. Definir la Condición Estructural del Evento "Dip-Buying en Tendencia Alcista" (Patrón MeLi)
# - Tendencia de fondo alcista: Close > SMA_100
# - Retroceso: RSI < 48 o distancia a soporte corta
# - Vela de absorción / mecha de rechazo: lower_wick_ratio > 0.25 y close > open (o rebote desde soporte)
is_uptrend = (df["close"] > df["SMA_100"])
is_pullback = (df["RSI_14"] < 48) | (df["sr_dist_support_pct"] < 1.5)
has_rejection = (df["lower_wick_ratio"] > 0.20) | (df["sr_bounce_sup"] == 1.0)

df["is_event_setup"] = (is_uptrend & is_pullback & has_rejection).astype(int)

print(f"\nTotal velas en el universo: {len(df):,}")
print(f"Velas que cumplen el Setup de Evento Institucional: {df['is_event_setup'].sum():,} ({df['is_event_setup'].mean()*100:.2f}%)")

# Tasa base sin modelo en todo el universo vs en el evento filtrado
base_rate_all = df["target_quality_trade"].mean() * 100.0
base_rate_event = df[df["is_event_setup"] == 1]["target_quality_trade"].mean() * 100.0

print(f"\nTASA BASE DE ÉXITO NATURAL (Take Profit 3.5x ATR antes de Stop Loss 1.8x ATR):")
print(f"  * En todas las velas continuas (Ruido):     {base_rate_all:.2f}%")
print(f"  * En las velas de Setup Filtrado (MeLi Dip): {base_rate_event:.2f}% (Ventaja natural de {base_rate_event - base_rate_all:+.2f}%)")

# 2. Split Cronológico Purgado sobre el Universo Filtrado de Eventos
df_event = df[df["is_event_setup"] == 1].copy().reset_index(drop=True)

train_end = pd.Timestamp("2024-12-15 00:00:00", tz="UTC")
calib_start = pd.Timestamp("2025-01-02 00:00:00", tz="UTC")
calib_end = pd.Timestamp("2025-06-15 00:00:00", tz="UTC")
test_start = pd.Timestamp("2025-07-01 00:00:00", tz="UTC")

train_df = df_event[df_event[time_col] < train_end].copy()
calib_df = df_event[(df_event[time_col] >= calib_start) & (df_event[time_col] < calib_end)].copy()
test_df = df_event[df_event[time_col] >= test_start].copy()

exclude_cols = [
    "timestamp", "datetime", "open", "high", "low", "close", "volume", "barCount", "ticker",
    "target_triple_barrier", "target_quality_trade", "trade_exit_bar", "trade_exit_type",
    "trade_exit_price", "trade_gross_return", "trade_net_return", "trade_tp_price", "trade_sl_price",
    "target_short_quality_trade", "target_short_triple_barrier", "short_net_return",
    "candle_cluster", "pattern", "is_event_setup"
]
feature_cols = [c for c in df_event.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_event[c])]

X_tr = train_df[feature_cols].fillna(0.0)
y_tr = train_df["target_quality_trade"].astype(int)

X_cal = calib_df[feature_cols].fillna(0.0)
y_cal = calib_df["target_quality_trade"].astype(int)

X_te = test_df[feature_cols].fillna(0.0)
y_te = test_df["target_quality_trade"].astype(int)

print(f"\nPartición Purgada sobre Eventos Filtrados:")
print(f"  * Train: {len(X_tr):,} eventos | Tasa positiva: {y_tr.mean()*100:.2f}%")
print(f"  * Calib: {len(X_cal):,} eventos | Tasa positiva: {y_cal.mean()*100:.2f}%")
print(f"  * Test:  {len(X_te):,} eventos | Tasa positiva: {y_te.mean()*100:.2f}%")

# Entrenar LightGBM enfocado en filtrar falsos rebotes
base_clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=25,
    max_depth=5,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.6,
    random_state=42,
    verbose=-1
)
base_clf.fit(X_tr, y_tr)

# Calibración Sigmoide
calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv="prefit")
calibrated_clf.fit(X_cal, y_cal)

probs_te = calibrated_clf.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, probs_te)
pr_auc = average_precision_score(y_te, probs_te)

print("\n" + "="*80)
print(f"DISCRIMINACIÓN OUT-OF-SAMPLE ENTRENANDO SOLO EN EL UNIVERSO DE EVENTOS:")
print(f"  * ROC-AUC OOS: {auc:.4f}")
print(f"  * PR-AUC OOS:  {pr_auc:.4f} (Base azar: {y_te.mean():.4f})")
print("="*80)

# Deciles sobre el universo de eventos
test_df["prob"] = probs_te
test_df["decil"] = pd.qcut(test_df["prob"], q=5, labels=False, duplicates="drop")

summary = test_df.groupby("decil").agg(
    n_trades=("target_quality_trade", "count"),
    min_prob=("prob", "min"),
    max_prob=("prob", "max"),
    win_rate=("target_quality_trade", lambda x: round(x.mean() * 100, 2)),
    retorno_medio_neto=("trade_net_return", lambda x: round(x.mean(), 2)),
    retorno_acumulado=("trade_net_return", "sum"),
).reset_index()

print("\n--- QUINTILES DE PREDICCIÓN SOBRE EL UNIVERSO DE EVENTOS (PURGED OOS) ---")
print(summary.to_string(index=False))
