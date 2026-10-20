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
from src.utils.exploration_utils import engineer_features, fractal_engineering
from src.utils.stock_sr import compute_stock_support_resistance_features
from src.models.labeling import generate_triple_barrier_labels

print("1. Cargando datos de NVDA...")
df = pd.read_parquet("data/processed/NVDA_1hour.parquet")

print("2. Generando etiquetas Triple Barrier (TP=3.5x ATR, SL=1.8x ATR, horizon=35)...")
df = generate_triple_barrier_labels(df, k_tp=3.5, k_sl=1.8, horizon_bars=35)
df = df.dropna(subset=["target_quality_trade", "target_short_quality_trade"]).reset_index(drop=True)

print("3. Ejecutando Feature Engineering completo (Velas + Clusters + Fractales + S/R Estructural)...")
# A. Feature engineering de velas y clusters K-Means del usuario
df_eng, _ = engineer_features(df, n_candles=4, n_clusters=6, fit_clusterer=True, random_state=42)
# B. Fractal engineering del usuario
df_eng = fractal_engineering(df_eng)
# C. Soporte y Resistencia estructural
df_eng = compute_stock_support_resistance_features(df_eng)

# Identificar columnas numéricas para el modelo
exclude_cols = [
    'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'barCount', 'ticker',
    'target_triple_barrier', 'target_quality_trade', 'trade_exit_bar', 'trade_exit_type',
    'trade_exit_price', 'trade_gross_return', 'trade_net_return', 'trade_tp_price', 'trade_sl_price',
    'target_short_quality_trade', 'target_short_triple_barrier', 'short_net_return',
    'candle_cluster', 'pattern'
]
feature_cols = [c for c in df_eng.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_eng[c])]
print(f"Features numéricas disponibles para entrenamiento: {len(feature_cols)}")

# Split temporal Train (75%) / Test OOS (25%)
split_idx = int(len(df_eng) * 0.75)
train_df = df_eng.iloc[:split_idx]
test_df = df_eng.iloc[split_idx:]

X_tr = train_df[feature_cols].fillna(0.0)
y_tr_buy = train_df["target_quality_trade"].astype(int)
y_tr_sell = train_df["target_short_quality_trade"].astype(int)

X_te = test_df[feature_cols].fillna(0.0)
y_te_buy = test_df["target_quality_trade"].astype(int)
y_te_sell = test_df["target_short_quality_trade"].astype(int)

print(f"Train size: {len(X_tr)} (Buy positivos: {y_tr_buy.mean()*100:.1f}%, Sell positivos: {y_tr_sell.mean()*100:.1f}%)")
print(f"Test size:  {len(X_te)} (Buy positivos: {y_te_buy.mean()*100:.1f}%, Sell positivos: {y_te_sell.mean()*100:.1f}%)")

def train_calibrated_model(X, y):
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1
    )
    calibrated = CalibratedClassifierCV(clf, method='isotonic', cv=3)
    calibrated.fit(X, y)
    return calibrated

print("\n4. Entrenando modelo BUY calibrado...")
model_buy = train_calibrated_model(X_tr, y_tr_buy)
raw_probs_buy = model_buy.predict_proba(X_te)[:, 1]

print("5. Entrenando modelo SELL calibrado...")
model_sell = train_calibrated_model(X_tr, y_tr_sell)
raw_probs_sell = model_sell.predict_proba(X_te)[:, 1]

# 6. APLICAR LÓGICA 'WINNER TAKES ALL' (Idéntico a notebooks/training-model EURUSD.ipynb)
print("6. Aplicando lógica 'Winner Takes All'...")
final_prob_buy = np.where(raw_probs_buy > raw_probs_sell, raw_probs_buy, 0.0)
final_prob_sell = np.where(raw_probs_sell > raw_probs_buy, raw_probs_sell, 0.0)

auc_buy = roc_auc_score(y_te_buy, raw_probs_buy)
auc_sell = roc_auc_score(y_te_sell, raw_probs_sell)

print("\n" + "="*70)
print(f"RESULTADOS OUT-OF-SAMPLE DUAL-MODEL CON LAS {len(feature_cols)} FEATURES:")
print(f"  * BUY ROC-AUC (Crudo):  {auc_buy:.4f}")
print(f"  * SELL ROC-AUC (Crudo): {auc_sell:.4f}")
print("="*70)

# 7. Evaluación de Deciles y Umbrales para BUY
test_results = test_df.copy()
test_results["prob_buy"] = final_prob_buy
test_results["prob_sell"] = final_prob_sell

print("\n--- ANÁLISIS DE UMBRALES DE UTILIDAD MÁXIMA (BUY TRADES) ---")
thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
th_rows = []
for th in thresholds:
    active = test_results[test_results["prob_buy"] >= th]
    if len(active) > 0:
        wr = (active["target_quality_trade"] == 1).mean() * 100.0
        ret_mean = active["trade_net_return"].mean()
        tot_pnl = active["trade_net_return"].sum()
        th_rows.append({
            "Umbral": th,
            "Trades": len(active),
            "Pct Test %": round((len(active) / len(test_results)) * 100, 1),
            "Win Rate %": round(wr, 2),
            "Retorno Medio %": round(ret_mean, 2),
            "Retorno Acumulado %": round(tot_pnl, 2),
        })
print(pd.DataFrame(th_rows).to_string(index=False))

print("\n--- ANÁLISIS DE UMBRALES DE UTILIDAD MÁXIMA (SELL TRADES / SHORTS) ---")
th_rows_sell = []
for th in thresholds:
    active = test_results[test_results["prob_sell"] >= th]
    if len(active) > 0:
        wr = (active["target_short_quality_trade"] == 1).mean() * 100.0
        ret_mean = active["short_net_return"].mean()
        tot_pnl = active["short_net_return"].sum()
        th_rows_sell.append({
            "Umbral": th,
            "Trades": len(active),
            "Pct Test %": round((len(active) / len(test_results)) * 100, 1),
            "Win Rate %": round(wr, 2),
            "Retorno Medio %": round(ret_mean, 2),
            "Retorno Acumulado %": round(tot_pnl, 2),
        })
print(pd.DataFrame(th_rows_sell).to_string(index=False))
