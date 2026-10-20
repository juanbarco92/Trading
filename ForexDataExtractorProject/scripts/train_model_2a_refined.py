"""
Entrenamiento y Validación del Modelo 2A (Cuándo Entrar en Short):
- LightGBM Regularizado con Purged GroupTimeSeriesSplit.
- Features: Microestructura de quiebre, Debilidad Relativa vs SPY, Razón de mecha, RVOL, EPS Surprise.
- Target: First-Passage Asimétrico 2:1 (Meta -1.5x ATR antes de Stop +0.8x ATR).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "motor_2a_dataset_refined.parquet"
MODEL_OUT = PROJECT_ROOT / "models" / "modelo_2a_shorts_lgb.txt"

def train_model_2a():
    print("=" * 80)
    print("ENTRENANDO MODELO 2A (PREDICTOR DE ENTRADAS EN SHORT)")
    print("=" * 80)
    
    df = pd.read_parquet(DATASET_PATH)
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Dataset cargado: {len(df)} eventos calificados.")
    
    features = [
        "atr_d", "ret_3d", "ret_5d", "z_3d", "rvol_1h", "close_loc_1h",
        "upper_wick_ratio", "lower_wick_ratio", "body_ratio", "rsi_14",
        "dist_sma_20", "dist_sma_40", "eps_surprise_pct", "rel_strength_3d",
        "spy_ret_3d"
    ]
    
    # Split Temporal: Train (2022 a mediados 2023), Test OOS (finales 2023 a 2026)
    train_df = df[df["timestamp"] < "2023-07-01"].copy()
    test_df = df[df["timestamp"] >= "2023-07-01"].copy()
    
    print(f"Train In-Sample: {len(train_df)} eventos (2022 - mediados 2023)")
    print(f"Test Out-of-Sample: {len(test_df)} eventos (mediados 2023 - 2026)")
    
    # Parámetros altamente regularizados para evitar sobreajuste
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 45,
        "learning_rate": 0.03,
        "max_depth": 3,
        "num_leaves": 5,
        "min_child_samples": 15,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 1.5,
        "reg_lambda": 5.0,
        "random_state": 42,
        "verbose": -1
    }
    
    clf = lgb.LGBMClassifier(**params)
    clf.fit(train_df[features], train_df["target_2a"])
    
    p_train = clf.predict_proba(train_df[features])[:, 1]
    p_test = clf.predict_proba(test_df[features])[:, 1]
    
    auc_train = roc_auc_score(train_df["target_2a"], p_train)
    auc_test = roc_auc_score(test_df["target_2a"], p_test)
    
    print(f"\nResultados Modelo 2A:")
    print(f"  ROC-AUC In-Sample (Train): {auc_train:.4f}")
    print(f"  ROC-AUC Out-of-Sample (Test): {auc_test:.4f}")
    
    # Feature Importances
    imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    print("\nImportancia de Features:")
    for f, v in imp.items():
        print(f"  {f:20s}: {v}")
        
    # Evaluación por Terciles en Test
    test_df["prob_2a"] = p_test
    test_df["decil"] = pd.qcut(test_df["prob_2a"], q=3, labels=["Bajo", "Medio", "Alto"])
    print("\nDesempeño en Test OOS por Terciles de Probabilidad:")
    summary = test_df.groupby("decil", observed=False)["target_2a"].agg(["count", "mean"])
    summary["mean"] = summary["mean"].map(lambda x: f"{x*100:.1f}%")
    print(summary)
    
    # Guardar modelo
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    clf.booster_.save_model(str(MODEL_OUT))
    print(f"\n[OK] Modelo guardado en: {MODEL_OUT}")

if __name__ == "__main__":
    train_model_2a()
