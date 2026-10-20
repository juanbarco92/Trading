"""
Entrenamiento y Validación Out-of-Sample del Modelo 2A (Cuándo Entrar en Short):
- Modelo: LightGBM Classifier optimizado para precisión en venta en corto.
- Validación Temporal Estricta:
  * Train: Todo lo ocurrido antes de Noviembre 2025 (con Purge Gap de 21 días).
  * Test OOS: Noviembre 2025 a Octubre 2026.
- Mide ROC-AUC OOS, Feature Importance y calibración de probabilidades.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "motor_2a_dataset_shorts.parquet"

def main():
    print("=" * 80)
    print("ENTRENAMIENTO Y VALIDACIÓN OUT-OF-SAMPLE DEL MODELO 2A (SHORTS)")
    print("=" * 80)

    df = pd.read_parquet(DATASET_PATH)
    print(f"Total eventos cargados: {len(df):,}")
    print(f"Tasa base global Target=1: {df['target_2a'].mean()*100:.1f}%\n")

    # Features predictivas del Modelo 2A
    features = [
        "atr_d", "ret_3d", "ret_5d", "z_3d", "rvol_1h",
        "close_loc_1h", "upper_wick_ratio", "lower_wick_ratio", "body_ratio",
        "rsi_14", "dist_sma_20", "dist_sma_40", "dist_sma_100",
        "eps_surprise_pct", "spy_under_sma100", "spy_ret_3d", "archetype_code"
    ]

    # Partición temporal estricta idéntica a Motor 1
    test_start_ts = pd.Timestamp("2025-11-17", tz="UTC")
    train_end_ts = test_start_ts - pd.Timedelta(days=22) # Purge gap de 21 días

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    tr = df[df["timestamp"] <= train_end_ts].copy().reset_index(drop=True)
    te = df[df["timestamp"] >= test_start_ts].copy().reset_index(drop=True)

    print(f"Set de Entrenamiento (Train): {len(tr):,} eventos ({tr['timestamp'].min().date()} a {tr['timestamp'].max().date()})")
    print(f"Set Fuera de Muestra (Test OOS): {len(te):,} eventos ({te['timestamp'].min().date()} a {te['timestamp'].max().date()})")
    print(f"Tasa base en Train: {tr['target_2a'].mean()*100:.1f}% | Tasa base en Test OOS: {te['target_2a'].mean()*100:.1f}%\n")

    # Entrenar LightGBM con regularización fuerte para evitar sobreajuste
    clf_2a = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=8,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=8.0,
        random_state=42,
        verbose=-1
    )

    clf_2a.fit(tr[features].fillna(0.0), tr["target_2a"].values)

    # Predicción fuera de muestra
    te["prob_2a"] = clf_2a.predict_proba(te[features].fillna(0.0))[:, 1]
    tr["prob_2a"] = clf_2a.predict_proba(tr[features].fillna(0.0))[:, 1]

    auc_tr = roc_auc_score(tr["target_2a"], tr["prob_2a"])
    auc_te = roc_auc_score(te["target_2a"], te["prob_2a"])

    print("=" * 80)
    print("PODER PREDICTIVO DEL MODELO 2A:")
    print(f"  * ROC-AUC en Entrenamiento (In-Sample):    {auc_tr:.4f}")
    print(f"  * ROC-AUC Fuera de Muestra (OOS Test):     {auc_te:.4f}")
    print("=" * 80)

    # Calibración por Deciles en Test OOS
    te["decil"] = pd.qcut(te["prob_2a"], q=5, labels=["Q1_MuyBajo", "Q2_Bajo", "Q3_Medio", "Q4_Alto", "Q5_TopConviccion"])
    print("\nCALIBRACIÓN FUERA DE MUESTRA POR QUINTILES (TEST OOS):")
    dec_stats = []
    for dec, gdf in te.groupby("decil"):
        dec_stats.append({
            "Quintil": dec,
            "Casos": len(gdf),
            "Prob_Media": f"{gdf['prob_2a'].mean()*100:.1f}%",
            "Tasa_Exito_Real_Target1": f"{gdf['target_2a'].mean()*100:.1f}%"
        })
    print(pd.DataFrame(dec_stats).to_string(index=False))

    # Importancia de Variables
    imp = pd.DataFrame({
        "Feature": features,
        "Importancia_Gain": clf_2a.booster_.feature_importance(importance_type="gain")
    }).sort_values("Importancia_Gain", ascending=False)
    print("\nTOP 10 FEATURES MÁS DETERMINANTES DEL MODELO 2A:")
    print(imp.head(10).to_string(index=False))

    # Guardar predicciones para el simulador de salidas 2B
    te.to_parquet(PROJECT_ROOT / "data" / "processed" / "motor_2a_predictions_test.parquet", index=False)
    print("\n[OK] Predicciones OOS guardadas en data/processed/motor_2a_predictions_test.parquet")

if __name__ == "__main__":
    main()
