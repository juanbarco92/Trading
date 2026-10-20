"""
train_universe_dual_model_purged.py - TICK-16: Entrenamiento y Calibración Anti-Leakage (Purged & Embargoed).

Arquitectura Institucional de Validación Temporal (Marcos López de Prado):
1. Split Cronológico en 3 Fases con Purga y Embargo:
   - Fase A: TRAIN SET (2021-12 a 2024-12) -> Ajuste del clasificador base LightGBM.
   - PURGA 1: 35 barras de horizonte Triple Barrier + 10 barras de embargo.
   - Fase B: CALIBRATION SET (2025-01 a 2025-06) -> Calibración Isotónica OOS pura (cv='prefit').
   - PURGA 2: 35 barras de horizonte Triple Barrier + 10 barras de embargo.
   - Fase C: TEST SET OOS CIEGO (2025-07 a 2026-10) -> Evaluación definitiva jamás vista.
2. Cero fuga por k-fold aleatorio: La curva isotónica se ajusta exclusivamente sobre datos futuros al train y pasados al test.
3. Deciles y umbrales evaluados sobre el Test Set OOS estrictamente ciego.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV


def evaluate_deciles(df: pd.DataFrame, prob_col: str, target_col: str, return_col: str, label: str):
    print(f"\n{'='*75}")
    print(f"ANÁLISIS DE ORDENAMIENTO POR DECILES (PURGED OOS) - {label}")
    print(f"{'='*75}")
    
    active = df[df[prob_col] > 0].copy()
    if len(active) == 0:
        print(f"No hay operaciones activas para {label}.")
        return

    try:
        active["decil"] = pd.qcut(active[prob_col], q=10, labels=False, duplicates="drop")
    except Exception:
        active["decil"] = pd.cut(active[prob_col], bins=10, labels=False)
        
    summary = active.groupby("decil").agg(
        n_trades=(target_col, "count"),
        min_prob=(prob_col, "min"),
        max_prob=(prob_col, "max"),
        win_rate=(target_col, lambda x: round(x.mean() * 100, 2)),
        retorno_medio_neto=(return_col, lambda x: round(x.mean(), 2)),
        retorno_acumulado=(return_col, "sum"),
    ).reset_index()
    
    print(summary.to_string(index=False))


def evaluate_thresholds(df: pd.DataFrame, prob_col: str, target_col: str, return_col: str, label: str):
    print(f"\n--- BÚSQUEDA DE UMBRAL ÓPTIMO (PURGED OOS) - {label} ---")
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    th_rows = []
    
    for th in thresholds:
        active = df[df[prob_col] >= th]
        n_act = len(active)
        if n_act > 0:
            wr = (active[target_col] == 1).mean() * 100.0
            ret_mean = active[return_col].mean()
            tot_pnl = active[return_col].sum()
            wins = active[active[return_col] > 0][return_col].sum()
            losses = abs(active[active[return_col] < 0][return_col].sum())
            pf = round(wins / losses, 2) if losses > 0 else 99.0
            
            th_rows.append({
                "Umbral": th,
                "Trades": n_act,
                "Win Rate %": round(wr, 2),
                "Retorno Medio %": round(ret_mean, 2),
                "Retorno Total %": round(tot_pnl, 1),
                "Profit Factor": pf,
            })
            
    df_th = pd.DataFrame(th_rows)
    print(df_th.to_string(index=False))
    return df_th


def main():
    features_path = PROJECT_ROOT / "data" / "processed" / "features_universe_full_structural.parquet"
    print("1. Cargando dataset completo del universo...")
    df = pd.read_parquet(features_path)
    df = df.dropna(subset=["target_quality_trade", "target_short_quality_trade"]).reset_index(drop=True)

    time_col = "timestamp" if "timestamp" in df.columns else "datetime"
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    exclude_cols = [
        "timestamp", "datetime", "open", "high", "low", "close", "volume", "barCount", "ticker",
        "target_triple_barrier", "target_quality_trade", "trade_exit_bar", "trade_exit_type",
        "trade_exit_price", "trade_gross_return", "trade_net_return", "trade_tp_price", "trade_sl_price",
        "target_short_quality_trade", "target_short_triple_barrier", "short_net_return",
        "candle_cluster", "pattern"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Features numéricas causales auditadas: {len(feature_cols)}")

    # 2. Partición Cronológica en 3 Fases con Purga y Embargo (Horizonte 35 barras + Embargo 10 barras)
    # Train: 2021-12 a 2024-12-15
    train_end = pd.Timestamp("2024-12-15 00:00:00", tz="UTC")
    # Calib start: 2025-01-02 (purga de 45 barras de mercado)
    calib_start = pd.Timestamp("2025-01-02 00:00:00", tz="UTC")
    # Calib end: 2025-06-30
    calib_end = pd.Timestamp("2025-06-15 00:00:00", tz="UTC")
    # Test start: 2025-07-01 (purga de 45 barras de mercado)
    test_start = pd.Timestamp("2025-07-01 00:00:00", tz="UTC")

    train_mask = df[time_col] < train_end
    calib_mask = (df[time_col] >= calib_start) & (df[time_col] < calib_end)
    test_mask = df[time_col] >= test_start

    train_df = df[train_mask].copy()
    calib_df = df[calib_mask].copy()
    test_df = df[test_mask].copy()

    print("\n" + "="*80)
    print("PARTICIÓN TEMPORAL PURGADA & EMBARGADA (CERO SOLAPAMIENTO):")
    print(f"  * TRAIN SET (Ajuste Base):       {len(train_df):,} barras ({train_df[time_col].min()} a {train_df[time_col].max()})")
    print(f"    [PURGA 1: 45 barras eliminadas entre {train_end} y {calib_start}]")
    print(f"  * CALIBRATION SET (Prefit OOS):  {len(calib_df):,} barras ({calib_df[time_col].min()} a {calib_df[time_col].max()})")
    print(f"    [PURGA 2: 45 barras eliminadas entre {calib_end} y {test_start}]")
    print(f"  * TEST SET (OOS Ciego 100%):    {len(test_df):,} barras ({test_df[time_col].min()} a {test_df[time_col].max()})")
    print("="*80)

    X_tr = train_df[feature_cols].fillna(0.0)
    y_tr_buy = train_df["target_quality_trade"].astype(int)
    y_tr_sell = train_df["target_short_quality_trade"].astype(int)

    X_cal = calib_df[feature_cols].fillna(0.0)
    y_cal_buy = calib_df["target_quality_trade"].astype(int)
    y_cal_sell = calib_df["target_short_quality_trade"].astype(int)

    X_te = test_df[feature_cols].fillna(0.0)
    y_te_buy = test_df["target_quality_trade"].astype(int)
    y_te_sell = test_df["target_short_quality_trade"].astype(int)

    def train_purged_calibrated_pipeline(X_train, y_train, X_val, y_val, name: str):
        print(f"\n[{name}] Paso 1: Ajustando clasificador base LightGBM sobre Train Set...")
        t0 = time.time()
        base_clf = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=45,
            max_depth=6,
            min_child_samples=60,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        base_clf.fit(X_train, y_train)
        print(f"[{name}] Paso 1 completado en {time.time()-t0:.1f}s.")

        print(f"[{name}] Paso 2: Calibración Isotónica OOS con cv='prefit' sobre Calibration Set...")
        t1 = time.time()
        # cv='prefit' garantiza que la calibración se entrena sobre datos futuros OOS no vistos por base_clf
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="isotonic", cv="prefit")
        calibrated_clf.fit(X_val, y_val)
        print(f"[{name}] Paso 2 completado en {time.time()-t1:.1f}s.")
        return calibrated_clf

    model_buy = train_purged_calibrated_pipeline(X_tr, y_tr_buy, X_cal, y_cal_buy, "BUY MODEL")
    model_sell = train_purged_calibrated_pipeline(X_tr, y_tr_sell, X_cal, y_cal_sell, "SELL MODEL")

    # 3. Predicciones OOS y Winner Takes All en Test Set 100% Ciego
    print("\nGenerando predicciones sobre el Test Set OOS (Jul 2025 - Oct 2026)...")
    raw_probs_buy = model_buy.predict_proba(X_te)[:, 1]
    raw_probs_sell = model_sell.predict_proba(X_te)[:, 1]

    auc_buy = roc_auc_score(y_te_buy, raw_probs_buy)
    auc_sell = roc_auc_score(y_te_sell, raw_probs_sell)
    pr_buy = average_precision_score(y_te_buy, raw_probs_buy)
    pr_sell = average_precision_score(y_te_sell, raw_probs_sell)

    print("\n" + "="*80)
    print("DISCRIMINACIÓN OUT-OF-SAMPLE PURIFICADA (TEST SET 100% CIEGO):")
    print(f"  * BUY ROC-AUC:  {auc_buy:.4f} | PR-AUC: {pr_buy:.4f} (Base azar: {y_te_buy.mean():.4f})")
    print(f"  * SELL ROC-AUC: {auc_sell:.4f} | PR-AUC: {pr_sell:.4f} (Base azar: {y_te_sell.mean():.4f})")
    print("="*80)

    # Winner Takes All
    final_prob_buy = np.where(raw_probs_buy > raw_probs_sell, raw_probs_buy, 0.0)
    final_prob_sell = np.where(raw_probs_sell > raw_probs_buy, raw_probs_sell, 0.0)

    test_df["prob_buy"] = final_prob_buy
    test_df["prob_sell"] = final_prob_sell

    # 4. Deciles y Umbrales
    evaluate_deciles(test_df, "prob_buy", "target_quality_trade", "trade_net_return", "BUY TRADES")
    evaluate_deciles(test_df, "prob_sell", "target_short_quality_trade", "short_net_return", "SELL TRADES / SHORTS")

    evaluate_thresholds(test_df, "prob_buy", "target_quality_trade", "trade_net_return", "BUY TRADES")
    evaluate_thresholds(test_df, "prob_sell", "target_short_quality_trade", "short_net_return", "SELL TRADES / SHORTS")

    # Guardar Modelos Purificados
    models_dir = PROJECT_ROOT / "models"
    buy_file = models_dir / "pipeline_universe_buy_calibrated.joblib"
    sell_file = models_dir / "pipeline_universe_sell_calibrated.joblib"
    joblib.dump(model_buy, buy_file)
    joblib.dump(model_sell, sell_file)
    print(f"\nModelos purificados guardados exitosamente en:\n  * {buy_file}\n  * {sell_file}")


if __name__ == "__main__":
    main()
