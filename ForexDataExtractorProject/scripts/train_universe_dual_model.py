"""
train_universe_dual_model.py - Entrenamiento Dual Calibrado (Buy & Sell) Multi-Activo.

Implementa la arquitectura institucional inspirada en notebooks/training-model EURUSD.ipynb:
1. Dataset completo de 21 activos con 580+ features del usuario (velas, clusters, fractales, S/R, régimen).
2. Split temporal estrictamente cronológico OOS (Train: 2022-2024, Test OOS: 2025-2026).
3. Modelos LightGBM con CalibratedClassifierCV(method='isotonic', cv=3).
4. Lógica 'Winner Takes All' (WTA) para resolución de conflictos Long vs Short.
5. Ordenamiento de deciles y búsqueda de umbral óptimo de utilidad máxima.
6. Guardado de artefactos entrenados en models/.
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
    print(f"ANÁLISIS DE ORDENAMIENTO POR DECILES - {label}")
    print(f"{'='*75}")
    
    # Filtrar solo casos donde prob > 0 (afectados por Winner Takes All)
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
    print(f"\n--- BÚSQUEDA DE UMBRAL ÓPTIMO (MAX UTILITY) - {label} ---")
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    th_rows = []
    
    for th in thresholds:
        active = df[df[prob_col] >= th]
        n_act = len(active)
        if n_act > 0:
            wr = (active[target_col] == 1).mean() * 100.0
            ret_mean = active[return_col].mean()
            tot_pnl = active[return_col].sum()
            # Profit factor aproximado
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


def extract_calibrated_importances(calibrated_clf, feature_names):
    """Extrae las importancias promediadas a través de los folds de CalibratedClassifierCV."""
    imps = []
    for cal in calibrated_clf.calibrated_classifiers_:
        estimator = getattr(cal, "estimator", getattr(cal, "base_estimator", None))
        if estimator and hasattr(estimator, "feature_importances_"):
            imps.append(estimator.feature_importances_)
    if imps:
        avg_imp = np.mean(imps, axis=0)
        df_imp = pd.DataFrame({"Feature": feature_names, "Importance": avg_imp})
        return df_imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    return pd.DataFrame()


def main():
    features_path = PROJECT_ROOT / "data" / "processed" / "features_universe_full_structural.parquet"
    if not features_path.exists():
        print(f"Error: {features_path} no existe aún. Debe completarse la generación de features.")
        return

    print("1. Cargando dataset completo del universo...")
    df = pd.read_parquet(features_path)
    print(f"Filas cargadas: {len(df):,}, Columnas: {len(df.columns)}")

    # Filtrar etiquetas válidas (sin nan en targets)
    df = df.dropna(subset=["target_quality_trade", "target_short_quality_trade"]).reset_index(drop=True)
    print(f"Filas con etiquetas completas: {len(df):,}")

    # Columnas a excluir del modelado
    exclude_cols = [
        "timestamp", "datetime", "open", "high", "low", "close", "volume", "barCount", "ticker",
        "target_triple_barrier", "target_quality_trade", "trade_exit_bar", "trade_exit_type",
        "trade_exit_price", "trade_gross_return", "trade_net_return", "trade_tp_price", "trade_sl_price",
        "target_short_quality_trade", "target_short_triple_barrier", "short_net_return",
        "candle_cluster", "pattern"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Features numéricas disponibles para modelado: {len(feature_cols)}")

    # 2. Split Temporal Estrictamente Cronológico OOS
    # Usamos timestamp para ordenar y dividir sin ninguna fuga temporal
    time_col = "timestamp" if "timestamp" in df.columns else "datetime"
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    
    unique_times = df[time_col].drop_duplicates().sort_values().reset_index(drop=True)
    split_time_idx = int(len(unique_times) * 0.75)
    split_cutoff_date = unique_times.iloc[split_time_idx]
    
    train_mask = df[time_col] < split_cutoff_date
    test_mask = df[time_col] >= split_cutoff_date
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print("\n" + "="*75)
    print(f"SPLIT CRONOLÓGICO OUT-OF-SAMPLE:")
    print(f"  * Fecha de corte OOS:  {split_cutoff_date}")
    print(f"  * Train set: {len(train_df):,} barras ({train_df[time_col].min()} a {train_df[time_col].max()})")
    print(f"  * Test set:  {len(test_df):,} barras ({test_df[time_col].min()} a {test_df[time_col].max()})")
    print(f"  * Tasa positiva Buy  (Train): {train_df['target_quality_trade'].mean()*100:.2f}%")
    print(f"  * Tasa positiva Sell (Train): {train_df['target_short_quality_trade'].mean()*100:.2f}%")
    print(f"  * Tasa positiva Buy  (Test):  {test_df['target_quality_trade'].mean()*100:.2f}%")
    print(f"  * Tasa positiva Sell (Test):  {test_df['target_short_quality_trade'].mean()*100:.2f}%")
    print("="*75)

    X_tr = train_df[feature_cols].fillna(0.0)
    y_tr_buy = train_df["target_quality_trade"].astype(int)
    y_tr_sell = train_df["target_short_quality_trade"].astype(int)

    X_te = test_df[feature_cols].fillna(0.0)
    y_te_buy = test_df["target_quality_trade"].astype(int)
    y_te_sell = test_df["target_short_quality_trade"].astype(int)

    # 3. Modelos LightGBM con CalibratedClassifierCV
    def create_and_train_model(X, y, name: str):
        print(f"\nEntrenando clasificador LightGBM + Isotonic Calibration ({name})...")
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
        calibrated_clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
        calibrated_clf.fit(X, y)
        print(f"[{name}] Completado en {time.time()-t0:.1f}s.")
        return calibrated_clf

    model_buy = create_and_train_model(X_tr, y_tr_buy, "BUY MODEL")
    model_sell = create_and_train_model(X_tr, y_tr_sell, "SELL MODEL")

    # 4. Predicciones OOS y Winner Takes All
    print("\nGenerando predicciones de probabilidades calibradas en Test OOS...")
    raw_probs_buy = model_buy.predict_proba(X_te)[:, 1]
    raw_probs_sell = model_sell.predict_proba(X_te)[:, 1]

    auc_buy = roc_auc_score(y_te_buy, raw_probs_buy)
    auc_sell = roc_auc_score(y_te_sell, raw_probs_sell)
    pr_buy = average_precision_score(y_te_buy, raw_probs_buy)
    pr_sell = average_precision_score(y_te_sell, raw_probs_sell)

    print("\n" + "="*75)
    print(f"DISCRIMINACIÓN OUT-OF-SAMPLE (UNIVERSO MULTI-ACTIVO):")
    print(f"  * BUY ROC-AUC:  {auc_buy:.4f} | PR-AUC: {pr_buy:.4f} (Base: {y_te_buy.mean():.4f})")
    print(f"  * SELL ROC-AUC: {auc_sell:.4f} | PR-AUC: {pr_sell:.4f} (Base: {y_te_sell.mean():.4f})")
    print("="*75)

    # Aplicar Winner Takes All
    print("Aplicando lógica institucional 'Winner Takes All'...")
    final_prob_buy = np.where(raw_probs_buy > raw_probs_sell, raw_probs_buy, 0.0)
    final_prob_sell = np.where(raw_probs_sell > raw_probs_buy, raw_probs_sell, 0.0)

    test_df["prob_buy"] = final_prob_buy
    test_df["prob_sell"] = final_prob_sell

    # 5. Evaluación de Deciles y Umbrales
    evaluate_deciles(test_df, "prob_buy", "target_quality_trade", "trade_net_return", "BUY TRADES")
    evaluate_deciles(test_df, "prob_sell", "target_short_quality_trade", "short_net_return", "SELL TRADES / SHORTS")

    th_buy = evaluate_thresholds(test_df, "prob_buy", "target_quality_trade", "trade_net_return", "BUY TRADES")
    th_sell = evaluate_thresholds(test_df, "prob_sell", "target_short_quality_trade", "short_net_return", "SELL TRADES / SHORTS")

    # 6. Guardar Modelos
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    
    buy_file = models_dir / "pipeline_universe_buy_calibrated.joblib"
    sell_file = models_dir / "pipeline_universe_sell_calibrated.joblib"
    
    print(f"\nGuardando pipelines calibrados...")
    joblib.dump(model_buy, buy_file)
    joblib.dump(model_sell, sell_file)
    print(f"Modelos guardados en:\n  * {buy_file}\n  * {sell_file}")

    # 7. Importancias de Características
    imp_buy = extract_calibrated_importances(model_buy, feature_cols)
    if not imp_buy.empty:
        print(f"\n--- TOP 25 CARACTERÍSTICAS MÁS IMPORTANTES (BUY MODEL) ---")
        print(imp_buy.head(25).to_string(index=False))

    imp_sell = extract_calibrated_importances(model_sell, feature_cols)
    if not imp_sell.empty:
        print(f"\n--- TOP 25 CARACTERÍSTICAS MÁS IMPORTANTES (SELL MODEL) ---")
        print(imp_sell.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
