"""
train_swing_model.py - TICK-10: Entrenamiento y Calibración del Modelo (LightGBM)

Pipeline institucional para el entrenamiento del clasificador Swing Trading:
1. Carga de las 20 features ortogonales podadas y el target Triple Barrier (TICK-07/08).
2. Validación Cruzada Out-of-Sample mediante PurgedKFold con Embargo (TICK-09).
3. Calibración de Probabilidades con CalibratedClassifierCV (Sigmoid / Platt Scaling / Isotonic).
4. Barrido de umbrales de decisión (Threshold Sweeping) y análisis de Expectativa Matemática ($E[R]$).
5. Auditoría de importancia de variables (Gain & Split Importance).
6. Persistencia del modelo final de producción en models/equities/swing_lgbm_calibrated.joblib.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Agregar directorio raíz al PYTHONPATH si es ejecutado como script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.models.validation import PurgedKFold, compute_t1_series


# 20 Variables Ortogonales Estacionarias
DEFAULT_FEATURES: List[str] = [
    "rvol",
    "rvol_surge",
    "vwap_dist_pct",
    "is_above_vwap",
    "volume_zscore",
    "is_bullish_absorption",
    "is_market_bullish",
    "spy_dist_ema50_pct",
    "alpha_1d",
    "alpha_5d",
    "is_outperforming_spy",
    "rs_divergence_bullish",
    "atr_pct",
    "vol_regime",
    "is_vol_compression",
    "is_vol_expansion",
    "dist_to_res20_atr",
    "dist_to_sup20_atr",
    "return_1h",
    "return_4h",
]


def load_training_data(
    data_path: Union[str, Path] = "data/processed/labeled_universe_1hour.parquet",
    features: Optional[List[str]] = None,
    target_col: str = "target_quality_trade",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """
    Carga y limpia el dataset para entrenamiento.

    Retorna:
    --------
    X : pd.DataFrame
        Matriz de características numéricas ortogonales.
    y : pd.Series
        Variable objetivo binaria (target_col).
    t0 : pd.Series
        Timestamp de inicio / entrada.
    t1 : pd.Series
        Timestamp de cierre / salida del evento.
    meta : pd.DataFrame
        Metadatos complementarios para evaluación y backtesting.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {path}")

    df = pd.read_parquet(path)

    # Filtrar filas sin target válido (las últimas barras de cada ticker)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    feat_cols = features if features is not None else DEFAULT_FEATURES
    missing_feats = [f for f in feat_cols if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Faltan características requeridas en el dataset: {missing_feats}")

    X = df[feat_cols].astype(np.float32)
    y = df[target_col].astype(int)
    t0 = pd.to_datetime(df["timestamp"])
    t1 = compute_t1_series(df)

    meta_cols = [
        "ticker",
        "timestamp",
        "trade_net_return",
        "trade_gross_return",
        "trade_exit_type",
        "trade_exit_bar",
        "close",
        "target_short_quality_trade",
        "short_net_return",
        "alpha_5d",
        "is_market_bullish",
    ]
    meta = df[[col for col in meta_cols if col in df.columns]].copy()

    return X, y, t0, t1, meta


def get_base_lgbm_classifier(min_child_samples: int = 80, random_state: int = 42) -> lgb.LGBMClassifier:
    """Configura el clasificador base LightGBM con hiperparámetros contra sobreajuste financiero."""
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=min_child_samples,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def evaluate_out_of_sample_cv(
    X: pd.DataFrame,
    y: pd.Series,
    t0: pd.Series,
    t1: pd.Series,
    meta: pd.DataFrame,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    min_child_samples: int = 80,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Ejecuta validación cruzada PurgedKFold, prediciendo probabilidades Out-of-Sample estrictas.

    Retorna:
    --------
    results_df : pd.DataFrame
        DataFrame con cada observación y su probabilidad Out-of-Sample predicha.
    metrics : dict
        Diccionario con las métricas cuantitativas globales y por umbral.
    """
    cv = PurgedKFold(n_splits=n_splits, timestamps=t0, t1=t1, pct_embargo=pct_embargo)

    n_samples = len(X)
    oos_probs = np.full(n_samples, np.nan, dtype=np.float64)
    fold_metrics = []

    print(f"Ejecutando Purged K-Fold Cross-Validation ({n_splits} Folds, Embargo {pct_embargo*100:.1f}%)...")

    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # 1. Ajustar clasificador base
        base_clf = get_base_lgbm_classifier(min_child_samples=min_child_samples, random_state=random_state + fold_num)

        # 2. Calibración dentro del fold usando PurgedKFold interno para evitar fuga en calibración
        inner_cv = PurgedKFold(
            n_splits=3,
            timestamps=t0.iloc[train_idx],
            t1=t1.iloc[train_idx],
            pct_embargo=pct_embargo,
        )
        calibrated_clf = CalibratedClassifierCV(
            estimator=base_clf,
            method="sigmoid",
            cv=inner_cv,
        )
        calibrated_clf.fit(X_train, y_train)

        # 3. Predicciones estrictamente fuera de muestra
        probs_test = calibrated_clf.predict_proba(X_test)[:, 1]
        oos_probs[test_idx] = probs_test

        # Métricas individuales del fold
        fold_auc = roc_auc_score(y_test, probs_test)
        fold_pr_auc = average_precision_score(y_test, probs_test)
        fold_brier = brier_score_loss(y_test, probs_test)

        fold_metrics.append({
            "fold": fold_num + 1,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "roc_auc": round(float(fold_auc), 4),
            "pr_auc": round(float(fold_pr_auc), 4),
            "brier_score": round(float(fold_brier), 4),
        })
        print(f"  Fold {fold_num + 1}/{n_splits} completado -> ROC-AUC: {fold_auc:.4f} | PR-AUC: {fold_pr_auc:.4f} | Brier: {fold_brier:.4f}")

    results_df = meta.copy()
    results_df["target_quality_trade"] = y.values
    results_df["prob_calibrated"] = oos_probs

    # 4. Análisis global Out-of-Sample
    valid_mask = ~np.isnan(oos_probs)
    y_valid = y.iloc[valid_mask].values
    probs_valid = oos_probs[valid_mask]
    returns_valid = results_df.loc[valid_mask, "trade_net_return"].values

    global_auc = float(roc_auc_score(y_valid, probs_valid))
    global_pr_auc = float(average_precision_score(y_valid, probs_valid))
    global_brier = float(brier_score_loss(y_valid, probs_valid))
    global_logloss = float(log_loss(y_valid, probs_valid))

    # 5. Barrido de Umbrales de Probabilidad y Expectativa ($E[R]$)
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    threshold_analysis = []

    for th in thresholds:
        pred_bin = (probs_valid >= th).astype(int)
        n_signals = int(np.sum(pred_bin))

        if n_signals > 0:
            prec = float(precision_score(y_valid, pred_bin, zero_division=0))
            rec = float(recall_score(y_valid, pred_bin, zero_division=0))
            active_returns = returns_valid[pred_bin == 1]
            mean_net_return = float(np.mean(active_returns))
            win_rate = float(np.mean(active_returns > 0))
            # Ganancia media en ganadores vs pérdida media en perdedores
            wins = active_returns[active_returns > 0]
            losses = active_returns[active_returns <= 0]
            avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
            avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
            profit_factor = (
                float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 999.0
            )
            # Expectativa matemática en % por trade
            expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)
        else:
            prec, rec, mean_net_return, win_rate = 0.0, 0.0, 0.0, 0.0
            avg_win, avg_loss, profit_factor, expectancy = 0.0, 0.0, 0.0, 0.0

        threshold_analysis.append({
            "umbral": th,
            "total_senales": n_signals,
            "pct_del_universo": round((n_signals / len(y_valid)) * 100, 2),
            "precision_oos": round(prec * 100, 2),
            "recall_oos": round(rec * 100, 2),
            "win_rate_neto": round(win_rate * 100, 2),
            "retorno_medio_neto": round(mean_net_return, 2),
            "ganancia_media_win": round(avg_win, 2),
            "perdida_media_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "expectativa_matematica_pct": round(expectancy, 2),
        })

    # 6. Desempeño por activo con umbral estándar (P >= 0.55)
    ticker_breakdown = []
    th_eval = 0.55
    for ticker, group in results_df.groupby("ticker"):
        t_y = group["target_quality_trade"].values
        t_prob = group["prob_calibrated"].values
        t_ret = group["trade_net_return"].values

        mask_sig = t_prob >= th_eval
        n_sig = int(np.sum(mask_sig))

        if n_sig > 0:
            t_prec = float(np.mean(t_y[mask_sig])) * 100.0
            t_net_ret = float(np.mean(t_ret[mask_sig]))
            t_wr = float(np.mean(t_ret[mask_sig] > 0)) * 100.0
        else:
            t_prec, t_net_ret, t_wr = 0.0, 0.0, 0.0

        ticker_breakdown.append({
            "ticker": ticker,
            "total_velas": len(group),
            "senales_p55": n_sig,
            "precision_p55": round(t_prec, 2),
            "win_rate_neto_p55": round(t_wr, 2),
            "retorno_medio_neto": round(t_net_ret, 2),
        })

    summary_metrics = {
        "global_roc_auc": round(global_auc, 4),
        "global_pr_auc": round(global_pr_auc, 4),
        "global_brier_score": round(global_brier, 4),
        "global_log_loss": round(global_logloss, 4),
        "fold_metrics": fold_metrics,
        "threshold_analysis": threshold_analysis,
        "ticker_breakdown": ticker_breakdown,
    }

    return results_df, summary_metrics


def train_final_calibrated_model(
    X: pd.DataFrame,
    y: pd.Series,
    t0: pd.Series,
    t1: pd.Series,
    pct_embargo: float = 0.01,
    min_child_samples: int = 80,
    save_path: Union[str, Path] = "models/equities/swing_lgbm_calibrated.joblib",
    random_state: int = 42,
) -> Tuple[CalibratedClassifierCV, pd.DataFrame]:
    """
    Entrena el modelo de producción final calibrado sobre todo el universo disponible
    y calcula la importancia de variables.
    """
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nEntrenando modelo de producción final calibrado sobre {len(X):,} observaciones...")
    t0_time = time.time()

    # 1. Ajustar clasificador base para extraer importancias de variables
    base_model = get_base_lgbm_classifier(min_child_samples=min_child_samples, random_state=random_state)
    base_model.fit(X, y)

    # 2. Importancia de características
    feature_names = X.columns.tolist()
    gain_importance = base_model.booster_.feature_importance(importance_type="gain")
    split_importance = base_model.booster_.feature_importance(importance_type="split")

    feat_imp_df = pd.DataFrame({
        "feature": feature_names,
        "gain_importance": gain_importance,
        "split_importance": split_importance,
    })
    total_gain = feat_imp_df["gain_importance"].sum()
    if total_gain > 0:
        feat_imp_df["gain_pct"] = (feat_imp_df["gain_importance"] / total_gain) * 100.0
    else:
        feat_imp_df["gain_pct"] = 0.0
    feat_imp_df = feat_imp_df.sort_values("gain_pct", ascending=False).reset_index(drop=True)

    # 3. Calibrar usando PurgedKFold interno sobre el dataset completo
    cv = PurgedKFold(n_splits=5, timestamps=t0, t1=t1, pct_embargo=pct_embargo)
    calibrated_model = CalibratedClassifierCV(
        estimator=get_base_lgbm_classifier(min_child_samples=min_child_samples, random_state=random_state),
        method="sigmoid",
        cv=cv,
    )
    calibrated_model.fit(X, y)

    # 4. Persistir modelo
    joblib.dump(calibrated_model, save_p)
    elapsed = time.time() - t0_time
    print(f"Modelo calibrado guardado en {save_p} ({save_p.stat().st_size / (1024*1024):.2f} MB) en {elapsed:.2f}s.")

    return calibrated_model, feat_imp_df


def main():
    print("=" * 80)
    print("ENTRENAMIENTO Y CALIBRACIÓN DE MODELO LIGHTGBM (TICK-10)")
    print("=" * 80)

    # =========================================================================
    # 1. ENTRENAMIENTO Y CALIBRACIÓN: MODELO LONG (COMPRA)
    # =========================================================================
    print("=" * 80)
    print("FASE 1: ENTRENAMIENTO DEL MODELO LONG (SWING COMPRADOR)")
    print("=" * 80)
    data_path = "data/processed/labeled_universe_1hour.parquet"
    X_long, y_long, t0_long, t1_long, meta_long = load_training_data(data_path, target_col="target_quality_trade")
    print(f"Datos Long: {len(X_long):,} filas | {X_long.shape[1]} features | Tasa base: {y_long.mean()*100:.2f}%")

    results_long_df, metrics_long = evaluate_out_of_sample_cv(
        X_long, y_long, t0_long, t1_long, meta_long, n_splits=5, pct_embargo=0.01
    )

    print("\n--- MÉTRICAS GLOBALES OUT-OF-SAMPLE (LONGS - PURGED K-FOLD) ---")
    print(f"  * ROC-AUC:      {metrics_long['global_roc_auc']:.4f}")
    print(f"  * PR-AUC:       {metrics_long['global_pr_auc']:.4f} (Línea base: {y_long.mean():.4f})")
    print(f"  * Brier Score:  {metrics_long['global_brier_score']:.4f}")

    th_long_df = pd.DataFrame(metrics_long["threshold_analysis"])
    print("\n--- BARRIDO DE UMBRALES LONG ---")
    print(th_long_df[["umbral", "total_senales", "pct_del_universo", "precision_oos", "win_rate_neto", "retorno_medio_neto", "profit_factor"]].to_string(index=False))

    model_long_path = "models/equities/swing_lgbm_calibrated.joblib"
    calibrated_long_model, feat_imp_long = train_final_calibrated_model(
        X_long, y_long, t0_long, t1_long, save_path=model_long_path
    )

    # =========================================================================
    # 2. ENTRENAMIENTO Y CALIBRACIÓN: MODELO SHORT (VENTA EN CORTO)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FASE 2: ENTRENAMIENTO DEL MODELO SHORT (SWING VENDEDOR)")
    print("=" * 80)
    X_short, y_short, t0_short, t1_short, meta_short = load_training_data(data_path, target_col="target_short_quality_trade")
    print(f"Datos Short: {len(X_short):,} filas | {X_short.shape[1]} features | Tasa base: {y_short.mean()*100:.2f}%")

    results_short_df, metrics_short = evaluate_out_of_sample_cv(
        X_short, y_short, t0_short, t1_short, meta_short, n_splits=5, pct_embargo=0.01
    )

    print("\n--- MÉTRICAS GLOBALES OUT-OF-SAMPLE (SHORTS - PURGED K-FOLD) ---")
    print(f"  * ROC-AUC:      {metrics_short['global_roc_auc']:.4f}")
    print(f"  * PR-AUC:       {metrics_short['global_pr_auc']:.4f} (Línea base: {y_short.mean():.4f})")
    print(f"  * Brier Score:  {metrics_short['global_brier_score']:.4f}")

    th_short_df = pd.DataFrame(metrics_short["threshold_analysis"])
    print("\n--- BARRIDO DE UMBRALES SHORT ---")
    print(th_short_df[["umbral", "total_senales", "pct_del_universo", "precision_oos", "win_rate_neto", "retorno_medio_neto", "profit_factor"]].to_string(index=False))

    model_short_path = "models/equities/swing_lgbm_short_calibrated.joblib"
    calibrated_short_model, feat_imp_short = train_final_calibrated_model(
        X_short, y_short, t0_short, t1_short, save_path=model_short_path
    )

    # =========================================================================
    # 3. CONSOLIDACIÓN DE PREDICCIONES OOS BIDIRECCIONALES
    # =========================================================================
    combined_oos = results_long_df.copy()
    combined_oos["prob_short_calibrated"] = results_short_df["prob_calibrated"]

    oos_save_path = Path("data/processed/oos_predictions.parquet")
    combined_oos.to_parquet(oos_save_path, index=False)
    print(f"\nPredicciones Out-of-Sample Bidireccionales guardadas en {oos_save_path} ({oos_save_path.stat().st_size / (1024*1024):.2f} MB)")

    # Guardar reporte JSON
    report_path = Path("data/processed/model_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"long_metrics": metrics_long, "short_metrics": metrics_short}, f, indent=4)
    print(f"Reporte completo de evaluación guardado en {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
