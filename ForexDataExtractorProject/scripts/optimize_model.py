"""
optimize_model.py - Diagnóstico y Optimización de Hiperparámetros y Señales Out-of-Sample.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
from src.models.train_swing_model import load_training_data, DEFAULT_FEATURES
from src.models.validation import PurgedKFold


def main():
    X, y, t0, t1, meta = load_training_data()
    print(f"Total datos: {len(X):,} filas | Tasa base: {y.mean()*100:.2f}%")

    cv = PurgedKFold(n_splits=5, timestamps=t0, t1=t1, pct_embargo=0.01)

    # 1. Comparar familias de modelos
    models = {
        "LogReg (L2 C=0.01)": LogisticRegression(C=0.01, max_iter=200, random_state=42),
        "LogReg (L1 C=0.01)": LogisticRegression(C=0.01, penalty="l1", solver="liblinear", random_state=42),
        "RandomForest (depth=3)": RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=200, random_state=42, n_jobs=-1),
        "LightGBM (depth=2, leaves=4)": lgb.LGBMClassifier(max_depth=2, num_leaves=4, min_child_samples=300, learning_rate=0.02, n_estimators=100, reg_alpha=1.0, reg_lambda=5.0, random_state=42, n_jobs=-1, verbose=-1),
        "LightGBM (depth=3, leaves=7)": lgb.LGBMClassifier(max_depth=3, num_leaves=7, min_child_samples=300, learning_rate=0.02, n_estimators=100, reg_alpha=1.0, reg_lambda=5.0, random_state=42, n_jobs=-1, verbose=-1),
        "LightGBM (depth=4, leaves=15)": lgb.LGBMClassifier(max_depth=4, num_leaves=15, min_child_samples=200, learning_rate=0.03, n_estimators=150, reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1),
    }

    results = []
    for name, model in models.items():
        oos_preds = np.zeros(len(X))
        auc_scores = []
        for tr, te in cv.split(X):
            model.fit(X.iloc[tr], y.iloc[tr])
            p = model.predict_proba(X.iloc[te])[:, 1]
            oos_preds[te] = p
            auc_scores.append(roc_auc_score(y.iloc[te], p))

        global_auc = roc_auc_score(y, oos_preds)
        global_prauc = average_precision_score(y, oos_preds)

        # Retorno en top 10% de predicciones de cada modelo
        top10_th = np.percentile(oos_preds, 90)
        top10_mask = oos_preds >= top10_th
        top10_prec = np.mean(y[top10_mask]) * 100.0
        top10_ret = np.mean(meta.loc[top10_mask, "trade_net_return"])
        top10_wr = np.mean(meta.loc[top10_mask, "trade_net_return"] > 0) * 100.0

        results.append({
            "Modelo": name,
            "Mean Fold AUC": round(float(np.mean(auc_scores)), 4),
            "Global AUC": round(float(global_auc), 4),
            "Global PR-AUC": round(float(global_prauc), 4),
            "Top 10% Precisión": f"{top10_prec:.2f}%",
            "Top 10% Win Rate": f"{top10_wr:.2f}%",
            "Top 10% Retorno Neto": f"{top10_ret:.2f}%",
        })

    print("\n--- COMPARATIVA DE MODELOS OUT-OF-SAMPLE (PURGED K-FOLD) ---")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
