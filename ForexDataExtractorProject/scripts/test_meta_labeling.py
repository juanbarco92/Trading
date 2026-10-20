"""
test_meta_labeling.py - Prueba de Meta-Labeling (López de Prado) sobre eventos candidatos.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score
from src.models.train_swing_model import DEFAULT_FEATURES
from src.models.validation import PurgedKFold


def main():
    df = pd.read_parquet("data/processed/labeled_universe_1hour.parquet").dropna(subset=["target_quality_trade"])
    
    # Definir el generador primario de eventos institucionales:
    # 1. Breakout estructural o rebote en soporte
    # 2. Con volumen relativo anormal (RVOL > 1.1 o Z-Score > 0.5)
    # 3. Mercado macro alcista (SPY > EMA50)
    event_mask = (
        (df["is_market_bullish"] == 1) &
        ((df["dist_to_res20_atr"] < 0.8) | (df["dist_to_sup20_atr"] < 1.0) | (df["is_vol_compression"] == 1)) &
        (df["rvol"] > 1.0)
    )

    events_df = df[event_mask].reset_index(drop=True)
    print(f"Total eventos candidatos filtrados: {len(events_df):,} de {len(df):,} ({len(events_df)/len(df)*100:.1f}%)")
    print(f"Tasa base en eventos: {events_df['target_quality_trade'].mean() * 100:.2f}%\n")

    X = events_df[DEFAULT_FEATURES].astype(np.float32)
    y = events_df["target_quality_trade"].astype(int)
    t0 = pd.to_datetime(events_df["timestamp"])
    t1 = pd.to_datetime(events_df["exit_timestamp"]) if "exit_timestamp" in events_df else t0 + pd.Timedelta(hours=10)

    cv = PurgedKFold(n_splits=5, timestamps=t0, t1=t1, pct_embargo=0.01)

    oos_preds = np.zeros(len(X))
    for tr, te in cv.split(X):
        clf = lgb.LGBMClassifier(
            max_depth=3,
            num_leaves=7,
            min_child_samples=50,
            learning_rate=0.02,
            n_estimators=100,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X.iloc[tr], y.iloc[tr])
        oos_preds[te] = clf.predict_proba(X.iloc[te])[:, 1]

    auc = roc_auc_score(y, oos_preds)
    pr_auc = average_precision_score(y, oos_preds)
    print(f"Meta-Model Out-of-Sample ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")

    # Barrido de percentiles de convicción
    for pct in [50, 70, 80, 85, 90, 95]:
        th = np.percentile(oos_preds, pct)
        mask = oos_preds >= th
        n_sel = np.sum(mask)
        sub_y = y[mask]
        sub_ret = events_df.loc[mask, "trade_net_return"]
        tp_rate = sub_y.mean() * 100
        wr = (sub_ret > 0).mean() * 100
        ret = sub_ret.mean()
        wins = sub_ret[sub_ret > 0]
        losses = sub_ret[sub_ret <= 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999.0

        print(f"Top {100-pct:>2}% Señales (P>={th:.3f}) | N: {n_sel:>5} | TP Rate: {tp_rate:>5.2f}% | WinRate: {wr:>5.2f}% | RetNeto: {ret:>+5.2f}% | PF: {pf:.2f}")


if __name__ == "__main__":
    main()
