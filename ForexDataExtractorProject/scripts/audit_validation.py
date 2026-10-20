"""
audit_validation.py - Auditoría Cuantitativa de Validación Cruzada (TICK-09).
Aplica PurgedKFold y PurgedWalkForwardCV sobre el dataset maestro de 83.590 velas
y verifica formalmente la ausencia de fuga de información (Cero solapamiento).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.models.validation import PurgedKFold, PurgedWalkForwardCV, verify_no_leakage


def main():
    data_path = Path("data/processed/labeled_universe_1hour.parquet")
    if not data_path.exists():
        print(f"Error: {data_path} no existe.")
        return

    print("=" * 80)
    print("AUDITORÍA DE VALIDACIÓN CRUZADA: PURGED & EMBARGOED CV (TICK-09)")
    print("=" * 80)

    # Filtrar velas con target válido (las últimas 26 velas de cada ticker no tienen futuro)
    df = pd.read_parquet(data_path).dropna(subset=["target_quality_trade"]).reset_index(drop=True)
    print(f"Dataset cargado: {len(df):,} velas válidas con target en {len(df['ticker'].unique())} activos.")

    # 1. Auditoría de PurgedKFold (5 Folds)
    print("\n--- 1. PURGED K-FOLD (5 Folds, Embargo = 26 barras / 1%) ---")
    cv_kfold = PurgedKFold(n_splits=5, pct_embargo=0.01)
    t0, t1 = cv_kfold._resolve_time_series(df)

    kfold_records = []
    for fold, (train_idx, test_idx) in enumerate(cv_kfold.split(df)):
        audit = verify_no_leakage(train_idx, test_idx, t0, t1)
        test_t0 = audit["test_t0_min"].strftime("%Y-%m-%d %H:%M")
        test_t1 = audit["test_t1_max"].strftime("%Y-%m-%d %H:%M")

        kfold_records.append({
            "Fold": fold + 1,
            "Inicio Test": test_t0,
            "Fin Test": test_t1,
            "Muestras Train": f"{audit['train_samples']:,}",
            "Muestras Test": f"{audit['test_samples']:,}",
            "Solapamientos": audit["temporal_overlap_count"],
            "Fuga Cero (Leakage Free)": "SI [OK]" if audit["is_leakage_free"] else "ERROR",
        })

    print(pd.DataFrame(kfold_records).to_string(index=False))

    # 2. Auditoría de PurgedWalkForwardCV (4 Folds expansivos)
    print("\n--- 2. PURGED WALK-FORWARD CV (Ventana Expansiva OOS) ---")
    cv_wf = PurgedWalkForwardCV(n_splits=5)
    wf_t0, wf_t1 = cv_wf._resolve_time_series(df)

    wf_records = []
    for step, (train_idx, test_idx) in enumerate(cv_wf.split(df)):
        audit = verify_no_leakage(train_idx, test_idx, wf_t0, wf_t1)
        test_t0 = audit["test_t0_min"].strftime("%Y-%m-%d %H:%M")
        test_t1 = audit["test_t1_max"].strftime("%Y-%m-%d %H:%M")

        wf_records.append({
            "Paso Walk-Forward": f"Paso {step + 1}",
            "Inicio Test": test_t0,
            "Fin Test": test_t1,
            "Train Acumulado": f"{audit['train_samples']:,}",
            "Test OOS": f"{audit['test_samples']:,}",
            "Solapamientos": audit["temporal_overlap_count"],
            "Fuga Cero (Leakage Free)": "SI [OK]" if audit["is_leakage_free"] else "ERROR",
        })

    print(pd.DataFrame(wf_records).to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
