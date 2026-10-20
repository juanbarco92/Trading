"""
audit_feature_lookahead.py - Test Automatizado de Permutación Futura para Detección de Lookahead Bias.

Metodología estándar de fondos cuantitativos:
1. Toma una serie histórica de N barras.
2. Calcula las características base hasta la barra T_split (p. ej. barra 300).
3. Modifica aleatoriamente / permuta todas las barras futuras (T_split + 1 hasta el final).
4. Recalcula las características con el futuro alterado.
5. Compara las características en el pasado (t <= T_split):
   - Si alguna feature en t <= T_split cambia de valor -> ¡TIENE LOOKAHEAD BIAS (FUGA DEL FUTURO)!
   - Si todas las features en t <= T_split se mantienen 100% idénticas -> ¡CERO FUGA MATEMÁTICA!
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.utils.exploration_utils import engineer_features, fractal_engineering
from src.utils.stock_sr import compute_stock_support_resistance_features


def run_lookahead_audit():
    print("="*80)
    print("INICIANDO AUDITORÍA FORENSE DE LOOKAHEAD BIAS (TEST DE PERMUTACIÓN FUTURA)")
    print("="*80)

    # 1. Cargar muestra de 600 velas de NVDA
    df_raw = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "NVDA_1hour.parquet").head(600).copy()
    if "datetime" not in df_raw.columns and "timestamp" in df_raw.columns:
        df_raw["datetime"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("datetime").reset_index(drop=True)

    T_SPLIT = 300
    print(f"Total barras muestra: {len(df_raw)}, Punto de corte temporal T_split: barra {T_SPLIT}")

    # --- FASE 1: Cálculo Base ---
    print("\n1. Calculando features base sobre la serie original...")
    df_base, clusterer = engineer_features(df_raw.copy(), n_candles=4, n_clusters=6, fit_clusterer=True, random_state=42)
    df_base = fractal_engineering(df_base)
    df_base = compute_stock_support_resistance_features(df_base)

    # Seleccionar columnas numéricas a auditar
    exclude = ['timestamp', 'datetime', 'candle_cluster', 'pattern']
    audit_cols = [c for c in df_base.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_base[c])]
    print(f"Features a auditar: {len(audit_cols)}")

    base_past = df_base.iloc[:T_SPLIT][audit_cols].copy()

    # --- FASE 2: Alterar el Futuro (t > T_SPLIT) ---
    print(f"\n2. Alterando radicalmente los datos futuros (barras {T_SPLIT} a {len(df_raw)-1})...")
    df_perturbed = df_raw.copy()
    np.random.seed(999)
    n_future = len(df_raw) - T_SPLIT
    
    # Inyectar una explosión de precios y volatilidad artificial en el futuro
    multiplier = 1.0 + np.random.uniform(0.5, 2.0, size=n_future)
    for col in ['open', 'high', 'low', 'close']:
        df_perturbed.loc[T_SPLIT:, col] = df_perturbed.loc[T_SPLIT:, col] * multiplier
        # Asegurar consistencia de barra (high >= max(open, close), low <= min(open, close))
        df_perturbed.loc[T_SPLIT:, 'high'] = df_perturbed.loc[T_SPLIT:, ['open', 'high', 'close']].max(axis=1)
        df_perturbed.loc[T_SPLIT:, 'low'] = df_perturbed.loc[T_SPLIT:, ['open', 'low', 'close']].min(axis=1)
    df_perturbed.loc[T_SPLIT:, 'volume'] = df_perturbed.loc[T_SPLIT:, 'volume'] * 10.0

    # --- FASE 3: Recalcular Features sobre la Serie Alterada ---
    print("3. Recalculando features sobre la serie con futuro alterado...")
    df_new, _ = engineer_features(df_perturbed.copy(), n_candles=4, n_clusters=6, clusterer=clusterer, fit_clusterer=False)
    df_new = fractal_engineering(df_new)
    df_new = compute_stock_support_resistance_features(df_new)

    new_past = df_new.iloc[:T_SPLIT][audit_cols].copy()

    # --- FASE 4: Comparación Forense ---
    print("\n4. Evaluando discrepancias en el pasado (t <= T_split)...")
    leaking_features = []

    for col in audit_cols:
        s_base = base_past[col].to_numpy(dtype=np.float64)
        s_new = new_past[col].to_numpy(dtype=np.float64)

        # Manejar NaNs iguales
        nan_mask_base = np.isnan(s_base)
        nan_mask_new = np.isnan(s_new)
        if not np.array_equal(nan_mask_base, nan_mask_new):
            leaking_features.append((col, "Diferencia en patrón de NaNs", 999.0))
            continue

        valid = ~nan_mask_base
        if np.sum(valid) == 0:
            continue

        diff = np.abs(s_base[valid] - s_new[valid])
        max_diff = np.max(diff)

        if max_diff > 1e-5:
            # Encontrar la primera barra donde ocurre la divergencia
            first_divergent_bar = int(np.where(diff > 1e-5)[0][0])
            bars_before_split = T_SPLIT - first_divergent_bar
            leaking_features.append((col, f"Diferencia máx: {max_diff:.6f} a {bars_before_split} barras antes del corte", max_diff))

    print("\n" + "="*80)
    print("RESULTADO DE LA AUDITORÍA DE LOOKAHEAD BIAS:")
    print("="*80)

    if not leaking_features:
        print("[OK] Ninguna variable cambio de valor. Cero Lookahead Bias detectado.")
    else:
        print(f"[ALERTA CRITICA] Se detectaron {len(leaking_features)} caracteristicas con fuga del futuro:")
        for name, detail, max_d in sorted(leaking_features, key=lambda x: x[2], reverse=True)[:40]:
            print(f"  * {name:<35} -> {detail}")
        if len(leaking_features) > 40:
            print(f"  ... y {len(leaking_features)-40} variables mas.")

    return leaking_features


if __name__ == "__main__":
    run_lookahead_audit()
