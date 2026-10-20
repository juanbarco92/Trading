"""
label_universe.py - Genera el dataset maestro etiquetado con Triple Barrier y Fricción Realista.
Guarda el resultado en data/processed/labeled_universe_1hour.parquet.
"""

import time
import sys
from pathlib import Path

# Agregar directorio raíz al PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.models.labeling import generate_triple_barrier_labels


def main():
    input_path = Path("data/processed/features_universe_1hour.parquet")
    output_path = Path("data/processed/labeled_universe_1hour.parquet")

    if not input_path.exists():
        print(f"Error: {input_path} no encontrado.")
        return

    print("=" * 70)
    print("ETIQUETADO INSTITUCIONAL TRIPLE BARRIER (TICK-07 & TICK-08)")
    print("=" * 70)

    t0 = time.time()
    df = pd.read_parquet(input_path)
    print(f"Cargadas {len(df):,} filas de {input_path.name}...")

    # Parámetros calibrados para Swing Trading de Alto Beta
    k_tp = 3.5          # 3.5x ATR (TP expandido para capturar momentum multidiario)
    k_sl = 1.8          # 1.8x ATR (SL con margen para respirar y evitar premature stops)
    horizon_bars = 35   # 35 velas horarias (~5 días hábiles, swing semanal completo)
    friction_pct = 0.0005  # 5 bps por lado = 10 bps round-trip

    labeled_df = generate_triple_barrier_labels(
        df,
        k_tp=k_tp,
        k_sl=k_sl,
        horizon_bars=horizon_bars,
        friction_pct=friction_pct,
        conservative_intrabar=True,
        adjust_barriers_for_friction=True,
    )

    elapsed = time.time() - t0
    print(f"Etiquetado completado en {elapsed:.2f} segundos.")

    # Guardar en parquet
    labeled_df.to_parquet(output_path, index=False)
    print(f"Guardado dataset etiquetado en {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")

    # Análisis de distribución
    valid = labeled_df.dropna(subset=["target_triple_barrier"])
    print("\n--- DISTRIBUCIÓN GLOBAL DE CLASES TRIPLE BARRIER (LONGS) ---")
    tb_dist = valid["target_triple_barrier"].value_counts().rename({1.0: "TP (+1)", -1.0: "SL (-1)", 0.0: "TIME (0)"})
    tb_pct = (valid["target_triple_barrier"].value_counts(normalize=True) * 100).round(2).rename({1.0: "TP (+1) %", -1.0: "SL (-1) %", 0.0: "TIME (0) %"})
    summary_tb = pd.DataFrame({"Conteo": tb_dist, "Porcentaje": tb_pct})
    print(summary_tb)

    print("\n--- DISTRIBUCIÓN DE CLASE BINARIA TARGET (target_quality_trade - LONGS) ---")
    q_dist = valid["target_quality_trade"].value_counts().rename({1.0: "Éxito Swing Long (1)", 0.0: "Fallo / Neutro (0)"})
    q_pct = (valid["target_quality_trade"].value_counts(normalize=True) * 100).round(2).rename({1.0: "Éxito Long %", 0.0: "Fallo %"})
    summary_q = pd.DataFrame({"Conteo": q_dist, "Porcentaje": q_pct})
    print(summary_q)

    if "target_short_quality_trade" in valid.columns:
        print("\n--- DISTRIBUCIÓN DE CLASE BINARIA TARGET (target_short_quality_trade - SHORTS) ---")
        qs_dist = valid["target_short_quality_trade"].value_counts().rename({1.0: "Éxito Swing Short (1)", 0.0: "Fallo / Neutro (0)"})
        qs_pct = (valid["target_short_quality_trade"].value_counts(normalize=True) * 100).round(2).rename({1.0: "Éxito Short %", 0.0: "Fallo %"})
        summary_qs = pd.DataFrame({"Conteo": qs_dist, "Porcentaje": qs_pct})
        print(summary_qs)

    print("\n--- TASA DE ÉXITO (LONGS vs SHORTS) POR ACTIVO (%) ---")
    by_ticker = valid.groupby("ticker").agg(
        total_velas=("target_quality_trade", "count"),
        exito_long_pct=("target_quality_trade", lambda x: round(x.mean() * 100, 2)),
        exito_short_pct=("target_short_quality_trade", lambda x: round(x.mean() * 100, 2)),
        retorno_long_neto=("trade_net_return", lambda x: round(x.mean(), 2)),
        retorno_short_neto=("short_net_return", lambda x: round(x.mean(), 2)),
    )
    print(by_ticker)
    print("=" * 70)


if __name__ == "__main__":
    main()
