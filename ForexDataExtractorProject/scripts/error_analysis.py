"""
error_analysis.py - Análisis Cuantitativo Profundo de Errores y Diagnóstico del Modelo
Desglosa:
1. Rendimiento por año y por qué 2023-2024 fueron buenos y 2022/2025/2026 planos/negativos.
2. Comparativa estadística de features: Ganadores vs Perdedores (Cohen's d y KS-test).
3. Auditoría del modelo ML: Por qué el ROC-AUC está cerca de 0.50 (análisis de falsos positivos).
4. Anatomía de los trades fallidos: ¿Falsas rupturas? ¿Horas del día? ¿Régimen macro?
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

def main():
    print("=" * 90)
    print("AUDITORÍA Y ANÁLISIS DE ERRORES DEL SISTEMA CUANTITATIVO")
    print("=" * 90)

    # 1. Cargar datos etiquetados y predicciones OOS
    labeled_path = Path("data/processed/labeled_universe_1hour.parquet")
    oos_path = Path("data/processed/oos_predictions.parquet")

    df = pd.read_parquet(labeled_path).dropna(subset=["target_quality_trade"]).reset_index(drop=True)
    if oos_path.exists():
        oos = pd.read_parquet(oos_path)[["timestamp", "ticker", "prob_calibrated", "prob_short_calibrated"]]
        df = df.merge(oos, on=["timestamp", "ticker"], how="left")

    df["year"] = pd.to_datetime(df["timestamp"]).dt.year
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    # Filtrar universo de líderes alto beta
    high_beta = ["PLTR", "NVDA", "TSLA", "MELI", "AMD", "ARM", "COIN", "MSTR", "META", "SHOP", "SMCI"]
    df_hb = df[df["ticker"].isin(high_beta)].copy()

    # Definir la señal de entrada evaluada
    df_hb["is_signal"] = (
        (df_hb["dist_to_res20_atr"] < 0.5) &
        (df_hb["is_market_bullish"] == 1) &
        (df_hb["rvol"] > 1.2) &
        (df_hb["alpha_5d"] > 0)
    )

    signals = df_hb[df_hb["is_signal"]].copy()
    print(f"Total de observaciones analizadas: {len(df_hb):,} velas en 11 activos líderes.")
    print(f"Total de señales generadas por la estrategia: {len(signals):,} señales de compra.\n")

    # =========================================================================
    # SECCIÓN 1: DESGLOSE DE MÉTRICAS DE SEÑAL AÑO POR AÑO
    # =========================================================================
    print("--- 1. DESGLOSE DE SEÑALES AÑO POR AÑO ---")
    annual_diag = []
    for yr, grp in signals.groupby("year"):
        n_sigs = len(grp)
        wins = grp[grp["target_quality_trade"] == 1]
        losses = grp[grp["trade_exit_type"] == "SL"]
        timeouts = grp[grp["trade_exit_type"] == "TIME"]

        wr = (len(wins) / n_sigs) * 100.0 if n_sigs > 0 else 0.0
        avg_ret = grp["trade_net_return"].mean()
        avg_win_ret = wins["trade_net_return"].mean() if len(wins) > 0 else 0.0
        avg_loss_ret = losses["trade_net_return"].mean() if len(losses) > 0 else 0.0

        annual_diag.append({
            "Año": yr,
            "Señales": n_sigs,
            "Win Rate %": round(wr, 2),
            "Retorno Medio %": round(avg_ret, 2),
            "Ganancia Media (Wins) %": round(avg_win_ret, 2),
            "Pérdida Media (Losses) %": round(avg_loss_ret, 2),
            "Ratio Win/Loss": round(abs(avg_win_ret / avg_loss_ret), 2) if avg_loss_ret != 0 else 0,
            "TPs": len(wins),
            "SLs": len(losses),
            "Timeouts": len(timeouts),
        })

    print(pd.DataFrame(annual_diag).to_string(index=False))

    # =========================================================================
    # SECCIÓN 2: DISCRIMINACIÓN DE FEATURES (GANADORES VS PERDEDORES)
    # =========================================================================
    print("\n" + "=" * 90)
    print("--- 2. DISCRIMINACIÓN DE CARACTERÍSTICAS: TRADES GANADORES VS PERDEDORES ---")
    print("¿Cuáles de las 20 variables realmente separan un trade ganador de uno perdedor?")
    print("=" * 90)

    feature_cols = [
        "rvol", "rvol_surge", "vwap_dist_pct", "is_above_vwap", "volume_zscore",
        "is_bullish_absorption", "is_market_bullish", "spy_dist_ema50_pct",
        "alpha_1d", "alpha_5d", "is_outperforming_spy", "rs_divergence_bullish",
        "atr_pct", "vol_regime", "is_vol_compression", "is_vol_expansion",
        "dist_to_res20_atr", "dist_to_sup20_atr", "return_1h", "return_4h"
    ]

    wins_all = signals[signals["target_quality_trade"] == 1]
    losses_all = signals[signals["target_quality_trade"] == 0]

    feature_discrim = []
    for feat in feature_cols:
        w_vals = wins_all[feat].dropna()
        l_vals = losses_all[feat].dropna()

        if len(w_vals) > 10 and len(l_vals) > 10:
            # Cohen's d (tamaño del efecto)
            mean_diff = w_vals.mean() - l_vals.mean()
            pooled_std = np.sqrt((w_vals.var() + l_vals.var()) / 2.0) + 1e-9
            cohen_d = mean_diff / pooled_std

            # KS test (diferencia de distribuciones)
            ks_stat, ks_pval = stats.ks_2samp(w_vals, l_vals)

            feature_discrim.append({
                "Feature": feat,
                "Media (Ganadores)": round(w_vals.mean(), 3),
                "Media (Perdedores)": round(l_vals.mean(), 3),
                "Diferencia %": round(((w_vals.mean() - l_vals.mean()) / (abs(l_vals.mean()) + 1e-6)) * 100.0, 1),
                "Cohen's d (Poder de Separación)": round(cohen_d, 3),
                "KS Stat": round(ks_stat, 3),
                "P-Valor Significativo": "SÍ" if ks_pval < 0.05 else "NO (Ruido)",
            })

    discrim_df = pd.DataFrame(feature_discrim).sort_values(by="KS Stat", ascending=False)
    print(discrim_df.to_string(index=False))

    # =========================================================================
    # SECCIÓN 3: AUDITORÍA DEL MODELO LIGHTGBM Y LAS PROBABILIDADES
    # =========================================================================
    print("\n" + "=" * 90)
    print("--- 3. AUDITORÍA DEL MODELO MACHINE LEARNING: ¿POR QUÉ FALLA EN DISCRIMINAR? ---")
    print("=" * 90)
    if "prob_calibrated" in signals.columns:
        prob_w = wins_all["prob_calibrated"].dropna()
        prob_l = losses_all["prob_calibrated"].dropna()
        print(f"Probabilidad media predicha en trades Ganadores: {prob_w.mean():.4f}")
        print(f"Probabilidad media predicha en trades Perdedores: {prob_l.mean():.4f}")
        print(f"Diferencia entre ganador y perdedor: {(prob_w.mean() - prob_l.mean()):.4f} (¡Prácticamente CERO!)")

        # Correlación de la probabilidad con el retorno real
        corr_prob_ret = signals["prob_calibrated"].corr(signals["trade_net_return"])
        print(f"Correlación entre Probabilidad del Modelo y Retorno Real: {corr_prob_ret:.4f}")

    # =========================================================================
    # SECCIÓN 4: ANATOMÍA DE LAS PÉRDIDAS (FALSAS RUPTURAS)
    # =========================================================================
    print("\n" + "=" * 90)
    print("--- 4. ANATOMÍA DE LOS TRADES FALLIDOS (¿CUÁNDO OCURREN LOS SLs?) ---")
    print("=" * 90)

    # A. Por hora del día
    hour_summary = signals.groupby("hour").agg(
        total_signals=("target_quality_trade", "count"),
        win_rate=("target_quality_trade", lambda x: round(x.mean() * 100, 1)),
        avg_ret=("trade_net_return", lambda x: round(x.mean(), 2)),
    ).reset_index()
    print("\nDesempeño según Hora del Día (UTC):")
    print(hour_summary.to_string(index=False))

    # B. Por Activo Individual
    ticker_summary = signals.groupby("ticker").agg(
        total_signals=("target_quality_trade", "count"),
        win_rate=("target_quality_trade", lambda x: round(x.mean() * 100, 1)),
        retorno_medio=("trade_net_return", lambda x: round(x.mean(), 2)),
        retorno_total=("trade_net_return", "sum"),
    ).sort_values("retorno_total", ascending=False).reset_index()
    print("\nDesempeño por Activo en Señales de Breakout:")
    print(ticker_summary.to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    main()
