"""
Análisis Cuantitativo de Errores - Modelo 2A (Ingreso) Exclusivo para Arquetipo 2 (Breakdown Estructural)
Audita a fondo los falsos quiebres (falsos positivos / bear traps) frente a los quiebres genuinos.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_breakdown_estructural_isolated import extract_breakdown_only_events
from scripts.run_motor_2_walkforward_pipeline import PARQUET_IBKR

def run_error_analysis_2a():
    print("=" * 90)
    print("ANALISIS CUANTITATIVO DE ERRORES: MODELO 2A (INGRESO) - ARQUETIPO 2 (BREAKDOWN)")
    print("=" * 90)
    
    df_raw = pd.read_parquet(PARQUET_IBKR)
    df_events = extract_breakdown_only_events(df_raw)
    
    # Adicionar métricas diagnósticas adicionales para el análisis de causa raíz
    # (calculadas a posteriori sobre df_raw para cruzar con los fallos)
    print("\nCalculando métricas diagnósticas profundas de microestructura para el análisis...")
    raw_by_tk = {tk: grp.sort_values("timestamp").reset_index(drop=True) for tk, grp in df_raw.groupby("ticker")}
    
    extra_diag = []
    for _, row in df_events.iterrows():
        tk = row["ticker"]
        stk_df = raw_by_tk.get(tk)
        if stk_df is None:
            continue
        ts_e = row["timestamp"]
        matches = np.where(stk_df["timestamp"].to_numpy() == ts_e)[0]
        if len(matches) == 0:
            continue
        i = matches[0]
        
        c = stk_df["close"].to_numpy()
        o = stk_df["open"].to_numpy()
        h = stk_df["high"].to_numpy()
        l = stk_df["low"].to_numpy()
        v = stk_df["volume"].to_numpy()
        
        # 1. ¿Cuántas velas rojas consecutivas llevaba antes del quiebre? (Sobreextensión previa)
        consec_red = 0
        for step in range(1, 8):
            if i - step >= 0 and c[i - step] < o[i - step]:
                consec_red += 1
            else:
                break
                
        # 2. Profundidad del quiebre: ¿cuánto perforó el soporte de 40 barras?
        min_40 = np.min(l[max(0, i-40): i])
        penetration_pct = (min_40 - c[i]) / min_40 # Cuánto cerró por debajo
        
        # 3. Ratio de volumen con respecto a la vela inmediatamente anterior
        v_acceleration = v[i] / (v[i-1] + 1e-6) if i > 0 else 1.0
        
        # 4. Rango de consolidación de 40 barras (Tightness): (Max - Min) / ATR
        max_40 = np.max(h[max(0, i-40): i])
        consolidation_width = (max_40 - min_40) / (c[i] * row["atr_d"] + 1e-6)
        
        # 5. Distancia a SMA 200 horaria (~1 mes)
        sma_200 = np.mean(c[max(0, i-200): i+1]) if i >= 50 else c[i]
        dist_sma_200 = (c[i] - sma_200) / sma_200
        
        # 6. Retorno de las últimas 7 barras (1 día previo)
        ret_1d_prev = (c[i] - c[max(0, i-7)]) / c[max(0, i-7)]
        
        extra_diag.append({
            "event_id": row["event_id"],
            "consec_red_bars": consec_red,
            "penetration_pct": penetration_pct * 100.0,
            "v_acceleration": v_acceleration,
            "consolidation_width_atr": consolidation_width,
            "dist_sma_200": dist_sma_200 * 100.0,
            "ret_1d_prev_pct": ret_1d_prev * 100.0
        })
        
    df_diag = pd.DataFrame(extra_diag)
    df_events = pd.merge(df_events, df_diag, on="event_id")
    
    # 2. Pipeline Walk-Forward ML (Modelo 2A)
    windows = [
        ("Ventana 1 (2022)", list(range(2020, 2022)), [2022]),
        ("Ventana 2 (2023)", list(range(2020, 2023)), [2023]),
        ("Ventana 3 (2024)", list(range(2020, 2024)), [2024]),
        ("Ventana 4 (2025-26)", list(range(2020, 2025)), [2025, 2026])
    ]
    
    features_base = [
        "atr_d", "ret_3d", "ret_5d", "z_3d", "rvol_1h", "close_loc_1h",
        "upper_wick_ratio", "lower_wick_ratio", "body_ratio", "rsi_14",
        "dist_sma_20", "dist_sma_40", "rel_strength_3d",
        "spy_ret_3d", "spy_ret_5d", "dist_spy_sma200",
        "upside_tail_risk_20d", "rvol_break_power", "marubozu_bear_strength"
    ]
    
    oos_preds = []
    feature_importances = []
    
    for win_name, train_years, test_years in windows:
        train_df = df_events[df_events["year"].isin(train_years)].copy()
        test_df = df_events[df_events["year"].isin(test_years)].copy()
        
        clf = lgb.LGBMClassifier(
            n_estimators=70, learning_rate=0.03, max_depth=3, num_leaves=7,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        clf.fit(train_df[features_base], train_df["target_2a"])
        test_df["prob_2a"] = clf.predict_proba(test_df[features_base])[:, 1]
        test_df["window_name"] = win_name
        oos_preds.append(test_df)
        feature_importances.append(clf.feature_importances_)
        
    oos_df = pd.concat(oos_preds, ignore_index=True)
    
    # Promedio de importancia de features
    avg_fi = np.mean(feature_importances, axis=0)
    fi_df = pd.DataFrame({"Feature": features_base, "Importance": avg_fi}).sort_values("Importance", ascending=False)
    
    print("\n" + "=" * 90)
    print("1. FEATURE IMPORTANCE EN MODELO 2A (LIGHTGBM EN BREAKDOWNS PUROS):")
    print("=" * 90)
    print(fi_df.to_string(index=False))
    
    # 3. Matriz de Confusión Cuantitativa (Terciles de Probabilidad)
    prob_p50 = oos_df["prob_2a"].median()
    oos_df["pred_enter"] = oos_df["prob_2a"] >= prob_p50
    
    tp = oos_df[(oos_df["pred_enter"]) & (oos_df["target_2a"] == 1)] # Quiebres genuinos capturados
    fp = oos_df[(oos_df["pred_enter"]) & (oos_df["target_2a"] == 0)] # TRAMPAS ALCISTAS (Bear Traps)
    fn = oos_df[(~oos_df["pred_enter"]) & (oos_df["target_2a"] == 1)] # Quiebres genuinos ignorados
    tn = oos_df[(~oos_df["pred_enter"]) & (oos_df["target_2a"] == 0)] # Trampas evitadas con éxito
    
    print("\n" + "=" * 90)
    print("2. MATRIZ DE DECISIÓN DE INGRESO (MODELO 2A OOS 2022-2026):")
    print("=" * 90)
    print(f"Total eventos de Breakdown OOS evaluados: {len(oos_df)}")
    print(f"  - True Positives (Entramos y fue quiebre genuino):  {len(tp):4d} ({len(tp)/len(oos_df)*100:4.1f}%)")
    print(f"  - False Positives (Entramos y fue trampa alcista): {len(fp):4d} ({len(fp)/len(oos_df)*100:4.1f}%)  <-- ¡LA FUGA PRINCIPAL!")
    print(f"  - False Negatives (Descartamos y sí cayó):          {len(fn):4d} ({len(fn)/len(oos_df)*100:4.1f}%)")
    print(f"  - True Negatives (Descartamos trampa con éxito):   {len(tn):4d} ({len(tn)/len(oos_df)*100:4.1f}%)")
    
    # -------------------------------------------------------------------------
    # 4. ANATOMÍA FORENSE: FALSE POSITIVES (TRAMPAS) VS TRUE POSITIVES (GENUINOS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("3. ANATOMÍA FORENSE: ¿POR QUÉ FALLAN LOS BREAKDOWNS DONDE ENTRAMOS?")
    print("   Comparativa: True Positives (Genuinos) vs False Positives (Trampas)")
    print("=" * 90)
    
    all_check_cols = features_base + [
        "consec_red_bars", "penetration_pct", "v_acceleration",
        "consolidation_width_atr", "dist_sma_200", "ret_1d_prev_pct"
    ]
    
    diff_rows = []
    for col in all_check_cols:
        v_tp = tp[col].dropna()
        v_fp = fp[col].dropna()
        
        t_stat, p_val = stats.mannwhitneyu(v_tp, v_fp)
        diff_rows.append({
            "Variable": col,
            "Media_Genuinos(TP)": round(v_tp.mean(), 4),
            "Media_Trampas(FP)": round(v_fp.mean(), 4),
            "Diferencia": round(v_tp.mean() - v_fp.mean(), 4),
            "p-value": round(p_val, 4),
            "Significativo": "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.10 else ""))
        })
    diff_df = pd.DataFrame(diff_rows).sort_values("p-value").reset_index(drop=True)
    print(diff_df.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # 5. DIAGNÓSTICO ESPECÍFICO DE CONDICIONES PATOLÓGICAS EN FALSOS QUIEBRES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("4. PATOLOGÍAS CLAVE IDENTIFICADAS EN LAS TRAMPAS ALCISTAS (FP):")
    print("=" * 90)
    
    # Patología A: Sobre-extensión previa (Llegó al soporte "con la lengua afuera")
    print("A. Trampa por Sobreextensión Previa (Vender cuando ya cayó demasiado antes del quiebre):")
    print(f"   - % de Trampas con ret_3d <= -15%:   {(fp['ret_3d'] <= -0.15).mean()*100:.1f}% vs {(tp['ret_3d'] <= -0.15).mean()*100:.1f}% en Genuinos")
    print(f"   - % de Trampas con RSI-14 < 25:      {(fp['rsi_14'] < 25).mean()*100:.1f}% vs {(tp['rsi_14'] < 25).mean()*100:.1f}% en Genuinos")
    print(f"   - % de Trampas con consec_red >= 4:  {(fp['consec_red_bars'] >= 4).mean()*100:.1f}% vs {(tp['consec_red_bars'] >= 4).mean()*100:.1f}% en Genuinos")
    
    # Patología B: Falta de Penetración (El quiebre apenas rozó el soporte y cerró pegado)
    print("\nB. Trampa de Quiebre Tímido (Penetración < 0.3% del soporte):")
    print(f"   - Penetración media en Genuinos: {tp['penetration_pct'].mean():.2f}%")
    print(f"   - Penetración media en Trampas:  {fp['penetration_pct'].mean():.2f}%")
    print(f"   - % de Trampas con penetración < 0.30%: {(fp['penetration_pct'] < 0.30).mean()*100:.1f}% vs {(tp['penetration_pct'] < 0.30).mean()*100:.1f}% en Genuinos")
    
    # Patología C: Divergencia con SPY (El mercado general estaba subiendo mientras la acción intentaba romper)
    print("\nC. Interacción con el SPY (Intentar romper soporte mientras el mercado general rebota):")
    print(f"   - Retorno 3d de SPY en Genuinos: {tp['spy_ret_3d'].mean()*100:+.2f}%")
    print(f"   - Retorno 3d de SPY en Trampas:  {fp['spy_ret_3d'].mean()*100:+.2f}%")
    print(f"   - % de Trampas con SPY en rally 3d (> +1.5%): {(fp['spy_ret_3d'] > 0.015).mean()*100:.1f}% vs {(tp['spy_ret_3d'] > 0.015).mean()*100:.1f}% en Genuinos")
    
    print("=" * 90)

if __name__ == "__main__":
    run_error_analysis_2a()
