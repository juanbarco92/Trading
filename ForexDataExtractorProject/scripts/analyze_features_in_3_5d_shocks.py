"""
analyze_features_in_3_5d_shocks.py - Radiografía Multivariada de las 584 Features en Golpes de 3 a 5 Días.

Objetivo:
Responder a la pregunta clave:
Cuando aislamos el momento del "Golpe Duro de 3 a 5 días" (z_3d <= -2.0 o z_5d <= -2.0),
¿qué pasa con las 584 variables construidas en el motor de features?
¿Cuáles de nuestras variables separan los golpes que SE RECUPERAN FUERTE de los que SIGUEN CAYENDO?
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


def deduplicate_events(df_events: pd.DataFrame, time_col: str, min_gap_hours: int = 72) -> pd.DataFrame:
    """Deja al menos min_gap_hours entre episodios de un mismo ticker."""
    kept = []
    for ticker, grp in df_events.sort_values(time_col).groupby("ticker"):
        last_time = None
        for idx, row in grp.iterrows():
            t = row[time_col]
            if last_time is None or (t - last_time).total_seconds() / 3600.0 >= min_gap_hours:
                kept.append(idx)
                last_time = t
    return df_events.loc[kept].sort_values(time_col).reset_index(drop=True)


def main():
    print("=" * 95)
    print("RADIOGRAFÍA DE LAS 584 FEATURES DENTRO DE LOS GOLPES DUROS DE 3 A 5 DÍAS")
    print("=" * 95)

    feat_path = PROJECT_ROOT / "data" / "processed" / "features_universe_full_structural.parquet"
    df = pd.read_parquet(feat_path)
    time_col = "timestamp" if "timestamp" in df.columns else "datetime"
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(["ticker", time_col]).reset_index(drop=True)

    # Extraer SPY para Alpha
    spy = df[df["ticker"] == "SPY"].sort_values(time_col).set_index(time_col)
    spy_open_next = spy["open"].shift(-1)
    spy_fut_10d = (spy["close"].shift(-70) - spy_open_next) / spy_open_next * 100.0
    spy_fut_15d = (spy["close"].shift(-105) - spy_open_next) / spy_open_next * 100.0
    spy_ret_3d = spy["close"].pct_change(21)

    # Filtrar solo acciones individuales
    stocks = df[~df["ticker"].isin(["SPY", "QQQ"])].copy().reset_index(drop=True)
    stocks["spy_fut_10d"] = stocks[time_col].map(spy_fut_10d)
    stocks["spy_fut_15d"] = stocks[time_col].map(spy_fut_15d)
    stocks["spy_ret_3d"] = stocks[time_col].map(spy_ret_3d)

    g = stocks.groupby("ticker", group_keys=False)
    stocks["ret_3d"] = g["close"].pct_change(21)
    stocks["ret_5d"] = g["close"].pct_change(35)
    stocks["vol_3d"] = g["ret_3d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["vol_5d"] = g["ret_5d"].transform(lambda s: s.rolling(420, min_periods=70).std())

    stocks["z_3d"] = stocks["ret_3d"] / stocks["vol_3d"].replace(0, np.nan)
    stocks["z_5d"] = stocks["ret_5d"] / stocks["vol_5d"].replace(0, np.nan)

    # Desenlace futuro desde Open[t+1]
    open_next = g["open"].shift(-1)
    stocks["fut_ret_10d"] = (g["close"].shift(-70) - open_next) / open_next * 100.0
    stocks["fut_ret_15d"] = (g["close"].shift(-105) - open_next) / open_next * 100.0
    stocks["fut_alpha_10d"] = stocks["fut_ret_10d"] - stocks["spy_fut_10d"]
    stocks["fut_alpha_15d"] = stocks["fut_ret_15d"] - stocks["spy_fut_15d"]

    # Excursión máxima adversa en 10 días (para medir si no colapsa)
    fut_min_low_10d = g["low"].transform(lambda s: s.shift(-1).rolling(70, min_periods=35).min().shift(-69))
    stocks["mae_10d_pct"] = (fut_min_low_10d - open_next) / open_next * 100.0

    # Target de Recuperación Exitosa en 10-15 días:
    # 1 si supera al SPY con Alpha > +1.5% y Retorno absoluto > +2.0%, 0 en caso contrario
    stocks["target_recovery"] = ((stocks["fut_alpha_10d"] > 1.0) & (stocks["fut_ret_10d"] > 2.0)).astype(int)

    # Aislar el universo de Golpes Duros de 3 a 5 días (z_3d <= -2.0 o z_5d <= -2.0)
    shock_mask = (stocks["z_3d"] <= -2.0) | (stocks["z_5d"] <= -2.0)
    shocks_all = stocks[shock_mask].dropna(subset=["fut_ret_10d", "fut_alpha_10d"]).copy()
    shocks_dedup = deduplicate_events(shocks_all, time_col, min_gap_hours=48)

    print(f"Total barras en golpe de 3-5 días: {len(shocks_all):,}")
    print(f"Total episodios deduplicados (min 48h): {len(shocks_dedup):,}")
    print(f"Tasa base de Recuperación Fuerte (Ret > +2% y Alpha > +1% en 10d): {shocks_dedup['target_recovery'].mean()*100:.1f}%\n")

    # Identificar las 584 features construidas por el usuario
    exclude_prefixes = ("target_", "trade_", "short_", "fut_", "spy_fut", "mae_", "mfe_")
    exclude_exact = {
        "timestamp", "datetime", "open", "high", "low", "close", "volume", "barCount",
        "ticker", "candle_cluster", "pattern", "ret_3d", "ret_5d", "vol_3d", "vol_5d", "z_3d", "z_5d"
    }

    feature_cols = [
        c for c in shocks_dedup.columns
        if c not in exclude_exact
        and not c.startswith(exclude_prefixes)
        and pd.api.types.is_numeric_dtype(shocks_dedup[c])
    ]
    print(f"Analizando poder predictivo de {len(feature_cols)} features construidas dentro del evento...\n")

    # =========================================================================
    # 1. RANKING UNIVARIADO DE LAS 584 FEATURES DENTRO DEL GOLPE DE 3-5 DÍAS
    # =========================================================================
    results = []
    y_rec = shocks_dedup["target_recovery"].values
    ret_10d = shocks_dedup["fut_ret_10d"].values
    alpha_10d = shocks_dedup["fut_alpha_10d"].values

    for col in feature_cols:
        s = shocks_dedup[col].replace([np.inf, -np.inf], np.nan)
        if s.isna().mean() > 0.30 or s.nunique() < 2:
            continue
        s_filled = s.fillna(s.median())
        try:
            auc_raw = roc_auc_score(y_rec, s_filled)
            # Poder discriminante sin importar el signo (si es <0.5 invierte la dirección)
            auc_power = max(auc_raw, 1.0 - auc_raw)
            direction = "MAYOR es mejor (+)" if auc_raw >= 0.5 else "MENOR es mejor (-)"
            corr_ret = np.corrcoef(s_filled, ret_10d)[0, 1]
            corr_alpha = np.corrcoef(s_filled, alpha_10d)[0, 1]

            # Diferencia entre el cuartil favorable y el cuartil desfavorable
            q75 = np.percentile(s_filled, 75)
            q25 = np.percentile(s_filled, 25)
            if auc_raw >= 0.5:
                best_q_ret = ret_10d[s_filled >= q75].mean()
                worst_q_ret = ret_10d[s_filled <= q25].mean()
                best_q_wr = y_rec[s_filled >= q75].mean() * 100.0
                worst_q_wr = y_rec[s_filled <= q25].mean() * 100.0
            else:
                best_q_ret = ret_10d[s_filled <= q25].mean()
                worst_q_ret = ret_10d[s_filled >= q75].mean()
                best_q_wr = y_rec[s_filled <= q25].mean() * 100.0
                worst_q_wr = y_rec[s_filled >= q75].mean() * 100.0

            results.append({
                "Feature": col,
                "AUC_Univariado": round(auc_power, 4),
                "Direccion": direction,
                "Corr_Ret10d": round(corr_ret, 4),
                "Corr_Alpha10d": round(corr_alpha, 4),
                "Ret10d_Mejor_Q %": round(best_q_ret, 2),
                "Ret10d_Peor_Q %": round(worst_q_ret, 2),
                "Spread_Ret %": round(best_q_ret - worst_q_ret, 2),
                "WR_Mejor_Q %": round(best_q_wr, 1),
                "WR_Peor_Q %": round(worst_q_wr, 1),
            })
        except Exception:
            continue

    df_res = pd.DataFrame(results).sort_values("AUC_Univariado", ascending=False).reset_index(drop=True)
    print("=" * 95)
    print("TOP 25 VARIABLES DE TU MOTOR QUE MÁS SEPARAN EL ÉXITO DEL FRACASO EN GOLPES DE 3-5 DÍAS:")
    print("=" * 95)
    print(df_res.head(25).to_string(index=False))

    # =========================================================================
    # 2. COMPARATIVA DE CORRELACIÓN: EN RUIDO CONTINUO VS DENTRO DEL GOLPE 3-5D
    # =========================================================================
    print("\n" + "=" * 95)
    print("2. ¿POR QUÉ ANTES PARECÍAN RUIDO Y EN EL GOLPE DE 3-5 DÍAS COBRAN VIDA?")
    print("=" * 95)
    top_10_cols = df_res.head(10)["Feature"].tolist()
    comp_rows = []
    valid_all = stocks.dropna(subset=["fut_ret_10d"])
    for col in top_10_cols:
        c_all = abs(valid_all[col].corr(valid_all["fut_ret_10d"]))
        c_shock = abs(shocks_dedup[col].corr(shocks_dedup["fut_ret_10d"]))
        ratio = c_shock / c_all if c_all > 0 else 1.0
        comp_rows.append({
            "Feature": col,
            "|Corr| en 172k Velas Continuas": round(c_all, 4),
            "|Corr| Dentro del Golpe 3-5 Días": round(c_shock, 4),
            "Multiplicador de Señal": f"{ratio:.1f}x más fuerte",
        })
    print(pd.DataFrame(comp_rows).to_string(index=False))

    # =========================================================================
    # 3. PRUEBA DE MODELO MULTIVARIADO (SPLIT TEMPORAL CIEGO: TRAIN <=2024 vs TEST 2025-2026)
    # =========================================================================
    print("\n" + "=" * 95)
    print("3. PRUEBA MULTIVARIADA OUT-OF-SAMPLE (ENTRENANDO SOLO EN LOS GOLPES DE 3-5 DÍAS)")
    print("=" * 95)

    # Seleccionar top 35 features pero eliminando precios absolutos (que no son estacionarios entre acciones)
    non_stationary_keywords = ["_upper", "_lower", "_middle", "_price", "_support", "_resistance", "_closest", "_trend_line", "_channel", "_ma", "SMA_20", "SMA_40", "SMA_100", "EMA_"]
    stationary_features = [
        f for f in df_res["Feature"].tolist()
        if not any(f.endswith(kw) for kw in ["_upper", "_lower", "_middle", "_price", "_support", "_resistance", "_closest", "_trend_line", "_channel"])
        and f not in ["SMA_20", "SMA_40", "SMA_100", "EMA_12", "EMA_26", "ATR_14", "ATR_21"]
    ]
    selected_feats = stationary_features[:30]
    print(f"Top 15 Features Estacionarias Seleccionadas:\n  {selected_feats[:15]}\n")

    train_mask = shocks_dedup[time_col] < pd.Timestamp("2025-01-01", tz="UTC")
    test_mask = shocks_dedup[time_col] >= pd.Timestamp("2025-01-15", tz="UTC")

    tr_df = shocks_dedup[train_mask].copy()
    te_df = shocks_dedup[test_mask].copy()

    X_tr = tr_df[selected_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_tr = tr_df["target_recovery"].values

    X_te = te_df[selected_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_te = te_df["target_recovery"].values

    clf = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    probs_te = clf.predict_proba(X_te)[:, 1]
    auc_oos = roc_auc_score(y_te, probs_te)

    print(f"Train Set (2022-2024): {len(tr_df)} eventos | Test Set OOS (2025-2026): {len(te_df)} eventos")
    print(f"ROC-AUC Out-of-Sample (2025-2026): {auc_oos:.4f}\n")

    te_df["prob"] = probs_te
    te_df["quintil"] = pd.qcut(te_df["prob"], q=5, labels=False, duplicates="drop")

    q_summary = te_df.groupby("quintil").agg(
        eventos=("target_recovery", "count"),
        prob_min=("prob", lambda s: round(s.min(), 3)),
        prob_max=("prob", lambda s: round(s.max(), 3)),
        exito_fuerte_pct=("target_recovery", lambda s: round(s.mean() * 100, 1)),
        win_rate_ret10d=("fut_ret_10d", lambda s: round((s > 0).mean() * 100, 1)),
        ret_10d_medio=("fut_ret_10d", lambda s: round(s.mean(), 2)),
        alpha_10d_medio=("fut_alpha_10d", lambda s: round(s.mean(), 2)),
        ret_15d_medio=("fut_ret_15d", lambda s: round(s.mean(), 2)),
    ).reset_index()

    print("--- ORDENAMIENTO POR QUINTILES EN TEST OUT-OF-SAMPLE (2025-2026) ---")
    print(q_summary.to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    main()
