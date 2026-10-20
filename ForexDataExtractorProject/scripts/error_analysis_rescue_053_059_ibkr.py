"""
Palanca #2: Error Analysis en la Zona Gris (`0.53 <= prob_1a < 0.59`) para Rescatar Señales Ganadoras
empujando su probabilidad por encima de `>= 0.59` (vs simplemente bajar el threshold).

Preguntas que responde este script (100% en Producción Estricta IBKR):
1. RADIOGRAFÍA DE ERROR ANALYSIS EN LA ZONA `0.53 <= prob_1a < 0.59`:
   - Dentro de las 493 señales que quedaron entre `0.53` y `0.59`, ¿qué diferencia exactamente a las
     JOYAS OCULTAS (que ganan >= +3.0% en IBKR) de las TRAMPAS (que pierden dinero / tocan Stop Loss)?
2. NUEVAS VARIABLES 100% CAUSALES EN IBKR DESCUBIERTAS EN EL ERROR ANALYSIS:
   - `rel_strength_60d_vs_spy`: Liderazgo secular previo de 60 días frente al SPY (`ret_60d - spy_ret_60d`).
     (Los fondos institucionales compran agresivamente el dip en líderes que venían ganándole al SPY,
      pero abandonan a las rezagadas crónicas).
   - `floor_defense_3h`: Defensa institucional del piso en las últimas 3 horas (cuántas de las últimas 3 velas
     horarias dejaron mecha inferior / cerraron en la mitad superior sin romper el mínimo previo).
   - `bounce_thrust_2h`: Impulso de giro en las últimas 2 horas acompañado de volumen relativo.
   - `atr_norm_drop_3d`: Magnitud de la caída de 3 días medida en múltiplos de su propio ATR diario (`|ret_3d| / atr_d`).
     (Una caída de 3.5x su ATR diario tiene mucha más energía elástica de rebote que una de 1.5x ATR).
   - `vix_cooling_3h`: Si el `^VIX` ya dejó de acelerar en las últimas 3 horas mientras la acción defiende soporte.
3. COMPARACIÓN DIRECTA EN PORTAFOLIO IBKR ($25,000):
   - A) No hacer nada (Corte fijo 0.59 actual: 63 trades, +31.6%).
   - B) Simplemente bajar el threshold a 0.55 a lo bruto (para ver el efecto de meter ruido y llenar cupos).
   - C) Re-entrenar Modelo 1A con las variables del Error Analysis (empujando las Joyas de 0.53-0.58 a >= 0.59).
   - D) Modelo 1A Mejorado + Asignación de Cupos en Dos Niveles (Tiered Slots en IBKR: hasta 4 cupos para >= 0.575,
        y el 5to/6to cupo reservado exclusivamente para Francotiradores >= 0.60).
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_portfolio_motor1_dual_ml import run_portfolio_simulation
from scripts.upgrade_model_1b_first_passage_and_perception import simulate_calibrated_1b_ibkr_exit


def run_tiered_ibkr_portfolio(
    signals_df: pd.DataFrame,
    initial_capital: float = 25000.0,
    max_total_slots: int = 5,
    sniper_reserved_slots: int = 1,
    sniper_prob_thresh: float = 0.605,
):
    """
    Simulador de Portafolio IBKR con Reserva de Cupos para Francotirador (Tiered Slot Allocation):
    - Si llega una señal estándar (`prob_1a < sniper_prob_thresh`), solo puede entrar si hay menos de
      `(max_total_slots - sniper_reserved_slots)` posiciones abiertas.
    - Los últimos `sniper_reserved_slots` cupos del portafolio están reservados exclusivamente para señales
      Francotirador (`prob_1a >= sniper_prob_thresh`), evitando que una señal de convicción media secuestre
      el último cupo antes de que llegue una oportunidad extraordinaria!
    """
    df_sorted = signals_df.sort_values(["entry_ts", "prob_1a"], ascending=[True, False]).reset_index(drop=True)

    cash = initial_capital
    open_positions = []
    closed_trades = []
    equity_curve = []

    for _, row in df_sorted.iterrows():
        cur_ts = pd.Timestamp(row["entry_ts"])

        # 1. Cerrar posiciones que ya terminaron antes o en `cur_ts`
        still_open = []
        for pos in open_positions:
            if pos["exit_ts"] <= cur_ts:
                cash += pos["returned_cash"]
                closed_trades.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. Verificar si hay cupo según el nivel de convicción de `prob_1a`
        is_sniper = float(row["prob_1a"]) >= sniper_prob_thresh
        allowed_slots = max_total_slots if is_sniper else (max_total_slots - sniper_reserved_slots)

        if len(open_positions) >= allowed_slots:
            continue
        if any(p["ticker"] == row["ticker"] for p in open_positions):
            continue

        slots_free = max_total_slots - len(open_positions)
        if slots_free <= 0:
            continue
        alloc = min(cash / slots_free, (cash + sum(p["invested"] for p in open_positions)) / max_total_slots)
        if alloc < 500 or cash < alloc:
            continue

        cash -= alloc
        ret_net = float(row["net_ret_pct"]) / 100.0
        returned = alloc * (1.0 + ret_net)

        open_positions.append({
            "ticker": row["ticker"],
            "entry_ts": cur_ts,
            "exit_ts": pd.Timestamp(row["exit_ts"]),
            "invested": alloc,
            "returned_cash": returned,
            "net_ret_pct": float(row["net_ret_pct"]),
            "prob_1a": float(row["prob_1a"]),
        })
        equity_curve.append({"ts": cur_ts, "equity": cash + sum(p["invested"] for p in open_positions)})

    for pos in sorted(open_positions, key=lambda x: x["exit_ts"]):
        cash += pos["returned_cash"]
        closed_trades.append(pos)
        equity_curve.append({"ts": pos["exit_ts"], "equity": cash})

    if not closed_trades:
        return {"trades_executed": 0, "win_rate_pct": 0.0, "avg_trade_ret_pct": 0.0, "profit_factor": 0.0,
                "final_capital": initial_capital, "total_return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}

    tdf = pd.DataFrame(closed_trades)
    wins = tdf[tdf["net_ret_pct"] > 0]["net_ret_pct"].sum()
    losses = abs(tdf[tdf["net_ret_pct"] <= 0]["net_ret_pct"].sum())
    pf = wins / losses if losses > 0 else 9.99

    eq_df = pd.DataFrame(equity_curve).sort_values("ts").reset_index(drop=True)
    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"] * 100.0
    max_dd = float(eq_df["dd"].min())

    eq_df["day"] = eq_df["ts"].dt.floor("D")
    daily_eq = eq_df.groupby("day")["equity"].last().pct_change().dropna()
    sharpe = float((daily_eq.mean() / daily_eq.std()) * np.sqrt(252)) if len(daily_eq) > 5 and daily_eq.std() > 0 else 0.0

    return {
        "trades_executed": len(tdf),
        "win_rate_pct": float((tdf["net_ret_pct"] > 0).mean() * 100.0),
        "avg_trade_ret_pct": float(tdf["net_ret_pct"].mean()),
        "profit_factor": float(pf),
        "final_capital": float(cash),
        "total_return_pct": float((cash - initial_capital) / initial_capital * 100.0),
        "max_dd_pct": max_dd,
        "sharpe": sharpe,
    }


def main():
    t0 = time.time()
    cache_1a_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_1a.parquet"
    cache_meli_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_meli.parquet"
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    macro_path = PROJECT_ROOT / "data" / "processed" / "macro_perception_1hour.parquet"

    print("[1/4] Cargando datos y calculando nuevas variables candidatas de microestructura IBKR...")
    df_1a = pd.read_parquet(cache_1a_path)
    df_meli = pd.read_parquet(cache_meli_path)

    df_raw = pd.read_parquet(hourly_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    df_raw["hour_key"] = df_raw["timestamp"].dt.floor("h")

    macro_df = pd.read_parquet(macro_path)
    macro_df["hour_key"] = pd.to_datetime(macro_df["hour_key"], utc=True)
    macro_df = macro_df.set_index("hour_key")

    for mcol in ["vix", "vix_ret_3d", "vix_slope_3h", "tnx_ret_3d"]:
        df_raw[mcol] = df_raw["hour_key"].map(macro_df[mcol]).ffill().fillna(0.0)

    # Mapa de retorno de 60 días del SPY para calcular Liderazgo Secular Previo (`rel_strength_60d_vs_spy`)
    spy_df = df_raw[df_raw["ticker"] == "SPY"].sort_values("timestamp").drop_duplicates("hour_key").set_index("hour_key")
    spy_ret_60d_map = spy_df["close"].pct_change(420).fillna(0.0)

    ticker_dfs = {
        tk: gdf.sort_values("timestamp").reset_index(drop=True)
        for tk, gdf in df_raw[~df_raw["ticker"].isin(["SPY", "QQQ"])].groupby("ticker")
    }

    # Calcular nuevas variables 100% causales al cierre de la barra de señal `i0` para cada candidato en `df_1a`
    rel_str_60d_list = []
    atr_norm_drop_3d_list = []
    floor_defense_3h_list = []
    bounce_thrust_2h_list = []
    green_vol_ratio_5d_list = []
    elastic_potential_list = []

    for _, row in df_1a.iterrows():
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        h_key = pd.Timestamp(row["timestamp"]).floor("h")
        spy_r60 = float(spy_ret_60d_map.get(h_key, 0.0))

        # 1. Liderazgo secular previo de 60 días frente al SPY (`ret_60d - spy_ret_60d`)
        rel_str_60d = float(row["ret_60d"]) - spy_r60

        # 2. Caída de 3 días normalizada por su propio ATR diario (`|ret_3d| / atr_d`)
        atr_d = max(float(row["atr_d"]), 0.015)
        atr_norm_drop = abs(float(row["ret_3d"])) / atr_d

        # 3. Defensa institucional del piso en las últimas 3 velas horarias (`i0-2, i0-1, i0`)
        win3 = tdf.iloc[max(0, i0 - 2) : i0 + 1]
        rng3 = (win3["high"] - win3["low"]).replace(0, np.nan)
        cloc3 = ((win3["close"] - win3["low"]) / rng3).fillna(0.5)
        lwick3 = ((np.minimum(win3["open"], win3["close"]) - win3["low"]) / rng3).fillna(0.0)
        # Promedio de absorción en las últimas 3 horas (no solo la última hora)
        floor_def_3h = float((cloc3 + lwick3).mean())

        # 4. Impulso de giro en las últimas 2 horas (`bounce_thrust_2h`)
        if i0 >= 2:
            ret_2h = (float(tdf.loc[i0, "close"]) - float(tdf.loc[i0 - 2, "low"])) / (float(tdf.loc[i0 - 2, "low"]) + 1e-9)
            thrust_2h = (ret_2h / atr_d) * float(row["rvol_1h"])
        else:
            thrust_2h = 0.0

        # 5. Proporción de volumen en velas verdes durante los últimos 5 días (35 barras)
        win35 = tdf.iloc[max(0, i0 - 34) : i0 + 1]
        g_vol = win35[win35["close"] >= win35["open"]]["volume"].sum()
        t_vol = win35["volume"].sum() + 1e-9
        green_v_ratio = float(g_vol / t_vol)

        # 6. Potencial elástico combinado: Caída en múltiplos de ATR * Escudo Fundamental * Defensa de Piso 3H
        elastic_pot = atr_norm_drop * float(row["fundamental_shield"]) * floor_def_3h

        rel_str_60d_list.append(rel_str_60d)
        atr_norm_drop_3d_list.append(atr_norm_drop)
        floor_defense_3h_list.append(floor_def_3h)
        bounce_thrust_2h_list.append(thrust_2h)
        green_vol_ratio_5d_list.append(green_v_ratio)
        elastic_potential_list.append(elastic_pot)

    df_1a["rel_strength_60d_vs_spy"] = rel_str_60d_list
    df_1a["atr_norm_drop_3d"] = atr_norm_drop_3d_list
    df_1a["floor_defense_3h"] = floor_defense_3h_list
    df_1a["bounce_thrust_2h"] = bounce_thrust_2h_list
    df_1a["green_vol_ratio_5d"] = green_vol_ratio_5d_list
    df_1a["elastic_potential"] = elastic_potential_list

    # También preparar el Nuevo Modelo 1B (`ROC-AUC = 0.6324`) para simular todas las salidas con precisión
    ev_1a_lookup = df_1a.set_index("event_id")
    fp_targets, vix_cooling_list, vix_slope_50_list, tnx_50_list = [], [], [], []
    down_up_entry_list, sys_pull_entry_list, fund_shield_entry_list = [], [], []

    for _, mrow in df_meli.iterrows():
        ev_id = int(mrow["event_id"])
        r1a = ev_1a_lookup.loc[ev_id]
        tk = r1a["ticker"]
        i0 = int(r1a["bar_idx"])
        trig_step = int(r1a["meli_trigger_step"])
        entry_p = float(r1a["entry_p"])
        shock_drop = float(r1a["shock_drop"])
        tdf = ticker_dfs[tk]

        vix_at_entry = float(r1a["vix"]) if "vix" in r1a else 20.0
        cur_bar_idx = min(len(tdf) - 1, i0 + 1 + trig_step)
        cur_bar = tdf.iloc[cur_bar_idx]
        vix_at_50 = float(cur_bar["vix"])
        vix_cooling_list.append((vix_at_50 - vix_at_entry) / max(vix_at_entry, 10.0))
        vix_slope_50_list.append(float(cur_bar["vix_slope_3h"]))
        tnx_50_list.append(float(cur_bar["tnx_ret_3d"]))
        down_up_entry_list.append(float(r1a["down_up_vol_ratio_15d"]))
        sys_pull_entry_list.append(float(r1a["systemic_pull_z_3d"]))
        fund_shield_entry_list.append(float(r1a["fundamental_shield"]))

        fut_rem = tdf.iloc[i0 + 1 + trig_step + 1 : i0 + 106]
        rem_h = fut_rem["high"].to_numpy()
        rem_c = fut_rem["close"].to_numpy()
        target_85_p = entry_p * (1.0 + 0.85 * shock_drop)
        stop_be_p = entry_p * (1.0 + 0.05 * shock_drop)
        fp_win = 0
        for h_val, c_val in zip(rem_h, rem_c):
            if h_val >= target_85_p:
                fp_win = 1
                break
            if c_val <= stop_be_p:
                fp_win = 0
                break
        fp_targets.append(fp_win)

    df_meli["keep_running_fp"] = fp_targets
    df_meli["vix_cooling_since_entry"] = vix_cooling_list
    df_meli["vix_slope_at_50"] = vix_slope_50_list
    df_meli["tnx_ret_3d_at_50"] = tnx_50_list
    df_meli["entry_down_up_vol_ratio"] = down_up_entry_list
    df_meli["entry_systemic_pull_z_3d"] = sys_pull_entry_list
    df_meli["entry_fundamental_shield"] = fund_shield_entry_list

    feats_base = [
        "SMA_100_pct_momentum", "SMA_40_pct_momentum", "close_diff_SMA_100", "close_diff_SMA_40",
        "ratio_SMA_40_SMA_100", "ratio_SMA_100_SMA_40", "ratio_SMA_100_SMA_10",
        "norm_diff_SMA_40_SMA_100", "norm_diff_SMA_10_SMA_100", "slope_SMA_100",
        "sr_dist_support_pct", "sr_dist_resistance_pct", "sr_dist_support_atr", "sr_dist_resistance_atr",
        "sr_support_touches", "sr_resistance_touches", "sr_channel_position",
        "ATR_14_pct", "ATR_21_pct", "ATR_21_relative", "BB_20_20_bandwidth", "BB_20_20_percent_b",
        "RSI_14", "close_loc_1h", "lower_wick_ratio", "body_ratio", "rvol_1h",
        "z_3d", "z_5d", "ret_60d", "spy_ret_3d",
        "days_since_earnings", "eps_surprise_pct", "eps_positive", "eps_beat_4q_ratio", "eps_surprise_4q_mean",
        "spy_ret_5d", "sector_ret_3d", "sector_ret_5d",
        "excess_vs_spy_3d", "excess_vs_sector_3d", "sector_vs_spy_3d",
        "idiosyncratic_ratio_3d", "is_earnings_window", "earnings_window_x_surprise",
        "is_bad_earnings_shock", "is_good_earnings_shock", "shock_type_code",
        "spy_z_3d", "sector_z_3d", "systemic_pull_ret_3d", "systemic_pull_z_3d",
        "pre_shock_trend_55d", "rvol_3h_max", "inst_absorption_score", "weak_volume_flag",
        "support_touch_density", "fundamental_shield", "sector_pull_x_fundamentals", "sector_pull_x_absorption",
        "last_earnings_reaction_1d", "last_earnings_reaction_3d", "guidance_disappointment_flag",
        "cmf_15d", "down_up_vol_ratio_15d",
    ]
    new_rescue_feats = [
        "rel_strength_60d_vs_spy",
        "atr_norm_drop_3d",
        "floor_defense_3h",
        "bounce_thrust_2h",
        "green_vol_ratio_5d",
        "elastic_potential",
    ]
    feats_upgraded_1a = feats_base + new_rescue_feats

    test_start_ts = pd.Timestamp("2025-11-17", tz="UTC")
    train_end_ts = test_start_ts - pd.Timedelta(days=22)

    tr_1a = df_1a[df_1a["timestamp"] <= train_end_ts].copy().reset_index(drop=True)
    te_1a = df_1a[df_1a["timestamp"] >= test_start_ts].copy().reset_index(drop=True)
    target_col = "target_1a_strict" if "target_1a_strict" in tr_1a.columns else "target_1a"

    # 1. Entrenar Modelo 1A Base (sin las nuevas features) y obtener Out-of-Fold en Train para hacer el Error Analysis SIN FUGA!
    kf = KFold(n_splits=5, shuffle=False)
    oof_base = np.zeros(len(tr_1a))
    for idx_tr, idx_val in kf.split(tr_1a):
        m_f = lgb.LGBMClassifier(
            n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=12,
            min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
        )
        m_f.fit(tr_1a.iloc[idx_tr][feats_base].fillna(0.0), tr_1a.iloc[idx_tr][target_col].values)
        oof_base[idx_val] = m_f.predict_proba(tr_1a.iloc[idx_val][feats_base].fillna(0.0))[:, 1]
    tr_1a["prob_1a_base"] = oof_base

    clf_1a_base = lgb.LGBMClassifier(
        n_estimators=160, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1a_base.fit(tr_1a[feats_base].fillna(0.0), tr_1a[target_col].values)
    te_1a["prob_1a_base"] = clf_1a_base.predict_proba(te_1a[feats_base].fillna(0.0))[:, 1]
    auc_1a_base = roc_auc_score(te_1a[target_col].values, te_1a["prob_1a_base"].values)

    # =========================================================================
    # PASO 1: ERROR ANALYSIS EN TRAIN (Out-of-Fold) SOBRE LA ZONA GRIS `0.53 <= prob_1a_base < 0.59`
    # =========================================================================
    grey_train = tr_1a[(tr_1a["prob_1a_base"] >= 0.53) & (tr_1a["prob_1a_base"] < 0.59)].copy()
    gems_train = grey_train[grey_train[target_col] == 1]
    traps_train = grey_train[grey_train[target_col] == 0]

    print("\n" + "=" * 124)
    print(f"1. ERROR ANALYSIS EN LA ZONA GRIS (`0.53 <= prob_1a < 0.59` en Train OOF: {len(grey_train)} casos):")
    print(f"   ¿Qué diferencia a las {len(gems_train)} JOYAS OCULTAS (Target=1) de las {len(traps_train)} TRAMPAS (Target=0)?")
    print("=" * 124)
    diag_cols = [
        ("atr_norm_drop_3d", "Caída de 3d en múltiplos de ATR (|ret_3d|/ATR)"),
        ("elastic_potential", "Potencial Elástico (Caída ATR x Fundamentales x Defensa 3H)"),
        ("rel_strength_60d_vs_spy", "Liderazgo Previo 60d vs SPY (ret_60d - spy_ret_60d)"),
        ("floor_defense_3h", "Defensa Institucional del Piso en 3 Horas"),
        ("bounce_thrust_2h", "Impulso de Giro en últimas 2 Horas"),
        ("systemic_pull_z_3d", "Arrastre Sistémico / Sectorial Z-Score"),
        ("fundamental_shield", "Escudo Fundamental (EPS + Racha 4Q + Guía)"),
    ]
    ea_rows = []
    for col, desc in diag_cols:
        m_gem = float(gems_train[col].mean())
        m_trap = float(traps_train[col].mean())
        diff_pct = (m_gem - m_trap) / (abs(m_trap) + 1e-6) * 100.0
        ea_rows.append({
            "Variable_Candidata": col,
            "Descripción": desc,
            "Media_Joyas_Ganadoras": round(m_gem, 4),
            "Media_Trampas_Perdedoras": round(m_trap, 4),
            "Diferencia_%": f"{diff_pct:+.1f}%",
        })
    print(pd.DataFrame(ea_rows).to_string(index=False))

    # =========================================================================
    # PASO 2: RE-ENTRENAR MODELO 1A CON LAS NUEVAS VARIABLES DEL ERROR ANALYSIS
    # =========================================================================
    clf_1a_new = lgb.LGBMClassifier(
        n_estimators=165, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=45, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1a_new.fit(tr_1a[feats_upgraded_1a].fillna(0.0), tr_1a[target_col].values)
    te_1a["prob_1a_new"] = clf_1a_new.predict_proba(te_1a[feats_upgraded_1a].fillna(0.0))[:, 1]
    auc_1a_new = roc_auc_score(te_1a[target_col].values, te_1a["prob_1a_new"].values)

    # Verificar cuántas Joyas de `0.53-0.59` fueron promovidas a `>= 0.59` en Test OOS!
    grey_test = te_1a[(te_1a["prob_1a_base"] >= 0.53) & (te_1a["prob_1a_base"] < 0.59)].copy()
    promoted = grey_test[grey_test["prob_1a_new"] >= 0.59]
    demoted = grey_test[grey_test["prob_1a_new"] < 0.53]

    print("\n" + "=" * 124)
    print(f"2. EFECTO DEL RE-ENTRENAMIENTO DEL MODELO 1A (ROC-AUC OOS: {auc_1a_base:.4f} -> {auc_1a_new:.4f}):")
    print(f"   * De las {len(grey_test)} señales en la zona gris (0.53-0.59) en Test OOS:")
    print(f"     - PROMOVIDAS a `prob_1a_new >= 0.59`: {len(promoted)} señales (Tasa de Éxito Target_1A = {promoted[target_col].mean()*100:.1f}%)")
    print(f"     - EXPULSADAS hacia abajo (`< 0.53`):  {len(demoted)} señales (Tasa de Éxito Target_1A = {demoted[target_col].mean()*100:.1f}%)")
    print("=" * 124)

    # Entrenar Nuevo Modelo 1B (`ROC-AUC = 0.6324`) y simular las salidas estrictas en Test OOS
    prob_1a_map = {int(r["event_id"]): float(r["prob_1a_base"]) for _, r in pd.concat([tr_1a, te_1a]).iterrows()}
    df_meli["model_1a_prob"] = df_meli["event_id"].map(prob_1a_map).fillna(0.5)

    feats_1b_upgraded = [
        "model_1a_prob", "shock_type_code",
        "steps_to_reach_50", "velocity_pct_per_bar", "RSI_14_at_50", "rsi_slope_3h_at_50",
        "rvol_1h_at_50", "close_loc_1h_at_50", "upper_wick_at_50",
        "sr_dist_res_atr_at_50", "sr_channel_pos_at_50", "BB_percent_b_at_50", "shock_drop",
        "close_diff_SMA_100_at_50",
        "spy_rebound_since_entry", "sector_rebound_since_entry", "rebound_excess_vs_sector",
        "entry_z_3d", "entry_z_5d", "entry_spy_ret_3d", "entry_sector_ret_3d",
        "entry_excess_vs_sector_3d", "entry_idiosyncratic_ratio",
        "entry_days_since_earnings", "entry_eps_surprise_pct", "entry_eps_surprise_4q_mean",
        "entry_is_bad_earnings", "entry_is_good_earnings",
        "entry_SMA_100_momentum", "entry_close_diff_SMA_100",
        "entry_last_earnings_reaction_3d", "entry_cmf_15d",
        "vix_cooling_since_entry", "vix_slope_at_50", "tnx_ret_3d_at_50",
        "entry_down_up_vol_ratio", "entry_systemic_pull_z_3d", "entry_fundamental_shield"
    ]
    tr_m = df_meli[df_meli["timestamp"] <= train_end_ts].copy().reset_index(drop=True)
    te_m = df_meli[df_meli["timestamp"] >= test_start_ts].copy().reset_index(drop=True)

    clf_1b_new = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=35, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b_new.fit(tr_m[feats_1b_upgraded].fillna(0.0), tr_m["keep_running_fp"].values)
    prob_new_tr = clf_1b_new.predict_proba(tr_m[feats_1b_upgraded].fillna(0.0))[:, 1]
    prob_new_te = clf_1b_new.predict_proba(te_m[feats_1b_upgraded].fillna(0.0))[:, 1]
    q70_1b = float(np.quantile(prob_new_tr, 0.70))
    q80_1b = float(np.quantile(prob_new_tr, 0.80))
    prob_1b_new_map = {int(r["event_id"]): float(p) for r, p in zip(te_m.to_dict("records"), prob_new_te)}

    # Simular la salida estricta (Palanca #1: Cierre Horario -> Open[t+1] con Nuevo 1B) para todos los candidatos de Test
    base_shield_mask = (
        (te_1a["weak_volume_flag"] == 0)
        & ~((te_1a["last_earnings_reaction_3d"] <= -0.05) & (te_1a["cmf_15d"] <= -0.05))
        & ~((te_1a["tnx_ret_3d"] >= 0.030) & (te_1a["vix_slope_3h"] >= 0.04))
    )
    te_clean = te_1a[base_shield_mask].copy().reset_index(drop=True)

    sim_rows = []
    for _, row in te_clean.iterrows():
        ev_id = int(row["event_id"])
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        fut_df = tdf.iloc[i0 + 1 : i0 + 106]
        trig_step = int(row["meli_trigger_step"])
        p_1b = prob_1b_new_map.get(ev_id, 0.50)

        r_net, x_ts, x_rsn = simulate_calibrated_1b_ibkr_exit(
            fut_df, float(row["entry_p"]), float(row["sl_p"]), float(row["atr_d"]),
            float(row["shock_drop"]), int(row["shock_type_code"]), p_1b, trig_step,
            q25_1b=q70_1b, q75_1b=q80_1b, mode="calibrated_hourly_close"
        )
        d = row.to_dict()
        d.update({"prob_1b": p_1b, "net_ret_pct": r_net, "exit_ts": x_ts, "exit_reason": x_rsn})
        sim_rows.append(d)

    df_sim = pd.DataFrame(sim_rows).sort_values("entry_ts").reset_index(drop=True)

    # Crear dos copias: una donde `prob_1a` es `prob_1a_base` y otra donde `prob_1a` es `prob_1a_new`
    df_base_1a = df_sim.copy()
    df_base_1a["prob_1a"] = df_base_1a["prob_1a_base"]

    df_new_1a = df_sim.copy()
    df_new_1a["prob_1a"] = df_new_1a["prob_1a_new"]

    print("\n" + "=" * 128)
    print("3. COMPARACIÓN EN PORTAFOLIO ESTRICTO IBKR ($25,000, 10 bps): ¿Bajar Threshold vs Error Analysis + Nuevas Variables?")
    print("=" * 128)

    # Evaluar las opciones
    p_configs = [
        ("1. Modelo 1A Base con Corte Fijo >= 0.59 (5 Slots)",
         run_portfolio_simulation(df_base_1a[df_base_1a["prob_1a"] >= 0.59], 25000.0, 5),
         len(df_base_1a[df_base_1a["prob_1a"] >= 0.59])),
        ("2. OPCIÓN BRUTA: Simplemente bajar el threshold a >= 0.55 en Modelo Base (5 Slots)",
         run_portfolio_simulation(df_base_1a[df_base_1a["prob_1a"] >= 0.55], 25000.0, 5),
         len(df_base_1a[df_base_1a["prob_1a"] >= 0.55])),
        ("3. ERROR ANALYSIS (Nuevo 1A con Liderazgo 60d + Elasticidad ATR) Corte >= 0.59 (5 Slots)",
         run_portfolio_simulation(df_new_1a[df_new_1a["prob_1a"] >= 0.59], 25000.0, 5),
         len(df_new_1a[df_new_1a["prob_1a"] >= 0.59])),
        ("4. ERROR ANALYSIS (Nuevo 1A) Corte >= 0.58 + Arquetipo 1.1 (5 Slots)",
         run_portfolio_simulation(df_new_1a[(df_new_1a["prob_1a"] >= 0.59) | ((df_new_1a["prob_1a"] >= 0.575) & (df_new_1a["shock_type_code"].isin([3, 4])))], 25000.0, 5),
         len(df_new_1a[(df_new_1a["prob_1a"] >= 0.59) | ((df_new_1a["prob_1a"] >= 0.575) & (df_new_1a["shock_type_code"].isin([3, 4])))])),
        ("5. ERROR ANALYSIS + CUPOS ESCALONADOS (5 Slots: 4 Estándar >=0.575 + 1 Reservado Francotirador >=0.605)",
         run_tiered_ibkr_portfolio(df_new_1a[(df_new_1a["prob_1a"] >= 0.59) | ((df_new_1a["prob_1a"] >= 0.575) & (df_new_1a["shock_type_code"].isin([3, 4])))], 25000.0, max_total_slots=5, sniper_reserved_slots=1, sniper_prob_thresh=0.605),
         len(df_new_1a[(df_new_1a["prob_1a"] >= 0.59) | ((df_new_1a["prob_1a"] >= 0.575) & (df_new_1a["shock_type_code"].isin([3, 4])))])),
        ("6. ERROR ANALYSIS + CUPOS ESCALONADOS (6 Slots: 4 Estándar >=0.575 + 2 Reservados Francotirador >=0.60)",
         run_tiered_ibkr_portfolio(df_new_1a[(df_new_1a["prob_1a"] >= 0.59) | ((df_new_1a["prob_1a"] >= 0.575) & (df_new_1a["shock_type_code"].isin([3, 4])))], 25000.0, max_total_slots=6, sniper_reserved_slots=2, sniper_prob_thresh=0.600),
         len(df_new_1a[(df_new_1a["prob_1a"] >= 0.59) | ((df_new_1a["prob_1a"] >= 0.575) & (df_new_1a["shock_type_code"].isin([3, 4])))])),
    ]

    out_rows = []
    for label, res, n_sig in p_configs:
        out_rows.append({
            "Estrategia_de_Rescate_IBKR": label,
            "Señales": n_sig,
            "Trades": res["trades_executed"],
            "Trades/Mes": round(res["trades_executed"] / 11.0, 1),
            "WR %": round(res["win_rate_pct"], 1),
            "Ret_Medio %": round(res["avg_trade_ret_pct"], 2),
            "PF": round(res["profit_factor"], 2),
            "Capital_Final": f"${res['final_capital']:,.0f}",
            "Retorno_11m": f"+{res['total_return_pct']:.1f}%",
            "Max_DD %": f"{res['max_dd_pct']:.2f}%",
            "Sharpe": round(res["sharpe"], 2),
        })
    print(pd.DataFrame(out_rows).to_string(index=False))
    print("=" * 128)
    print(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
