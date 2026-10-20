"""
Experimento Cuantitativo: Arquetipo 3 (Rollover / Exhaustion Top) + Gestión con Runner Parcial (50/50)
Valida el impacto en OOS Walk-Forward antes de consolidar en el pipeline de producción.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_motor_2_walkforward_pipeline import (
    PARQUET_IBKR, EARNINGS_PARQUET
)

FEATURES = [
    "atr_d", "ret_3d", "ret_5d", "z_3d", "rvol_1h", "close_loc_1h",
    "upper_wick_ratio", "lower_wick_ratio", "body_ratio", "rsi_14",
    "dist_sma_20", "dist_sma_40", "eps_surprise_pct", "archetype_code",
    "dist_spy_sma200", "spy_ret_3d", "rel_strength_3d",
    "upside_tail_risk_20d", "rvol_break_power", "marubozu_bear_strength"
]

def extract_motor_2_events_v3(df: pd.DataFrame, earn_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[1/4] Extrayendo eventos bajistas causales con Arquetipo 3 (Rollover)...")
    t0 = time.time()
    
    # SPY benchmark
    df_spy = df[df["ticker"] == "SPY"].sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    spy_c = df_spy["close"]
    spy_sma200_d = spy_c.rolling(1400, min_periods=200).mean() # SMA 200 diaria
    spy_ret_3d = spy_c.pct_change(21).fillna(0.0)
    spy_ret_5d = spy_c.pct_change(35).fillna(0.0)
    
    earn_df["earnings_date"] = pd.to_datetime(earn_df["earnings_date"], utc=True)
    earn_map = {}
    for tk, gdf in earn_df.groupby("ticker"):
        earn_map[tk] = gdf.set_index("earnings_date")["eps_surprise_pct"].to_dict()
            
    events = []
    event_id = 0
    tickers = [t for t in df["ticker"].unique() if t not in ["SPY", "QQQ"]]
    
    for ticker in tickers:
        tdf = df[df["ticker"] == ticker].sort_values("timestamp").reset_index(drop=True)
        if len(tdf) < 200:
            continue
            
        c = tdf["close"].to_numpy()
        o = tdf["open"].to_numpy()
        h = tdf["high"].to_numpy()
        l = tdf["low"].to_numpy()
        v = tdf["volume"].to_numpy()
        ts = tdf["timestamp"].to_numpy()
        sec = tdf["sector"].iloc[0] if "sector" in tdf.columns else "General"
        tk_earn = earn_map.get(ticker, {})
        
        sma_20 = pd.Series(c).rolling(20, min_periods=10).mean().to_numpy()
        sma_40 = pd.Series(c).rolling(40, min_periods=20).mean().to_numpy()
        
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        tr = np.insert(tr, 0, h[0] - l[0])
        atr_d = pd.Series(tr).rolling(98, min_periods=20).mean().to_numpy() / (c + 1e-6)
        
        ret_1d = np.zeros(len(tdf))
        ret_1d[7:] = (c[7:] - c[:-7]) / (c[:-7] + 1e-6)
        roll_max_up = pd.Series(ret_1d).rolling(140, min_periods=20).max().fillna(0.05).to_numpy()
        
        ret_3d = np.zeros(len(tdf))
        ret_5d = np.zeros(len(tdf))
        ret_3d[21:] = (c[21:] - c[:-21]) / (c[:-21] + 1e-6)
        ret_5d[35:] = (c[35:] - c[:-35]) / (c[:-35] + 1e-6)
        roll_std_3d = pd.Series(ret_3d).rolling(250, min_periods=50).std().fillna(0.04).to_numpy()
        z_3d = ret_3d / (roll_std_3d + 1e-6)
        
        rng = np.maximum(h - l, 1e-6)
        close_loc = (c - l) / rng
        upper_wick = (h - np.maximum(o, c)) / rng
        lower_wick = (np.minimum(o, c) - l) / rng
        body_ratio = np.abs(c - o) / rng
        rvol_1h = v / (pd.Series(v).rolling(35, min_periods=10).mean().to_numpy() + 1e-6)
        
        delta = pd.Series(c).diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14, min_periods=14).mean() / (loss.rolling(14, min_periods=14).mean() + 1e-9)
        rsi_14 = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0).to_numpy()
        
        for i in range(120, len(tdf) - 20):
            if atr_d[i] < 0.020:
                continue
                
            cur_ts = pd.Timestamp(ts[i])
            
            # TRIGGER 1: PEAD NEGATIVO INMEDIATO
            is_pead = False
            last_surprise = 0.0
            for e_date, surprise in tk_earn.items():
                diff_d = (cur_ts - e_date).total_seconds() / 86400.0
                if 0.0 <= diff_d <= 1.2:
                    gap_down = (o[i] - c[i-1]) / (c[i-1] + 1e-6)
                    if surprise <= -0.02 or (gap_down <= -0.015 and rvol_1h[i] >= 1.3):
                        if c[i] < o[i] or close_loc[i] <= 0.40 or upper_wick[i] >= 0.15:
                            is_pead = True
                            last_surprise = surprise
                            break
                            
            # TRIGGER 2: BREAKDOWN ESTRUCTURAL (40 barras)
            min_prev_40 = np.min(l[max(0, i-40): i])
            is_breakdown = False
            if not is_pead:
                if c[i] < min_prev_40 and rvol_1h[i] >= 1.6 and close_loc[i] <= 0.30:
                    is_breakdown = True
                    
            # TRIGGER 3: ROLLOVER / EXHAUSTION TOP (Reversión desde Techo / Ruptura de Soporte Local de 10 barras)
            is_rollover = False
            if not is_pead and not is_breakdown:
                max_prev_20 = np.max(h[max(0, i-20): i])
                dist_to_max20 = (c[i] - max_prev_20) / max_prev_20
                
                # Acción cerca del techo de los últimos 3 días (a menos de 4.5% del máximo)
                if dist_to_max20 >= -0.045:
                    min_prev_10 = np.min(l[max(0, i-10): i])
                    # Caso A: Rompe el soporte local de las últimas 10 barras con volumen
                    if c[i] < min_prev_10 and rvol_1h[i] >= 1.3 and close_loc[i] <= 0.30 and c[i] < o[i]:
                        is_rollover = True
                    # Caso B: Pin Bar / Rejection masivo en el techo (probó máximo y cerró en mínimos con volumen)
                    elif h[i] >= max_prev_20 * 0.995 and upper_wick[i] >= 0.30 and close_loc[i] <= 0.25 and c[i] < o[i] and rvol_1h[i] >= 1.3:
                        is_rollover = True
                        
            if not (is_pead or is_breakdown or is_rollover):
                continue
                
            archetype_code = 1 if is_pead else (2 if is_breakdown else 3)
            archetype_name = "PEAD_Negative" if is_pead else ("Breakdown_Short" if is_breakdown else "Rollover_Top")
            entry_p = o[i + 1]
            if entry_p <= 0:
                continue
                
            # TARGET ASIMETRICO 2:1 RÁPIDO (14 BARRAS / 2 DÍAS MÁXIMO)
            stop_loss_p = max(h[i], entry_p * (1.0 + 0.7 * atr_d[i]))
            target_drop_p = entry_p * (1.0 - 1.0 * atr_d[i])
            
            fut_h = h[i + 1 : i + 1 + 14]
            fut_l = l[i + 1 : i + 1 + 14]
            
            target_hit = 0
            for step in range(len(fut_h)):
                if fut_l[step] <= target_drop_p:
                    target_hit = 1
                    break
                if fut_h[step] >= stop_loss_p:
                    target_hit = 0
                    break
                    
            spy_val = spy_c.get(cur_ts, np.nan)
            spy_sma_val = spy_sma200_d.get(cur_ts, np.nan)
            dist_spy_sma200 = (spy_val - spy_sma_val) / (spy_sma_val + 1e-6) if not np.isnan(spy_sma_val) else 0.0
            spy_r3 = float(spy_ret_3d.get(cur_ts, 0.0))
            spy_r5 = float(spy_ret_5d.get(cur_ts, 0.0))
            rel_strength_3d = ret_3d[i] - spy_r3
            
            event_id += 1
            events.append({
                "event_id": event_id,
                "ticker": ticker,
                "sector": sec,
                "timestamp": cur_ts,
                "entry_ts": pd.Timestamp(ts[i + 1]),
                "year": cur_ts.year,
                "archetype_code": archetype_code,
                "archetype_name": archetype_name,
                "entry_p": entry_p,
                "target_drop_p": target_drop_p,
                "stop_loss_p": stop_loss_p,
                "target_2a": target_hit,
                # Features
                "atr_d": atr_d[i],
                "ret_3d": ret_3d[i],
                "ret_5d": ret_5d[i],
                "z_3d": z_3d[i],
                "rvol_1h": rvol_1h[i],
                "close_loc_1h": close_loc[i],
                "upper_wick_ratio": upper_wick[i],
                "lower_wick_ratio": lower_wick[i],
                "body_ratio": body_ratio[i],
                "rsi_14": rsi_14[i],
                "dist_sma_20": (c[i] - sma_20[i]) / (sma_20[i] + 1e-6),
                "dist_sma_40": (c[i] - sma_40[i]) / (sma_40[i] + 1e-6),
                "eps_surprise_pct": last_surprise,
                "dist_spy_sma200": dist_spy_sma200,
                "spy_ret_3d": spy_r3,
                "spy_ret_5d": spy_r5,
                "rel_strength_3d": rel_strength_3d,
                "macro_regime": "Bear_Crash" if spy_r3 <= -0.02 else ("Bull_Trend" if spy_r3 >= 0.015 else "Neutral_Chop"),
                # Nuevas features agnósticas
                "upside_tail_risk_20d": roll_max_up[i],
                "rvol_break_power": rvol_1h[i] * (1.0 - close_loc[i]),
                "marubozu_bear_strength": body_ratio[i] * (1.0 - upper_wick[i]) if c[i] < o[i] else 0.0
            })
            
    df_events = pd.DataFrame(events)
    print(f"[OK] {len(df_events):,} eventos extraídos en {time.time() - t0:.1f}s.")
    print("Desglose de eventos por Arquetipo:")
    print(df_events["archetype_name"].value_counts())
    return df_events

def run_walk_forward_ml_v3(df_events: pd.DataFrame):
    windows = [
        ("Ventana 1 (2022)", list(range(2020, 2022)), [2022]),
        ("Ventana 2 (2023)", list(range(2020, 2023)), [2023]),
        ("Ventana 3 (2024)", list(range(2020, 2024)), [2024]),
        ("Ventana 4 (2025-26)", list(range(2020, 2025)), [2025, 2026])
    ]
    
    oos_preds = []
    for win_name, train_years, test_years in windows:
        train_mask = df_events["year"].isin(train_years)
        test_mask = df_events["year"].isin(test_years)
        
        train_df = df_events[train_mask].copy()
        test_df = df_events[test_mask].copy()
        if len(train_df) < 50 or len(test_df) == 0:
            continue
            
        X_train = train_df[FEATURES]
        y_train = train_df["target_2a"]
        X_test = test_df[FEATURES]
        y_test = test_df["target_2a"]
        
        clf = lgb.LGBMClassifier(
            n_estimators=70,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=7,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        clf.fit(X_train, y_train)
        
        test_df = test_df.copy()
        test_df["prob_2a"] = clf.predict_proba(X_test)[:, 1]
        test_df["window_name"] = win_name
        oos_preds.append(test_df)
        
    oos_df = pd.concat(oos_preds, ignore_index=True)
    return oos_df

def simulate_dynamic_trading_with_runners(oos_df: pd.DataFrame, df_raw: pd.DataFrame, runner_ratio=0.50):
    """
    Simula ejecución dinámica con opción de Runner Parcial:
    - 50% de la posición se toma en Target 1.2x ATR.
    - El 50% restante pasa a Trailing Stop (mínimo Breakeven) para capturar colapsos severos.
    """
    t0 = time.time()
    trades = []
    
    # Pre-indexar datos por ticker
    raw_by_tk = {tk: grp.sort_values("timestamp").reset_index(drop=True) for tk, grp in df_raw.groupby("ticker")}
    
    for _, row in oos_df.iterrows():
        tk = row["ticker"]
        tdf = raw_by_tk.get(tk)
        if tdf is None:
            continue
            
        ts_entry = row["entry_ts"]
        sub = tdf[tdf["timestamp"] >= ts_entry].head(15).reset_index(drop=True)
        if len(sub) < 2:
            continue
            
        entry_p = sub.iloc[0]["open"]
        atr = row["atr_d"] * entry_p
        
        # Parámetros dinámicos
        hard_stop_p = max(sub.iloc[0]["high"], entry_p + (0.70 * atr))
        target_p = entry_p - (1.20 * atr)
        be_trigger_p = entry_p - (0.50 * atr)
        
        stop_p = hard_stop_p
        be_active = False
        target_reached = False
        target_step = 0
        
        exit_p_part1 = None
        exit_p_runner = None
        exit_reason_part1 = None
        exit_reason_runner = None
        duration_bars = 14
        
        lowest_since_target = target_p
        
        for step in range(1, len(sub)):
            bar = sub.iloc[step]
            bh, bl, bc, bo = bar["high"], bar["low"], bar["close"], bar["open"]
            rng = max(bh - bl, 1e-6)
            lower_wick = (min(bo, bc) - bl) / rng
            gain_now = (entry_p - bl)
            
            # Si no hemos tocado target aún
            if not target_reached:
                # 1. Stop Loss inicial o Breakeven Stop
                if bh >= stop_p:
                    exit_p_part1 = stop_p
                    exit_p_runner = stop_p
                    exit_reason_part1 = "breakeven_stop" if be_active else "hard_stop"
                    exit_reason_runner = exit_reason_part1
                    duration_bars = step + 1
                    break
                    
                # 2. Target Hit 1.2x ATR
                if bl <= target_p:
                    target_reached = True
                    target_step = step
                    exit_p_part1 = target_p
                    exit_reason_part1 = "target_hit"
                    
                    if runner_ratio == 0.0:
                        # Modo 100% fijo (como antes)
                        exit_p_runner = target_p
                        exit_reason_runner = "target_hit"
                        duration_bars = step + 1
                        break
                    else:
                        # Modo Runner Parcial: Movemos stop del runner a Breakeven inmediatamente
                        stop_p = entry_p # Breakeven para el runner
                        lowest_since_target = bl
                        continue
                        
                # 3. Activación de Breakeven
                if gain_now >= (0.50 * atr) and not be_active:
                    be_active = True
                    stop_p = entry_p * 1.0005 # Breakeven seguro
                    
                # 4. Absorción / Martillo
                if gain_now >= (0.60 * atr) and lower_wick >= 0.40 and bc > bo:
                    exit_p_part1 = bc
                    exit_p_runner = bc
                    exit_reason_part1 = "absorption_hammer"
                    exit_reason_runner = "absorption_hammer"
                    duration_bars = step + 1
                    break
            else:
                # GESTIÓN DEL RUNNER (50% restante)
                lowest_since_target = min(lowest_since_target, bl)
                # Trailing stop del runner: 0.8x ATR por encima del punto más bajo alcanzado, pero nunca peor que Breakeven
                trailing_stop = min(entry_p, lowest_since_target + (0.80 * atr))
                
                # ¿Tocó trailing stop?
                if bh >= trailing_stop:
                    exit_p_runner = trailing_stop
                    exit_reason_runner = "runner_trail_exit"
                    duration_bars = step + 1
                    break
                    
                # ¿Martillo de absorción durante el runner?
                if lower_wick >= 0.40 and bc > bo and (entry_p - bc) >= (1.5 * atr):
                    exit_p_runner = bc
                    exit_reason_runner = "runner_absorption"
                    duration_bars = step + 1
                    break
                    
        # Salida por tiempo (14 barras)
        if exit_p_part1 is None:
            last_c = sub.iloc[-1]["close"]
            exit_p_part1 = last_c
            exit_p_runner = last_c
            exit_reason_part1 = "time_decay_14b"
            exit_reason_runner = "time_decay_14b"
            duration_bars = len(sub)
        elif exit_p_runner is None:
            last_c = sub.iloc[-1]["close"]
            exit_p_runner = last_c
            exit_reason_runner = "runner_time_14b"
            duration_bars = len(sub)
            
        # Retorno combinado
        ret_part1 = (entry_p - exit_p_part1) / entry_p
        ret_runner = (entry_p - exit_p_runner) / entry_p
        
        gross_ret = (1.0 - runner_ratio) * ret_part1 + runner_ratio * ret_runner
        net_ret = gross_ret - 0.0010 # 10 bps
        
        trades.append({
            "event_id": row["event_id"],
            "ticker": tk,
            "sector": row["sector"],
            "archetype_name": row["archetype_name"],
            "year": row["year"],
            "macro_regime": row["macro_regime"],
            "window_name": row["window_name"],
            "entry_ts": ts_entry,
            "exit_ts": sub.iloc[min(len(sub)-1, duration_bars-1)]["timestamp"],
            "duration_bars": duration_bars,
            "prob_2a": row["prob_2a"],
            "target_2a": row["target_2a"],
            "entry_p": entry_p,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "net_ret_pct": net_ret * 100.0,
            "exit_reason": exit_reason_part1 if runner_ratio == 0 else f"{exit_reason_part1}+{exit_reason_runner}",
            "ret_part1": ret_part1,
            "ret_runner": ret_runner
        })
        
    trades_df = pd.DataFrame(trades).sort_values("entry_ts").reset_index(drop=True)
    return trades_df

def simulate_portfolio_slots_v3(trades_df: pd.DataFrame, initial_capital=25000.0, max_slots=5):
    active_slots = []
    slot_size = initial_capital / max_slots
    executed = []
    
    for _, tr in trades_df.iterrows():
        t_entry = tr["entry_ts"]
        t_exit = tr["exit_ts"]
        
        active_slots = [s_exit for s_exit in active_slots if s_exit > t_entry]
        
        if len(active_slots) < max_slots:
            active_slots.append(t_exit)
            pnl_usd = slot_size * tr["net_ret"]
            executed.append({
                **tr.to_dict(),
                "slot_allocated": slot_size,
                "pnl_usd": pnl_usd
            })
            
    exec_df = pd.DataFrame(executed).sort_values("exit_ts").reset_index(drop=True)
    if exec_df.empty:
        return exec_df, {}
        
    exec_df["cum_pnl"] = exec_df["pnl_usd"].cumsum()
    exec_df["equity"] = initial_capital + exec_df["cum_pnl"]
    exec_df["peak"] = exec_df["equity"].cummax()
    exec_df["drawdown_pct"] = (exec_df["equity"] - exec_df["peak"]) / exec_df["peak"] * 100.0
    
    total_trades = len(exec_df)
    winners = exec_df[exec_df["net_ret"] > 0]
    losers = exec_df[exec_df["net_ret"] <= 0]
    win_rate = len(winners) / total_trades if total_trades > 0 else 0.0
    
    sum_win = winners["pnl_usd"].sum()
    sum_loss = abs(losers["pnl_usd"].sum())
    profit_factor = sum_win / sum_loss if sum_loss > 0 else np.nan
    
    net_pnl = exec_df["cum_pnl"].iloc[-1]
    ret_pct = (net_pnl / initial_capital) * 100.0
    max_dd = exec_df["drawdown_pct"].min()
    
    stats = {
        "trades": total_trades,
        "win_rate": win_rate * 100.0,
        "profit_factor": profit_factor,
        "net_pnl": net_pnl,
        "ret_pct": ret_pct,
        "max_dd": max_dd
    }
    return exec_df, stats

def run_experiment():
    print("=" * 90)
    print("INICIANDO EXPERIMENTO: ARQUETIPO 3 (ROLLOVER) + RUNNER PARCIAL (50/50)")
    print("=" * 90)
    
    df_raw = pd.read_parquet(PARQUET_IBKR)
    earn_df = pd.read_parquet(EARNINGS_PARQUET)
    
    df_events = extract_motor_2_events_v3(df_raw, earn_df)
    oos_df = run_walk_forward_ml_v3(df_events)
    
    prob_cutoff = oos_df["prob_2a"].median()
    
    # -------------------------------------------------------------
    # Caso 1: Arquetipo 3 con salida Fija 100% (runner_ratio=0.0)
    # -------------------------------------------------------------
    print("\n--- CASO 1: Con Arquetipo 3 + Salida Fija 100% (-1.2x ATR) ---")
    trades_fixed = simulate_dynamic_trading_with_runners(oos_df, df_raw, runner_ratio=0.0)
    qual_fixed = trades_fixed[trades_fixed["prob_2a"] >= prob_cutoff].copy().reset_index(drop=True)
    exec_fixed, stats_fixed = simulate_portfolio_slots_v3(qual_fixed)
    
    print(f"Trades: {stats_fixed['trades']} | Win Rate: {stats_fixed['win_rate']:.1f}% | PF: {stats_fixed['profit_factor']:.2f}")
    print(f"PnL Total: ${stats_fixed['net_pnl']:+,.0f} USD ({stats_fixed['ret_pct']:+.1f}%) | Max DD: {stats_fixed['max_dd']:.2f}%")
    
    # Desglose por Arquetipo en Caso 1
    print("\nDesglose por Arquetipo (Caso 1):")
    for arch, grp in exec_fixed.groupby("archetype_name"):
        wr = (grp['net_ret'] > 0).mean() * 100
        pnl = grp['pnl_usd'].sum()
        print(f"  {arch:<20} -> Trades: {len(grp):<4} | WinRate: {wr:.1f}% | PnL: ${pnl:+,.0f} USD")
        
    # -------------------------------------------------------------
    # Caso 2: Con Arquetipo 3 + RUNNER PARCIAL (50% Take Profit / 50% Trailing Runner)
    # -------------------------------------------------------------
    print("\n--- CASO 2: Con Arquetipo 3 + RUNNER PARCIAL (50/50) ---")
    trades_runner = simulate_dynamic_trading_with_runners(oos_df, df_raw, runner_ratio=0.50)
    qual_runner = trades_runner[trades_runner["prob_2a"] >= prob_cutoff].copy().reset_index(drop=True)
    exec_runner, stats_runner = simulate_portfolio_slots_v3(qual_runner)
    
    print(f"Trades: {stats_runner['trades']} | Win Rate: {stats_runner['win_rate']:.1f}% | PF: {stats_runner['profit_factor']:.2f}")
    print(f"PnL Total: ${stats_runner['net_pnl']:+,.0f} USD ({stats_runner['ret_pct']:+.1f}%) | Max DD: {stats_runner['max_dd']:.2f}%")
    
    # Desglose por Arquetipo en Caso 2
    print("\nDesglose por Arquetipo (Caso 2 con Runner):")
    for arch, grp in exec_runner.groupby("archetype_name"):
        wr = (grp['net_ret'] > 0).mean() * 100
        pnl = grp['pnl_usd'].sum()
        print(f"  {arch:<20} -> Trades: {len(grp):<4} | WinRate: {wr:.1f}% | PnL: ${pnl:+,.0f} USD")
        
    # Desglose Anual en Caso 2
    print("\nAtribución Anual (Caso 2 con Runner):")
    for yr, grp in exec_runner.groupby("year"):
        wr = (grp['net_ret'] > 0).mean() * 100
        pnl = grp['pnl_usd'].sum()
        print(f"  Año {yr} -> Trades: {len(grp):<4} | WinRate: {wr:.1f}% | PnL: ${pnl:+,.0f} USD")
    print("=" * 90)

if __name__ == "__main__":
    run_experiment()
