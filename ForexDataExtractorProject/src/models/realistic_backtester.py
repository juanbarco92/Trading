"""
realistic_backtester.py - TICK-17: Motor de Backtesting con Ejecución Causal en Open[t+1].

Elimina 100% el sesgo de barra simultánea (same-bar bias):
1. Las señales calculadas al cierre de la barra t se ejecutan OBLIGATORIAMENTE en Open[t+1].
2. Se modela el deslizamiento real y gaps de apertura entre Close[t] y Open[t+1].
3. La evaluación de barreras de Take Profit y Stop Loss comienza únicamente a partir de la barra t+1.
4. Resolución pesimista intra-vela: si en una misma barra se tocan TP y SL, se asume siempre Stop Loss.
5. Fricción institucional de 10 bps round-trip y slippage en gaps de apertura.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def run_realistic_portfolio_backtest(
    df: pd.DataFrame,
    threshold_buy: float = 0.45,
    threshold_sell: float = 0.45,
    initial_capital: float = 25000.0,
    max_position_size: float = 8000.0,
    risk_per_trade_pct: float = 0.015,   # 1.5% de riesgo por trade
    max_open_positions: int = 6,
    friction_pct: float = 0.0010,        # 10 bps round-trip
    filter_morning_traps: bool = True,
    max_horizon_bars: int = 35,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta una simulación de cartera institucional con ejecución estricta en Open[t+1].
    """
    time_col = "timestamp" if "timestamp" in df.columns else "datetime"
    df_sorted = df.copy()
    df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
    df_sorted = df_sorted.sort_values(time_col).reset_index(drop=True)

    # Identificar señales al cierre de la vela t
    signals_buy = (df_sorted["prob_buy"] >= threshold_buy)
    signals_sell = (df_sorted["prob_sell"] >= threshold_sell)

    if filter_morning_traps and "hour_of_day" in df_sorted.columns:
        # Filtrar trampas de apertura matutina (13-14 UTC / 9:30-10:30 EST)
        signals_buy = signals_buy & (df_sorted["hour_of_day"] > 14)
        signals_sell = signals_sell & (df_sorted["hour_of_day"] > 14)

    df_sorted["signal_buy"] = signals_buy.astype(int)
    df_sorted["signal_sell"] = signals_sell.astype(int)

    # Diccionario de barras ordenadas por ticker para acceso O(1)
    # y mapeo cronológico por timestamps
    unique_timestamps = df_sorted[time_col].drop_duplicates().sort_values().to_numpy()
    time_grouped = df_sorted.groupby(time_col)

    capital = initial_capital
    peak_capital = initial_capital
    open_positions: List[Dict] = []
    closed_trades: List[Dict] = []
    pending_orders: List[Dict] = []  # Órdenes generadas en t para ejecutarse en t+1

    for step_idx, current_time in enumerate(unique_timestamps):
        current_bars = time_grouped.get_group(current_time)
        bar_dict = {row["ticker"]: row for _, row in current_bars.iterrows()}

        # -------------------------------------------------------------
        # PASO 1: Ejecutar órdenes pendientes generadas en la barra t-1
        # Se ejecutan exactamente en Open[t] de la barra actual
        # -------------------------------------------------------------
        for order in pending_orders:
            ticker = order["ticker"]
            if ticker in bar_dict:
                bar = bar_dict[ticker]
                open_price = bar["open"]
                side = order["side"]
                atr_val = order["atr"]

                # Precio real de ejecución con fricción
                if side == "BUY":
                    entry_price = open_price * (1.0 + friction_pct / 2.0)
                    tp_price = entry_price + 3.5 * atr_val
                    sl_price = entry_price - 1.8 * atr_val
                else:
                    entry_price = open_price * (1.0 - friction_pct / 2.0)
                    tp_price = entry_price - 3.5 * atr_val
                    sl_price = entry_price + 1.8 * atr_val

                # Calcular tamaño de posición
                sl_dist_pct = abs(entry_price - sl_price) / entry_price
                target_pos_size = min(capital * (risk_per_trade_pct / max(sl_dist_pct, 0.01)), max_position_size)
                target_pos_size = min(target_pos_size, capital - 100.0)

                if target_pos_size >= 500.0 and len(open_positions) < max_open_positions:
                    capital -= target_pos_size
                    open_positions.append({
                        "ticker": ticker,
                        "side": side,
                        "entry_time": current_time,
                        "entry_price": entry_price,
                        "tp_price": tp_price,
                        "sl_price": sl_price,
                        "position_size_usd": target_pos_size,
                        "bars_held": 0,
                    })

        pending_orders = []  # Limpiar órdenes ejecutadas

        # -------------------------------------------------------------
        # PASO 2: Gestionar salidas de posiciones activas en la barra t
        # -------------------------------------------------------------
        remaining_positions = []
        for pos in open_positions:
            ticker = pos["ticker"]
            if ticker in bar_dict:
                bar = bar_dict[ticker]
                pos["bars_held"] += 1
                b_high = bar["high"]
                b_low = bar["low"]
                b_open = bar["open"]
                b_close = bar["close"]

                exit_triggered = False
                exit_price = b_close
                exit_reason = "TIME_EXPIRY"

                if pos["side"] == "BUY":
                    # Chequear si tocan ambas barreras en la misma barra -> Pesimista: SL
                    touches_tp = (b_high >= pos["tp_price"])
                    touches_sl = (b_low <= pos["sl_price"])

                    if touches_tp and touches_sl:
                        # Ambigüedad resuelta a favor del peor caso (Stop Loss)
                        exit_price = min(pos["sl_price"], b_open) * (1.0 - friction_pct / 2.0)
                        exit_reason = "STOP_LOSS_AMBIGUOUS"
                        exit_triggered = True
                    elif touches_sl:
                        exit_price = min(pos["sl_price"], b_open) * (1.0 - friction_pct / 2.0)
                        exit_reason = "STOP_LOSS"
                        exit_triggered = True
                    elif touches_tp:
                        exit_price = pos["tp_price"] * (1.0 - friction_pct / 2.0)
                        exit_reason = "TAKE_PROFIT"
                        exit_triggered = True
                    elif pos["bars_held"] >= max_horizon_bars:
                        exit_price = b_close * (1.0 - friction_pct / 2.0)
                        exit_reason = "TIME_EXPIRY"
                        exit_triggered = True

                    if exit_triggered:
                        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
                        pnl_usd = pos["position_size_usd"] * pnl_pct
                        capital += pos["position_size_usd"] + pnl_usd
                        closed_trades.append({
                            "ticker": ticker,
                            "side": "BUY",
                            "entry_time": pos["entry_time"],
                            "exit_time": current_time,
                            "entry_price": pos["entry_price"],
                            "exit_price": exit_price,
                            "pnl_pct": pnl_pct * 100.0,
                            "pnl_usd": pnl_usd,
                            "exit_reason": exit_reason,
                            "bars_held": pos["bars_held"],
                            "capital_after": capital,
                        })
                    else:
                        remaining_positions.append(pos)

                elif pos["side"] == "SELL":
                    touches_tp = (b_low <= pos["tp_price"])
                    touches_sl = (b_high >= pos["sl_price"])

                    if touches_tp and touches_sl:
                        exit_price = max(pos["sl_price"], b_open) * (1.0 + friction_pct / 2.0)
                        exit_reason = "STOP_LOSS_AMBIGUOUS"
                        exit_triggered = True
                    elif touches_sl:
                        exit_price = max(pos["sl_price"], b_open) * (1.0 + friction_pct / 2.0)
                        exit_reason = "STOP_LOSS"
                        exit_triggered = True
                    elif touches_tp:
                        exit_price = pos["tp_price"] * (1.0 + friction_pct / 2.0)
                        exit_reason = "TAKE_PROFIT"
                        exit_triggered = True
                    elif pos["bars_held"] >= max_horizon_bars:
                        exit_price = b_close * (1.0 + friction_pct / 2.0)
                        exit_reason = "TIME_EXPIRY"
                        exit_triggered = True

                    if exit_triggered:
                        pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]
                        pnl_usd = pos["position_size_usd"] * pnl_pct
                        capital += pos["position_size_usd"] + pnl_usd
                        closed_trades.append({
                            "ticker": ticker,
                            "side": "SELL",
                            "entry_time": pos["entry_time"],
                            "exit_time": current_time,
                            "entry_price": pos["entry_price"],
                            "exit_price": exit_price,
                            "pnl_pct": pnl_pct * 100.0,
                            "pnl_usd": pnl_usd,
                            "exit_reason": exit_reason,
                            "bars_held": pos["bars_held"],
                            "capital_after": capital,
                        })
                    else:
                        remaining_positions.append(pos)
            else:
                remaining_positions.append(pos)

        open_positions = remaining_positions

        # -------------------------------------------------------------
        # PASO 3: Evaluar señales al cierre de la barra t
        # Las señales aprobadas se programan para ejecutarse en Open[t+1]
        # -------------------------------------------------------------
        available_slots = max_open_positions - len(open_positions)
        if available_slots > 0 and capital > 1000.0:
            open_tickers = {p["ticker"] for p in open_positions}
            
            b_cands = current_bars[current_bars["signal_buy"] == 1].sort_values("prob_buy", ascending=False)
            s_cands = current_bars[current_bars["signal_sell"] == 1].sort_values("prob_sell", ascending=False)

            for _, r in b_cands.iterrows():
                if available_slots <= 0:
                    break
                t_cand = r["ticker"]
                if t_cand not in open_tickers:
                    atr_cand = max(r.get("atr14", r["close"] * 0.02), 0.01)
                    pending_orders.append({
                        "ticker": t_cand,
                        "side": "BUY",
                        "signal_time": current_time,
                        "atr": atr_cand,
                    })
                    open_tickers.add(t_cand)
                    available_slots -= 1

            for _, r in s_cands.iterrows():
                if available_slots <= 0:
                    break
                t_cand = r["ticker"]
                if t_cand not in open_tickers:
                    atr_cand = max(r.get("atr14", r["close"] * 0.02), 0.01)
                    pending_orders.append({
                        "ticker": t_cand,
                        "side": "SELL",
                        "signal_time": current_time,
                        "atr": atr_cand,
                    })
                    open_tickers.add(t_cand)
                    available_slots -= 1

    # Liquidación al final
    for pos in open_positions:
        ticker = pos["ticker"]
        fin_bar = df_sorted[df_sorted["ticker"] == ticker].iloc[-1]
        exit_p = fin_bar["close"]
        pnl_pct = (exit_p - pos["entry_price"]) / pos["entry_price"] if pos["side"] == "BUY" else (pos["entry_price"] - exit_p) / pos["entry_price"]
        pnl_usd = pos["position_size_usd"] * pnl_pct
        capital += pos["position_size_usd"] + pnl_usd
        closed_trades.append({
            "ticker": ticker,
            "side": pos["side"],
            "entry_time": pos["entry_time"],
            "exit_time": fin_bar[time_col],
            "entry_price": pos["entry_price"],
            "exit_price": exit_p,
            "pnl_pct": pnl_pct * 100.0,
            "pnl_usd": pnl_usd,
            "exit_reason": "FINAL_CLOSE",
            "bars_held": pos["bars_held"],
            "capital_after": capital,
        })

    trades_df = pd.DataFrame(closed_trades)
    
    # Resumen Anual
    annual_summary = []
    if len(trades_df) > 0:
        trades_df["exit_year"] = pd.to_datetime(trades_df["exit_time"]).dt.year
        for year, grp in trades_df.groupby("exit_year"):
            y_wins = grp[grp["pnl_usd"] > 0]
            y_losses = grp[grp["pnl_usd"] <= 0]
            y_wr = len(y_wins) / len(grp) * 100.0
            y_pnl = grp["pnl_usd"].sum()
            y_pf = y_wins["pnl_usd"].sum() / abs(y_losses["pnl_usd"].sum()) if len(y_losses) > 0 and abs(y_losses["pnl_usd"].sum()) > 0 else 99.0
            annual_summary.append({
                "Año": year,
                "Trades": len(grp),
                "Win Rate %": round(y_wr, 2),
                "PnL USD": round(y_pnl, 2),
                "Profit Factor": round(y_pf, 2),
                "Retorno Medio Trade %": round(grp["pnl_pct"].mean(), 2),
            })

    annual_df = pd.DataFrame(annual_summary)
    return trades_df, annual_df
