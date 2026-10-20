"""
run_expanded_swing_backtest.py - Master Backtest Institucional del Universo Expandido (21 Activos)
Evalúa combinaciones de Sizing ($1R=$1,000 vs $2R=$2,000), Long-Only vs Bidireccional (Long+Short),
Gestión de Breakeven (+1.8x ATR), y desglose anual de rendimiento (CAGR / EA).
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def simulate_portfolio(
    df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    risk_per_trade_usd: float = 2_000.0,
    max_open_positions: int = 7,
    max_pos_capital_pct: float = 0.25,
    friction_pct: float = 0.0005,
    k_tp: float = 3.5,
    k_sl: float = 1.8,
    use_breakeven: bool = True,
    breakeven_atr: float = 1.8,
    max_holding_bars: int = 35,
    friday_filter: bool = True,
) -> Tuple[pd.DataFrame, List[dict], dict]:
    """
    Simulador portfolio bar-by-bar con gestión de riesgo institucional,
    soporte Long y Short, y activación de Breakeven.
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    cash = initial_capital
    active_positions: Dict[str, dict] = {}
    closed_trades: List[dict] = []
    equity_records: List[dict] = []

    grouped = df_sorted.groupby("timestamp", sort=False)

    for current_time, bar_group in grouped:
        prices_now = dict(zip(bar_group["ticker"], bar_group["close"]))
        highs_now = dict(zip(bar_group["ticker"], bar_group["high"]))
        lows_now = dict(zip(bar_group["ticker"], bar_group["low"]))
        opens_now = dict(zip(bar_group["ticker"], bar_group["open"]))

        # 1. GESTIÓN DE POSICIONES ABIERTAS
        tickers_to_close = []

        for ticker, pos in active_positions.items():
            if ticker not in highs_now:
                continue

            h = highs_now[ticker]
            l = lows_now[ticker]
            o = opens_now[ticker]
            c = prices_now[ticker]
            atr = pos["atr"]

            pos["bars_held"] += 1
            direction = pos["direction"]

            exit_triggered = False
            raw_exit_price = c
            exit_type = "TIME"

            if direction == 1:  # LONG
                if use_breakeven and not pos["is_breakeven"]:
                    if h >= pos["entry_price_raw"] + (breakeven_atr * atr):
                        pos["sl_price"] = max(pos["sl_price"], pos["entry_price_raw"])
                        pos["is_breakeven"] = True

                if l <= pos["sl_price"]:
                    exit_triggered = True
                    raw_exit_price = min(pos["sl_price"], o)
                    exit_type = "BREAKEVEN" if pos["is_breakeven"] and raw_exit_price >= pos["entry_price_raw"] else "SL"
                elif h >= pos["tp_price"]:
                    exit_triggered = True
                    raw_exit_price = max(pos["tp_price"], o)
                    exit_type = "TP"
                elif pos["bars_held"] >= max_holding_bars:
                    exit_triggered = True
                    raw_exit_price = c
                    exit_type = "TIME"

            else:  # SHORT
                if use_breakeven and not pos["is_breakeven"]:
                    if l <= pos["entry_price_raw"] - (breakeven_atr * atr):
                        pos["sl_price"] = min(pos["sl_price"], pos["entry_price_raw"])
                        pos["is_breakeven"] = True

                if h >= pos["sl_price"]:
                    exit_triggered = True
                    raw_exit_price = max(pos["sl_price"], o)
                    exit_type = "BREAKEVEN" if pos["is_breakeven"] and raw_exit_price <= pos["entry_price_raw"] else "SL"
                elif l <= pos["tp_price"]:
                    exit_triggered = True
                    raw_exit_price = min(pos["tp_price"], o)
                    exit_type = "TP"
                elif pos["bars_held"] >= max_holding_bars:
                    exit_triggered = True
                    raw_exit_price = c
                    exit_type = "TIME"

            if exit_triggered:
                tickers_to_close.append((ticker, raw_exit_price, exit_type))

        # Procesar cierres
        for ticker, raw_exit_price, exit_type in tickers_to_close:
            pos = active_positions.pop(ticker)
            direction = pos["direction"]
            shares = pos["shares"]

            if direction == 1:
                exit_price_net = raw_exit_price * (1.0 - friction_pct)
                pnl_usd = (exit_price_net - pos["entry_price_net"]) * shares
            else:
                exit_price_net = raw_exit_price * (1.0 + friction_pct)
                pnl_usd = (pos["entry_price_net"] - exit_price_net) * shares

            cash += pos["margin_locked"] + pnl_usd
            r_mult = pnl_usd / pos["risk_usd"] if pos["risk_usd"] > 0 else 0.0

            closed_trades.append({
                "ticker": ticker,
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_time": pos["entry_time"],
                "exit_time": current_time,
                "entry_price": pos["entry_price_net"],
                "exit_price": exit_price_net,
                "exit_type": exit_type,
                "shares": shares,
                "pnl_usd": pnl_usd,
                "r_multiple": r_mult,
                "holding_bars": pos["bars_held"],
            })

        # 2. GESTIÓN DE ENTRADAS
        is_friday_afternoon = (current_time.dayofweek == 4) and (current_time.hour >= 18)
        can_open_new = not (friday_filter and is_friday_afternoon)

        if can_open_new and len(active_positions) < max_open_positions:
            available_slots = max_open_positions - len(active_positions)

            candidate_rows = []
            for _, row in bar_group.iterrows():
                sig = row.get("signal_dir", 0)
                if sig != 0 and row["ticker"] not in active_positions:
                    candidate_rows.append(row)

            # Priorizar por RVOL y Convicción
            candidate_rows = sorted(
                candidate_rows,
                key=lambda r: r.get("prob_calibrated", 0.5) if r.get("signal_dir", 0) == 1 else r.get("prob_short_calibrated", 0.5),
                reverse=True,
            )[:available_slots]

            for row in candidate_rows:
                ticker = row["ticker"]
                direction = int(row["signal_dir"])
                close_p = row["close"]
                atr = row.get("atr14", close_p * 0.02)

                entry_price_raw = close_p
                entry_price_net = close_p * (1.0 + (friction_pct if direction == 1 else -friction_pct))

                stop_dist = max(k_sl * atr, entry_price_net * 0.005)
                shares_by_risk = risk_per_trade_usd / stop_dist
                max_cap = min(initial_capital * max_pos_capital_pct, cash)
                shares_by_cap = max_cap / entry_price_net
                shares = int(min(shares_by_risk, shares_by_cap))

                margin_req = shares * entry_price_net

                if shares > 0 and margin_req <= cash:
                    cash -= margin_req

                    if direction == 1:
                        sl_p = entry_price_raw - (k_sl * atr)
                        tp_p = entry_price_raw + (k_tp * atr)
                    else:
                        sl_p = entry_price_raw + (k_sl * atr)
                        tp_p = entry_price_raw - (k_tp * atr)

                    active_positions[ticker] = {
                        "direction": direction,
                        "entry_time": current_time,
                        "entry_price_raw": entry_price_raw,
                        "entry_price_net": entry_price_net,
                        "sl_price": sl_p,
                        "tp_price": tp_p,
                        "atr": atr,
                        "shares": shares,
                        "risk_usd": risk_per_trade_usd,
                        "invested_capital": margin_req,
                        "margin_locked": margin_req,
                        "bars_held": 0,
                        "is_breakeven": False,
                    }

        # 3. VALORACIÓN DE CARTERA
        open_val = 0.0
        for ticker, pos in active_positions.items():
            curr_p = prices_now.get(ticker, pos["entry_price_net"])
            if pos["direction"] == 1:
                open_val += pos["shares"] * curr_p
            else:
                pnl = (pos["entry_price_net"] - curr_p) * pos["shares"]
                open_val += pos["margin_locked"] + pnl

        tot_eq = cash + open_val
        equity_records.append({
            "timestamp": current_time,
            "cash": cash,
            "open_positions": len(active_positions),
            "total_equity": tot_eq,
        })

    eq_df = pd.DataFrame(equity_records)
    if eq_df.empty:
        return eq_df, closed_trades, {}

    # Calcular métricas
    final_eq = eq_df["total_equity"].iloc[-1]
    net_profit = final_eq - initial_capital
    ret_pct = (net_profit / initial_capital) * 100.0

    eq_df["peak"] = eq_df["total_equity"].cummax()
    eq_df["drawdown"] = (eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]
    max_dd = float(eq_df["drawdown"].min() * 100.0)

    # Duración en años
    t_start = eq_df["timestamp"].iloc[0]
    t_end = eq_df["timestamp"].iloc[-1]
    years = max((t_end - t_start).days / 365.25, 0.5)
    cagr = ((final_eq / initial_capital) ** (1.0 / years) - 1.0) * 100.0

    # Retornos diarios para Sharpe
    daily_eq = eq_df.set_index("timestamp")["total_equity"].resample("1D").last().dropna()
    daily_rets = daily_eq.pct_change().dropna()
    sharpe = float((daily_rets.mean() / (daily_rets.std() + 1e-9)) * np.sqrt(252)) if len(daily_rets) > 0 else 0.0

    # Métricas de trades
    n_trades = len(closed_trades)
    if n_trades > 0:
        trades_df = pd.DataFrame(closed_trades)
        wins = trades_df[trades_df["pnl_usd"] > 0]
        losses = trades_df[trades_df["pnl_usd"] <= 0]
        win_rate = (len(wins) / n_trades) * 100.0
        tot_win = wins["pnl_usd"].sum()
        tot_loss = abs(losses["pnl_usd"].sum())
        pf = tot_win / tot_loss if tot_loss > 0 else 999.0
        exp_r = trades_df["r_multiple"].mean()
    else:
        win_rate, pf, exp_r = 0.0, 0.0, 0.0

    metrics = {
        "retorno_total_pct": round(ret_pct, 2),
        "ganancia_neta_usd": round(net_profit, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_trades": n_trades,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 2),
        "expectativa_r": round(exp_r, 2),
    }

    return eq_df, closed_trades, metrics


def main():
    print("=" * 85)
    print("BACKTESTING MAESTRO: UNIVERSO EXPANDIDO (21 ACTIVOS: 2022 - 2026)")
    print("=" * 85)

    data_path = Path("data/processed/labeled_universe_1hour.parquet")
    oos_path = Path("data/processed/oos_predictions.parquet")

    df = pd.read_parquet(data_path).dropna(subset=["target_quality_trade"]).reset_index(drop=True)
    if oos_path.exists():
        oos_df = pd.read_parquet(oos_path)[["timestamp", "ticker", "prob_calibrated", "prob_short_calibrated"]]
        df = df.merge(oos_df, on=["timestamp", "ticker"], how="left")
    else:
        df["prob_calibrated"] = 0.33
        df["prob_short_calibrated"] = 0.30

    print(f"Cargadas {len(df):,} velas horarias en 21 activos.")

    # Generación de Señales Cuantitativas Institucionales
    # Long: Convicción ML Long + Fuerza Relativa (Alpha > 0) + Mercado General Alcista
    long_mask = (
        (df["prob_calibrated"] >= 0.35) &
        (df["alpha_5d"] > 0) &
        (df["is_market_bullish"] == 1) &
        (df["rvol"] > 1.1)
    )

    # Short: Convicción ML Short + Debilidad Relativa Extrema (Alpha < -3) + Mercado Bajista
    short_mask = (
        (df["prob_short_calibrated"] >= 0.35) &
        (df["alpha_5d"] < -3.0) &
        (df["is_market_bullish"] == 0) &
        (df["dist_to_sup20_atr"] < 0.8)
    )

    df["signal_dir"] = 0
    df.loc[long_mask, "signal_dir"] = 1
    df.loc[short_mask, "signal_dir"] = -1

    n_longs = int(np.sum(long_mask))
    n_shorts = int(np.sum(short_mask))
    print(f"Señales generadas: {n_longs:,} Longs | {n_shorts:,} Shorts.")

    # Experimentos Clave
    experiments = [
        # (Nombre, Riesgo, MaxPos, Breakeven, BreakevenATR, PermitirShorts, FiltrarLeaders)
        ("1. Base Conservadora ($1R = $1,000 | 5 Pos | Long Only)", 1000.0, 5, False, 1.8, False, False),
        ("2. Sizing Optimizado ($2R = $2,000 | 7 Pos | Long Only)", 2000.0, 7, False, 1.8, False, False),
        ("3. Sizing 2% + Breakeven en +1.8R (Long Only)", 2000.0, 7, True, 1.8, False, False),
        ("4. Sizing 2% + Breakeven + Shorts Bidireccionales", 2000.0, 7, True, 1.8, True, False),
        ("5. INSTITUCIONAL ALTO BETA (Líderes + 2% + Breakeven + Shorts)", 2000.0, 7, True, 1.8, True, True),
    ]

    leaders = ["PLTR", "COIN", "ARM", "MSTR", "AMD", "TSLA", "NVDA", "MELI", "META", "SHOP", "SMCI"]

    results_table = []
    annual_breakdowns = {}

    for name, risk, max_pos, be, be_atr, allow_shorts, only_leaders in experiments:
        df_run = df.copy()
        if not allow_shorts:
            df_run.loc[df_run["signal_dir"] == -1, "signal_dir"] = 0
        if only_leaders:
            df_run = df_run[df_run["ticker"].isin(leaders + ["SPY"])].copy()

        eq_df, tr, met = simulate_portfolio(
            df_run,
            initial_capital=100_000.0,
            risk_per_trade_usd=risk,
            max_open_positions=max_pos,
            k_tp=3.5,
            k_sl=1.8,
            use_breakeven=be,
            breakeven_atr=be_atr,
            max_holding_bars=35,
            friday_filter=True,
        )

        # Desglose de rendimiento por año
        s = eq_df.set_index("timestamp")["total_equity"].resample("YE").last()
        first = pd.Series([100_000.0], index=[pd.Timestamp("2021-12-31", tz="UTC")])
        fy = pd.concat([first, s])
        ann_rets = fy.pct_change().dropna() * 100.0
        ann_rets.index = ann_rets.index.year
        annual_breakdowns[name] = ann_rets

        results_table.append({
            "Estrategia": name,
            "Retorno Total": f"{met['retorno_total_pct']:+.2f}%",
            "CAGR (EA)": f"{met['cagr_pct']:.2f}%",
            "Sharpe": met["sharpe_ratio"],
            "Max DD %": f"{met['max_drawdown_pct']:.2f}%",
            "Win Rate": f"{met['win_rate_pct']:.2f}%",
            "Profit Factor": met["profit_factor"],
            "Trades": met["total_trades"],
            "Ganancia USD": f"${met['ganancia_neta_usd']:+,.0f}",
        })

    print("\n--- COMPARATIVA MAESTRA DE RENDIMIENTO (2022 - 2026) ---")
    summary_df = pd.DataFrame(results_table)
    print(summary_df.to_string(index=False))

    print("\n--- DESGLOSE DE RENTABILIDAD ANUAL (%) POR AÑO ---")
    ann_df = pd.DataFrame(annual_breakdowns).round(2)
    print(ann_df.to_string())
    print("=" * 85)


if __name__ == "__main__":
    main()
