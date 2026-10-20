"""
test_active_trailing_and_shorts.py - Simulación con Trailing Stop, Breakeven y Operativa Short.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def simulate_bar_by_bar_active(
    df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    risk_per_trade_usd: float = 1_000.0,
    max_open_positions: int = 5,
    max_pos_capital_pct: float = 0.25,
    friction_pct: float = 0.0005,
    use_breakeven: bool = True,
    breakeven_atr: float = 1.5,
    use_trailing: bool = True,
    trailing_atr: float = 1.5,
    max_holding_bars: int = 26,
    friday_filter: bool = True,
) -> Tuple[pd.DataFrame, List[dict], dict]:
    """
    Simulador activo vela a vela con soporte para:
    1. Compras (Longs) y Ventas en Corto (Shorts).
    2. Subida de Stop a Breakeven ($+0.0R$) al alcanzar +1.5x ATR.
    3. Trailing Stop dinámico.
    4. Filtro de fin de semana (no abrir posiciones viernes después de las 14:00 EST).
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    cash = initial_capital
    active_positions: Dict[str, dict] = {}
    closed_trades: List[dict] = []
    equity_records: List[dict] = []

    # Agrupar por timestamp
    grouped = df_sorted.groupby("timestamp", sort=False)

    for current_time, bar_group in grouped:
        prices_now = dict(zip(bar_group["ticker"], bar_group["close"]))
        highs_now = dict(zip(bar_group["ticker"], bar_group["high"]))
        lows_now = dict(zip(bar_group["ticker"], bar_group["low"]))
        opens_now = dict(zip(bar_group["ticker"], bar_group["open"]))

        # 1. GESTIÓN ACTIVA DE POSICIONES ABIERTAS (Check SL, TP, Breakeven, Trailing)
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
            direction = pos["direction"]  # 1 for Long, -1 for Short

            exit_triggered = False
            raw_exit_price = c
            exit_type = "TIME"

            if direction == 1:  # LONG
                # A. Comprobar Breakeven / Trailing trigger
                if use_breakeven and not pos["is_breakeven"]:
                    if h >= pos["entry_price_raw"] + (breakeven_atr * atr):
                        pos["sl_price"] = max(pos["sl_price"], pos["entry_price_raw"])
                        pos["is_breakeven"] = True

                if use_trailing and pos["is_breakeven"]:
                    # Trailing stop por debajo del máximo alcanzado
                    pos["peak_price"] = max(pos["peak_price"], h)
                    trailing_stop = pos["peak_price"] - (trailing_atr * atr)
                    pos["sl_price"] = max(pos["sl_price"], trailing_stop)

                # B. Comprobar toques
                # 1. Stop Loss (o Trailing / Breakeven)
                if l <= pos["sl_price"]:
                    exit_triggered = True
                    raw_exit_price = min(pos["sl_price"], o)
                    exit_type = "TRAILING_SL" if pos["is_breakeven"] else "SL"
                # 2. Take Profit fijo
                elif h >= pos["tp_price"]:
                    exit_triggered = True
                    raw_exit_price = max(pos["tp_price"], o)
                    exit_type = "TP"
                # 3. Tiempo límite
                elif pos["bars_held"] >= max_holding_bars:
                    exit_triggered = True
                    raw_exit_price = c
                    exit_type = "TIME"

            else:  # SHORT
                # A. Breakeven / Trailing trigger para cortos
                if use_breakeven and not pos["is_breakeven"]:
                    if l <= pos["entry_price_raw"] - (breakeven_atr * atr):
                        pos["sl_price"] = min(pos["sl_price"], pos["entry_price_raw"])
                        pos["is_breakeven"] = True

                if use_trailing and pos["is_breakeven"]:
                    pos["trough_price"] = min(pos["trough_price"], l)
                    trailing_stop = pos["trough_price"] + (trailing_atr * atr)
                    pos["sl_price"] = min(pos["sl_price"], trailing_stop)

                # B. Comprobar toques para cortos
                if h >= pos["sl_price"]:
                    exit_triggered = True
                    raw_exit_price = max(pos["sl_price"], o)
                    exit_type = "TRAILING_SL" if pos["is_breakeven"] else "SL"
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

            if direction == 1:  # Long exit
                exit_price_net = raw_exit_price * (1.0 - friction_pct)
                proceeds = shares * exit_price_net
                pnl_usd = proceeds - pos["invested_capital"]
            else:  # Short exit
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

        # 2. GESTIÓN DE ENTRADAS (LONGS Y SHORTS)
        # Filtro de viernes: no abrir nuevas posiciones después de las 14:00 (19:00 UTC)
        is_friday_afternoon = (current_time.dayofweek == 4) and (current_time.hour >= 18)
        can_open_new = not (friday_filter and is_friday_afternoon)

        if can_open_new and len(active_positions) < max_open_positions:
            available_slots = max_open_positions - len(active_positions)

            # Buscar señales en bar_group
            candidate_rows = []
            for _, row in bar_group.iterrows():
                sig = row.get("signal_dir", 0)  # +1 for Long, -1 for Short, 0 for None
                if sig != 0 and row["ticker"] not in active_positions:
                    candidate_rows.append(row)

            # Priorizar por volumen relativo (RVOL)
            candidate_rows = sorted(
                candidate_rows, key=lambda r: r.get("rvol", 1.0), reverse=True
            )[:available_slots]

            for row in candidate_rows:
                ticker = row["ticker"]
                direction = int(row["signal_dir"])
                close_p = row["close"]
                atr = row.get("atr14", close_p * 0.02)

                entry_price_raw = close_p
                entry_price_net = close_p * (1.0 + (friction_pct if direction == 1 else -friction_pct))

                stop_dist = max(1.5 * atr, entry_price_net * 0.005)
                shares_by_risk = risk_per_trade_usd / stop_dist
                max_cap = min(initial_capital * max_pos_capital_pct, cash)
                shares_by_cap = max_cap / entry_price_net
                shares = int(min(shares_by_risk, shares_by_cap))

                margin_req = shares * entry_price_net

                if shares > 0 and margin_req <= cash:
                    cash -= margin_req

                    if direction == 1:
                        sl_p = entry_price_raw - (1.5 * atr)
                        tp_p = entry_price_raw + (3.0 * atr)
                    else:
                        sl_p = entry_price_raw + (1.5 * atr)
                        tp_p = entry_price_raw - (3.0 * atr)

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
                        "peak_price": entry_price_raw,
                        "trough_price": entry_price_raw,
                    }

        # 3. VALORACIÓN DE CARTERA
        open_val = 0.0
        for ticker, pos in active_positions.items():
            curr_p = prices_now.get(ticker, pos["entry_price_net"])
            if pos["direction"] == 1:
                open_val += pos["shares"] * curr_p
            else:
                unrealized_pnl = (pos["entry_price_net"] - curr_p) * pos["shares"]
                open_val += pos["margin_locked"] + unrealized_pnl

        total_equity = cash + open_val
        equity_records.append({
            "timestamp": current_time,
            "total_equity": total_equity,
            "cash": cash,
            "open_positions": len(active_positions),
        })

    eq_df = pd.DataFrame(equity_records).drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    eq_df["peak"] = eq_df["total_equity"].cummax()
    eq_df["drawdown_pct"] = ((eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]) * 100.0

    # Métricas
    final_eq = eq_df["total_equity"].iloc[-1]
    net_pnl = final_eq - initial_capital
    ret_pct = (net_pnl / initial_capital) * 100.0
    duration_years = (eq_df["timestamp"].iloc[-1] - eq_df["timestamp"].iloc[0]).days / 365.25
    cagr = (((final_eq / initial_capital) ** (1.0 / duration_years)) - 1.0) * 100.0

    daily = eq_df.set_index("timestamp")["total_equity"].resample("D").last().dropna()
    daily_rets = daily.pct_change().dropna()
    rf = 0.04 / 252.0
    sharpe = float(((daily_rets - rf).mean() / daily_rets.std()) * np.sqrt(252)) if daily_rets.std() > 0 else 0.0

    n_tr = len(closed_trades)
    if n_tr > 0:
        pnls = [t["pnl_usd"] for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / n_tr * 100.0
        pf = sum(wins) / abs(sum(losses)) if len(losses) > 0 and sum(losses) != 0 else 999.0
        r_arr = [t["r_multiple"] for t in closed_trades]
        exp_r = np.mean(r_arr)
    else:
        wr, pf, exp_r = 0.0, 0.0, 0.0

    metrics = {
        "capital_final": round(final_eq, 2),
        "ganancia_neta": round(net_pnl, 2),
        "retorno_total_pct": round(ret_pct, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(float(eq_df["drawdown_pct"].min()), 2),
        "total_trades": n_tr,
        "win_rate_pct": round(wr, 2),
        "profit_factor": round(pf, 2),
        "expectativa_r": round(float(exp_r), 2),
    }

    return eq_df, closed_trades, metrics


def main():
    print("=" * 80)
    print("SIMULACIÓN ACTIVA: TRAILING STOP + BREAKEVEN + MÓDULO SHORTS (2022 - 2026)")
    print("=" * 80)

    data_path = Path("data/processed/labeled_universe_1hour.parquet")
    df = pd.read_parquet(data_path).dropna(subset=["target_quality_trade"]).reset_index(drop=True)

    # 1. Definir señales Long y Short
    # Long: Breakout + Bull Market + RVOL > 1.2
    long_mask = (
        (df["dist_to_res20_atr"] < 0.5) &
        (df["is_market_bullish"] == 1) &
        (df["rvol"] > 1.2)
    )

    # Short: Breakdown soporte + Bear Market (SPY < EMA50) + Alpha negativo + RVOL > 1.2
    short_mask = (
        (df["dist_to_sup20_atr"] < 0.5) &
        (df["is_market_bullish"] == 0) &
        (df["alpha_5d"] < 0) &
        (df["rvol"] > 1.2)
    )

    df["signal_dir"] = 0
    df.loc[long_mask, "signal_dir"] = 1
    df.loc[short_mask, "signal_dir"] = -1

    n_longs = int(np.sum(long_mask))
    n_shorts = int(np.sum(short_mask))
    print(f"Señales detectadas: {n_longs:,} Longs | {n_shorts:,} Shorts en el período 2022-2026.")

    # Experimentos
    configs = [
        ("A. Long-Only Estático (Base previa)", 1000.0, False, False, False, 1),
        ("B. Long-Only + Breakeven + Trailing", 1000.0, True, True, True, 1),
        ("C. Long + Short + Breakeven + Trailing ($1R = $1,000)", 1000.0, True, True, True, 0),
        ("D. Long + Short + Breakeven + Trailing ($1R = $2,000 / 2%)", 2000.0, True, True, True, 0),
    ]

    results = []
    for name, risk, be, trail, fri, allow_dir in configs:
        df_run = df.copy()
        if allow_dir == 1:
            df_run.loc[df_run["signal_dir"] == -1, "signal_dir"] = 0

        eq_df, tr, met = simulate_bar_by_bar_active(
            df_run,
            initial_capital=100_000.0,
            risk_per_trade_usd=risk,
            max_open_positions=5,
            use_breakeven=be,
            use_trailing=trail,
            friday_filter=fri,
        )

        # Retornos anuales
        s = eq_df.set_index("timestamp")["total_equity"].resample("YE").last()
        first = pd.Series([100_000.0], index=[pd.Timestamp("2021-12-31", tz="UTC")])
        fy = pd.concat([first, s])
        ann_rets = fy.pct_change().dropna() * 100.0

        results.append({
            "Configuración": name,
            "Retorno Total": f"{met['retorno_total_pct']:+.2f}%",
            "CAGR %": f"{met['cagr_pct']:.2f}%",
            "Sharpe": met["sharpe_ratio"],
            "Max DD %": f"{met['max_drawdown_pct']:.2f}%",
            "Win Rate": f"{met['win_rate_pct']:.2f}%",
            "Profit Factor": met["profit_factor"],
            "Total Trades": met["total_trades"],
            "Peor Año": f"{ann_rets.min():+.2f}%",
            "Mejor Año": f"{ann_rets.max():+.2f}%",
        })

    print("\n--- COMPARATIVA DE IMPACTO: BREAKEVEN, TRAILING, SHORTS Y TAMAÑO ---")
    print(pd.DataFrame(results).to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
