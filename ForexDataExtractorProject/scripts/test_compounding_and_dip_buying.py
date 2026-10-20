"""
test_compounding_and_dip_buying.py
Prueba dos palancas cruciales para alcanzar >= 30% - 40% EA:
1. Sizing porcentual compuesto (arriesgar el 2.0% del capital actual, no una cantidad fija de $2,000 que se diluye a 0.8% cuando la cuenta crece).
2. Motor Dual:
   - Estrategia A: Breakout Momentum en tendencia alcista.
   - Estrategia B: Compra de Pánico / Capitulación (Efecto MeLi / Dip-Buying en líderes con absorción de volumen).
   - Estrategia C: Shorts en colapsos macro.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def simulate_dynamic_compounding(
    df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    risk_pct_per_trade: float = 0.02,  # 2.0% del capital VIVO (compuesto)
    max_open_positions: int = 7,
    max_pos_capital_pct: float = 0.25,
    friction_pct: float = 0.0005,
    k_tp: float = 3.5,
    k_sl: float = 1.8,
    max_holding_bars: int = 35,
    friday_filter: bool = True,
):
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    cash = initial_capital
    active_positions = {}
    closed_trades = []
    equity_records = []

    grouped = df_sorted.groupby("timestamp", sort=False)

    for current_time, bar_group in grouped:
        prices_now = dict(zip(bar_group["ticker"], bar_group["close"]))
        highs_now = dict(zip(bar_group["ticker"], bar_group["high"]))
        lows_now = dict(zip(bar_group["ticker"], bar_group["low"]))
        opens_now = dict(zip(bar_group["ticker"], bar_group["open"]))

        # 1. GESTIÓN DE SALIDAS
        tickers_to_close = []
        for ticker, pos in active_positions.items():
            if ticker not in highs_now:
                continue

            h = highs_now[ticker]
            l = lows_now[ticker]
            o = opens_now[ticker]
            c = prices_now[ticker]

            pos["bars_held"] += 1
            direction = pos["direction"]

            exit_triggered = False
            raw_exit_price = c
            exit_type = "TIME"

            if direction == 1:
                if l <= pos["sl_price"]:
                    exit_triggered = True
                    raw_exit_price = min(pos["sl_price"], o)
                    exit_type = "SL"
                elif h >= pos["tp_price"]:
                    exit_triggered = True
                    raw_exit_price = max(pos["tp_price"], o)
                    exit_type = "TP"
                elif pos["bars_held"] >= max_holding_bars:
                    exit_triggered = True
                    raw_exit_price = c
                    exit_type = "TIME"
            else:
                if h >= pos["sl_price"]:
                    exit_triggered = True
                    raw_exit_price = max(pos["sl_price"], o)
                    exit_type = "SL"
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
                "pnl_usd": pnl_usd,
                "r_multiple": r_mult,
                "exit_type": exit_type,
            })

        # Calcular equity actual antes de dimensionar nuevas órdenes
        open_val = 0.0
        for ticker, pos in active_positions.items():
            curr_p = prices_now.get(ticker, pos["entry_price_net"])
            if pos["direction"] == 1:
                open_val += pos["shares"] * curr_p
            else:
                pnl = (pos["entry_price_net"] - curr_p) * pos["shares"]
                open_val += pos["margin_locked"] + pnl
        current_equity = cash + open_val

        # 2. ENTRADAS CON SIZING COMPUESTO DINÁMICO
        is_friday_afternoon = (current_time.dayofweek == 4) and (current_time.hour >= 18)
        can_open_new = not (friday_filter and is_friday_afternoon)

        if can_open_new and len(active_positions) < max_open_positions:
            available_slots = max_open_positions - len(active_positions)

            candidate_rows = []
            for _, row in bar_group.iterrows():
                sig = row.get("signal_dir", 0)
                if sig != 0 and row["ticker"] not in active_positions:
                    candidate_rows.append(row)

            candidate_rows = sorted(
                candidate_rows, key=lambda r: r.get("rvol", 1.0), reverse=True
            )[:available_slots]

            # El riesgo se calcula sobre el EQUITY VIVO (Compuesto)
            risk_usd_now = current_equity * risk_pct_per_trade

            for row in candidate_rows:
                ticker = row["ticker"]
                direction = int(row["signal_dir"])
                close_p = row["close"]
                atr = row.get("atr14", close_p * 0.02)

                entry_price_raw = close_p
                entry_price_net = close_p * (1.0 + (friction_pct if direction == 1 else -friction_pct))

                stop_dist = max(k_sl * atr, entry_price_net * 0.005)
                shares_by_risk = risk_usd_now / stop_dist
                max_cap = min(current_equity * max_pos_capital_pct, cash)
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
                        "risk_usd": risk_usd_now,
                        "margin_locked": margin_req,
                        "bars_held": 0,
                    }

        # Registro de equity
        open_val_final = 0.0
        for ticker, pos in active_positions.items():
            curr_p = prices_now.get(ticker, pos["entry_price_net"])
            if pos["direction"] == 1:
                open_val_final += pos["shares"] * curr_p
            else:
                pnl = (pos["entry_price_net"] - curr_p) * pos["shares"]
                open_val_final += pos["margin_locked"] + pnl
        equity_records.append({
            "timestamp": current_time,
            "total_equity": cash + open_val_final,
        })

    eq_df = pd.DataFrame(equity_records)
    final_eq = eq_df["total_equity"].iloc[-1]
    net_profit = final_eq - initial_capital
    ret_pct = (net_profit / initial_capital) * 100.0

    eq_df["peak"] = eq_df["total_equity"].cummax()
    eq_df["drawdown"] = (eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]
    max_dd = float(eq_df["drawdown"].min() * 100.0)

    t_start = eq_df["timestamp"].iloc[0]
    t_end = eq_df["timestamp"].iloc[-1]
    years = max((t_end - t_start).days / 365.25, 0.5)
    cagr = ((final_eq / initial_capital) ** (1.0 / years) - 1.0) * 100.0

    daily_eq = eq_df.set_index("timestamp")["total_equity"].resample("1D").last().dropna()
    daily_rets = daily_eq.pct_change().dropna()
    sharpe = float((daily_rets.mean() / (daily_rets.std() + 1e-9)) * np.sqrt(252)) if len(daily_rets) > 0 else 0.0

    tr_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()
    wr = (tr_df["pnl_usd"] > 0).mean() * 100.0 if not tr_df.empty else 0.0
    wins = tr_df[tr_df["pnl_usd"] > 0]["pnl_usd"].sum() if not tr_df.empty else 0.0
    losses = abs(tr_df[tr_df["pnl_usd"] <= 0]["pnl_usd"].sum()) if not tr_df.empty else 1.0
    pf = wins / losses if losses > 0 else 999.0

    return eq_df, {
        "retorno_total_pct": round(ret_pct, 2),
        "ganancia_neta": round(net_profit, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "win_rate_pct": round(wr, 2),
        "profit_factor": round(pf, 2),
        "total_trades": len(closed_trades),
    }


def main():
    data_path = Path("data/processed/labeled_universe_1hour.parquet")
    df = pd.read_parquet(data_path).dropna(subset=["target_quality_trade"]).reset_index(drop=True)

    high_beta_leaders = ["PLTR", "NVDA", "TSLA", "MELI", "AMD", "ARM", "COIN", "MSTR", "META", "SHOP", "SMCI"]
    df = df[df["ticker"].isin(high_beta_leaders + ["SPY"])].copy()

    # MOTOR 1: Breakout Momentum en Líderes
    sig_breakout = (
        (df["dist_to_res20_atr"] < 0.5) &
        (df["is_market_bullish"] == 1) &
        (df["alpha_5d"] > 0) &
        (df["rvol"] > 1.2)
    )

    # MOTOR 2: Pánico / Capitulación y Rebote (Efecto MeLi / Dip-Buying)
    # Condición: La acción líder está en retroceso/pánico pero muestra absorción institucional de volumen
    sig_dip_buy = (
        (df["dist_to_sup20_atr"] < 0.8) &
        (df["is_bullish_absorption"] == 1) &
        (df["alpha_5d"] > -2.0) &
        (df["rvol"] > 1.3)
    )

    # MOTOR 3: Shorts Selectivos en colapsos con debilidad extrema
    sig_short = (
        (df["dist_to_sup20_atr"] < 0.5) &
        (df["is_market_bullish"] == 0) &
        (df["alpha_5d"] < -4.0) &
        (df["rvol"] > 1.3)
    )

    # Experimentos de Sizing Compuesto
    scenarios = [
        ("A. Sizing Fijo $2,000 (No compuesto / diluido)", 0.02, False, False, False),
        ("B. Sizing Compuesto 2.0% (Breakouts)", 0.02, True, False, False),
        ("C. Sizing Compuesto 2.5% (Breakouts)", 0.025, True, False, False),
        ("D. Sizing Compuesto 2.0% + Motor Dip-Buying MeLi", 0.02, True, True, False),
        ("E. Sizing Compuesto 2.5% + Motor Dip-Buying MeLi", 0.025, True, True, False),
        ("F. Sistema Tri-Motor Completo: Breakout + Dip-Buying + Shorts (2.5% Compuesto)", 0.025, True, True, True),
    ]

    results = []
    annual_tables = {}

    for name, r_pct, is_comp, use_dip, use_sh in scenarios:
        df_run = df.copy()
        df_run["signal_dir"] = 0

        # Activar señales según escenario
        df_run.loc[sig_breakout, "signal_dir"] = 1
        if use_dip:
            df_run.loc[sig_dip_buy, "signal_dir"] = 1
        if use_sh:
            df_run.loc[sig_short, "signal_dir"] = -1

        if is_comp:
            eq_df, met = simulate_dynamic_compounding(
                df_run,
                initial_capital=100_000.0,
                risk_pct_per_trade=r_pct,
                max_open_positions=7,
                max_pos_capital_pct=0.25,
                k_tp=3.5,
                k_sl=1.8,
                max_holding_bars=35,
            )
        else:
            # Replicar el script previo sin interés compuesto (fijo 2k)
            from scripts.test_active_trailing_and_shorts import simulate_bar_by_bar_active
            eq_df, _, met_raw = simulate_bar_by_bar_active(
                df_run,
                initial_capital=100_000.0,
                risk_per_trade_usd=2000.0,
                max_open_positions=7,
                use_breakeven=False,
                use_trailing=False,
            )
            met = {
                "retorno_total_pct": met_raw["retorno_total_pct"],
                "ganancia_neta": met_raw["ganancia_neta"],
                "cagr_pct": met_raw["cagr_pct"],
                "max_drawdown_pct": met_raw["max_drawdown_pct"],
                "sharpe_ratio": met_raw["sharpe_ratio"],
                "win_rate_pct": met_raw["win_rate_pct"],
                "profit_factor": met_raw["profit_factor"],
                "total_trades": met_raw["total_trades"],
            }

        s = eq_df.set_index("timestamp")["total_equity"].resample("YE").last()
        first = pd.Series([100_000.0], index=[pd.Timestamp("2021-12-31", tz="UTC")])
        fy = pd.concat([first, s])
        ann = fy.pct_change().dropna() * 100.0
        ann.index = ann.index.year
        annual_tables[name] = ann

        results.append({
            "Escenario": name,
            "Retorno Total": f"{met['retorno_total_pct']:+.2f}%",
            "CAGR %": f"{met['cagr_pct']:.2f}%",
            "Sharpe": met["sharpe_ratio"],
            "Max DD %": f"{met['max_drawdown_pct']:.2f}%",
            "Trades": met["total_trades"],
            "Win Rate": f"{met['win_rate_pct']:.2f}%",
            "Profit Factor": met["profit_factor"],
            "Ganancia USD": f"${met['ganancia_neta']:+,.0f}",
        })

    print("=" * 100)
    print("COMPARATIVA DE ESCENARIOS DE ALTO RENDIMIENTO (COMPOUNDING & DIP-BUYING)")
    print("=" * 100)
    print(pd.DataFrame(results).to_string(index=False))

    print("\n--- DESGLOSE ANUAL (%) POR AÑO ---")
    print(pd.DataFrame(annual_tables).round(2).to_string())
    print("=" * 100)


if __name__ == "__main__":
    main()
