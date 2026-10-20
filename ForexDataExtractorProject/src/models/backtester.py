"""
backtester.py - TICK-11: Motor de Backtesting Vectorizado y Reporte de Métricas Institucionales.

Simulador de cartera cuantitativo para Swing Trading en acciones y ETFs (2022-2026):
1. Dimensionamiento de posición dinámico por Riesgo Fijo en Dólares ($1R$).
   - Shares = min(Risk_USD / (1.5 * ATR), Max_Capital_Per_Pos / P_entry).
2. Control estricto de concurrencia y asignación de capital:
   - Máximo N posiciones simultáneas (ej. 4 a 6 activos).
   - Bloqueo de capital durante el horizonte del swing (2 a 5 días).
   - Cero re-entradas duplicadas en el mismo activo mientras la posición esté abierta.
3. Fricción sintética: Descuento de comisiones y slippage de IBKR en entrada y salida.
4. Métricas de grado institucional:
   - Sharpe Ratio, Sortino Ratio, Calmar Ratio, CAGR %, Max Drawdown %.
   - Win Rate, Profit Factor, Expectativa Matemática en R ($E[R]$) y en USD.
   - Comparativa de referencia contra la estrategia pasiva Buy & Hold del SPY.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class Trade:
    ticker: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_type: str
    shares: float
    risk_usd: float
    invested_capital: float
    pnl_usd: float
    pnl_pct: float
    r_multiple: float
    holding_bars: int


@dataclass
class PortfolioState:
    cash: float
    initial_capital: float = 100_000.0
    risk_per_trade_usd: float = 1_000.0  # 1R fijo
    max_open_positions: int = 5
    max_pos_capital_pct: float = 0.25     # Máximo 25% de la cuenta por acción
    friction_pct: float = 0.0005          # 5 bps por lado = 10 bps round-trip
    active_positions: Dict[str, dict] = field(default_factory=dict)
    closed_trades: List[Trade] = field(default_factory=list)


def run_portfolio_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    risk_per_trade_usd: float = 1_000.0,
    max_open_positions: int = 5,
    max_pos_capital_pct: float = 0.25,
    friction_pct: float = 0.0005,
    signal_col: str = "signal",
) -> Tuple[pd.DataFrame, List[Trade], dict]:
    """
    Ejecuta la simulación cronológica de la cartera a lo largo de toda la historia.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con datos horarios ordenados, enriquecido con 'signal', 'atr14', 'trade_exit_bar',
        'trade_exit_price', 'trade_exit_type', 'trade_net_return'.
    initial_capital : float
        Capital inicial de la cuenta en USD (default $100,000).
    risk_per_trade_usd : float
        Riesgo fijo en dólares por operación ($1R, default $1,000 = 1%).
    max_open_positions : int
        Número máximo de posiciones swing abiertas en simultáneo (default 5).
    max_pos_capital_pct : float
        Límite máximo de capital invertido en un único activo (default 25%).
    friction_pct : float
        Fricción porcentual por lado (default 0.0005 = 5 bps).
    signal_col : str
        Nombre de la columna booleana que activa la orden de entrada.

    Retorna:
    --------
    equity_df : pd.DataFrame
        Curva de capital horaria (timestamp, total_equity, cash, invested, drawdown_pct).
    closed_trades : List[Trade]
        Lista de todas las operaciones ejecutadas con detalle de P&L y ratio R.
    summary_metrics : dict
        Métricas cuantitativas institucionales y comparativa con el SPY.
    """
    if signal_col not in df.columns:
        raise ValueError(f"Columna de señal '{signal_col}' no encontrada en el DataFrame.")

    # Asegurar orden cronológico estricto
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    cash = initial_capital
    active_positions: Dict[str, dict] = {}  # ticker -> position dict
    closed_trades: List[Trade] = []
    equity_records: List[dict] = []

    # Mapeo rápido de precios de cierre para valoración de cartera
    # Agrupar por timestamp para simulación sincrónica
    grouped = df_sorted.groupby("timestamp", sort=False)

    for current_time, bar_group in grouped:
        prices_now = dict(zip(bar_group["ticker"], bar_group["close"]))

        # 1. GESTIÓN DE SALIDAS (Comprobar si alguna posición activa venció)
        tickers_to_close = []
        for ticker, pos in active_positions.items():
            if current_time >= pos["exit_time"]:
                tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            pos = active_positions.pop(ticker)
            # Liquidación a precio de salida neto de fricción
            exit_price_net = pos["raw_exit_price"] * (1.0 - friction_pct)
            proceeds = pos["shares"] * exit_price_net
            cash += proceeds

            pnl_usd = proceeds - pos["invested_capital"]
            pnl_pct = (pnl_usd / pos["invested_capital"]) * 100.0
            r_mult = pnl_usd / pos["risk_usd"] if pos["risk_usd"] > 0 else 0.0

            closed_trades.append(
                Trade(
                    ticker=ticker,
                    entry_time=pos["entry_time"],
                    entry_price=pos["entry_price_net"],
                    exit_time=current_time,
                    exit_price=exit_price_net,
                    exit_type=pos["exit_type"],
                    shares=pos["shares"],
                    risk_usd=pos["risk_usd"],
                    invested_capital=pos["invested_capital"],
                    pnl_usd=pnl_usd,
                    pnl_pct=pnl_pct,
                    r_multiple=r_mult,
                    holding_bars=pos["holding_bars"],
                )
            )

        # 2. GESTIÓN DE ENTRADAS (Nuevas señales en la vela actual)
        signals_now = bar_group[bar_group[signal_col] == True]

        if len(signals_now) > 0 and len(active_positions) < max_open_positions:
            # Filtrar activos que ya tengan una posición abierta
            candidate_rows = [
                row for _, row in signals_now.iterrows() if row["ticker"] not in active_positions
            ]

            # Si hay más candidatos que cupos libres, ordenar por prioridad (probabilidad o rvol)
            available_slots = max_open_positions - len(active_positions)
            if "prob_calibrated" in signals_now.columns:
                candidate_rows = sorted(
                    candidate_rows, key=lambda r: r.get("prob_calibrated", 0.0), reverse=True
                )[:available_slots]
            else:
                candidate_rows = candidate_rows[:available_slots]

            for row in candidate_rows:
                ticker = row["ticker"]
                close_price = row["close"]
                atr = row.get("atr14", close_price * (row.get("atr_pct", 2.0) / 100.0))

                entry_price_net = close_price * (1.0 + friction_pct)
                # Distancia de Stop Loss: 1.5 * ATR
                stop_dist = max(1.5 * atr, entry_price_net * 0.005)  # Mínimo 0.5% de distancia

                # Tamaño de posición según riesgo en dólares $1R
                shares_by_risk = risk_per_trade_usd / stop_dist

                # Límite de capital por posición (máximo 25% de la cuenta o efectivo disponible)
                max_capital_pos = min(initial_capital * max_pos_capital_pct, cash)
                shares_by_capital = max_capital_pos / entry_price_net
                shares = int(min(shares_by_risk, shares_by_capital))

                cost = shares * entry_price_net

                if shares > 0 and cost <= cash:
                    cash -= cost

                    # Calcular fecha/hora de salida según trade_exit_bar
                    exit_bars = int(row.get("trade_exit_bar", 26))
                    # Estimación de timestamp de salida
                    exit_time_val = row.get("exit_timestamp", current_time + pd.Timedelta(hours=exit_bars))
                    if isinstance(exit_time_val, str):
                        exit_time_val = pd.to_datetime(exit_time_val)

                    raw_exit = row.get("trade_exit_price", close_price)
                    exit_type = str(row.get("trade_exit_type", "TIME"))

                    active_positions[ticker] = {
                        "entry_time": current_time,
                        "entry_price_net": entry_price_net,
                        "exit_time": exit_time_val,
                        "raw_exit_price": raw_exit,
                        "exit_type": exit_type,
                        "shares": shares,
                        "risk_usd": risk_per_trade_usd,
                        "invested_capital": cost,
                        "holding_bars": exit_bars,
                    }

        # 3. VALORACIÓN DE CARTERA
        open_positions_val = 0.0
        for ticker, pos in active_positions.items():
            curr_p = prices_now.get(ticker, pos["entry_price_net"])
            open_positions_val += pos["shares"] * curr_p

        total_equity = cash + open_positions_val
        equity_records.append({
            "timestamp": current_time,
            "cash": cash,
            "invested_value": open_positions_val,
            "total_equity": total_equity,
            "open_positions_count": len(active_positions),
        })

    equity_df = pd.DataFrame(equity_records).drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # 4. CÁLCULO DE DRAWDOWN Y RETORNOS DE CARTERA
    equity_df["peak_equity"] = equity_df["total_equity"].cummax()
    equity_df["drawdown_usd"] = equity_df["total_equity"] - equity_df["peak_equity"]
    equity_df["drawdown_pct"] = (equity_df["drawdown_usd"] / equity_df["peak_equity"]) * 100.0

    # 5. CÁLCULO DE MÉTRICAS INSTITUCIONALES Y BENCHMARK (SPY)
    metrics = compute_backtest_metrics(equity_df, closed_trades, df_sorted, initial_capital)

    return equity_df, closed_trades, metrics


def compute_backtest_metrics(
    equity_df: pd.DataFrame,
    closed_trades: List[Trade],
    raw_df: pd.DataFrame,
    initial_capital: float = 100_000.0,
) -> dict:
    """Calcula las métricas estadísticas institucionales completas."""
    if len(equity_df) == 0:
        return {}

    final_equity = equity_df["total_equity"].iloc[-1]
    total_net_pnl = final_equity - initial_capital
    total_return_pct = (total_net_pnl / initial_capital) * 100.0

    # Duración en años
    t_start = equity_df["timestamp"].iloc[0]
    t_end = equity_df["timestamp"].iloc[-1]
    duration_days = max(1, (t_end - t_start).days)
    duration_years = duration_days / 365.25

    # CAGR %
    cagr_pct = (((final_equity / initial_capital) ** (1.0 / duration_years)) - 1.0) * 100.0 if duration_years > 0 else 0.0

    # Retornos periódicos para Sharpe / Sortino (re-muestreo a diario)
    daily_equity = equity_df.set_index("timestamp")["total_equity"].resample("D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        # Sharpe Ratio anualizado (Tasa libre de riesgo Rf = 4.0% anual = 0.04 / 252)
        rf_daily = 0.04 / 252.0
        excess_daily = daily_returns - rf_daily
        sharpe_ratio = float((excess_daily.mean() / daily_returns.std()) * np.sqrt(252))

        # Sortino Ratio (Downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 1 else daily_returns.std()
        sortino_ratio = float((excess_daily.mean() / max(downside_std, 1e-6)) * np.sqrt(252))
    else:
        sharpe_ratio, sortino_ratio = 0.0, 0.0

    # Max Drawdown
    max_dd_pct = float(equity_df["drawdown_pct"].min())
    max_dd_usd = float(equity_df["drawdown_usd"].min())
    calmar_ratio = abs(cagr_pct / max_dd_pct) if max_dd_pct != 0 else 0.0

    # Estadísticas de Trades
    n_trades = len(closed_trades)
    if n_trades > 0:
        pnl_arr = np.array([t.pnl_usd for t in closed_trades])
        r_arr = np.array([t.r_multiple for t in closed_trades])
        bars_arr = np.array([t.holding_bars for t in closed_trades])

        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr <= 0]

        win_rate_pct = float(len(wins) / n_trades) * 100.0
        avg_trade_usd = float(np.mean(pnl_arr))
        avg_win_usd = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss_usd = float(np.mean(losses)) if len(losses) > 0 else 0.0

        profit_factor = (
            float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 999.0
        )
        avg_r_multiple = float(np.mean(r_arr))
        expectancy_r = float((win_rate_pct / 100.0 * np.mean(r_arr[r_arr > 0])) + ((1.0 - win_rate_pct / 100.0) * np.mean(r_arr[r_arr <= 0]))) if len(wins) > 0 and len(losses) > 0 else avg_r_multiple

        exit_counts = pd.Series([t.exit_type for t in closed_trades]).value_counts().to_dict()
        avg_holding_bars = float(np.mean(bars_arr))
        trades_per_month = float(n_trades / (duration_years * 12.0)) if duration_years > 0 else 0.0
    else:
        win_rate_pct, avg_trade_usd, avg_win_usd, avg_loss_usd = 0.0, 0.0, 0.0, 0.0
        profit_factor, avg_r_multiple, expectancy_r, avg_holding_bars = 0.0, 0.0, 0.0, 0.0
        trades_per_month = 0.0
        exit_counts = {}

    # 6. BENCHMARK SPY (Buy & Hold)
    spy_metrics = compute_spy_benchmark(raw_df, t_start, t_end)

    return {
        "capital_inicial": initial_capital,
        "capital_final": round(final_equity, 2),
        "ganancia_neta_usd": round(total_net_pnl, 2),
        "retorno_total_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "calmar_ratio": round(calmar_ratio, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "total_trades": n_trades,
        "trades_por_mes": round(trades_per_month, 1),
        "win_rate_pct": round(win_rate_pct, 2),
        "profit_factor": round(profit_factor, 2),
        "ganancia_media_usd": round(avg_win_usd, 2),
        "perdida_media_usd": round(avg_loss_usd, 2),
        "expectativa_r": round(expectancy_r, 2),
        "avg_r_multiple": round(avg_r_multiple, 2),
        "duracion_media_velas": round(avg_holding_bars, 1),
        "desglose_salidas": exit_counts,
        "benchmark_spy": spy_metrics,
    }


def compute_spy_benchmark(raw_df: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp) -> dict:
    """Calcula el rendimiento del Buy & Hold de SPY sobre el período de simulación exacto."""
    spy_data = raw_df[raw_df["ticker"] == "SPY"].sort_values("timestamp")
    if len(spy_data) == 0:
        return {}

    sub = spy_data[(spy_data["timestamp"] >= t_start) & (spy_data["timestamp"] <= t_end)]
    if len(sub) < 2:
        return {}

    p_in = sub["close"].iloc[0]
    p_out = sub["close"].iloc[-1]
    spy_total_return = ((p_out - p_in) / p_in) * 100.0

    duration_years = max(0.1, (t_end - t_start).days / 365.25)
    spy_cagr = (((p_out / p_in) ** (1.0 / duration_years)) - 1.0) * 100.0

    # Drawdown SPY
    peaks = sub["close"].cummax()
    spy_dd = ((sub["close"] - peaks) / peaks) * 100.0
    spy_max_dd = float(spy_dd.min())

    # Sharpe SPY
    daily_spy = sub.set_index("timestamp")["close"].resample("D").last().dropna()
    daily_spy_ret = daily_spy.pct_change().dropna()
    rf_daily = 0.04 / 252.0
    spy_sharpe = float(((daily_spy_ret - rf_daily).mean() / daily_spy_ret.std()) * np.sqrt(252)) if daily_spy_ret.std() > 0 else 0.0

    return {
        "spy_retorno_total_pct": round(spy_total_return, 2),
        "spy_cagr_pct": round(spy_cagr, 2),
        "spy_max_drawdown_pct": round(spy_max_dd, 2),
        "spy_sharpe_ratio": round(spy_sharpe, 2),
    }
