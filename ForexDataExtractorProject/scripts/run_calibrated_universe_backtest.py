"""
run_calibrated_universe_backtest.py - Master Backtest Multi-Activo con Umbrales Óptimos y Deciles.

Ejecuta el backtest institucional completo:
- Aplica los modelos duales calibrados (Buy & Sell) y 'Winner Takes All'.
- Opera EXCLUSIVAMENTE cuando la probabilidad predicha supera el umbral óptimo T*.
- Filtra trampas de apertura matutina (9:30 - 10:30 EST) identificadas en el Error Analysis.
- Simulación de cartera con gestión de capital institucional (máx 8 posiciones simultáneas,
  riesgo controlado por posición, comisiones IBKR de 10 bps round-trip y slippage).
- Desglose riguroso año a año (2022 a 2026) para verificar consistencia absoluta sin depender de secuencias alcistas.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd


def run_portfolio_backtest(
    df: pd.DataFrame,
    threshold_buy: float = 0.50,
    threshold_sell: float = 0.50,
    initial_capital: float = 10000.0,
    risk_per_trade_pct: float = 0.02,  # 2% de riesgo de capital por trade
    max_open_positions: int = 8,
    friction_pct: float = 0.0010,       # 10 bps round-trip
    filter_morning_traps: bool = True,
):
    print("\n" + "="*80)
    print(f"EJECUTANDO MASTER BACKTEST CON UMBRALES:")
    print(f"  * Umbral Compra (Buy):  {threshold_buy:.2f}")
    print(f"  * Umbral Venta (Sell):  {threshold_sell:.2f}")
    print(f"  * Capital Inicial:      ${initial_capital:,.2f} USD")
    print(f"  * Riesgo por Trade:     {risk_per_trade_pct*100:.1f}%")
    print(f"  * Posiciones Máx:       {max_open_positions}")
    print(f"  * Filtro Trampa Mañana: {'Activado' if filter_morning_traps else 'Desactivado'}")
    print("="*80)

    time_col = "timestamp" if "timestamp" in df.columns else "datetime"
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # Identificar señales válidas
    signals_buy = (df["prob_buy"] >= threshold_buy)
    signals_sell = (df["prob_sell"] >= threshold_sell)

    if filter_morning_traps and "hour_of_day" in df.columns:
        # Filtrar horas de trampas matutinas (13-14 UTC / 9:30-10:30 EST)
        signals_buy = signals_buy & (df["hour_of_day"] > 14)
        signals_sell = signals_sell & (df["hour_of_day"] > 14)

    df["signal_buy"] = signals_buy.astype(int)
    df["signal_sell"] = signals_sell.astype(int)

    # Simulación cronológica vela a vela
    capital = initial_capital
    peak_capital = initial_capital
    open_positions = []  # Lista de dicts con posiciones activas
    closed_trades = []

    # Agrupar datos por timestamp único para simular concurrencia multi-activo
    unique_timestamps = df[time_col].drop_duplicates().sort_values().to_numpy()
    
    # Mapear barras por timestamp
    time_grouped = df.groupby(time_col)

    for current_time in unique_timestamps:
        current_bars = time_grouped.get_group(current_time)
        current_bar_dict = {row["ticker"]: row for _, row in current_bars.iterrows()}
        
        # 1. Gestionar salidas de posiciones abiertas existentes
        remaining_positions = []
        for pos in open_positions:
            ticker = pos["ticker"]
            if ticker in current_bar_dict:
                bar = current_bar_dict[ticker]
                pos["bars_held"] += 1
                cur_high = bar["high"]
                cur_low = bar["low"]
                cur_close = bar["close"]
                
                exit_triggered = False
                exit_price = cur_close
                exit_reason = "HORIZON"
                
                if pos["side"] == "BUY":
                    # Chequear TP
                    if cur_high >= pos["tp_price"]:
                        exit_price = pos["tp_price"] * (1.0 - friction_pct / 2.0)
                        exit_reason = "TAKE_PROFIT"
                        exit_triggered = True
                    # Chequear SL
                    elif cur_low <= pos["sl_price"]:
                        # Posible gap-down por debajo de SL
                        exit_price = min(pos["sl_price"], bar["open"]) * (1.0 - friction_pct / 2.0)
                        exit_reason = "STOP_LOSS"
                        exit_triggered = True
                    # Chequear tiempo
                    elif pos["bars_held"] >= pos["max_horizon"]:
                        exit_price = cur_close * (1.0 - friction_pct / 2.0)
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
                    # Chequear TP (para short el TP está por debajo)
                    if cur_low <= pos["tp_price"]:
                        exit_price = pos["tp_price"] * (1.0 + friction_pct / 2.0)
                        exit_reason = "TAKE_PROFIT"
                        exit_triggered = True
                    # Chequear SL (para short el SL está por encima)
                    elif cur_high >= pos["sl_price"]:
                        exit_price = max(pos["sl_price"], bar["open"]) * (1.0 + friction_pct / 2.0)
                        exit_reason = "STOP_LOSS"
                        exit_triggered = True
                    elif pos["bars_held"] >= pos["max_horizon"]:
                        exit_price = cur_close * (1.0 + friction_pct / 2.0)
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
        if capital > peak_capital:
            peak_capital = capital

        # 2. Apertura de nuevas posiciones si hay cupo disponible
        available_slots = max_open_positions - len(open_positions)
        if available_slots > 0 and capital > 1000.0:
            open_tickers = {p["ticker"] for p in open_positions}
            
            # Candidatos a compra y venta ordenados por mayor probabilidad
            buy_candidates = current_bars[current_bars["signal_buy"] == 1].sort_values("prob_buy", ascending=False)
            sell_candidates = current_bars[current_bars["signal_sell"] == 1].sort_values("prob_sell", ascending=False)
            
            for _, row in buy_candidates.iterrows():
                if available_slots <= 0:
                    break
                t_cand = row["ticker"]
                if t_cand not in open_tickers:
                    # Tamaño de posición: asignación de riesgo fijo (2% del capital / distancia SL)
                    cur_close = row["close"]
                    cur_atr = max(row.get("atr14", cur_close * 0.02), 0.01)
                    sl_dist = 1.8 * cur_atr
                    sl_dist_pct = sl_dist / cur_close
                    
                    # Posición en USD para arriesgar exactamente risk_per_trade_pct del capital
                    target_pos_size = min(capital * (risk_per_trade_pct / max(sl_dist_pct, 0.01)), capital * 0.25)
                    target_pos_size = min(target_pos_size, capital - 100.0)
                    
                    if target_pos_size >= 500.0:
                        entry_p = cur_close * (1.0 + friction_pct / 2.0)
                        open_positions.append({
                            "ticker": t_cand,
                            "side": "BUY",
                            "entry_time": current_time,
                            "entry_price": entry_p,
                            "tp_price": cur_close + 3.5 * cur_atr,
                            "sl_price": cur_close - 1.8 * cur_atr,
                            "position_size_usd": target_pos_size,
                            "bars_held": 0,
                            "max_horizon": 35,
                        })
                        capital -= target_pos_size
                        open_tickers.add(t_cand)
                        available_slots -= 1

            for _, row in sell_candidates.iterrows():
                if available_slots <= 0:
                    break
                t_cand = row["ticker"]
                if t_cand not in open_tickers:
                    cur_close = row["close"]
                    cur_atr = max(row.get("atr14", cur_close * 0.02), 0.01)
                    sl_dist = 1.8 * cur_atr
                    sl_dist_pct = sl_dist / cur_close
                    
                    target_pos_size = min(capital * (risk_per_trade_pct / max(sl_dist_pct, 0.01)), capital * 0.25)
                    target_pos_size = min(target_pos_size, capital - 100.0)
                    
                    if target_pos_size >= 500.0:
                        entry_p = cur_close * (1.0 - friction_pct / 2.0)
                        open_positions.append({
                            "ticker": t_cand,
                            "side": "SELL",
                            "entry_time": current_time,
                            "entry_price": entry_p,
                            "tp_price": cur_close - 3.5 * cur_atr,
                            "sl_price": cur_close + 1.8 * cur_atr,
                            "position_size_usd": target_pos_size,
                            "bars_held": 0,
                            "max_horizon": 35,
                        })
                        capital -= target_pos_size
                        open_tickers.add(t_cand)
                        available_slots -= 1

    # Liquidar posiciones abiertas al final
    for pos in open_positions:
        ticker = pos["ticker"]
        final_row = df[df["ticker"] == ticker].iloc[-1]
        exit_p = final_row["close"]
        if pos["side"] == "BUY":
            pnl_pct = (exit_p - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - exit_p) / pos["entry_price"]
        pnl_usd = pos["position_size_usd"] * pnl_pct
        capital += pos["position_size_usd"] + pnl_usd
        closed_trades.append({
            "ticker": ticker,
            "side": pos["side"],
            "entry_time": pos["entry_time"],
            "exit_time": final_row[time_col],
            "entry_price": pos["entry_price"],
            "exit_price": exit_p,
            "pnl_pct": pnl_pct * 100.0,
            "pnl_usd": pnl_usd,
            "exit_reason": "BACKTEST_END",
            "bars_held": pos["bars_held"],
            "capital_after": capital,
        })

    trades_df = pd.DataFrame(closed_trades)
    
    # Métricas y Resumen Anual
    print("\n" + "="*80)
    print("RESULTADOS GLOBALES DEL MASTER BACKTEST:")
    total_trades = len(trades_df)
    if total_trades == 0:
        print("No se ejecutaron trades con estos filtros.")
        return

    wins = trades_df[trades_df["pnl_usd"] > 0]
    losses = trades_df[trades_df["pnl_usd"] <= 0]
    win_rate = len(wins) / total_trades * 100.0
    profit_factor = wins["pnl_usd"].sum() / abs(losses["pnl_usd"].sum()) if len(losses) > 0 and abs(losses["pnl_usd"].sum()) > 0 else 99.0
    total_return_pct = (capital - initial_capital) / initial_capital * 100.0

    print(f"  * Capital Final:      ${capital:,.2f} USD")
    print(f"  * Retorno Total:      {total_return_pct:+.2f}%")
    print(f"  * Total Trades:       {total_trades}")
    print(f"  * Win Rate:           {win_rate:.2f}%")
    print(f"  * Profit Factor:      {profit_factor:.2f}")
    print(f"  * Ganancia Promedio:  ${wins['pnl_usd'].mean():,.2f} USD")
    print(f"  * Pérdida Promedio:   ${losses['pnl_usd'].mean():,.2f} USD")
    print("="*80)

    # Desglose por Año
    trades_df["exit_year"] = pd.to_datetime(trades_df["exit_time"]).dt.year
    annual_summary = []
    
    for year, grp in trades_df.groupby("exit_year"):
        y_wins = grp[grp["pnl_usd"] > 0]
        y_losses = grp[grp["pnl_usd"] <= 0]
        y_pnl = grp["pnl_usd"].sum()
        y_wr = len(y_wins) / len(grp) * 100.0
        y_pf = y_wins["pnl_usd"].sum() / abs(y_losses["pnl_usd"].sum()) if len(y_losses) > 0 and abs(y_losses["pnl_usd"].sum()) > 0 else 99.0
        annual_summary.append({
            "Año": year,
            "Trades": len(grp),
            "Win Rate %": round(y_wr, 2),
            "PnL USD": round(y_pnl, 2),
            "Profit Factor": round(y_pf, 2),
            "Retorno Medio Trade %": round(grp["pnl_pct"].mean(), 2),
        })

    print("\n--- DESGLOSE ANUAL RIGUROSO (2022 - 2026) ---")
    df_annual = pd.DataFrame(annual_summary)
    print(df_annual.to_string(index=False))
    print("="*80)

    return trades_df


def main():
    features_path = PROJECT_ROOT / "data" / "processed" / "features_universe_full_structural.parquet"
    models_dir = PROJECT_ROOT / "models"
    buy_file = models_dir / "pipeline_universe_buy_calibrated.joblib"
    sell_file = models_dir / "pipeline_universe_sell_calibrated.joblib"

    if not features_path.exists() or not buy_file.exists() or not sell_file.exists():
        print("Esperando a que existan los features y modelos entrenados...")
        return

    print("Cargando dataset y modelos entrenados...")
    df = pd.read_parquet(features_path)
    model_buy = joblib.load(buy_file)
    model_sell = joblib.load(sell_file)

    feature_cols = list(model_buy.feature_names_in_)
    print(f"Alineando {len(feature_cols)} features con el modelo entrenado...")
    X = df[feature_cols].fillna(0.0)

    print("Generando probabilidades en todo el histórico...")
    raw_buy = model_buy.predict_proba(X)[:, 1]
    raw_sell = model_sell.predict_proba(X)[:, 1]

    # Winner Takes All
    df["prob_buy"] = np.where(raw_buy > raw_sell, raw_buy, 0.0)
    df["prob_sell"] = np.where(raw_sell > raw_buy, raw_sell, 0.0)

    # Ejecutar backtest con umbral 0.42 (Alta Frecuencia Institucional)
    print("\n" + "#"*80 + "\nESCENARIO A: UMBRAL 0.42 (DECILES 8-9)\n" + "#"*80)
    run_portfolio_backtest(
        df,
        threshold_buy=0.42,
        threshold_sell=0.42,
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        max_open_positions=8,
        friction_pct=0.0010,
        filter_morning_traps=True,
    )

    # Ejecutar backtest con umbral 0.45 (Máxima Precisión Institucional)
    print("\n" + "#"*80 + "\nESCENARIO B: UMBRAL 0.45 (DECIL 9 - MÁXIMA PRECISIÓN)\n" + "#"*80)
    run_portfolio_backtest(
        df,
        threshold_buy=0.45,
        threshold_sell=0.45,
        initial_capital=10000.0,
        risk_per_trade_pct=0.02,
        max_open_positions=8,
        friction_pct=0.0010,
        filter_morning_traps=True,
    )


if __name__ == "__main__":
    main()

