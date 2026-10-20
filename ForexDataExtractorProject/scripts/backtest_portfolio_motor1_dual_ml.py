"""
backtest_portfolio_motor1_dual_ml.py -
Simulación Histórica Realista de Portafolio (Walk-Forward 2022-2026 y OOS 2025-2026)
para el Motor 1 (Arquitectura Dual en Cascada: Modelo 1A Entrada + Modelo 1B Salida Condicionada al Tipo de Caída).

Reglas Estrictas de Realismo (Zero Lookahead Bias):
1. Universo elegible: Acciones líquidas con ATR Diario >= 2.2% en el momento de la señal (causal).
2. Ejecución: Entrada estricta en Open[t+1] + 5 bps de slippage/comisión; Salida con -5 bps (10 bps round-trip).
3. El Modelo 1B usa explícitamente:
   - La probabilidad del Modelo 1A (`prob_1a`).
   - El Tipo de Caída de la que viene (Post-Earnings Negativo/Positivo, Caída Macro SPY, Caída Sectorial, Caída Idiosincrática).
   - La velocidad y velas de recuperación en vivo.
4. Regla de Cobro Adaptativa según Tipo de Caída + Modelo 1B:
   - Si viene de Castigo de Balance (Post-Earnings) o `prob_1b` baja (< 0.42):
     Asegura ganancias temprano desde el 35%-45% de recuperación con candado estricto (-0.40x ATR) o cobro directo en 55%.
   - Si viene de Caída Idiosincrática / Sectorial con `prob_1b` media (0.42 - 0.54):
     Candado Progresivo + Trailing Medio (-0.65x ATR desde el 50% de recuperación).
   - Si viene de Caída Macro (SPY) / Sectorial con `prob_1b` alta (>= 0.54):
     Suelta la rienda con Trailing Dinámico holgado (-0.90x ATR sin techo fijo) para capturar recuperaciones del 85%-120%+.
5. Gestión de Portafolio Real:
   - Capital inicial: $25,000 USD (sin apalancamiento, 100% cash).
   - Máximo N posiciones abiertas simultáneas (ej. 5 slots de 20% cada uno).
   - Si una acción ya está en cartera, no se duplica posición.
"""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from scripts.evaluate_multisector_motor1 import compute_core_user_features_per_ticker, deduplicate_events


SHOCK_TYPE_MAP = {
    "1_PostEarnings_Negativo": 1,
    "2_PostEarnings_Positivo": 2,
    "3_Caida_Macro_Mercado": 3,
    "4_Caida_Sectorial": 4,
    "5_Caida_Idiosincratica": 5,
}


def classify_shock_origin(row) -> str:
    if row["days_since_earnings"] <= 5.0:
        if row["eps_surprise_pct"] < 0:
            return "1_PostEarnings_Negativo"
        else:
            return "2_PostEarnings_Positivo"
    if row["spy_ret_3d"] <= -0.022 or row["spy_ret_5d"] <= -0.030:
        return "3_Caida_Macro_Mercado"
    if row["sector_ret_3d"] <= -0.022 or row["sector_ret_5d"] <= -0.030:
        return "4_Caida_Sectorial"
    return "5_Caida_Idiosincratica"


def simulate_adaptive_exit_by_shock_and_1b(
    fut_df: pd.DataFrame,
    entry_p: float,
    sl_p: float,
    atr_d: float,
    shock_drop: float,
    shock_type_code: int,
    prob_1b: float,
):
    """
    Simula hora por hora la salida usando TANTO el Tipo de Caída como la probabilidad del Modelo 1B:
    - En caídas Post-Earnings ( códigos 1 y 2 ) o cuando Modelo 1B ve agotamiento (prob_1b < 0.43):
      El techo suele estar en 45%-55%. Apenas toca 40% asegura +25%, y apenas toca 52% ajusta trailing muy corto (-0.35x ATR).
    - En caídas Idiosincráticas o prob_1b media (0.43 <= prob_1b < 0.53):
      Al tocar 45% pone breakeven (+0.3%), al tocar 55% activa trailing medio (-0.65x ATR).
    - En caídas Macro / Sectoriales o con alta convicción de Modelo 1B (prob_1b >= 0.53):
      Al tocar 45% pone breakeven, y desde el 55% deja correr con trailing holgado (-0.90x ATR) sin techo fijo para capturar el 85%-120%+.
    """
    highs = fut_df["high"].to_numpy()
    closes = fut_df["close"].to_numpy()
    timestamps = fut_df["timestamp"].to_numpy()

    floor_p = None
    peak_p = entry_p
    exit_p = closes[-1]
    exit_ts = timestamps[-1]
    exit_reason = "TimeStop_15d"

    is_earnings_shock = shock_type_code in (1, 2)
    is_macro_or_sector = shock_type_code in (3, 4)

    for step in range(len(fut_df)):
        if highs[step] > peak_p:
            peak_p = highs[step]
        max_rec = ((peak_p - entry_p) / entry_p) / shock_drop

        # Regla adaptativa de candado / trailing según Tipo de Caída + Modelo 1B
        if is_earnings_shock or prob_1b < 0.43:
            # Modo Cosecha Rápida (Post-Earnings o Agotamiento detectado por 1B)
            if max_rec >= 0.52:
                trail_dist = 0.35 * atr_d
                floor_p = max(floor_p or 0.0, peak_p * (1.0 - trail_dist), entry_p * (1.0 + 0.42 * shock_drop))
            elif max_rec >= 0.38:
                floor_p = max(floor_p or 0.0, entry_p * (1.0 + 0.22 * shock_drop))
        elif is_macro_or_sector and prob_1b >= 0.52:
            # Modo Home Run (Caída Macro/Sectorial con fuerte respaldo de 1A+1B)
            if max_rec >= 0.55:
                trail_dist = 0.90 * atr_d
                floor_p = max(floor_p or 0.0, peak_p * (1.0 - trail_dist), entry_p * (1.0 + 0.38 * shock_drop))
            elif max_rec >= 0.42:
                floor_p = max(floor_p or 0.0, entry_p * 1.003)
        else:
            # Modo Equilibrado (Caída Idiosincrática / Sectorial estándar)
            if max_rec >= 0.52:
                trail_dist = 0.65 * atr_d
                floor_p = max(floor_p or 0.0, peak_p * (1.0 - trail_dist), entry_p * (1.0 + 0.38 * shock_drop))
            elif max_rec >= 0.40:
                floor_p = max(floor_p or 0.0, entry_p * 1.003)

        # Verificar salida por Candado / Trailing
        if floor_p is not None and closes[step] <= floor_p and step > 0:
            exit_p = floor_p
            exit_ts = timestamps[step]
            exit_reason = "Adaptive_1B_Trail"
            break

        # Verificar Stop Loss Estructural al cierre diario (cada 7 barras horarias)
        if (step + 1) % 7 == 0 and closes[step] <= sl_p:
            exit_p = closes[step]
            exit_ts = timestamps[step]
            exit_reason = "DailyClose_SL"
            break

    net_ret_pct = ((exit_p * 0.9995 - entry_p) / entry_p) * 100.0
    return net_ret_pct, pd.Timestamp(exit_ts), exit_reason


def run_portfolio_simulation(
    trades_df: pd.DataFrame,
    initial_capital: float = 25000.0,
    max_slots: int = 5,
):
    """
    Simula una cuenta real de trading ($25,000 USD iniciales) con máximo `max_slots` posiciones abiertas al mismo tiempo.
    Ordena las señales cronológicamente por `entry_ts` (y desempata por mayor `prob_1a`).
    """
    if trades_df.empty:
        return {}

    candidates = trades_df.sort_values(["entry_ts", "prob_1a"], ascending=[True, False]).reset_index(drop=True)
    cash = initial_capital
    open_positions = []  # lista de dicts: {'ticker', 'exit_ts', 'invested', 'proceeds', 'ret_pct'}
    executed_trades = []
    equity_curve = []

    for _, row in candidates.iterrows():
        cur_ts = row["entry_ts"]

        # 1. Cerrar todas las posiciones cuya fecha de salida (`exit_ts`) sea <= `cur_ts`
        still_open = []
        for pos in open_positions:
            if pos["exit_ts"] <= cur_ts:
                cash += pos["proceeds"]
                executed_trades.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. Verificar si hay slot disponible y si el ticker NO está ya abierto en cartera
        active_tickers = {p["ticker"] for p in open_positions}
        if len(open_positions) < max_slots and row["ticker"] not in active_tickers:
            # Asignar 1 / slots_libres del efectivo disponible (igual peso relativo respecto a la equidad actual)
            slots_free = max_slots - len(open_positions)
            alloc = cash / slots_free
            if alloc >= 500.0:
                cash -= alloc
                ret_frac = row["net_ret_pct"] / 100.0
                proceeds = alloc * (1.0 + ret_frac)
                open_positions.append({
                    "ticker": row["ticker"],
                    "sector": row["sector"],
                    "shock_origin_type": row["shock_origin_type"],
                    "entry_ts": cur_ts,
                    "exit_ts": row["exit_ts"],
                    "invested": alloc,
                    "proceeds": proceeds,
                    "net_ret_pct": row["net_ret_pct"],
                    "pnl_usd": proceeds - alloc,
                    "prob_1a": row["prob_1a"],
                    "prob_1b": row["prob_1b"],
                    "exit_reason": row["exit_reason"],
                })

        # Registrar equidad marcada al costo + cierres realizados
        cur_eq = cash + sum(p["invested"] for p in open_positions)
        equity_curve.append({"timestamp": cur_ts, "equity": cur_eq})

    # Cerrar remanentes al final del período manteniendo en la equidad las posiciones aún abiertas
    remaining = sorted(open_positions, key=lambda x: x["exit_ts"])
    for idx_rem, pos in enumerate(remaining):
        cash += pos["proceeds"]
        executed_trades.append(pos)
        still_invested = sum(p["invested"] for p in remaining[idx_rem + 1 :])
        equity_curve.append({"timestamp": pos["exit_ts"], "equity": cash + still_invested})

    exec_df = pd.DataFrame(executed_trades)
    eq_df = pd.DataFrame(equity_curve).sort_values("timestamp").reset_index(drop=True)
    if exec_df.empty:
        return {}

    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd_pct"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"] * 100.0
    max_dd_pct = eq_df["dd_pct"].min()

    rets = exec_df["net_ret_pct"].to_numpy()
    wins = exec_df.loc[exec_df["pnl_usd"] > 0, "pnl_usd"].sum()
    losses = abs(exec_df.loc[exec_df["pnl_usd"] <= 0, "pnl_usd"].sum())
    pf = wins / losses if losses > 0 else 99.0

    days_span = max((exec_df["exit_ts"].max() - exec_df["entry_ts"].min()).days, 30)
    years_span = days_span / 365.25
    total_return_pct = (cash / initial_capital - 1.0) * 100.0
    cagr_pct = ((cash / initial_capital) ** (1.0 / years_span) - 1.0) * 100.0 if cash > 0 else -100.0

    # Sharpe anualizado aproximado sobre retornos de portafolio
    eq_daily = eq_df.set_index("timestamp")["equity"].resample("1D").last().ffill()
    daily_rets = eq_daily.pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0.0

    return {
        "final_capital": cash,
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "max_dd_pct": max_dd_pct,
        "sharpe": sharpe,
        "trades_executed": len(exec_df),
        "win_rate_pct": (rets > 0).mean() * 100.0,
        "avg_trade_ret_pct": rets.mean(),
        "profit_factor": pf,
        "exec_df": exec_df,
        "eq_df": eq_df,
    }


def main():
    t0 = time.time()
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    earn_path = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"

    print("[1/4] Preparando datos horarios multi-sector (314 empresas de alto ATR) + S&P 500 + Earnings...")
    df = pd.read_parquet(hourly_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    df["hour_key"] = df["timestamp"].dt.floor("h")

    spy = df[df["ticker"] == "SPY"].sort_values("timestamp").drop_duplicates("hour_key").set_index("hour_key")
    spy_close_map = spy["close"]
    spy_ret_3d_map = spy["close"].pct_change(21)
    spy_ret_5d_map = spy["close"].pct_change(35)

    stocks = df[~df["ticker"].isin(["SPY", "QQQ"])].copy().reset_index(drop=True)
    stocks["spy_close"] = stocks["hour_key"].map(spy_close_map).ffill()
    stocks["spy_ret_3d"] = stocks["hour_key"].map(spy_ret_3d_map).fillna(0.0)
    stocks["spy_ret_5d"] = stocks["hour_key"].map(spy_ret_5d_map).fillna(0.0)

    g = stocks.groupby("ticker", group_keys=False)
    stocks["ret_1h"] = g["close"].pct_change(1)
    stocks["ret_3d"] = g["close"].pct_change(21)
    stocks["ret_5d"] = g["close"].pct_change(35)
    stocks["ret_60d"] = g["close"].pct_change(420)
    stocks["vol_3d"] = g["ret_3d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["vol_5d"] = g["ret_5d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["z_3d"] = stocks["ret_3d"] / stocks["vol_3d"].replace(0, np.nan)
    stocks["z_5d"] = stocks["ret_5d"] / stocks["vol_5d"].replace(0, np.nan)

    sec_hourly = stocks.groupby(["sector", "hour_key"]).agg(
        sector_ret_1h=("ret_1h", "mean"),
        sector_ret_3d=("ret_3d", "mean"),
        sector_ret_5d=("ret_5d", "mean"),
    ).reset_index().sort_values(["sector", "hour_key"]).reset_index(drop=True)
    sec_hourly["sector_idx"] = sec_hourly.groupby("sector")["sector_ret_1h"].transform(
        lambda s: (1.0 + s.fillna(0.0)).cumprod()
    )

    stocks = stocks.merge(
        sec_hourly[["sector", "hour_key", "sector_ret_3d", "sector_ret_5d", "sector_idx"]],
        on=["sector", "hour_key"],
        how="left",
    )

    stocks["excess_vs_spy_3d"] = stocks["ret_3d"] - stocks["spy_ret_3d"]
    stocks["excess_vs_sector_3d"] = stocks["ret_3d"] - stocks["sector_ret_3d"]
    stocks["sector_vs_spy_3d"] = stocks["sector_ret_3d"] - stocks["spy_ret_3d"]
    stocks["idiosyncratic_ratio_3d"] = (
        (stocks["ret_3d"] - stocks["sector_ret_3d"]) / stocks["ret_3d"].clip(upper=-0.005)
    ).clip(0.0, 1.5)

    high_1d = g["high"].transform(lambda s: s.rolling(7, min_periods=3).max())
    low_1d = g["low"].transform(lambda s: s.rolling(7, min_periods=3).min())
    stocks["atr_daily_pct"] = ((high_1d - low_1d) / stocks["close"] * 100.0).groupby(stocks["ticker"]).transform(
        lambda s: s.rolling(98, min_periods=21).mean()
    )

    valid_tickers = stocks.groupby("ticker")["atr_daily_pct"].max()
    eligible_tickers = valid_tickers[valid_tickers >= 2.2].index.tolist()
    stocks = stocks[stocks["ticker"].isin(eligible_tickers)].copy().reset_index(drop=True)

    ticker_dfs = {}
    shock_candidates = []
    for tk, sub in stocks.groupby("ticker"):
        sub_feat = compute_core_user_features_per_ticker(sub)
        rng = (sub_feat["high"] - sub_feat["low"]).replace(0, np.nan)
        sub_feat["upper_wick_ratio"] = (sub_feat["high"] - np.maximum(sub_feat["open"], sub_feat["close"])) / rng
        sub_feat["rsi_slope_3h"] = sub_feat["RSI_14"].diff(3)
        sub_feat = sub_feat.reset_index(drop=True)
        ticker_dfs[tk] = sub_feat

        mask = (
            (sub_feat["atr_daily_pct"] >= 2.2)
            & ((sub_feat["z_3d"] <= -2.0) | (sub_feat["z_5d"] <= -2.0))
            & ((sub_feat["close_loc_1h"] >= 0.55) | (sub_feat["lower_wick_ratio"] >= 0.25))
        )
        sc = sub_feat[mask].copy()
        if not sc.empty:
            sc["bar_idx"] = sc.index
            shock_candidates.append(sc)

    shocks_raw = pd.concat(shock_candidates, ignore_index=True)

    if earn_path.exists():
        earn_df = pd.read_parquet(earn_path)
        earn_df["earnings_date"] = pd.to_datetime(earn_df["earnings_date"], utc=True)
        earn_df = earn_df.sort_values(["ticker", "earnings_date"]).reset_index(drop=True)
        earn_df["eps_beat_4q_ratio"] = earn_df.groupby("ticker")["eps_beat"].transform(lambda s: s.rolling(4, min_periods=1).mean())
        earn_df["eps_surprise_4q_mean"] = earn_df.groupby("ticker")["eps_surprise_pct"].transform(lambda s: s.rolling(4, min_periods=1).mean().clip(-100, 200))

        shocks_raw = shocks_raw.sort_values("timestamp").reset_index(drop=True)
        earn_sorted = earn_df[["ticker", "earnings_date", "eps_surprise_pct", "eps_positive", "eps_beat", "eps_beat_4q_ratio", "eps_surprise_4q_mean"]].sort_values("earnings_date").reset_index(drop=True)
        shocks_raw = pd.merge_asof(shocks_raw, earn_sorted, left_on="timestamp", right_on="earnings_date", by="ticker", direction="backward")
        shocks_raw["days_since_earnings"] = ((shocks_raw["timestamp"] - shocks_raw["earnings_date"]).dt.total_seconds() / 86400.0).clip(0, 120).fillna(45.0)
        shocks_raw["eps_surprise_pct"] = shocks_raw["eps_surprise_pct"].clip(-100, 200).fillna(0.0)
        shocks_raw["eps_positive"] = shocks_raw["eps_positive"].fillna(1.0)
        shocks_raw["eps_beat_4q_ratio"] = shocks_raw["eps_beat_4q_ratio"].fillna(0.5)
        shocks_raw["eps_surprise_4q_mean"] = shocks_raw["eps_surprise_4q_mean"].fillna(0.0)

    shocks = deduplicate_events(shocks_raw, "timestamp", min_gap_hours=48)
    shocks["is_earnings_window"] = (shocks["days_since_earnings"] <= 5.0).astype(float)
    shocks["earnings_window_x_surprise"] = shocks["is_earnings_window"] * shocks["eps_surprise_pct"]
    shocks["is_bad_earnings_shock"] = ((shocks["days_since_earnings"] <= 5.0) & (shocks["eps_surprise_pct"] < 0)).astype(float)
    shocks["is_good_earnings_shock"] = ((shocks["days_since_earnings"] <= 5.0) & (shocks["eps_surprise_pct"] >= 0)).astype(float)
    shocks["shock_origin_type"] = shocks.apply(classify_shock_origin, axis=1)
    shocks["shock_type_code"] = shocks["shock_origin_type"].map(SHOCK_TYPE_MAP)

    print("[2/4] Construyendo eventos y snapshots para el Modelo 1A y Modelo 1B...")
    valid_shocks = []
    meli_snapshots = []

    for event_id, row in shocks.iterrows():
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        if i0 + 110 >= len(tdf):
            continue

        atr_d = row["atr_daily_pct"] / 100.0
        shock_drop = abs(min(row["ret_3d"], row["ret_5d"], -0.04))
        entry_p = tdf.loc[i0 + 1, "open"] * 1.0005
        entry_ts = pd.Timestamp(tdf.loc[i0 + 1, "timestamp"])
        sl_p = entry_p * (1.0 - 1.5 * atr_d)
        spy_entry = tdf.loc[i0 + 1, "spy_close"]
        sec_entry = tdf.loc[i0 + 1, "sector_idx"]

        fut = tdf.iloc[i0 + 1 : i0 + 106]
        highs = fut["high"].to_numpy()
        closes = fut["close"].to_numpy()

        max_rec = 0.0
        floor_trail = None
        peak_t = entry_p
        exit_trail = closes[-1]
        for step in range(len(fut)):
            if highs[step] > peak_t:
                peak_t = highs[step]
            m_rec = ((peak_t - entry_p) / entry_p) / shock_drop
            if m_rec > max_rec:
                max_rec = m_rec
            if m_rec >= 0.50:
                floor_trail = max(floor_trail or 0.0, peak_t * (1.0 - 0.8 * atr_d))
            if floor_trail is not None and closes[step] <= floor_trail and step > 0:
                exit_trail = floor_trail
                break
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                exit_trail = closes[step]
                break
        ret_trail = ((exit_trail * 0.9995 - entry_p) / entry_p) * 100.0

        # Capturar hito del 45% de recuperación para preguntar al Modelo 1B en tiempo real
        for step in range(len(fut) - 7):
            if (step + 1) % 7 == 0 and closes[step] <= sl_p:
                break
            cur_h = highs[step]
            rec_here = ((cur_h - entry_p) / entry_p) / shock_drop
            if rec_here >= 0.45:
                rem_highs = highs[step + 1 :]
                rem_closes = closes[step + 1 :]
                if len(rem_highs) == 0:
                    break
                hit_85_after = np.max(rem_highs) >= entry_p * (1.0 + 0.85 * shock_drop)
                crashed_after = np.min(rem_closes) <= entry_p * (1.0 + 0.15 * shock_drop)
                keep_running = 1 if (hit_85_after and not crashed_after) else 0

                cur_bar = tdf.loc[i0 + 1 + step]
                spy_rebound = (cur_bar["spy_close"] - spy_entry) / spy_entry if spy_entry > 0 else 0.0
                sec_rebound = (cur_bar["sector_idx"] - sec_entry) / sec_entry if sec_entry > 0 else 0.0
                stock_rebound = 0.45 * shock_drop

                meli_snapshots.append({
                    "event_id": event_id,
                    "timestamp": cur_bar["timestamp"],
                    "steps_to_reach_50": step + 1,
                    "velocity_pct_per_bar": (stock_rebound * 100.0) / max(step + 1, 1),
                    "RSI_14_at_50": cur_bar["RSI_14"],
                    "rsi_slope_3h_at_50": cur_bar["rsi_slope_3h"],
                    "rvol_1h_at_50": cur_bar["rvol_1h"],
                    "close_loc_1h_at_50": cur_bar["close_loc_1h"],
                    "upper_wick_at_50": cur_bar["upper_wick_ratio"],
                    "sr_dist_res_atr_at_50": cur_bar["sr_dist_resistance_atr"],
                    "sr_channel_pos_at_50": cur_bar["sr_channel_position"],
                    "close_diff_SMA_100_at_50": cur_bar["close_diff_SMA_100"],
                    "BB_percent_b_at_50": cur_bar["BB_20_20_percent_b"],
                    "spy_rebound_since_entry": spy_rebound,
                    "sector_rebound_since_entry": sec_rebound,
                    "rebound_excess_vs_sector": stock_rebound - sec_rebound,
                    "shock_drop": shock_drop,
                    "shock_type_code": row["shock_type_code"],
                    "entry_z_3d": row["z_3d"],
                    "entry_z_5d": row["z_5d"],
                    "entry_spy_ret_3d": row["spy_ret_3d"],
                    "entry_sector_ret_3d": row["sector_ret_3d"],
                    "entry_excess_vs_sector_3d": row["excess_vs_sector_3d"],
                    "entry_idiosyncratic_ratio": row["idiosyncratic_ratio_3d"],
                    "entry_days_since_earnings": row["days_since_earnings"],
                    "entry_eps_surprise_pct": row["eps_surprise_pct"],
                    "entry_eps_surprise_4q_mean": row["eps_surprise_4q_mean"],
                    "entry_is_bad_earnings": row["is_bad_earnings_shock"],
                    "entry_is_good_earnings": row["is_good_earnings_shock"],
                    "entry_SMA_100_momentum": row["SMA_100_pct_momentum"],
                    "entry_close_diff_SMA_100": row["close_diff_SMA_100"],
                    "keep_running": keep_running,
                })
                break

        row_dict = row.to_dict()
        row_dict["event_id"] = event_id
        row_dict["entry_ts"] = entry_ts
        row_dict["entry_p"] = entry_p
        row_dict["sl_p"] = sl_p
        row_dict["atr_d"] = atr_d
        row_dict["shock_drop"] = shock_drop
        row_dict["target_1a"] = 1 if (ret_trail > 1.0 and max_rec >= 0.55) else 0
        valid_shocks.append(row_dict)

    df_1a = pd.DataFrame(valid_shocks).sort_values("timestamp").reset_index(drop=True)
    df_meli = pd.DataFrame(meli_snapshots).sort_values("timestamp").reset_index(drop=True)

    feats_1a = [
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
        "is_bad_earnings_shock", "is_good_earnings_shock", "shock_type_code"
    ]
    feats_1a = [f for f in feats_1a if f in df_1a.columns]

    feats_1b = [
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
        "entry_SMA_100_momentum", "entry_close_diff_SMA_100"
    ]

    print("[3/4] Entrenando Modelos 1A y 1B (70% Train / 30% Out-of-Sample Ciego) y simulando salidas...")
    split_idx = int(len(df_1a) * 0.70)
    split_date = df_1a["timestamp"].iloc[split_idx]
    tr_1a = df_1a.iloc[:split_idx].copy().reset_index(drop=True)
    te_1a = df_1a.iloc[split_idx:].copy().reset_index(drop=True)

    # OOF Probabilities en Train para entrenar Modelo 1B sin sesgo
    oof_probs_1a = np.zeros(len(tr_1a))
    kf = KFold(n_splits=5, shuffle=False)
    for idx_tr, idx_val in kf.split(tr_1a):
        m_fold = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
            min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
        )
        m_fold.fit(tr_1a.iloc[idx_tr][feats_1a].fillna(0.0), tr_1a.iloc[idx_tr]["target_1a"].values)
        oof_probs_1a[idx_val] = m_fold.predict_proba(tr_1a.iloc[idx_val][feats_1a].fillna(0.0))[:, 1]

    clf_1a = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=4, num_leaves=12,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1a.fit(tr_1a[feats_1a].fillna(0.0), tr_1a["target_1a"].values)
    probs_te_1a = clf_1a.predict_proba(te_1a[feats_1a].fillna(0.0))[:, 1]

    tr_1a["prob_1a"] = oof_probs_1a
    te_1a["prob_1a"] = probs_te_1a

    # Umbrales aprendidos ESTRICTAMENTE en el conjunto de Train (sin mirar Test)
    q90_thresh = np.quantile(tr_1a["prob_1a"], 0.90)
    q80_thresh = np.quantile(tr_1a["prob_1a"], 0.80)
    q70_thresh = np.quantile(tr_1a["prob_1a"], 0.70)

    prob_1a_map = {}
    for _, r in tr_1a.iterrows():
        prob_1a_map[int(r["event_id"])] = float(r["prob_1a"])
    for _, r in te_1a.iterrows():
        prob_1a_map[int(r["event_id"])] = float(r["prob_1a"])

    df_meli["model_1a_prob"] = df_meli["event_id"].map(prob_1a_map).fillna(0.5)
    tr_m = df_meli[df_meli["timestamp"] < split_date].copy()
    te_m = df_meli[df_meli["timestamp"] >= split_date].copy()

    clf_1b = lgb.LGBMClassifier(
        n_estimators=140, learning_rate=0.03, max_depth=3, num_leaves=8,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=8.0, random_state=42, verbose=-1
    )
    clf_1b.fit(tr_m[feats_1b].fillna(0.0), tr_m["keep_running"].values)
    te_m["prob_1b"] = clf_1b.predict_proba(te_m[feats_1b].fillna(0.0))[:, 1]
    prob_1b_map = {int(r["event_id"]): float(r["prob_1b"]) for _, r in te_m.iterrows()}

    # Simular barra por barra en el período OOS la salida adaptativa guiada por Tipo de Caída + Modelo 1B
    oos_sim_rows = []
    for _, row in te_1a.iterrows():
        ev_id = int(row["event_id"])
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        fut_df = tdf.iloc[i0 + 1 : i0 + 106]
        p_1b = prob_1b_map.get(ev_id, 0.50)

        net_ret, exit_ts, exit_reason = simulate_adaptive_exit_by_shock_and_1b(
            fut_df=fut_df,
            entry_p=float(row["entry_p"]),
            sl_p=float(row["sl_p"]),
            atr_d=float(row["atr_d"]),
            shock_drop=float(row["shock_drop"]),
            shock_type_code=int(row["shock_type_code"]),
            prob_1b=p_1b,
        )
        r_dict = row.to_dict()
        r_dict["prob_1b"] = p_1b
        r_dict["net_ret_pct"] = net_ret
        r_dict["exit_ts"] = exit_ts
        r_dict["exit_reason"] = exit_reason
        oos_sim_rows.append(r_dict)

    oos_df = pd.DataFrame(oos_sim_rows).sort_values("entry_ts").reset_index(drop=True)

    # Benchmark SPY Buy & Hold en exactamente el mismo período OOS
    oos_start = oos_df["entry_ts"].min()
    oos_end = oos_df["exit_ts"].max()
    spy_oos = spy[(spy.index >= oos_start) & (spy.index <= oos_end)]
    spy_bh_ret = ((spy_oos["close"].iloc[-1] / spy_oos["close"].iloc[0]) - 1.0) * 100.0 if len(spy_oos) > 1 else 0.0
    spy_peak = spy_oos["close"].cummax()
    spy_max_dd = ((spy_oos["close"] - spy_peak) / spy_peak * 100.0).min() if len(spy_oos) > 1 else 0.0

    print("\n" + "=" * 120)
    print(f"4. SIMULACIÓN DE PORTAFOLIO OUT-OF-SAMPLE CIEGO ({oos_start.strftime('%Y-%m-%d')} a {oos_end.strftime('%Y-%m-%d')})")
    print(f"   Capital Inicial: $25,000 USD | Fricción: 10 bps round-trip | Ejecución: Open[t+1] | Sin Apalancamiento (Cash)")
    print(f"   Benchmark S&P 500 (SPY Buy & Hold en el mismo período): Retorno = +{spy_bh_ret:.2f}% | Max Drawdown = {spy_max_dd:.2f}%")
    print("=" * 120)

    configs = [
        ("1. Sin Filtro ML (Todos los Golpes, 5 Slots)", oos_df, 5),
        (f"2. Modelo 1A Moderado (prob >= {q70_thresh:.2f}, 5 Slots)", oos_df[oos_df["prob_1a"] >= q70_thresh], 5),
        (f"3. Modelo 1A Óptimo Decil 9 (prob >= {q80_thresh:.2f}, 5 Slots)", oos_df[oos_df["prob_1a"] >= q80_thresh], 5),
        (f"4. Modelo 1A Óptimo Decil 9 (prob >= {q80_thresh:.2f}, 8 Slots)", oos_df[oos_df["prob_1a"] >= q80_thresh], 8),
        (f"5. Modelo 1A Francotirador Extremo (prob >= {q90_thresh:.2f}, 5 Slots)", oos_df[oos_df["prob_1a"] >= q90_thresh], 5),
    ]

    summary_rows = []
    best_res = None
    for name, sub_candidates, slots in configs:
        res = run_portfolio_simulation(sub_candidates, initial_capital=25000.0, max_slots=slots)
        if not res:
            continue
        if "3. Modelo 1A Óptimo Decil 9" in name:
            best_res = res
        summary_rows.append({
            "Configuración_Portafolio": name,
            "Trades_Ejecutados": res["trades_executed"],
            "Win_Rate %": round(res["win_rate_pct"], 1),
            "Ret_Medio_Trade %": round(res["avg_trade_ret_pct"], 2),
            "Profit_Factor": round(res["profit_factor"], 2),
            "Capital_Final ($)": f"${res['final_capital']:,.0f}",
            "Retorno_Total %": f"+{res['total_return_pct']:.1f}%",
            "Max_Drawdown %": f"{res['max_dd_pct']:.2f}%",
            "Sharpe_Anual": round(res["sharpe"], 2),
        })

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("=" * 120)

    if best_res is not None:
        exec_df = best_res["exec_df"]
        print(f"\nDESGLOSE POR TIPO DE CAÍDA EN EL PORTAFOLIO ÓPTIMO (prob_1a >= {q80_thresh:.2f} + SALIDA ADAPTATIVA 1B, 5 SLOTS):")
        by_origin = exec_df.groupby("shock_origin_type").agg(
            Trades=("net_ret_pct", "count"),
            Win_Rate_pct=("net_ret_pct", lambda s: round((s > 0).mean() * 100, 1)),
            Ret_Medio_pct=("net_ret_pct", lambda s: round(s.mean(), 2)),
            PnL_Total_USD=("pnl_usd", lambda s: round(s.sum(), 0)),
        ).reset_index()
        print(by_origin.to_string(index=False))

        print("\nDESGLOSE MENSUAL DE LA CURVA DE CAPITAL (DECIL 9 + SALIDA 1B, 5 SLOTS):")
        exec_df["month"] = exec_df["exit_ts"].dt.strftime("%Y-%m")
        monthly = exec_df.groupby("month").agg(
            Trades_Cerrados=("net_ret_pct", "count"),
            Win_Rate_pct=("net_ret_pct", lambda s: round((s > 0).mean() * 100, 1)),
            Ret_Medio_Trade_pct=("net_ret_pct", lambda s: round(s.mean(), 2)),
            Ganancia_Neta_USD=("pnl_usd", lambda s: round(s.sum(), 0)),
        ).reset_index()
        print(monthly.to_string(index=False))

        print("\nÚLTIMAS 12 OPERACIONES REALES EJECUTADAS POR EL PORTAFOLIO EN OOS:")
        cols_show = ["ticker", "sector", "shock_origin_type", "entry_ts", "exit_ts", "prob_1a", "prob_1b", "net_ret_pct", "pnl_usd", "exit_reason"]
        tail_df = exec_df[cols_show].tail(12).copy()
        tail_df["entry_ts"] = tail_df["entry_ts"].dt.strftime("%Y-%m-%d %H:%M")
        tail_df["exit_ts"] = tail_df["exit_ts"].dt.strftime("%Y-%m-%d %H:%M")
        tail_df["prob_1a"] = tail_df["prob_1a"].round(3)
        tail_df["prob_1b"] = tail_df["prob_1b"].round(3)
        tail_df["net_ret_pct"] = tail_df["net_ret_pct"].round(2)
        tail_df["pnl_usd"] = tail_df["pnl_usd"].round(0)
        print(tail_df.to_string(index=False))

    print("=" * 120)
    print(f"Tiempo total de simulación: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
