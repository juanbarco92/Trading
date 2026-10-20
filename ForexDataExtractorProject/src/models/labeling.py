"""
labeling.py - TICK-07 & TICK-08: Triple Barrier Labeling con Fricción Sintética y Costes Reales.

Implementación institucional del Triple Barrier Method (Marcos López de Prado)
calibrado para Swing Trading en acciones y ETFs (velas 1H, horizonte 2-5 días):

1. Barrera Superior (Take Profit):
   P_entry + (k_tp * ATR14) + margen de fricción
2. Barrera Inferior (Stop Loss):
   P_entry - (k_sl * ATR14) - margen de fricción
3. Barrera Temporal (Vertical Barrier):
   Límite de velas (horizon_bars = 26, aprox. 4 días de mercado).
4. Inyección de Fricción Sintética (TICK-08):
   - Compra ejecutada a P_entry * (1 + friction_pct)
   - Venta ejecutada a P_exit * (1 - friction_pct)
   - Penalización de slippage en Stop Loss ante gap-downs de apertura: min(SL, Open_exit)
   - Clases estrictas: Solo operaciones con retorno neto positivo que tocan TP
     se marcan como `target_quality_trade = 1`.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
from numpy.lib.stride_tricks import sliding_window_view


def compute_atr_series(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calcula el ATR estándar sobre un DataFrame con 'high', 'low', 'close'."""
    h = df["high"]
    l = df["low"]
    c_prev = df["close"].shift(1).fillna(df["open"])

    tr1 = h - l
    tr2 = (h - c_prev).abs()
    tr3 = (l - c_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=max(1, window // 2)).mean().bfill()
    return atr


def _label_single_series(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    atr: np.ndarray,
    k_tp: float = 3.0,
    k_sl: float = 1.5,
    horizon_bars: int = 26,
    friction_pct: float = 0.0005,
    conservative_intrabar: bool = True,
    adjust_barriers_for_friction: bool = True,
) -> dict:
    """
    Etiqueta de forma 100% vectorizada una serie univariada de un ticker.
    """
    n = len(close)
    valid_len = n - horizon_bars

    # Salidas pre-asignadas
    labels = np.full(n, np.nan, dtype=np.float64)
    quality = np.full(n, np.nan, dtype=np.float64)
    exit_bars = np.full(n, np.nan, dtype=np.float64)
    exit_types = np.array(["INCOMPLETE"] * n, dtype=object)
    exit_prices = np.full(n, np.nan, dtype=np.float64)
    gross_returns = np.full(n, np.nan, dtype=np.float64)
    net_returns = np.full(n, np.nan, dtype=np.float64)
    tp_prices = np.full(n, np.nan, dtype=np.float64)
    sl_prices = np.full(n, np.nan, dtype=np.float64)

    if valid_len <= 0:
        return {
            "target_triple_barrier": labels,
            "target_quality_trade": quality,
            "trade_exit_bar": exit_bars,
            "trade_exit_type": exit_types,
            "trade_exit_price": exit_prices,
            "trade_gross_return": gross_returns,
            "trade_net_return": net_returns,
            "trade_tp_price": tp_prices,
            "trade_sl_price": sl_prices,
        }

    # Slice para filas válidas de entrada
    p_entry = close[:valid_len]
    atr_entry = atr[:valid_len]

    # Cálculo de barreras con/sin ajuste de fricción
    if adjust_barriers_for_friction:
        # El precio de salida debe cubrir los 2 * friction_pct de ida y vuelta
        tp = p_entry + (k_tp * atr_entry) + (2.0 * friction_pct * p_entry)
        sl = p_entry - (k_sl * atr_entry) - (2.0 * friction_pct * p_entry)
    else:
        tp = p_entry + (k_tp * atr_entry)
        sl = p_entry - (k_sl * atr_entry)

    tp_prices[:valid_len] = tp
    sl_prices[:valid_len] = sl

    # Ventana deslizante para las siguientes horizon_bars velas (high[1:], low[1:], etc.)
    # shape: (n - horizon_bars, horizon_bars)
    future_highs = sliding_window_view(high[1:], horizon_bars)[:valid_len]
    future_lows = sliding_window_view(low[1:], horizon_bars)[:valid_len]
    future_opens = sliding_window_view(open_[1:], horizon_bars)[:valid_len]
    future_closes = sliding_window_view(close[1:], horizon_bars)[:valid_len]

    # Evaluaciones booleanas de toque de barreras
    tp_hits = future_highs >= tp[:, None]
    sl_hits = future_lows <= sl[:, None]

    tp_any = tp_hits.any(axis=1)
    sl_any = sl_hits.any(axis=1)

    # Índice del primer toque (0 a horizon_bars - 1)
    sentinel = 9999
    tp_first = np.where(tp_any, np.argmax(tp_hits, axis=1), sentinel)
    sl_first = np.where(sl_any, np.argmax(sl_hits, axis=1), sentinel)

    # Vectorización de resolución de eventos
    # 1. TP primero
    hit_tp_first = (tp_first < sl_first) & (tp_first < sentinel)
    # 2. SL primero
    hit_sl_first = (sl_first < tp_first) & (sl_first < sentinel)
    # 3. Conflicto misma vela (ambos tocaron en el mismo paso)
    conflict_same_bar = (tp_first == sl_first) & (tp_first < sentinel)
    # 4. Barrera temporal expirada (ninguno tocó)
    time_expired = (tp_first == sentinel) & (sl_first == sentinel)

    # Aplicar asignaciones
    row_indices = np.arange(valid_len)

    # Caso 1: TP primero
    if np.any(hit_tp_first):
        idx = row_indices[hit_tp_first]
        step = tp_first[hit_tp_first]
        labels[idx] = 1.0
        exit_bars[idx] = step + 1
        exit_types[idx] = "TP"
        # En Take Profit limit order, el fill es el TP (o Open si hubo gap alcista favorable)
        open_at_exit = future_opens[hit_tp_first, step]
        exit_prices[idx] = np.maximum(tp[idx], open_at_exit)

    # Caso 2: SL primero
    if np.any(hit_sl_first):
        idx = row_indices[hit_sl_first]
        step = sl_first[hit_sl_first]
        labels[idx] = -1.0
        exit_bars[idx] = step + 1
        exit_types[idx] = "SL"
        # En Stop Loss, penalizar gaps bajistas: min(SL, Open)
        open_at_exit = future_opens[hit_sl_first, step]
        exit_prices[idx] = np.minimum(sl[idx], open_at_exit)

    # Caso 3: Conflicto en la misma vela
    if np.any(conflict_same_bar):
        idx = row_indices[conflict_same_bar]
        step = sl_first[conflict_same_bar]
        exit_bars[idx] = step + 1
        open_at_exit = future_opens[conflict_same_bar, step]

        if conservative_intrabar:
            # Regla conservadora institucional: asumir peor caso (SL ejecutado primero)
            labels[idx] = -1.0
            exit_types[idx] = "SL"
            exit_prices[idx] = np.minimum(sl[idx], open_at_exit)
        else:
            labels[idx] = 1.0
            exit_types[idx] = "TP"
            exit_prices[idx] = np.maximum(tp[idx], open_at_exit)

    # Caso 4: Expiración temporal
    if np.any(time_expired):
        idx = row_indices[time_expired]
        step = horizon_bars - 1
        labels[idx] = 0.0
        exit_bars[idx] = horizon_bars
        exit_types[idx] = "TIME"
        # Venta a precio de cierre al vencimiento del horizonte
        exit_prices[idx] = future_closes[time_expired, step]

    # TICK-08: Inyección de costes de transacción y retornos netos (LONG)
    p_in_net = p_entry * (1.0 + friction_pct)
    p_out_net = exit_prices[:valid_len] * (1.0 - friction_pct)

    gross_returns[:valid_len] = ((exit_prices[:valid_len] - p_entry) / p_entry) * 100.0
    net_returns[:valid_len] = ((p_out_net - p_in_net) / p_in_net) * 100.0

    # Clasificación binaria estricta (LONG):
    # 1 si tocó TP primero Y el retorno neto es estrictamente positivo (> 0)
    # 0 en cualquier otro caso (SL o TIME)
    is_win = (labels[:valid_len] == 1.0) & (net_returns[:valid_len] > 0.0)
    quality[:valid_len] = is_win.astype(np.float64)

    # === EVALUACIÓN BIDIRECCIONAL: ETIQUETADO DE SHORTS (VENTA EN CORTO) ===
    short_labels = np.full(n, np.nan, dtype=np.float64)
    short_quality = np.full(n, np.nan, dtype=np.float64)
    short_exit_bars = np.full(n, np.nan, dtype=np.float64)
    short_exit_types = np.array(["INCOMPLETE"] * n, dtype=object)
    short_exit_prices = np.full(n, np.nan, dtype=np.float64)
    short_net_returns = np.full(n, np.nan, dtype=np.float64)

    if adjust_barriers_for_friction:
        tp_short = p_entry - (k_tp * atr_entry) - (2.0 * friction_pct * p_entry)
        sl_short = p_entry + (k_sl * atr_entry) + (2.0 * friction_pct * p_entry)
    else:
        tp_short = p_entry - (k_tp * atr_entry)
        sl_short = p_entry + (k_sl * atr_entry)

    tp_short_hits = future_lows <= tp_short[:, None]
    sl_short_hits = future_highs >= sl_short[:, None]

    tp_short_any = tp_short_hits.any(axis=1)
    sl_short_any = sl_short_hits.any(axis=1)

    tp_short_first = np.where(tp_short_any, np.argmax(tp_short_hits, axis=1), sentinel)
    sl_short_first = np.where(sl_short_any, np.argmax(sl_short_hits, axis=1), sentinel)

    hit_tp_short_first = (tp_short_first < sl_short_first) & (tp_short_first < sentinel)
    hit_sl_short_first = (sl_short_first < tp_short_first) & (sl_short_first < sentinel)
    conflict_short = (tp_short_first == sl_short_first) & (tp_short_first < sentinel)
    time_expired_short = (tp_short_first == sentinel) & (sl_short_first == sentinel)

    if np.any(hit_tp_short_first):
        idx_s = row_indices[hit_tp_short_first]
        step_s = tp_short_first[hit_tp_short_first]
        short_labels[idx_s] = 1.0
        short_exit_bars[idx_s] = step_s + 1
        short_exit_types[idx_s] = "TP"
        open_at_exit_s = future_opens[hit_tp_short_first, step_s]
        short_exit_prices[idx_s] = np.minimum(tp_short[idx_s], open_at_exit_s)

    if np.any(hit_sl_short_first):
        idx_s = row_indices[hit_sl_short_first]
        step_s = sl_short_first[hit_sl_short_first]
        short_labels[idx_s] = -1.0
        short_exit_bars[idx_s] = step_s + 1
        short_exit_types[idx_s] = "SL"
        open_at_exit_s = future_opens[hit_sl_short_first, step_s]
        short_exit_prices[idx_s] = np.maximum(sl_short[idx_s], open_at_exit_s)

    if np.any(conflict_short):
        idx_s = row_indices[conflict_short]
        step_s = sl_short_first[conflict_short]
        short_exit_bars[idx_s] = step_s + 1
        open_at_exit_s = future_opens[conflict_short, step_s]
        short_labels[idx_s] = -1.0
        short_exit_types[idx_s] = "SL"
        short_exit_prices[idx_s] = np.maximum(sl_short[idx_s], open_at_exit_s)

    if np.any(time_expired_short):
        idx_s = row_indices[time_expired_short]
        step_s = horizon_bars - 1
        short_labels[idx_s] = 0.0
        short_exit_bars[idx_s] = horizon_bars
        short_exit_types[idx_s] = "TIME"
        short_exit_prices[idx_s] = future_closes[time_expired_short, step_s]

    p_in_short_net = p_entry * (1.0 - friction_pct)
    p_out_short_net = short_exit_prices[:valid_len] * (1.0 + friction_pct)
    short_net_returns[:valid_len] = ((p_in_short_net - p_out_short_net) / p_in_short_net) * 100.0

    is_short_win = (short_labels[:valid_len] == 1.0) & (short_net_returns[:valid_len] > 0.0)
    short_quality[:valid_len] = is_short_win.astype(np.float64)

    return {
        "target_triple_barrier": labels,
        "target_quality_trade": quality,
        "trade_exit_bar": exit_bars,
        "trade_exit_type": exit_types,
        "trade_exit_price": exit_prices,
        "trade_gross_return": gross_returns,
        "trade_net_return": net_returns,
        "trade_tp_price": tp_prices,
        "trade_sl_price": sl_prices,
        "target_short_quality_trade": short_quality,
        "target_short_triple_barrier": short_labels,
        "short_net_return": short_net_returns,
    }


def generate_triple_barrier_labels(
    df: pd.DataFrame,
    k_tp: float = 3.0,
    k_sl: float = 1.5,
    horizon_bars: int = 26,
    friction_pct: float = 0.0005,
    conservative_intrabar: bool = True,
    adjust_barriers_for_friction: bool = True,
    atr_window: int = 14,
) -> pd.DataFrame:
    """
    Genera el etiquetado Triple Barrier con fricción realista para un DataFrame multi-activo.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con velas (requiere 'open', 'high', 'low', 'close', opcionalmente 'ticker', 'atr14').
    k_tp : float
        Multiplicador de ATR para Take Profit (Upper Barrier), default 3.0.
    k_sl : float
        Multiplicador de ATR para Stop Loss (Lower Barrier), default 1.5.
    horizon_bars : int
        Número de velas horarias límite antes de cerrar la posición por tiempo (default 26 velas = 4 días).
    friction_pct : float
        Fricción porcentual por lado (default 0.0005 = 5 bps por lado = 10 bps round-trip).
    conservative_intrabar : bool
        Si True, cuando una misma vela toca tanto TP como SL, asume que SL ocurrió primero.
    adjust_barriers_for_friction : bool
        Si True, las barreras de TP y SL exigen cubrir la fricción de entrada y salida.
    atr_window : int
        Ventana para calcular ATR si no existe la columna 'atr14'.

    Retorna:
    --------
    pd.DataFrame con las columnas de target añadidas.
    """
    df = df.copy()

    # Validar columnas requeridas
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en DataFrame: {missing}")

    has_ticker = "ticker" in df.columns

    # Calcular o reutilizar ATR(14)
    if "atr14" not in df.columns:
        if "atr_pct" in df.columns:
            df["atr14"] = df["close"] * (df["atr_pct"] / 100.0)
        else:
            if has_ticker:
                atr_series = pd.Series(index=df.index, dtype=float)
                for _, grp in df.groupby("ticker"):
                    atr_series.loc[grp.index] = compute_atr_series(grp, window=atr_window)
                df["atr14"] = atr_series
            else:
                df["atr14"] = compute_atr_series(df, window=atr_window)

    # Columnas de salida
    out_cols = [
        "target_triple_barrier",
        "target_quality_trade",
        "trade_exit_bar",
        "trade_exit_type",
        "trade_exit_price",
        "trade_gross_return",
        "trade_net_return",
        "trade_tp_price",
        "trade_sl_price",
        "target_short_quality_trade",
        "target_short_triple_barrier",
        "short_net_return",
    ]

    for col in out_cols:
        df[col] = np.nan if col != "trade_exit_type" else "INCOMPLETE"

    if has_ticker:
        # Procesar cada ticker de forma independiente para evitar fuga entre activos
        for ticker, group_indices in df.groupby("ticker").groups.items():
            sub = df.loc[group_indices]
            res = _label_single_series(
                close=sub["close"].to_numpy(dtype=np.float64),
                high=sub["high"].to_numpy(dtype=np.float64),
                low=sub["low"].to_numpy(dtype=np.float64),
                open_=sub["open"].to_numpy(dtype=np.float64),
                atr=sub["atr14"].to_numpy(dtype=np.float64),
                k_tp=k_tp,
                k_sl=k_sl,
                horizon_bars=horizon_bars,
                friction_pct=friction_pct,
                conservative_intrabar=conservative_intrabar,
                adjust_barriers_for_friction=adjust_barriers_for_friction,
            )
            for col in out_cols:
                df.loc[group_indices, col] = res[col]
    else:
        res = _label_single_series(
            close=df["close"].to_numpy(dtype=np.float64),
            high=df["high"].to_numpy(dtype=np.float64),
            low=df["low"].to_numpy(dtype=np.float64),
            open_=df["open"].to_numpy(dtype=np.float64),
            atr=df["atr14"].to_numpy(dtype=np.float64),
            k_tp=k_tp,
            k_sl=k_sl,
            horizon_bars=horizon_bars,
            friction_pct=friction_pct,
            conservative_intrabar=conservative_intrabar,
            adjust_barriers_for_friction=adjust_barriers_for_friction,
        )
        for col in out_cols:
            df[col] = res[col]

    return df


def generate_target_variable(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Alias retrocompatible con src/utils/exploration_utils.py.
    """
    return generate_triple_barrier_labels(df, **kwargs)
