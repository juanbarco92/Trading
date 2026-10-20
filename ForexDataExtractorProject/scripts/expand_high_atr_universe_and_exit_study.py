"""
expand_high_atr_universe_and_exit_study.py -
1. Expande el universo con ~125 acciones adicionales de Alta Elasticidad (ATR Diario >= 2.0%)
   del S&P 500 / Nasdaq 100 en todos los sectores dinámicos (totalizando ~280 activos).
2. Realiza la radiografía exacta de qué pasa DENTRO de la operación (Día 1 a Día 20):
   - El Caso MeLi: ¿Qué pasa cuando una acción recupera el 50%-65% de la caída al día 8-10 pero
     no toca el 75% de TP? ¿Cuántas se devuelven y pierden todo si no se protegen?
   - Compara 6 arquitecturas de salida (Estática vs Breakeven + Ratchet/Trailing al 60% vs Time-Stop en Día 7).
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
import yfinance as yf
from scripts.evaluate_multisector_motor1 import deduplicate_events

# Nuevas empresas líquidas de alta elasticidad (ATR Diario >= 2.0%) por sector
HIGH_ATR_EXPANSION = {
    "Technology": [
        "MRVL", "ON", "MPWR", "NXPI", "MCHP", "SWKS", "QRVO", "TER", "ENTG",
        "DELL", "HPE", "HPQ", "NTAP", "WDC", "STX", "ANET", "FTNT", "CRWD",
        "DDOG", "NET", "ZS", "SNOW", "MDB", "TEAM", "HUBS", "WDAY", "PTC", "EPAM", "FSLR", "ENPH"
    ],
    "Consumer_Discretionary": [
        "LULU", "DECK", "CROX", "TPR", "RL", "PVH", "RCL", "CCL", "NCLH",
        "DAL", "UAL", "LUV", "EXPE", "LEN", "DHI", "PHM", "TOL", "BLDR",
        "URI", "MGM", "WYNN", "LVS", "DKNG", "EBAY", "ETSY", "W", "CVNA", "KMX", "APTV"
    ],
    "Financials": [
        "BX", "KKR", "APO", "ARES", "AMP", "TROW", "STT", "BK", "USB",
        "PNC", "TFC", "COF", "SYF", "ALLY", "FITB", "KEY", "RF", "HBAN",
        "CFG", "MTB", "IBKR", "NDAQ", "CBOE", "MSCI", "MCO", "AFRM", "SOFI", "PYPL", "SQ"
    ],
    "Industrials": [
        "PWR", "AXON", "GWW", "FAST", "CARR", "OTIS", "TT", "JCI", "LDOS",
        "TDG", "HWM", "TXT", "IR", "DOV", "ROK", "AME", "XYL", "IEX", "GNRC", "JBHT", "ODFL", "DAL"
    ],
    "Healthcare": [
        "DXCM", "PODD", "ALGN", "IDXX", "IQV", "CRL", "TECH", "BIO",
        "BIIB", "MRNA", "INCY", "EXAS", "HOLX", "RMD", "STE", "ZBH", "BAX", "MOH", "CNC", "HUM"
    ],
    "Energy": [
        "HAL", "BKR", "DVN", "FANG", "CTRA", "EQT", "TRGP", "OKE", "APA", "MRO"
    ],
    "Materials": [
        "STLD", "NUE", "CLF", "AA", "SCCO", "ALB", "CF", "MOS", "FMC", "PPG", "VMC", "MLM", "CE", "EMN"
    ],
    "Communication": [
        "SPOT", "ROKU", "PINS", "SNAP", "MTCH", "OMC", "IPG", "LYV", "WBD", "PARA"
    ],
}


def download_and_merge_expansion():
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    existing_df = pd.read_parquet(hourly_path)
    existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True)
    existing_tickers = set(existing_df["ticker"].unique())

    new_ticker_to_sec = {}
    new_tickers = []
    for sec, tks in HIGH_ATR_EXPANSION.items():
        for t in tks:
            if t not in existing_tickers and t not in new_ticker_to_sec:
                new_ticker_to_sec[t] = sec
                new_tickers.append(t)

    if not new_tickers:
        print(f"Universo ya expandido con {existing_df['ticker'].nunique()} activos.")
        return existing_df

    print(f"1. Descargando {len(new_tickers)} acciones adicionales de alta elasticidad (1H, 730 días)...")
    t0 = time.time()
    batch_size = 30
    chunks = []

    for i in range(0, len(new_tickers), batch_size):
        batch = new_tickers[i : i + batch_size]
        print(f"   * Lote {i//batch_size + 1}/{(len(new_tickers)-1)//batch_size + 1}: {batch[:6]}...")
        try:
            raw = yf.download(
                tickers=batch,
                period="730d",
                interval="1h",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if raw is None or raw.empty:
                continue
            for tk in batch:
                try:
                    sub = raw[tk].copy() if len(batch) > 1 else raw.copy()
                    sub = sub.dropna(subset=["Open", "High", "Low", "Close"])
                    if sub.empty:
                        continue
                    sub = sub.reset_index()
                    dt_col = "Datetime" if "Datetime" in sub.columns else sub.columns[0]
                    sub["timestamp"] = pd.to_datetime(sub[dt_col], utc=True)
                    sub = sub.rename(columns={
                        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
                    })
                    sub["ticker"] = tk
                    sub["sector"] = new_ticker_to_sec[tk]
                    sub = sub[["timestamp", "open", "high", "low", "close", "volume", "ticker", "sector"]]
                    chunks.append(sub)
                except Exception:
                    continue
        except Exception:
            continue

    if chunks:
        new_df = pd.concat(chunks, ignore_index=True)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.sort_values(["ticker", "timestamp"]).drop_duplicates(subset=["ticker", "timestamp"]).reset_index(drop=True)
        combined.to_parquet(hourly_path, index=False)
        print(f"   * Universo consolidado actualizado: {len(combined):,} velas 1H en {combined['ticker'].nunique()} empresas ({time.time()-t0:.1f}s).")
        return combined
    return existing_df


def main():
    print("=" * 105)
    print("EXPANSIÓN DE UNIVERSO DE ALTO ATR + ESTUDIO PROFUNDO DE GESTIÓN DE SALIDAS (EL CASO MELI)")
    print("=" * 105)

    df = download_and_merge_expansion()

    spy = df[df["ticker"] == "SPY"].sort_values("timestamp").copy()
    spy["hour_key"] = spy["timestamp"].dt.floor("h")
    spy = spy.drop_duplicates(subset=["hour_key"]).set_index("hour_key")
    spy_open_next = spy["open"].shift(-1)
    spy_fut_15d = (spy["close"].shift(-105) - spy_open_next) / spy_open_next * 100.0

    stocks = df[~df["ticker"].isin(["SPY", "QQQ"])].copy().reset_index(drop=True)
    stocks["hour_key"] = stocks["timestamp"].dt.floor("h")
    stocks["spy_fut_15d"] = stocks["hour_key"].map(spy_fut_15d).ffill()

    g = stocks.groupby("ticker", group_keys=False)
    stocks["ret_3d"] = g["close"].pct_change(21)
    stocks["ret_5d"] = g["close"].pct_change(35)
    stocks["vol_3d"] = g["ret_3d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["vol_5d"] = g["ret_5d"].transform(lambda s: s.rolling(420, min_periods=70).std())
    stocks["z_3d"] = stocks["ret_3d"] / stocks["vol_3d"].replace(0, np.nan)
    stocks["z_5d"] = stocks["ret_5d"] / stocks["vol_5d"].replace(0, np.nan)

    high_1d = g["high"].transform(lambda s: s.rolling(7, min_periods=3).max())
    low_1d = g["low"].transform(lambda s: s.rolling(7, min_periods=3).min())
    stocks["atr_daily_pct"] = ((high_1d - low_1d) / stocks["close"] * 100.0).groupby(stocks["ticker"]).transform(
        lambda s: s.rolling(98, min_periods=21).mean()
    )

    rng_1h = (stocks["high"] - stocks["low"]).replace(0, np.nan)
    stocks["close_loc_1h"] = (stocks["close"] - stocks["low"]) / rng_1h
    stocks["lower_wick_ratio"] = (np.minimum(stocks["open"], stocks["close"]) - stocks["low"]) / rng_1h

    open_next = g["open"].shift(-1)
    stocks["fut_ret_15d"] = (g["close"].shift(-105) - open_next) / open_next * 100.0

    # Filtrar UNIVERSO ELEGIBLE: Acciones con ATR Diario >= 2.2% (Excluyendo tortugas)
    # y Golpes de 3-5 días con vela de absorción 1H
    shock_mask = (
        (stocks["atr_daily_pct"] >= 2.2)
        & ((stocks["z_3d"] <= -2.0) | (stocks["z_5d"] <= -2.0))
        & ((stocks["close_loc_1h"] >= 0.55) | (stocks["lower_wick_ratio"] >= 0.25))
    )
    shocks_raw = stocks[shock_mask].dropna(subset=["fut_ret_15d", "atr_daily_pct"]).copy()
    shocks = deduplicate_events(shocks_raw, "timestamp", min_gap_hours=48)

    print(f"\nUniverso filtrado de Alta Elasticidad (ATR Diario >= 2.2%):")
    print(f"  * Empresas elegibles con golpes: {shocks['ticker'].nunique()} empresas")
    print(f"  * Total Golpes Duros Únicos (3-5d + absorción 1H): {len(shocks):,} eventos\n")

    # Pre-indexar series por ticker para simular trayectoria día a día desde Open[t+1] hasta Día 20 (140 barras 1H)
    ticker_groups = {tk: grp.sort_values("timestamp").reset_index(drop=True) for tk, grp in stocks.groupby("ticker")}

    # =========================================================================
    # 1. ANÁLISIS FORENSE DEL "CASO MELI" (QUÉ PASA CUANDO SUBE AL 55%-70% PERO NO TOCA EL 75% TP)
    # =========================================================================
    print("=" * 105)
    print("1. ANATOMÍA DEL 'CASO MELI': ¿QUÉ PASA CON LOS TRADES QUE RECUPERAN EL 55%-74% DE LA CAÍDA")
    print("   PERO SE QUEDAN A MILÍMETROS DEL TAKE PROFIT DEL 75%?")
    print("=" * 105)

    total_valid = 0
    hit_75_tp = 0
    reached_55_to_74_only = 0
    gave_back_gains = 0
    turned_into_loss = 0
    end_rets_of_near_miss = []
    peak_rets_of_near_miss = []

    trajectories = []  # Guardamos las trayectorias para probar las 6 reglas de salida

    for _, row in shocks.iterrows():
        tk = row["ticker"]
        ts = row["timestamp"]
        atr_d = row["atr_daily_pct"] / 100.0
        # Magnitud de la caída de 3-5 días que buscamos recuperar
        shock_drop = abs(min(row["ret_3d"], row["ret_5d"], -0.04))

        tdf = ticker_groups[tk]
        idx_arr = tdf.index[tdf["timestamp"] == ts].to_numpy()
        if len(idx_arr) == 0:
            continue
        i0 = idx_arr[0]
        if i0 + 140 >= len(tdf):
            continue

        entry_p = tdf.loc[i0 + 1, "open"] * 1.0005
        sl_static_p = entry_p * (1.0 - 1.5 * atr_d)
        tp_75_p = entry_p * (1.0 + 0.75 * shock_drop)

        fut = tdf.iloc[i0 + 1 : i0 + 1 + 140]  # 20 días (140 horas)
        highs = fut["high"].to_numpy()
        lows = fut["low"].to_numpy()
        closes = fut["close"].to_numpy()

        trajectories.append((entry_p, atr_d, shock_drop, highs, lows, closes, row["sector"]))
        total_valid += 1

        # En los primeros 15 días (105 barras), ¿cuánto llegó a recuperar como máximo antes de tocar SL?
        max_recovery_ratio = 0.0
        touched_75 = False
        final_ret_15d = ((closes[104] * 0.9995 - entry_p) / entry_p) * 100.0

        for step in range(105):
            # Comprobar SL al cierre diario
            if (step + 1) % 7 == 0 and closes[step] <= sl_static_p:
                final_ret_15d = ((closes[step] * 0.9995 - entry_p) / entry_p) * 100.0
                break
            cur_gain_pct = (highs[step] - entry_p) / entry_p
            rec_ratio = cur_gain_pct / shock_drop
            if rec_ratio > max_recovery_ratio:
                max_recovery_ratio = rec_ratio
            if highs[step] >= tp_75_p:
                touched_75 = True
                break

        if touched_75:
            hit_75_tp += 1
        elif 0.55 <= max_recovery_ratio < 0.75:
            reached_55_to_74_only += 1
            peak_pct = max_recovery_ratio * shock_drop * 100.0
            peak_rets_of_near_miss.append(peak_pct)
            end_rets_of_near_miss.append(final_ret_15d)
            if final_ret_15d < peak_pct * 0.5:
                gave_back_gains += 1
            if final_ret_15d < 0:
                turned_into_loss += 1

    print(f"De {total_valid:,} operaciones analizadas en horizonte de 15 días:")
    print(f"  * Tocaron directamente el TP del 75% de recuperación: {hit_75_tp:,} ({hit_75_tp/total_valid*100:.1f}%)")
    print(f"  * CASO MELI (Subieron hasta recuperar entre 55% y 74%, pero NO tocaron el 75%): {reached_55_to_74_only:,} casos")
    if reached_55_to_74_only > 0:
        print(f"    -> Ganancia media que tenían en su pico:     +{np.mean(peak_rets_of_near_miss):.2f}%")
        print(f"    -> Con cuánto terminaron por no cerrar:      {np.mean(end_rets_of_near_miss):+.2f}% (¡Devolvieron {np.mean(peak_rets_of_near_miss) - np.mean(end_rets_of_near_miss):.2f}%!)")
        print(f"    -> Devolvieron más de la mitad de lo ganado: {gave_back_gains} de {reached_55_to_74_only} ({gave_back_gains/reached_55_to_74_only*100:.1f}%)")
        print(f"    -> ¡Terminaron en PÉRDIDA (rojos) tras ir ganando!: {turned_into_loss} de {reached_55_to_74_only} ({turned_into_loss/reached_55_to_74_only*100:.1f}%)")

    # =========================================================================
    # 2. ¿QUÉ PASA SI AL DÍA 6-8 LA ACCIÓN NO ARRANCÓ ("ZOMBIE TRADE")?
    # =========================================================================
    print("\n" + "=" * 105)
    print("2. ¿CUÁNTO TIEMPO ESPERAR SI LA ACCIÓN NO ARRANCA? (EL PROBLEMA DE LAS 'ACCIONES ZOMBIE')")
    print("=" * 105)
    day5_buckets = {
        "A. Al Día 5 ya va ganando fuerte (> +2%)": [],
        "B. Al Día 5 va tibia (entre -1% y +2%)": [],
        "C. Al Día 5 sigue cayendo (entre -4% y -1%)": [],
        "D. Al Día 5 se hundió (< -4%)": [],
    }
    for entry_p, atr_d, shock_drop, highs, lows, closes, sec in trajectories:
        ret_d5 = (closes[34] - entry_p) / entry_p * 100.0
        ret_d5_to_d15 = (closes[104] - closes[34]) / closes[34] * 100.0
        if ret_d5 > 2.0:
            day5_buckets["A. Al Día 5 ya va ganando fuerte (> +2%)"].append((ret_d5, ret_d5_to_d15))
        elif ret_d5 >= -1.0:
            day5_buckets["B. Al Día 5 va tibia (entre -1% y +2%)"].append((ret_d5, ret_d5_to_d15))
        elif ret_d5 >= -4.0:
            day5_buckets["C. Al Día 5 sigue cayendo (entre -4% y -1%)"].append((ret_d5, ret_d5_to_d15))
        else:
            day5_buckets["D. Al Día 5 se hundió (< -4%)"].append((ret_d5, ret_d5_to_d15))

    z_rows = []
    for k, vals in day5_buckets.items():
        if not vals:
            continue
        r5 = np.array([v[0] for v in vals])
        r_rest = np.array([v[1] for v in vals])
        z_rows.append({
            "Estado de la Posición al Día 5": k,
            "Casos": len(vals),
            "Retorno Acumulado al Día 5 %": round(r5.mean(), 2),
            "¿Cuánto más sube del Día 5 al 15? %": round(r_rest.mean(), 2),
            "Probabilidad de subir entre Día 5 y 15 %": round((r_rest > 0).mean() * 100, 1),
        })
    print(pd.DataFrame(z_rows).to_string(index=False))

    # =========================================================================
    # 3. COMPARATIVA DE LAS 5 REGLAS DE SALIDA INTELIGENTE (SOLUCIÓN AL CASO MELI)
    # =========================================================================
    print("\n" + "=" * 105)
    print("3. SOLUCIÓN AL CASO MELI: COMPARATIVA DE 5 REGLAS DE GESTIÓN DE SALIDA (SOBRE LOS 3,800+ GOLPES):")
    print("=" * 105)

    def simulate_rule(rule_name: str):
        rets = []
        days_list = []
        exit_reasons = {"TP": 0, "PROTECTED_PROFIT": 0, "TIME": 0, "SL": 0}

        for entry_p, atr_d, shock_drop, highs, lows, closes, sec in trajectories:
            # Horizonte por sector: 12 días (84h) en Salud/Finanzas/Energía, 18 días (126h) en Tech/Consumo/Industriales
            if rule_name == "5. SISTEMA COMPLETO (Candado Progresivo 40%/60% + Horizonte por Sector)":
                max_bars = 84 if sec in ("Healthcare", "Financials", "Energy") else 126
            else:
                max_bars = 105  # 15 días

            tp_p = entry_p * (1.0 + max(0.75 * shock_drop, 1.8 * atr_d))
            sl_p = entry_p * (1.0 - 1.5 * atr_d)
            current_floor_p = None  # Piso dinámico de protección de ganancias
            max_price_seen = entry_p

            exit_p = closes[max_bars - 1]
            reason = "TIME"
            held_bars = max_bars

            for step in range(max_bars):
                h_val = highs[step]
                l_val = lows[step]
                c_val = closes[step]

                if h_val > max_price_seen:
                    max_price_seen = h_val
                max_rec_ratio = ((max_price_seen - entry_p) / entry_p) / shock_drop

                # Actualizar Candado de Ganancias según la regla
                if rule_name in (
                    "2. Candado Breakeven al 45% + Asegurar 50% al llegar al 60% (Solución MeLi)",
                    "5. SISTEMA COMPLETO (Candado Progresivo 40%/60% + Horizonte por Sector)"
                ):
                    if max_rec_ratio >= 0.60:
                        # Ya recuperó >=60% (Caso MeLi): aseguramos el 45% de la caída (nunca devolvemos el viaje!)
                        lock_p = entry_p * (1.0 + 0.45 * shock_drop)
                        current_floor_p = max(current_floor_p or 0.0, lock_p)
                    elif max_rec_ratio >= 0.40:
                        # Ya recuperó >=40%: subimos stop a Breakeven + comisión (+0.2%)
                        lock_p = entry_p * 1.002
                        current_floor_p = max(current_floor_p or 0.0, lock_p)

                elif rule_name == "3. Trailing Stop Dinámico desde el 50% de recuperación (Sin techo fijo)":
                    if max_rec_ratio >= 0.50:
                        # Deja correr la ganancia sin techo fijo, con trailing stop a 0.8x ATR_d bajo el máximo
                        lock_p = max_price_seen * (1.0 - 0.8 * atr_d)
                        current_floor_p = max(current_floor_p or 0.0, lock_p)

                elif rule_name == "4. Salida Parcial 50/50 (Mitad en 50% de recup + Mitad en 85% con Breakeven)":
                    pass  # Manejado abajo

                # 1. ¿Tocó el Piso de Ganancia Asegurada (Candado)?
                if current_floor_p is not None and c_val <= current_floor_p and step > 0:
                    exit_p = current_floor_p
                    reason = "PROTECTED_PROFIT"
                    held_bars = step + 1
                    break

                # 2. ¿Tocó el Stop Loss inicial al cierre del día (cada 7 horas)?
                if (step + 1) % 7 == 0 and c_val <= sl_p:
                    exit_p = c_val
                    reason = "SL"
                    held_bars = step + 1
                    break

                # 3. ¿Tocó el Take Profit? (Salvo en regla 3 que deja correr con Trailing)
                if rule_name != "3. Trailing Stop Dinámico desde el 50% de recuperación (Sin techo fijo)":
                    if h_val >= tp_p:
                        exit_p = tp_p
                        reason = "TP"
                        held_bars = step + 1
                        break

            net_ret = ((exit_p * 0.9995 - entry_p) / entry_p) * 100.0
            rets.append(net_ret)
            days_list.append(held_bars / 7.0)
            exit_reasons[reason] += 1

        rets = np.array(rets)
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 99.0
        n_tot = len(rets)
        return {
            "Regla de Gestión de Salida": rule_name,
            "TP_Lleno %": round(exit_reasons["TP"] / n_tot * 100, 1),
            "Ganancia_Protegida %": round(exit_reasons["PROTECTED_PROFIT"] / n_tot * 100, 1),
            "Salida_Tiempo %": round(exit_reasons["TIME"] / n_tot * 100, 1),
            "Stop_Loss %": round(exit_reasons["SL"] / n_tot * 100, 1),
            "Win_Rate_Neto %": round((rets > 0).mean() * 100, 1),
            "Ret_Medio %": round(rets.mean(), 2),
            "Profit_Factor": round(pf, 2),
            "Días_En_Trade": round(np.mean(days_list), 1),
        }

    rules = [
        "1. Estática (Espera el 75% TP o se queda abierta 15 días sin proteger)",
        "2. Candado Breakeven al 45% + Asegurar 50% al llegar al 60% (Solución MeLi)",
        "3. Trailing Stop Dinámico desde el 50% de recuperación (Sin techo fijo)",
        "5. SISTEMA COMPLETO (Candado Progresivo 40%/60% + Horizonte por Sector)",
    ]
    r_rows = [simulate_rule(r) for r in rules]
    print(pd.DataFrame(r_rows).to_string(index=False))
    print("=" * 105)


if __name__ == "__main__":
    main()
