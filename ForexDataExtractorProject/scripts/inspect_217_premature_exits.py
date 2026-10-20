"""
Inspección forense de los 217 casos de Salida Prematura en las señales que llegaron al +45% ("Momento MeLi").
¿Por qué salen temprano? ¿En qué barra salen, qué regla exacta los saca, hasta dónde caen antes de volar al +17.29%,
y cómo distingue el Modelo 1B los que siguen subiendo de los que se devuelven?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    cache_1a_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_1a.parquet"
    hourly_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"

    df_1a = pd.read_parquet(cache_1a_path)
    df_raw = pd.read_parquet(hourly_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    ticker_dfs = {
        tk: gdf.sort_values("timestamp").reset_index(drop=True)
        for tk, gdf in df_raw[~df_raw["ticker"].isin(["SPY", "QQQ"])].groupby("ticker")
    }

    test_start_ts = pd.Timestamp("2025-11-17", tz="UTC")
    te = df_1a[df_1a["timestamp"] >= test_start_ts].copy().reset_index(drop=True)

    # Analizar todos los eventos de Test que llegaron a `meli_trigger_step < 900` (el 45% de recuperación)
    meli_events = te[te["meli_trigger_step"] < 900].copy().reset_index(drop=True)
    print(f"Total eventos en Test que tocaron el +45% de recuperación ('Momento MeLi'): {len(meli_events)}")

    records = []
    for _, row in meli_events.iterrows():
        tk = row["ticker"]
        i0 = int(row["bar_idx"])
        tdf = ticker_dfs[tk]
        fut = tdf.iloc[i0 + 1 : i0 + 106]
        entry_p = float(row["entry_p"])
        atr_d = float(row["atr_d"])
        shock_drop = float(row["shock_drop"])
        trig_step = int(row["meli_trigger_step"])

        highs = fut["high"].to_numpy()
        lows = fut["low"].to_numpy()
        closes = fut["close"].to_numpy()
        opens = fut["open"].to_numpy()

        # En el momento exacto en que termina la barra `trig_step` (cuando tocó >= 45% de recuperación):
        p_at_meli_close = closes[trig_step]
        ret_at_meli_close = (p_at_meli_close - entry_p) / entry_p * 100.0

        # ¿Qué pasa DESPUÉS de `trig_step` (desde `trig_step + 1` hasta el día 15)?
        rem_highs = highs[trig_step + 1 :]
        rem_lows = lows[trig_step + 1 :]
        rem_closes = closes[trig_step + 1 :]
        if len(rem_highs) == 0:
            continue

        # Pico máximo posterior (MFE después del Momento MeLi) y en qué barra ocurre
        best_post_idx = int(np.argmax(rem_highs))
        mfe_post_p = rem_highs[best_post_idx]
        mfe_post_pct = (mfe_post_p - entry_p) / entry_p * 100.0
        mfe_post_rec = (mfe_post_pct / 100.0) / shock_drop

        # ¿Cuánto retrocede (Drawdown desde `p_at_meli_close` o desde el pico intermedio) ANTES de alcanzar ese `mfe_post_p`?
        lows_before_mfe = rem_lows[: best_post_idx + 1]
        closes_before_mfe = rem_closes[: best_post_idx + 1]
        min_low_before_mfe = float(np.min(lows_before_mfe))
        min_close_before_mfe = float(np.min(closes_before_mfe))

        dip_from_meli_low_atr = (p_at_meli_close - min_low_before_mfe) / (entry_p * atr_d)
        dip_from_meli_close_atr = (p_at_meli_close - min_close_before_mfe) / (entry_p * atr_d)
        min_rec_before_mfe = ((min_low_before_mfe - entry_p) / entry_p) / shock_drop

        # ¿Y si simplemente mantenemos hasta el día 5, día 8, día 10 o día 15 con un Stop en Break-Even (-0.2 ATR)?
        ret_hold_5d = (closes[min(len(closes) - 1, 34)] - entry_p) / entry_p * 100.0
        ret_hold_8d = (closes[min(len(closes) - 1, 55)] - entry_p) / entry_p * 100.0
        ret_hold_10d = (closes[min(len(closes) - 1, 69)] - entry_p) / entry_p * 100.0
        ret_hold_15d = (closes[-1] - entry_p) / entry_p * 100.0

        records.append({
            "shock_type_code": int(row["shock_type_code"]),
            "shock_drop_pct": shock_drop * 100.0,
            "atr_d_pct": atr_d * 100.0,
            "trig_step": trig_step,
            "ret_at_meli_close": ret_at_meli_close,
            "mfe_post_pct": mfe_post_pct,
            "mfe_post_rec": mfe_post_rec,
            "bars_after_meli_to_mfe": best_post_idx + 1,
            "dip_from_meli_low_atr": dip_from_meli_low_atr,
            "dip_from_meli_close_atr": dip_from_meli_close_atr,
            "min_rec_before_mfe": min_rec_before_mfe,
            "ret_hold_5d": ret_hold_5d,
            "ret_hold_8d": ret_hold_8d,
            "ret_hold_10d": ret_hold_10d,
            "ret_hold_15d": ret_hold_15d,
        })

    rdf = pd.DataFrame(records)
    big_runners = rdf[rdf["mfe_post_rec"] >= 0.85]
    stalled = rdf[rdf["mfe_post_rec"] < 0.85]

    print("\n" + "=" * 115)
    print(f"ANATOMÍA DESPUÉS DEL 'MOMENTO MELI' (+45% Recuperación): ¿Por qué nos saca el Trailing antes del pico?")
    print("=" * 115)
    print(f"1. Casos que SIGUEN CORRIENDO hasta >= 85% de recuperación de la caída: {len(big_runners)} de {len(rdf)} ({len(big_runners)/len(rdf)*100:.1f}%)")
    print(f"   * Retorno al cerrar la vela del Momento MeLi: +{big_runners['ret_at_meli_close'].mean():.2f}%")
    print(f"   * Pico Máximo Posterior (MFE):                +{big_runners['mfe_post_pct'].mean():.2f}% (Recupera {big_runners['mfe_post_rec'].mean()*100:.0f}% de la caída)")
    print(f"   * ¿Cuántas barras (horas) tarda desde el Momento MeLi hasta el pico máximo?: {big_runners['bars_after_meli_to_mfe'].median():.0f} barras (~{big_runners['bars_after_meli_to_mfe'].median()/7:.1f} días)")
    print(f"   * ¡DATO CLAVE! ¿Cuánto retrocede (pullback intradía) ANTES de llegar a ese pico máximo?:")
    print(f"     - Retroceso intradía mediano (Low):  {big_runners['dip_from_meli_low_atr'].median():.2f} ATR (Percentil 75: {big_runners['dip_from_meli_low_atr'].quantile(0.75):.2f} ATR)")
    print(f"     - Nivel mínimo de recuperación que pisa antes de despegar (Mediana): {big_runners['min_rec_before_mfe'].median()*100:.1f}% de la caída (P25: {big_runners['min_rec_before_mfe'].quantile(0.25)*100:.1f}%)")

    print(f"\n2. Casos que SE ESTANCAN o REVIERTEN (< 85% de recuperación): {len(stalled)} de {len(rdf)} ({len(stalled)/len(rdf)*100:.1f}%)")
    print(f"   * Retorno al cerrar la vela del Momento MeLi: +{stalled['ret_at_meli_close'].mean():.2f}%")
    print(f"   * Pico Máximo Posterior (MFE):                +{stalled['mfe_post_pct'].mean():.2f}% (Recupera {stalled['mfe_post_rec'].mean()*100:.0f}% de la caída)")
    print(f"   * Retorno si no vendes y esperas al Día 15:   {stalled['ret_hold_15d'].mean():+.2f}% (Win Rate al día 15: {(stalled['ret_hold_15d']>0).mean()*100:.1f}%)")
    print("=" * 115)


if __name__ == "__main__":
    main()
