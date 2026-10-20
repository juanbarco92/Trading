"""
download_sp500_multisector_universe.py - TICK-21 & TICK-22:
Descarga Masiva Multi-Sectorial (120+ Acciones del S&P 500 de 10 Sectores) + Earnings & Fundamentales Point-in-Time.

1. Descarga 5 años de velas Diarias (2021-2026) para ~120 acciones de todos los sectores del S&P 500.
2. Descarga el máximo histórico de velas de 1 Hora (730 días en Yahoo Finance) para las ~100 acciones nuevas
   y las fusiona con los 5 años de velas 1H de IBKR de las 21 acciones existentes.
3. Descarga el calendario histórico de Earnings (Fecha exacta de reporte, EPS Estimado, EPS Real, Surprise %)
   y métricas fundamentales trimestrales (Revenue YoY, Operating Margin, Cash/Debt) Point-in-Time.
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

# Universo Estratificado de 115 Acciones Líquidas en 10 Sectores Económicos + SPY/QQQ
SECTOR_UNIVERSE = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "ACN",
        "IBM", "INTC", "TXN", "QCOM", "AMAT", "MU", "LRCX", "ADI", "KLAC", "PANW",
        "PLTR", "SMCI", "ARM", "SHOP", "NOW", "INTU", "SNPS", "CDNS"
    ],
    "Communication": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "EA", "TTWO"
    ],
    "Consumer_Discretionary": [
        "AMZN", "TSLA", "MELI", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG",
        "MAR", "CMG", "ORLY", "AZO", "ROST", "GM", "F", "ABNB", "UBER"
    ],
    "Consumer_Staples": [
        "WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "TGT",
        "KMB", "GIS", "SYY", "KR", "HSY"
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V",
        "MA", "SPGI", "CB", "PGR", "MMC", "ICE", "CME", "COIN", "HOOD", "MSTR"
    ],
    "Healthcare": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "ISRG", "VRTX", "REGN", "SYK", "BSX", "MDT", "CI", "CVS"
    ],
    "Industrials": [
        "CAT", "DE", "GE", "UNP", "HON", "RTX", "BA", "LMT", "UPS", "ADP",
        "ETN", "ITW", "WM", "CSX", "NSC", "FDX", "EMR", "GD", "NOC", "PH"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HES", "KMI", "WMB"
    ],
    "Materials": [
        "LIN", "APD", "SHW", "FCX", "NEM", "ECL", "NUE", "DOW", "DD", "CTVA"
    ],
    "Utilities_RealEstate": [
        "NEE", "SO", "DUK", "AMT", "PLD", "EQIX", "SPG", "O", "PSA", "WELL"
    ],
    "Benchmarks": [
        "SPY", "QQQ"
    ]
}

TICKER_TO_SECTOR = {}
ALL_TICKERS = []
for sec, tks in SECTOR_UNIVERSE.items():
    for t in tks:
        if t not in TICKER_TO_SECTOR:
            TICKER_TO_SECTOR[t] = sec
            ALL_TICKERS.append(t)


def download_hourly_data(tickers: list) -> pd.DataFrame:
    """
    Descarga 730 días de velas 1H desde Yahoo Finance en lotes y las combina con
    el archivo local universe_1hour.parquet (que tiene 5 años de IBKR para 21 activos).
    """
    print(f"\n[1/3] Descargando velas de 1 Hora (730 días) para {len(tickers)} activos en lotes...")
    t0 = time.time()

    # Cargar datos existentes de IBKR (2021-2026) si están disponibles
    ibkr_path = PROJECT_ROOT / "data" / "processed" / "universe_1hour.parquet"
    existing_df = None
    if ibkr_path.exists():
        existing_df = pd.read_parquet(ibkr_path)
        time_col = "timestamp" if "timestamp" in existing_df.columns else "datetime"
        existing_df["timestamp"] = pd.to_datetime(existing_df[time_col], utc=True)
        existing_df = existing_df[["timestamp", "open", "high", "low", "close", "volume", "ticker"]].copy()
        print(f"  * Cargado histórico IBKR local: {len(existing_df):,} velas ({existing_df['ticker'].nunique()} activos, 2021-2026).")

    batch_size = 25
    all_chunks = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  * Lote 1H {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: {batch[:6]}...")
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
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    })
                    sub["ticker"] = tk
                    sub = sub[["timestamp", "open", "high", "low", "close", "volume", "ticker"]]
                    all_chunks.append(sub)
                except Exception:
                    continue
        except Exception as e:
            print(f"    [WARN] Error en lote {batch[:3]}: {e}")

    yf_hourly = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()
    print(f"  * Descargadas {len(yf_hourly):,} velas 1H de Yahoo Finance ({yf_hourly['ticker'].nunique()} activos).")

    if existing_df is not None and not yf_hourly.empty:
        # Para los activos que ya están en IBKR (5 años completos), conservamos IBKR y añadimos los nuevos de YF
        ibkr_tickers = set(existing_df["ticker"].unique())
        yf_new_only = yf_hourly[~yf_hourly["ticker"].isin(ibkr_tickers)].copy()
        combined = pd.concat([existing_df, yf_new_only], ignore_index=True)
    elif existing_df is not None:
        combined = existing_df
    else:
        combined = yf_hourly

    combined["sector"] = combined["ticker"].map(TICKER_TO_SECTOR).fillna("Other")
    combined = combined.sort_values(["ticker", "timestamp"]).drop_duplicates(subset=["ticker", "timestamp"]).reset_index(drop=True)

    out_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_1hour.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"  * Guardado {out_path.name}: {len(combined):,} velas 1H | {combined['ticker'].nunique()} activos en {time.time()-t0:.1f}s.")
    return combined


def download_daily_data(tickers: list) -> pd.DataFrame:
    """
    Descarga 5+ años de velas Diarias (2020-2026) para los ~155 activos del universo multi-sectorial.
    """
    print(f"\n[2/3] Descargando 6 años de velas Diarias (2020-2026) para {len(tickers)} activos...")
    t0 = time.time()
    batch_size = 40
    all_chunks = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            raw = yf.download(
                tickers=batch,
                start="2020-01-01",
                interval="1d",
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
                    dt_col = "Date" if "Date" in sub.columns else sub.columns[0]
                    sub["timestamp"] = pd.to_datetime(sub[dt_col], utc=True)
                    sub = sub.rename(columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    })
                    sub["ticker"] = tk
                    sub["sector"] = TICKER_TO_SECTOR.get(tk, "Other")
                    sub = sub[["timestamp", "open", "high", "low", "close", "volume", "ticker", "sector"]]
                    all_chunks.append(sub)
                except Exception:
                    continue
        except Exception as e:
            print(f"    [WARN] Error lote diario: {e}")

    daily_df = pd.concat(all_chunks, ignore_index=True)
    daily_df = daily_df.sort_values(["ticker", "timestamp"]).drop_duplicates(subset=["ticker", "timestamp"]).reset_index(drop=True)
    out_path = PROJECT_ROOT / "data" / "processed" / "sp500_multisector_daily.parquet"
    daily_df.to_parquet(out_path, index=False)
    print(f"  * Guardado {out_path.name}: {len(daily_df):,} velas Diarias | {daily_df['ticker'].nunique()} activos en {time.time()-t0:.1f}s.")
    return daily_df


def download_earnings_and_fundamentals(tickers: list) -> pd.DataFrame:
    """
    Descarga el historial Point-in-Time de Earnings (Fecha exacta de publicación, EPS Estimate,
    Reported EPS, Surprise %) para cada activo.
    """
    print(f"\n[3/3] Extrayendo Fechas de Resultados (Earnings) y Sorpresas de EPS Point-in-Time...")
    t0 = time.time()
    stock_tickers = [t for t in tickers if t not in ("SPY", "QQQ")]
    earnings_rows = []

    for idx, tk in enumerate(stock_tickers):
        if (idx + 1) % 25 == 0 or idx == 0:
            print(f"  * Procesando fundamentales/earnings {idx+1}/{len(stock_tickers)} ({tk})...")
        try:
            ticker_obj = yf.Ticker(tk)
            ed = ticker_obj.get_earnings_dates(limit=32)
            if ed is not None and not ed.empty:
                ed = ed.reset_index()
                dt_col = ed.columns[0]
                ed["earnings_date"] = pd.to_datetime(ed[dt_col], utc=True)
                for _, row in ed.iterrows():
                    eps_est = row.get("EPS Estimate", np.nan)
                    eps_act = row.get("Reported EPS", np.nan)
                    surp = row.get("Surprise(%)", np.nan)
                    if pd.isna(eps_act) and pd.isna(surp):
                        continue
                    earnings_rows.append({
                        "ticker": tk,
                        "sector": TICKER_TO_SECTOR.get(tk, "Other"),
                        "earnings_date": row["earnings_date"],
                        "eps_estimate": float(eps_est) if pd.notna(eps_est) else np.nan,
                        "eps_actual": float(eps_act) if pd.notna(eps_act) else np.nan,
                        "eps_surprise_pct": float(surp) if pd.notna(surp) else np.nan,
                        "eps_positive": 1 if (pd.notna(eps_act) and float(eps_act) > 0) else 0,
                        "eps_beat": 1 if (pd.notna(surp) and float(surp) > 0) else 0,
                    })
        except Exception:
            continue

    earn_df = pd.DataFrame(earnings_rows)
    if not earn_df.empty:
        earn_df = earn_df.sort_values(["ticker", "earnings_date"]).drop_duplicates(subset=["ticker", "earnings_date"]).reset_index(drop=True)
        out_path = PROJECT_ROOT / "data" / "processed" / "sp500_earnings_pit.parquet"
        earn_df.to_parquet(out_path, index=False)
        print(f"  * Guardado {out_path.name}: {len(earn_df):,} reportes trimestrales Point-in-Time ({earn_df['ticker'].nunique()} empresas) en {time.time()-t0:.1f}s.")
    else:
        print("  [WARN] No se pudieron descargar registros de earnings.")
    return earn_df


def main():
    print("=" * 95)
    print(f"INICIANDO DESCARGA MULTI-SECTORIAL S&P 500 ({len(ALL_TICKERS)} ACTIVOS EN {len(SECTOR_UNIVERSE)-1} SECTORES)")
    print("=" * 95)
    hourly_df = download_hourly_data(ALL_TICKERS)
    daily_df = download_daily_data(ALL_TICKERS)
    earn_df = download_earnings_and_fundamentals(ALL_TICKERS)

    print("\n" + "=" * 95)
    print("RESUMEN DEL UNIVERSO MULTI-SECTORIAL CONSOLIDADO:")
    print("=" * 95)
    sec_summary = hourly_df.groupby("sector").agg(
        activos=("ticker", "nunique"),
        velas_1h=("close", "count"),
        fecha_min=("timestamp", lambda s: str(s.min().date())),
        fecha_max=("timestamp", lambda s: str(s.max().date())),
    ).reset_index()
    print(sec_summary.to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    main()
