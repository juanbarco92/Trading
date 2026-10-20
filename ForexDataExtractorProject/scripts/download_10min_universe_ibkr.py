"""
Descargador en Segundo Plano de Barras de 10 Minutos en Interactive Brokers:
- Descarga barras de 10 minutos (2020 a 2026) para los activos de alta volatilidad.
- Guarda en data_ibkr/{ticker}/10mins/{YYYYMM}.csv.
- Salta automáticamente los bloques que ya existen.
- Pacing seguro de 0.8s entre llamadas para estabilidad total de TWS.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_ibkr"

# Tickers prioritarios para Motor 2 (Shorts de alta volatilidad y liquidez)
PRIORITY_TICKERS = [
    "MELI", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL", "MSFT", "AAPL",
    "COIN", "MSTR", "CVNA", "PLTR", "SHOP", "ROKU", "HOOD", "TTD", "UBER",
    "CAT", "GE", "VRT", "CLF", "FCX", "MP", "XOM", "CVX", "PBF", "LLY", "UNH",
    "SPY", "QQQ"
]

class IBKR10MinDownloader(EClient, EWrapper):
    def __init__(self, pacing_interval: float = 0.8):
        EClient.__init__(self, self)
        self.pacing_interval = pacing_interval
        self.connected_event = False
        self.current_bars = []
        self.request_finished = False
        self.has_error = False
        self.last_error = ""

    def connectAck(self):
        self.connected_event = True

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode in [2104, 2106, 2158]: # Mensajes informativos
            return
        if errorCode == 162: # Pacing violation
            print(f"  [PACING IBKR] Violación de ritmo ({errorString}). Pausando 15s...")
            time.sleep(15.0)
            return
        self.has_error = True
        self.last_error = f"Error {errorCode}: {errorString}"

    def historicalData(self, reqId: int, bar: BarData):
        self.current_bars.append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "barCount": getattr(bar, "barCount", 0),
            "wap": getattr(bar, "wap", getattr(bar, "average", 0.0))
        })

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        self.request_finished = True

def create_stock_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c

def get_monthly_ranges(start_year=2020, end_year=2026):
    months = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            if y == 2026 and m > 10:
                break
            if m == 12:
                next_y = y + 1
                next_m = 1
            else:
                next_y = y
                next_m = m + 1
            end_dt = datetime(next_y, next_m, 1, 0, 0, 0)
            end_str = end_dt.strftime("%Y%m%d 00:00:00 US/Eastern")
            month_key = f"{y}{m:02d}"
            months.append((month_key, end_str))
    return months

def download_symbol_10min(app: IBKR10MinDownloader, symbol: str):
    out_dir = DATA_DIR / symbol / "10mins"
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = create_stock_contract(symbol)
    monthly_blocks = get_monthly_ranges(2020, 2026)

    print(f"\n[10-MIN] Iniciando descarga para {symbol} ({len(monthly_blocks)} bloques mensuales)...")
    success_count = 0

    for i, (month_key, end_str) in enumerate(monthly_blocks, 1):
        csv_file = out_dir / f"{month_key}.csv"
        if csv_file.exists() and csv_file.stat().st_size > 100:
            success_count += 1
            continue

        app.current_bars = []
        app.request_finished = False
        app.has_error = False

        app.reqHistoricalData(
            reqId=i,
            contract=contract,
            endDateTime=end_str,
            durationStr="1 M",
            barSizeSetting="10 mins",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

        t_wait = 0
        while not app.request_finished and not app.has_error and t_wait < 10.0:
            time.sleep(0.1)
            t_wait += 0.1

        if app.current_bars:
            df = pd.DataFrame(app.current_bars)
            df.to_csv(csv_file, index=False)
            success_count += 1
            print(f"  [{symbol}] {month_key}: {len(df)} barras de 10 min guardadas.")
        else:
            # Guardar archivo vacío si no cotizaba aún
            pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "barCount", "wap"]).to_csv(csv_file, index=False)

        time.sleep(app.pacing_interval)

    print(f"[OK] {symbol}: {success_count}/{len(monthly_blocks)} meses de 10 minutos procesados.")

def main():
    print("=" * 80)
    print("DESCARGA DE BARRAS DE 10 MINUTOS EN SEGUNDO PLANO (IBKR 2020 - 2026)")
    print(f"Activos prioritarios: {len(PRIORITY_TICKERS)}")
    print("=" * 80)

    app = IBKR10MinDownloader(pacing_interval=0.8)
    app.connect("127.0.0.1", 7496, clientId=110)

    import threading
    t = threading.Thread(target=app.run, daemon=True)
    t.start()

    t0 = time.time()
    while not app.connected_event and time.time() - t0 < 5.0:
        time.sleep(0.1)

    if not app.connected_event:
        print("[ERROR] No se pudo conectar a TWS en 127.0.0.1:7496")
        return

    print("[OK] Conectado a TWS Live (7496). Comenzando descarga en segundo plano...\n")

    for sym in PRIORITY_TICKERS:
        try:
            download_symbol_10min(app, sym)
        except Exception as e:
            print(f"  [ALERTA] Error con {sym}: {e}")
        time.sleep(1.0)

    print("\n[FIN] Descarga de barras de 10 minutos completada.")
    app.disconnect()

if __name__ == "__main__":
    main()
