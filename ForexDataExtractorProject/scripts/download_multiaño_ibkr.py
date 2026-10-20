"""
Script de Descarga Multiaño Inteligente desde Interactive Brokers:
- Canasta representativa de 30 activos líderes y alto beta multisectorial + SPY + QQQ + VIX.
- Rango histórico: Enero 2020 a Octubre 2026 (7 años completos que cubren COVID Crash, Bull 2021, Bear 2022 y Rally 2023-2026).
- Respeta pacing de IBKR con auto-guardado en data_ibkr/{ticker}/1hour/{YYYYMM}.csv.
- Salta automáticamente los bloques mensuales que ya estén descargados.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data_ibkr"

# Universo balanceado multisectorial para la Prueba de Fuego Multiaño (2020 - 2026)
BENCHMARKS = ["SPY", "QQQ"]
INDICES = [("VIX", "CBOE")]  # Índice VIX de CBOE

# 30 activos seleccionados por liquidez y alto ATR (3 por cada sector clave)
UNIVERSE_STOCKS = [
    # Technology
    "NVDA", "AAPL", "MSFT", "AMD",
    # Consumer Discretionary & Growth
    "MELI", "AMZN", "TSLA", "CVNA",
    # Financials & FinTech
    "JPM", "COIN", "MSTR", "HOOD",
    # Communication
    "META", "GOOGL", "ROKU", "TTD",
    # Industrials
    "CAT", "GE", "VRT",
    # Energy
    "XOM", "CVX", "PBF",
    # Healthcare
    "LLY", "UNH", "VKTX",
    # Materials
    "CLF", "FCX", "MP",
    # Consumer Staples & Real Estate (Elasticos)
    "TGT", "SPG"
]

class IBKRBatchDownloader(EClient, EWrapper):
    def __init__(self, pacing_interval: float = 0.6):
        EClient.__init__(self, self)
        self.pacing_interval = pacing_interval
        self.connected_event = False
        self.current_bars = []
        self.request_finished = False
        self.has_error = False
        self.last_error = ""

    def connectAck(self):
        pass

    def nextValidId(self, orderId: int):
        self.connected_event = True

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        # Ignorar mensajes de conectividad normales
        if errorCode in [2104, 2106, 2158]:
            return
        # Pacing violation (código 162 con texto pacing)
        if errorCode == 162 and "pacing violation" in errorString.lower():
            print(f"[PACING ALERT] Pausando 15s por límite de velocidad de IBKR...")
            time.sleep(15)
        self.last_error = f"{errorCode}: {errorString}"
        if errorCode in [200, 321, 162]:
            self.has_error = True
            self.request_finished = True

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


def make_stock_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c


def make_index_contract(symbol: str, exchange: str = "CBOE") -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "IND"
    c.exchange = exchange
    c.currency = "USD"
    return c


def get_monthly_slices(start_year: int = 2020, end_year: int = 2026) -> List[Tuple[str, str]]:
    """Genera lista de cortes mensuales: (YYYYMM, endDateTime en IBKR)"""
    slices = []
    # De 2020 a octubre 2026
    current_dt = datetime(start_year, 1, 1)
    end_dt = datetime(end_year, 10, 20)

    while current_dt < end_dt:
        # Fin de mes
        if current_dt.month == 12:
            next_m = datetime(current_dt.year + 1, 1, 1)
        else:
            next_m = datetime(current_dt.year, current_dt.month + 1, 1)
        
        month_label = current_dt.strftime("%Y%m")
        # Fecha límite para la petición a IBKR: primer día del mes siguiente a las 00:00 EST
        end_str = next_m.strftime("%Y%m%d 00:00:00 US/Eastern")
        slices.append((month_label, end_str))
        current_dt = next_m
    return slices


def download_symbol(app: IBKRBatchDownloader, symbol: str, is_index: bool = False, exchange: str = "SMART"):
    out_dir = DATA_DIR / symbol / "1hour"
    out_dir.mkdir(parents=True, exist_ok=True)

    contract = make_index_contract(symbol, exchange) if is_index else make_stock_contract(symbol)
    monthly_slices = get_monthly_slices(start_year=2020, end_year=2026)

    # Identificar meses pendientes
    pending = []
    for m_label, end_str in monthly_slices:
        target_file = out_dir / f"{m_label}.csv"
        # Si ya existe y tiene más de 100 bytes, se salta
        if target_file.exists() and target_file.stat().st_size > 500:
            continue
        pending.append((m_label, end_str, target_file))

    if not pending:
        print(f"[OK COMPLETO] {symbol}: Todos los 82 meses (2020-2026) ya están descargados.")
        return

    print(f"\nDescargando {symbol} ({len(pending)} meses pendientes de {len(monthly_slices)})...")

    req_id = 5000
    for idx, (m_label, end_str, target_file) in enumerate(pending, 1):
        req_id += 1
        app.current_bars = []
        app.request_finished = False
        app.has_error = False

        app.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime=end_str,
            durationStr="1 M",
            barSizeSetting="1 hour",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

        t_wait = time.time()
        while not app.request_finished and time.time() - t_wait < 10.0:
            time.sleep(0.05)

        if app.current_bars:
            df = pd.DataFrame(app.current_bars)
            df.to_csv(target_file, index=False)
            print(f"  [{idx}/{len(pending)}] {symbol} {m_label}: {len(df)} barras guardadas.", end="\r")
        else:
            if not app.has_error:
                # Mes vacío (ej. la acción no cotizaba en 2020)
                # Guardar archivo vacío indicador para no re-pedirlo en cada corrida
                target_file.write_text("date,open,high,low,close,volume,barCount,wap\n")
                print(f"  [{idx}/{len(pending)}] {symbol} {m_label}: Sin datos (posible pre-IPO).", end="\r")

        time.sleep(app.pacing_interval)

    print(f"\n  [OK] Finalizada descarga de {symbol}.")


def main():
    print("=" * 80)
    print("INICIANDO DESCARGA MULTIAÑO (2020-2026) DESDE INTERACTIVE BROKERS")
    print(f"Directorio de destino: {DATA_DIR}")
    print("=" * 80)

    app = IBKRBatchDownloader(pacing_interval=0.6)
    app.connect("127.0.0.1", 7496, clientId=95)

    import threading
    t = threading.Thread(target=app.run, daemon=True)
    t.start()

    t0 = time.time()
    while not app.connected_event and time.time() - t0 < 5.0:
        time.sleep(0.1)

    if not app.connected_event:
        print("[ERROR] No se pudo conectar a TWS en 127.0.0.1:7496")
        return

    print("[OK] Conectado exitosamente a IBKR TWS.")

    # 1. Benchmarks e Índice VIX
    for bmk in BENCHMARKS:
        download_symbol(app, bmk, is_index=False)

    for idx_sym, exch in INDICES:
        download_symbol(app, idx_sym, is_index=True, exchange=exch)

    # 2. Acciones del Universo
    for sym in UNIVERSE_STOCKS:
        download_symbol(app, sym, is_index=False)

    print("\n" + "=" * 80)
    print("¡DESCARGA MULTIAÑO DE CANASTA REPRESENTATIVA COMPLETADA CON ÉXITO!")
    print("=" * 80)
    app.disconnect()


if __name__ == "__main__":
    main()
