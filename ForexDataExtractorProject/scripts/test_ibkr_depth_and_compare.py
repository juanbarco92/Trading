"""
Script de prueba de profundidad histórica y comparación IBKR vs Yahoo:
1. Pide a IBKR una ventana histórica de 2022 a 2023 (bloque de 1 mes y verificar headTimestamp).
2. Pide a IBKR una semana reciente de AAPL (octubre 2026).
3. Compara exactamente las barras horarias de esa misma semana con Yahoo Finance.
"""

import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData

class DepthAndCompareApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected_event = False
        self.bars = {}
        self.head_timestamps = {}
        self.done = {}

    def connectAck(self):
        pass

    def nextValidId(self, orderId: int):
        self.connected_event = True

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2158]:
            print(f"[IBKR ERROR/MSG] reqId={reqId}, code={errorCode}: {errorString}")

    def headTimestamp(self, reqId: int, headTimestamp: str):
        print(f"[OK] headTimestamp para reqId={reqId}: {headTimestamp}")
        self.head_timestamps[reqId] = headTimestamp
        self.done[reqId] = True

    def historicalData(self, reqId: int, bar: BarData):
        if reqId not in self.bars:
            self.bars[reqId] = []
        self.bars[reqId].append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "barCount": bar.barCount,
            "wap": getattr(bar, 'wap', getattr(bar, 'average', 0.0))
        })

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        print(f"[OK] historicalDataEnd para reqId={reqId}. Total barras: {len(self.bars.get(reqId, []))}")
        self.done[reqId] = True


def make_stock_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c


def main():
    app = DepthAndCompareApp()
    app.connect("127.0.0.1", 7496, clientId=89)

    t_thread = threading.Thread(target=app.run, daemon=True)
    t_thread.start()

    # Esperar conexión
    t0 = time.time()
    while not app.connected_event and time.time() - t0 < 4.0:
        time.sleep(0.1)

    if not app.connected_event:
        print("[FALLO] No se pudo conectar a IBKR en 7496.")
        return

    print("[OK] Conectado a TWS Live (7496).")

    # 1. Chequear headTimestamp de AAPL (Fecha más antigua disponible en IBKR)
    print("\n" + "=" * 60)
    print("TEST 1: Consultando fecha más antigua disponible en IBKR (reqHeadTimeStamp)...")
    print("=" * 60)
    c_aapl = make_stock_contract("AAPL")
    app.reqHeadTimeStamp(101, c_aapl, "TRADES", 1, 1)

    # 2. Descargar 1 mes histórico del pasado remoto (Ej. Enero de 2020) para comprobar profundidad
    print("\n" + "=" * 60)
    print("TEST 2: Solicitando bloque histórico de 1 mes de ENERO 2020 (hace casi 7 años)...")
    print("=" * 60)
    app.reqHistoricalData(
        reqId=102,
        contract=c_aapl,
        endDateTime="20200201 00:00:00 US/Eastern",
        durationStr="1 M",
        barSizeSetting="1 hour",
        whatToShow="TRADES",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
    )

    # 3. Descargar la última semana de trading (Octubre 2026) para comparar con Yahoo
    print("\n" + "=" * 60)
    print("TEST 3: Solicitando semana reciente (10 Días) de AAPL para comparar con Yahoo...")
    print("=" * 60)
    app.reqHistoricalData(
        reqId=103,
        contract=c_aapl,
        endDateTime="",
        durationStr="10 D",
        barSizeSetting="1 hour",
        whatToShow="TRADES",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
    )

    # Esperar que terminen las peticiones
    t_wait = time.time()
    while time.time() - t_wait < 10.0:
        if all(k in app.done for k in [101, 102, 103]):
            break
        time.sleep(0.5)

    app.disconnect()

    print("\n" + "=" * 80)
    print("RESULTADOS DE LA PRUEBA:")
    print("=" * 80)

    # Test 1
    head_ts = app.head_timestamps.get(101, "No recibida")
    print(f"1. Fecha de inicio más antigua disponible en IBKR para AAPL: {head_ts}")

    # Test 2
    b_2020 = app.bars.get(102, [])
    print(f"2. Barras recibidas de Enero 2020: {len(b_2020)}")
    if b_2020:
        df_2020 = pd.DataFrame(b_2020)
        print(f"   Primera barra 2020: {df_2020.iloc[0]['date']} | Open={df_2020.iloc[0]['open']} | Close={df_2020.iloc[0]['close']}")
        print(f"   Última barra 2020:  {df_2020.iloc[-1]['date']} | Open={df_2020.iloc[-1]['open']} | Close={df_2020.iloc[-1]['close']}")

    # Test 3: Comparación con Yahoo
    b_recent = app.bars.get(103, [])
    print(f"\n3. Barras recientes recibidas de IBKR: {len(b_recent)}")
    if b_recent:
        df_ibkr = pd.DataFrame(b_recent)
        print("\nÚltimas 5 barras horarias en IBKR:")
        print(df_ibkr[['date', 'open', 'high', 'low', 'close', 'volume']].tail(5).to_string(index=False))

        # Descargar de Yahoo para comparar
        print("\nDescargando las mismas barras desde Yahoo Finance...")
        try:
            yf_df = yf.download("AAPL", period="10d", interval="1h", progress=False)
            if not yf_df.empty:
                if isinstance(yf_df.columns, pd.MultiIndex):
                    yf_df.columns = yf_df.columns.get_level_values(0)
                yf_df = yf_df.reset_index()
                date_col = "Datetime" if "Datetime" in yf_df.columns else "Date"
                print("\nÚltimas 5 barras horarias en Yahoo:")
                print(yf_df[[date_col, 'Open', 'High', 'Low', 'Close', 'Volume']].tail(5).to_string(index=False))
        except Exception as e:
            print(f"Error descargando Yahoo: {e}")


if __name__ == "__main__":
    main()
