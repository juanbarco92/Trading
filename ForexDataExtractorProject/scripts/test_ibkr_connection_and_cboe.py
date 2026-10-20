"""
Script de prueba de conexión a IBKR (TWS / Gateway):
1. Prueba de conexión en puerto 7496 (Live) / 7497 (Paper).
2. Descarga de 5 días de velas horarias (1H) de una acción (AAPL).
3. Prueba de solicitud del índice VIX en CBOE (IND / CBOE / USD).
"""

import sys
import time
from pathlib import Path
from datetime import datetime

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData

class IBKRTestApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected_event = False
        self.bars_received = {}
        self.errors = []
        self.req_status = {}

    def connectAck(self):
        print("[INFO] Conexión física establecida (connectAck recibido).")

    def nextValidId(self, orderId: int):
        print(f"[OK] Conectado exitosamente a IBKR. NextValidOrderId: {orderId}")
        self.connected_event = True

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        # Ignorar mensajes informativos (2104, 2106, 2158)
        if errorCode in [2104, 2106, 2158]:
            return
        msg = f"[ERROR/MSG] reqId={reqId}, code={errorCode}: {errorString}"
        print(msg)
        self.errors.append((reqId, errorCode, errorString))

    def historicalData(self, reqId: int, bar: BarData):
        if reqId not in self.bars_received:
            self.bars_received[reqId] = []
        self.bars_received[reqId].append(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        count = len(self.bars_received.get(reqId, []))
        print(f"[OK] Descarga completada para reqId={reqId}. Barras recibidas: {count}")
        self.req_status[reqId] = "DONE"


def create_stock_contract(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c


def create_index_contract(symbol: str, exchange: str = "CBOE") -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "IND"
    c.exchange = exchange
    c.currency = "USD"
    return c


def run_test():
    ports_to_try = [7496, 7497]
    app = None
    connected_port = None

    for port in ports_to_try:
        print(f"\nIntentando conectar a IBKR en 127.0.0.1:{port} (clientId=88)...")
        test_app = IBKRTestApp()
        test_app.connect("127.0.0.1", port, clientId=88)
        
        # Esperar confirmación
        t_start = time.time()
        while time.time() - t_start < 3.0:
            test_app.run() if hasattr(test_app, "run_once") else None
            # En ibapi python, EClient corre en hilo o necesita loop
            time.sleep(0.2)
            if test_app.connected_event:
                app = test_app
                connected_port = port
                break
        
        if app and app.connected_event:
            break
        else:
            test_app.disconnect()

    if not app or not app.connected_event:
        # Si el loop de ibapi no procesó en hilo, usemos un hilo para app.run()
        print("[INFO] Probando con hilo en segundo plano para app.run()...")
        import threading
        for port in ports_to_try:
            test_app = IBKRTestApp()
            test_app.connect("127.0.0.1", port, clientId=88)
            t = threading.Thread(target=test_app.run, daemon=True)
            t.start()
            
            t_start = time.time()
            while time.time() - t_start < 4.0:
                if test_app.connected_event:
                    app = test_app
                    connected_port = port
                    break
                time.sleep(0.2)
            
            if app and app.connected_event:
                break
            else:
                test_app.disconnect()

    if not app or not app.connected_event:
        print("\n[FALLO] No se pudo conectar a TWS en los puertos 7496 ni 7497.")
        print("Verifica en TWS: Configuración -> API -> Ajustes -> Habilitar clientes ActiveX y Socket.")
        return

    print(f"\n" + "=" * 60)
    print(f"PASO 1.1: Solicitando datos de prueba para ACCIÓN (AAPL, STK, 1 hour)...")
    print("=" * 60)
    c_aapl = create_stock_contract("AAPL")
    req_id_aapl = 1001
    app.reqHistoricalData(
        reqId=req_id_aapl,
        contract=c_aapl,
        endDateTime="",
        durationStr="5 D",
        barSizeSetting="1 hour",
        whatToShow="TRADES",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
    )

    print("\n" + "=" * 60)
    print(f"PASO 1.2: Solicitando datos de prueba para ÍNDICE CBOE (VIX, IND, CBOE, 1 hour)...")
    print("=" * 60)
    c_vix = create_index_contract("VIX", exchange="CBOE")
    req_id_vix = 1002
    app.reqHistoricalData(
        reqId=req_id_vix,
        contract=c_vix,
        endDateTime="",
        durationStr="5 D",
        barSizeSetting="1 hour",
        whatToShow="TRADES",
        useRTH=1,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
    )

    # Esperar respuestas
    t_wait = time.time()
    while time.time() - t_wait < 8.0:
        if req_id_aapl in app.req_status and req_id_vix in app.req_status:
            break
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("RESUMEN DEL TEST DE CONEXIÓN Y PERMISOS IBKR:")
    print("=" * 60)

    # Evaluar AAPL
    bars_aapl = app.bars_received.get(req_id_aapl, [])
    if bars_aapl:
        print(f"[ÉXITO AAPL] Recibidas {len(bars_aapl)} barras horarias de AAPL.")
        print(f"  Última barra: Fecha={bars_aapl[-1].date}, Open={bars_aapl[-1].open}, Close={bars_aapl[-1].close}, Volume={bars_aapl[-1].volume}")
    else:
        print("[FALLO AAPL] No se recibieron barras de AAPL.")

    # Evaluar VIX
    bars_vix = app.bars_received.get(req_id_vix, [])
    if bars_vix:
        print(f"[ÉXITO CBOE VIX] Recibidas {len(bars_vix)} barras horarias de VIX.")
        print(f"  Última barra VIX: Fecha={bars_vix[-1].date}, Close={bars_vix[-1].close}")
    else:
        print("[ESTADO CBOE VIX] No se recibieron datos de VIX CBOE.")
        vix_errs = [e for e in app.errors if e[0] == req_id_vix]
        if vix_errs:
            print(f"  Detalle de error IBKR: {vix_errs[0][1]} - {vix_errs[0][2]}")

    print("\nDesconectando...")
    app.disconnect()


if __name__ == "__main__":
    run_test()
