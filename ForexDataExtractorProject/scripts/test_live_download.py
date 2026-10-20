"""
test_live_download.py - Diagnóstico de petición de datos históricos en vivo.
"""

import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from src.utils.utils import stockContract


class LiveTest(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.bars = []
        self.error_received = None

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        print(f"  [IBKR MSG] Code {errorCode} (reqId {reqId}): {errorString}")
        if errorCode in [162, 200, 321, 502]:
            self.error_received = errorCode

    def historicalData(self, reqId, bar):
        self.bars.append(bar)

    def historicalDataEnd(self, reqId, start, end):
        print(f"  [OK] Descarga completada con {len(self.bars)} velas recibidas! (Rango: {start} a {end})")
        self.disconnect()


def run():
    print("Conectando a TWS Live (127.0.0.1:7496, clientId=55)...")
    app = LiveTest()
    app.connect("127.0.0.1", 7496, clientId=55)

    contract = stockContract("AMD")

    time.sleep(1)
    if not app.isConnected():
        print("Error: No se pudo conectar a TWS en puerto 7496.")
        return

    print("Solicitando 1 día de velas de 1 hora para AMD...")
    app.reqHistoricalData(
        1,
        contract,
        "",
        "2 D",
        "1 hour",
        "TRADES",
        1,
        1,
        False,
        [],
    )

    t0 = time.time()
    while time.time() - t0 < 8:
        app.run()
        if len(app.bars) > 0 or app.error_received:
            break
        time.sleep(0.5)

    app.disconnect()


if __name__ == "__main__":
    run()
