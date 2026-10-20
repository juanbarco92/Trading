from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import threading
import time

class TestApp(EWrapper, EClient):
    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        self.connected_ok = False

    def nextValidId(self, orderId: int):
        print(f"[OK] Conectado exitosamente a IBKR TWS puerto 7496! OrderId: {orderId}")
        self.connected_ok = True
        self.disconnect()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2158]:
            print(f"[IBKR {errorCode}]: {errorString}")

app = TestApp()
app.connect("127.0.0.1", 7496, clientId=99)
t = threading.Thread(target=app.run, daemon=True)
t.start()
time.sleep(3)
if not app.connected_ok:
    print("[TIMEOUT] No se recibio confirmacion de conexion.")
    app.disconnect()
