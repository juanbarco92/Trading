"""
check_ib_connection.py - Comprueba rápidamente si TWS o IB Gateway están abiertos y en qué puerto.
"""

import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper


class IBTest(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected_ok = False

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        pass

    def nextValidId(self, orderId):
        self.connected_ok = True
        print(f"  [OK] Conectado exitosamente con ID: {orderId}")
        self.disconnect()


def test():
    ports = [7496, 7497, 4001, 4002]
    found = False
    for p in ports:
        print(f"Probando puerto {p}...")
        app = IBTest()
        try:
            app.connect("127.0.0.1", p, clientId=98)
            time.sleep(1)
            if app.isConnected():
                print(f"-> Puerto activo encontrado: {p}")
                found = True
                app.run()
                break
        except Exception as e:
            print(f"  Fallo en puerto {p}: {e}")

    if not found:
        print("\n[AVISO] No se detectó ninguna sesión activa de TWS o IB Gateway en los puertos habituales (7496/7497/4001/4002).")
        return False
    return True


if __name__ == "__main__":
    test()
