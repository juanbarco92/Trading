"""
Descarga ampliada de 250 acciones en Interactive Brokers (2020 a 2026):
- Continúa exactamente donde quedó la descarga anterior (salta los 30 ya descargados).
- Descarga los siguientes ~220 activos con mayor historial de señales y ATR >= 2.2%.
- Guarda en data_ibkr/{ticker}/1hour/{YYYYMM}.csv.
- Pacing seguro de 0.6s para operar en segundo plano.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_multiaño_ibkr import IBKRBatchDownloader, download_symbol, DATA_DIR

# Cargar candidatos ordenados por número de señales y elasticidad
strict_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_1a.parquet"
df_1a = pd.read_parquet(strict_path)

mean_atr = df_1a.groupby('ticker')['atr_d'].mean()
count_sigs = df_1a.groupby('ticker').size()
df_meta = pd.DataFrame({'atr': mean_atr, 'sigs': count_sigs}).sort_values('sigs', ascending=False)

# Ya descargados previamente
downloaded_already = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and len(list((d / "1hour").glob("*.csv"))) >= 50]

# Seleccionar los siguientes 220 más activos que no estén completos
tickers_to_download = [
    t for t in df_meta.index
    if t not in downloaded_already and df_meta.loc[t, 'atr'] >= 0.022
][:220]

def main():
    print("=" * 80)
    print(f"INICIANDO DESCARGA AMPLIADA DE {len(tickers_to_download)} ACTIVOS EN IBKR (2020 - 2026)")
    print(f"Directorio de destino: {DATA_DIR}")
    print(f"Activos ya completados previamente: {len(downloaded_already)}")
    print("=" * 80)

    app = IBKRBatchDownloader(pacing_interval=0.6)
    app.connect("127.0.0.1", 7496, clientId=96)

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

    for i, sym in enumerate(tickers_to_download, 1):
        print(f"--- Activo [{i}/{len(tickers_to_download)}]: {sym} ---")
        try:
            download_symbol(app, sym, is_index=False)
        except Exception as e:
            print(f"  [ALERTA] Error con {sym}: {e}")
        time.sleep(1.0)

    print("\n" + "=" * 80)
    print("¡DESCARGA AMPLIADA FINALIZADA CON ÉXITO!")
    print("=" * 80)
    app.disconnect()

if __name__ == "__main__":
    main()
