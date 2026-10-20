"""
download_next_batch_200_ibkr.py - Descarga del Segundo Lote de 200 Acciones en IBKR (2020 a 2026)
================================================================================================
- Diseñado para encadenarse automáticamente:
  1. Detecta si la tarea previa de descarga está activa. Si está activa, espera pacientemente a que termine.
  2. Una vez libre la conexión de TWS (puerto 7496), inicia la descarga del siguiente lote de 200 acciones líquidas con ATR >= 2.0%.
  3. Esto llevará el universo total de ~250 acciones a más de ~450 acciones con 7 años de velas horarias limpias.
"""

import sys
import time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_multiaño_ibkr import IBKRBatchDownloader, download_symbol, DATA_DIR

def get_candidates():
    strict_path = PROJECT_ROOT / "data" / "processed" / "strict_prod_df_1a.parquet"
    df_1a = pd.read_parquet(strict_path)
    
    mean_atr = df_1a.groupby('ticker')['atr_d'].mean()
    count_sigs = df_1a.groupby('ticker').size()
    df_meta = pd.DataFrame({'atr': mean_atr, 'sigs': count_sigs}).sort_values('sigs', ascending=False)
    
    # Activos ya completados en disco
    downloaded = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and len(list((d / "1hour").glob("*.csv"))) >= 50]
    
    # Seleccionar 200 activos que no estén completados
    pending = [t for t in df_meta.index if t not in downloaded and df_meta.loc[t, 'atr'] >= 0.020][:200]
    return downloaded, pending

def main():
    print("=" * 80)
    print("ENCADENADOR DE DESCARGA: SEGUNDO LOTE DE 200 ACCIONES EN IBKR (2020 - 2026)")
    print("=" * 80)
    
    # Comprobar si hay otra descarga activa en background
    import psutil
    while True:
        current_pid = os.getpid() if 'os' in globals() else -1
        active_downloaders = 0
        import os
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmd = " ".join(p.info['cmdline'] or [])
                if "download_extended_universe_ibkr.py" in cmd and p.info['pid'] != os.getpid():
                    active_downloaders += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        if active_downloaders == 0:
            print("[INFO] No hay otras descargas de IBKR activas. Comenzando lote nuevo inmediatamente.")
            break
        else:
            print(f"[ESPERANDO] Hay {active_downloaders} proceso de descarga en curso. Esperando 30s antes de reintentar...")
            time.sleep(30)
            
    downloaded, pending = get_candidates()
    print(f"\nActivos ya completados en disco: {len(downloaded)}")
    print(f"Nuevos activos a descargar en este lote: {len(pending)}")
    print("Primeros 15 de este lote:", pending[:15])
    
    if len(pending) == 0:
        print("[FIN] Todos los activos del universo ya están descargados.")
        return

    app = IBKRBatchDownloader(pacing_interval=0.6)
    app.connect("127.0.0.1", 7496, clientId=97) # ClientId distinto para evitar colisiones

    import threading
    t = threading.Thread(target=app.run, daemon=True)
    t.start()

    t0 = time.time()
    while not app.connected_event and time.time() - t0 < 5.0:
        time.sleep(0.1)

    if not app.connected_event:
        print("[ERROR] No se pudo conectar a TWS en 127.0.0.1:7496")
        return

    print("[OK] Conectado a TWS Live (7496) con clientId=97. Iniciando descarga del lote de 200...\n")

    for i, sym in enumerate(pending, 1):
        print(f"--- [LOTE 2] Activo [{i}/{len(pending)}]: {sym} ---")
        try:
            download_symbol(app, sym, is_index=False)
        except Exception as e:
            print(f"  [ALERTA] Error con {sym}: {e}")
        time.sleep(1.0)

    print("\n" + "=" * 80)
    print("¡LOTE DE 200 ACCIONES ADICIONALES FINALIZADO CON ÉXITO!")
    print("=" * 80)
    app.disconnect()

if __name__ == "__main__":
    main()
