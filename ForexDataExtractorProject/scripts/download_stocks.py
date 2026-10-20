"""
Script de conveniencia para descargar datos históricos de Acciones y ETFs desde Interactive Brokers.
Asegúrate de tener TWS o IB Gateway iniciado con la API habilitada.
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Asegurar que el path raíz del proyecto está en sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from src.utils.utils import stockContract
from src.data_extraction.download_data import download_historical_data

if __name__ == "__main__":
    # 1. Lista de acciones o ETFs a descargar (Universo Núcleo + Alto Beta & Crecimiento)
    symbols = [
        "SPY", "QQQ",          # Benchmarks de mercado indispensables
        "MELI",                # Crecimiento / Reversión institucional
        "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA",  # Mega-caps líderes
        "AMD", "PLTR", "COIN", "ARM", "MSTR", "HOOD", "SHOP", "SMCI", "UBER",  # Alto Beta / Momentum explosivo
        "JPM", "LLY",          # Diversificación sectorial (Finanzas y Salud)
    ]

    # 2. Parámetros de descarga para Swing Trading / Monitoreo Frecuente
    candle_size = "1 hour"  # Opciones: '1 hour', '1 day', '15 mins'
    chunk_size = "1 M"      # Tamaño de bloque por petición a IBKR
    start_date = datetime(2022, 1, 1)  # Fecha mínima hacia el pasado

    # 3. Filtrar acciones que ya están descargadas (auto-resume inteligente)
    symbols_to_download = []
    clean_size = candle_size.replace(" ", "")
    for sym in symbols:
        sym_dir = root_dir / "data" / sym / clean_size
        cutoff_file = sym_dir / f"{start_date.strftime('%Y%m')}.csv"
        csv_files = list(sym_dir.glob("*.csv")) if sym_dir.exists() else []
        # ARM salió a bolsa en Sep 2023 (38 meses máx posible)
        is_arm_complete = (sym == "ARM" and len(csv_files) >= 35)
        if (cutoff_file.exists() and len(csv_files) >= 50) or is_arm_complete:
            print(f"[OK] {sym} ya está descargado ({len(csv_files)} meses). Omitiendo...")
        else:
            symbols_to_download.append(sym)

    if not symbols_to_download:
        print("\n¡Todos los activos del universo ya están descargados por completo!")
        sys.exit(0)

    # 4. Construir configuración para los activos pendientes
    contracts_config = {}
    for sym in symbols_to_download:
        contract = stockContract(sym)
        contracts_config[contract] = [
            {
                "candlesize": candle_size,
                "duration": chunk_size,
                "startDate": start_date,
                "whatToShow": "TRADES",  # Captura volumen real y transacciones
                "useRTH": 1,             # 1 = Regular Trading Hours (9:30 AM a 4:00 PM EST)
            }
        ]

    print("=" * 60)
    print(f"Descargando activos pendientes: {symbols_to_download}")
    print(f"Temporalidad: {candle_size} | Desde: {start_date.date()} | Modo: TRADES (Volumen Real)")
    print("=" * 60)

    # 4. Iniciar descarga conectando a TWS / Gateway
    # Puerto estándar: 7497 (TWS Paper Trading), 7496 (TWS Live), 4002 (Gateway Paper)
    download_historical_data(
        contracts_config,
        host="127.0.0.1",
        port=7496,
        client_id=10,
        pacing_interval=1.0,  # Pausa de 1s para respetar los límites de velocidad de IBKR
    )
