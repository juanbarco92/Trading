import os
import sys
import logging
import threading
import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from collections import defaultdict
import math # For floor function
import traceback # For detailed error printing
import json # For loading metadata

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.utils import iswrapper
from ibapi.contract import Contract
from ibapi.common import BarData # Import BarData

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
    
import pytz
import pandas as pd
import numpy as np
import joblib

# --- Configuración del Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
log_handler = logging.FileHandler('realtime_processor.log', mode='a') # 'a' para añadir, 'w' para sobrescribir
log_handler.setFormatter(log_formatter)

# Configurar también un handler para la consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO) # Nivel mínimo a registrar (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# Limpiar handlers existentes para evitar duplicados si se recarga el módulo
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(log_handler)
logger.addHandler(console_handler) # Añadir handler de consola

# --- Intenta importar la función de ingeniería de características ---
try:
    # Asume que feature_engineering.py está en el mismo directorio o en PYTHONPATH
    from ..utils.exploration_utils import engineer_features
    FEATURE_ENGINEERING_AVAILABLE = True
    logger.info("Función 'engineer_features' cargada correctamente.")
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    engineer_features = None # Define como None si no se puede importar
    logger.warning("ADVERTENCIA: No se pudo importar 'engineer_features' desde feature_engineering.py. La predicción estará deshabilitada.")
except Exception as e:
    FEATURE_ENGINEERING_AVAILABLE = False
    engineer_features = None
    logger.error(f"ERROR al importar 'engineer_features': {e}. La predicción estará deshabilitada.", exc_info=True)


# Define types for clarity
FiveMinBarDict = Dict[str, Optional[Union[float, int, datetime]]] # Type for the aggregated bar
ContractRequestDetails = List[Dict[str, str]] # e.g., [{'candlesize_label': '5 mins'}]
ContractDict = Dict[Contract, ContractRequestDetails]


# --- Constantes para Modelos y Datos Históricos ---
MODEL_BASE_PATH = "models/{ticker}/" # Ajusta si los modelos están en otro lugar
CLUSTERER_FILENAME_TPL = "forex_candle_clusterer_{ticker}_{interval}.joblib"
PIPELINE_FILENAME_TPL = "forex_rf_pipeline_{ticker}_{interval}.joblib"
METADATA_FILENAME_TPL = "forex_rf_pipeline_{ticker}_{interval}_metadata.json" # Añadir plantilla para metadata

REQUIRED_HISTORICAL_BARS = 150 # Número de barras necesarias para ingeniería de características + buffer
HISTORICAL_DATA_BAR_SIZE_MAP = { # Mapeo para la API de IBKR
    '5 mins': '5 mins',
    # Añade otros si es necesario
}

def make_save_path(ticket, candlesize_label) -> str:
    """Make path for saving csv files based on ticker and descriptive label."""
    path = os.path.sep.join(
        [
            "data_realtime_aggregated", # Different base directory
            ticket,
            candlesize_label.replace(" ", "")
        ]
    )
    return path


class RealTimeAggregatorSaver(EClient, EWrapper):

    # --- Configuration ---
    WHAT_TO_SHOW = "MIDPOINT"
    USERTH = 1
    BAR_SIZE_SECONDS = 5 # IBKR API requirement
    AGGREGATION_MINUTES = 5 # Target aggregation interval
    SAVE_INTERVAL = 'hourly' # Options: 'hourly', 'daily' for saving aggregated bars
    HISTORICAL_DATA_END_DATETIME = '' # Solicitar datos hasta 'ahora'
    HISTORICAL_DATA_DURATION = '2 D' # Pedir 2 días de datos históricos de 5 min

    N_CANDLES_PARAM = 15
    N_CLUSTERS_PARAM = 5 
    MIN_PATTERN_THRESHOLD_PARAM = 100 

    # --- End Configuration ---

    def __init__(self, contracts: ContractDict):
        EClient.__init__(self, self)
        self.next_valid_order_id = None
        self.started = False # Flag para requests en tiempo real
        self.historical_requests_started = False # Flag para requests históricos
        self.request_id_counter = 0
        self.contracts_config = contracts
        self.requests = {} # Map reqId back to contract and config details {reqId: {'contract': Contract, 'candlesize_label': str, 'type': 'historical'|'realtime', 'aggregator_key': str}}

        # Data storage
        self.aggregated_bar_accumulators = defaultdict(list)
        self.current_5min_bars = defaultdict(lambda: None)
        self.last_save_period = defaultdict(str)
        self.historical_data = defaultdict(pd.DataFrame) # Almacena histórico + nuevas barras {aggregator_key: DataFrame}
        self.historical_data_received = defaultdict(threading.Event) # Señaliza cuándo se recibe el histórico inicial
        self.models = {} # Almacena modelos cargados {aggregator_key: {'clusterer': obj, 'pipeline': obj, 'metadata': dict}}
        self.last_historical_bar_time = {}

        self._lock = threading.Lock() # Lock para proteger acceso a datos compartidos

        # --- Connection State ---
        self.connection_lost_unexpectedly = False # Flag para indicar pérdida inesperada
        self._disconnect_called = False # Flag para saber si la desconexión fue intencional

        # --- Cargar Modelos ---
        self._load_models()

    def _load_models(self):
        """Carga los modelos de predicción para los contratos configurados."""
        if not FEATURE_ENGINEERING_AVAILABLE:
            logger.info("Saltando carga de modelos porque la ingeniería de características no está disponible.")
            return

        logger.info("Cargando modelos de predicción...")
        for contract, request_details_list in self.contracts_config.items():
             for details in request_details_list:
                candlesize_label = details.get('candlesize_label', f'{self.AGGREGATION_MINUTES} mins')
                ticker_base = f"{contract.symbol}{contract.currency}"
                aggregator_key = f"{ticker_base}_{candlesize_label}"

                # Formatear nombres como en el notebook (lowercase, sin espacios)
                ticker_lower = ticker_base.lower()
                interval_simple = candlesize_label.replace(" ", "") # ej. "5mins"

                clusterer_path = os.path.join(MODEL_BASE_PATH.format(ticker=ticker_base), CLUSTERER_FILENAME_TPL.format(ticker=ticker_lower, interval=interval_simple))
                pipeline_path = os.path.join(MODEL_BASE_PATH.format(ticker=ticker_base), PIPELINE_FILENAME_TPL.format(ticker=ticker_lower, interval=interval_simple))
                metadata_path = os.path.join(MODEL_BASE_PATH.format(ticker=ticker_base), METADATA_FILENAME_TPL.format(ticker=ticker_lower, interval=interval_simple)) # Ruta metadata

                try:
                    clusterer = joblib.load(clusterer_path)
                    pipeline = joblib.load(pipeline_path)
                    # Cargar metadata
                    metadata = {}
                    if os.path.exists(metadata_path):
                         with open(metadata_path, 'r') as f:
                              metadata = json.load(f)
                         logger.info(f"  - Metadata cargada desde: {metadata_path}")
                    else:
                         logger.warning(f"  - Archivo de metadata no encontrado en: {metadata_path}")

                    self.models[aggregator_key] = {
                        'clusterer': clusterer,
                        'pipeline': pipeline,
                        'metadata': metadata # Guardar metadata
                    }
                    logger.info(f"Modelos y metadata cargados para {aggregator_key}:")
                    logger.info(f"  - Clusterer: {clusterer_path}")
                    logger.info(f"  - Pipeline: {pipeline_path}")
                    # Loggear umbrales cargados
                    buy_thresh = metadata.get('optimal_buy_threshold')
                    sell_thresh = metadata.get('optimal_sell_threshold')
                    logger.info(f"    - Umbral Compra Óptimo: {buy_thresh}")
                    logger.info(f"    - Umbral Venta Óptimo: {sell_thresh}")

                except FileNotFoundError:
                    logger.warning(f"No se encontraron archivos de modelo para {aggregator_key}. Rutas buscadas:")
                    logger.warning(f"  - {clusterer_path}")
                    logger.warning(f"  - {pipeline_path}")
                    logger.warning(f"  La predicción para {aggregator_key} estará deshabilitada.")
                except Exception as e:
                    logger.error(f"ERROR al cargar modelos para {aggregator_key}: {e}", exc_info=True)
                    logger.error(f"  La predicción para {aggregator_key} estará deshabilitada.")


    # --- Connection Handling ---
    @iswrapper
    def connectAck(self):
        logger.info("Conectado a IBKR.")
        # Resetear flags al conectar/reconectar
        self.connection_lost_unexpectedly = False
        self._disconnect_called = False

    @iswrapper
    def nextValidId(self, order_id: int):
        super().nextValidId(order_id)
        self.next_valid_order_id = order_id
        logger.info(f"nextValidId: {order_id}")
        # Solicitar datos históricos PRIMERO
        self.start_historical_data_requests()
        # Los requests en tiempo real se iniciarán después de recibir los históricos

    def next_request_id(self) -> int:
        # Usar lock para seguridad en threads si se llama desde múltiples sitios (aunque aquí parece secuencial)
        with self._lock:
            self.request_id_counter += 1
            return self.request_id_counter

    # --- Request Initiation ---
    def start_historical_data_requests(self):
        """Calcula la espera hasta el próximo intervalo, espera, y luego solicita los datos históricos."""
        if self.historical_requests_started:
            return
        self.historical_requests_started = True # Marcar como iniciado para evitar re-entradas

        # --- Calcular y ejecutar espera ---
        now_utc = datetime.now(pytz.utc)
        now_ts = int(now_utc.timestamp())
        aggregation_seconds = self.AGGREGATION_MINUTES * 60

        # Calcular el timestamp de inicio del *próximo* intervalo de 5 minutos
        # Usamos math.ceil para redondear hacia arriba al siguiente múltiplo
        next_interval_start_ts = math.ceil(now_ts / aggregation_seconds) * aggregation_seconds

        # Si ya estamos exactamente en el inicio de un intervalo, no esperamos (o esperamos 0s)
        # Y el tiempo final del histórico será este mismo inicio de intervalo.
        if now_ts == next_interval_start_ts:
             seconds_to_wait = 0
             historical_end_dt_utc = now_utc # Usar el tiempo actual exacto
        else:
             seconds_to_wait = next_interval_start_ts - now_ts
             historical_end_dt_utc = datetime.fromtimestamp(next_interval_start_ts, tz=pytz.utc)

        if seconds_to_wait > 0:
            wait_until_dt_str = historical_end_dt_utc.strftime('%Y-%m-%d %H:%M:%S %Z')
            logger.info(f"Esperando {seconds_to_wait:.2f} segundos hasta el próximo intervalo de {self.AGGREGATION_MINUTES} minutos ({wait_until_dt_str})...")
            time.sleep(seconds_to_wait)
            logger.info(f"Espera completada. Procediendo a solicitar datos históricos.")
            # Recalcular el tiempo final después de la espera para mayor precisión si es necesario,
            # aunque usar next_interval_start_ts debería ser correcto.
            historical_end_dt_utc = datetime.fromtimestamp(next_interval_start_ts, tz=pytz.utc)
        else:
             logger.info(f"Ya estamos en un inicio de intervalo ({historical_end_dt_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}). No se espera.")

        # Formatear el tiempo final para la API: YYYYMMDD HH:MM:SS UTC
        # Asegúrate de que tu versión de Python/pytz maneje %Z correctamente para añadir 'UTC'
        # Si no, puedes añadirlo manualmente.
        historical_end_datetime_str = historical_end_dt_utc.strftime('%Y%m%d %H:%M:%S') + " UTC"
        logger.info(f"Solicitando datos históricos hasta: {historical_end_datetime_str}")
        # --- Fin de la lógica de espera ---

        logger.info("Solicitando datos históricos iniciales...")
        if not self.contracts_config:
            logger.warning("No hay contratos configurados. Saltando solicitud de datos históricos.")
            self.start_realtime_requests() # Iniciar tiempo real directamente
            return

        expected_historical_requests = 0
        for contract, request_details_list in self.contracts_config.items():
            for details in request_details_list:
                candlesize_label = details.get('candlesize_label', f'{self.AGGREGATION_MINUTES} mins')
                ticker = f"{contract.symbol}{contract.currency}"
                aggregator_key = f"{ticker}_{candlesize_label}"

                api_bar_size = HISTORICAL_DATA_BAR_SIZE_MAP.get(candlesize_label)
                if not api_bar_size:
                    logger.warning(f"No se encontró mapeo de tamaño de barra para '{candlesize_label}'. Saltando histórico para {aggregator_key}.")
                    continue

                self.historical_data_received[aggregator_key].clear()
                expected_historical_requests += 1

                reqId = self.next_request_id()
                self.requests[reqId] = {
                    'contract': contract,
                    'candlesize_label': candlesize_label,
                    'type': 'historical',
                    'aggregator_key': aggregator_key
                }

                logger.info(f"Solicitando datos históricos para {aggregator_key} ({api_bar_size}, Dur: {self.HISTORICAL_DATA_DURATION}) con reqId {reqId}")
                self.reqHistoricalData(
                    reqId=reqId,
                    contract=contract,
                    endDateTime=historical_end_datetime_str, # Usar el tiempo final calculado
                    durationStr=self.HISTORICAL_DATA_DURATION,
                    barSizeSetting=api_bar_size,
                    whatToShow=self.WHAT_TO_SHOW,
                    useRTH=self.USERTH,
                    formatDate=1, # 1 for UTC timestamps yyyymmdd hh:mm:ss
                    keepUpToDate=False,
                    chartOptions=[]
                )

        # Iniciar hilo para esperar los datos históricos (sin cambios aquí)
        if expected_historical_requests > 0:
            logger.info(f"Esperando {expected_historical_requests} streams de datos históricos...")
            wait_thread = threading.Thread(target=self._wait_for_historical_data, args=(list(self.historical_data_received.keys()),), daemon=True)
            wait_thread.start()
        else:
             logger.info("No se solicitaron datos históricos, iniciando requests en tiempo real.")
             self.start_realtime_requests()

    def _wait_for_historical_data(self, keys_to_wait_for: List[str]):
        """Espera a que todos los eventos de datos históricos se activen."""
        timeout_seconds = 300 # 5 minutos de timeout total
        start_time = time.time()
        all_received = True

        for key in keys_to_wait_for:
            event = self.historical_data_received[key]
            remaining_time = timeout_seconds - (time.time() - start_time)
            if remaining_time <= 0:
                 logger.error(f"TIMEOUT general esperando datos históricos antes de procesar {key}.")
                 all_received = False
                 break # Salir del bucle

            if event.wait(timeout=remaining_time):
                logger.info(f"Datos históricos iniciales recibidos para {key}.")
            else:
                 logger.error(f"TIMEOUT esperando datos históricos específicos para {key}.")
                 all_received = False
                 # Podríamos decidir continuar esperando otros, pero por simplicidad, marcamos fallo general

        if all_received:
            logger.info("Todos los datos históricos iniciales necesarios han sido recibidos.")
        else:
            logger.warning("ADVERTENCIA: No se recibieron todos los datos históricos iniciales (timeout o error). Las predicciones pueden fallar para algunos instrumentos.")

        logger.info("Iniciando requests de datos en tiempo real...")
        self.start_realtime_requests() # Iniciar tiempo real ahora

    def start_realtime_requests(self):
        """Inicia las solicitudes de barras en tiempo real (5 segundos)."""
        if self.started:
            return
        self.started = True
        logger.info("Iniciando requests de datos en tiempo real (barras de 5 segundos)...")

        for contract, request_details_list in self.contracts_config.items():
            for details in request_details_list:
                candlesize_label = details.get('candlesize_label', f'{self.AGGREGATION_MINUTES} mins')
                ticker = f"{contract.symbol}{contract.currency}"
                aggregator_key = f"{ticker}_{candlesize_label}"

                # Verificar si el histórico se recibió (aunque _wait_for_historical_data debería asegurar esto)
                if not self.historical_data_received[aggregator_key].is_set():
                     logger.warning(f"Iniciando tiempo real para {aggregator_key} pero los datos históricos aún no están marcados como recibidos.")

                reqId = self.next_request_id()
                self.requests[reqId] = {
                    'contract': contract,
                    'candlesize_label': candlesize_label,
                    'type': 'realtime',
                    'aggregator_key': aggregator_key
                }

                logger.info(f"Solicitando barras en tiempo real para {aggregator_key} (reqId {reqId})")
                self.reqRealTimeBars(
                    reqId=reqId,
                    contract=contract,
                    barSize=self.BAR_SIZE_SECONDS, # Siempre 5 segundos para agregar
                    whatToShow=self.WHAT_TO_SHOW,
                    useRTH=self.USERTH,
                    realTimeBarsOptions=[]
                )

    # --- Data Handling Callbacks ---
    @iswrapper
    def historicalData(self, reqId: int, bar: BarData):
        """Callback recibiendo datos históricos."""
        request_info = self.requests.get(reqId)
        if not request_info or request_info.get('type') != 'historical':
            return # Ignorar si no es una solicitud histórica nuestra

        aggregator_key = request_info['aggregator_key']

        # Almacenar temporalmente (procesar en historicalDataEnd)
        if not hasattr(self, '_temp_historical_bars'):
             self._temp_historical_bars = defaultdict(list)

        try:
            # --- Nueva lógica de parseo inspirada en preprocess_data ---
            dt_obj_utc = None
            bar_date_str = bar.date # ej: '20250418 14:05:00 US/Eastern' o '1678886400'

            # Intentar parsear como 'YYYYMMDD HH:MM:SS TZ' primero
            try:
                # Usamos pd.to_datetime que puede manejar '%Z' para la zona horaria
                # errors='coerce' -> NaT si falla; utc=False -> respeta la zona original
                localized_dt = pd.to_datetime(bar_date_str, format='%Y%m%d %H:%M:%S %Z', errors='coerce', utc=False)

                if not pd.isna(localized_dt):
                    # Si el parseo fue exitoso, convertir a UTC
                    dt_obj_utc = localized_dt.tz_convert('UTC')
                    # Convertir de Timestamp de pandas a datetime de Python
                    dt_obj_utc = dt_obj_utc.to_pydatetime()
                else:
                    # Si falló el formato con TZ, intenta otros formatos
                    raise ValueError("No coincide con formato YYYYMMDD HH:MM:SS TZ")

            except ValueError:
                 # Fallback: Intentar como 'YYYYMMDD HH:MM:SS' (asumiendo UTC como pide formatDate=1)
                 try:
                     naive_dt = datetime.strptime(bar_date_str, '%Y%m%d %H:%M:%S')
                     dt_obj_utc = pytz.utc.localize(naive_dt)
                 except ValueError:
                     # Fallback: Intentar como timestamp UNIX
                     try:
                         dt_obj_utc = datetime.fromtimestamp(int(bar_date_str), tz=pytz.utc)
                     except ValueError:
                         # Fallback: Intentar como solo fecha 'YYYYMMDD' (menos probable)
                         try:
                             naive_dt = datetime.strptime(bar_date_str, '%Y%m%d')
                             dt_obj_utc = pytz.utc.localize(naive_dt) # Asumir UTC
                         except ValueError:
                              logger.error(f"Formato de fecha histórica no reconocido '{bar.date}' para reqId {reqId}. Saltando barra.")
                              return # Saltar si ningún formato funciona

            # --- Fin nueva lógica de parseo ---

            # Si dt_obj_utc no se pudo determinar, salir (aunque los fallbacks deberían cubrirlo)
            if dt_obj_utc is None:
                 logger.error(f"No se pudo determinar datetime UTC para '{bar.date}' (reqId {reqId}).")
                 return

            self._temp_historical_bars[aggregator_key].append({
                "datetime_utc": dt_obj_utc,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume) if bar.volume != -1 else 0.0, # Usar 0 si el volumen es -1
                "barCount": int(bar.barCount) if bar.barCount != -1 else 0 # Usar 0 si count es -1
            })
        except Exception as e:
             logger.error(f"ERROR procesando barra histórica para reqId {reqId}: {e}", exc_info=True)
             logger.error(f"Barra problemática: {bar}")


    @iswrapper
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Marca el fin de la recepción de datos históricos para una solicitud."""
        super().historicalDataEnd(reqId, start, end)
        request_info = self.requests.get(reqId)
        if not request_info or request_info.get('type') != 'historical':
            return # Ignorar

        aggregator_key = request_info['aggregator_key']
        logger.info(f"Recepción de datos históricos finalizada para {aggregator_key} (reqId {reqId}). Periodo: {start} - {end}")

        # Procesar las barras recolectadas
        temp_bars = self._temp_historical_bars.pop(aggregator_key, [])

        if temp_bars:
             try:
                 df = pd.DataFrame(temp_bars)
                 df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
                 df.set_index('datetime_utc', inplace=True)
                 df.sort_index(inplace=True)
                 # Eliminar duplicados por si acaso (manteniendo el último)
                 df = df[~df.index.duplicated(keep='last')]

                 # Asegurar tipos correctos
                 df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float, 'barCount': int})

                 with self._lock: # Proteger acceso a self.historical_data
                    self.historical_data[aggregator_key] = df
                    if not df.empty:
                        last_ts = df.index[-1]
                        self.last_historical_bar_time[aggregator_key] = last_ts
                        logger.info(f"Stored last historical bar time for {aggregator_key}: {last_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                 logger.info(f"Almacenadas {len(df)} barras históricas para {aggregator_key}.")

             except Exception as e:
                 logger.error(f"ERROR procesando DataFrame histórico para {aggregator_key}: {e}", exc_info=True)
                 # Marcar como recibido pero vacío para evitar bloqueo infinito
                 with self._lock:
                     self.historical_data[aggregator_key] = pd.DataFrame() # DataFrame vacío
        else:
             logger.warning(f"No se recibieron barras temporales para {aggregator_key} (reqId {reqId}) al finalizar el histórico.")
             # Marcar como recibido pero vacío
             with self._lock:
                 self.historical_data[aggregator_key] = pd.DataFrame()

        # Señalizar que los datos para esta clave están listos (o se intentó cargarlos)
        self.historical_data_received[aggregator_key].set()


    @iswrapper
    def realtimeBar(self, reqId: int, time_ts: int, open_: float, high: float, low: float, close: float, volume: int, wap: float, count: int):
        """Callback recibiendo barras de 5 segundos y agregándolas."""
        # Renombrado 'time' a 'time_ts' para evitar conflicto con el módulo 'time'
        super().realtimeBar(reqId, time_ts, open_, high, low, close, volume, wap, count)

        request_info = self.requests.get(reqId)
        if not request_info:
            return

        # Asegurarse de que es un request de tiempo real (aunque el callback lo diferencia)
        if request_info.get('type') != 'realtime':
             logger.warning(f"Received realtimeBar callback for non-realtime reqId: {reqId}")
             return

        aggregator_key = request_info['aggregator_key']

        # Convertir timestamp de IB (segundos desde epoch UTC) a datetime object (UTC)
        bar_5s_time_utc = datetime.fromtimestamp(time_ts, tz=pytz.utc)

        # Determinar la hora de inicio del intervalo de 5 minutos al que pertenece esta barra
        aggregation_seconds = self.AGGREGATION_MINUTES * 60
        interval_start_timestamp = math.floor(time_ts / aggregation_seconds) * aggregation_seconds
        interval_start_time_utc = datetime.fromtimestamp(interval_start_timestamp, tz=pytz.utc)

        completed_bar_to_process: Optional[FiveMinBarDict] = None

        with self._lock:
            is_potential_first_bar_for_interval = self.current_5min_bars.get(aggregator_key) is None
            last_hist_time = self.last_historical_bar_time.get(aggregator_key)

        # Realizar la comprobación fuera del lock
        if is_potential_first_bar_for_interval and last_hist_time is not None and interval_start_time_utc == last_hist_time:
            logger.info(f"[{aggregator_key}] Ignorando primera realtimeBar @ {interval_start_time_utc.strftime('%Y-%m-%d %H:%M:%S')} - solapa con última barra histórica.")
            # Simplemente saltamos el procesamiento de esta barra de 5s.
            # La *siguiente* barra de 5s para este intervalo iniciará correctamente la agregación.
            return
        
        with self._lock:
            current_agg_bar = self.current_5min_bars[aggregator_key]

            # --- Comprobar si esta barra de 5s inicia un NUEVO intervalo de 5 minutos ---
            if current_agg_bar is None or current_agg_bar['start_time_utc'] != interval_start_time_utc:

                # 1. Finalizar la barra ANTERIOR (si existe)
                if current_agg_bar is not None:
                    logger.info(f"Completada barra {self.AGGREGATION_MINUTES}-min para {aggregator_key} @ {current_agg_bar['start_time_utc'].strftime('%Y-%m-%d %H:%M:%S')} "
                          f"O:{current_agg_bar['open']:.5f} H:{current_agg_bar['high']:.5f} L:{current_agg_bar['low']:.5f} C:{current_agg_bar['close']:.5f} V:{current_agg_bar['volume']:.0f}")

                    # Añadir la barra completada al acumulador para guardar
                    self.aggregated_bar_accumulators[aggregator_key].append(current_agg_bar)
                    # Marcarla para procesar (guardado y predicción) fuera del lock
                    completed_bar_to_process = current_agg_bar.copy() # Copiar para evitar modificaciones

                # 2. Iniciar la NUEVA barra
                self.current_5min_bars[aggregator_key] = {
                    "start_time_utc": interval_start_time_utc,
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(volume), # Iniciar volumen como float
                    "barCount": int(count)
                }

            # --- Else: Actualizar el intervalo ACTUAL de 5 minutos ---
            else:
                # Asegurar que los valores existen antes de usar max/min
                current_agg_bar['high'] = max(current_agg_bar.get('high', -np.inf), float(high))
                current_agg_bar['low'] = min(current_agg_bar.get('low', np.inf), float(low))
                current_agg_bar['close'] = float(close) # El último cierre es el cierre del intervalo
                current_agg_bar['volume'] = current_agg_bar.get('volume', 0.0) + float(volume) # Acumular como float
                current_agg_bar['barCount'] = current_agg_bar.get('barCount', 0) + int(count)


        # --- Procesar la barra completada (fuera del lock) ---
        if completed_bar_to_process:
            # 1. Comprobar y guardar datos agregados
            self.check_and_save_aggregated_data(aggregator_key, completed_bar_to_process['start_time_utc'])

            # 2. Procesar para predicción (si está disponible)
            if FEATURE_ENGINEERING_AVAILABLE and aggregator_key in self.models:
                 # Verificar si el histórico inicial ya se recibió para esta clave
                 if self.historical_data_received[aggregator_key].is_set():
                     self.process_new_bar_for_prediction(aggregator_key, completed_bar_to_process)
                 else:
                     logger.info(f"Saltando predicción para {aggregator_key}, aún esperando datos históricos iniciales.")


    def process_new_bar_for_prediction(self, aggregator_key: str, completed_bar: FiveMinBarDict):
        """Añade la nueva barra al histórico, ejecuta ingeniería de características y predice."""
        if not engineer_features: # Doble check por si acaso
             logger.error("Intento de predicción pero engineer_features no está disponible.")
             return

        try:
            # --- Obtener y actualizar DataFrame histórico (con lock) ---
            with self._lock:
                hist_df = self.historical_data.get(aggregator_key)
                if hist_df is None or hist_df.empty:
                    logger.error(f"No hay datos históricos (o está vacío) para {aggregator_key} al intentar predecir.")
                    return # No se puede continuar sin histórico

                # Crear fila para la nueva barra
                new_bar_time = completed_bar['start_time_utc']
                new_row_data = {
                    'open': completed_bar['open'],
                    'high': completed_bar['high'],
                    'low': completed_bar['low'],
                    'close': completed_bar['close'],
                    'volume': completed_bar['volume'],
                    'barCount': completed_bar['barCount']
                }

                # Comprobar si el índice ya existe (evitar duplicados estrictos)
                if new_bar_time in hist_df.index:
                    logger.warning(f"Timestamp duplicado {new_bar_time} detectado para {aggregator_key}. Actualizando fila existente.")
                    # Actualizar la fila existente en lugar de añadir
                    for col, value in new_row_data.items():
                         hist_df.loc[new_bar_time, col] = value
                else:
                    # Añadir la nueva fila
                    new_row = pd.DataFrame([new_row_data], index=[new_bar_time])
                    new_row.index.name = 'datetime_utc'
                    hist_df = pd.concat([hist_df, new_row])
                    hist_df.sort_index(inplace=True) # Asegurar orden

                # --- Recortar DataFrame para mantener tamaño manejable ---
                max_len = REQUIRED_HISTORICAL_BARS + 50
                if len(hist_df) > max_len:
                    hist_df = hist_df.iloc[-max_len:]

                # Actualizar el DataFrame almacenado
                self.historical_data[aggregator_key] = hist_df

                # --- Comprobar si hay suficientes datos ---
                if len(hist_df) < REQUIRED_HISTORICAL_BARS:
                    logger.info(f"No hay suficientes datos históricos para {aggregator_key} ({len(hist_df)}/{REQUIRED_HISTORICAL_BARS}). Saltando predicción.")
                    return

                # --- Crear copia para ingeniería de características ---
                df_for_features = hist_df.copy()

            # --- Añadir Componentes de Tiempo (similar a preprocess_data) ---
            # Asegurarse de que el índice es datetime y UTC
            if pd.api.types.is_datetime64_any_dtype(df_for_features.index) and df_for_features.index.tz == pytz.UTC:
                logger.debug(f"    -> Añadiendo componentes de tiempo a df_for_features para {aggregator_key}...")
                df_for_features['Year'] = df_for_features.index.year
                df_for_features['Month'] = df_for_features.index.strftime('%Y%m') # Formato YYYYMM
                df_for_features['Day'] = df_for_features.index.day
                # df_for_features['Date'] = df_for_features.index.strftime('%Y%m%d') # Descomentar si es necesario
                df_for_features['Hour'] = df_for_features.index.hour # Hora UTC (0-23)
                df_for_features['Minute'] = df_for_features.index.minute # Minuto UTC (0-59)
                df_for_features['DayOfWeek'] = df_for_features.index.dayofweek # Lunes=0, Domingo=6
                logger.debug(f"    -> Componentes de tiempo añadidos.")
            else:
                logger.warning(f"    -> El índice de df_for_features no es datetime UTC. Saltando adición de componentes de tiempo.")
            # --- Fin Añadir Componentes de Tiempo ---

            # --- Ejecutar Ingeniería de Características (fuera del lock) ---
            required_cols = ['open', 'high', 'low', 'close', 'volume'] # Columnas básicas OHLCV
            # Añadir columnas de tiempo que engineer_features pueda necesitar
            time_cols = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'DayOfWeek']
            required_cols.extend([col for col in time_cols if col in df_for_features.columns])

            if not all(col in df_for_features.columns for col in required_cols):
                 # Crear lista de las que realmente faltan
                 missing_required = [col for col in required_cols if col not in df_for_features.columns]
                 logger.error(f"Faltan columnas requeridas ({missing_required}) en df_for_features para {aggregator_key}.")
                 return

            # --- Obtener el clusterer cargado ---
            models = self.models.get(aggregator_key)
            if not models or 'clusterer' not in models:
                 logger.error(f"Clusterer no encontrado para {aggregator_key}. No se puede ejecutar engineer_features correctamente.")
                 return
            fitted_clusterer = models['clusterer']
            # --- Fin obtener clusterer ---

            start_time_fe = time.time()
            # --- Llamada CORREGIDA a engineer_features ---
            try:
                # Asegúrate que engineer_features acepta 'clusterer' y 'fit_clusterer'
                result_from_engineer = engineer_features(
                    df_for_features,
                    n_candles=self.N_CANDLES_PARAM,
                    n_clusters=self.N_CLUSTERS_PARAM,
                    min_pattern_threshold=self.MIN_PATTERN_THRESHOLD_PARAM,
                    clusterer=fitted_clusterer, # <--- PASAR EL CLUSTERER CARGADO
                    fit_clusterer=False        # <--- NO REAJUSTAR
                )
            except Exception as fe_error:
                 logger.error(f"Error durante la llamada a engineer_features para {aggregator_key}: {fe_error}", exc_info=True)
                 return
            # --- Fin Llamada CORREGIDA ---

            if isinstance(result_from_engineer, tuple):
                 features_df = result_from_engineer[0]
                 logger.debug(f"engineer_features devolvió una tupla. Usando el primer elemento como DataFrame. Contenido completo: {result_from_engineer}")
            elif isinstance(result_from_engineer, pd.DataFrame):
                 features_df = result_from_engineer
            else:
                 logger.error(f"Tipo inesperado devuelto por engineer_features: {type(result_from_engineer)}. Se esperaba DataFrame o Tupla.")
                 return

            end_time_fe = time.time()
            duration_fe = end_time_fe - start_time_fe
            logger.info(f"    -> Tiempo Ingeniería Características ({aggregator_key}): {duration_fe:.4f} segundos")

            latest_features = features_df.iloc[-1:]

            if latest_features.empty:
                 logger.error(f"La ingeniería de características devolvió un resultado vacío para {aggregator_key}.")
                 return

            # Verificar si 'pattern' se creó (opcional pero bueno para debug)
            if 'pattern' not in features_df.columns:
                 logger.warning(f"La columna 'pattern' NO fue creada por engineer_features para {aggregator_key}.")
            else:
                 logger.debug(f"Columna 'pattern' creada por engineer_features para {aggregator_key}.")

            start_time_pre = time.time()
            self.make_prediction(aggregator_key, latest_features)
            end_time_pre = time.time()
            duration_pre = end_time_pre - start_time_pre
            logger.info(f"    -> Tiempo Predicción ({aggregator_key}): {duration_pre:.4f} segundos")

        except Exception as e:
            logger.error(f"ERROR durante el procesamiento para predicción de {aggregator_key}: {e}", exc_info=True)


    def make_prediction(self, aggregator_key: str, latest_features: pd.DataFrame):
        """Usa los modelos cargados para predecir sobre las últimas características."""
        models = self.models.get(aggregator_key)
        if not models:
            logger.error(f"Modelos no encontrados para {aggregator_key} en el paso de predicción.")
            return

        pipeline = models.get('pipeline')

        if not pipeline: # El pipeline es esencial
             logger.error(f"Pipeline de RF no cargado para {aggregator_key}.")
             return

        try:
            prediction_features = latest_features.copy()

            # Asegurarse de que todas las columnas que espera el pipeline están presentes ANTES de predecir
            final_required_cols = pipeline.feature_names_in_
            missing_final_cols = [col for col in final_required_cols if col not in prediction_features.columns]

            if missing_final_cols:
                 logger.error(f"Faltan columnas finales para la predicción del pipeline en {aggregator_key}: {missing_final_cols}")
                 return

            # Seleccionar solo las columnas que el pipeline espera en el orden correcto
            logger.debug(f"Columnas finales para predicción ({aggregator_key}): {final_required_cols}")
            prediction_input = prediction_features[final_required_cols]

            prediction = pipeline.predict(prediction_input)
            prediction_proba = pipeline.predict_proba(prediction_input)

            bar_time_str = latest_features.index[0].strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"--- Predicción para {aggregator_key} @ {bar_time_str} ---")
            logger.info(f"    Predicción   : {prediction[0]}")
            if hasattr(pipeline, 'classes_'):
                classes = pipeline.classes_
                proba_dict = {f"Prob({c})": f"{p:.4f}" for c, p in zip(classes, prediction_proba[0])}
                logger.info(f"    Probabilidades: {proba_dict}")
            else:
                 logger.info(f"    Probabilidades: {prediction_proba[0]}")
            logger.info(f"---------------------------------------------------------")

        except Exception as e:
            logger.error(f"ERROR durante la predicción del modelo para {aggregator_key}: {e}", exc_info=True)


    # --- Saving Logic ---
    def check_and_save_aggregated_data(self, aggregator_key: str, completed_bar_start_time_utc: datetime):
        if self.SAVE_INTERVAL == 'hourly':
            current_period_key = completed_bar_start_time_utc.strftime('%Y%m%d_%H')
        elif self.SAVE_INTERVAL == 'daily':
            current_period_key = completed_bar_start_time_utc.strftime('%Y%m%d')
        else: # Default daily
            current_period_key = completed_bar_start_time_utc.strftime('%Y%m%d')

        last_period = self.last_save_period.get(aggregator_key)

        if last_period != current_period_key:
            bars_to_save = []
            previous_period_key = last_period

            with self._lock:
                if last_period is not None:
                    # Copiar solo si hay barras acumuladas para este periodo
                    if self.aggregated_bar_accumulators[aggregator_key]:
                        bars_to_save = self.aggregated_bar_accumulators[aggregator_key][:] # Copiar
                        self.aggregated_bar_accumulators[aggregator_key] = [] # Limpiar acumulador
                self.last_save_period[aggregator_key] = current_period_key

            if bars_to_save and previous_period_key:
                logger.info(f"Cambio de periodo detectado para {aggregator_key}. Guardando datos agregados para el periodo {previous_period_key}.")
                self.save_aggregated_data_batch(aggregator_key, bars_to_save, previous_period_key)
            elif last_period is None:
                 logger.info(f"Periodo de guardado inicializado para {aggregator_key} a {current_period_key}.")


    def save_aggregated_data_batch(self, aggregator_key: str, bars: List[FiveMinBarDict], period_key: str):
        """Guarda un lote de barras agregadas completas en un archivo CSV."""
        if not bars:
            return

        try:
            parts = aggregator_key.split('_')
            ticker = parts[0]
            # Unir el resto por si el candlesize_label tuviera '_'
            candlesize_label = '_'.join(parts[1:])

            df = pd.DataFrame(bars)
            # Asegurarse de que 'start_time_utc' existe antes de usarlo
            if 'start_time_utc' not in df.columns:
                 logger.error(f"Columna 'start_time_utc' no encontrada en el batch para {aggregator_key}, periodo {period_key}. No se puede guardar.")
                 return

            df['datetime_utc'] = pd.to_datetime(df['start_time_utc'], utc=True)
            df.set_index('datetime_utc', inplace=True)

            df = pd.DataFrame(bars)
            # Asegurarse de que 'start_time_utc' existe antes de usarlo
            if 'start_time_utc' not in df.columns:
                 logger.error(f"Columna 'start_time_utc' no encontrada en el batch para {aggregator_key}, periodo {period_key}. No se puede guardar.")
                 return

            df['datetime_utc'] = pd.to_datetime(df['start_time_utc'], utc=True)
            df.set_index('datetime_utc', inplace=True)
            df.drop(columns=['start_time_utc'], inplace=True)

            save_dir = make_save_path(ticker, candlesize_label)
            filename = f"{period_key}.csv"
            full_path = os.path.join(save_dir, filename)

            os.makedirs(save_dir, exist_ok=True)

            logger.info(f"Guardando {len(df)} barras agregadas en {full_path}...")
            columns_order = ['open', 'high', 'low', 'close', 'volume', 'barCount']
            # Añadir columnas faltantes con NaN para asegurar consistencia
            for col in columns_order:
                if col not in df.columns:
                    df[col] = np.nan

            # Reordenar y guardar
            df[columns_order].to_csv(full_path, index=True, index_label='datetime_utc', float_format='%.5f')
            logger.info(f"Datos agregados guardados exitosamente en {full_path}")

        except Exception as e:
            logger.error(f"ERROR guardando datos agregados para {aggregator_key} periodo {period_key}: {e}", exc_info=True)


    # --- Error Handling ---
    @iswrapper
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """Maneja errores de la API."""
        super().error(reqId, errorCode, errorString, advancedOrderRejectJson)

        # Códigos que indican una pérdida de conexión que podría requerir reconexión
        # 1100: Connectivity between IB and TWS has been lost.
        # 1101: Connectivity between IB and TWS has been restored- data lost.
        # 1102: Connectivity between IB and TWS has been restored- data maintained. (Puede no requerir acción si los datos se mantienen)
        # 2103: A market data farm connection is broken
        # 2105: A historical data farm connection is broken
        # 2110: Connectivity between TWS and server is broken. Backfill data is being downloaded. (TWS intenta recuperar, pero la API podría necesitar reconectar)
        # 504: Not connected
        # 502: Couldn't connect to TWS. Confirm that "Enable ActiveX and Socket Clients" is enabled... (Error de conexión inicial)
        critical_error_codes = [1100, 1101, 2103, 2105, 2110, 504, 502]

        if errorCode in critical_error_codes:
             logger.error(f"ERROR CRÍTICO API (posible desconexión): reqId={reqId}, Code={errorCode}, Msg={errorString}")
             # Marcar como pérdida inesperada solo si no estamos ya desconectando intencionalmente
             if not self._disconnect_called:
                 self.connection_lost_unexpectedly = True
             return # No procesar más como error genérico

        # Ignorar códigos de error informativos o comunes que no indican un problema real
        # 2104: Market data farm connection is OK
        # 2106: Historical market data farm connection is OK
        # 2158: Sec-def data farm connection is OK
        # 2108: Market data farm connection is inactive but should be available upon demand
        # 2119: Warning: Approaching max rate of 50 messages per second
        # 10167: Requested market data is not subscribed
        # 2109: Historical data farm connection is inactive but should be available upon demand
        if errorCode in [2104, 2106, 2158, 2108, 2119, 10167, 2109]:
             return

        request_info = self.requests.get(reqId, {})
        req_type = request_info.get('type', 'unknown')
        agg_key = request_info.get('aggregator_key', 'unknown')

        log_message = f"ERROR API: reqId={reqId} (Type: {req_type}, Key: {agg_key}), Code={errorCode}, Msg={errorString}"
        if advancedOrderRejectJson:
             log_message += f"\n       Advanced Order Reject: {advancedOrderRejectJson}"

        logger.error(log_message)

        # Manejo específico para errores de datos históricos que podrían bloquear el inicio
        if req_type == 'historical':
            if errorCode in [162, 200, 321, 322, 10167, 10285]:
                 logger.warning(f"       -> Marcando histórico para {agg_key} como fallido y recibido (para desbloquear espera) debido al error {errorCode}.")
                 if agg_key in self.historical_data_received:
                     self.historical_data_received[agg_key].set()
                 else:
                     logger.warning(f"       -> No se pudo encontrar el evento para {agg_key} para marcar como recibido.")


    # --- Control Methods ---
    def websocket_con(self):
        logger.info("Iniciando bucle de eventos de EClient...")
        # Resetear flag al iniciar el bucle
        self.connection_lost_unexpectedly = False
        self._disconnect_called = False
        try:
            self.run()
        except Exception as e:
             logger.error(f"Excepción dentro del bucle EClient.run(): {e}", exc_info=True)
             if not self._disconnect_called:
                 self.connection_lost_unexpectedly = True
        finally:
            logger.info("Bucle de eventos de EClient finalizado.")
            if not self._disconnect_called and self.isConnected():
                 logger.warning("EClient.run() terminó pero aún estábamos conectados. Considerado desconexión inesperada.")
                 self.connection_lost_unexpectedly = True
            elif not self.isConnected() and not self._disconnect_called:
                 logger.warning("EClient.run() terminó y no estábamos conectados. Considerado desconexión inesperada.")
                 self.connection_lost_unexpectedly = True


    def disconnect_from_ib(self):
        logger.info("Desconectando de IBKR...")
        self._disconnect_called = True # Marcar desconexión intencional
        self.connection_lost_unexpectedly = False # Asegurar que no se marque como inesperada

        logger.info("Guardando barras agregadas restantes antes de salir...")
        with self._lock:
            for key, current_bar in list(self.current_5min_bars.items()):
                if current_bar:
                    logger.info(f"Añadiendo última barra parcial para {key} @ {current_bar['start_time_utc']}")
                    if all(k in current_bar for k in ['start_time_utc', 'open', 'high', 'low', 'close', 'volume', 'barCount']):
                        self.aggregated_bar_accumulators[key].append(current_bar)
                    else:
                        logger.warning(f"Barra parcial para {key} incompleta, no se añadirá al acumulador final: {current_bar}")

            for key, bars in list(self.aggregated_bar_accumulators.items()):
                 if bars:
                      last_bar_time = bars[-1]['start_time_utc']
                      last_period_key = ""
                      if self.SAVE_INTERVAL == 'hourly':
                          last_period_key = last_bar_time.strftime('%Y%m%d_%H')
                      else:
                          last_period_key = last_bar_time.strftime('%Y%m%d')

                      logger.info(f"Guardando lote final para {key}, periodo aprox. {last_period_key}")
                      self.save_aggregated_data_batch(key, bars, f"{last_period_key}_final")

            logger.info("Limpiando estructuras de datos internas...")
            self.historical_data.clear()
            self.current_5min_bars.clear()
            self.aggregated_bar_accumulators.clear()
            self.models.clear()
            self.requests.clear()
            self.last_save_period.clear()
            for event in self.historical_data_received.values():
                event.set()
            self.historical_data_received.clear()

        if self.isConnected():
            self.disconnect()
            logger.info("Desconexión de EClient solicitada.")
        else:
            logger.info("Ya estaba desconectado.")
        logger.info("Proceso de desconexión finalizado.")


# --- Example Usage ---
if __name__ == "__main__":
    eurusd_contract = Contract()
    eurusd_contract.symbol = "EUR"  # Base currency
    eurusd_contract.secType = "CASH"
    eurusd_contract.currency = "USD"  # Quote currency
    eurusd_contract.exchange = "IDEALPRO"

    # usdjpy_contract = Contract()
    # usdjpy_contract.symbol = "USD"  # Base currency
    # usdjpy_contract.secType = "CASH"
    # usdjpy_contract.currency = "JPY"  # Quote currency
    # usdjpy_contract.exchange = "IDEALPRO"

    contracts_to_stream = {
        eurusd_contract: [
            {'candlesize_label': '5 mins'} # Label for the 5-min aggregated data
        ]
    }

    app = None # Inicializar app a None
    api_thread = None
    try:
        app = RealTimeAggregatorSaver(contracts=contracts_to_stream) # Crear instancia aquí

        logger.info("Intentando conectar a IBKR TWS/Gateway...")
        ib_host = '127.0.0.1'
        ib_port = 7497
        client_id = 27 # Elige un ID de cliente único
        app.connect(host=ib_host, port=ib_port, clientId=client_id)

        time.sleep(3)

        if not app.isConnected():
             raise ConnectionError(f"No se pudo establecer conexión con IBKR en {ib_host}:{ib_port} con clientId {client_id}")

        logger.info("Iniciando hilo de la API...")
        api_thread = threading.Thread(target=app.websocket_con, daemon=True)
        api_thread.start()

        while api_thread.is_alive():
            time.sleep(1)

        logger.info("El hilo de la API ha terminado.")

    except KeyboardInterrupt:
        logger.info("\nInterrupción de teclado recibida. Desconectando...")
    except ConnectionRefusedError:
         logger.error(f"\nERROR: No se pudo conectar a {ib_host}:{ib_port}. ¿Está TWS/Gateway ejecutándose y la API habilitada?")
    except ConnectionError as ce:
         logger.error(f"\nERROR de conexión: {ce}")
    except Exception as e:
        logger.error(f"\nERROR inesperado en el bloque principal: {e}", exc_info=True)
    finally:
        logger.info("Iniciando secuencia de finalización...")
        if app is not None:
             if app.isConnected():
                  logger.info("Llamando a disconnect_from_ib()...")
                  app.disconnect_from_ib()
                  if api_thread and api_thread.is_alive():
                       logger.info("Esperando a que el hilo de la API finalice...")
                       api_thread.join(timeout=10)
                       if api_thread.is_alive():
                            logger.warning("ADVERTENCIA: El hilo de la API no finalizó limpiamente después de 10 segundos.")
             else:
                  logger.info("La aplicación no estaba conectada o ya se desconectó antes del bloque finally.")
        else:
             logger.info("El objeto de la aplicación (app) no fue creado debido a un error anterior.")

        logger.info("Script finalizado.")