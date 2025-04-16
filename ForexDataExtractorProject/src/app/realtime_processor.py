import os
import sys
import threading
import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from collections import defaultdict
import math # For floor function

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.utils import iswrapper
from ibapi.contract import Contract
from ibapi.common import BarData # Import BarData

import pytz
import pandas as pd

# Define types for clarity
FiveMinBarDict = Dict[str, Optional[Union[float, int, datetime]]] # Type for the aggregated bar
ContractRequestDetails = List[Dict[str, str]] # e.g., [{'candlesize_label': '5 mins'}]
ContractDict = Dict[Contract, ContractRequestDetails]

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
    # --- End Configuration ---

    def __init__(self, contracts: ContractDict):
        EClient.__init__(self, self)
        self.next_valid_order_id = None
        self.started = False
        self.request_id_counter = 0
        self.contracts_config = contracts
        self.requests = {} # Map reqId back to contract and config details

        # Data storage for aggregated bars
        self.aggregated_bar_accumulators = defaultdict(list) # Stores completed 5-min bars { "TICKER_candlesizeLabel": [FiveMinBarDict] }
        self.current_5min_bars = defaultdict(lambda: None) # Stores the currently forming 5-min bar { "TICKER_candlesizeLabel": FiveMinBarDict }
        self.last_save_period = defaultdict(str) # Track last save period key for aggregated data

        self._lock = threading.Lock() # Lock for thread safety

    # --- Connection Handling (same as RealTimeDataSaver) ---
    @iswrapper
    def connectAck(self):
        print("Connected to IBKR.")

    @iswrapper
    def nextValidId(self, order_id: int):
        super().nextValidId(order_id)
        self.next_valid_order_id = order_id
        print(f"nextValidId: {order_id}")
        self.start_realtime_requests()

    def next_request_id(self) -> int:
        self.request_id_counter += 1
        return self.request_id_counter

    # --- Request Initiation (same as RealTimeDataSaver) ---
    def start_realtime_requests(self):
        if self.started:
            return
        self.started = True
        print("Starting real-time data requests (for 5-second bars)...")

        for contract, request_details_list in self.contracts_config.items():
            for details in request_details_list:
                # 'candlesize_label' should now reflect the target aggregation (e.g., '5 mins')
                candlesize_label = details.get('candlesize_label', f'{self.AGGREGATION_MINUTES} mins')

                reqId = self.next_request_id()
                self.requests[reqId] = {'contract': contract, 'candlesize_label': candlesize_label}

                print(f"Requesting real-time bars for {contract.symbol}{contract.currency} (Label: {candlesize_label}) with reqId {reqId}")
                self.reqRealTimeBars(
                    reqId=reqId,
                    contract=contract,
                    barSize=self.BAR_SIZE_SECONDS, # Still request 5-second bars
                    whatToShow=self.WHAT_TO_SHOW,
                    useRTH=self.USERTH,
                    realTimeBarsOptions=[]
                )

    # --- Data Handling Callbacks ---
    @iswrapper
    def realtimeBar(self, reqId: int, time: int, open_: float, high: float, low: float, close: float, volume: int, wap: float, count: int):
        """Callback receiving 5-second bars and aggregating them into 5-minute bars."""
        super().realtimeBar(reqId, time, open_, high, low, close, volume, wap, count)

        request_info = self.requests.get(reqId)
        if not request_info:
            print(f"Warning: Received realtimeBar for unknown reqId: {reqId}")
            return

        contract = request_info['contract']
        candlesize_label = request_info['candlesize_label']
        ticker = f"{contract.symbol}{contract.currency}"
        aggregator_key = f"{ticker}_{candlesize_label}"

        # Convert IB timestamp (seconds since epoch UTC) to datetime object (UTC)
        bar_5s_time_utc = datetime.fromtimestamp(time, tz=pytz.utc)
        print(f"-> Received 5s bar for {ticker} at {bar_5s_time_utc.strftime('%Y-%m-%d %H:%M:%S')} C:{close:.5f}")
        # Determine the start time of the 5-minute interval this bar belongs to
        aggregation_seconds = self.AGGREGATION_MINUTES * 60
        interval_start_timestamp = math.floor(time / aggregation_seconds) * aggregation_seconds
        interval_start_time_utc = datetime.fromtimestamp(interval_start_timestamp, tz=pytz.utc)

        completed_bar_to_process: Optional[FiveMinBarDict] = None

        with self._lock:
            current_agg_bar = self.current_5min_bars[aggregator_key]

            # --- Check if this 5s bar starts a NEW 5-minute interval ---
            if current_agg_bar is None or current_agg_bar['start_time_utc'] != interval_start_time_utc:

                # 1. Finalize the PREVIOUS bar (if it exists)
                if current_agg_bar is not None:
                    # --- MODIFIED PRINT STATEMENT ---
                    print(f"Completed 5-min bar for {aggregator_key} ending {current_agg_bar['start_time_utc'].strftime('%Y-%m-%d %H:%M:%S')} "
                          f"O:{current_agg_bar['open']:.5f} H:{current_agg_bar['high']:.5f} L:{current_agg_bar['low']:.5f} C:{current_agg_bar['close']:.5f} V:{current_agg_bar['volume']}")
                    # --- END MODIFICATION ---

                    # Add the completed bar to the accumulator for saving
                    self.aggregated_bar_accumulators[aggregator_key].append(current_agg_bar)
                    # Store it to be processed for saving outside the lock
                    completed_bar_to_process = current_agg_bar

                # 2. Start the NEW bar
                self.current_5min_bars[aggregator_key] = {
                    "start_time_utc": interval_start_time_utc,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "barCount": count # Initialize bar count
                }
                # Optional: Print when a new bar starts
                # print(f"Starting new 5-min bar for {aggregator_key} at {interval_start_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")

            # --- Else: Update the CURRENT 5-minute interval ---
            else:
                # Ensure values exist before using max/min (should always exist here, but good practice)
                if current_agg_bar.get('high') is not None:
                    current_agg_bar['high'] = max(current_agg_bar['high'], high)
                else:
                    current_agg_bar['high'] = high

                if current_agg_bar.get('low') is not None:
                    current_agg_bar['low'] = min(current_agg_bar['low'], low)
                else:
                    current_agg_bar['low'] = low

                current_agg_bar['close'] = close # Last close becomes the interval's close
                current_agg_bar['volume'] = current_agg_bar.get('volume', 0) + volume
                current_agg_bar['barCount'] = current_agg_bar.get('barCount', 0) + count


        # --- Process saving for the completed bar (outside the lock) ---
        if completed_bar_to_process:
            # Use the START time of the completed bar for period checking
            self.check_and_save_aggregated_data(aggregator_key, completed_bar_to_process['start_time_utc'])


    # --- Saving Logic (Adapted for Aggregated Bars) ---
    def check_and_save_aggregated_data(self, aggregator_key: str, completed_bar_start_time_utc: datetime):
        """Checks if the completed bar belongs to a new save period and triggers saving."""
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
                    # Get the *aggregated* bars accumulated for the previous period
                    bars_to_save = self.aggregated_bar_accumulators[aggregator_key][:] # Copy
                    self.aggregated_bar_accumulators[aggregator_key] = [] # Clear accumulator
                self.last_save_period[aggregator_key] = current_period_key

            if bars_to_save and previous_period_key:
                print(f"Change detected for {aggregator_key}. Saving aggregated data for period {previous_period_key}.")
                self.save_aggregated_data_batch(aggregator_key, bars_to_save, previous_period_key)
            elif last_period is None:
                 print(f"Initialized save period for {aggregator_key} to {current_period_key}.")


    def save_aggregated_data_batch(self, aggregator_key: str, bars: List[FiveMinBarDict], period_key: str):
        """Saves a batch of completed aggregated (5-minute) bar data to a CSV file."""
        if not bars:
            print(f"No aggregated bars to save for {aggregator_key} period {period_key}.")
            return

        try:
            parts = aggregator_key.split('_')
            ticker = parts[0]
            candlesize_label = '_'.join(parts[1:])

            df = pd.DataFrame(bars)
            # Rename 'start_time_utc' to 'datetime' for consistency if desired
            df.rename(columns={'start_time_utc': 'datetime_utc'}, inplace=True)
            # Format datetime for CSV
            df['datetime_utc'] = df['datetime_utc'].dt.strftime('%Y-%m-%d %H:%M:%S')

            save_dir = make_save_path(ticker, candlesize_label)
            filename = f"{period_key}.csv" # e.g., "20231027_5min.csv" or "20231027_15_5min.csv"
            full_path = os.path.join(save_dir, filename)

            os.makedirs(save_dir, exist_ok=True)

            print(f"Saving {len(bars)} aggregated bars to {full_path}...")
            # Select columns in desired order for CSV
            columns_order = ['datetime_utc', 'open', 'high', 'low', 'close', 'volume', 'barCount']
            # Ensure all columns exist before selecting, fill missing with NaN if necessary
            for col in columns_order:
                if col not in df.columns:
                    df[col] = pd.NA # Or appropriate default like 0 for volume/barCount
            df[columns_order].to_csv(full_path, index=False)
            print(f"Successfully saved aggregated data to {full_path}")

        except Exception as e:
            print(f"Error saving aggregated data for {aggregator_key} period {period_key}: {e}")

    # --- Error Handling (same as RealTimeDataSaver) ---
    @iswrapper
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        super().error(reqId, errorCode, errorString, advancedOrderRejectJson)
        print(f"Error: reqId={reqId}, Code={errorCode}, Msg={errorString}")
        if advancedOrderRejectJson:
             print(f"Advanced Order Reject: {advancedOrderRejectJson}")

    # --- Control Methods (same as RealTimeDataSaver) ---
    def websocket_con(self):
        print("Starting EClient event loop...")
        self.run()

    def disconnect_from_ib(self):
        print("Disconnecting from IBKR...")
        # --- Add logic to save any remaining data before disconnecting ---
        print("Saving any remaining aggregated bars before exit...")
        with self._lock:
            # Process the very last partially formed bar if it exists
            for key, current_bar in list(self.current_5min_bars.items()): # Use list for safe iteration
                if current_bar:
                    print(f"Adding final partially formed bar for {key} ending {current_bar['start_time_utc']}")
                    self.aggregated_bar_accumulators[key].append(current_bar)
                    del self.current_5min_bars[key] # Clear it after adding

            # Trigger save for all accumulators
            for key, bars in list(self.aggregated_bar_accumulators.items()): # Use list for safe iteration
                 if bars:
                      # Determine the period key for the last bar in the list
                      last_bar_time = bars[-1]['start_time_utc']
                      last_period = self.last_save_period.get(key, last_bar_time.strftime('%Y%m%d_%H' if self.SAVE_INTERVAL == 'hourly' else '%Y%m%d'))
                      self.save_aggregated_data_batch(key, bars, f"{last_period}_final") # Add suffix
                      del self.aggregated_bar_accumulators[key] # Clear after saving attempt

        # --- End save logic ---
        # Now disconnect
        if self.isConnected():
            self.disconnect()
        print("Disconnected.")


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create and configure Contract objects first
    eurusd_contract = Contract()
    eurusd_contract.symbol = "EUR"  # Base currency
    eurusd_contract.secType = "CASH"
    eurusd_contract.currency = "USD"  # Quote currency
    eurusd_contract.exchange = "IDEALPRO"

    usdjpy_contract = Contract()
    usdjpy_contract.symbol = "USD"  # Base currency
    usdjpy_contract.secType = "CASH"
    usdjpy_contract.currency = "JPY"  # Quote currency
    usdjpy_contract.exchange = "IDEALPRO"

    # 2. Define the stream configuration using the configured contracts as keys
    contracts_to_stream = {
        eurusd_contract: [
            {'candlesize_label': '5 mins'} # Label for the 5-min aggregated data
        ],
        usdjpy_contract: [
            {'candlesize_label': '5 mins'}
        ]
    }

    # 3. Create the application instance
    app = RealTimeAggregatorSaver(contracts=contracts_to_stream)

    # --- Connection and Execution Logic (remains the same) ---
    try:
        # Ensure TWS/Gateway is running and API connection is enabled
        print("Attempting to connect to IBKR...")
        app.connect(host='127.0.0.1', port=7497, clientId=26) # Use a unique clientId

        # Check connection status briefly after connect returns (optional)
        # time.sleep(2) # Give a moment for connection callbacks
        # if not app.isConnected():
        #     print("Connection failed. Please check TWS/Gateway and API settings.")
        #     sys.exit(1) # Exit if connection failed immediately

        print("Starting API thread...")
        api_thread = threading.Thread(target=app.websocket_con, daemon=True)
        api_thread.start()

        print("Main thread waiting for API thread to finish (or KeyboardInterrupt)...")
        while api_thread.is_alive():
            time.sleep(1)
        print("API thread finished.") # Should only print if disconnected properly or error in thread

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Disconnecting...")
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")
    finally:
        print("Ensuring disconnection...")
        # Ensure disconnect is called even if connection failed initially
        if 'app' in locals() and app.isConnected():
             app.disconnect_from_ib()
        elif 'app' in locals():
             print("App was not connected or already disconnected.")
        else:
             print("App object not created.")
        print("Exiting script.")