import os
import sys
import time
import argparse
import logging
from typing import Dict, List, Union
from datetime import datetime
from collections import defaultdict

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.utils import iswrapper
from ibapi.contract import Contract
from ibapi.common import BarData

from dateutil import parser
import pytz
import pandas as pd

from pathlib import Path

# Ensure project root directory is in sys.path
root_dir = str(Path(__file__).resolve().parents[2])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.utils.utils import stockContract, fxPair, futureContract

BarDataList = List[BarData]
ContractRequestDetails = List[Dict[str, Union[str, int, datetime]]]
ContractDict = Dict[Contract, ContractRequestDetails]


def get_ticket_from_contract(contract: Contract) -> str:
    """
    Extracts a clean filesystem ticket/identifier from an IBKR Contract.
    - Stocks/ETFs (STK): e.g. 'SPY', 'QQQ', 'AAPL'
    - Futures (FUT): e.g. 'MES', 'MNQ'
    - Forex (CASH): e.g. 'EURUSD', 'GBPUSD'
    """
    sec_type = getattr(contract, "secType", "").upper()
    symbol = getattr(contract, "symbol", "").upper()
    currency = getattr(contract, "currency", "").upper()

    if sec_type in ["STK", "FUT", "IND"]:
        return symbol
    elif sec_type == "CASH":
        return f"{symbol}{currency}"
    return symbol or "UNKNOWN"


def make_download_path(ticket: str, candlesize: str, base_dir: str = "data") -> str:
    """Make path for saving csv files: base_dir/<ticket>/<clean_candlesize>/"""
    clean_size = candlesize.replace(" ", "")
    return os.path.join(base_dir, ticket, clean_size)


def parse_bar_date(date_val: Union[str, datetime]) -> datetime:
    """
    Safely parses bar dates returned by IBKR:
    - Daily bars: 'YYYYMMDD' (e.g. '20240115')
    - Intraday bars: 'YYYYMMDD HH:MM:SS' or 'YYYYMMDD HH:MM:SS Timezone'
    - Head timestamp: 'YYYYMMDD-HH:MM:SS'
    """
    if isinstance(date_val, datetime):
        return date_val

    date_str = str(date_val).strip()
    if not date_str:
        return datetime.utcnow()

    # 1. Daily bar format: 'YYYYMMDD'
    if len(date_str) == 8 and date_str.isdigit():
        return datetime.strptime(date_str, "%Y%m%d")

    # 2. IBKR headTimestamp format: 'YYYYMMDD-HH:MM:SS'
    if "-" in date_str and len(date_str.split("-")) == 2:
        d_part, t_part = date_str.split("-")
        if len(d_part) == 8 and len(t_part) == 8:
            return datetime.strptime(date_str, "%Y%m%d-%H:%M:%S")

    # 3. Intraday bar format: 'YYYYMMDD HH:MM:SS [Timezone]'
    parts = date_str.split()
    if len(parts) >= 2:
        dt_str = f"{parts[0]} {parts[1]}"
        try:
            return datetime.strptime(dt_str, "%Y%m%d %H:%M:%S")
        except ValueError:
            pass

    # 4. Fallback to general parser
    return parser.parse(date_str)


class MarketDataDownloader(EClient, EWrapper):
    """
    Robust multi-asset historical data downloader for Interactive Brokers.
    Supports Stocks/ETFs (STK), Futures (FUT), and Forex (CASH).
    """

    timezone = pytz.timezone("America/New_York")

    FORMAT_DATE = 1
    DEFAULT_CHUNK_SIZE = "1 M"
    DEFAULT_START_DATE = datetime(2020, 1, 1)

    def __init__(self, contracts: ContractDict, pacing_interval: float = 1.0):
        EClient.__init__(self, self)
        self.next_valid_order_id = None
        self.started = False
        self.request_id = 0
        self.contracts = contracts
        self.requests = {}
        self.request_details = {}
        self.bar_data = defaultdict(lambda: defaultdict(list))
        self.pacing_interval = pacing_interval
        self.active_requests_count = 0

    def _get_default_what_to_show(self, contract: Contract) -> str:
        """Determines default data stream type based on asset class."""
        sec_type = getattr(contract, "secType", "CASH").upper()
        if sec_type in ["STK", "FUT", "IND"]:
            return "TRADES"  # Actual traded volume and prices for equities & futures
        return "MIDPOINT"  # Standard for OTC Forex

    def _get_default_use_rth(self, contract: Contract) -> int:
        """
        Determines whether to filter to Regular Trading Hours (useRTH).
        - For STK: 1 (09:30 - 16:00 EST) is standard to exclude illiquid overnight noise.
        - For CASH / FUT: 0 (24h continuous).
        """
        sec_type = getattr(contract, "secType", "CASH").upper()
        if sec_type == "STK":
            return 1
        return 0

    @iswrapper
    def connectAck(self):
        logging.info("Connected to IBKR.")
        print("Connected to IBKR.")

    @iswrapper
    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        """IBKR Error callback with intelligent log filtering."""
        # Informational codes (not real errors)
        info_codes = {2104, 2106, 2158, 2108, 2174}
        if errorCode in info_codes:
            logging.debug(f"[IBKR INFO {errorCode}] reqId {reqId}: {errorString}")
            return

        # Pacing violation warning
        if errorCode == 162:
            print(f"[IBKR PACING WARNING {errorCode}] reqId {reqId}: {errorString}. Consider increasing pacing_interval.")
            logging.warning(f"[IBKR PACING WARNING {errorCode}] reqId {reqId}: {errorString}")
            return

        print(f"[IBKR ERROR {errorCode}] reqId {reqId}: {errorString}")
        logging.error(f"[IBKR ERROR {errorCode}] reqId {reqId}: {errorString}")

    @iswrapper
    def nextValidId(self, order_id: int):
        super().nextValidId(order_id)
        self.next_valid_order_id = order_id
        logging.info(f"nextValidId: {order_id}")
        self.start()

    def next_request_id(self, contract: Contract = None) -> int:
        self.request_id += 1
        return self.request_id

    def reqHeadTimeStampForContract(self, contract: Contract, request_details: Dict[str, Union[str, int]]):
        """Request the earliest available timestamp for a contract."""
        reqId = self.next_request_id(contract)
        self.requests[reqId] = {"contract": contract, **request_details}

        what_to_show = request_details.get("whatToShow", self._get_default_what_to_show(contract))
        use_rth = request_details.get("useRTH", self._get_default_use_rth(contract))

        self.reqHeadTimeStamp(
            reqId,
            contract,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=self.FORMAT_DATE,
        )

    def start(self):
        if self.started:
            return

        self.started = True
        print(f"Starting extraction for {len(self.contracts)} contract(s)...")

        for contract, requests in self.contracts.items():
            for request in requests:
                self.active_requests_count += 1
                self.historicalDataRequest({"contract": contract, **request})

    @iswrapper
    def headTimestamp(self, reqId: int, headTimestamp: str) -> None:
        contract_details = self.requests.get(reqId)
        if not contract_details:
            return

        ts = parse_bar_date(headTimestamp)
        ticket = get_ticket_from_contract(contract_details["contract"])
        candlesize = contract_details.get("candlesize", "1 hour")
        print(f"Earliest data available in IBKR for {ticket} ({candlesize}): {ts}")

        self.historicalDataRequest(contract_details)

    def historicalDataRequest(self, contract_details: Dict) -> None:
        contract = contract_details["contract"]
        cid = self.next_request_id(contract)
        self.request_details[cid] = contract_details

        what_to_show = contract_details.get("whatToShow", self._get_default_what_to_show(contract))
        use_rth = contract_details.get("useRTH", self._get_default_use_rth(contract))
        chunk_size = contract_details.get("duration", self.DEFAULT_CHUNK_SIZE)
        candlesize = contract_details.get("candlesize", "1 hour")

        self.reqHistoricalData(
            cid,
            contract,
            "",  # empty string requests up to current/latest time
            chunk_size,
            candlesize,
            what_to_show,
            use_rth,
            self.FORMAT_DATE,
            False,
            [],
        )

    @iswrapper
    def historicalData(self, reqId: int, bar) -> None:
        contract_details = self.request_details.get(reqId)
        if not contract_details:
            return

        contract = contract_details["contract"]
        ticket = get_ticket_from_contract(contract)
        candlesize = contract_details.get("candlesize", "1 hour")

        date = parse_bar_date(bar.date)
        month_key = date.strftime("%Y%m")

        self.bar_data[f"{ticket}_{candlesize}"][month_key].append(bar)

    @iswrapper
    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        super().historicalDataEnd(reqId, start, end)

        contract_details = self.request_details.get(reqId)
        if not contract_details:
            return

        contract = contract_details["contract"]
        ticket = get_ticket_from_contract(contract)
        candlesize = contract_details.get("candlesize", "1 hour")

        start_dt = parse_bar_date(start)
        end_dt = parse_bar_date(end)
        month_key_end = end_dt.strftime("%Y%m")

        # Save buffer for the month that just completed
        bars_to_save = self.bar_data[f"{ticket}_{candlesize}"].get(month_key_end, [])
        if bars_to_save:
            self.save_data(contract_details, bars_to_save, month_key_end)
            del self.bar_data[f"{ticket}_{candlesize}"][month_key_end]

        start_cutoff = contract_details.get("startDate", self.DEFAULT_START_DATE)
        if isinstance(start_cutoff, str):
            start_cutoff = parse_bar_date(start_cutoff)

        if start_dt <= start_cutoff:
            print(f"Finished downloading data for {ticket} {candlesize}. Reached start cutoff: {start_cutoff.date()}.")
            # Save any remaining monthly buffers
            remaining_keys = list(self.bar_data[f"{ticket}_{candlesize}"].keys())
            for m_key in remaining_keys:
                if self.bar_data[f"{ticket}_{candlesize}"][m_key]:
                    self.save_data(contract_details, self.bar_data[f"{ticket}_{candlesize}"][m_key], m_key)
                    del self.bar_data[f"{ticket}_{candlesize}"][m_key]

            self.active_requests_count -= 1
            if self.active_requests_count <= 0:
                print("All download tasks completed successfully.")
                self.disconnect()
        else:
            new_end_date = start_dt.strftime("%Y%m%d %H:%M:%S")
            print(f"Requesting chunk up to {new_end_date} for {ticket} {candlesize} (Pacing pause {self.pacing_interval}s)...")
            time.sleep(self.pacing_interval)

            what_to_show = contract_details.get("whatToShow", self._get_default_what_to_show(contract))
            use_rth = contract_details.get("useRTH", self._get_default_use_rth(contract))
            chunk_size = contract_details.get("duration", self.DEFAULT_CHUNK_SIZE)

            new_cid = self.next_request_id(contract)
            self.request_details[new_cid] = contract_details
            self.reqHistoricalData(
                new_cid,
                contract,
                new_end_date,
                chunk_size,
                candlesize,
                what_to_show,
                use_rth,
                self.FORMAT_DATE,
                False,
                [],
            )

    def save_data(self, contract_details: Dict, bars: BarDataList, month_key_end: str) -> None:
        if not bars:
            return

        data = [
            [b.date, b.open, b.high, b.low, b.close, b.volume, b.barCount]
            for b in bars
        ]

        df = pd.DataFrame(
            data,
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "barCount",
            ],
        )

        contract = contract_details["contract"]
        ticket = get_ticket_from_contract(contract)
        candlesize = contract_details.get("candlesize", "1 hour")

        folder_path = make_download_path(ticket, candlesize)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{month_key_end}.csv")

        # If file already exists, merge and remove duplicate timestamps
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["date"]).sort_values(by="date")

        df.to_csv(file_path, index=False)
        print(f"[{ticket} {candlesize}] Saved {len(df)} bars to {file_path}")

    def websocket_con(self):
        self.run()

    def disconnect_from_ib(self):
        self.disconnect()


# Backwards compatibility alias
ForexDataExtractor = MarketDataDownloader


def download_historical_data(
    contracts_config: ContractDict,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
    pacing_interval: float = 1.0,
):
    """
    Convenience function to initiate historical downloads sequentially for any set of contracts.
    Downloading sequentially guarantees zero pacing collisions with IBKR.
    """
    total_contracts = len(contracts_config)
    for idx, (contract, req_list) in enumerate(contracts_config.items(), 1):
        ticket = get_ticket_from_contract(contract)
        print(f"\n{'='*50}")
        print(f"[{idx}/{total_contracts}] >>> Starting download for {ticket}...")
        print(f"{'='*50}")
        single_config = {contract: req_list}
        app = MarketDataDownloader(single_config, pacing_interval=pacing_interval)
        app.connect(host, port, clientId=client_id)
        try:
            app.websocket_con()
        except KeyboardInterrupt:
            print(f"\nDownload cancelled by user at {ticket}.")
            break
        finally:
            app.disconnect_from_ib()
        time.sleep(pacing_interval)
    print("\nAll download tasks completed successfully!")


if __name__ == "__main__":
    parser_cli = argparse.ArgumentParser(description="Download Historical Market Data from Interactive Brokers.")
    parser_cli.add_argument("--symbols", type=str, default="SPY,QQQ", help="Comma-separated ticker symbols (e.g. SPY,QQQ,AAPL)")
    parser_cli.add_argument("--secType", type=str, default="STK", choices=["STK", "CASH", "FUT"], help="Security type: STK, CASH, or FUT")
    parser_cli.add_argument("--candlesize", type=str, default="1 hour", help="Bar size (e.g. '1 hour', '1 day', '15 mins', '5 mins')")
    parser_cli.add_argument("--duration", type=str, default="1 M", help="Chunk duration per IBKR request (e.g. '1 M', '1 Y')")
    parser_cli.add_argument("--start", type=str, default="2022-01-01", help="Historical start cutoff date (YYYY-MM-DD)")
    parser_cli.add_argument("--host", type=str, default="127.0.0.1", help="IBKR TWS/Gateway IP host")
    parser_cli.add_argument("--port", type=int, default=7497, help="IBKR port (7497 paper, 7496 live, 4002 gateway paper, 4001 gateway live)")
    parser_cli.add_argument("--client-id", type=int, default=1, help="IBKR client ID")
    parser_cli.add_argument("--pacing", type=float, default=1.0, help="Pacing pause in seconds between requests")

    args = parser_cli.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start_date = datetime.strptime(args.start, "%Y-%m-%d")

    contracts_dict: ContractDict = {}

    for sym in symbols:
        if args.secType == "STK":
            contract = stockContract(sym)
        elif args.secType == "CASH":
            contract = fxPair(sym)
        elif args.secType == "FUT":
            contract = futureContract(sym)
        else:
            raise ValueError(f"Unsupported secType: {args.secType}")

        contracts_dict[contract] = [
            {
                "candlesize": args.candlesize,
                "duration": args.duration,
                "startDate": start_date,
            }
        ]

    print(f"Initiating download for: {symbols} (Type: {args.secType}, Bar: {args.candlesize}, From: {args.start})")
    download_historical_data(
        contracts_dict,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        pacing_interval=args.pacing,
    )