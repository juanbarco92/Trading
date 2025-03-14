import os
import sys

from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.utils import iswrapper
from ibapi.contract import Contract
from ibapi.common import BarData

from dateutil import parser
import pytz
import pandas as pd

BarDataList = List[BarData]
ContractRequestDetails = List[Dict[str, str]]
ContractDict = Dict[Contract, ContractRequestDetails]

def make_download_path(ticket, candlesize) -> str:
    """Make path for saving csv files.
    Files to be stored in base_directory/<security_type>/<size>/<symbol>/
    """
    path = os.path.sep.join(
        [
            "data",
            ticket,
            candlesize.replace(" ", "")
        ]
    )
    return path

class ForexDataExtractor(EClient, EWrapper):

    timezone = pytz.timezone("America/New_York")  # Example: 

    WHAT_TO_SHOW = "MIDPOINT"
    USERTH = 0  
    FORMAT_DATE = 1

    START_DATE = datetime(2010, 1, 1)
    CHUNK_SIZE = '1 M' 

    def __init__(self, contracts: ContractDict):
        EClient.__init__(self, self)
        self.next_valid_order_id = None
        self.started = False
        self.request_id = 0
        self.contracts = contracts
        self.requests = {}
        self.request_details = {}
        self.bar_data = defaultdict(lambda: defaultdict(list))
        self.previous_month_key = None
        
        
    #Callback triggered when connection successful
    @iswrapper
    def connectAck(self):
        print("Connected")

    #Callback triggered when run() inovked
    @iswrapper
    def nextValidId(self, order_id: int):
        super().nextValidId(order_id)

        self.next_valid_order_id = order_id
        print(f"nextValidId: {order_id}")
        # we can start now
        self.start()

    def next_request_id(self, contract: Contract) -> int:
        self.request_id += 1
        
        return self.request_id
    
    
    def reqHeadTimeStampForContract(self, contract: Contract, request_details: Dict[str, str]):
        """Request the head timestamp for a contract with specific request details."""
        reqId = self.next_request_id(contract)
        self.requests[reqId] = {'contract': contract, **request_details}
        
        self.reqHeadTimeStamp(
            reqId, 
            contract, 
            whatToShow=self.WHAT_TO_SHOW, 
            useRTH=self.USERTH, 
            formatDate=self.FORMAT_DATE
        )

        

    def start(self):
        if self.started:
            return

        self.started = True
        print(self.started)

        for contract, requests in self.contracts.items():
            for request in requests:
                self.reqHeadTimeStampForContract(contract, request)

    
    @iswrapper
    def headTimestamp(self, reqId: int, headTimestamp: str) -> None:
        contract = self.requests.get(reqId)
        ts = datetime.strptime(headTimestamp, "%Y%m%d-%H:%M:%S")
        print(f"Head Timestamp for {contract} is {ts}")

        self.historicalDataRequest(contract)


    def historicalDataRequest(self, contract: Contract) -> None:
        cid = self.next_request_id(contract)
        self.request_details[cid] = contract

        self.reqHistoricalData(
            cid,  # tickerId, used to identify incoming data
            contract.get('contract'),
            '',  # always go to midnight
            self.CHUNK_SIZE,  # amount of time to go back
            contract.get('candlesize'),  # bar size
            self.WHAT_TO_SHOW,  # historical data type
            self.USERTH,  # useRTH (regular trading hours)
            self.FORMAT_DATE,  # format the date in yyyyMMdd HH:mm:ss
            False,  # keep up to date after snapshot
            [],  # chart options
        )

    @iswrapper
    def historicalData(self, reqId: int, bar) -> None:
        # Extract the month and year from the bar's date
        bar_date_pars = bar.date.rsplit(' ', 1)[0]
        print(bar_date_pars)
        date = datetime.strptime(bar_date_pars, "%Y%m%d %H:%M:%S")
        month_key = date.strftime('%Y%m')  # Format as 'YYYYMM'

        contract_details = self.request_details.get(reqId)

        contract = contract_details['contract']
        base_currency = contract.symbol 
        quote_currency = contract.currency  # 'JPY' - this index needs confirmation based on actual structure
        ticket = base_currency + quote_currency

        # Extracting candlesize
        candlesize = contract_details['candlesize']
        self.bar_data[ticket + "_" + candlesize][month_key].append(bar)


    @iswrapper
    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        super().historicalDataEnd(reqId, start, end)
        
        contract_details = self.request_details.get(reqId)
                
        # Add to the request details the start and end date of the specific request
        self.request_details[reqId]['start_date'] = start
        self.request_details[reqId]['end_date'] = end

        # Assuming the format and timezone part is consistent
        start_pars = start.rsplit(' ', 1)[0]  # Split on last space and take the first part
        start_pars = datetime.strptime(start_pars, "%Y%m%d %H:%M:%S")

        end_pars = end.rsplit(' ', 1)[0]  # Split on last space and take the first part
        end_pars = datetime.strptime(end_pars, "%Y%m%d %H:%M:%S")
        month_key_end = end_pars.strftime('%Y%m')
        
        contract = contract_details['contract']
        base_currency = contract.symbol 
        quote_currency = contract.currency  # 'JPY' - this index needs confirmation based on actual structure
        ticket = base_currency + quote_currency
        
        # Extracting candlesize
        candlesize = contract_details['candlesize']

        print(month_key_end,ticket,candlesize)

        self.save_data(contract_details, self.bar_data[ticket + "_" + candlesize][month_key_end], month_key_end)
        del self.bar_data[ticket + "_" + candlesize][month_key_end]

        if start_pars < self.START_DATE:
            print(f"Reached the START_DATE for {ticket}{candlesize}. No more requests needed.")
        else:
            # Update the last end date for the next chunk
            new_end_date = start

            print(f"Requesting more data up to {new_end_date} for {ticket}{candlesize}")

            new_cid = self.next_request_id(contract)
            self.request_details[new_cid] = contract_details
            print(new_cid)
            self.reqHistoricalData(
                new_cid,
                contract,
                new_end_date,
                self.CHUNK_SIZE,  # Assuming you continue with 1-month chunks
                candlesize,
                self.WHAT_TO_SHOW,
                self.USERTH,
                self.FORMAT_DATE,
                False,
                []
            )

    def save_data(self, contract_details, bars: BarDataList, month_key_end) -> None:
        
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

        # Extracting Ticket
        # Assuming the format and positions are known and fixed

        contract = contract_details['contract']
        base_currency = contract.symbol 
        quote_currency = contract.currency  # 'JPY' - this index needs confirmation based on actual structure
        ticket = base_currency + quote_currency

        # Extracting candlesize
        candlesize = contract_details['candlesize']
        

        path = os.path.sep.join(
                [make_download_path(ticket,candlesize), f"{month_key_end}.csv"]
                )
        if not os.path.exists(make_download_path(ticket,candlesize)):
            os.makedirs(make_download_path(ticket,candlesize), exist_ok=True)

        df.to_csv(path, index=False)

    def websocket_con(self):
        self.run()

    def disconnect_from_ib(self):
        self.disconnect()
        