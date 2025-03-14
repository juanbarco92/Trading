# src/data_extraction/extractor.py

import logging
import json

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.utils import iswrapper

from datetime import datetime

class ForexDataExtractor(EClient, EWrapper):
    def __init__(self, contracts):
        EClient.__init__(self, self)
        self.request_id = 0
        self.data_buffers  = {}
        self.started = False
        self.next_valid_order_id = None
        self.contracts = contracts

    def connect_to_ib(self):
        self.connect(self.ip_address, self.port, clientId = self.client_id)
        print("Connected to IB.")

    @iswrapper
    def connectAck(self):
        logging.info("Connected")

    def load_metadata(self):
        try:
            with open('data/metadata.json', 'r') as file:
                self.metadata = json.load(file)
        except FileNotFoundError:
            self.metadata = {}

    @iswrapper
    def nextValidId(self, order_id: int):
        super().nextValidId(order_id)

        self.next_valid_order_id = order_id
        logging.info(f"nextValidId: {order_id}")
        # we can start now
        self.start()

    def start(self):
        if self.started:
            return

        self.started = True
        for contract in self.contracts:
            self.reqHeadTimeStamp(
                self.next_request_id(contract), contract, self.args.data_type, 0, 1
            )

    def update_metadata(self, pair, candle_size, last_bar_timestamp):
        if pair not in self.metadata:
            self.metadata[pair] = {}
        self.metadata[pair][candle_size] = {"last_retrieved": last_bar_timestamp}
        with open('data/metadata.json', 'w') as file:
            json.dump(self.metadata, file, indent=4)    


    def extract_data(self, contract, query_parameters):
        """
        Initiates a request to extract data for a given contract using specified query parameters.
        
        Parameters:
        - contract (Contract): The contract for which to extract data.
        - query_parameters (dict): Specifications for the data query, such as duration, 
                                   bar size, what to show, and useRTH (Regular Trading Hours).
                                   
        Note: Ensure the connection to IBapi is established before calling this method.
        """
        # This will depend on the specifics of your data needs and the IBapi
        self.reqHistoricalData(reqId=self.nextRequestId(), 
                               contract=contract,
                               endDateTime='',
                               durationStr=query_parameters['durationStr'],
                               barSizeSetting=query_parameters['barSizeSetting'],
                               whatToShow=query_parameters['whatToShow'],
                               useRTH=query_parameters['useRTH'],
                               formatDate=query_parameters['formatDate'],
                               keepUpToDate=False,
                               chartOptions=[])
        
        #update attributes for metadata storing
        self.pair = query_parameters['pair']
        self.candle_size = query_parameters['barSizeSetting']

        print(self.last_req_id)
        
    def nextRequestId(self):
        """
        Generates a unique request ID for each data request.

        Returns:
            int: A unique request ID.
        """
        self.last_req_id += 1
        return self.last_req_id
    
    def historicalData(self, reqId, bar):
        """Called when a single data bar is received."""
        if reqId not in self.data_buffers:
            self.data_buffers[reqId] = []
        self.data_buffers[reqId].append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])
        self.last_bar_timestamp = bar.date

    def historicalDataEnd(self, reqId, start, end):
        """Called when the historical data request is completed."""
        print(f"Data request {reqId} completed from {start} to {end}")
        print(list(self.data_buffers.keys()))
        self.write_to_csv(reqId)

        self.update_metadata(self.pair, self.candle_size, self.last_bar_timestamp)
        # Clear the buffer for this reqId after processing
        del self.data_buffers[reqId]

    def write_to_csv(self, reqId):
        """Writes collected data for a given reqId to a CSV file."""
        import csv
        filename = f"data/data_{reqId}.csv"
        # Ensure reqId is in data_buffers and has data
        if reqId in self.data_buffers and self.data_buffers[reqId]:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
                writer.writerows(self.data_buffers[reqId])  # Corrected to write rows from buffer
        else:
            print(f"No data to write for reqId {reqId}.")



    def websocket_con(self):
        self.run()

    def disconnect_from_ib(self):
        self.disconnect()


