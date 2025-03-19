# scripts/testing.py

### update root_path
from pathlib import Path
import sys


path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))


### start testing
#from src.data_extraction.extractor import ForexDataExtractor
from src.data_extraction.download_data import ForexDataExtractor
from src.utils.utils import fxPair
import threading
import time
from datetime import datetime

# Defining current time
now = datetime.now()

#create list of desired data

pair = 'EUR/USD'
contract1 = fxPair(pair)

pair = 'USD/JPY'
contract2 = fxPair(pair)

contracts = {
    contract1:[
        {
         'candlesize':'1 min',
         'duration':'1 D'
        }
    ]
}





# Instatiate Forex Extractor Class
app = ForexDataExtractor(contracts)
app.connect("127.0.0.1", port=7497, clientId=1)
app.websocket_con()
# starting a separate daemon thread to execute the websocket connection
#con_thread = threading.Thread(target=app.websocket_con, daemon=True)
#con_thread.start()
#time.sleep(4000)





# # Define common query parameters for data extraction
# query_parameters = {
#         "endtime": datetime.now().strftime("%Y-%m-%d"),
#         "barSizeSetting": candlesize,
#         "durationStr": "1 M",
#         "whatToShow": "MIDPOINT",
#         "useRTH": 1,
#         "formatDate": 1 , 
#         "keepUpToDate": 0,
#         "pair": pair
# }

# # extract desired data
# app.extract_data(contract=contract,
#                 query_parameters=query_parameters)



app.disconnect_from_ib()

