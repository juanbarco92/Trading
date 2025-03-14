from ibapi.contract import Contract

from ibapi.contract import Contract

def fxPair(pair, sec_type="CASH", exchange="IDEALPRO"):
    """
    Creates a Contract object for a forex pair for use with the Interactive Brokers API.

    This function constructs a contract object representing a forex currency pair,
    which can be used to request data or place orders via the IB API. The function
    expects the currency pair in the format "BaseCurrency/QuoteCurrency" (e.g., "EUR/USD").

    Parameters:
    - pair (str): The forex pair in "BaseCurrency/QuoteCurrency" format. 
                  The base currency is the first currency in a forex pair, 
                  and the quote currency is the second currency. For example, 
                  in the forex pair "EUR/USD", EUR is the base currency, and USD is the quote currency.
    - sec_type (str, optional): The security type. Defaults to "CASH", which is 
                                suitable for forex pairs in the Interactive Brokers API.
    - exchange (str, optional): The exchange where the currency pair is traded. 
                                Defaults to "IDEALPRO", which is IB's electronic 
                                trading platform for forex.

    Returns:
    - Contract: An IBapi contract object configured for the specified forex pair.

    """
    contract = Contract()
    contract.symbol = pair.split("/")[0]
    contract.secType = sec_type
    contract.currency = pair.split("/")[1]
    contract.exchange = exchange
    return contract


