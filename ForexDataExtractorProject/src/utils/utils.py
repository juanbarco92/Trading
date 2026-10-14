from ibapi.contract import Contract

def fxPair(pair: str, sec_type: str = "CASH", exchange: str = "IDEALPRO") -> Contract:
    """
    Creates a Contract object for a forex pair for use with the Interactive Brokers API.

    Parameters:
    - pair (str): The forex pair in "BaseCurrency/QuoteCurrency" format (e.g., "EUR/USD").
    - sec_type (str, optional): Defaults to "CASH".
    - exchange (str, optional): Defaults to "IDEALPRO".

    Returns:
    - Contract: An IBapi contract object configured for the specified forex pair.
    """
    contract = Contract()
    parts = pair.split("/")
    contract.symbol = parts[0]
    contract.secType = sec_type
    contract.currency = parts[1]
    contract.exchange = exchange
    return contract


def stockContract(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
    """
    Creates a Contract object for a Stock or ETF (e.g. 'SPY', 'QQQ', 'AAPL')
    for use with the Interactive Brokers API.

    Parameters:
    - symbol (str): Ticker symbol.
    - exchange (str, optional): Routing exchange. Defaults to "SMART".
    - currency (str, optional): Currency. Defaults to "USD".

    Returns:
    - Contract: An IBapi contract object configured for the stock/ETF.
    """
    contract = Contract()
    contract.symbol = symbol.upper()
    contract.secType = "STK"
    contract.currency = currency
    contract.exchange = exchange
    return contract


def futureContract(symbol: str, last_trade_date: str = "", exchange: str = "CME", currency: str = "USD") -> Contract:
    """
    Creates a Contract object for a Futures contract (e.g. 'MES', 'MNQ', 'ES')
    for use with the Interactive Brokers API.

    Parameters:
    - symbol (str): Future underlying symbol (e.g. 'MES').
    - last_trade_date (str, optional): Expiration date or month (YYYYMM or YYYYMMDD).
    - exchange (str, optional): Defaults to "CME".
    - currency (str, optional): Defaults to "USD".

    Returns:
    - Contract: An IBapi contract object configured for the future.
    """
    contract = Contract()
    contract.symbol = symbol.upper()
    contract.secType = "FUT"
    contract.currency = currency
    contract.exchange = exchange
    if last_trade_date:
        contract.lastTradeDateOrContractMonth = last_trade_date
    return contract
