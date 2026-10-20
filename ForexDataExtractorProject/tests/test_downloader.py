import unittest
from datetime import datetime
from src.utils.utils import stockContract, fxPair, futureContract
from src.data_extraction.download_data import (
    get_ticket_from_contract,
    parse_bar_date,
    make_download_path,
    MarketDataDownloader,
)


class TestMarketDataDownloader(unittest.TestCase):
    def test_get_ticket_from_contract(self):
        stk = stockContract("SPY")
        self.assertEqual(get_ticket_from_contract(stk), "SPY")

        fut = futureContract("MES")
        self.assertEqual(get_ticket_from_contract(fut), "MES")

        fx = fxPair("EUR/USD")
        self.assertEqual(get_ticket_from_contract(fx), "EURUSD")

    def test_parse_bar_date(self):
        # Daily format
        d1 = parse_bar_date("20240115")
        self.assertEqual(d1, datetime(2024, 1, 15))

        # Intraday with timezone
        d2 = parse_bar_date("20240115 14:30:00 US/Eastern")
        self.assertEqual(d2, datetime(2024, 1, 15, 14, 30, 0))

        # Intraday standard
        d3 = parse_bar_date("20240115 09:30:00")
        self.assertEqual(d3, datetime(2024, 1, 15, 9, 30, 0))

        # Head timestamp
        d4 = parse_bar_date("20200101-00:00:00")
        self.assertEqual(d4, datetime(2020, 1, 1, 0, 0, 0))

    def test_make_download_path(self):
        path = make_download_path("SPY", "1 hour", base_dir="data")
        self.assertIn("SPY", path)
        self.assertIn("1hour", path)

    def test_default_streams(self):
        downloader = MarketDataDownloader(contracts={})
        stk = stockContract("AAPL")
        fx = fxPair("EUR/USD")

        self.assertEqual(downloader._get_default_what_to_show(stk), "TRADES")
        self.assertEqual(downloader._get_default_what_to_show(fx), "MIDPOINT")

        self.assertEqual(downloader._get_default_use_rth(stk), 1)
        self.assertEqual(downloader._get_default_use_rth(fx), 0)


if __name__ == "__main__":
    unittest.main()
