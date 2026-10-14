# tests/test_extractor.py
import unittest
from unittest.mock import patch
from src.data_extraction.extractor import ForexDataExtractor

class TestForexDataExtractor(unittest.TestCase):
    @patch('src.data_extraction.extractor.EClient.connect')
    def test_connection(self, mock_connect):
        extractor = ForexDataExtractor(contracts={})
        extractor.ip_address = '127.0.0.1'
        extractor.port = 7497
        extractor.client_id = 1

        extractor.connect_to_ib()
        mock_connect.assert_called_with('127.0.0.1', 7497, clientId=1)

if __name__ == '__main__':
    unittest.main()