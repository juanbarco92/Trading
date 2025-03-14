# tests/test_extractor.py
import unittest
from unittest.mock import patch
from src.data_extraction.extractor import ForexDataExtractor

class TestForexDataExtractor(unittest.TestCase):
    @patch('src.data_extraction.extractor.EClient')
    def test_connection(self, MockedEClient):
        # Initialize the extractor
        extractor = ForexDataExtractor()

        # Perform the connection
        extractor.connect_to_ib()

        # Assert that EClient's connect method was called with the correct parameters
        MockedEClient.connect.assert_called_with(extractor.ip_address, extractor.port, extractor.client_id)

if __name__ == '__main__':
    unittest.main()