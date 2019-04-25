import unittest
import numpy as np
from StockAnalyzer import Ticker

class TickerTestCase(unittest.TestCase):

    def setUp(self):
        self.ticker = Ticker('AAPL')

    def test_ticker_creates_data_frame_with_appropriate_indices(self):
        self.ticker.join_data_into_dataframes()
        inds = self.ticker.df.index
        max_ind_len = np.max(np.array([len(self.ticker.data[key].index) for key in self.ticker.data.keys()]))
        self.assertEqual(len(inds), max_ind_len)


if __name__ == '__main__':
    unittest.main()
