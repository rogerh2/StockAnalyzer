import unittest
import numpy as np
import pandas as pd
from StockAnalyzer import create_between_day_change_col


class UtilUnitTests(unittest.TestCase):

    def test_time_str_conversion(self):
        opens = np.array([1.1, np.nan, np.nan, 1.1, 1.1, 1.1, 1.1, 1.1, np.nan, np.nan])
        closes = np.array([1.0, np.nan, np.nan, 1.2, 1.3, 1.4, 1.5, 1.6, np.nan, np.nan])
        true_ans = np.array([np.nan, np.nan, np.nan, 0.1, -0.1, -0.2, -0.3, -0.4, np.nan, np.nan])
        df = pd.DataFrame({'1. open': opens, '4. close':closes})
        calculated_ans = create_between_day_change_col(df)
        np.testing.assert_array_almost_equal(true_ans, calculated_ans, 0.001)


if __name__ == '__main__':
    unittest.main()