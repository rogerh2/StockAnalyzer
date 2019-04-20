import unittest
from util import convert_utc_str_to_est_str
from util import convert_utc_str_to_timestamp

class UtilUnitTests(unittest.TestCase):

    def test_time_str_conversion(self):
        utc_date_str = "2019-04-11T03:20:59+00:00"
        est_date_str = "2019-04-10"
        utc_fmt = '%Y-%m-%dT%H:%M:%S'
        est_fmt = "%Y-%m-%d"
        test_est_str = convert_utc_str_to_est_str(utc_date_str, utc_fmt, est_fmt)
        self.assertEqual(est_date_str, test_est_str)

    def test_time_str_to_timestamp_conversion(self):
        utc_date_str = "2019-04-11T03:20:59+00:00"
        ts = 1554952859
        utc_fmt = '%Y-%m-%dT%H:%M:%S'
        test_est_str = convert_utc_str_to_timestamp(utc_date_str, utc_fmt)
        self.assertEqual(ts, test_est_str)


if __name__ == '__main__':
    unittest.main()
