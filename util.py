import numpy as np
import pandas as pd
import scipy.stats
import pytz
from matplotlib import pyplot as plt
from datetime import datetime as dt

def num2str(num, digits):
    # This function formats numbers as strings with the desired number of digits
    fmt_str = "{:0." + str(digits) + "f}"
    num_str = fmt_str.format(num)

    return num_str

def mean_confidence_interval(data, confidence=0.95):
    a = data
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def progress_printer(total_len, current_ind, start_ind=0, digit_resolution=1, print_resolution=None, tsk='Task', suppress_output=False):

    if print_resolution is None:
        # Print resolutions is the number of digits to print whereas digit resolution is how small of changes should be
        # registered, in most cases these are the same
        print_resolution = digit_resolution

    if not suppress_output:
        progress_percent = 100*(current_ind-start_ind)/(total_len-start_ind)
        resolution = 10**-(digit_resolution+2)

        if 1 >= (total_len - start_ind)*resolution:
            print (tsk + ' is ' + num2str(progress_percent, print_resolution) + '% Complete')
        else:
            relevant_inds = range(start_ind, total_len, round((total_len - start_ind)*resolution))
            if current_ind in relevant_inds:
                print(tsk + ' is ' + num2str(progress_percent, print_resolution) + '% Complete')

    else:
        pass

def convert_utc_str_to_est_str(naive_datetime_str, from_fmt, to_fmt):
    # This function converts utc dates to etc
    naive_datetime = dt.strptime(naive_datetime_str[0:19], from_fmt)
    utc = pytz.UTC
    est = pytz.timezone('America/New_York')
    utc_date = utc.localize(naive_datetime)
    est_date = utc_date.astimezone(est)
    est_date_str = est_date.strftime(to_fmt)
    return est_date_str

def convert_utc_str_to_timestamp(naive_datetime_str, from_fmt):
    naive_datetime = dt.strptime(naive_datetime_str[0:19], from_fmt)
    utc = pytz.UTC
    utc_date = utc.localize(naive_datetime)
    ts = utc_date.timestamp()
    return ts

def eliminate_nans_equally(arr_list):
    arr_len = len(arr_list)

    for i in range(0, arr_len):
        current_mask = ~np.isnan(arr_list[i])
        for j in range(0, arr_len):
            arr_list[j] = arr_list[j][current_mask]

    return arr_list

def inv_t(t):
    ans = 1 / t
    return ans

def inv_sq_t(t):
    ans = -inv_t(t**2)
    return ans

def exp_decay(t):
    ans = np.exp(-t)
    return ans

def unity(t):
    return t


class Stats:

    def __init__(self, data_path):
        ticker = pd.read_csv(data_path, index_col=0)
        daily_change_col = ticker['4. close'] - ticker['1. open']
        daily_change_col.name = 'daily_change'
        self.ticker = ticker.join(daily_change_col)


    def create_time_dependent_arr(self, header_to_convert, t_func=inv_t, start_val=20):
        arr_to_convert = self.ticker[header_to_convert].values / np.max(self.ticker[header_to_convert].fillna(0).values)

        new_arr = np.array([])

        for i in range(start_val, len(arr_to_convert)):
            current_term = arr_to_convert[i]
            for j in range(1, start_val):
                t = start_val - j
                current_term += arr_to_convert[i-t] * t_func(t)

            new_arr = np.append(new_arr, current_term)

        return new_arr

    def fit_time_dependent_arr_to_arr(self, t_dependent_arr_header, arr_header, t_func=inv_t, start_val=20):
        arr = self.ticker[arr_header].values[start_val::]
        t_dependent_arr = self.create_time_dependent_arr(t_dependent_arr_header, t_func=t_func, start_val=start_val)
        no_nan_arrs = eliminate_nans_equally([arr, t_dependent_arr])
        daily_open_no_nan = no_nan_arrs[0]
        header_arr_no_nan = no_nan_arrs[1]

        coeff = np.polyfit(header_arr_no_nan, daily_open_no_nan, 1)

        return coeff, t_dependent_arr, arr

    def plot_time_dependent_arr_vs_arr(self, t_dependent_arr_header, arr_header, t_func=inv_t, start_val=20):
        coeff, t_dependent_arr, arr = stat.fit_time_dependent_arr_to_arr(t_dependent_arr_header, arr_header, t_func=t_func, start_val=start_val)
        plt.plot(arr - coeff[0] * t_dependent_arr, '--bo')
        plt.title('Zeroed')
        plt.figure()
        plt.plot(coeff[0] * t_dependent_arr, arr, 'rx')
        plt.title('Fit')
        plt.figure()
        plt.plot(arr)
        plt.title(arr_header)
        plt.figure()
        plt.plot(t_dependent_arr)
        plt.title(t_dependent_arr_header)

        plt.show()


    def analyze_correlations(self, corr_header):
        ticker = self.ticker
        corr_data_open = ticker.corr()[corr_header]

        for data_name in corr_data_open.index.values:
            if (np.abs(corr_data_open.loc[data_name]) > 0.4) and (np.abs(corr_data_open.loc[data_name]) < 1):
                ticker.plot(x=data_name, y=corr_header, style=['rx'])
                plt.title(data_name + 'vs ' + corr_header + ' ' + num2str(corr_data_open.loc[data_name], 2))

        plt.show()

    def plot_autocorrelation(self, header):
        data = self.ticker[header]
        plt.figure()
        plt.plot(data.values[0:-1], data.values[1::], 'rx')
        plt.title('Daily Change Autocorellation: ' + num2str(data.autocorr(), 3))
        plt.show()




if __name__ == "__main__":
    stat = Stats('/Users/rjh2nd/PycharmProjects/StockAnalyzer/Stock Data/Aspen Aerogels_from 2019-04-29.csv')
    # _ = stat.analyze_correlations('1. open')
    stat.plot_time_dependent_arr_vs_arr('aerogel', '1. open', t_func=inv_t)