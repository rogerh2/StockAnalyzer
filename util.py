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

def analyze_correlations(data_path):
    ticker = pd.read_csv(data_path, index_col=0)
    daily_change_col = ticker['4. close'] - ticker['1. open']
    daily_change_col.name = 'daily_change'
    ticker_full = ticker.join(daily_change_col)
    corr_data_change = ticker_full.corr()['daily_change'][0:-1]
    corr_data_open = ticker_full.corr()['1. open'][0:-1]

    for data_name in corr_data_change.index.values:
        if np.abs(corr_data_change.loc[data_name]) > 0.4:
            ticker_full.plot(x=data_name, y='daily_change', style=['rx'])
            plt.title(data_name + 'vs daily_change: ' + num2str(corr_data_change.loc[data_name], 2))

        if (np.abs(corr_data_open.loc[data_name]) > 0.4) and (np.abs(corr_data_open.loc[data_name]) < 1):
            ticker.plot(x=data_name, y='1. open', style=['rx'])
            plt.title(data_name + 'vs open: ' + num2str(corr_data_open.loc[data_name], 2))

    plt.figure()
    plt.plot(ticker_full.daily_change.values[0:-1], ticker_full.daily_change.values[1::], 'rx')
    plt.title('Daily Change Autocorellation: ' + num2str(ticker_full.daily_change.autocorr(), 3))

    plt.show()


if __name__ == "__main__":
    analyze_correlations('/Users/rjh2nd/PycharmProjects/StockAnalyzer/Stock Data/stock_data_forRegional Health_from 2019-04-29.csv')