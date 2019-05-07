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