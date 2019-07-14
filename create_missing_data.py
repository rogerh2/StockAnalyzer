import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
from constants import STOCK_DATA_PATH
from constants import FMT

all_files = os.listdir(STOCK_DATA_PATH)
create_new_files = False

if create_new_files:
    # create files for missing days (helpful for data gathered on weekends)
    for file in all_files:
        if file[0] == '.':
            #This helps avoid Mac's hidden files
            continue
        day = file[0:10]
        file_name_base = file[10::]
        current_fname_day = dt.strptime(day, FMT) - timedelta(days=1)
        current_fname_day_str = current_fname_day.strftime(FMT)
        new_file_name = current_fname_day_str + file_name_base

        old_data = pd.read_csv(STOCK_DATA_PATH + file, index_col=0)
        num_trading_days = len(old_data['1. open'].dropna().values)
        if (num_trading_days > 99) and ((not np.isnan(old_data['1. open'].values[-1])) or (day == '2019-07-06')) and (not new_file_name in all_files):
            new_data = old_data.iloc[0:-1, ::]
            new_data.to_csv(STOCK_DATA_PATH + new_file_name)
            print('Creating new file: ' + new_file_name)
else:
    # Create lists as input to create training/validation/and test data
    all_days = list(set([file[0:10] for file in all_files]))
    num_test_files = 7
    traning_len = int(8 * (len(all_days)-num_test_files)/10)
    print('training: ' + str(all_days[0:traning_len]))
    print('validation: ' + str(all_days[traning_len:-num_test_files]))
    print('test: ' + str(all_days[-num_test_files::]))