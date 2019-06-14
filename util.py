import numpy as np
import scipy.stats
import pytz
import keras
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime as dt
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from sklearn.utils import resample
from sklearn.utils import shuffle

def num2str(num, digits):
    # This function formats numbers as strings with the desired number of digits
    fmt_str = "{:0." + str(digits) + "f}"
    num_str = fmt_str.format(num)

    return num_str

def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), scipy.stats.sem(data)
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

def plot_informative_lines(ax, horz_pos=0, vert_pos=None, style='r--'):
    fig = ax.get_figure()
    plt.figure(fig.number)
    plt.sca(ax)
    left, right = plt.xlim()
    top, bot = plt.ylim()
    if vert_pos is not None:
        plt.plot([vert_pos, vert_pos], [bot, top], style)
    if horz_pos is not None:
        plt.plot([left, right], [horz_pos, horz_pos], style)
    plt.xlim(left, right)
    plt.ylim(top, bot)

def balance_classes(data, upsample_mask, new_sample_size, upsample=True, seed=1):
    data_to_sample = data[upsample_mask, :]
    if  new_sample_size > np.sum(upsample_mask):
        resampled_data = resample(data_to_sample, replace=upsample, n_samples=(new_sample_size - np.sum(upsample_mask)), random_state=seed)
        balanced_data = shuffle(np.vstack((data, resampled_data)), random_state=seed)
    else:
        balanced_data = data


    return balanced_data

def merge_dicts(dicts):
    super_dict = {}
    for d in dicts:
        for k, v in d.items():  # d.items() in Python 3+
            super_dict.setdefault(k, v)

    return super_dict


class BaseNN:
    # This class is meant to serve as a parent class to neural net based machine learning classes

    def __init__(self, model_type=models.Sequential(), model_path=None, seed=7):
        np.random.seed(seed)
        if model_path is None:
            self.model = model_type
        else:
            self.model = models.load_model(model_path)

        self.seed = seed

    def train_model(self, training_input, training_output, epochs, file_name=None, retrain_model=False, shuffle=True, val_split=0.25, batch_size=96, training_patience=2, min_training_delta=0, training_e_stop_monitor='val_loss'):
        if retrain_model:
            print('re-trianing model')
            self.model.reset_states()

        estop = keras.callbacks.EarlyStopping(monitor=training_e_stop_monitor, min_delta=min_training_delta, patience=training_patience, verbose=0, mode='auto')

        hist = self.model.fit(training_input, training_output, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=shuffle, validation_split=val_split, callbacks=[estop])

        if not file_name is None:
            self.model.save(file_name)

        return hist

    def test_model(self, test_input, test_output, show_plots=True, x_indices=None, prediction_names=('Predicted', 'Measured'), prediction_styles=('rx--', 'b.--'), x_label=None, y_label=None, title=None):
        prediction = self.model.predict(test_input)
        if prediction.shape[1] == 0:
            prediction = prediction[::, 0] # For some reason the predictions come out 2D (e.g. [[p1,...,pn]] vs [p1,...,pn]]

        if show_plots:
            # Plot the price and the predicted price vs time
            if x_indices is None:
                plt.plot(prediction, prediction_styles[0])
                plt.plot(test_output, prediction_styles[1])
                plt.legend(prediction_names[0], prediction_names[1])
            else:
                df = pd.DataFrame(data={prediction_names[0]: test_output, prediction_names[1]: prediction}, index=x_indices)
                df.Predicted.plot(style=prediction_styles[0])
                df.Actual.plot(style=prediction_styles[1])

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)

            # Plot the correlation between price and predicted
            plt.figure()
            plt.plot(prediction, test_output, 'b.')
            plt.ylabel(prediction_names[0])
            plt.xlabel(prediction_names[1])
            plt.title('Correlation Between ' + prediction_names[0] + ' and ' + prediction_names[1])

            plt.show()


        return {prediction_names[0]:prediction, prediction_names[1]:test_output}

class ClassifierNN(BaseNN):

    def __init__(self, model_type=Sequential(), N=3, input_size=17, model_path=None, seed=7):
        super(ClassifierNN, self).__init__(model_type, model_path, seed)
        if model_path is None:
            self.model.add(Dense(30, input_dim=input_size, activation='relu'))
            self.model.add(LeakyReLU())
            self.model.add(Dense(N, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __call__(self):
        return self.model