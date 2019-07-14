import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from util import BaseNN
from util import balance_classes
from util import get_current_date
from sklearn.utils import resample
from constants import NN_TRAINING_DATA_PATH
from constants import MODEL_PATH

# dataset = pandas.read_csv('/Users/rjh2nd/Dropbox (Personal)/Programing/databases/iris.data').values
# # encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y = shuffle(np_utils.to_categorical(encoded_Y), random_state=1)

INPUT_SIZE=17

def create_data_for_train_test(data):
    dataset = data.values
    X = shuffle(dataset[:,0:(dataset.shape[1]-1)].astype(float), random_state=2)
    shuffled_nan_mask = ~np.isnan(X).any(axis=1)
    X = X[shuffled_nan_mask, :]
    Y = shuffle(dataset[:, (dataset.shape[1]-1)], random_state=2)[shuffled_nan_mask]
    neg_y_mask = Y < 0
    low_y_mask = (Y > 0) * (Y < 5)
    hi_y_mask = (Y > 5)

    Y_unbalanced = np.hstack((neg_y_mask.reshape(len(Y), 1), low_y_mask.reshape(len(Y), 1), hi_y_mask.reshape(len(Y), 1)))
    hi_balance_mask = hi_y_mask
    X_unbalanced = X

    X_hi_balanced = balance_classes(X_unbalanced, hi_balance_mask, np.sum(neg_y_mask))
    Y_hi_balanced = balance_classes(Y_unbalanced, hi_balance_mask, np.sum(neg_y_mask))
    low_balance_mask = Y_hi_balanced[:, 1]
    X_fit = balance_classes(X_hi_balanced, low_balance_mask, np.sum(neg_y_mask))
    Y_fit = balance_classes(Y_hi_balanced, low_balance_mask, np.sum(neg_y_mask))

    return X_fit, Y_fit

def normalize_rows(df, col_list, norm_list):
    for col, norm in zip(col_list, norm_list):
        if type(norm) == str:
            df[col] = df[col].values / df[norm]
        else:
            df[col] = df[col] / norm

    return df

def save_pred_data_as_csv(df, xlsx_name):
    # Create the Excel file
    writer = pd.ExcelWriter(xlsx_name + '.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    # Light red fill with dark red text.
    red_format = workbook.add_format({'bg_color': '#FFC7CE',
                                   'font_color': '#9C0006'})
    # Light yellow fill with dark yellow text.
    yellow_format = workbook.add_format({'bg_color': '#FFEB9C',
                                   'font_color': '#9C6500'})
    # Green fill with dark green text.
    green_format = workbook.add_format({'bg_color': '#C6EFCE',
                                   'font_color': '#006100'})

    sheet_len = str(df.shape[0] + 1)

    # Wroksheet conditions
    worksheet.conditional_format('B2:D' + sheet_len, {'type': 'data_bar'})
    worksheet.conditional_format('E2:E' + sheet_len, {'type': '3_color_scale'})
    worksheet.conditional_format('A2:A' + sheet_len,
                                 {'type': 'formula',
                                  'criteria': '=$B2>$E2',
                                  'format': red_format
                                  })
    worksheet.conditional_format('A2:A' + sheet_len,
                                 {'type': 'formula',
                                  'criteria': '=AND($E2>$B2, $C2>$D2)',
                                  'format': yellow_format
                                  })
    worksheet.conditional_format('A2:A' + sheet_len,
                                 {'type': 'formula',
                                  'criteria': '=AND($E2>$B2, $C2<$D2)',
                                  'format': green_format
                                  })
    writer.save()


class ClassifierNN(BaseNN):

    def __init__(self, model_type=Sequential(), model_path=None, seed=7, number_of_classes=3):
        # N is the number
        super(ClassifierNN, self).__init__(model_type, model_path, seed)
        if model_path is None:
            self.model.add(Dense(30, input_dim=INPUT_SIZE, activation='relu'))
            self.model.add(LeakyReLU())
            self.model.add(Dense(number_of_classes, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __call__(self):
        return self.model


if __name__ == "__main__":
    train = True
    #TODO move to StockAnalyzer
    prefix = 'daily_change'
    if train:
        train_dataset = pd.read_csv(NN_TRAINING_DATA_PATH + prefix + '_training_data.csv', index_col=0).drop_duplicates(subset ="Previous Movement", keep ='first', inplace = False)
        train_dataset = normalize_rows(train_dataset, ['Previous Movement'], [100])

        val_dataset = pd.read_csv(NN_TRAINING_DATA_PATH + prefix + '_val_data.csv', index_col=0).drop_duplicates(subset ="Previous Movement", keep ='first', inplace = False)
        val_dataset = normalize_rows(val_dataset, ['Previous Movement'], [100])

        test_dataset = pd.read_csv(NN_TRAINING_DATA_PATH + prefix + '_test_data.csv', index_col=0)
        test_dataset = normalize_rows(test_dataset, ['Previous Movement'], [100])

        X_train, Y_train = create_data_for_train_test(train_dataset)
        X_val, Y_val = create_data_for_train_test(val_dataset)
        X_test = test_dataset.values[:, 0:INPUT_SIZE]
        Y_test = test_dataset.values[:, INPUT_SIZE]

        X_fit = np.vstack((X_train, X_val))
        Y_fit = np.vstack((Y_train, Y_val))

        train_val_split = X_val.shape[0] / X_fit.shape[0]

        class_nn = ClassifierNN()
        class_nn.train_model(X_fit, Y_fit, 30, batch_size=5, training_patience=200, val_split=train_val_split, file_name=MODEL_PATH + 'stock_daily_change_predictor_' + get_current_date() + '.h5')
        test_data = class_nn.test_model(X_test, Y_test, show_plots=False)
        df1 = pd.DataFrame(test_data['Predicted'], index=test_dataset.index.values)
        pos_comb = df1[1].values + df1[2].values
        df1['pos_comb'] = pos_comb
        df2 = pd.DataFrame(test_data['Measured'], index=test_dataset.index.values)
        df = pd.concat((df1, df2), axis=1)
        df.index = test_dataset.index.values
        save_pred_data_as_csv(df, prefix + '_test_results_100')

        # estimator = KerasClassifier(build_fn=class_nn, epochs=200, batch_size=5, verbose=2)
        # kfold = KFold(n_splits=10, shuffle=True, random_state=class_nn.seed)
        #
        # results = cross_val_score(estimator, X_fit, Y_fit, cv=kfold)
        # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    else:
        class_nn = ClassifierNN(model_path='/Users/rjh2nd/PycharmProjects/StockAnalyzer/models/stock_daily_change_predictor_20190610.h5')
        dataset = pd.read_csv('/Users/rjh2nd/PycharmProjects/StockAnalyzer/pred_data.csv', index_col=0)
        dataset = normalize_rows(dataset, ['Previous Movement'], [100])
        X = dataset.values[:, 0:INPUT_SIZE]
        prediction = class_nn.model.predict(X)
        pred_df = pd.DataFrame(data=prediction, columns=['negative', 'weak positive', 'strong positive'], index=dataset.index.values)
        pred_df['total positive'] = pred_df['weak positive'].values + pred_df['strong positive'].values
        save_pred_data_as_csv(pred_df, '/Users/rjh2nd/Dropbox (Personal)/StockAnalyzer/data/20190612/Predictions_20190612')

