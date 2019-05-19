import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from util import BaseNN
from util import balance_classes
from sklearn.utils import resample

# dataset = pandas.read_csv('/Users/rjh2nd/Dropbox (Personal)/Programing/databases/iris.data').values
dataset = pandas.read_csv('/Users/rjh2nd/PycharmProjects/StockAnalyzer/training_data.csv', index_col=0).values
X = shuffle(dataset[:,0:17].astype(float), random_state=2)
shuffled_nan_mask = ~np.isnan(X).any(axis=1)
X = X[shuffled_nan_mask, :]
Y = shuffle(dataset[:, 17], random_state=2)[shuffled_nan_mask]
# # encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# # convert integers to dummy variables (i.e. one hot encoded)
# dummy_y = shuffle(np_utils.to_categorical(encoded_Y), random_state=1)
neg_y_mask = Y < 0
low_y_mask = (Y > 0) * (Y < 5)
hi_y_mask = (Y > 5)

dummy_y = np.hstack((neg_y_mask.reshape(len(Y), 1), low_y_mask.reshape(len(Y), 1), hi_y_mask.reshape(len(Y), 1)))
dummy_y = dummy_y

X_test = X[600::, :]
Y_test = dummy_y[600::, :]


hi_balance_mask = hi_y_mask[0:600]
X_unbalanced = X[0:600, :]
Y_unbalanced = dummy_y[0:600, :]

X_hi_balanced = balance_classes(X_unbalanced, hi_balance_mask, np.sum(neg_y_mask))
Y_hi_balanced = balance_classes(Y_unbalanced, hi_balance_mask, np.sum(neg_y_mask))

low_balance_mask = Y_hi_balanced[:, 1]
X_fit = balance_classes(X_hi_balanced, low_balance_mask, np.sum(neg_y_mask))
Y_fit = balance_classes(Y_hi_balanced, low_balance_mask, np.sum(neg_y_mask))


class ClassifierNN(BaseNN):

    def __init__(self, model_type=Sequential(), model_path=None, seed=7):
        super(ClassifierNN, self).__init__(model_type, model_path, seed)
        self.model.add(Dense(8, input_dim=17, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __call__(self):
        return self.model

class_nn = ClassifierNN()
class_nn.train_model(X_fit, Y_fit, 200, batch_size=5, training_patience=200, val_split=0.33)
test_data = class_nn.test_model(X_test, Y_test, show_plots=False)

estimator = KerasClassifier(build_fn=class_nn, epochs=200, batch_size=5, verbose=2)
kfold = KFold(n_splits=10, shuffle=True, random_state=class_nn.seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

