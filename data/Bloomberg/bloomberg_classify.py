import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn import preprocessing


def get_x_y(vals, commodity_index):
    X = []
    y = []
    for i in range(1000, len(vals) - 1):
        X.append(vals[i-1000:i])
        y.append(np.array([i > j for i, j in zip(vals[i], vals[i - 1])])[commodity_index])
    return np.array(X), np.array(y).astype(int)


df = pd.read_csv('PriceDate.csv')
df = df.drop(['Date'], axis=1)
train_vals = df.iloc[:4000].values
test_vals = df.iloc[4000:].values


avg_acc = 0
for commodity_index in range(32):
    X_train, y_train = get_x_y(train_vals, commodity_index)
    X_train = np.swapaxes(X_train, 1, 2)
    nsamples, nx, ny = X_train.shape
    d2_X_train = X_train.reshape((nsamples,nx*ny))
    d2_X_train = preprocessing.normalize(d2_X_train)
    X_test, y_test = get_x_y(test_vals, commodity_index)
    X_test = np.swapaxes(X_test, 1, 2)
    nsamples, nx, ny = X_test.shape
    d2_X_test = X_test.reshape((nsamples,nx*ny))
    d2_X_test = preprocessing.normalize(d2_X_test)

    clf = RandomForestClassifier()
    clf.fit(d2_X_train, y_train)
    pred = clf.predict(d2_X_test)
    acc = metrics.accuracy_score(y_test, pred)
    print(f'acc for commodity {commodity_index} is {acc}')
    avg_acc += acc

avg_acc /= 32
print(avg_acc)

