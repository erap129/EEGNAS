import pandas as pd
import numpy as np


def get_x_y(vals):
    X = []
    y = []
    for i in range(1000, len(vals) - 1):
        X.append(vals[i-1000:i])
        y.append(np.array([i > j for i, j in zip(vals[i], vals[i - 1])])[0])
    return np.array(X), np.array(y).astype(int)


df = pd.read_csv('PriceDate.csv')
df = df.drop(['Date'], axis=1)
train_vals = df.iloc[:4000].values
test_vals = df.iloc[4000:].values

X_train, y_train = get_x_y(train_vals)
X_train = np.swapaxes(X_train, 1, 2)
X_test, y_test = get_x_y(test_vals)
X_test = np.swapaxes(X_test, 1, 2)
np.save('X_train_gold.npy', X_train)
np.save('y_train_gold.npy', y_train)
np.save('X_test_gold.npy', X_test)
np.save('y_test_gold.npy', y_test)