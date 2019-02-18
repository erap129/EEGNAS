import pandas as pd
import numpy as np


def get_x_y(vals):
    X = []
    y = []
    for i in range(1000, len(vals) - 1):
        X.append(vals[i-1000:i-50])
        # y.append(np.array([i > j for i, j in zip(np.average(vals[i-50:i], axis=0),
        #                                          np.average(vals[i-100:i - 50], axis=0))])[0])
        y.append(np.array([get_labels(vals, i, j) for i, j in zip(np.average(vals[i-50:i], axis=0),
                                                 np.average(vals[i-100:i - 50], axis=0))])[0])
    return np.array(X), np.array(y).astype(int)


def get_labels(all_vals, value_a, value_b):
    mean_val = np.mean(all_vals[:, 0])
    if value_a > 0.02 * mean_val + value_b:  # price gone up
        return 2
    elif value_b > 0.02 * mean_val + value_a:  # price gone down
        return 1
    else:  # price hasn't changed
        return 0

def get_bloomberg(data_folder):
    df = pd.read_csv(f'{data_folder}/PriceDate.csv')
    df = df.drop(['Date'], axis=1)
    train_vals = df.iloc[:4000].values
    test_vals = df.iloc[4000:].values

    X_train, y_train = get_x_y(train_vals)
    X_train = np.swapaxes(X_train, 1, 2)
    X_test, y_test = get_x_y(test_vals)
    X_test = np.swapaxes(X_test, 1, 2)
    return X_train, y_train, X_test, y_test
