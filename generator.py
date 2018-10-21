import random
import numpy as np
from sklearn.model_selection import train_test_split


def cropped_generator(X, y, batch_size, crop_len, n_chans, n_classes):
    batch_X = np.zeros((batch_size, n_chans, crop_len, 1))
    batch_y = np.zeros((batch_size, n_classes))

    while True:
        for i in range(batch_size):
            index = random.randint(0, len(X) - 1)
            chosen_X = np.swapaxes(X[index], 0, 1)
            rand_ind = random.randint(0, len(chosen_X) - crop_len - 1)
            cut_X = chosen_X[rand_ind:rand_ind+crop_len]
            batch_X[i] = np.swapaxes(cut_X, 0, 1)
            batch_y[i] = y[index]
        yield batch_X, batch_y


def cropped_predictor(model, X_test, crop_len, n_classes):
    predictions = np.zeros((len(X_test), n_classes))
    for i in range(len(X_test)):
        X_test_swapped = np.swapaxes(X_test[i], 0, 1)
        preds = np.zeros((1, 4))
        X_combined = np.zeros((len(X_test_swapped) - crop_len, 22, crop_len, 1))
        for j in range(len(X_test_swapped) - crop_len):
            X_check_swapped = X_test_swapped[j:j+crop_len]
            X_check = np.swapaxes(X_check_swapped, 0, 1)
            X_combined[j] = X_check
        one_pred = np.mean(model.predict(X_combined), axis=0)
        predictions[i] = one_pred
    return predictions


def binary_example_generator():
    X_train = np.zeros((10, 2, 1000))  # create signal with 2 channels and 1000 samples
    y_train = np.zeros((10, 2))
    for i in range(5):  # set 5 zero examples
        X_train[i, 0, :] = np.ones((1, 1000))
        y_train[i] = [1, 0]

    for i in range(5, 10):  # set 5 one examples
        X_train[i, 1, :] = np.ones((1, 1000))
        y_train[i] = [0, 1]

    X_test = np.zeros((5, 2, 1000))  # create signal with 2 channels and 1000 samples
    y_test = np.zeros((5, 2))
    for i in range(3):  # set 3 zero examples
        X_test[i, 0, :] = np.ones((1, 1000))
        y_test[i] = [1, 0]

    for i in range(2, 5):  # set 2 one examples
        X_test[i, 1, :] = np.ones((1, 1000))
        y_test[i] = [0, 1]

    X_train = X_train[:, :, :, np.newaxis]
    X_test = X_test[:, :, :, np.newaxis]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    return X_train, y_train, X_val, y_val, X_test, y_test


def four_class_example_generator():
    X_train = np.zeros((40, 4, 1000))  # create signal with 4 channels and 1000 samples
    y_train = np.zeros((40, 4))
    for i in range(4):
        for j in range(10):
            X_train[i*10 + j, i, :] = np.ones((1, 1000))
            y_train[i*10 + j][i] = 1

    X_test = np.zeros((20, 4, 1000))  # create signal with 4 channels and 1000 samples
    y_test = np.zeros((20, 4))
    for i in range(4):
        for j in range(5):
            X_test[i * 5 + j, i, :] = np.ones((1, 1000))
            y_test[i * 5 + j][i] = 1

    X_train = X_train[:, :, :, np.newaxis]
    X_test = X_test[:, :, :, np.newaxis]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    return X_train, y_train, X_val, y_val, X_test, y_test