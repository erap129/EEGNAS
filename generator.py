import random
import numpy as np


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
        print('X_test_swapped.shape is', X_test_swapped.shape)
        preds = np.zeros((1, 4))
        X_combined = np.zeros((len(X_test_swapped) - crop_len, 22, crop_len, 1))
        for j in range(len(X_test_swapped) - crop_len):
            X_check_swapped = X_test_swapped[j:j+crop_len]
            X_check = np.swapaxes(X_check_swapped, 0, 1)
            X_combined[j] = X_check
        one_pred = np.mean(model.predict(X_combined), axis=0)
        predictions[i] = one_pred
    print('predictions is:', predictions)
    return predictions


