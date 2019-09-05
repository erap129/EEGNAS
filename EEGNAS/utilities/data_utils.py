from copy import deepcopy

import torch
from braindecode.torch_ext.util import np_to_var
from EEGNAS import global_vars
import numpy as np
from scipy.io import savemat


def get_dummy_input():
    input_shape = (2, global_vars.get('eeg_chans'), global_vars.get('input_height'), global_vars.get('input_width'))
    return np_to_var(np.random.random(input_shape).astype(np.float32))


def prepare_data_for_NN(X):
    if X.ndim == 3:
        X = X[:, :, :, None]
    X = np_to_var(X, pin_memory=global_vars.get('pin_memory'))
    if torch.cuda.is_available():
        with torch.cuda.device(0):
            X = X.cuda()
    return X


def split_sequence(sequence, n_steps, n_steps_ahead, start_point, jumps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix % jumps != 0:
            continue
        if end_ix + n_steps_ahead - 1 > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+n_steps_ahead]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def split_parallel_sequences(sequences, n_steps, n_steps_ahead, start_point, jumps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix % jumps != 0:
            continue
        if end_ix + n_steps_ahead - 1 > len(sequences) - 1:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:end_ix+n_steps_ahead, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def noise_input(data, devs_id):
    noise_data = deepcopy(data)
    for id_in_batch in range(data.shape[0]):
        noise_steps = np.random.choice(range(global_vars.get('steps')), size=int(global_vars.get('steps')
                                                                                 * global_vars.get('noise_ratio')), replace=False)
        b_mean = np.mean(data[id_in_batch])
        b_std = np.std(data[id_in_batch])
        for dev_id in range(len(devs_id)):
            noise_data[id_in_batch][dev_id][noise_steps] = np.random.normal(b_mean, b_std)
    return noise_data


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def calc_regression_accuracy(y_pred, y_real, threshold):
    actual = []
    predicted = []
    for yp, yr in zip(y_pred, y_real):
        predicted.append((yp > threshold).astype('int'))
        actual.append((yr > threshold).astype('int'))
    return actual, predicted


def write_dict(dict, filename):
    with open(filename, 'w') as f:
        all_keys = []
        for _, inner_dict in sorted(dict.items()):
            for K, _ in sorted(inner_dict.items()):
                all_keys.append(K)
        for K in all_keys:
            f.write(f"{K}\t{global_vars.get(K)}\n")


def export_data_to_file(dataset, format, classes=None):
    if classes is None:
        X_data = dataset.X
        y_data = dataset.y
        class_str = ''
    else:
        X_data = []
        y_data = []
        for class_idx in classes:
            X_data.extend(dataset.X[np.where(dataset.y == class_idx)])
            y_data.extend(dataset.y[np.where(dataset.y == class_idx)])
        class_str = f'_classes_{str(classes)}'
        X_data, y_data = unison_shuffled_copies(np.array(X_data), np.array(y_data))
    if format == 'numpy':
        np.save(f'data/export_data/X_all_{global_vars.get("dataset")}', X_data)
        np.save(f'data/export_data/y_all_{global_vars.get("dataset")}', y_data)
    elif format == 'matlab':
        X_data = np.transpose(X_data, [1, 2, 0])
        savemat(f'data/export_data/X_all_{global_vars.get("dataset")}{class_str}.mat', {'data': X_data})
        savemat(f'data/export_data/y_all_{global_vars.get("dataset")}{class_str}.mat', {'data': y_data})