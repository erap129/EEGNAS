from copy import deepcopy

import torch
from braindecode.torch_ext.util import np_to_var
import global_vars
import numpy as np


def get_dummy_input():
    input_shape = (2, global_vars.get('eeg_chans'), global_vars.get('input_time_len'), 1)
    return np_to_var(np.random.random(input_shape).astype(np.float32))


def prepare_data_for_NN(X):
    if X.ndim == 3:
        X = X[:, :, :, None]
    X = np_to_var(X, pin_memory=global_vars.get('pin_memory'))
    if torch.cuda.is_available():
        with torch.cuda.device(0):
            X = X.cuda()
    return X


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
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