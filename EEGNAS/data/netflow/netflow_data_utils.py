import os

import pywt
from datetime import datetime
import json
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from EEGNAS import global_vars
from EEGNAS.utilities.data_utils import split_parallel_sequences, unison_shuffled_copies, split_sequence
plt.interactive(False)
import pandas as pd


def preprocess_netflow_data(files, n_before, n_ahead, jumps, buffer):
    all_X = []
    all_y = []
    all_datetimes_X = []
    all_datetimes_Y = []
    for file in files:
        all_data = get_whole_netflow_data(file)
        all_data.fillna(method='ffill', inplace=True)
        all_data.fillna(method='bfill', inplace=True)
        sample_list, y = split_parallel_sequences(all_data.values, n_before, n_ahead, jumps, buffer)
        datetimes_X, datetimes_Y = split_sequence(all_data.index, n_before, n_ahead, jumps, buffer)
        all_datetimes_X.extend(datetimes_X)
        all_datetimes_Y.extend(datetimes_Y)
        num_handovers = sample_list.shape[2] - 1
        all_X.extend(sample_list.swapaxes(1, 2)[:, :num_handovers])
        all_y.extend(y.swapaxes(1, 2)[:, num_handovers])
    max_handovers = global_vars.get('max_handovers')
    if not max_handovers:
        max_handovers = max(x.shape[0] for x in all_X)
    for idx in range(len(all_X)):
        if all_X[idx].shape[0] < max_handovers:
            all_X[idx] = np.pad(all_X[idx], pad_width=((max_handovers-all_X[idx].shape[0],0),(0,0)), mode='constant')
        elif all_X[idx].shape[0] > max_handovers:
            all_X[idx] = all_X[idx][:max_handovers]
    return np.stack(all_X, axis=0), np.stack(all_y, axis=0), \
           np.stack(all_datetimes_X, axis=0), np.stack(all_datetimes_Y, axis=0)


def get_whole_netflow_data(file):
    orig_df = pd.read_csv(file)
    own_as_num = os.path.basename(file).split('_')[0]
    vols = {}
    dfs = []
    for index, row in orig_df.iterrows():
        vols[row['id']] = [row['ts'], row['vol']]
    idx = 0
    for key, value in vols.items():
        datetimes = [datetime.utcfromtimestamp(int(tm)) for tm in json.loads(value[0])]
        df = pd.DataFrame(list(zip(datetimes, json.loads(value[1]))), columns=['ts', orig_df.iloc[idx]['id']])
        df = df.sort_values(by='ts')
        df.index = pd.to_datetime(df['ts'])
        df = df.drop(columns=['ts'])
        df = df.resample('H').pad()
        dfs.append(df)
        idx += 1
    all_data = pd.concat(dfs, axis=1)
    all_data = all_data.dropna(axis=1, how='any')
    all_data['sum'] = all_data.drop(labels=int(own_as_num), axis=1, errors='ignore').sum(axis=1)
    all_data = all_data[np.flatnonzero(df.index.hour == global_vars.get('start_hour'))[0]:]
    return all_data


def get_netflow_threshold(file, stds):
    df = get_whole_netflow_data(file)
    return df['sum'].mean() + df['sum'].std() * stds


def get_time_freq(signal):
    scales = range(1, 64)
    waveletname = 'morl'
    coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
    coeff = coeff[:,:63]
    return coeff


def turn_dataset_to_timefreq(X):
    new_X = np.zeros((len(X), 10, 63, 63))
    for ex_idx, example in enumerate(X):
        for sig_idx, signal in enumerate(example):
            new_X[ex_idx, sig_idx] = get_time_freq(signal)
        print(f'converted example {ex_idx}/{len(X)} into TF')
    return new_X


def turn_netflow_into_classification(X, y, threshold, oversampling=True):
    res = []
    for example in y:
        found = False
        for measurement in example:
            if measurement > threshold:
                res.append(1)
                found = True
                break
        if not found:
            res.append(0)
    assert(len(res) == len(y))
    if oversampling:
        ros = RandomOverSampler(random_state=0)
        X_reshaped = X.reshape(X.shape[0], -1)
        X_oversample, res_oversample = ros.fit_sample(X_reshaped, res)
        X_oversample = X_oversample.reshape(-1, X.shape[1], X.shape[2])
        return unison_shuffled_copies(X_oversample, res_oversample)
    return np.array(X), np.array(res)


def count_overflows_in_data(dataset, threshold, start_idx=0, end_idx=24):
    overflow_count = 0
    for example in dataset.y:
        for measurement in example[start_idx:end_idx]:
            if measurement > threshold:
                overflow_count += 1
                break
    return overflow_count






