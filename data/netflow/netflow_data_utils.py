from datetime import datetime
import json
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from utilities.data_utils import split_parallel_sequences, unison_shuffled_copies

plt.interactive(False)
import pandas as pd


def fix_netflow_days(times, measurements):
    prev_time = times[0]
    for idx, time in enumerate(times[1:]):
        if time.hour != (prev_time.hour + 1) % 24:
            time_diff = (time.hour - prev_time.hour) % 24
            for m_idx, measurement_series in enumerate(measurements):
                measurements[m_idx] = np.insert(measurement_series, idx+1, [0 for i in range(time_diff)], axis=0)
        prev_time = time
    return measurements


def preprocess_netflow_data(file, n_before, n_ahead, start_point, jumps, fix_days=True):
    df = pd.read_csv(file)
    vols = {}
    data_sample = []
    for index, row in df.iterrows():
        vols[row['id']] = [row['ts'], row['vol']]
    for key, value in vols.items():
        df = pd.DataFrame([json.loads(value[0]), json.loads(value[1])], index=['ts', 'vol']).T
        df = df.sort_values(by='ts')
        data_sample.append(np.array(df['vol']))
    sum_arr = [sum(x) for x in zip(*data_sample)]
    data_sample.append(np.array(sum_arr))
    data_time = [datetime.utcfromtimestamp(int(tm)) for tm in df['ts']]
    if fix_days:
        data_sample = fix_netflow_days(data_time, data_sample)
    data_sample = list(map(lambda x: x.reshape((len(x), 1)), data_sample))
    dataset = np.hstack(data_sample)
    sample_list, y = split_parallel_sequences(dataset, n_before, n_ahead, start_point, jumps)
    X = sample_list.swapaxes(1, 2)[:, :10]
    y = y.swapaxes(1, 2)[:, 10]
    return X, y


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
        # plt.plot(example)
        # plt.show()
        for measurement in example[start_idx:end_idx]:
            if measurement > threshold:
                overflow_count += 1
                break
    return overflow_count






