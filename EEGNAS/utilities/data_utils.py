import os
from copy import deepcopy

import mne
import torch
from braindecode.torch_ext.util import np_to_var
from mne.time_frequency import tfr_morlet
from oct2py import octave

from EEGNAS import global_vars
import numpy as np
from scipy.io import savemat
from PIL import Image
from EEGNAS.utilities.misc import create_folder
from sktime.utils.load_data import load_from_tsfile_to_dataframe


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


def export_data_to_file(dataset, format, out_folder, classes=None):
    create_folder(out_folder)
    for segment in dataset.keys():
        if classes is None:
            X_data = dataset[segment].X
            y_data = dataset[segment].y
            class_str = ''
        else:
            X_data = []
            y_data = []
            for class_idx in classes:
                X_data.extend(dataset[segment].X[np.where(dataset[segment].y == class_idx)])
                y_data.extend(dataset[segment].y[np.where(dataset[segment].y == class_idx)])
            class_str = f'_classes_{str(classes)}'
            X_data, y_data = unison_shuffled_copies(np.array(X_data), np.array(y_data))
        if format == 'numpy':
            np.save(f'{out_folder}/X_{segment}_{global_vars.get("dataset")}{class_str}', X_data)
            np.save(f'{out_folder}/y_{segment}_{global_vars.get("dataset")}{class_str}', y_data)
        elif format == 'matlab':
            X_data = np.transpose(X_data, [1, 2, 0])
            savemat(f'{out_folder}/X_{segment}_{global_vars.get("dataset")}{class_str}.mat', {'data': X_data})
            savemat(f'{out_folder}/y_{segment}_{global_vars.get("dataset")}{class_str}.mat', {'data': y_data})


def EEG_to_TF(dataset, out_folder, dim):
    ch_names = [str(i) for i in range(global_vars.get('eeg_chans'))]
    ch_types = ['eeg' for i in range(global_vars.get('eeg_chans'))]
    info = mne.create_info(ch_names=ch_names, sfreq=global_vars.get('frequency'), ch_types=ch_types)
    n_cycles = 7  # number of cycles in Morlet wavelet
    freqs = np.arange(7, 40, 1)  # frequencies of interest
    create_folder(out_folder)
    for segment in dataset.keys():
        TF_array = np.zeros((len(dataset[segment].X), global_vars.get('eeg_chans'), dim, dim))
        for ex_idx, example in enumerate(dataset[segment].X):
            epochs = mne.EpochsArray(example[None, :, :], info=info)
            power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                                    return_itc=True, decim=3, n_jobs=1)
            for ch_idx, channel in enumerate(ch_names):
                fig = power.plot([power.ch_names.index(channel)])
                fig.set_frameon(False)
                fig.delaxes(fig.axes[1])
                for ax in fig.axes:
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                fig.suptitle('')
                fig.savefig(f'{out_folder}/tmp.png', bbox_inches='tight')
                img = Image.open(f'{out_folder}/tmp.png').convert('LA').resize((dim, dim))
                TF_array[ex_idx, ch_idx] = np.array(img)[:,:,0]
            print(f'created TF {ex_idx}/{len(dataset[segment].X)} in {segment} data')
        os.remove(f'{out_folder}/tmp.png')
        np.save(f'{out_folder}/X_{segment}', TF_array)
        np.save(f'{out_folder}/y_{segment}', dataset[segment].y)


def EEG_to_TF_matlab(dataset, out_folder):
    octave.addpath('eeglab/functions/guifunc');
    octave.addpath('eeglab/functions/popfunc');
    octave.addpath('eeglab/functions/adminfunc');
    octave.addpath('eeglab/functions/sigprocfunc');
    octave.addpath('eeglab/functions/miscfunc');
    octave.addpath('eeglab/functions/timefreqfunc');
    create_folder(out_folder)
    for segment in dataset.keys():
        TF_array = np.zeros((len(dataset[segment].X), global_vars.get('eeg_chans'), 49, 50))
        for ex_idx, example in enumerate(dataset[segment].X):
            for ch_idx, channel in enumerate(example):
                tf = octave.newtimef(channel.reshape(1, -1), 1125, [0, 4500], 250, [3, 0.5],
                                     'baseline', 0, 'plotphase', 'off', 'padratio', 1, 'ntimesout', 50)
                TF_array[ex_idx, ch_idx] = tf
                print(f'created TF for example {ex_idx}/{len(dataset[segment].X)}, channel {ch_idx}/{len(example)} in {segment} data\n')
        np.save(f'{out_folder}/X_{segment}_{global_vars.get("dataset")}_TF_matlab', TF_array)
        np.save(f'{out_folder}/y_{segment}_{global_vars.get("dataset")}_TF_matlab', dataset[segment].y)


def tensor_to_eeglab(X, filepath):
    savemat(filepath, {'data': np.transpose(X.cpu().detach().numpy().squeeze(), [1, 2, 0])})


def sktime_to_numpy(file):
    X_ts, y = load_from_tsfile_to_dataframe(file)
    X = np.zeros((len(X_ts), len(X_ts.columns), len(X_ts.iloc[0]['dim_0'])))
    for i in range(len(X_ts)):
        for col_idx, col in enumerate(X_ts.columns):
            X[i, col_idx] = X_ts.iloc[i][col].values
    return X, y.astype(np.float).astype('int')-1


def set_global_vars_by_sktime(file):
    X_ts, y = load_from_tsfile_to_dataframe(file)
    global_vars.set('input_height', len(X_ts.iloc[0]['dim_0']))
    global_vars.set('eeg_chans', len(X_ts.columns))
    global_vars.set('n_classes', len(np.unique(y)))



