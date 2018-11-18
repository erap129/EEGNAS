import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import configparser
from keras.utils import to_categorical
from BCI_IV_2a_loader import BCI_IV_2a
from collections import OrderedDict
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.splitters import split_into_two_sets
from datautil.signalproc import bandpass_cnt, exponential_running_standardize
from datautil.trial_segment import create_signal_target_from_raw_mne


def get_train_test(data_folder, subject_id, low_cut_hz, model=None):
    ival = [-500, 4000]  # this is the window around the event from which we will take data to feed to the classifier
    input_time_length = 1000
    max_epochs = 800  # max runs of the NN
    max_increase_epochs = 80  # ???
    batch_size = 60  # 60 examples will be processed each run of the NN
    high_cut_hz = 38  # cut off parts of signal higher than 38 hz
    factor_new = 1e-3  # ??? has to do with exponential running standardize
    init_block_size = 1000  # ???

    train_filename = 'A{:02d}T.gdf'.format(subject_id)
    test_filename = 'A{:02d}E.gdf'.format(subject_id)
    train_filepath = os.path.join(data_folder, train_filename)
    test_filepath = os.path.join(data_folder, test_filename)
    train_label_filepath = train_filepath.replace('.gdf', '.mat')
    test_label_filepath = test_filepath.replace('.gdf', '.mat')

    train_loader = BCI_IV_2a(train_filepath, train_label_filepath)
    test_loader = BCI_IV_2a(test_filepath, test_label_filepath)
    train_cnt = train_loader.load()
    print('raw training data is:', train_cnt.get_data())
    test_cnt = test_loader.load()

    train_cnt = train_cnt.drop_channels(['STI 014', 'EOG-left',
                                         'EOG-central', 'EOG-right'])
    assert len(train_cnt.ch_names) == 22

    # convert measurements to millivolt
    train_cnt = mne_apply(lambda a: a * 1e6, train_cnt)
    train_cnt = mne_apply(  # signal processing procedure that I don't understand
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    train_cnt = mne_apply(  # signal processing procedure that I don't understand
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        train_cnt)

    test_cnt = test_cnt.drop_channels(['STI 014', 'EOG-left',
                                         'EOG-central', 'EOG-right'])
    assert len(test_cnt.ch_names) == 22

    # convert measurements to millivolt
    test_cnt = mne_apply(lambda a: a * 1e6, test_cnt)
    test_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, test_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), test_cnt)
    test_cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=factor_new,
                                                  init_block_size=init_block_size,
                                                  eps=1e-4).T,
        test_cnt)

    marker_def = OrderedDict([('Left Hand', [1]), ('Right Hand', [2],),
                              ('Foot', [3]), ('Tongue', [4])])

    train_set = create_signal_target_from_raw_mne(train_cnt, marker_def, ival)
    test_set = create_signal_target_from_raw_mne(test_cnt, marker_def, ival)

    return train_set, test_set


def show_spectrogram(data):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(256/96, 256/96)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(data, NFFT=256, Fs=250)
    fig.canvas.draw()
    plt.show()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    print('data.shape is:', data.shape)


def create_all_spectrograms(dataset, im_size=256):
    n_chans = len(dataset.X[1])
    specs = np.zeros((dataset.X.shape[0], im_size, im_size, n_chans * 3))
    for i, trial in enumerate(dataset.X):
        for j, channel in enumerate(trial):
            fig = plt.figure(frameon=False)
            fig.set_size_inches((im_size - 10) / 96, (im_size - 10) / 96)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.specgram(channel, NFFT=256, Fs=250)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            specs[i, :, :, 3 * j:3 * (j + 1)] = data
            plt.close(fig)
        if i % 10 == 0:
            print('finished trial:', i)
    return specs


def create_spectrograms_from_raw(train_set, test_set):
    return create_all_spectrograms(train_set), create_all_spectrograms(test_set)


class cropped_set:
    def __init__(self, X, y):
        self.X = X
        self.y = y


def create_supercrops(train_set, test_set, crop_size):
    trial_len = train_set.X.shape[2]
    print('trial_len is:', trial_len)
    ncrops_per_trial = trial_len/crop_size
    if ncrops_per_trial % crop_size != 0:
        ncrops_per_trial += 1
    X_train_crops = int(train_set.X.shape[0] * ncrops_per_trial)
    X_trial_len = crop_size
    nchans = 22
    X_test_crops = int(test_set.X.shape[0] * ncrops_per_trial)
    new_train_set_X = np.zeros((X_train_crops, nchans, X_trial_len))
    new_train_set_y = np.zeros(X_train_crops)
    new_test_set_X = np.zeros((X_test_crops, nchans, X_trial_len))
    new_test_set_y = np.zeros(X_test_crops)
    for i, trial in enumerate(train_set.X):
        curr_loc = int(ncrops_per_trial * i)
        new_train_set_X[curr_loc] = trial[:, 0:crop_size]
        new_train_set_X[curr_loc + 1] = trial[:, trial_len - crop_size:]
        new_train_set_y[curr_loc] = train_set.y[i]
        new_train_set_y[curr_loc + 1] = train_set.y[i]
    for i, trial in enumerate(test_set.X):
        curr_loc = int(ncrops_per_trial * i)
        new_test_set_X[curr_loc] = trial[:, 0:crop_size]
        new_test_set_X[curr_loc + 1] = trial[:, trial_len - crop_size:]
        new_test_set_y[curr_loc] = test_set.y[i]
        new_test_set_y[curr_loc + 1] = test_set.y[i]
    return cropped_set(new_train_set_X, new_train_set_y), cropped_set(new_test_set_X, new_test_set_y)


def handle_subject_data(subject_id, cropping=False):
    config = configparser.ConfigParser()
    config.read('config.ini')
    train_set, test_set = get_train_test(config['DEFAULT']['data_folder'], subject_id, 0)
    if cropping:
        train_set, test_set = create_supercrops(train_set, test_set, crop_size=1000)
        print('train_set.X.shape is:', train_set.X.shape)
        print('train_set.y.shape is:', train_set.y.shape)
    train_set, valid_set = split_into_two_sets(
        train_set, first_set_fraction=1 - config['DEFAULT'].getfloat('valid_set_fraction'))
    X_train = train_set.X[:, :, :, np.newaxis]
    X_valid = valid_set.X[:, :, :, np.newaxis]
    X_test = test_set.X[:, :, :, np.newaxis]
    y_train = to_categorical(train_set.y, num_classes=4)
    y_valid = to_categorical(valid_set.y, num_classes=4)
    y_test = to_categorical(test_set.y, num_classes=4)
    return X_train, y_train, X_valid, y_valid, X_test, y_test