import os
from braindecode.datasets.bbci import BBCIDataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
matplotlib.use('Agg')
from BCI_IV_2a_loader import BCI_IV_2a
from collections import OrderedDict
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.datautil.signalproc import bandpass_cnt, exponential_running_standardize, highpass_cnt
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from sklearn.model_selection import train_test_split
from moabb.datasets import Cho2017, BNCI2014004
from moabb.paradigms import (LeftRightImagery, MotorImagery,
                             FilterBankMotorImagery)
from data.Bloomberg.bloomberg_preproc import get_bloomberg
from data.HumanActivity.human_activity_preproc import get_human_activity
from data.Opportunity.sliding_window import sliding_window
from sklearn import preprocessing
import pickle
import globals
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def get_ch_names():
    train_filename = 'A01T.gdf'
    test_filename = 'A01E.gdf'
    train_filepath = os.path.join('../data/BCI_IV/', train_filename)
    test_filepath = os.path.join('../data/BCI_IV/', test_filename)
    train_label_filepath = train_filepath.replace('.gdf', '.mat')

    train_loader = BCI_IV_2a(train_filepath, train_label_filepath)
    train_cnt = train_loader.load()
    return train_cnt.ch_names


def get_bci_iv_2a_train_val_test(data_folder,subject_id, low_cut_hz):
    ival = [-500, 4000]  # this is the window around the event from which we will take data to feed to the classifier
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
    train_set, valid_set = split_into_two_sets(
        train_set, first_set_fraction=1 - globals.get('valid_set_fraction'))

    return train_set, valid_set, test_set


def load_bbci_data(filename, low_cut_hz, debug=False):
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)

    log.info("Loading data...")
    cnt = loader.load()

    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                  clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))

    # now pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)

    # Further preprocessings as descibed in paper
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    # Trial interval, start at -500 already, since improved decoding for networks
    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset


def load_train_valid_test_bbci(
        train_filename, test_filename, low_cut_hz, debug=False):
    log.info("Loading train...")
    full_train_set = load_bbci_data(
        train_filename, low_cut_hz=low_cut_hz, debug=debug)

    log.info("Loading test...")
    test_set = load_bbci_data(
        test_filename, low_cut_hz=low_cut_hz, debug=debug)
    valid_set_fraction = 0.8
    train_set, valid_set = split_into_two_sets(full_train_set,
                                               valid_set_fraction)

    log.info("Train set with {:4d} trials".format(len(train_set.X)))
    if valid_set is not None:
        log.info("Valid set with {:4d} trials".format(len(valid_set.X)))
    log.info("Test set with  {:4d} trials".format(len(test_set.X)))

    return train_set, valid_set, test_set


def get_hg_train_val_test(data_folder, subject_id, low_cut_hz):
    return load_train_valid_test_bbci(f"{data_folder}train/{subject_id}.mat",
                                      f"{data_folder}test/{subject_id}.mat",
                                        low_cut_hz)


class DummySignalTarget:
    def __init__(self, X, y, y_type=np.longlong):
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=y_type)


def get_ner_train_val_test(data_folder):
    X = np.load(f"{data_folder}NER15/preproc/epochs.npy")
    y, User = np.load(f"{data_folder}NER15/preproc/infos.npy")
    X_test = np.load(f"{data_folder}NER15/preproc/test_epochs.npy")
    y_test = pd.read_csv(f"{data_folder}NER15/preproc/true_labels.csv").values.reshape(-1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=globals.get('valid_set_fraction'))
    train_set = DummySignalTarget(X_train, y_train)
    valid_set = DummySignalTarget(X_val, y_val)
    test_set = DummySignalTarget(X_test, y_test)
    return train_set, valid_set, test_set


def get_cho_train_val_test(subject_id):
    paradigm = LeftRightImagery()
    dataset = Cho2017()
    subjects = [subject_id]
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    y = np.where(y=='left_hand', 0, y)
    y = np.where(y=='right_hand', 1, y)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3) # test = 30%, same as paper
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=globals.get('valid_set_fraction'))
    train_set = DummySignalTarget(X_train, y_train)
    valid_set = DummySignalTarget(X_val, y_val)
    test_set = DummySignalTarget(X_test, y_test)
    return train_set, valid_set, test_set


def get_bci_iv_2b_train_val_test(subject_id):
    paradigm = LeftRightImagery()
    dataset = BNCI2014004()
    subjects = [subject_id]
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    y = np.where(y=='left_hand', 0, y)
    y = np.where(y=='right_hand', 1, y)
    train_indexes = np.logical_or(metadata.values[:, 1] == 'session_0',
                                       metadata.values[:, 1] == 'session_1',
                                       metadata.values[:, 1] == 'session_2')
    test_indexes = np.logical_or(metadata.values[:, 1] == 'session_3',
                                       metadata.values[:, 1] == 'session_4')
    X_train_val = X[train_indexes]
    y_train_val = y[train_indexes]
    X_test = X[test_indexes]
    y_test = y[test_indexes]
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=globals.get('valid_set_fraction'))
    train_set = DummySignalTarget(X_train, y_train)
    valid_set = DummySignalTarget(X_val, y_val)
    test_set = DummySignalTarget(X_test, y_test)
    return train_set, valid_set, test_set


def get_bloomberg_train_val_test(data_folder):
    X_train_val, y_train_val, X_test, y_test = get_bloomberg(f'{data_folder}/Bloomberg')
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=globals.get('valid_set_fraction'))
    train_set = DummySignalTarget(X_train, y_train)
    valid_set = DummySignalTarget(X_val, y_val)
    test_set = DummySignalTarget(X_test, y_test)
    return train_set, valid_set, test_set


def get_nyse_train_val_test(data_folder):
    df = pd.read_csv(f"{data_folder}/NYSE/prices-split-adjusted.csv", index_col=0)
    df["adj close"] = df.close  # Moving close to the last column
    df.drop(['close'], 1, inplace=True)  # Moving close to the last column
    df2 = pd.read_csv(f"{data_folder}/NYSE/fundamentals.csv")
    symbols = list(set(df.symbol))

    df = df[df.symbol == 'GOOG']
    df.drop(['symbol'], 1, inplace=True)
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1, 1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1, 1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1, 1))
    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1, 1))
    df['adj close'] = min_max_scaler.fit_transform(df['adj close'].values.reshape(-1, 1))
    amount_of_features = len(df.columns)  # 5
    data = df.as_matrix()
    sequence_length = 200 + 1  # index starting from 0
    result = []

    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.7 * result.shape[0])  # 90% split
    train = result[:int(row), :]  # 90% date, all features

    X_train_val = train[:, :-1]
    y_train_val = np.array([i > j for i,j in zip(train[:, -1][:, -1], train[:, -2][:, -1])]).astype(int) # did the price go up?

    X_test = result[int(row):, :-1]
    y_test = np.array([i > j for i,j in zip(result[int(row):, -1][:, -1], result[int(row):, -2][:, -1])]).astype(int)

    X_train_val = np.reshape(X_train_val, (X_train_val.shape[0], amount_of_features, X_train_val.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], amount_of_features, X_test.shape[1]))
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=0.2)
    train_set = DummySignalTarget(X_train, y_train)
    valid_set = DummySignalTarget(X_val, y_val)
    test_set = DummySignalTarget(X_test, y_test)
    return train_set, valid_set, test_set


def get_human_activity_train_val_test(data_folder, subject_id):
    X_train_val, y_train_val, X_test, y_test = get_human_activity(f'{data_folder}/HumanActivity', subject_id)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=globals.get('valid_set_fraction'))
    train_set = DummySignalTarget(X_train, y_train)
    valid_set = DummySignalTarget(X_val, y_val)
    test_set = DummySignalTarget(X_test, y_test)
    return train_set, valid_set, test_set


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


def get_opportunity_train_val_test(data_folder):
    with open(f'{data_folder}/Opportunity/oppChallenge_gestures.data', 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        X_train, y_train = data[0]
        X_test, y_test = data[1]
        X_test, y_test = opp_sliding_window(X_test, y_test, 128, 12)
        X_train_val, y_train_val = opp_sliding_window(X_train, y_train, 128, 12)
        X_test = np.swapaxes(X_test, 1, 2)
        X_train_val = np.swapaxes(X_train_val, 1, 2)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                          test_size=globals.get('valid_set_fraction'))
        train_set = DummySignalTarget(X_train, y_train)
        valid_set = DummySignalTarget(X_val, y_val)
        test_set = DummySignalTarget(X_test, y_test)
        return train_set, valid_set, test_set


def get_sonarsub_train_val_test(data_folder):
    X = np.load(f"{data_folder}SonarSub/data_file_aug.npy")
    y = np.load(f"{data_folder}SonarSub/labels_file_aug.npy")
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=globals.get('valid_set_fraction'))
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=globals.get('valid_set_fraction'))
    train_set = DummySignalTarget(X_train, y_train)
    valid_set = DummySignalTarget(X_val, y_val)
    test_set = DummySignalTarget(X_test, y_test)
    return train_set, valid_set, test_set

def get_train_val_test(data_folder, subject_id, low_cut_hz):
    if globals.get('dataset') == 'BCI_IV_2a':
        return get_bci_iv_2a_train_val_test(f"{data_folder}BCI_IV/", subject_id, low_cut_hz)
    elif globals.get('dataset') == 'HG':
        return get_hg_train_val_test(f"{data_folder}HG/", subject_id, low_cut_hz)
    elif globals.get('dataset') == 'NER15':
        return get_ner_train_val_test(data_folder)
    elif globals.get('dataset') == 'Cho':
        return get_cho_train_val_test(subject_id)
    elif globals.get('dataset') == 'BCI_IV_2b':
        return get_bci_iv_2b_train_val_test(subject_id)
    elif globals.get('dataset') == 'Bloomberg':
        return get_bloomberg_train_val_test(data_folder)
    elif globals.get('dataset') == 'NYSE':
        return get_nyse_train_val_test(data_folder)
    elif globals.get('dataset') == 'HumanActivity':
        return get_human_activity_train_val_test(data_folder, subject_id)
    elif globals.get('dataset') == 'Opportunity':
        return get_opportunity_train_val_test(data_folder)
    elif globals.get('dataset') == 'SonarSub':
        return get_sonarsub_train_val_test(data_folder)