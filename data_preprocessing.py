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
from moabb.datasets import Cho2017
from moabb.paradigms import (LeftRightImagery, MotorImagery,
                             FilterBankMotorImagery)
import globals
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def get_bci_iv_train_val_test(data_folder,subject_id, low_cut_hz):
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
    def __init__(self, X, y):
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)


def get_ner_train_val_test(data_folder):
    X = np.load(f"{data_folder}NER15/preproc/epochs.npy")
    y, User = np.load(f"{data_folder}NER15/preproc/infos.npy")
    y = [[val, 1-val] for val in y]
    X_test = np.load(f"{data_folder}NER15/preproc/test_epochs.npy")
    y_test = pd.read_csv(f"{data_folder}NER15/preproc/true_labels.csv").values.reshape(-1)
    y_test = [[val, 1 - val] for val in y_test]
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
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3) # test = 30%, same as paper
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      test_size=globals.get('valid_set_fraction'))
    train_set = DummySignalTarget(X_train, y_train)
    valid_set = DummySignalTarget(X_val, y_val)
    test_set = DummySignalTarget(X_test, y_test)
    return train_set, valid_set, test_set



def get_train_val_test(data_folder, subject_id, low_cut_hz):
    if globals.get('dataset') == 'BCI_IV':
        return get_bci_iv_train_val_test(f"{data_folder}BCI_IV/", subject_id, low_cut_hz)
    elif globals.get('dataset') == 'HG':
        return get_hg_train_val_test(f"{data_folder}HG/", subject_id, low_cut_hz)
    elif globals.get('dataset') == 'NER15':
        return get_ner_train_val_test(data_folder)
    elif globals.get('dataset') == 'Cho':
        return get_cho_train_val_test(subject_id)


