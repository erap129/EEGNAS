import pandas as pd
import numpy as np
import logging
import glob
import mne
import re

import resampy
from braindecode.datautil.iterators import get_balanced_batches
from numpy.random.mtrand import RandomState

log = logging.getLogger(__name__)


def create_set(X, y, inds):
    """
    X list and y nparray
    :return:
    """
    new_X = []
    for i in inds:
        new_X.append(X[i])
    new_y = y[inds]
    return SignalAndTarget(new_X, new_y)


class TrainValidTestSplitter(object):
    def __init__(self, n_folds, i_test_fold, shuffle):
        self.n_folds = n_folds
        self.i_test_fold = i_test_fold
        self.rng = RandomState(39483948)
        self.shuffle = shuffle

    def split(self, X, y,):
        if len(X) < self.n_folds:
            raise ValueError("Less Trials: {:d} than folds: {:d}".format(
                len(X), self.n_folds
            ))
        folds = get_balanced_batches(len(X), self.rng, self.shuffle,
                                     n_batches=self.n_folds)
        test_inds = folds[self.i_test_fold]
        valid_inds = folds[self.i_test_fold - 1]
        all_inds = list(range(len(X)))
        train_inds = np.setdiff1d(all_inds, np.union1d(test_inds, valid_inds))
        assert np.intersect1d(train_inds, valid_inds).size == 0
        assert np.intersect1d(train_inds, test_inds).size == 0
        assert np.intersect1d(valid_inds, test_inds).size == 0
        assert np.array_equal(np.sort(
            np.union1d(train_inds, np.union1d(valid_inds, test_inds))),
            all_inds)

        train_set = create_set(X, y, train_inds)
        valid_set = create_set(X, y, valid_inds)
        test_set = create_set(X, y, test_inds)

        return train_set, valid_set, test_set


def create_preproc_functions(
        sec_to_cut_at_start, sec_to_cut_at_end, duration_recording_mins,
        max_abs_val, clip_before_resample, sampling_freq,
        divisor):
    preproc_functions = []
    if (sec_to_cut_at_start is not None) and (sec_to_cut_at_start > 0):
        preproc_functions.append(
            lambda data, fs: (data[:, int(sec_to_cut_at_start * fs):], fs))
    if (sec_to_cut_at_end is not None) and (sec_to_cut_at_end > 0):
        preproc_functions.append(
            lambda data, fs: (data[:, :-int(sec_to_cut_at_end * fs)], fs))
    preproc_functions.append(
        lambda data, fs: (
        data[:, :int(duration_recording_mins * 60 * fs)], fs))
    if (max_abs_val is not None) and (clip_before_resample):
        preproc_functions.append(lambda data, fs:
                                 (np.clip(data, -max_abs_val, max_abs_val),
                                  fs))

    preproc_functions.append(lambda data, fs: (resampy.resample(data, fs,
                                                                sampling_freq,
                                                                axis=1,
                                                                filter='kaiser_fast'),
                                               sampling_freq))
    if (max_abs_val is not None) and (not clip_before_resample):
        preproc_functions.append(lambda data, fs:
                                 (np.clip(data, -max_abs_val, max_abs_val),
                                  fs))
    if divisor is not None:
        preproc_functions.append(lambda data, fs: (data / divisor, fs))
    return preproc_functions


# ______________________________________________________________________________________________________________________
def natural_key(file_name):
    """ provides a human-like sorting key of a string """
    key = [int(token) if token.isdigit() else None for token in re.split(r'(\d+)', file_name)]
    return key


def session_key(file_name):
    """ sort the file name by session """
    return re.findall(r'(s\d*)_', file_name)


# ______________________________________________________________________________________________________________________
def time_key(file_name):
    """ provides a time-based sorting key """
    splits = file_name.split('/')
    [date] = re.findall(r'(\d{4}_\d{2}_\d{2})', splits[-2])
    date_id = [int(token) for token in date.split('_')]
    recording_id = natural_key(splits[-1])
    session_id = session_key(splits[-2])

    return date_id + session_id + recording_id


# ______________________________________________________________________________________________________________________
def read_all_file_names(path, extension, key="time"):
    """ read all files with specified extension from given path
    :param path: parent directory holding the files directly or in subdirectories
    :param extension: the type of the file, e.g. '.txt' or '.edf'
    :param key: the sorting of the files. natural e.g. 1, 2, 12, 21 (machine 1, 12, 2, 21) or by time since this is
    important for cv. time is specified in the edf file names
    """
    file_paths = glob.glob(path + '**/*' + extension, recursive=True)

    if key == 'time':
        return sorted(file_paths, key=time_key)

    elif key == 'natural':
        return sorted(file_paths, key=natural_key)


# ______________________________________________________________________________________________________________________
def fix_header(file_path):
    """ this was used to try to fix the corrupted header of recordings. not needed anymore since they were officially
    repaired by tuh
    """
    logging.warning("Couldn't open edf {}. Trying to fix the header ...".format(file_path))
    f = open(file_path, 'rb')
    content = f.read()
    f.close()

    header = content[:256]
    # print(header)

    # version = header[:8].decode('ascii')
    # patient_id = header[8:88].decode('ascii')
    # [age] = re.findall("Age:(\d+)", patient_id)
    # [sex] = re.findall("\s\w\s", patient_id)

    recording_id = header[88:168].decode('ascii')
    # startdate = header[168:176]
    # starttime = header[176:184]
    # n_bytes_in_header = header[184:192].decode('ascii')
    # reserved = header[192:236].decode('ascii')
    # THIS IS MESSED UP IN THE HEADER DESCRIPTION
    # duration = header[236:244].decode('ascii')
    # n_data_records = header[244:252].decode('ascii')
    # n_signals = header[252:].decode('ascii')

    date = recording_id[10:21]
    day, month, year = date.split('-')
    if month == 'JAN':
        month = '01'

    elif month == 'FEB':
        month = '02'

    elif month == 'MAR':
        month = '03'

    elif month == 'APR':
        month = '04'

    elif month == 'MAY':
        month = '05'

    elif month == 'JUN':
        month = '06'

    elif month == 'JUL':
        month = '07'

    elif month == 'AUG':
        month = '08'

    elif month == 'SEP':
        month = '09'

    elif month == 'OCT':
        month = '10'

    elif month == 'NOV':
        month = '11'

    elif month == 'DEC':
        month = '12'

    year = year[-2:]
    date = '.'.join([day, month, year])

    fake_time = '00.00.00'

    # n_bytes = int(n_bytes_in_header) - 256
    # n_signals = int(n_bytes / 256)
    # n_signals = str(n_signals) + '    '
    # n_signals = n_signals[:4]

    # new_header = version + patient_id + recording_id + date + fake_time + n_bytes_in_header + reserved +
    # new_header += n_data_records + duration + n_signals
    # new_content = (bytes(new_header, encoding="ascii") + content[256:])

    new_content = header[:168] + bytes(date + fake_time, encoding="ascii") + header[184:] + content[256:]

    # f = open(file_path, 'wb')
    # f.write(new_content)
    # f.close()


# ______________________________________________________________________________________________________________________
def get_patient_info(file_path):
    """ parse sex and age of patient from the patient_id in the header of the edf file
    :param file_path: path of the recording
    :return: sex (0=M, 1=F) and age of patient
    """
    f = open(file_path, 'rb')
    content = f.read()
    f.close()

    header = content[:88]
    patient_id = header[8:88].decode('ascii')
    # the headers "fixed" by tuh nedc data team show a '-' right before the age of the patient. therefore add this to
    # the regex and use the absolute value of the casted age
    [age] = re.findall("Age:(-?\d+)", patient_id)
    [sex] = re.findall("\s\w\s", patient_id)

    sex_id = 0 if sex.strip() == 'M' else 1

    return sex_id, abs(int(age))


# ______________________________________________________________________________________________________________________
def get_recording_length(file_path):
    """ some recordings were that huge that simply opening them with mne caused the program to crash. therefore, open
    the edf as bytes and only read the header. parse the duration from there and check if the file can safely be opened
    :param file_path: path of the directory
    :return: the duration of the recording
    """
    f = open(file_path, 'rb')
    header = f.read(256)
    f.close()

    return int(header[236:244].decode('ascii'))


# ______________________________________________________________________________________________________________________
def get_info_with_mne(file_path):
    """ read info from the edf file without loading the data. loading data is done in multiprocessing since it takes
    some time. getting info is done before because some files had corrupted headers or weird sampling frequencies
    that caused the multiprocessing workers to crash. therefore get and check e.g. sampling frequency and duration
    beforehand
    :param file_path: path of the recording file
    :return: file name, sampling frequency, number of samples, number of signals, signal names, duration of the rec
    """
    try:
        edf_file = mne.io.read_raw_edf(file_path, verbose='error')
    except ValueError:
        return None, None, None, None, None, None
        # fix_header(file_path)
        # try:
        #     edf_file = mne.io.read_raw_edf(file_path, verbose='error')
        #     logging.warning("Fixed it!")
        # except ValueError:
        #     return None, None, None, None, None, None

    # some recordings have a very weird sampling frequency. check twice before skipping the file
    sampling_frequency = int(edf_file.info['sfreq'])
    if sampling_frequency < 10:
        sampling_frequency = 1 / (edf_file.times[1] - edf_file.times[0])
        if sampling_frequency < 10:
            return None, sampling_frequency, None, None, None, None

    n_samples = edf_file.n_times
    signal_names = edf_file.ch_names
    n_signals = len(signal_names)
    # some weird sampling frequencies are at 1 hz or below, which results in division by zero
    duration = n_samples / max(sampling_frequency, 1)

    # TODO: return rec object?
    return edf_file, sampling_frequency, n_samples, n_signals, signal_names, duration


# ______________________________________________________________________________________________________________________
def load_data_with_mne(rec):
    """ loads the data using the mne library
    :param rec: recording object holding all necessary data of an eeg recording
    :return: a pandas dataframe holding the data of all electrodes as specified in the rec object
    """
    rec.raw_edf.load_data()
    signals = rec.raw_edf.get_data()

    data = pd.DataFrame(index=range(rec.n_samples), columns=rec.signal_names)
    for electrode_id, electrode in enumerate(rec.signal_names):
        data[electrode] = signals[electrode_id]

    # TODO: return rec object?
    return data


def load_data(fname, preproc_functions, sensor_types=['EEG']):
    cnt, sfreq, n_samples, n_channels, chan_names, n_sec = get_info_with_mne(
        fname)
    log.info("Load data...")
    cnt.load_data()
    selected_ch_names = []
    if 'EEG' in sensor_types:
        wanted_elecs = ['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1',
                        'FP2', 'FZ', 'O1', 'O2',
                        'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']

        for wanted_part in wanted_elecs:
            wanted_found_name = []
            for ch_name in cnt.ch_names:
                if ' ' + wanted_part + '-' in ch_name:
                    wanted_found_name.append(ch_name)
            assert len(wanted_found_name) == 1
            selected_ch_names.append(wanted_found_name[0])
    if 'EKG' in sensor_types:
        wanted_found_name = []
        for ch_name in cnt.ch_names:
            if 'EKG' in ch_name:
                wanted_found_name.append(ch_name)
        assert len(wanted_found_name) == 1
        selected_ch_names.append(wanted_found_name[0])

    cnt = cnt.reorder_channels(selected_ch_names)
    assert np.array_equal(cnt.ch_names, selected_ch_names), (
        "Actual chans:\n{:s}\nwanted chans:\n{:s}".format(
            str(cnt.ch_names), str(selected_ch_names)
        ))

    n_sensors = 0
    if 'EEG' in sensor_types:
        n_sensors += 21
    if 'EKG' in sensor_types:
        n_sensors += 1

    assert len(cnt.ch_names)  == n_sensors, (
        "Expected {:d} channel names, got {:d} channel names".format(
            n_sensors, len(cnt.ch_names)))

    # change from volt to mikrovolt
    data = (cnt.get_data() * 1e6).astype(np.float32)
    fs = cnt.info['sfreq']
    log.info("Preprocessing...")
    for fn in preproc_functions:
        log.info(fn)
        data, fs = fn(data, fs)
        data = data.astype(np.float32)
        fs = float(fs)
    return data

def get_all_sorted_file_names_and_labels(train_or_eval):
    normal_path = ('/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/'
        '{:s}/normal/'.format(train_or_eval))
    normal_file_names = read_all_file_names(normal_path, '.edf', key='time')
    abnormal_path = ('/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/edf/'
        '{:s}/abnormal/'.format(train_or_eval))
    abnormal_file_names = read_all_file_names(abnormal_path, '.edf', key='time')

    all_file_names = normal_file_names + abnormal_file_names

    all_file_names = sorted(all_file_names, key=time_key)

    abnorm_counts = [fname.count('abnormal') for fname in all_file_names]
    assert set(abnorm_counts) == set([1, 2])
    labels = np.array(abnorm_counts) == 2
    labels = labels.astype(np.int64)
    return all_file_names, labels

class DiagnosisSet(object):
    def __init__(self, n_recordings, max_recording_mins, preproc_functions,
                 train_or_eval='train', sensor_types=['EEG']):
        self.n_recordings = n_recordings
        self.max_recording_mins = max_recording_mins
        self.preproc_functions = preproc_functions
        self.train_or_eval = train_or_eval
        self.sensor_types = sensor_types

    def load(self, only_return_labels=False):
        log.info("Read file names")
        all_file_names, labels = get_all_sorted_file_names_and_labels(
            train_or_eval=self.train_or_eval)

        log.info("Read recording lengths...")

        # Read files in gradually until all used or until wanted number of
        # recordings reached
        cleaned_file_names = []
        cleaned_labels = []
        i_file = 0
        assert (self.n_recordings is None) or (self.n_recordings > 0)
        collected_all_wanted_files = self.n_recordings == 0
        all_files_used = False
        while (not collected_all_wanted_files) and (not all_files_used):
            fname = all_file_names[i_file]
            if self.max_recording_mins is not None:
                recording_length = get_recording_length(fname)
            if (self.max_recording_mins is None) or (
                recording_length < self.max_recording_mins * 60):
                cleaned_file_names.append(fname)
                cleaned_labels.append(labels[i_file])
            i_file += 1
            # also False if n_recordings is None, as expected
            collected_all_wanted_files = (
                self.n_recordings == len(cleaned_file_names))
            all_files_used = i_file == len(all_file_names)

        if self.n_recordings is not None:
            assert len(cleaned_file_names) == self.n_recordings
        assert len(cleaned_labels) == len(cleaned_file_names)

        if only_return_labels:
            return cleaned_labels

        X = []
        y = []
        n_files = len(cleaned_file_names)
        for i_fname, fname in enumerate(cleaned_file_names):
            log.info("Load {:d} of {:d}".format(i_fname + 1,n_files))
            x = load_data(fname, preproc_functions=self.preproc_functions,
                          sensor_types=self.sensor_types)
            assert x is not None
            X.append(x)
            y.append(cleaned_labels[i_fname])
        y = np.array(y)
        return X, y