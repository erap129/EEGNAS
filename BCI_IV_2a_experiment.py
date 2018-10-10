import os
import time
import datetime
import numpy as np
import pandas as pd
import keras
import keras_models
import tensorflow as tf
import platform
import io
from generator import cropped_generator, cropped_predictor
# import autosklearn.classification
import sklearn.metrics
from autokeras import ImageClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from BCI_IV_2a_loader import BCI_IV_2a
from collections import OrderedDict
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.splitters import split_into_two_sets
from datautil.signalproc import bandpass_cnt, exponential_running_standardize
from datautil.trial_segment import create_signal_target_from_raw_mne
from tpot import TPOTClassifier



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
    from scipy import signal
    import matplotlib.pyplot as plt

    # fs = 250
    # print('raw data is:', data)
    # print('length of raw data is:', len(data))
    # f, t, Sxx = signal.spectrogram(data, fs, nperseg=10)
    # plt.pcolormesh(t, f, Sxx)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    fig = plt.figure(frameon=False)
    fig.set_size_inches(256/96, 256/96)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(data, NFFT=256, Fs=250)
    buf = io.BytesIO()
    fig.savefig('testfig.png', dpi = 96)


def create_spectrograms_from_raw(train_set, test_set):
    n_chans = len(train_set[0]
    for trial in train_set.X:
        for channel in trial:
            fig = plt.figure(frameon=False)
            fig.set_size_inches(256 / 96, 256 / 96)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.specgram(channel, NFFT=256, Fs=250)
            buf = io.BytesIO()
            fig.savefig('testfig.png', dpi=96)


def run_keras_deep_model(train_set, test_set, row, cropped=False):
    print('--------------running keras model--------------')
    valid_set_fraction = 0.2
    print('train_set.X.shape is', train_set.X.shape)
    earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10)
    mcp = ModelCheckpoint('best_keras_model.hdf5', save_best_only=True, monitor='val_acc', mode='max')
    train_set_split, valid_set_split = split_into_two_sets(
        train_set, first_set_fraction=1 - valid_set_fraction)
    X_train = train_set_split.X[:, :, :, np.newaxis]
    X_valid = valid_set_split.X[:, :, :, np.newaxis]
    X_test = test_set.X[:, :, :, np.newaxis]
    y_train = to_categorical(train_set_split.y, num_classes=4)
    y_valid = to_categorical(valid_set_split.y, num_classes=4)
    start = time.time()
    if cropped:
        crop_len = 522
        model = keras_models.deep_model_cropped(train_set.X.shape[1],
                                                crop_len,
                                                4)
        train_generator = cropped_generator(X_train, y_train,
                                            32, crop_len, 22, 4)
        val_generator = cropped_generator(X_valid, y_valid,
                                          32, crop_len, 22, 4)
        model.fit_generator(generator=train_generator, steps_per_epoch=50, epochs=50,
                            validation_data=val_generator, validation_steps=20, callbacks=[earlystopping, mcp])
    else:
        model = keras_models.deep_model(train_set.X.shape[1],
                                        train_set.X.shape[2],
                                        4)
        print('X_train.shape is:', X_train.shape)
        print('y_train.shape is:', y_train.shape)
        print('X_valid.shape is:', X_valid.shape)
        print('y_valid.shape is:', y_valid.shape)
        model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid),
                  callbacks=[earlystopping, mcp])
    model.load_weights('best_keras_model.hdf5')
    end = time.time()
    if cropped:
        predictions = np.argmax(cropped_predictor(model, X_test, crop_len, n_classes=4), axis=1)
        res = sklearn.metrics.accuracy_score(predictions, test_set.y)
    else:
        y_test = to_categorical(test_set.y, num_classes=4)
        res = model.evaluate(X_test, y_test)[1] * 100
    print('accuracy for keras model:', res)
    print('runtime for keras model:', end - start)
    row = np.append(row, res)
    row = np.append(row, str(end - start))
    return row


def run_tpot_model(train_set, test_set, row):
    print('--------------running tpot model--------------')
    tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
    start = time.time()
    print(train_set.X.shape)
    print(train_set.y.shape)
    train_set_reshpX = train_set.X.reshape(train_set.X.shape[0], -1)
    test_set_reshpX = test_set.X.reshape(train_set.X.shape[0], -1)
    tpot.fit(train_set_reshpX, train_set.y)
    end = time.time()
    row = np.append(row, tpot.score(test_set_reshpX, test_set.y))
    row = np.append(row, str(end - start))
    return row


def run_autokeras_model(train_set, test_set, row):
    print('--------------running auto-keras model--------------')
    X_train = train_set.X[:, :, :, np.newaxis]
    X_train = np.swapaxes(X_train, 3, 1)
    X_test = test_set.X[:, :, :, np.newaxis]
    X_test = np.swapaxes(X_test, 3, 1)
    print("X_train.shape is: %s" % (str(X_train.shape)))
    start = time.time()
    clf = ImageClassifier(verbose=True, searcher_args={'trainer_args': {'max_iter_num': 5}})
    clf.fit(X_train, train_set.y, time_limit=12 * 60 * 60)
    clf.final_fit(X_train, train_set.y, X_test, test_set.y, retrain=False)
    end = time.time()
    y = clf.evaluate(X_test, test_set.y)
    row = np.append(row, str(y * 100))
    row = np.append(row, str(end - start))
    # clf.load_searcher().load_best_model().produce_keras_model().save('autokeras_model.h5')
    print(y)
    return row


def run_auto_sklearn_model(train_set, test_set, row):
    print('--------------running auto-sklearn model--------------')
    train_set_reshpX = train_set.X.reshape(train_set.X.shape[0], -1)
    test_set_reshpX = test_set.X.reshape(train_set.X.shape[0], -1)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='autosklearn_cv_example_tmp',
        output_folder='autosklearn_cv_example_out',
        delete_tmp_folder_after_terminate=True,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
    )
    # fit() changes the data in place, but refit needs the original data. We
    # therefore copy the data. In practice, one should reload the data
    start = time.time()
    automl.fit(train_set_reshpX.copy(), train_set.y.copy(), dataset_name='digits')
    # During fit(), models are fit on individual cross-validation folds. To use
    # all available data, we call refit() which trains all models in the
    # final ensemble on the whole dataset.
    automl.refit(train_set_reshpX.copy(), train_set.y.copy())
    end = time.time()
    predictions = automl.predict(test_set_reshpX)
    res = sklearn.metrics.accuracy_score(test_set.y, predictions)
    row = np.append(row, res)
    row = np.append(row, str(end - start))
    return row


def run_exp(train_set, test_set, subject):
    configs = ['keras', 'tpot', 'auto-keras', 'auto-sklearn']
    disabled = {'keras': True, 'tpot': True, 'auto-keras': False, 'auto-sklearn': True}
    now = str(datetime.datetime.now()).replace(":", "-")
    row = np.array([])
    row = np.append(row, now)
    row = np.append(row, str(subject))
    for config in configs:
        if disabled[config] is True:
            row = np.append(row, 0)
            row = np.append(row, 0)
            continue

        elif config == 'keras':
            row = run_keras_deep_model(train_set, test_set, row, cropped=False)

        elif config == 'tpot':
            row = run_tpot_model(train_set, test_set, row)

        elif config == 'auto-keras':
            row = run_autokeras_model(train_set, test_set, row)

        elif config == 'auto-sklearn':
            row = run_auto_sklearn_model(train_set, test_set, row)
    print('row is:', row)
    return row



if __name__ == '__main__':
    # if platform.node() == 'nvidia':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    #     config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
    #     sess = tf.Session(config=config)
    #     keras.backend.set_session(sess)

    data_folder = 'data/'
    low_cut_hz = 0

    spect_train, spect_test = get_train_test(data_folder, 1, 0)
    show_spectrogram(spect_train.X[0][20])

    # results = pd.DataFrame(columns=['date', 'subject', 'keras_acc', 'keras_runtime',
    #                                 'tpot_acc', 'tpot_runtime',
    #                                 'auto-keras_acc', 'auto-keras_runtime',
    #                                 'auto-sklearn_acc', 'auto-sklearn_runtime'])
    #
    # for subject_id in range(1, 10):
    #     train_set, test_set = get_train_test(data_folder, subject_id, low_cut_hz)
    #     row = run_exp(train_set, test_set, subject_id)
    #     results.loc[subject_id-1] = row
    #
    # now = str(datetime.datetime.now()).replace(":", "-")
    # header = True
    # if os.path.isfile('results'+now+'.csv'):
    #     header = False
    # results.to_csv('results'+now+'.csv', mode='a', header=header)