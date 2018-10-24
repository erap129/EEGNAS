import os
import time
import datetime
import numpy as np
import pandas as pd
import keras
import keras_models
import tensorflow as tf
import platform
from naiveNAS import NaiveNAS
import test_skip_connection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generator import binary_example_generator, four_class_example_generator
# import autosklearn.classification
import sklearn.metrics
from autokeras import ImageClassifier
# from autokeras import Image1DClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from BCI_IV_2a_loader import BCI_IV_2a
from collections import OrderedDict
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.splitters import split_into_two_sets
from datautil.signalproc import bandpass_cnt, exponential_running_standardize
from datautil.trial_segment import create_signal_target_from_raw_mne
from tpot import TPOTClassifier


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


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


def run_keras_model(train_set, test_set, row, cropped=False, mode='deep'):
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
    print('X_train.shape is:', X_train.shape)
    print('y_train.shape is:', y_train.shape)
    print('X_valid.shape is:', X_valid.shape)
    print('y_valid.shape is:', y_valid.shape)

    start = time.time()
    if mode == 'deep':
        model = keras_models.deep_model_mimic(train_set.X.shape[1], train_set.X.shape[2], 4, cropped=cropped)
    elif mode == 'shallow':
        model = keras_models.shallow_model_mimic(train_set.X.shape[1], train_set.X.shape[2], 4, cropped=cropped)
    if cropped:
        model = keras_models.convert_to_dilated(model)

    model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid), callbacks=[earlystopping, mcp])
    model.load_weights('best_keras_model.hdf5')
    end = time.time()

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


def run_autokeras_model(X_train, y_train, X_test, y_test, row=None):
    print('--------------running auto-keras model--------------')
    # if len(train_set.shape) == 3 and len(test_set.shape) == 3:
    #     X_train = train_set.X[:, :, :, np.newaxis]
    #     X_test = test_set.X[:, :, :, np.newaxis]
    # else:
    #     X_train = train_set.X
    #     X_test = test_set.X
    # X_train = np.swapaxes(X_train, 3, 1)[:, :, :, :]
    # X_test = np.swapaxes(X_test, 3, 1)[:, :, :, :]
    print("X_train.shape is: %s" % (str(X_train.shape)))
    start = time.time()
    clf = ImageClassifier(verbose=True, searcher_args={'trainer_args': {'max_iter_num': 5}})
    clf.fit(X_train, y_train, time_limit=12 * 60 * 60)
    clf.final_fit(X_train, y_train, X_test, y_test, retrain=False)
    end = time.time()
    y = clf.evaluate(X_test, y_test)
    print('autokeras result:', row * 100)
    if row is not None:
        row = np.append(row, str(y * 100))
        row = np.append(row, str(end - start))
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


def run_exp(train_set, test_set, subject, toggle):
    now = str(datetime.datetime.now()).replace(":", "-")
    row = np.array([])
    row = np.append(row, now)
    row = np.append(row, str(subject))
    for config in toggle.keys():
        if config == 'keras' and toggle[config]:
            row = run_keras_model(train_set, test_set, row, cropped=False, mode='deep')

        elif config == 'tpot' and toggle[config]:
            row = run_tpot_model(train_set, test_set, row)

        elif config == 'auto-keras' and toggle[config]:
            row = run_autokeras_model(train_set, test_set, row)

        elif config == 'auto-sklearn' and toggle[config]:
            row = run_auto_sklearn_model(train_set, test_set, row)
    print('row is:', row)
    return row


def automl_comparison():
    data_folder = 'data/'
    low_cut_hz = 0
    results = pd.DataFrame(columns=['date', 'subject'])
    toggle = {'keras': True, 'tpot': False, 'auto-keras': False, 'auto-sklearn': False}
    for setting in toggle.keys():
        if toggle[setting]:
            results[setting+'_acc'] = None
            results[setting+'_runtime'] = None

    for subject_id in range(1, 10):
        train_set, test_set = get_train_test(data_folder, subject_id, low_cut_hz)
        row = run_exp(train_set, test_set, subject_id, toggle)
        results.loc[subject_id - 1] = row

    now = str(datetime.datetime.now()).replace(":", "-")
    header = True
    if os.path.isfile('results' + now + '.csv'):
        header = False
    results.to_csv('results/results' + now + '.csv', mode='a', header=header)


def spectrogram_autokeras():
    global data_folder
    train_set, test_set = get_train_test(data_folder, 1, 0)
    train_specs, test_specs = create_spectrograms_from_raw(train_set=train_set, test_set=test_set)
    print(train_specs.shape)
    print(train_specs)
    train_specs = train_specs[:, :, :, 0:10]
    test_specs = test_specs[:, :, :, 0:10]
    print('train_specs.shape is:', train_specs.shape)
    run_autokeras_model(train_specs[:10], train_set.y[:10], test_specs[:10], test_set.y[:10])


def handle_subject_data(subject_id):
    train_set, test_set = get_train_test(data_folder, subject_id, 0)
    train_set, valid_set = split_into_two_sets(
        train_set, first_set_fraction=1 - valid_set_fraction)
    X_train = train_set.X[:, :, :, np.newaxis]
    X_valid = valid_set.X[:, :, :, np.newaxis]
    X_test = test_set.X[:, :, :, np.newaxis]
    y_train = to_categorical(train_set.y, num_classes=4)
    y_valid = to_categorical(valid_set.y, num_classes=4)
    y_test = to_categorical(test_set.y, num_classes=4)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def run_naive_nas(real_data=True, toy_data=False):
    global data_folder, valid_set_fraction
    now = str(datetime.datetime.now()).replace(":", "-")
    experiment_name = 'filter_experiment'
    folder_name = experiment_name+'_'+now
    createFolder(folder_name)
    accuracies = np.zeros(9)
    if real_data:
        for subject_id in range(1, 10):
            X_train, y_train, X_valid, y_valid, X_test, y_test = handle_subject_data(subject_id)
            naiveNAS = NaiveNAS(n_classes=4, input_time_len=1125, n_chans=22,
                                X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
                                X_test=X_test, y_test=y_test, subject_id=subject_id, cropping=False)
            accuracies[subject_id-1] = naiveNAS.find_best_model(folder_name, 'filter_experiment')
        np.savetxt('results/' + folder_name + '/accuracies.csv', accuracies, delimiter=',')

    if toy_data:
        X_train, y_train, X_val, y_val, X_test, y_test = four_class_example_generator()
        print(X_test)
        print(y_test)
        naiveNAS = NaiveNAS(n_classes=4, input_time_len=1000, n_chans=4,
                            X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val,
                            X_test=X_test, y_test=y_test)
        naiveNAS.find_best_model('filter_experiment')


def run_grid_search():
    X_train, y_train, X_valid, y_valid, X_test, y_test = handle_subject_data(subject_id)
    naiveNAS = NaiveNAS(n_classes=4, input_time_len=1125, n_chans=22,
                        X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
                        X_test=X_test, y_test=y_test, subject_id=subject_id, cropping=False)
    naiveNAS.grid_search()


def test_skip_connections():
    train_set, test_set = get_train_test(data_folder, 1, 0)
    train_set, valid_set = split_into_two_sets(
        train_set, first_set_fraction=1 - valid_set_fraction)
    skip_model = test_skip_connection.skip_model(train_set.X.shape[1], train_set.X.shape[2], 4)



if __name__ == '__main__':
    global data_folder, valid_set_fraction

    if platform.node() == 'nvidia':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)

    data_folder = 'data/'
    low_cut_hz = 0
    valid_set_fraction = 0.2
    # run_naive_nas()
    # test_skip_connections()
    # automl_comparison()
    run_grid_search()
