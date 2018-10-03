import os
import time
import datetime
import numpy as np
import pandas as pd
import keras_models
import autosklearn.classification
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
    ival = [-500, 4000]
    input_time_length = 1000
    max_epochs = 800
    max_increase_epochs = 80
    batch_size = 60
    high_cut_hz = 38
    factor_new = 1e-3
    init_block_size = 1000
    valid_set_fraction = 0.2

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
    train_cnt = mne_apply(
        lambda a: bandpass_cnt(a, low_cut_hz, high_cut_hz, train_cnt.info['sfreq'],
                               filt_order=3,
                               axis=1), train_cnt)
    train_cnt = mne_apply(
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



def run_exp(train_set, test_set):
    results = pd.DataFrame(columns=['date', 'keras_acc', 'keras_runtime',
                                    'tpot_acc', 'tpot_runtime',
                                    'auto-keras_acc', 'auto-keras_runtime'])
    configs = ['keras', 'tpot', 'auto-keras']
    now = str(datetime.datetime.now()).replace(":", "-")
    row = np.array([])
    row = np.append(row, now)
    for config in configs:
        if config == 'keras':
            # valid_set_fraction = 0.2
            # model = keras_models.deep_model(train_set.X.shape[1],
            #                                 train_set.X.shape[2],
            #                                 4)
            # train_set, valid_set = split_into_two_sets(
            #     train_set, first_set_fraction=1 - valid_set_fraction)
            # X_train = train_set.X[:, :, :, np.newaxis]
            # X_valid = valid_set.X[:, :, :, np.newaxis]
            # X_test = test_set.X[:, :, :, np.newaxis]
            # y_train = to_categorical(train_set.y, num_classes=4)
            # y_valid = to_categorical(valid_set.y, num_classes=4)
            # y_test = to_categorical(test_set.y, num_classes=4)
            # start = time.time()
            # earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10)
            # mcp = ModelCheckpoint('best_keras_model.hdf5', save_best_only=True, monitor='val_acc', mode='max')
            # model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid),
            #           callbacks=[earlystopping, mcp])
            # model.load_weights('best_keras_model.hdf5')
            # end = time.time()
            # res = model.evaluate(X_test, y_test, verbose=1)
            # row = np.append(row, res[1])
            # row = np.append(row, str(end-start))
            row = np.append(row, 0)
            row = np.append(row, 0)
        elif config == 'tpot':
            # tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
            # start = time.time()
            # print(train_set.X.shape)
            # print(train_set.y.shape)
            # train_set_reshpX = train_set.X.reshape(train_set.X.shape[0], -1)
            # test_set_reshpX = test_set.X.reshape(train_set.X.shape[0], -1)
            # tpot.fit(train_set_reshpX, train_set.y)
            # end = time.time()
            # print('time: ', (end - start))
            # print('accuracy :', tpot.score(test_set_reshpX, test_set.y))
            row = np.append(row, 0)
            row = np.append(row, 0)
        elif config == 'auto-keras':
            # X = train_set.X[:,:,:,np.newaxis]
            # clf = ImageClassifier(verbose=True, searcher_args={'trainer_args':{'max_iter_num' : 25}})
            # clf.fit(X, train_set.y, time_limit=12 * 60 * 60)
            # clf.final_fit(X, train_set.y, test_set.X, test_set.y, retrain=False)
            # y = clf.evaluate(test_set.X[:,:,:,np.newaxis], test_set.y)
            # clf.load_searcher().load_best_model().produce_keras_model().save('autokeras_model.h5')
            # print(y)
            row = np.append(row, 0)
            row = np.append(row, 0)
        elif config == 'auto-sklearn':
            automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=120,
                per_run_time_limit=30,
                tmp_folder='/tmp/autosklearn_cv_example_tmp',
                output_folder='/tmp/autosklearn_cv_example_out',
                delete_tmp_folder_after_terminate=False,
                resampling_strategy='cv',
                resampling_strategy_arguments={'folds': 5},
            )
            # fit() changes the data in place, but refit needs the original data. We
            # therefore copy the data. In practice, one should reload the data
            automl.fit(X_train.copy(), y_train.copy(), dataset_name='digits')
            # During fit(), models are fit on individual cross-validation folds. To use
            # all available data, we call refit() which trains all models in the
            # final ensemble on the whole dataset.
            automl.refit(X_train.copy(), y_train.copy())
            print(automl.show_models())
            predictions = automl.predict(X_test)
            print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

    results.loc[0] = row
    header = True
    if os.path.isfile('results'):
        header = False
    results.to_csv('results.csv', mode='a', header = header)



if __name__ == '__main__':
    data_folder = 'data/'
    subject_id = 1
    low_cut_hz = 0

    train_set, test_set = get_train_test(data_folder, subject_id, low_cut_hz)

    run_exp(train_set, test_set)
