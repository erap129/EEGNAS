import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import numpy as np
import pandas as pd
import keras_models
from naiveNAS import NaiveNAS
import matplotlib
matplotlib.use('Agg')
from generator import four_class_example_generator
import sklearn.metrics
from autokeras import ImageClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from braindecode.datautil.splitters import split_into_two_sets
from data_preprocessing import handle_subject_data
from tpot import TPOTClassifier
import time
import random

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def run_genetic_filters(exp_id, configuration):
    subjects = random.sample(range(9), configuration.getint('num_subjects'))
    exp_folder = str(exp_id) + '_evolution_' + '_'.join(configuration.values())
    merged_results_dict = {'subject': [], 'generation': [], 'val_acc': []}
    for subject_id in subjects:
        X_train, y_train, X_valid, y_valid, X_test, y_test = handle_subject_data(subject_id)
        naiveNAS = NaiveNAS(n_classes=4, input_time_len=1125, n_chans=22,
                            X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
                            X_test=X_test, y_test=y_test, configuration=configuration, subject_id=subject_id, cropping=False)
        results_dict = naiveNAS.evolution_filters()
        for key in results_dict.keys():
            merged_results_dict[key] = merged_results_dict[key], results_dict[key]
    createFolder(exp_folder)
    pd.DataFrame.from_dict(merged_results_dict, orient='index').to_csv(exp_folder + 'results.csv')


def run_keras_model(X_train, y_train, X_valid, y_valid, X_test, y_test, row, cropping=False, mode='deep'):
    print('--------------running keras model--------------')
    earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10)
    mcp = ModelCheckpoint('best_keras_model.hdf5', save_best_only=True, monitor='val_acc', mode='max')
    start = time.time()
    if mode == 'deep':
        model = keras_models.deep_model_mimic(X_train.shape[1], X_train.shape[2], 4, cropped=cropping)
    elif mode == 'shallow':
        model = keras_models.shallow_model_mimic(X_train.shape[1], X_train.shape[2], 4, cropped=cropping)
    if cropping:
        model = keras_models.convert_to_dilated(model)

    model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid), callbacks=[earlystopping, mcp])
    model.load_weights('best_keras_model.hdf5')
    end = time.time()
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


def run_exp(X_train, y_train, X_valid, y_valid, X_test, y_test, subject, toggle, cropping=False):
    now = str(datetime.datetime.now()).replace(":", "-")
    row = np.array([])
    row = np.append(row, now)
    row = np.append(row, str(subject))
    for config in toggle.keys():
        if config == 'keras' and toggle[config]:
            row = run_keras_model(X_train, y_train, X_valid, y_valid, X_test, y_test, row, cropping=cropping, mode='deep')
    print('row is:', row)
    return row


def automl_comparison(cropping=False):
    data_folder = 'data/'
    low_cut_hz = 0
    results = pd.DataFrame(columns=['date', 'subject'])
    toggle = {'keras': True, 'tpot': False, 'auto-keras': False, 'auto-sklearn': False}
    for setting in toggle.keys():
        if toggle[setting]:
            results[setting+'_acc'] = None
            results[setting+'_runtime'] = None

    for subject_id in range(1, 10):
        X_train, y_train, X_valid, y_valid, X_test, y_test = handle_subject_data(subject_id, cropping=cropping)
        row = run_exp(X_train, y_train, X_valid, y_valid, X_test, y_test, subject_id, toggle, cropping=cropping)
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


def run_naive_nas(real_data=True, toy_data=False):
    global data_folder, valid_set_fraction
    now = str(datetime.datetime.now()).replace(":", "-")
    accuracies = np.zeros(9)
    if real_data:
        for subject_id in range(1, 10):
            X_train, y_train, X_valid, y_valid, X_test, y_test = handle_subject_data(subject_id)
            naiveNAS = NaiveNAS(n_classes=4, input_time_len=1125, n_chans=22,
                                X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
                                X_test=X_test, y_test=y_test, subject_id=subject_id, cropping=False)
            accuracies[subject_id-1] = naiveNAS.find_best_model_evolution()
        np.savetxt('results/naive_nas_'+now+'.csv', accuracies, delimiter=',')

    if toy_data:
        X_train, y_train, X_val, y_val, X_test, y_test = four_class_example_generator()
        print(X_test)
        print(y_test)
        naiveNAS = NaiveNAS(n_classes=4, input_time_len=1000, n_chans=4,
                            X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val,
                            X_test=X_test, y_test=y_test)
        naiveNAS.find_best_model('filter_experiment')


def run_grid_search(subject_id, cropping=False):
    X_train, y_train, X_valid, y_valid, X_test, y_test = handle_subject_data(subject_id, cropping=cropping)
    if cropping:
        input_time_len = 1000
    else:
        input_time_len = 1125
    naiveNAS = NaiveNAS(n_classes=4, input_time_len=input_time_len, n_chans=22,
                        X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid,
                        X_test=X_test, y_test=y_test, subject_id=subject_id, cropping=False)
    naiveNAS.grid_search_filters(1, 21, 1)
