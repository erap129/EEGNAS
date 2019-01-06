def find_best_model_evolution(self):
    nb_evolution_steps = 10
    tournament = \
        gp.TournamentOptimizer(
            population_sz=10,
            init_fn=random_model,
            mutate_fn=mutate_net,
            naive_nas=self)

    for i in range(nb_evolution_steps):
        print('\nEvolution step:{}'.format(i))
        print('================')
        top_model = tournament.step()
        # keep track of the experiment results & corresponding architectures
        name = "tourney_{}".format(i)
        res, _ = self.evaluate_model(top_model, test=True)
    return res


def find_best_model_bb(self, folder_name, time_limit=12 * 60 * 60):
    curr_models = queue.Queue()
    operations = [self.factor_filters, self.add_conv_maxpool_block]
    initial_model = self.base_model()
    curr_models.put(initial_model)
    res = self.evaluate_model(initial_model)
    total_results = []
    result_tree = treelib.Tree()
    result_tree.create_node(data=res, identifier=initial_model.name)
    start_time = time.time()
    while not curr_models.empty() and time.time() - start_time < time_limit:
        curr_model = curr_models.get_nowait()
        for op in operations:
            model = op(curr_model)
            res = self.evaluate_model(model)
            print('node name is:', model.name)
            result_tree.create_node(data=res, identifier=model.name, parent=curr_model.name)
            result_tree.show()
            print('model accuracy:', res[1] * 100)
            total_results.append(res[1])
            curr_models.put(model)
    print(total_results)
    return total_results.max()


def find_best_model_simanneal(self, folder_name, experiment=None, time_limit=12 * 60 * 60):
    curr_model = self.base_model()
    earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5)
    mcp = ModelCheckpoint('best_keras_model.hdf5', save_best_only=True, monitor='val_acc', mode='max')
    start_time = time.time()
    curr_acc = 0
    num_of_ops = 0
    temperature = 10
    coolingRate = 0.003
    operations = [self.factor_filters, self.add_conv_maxpool_block]
    total_time = 0
    if experiment == 'filter_experiment':
        results = pd.DataFrame(columns=['conv1 filters', 'conv2 filters', 'conv3 filters',
                                        'accuracy', 'runtime', 'switch probability', 'temperature'])
    while time.time() - start_time < time_limit and not self.finalize_flag:
        K.clear_session()
        op_index = random.randint(0, len(operations) - 1)
        num_of_ops += 1
        mcp = ModelCheckpoint('keras_models/best_keras_model' + str(num_of_ops) + '.hdf5',
                              save_best_only=True, monitor='val_acc', mode='max', save_weights_only=True)
        model = operations[op_index](curr_model)
        finalized_model = self.finalize_model(model.layer_collection)
        if self.cropping:
            finalized_model.model = convert_to_dilated(model.model)
        start = time.time()
        finalized_model.model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                                  callbacks=[earlystopping, mcp])
        finalized_model.model.load_weights('keras_models/best_keras_model' + str(num_of_ops) + '.hdf5')
        end = time.time()
        total_time += end - start
        res = finalized_model.model.evaluate(self.X_test, self.y_test)[1] * 100
        curr_model = model
        if res >= curr_acc:
            curr_model = model
            curr_acc = res
            probability = -1
        else:
            probability = np.exp((res - curr_acc) / temperature)
            rand = np.random.choice(a=[1, 0], p=[probability, 1 - probability])
            if rand == 1:
                curr_model = model
        temperature *= (1 - coolingRate)
        print('train time in seconds:', end - start)
        print('model accuracy:', res * 100)
        if experiment == 'filter_experiment':
            results.loc[num_of_ops - 1] = np.array([int(finalized_model.model.get_layer('conv1').filters),
                                                    int(finalized_model.model.get_layer('conv2').filters),
                                                    int(finalized_model.model.get_layer('conv3').filters),
                                                    res,
                                                    str(end - start),
                                                    str(probability),
                                                    str(temperature)])
            print(results)
    final_model = self.finalize_model(curr_model.layer_collection)
    final_model.model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                          callbacks=[earlystopping])
    res = final_model.model.evaluate(self.X_test, self.y_test)[1] * 100
    print('model accuracy:', res)
    print('average train time per model', total_time / num_of_ops)
    now = str(datetime.datetime.now()).replace(":", "-")
    if experiment == 'filter_experiment':
        if not os.path.isdir('results/' + folder_name):
            createFolder('results/' + folder_name)
        results.to_csv('results/' + folder_name + '/subject_' + str(self.subject_id) + experiment + '_' + now + '.csv',
                       mode='a')


def grid_search_filters(self, lo, hi, jumps):
    model = target_model()
    start_time = time.time()
    num_of_ops = 0
    total_time = 0
    results = pd.DataFrame(columns=['conv1 filters', 'conv2 filters', 'conv3 filters',
                                    'test acc', 'val acc', 'train acc', 'runtime'])
    for first_filt in range(lo, hi, jumps):
        for second_filt in range(lo, hi, jumps):
            for third_filt in range(lo, hi, jumps):
                K.clear_session()
                num_of_ops += 1
                model = set_target_model_filters(model, first_filt, second_filt, third_filt)
                run_time, res_test, res_val, res_train, _ = self.evaluate_model(model)
                total_time += run_time
                print('train time in seconds:', run_time)
                print('model accuracy:', res_test)
                results.loc[num_of_ops - 1] = np.array([int(model.model.get_layer('conv1').filters),
                                                        int(model.model.get_layer('conv2').filters),
                                                        int(model.model.get_layer('conv3').filters),
                                                        res_test,
                                                        res_val,
                                                        res_train,
                                                        str(run_time)])
                print(results)

    total_time = time.time() - start_time
    print('average train time per model', total_time / num_of_ops)
    now = str(datetime.datetime.now()).replace(":", "-")
    results.to_csv('results/filter_gridsearch_' + now + '.csv', mode='a')


def grid_search_kernel_size(self, lo, hi, jumps):
    model = target_model()
    start_time = time.time()
    num_of_ops = 0
    total_time = 0
    results = pd.DataFrame(columns=['conv1 kernel size', 'conv2 kernel size', 'conv3 kernel size',
                                    'accuracy', 'train acc', 'runtime'])
    for first_size in range(lo, hi, jumps):
        for second_size in range(lo, hi, jumps):
            for third_size in range(lo, hi, jumps):
                K.clear_session()
                num_of_ops += 1
                model = set_target_model_kernel_sizes(model, first_size, second_size, third_size)
                run_time, res, res_train = self.run_one_model(model)
                total_time += time
                print('train time in seconds:', time)
                print('model accuracy:', res)
                results.loc[num_of_ops - 1] = np.array([int(model.model.get_layer('conv1').kernel_size[1]),
                                                        int(model.model.get_layer('conv2').kernel_size[1]),
                                                        int(model.model.get_layer('conv3').kernel_size[1]),
                                                        res_train,
                                                        str(run_time)])
                print(results)

    total_time = time.time() - start_time
    print('average train time per model', total_time / num_of_ops)
    now = str(datetime.datetime.now()).replace(":", "-")
    results.to_csv('results/kernel_size_gridsearch_' + now + '.csv', mode='a')




# def folder_renamer():
#     if not globals.config['DEFAULT'].getboolean('success'):
#         new_exp_folder = exp_folder + '_fail'
#     else:
#         new_exp_folder = exp_folder
#     os.rename(exp_folder, new_exp_folder)
#
#
# atexit.register(folder_renamer())



def add_conv_maxpool_block(layer_collection, conv_width=10, conv_filter_num=50, dropout=False,
                           pool_width=3, pool_stride=3, conv_layer_name=None, random_values=True):
    layer_collection = copy.deepcopy(layer_collection)
    if random_values:
        conv_time = random.randint(5, 10)
        conv_filter_num = random.randint(0, 50)
        pool_time = 2
        pool_stride = 2

    topo_layers = create_topo_layers(layer_collection.values())
    last_layer_id = topo_layers[-1]
    if dropout:
        dropout = DropoutLayer()
        layer_collection[dropout.id] = dropout
        layer_collection[last_layer_id].make_connection(dropout)
        last_layer_id = dropout.id
    conv_layer = ConvLayer(kernel_time=conv_width, kernel_eeg_chan=1,
                           filter_num=conv_filter_num, name=conv_layer_name)
    layer_collection[conv_layer.id] = conv_layer
    layer_collection[last_layer_id].make_connection(conv_layer)
    batchnorm_layer = BatchNormLayer()
    layer_collection[batchnorm_layer.id] = batchnorm_layer
    conv_layer.make_connection(batchnorm_layer)
    activation_layer = ActivationLayer()
    layer_collection[activation_layer.id] = activation_layer
    batchnorm_layer.make_connection(activation_layer)
    maxpool_layer = PoolingLayer(pool_time=pool_width, stride_time=pool_stride, mode='MAX')
    layer_collection[maxpool_layer.id] = maxpool_layer
    activation_layer.make_connection(maxpool_layer)
    return layer_collection
    # return MyModel.new_model_from_structure(layer_collection, name=model.name + '->add_conv_maxpool')

def add_skip_connection_concat(model):
    topo_layers = create_topo_layers(model.layer_collection.values())
    to_concat = random.sample(model.layer_collection.keys(), 2)  # choose 2 random layer id's
    # first_layer_index = topo_layers.index(np.min(to_concat))
    # second_layer_index = topo_layers.index(np.max(to_concat))
    # first_layer_index = topo_layers[first_layer_index]
    # second_layer_index = topo_layers[second_layer_index]
    first_layer_index = np.min(to_concat)
    second_layer_index = np.max(to_concat)
    first_shape = model.model.get_layer(str(first_layer_index)).output.shape
    second_shape = model.model.get_layer(str(second_layer_index)).output.shape
    print('first layer shape is:', first_shape)
    print('second layer shape is:', second_shape)

    height_diff = int(first_shape[1]) - int(second_shape[1])
    width_diff = int(first_shape[2]) - int(second_shape[2])
    height_crop_top = height_crop_bottom = np.abs(int(height_diff / 2))
    width_crop_left = width_crop_right = np.abs(int(width_diff / 2))
    if height_diff % 2 == 1:
        height_crop_top += 1
    if width_diff % 2 == 1:
        width_crop_left += 1
    if height_diff < 0:
        ChosenHeightClass = ZeroPadLayer
    else:
        ChosenHeightClass = CroppingLayer
    if width_diff < 0:
        ChosenWidthClass = ZeroPadLayer
    else:
        ChosenWidthClass = CroppingLayer
    first_layer = model.layer_collection[first_layer_index]
    second_layer = model.layer_collection[second_layer_index]
    next_layer = first_layer
    if height_diff != 0:
        heightChanger = ChosenHeightClass(height_crop_top, height_crop_bottom, 0, 0)
        model.layer_collection[heightChanger.id] = heightChanger
        first_layer.make_connection(heightChanger)
        next_layer = heightChanger
    if width_diff != 0:
        widthChanger = ChosenWidthClass(0, 0, width_crop_left, width_crop_right)
        model.layer_collection[widthChanger.id] = widthChanger
        next_layer.make_connection(widthChanger)
        next_layer = widthChanger
    concat = ConcatLayer(next_layer.id, second_layer_index)
    model.layer_collection[concat.id] = concat
    next_layer.connections.append(concat)
    for lay in second_layer.connections:
        concat.connections.append(lay)
        if not isinstance(lay, ConcatLayer):
            lay.parent = concat
        else:
            if lay.second_layer_index == second_layer_index:
                lay.second_layer_index = concat.id
            if lay.first_layer_index == second_layer_index:
                lay.first_layer_index = concat.id
    second_layer.connections = []
    second_layer.connections.append(concat)
    return model

def edit_conv_layer(model, mode):
    layer_collection = copy.deepcopy(model.layer_collection)
    conv_indices = [layer.id for layer in layer_collection.values() if isinstance(layer, ConvLayer)]
    try:
        random_conv_index = random.randint(2, len(conv_indices) - 2) # don't include last conv
    except ValueError:
        return model
    factor = 1 + random.uniform(0,1)
    if mode == 'filters':
        layer_collection[conv_indices[random_conv_index]].filter_num = \
            int(layer_collection[conv_indices[random_conv_index]].filter_num * factor)
    elif mode == 'kernels':
        layer_collection[conv_indices[random_conv_index]].kernel_width = \
            int(layer_collection[conv_indices[random_conv_index]].kernel_width * factor)
    return MyModel.new_model_from_structure(layer_collection, name=model.name + '->factor_filters')

def factor_filters(model):
    return edit_conv_layer(model, mode='filters')

def factor_kernels(model):
    return edit_conv_layer(model, mode='kernels')

def set_target_model_filters(model, filt1, filt2, filt3):
    conv_indices = [layer.id for layer in model.layer_collection.values() if isinstance(layer, ConvLayer)]
    conv_indices = conv_indices[2:len(conv_indices)]  # take only relevant indices
    model.layer_collection[conv_indices[0]].filter_num = filt1
    model.layer_collection[conv_indices[1]].filter_num = filt2
    model.layer_collection[conv_indices[2]].filter_num = filt3
    return model.new_model_from_structure(model.layer_collection)

def set_target_model_kernel_sizes(model, size1, size2, size3):
    conv_indices = [layer.id for layer in model.layer_collection.values() if isinstance(layer, ConvLayer)]
    conv_indices = conv_indices[2:len(conv_indices)]  # take only relevant indices
    model.layer_collection[conv_indices[0]].kernel_width = size1
    model.layer_collection[conv_indices[1]].kernel_width = size2
    model.layer_collection[conv_indices[2]].kernel_width = size3
    return model.new_model_from_structure(model.layer_collection)

def mutate_net(model):
    operations = [add_conv_maxpool_block, factor_filters, factor_kernels, add_skip_connection_concat]
    op_index = random.randint(0, len(operations) - 1)
    try:
        model = operations[op_index](model)
    except ValueError as e:
        print('exception occured while performing ', operations[op_index].__name__)
        print('error message: ', str(e))
        print('trying another mutation...')
        return mutate_net(model)
    return model



# class CroppingLayer(Layer):
#     @initializer
#     def __init__(self, height_crop_top, height_crop_bottom, width_crop_left, width_crop_right):
#         Layer.__init__(self)


def create_topo_layers(layers):
    # layer_dict = {}
    # for layer in layers:
    #     layer_dict[layer.id] = {x.id for x in layer.connections}
    # return list(reversed(toposort_flatten(layer_dict)))
    ans = []
    for layer in layers:
        ans.append(layer.id)
    return ans

    def test_check_pickle(self):
        test1 = pickle.dumps(ConvLayer(kernel_eeg_chan=2))
        test2 = pickle.dumps(ConvLayer(kernel_eeg_chan=1))
        test3 = pickle.dumps(ActivationLayer())
        test4 = pickle.dumps(ActivationLayer())
        assert(test3 == test4)
        assert(test1 != test2)

        model1 = pickle.dumps(uniform_model(10, ActivationLayer))
        model2 = pickle.dumps(uniform_model(10, ActivationLayer))
        model_set = set()
        model_set.add(model1)
        model_set.add(model2)
        assert(len(model_set) == 1)
        model3 = pickle.dumps([ConvLayer(kernel_eeg_chan=2), ConvLayer(kernel_eeg_chan=1), ActivationLayer()])
        model_set.add(model3)
        assert(len(model_set) == 2)

        model1 = pickle.loads(model1)
        for layer in model1:
            assert(pickle.dumps(layer) == pickle.dumps(ActivationLayer()))


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


from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Activation, MaxPool2D, Lambda, Dropout, BatchNormalization
from keras.models import model_from_json, Model
import numpy as np
import keras.backend as K


def shallow_model_mimic(n_chans, input_time_length, n_classes, n_filters_time=40, n_filters_spat=40,
                        filter_time_length=25, cropped=False):
    model = Sequential()
    model.add(Conv2D(name='temporal_convolution', filters=n_filters_time, input_shape=(n_chans, input_time_length, 1),
                     kernel_size=(1, filter_time_length), strides=(1, 1)))
    model.add(Conv2D(name='spatial_filter', filters=n_filters_spat, kernel_size=(n_chans, 1), strides=(1, 1)))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5))
    model.add(Lambda(lambda x: x ** 2))  # squaring layer
    model.add(Lambda(dilation_pool, arguments={'window_shape': (1, 75), 'strides': (1, 15), 'dilation_rate': (1, 1), 'pooling_type': 'AVG'}))
    model.add(Lambda(lambda x: K.log(K.clip(x, min_value=1e-6, max_value=None))))
    model.add(Dropout(0.5))
    if cropped:
        final_kernel_size = 30
    else:
        final_kernel_size = int(model.layers[-1].output_shape[2])
    model.add(Conv2D(filters=n_classes, kernel_size=(1, final_kernel_size), strides=(1, 1)))
    model.add(Activation('softmax'))
    if cropped:
        model.add(Lambda(mean_layer))
    model.add(Flatten())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def dilation_pool(x, window_shape, strides, dilation_rate, pooling_type='MAX'):
    import tensorflow as tf
    return tf.nn.pool(x, window_shape=window_shape, strides=strides, dilation_rate=dilation_rate,
                      pooling_type=pooling_type, padding='VALID')


def mean_layer(x):
    import keras.backend as K
    return K.mean(x, axis=2)


# trying to mimic exactly what was done in pytorch in the paper implementation
def deep_model_mimic(n_chans, input_time_length, n_classes, n_filters_time=25, n_filters_spat=25, filter_time_length=10,
                  n_filters_2=50, filter_len_2=10, n_filters_3=100, filter_len_3=10, n_filters_4=200,
                  filter_len_4=10, cropped=False):
    model = Sequential()
    model.add(Conv2D(name='temporal_convolution', filters=n_filters_time, input_shape=(n_chans, input_time_length, 1),
                     kernel_size=(1, filter_time_length), strides=(1,1)))
    model.add(Conv2D(name='spatial_filter', filters=n_filters_spat, kernel_size=(n_chans, 1), strides=(1,1)))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5))
    model.add(Activation('elu'))
    model.add(Lambda(dilation_pool, arguments={'window_shape': (1,3), 'strides': (1,3), 'dilation_rate': (1,1)}))

    model.add(Conv2D(filters=n_filters_2, kernel_size=(1, filter_len_2), strides=(1, 1)))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5))
    model.add(Activation('elu'))
    model.add(Lambda(dilation_pool, arguments={'window_shape': (1, 3), 'strides': (1, 3), 'dilation_rate': (1,1)}))

    model.add(Conv2D(filters=n_filters_3, kernel_size=(1, filter_len_3), strides=(1, 1)))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5))
    model.add(Activation('elu'))
    model.add(Lambda(dilation_pool, arguments={'window_shape': (1, 3), 'strides': (1, 3), 'dilation_rate': (1,1)}))

    model.add(Dropout(0.5))
    model.add(Conv2D(filters=n_filters_4, kernel_size=(1, filter_len_4), strides=(1, 1)))
    model.add(BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5))
    model.add(Activation('elu'))
    model.add(Lambda(dilation_pool, arguments={'window_shape': (1, 3), 'strides': (1, 3), 'dilation_rate': (1,1)}))
    if cropped:
        final_kernel_size = 2
    else:
        final_kernel_size = int(model.layers[-1].output_shape[2])
    model.add(Conv2D(filters=n_classes, kernel_size=(1, final_kernel_size), strides=(1, 1)))
    model.add(Activation('softmax'))
    if cropped:
        model.add(Lambda(mean_layer))
    model.add(Flatten())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def convert_to_dilated(model):
    axis = [0,1]
    stride_so_far = np.array([1,1])
    for layer in model.layers:
        if hasattr(layer, 'dilation_rate') or (hasattr(layer, 'arguments') and 'dilation_rate' in layer.arguments):
            if hasattr(layer, 'arguments'):
                dilation_rate = layer.arguments['dilation_rate']
            else:
                dilation_rate = layer.dilation_rate
            assert dilation_rate == 1 or (dilation_rate == (1, 1)), (
                "Dilation should equal 1 before conversion, maybe the model is "
                "already converted?")
            new_dilation = [1, 1]
            for ax in axis:
                new_dilation[ax] = int(stride_so_far[ax])
            if hasattr(layer, 'arguments'):
                layer.arguments['dilation_rate'] = tuple(new_dilation)
            else:
                layer.dilation_rate = tuple(new_dilation)
        if hasattr(layer, 'strides') or (hasattr(layer, 'arguments') and 'strides' in layer.arguments):
            if hasattr(layer, 'arguments'):
                strides = layer.arguments['strides']
            else:
                strides = layer.strides
            if not hasattr(strides, '__len__'):
                strides = (strides, strides)
            stride_so_far *= np.array(strides)
            new_stride = list(strides)
            for ax in axis:
                new_stride[ax] = 1
            if hasattr(layer, 'arguments'):
                layer.arguments['strides'] = tuple(new_stride)
            else:
                layer.strides = tuple(new_stride)
    new_model = model_from_json(model.to_json())
    print(model.layers)
    new_model.summary()
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return new_model



def deep_model_cropped(n_chans, input_time_length, n_classes, n_filters_time=25, n_filters_spat=25, filter_time_length=10,
                  n_filters_2=50, filter_len_2=10, n_filters_3=100, filter_len_3=10, n_filters_4=200,
                  filter_len_4=10, n_filters_5=400, filter_len_5=4):
    model = Sequential()
    model.add(Conv2D(name='temporal_convolution', filters=n_filters_time, input_shape=(n_chans, input_time_length, 1),
                     kernel_size=(1, filter_time_length), strides=(1,1), activation='elu'))
    model.add(Conv2D(name='spatial_filter', filters=n_filters_spat, kernel_size=(n_chans, 1), strides=(1,1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1,3), strides=(1,3)))

    model.add(Conv2D(filters=n_filters_2, kernel_size=(1, filter_len_2), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1,3), strides=(1,3)))

    model.add(Conv2D(filters=n_filters_3, kernel_size=(1, filter_len_3), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Conv2D(filters=n_filters_4, kernel_size=(1, filter_len_4), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Dropout(0.5))

    model.add(Conv2D(filters=n_classes, kernel_size=(1, 2), strides=(1, 1), activation='softmax'))
    model.add(Flatten())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model




def deep_model(n_chans, input_time_length, n_classes, n_filters_time=25, n_filters_spat=25, filter_time_length=10,
                  n_filters_2=50, filter_len_2=10, n_filters_3=100, filter_len_3=10, n_filters_4=200,
                  filter_len_4=10, n_filters_5=400, filter_len_5=4):
    model = Sequential()
    model.add(Conv2D(name='temporal_convolution', filters=n_filters_time, input_shape=(n_chans, input_time_length, 1),
                     kernel_size=(1, filter_time_length), strides=(1,1), activation='elu'))

    # note that this is a different implementation from the paper!
    # they didn't put an activation function between the first two convolutions
    # Also, in the paper they implemented batch-norm before each non-linearity - which I didn't do!
    # Also, they added dropout for each input the conv layers except the first! I dropped out only in the end

    model.add(Conv2D(name='spatial_filter', filters=n_filters_spat, kernel_size=(n_chans, 1), strides=(1,1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1,3), strides=(1,3)))

    model.add(Conv2D(filters=n_filters_2, kernel_size=(1, filter_len_2), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1,3), strides=(1,3)))

    model.add(Conv2D(filters=n_filters_3, kernel_size=(1, filter_len_3), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Conv2D(filters=n_filters_4, kernel_size=(1, filter_len_4), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Dropout(0.5))

    model.add(Conv2D(filters=n_filters_5, kernel_size=(1, filter_len_5), strides=(1, 1), activation='elu'))
    model.add(MaxPool2D(pool_size=(1, 3), strides=(1, 3)))

    model.add(Dropout(0.5))

    model.add(Conv2D(filters=n_classes, kernel_size=(1, 2), strides=(1, 1), activation='softmax'))
    model.add(Flatten())
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

