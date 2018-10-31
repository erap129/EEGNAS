from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPool2D, Lambda, Dropout, Input, Cropping2D, Concatenate, ZeroPadding2D, BatchNormalization
import pandas as pd
import numpy as np
import time
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from toposort import toposort_flatten
from keras_models import dilation_pool, convert_to_dilated, mean_layer
import os
import keras.backend as K
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import random
import queue
import platform
import datetime
import copy
import treelib
import shutil
import tensorflow as tf
WARNING = '\033[93m'
ENDC = '\033[0m'


class Tree(object):
    def __init__(self):
        self.children = []
        self.data = None


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def delete_from_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


class Layer:
    running_id = 0

    def __init__(self, name=None):
        self.id = Layer.running_id
        self.connections = []
        self.parent = None
        self.keras_layer = None
        self.name = name
        Layer.running_id += 1

    def make_connection(self, other):
        self.connections.append(other)
        other.parent = self


class LambdaLayer(Layer):
    def __init__(self, function):
        Layer.__init__(self)
        self.function = function

class InputLayer(Layer):
    def __init__(self, shape_height, shape_width):
        Layer.__init__(self)
        self.shape_height = shape_height
        self.shape_width = shape_width


class FlattenLayer(Layer):
    def __init__(self):
        Layer.__init__(self)


class DropoutLayer(Layer):
    def __init__(self, rate=0.5):
        Layer.__init__(self)
        self.rate = rate


class BatchNormLayer(Layer):
    def __init__(self, axis=3, momentum=0.1, epsilon=1e-5):
        Layer.__init__(self)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon


class ActivationLayer(Layer):
    def __init__(self, activation_type='elu'):
        Layer.__init__(self)
        self.activation_type = activation_type


class ConvLayer(Layer):
    def __init__(self, kernel_width, kernel_height, filter_num, name=None):
        Layer.__init__(self, name)
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.filter_num = filter_num


class PoolingLayer(Layer):
    def __init__(self, pool_width, stride_width, mode):
        Layer.__init__(self)
        self.pool_width = pool_width
        self.stride_width = stride_width
        self.mode = mode


class CroppingLayer(Layer):
    def __init__(self, height_crop_top, height_crop_bottom, width_crop_left, width_crop_right):
        Layer.__init__(self)
        self.height_crop_top = height_crop_top
        self.height_crop_bottom = height_crop_bottom
        self.width_crop_left = width_crop_left
        self.width_crop_right = width_crop_right


class ZeroPadLayer(Layer):
    def __init__(self, height_pad_top, height_pad_bottom, width_pad_left, width_pad_right):
        Layer.__init__(self)
        self.height_pad_top = height_pad_top
        self.height_pad_bottom = height_pad_bottom
        self.width_pad_left = width_pad_left
        self.width_pad_right = width_pad_right


class ConcatLayer(Layer):
    def __init__(self, first_layer_index, second_layer_index):
        Layer.__init__(self)
        self.first_layer_index = first_layer_index
        self.second_layer_index = second_layer_index


def print_structure(structure):
    print('---------------model structure-----------------')
    for layer in structure:
        print('layer type:', layer.__class__.__name__, 'layer id:', layer.id)
    print('-----------------------------------------------')


def create_topo_layers(layers):
    layer_dict = {}
    for layer in layers:
        layer_dict[layer.id] = {x.id for x in layer.connections}
    return list(reversed(toposort_flatten(layer_dict)))


class MyModel:
    def __init__(self, model, layer_collection={}, name=None):
        self.layer_collection = layer_collection
        self.model = model
        self.name = name

    @staticmethod
    def new_model_from_structure(layer_collection):
        pool_counter = 0
        topo_layers = create_topo_layers(layer_collection.values())
        for i in topo_layers:
            layer = layer_collection[i]
            if isinstance(layer, InputLayer):
                keras_layer = Input(shape=(layer.shape_height, layer.shape_width, 1), name=str(layer.id))

            elif isinstance(layer, PoolingLayer):
                pool_counter += 1
                keras_layer = Lambda(dilation_pool, name='pooling_'+str(pool_counter), arguments={'window_shape': (1, layer.pool_width), 'strides':
                    (1, layer.stride_width), 'dilation_rate': (1, 1), 'pooling_type': layer.mode})

            elif isinstance(layer, ConvLayer):
                if(layer.kernel_width == 'down_to_one'):
                    topo_layers = create_topo_layers(layer_collection.values())
                    before_conv_layer_id = topo_layers[(np.where(np.array(topo_layers) == layer.id)[0] - 1)[0]]
                    layer.kernel_width = int(layer_collection[before_conv_layer_id].keras_layer.shape[2])
                keras_layer = Conv2D(filters=layer.filter_num, kernel_size=(layer.kernel_height, layer.kernel_width),
                                     strides=(1, 1), activation='elu', name=str(layer.id))

            elif isinstance(layer, CroppingLayer):
                keras_layer = Cropping2D(((layer.height_crop_top, layer.height_crop_bottom),
                                          (layer.width_crop_left, layer.width_crop_right)), name=str(layer.id))

            elif isinstance(layer, ZeroPadLayer):
                keras_layer = ZeroPadding2D(((layer.height_pad_top, layer.height_pad_bottom),
                                             (layer.width_pad_left, layer.width_pad_right)), name=str(layer.id))

            elif isinstance(layer, ConcatLayer):
                keras_layer = Concatenate(name=str(layer.id))(
                    [layer_collection[layer.first_layer_index].keras_layer,
                     layer_collection[layer.second_layer_index].keras_layer])

            elif isinstance(layer, BatchNormLayer):
                keras_layer = BatchNormalization(axis=layer.axis, momentum=layer.momentum, epsilon=layer.epsilon)

            elif isinstance(layer, ActivationLayer):
                keras_layer = Activation(layer.activation_type)

            elif isinstance(layer, DropoutLayer):
                keras_layer = Dropout(layer.rate)

            elif isinstance(layer, FlattenLayer):
                keras_layer = Flatten()

            elif isinstance(layer, LambdaLayer):
                keras_layer = Lambda(layer.function)

            layer.keras_layer = keras_layer
            if layer.name is not None:
                layer.keras_layer.name = layer.name

            if layer.parent is not None:
                try:
                    layer.keras_layer = layer.keras_layer(layer.parent.keras_layer)
                except TypeError as e:
                    print('couldnt connect a keras layer with tensor...error was:', e)
                except ValueError as e:
                    print('couldnt connect a keras layer with tensor...error was:', e)
        model = MyModel(model=Model(inputs=layer_collection[0].keras_layer, output=layer.keras_layer),
                        layer_collection=layer_collection)
        model.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.model.summary()
        try:
            assert(len(model.model.layers) == len(layer_collection) == len(topo_layers))
        except AssertionError:
            print('what happened now..?')
        return model


class NaiveNAS:
    def __init__(self, n_classes, input_time_len, n_chans,
                 X_train, y_train, X_valid, y_valid, X_test, y_test, subject_id, cropping=False):
        self.n_classes = n_classes
        self.n_chans = n_chans
        self.input_time_len = input_time_len
        self.X_train = X_train
        self.X_test = X_test
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_test = y_test
        self.y_valid = y_valid
        self.finalize_flag = 0
        self.cropping = cropping
        self.subject_id = subject_id

    def evaluate_model(self, model):
        finalized_model = self.finalize_model(model.layer_collection)
        if self.cropping:
            finalized_model.model = convert_to_dilated(model.model)
        best_model_name = 'keras_models/best_keras_model' + str(time.time()) + '.hdf5'
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5)
        mcp = ModelCheckpoint(best_model_name, save_best_only=True, monitor='val_acc', mode='max')
        finalized_model.model.fit(self.X_train, self.y_train, epochs=50,
                                  validation_data=(self.X_valid, self.y_valid),
                                  callbacks=[earlystopping, mcp])
        finalized_model.model.load_weights(best_model_name)
        res = finalized_model.model.evaluate(self.X_test, self.y_test)[1] * 100
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
                model.name += op.__name__
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
        while time.time()-start_time < time_limit and not self.finalize_flag:
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
                rand = np.random.choice(a=[1, 0], p=[probability, 1-probability])
                if rand == 1:
                    curr_model = model
            temperature *= (1-coolingRate)
            print('train time in seconds:', end-start)
            print('model accuracy:', res * 100)
            if experiment == 'filter_experiment':
                results.loc[num_of_ops - 1] = np.array([int(finalized_model.model.get_layer('conv1').filters),
                                                    int(finalized_model.model.get_layer('conv2').filters),
                                                    int(finalized_model.model.get_layer('conv3').filters),
                                                    res,
                                                    str(end-start),
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
            if not os.path.isdir('results/'+folder_name):
                createFolder('results/'+folder_name)
            results.to_csv('results/'+folder_name+'/subject_' + str(self.subject_id) + experiment + '_' + now + '.csv', mode='a')

    def run_one_model(self, model):
        if self.cropping:
            model.model = convert_to_dilated(model.model)
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5)
        mcp_filepath = 'keras_models/best_keras_model_' + str(datetime.datetime.now()).replace(":", "-") + '.hdf5'
        mcp = ModelCheckpoint(
            mcp_filepath, save_best_only=True, monitor='val_acc', mode='max', save_weights_only=True)
        start = time.time()
        model.model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                        callbacks=[earlystopping, mcp])
        model.model.load_weights(mcp_filepath)
        res = model.model.evaluate(self.X_test, self.y_test)[1] * 100
        res_train = model.model.evaluate(self.X_train, self.y_train)[1] * 100
        res_val = model.model.evaluate(self.X_valid, self.y_valid)[1] * 100
        end = time.time()
        final_time = end - start
        os.remove(mcp_filepath)
        return final_time, res, res_train, res_val

    def grid_search_filters(self, lo, hi, jumps):
        model = self.target_model()
        start_time = time.time()
        num_of_ops = 0
        total_time = 0
        results = pd.DataFrame(columns=['conv1 filters', 'conv2 filters', 'conv3 filters',
                                         'accuracy', 'train acc', 'runtime'])
        for first_filt in range(lo, hi, jumps):
            for second_filt in range(lo, hi, jumps):
                for third_filt in range(lo, hi, jumps):
                    K.clear_session()
                    num_of_ops += 1
                    model = self.set_target_model_filters(model, first_filt, second_filt, third_filt)
                    run_time, res, res_train = self.run_one_model(model)
                    total_time += time
                    print('train time in seconds:', time)
                    print('model accuracy:', res)
                    results.loc[num_of_ops - 1] = np.array([int(model.model.get_layer('conv1').filters),
                                                            int(model.model.get_layer('conv2').filters),
                                                            int(model.model.get_layer('conv3').filters),
                                                            res_train,
                                                            str(run_time)])
                    print(results)

        total_time = time.time() - start_time
        print('average train time per model', total_time / num_of_ops)
        now = str(datetime.datetime.now()).replace(":", "-")
        results.to_csv('results/filter_gridsearch_' + now + '.csv', mode='a')

    def grid_search_kernel_size(self, lo, hi, jumps):
        model = self.target_model()
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
                    model = self.set_target_model_kernel_sizes(model, first_size, second_size, third_size)
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

    def base_model(self, n_filters_time=25, n_filters_spat=25, filter_time_length=10, finalize=False):
        layer_collection = {}
        inputs = InputLayer(shape_height=self.n_chans, shape_width=self.input_time_len)
        layer_collection[inputs.id] = inputs
        conv_time = ConvLayer(kernel_width=filter_time_length, kernel_height=1, filter_num=n_filters_time)
        layer_collection[conv_time.id] = conv_time
        inputs.make_connection(conv_time)
        conv_spat = ConvLayer(kernel_width=1, kernel_height=self.n_chans, filter_num=n_filters_spat)
        layer_collection[conv_spat.id] = conv_spat
        conv_time.make_connection(conv_spat)
        batchnorm = BatchNormLayer()
        layer_collection[batchnorm.id] = batchnorm
        conv_spat.make_connection(batchnorm)
        elu = ActivationLayer()
        layer_collection[elu.id] = elu
        batchnorm.make_connection(elu)
        maxpool = PoolingLayer(pool_width=3, stride_width=3, mode='MAX')
        layer_collection[maxpool.id] = maxpool
        elu.make_connection(maxpool)
        return MyModel(model=None, layer_collection=layer_collection, name='base')

    def target_model(self):
        base_model = self.base_model()
        model = self.add_conv_maxpool_block(base_model, conv_filter_num=50, conv_layer_name='conv1')
        model = self.add_conv_maxpool_block(model, conv_filter_num=100, conv_layer_name='conv2')
        model = self.add_conv_maxpool_block(model, conv_filter_num=200, dropout=True, conv_layer_name='conv3')
        return model

    def finalize_model(self, layer_collection):
        layer_collection = copy.deepcopy(layer_collection)
        topo_layers = create_topo_layers(layer_collection.values())
        last_layer_id = topo_layers[-1]
        if self.cropping:
            final_conv_width = 2
        else:
            final_conv_width = 'down_to_one'
        conv_layer = ConvLayer(kernel_width=final_conv_width, kernel_height=1, filter_num=self.n_classes)
        layer_collection[conv_layer.id] = conv_layer
        layer_collection[last_layer_id].make_connection(conv_layer)
        softmax = ActivationLayer('softmax')
        layer_collection[softmax.id] = softmax
        conv_layer.make_connection(softmax)
        if self.cropping:
            mean = LambdaLayer(mean_layer)
            layer_collection[mean.id] = mean
            softmax.make_connection(mean)
        flatten = FlattenLayer()
        layer_collection[flatten.id] = flatten
        if self.cropping:
            mean.make_connection(flatten)
        else:
            softmax.make_connection(flatten)
        return MyModel.new_model_from_structure(layer_collection)


    def add_conv_maxpool_block(self, model, conv_width=10, conv_filter_num=50,
                               pool_width=3, pool_stride=3, dropout=False, conv_layer_name=None, random_values=True):
        if random_values:
            conv_width = random.randint(5, 10)
            conv_filter_num = random.randint(0, 50)
            pool_width = random.randint(1, 3)
            pool_stride = random.randint(1,3)

        topo_layers = create_topo_layers(model.layer_collection.values())
        last_layer_id = topo_layers[-1]
        if dropout:
            dropout = DropoutLayer()
            model.layer_collection[dropout.id] = dropout
            model.layer_collection[last_layer_id].make_connection(dropout)
            last_layer_id = dropout.id
        conv_layer = ConvLayer(kernel_width=conv_width, kernel_height=1,
                               filter_num=conv_filter_num, name=conv_layer_name)
        model.layer_collection[conv_layer.id] = conv_layer
        model.layer_collection[last_layer_id].make_connection(conv_layer)
        batchnorm_layer = BatchNormLayer()
        model.layer_collection[batchnorm_layer.id] = batchnorm_layer
        conv_layer.make_connection(batchnorm_layer)
        activation_layer = ActivationLayer()
        model.layer_collection[activation_layer.id] = activation_layer
        batchnorm_layer.make_connection(activation_layer)
        maxpool_layer = PoolingLayer(pool_width=pool_width, stride_width=pool_stride, mode='MAX')
        model.layer_collection[maxpool_layer.id] = maxpool_layer
        activation_layer.make_connection(maxpool_layer)
        model.name += '->add_conv_maxpool'
        return model

    def add_skip_connection_concat(self, model):
        topo_layers = create_topo_layers(model.layer_collection.values())
        to_concat = random.sample(range(Layer.running_id), 2)  # choose 2 random layer id's
        first_layer_index = np.min(to_concat)
        second_layer_index = np.max(to_concat)
        first_layer_index = topo_layers[first_layer_index]
        second_layer_index = topo_layers[second_layer_index]
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
            next_layer = heightChange
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

    def factor_filters(self, model):
        conv_indices = [layer.id for layer in model.layer_collection.values() if isinstance(layer, ConvLayer)]
        try:
            random_conv_index = random.randint(2, len(conv_indices) - 1)
        except ValueError:
            return model
        factor = random.randint(2, 4)
        model.layer_collection[conv_indices[random_conv_index]].filter_num *= factor
        model.name = model.name + '->factor_filters'
        return model

    def add_remove_filters(self, model):
        conv_indices = [layer.id for layer in model.layer_collection.values() if isinstance(layer, ConvLayer)]
        try:
            random_conv_index = random.randint(2, len(conv_indices) - 2)  # don't include first 2 convs or last conv
            random_conv_index = conv_indices[random_conv_index]
        except ValueError:
            return model
        print('layer to add/remove filter is:', str(random_conv_index))
        add_remove = [-1, 1]
        rand = random.randint(0, 1)
        to_add = add_remove[rand]
        if (model.layer_collection[random_conv_index].filter_num == 1 and to_add == -1) or\
                (model.layer_collection[random_conv_index].filter_num == 300 and to_add == 1):
            return self.add_remove_filters(model)  # if zero or 300 filters, start over...
        model.layer_collection[random_conv_index].filter_num += to_add
        return model

    def set_target_model_filters(self, model, filt1, filt2, filt3):
        conv_indices = [layer.id for layer in model.layer_collection.values() if isinstance(layer, ConvLayer)]
        conv_indices = conv_indices[2:len(conv_indices)-1]  # take only relevant indices
        model.layer_collection[conv_indices[0]].filter_num = filt1
        model.layer_collection[conv_indices[1]].filter_num = filt2
        model.layer_collection[conv_indices[2]].filter_num = filt3
        return model

    def set_target_model_kernel_sizes(self, model, size1, size2, size3):
        conv_indices = [layer.id for layer in model.layer_collection.values() if isinstance(layer, ConvLayer)]
        conv_indices = conv_indices[2:len(conv_indices)-1]  # take only relevant indices
        model.layer_collection[conv_indices[0]].kernel_width = size1
        model.layer_collection[conv_indices[1]].kernel_width = size2
        model.layer_collection[conv_indices[2]].kernel_width = size3
        return model


