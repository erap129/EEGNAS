from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPool2D, Lambda, Dropout, Input, Cropping2D, Concatenate, ZeroPadding2D
from keras.utils import to_categorical
import copy
import numpy as np
import time
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from toposort import toposort_flatten
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import random


class Layer:
    running_id = 0

    def __init__(self):
        self.id = Layer.running_id
        self.connections = []
        self.parent = None
        self.keras_layer = None
        Layer.running_id += 1

    def make_connection(self, other):
        self.connections.append(other)
        other.parent = self


class InputLayer(Layer):
    def __init__(self, shape_height, shape_width):
        Layer.__init__(self)
        self.shape_height = shape_height
        self.shape_width = shape_width


class ConvLayer(Layer):
    def __init__(self, kernel_width, kernel_height, filter_num):
        Layer.__init__(self)
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.filter_num = filter_num


class MaxPoolLayer(Layer):
    def __init__(self, pool_width, stride_width):
        Layer.__init__(self)
        self.pool_width = pool_width
        self.stride_width = stride_width


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
    print('about to sort:', layer_dict)
    return list(reversed(toposort_flatten(layer_dict)))


class MyModel(Model):
    def __init__(self, layer_collection={}, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)
        self.layer_collection = layer_collection

    @staticmethod
    def new_model_from_structure(layer_collection):
        topo_layers = create_topo_layers(layer_collection.values())
        for i in topo_layers:
            layer = layer_collection[i]
            if isinstance(layer, InputLayer):
                keras_layer = Input(shape=(layer.shape_height, layer.shape_width, 1), name=str(layer.id))
                layer.keras_layer = keras_layer
            elif isinstance(layer, MaxPoolLayer):
                keras_layer = MaxPool2D(pool_size=(1, layer.pool_width), strides=(1, layer.stride_width),
                                        name=str(layer.id))
                layer.keras_layer = keras_layer
            elif isinstance(layer, ConvLayer):
                keras_layer = Conv2D(filters=layer.filter_num, kernel_size=(layer.kernel_height, layer.kernel_width),
                                     strides=(1, 1), activation='elu', name=str(layer.id))
                layer.keras_layer = keras_layer
            elif isinstance(layer, CroppingLayer):
                keras_layer = Cropping2D(((layer.height_crop_top, layer.height_crop_bottom),
                                          (layer.width_crop_left, layer.width_crop_right)), name=str(layer.id))
                layer.keras_layer = keras_layer
            elif isinstance(layer, ZeroPadLayer):
                keras_layer = ZeroPadding2D(((layer.height_pad_top, layer.height_pad_bottom),
                                             (layer.width_pad_left, layer.width_pad_right)), name=str(layer.id))
                layer.keras_layer = keras_layer
            elif isinstance(layer, ConcatLayer):
                print('about to concatenate layers:', layer.first_layer_index, 'and:', layer.second_layer_index)
                keras_layer = Concatenate(name=str(layer.id))(
                    [layer_collection[layer.first_layer_index].keras_layer,
                     layer_collection[layer.second_layer_index].keras_layer])
                layer.keras_layer = keras_layer
            if layer.parent is not None:
                try:
                    layer.keras_layer = layer.keras_layer(layer.parent.keras_layer)
                except TypeError:
                    print(layer.keras_layer.__class__.__name__)
                except ValueError:
                    print(layer.keras_layer.__class__.__name__)
        model = MyModel(layer_collection=layer_collection,
                       inputs=layer_collection[0].keras_layer, output=layer.keras_layer)
        try:
            assert(len(model.layers) == len(layer_collection) == len(topo_layers))
        except AssertionError:
            print('what happened now..?')
        return model


class NaiveNAS:
    def __init__(self, n_classes, input_time_len, n_chans,
                 X_train, y_train, X_valid, y_valid, X_test, y_test):
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

    def find_best_model(self, time_limit = 1 * 60 * 60):
        curr_model = self.base_model()
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=3)
        start_time = time.time()
        curr_acc = 0
        num_of_ops = 0
        temperature = 10000
        coolingRate = 0.003
        operations = [self.add_conv_maxpool_block, self.add_filters]

        while time.time()-start_time < time_limit and not self.finalize_flag:
            op_index = random.randint(0, len(operations) - 1)
            num_of_ops += 1
            # model = operations[op_index](curr_model, num_of_ops)
            model = self.add_skip_connection_concat(curr_model)
            final_model = self.finalize_model(model)
            final_model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                      callbacks=[earlystopping])
            res = final_model.evaluate(self.X_test, self.y_test) * 100
            curr_model = model
            # if res[1] >= curr_acc:
            #     curr_model = model
            # else:
            #     probability = np.exp((res[1] - curr_acc) / temperature)
            #     rand = np.random.choice(a=1, p=[1-probability, probability])
            #     if rand == 1:
            #         curr_model = model
            # temperature *= (1-coolingRate)
            print('model accuracy:', res[1] * 100)

        final_model = self.finalize_model(curr_model)
        final_model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                        callbacks=[earlystopping])
        res = final_model.evaluate(self.X_test, self.y_test) * 100
        print('model accuracy:', res[1] * 100)

    def base_model(self, n_filters_time=25, n_filters_spat=25, filter_time_length=10):
        layer_collection = {}
        inputs = InputLayer(shape_height=self.n_chans, shape_width=self.input_time_len)
        layer_collection[inputs.id] = inputs
        conv_time = ConvLayer(kernel_width=filter_time_length, kernel_height=1, filter_num=n_filters_time)
        layer_collection[conv_time.id] = conv_time
        inputs.make_connection(conv_time)
        conv_spat = ConvLayer(kernel_width=1, kernel_height=self.n_chans, filter_num=n_filters_spat)
        layer_collection[conv_spat.id] = conv_spat
        conv_time.make_connection(conv_spat)
        maxpool = MaxPoolLayer(pool_width=3, stride_width=3)
        layer_collection[maxpool.id] = maxpool
        conv_spat.make_connection(maxpool)
        return MyModel.new_model_from_structure(layer_collection)

    def finalize_model(self, model):
        output = model.layers[-1].output
        flatten_layer = Flatten()(output)
        prediction_layer = Dense(self.n_classes, activation='softmax', name='prediction_layer')(flatten_layer)
        model = MyModel(layer_collection=model.layer_collection, input=model.layers[0].input, output=prediction_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        plot_model(model, to_file='model.png')
        return model

    def add_conv_maxpool_block(self, model, num_of_ops):
        conv_width = random.randint(5, 10)
        conv_filter_num = random.randint(50, 100)
        maxpool_len = random.randint(3, 5)
        maxpool_stride = random.randint(1,3)

        output = model.layers[-1].output
        try:
            conv_layer = Conv2D(filters=conv_filter_num, kernel_size=(1, conv_width),
                                strides=(1, 1), activation='elu', name='convolution-'+str(num_of_ops))(output)
            maxpool_layer = MaxPool2D(pool_size=(1, maxpool_len), strides=(1, maxpool_stride))(conv_layer)
            model = MyModel(structure=model.structure, input=model.layers[0].input, output=maxpool_layer)
        except ValueError as e:
            print('failed to build new network with exception:', str(e))
            print('finalizing network')
            self.finalize_flag = 1

        model.structure.append(ConvLayer(kernel_width=1, kernel_height=conv_width, filter_num=conv_filter_num))
        model.structure.append(MaxPoolLayer(pool_width=maxpool_len, stride_width=maxpool_stride))

        return model

    def add_skip_connection_concat(self, model):
        topo_layers = create_topo_layers(model.layer_collection.values())
        to_concat = random.sample(range(Layer.running_id), 2)  # choose 2 random layer id's
        first_layer_index = np.min(to_concat)
        second_layer_index = np.max(to_concat)
        first_layer_index = topo_layers[first_layer_index]
        second_layer_index = topo_layers[second_layer_index]
        try:
            first_layer = model.get_layer(str(first_layer_index))
            second_layer = model.get_layer(str(second_layer_index))
        except ValueError:
            print('lala')
        first_shape = model.get_layer(str(first_layer_index)).output.shape
        second_shape = model.get_layer(str(second_layer_index)).output.shape
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

        return model.new_model_from_structure(model.layer_collection)


    def add_filters(self, model):
        conv_indices = [i for i, layer in enumerate(model.structure) if isinstance(layer, ConvLayer)]
        try:
            random_conv_index = random.randint(2, len(conv_indices) - 1)
        except ValueError:
            return model
        print('layer to widen is:', str(random_conv_index))
        factor = random.randint(2, 4)
        model.structure[conv_indices[random_conv_index]].filter_num *= factor
        return model.new_model_from_structure(copy.deepcopy(model.structure),
                                              copy.deepcopy(model.layer_collection), self)

