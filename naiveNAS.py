from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, Activation, MaxPool2D, Lambda, Dropout, Input, Cropping2D, Concatenate, ZeroPadding2D
from keras.utils import to_categorical
import copy
from simanneal import Annealer
import numpy as np
import time
from keras.callbacks import EarlyStopping

import random


class Layer:
    running_id = 0
    def __init__(self):
        self.id = Layer.running_id
        Layer.running_id += 1


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

class MyModel(Model):
    def __init__(self, structure=[], *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)
        self.structure = structure

    def new_model_from_structure(self, structure):
        model_layers = []

        for layer in structure:
            if isinstance(layer, InputLayer):
                keras_layer = Input(shape=(layer.shape_height, layer.shape_width, 1))
                x = keras_layer
            elif isinstance(layer, MaxPoolLayer):
                keras_layer = MaxPool2D(pool_size=(1, layer.pool_width), strides=(1, layer.stride_width))
                x = keras_layer(x)
            elif isinstance(layer, ConvLayer):
                keras_layer = Conv2D(filters=layer.filter_num, kernel_size=(layer.kernel_height, layer.kernel_width),
                           strides=(1, 1), activation='elu')
                x = keras_layer(x)
            elif isinstance(layer, CroppingLayer):
                keras_layer = Cropping2D(((layer.height_crop_top, layer.height_crop_bottom),
                                (layer.width_crop_left, layer.width_crop_right)))
                x = keras_layer(x)
            elif isinstance(layer, ZeroPadLayer):
                keras_layer = ZeroPadding2D(((layer.height_pad_top, layer.height_pad_bottom),
                                   (layer.width_pad_left, layer.width_pad_right)))
                x = keras_layer(x)
            elif isinstance(layer, ConcatLayer):
                print_structure(structure)
                print('about to concatenate layers:', layer.first_layer_index, 'and:', layer.second_layer_index)
                keras_layer = Concatenate()([next((x for x, id in model_layers if id == layer.first_layer_index), None),
                                             next((x for x, id in model_layers if id == layer.second_layer_index), None)])
                x = keras_layer
            model_layers.append((x, layer.id))
        return MyModel(structure=structure, inputs=model_layers[0][0], output=x)


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
            model = self.add_skip_connection_concat(curr_model, num_of_ops)
            final_model = self.finalize_model(model)
            final_model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                      callbacks=[earlystopping])
            res = final_model.evaluate(self.X_test, self.y_test) * 100
            if res[1] >= curr_acc:
                curr_model = model
            else:
                probability = np.exp((res[1] - curr_acc) / temperature)
                rand = np.random.choice(a=1, p=[1-probability, probability])
                if rand == 1:
                    curr_model = model
            temperature *= (1-coolingRate)
            print('model accuracy:', res[1] * 100)

        final_model = self.finalize_model(curr_model)
        final_model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                        callbacks=[earlystopping])
        res = final_model.evaluate(self.X_test, self.y_test) * 100
        print('model accuracy:', res[1] * 100)

    def base_model(self, n_filters_time=25, n_filters_spat=25, filter_time_length=10):
        inputs = Input(shape=(self.n_chans, self.input_time_len, 1))
        temporal_conv = Conv2D(name='temporal_convolution', filters=n_filters_time, input_shape=(self.n_chans, self.input_time_len, 1),
                         kernel_size=(1, filter_time_length), strides=(1,1), activation='elu')(inputs)

        # note that this is a different implementation from the paper!
        # they didn't put an activation function between the first two convolutions

        # Also, in the paper they implemented batch-norm before each non-linearity - which I didn't do!

        # Also, they added dropout for each input the conv layers except the first! I dropped out only in the end

        spatial_conv = Conv2D(name='spatial_convolution', filters=n_filters_spat,
                              kernel_size=(self.n_chans, 1), strides=(1,1), activation='elu')(temporal_conv)
        maxpool = MaxPool2D(pool_size=(1, 3), strides=(1, 3))(spatial_conv)
        model = MyModel(inputs=inputs, outputs=maxpool)
        model.structure.clear()
        model.structure.append(InputLayer(shape_height=self.n_chans, shape_width=self.input_time_len))
        model.structure.append(ConvLayer(kernel_width=filter_time_length, kernel_height=1, filter_num=n_filters_time))
        model.structure.append(ConvLayer(kernel_width=1, kernel_height=self.n_chans, filter_num=n_filters_spat))
        model.structure.append(MaxPoolLayer(pool_width=3, stride_width=3))
        # model.structure.append('input-'+str(self.n_chans)+'-'+str(self.input_time_len))
        # model.structure.append('convolution-'+str(n_filters_time)+'-1-'+str(filter_time_length))
        # model.structure.append('convolution-'+str(n_filters_spat)+'-'+str(self.n_chans)+'-1')
        # model.structure.append('maxpool-3-3')
        return model

    def finalize_model(self, model):
        output = model.layers[-1].output
        flatten_layer = Flatten()(output)
        prediction_layer = Dense(self.n_classes, activation='softmax', name='prediction_layer')(flatten_layer)
        model = MyModel(structure=model.structure, input=model.layers[0].input, output=prediction_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
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

        # model.structure.append('convolution-'+str(conv_filter_num)+'-1-'+str(conv_width))
        # model.structure.append('maxpool-'+str(maxpool_len)+'-'+str(maxpool_stride))
        return model

    def add_skip_connection_concat(self, model, num_of_ops):
        to_concat = random.sample(range(len(model.structure)), 2)
        first_layer_index = np.min(to_concat)
        second_layer_index = np.max(to_concat)
        first_layer = model.layers[first_layer_index]
        second_layer = model.layers[second_layer_index]
        first_shape = first_layer.output_shape
        second_shape = second_layer.output_shape
        print('first layer shape is:', first_shape)
        print('second layer shape is:', second_shape)
        print('first layer index for concatLayer is:', first_layer_index, 'second layer index is:', second_layer_index)
        print('-----------model summary in add skip connection------------')
        model.summary()
        print('-----------model structure in add skip connection----------')
        print_structure(model.structure)
        height_diff = second_shape[1] - first_shape[1]
        width_diff = second_shape[2] - first_shape[2]
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
        layerdiff = 0
        if height_diff != 0:
            layerdiff += 1
            heightChanger = ChosenHeightClass(height_crop_top, height_crop_bottom, 0, 0)
            model.structure.insert(second_layer_index + layerdiff, heightChanger)
        if width_diff != 0:
            layerdiff += 1
            widthChanger = ChosenWidthClass(0, 0, width_crop_left, width_crop_right)
            model.structure.insert(second_layer_index + layerdiff, widthChanger)
        model.structure.insert(second_layer_index + layerdiff + 1, ConcatLayer(
            model.structure[first_layer_index].id, model.structure[second_layer_index + layerdiff].id))
        # model.structure.insert(second_layer_index + 1, 'cropping-' + str(height_diff / 2) + '-' +
        #                        str(width_diff / 2))
        # model.structure.insert(second_layer_index+2, 'concatenate-'+str(first_layer_index)+'-'+
        #                         str(second_layer_index+1))
        return model.new_model_from_structure(copy.deepcopy(model.structure))


    def add_filters(self, model):
        conv_indices = [i for i, layer in enumerate(model.structure) if isinstance(layer, ConvLayer)]
        try:
            random_conv_index = random.randint(2, len(conv_indices) - 1)
        except ValueError:
            return model
        print('layer to widen is:', str(random_conv_index))
        factor = random.randint(2, 4)
        # convolution, depth, height, width = model.structure[conv_indices[random_conv_index]].split('-')
        # model.structure[conv_indices[random_conv_index]] = convolution+'-'+str(int(depth)*factor)+'-'+height+'-'+width
        model.structure[conv_indices[random_conv_index]].filter_num *= factor
        return model.new_model_from_structure(copy.deepcopy(model.structure), self)

