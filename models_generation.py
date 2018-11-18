from keras.models import Model
from keras.layers import Conv2D, Flatten, Activation, Lambda, Dropout, Input, Cropping2D, Concatenate, ZeroPadding2D, BatchNormalization
import numpy as np
from toposort import toposort_flatten
from keras_models import dilation_pool, mean_layer
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var
from utils import initializer
import os
from torch import nn
from torch.nn import init
from torchsummary import summary
from torch.nn.functional import elu
import configparser
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import random
import copy
WARNING = '\033[93m'
ENDC = '\033[0m'

def print_structure(structure):
    print('---------------model structure-----------------')
    for layer in structure.values():
        attrs = vars(layer)
        print('layer type:', layer.__class__.__name__)
    print('-----------------------------------------------')


def create_topo_layers(layers):
    layer_dict = {}
    for layer in layers:
        layer_dict[layer.id] = {x.id for x in layer.connections}
    return list(reversed(toposort_flatten(layer_dict)))


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
    # @initializer
    def __init__(self, kernel_eeg_chan, kernel_time, filter_num, name=None):
        Layer.__init__(self, name)
        self.kernel_eeg_chan = kernel_eeg_chan
        self.kernel_time = kernel_time
        self.filter_num = filter_num


class PoolingLayer(Layer):
    # @initializer
    def __init__(self, pool_time, stride_time, mode, stride_eeg_chan=1, pool_eeg_chan=1):
        Layer.__init__(self)
        self.pool_time = pool_time
        self.stride_time = stride_time
        self.mode = mode
        self.stride_eeg_chan = stride_eeg_chan
        self.pool_eeg_chan = pool_eeg_chan


# class CroppingLayer(Layer):
#     @initializer
#     def __init__(self, height_crop_top, height_crop_bottom, width_crop_left, width_crop_right):
#         Layer.__init__(self)



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


class MyModel:
    n_chans = 22
    input_time = 1125

    def __init__(self, model, layer_collection={}, name=None):
        self.layer_collection = layer_collection
        self.model = model
        self.name = name

    @staticmethod
    def new_model_from_structure_pytorch(layer_collection, name=None):
        config = configparser.ConfigParser()
        config.read('config.ini')
        topo_layers = create_topo_layers(layer_collection.values())
        model = nn.Sequential()
        model.add_module('dimshuffle', Expression(MyModel._transpose_time_to_spat))
        activations = {'elu': nn.ELU, 'softmax': nn.Softmax}
        for index, i in enumerate(topo_layers):
            layer = layer_collection[i]
            if index > 0:
                out = model.forward(np_to_var(np.ones(
                    (1, MyModel.n_chans, MyModel.input_time, 1),
                    dtype=np.float32)))
                prev_channels = out.cpu().data.numpy().shape[1]
                prev_time = out.cpu().data.numpy().shape[2]
                prev_eeg_channels = out.cpu().data.numpy().shape[3]
            else:
                prev_eeg_channels = MyModel.n_chans
                prev_time = MyModel.input_time
                prev_channels = 1

            if isinstance(layer, PoolingLayer):
                model.add_module('pool_%d' % (layer.id), nn.MaxPool2d(kernel_size=(layer.pool_time, layer.pool_eeg_chan),
                                                                          stride=(layer.stride_time, 1)))

            elif isinstance(layer, ConvLayer):
                if layer.kernel_time == 'down_to_one':
                    layer.kernel_time = prev_time
                    layer.kernel_eeg_chan = prev_eeg_channels
                    conv_name = 'conv_classifier'
                else:
                    conv_name = 'conv_%d' % (layer.id)
                model.add_module(conv_name, nn.Conv2d(prev_channels, layer.filter_num,
                                                    (layer.kernel_time, layer.kernel_eeg_chan),
                                                    stride=1))

            elif isinstance(layer, BatchNormLayer):
                model.add_module('bnorm_%d' % (layer.id), nn.BatchNorm2d(prev_channels,
                                                                            momentum=config['DEFAULT'].getfloat(
                                                                                'batch_norm_alpha'),
                                                                            affine=True, eps=1e-5), )

            elif isinstance(layer, ActivationLayer):
                model.add_module('%s_%d' % (layer.activation_type, layer.id), activations[layer.activation_type]())


            elif isinstance(layer, DropoutLayer):
                model.add_module('dropout_%d' % (layer.id), nn.Dropout(p=config['DEFAULT'].getfloat('dropout_p')))

            elif isinstance(layer, FlattenLayer):
                model.add_module('squeeze', Expression(MyModel._squeeze_final_output))

        init.xavier_uniform_(model.conv_classifier.weight, gain=1)
        init.constant_(model.conv_classifier.bias, 0)
        summary(model, (22, 1125, 1))
        return model

    # remove empty dim at end and potentially remove empty time dim
    # do not just use squeeze as we never want to remove first dim
    @staticmethod
    def _squeeze_final_output(x):
        assert x.size()[3] == 1
        x = x[:, :, :, 0]
        if x.size()[2] == 1:
            x = x[:, :, 0]
        return x

    @staticmethod
    def _transpose_time_to_spat(x):
        return x.permute(0, 3, 2, 1)


def random_model(n_chans, input_time_len):
    layer_collection = {}

    # random variables
    filter_time_length = random.randint(1, 50)
    n_filters_time = random.randint(1, 25)
    n_filters_spat = random.randint(1, 25)
    batchnorm_prob = random.random()
    activation_prob = random.random()
    pool_prob = random.random()

    inputs = InputLayer(shape_height=n_chans, shape_width=input_time_len)
    layer_collection[inputs.id] = inputs
    conv_time = ConvLayer(kernel_width=filter_time_length, kernel_height=1, filter_num=n_filters_time)
    layer_collection[conv_time.id] = conv_time
    inputs.make_connection(conv_time)
    conv_spat = ConvLayer(kernel_width=1, kernel_height=n_chans, filter_num=n_filters_spat)
    layer_collection[conv_spat.id] = conv_spat
    conv_time.make_connection(conv_spat)

    if random.random() > batchnorm_prob:
        batchnorm = BatchNormLayer()
        layer_collection[batchnorm.id] = batchnorm
        conv_spat.make_connection(batchnorm)
    else:
        batchnorm = conv_spat

    if random.random() > activation_prob:
        elu = ActivationLayer()
        layer_collection[elu.id] = elu
        batchnorm.make_connection(elu)
    else:
        elu = batchnorm

    if random.random() > pool_prob:
        maxpool = PoolingLayer(pool_width=3, stride_width=3, mode='MAX')
        layer_collection[maxpool.id] = maxpool
        elu.make_connection(maxpool)

    return MyModel.new_model_from_structure(layer_collection=layer_collection, name='base')


def base_model(n_chans=22, input_time_len=1125, n_filters_time=25, n_filters_spat=25, filter_time_length=10):
    layer_collection = {}
    conv_time = ConvLayer(kernel_time=filter_time_length, kernel_eeg_chan=1, filter_num=n_filters_time)
    layer_collection[conv_time.id] = conv_time
    conv_spat = ConvLayer(kernel_time=1, kernel_eeg_chan=n_chans, filter_num=n_filters_spat)
    layer_collection[conv_spat.id] = conv_spat
    conv_time.make_connection(conv_spat)
    batchnorm = BatchNormLayer()
    layer_collection[batchnorm.id] = batchnorm
    conv_spat.make_connection(batchnorm)
    elu = ActivationLayer()
    layer_collection[elu.id] = elu
    batchnorm.make_connection(elu)
    maxpool = PoolingLayer(pool_time=3, stride_time=3, mode='MAX')
    layer_collection[maxpool.id] = maxpool
    elu.make_connection(maxpool)
    return layer_collection


def target_model():
    basemodel = base_model()
    model = add_conv_maxpool_block(basemodel, conv_filter_num=50, conv_layer_name='conv1', random_values=False)
    model = add_conv_maxpool_block(model, conv_filter_num=100, conv_layer_name='conv2', random_values=False)
    model = add_conv_maxpool_block(model, conv_filter_num=200, dropout=True, conv_layer_name='conv3',
                                   random_values=False)
    return model


def genetic_filter_experiment_model(num_blocks):
    layer_collection = base_model()
    for block in range(num_blocks):
        add_layer(layer_collection, DropoutLayer(), in_place=True)
        add_layer(layer_collection, ConvLayer(filter_num=random.randint(1, 1000), kernel_height=1, kernel_width=10),
                  in_place=True)
        add_layer(layer_collection, BatchNormLayer(), in_place=True)
        add_layer(layer_collection, PoolingLayer(stride_width=2, pool_width=2, mode='MAX'), in_place=True)
    return layer_collection


def add_layer(layer_collection, layer_to_add, in_place=False):
    if not in_place:
        for layer in layer_collection.values():
            layer.keras_layer = None
        layer_collection = copy.deepcopy(layer_collection)
    topo_layers = create_topo_layers(layer_collection.values())
    last_layer_id = topo_layers[-1]
    layer_collection[layer_to_add.id] = layer_to_add
    layer_collection[last_layer_id].make_connection(layer_to_add)
    return layer_collection


def breed(first_model, second_model, mutation_rate, breed_rate):
    conv_indices_first = [layer.id for layer in first_model.layer_collection.values() if isinstance(layer, ConvLayer)]
    conv_indices_second = [layer.id for layer in second_model.layer_collection.values() if isinstance(layer, ConvLayer)]
    if random.random() < breed_rate:
        cut_point = random.randint(2, len(conv_indices_first) - 2)  # don't include last conv
        for i in range(cut_point):
            second_model.layer_collection[conv_indices_second[i]].filter_num = first_model.layer_collection[conv_indices_first[i]].filter_num
    if random.random() < mutation_rate:
        random_rate = random.uniform(0.1,3)
        random_index = conv_indices_second[random.randint(2, len(conv_indices_second) - 2)]
        second_model.layer_collection[random_index].filter_num = \
            np.clip(int(second_model.layer_collection[random_index].filter_num * random_rate), 1, None)
    return second_model


def finalize_model(layer_collection, naive_nas):
    # for layer in layer_collection.values():
    #     layer.keras_layer = None
    layer_collection = copy.deepcopy(layer_collection)
    topo_layers = create_topo_layers(layer_collection.values())
    last_layer_id = topo_layers[-1]
    if naive_nas.cropping:
        final_conv_time = 2
    else:
        final_conv_time = 'down_to_one'
    conv_layer = ConvLayer(kernel_time=final_conv_time, kernel_eeg_chan=1, filter_num=naive_nas.n_classes)
    layer_collection[conv_layer.id] = conv_layer
    layer_collection[last_layer_id].make_connection(conv_layer)
    softmax = ActivationLayer('softmax')
    layer_collection[softmax.id] = softmax
    conv_layer.make_connection(softmax)
    if naive_nas.cropping:
        mean = LambdaLayer(mean_layer)
        layer_collection[mean.id] = mean
        softmax.make_connection(mean)
    flatten = FlattenLayer()
    layer_collection[flatten.id] = flatten
    if naive_nas.cropping:
        mean.make_connection(flatten)
    else:
        softmax.make_connection(flatten)
    return MyModel.new_model_from_structure_pytorch(layer_collection)


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

