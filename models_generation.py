import torch
import traceback
import numpy as np
from keras_models import mean_layer
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var
from braindecode.models.util import to_dense_prediction_model
import os
from torch import nn
from torch.nn import init
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import random
import copy
import globals
from torchsummary import summary
WARNING = '\033[93m'
ENDC = '\033[0m'

def print_structure(structure):
    print('---------------model structure-----------------')
    for layer in structure.values():
        attrs = vars(layer)
        print('layer type:', layer.__class__.__name__)
    print('-----------------------------------------------')


class IdentityModule(nn.Module):
    def forward(self, inputs):
        return inputs


class Layer():

    def __init__(self, name=None):
        self.connections = []
        self.parent = None
        self.name = name

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return self.__dict__ != other.__dict__

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
    def __init__(self, kernel_eeg_chan=None, kernel_time=None, filter_num=None, name=None):
        Layer.__init__(self, name)
        if kernel_eeg_chan is None:
            kernel_eeg_chan = random.randint(1, globals.config['DEFAULT']['eeg_chans'])
        if kernel_time is None:
            kernel_time = random.randint(1, 20)
        if filter_num is None:
            filter_num = random.randint(1, 50)
        self.kernel_eeg_chan = kernel_eeg_chan
        self.kernel_time = kernel_time
        self.filter_num = filter_num


class PoolingLayer(Layer):
    # @initializer
    def __init__(self, pool_time=None, stride_time=None, mode='max', stride_eeg_chan=1, pool_eeg_chan=1):
        Layer.__init__(self)
        if pool_time is None:
            pool_time = random.randint(1, 3)
        if stride_time is None:
            stride_time = random.randint(1, 3)
        self.pool_time = pool_time
        self.stride_time = stride_time
        self.mode = mode
        self.stride_eeg_chan = stride_eeg_chan
        self.pool_eeg_chan = pool_eeg_chan


class IdentityLayer(Layer):
    def __init__(self):
        Layer.__init__(self)


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

    def __init__(self, model, layer_collection={}, name=None):
        self.layer_collection = layer_collection
        self.model = model
        self.name = name

    @staticmethod
    def new_model_from_structure_pytorch(layer_collection, applyFix=False):
        config = globals.config
        model = nn.Sequential()
        if globals.config['DEFAULT']['channel_dim'] != 'channels' or globals.config['DEFAULT']['exp_type'] == 'target':
            model.add_module('dimshuffle', Expression(MyModel._transpose_time_to_spat))
        if globals.config['DEFAULT']['time_factor'] != -1:
            model.add_module('stack_by_time', Expression(MyModel._stack_input_by_time))
        activations = {'elu': nn.ELU, 'softmax': nn.Softmax}
        for i in range(len(layer_collection)):
            layer = layer_collection[i]
            if i > 0:
                out = model.forward(np_to_var(np.ones(
                    (1, MyModel.n_chans, globals.config['DEFAULT']['input_time_len'], 1),
                    dtype=np.float32)))
                prev_channels = out.cpu().data.numpy().shape[1]
                prev_time = out.cpu().data.numpy().shape[2]
                prev_eeg_channels = out.cpu().data.numpy().shape[3]
            else:
                prev_eeg_channels = MyModel.n_chans
                prev_time = globals.config['DEFAULT']['input_time_len']
                prev_channels = 1

            if isinstance(layer, PoolingLayer):
                while applyFix and (prev_time-layer.pool_time) / layer.stride_time < 1:
                    if random.uniform(0,1) < 0.5 and layer.pool_time > 1:
                        layer.pool_time -= 1
                    elif layer.stride_time > 1:
                        layer.stride_time -=1
                if globals.config['DEFAULT']['channel_dim'] == 'channels':
                    layer.pool_eeg_chan = 1
                model.add_module('%s_%d' % (type(layer).__name__, i), nn.MaxPool2d(kernel_size=(layer.pool_time, layer.pool_eeg_chan),
                                                                          stride=(layer.stride_time, 1)))

            elif isinstance(layer, ConvLayer):
                if layer.kernel_time == 'down_to_one':
                    layer.kernel_time = prev_time
                    layer.kernel_eeg_chan = prev_eeg_channels
                    conv_name = 'conv_classifier'
                else:
                    conv_name = '%s_%d' % (type(layer).__name__, i)
                    if applyFix and layer.kernel_eeg_chan > prev_eeg_channels:
                        layer.kernel_eeg_chan = prev_eeg_channels
                    if applyFix and layer.kernel_time > prev_time:
                        layer.kernel_time = prev_time
                if globals.config['DEFAULT']['channel_dim'] == 'channels':
                    layer.kernel_eeg_chan = 1
                model.add_module(conv_name, nn.Conv2d(prev_channels, layer.filter_num,
                                                    (layer.kernel_time, layer.kernel_eeg_chan),
                                                    stride=1))

            elif isinstance(layer, BatchNormLayer):
                model.add_module('%s_%d' % (type(layer).__name__, i), nn.BatchNorm2d(prev_channels,
                                                                    momentum=config['DEFAULT']['batch_norm_alpha'],
                                                                    affine=True, eps=1e-5), )

            elif isinstance(layer, ActivationLayer):
                model.add_module('%s_%d' % (layer.activation_type, i), activations[layer.activation_type]())


            elif isinstance(layer, DropoutLayer):
                model.add_module('%s_%d' % (type(layer).__name__, i), nn.Dropout(p=config['DEFAULT']['dropout_p']))

            elif isinstance(layer, IdentityLayer):
                model.add_module('%s_%d' % (type(layer).__name__, i), IdentityModule())

            elif isinstance(layer, FlattenLayer):
                model.add_module('squeeze', Expression(MyModel._squeeze_final_output))

        if applyFix:
            return layer_collection

        init.xavier_uniform_(list(model._modules.items())[-3][1].weight, gain=1)
        init.constant_(list(model._modules.items())[-3][1].bias, 0)
        if not globals.config['DEFAULT']['cuda']:
            summary(model, (22, globals.config['DEFAULT']['input_time_len'], 1))
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
        return MyModel(layer_collection=layer_collection, model=model)

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

    @staticmethod
    def _stack_input_by_time(x):
        if globals.config['DEFAULT']['channel_dim'] == 'one':
            return x.view(x.shape[0], -1, int(x.shape[2] / globals.config['DEFAULT']['time_factor']), x.shape[3])
        else:
            return x.view(x.shape[0], x.shape[1], int(x.shape[2] / globals.config['DEFAULT']['time_factor']), -1)


def check_legal_model(layer_collection):
    try:
        finalize_model(layer_collection)
        return True
    except Exception as e:
        print('check legal model failed. Exception message: %s' % (str(e)))
        print(traceback.format_exc())
        return False


def random_layer():
    layers = [DropoutLayer, BatchNormLayer, ActivationLayer, ConvLayer, PoolingLayer, IdentityLayer]
    return layers[random.randint(0, 5)]()


def random_model(n_layers):
    layer_collection = []
    for i in range(n_layers):
        layer_collection.append(random_layer())
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return random_model(n_layers)


def uniform_model(n_layers, layer_type):
    layer_collection = []
    for i in range(n_layers):
        layer = layer_type()
        layer_collection.append(layer)
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return uniform_model(n_layers, layer_type)


def add_layer_to_state(new_model_state, layer, index, old_model_state):
    if type(layer).__name__ in ['BatchNormLayer', 'ConvLayer', 'PoolingLayer']:
        for k, v in old_model_state.items():
            if k.startswith('%s_%d' % (type(layer).__name__, index)) and \
                    k in new_model_state.keys() and new_model_state[k].shape == v.shape:
                new_model_state[k] = v


def breed_layers(first_model, second_model, first_model_state=None, second_model_state=None, cut_point=None):
    second_model = copy.deepcopy(second_model)
    save_weights = False
    if random.random() < globals.config['evolution']['breed_rate']:
        if cut_point is None:
            cut_point = random.randint(0, len(first_model) - 1)
        for i in range(cut_point):
            second_model[i] = first_model[i]
        if globals.config['evolution']['inherit_breeding_weights']:
            save_weights = True
    if random.random() < globals.config['evolution']['mutation_rate']:
        while True:
            rand_layer = random.randint(0, globals.config['evolution']['num_layers'] - 1)
            second_model[rand_layer] = random_layer()
            if check_legal_model(second_model):
                break
    new_model = MyModel.new_model_from_structure_pytorch(second_model, applyFix=True)
    if save_weights:
        finalized_new_model = finalize_model(new_model)
        finalized_new_model_state = finalized_new_model.model.state_dict()
        for i in range(cut_point):
            add_layer_to_state(finalized_new_model_state, second_model[i], i, first_model_state)
        for i in range(cut_point+1, globals.config['evolution']['num_layers']):
            add_layer_to_state(finalized_new_model_state, second_model[i-cut_point], i, second_model_state)
    else:
        finalized_new_model_state = None
    return new_model, finalized_new_model_state


def breed_filters(first, second):
    first_model = first['model']
    second_model = second['model']
    conv_indices_first = [layer.id for layer in first_model.values() if isinstance(layer, ConvLayer)]
    conv_indices_second = [layer.id for layer in second_model.values() if isinstance(layer, ConvLayer)]
    if random.random() < globals.config['evolution']['breed_rate']:
        cut_point = random.randint(0, len(conv_indices_first) - 1)
        for i in range(cut_point):
            second_model[conv_indices_second[i]].filter_num = first_model[conv_indices_first[i]].filter_num
    if random.random() < globals.config['DEFAULT']['mutation_rate']:
        random_rate = random.uniform(0.1,3)
        random_index = conv_indices_second[random.randint(2, len(conv_indices_second) - 2)]
        second_model[random_index].filter_num = \
            np.clip(int(second_model[random_index].filter_num * random_rate), 1, None)
    return second_model


def base_model(n_chans=22, n_filters_time=25, n_filters_spat=25,
               filter_time_length=10, random_filters=False):
    if random_filters:
        min_filt = globals.config['evolution']['random_filter_range_min']
        max_filt = globals.config['evolution']['random_filter_range_max']
    layer_collection = []
    conv_time = ConvLayer(kernel_time=filter_time_length, kernel_eeg_chan=1, filter_num=
                random.randint(min_filt, max_filt) if random_filters else n_filters_time)
    layer_collection.append(conv_time)
    conv_spat = ConvLayer(kernel_time=1, kernel_eeg_chan=n_chans, filter_num=
        random.randint(min_filt, max_filt) if random_filters else n_filters_spat)
    layer_collection.append(conv_spat)
    batchnorm = BatchNormLayer()
    layer_collection.append(batchnorm)
    elu = ActivationLayer()
    layer_collection.append(elu)
    maxpool = PoolingLayer(pool_time=3, stride_time=3, mode='MAX')
    layer_collection.append(maxpool)
    return layer_collection


def target_model():
    basemodel = base_model()
    model = add_conv_maxpool_block(basemodel, conv_filter_num=50, conv_layer_name='conv1', random_values=False)
    model = add_conv_maxpool_block(model, conv_filter_num=100, conv_layer_name='conv2', random_values=False)
    model = add_conv_maxpool_block(model, conv_filter_num=200, dropout=True, conv_layer_name='conv3',
                                   random_values=False)
    return model


def genetic_filter_experiment_model(num_blocks):
    layer_collection = base_model(random_filters=True)
    min_filt = globals.config['evolution']['random_filter_range_min']
    max_filt = globals.config['evolution']['random_filter_range_max']
    for block in range(num_blocks):
        add_layer(layer_collection, DropoutLayer(), in_place=True)
        add_layer(layer_collection, ConvLayer(filter_num=random.randint(min_filt, max_filt), kernel_eeg_chan=1, kernel_time=10),
                  in_place=True)
        add_layer(layer_collection, BatchNormLayer(), in_place=True)
        add_layer(layer_collection, PoolingLayer(stride_time=2, pool_time=2, mode='MAX'), in_place=True)
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


def finalize_model(layer_collection):
    layer_collection = copy.deepcopy(layer_collection)
    if globals.config['DEFAULT']['cropping']:
        final_conv_time = globals.config['DEFAULT']['final_conv_size']
    else:
        final_conv_time = 'down_to_one'
    conv_layer = ConvLayer(kernel_time=final_conv_time, kernel_eeg_chan=1,
                           filter_num=globals.config['DEFAULT']['n_classes'])
    layer_collection.append(conv_layer)
    softmax = ActivationLayer('softmax')
    layer_collection.append(softmax)
    if globals.config['DEFAULT']['cropping']:
        mean = LambdaLayer(mean_layer)
        layer_collection.append(mean)
    flatten = FlattenLayer()
    layer_collection.append(flatten)
    return MyModel.new_model_from_structure_pytorch(layer_collection)


def add_conv_maxpool_block(layer_collection, conv_width=10, conv_filter_num=50, dropout=False,
                           pool_width=3, pool_stride=3, conv_layer_name=None, random_values=True):
    layer_collection = copy.deepcopy(layer_collection)
    if random_values:
        conv_time = random.randint(5, 10)
        conv_filter_num = random.randint(0, 50)
        pool_time = 2
        pool_stride = 2

    if dropout:
        dropout = DropoutLayer()
        layer_collection.append(dropout)
    conv_layer = ConvLayer(kernel_time=conv_width, kernel_eeg_chan=1,
                           filter_num=conv_filter_num, name=conv_layer_name)
    layer_collection.append(conv_layer)
    batchnorm_layer = BatchNormLayer()
    layer_collection.append(batchnorm_layer)
    activation_layer = ActivationLayer()
    layer_collection.append(activation_layer)
    maxpool_layer = PoolingLayer(pool_time=pool_width, stride_time=pool_stride, mode='MAX')
    layer_collection.append(maxpool_layer)
    return layer_collection

