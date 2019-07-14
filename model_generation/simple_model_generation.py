import copy
import random
import numpy as np
from braindecode.torch_ext.util import np_to_var
from model_generation.abstract_layers import *
from model_generation.custom_modules import *
from torch import nn
from torch.nn import init

from model_generation.custom_modules import _squeeze_final_output


def new_model_from_structure_pytorch(layer_collection, applyFix=False, check_model=False):
    model = nn.Sequential()
    if globals.get('channel_dim') != 'channels' or globals.get('exp_type') == 'target':
        model.add_module('dimshuffle', _transpose(shape=[0, 3, 2, 1]))
    if globals.get('time_factor') != -1:
        model.add_module('stack_by_time', Expression(_stack_input_by_time))
    activations = {'elu': nn.ELU, 'softmax': nn.Softmax, 'sigmoid': nn.Sigmoid}
    input_shape = (2, globals.get('eeg_chans'), globals.get('input_time_len'), 1)
    for i in range(len(layer_collection)):
        layer = layer_collection[i]
        if i > 0:
            out = model.forward(np_to_var(np.ones(
                input_shape,
                dtype=np.float32)))
            prev_channels = out.cpu().data.numpy().shape[1]
            prev_time = out.cpu().data.numpy().shape[2]
            prev_eeg_channels = out.cpu().data.numpy().shape[3]
        else:
            prev_eeg_channels = globals.get('eeg_chans')
            prev_time = globals.get('input_time_len')
            prev_channels = 1
            if globals.get('channel_dim') == 'channels':
                prev_channels = globals.get('eeg_chans')
                prev_eeg_channels = 1
        if isinstance(layer, PoolingLayer):
            while applyFix and (prev_time-layer.pool_time) / layer.stride_time < 1:
                if random.uniform(0,1) < 0.5 and layer.pool_time > 1:
                    layer.pool_time -= 1
                elif layer.stride_time > 1:
                    layer.stride_time -= 1
                if layer.pool_time == 1 and layer.stride_time == 1:
                    break
            if globals.get('channel_dim') == 'channels':
                layer.pool_eeg_chan = 1
            model.add_module('%s_%d' % (type(layer).__name__, i), nn.MaxPool2d(kernel_size=(int(layer.pool_time), int(layer.pool_eeg_chan)),
                                                                      stride=(int(layer.stride_time), 1)))

        elif isinstance(layer, ConvLayer):
            if layer.kernel_time == 'down_to_one' or i >= globals.get('num_layers'):
                layer.kernel_time = prev_time
                layer.kernel_eeg_chan = prev_eeg_channels
                conv_name = 'conv_classifier'
            else:
                conv_name = '%s_%d' % (type(layer).__name__, i)
                if applyFix and layer.kernel_eeg_chan > prev_eeg_channels:
                    layer.kernel_eeg_chan = prev_eeg_channels
                if applyFix and layer.kernel_time > prev_time:
                    layer.kernel_time = prev_time
            if globals.get('channel_dim') == 'channels':
                layer.kernel_eeg_chan = 1
            model.add_module(conv_name, nn.Conv2d(prev_channels, layer.filter_num,
                                                (layer.kernel_time, layer.kernel_eeg_chan),
                                                stride=1))

        elif isinstance(layer, BatchNormLayer):
            model.add_module('%s_%d' % (type(layer).__name__, i), nn.BatchNorm2d(prev_channels,
                                                                momentum=globals.get('batch_norm_alpha'),
                                                                affine=True, eps=1e-5), )

        elif isinstance(layer, ActivationLayer):
            model.add_module('%s_%d' % (layer.activation_type, i), activations[layer.activation_type]())


        elif isinstance(layer, DropoutLayer):
            model.add_module('%s_%d' % (type(layer).__name__, i), nn.Dropout(p=globals.get('dropout_p')))

        elif isinstance(layer, IdentityLayer):
            model.add_module('%s_%d' % (type(layer).__name__, i), IdentityModule())

        elif isinstance(layer, FlattenLayer):
            model.add_module('squeeze', _squeeze_final_output())

    if applyFix:
        return layer_collection
    if check_model:
        return
    init.xavier_uniform_(list(model._modules.items())[-3][1].weight, gain=1)
    init.constant_(list(model._modules.items())[-3][1].bias, 0)
    return model


def check_legal_model(layer_collection):
    if globals.get('channel_dim') == 'channels':
        input_chans = 1
    else:
        input_chans = globals.get('eeg_chans')
    input_time = globals.get('input_time_len')
    for layer in layer_collection:
        if type(layer) == ConvLayer:
            input_time = (input_time - layer.kernel_time) + 1
            input_chans = (input_chans - layer.kernel_eeg_chan) + 1
        elif type(layer) == PoolingLayer:
            input_time = (input_time - layer.pool_time) / layer.stride_time + 1
            input_chans = (input_chans - layer.pool_eeg_chan) / layer.stride_eeg_chan + 1
        if input_time < 1 or input_chans < 1:
            print(f"illegal model, input_time={input_time}, input_chans={input_chans}")
            return False
    if globals.get('cropping'):
        return check_legal_cropping_model(layer_collection)
    return True


def uniform_model(n_layers, layer_type):
    layer_collection = []
    for i in range(n_layers):
        layer = layer_type()
        layer_collection.append(layer)
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return uniform_model(n_layers, layer_type)


def custom_model(layers):
    layer_collection = []
    for layer in layers:
        layer_collection.append(layer())
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return custom_model(layers)


def random_layer():
    layers = [DropoutLayer, BatchNormLayer, ActivationLayer, ConvLayer, PoolingLayer, IdentityLayer]
    return layers[random.randint(0, 5)]()


def random_model(n_layers):
    layer_collection = []
    for i in range(n_layers):
        if globals.get('simple_start'):
            layer_collection.append(IdentityLayer())
        else:
            layer_collection.append(random_layer())
    if check_legal_model(layer_collection):
        return layer_collection
    else:
        return random_model(n_layers)


def add_layer_to_state(new_model_state, layer, index, old_model_state):
    if type(layer).__name__ in ['BatchNormLayer', 'ConvLayer', 'PoolingLayer']:
        for k, v in old_model_state.items():
            if '%s_%d' % (type(layer).__name__, index) in k and \
                    k in new_model_state.keys() and new_model_state[k].shape == v.shape:
                new_model_state[k] = v


def finalize_model(layer_collection):
    if globals.get('grid'):
        return ModelFromGrid(layer_collection)
    layer_collection = copy.deepcopy(layer_collection)
    if globals.get('cropping'):
        final_conv_time = globals.get('final_conv_size')
    else:
        final_conv_time = 'down_to_one'
    conv_layer = ConvLayer(kernel_time=final_conv_time, kernel_eeg_chan=1,
                           filter_num=globals.get('n_classes'))
    layer_collection.append(conv_layer)
    if globals.get('problem') == 'classification':
        activation = ActivationLayer('softmax')
        layer_collection.append(activation)
    flatten = FlattenLayer()
    layer_collection.append(flatten)
    return new_model_from_structure_pytorch(layer_collection)
