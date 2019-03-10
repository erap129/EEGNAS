import torch
import numpy as np
from Bio.pairwise2 import format_alignment
from braindecode.torch_ext.modules import Expression
from braindecode.torch_ext.util import np_to_var
from braindecode.models import deep4, shallow_fbcsp, eegnet
import os
from torch import nn
from torch.nn import init
import random
import copy
import globals
from Bio import pairwise2
from collections import defaultdict
import networkx as nx
from networkx.classes.function import create_empty_copy
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
WARNING = '\033[93m'
ENDC = '\033[0m'

def print_structure(structure):
    print('---------------model structure-----------------')
    for layer in structure.values():
        attrs = vars(layer)
        print('layer type:', layer.__class__.__name__)
    print('-----------------------------------------------')


class Lattice:
    def __init__(self, layers, connections):
        self.layers = layers
        self.connections = connections


class IdentityModule(nn.Module):
    def forward(self, inputs):
        return inputs


class Layer():
    def __init__(self, name=None):
        self.connections = []
        self.name = name

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return self.__dict__ != other.__dict__

    def make_connection(self, other):
        self.connections.append(other)


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
            kernel_eeg_chan = random.randint(1, globals.get('kernel_height_max'))
            # kernel_eeg_chan = random.randint(1, globals.get('eeg_chans'))
        if kernel_time is None:
            kernel_time = random.randint(1, globals.get('kernel_time_max'))
        if filter_num is None:
            filter_num = random.randint(1, globals.get('filter_num_max'))
        if globals.get('channel_dim') == 'channels':
            kernel_eeg_chan = 1
        self.kernel_eeg_chan = kernel_eeg_chan
        self.kernel_time = kernel_time
        self.filter_num = filter_num


class PoolingLayer(Layer):
    # @initializer
    def __init__(self, pool_time=None, stride_time=None, mode='max', stride_eeg_chan=1, pool_eeg_chan=1):
        Layer.__init__(self)
        if pool_time is None:
            pool_time = random.randint(1, globals.get('pool_time_max'))
        if stride_time is None:
            stride_time = random.randint(1, globals.get('pool_time_max'))
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


def string_representation(layer_collection):
    translation = {FlattenLayer: 'f',
                   DropoutLayer: 'd',
                   BatchNormLayer: 'b',
                   ConvLayer: 'c',
                   PoolingLayer: 'p',
                   ActivationLayer: 'a',
                   IdentityLayer: 'i'}
    rep = ''
    for layer in layer_collection:
        rep += translation[type(layer)]
    return rep


def get_layer(layer_collection, layer_type, order):
    count = 0
    for layer in layer_collection:
        if type(layer) == layer_type:
            count += 1
            if count == order:
                return layer
    print(f"searched for layer {str(layer_type)} but didn't find.\n"
          f"the layer collection is: {str(layer_collection)}\n"
          f"the order is: {order}")


def layer_comparison(layer_type, layer1_order, layer2_order, layer_collection1, layer_collection2, attrs, output):
    score = 0
    layer1 = get_layer(layer_collection1, layer_type, layer1_order)
    layer2 = get_layer(layer_collection2, layer_type, layer2_order)
    for attr in attrs:
        added_value = 1 / (abs(getattr(layer1, attr) - getattr(layer2, attr)) + 1) * 5
        score += added_value
        output.append(f"{layer_type.__name__}_{layer1_order}_{layer2_order}"
                      f" with attribute {attr}, added value : 1 / abs({getattr(layer1, attr)}"
                      f" - {getattr(layer2, attr)}) + 1 * 5 = {added_value:.3f}")
    return score


def network_similarity(layer_collection1, layer_collection2, return_output=False):
    str1 = string_representation(layer_collection1)
    str2 = string_representation(layer_collection2)
    alignment = pairwise2.align.globalms(str1, str2, 2, -1, -.5, -.1)[0]
    output = ['-' * 50]
    output.append(format_alignment(*alignment))
    score = alignment[2]
    str1_orders = defaultdict(lambda:0)
    str2_orders = defaultdict(lambda:0)
    for x,y in (zip(alignment[0], alignment[1])):
        str1_orders[x] += 1
        str2_orders[y] += 1
        if x == y == 'c':
            score += layer_comparison(ConvLayer, str1_orders['c'], str2_orders['c'],
                                      layer_collection1, layer_collection2,
                                      ['kernel_eeg_chan', 'filter_num', 'kernel_time'], output)
        if x == y == 'p':
            score += layer_comparison(PoolingLayer, str1_orders['p'], str2_orders['p'],
                                      layer_collection1, layer_collection2,
                                      ['pool_time', 'stride_time'], output)
    output.append(f"final similarity: {score:.3f}")
    output.append('-' * 50)
    if return_output:
        return score, '\n'.join(output)
    else:
        return score


def new_model_from_structure_pytorch(layer_collection, applyFix=False, check_model=False):
    model = nn.Sequential()
    if globals.get('channel_dim') != 'channels' or globals.get('exp_type') == 'target':
        model.add_module('dimshuffle', Expression(MyModel._transpose_time_to_spat))
    if globals.get('time_factor') != -1:
        model.add_module('stack_by_time', Expression(MyModel._stack_input_by_time))
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
            model.add_module('squeeze', Expression(MyModel._squeeze_final_output))

    if applyFix:
        return layer_collection
    if check_model:
        return
    init.xavier_uniform_(list(model._modules.items())[-3][1].weight, gain=1)
    init.constant_(list(model._modules.items())[-3][1].bias, 0)
    return model


def generate_conv_layer(layer, in_chans, prev_time):
    if layer.kernel_time == 'down_to_one':
        layer.kernel_time = prev_time
        layer.filter_num = globals.get('n_classes')
    if globals.get('channel_dim') == 'channels':
        layer.kernel_eeg_chan = 1
    return nn.Conv2d(in_chans, layer.filter_num, (layer.kernel_time, layer.kernel_eeg_chan), stride=1)


def generate_pooling_layer(layer, in_chans, prev_time):
    if globals.get('channel_dim') == 'channels':
        layer.pool_eeg_chan = 1
    return nn.MaxPool2d(kernel_size=(int(layer.pool_time), int(layer.pool_eeg_chan)),
                                  stride=(int(layer.stride_time), 1))


def generate_batchnorm_layer(layer, in_chans, prev_time):
    return nn.BatchNorm2d(in_chans, momentum=globals.get('batch_norm_alpha'), affine=True, eps=1e-5)


def generate_activation_layer(layer, in_chans, prev_time):
    activations = {'elu': nn.ELU, 'softmax': nn.Softmax, 'sigmoid': nn.Sigmoid}
    return activations[layer.activation_type]()


def generate_dropout_layer(layer, in_chans, prev_time):
    return nn.Dropout(p=globals.get('dropout_p'))


def generate_identity_layer(layer, in_chans, prev_time):
    return IdentityModule()


def generate_flatten_layer(layer, in_chans, prev_time):
    return Expression(MyModel._squeeze_final_output)


class ModelFromGrid(torch.nn.Module):
    def __init__(self, layer_grid):
        super(ModelFromGrid, self).__init__()
        self.generate_pytorch_layer = {
            ConvLayer: generate_conv_layer,
            PoolingLayer: generate_pooling_layer,
            BatchNormLayer: generate_batchnorm_layer,
            ActivationLayer: generate_activation_layer,
            DropoutLayer: generate_dropout_layer,
            IdentityLayer: generate_identity_layer,
            FlattenLayer: generate_flatten_layer
        }
        layers = layer_grid.copy()
        self.pytorch_layers = nn.ModuleDict({})
        layers.add_node('output_softmax')
        layers.add_node('output_flatten')
        layers.nodes['output_conv']['layer'] = ConvLayer(kernel_time='down_to_one')
        layers.nodes['output_softmax']['layer'] = ActivationLayer('softmax')
        layers.nodes['output_flatten']['layer'] = FlattenLayer()
        layers.add_edge('output_conv', 'output_softmax')
        layers.add_edge('output_softmax', 'output_flatten')
        input_chans = globals.get('eeg_chans')
        input_time = globals.get('input_time_len')
        input_shape = {'time': input_time, 'chans': input_chans}
        # descendants = nx.descendants(layers, 'input')
        # descendants.add('input')
        descendants = list(set([item for sublist in nx.all_simple_paths(layers, 'input', 'output_flatten')
                                for item in sublist]))
        to_remove = []
        for node in list(layers.nodes):
            if node not in descendants:
                to_remove.append(node)
        for node in to_remove:
            layers.remove_node(node)
        self.sorted_nodes = list(nx.topological_sort(layers))
        self.predecessors = {}
        self.fixes = {}
        self.fixed_tensors = {}
        self.tensors = {}
        layers.nodes['input']['shape'] = input_shape
        for node in self.sorted_nodes[1:]:
            predecessors = list(layers.predecessors(node))
            self.predecessors[node] = predecessors
            if len(predecessors) == 0:
                continue
            self.calc_shape_multi(predecessors, node, layers)

    def calc_shape_multi(self, predecessors, node, layers):
        pred_shapes = [layers.nodes[pred]['shape']['time'] for pred in predecessors]
        min_time = int(min(pred_shapes))
        pred_chans = [layers.nodes[pred]['shape']['chans'] for pred in predecessors]
        sum_chans = int(sum(pred_chans))
        for pred in predecessors:
            self.fixes[(pred, node)] = layers.nodes[pred]['shape']['time'] - min_time
        self.pytorch_layers[str(node)] = self.generate_pytorch_layer[type(layers.nodes[node]['layer'])]\
            (layers.nodes[node]['layer'], sum_chans, min_time)
        layers.nodes[node]['shape'] = calc_shape_channels({'time': min_time, 'chans': sum_chans}, layers.nodes[node]['layer'])

    def forward(self, X):
        self.tensors['input'] = X
        for node in self.sorted_nodes[1:]:
            predecessors = self.predecessors[node]
            if len(predecessors) == 1:
                self.tensors[node] = self.pytorch_layers[str(node)](self.tensors[predecessors[0]])
            else:
                for pred in predecessors:
                    if (pred, node) in self.fixes.keys():
                        fix_amount = self.fixes[(pred, node)]
                        self.fixed_tensors[(pred, node)] = self.tensors[pred]
                        while fix_amount > 0:
                            self.fixed_tensors[(pred, node)] = nn.MaxPool2d((fix_amount+1,1), 1)(self.fixed_tensors[(pred, node)])
                            fix_amount -= fix_amount
                to_concat = []
                for pred in predecessors:
                    if (pred, node) in self.fixes.keys():
                        to_concat.append(self.fixed_tensors[(pred, node)])
                    else:
                        to_concat.append(self.tensors[pred])
                self.tensors[node] = self.pytorch_layers[str(node)](torch.cat(tuple(to_concat), dim=1))
            if node == 'output_flatten':
                return self.tensors[node]


class MyModel:
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
            return x.view(x.shape[0], -1, int(x.shape[2] / globals.get('time_factor')), x.shape[3])
        else:
            return x.view(x.shape[0], x.shape[1], int(x.shape[2] / globals.get('time_factor')), -1)


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
    return True


def calc_shape(in_shape, layer):
    input_time = in_shape['time']
    input_chans = in_shape['chans']
    if type(layer) == ConvLayer:
        input_time = (input_time - layer.kernel_time) + 1
        input_chans = (input_chans - layer.kernel_eeg_chan) + 1
    elif type(layer) == PoolingLayer:
        input_time = (input_time - layer.pool_time) / layer.stride_time + 1
        input_chans = (input_chans - layer.pool_eeg_chan) / layer.stride_eeg_chan + 1
    if input_time < 1 or input_chans < 1:
        raise ValueError(f"illegal model, input_time={input_time}, input_chans={input_chans}")
    return {'time': input_time, 'chans': input_chans}


def calc_shape_channels(in_shape, layer):
    input_time = in_shape['time']
    input_chans = in_shape['chans']
    if type(layer) == ConvLayer:
        input_time = (input_time - layer.kernel_time) + 1
        input_chans = layer.filter_num
    elif type(layer) == PoolingLayer:
        input_time = int((input_time - layer.pool_time) / layer.stride_time) + 1
    if input_time < 1:
        raise ValueError(f"illegal model, input_time={input_time}, input_chans={input_chans}")
    return {'time': input_time, 'chans': input_chans}


def check_legal_grid_model(layer_grid):
    if not nx.is_directed_acyclic_graph(layer_grid):
        return False
    input_chans = globals.get('eeg_chans')
    input_time = globals.get('input_time_len')
    input_shape = {'time': input_time, 'chans': input_chans}
    layers = layer_grid.copy()
    layers.nodes['input']['shape'] = input_shape
    descendants = nx.descendants(layers, 'input')
    descendants.add('input')
    to_remove = []
    for node in list(layers.nodes):
        if node not in descendants:
            to_remove.append(node)
    for node in to_remove:
        layers.remove_node(node)
    nodes_to_check = list(nx.topological_sort(layers))
    for node in nodes_to_check[1:]:
        predecessors = list(layers.predecessors(node))
        try:
            pred_shapes = [layers.nodes[pred]['shape']['time'] for pred in predecessors]
            min_time = int(min(pred_shapes))
            sum_chans = int(sum([layers.nodes[pred]['shape']['chans'] for pred in predecessors]))
            layers.nodes[node]['shape'] = calc_shape_channels({'time': min_time, 'chans': sum_chans},
                                                              layers.nodes[node]['layer'])
        except ValueError:
            return False
    return True


def add_random_connection(layer_grid):
    i1 = j1 = i2 = j2 = 0
    while i1 == i2 and j1 == j2:
        i1 = random.randint(0, layer_grid.graph['height'] - 1)
        i2 = random.randint(0, layer_grid.graph['height'] - 1)
        j1 = random.randint(0, layer_grid.graph['width'] - 1)
        j2 = random.randint(0, layer_grid.graph['width'] - 1)
    layer_grid.add_edge((i1, j1), (i2, j2))
    if not nx.is_directed_acyclic_graph(layer_grid):
        layer_grid.remove_edge((i1, j1), (i2, j2))
        add_random_connection(layer_grid)


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


def random_grid_model(dim):
    layer_grid = create_empty_copy(nx.to_directed(nx.grid_2d_graph(dim, dim)))
    for node in layer_grid.nodes.values():
        node['layer'] = random_layer()
    layer_grid.add_node('input')
    layer_grid.add_node('output_conv')
    layer_grid.nodes['output_conv']['layer'] = IdentityLayer()
    layer_grid.add_edge('input', (0,0))
    layer_grid.add_edge((0,dim-1), 'output_conv')
    for i in range(dim-1):
        layer_grid.add_edge((0,i), (0,i+1))
    layer_grid.graph['height'] = dim
    layer_grid.graph['width'] = dim
    if check_legal_grid_model(layer_grid):
        return layer_grid
    else:
        return random_grid_model(dim)


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


def add_layer_to_state(new_model_state, layer, index, old_model_state):
    if type(layer).__name__ in ['BatchNormLayer', 'ConvLayer', 'PoolingLayer']:
        for k, v in old_model_state.items():
            if '%s_%d' % (type(layer).__name__, index) in k and \
                    k in new_model_state.keys() and new_model_state[k].shape == v.shape:
                new_model_state[k] = v


def inherit_grid_states(dim, cut_point, child_model_state, first_model_state, second_model_state):
    for i in range(dim):
        for j in range(cut_point):
            for k, v in child_model_state.items():
                if f'({i}, {j})' in k and k in first_model_state.keys() and first_model_state[k].shape == v.shape:
                    child_model_state[k] = v
        for j in range(cut_point, dim):
            for k, v in child_model_state.items():
                if f'({i}, {j})' in k and k in second_model_state.keys() and second_model_state[k].shape == v.shape:
                    child_model_state[k] = v
    for k, v in child_model_state.items():
        if 'output' in k and k in second_model_state.keys() and second_model_state[k].shape == v.shape:
            child_model_state[k] = v


def breed_layers(mutation_rate, first_model, second_model, first_model_state=None, second_model_state=None, cut_point=None):
    second_model = copy.deepcopy(second_model)
    save_weights = False
    if random.random() < globals.get('breed_rate'):
        if cut_point is None:
            cut_point = random.randint(0, len(first_model) - 1)
        for i in range(cut_point):
            second_model[i] = first_model[i]
        save_weights = globals.get('inherit_weights_crossover') and globals.get('inherit_weights_normal')
    if random.random() < mutation_rate:
        while True:
            rand_layer = random.randint(0, globals.get('num_layers') - 1)
            second_model[rand_layer] = random_layer()
            if check_legal_model(second_model):
                break
    new_model = new_model_from_structure_pytorch(second_model, applyFix=True)
    if save_weights:
        finalized_new_model = finalize_model(new_model)
        if torch.cuda.device_count() > 1 and globals.get('parallel_gpu'):
            finalized_new_model.cuda()
            with torch.cuda.device(0):
                finalized_new_model = nn.DataParallel(finalized_new_model.cuda(), device_ids=
                    [int(s) for s in globals.get('gpu_select').split(',')])
        finalized_new_model_state = finalized_new_model.state_dict()
        if None not in [first_model_state, second_model_state]:
            for i in range(cut_point):
                add_layer_to_state(finalized_new_model_state, second_model[i], i, first_model_state)
            for i in range(cut_point+1, globals.get('num_layers')):
                add_layer_to_state(finalized_new_model_state, second_model[i-cut_point], i, second_model_state)
    else:
        finalized_new_model_state = None
    return new_model, finalized_new_model_state


def breed_grid(mutation_rate, first_model, second_model, first_model_state=None, second_model_state=None, cut_point=None):
    globals.set('total_breedings', globals.get('total_breedings') + 1)
    child_model = copy.deepcopy(first_model)
    child_model_state = None
    if random.random() < globals.get('breed_rate'):
        if cut_point is None:
            cut_point = random.randint(0, first_model.graph['width'] - 1)
        for i in range(first_model.graph['width']):
            for j in range(cut_point, first_model.graph['width']):
                child_model.nodes[(i, j)]['layer'] = second_model.nodes[(i, j)]['layer']
                remove_edges = []
                add_edges = []
                for edge in child_model.edges([(i, j)]):
                    if (i, j) == edge[0]:
                        remove_edges.append(edge)
                for edge in second_model.edges([(i, j)]):
                    if (i, j) == edge[0]:
                        add_edges.append(edge)
                for edge in remove_edges:
                    child_model.remove_edge(edge[0], edge[1])
                for edge in add_edges:
                    child_model.add_edge(edge[0], edge[1])
        if not check_legal_grid_model(child_model):
            globals.set('failed_breedings', globals.get('failed_breedings') + 1)
            return None, None
        if globals.get('inherit_weights_crossover') and first_model_state is not None and second_model is not None:
            child_model_state = ModelFromGrid(child_model).state_dict()
            inherit_grid_states(first_model.graph['width'], cut_point, child_model_state,
                                first_model_state, second_model_state)
    if random.random() < mutation_rate:
        add_random_connection(second_model)
        i = random.randint(0, first_model.graph['width'] - 1)
        j = random.randint(0, first_model.graph['width'] - 1)
        child_model.nodes[(i, j)]['layer'] = random_layer()
    if check_legal_grid_model(child_model):
        return child_model, child_model_state
    else:
        globals.set('failed_breedings', globals.get('failed_breedings') + 1)
        return None, None


def target_model(model_name):
    input_time_len = globals.get('input_time_len')
    n_classes = globals.get('n_classes')
    eeg_chans = globals.get('eeg_chans')
    models = {'deep': deep4.Deep4Net(eeg_chans, n_classes, input_time_len, final_conv_length='auto'),
              'shallow': shallow_fbcsp.ShallowFBCSPNet(eeg_chans, n_classes, input_time_len, final_conv_length='auto'),
              'eegnet': eegnet.EEGNet(eeg_chans, n_classes, input_time_length=input_time_len, final_conv_length='auto')}
    final_conv_sizes = {'deep': 2, 'shallow': 30, 'eegnet': 2}
    globals.set('final_conv_size', final_conv_sizes[model_name])
    model = models[model_name].create_network()
    return model


def finalize_model(layer_collection):
    layer_collection = copy.deepcopy(layer_collection)
    if globals.get('cropping'):
        final_conv_time = globals.get('final_conv_size')
    else:
        final_conv_time = 'down_to_one'
    conv_layer = ConvLayer(kernel_time=final_conv_time, kernel_eeg_chan=1,
                           filter_num=globals.get('n_classes'))
    layer_collection.append(conv_layer)
    activation = ActivationLayer('softmax')
    layer_collection.append(activation)
    flatten = FlattenLayer()
    layer_collection.append(flatten)
    return new_model_from_structure_pytorch(layer_collection)


