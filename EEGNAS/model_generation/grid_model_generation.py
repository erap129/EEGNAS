from torch.nn import init

from EEGNAS.model_generation.abstract_layers import *
from EEGNAS.model_generation.custom_modules import *
import networkx as nx

from EEGNAS.model_generation.custom_modules import _squeeze_final_output
from EEGNAS.model_generation.simple_model_generation import random_layer


def generate_conv_layer(layer, in_chans, prev_time):
    if layer.kernel_time == 'down_to_one':
        layer.kernel_time = prev_time
        layer.filter_num = global_vars.get('n_classes')
    if global_vars.get('channel_dim') == 'channels':
        layer.kernel_eeg_chan = 1
    return nn.Conv2d(in_chans, layer.filter_num, (layer.kernel_time, layer.kernel_eeg_chan), stride=1)


def generate_pooling_layer(layer, in_chans, prev_time):
    if global_vars.get('channel_dim') == 'channels':
        layer.pool_eeg_chan = 1
    return nn.MaxPool2d(kernel_size=(int(layer.pool_time), int(layer.pool_eeg_chan)),
                                  stride=(int(layer.stride_time), 1))


def generate_batchnorm_layer(layer, in_chans, prev_time):
    return nn.BatchNorm2d(in_chans, momentum=global_vars.get('batch_norm_alpha'), affine=True, eps=1e-5)


def generate_activation_layer(layer, in_chans, prev_time):
    activations = {'elu': nn.ELU, 'softmax': nn.Softmax, 'sigmoid': nn.Sigmoid}
    return activations[layer.activation_type]()


def generate_dropout_layer(layer, in_chans, prev_time):
    return nn.Dropout(p=global_vars.get('dropout_p'))


def generate_identity_layer(layer, in_chans, prev_time):
    return IdentityModule()


def generate_averaging_layer(layer, in_chans, prev_time):
    return AveragingModule()


def generate_linear_weighted_avg(layer, in_chans, prev_time):
    return LinearWeightedAvg(global_vars.get('n_classes'))


def generate_flatten_layer(layer, in_chans, prev_time):
    return _squeeze_final_output()


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
            FlattenLayer: generate_flatten_layer,
            AveragingLayer: generate_averaging_layer,
            LinearWeightedAvg: generate_linear_weighted_avg
        }
        layers = layer_grid.copy()
        self.pytorch_layers = nn.ModuleDict({})
        if global_vars.get('grid_as_ensemble'):
            self.finalize_grid_network_weighted_avg(layers)
        else:
            self.finalize_grid_network_concatenation(layers)
        input_chans = global_vars.get('eeg_chans')
        input_time = global_vars.get('input_time_len')
        input_shape = {'time': input_time, 'chans': input_chans}
        if global_vars.get('grid_as_ensemble'):
            self.final_layer_name = 'output_softmax'
        else:
            self.final_layer_name = 'output_flatten'
        descendants = list(set([item for sublist in nx.all_simple_paths(layers, 'input', self.final_layer_name)
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
        try:
            layers.nodes['input']['shape'] = input_shape
        except Exception as e:
            print(e)
            pdb.set_trace()
        for node in self.sorted_nodes[1:]:
            predecessors = list(layers.predecessors(node))
            self.predecessors[node] = predecessors
            if len(predecessors) == 0:
                continue
            self.calc_shape_multi(predecessors, node, layers)
        if global_vars.get('grid_as_ensemble'):
            for row in range(global_vars.get('num_layers')[0]):
                init.xavier_uniform_(self.pytorch_layers[f'output_conv_{row}'].weight, gain=1)
                init.constant_(self.pytorch_layers[f'output_conv_{row}'].bias, 0)
        else:
            init.xavier_uniform_(self.pytorch_layers['output_conv'].weight, gain=1)
            init.constant_(self.pytorch_layers['output_conv'].bias, 0)

    def finalize_grid_network_weighted_avg(self, layers):
        for row in range(global_vars.get('num_layers')[0]):
            layers.add_node(f'output_flatten_{row}')
            layers.nodes[f'output_conv_{row}']['layer'] = ConvLayer(kernel_time='down_to_one')
            layers.nodes[f'output_flatten_{row}']['layer'] = FlattenLayer()
            layers.add_edge(f'output_conv_{row}', f'output_flatten_{row}')
        layers.add_node('averaging_layer')
        layers.add_node('output_softmax')
        layers.nodes['output_softmax']['layer'] = ActivationLayer('softmax')
        layers.nodes['averaging_layer']['layer'] = LinearWeightedAvg(global_vars.get('n_classes'))
        for row in range(global_vars.get('num_layers')[0]):
            layers.add_edge(f'output_flatten_{row}', 'averaging_layer')
        layers.add_edge('averaging_layer', 'output_softmax')

    def finalize_grid_network_concatenation(self, layers):
        layers.add_node('output_softmax')
        layers.add_node('output_flatten')
        layers.nodes['output_conv']['layer'] = ConvLayer(kernel_time='down_to_one')
        layers.nodes['output_softmax']['layer'] = ActivationLayer('softmax')
        layers.nodes['output_flatten']['layer'] = FlattenLayer()
        layers.add_edge('output_conv', 'output_softmax')
        layers.add_edge('output_softmax', 'output_flatten')

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
                if node != 'averaging_layer':
                    self.tensors[node] = self.pytorch_layers[str(node)](torch.cat(tuple(to_concat), dim=1))
                if global_vars.get('grid_as_ensemble'):
                    if node == 'averaging_layer':
                        self.tensors[node] = self.pytorch_layers[str(node)](*to_concat)
            if node == self.final_layer_name:
                return self.tensors[node]


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


def check_grid_shapes(layer_grid):
    input_chans = global_vars.get('eeg_chans')
    input_time = global_vars.get('input_time_len')
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


def check_legal_grid_model(layer_grid):
    if not nx.is_directed_acyclic_graph(layer_grid):
        return False
    if not check_grid_shapes(layer_grid):
        return False
    if not global_vars.get('grid_as_ensemble'):
        if len(list(nx.all_simple_paths(layer_grid, 'input', 'output_conv'))) == 0:
            return False
    return True


def random_grid_model(dim):
    layer_grid = nx.create_empty_copy(nx.to_directed(nx.grid_2d_graph(dim[0], dim[1])))
    for node in layer_grid.nodes.values():
        if global_vars.get('simple_start'):
            node['layer'] = IdentityLayer()
        else:
            node['layer'] = random_layer()
    layer_grid.add_node('input')
    layer_grid.nodes['input']['layer'] = IdentityLayer()
    if global_vars.get('grid_as_ensemble'):
        for row in range(dim[0]):
            layer_grid.add_node(f'output_conv_{row}')
            layer_grid.nodes[f'output_conv_{row}']['layer'] = IdentityLayer()
            layer_grid.add_edge((row, dim[1] - 1), f'output_conv_{row}')
    else:
        layer_grid.add_node('output_conv')
        layer_grid.nodes['output_conv']['layer'] = IdentityLayer()
        layer_grid.add_edge((0, dim[1]-1), 'output_conv')
    layer_grid.add_edge('input', (0, 0))
    for i in range(dim[1]-1):
        layer_grid.add_edge((0, i), (0, i+1))
    layer_grid.graph['height'] = dim[0]
    layer_grid.graph['width'] = dim[1]
    if global_vars.get('parallel_paths_experiment'):
        set_parallel_paths(layer_grid)
    if check_legal_grid_model(layer_grid):
        return layer_grid
    else:
        return random_grid_model(dim)


def set_parallel_paths(layer_grid):
    layers = [ConvLayer(), BatchNormLayer(), ActivationLayer(),
              ConvLayer(), BatchNormLayer(), ActivationLayer(),
              ConvLayer(), BatchNormLayer(), ActivationLayer(),DropoutLayer()]
    for i in range(layer_grid.graph['height']):
        layer_grid.add_edge('input', (i, 0))
        for j in range(layer_grid.graph['width']):
            layer_grid.nodes[(i,j)]['layer'] = layers[j]
            if j > 0:
                layer_grid.add_edge((i, j-1), (i, j))
        if not global_vars.get('grid_as_ensemble'):
            layer_grid.add_edge((i, layer_grid.graph['width'] - 1), 'output_conv')
