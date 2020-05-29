import pickle
from copy import deepcopy

from graphviz import Digraph
from torch.nn import Conv2d, MaxPool2d, ELU, Dropout, BatchNorm2d
import pandas as pd
from EEGNAS.model_generation.abstract_layers import IdentityLayer, ConvLayer, PoolingLayer, ActivationLayer
from EEGNAS.model_generation.custom_modules import IdentityModule

SHORT_NAMES = {Conv2d: 'C',
               MaxPool2d: 'M',
               ELU: 'E',
               Dropout: 'D',
               BatchNorm2d: 'B'}


def get_layer_stats(layer, delimiter):
    if type(layer) == ELU or type(layer) == BatchNorm2d or type(layer) == Dropout:
        return ''
    elif type(layer) == Conv2d:
        return f'{delimiter}f:{layer.out_channels},k:{layer.kernel_size[0]}'
    elif type(layer) == MaxPool2d:
        return f'{delimiter}k:{layer.kernel_size[0]},s:{layer.stride[0]}'
    else:
        return ''


def export_eegnas_table(models, filename):
    model_series = []
    for model_idx, model in enumerate(models):
        layer_list = []
        module_list = list(model._modules.values())[:-1]
        module_list = [l for l in module_list if type(l) != IdentityModule]
        for layer_idx, layer in enumerate(module_list):
            layer_str = f'{SHORT_NAMES[type(layer)]}'
            layer_str += get_layer_stats(layer, '   ')
            layer_list.append(layer_str)
        layer_series = pd.Series(layer_list)
        layer_series.name = f'Model {model_idx}'
        model_series.append(pd.Series(layer_list))
    df = pd.DataFrame(model_series).transpose()
    df.columns = [f'Model {i+1}' for i in range(len(models))]
    df.to_csv(filename)


def plot_eegnas_model(model, f, subgraph_idx, nodes):
    nodes = deepcopy(nodes)
    multiplier = 1
    module_list = list(model._modules.values())[:-1]
    module_list = [l for l in module_list if type(l) != IdentityModule]
    for layer_idx, layer in enumerate(module_list):
        if type(layer) == BatchNorm2d or type(layer) == Dropout or type(layer) == ELU:
            if layer_idx < len(module_list) - 1 and type(module_list[layer_idx + 1]) == type(layer):
                multiplier += 1
                continue
        layer_str = f'{SHORT_NAMES[type(layer)]}'
        layer_str += get_layer_stats(layer, ',')
        layer_str = f'<<B>{layer_str}</B>>'
        if multiplier > 1:
            f.node(f'{subgraph_idx}_{layer_idx}', label=layer_str, xlabel=f'<<B>X {multiplier}</B>>')
        else:
            f.node(f'{subgraph_idx}_{layer_idx}', label=layer_str)
        nodes.append(f'{subgraph_idx}_{layer_idx}')
        if type(layer) == BatchNorm2d or type(layer) == Dropout or type(layer) == ELU:
            if layer_idx < len(module_list) - 1 and type(module_list[layer_idx + 1]) != type(layer):
                multiplier = 1
    nodes.append('output')
    for idx in range(len(nodes) - 1):
        f.edge(nodes[idx], nodes[idx+1])


def create_ensemble_digraph(weighted_population, n_members):
    f = Digraph('EEGNAS model', filename='EEGNAS_model.gv', graph_attr={'dpi':'300'}, format='png')
    f.attr('node', shape='box')
    f.node(f'input', label='<<B>Input: (Bsize, 240, 22)</B>>')
    f.node(f'output', label='<<B>Output: (Bsize, 5, 22)</B>>')
    nodes = ['input']
    for i in range(n_members):
        plot_eegnas_model(weighted_population[i]['finalized_model'], f, i, nodes)
    f.render('test_eegnas_graphviz', view=False)



sum_path = "/home/user/Documents/eladr/netflowinsights/CDN_overflow_prediction/eegnas_models/195_10_input_height_240_normalized_handovers_all_inheritance_fold9_architectures_iteration_1.p"
per_path = '/home/user/Documents/eladr/netflowinsights/CDN_overflow_prediction/eegnas_models/197_10_input_height_240_normalized_per_handover_handovers_all_inheritance_fold9_architectures_iteration_1.p'
weighted_population_per = pickle.load(open(per_path, 'rb'))
weighted_population_sum = pickle.load(open(sum_path, 'rb'))
# export_eegnas_table([weighted_population_per[i]['finalized_model'] for i in range(5)], 'per_architectures.csv')
# export_eegnas_table([weighted_population_sum[i]['finalized_model'] for i in range(5)], 'sum_architectures.csv')
create_ensemble_digraph(weighted_population_per, 5)


