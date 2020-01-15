import pickle
from graphviz import Digraph
from torch.nn import Conv2d, MaxPool2d, ELU, Dropout, BatchNorm2d
import pandas as pd
from EEGNAS.model_generation.abstract_layers import IdentityLayer, ConvLayer, PoolingLayer, ActivationLayer
from EEGNAS.model_generation.custom_modules import IdentityModule

SHORT_NAMES = {Conv2d: 'Conv',
               MaxPool2d: 'MaxPool',
               ELU: 'ELU',
               Dropout: 'Dropout',
               BatchNorm2d: 'BatchNorm'}


def get_layer_stats(layer, delimiter):
    if type(layer) == ELU or type(layer) == BatchNorm2d or type(layer) == Dropout:
        return ''
    elif type(layer) == Conv2d:
        return f'{delimiter}F: {layer.out_channels}, K: {layer.kernel_size[0]}'
    elif type(layer) == MaxPool2d:
        return f'{delimiter}K: {layer.kernel_size[0]}, S: {layer.stride[0]}'
    else:
        return ''


def export_eegnas_table(models):
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
    df.to_csv('netflow_eegnas_part.csv')


def plot_eegnas_model(model, f, subgraph_idx):
    # f = Digraph('EEGNAS model', filename='EEGNAS_model.gv')
    nodes = []
    multiplier = 1
    module_list = list(model._modules.values())[:-1]
    module_list = [l for l in module_list if type(l) != IdentityModule]
    for layer_idx, layer in enumerate(module_list):
        if type(layer) == BatchNorm2d or type(layer) == Dropout or type(layer) == ELU:
            if layer_idx < len(module_list) - 1 and type(module_list[layer_idx + 1]) == type(layer):
                multiplier += 1
                continue
        layer_str = f'{SHORT_NAMES[type(layer)]}'
        layer_str += get_layer_stats(layer)
        # for param, value in layer.__dict__.items():
        #     if value is not None:
        #         layer_str += f'{param}: {value}\\n'
        # nodes_labels.append(layer_str)
        if multiplier > 1:
            f.node(f'{subgraph_idx}_{layer_idx}', label=layer_str, xlabel=f'X {multiplier}')
        else:
            f.node(f'{subgraph_idx}_{layer_idx}', label=layer_str)
        nodes.append(f'{subgraph_idx}_{layer_idx}')
        if type(layer) == BatchNorm2d or type(layer) == Dropout or type(layer) == ELU:
            if layer_idx < len(module_list) - 1 and type(module_list[layer_idx + 1]) != type(layer):
                multiplier = 1
    for idx in range(len(nodes) - 1):
        f.edge(nodes[idx], nodes[idx+1])
    # f.render('test_eegnas_graphviz.gv', view=False)


def create_ensemble_digraph():
    f = Digraph('EEGNAS model', filename='EEGNAS_model.gv')
    for i in range(5):
        plot_eegnas_model(weighted_population[i]['finalized_model'], f, i)
        f.render('test_eegnas_graphviz.gv', view=False)


path = 'netflow_EEGNAS_pop_files/411_1_input_height_240_normalized_architectures.p'
weighted_population = pickle.load(open(path, 'rb'))
export_eegnas_table([weighted_population[i]['finalized_model'] for i in range(5)])



