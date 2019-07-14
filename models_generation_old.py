import math
import pdb
import sys
import torch
import numpy as np
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.util import np_to_var
from braindecode.models import deep4, shallow_fbcsp, eegnet
import os
from torch import nn
from torch.nn import init
import random
import copy
import globals
from collections import defaultdict
import networkx as nx
from model_generation.abstract_layers import *
from model_generation.grid_model_generation import ModelFromGrid, check_legal_grid_model
from model_generation.simple_model_generation import check_legal_model, random_layer
from model_impls.oh_parkinson import OhParkinson
from model_impls.sleepClassifier import get_sleep_classifier
from networkx.classes.function import create_empty_copy
from model_generation.simple_model_generation import new_model_from_structure_pytorch


def print_structure(structure):
    print('---------------model structure-----------------')
    for layer in structure.values():
        attrs = vars(layer)
        print('layer type:', layer.__class__.__name__)
    print('-----------------------------------------------')


def check_legal_cropping_model(layer_collection):
    finalized_model = finalize_model(layer_collection)
    finalized_model_to_dilated(finalized_model)
    n_preds_per_input = get_n_preds_per_input(finalized_model)
    n_receptive_field = globals.get('input_time_len') - n_preds_per_input + 1
    n_preds_per_trial = globals.get('original_input_time_len') - n_receptive_field + 1
    return n_preds_per_trial <= n_preds_per_input * \
           (math.ceil(globals.get('original_input_time_len') / globals.get('input_time_len')))


def remove_random_connection(layer_grid):
    random_edge = random.choice(list(layer_grid.edges))
    layer_grid.remove_edge(*random_edge)
    if len(list(nx.all_simple_paths(layer_grid, 'input', 'output_conv'))) == 0:
        layer_grid.add_edge(*random_edge)  # don't leave the input and output unconnected
        return False
    else:
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


def add_parallel_connection(layer_grid):
    i1 = i2 = 0
    j = random.randint(0, layer_grid.graph['width'] - 2)
    while i1 == i2:
        i1 = random.randint(0, layer_grid.graph['height'] - 1)
        i2 = random.randint(0, layer_grid.graph['height'] - 1)
    layer_grid.add_edge((i1, j), (i2, j+1))


def swap_random_layer(layer_grid):
    i = random.randint(0, layer_grid.graph['height'] - 1)
    j = random.randint(0, layer_grid.graph['width'] - 1)
    layer_grid.nodes[(i, j)]['layer'] = random_layer()


def add_layer_to_state(new_model_state, layer, index, old_model_state):
    if type(layer).__name__ in ['BatchNormLayer', 'ConvLayer', 'PoolingLayer']:
        for k, v in old_model_state.items():
            if '%s_%d' % (type(layer).__name__, index) in k and \
                    k in new_model_state.keys() and new_model_state[k].shape == v.shape:
                new_model_state[k] = v


def copy_one_layer_states(str, child_model_state, other_model_state):
    copy_weights = True
    for k, v in child_model_state.items():
        if str in k and k in other_model_state.keys() and other_model_state[k].shape != v.shape:
            copy_weights = False
    for k, v in child_model_state.items():
        if str in k and k in other_model_state.keys() and other_model_state[k].shape == v.shape \
                and copy_weights:
            child_model_state[k] = other_model_state[k]


def inherit_grid_states(dim, cut_point, child_model_state, first_model_state, second_model_state):
    for i in range(dim):
        for j in range(cut_point):
            copy_one_layer_states(f'({i}, {j})', child_model_state, first_model_state)
        for j in range(cut_point, dim):
            copy_one_layer_states(f'({i}, {j})', child_model_state, second_model_state)
    copy_one_layer_states('output', child_model_state, second_model_state)


def mutate_layer(model, layer_index):
    old_layer = model[layer_index]
    model[layer_index] = random_layer()
    if not check_legal_model(model):
        model[layer_index] = old_layer


def mutate_models(model, mutation_rate):
    if random.random() < mutation_rate:
        while True:
            rand_layer = random.randint(0, len(model) - 1)
            model[rand_layer] = random_layer()
            if check_legal_model(model):
                break


def mutate_layers(model, mutation_rate):
    for layer_index in range(len(model)):
        if random.random() < mutation_rate:
            mutate_layer(model, layer_index)


def breed_layers(mutation_rate, first_model, second_model, first_model_state=None, second_model_state=None, cut_point=None):
    second_model = copy.deepcopy(second_model)
    save_weights = False
    if random.random() < globals.get('breed_rate'):
        if cut_point is None:
            cut_point = random.randint(0, len(first_model) - 1)
        for i in range(cut_point):
            second_model[i] = first_model[i]
        save_weights = globals.get('inherit_weights_crossover') and globals.get('inherit_weights_normal')
    this_module = sys.modules[__name__]
    getattr(this_module, globals.get('mutation_method'))(second_model, mutation_rate)
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
    if check_legal_model(new_model):
        return new_model, finalized_new_model_state, cut_point
    else:
        globals.set('failed_breedings', globals.get('failed_breedings') + 1)
        return None, None, None


def breed_grid(mutation_rate, first_model, second_model, first_model_state=None, second_model_state=None, cut_point=None):
    globals.set('total_breedings', globals.get('total_breedings') + 1)
    child_model = copy.deepcopy(first_model)
    child_model_state = None

    if random.random() < globals.get('breed_rate'):
        if cut_point is None:
            cut_point = random.randint(0, first_model.graph['width'] - 1)
        for i in range(first_model.graph['height']):
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
            return None, None, None
        if globals.get('inherit_weights_crossover') and first_model_state is not None and second_model_state is not None:
            child_model_state = ModelFromGrid(child_model).state_dict()
            inherit_grid_states(first_model.graph['width'], cut_point, child_model_state,
                                first_model_state, second_model_state)

    if random.random() < mutation_rate:
        mutations = {'add_random_connection': add_random_connection,
                     'remove_random_connection': remove_random_connection,
                     'add_parallel_connection': add_parallel_connection,
                     'swap_random_layer': swap_random_layer}
        available_mutations = list(set(mutations.keys()).intersection(set(globals.get('mutations'))))
        chosen_mutation = mutations[random.choice(available_mutations)]
        chosen_mutation(child_model)
    if check_legal_grid_model(child_model):
        return child_model, child_model_state, cut_point
    else:
        globals.set('failed_breedings', globals.get('failed_breedings') + 1)
        return None, None, None


def breed_two_ensembles(breeding_method, mutation_rate, first_ensemble, second_ensemble, first_ensemble_states=None,
                        second_ensemble_states=None, cut_point=None):
    if cut_point is None:
        if globals.get('grid'):
            cut_point = random.randint(0, globals.get('num_layers')[1] - 1)
        else:
            cut_point = random.randint(0, globals.get('num_layers') - 1)
    models = []
    states = []
    for m1, m2, s1, s2 in zip(first_ensemble, second_ensemble, first_ensemble_states, second_ensemble_states):
        assert(m1['perm_ensemble_role'] == m2['perm_ensemble_role'])
        combined, combined_state, _ = breeding_method(mutation_rate, m1['model'], m2['model'], s1, s2, cut_point)
        models.append(combined)
        states.append(combined_state)
    return models, states, cut_point


def target_model(model_name):
    input_time_len = globals.get('input_time_len')
    n_classes = globals.get('n_classes')
    eeg_chans = globals.get('eeg_chans')
    models = {'deep': deep4.Deep4Net(eeg_chans, n_classes, input_time_len, final_conv_length='auto'),
              'shallow': shallow_fbcsp.ShallowFBCSPNet(eeg_chans, n_classes, input_time_len, final_conv_length='auto'),
              'eegnet': eegnet.EEGNet(eeg_chans, n_classes, input_time_length=input_time_len, final_conv_length='auto'),
              'sleep_classifier': get_sleep_classifier(),
              'oh_parkinson': OhParkinson}
    final_conv_sizes = {'deep': 2, 'shallow': 30, 'eegnet': 2}
    final_conv_sizes = defaultdict(int, final_conv_sizes)
    globals.set('final_conv_size', final_conv_sizes[model_name])
    if model_name == 'sleep_classifier':
        return models[model_name]
    else:
        model = models[model_name].create_network()
        return model


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
    activation = ActivationLayer('softmax')
    layer_collection.append(activation)
    flatten = FlattenLayer()
    layer_collection.append(flatten)
    return new_model_from_structure_pytorch(layer_collection)


def finalized_model_to_dilated(model):
    to_dense_prediction_model(model)
    conv_classifier = model.conv_classifier
    model.conv_classifier = nn.Conv2d(conv_classifier.in_channels, conv_classifier.out_channels,
                                      (globals.get('final_conv_size'),
                                       conv_classifier.kernel_size[1]), stride=conv_classifier.stride,
                                      dilation=conv_classifier.dilation)


def get_n_preds_per_input(model):
    dummy_input = get_dummy_input()
    if globals.get('cuda'):
        model.cuda()
        dummy_input = dummy_input.cuda()
    out = model(dummy_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    return n_preds_per_input


def get_dummy_input():
    input_shape = (2, globals.get('eeg_chans'), globals.get('input_time_len'), 1)
    return np_to_var(np.random.random(input_shape).astype(np.float32))
