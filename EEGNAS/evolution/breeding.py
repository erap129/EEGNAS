import random

import EEGNAS.utilities.NAS_utils
import EEGNAS.utilities.NN_utils
from EEGNAS import global_vars
from EEGNAS.utilities import NAS_utils
import torch
from torch import nn
import sys
import numpy as np
import copy

from EEGNAS.model_generation.simple_model_generation import new_model_from_structure_pytorch, add_layer_to_state, \
    check_legal_model, random_layer, finalize_model


def breed_population(weighted_population, eegnas):
    if global_vars.get('grid'):
        breeding_method = models_generation.breed_grid
    else:
        breeding_method = breed_layers
    if global_vars.get('perm_ensembles'):
        breed_perm_ensembles(weighted_population, breeding_method, eegnas)
    else:
        breed_normal_population(weighted_population, breeding_method, eegnas)


def breed_perm_ensembles(weighted_population, breeding_method, eegnas):
    children = []
    ensembles = list(NAS_utils.chunks(list(range(len(weighted_population))), global_vars.get('ensemble_size')))
    while len(weighted_population) + len(children) < global_vars.get('pop_size'):
        breeders = random.sample(ensembles, 2)
        first_ensemble = [weighted_population[i] for i in breeders[0]]
        second_ensemble = [weighted_population[i] for i in breeders[1]]
        for ensemble in [first_ensemble, second_ensemble]:
            assert (len(np.unique([pop['perm_ensemble_id'] for pop in ensemble])) == 1)
        first_ensemble_states = [NAS_utils.get_model_state(pop) for pop in first_ensemble]
        second_ensemble_states = [NAS_utils.get_model_state(pop) for pop in second_ensemble]
        new_ensemble, new_ensemble_states, cut_point = breed_two_ensembles(breeding_method,
                                                                           mutation_rate=self.mutation_rate,
                                                                           first_ensemble=first_ensemble,
                                                                           second_ensemble=second_ensemble,
                                                                           first_ensemble_states=first_ensemble_states,
                                                                           second_ensemble_states=second_ensemble_states)
        if None not in new_ensemble:
            for new_model, new_model_state in zip(new_ensemble, new_ensemble_states):
                children.append({'model': new_model, 'model_state': new_model_state, 'age': 0,
                                 'first_parent_index': first_ensemble[0]['perm_ensemble_id'],
                                 'second_parent_index': second_ensemble[0]['perm_ensemble_id'],
                                 'parents': [first_ensemble[0], second_ensemble[0]],
                                 'cut_point': cut_point})
                EEGNAS.utilities.NAS_utils.hash_model(new_model, eegnas.models_set, eegnas.genome_set)
    weighted_population.extend(children)


def breed_normal_population(weighted_population, breeding_method, eegnas):
    children = []
    while len(weighted_population) + len(children) < global_vars.get('pop_size'):
        breeders = random.sample(range(len(weighted_population)), 2)
        first_breeder = weighted_population[breeders[0]]
        second_breeder = weighted_population[breeders[1]]
        first_model_state = NAS_utils.get_model_state(first_breeder)
        second_model_state = NAS_utils.get_model_state(second_breeder)
        new_model, new_model_state, cut_point = breeding_method(mutation_rate=eegnas.mutation_rate,
                                                                first_model=first_breeder['model'],
                                                                second_model=second_breeder['model'],
                                                                first_model_state=first_model_state,
                                                                second_model_state=second_model_state)
        if new_model is not None:
            children.append({'model': new_model, 'model_state': new_model_state, 'age': 0,
                             'parents': [first_breeder, second_breeder], 'cut_point': cut_point,
                             'first_parent_index': breeders[0], 'second_parent_index': breeders[1]})
            EEGNAS.utilities.NAS_utils.hash_model(new_model, eegnas.models_set, eegnas.genome_set)
    weighted_population.extend(children)


def breed_layers(mutation_rate, first_model, second_model, first_model_state=None, second_model_state=None, cut_point=None):
    second_model = copy.deepcopy(second_model)
    save_weights = False
    if random.random() < global_vars.get('breed_rate'):
        if cut_point is None:
            cut_point = random.randint(0, len(first_model) - 1)
        for i in range(cut_point):
            second_model[i] = first_model[i]
        save_weights = global_vars.get('inherit_weights_crossover') and global_vars.get('inherit_weights_normal')
    this_module = sys.modules[__name__]
    getattr(this_module, global_vars.get('mutation_method'))(second_model, mutation_rate)
    new_model = new_model_from_structure_pytorch(second_model, applyFix=True)
    if save_weights:
        finalized_new_model = finalize_model(new_model)
        if torch.cuda.device_count() > 1 and global_vars.get('parallel_gpu'):
            finalized_new_model.cuda()
            with torch.cuda.device(0):
                finalized_new_model = nn.DataParallel(finalized_new_model.cuda(), device_ids=
                    [int(s) for s in global_vars.get('gpu_select').split(',')])
        finalized_new_model_state = finalized_new_model.state_dict()
        if None not in [first_model_state, second_model_state]:
            for i in range(cut_point):
                add_layer_to_state(finalized_new_model_state, second_model[i], i, first_model_state)
            for i in range(cut_point+1, global_vars.get('num_layers')):
                add_layer_to_state(finalized_new_model_state, second_model[i-cut_point], i, second_model_state)
    else:
        finalized_new_model_state = None
    if check_legal_model(new_model):
        return new_model, finalized_new_model_state, cut_point
    else:
        global_vars.set('failed_breedings', global_vars.get('failed_breedings') + 1)
        return None, None, None


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


def mutate_layer(model, layer_index):
    old_layer = model[layer_index]
    model[layer_index] = random_layer()
    if not check_legal_model(model):
        model[layer_index] = old_layer
