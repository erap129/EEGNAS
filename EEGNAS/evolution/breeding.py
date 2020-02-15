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
from EEGNAS.utilities.misc import is_sublist


def breed_population(weighted_population, eegnas, bb_population=None):
    if global_vars.get('grid'):
        breeding_method = models_generation.breed_grid
    else:
        breeding_method = breed_layers
    if global_vars.get('perm_ensembles'):
        breed_perm_ensembles(weighted_population, breeding_method, eegnas)
    else:
        breed_normal_population(weighted_population, breeding_method, eegnas, bb_population)


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


def breed_normal_population(weighted_population, breeding_method, eegnas, bb_population):
    children = []
    while len(weighted_population) + len(children) < global_vars.get('pop_size'):
        breeders = random.sample(range(len(weighted_population)), 2)
        first_breeder = weighted_population[breeders[0]]
        second_breeder = weighted_population[breeders[1]]
        first_model_state = NAS_utils.get_model_state(first_breeder)
        second_model_state = NAS_utils.get_model_state(second_breeder)
        new_model, new_model_state, cut_point = breeding_method(mutation_rate=eegnas.mutation_rate,
                                                                first_individual=first_breeder,
                                                                second_individual=second_breeder,
                                                                first_model_state=first_model_state,
                                                                second_model_state=second_model_state,
                                                                bb_population=bb_population)
        if new_model is not None:
            children.append({'model': new_model, 'model_state': new_model_state, 'age': 0,
                             'parents': [first_breeder, second_breeder], 'cut_point': cut_point,
                             'first_parent_index': breeders[0], 'second_parent_index': breeders[1],
                             'weight_inheritance_alpha': first_breeder['weight_inheritance_alpha'] * 0.5 +
                             second_breeder['weight_inheritance_alpha'] * 0.5})
            EEGNAS.utilities.NAS_utils.hash_model(new_model, eegnas.models_set, eegnas.genome_set)
    weighted_population.extend(children)


def choose_cutpoint_by_bb(bb_population, first_model, second_model):
    loci = np.mean(np.array([calc_recombination_loci(bb_population, first_model),
                            calc_recombination_loci(bb_population, second_model)]),
                    axis=0)
    min_val = loci.min()
    indices = np.where(loci == min_val)
    return random.choice(indices)[0]


def calc_recombination_loci(bb_population, model):
    recombination_loci = [0 for i in range(len(model))]
    for bb in bb_population:
        is_sblst, sblst_idx = is_sublist(bb['bb'], model)
        if is_sblst:
            for idx in range(sblst_idx, sblst_idx+len(bb['bb'])):
                recombination_loci[idx] = max(recombination_loci[idx], bb['fitness'])
    return recombination_loci


def breed_layers(mutation_rate, first_individual, second_individual, first_model_state=None, second_model_state=None, cut_point=None, bb_population=None):
    first_model = first_individual['model']
    second_model = second_individual['model']
    if bb_population is not None:
        if random.random() < global_vars.get('puzzle_usage_rate'):
            cut_point = choose_cutpoint_by_bb(bb_population, first_model, second_model)
    second_model = copy.deepcopy(second_model)
    if cut_point is None:
        cut_point = random.randint(0, len(first_model) - 1)
        if global_vars.get('cut_point_modulo'):
            while (cut_point+1) % global_vars.get('cut_point_modulo') != 0:
                cut_point = random.randint(0, len(first_model) - 1)
    for i in range(cut_point):
        second_model[i] = first_model[i]
    save_weights = global_vars.get('inherit_weights_crossover') and global_vars.get('inherit_weights_normal')
    this_module = sys.modules[__name__]
    if type(global_vars.get('mutation_method')) == list:
        mutation_method = random.choice(global_vars.get('mutation_method'))
    else:
        mutation_method = global_vars.get('mutation_method')
    getattr(this_module, mutation_method)(second_model, mutation_rate)
    new_model = new_model_from_structure_pytorch(second_model, applyFix=True)
    if save_weights:
        finalized_new_model = finalize_model(new_model)
        finalized_new_model_state = finalized_new_model.state_dict()
        if None not in [first_model_state, second_model_state]:
            for i in range(cut_point):
                add_layer_to_state(finalized_new_model_state, second_model[i], i, first_model_state, first_individual['weight_inheritance_alpha'])
            for i in range(cut_point+1, global_vars.get('num_layers')):
                add_layer_to_state(finalized_new_model_state, second_model[i-cut_point], i, second_model_state, second_individual['weight_inheritance_alpha'])
    else:
        finalized_new_model_state = None
    if check_legal_model(new_model):
        return new_model, finalized_new_model_state, cut_point
    else:
        global_vars.set('failed_breedings', global_vars.get('failed_breedings') + 1)
        return None, None, None


def breed_layers_modules(first_model, second_model, first_model_state=None, second_model_state=None, cut_point=None):
    second_model = copy.deepcopy(second_model)
    if cut_point is None:
        cut_point = random.randint(0, len(first_model) - 1)
    for i in range(cut_point):
        second_model[i] = first_model[i]
    save_weights = global_vars.get('inherit_weights_crossover') and global_vars.get('inherit_weights_normal')
    if check_legal_model(second_model):
        if save_weights:
            finalized_new_model = finalize_model(second_model)
            finalized_new_model_state = finalized_new_model.state_dict()
            if None not in [first_model_state, second_model_state]:
                for i in range(cut_point):
                    add_layer_to_state(finalized_new_model_state, second_model[i], i, first_model_state)
                for i in range(cut_point+1, global_vars.get('num_layers')):
                    add_layer_to_state(finalized_new_model_state, second_model[i-cut_point], i, second_model_state)
        else:
            finalized_new_model_state = None
        return second_model, finalized_new_model_state, cut_point
    else:
        global_vars.set('failed_breedings', global_vars.get('failed_breedings') + 1)
        return None, None, None


def mutate_weight_inheritance_alpha(model, mutation_rate):
    if random.random() < mutation_rate:
        model['weight_inheritance_alpha'] += random.uniform(-1, 1) * 0.1
        model['weight_inheritance_alpha'] = min(model['weight_inheritance_alpha'], 1)
        model['weight_inheritance_alpha'] = max(model['weight_inheritance_alpha'], 0)


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
