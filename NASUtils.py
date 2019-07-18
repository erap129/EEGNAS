import pickle
import torch
from collections import defaultdict
from functools import reduce

import networkx as nx
import random
from braindecode.torch_ext.util import np_to_var
import globals
import numpy as np
from copy import deepcopy
from scipy.stats import spearmanr
from model_generation.grid_model_generation import random_grid_model
from model_generation.simple_model_generation import random_model


def get_metric_strs():
    result = []
    for evaluation_metric in globals.get('evaluation_metrics'):
        if evaluation_metric in ['raw', 'target']:
            continue
        result.append(f"train_{evaluation_metric}")
        result.append(f"val_{evaluation_metric}")
        result.append(f"test_{evaluation_metric}")
    return result


def add_evaluations_to_stats(stats, evaluations, str_prefix=''):
    for metric, valuedict in evaluations.items():
        metric_str = metric
        if metric in ['raw', 'target']:
            continue
        stats[f"{str_prefix}train_{metric_str}"] = valuedict['train']
        stats[f"{str_prefix}val_{metric_str}"] = valuedict['valid']
        stats[f"{str_prefix}test_{metric_str}"] = valuedict['test']


def add_evaluations_to_weighted_population(pop, evaluations, str_prefix=''):
    for metric, valuedict in evaluations.items():
        metric_str = metric
        pop[f"{str_prefix}train_{metric_str}"] = valuedict['train']
        pop[f"{str_prefix}val_{metric_str}"] = valuedict['valid']
        pop[f"{str_prefix}test_{metric_str}"] = valuedict['test']


def add_raw_to_weighted_population(pop, raw):
    for key, value in raw.items():
        pop[key] = value


def sum_evaluations_to_weighted_population(pop, evaluations, str_prefix=''):
    for metric, valuedict in evaluations.items():
        metric_str = metric
        if f"{str_prefix}train_{metric_str}" in pop:
            pop[f"{str_prefix}train_{metric_str}"] += valuedict['train']
            pop[f"{str_prefix}val_{metric_str}"] += valuedict['valid']
            pop[f"{str_prefix}test_{metric_str}"] += valuedict['test']


def get_model_state(model):
    if globals.get('cross_subject'):
        available_states = [x for x in model.keys() if 'model_state' in x]
        model_state_str = random.sample(available_states, 1)[0]
    else:
        model_state_str = 'model_state'
    return model[model_state_str]


def check_age(model):
    return globals.get('use_aging') and\
        random.random() < 1 - 1 / (model['age'] + 1)


def get_average_param(models, layer_type, attribute):
    attr_count = 0
    count = 0
    for model in models:
        layers = get_model_layers(model)
        for layer in layers:
            if isinstance(layer, layer_type):
                attr_count += getattr(layer, attribute)
                count += 1
    if count == 0:
        return 'NAN'
    return attr_count / count


def inject_dropout(weighted_population):
    for pop in weighted_population:
        layer_collection = pop['model']
        for i in range(len(layer_collection)):
            if random.uniform(0, 1) < globals.get('dropout_injection_rate'):
                old_layer = layer_collection[i]
                layer_collection[i] = models_generation.DropoutLayer()
                if not models_generation.check_legal_model(layer_collection):
                    layer_collection[i] = old_layer
        pop['model_state'] = None


def remove_from_models_hash(model, model_set, genome_set):
    if globals.get('grid'):
        for layer in model.nodes.values():
            remove_layer = True
            for other_model in model_set:
                if not equal_grid_models(model, other_model):
                    for other_layer in other_model.nodes.values():
                        if layer['layer'] == other_layer['layer']:
                            remove_layer = False
                            break
                if not remove_layer:
                    break
            if remove_layer and layer['layer'] in genome_set:
                genome_set.remove(layer['layer'])
        if model in model_set:
            model_set.remove(model)
    else:
        for layer in model:
            remove_layer = True
            for other_model in model_set:
                if model != other_model:
                    for other_layer in other_model:
                        if layer == other_layer:
                            remove_layer = False
                            break
                if not remove_layer:
                    break
            if remove_layer and layer in genome_set:
                genome_set.remove(layer)
        if model in model_set:
            model_set.remove(model)


def get_model_layers(model):
    if globals.get('grid'):
        return [layer['layer'] for layer in model.nodes.values()]
    else:
        return model


def hash_model(model, model_set, genome_set):
    if globals.get('grid'):
        add_model = True
        for other_model in model_set:
            if equal_grid_models(model, other_model):
                add_model = False
        if add_model:
            model_set.append(model)
    else:
        if model not in model_set:
            model_set.append(model)
    layers = get_model_layers(model)
    for layer in layers:
        if layer not in genome_set:
            genome_set.append(layer)


def count_layer_type_in_pop(models, layer_type):
    count = 0
    for model in models:
        layers = get_model_layers(model)
        for layer in layers:
            if isinstance(layer, layer_type):
                count += 1
    return count


def num_of_models_with_skip_connection(weighted_population):
    total = 0
    for pop in weighted_population:
        if len(list(nx.all_simple_paths(pop['model'], 'input', 'output_conv'))) > 1:
            total += 1
    return total


def equal_grid_models(layer_grid_1, layer_grid_2):
    for i in range(layer_grid_1.graph['height']):
        for j in range(layer_grid_2.graph['width']):
            if layer_grid_1.nodes[(i,j)]['layer'] != layer_grid_2.nodes[(i,j)]['layer']:
                return False
    for edge in layer_grid_1.edges:
        if edge not in layer_grid_2.edges:
            return False
    for edge in layer_grid_2.edges:
        if edge not in layer_grid_1.edges:
            return False
    return True


def initialize_population(models_set, genome_set, subject_id):
    if globals.get('grid'):
        model_init = random_grid_model
    else:
        model_init = random_model
    if globals.get('weighted_population_from_file'):
        folder = f"models/{globals.get('models_dir')}"
        if globals.get('cross_subject'):
            weighted_population = pickle.load(f'{folder}/weighted_population_'
                               f'{"_".join([i for i in range(1, globals.get("num_subjects") + 1)])}.p')
        else:
            weighted_population = pickle.load(f'{folder}/weighted_population_{subject_id}.p')
    else:
        weighted_population = []
        for i in range(globals.get('pop_size')):
            new_rand_model = model_init(globals.get('num_layers'))
            weighted_population.append({'model': new_rand_model, 'model_state': None, 'age': 0})
    for i in range(globals.get('pop_size')):
        hash_model(weighted_population[i]['model'], models_set, genome_set)
    return weighted_population


def ranking_correlations(weighted_population, stats):
    old_ensemble_iterations = globals.get('ensemble_iterations')
    fitness_funcs = {'ensemble_fitness': ensemble_fitness, 'normal_fitness': normal_fitness}
    for num_iterations in globals.get('ranking_correlation_num_iterations'):
        rankings = []
        globals.set('ensemble_iterations', num_iterations)
        for fitness_func in globals.get('ranking_correlation_fitness_funcs'):
            weighted_pop_copy = deepcopy(weighted_population)
            for i, pop in enumerate(weighted_pop_copy):
                pop['order'] = i
            fitness_funcs[fitness_func](weighted_pop_copy)
            weighted_pop_copy = sorted(weighted_pop_copy, key=lambda x: x['fitness'], reverse=True)
            ranking = [pop['order'] for pop in weighted_pop_copy]
            rankings.append(ranking)
        correlation = spearmanr(*rankings)
        stats[f'ranking_correlation_{num_iterations}'] = correlation[0]
    globals.set('ensemble_iterations', old_ensemble_iterations)


def sort_population(weighted_population, reverse):
    new_weighted_pop = []
    if globals.get('perm_ensembles'):
        ensemble_order = weighted_population[globals.get('pop_size')]
        del weighted_population[globals.get('pop_size')]
        for order in ensemble_order:
            pops = [weighted_population[i] for i in range(globals.get('pop_size'))
                    if weighted_population[i]['group_id'] == order['group_id']]
            new_weighted_pop.extend(pops)
        return new_weighted_pop
    else:
        return sorted(weighted_population, key=lambda x: x['fitness'], reverse=reverse)


def add_model_to_stats(pop, model_index, model_stats):
    if globals.get('grid'):
        if globals.get('grid_as_ensemble'):
            for key, value in pop['weighted_avg_params'].items():
                model_stats[key] = value
    else:
        for i, layer in enumerate(pop['model']):
            model_stats[f'layer_{i}'] = type(layer).__name__
            for key, value in vars(layer).items():
                model_stats[f'layer_{i}_{key}'] = value
    if globals.get('perm_ensembles'):
        model_stats['ensemble_role'] = (model_index % globals.get('ensemble_size'))
        assert pop['perm_ensemble_role'] == model_stats['ensemble_role']
        model_stats['perm_ensemble_id'] = pop['perm_ensemble_id']
    if globals.get('delete_finalized_models'):
        finalized_model = models_generation.finalize_model(pop['model'])
    else:
        finalized_model = pop['finalized_model']
    model_stats['trainable_params'] = pytorch_count_params(finalized_model)


def train_time_penalty(weighted_population):
    train_time_indices = [i[0] for i in sorted(enumerate
                                               (weighted_population), key=lambda x: x[1]['train_time'])]
    for rank, idx in enumerate(train_time_indices):
        weighted_population[idx]['fitness'] -= (rank / globals.get('pop_size')) *\
                                                weighted_population[idx]['fitness'] * globals.get('penalty_factor')


def pytorch_count_params(model):
    total_params = sum(reduce(lambda a, b: a*b, x.size()) for x in model.parameters())
    return total_params


def format_manual_ensemble_evaluations(avg_evaluations):
    for eval in avg_evaluations.items():
        for eval_spec in eval[1].items():
            if type(eval_spec[1] == list):
                avg_evaluations[eval[0]][eval_spec[0]] = np.mean(eval_spec[1], axis=0)
            else:
                avg_evaluations[eval[0]][eval_spec[0]] = np.mean(eval_spec[1])
    new_avg_evaluations = defaultdict(dict)
    for dataset in ['train', 'valid', 'test']:
        ensemble_preds = avg_evaluations['raw'][dataset]
        pred_labels = np.argmax(ensemble_preds, axis=1).squeeze()
        ensemble_targets = avg_evaluations['target'][dataset]
        ensemble_fit = getattr(utils, f'{globals.get("ga_objective")}_func')(pred_labels, ensemble_targets)
        objective_str = globals.get("ga_objective")
        new_avg_evaluations[f'ensemble_{objective_str}'][dataset] = ensemble_fit
    return new_avg_evaluations


def set_finetuning(model, X):
    child_idx = 0
    num_layers = len(list(model.children()))
    for child in model.children():
        if child_idx < num_layers - X:
            for param in child.parameters():
                param.requires_grad = False
        child_idx += 1


def evaluate_single_model(model, X, y, eval_func):
    if X.ndim == 3:
        X = X[:, :, :, None]
    model.eval()
    with torch.no_grad():
        X = np_to_var(X, pin_memory=globals.get('pin_memory'))
        if torch.cuda.is_available():
            with torch.cuda.device(0):
                X = X.cuda()
        preds = model(X)
        preds = preds.cpu().data.numpy()
        pred_labels = np.argmax(preds, axis=1).squeeze()
        return eval_func(pred_labels, y)





