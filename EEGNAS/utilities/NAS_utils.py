import pickle
import torch
from collections import defaultdict
from functools import reduce

import networkx as nx
import random
from braindecode.torch_ext.util import np_to_var
from sklearn.utils import shuffle

from EEGNAS import global_vars
import numpy as np
from copy import deepcopy
from scipy.stats import spearmanr

from EEGNAS.model_generation.abstract_layers import ConvLayer, PoolingLayer
from EEGNAS.model_generation.grid_model_generation import random_grid_model
from EEGNAS.model_generation.simple_model_generation import random_model, finalize_model
from EEGNAS.utilities.misc import is_sublist


def get_metric_strs():
    result = []
    for evaluation_metric in global_vars.get('evaluation_metrics'):
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
    if global_vars.get('cross_subject'):
        available_states = [x for x in model.keys() if 'model_state' in x]
        model_state_str = random.sample(available_states, 1)[0]
    else:
        model_state_str = 'model_state'
    return model[model_state_str]


def check_age(model):
    return global_vars.get('use_aging') and \
           random.random() > 1 / (model['age'] + 1)


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
            if random.uniform(0, 1) < global_vars.get('dropout_injection_rate'):
                old_layer = layer_collection[i]
                layer_collection[i] = models_generation.DropoutLayer()
                if not models_generation.check_legal_model(layer_collection):
                    layer_collection[i] = old_layer
        pop['model_state'] = None


def remove_from_models_hash(model, model_set, genome_set):
    if global_vars.get('grid'):
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
    if global_vars.get('grid'):
        model_init = random_grid_model
    else:
        model_init = random_model
    if global_vars.get('weighted_population_from_file'):
        folder = f"models/{global_vars.get('models_dir')}"
        if global_vars.get('cross_subject'):
            weighted_population = pickle.load(f'{folder}/weighted_population_'
                               f'{"_".join([i for i in range(1, global_vars.get("num_subjects") + 1)])}.p')
        else:
            weighted_population = pickle.load(f'{folder}/weighted_population_{subject_id}.p')
    else:
        weighted_population = []
        for i in range(global_vars.get('pop_size')):
            new_rand_model = model_init(global_vars.get('num_layers'))
            new_individual = {'model': new_rand_model, 'model_state': None, 'age': 0}
            if global_vars.get('weight_inheritance_alpha') == 'model':
                new_individual['weight_inheritance_alpha'] = random.random()
            elif global_vars.get('weight_inheritance_alpha') == 'layer':
                new_individual['weight_inheritance_alpha'] = np.array([random.random() for i in range(global_vars.get('num_layers'))])
            else:
                new_individual['weight_inheritance_alpha'] = 1
            weighted_population.append(new_individual)
    for i in range(global_vars.get('pop_size')):
        hash_model(weighted_population[i]['model'], models_set, genome_set)
    return weighted_population


def initialize_bb_population(weighted_population):
    bb_population = []
    while len(bb_population) < global_vars.get('bb_population_size'):
        rand_pop = random.choice(weighted_population)['model']
        rand_idx = random.randint(0, len(rand_pop) - 2)
        bb = rand_pop[rand_idx:rand_idx+2]
        bb_population.append(bb)
    return bb_population


def rank_bbs_by_weighted_population(bb_population, weighted_population):
    for bb in bb_population:
        bb['count'] = 0
        for pop in weighted_population:
            if is_sublist(bb['bb'], pop['model'])[0]:
                bb['fitness'] += pop['fitness']
                bb['count'] += 1
        bb['fitness'] /= bb['count']
        # try:
        #     bb['fitness'] /= bb['count']
        # except Exception as e:
        #     print
        #     is_sublist(bb['bb'], weighted_population[0]['model'])
        #     is_sublist(bb['bb'], weighted_population[1]['model'])
        #     is_sublist(bb['bb'], weighted_population[2]['model'])
        #     is_sublist(bb['bb'], weighted_population[3]['model'])
        #     is_sublist(bb['bb'], weighted_population[4]['model'])


def maintain_bb_population(bb_population, weighted_population):
    for bb_idx, bb in enumerate(bb_population):
        found = False
        for pop in weighted_population:
            if is_sublist(bb['bb'], pop['model'])[0]:
                found = True
                break
        if not found:
            rand_pop = random.choice(weighted_population)['model']
            rand_idx = random.randint(0, len(rand_pop) - 2)
            bb = rand_pop[rand_idx:rand_idx + 2]
            bb_population[bb_idx] = {'bb': bb, 'fitness': 0}
        else:
            if random.random() < global_vars.get('puzzle_expansion_rate'):
                for pop in weighted_population:
                    sblst, sblst_idx = is_sublist(bb['bb'], pop['model'])
                    if sblst:
                        if random.random() < 0.5:
                            if sblst_idx < len(pop['model']) - len(bb['bb']):
                                bb['bb'].append(pop['model'][sblst_idx + len(bb['bb'])])
                        else:
                            if sblst_idx > 0:
                                bb['bb'].insert(0, pop['model'][sblst_idx - 1])
                        break
            if random.random() < global_vars.get('puzzle_replacement_rate'):
                rand_pop = random.choice(weighted_population)['model']
                rand_idx = random.randint(0, len(rand_pop) - 2)
                bb = rand_pop[rand_idx:rand_idx + 2]
                bb_population[bb_idx] = {'bb': bb, 'fitness': 0}
    for elem in [[str(bbi) for bbi in bb['bb']] for bb in bb_population]:
        print(elem)


def ranking_correlations(weighted_population, stats):
    old_ensemble_iterations = global_vars.get('ensemble_iterations')
    fitness_funcs = {'ensemble_fitness': ensemble_fitness, 'normal_fitness': normal_fitness}
    for num_iterations in global_vars.get('ranking_correlation_num_iterations'):
        rankings = []
        global_vars.set('ensemble_iterations', num_iterations)
        for fitness_func in global_vars.get('ranking_correlation_fitness_funcs'):
            weighted_pop_copy = deepcopy(weighted_population)
            for i, pop in enumerate(weighted_pop_copy):
                pop['order'] = i
            fitness_funcs[fitness_func](weighted_pop_copy)
            weighted_pop_copy = sorted(weighted_pop_copy, key=lambda x: x['fitness'], reverse=True)
            ranking = [pop['order'] for pop in weighted_pop_copy]
            rankings.append(ranking)
        correlation = spearmanr(*rankings)
        stats[f'ranking_correlation_{num_iterations}'] = correlation[0]
    global_vars.set('ensemble_iterations', old_ensemble_iterations)


def sort_population(weighted_population, reverse):
    new_weighted_pop = []
    if global_vars.get('perm_ensembles'):
        ensemble_order = weighted_population[global_vars.get('pop_size')]
        del weighted_population[global_vars.get('pop_size')]
        for order in ensemble_order:
            pops = [weighted_population[i] for i in range(global_vars.get('pop_size'))
                    if weighted_population[i]['group_id'] == order['group_id']]
            new_weighted_pop.extend(pops)
        result = new_weighted_pop
    else:
        result = sorted(weighted_population, key=lambda x: x['fitness'], reverse=reverse)
    if global_vars.get('random_search'):
        return shuffle(result)
    else:
        return result


def add_model_to_stats(pop, model_index, model_stats):
    if global_vars.get('grid'):
        if global_vars.get('grid_as_ensemble'):
            for key, value in pop['weighted_avg_params'].items():
                model_stats[key] = value
    else:
        for i, layer in enumerate(pop['model']):
            model_stats[f'layer_{i}'] = type(layer).__name__
            for key, value in vars(layer).items():
                model_stats[f'layer_{i}_{key}'] = value
    if global_vars.get('perm_ensembles'):
        model_stats['ensemble_role'] = (model_index % global_vars.get('ensemble_size'))
        assert pop['perm_ensemble_role'] == model_stats['ensemble_role']
        model_stats['perm_ensemble_id'] = pop['perm_ensemble_id']
    if global_vars.get('delete_finalized_models'):
        finalized_model = finalize_model(pop['model'])
    else:
        finalized_model = pop['finalized_model']
    model_stats['trainable_params'] = pytorch_count_params(finalized_model)
    layer_stats = {'average_conv_width': (ConvLayer, 'kernel_width'),
                   'average_conv_height': (ConvLayer, 'kernel_height'),
                   'average_conv_filters': (ConvLayer, 'filter_num'),
                   'average_pool_width': (PoolingLayer, 'pool_height'),
                   'average_pool_stride': (PoolingLayer, 'stride_height')}
    for stat in layer_stats.keys():
        model_stats[stat] = get_average_param([pop['model']], layer_stats[stat][0], layer_stats[stat][1])


def train_time_penalty(weighted_population):
    train_time_indices = [i[0] for i in sorted(enumerate
                                               (weighted_population), key=lambda x: x[1]['train_time'])]
    for rank, idx in enumerate(train_time_indices):
        weighted_population[idx]['fitness'] -= (rank / global_vars.get('pop_size')) * \
                                               weighted_population[idx]['fitness'] * global_vars.get('penalty_factor')


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
        ensemble_fit = getattr(utils, f'{global_vars.get("ga_objective")}_func')(pred_labels, ensemble_targets)
        objective_str = global_vars.get("ga_objective")
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
        X = np_to_var(X, pin_memory=global_vars.get('pin_memory'))
        if torch.cuda.is_available():
            with torch.cuda.device(0):
                X = X.cuda()
        preds = model(X)
        preds = preds.cpu().data.numpy()
        pred_labels = np.argmax(preds, axis=1).squeeze()
        return eval_func(pred_labels, y)


def get_model_layers(model):
    if global_vars.get('grid'):
        return [layer['layer'] for layer in model.nodes.values()]
    else:
        return model


def hash_model(model, model_set, genome_set):
    if global_vars.get('grid'):
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