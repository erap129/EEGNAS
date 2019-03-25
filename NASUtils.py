import pickle

import networkx as nx
import random
import models_generation
import globals
from scipy.spatial.distance import pdist


def get_metric_strs():
    result = []
    for evaluation_metric in globals.get('evaluation_metrics'):
        if evaluation_metric == 'accuracy':
            evaluation_metric = 'acc'
        result.append(f"train_{evaluation_metric}")
        result.append(f"val_{evaluation_metric}")
        result.append(f"test_{evaluation_metric}")
    return result


def add_evaluations_to_stats(stats, evaluations, str_prefix=''):
    for metric, valuedict in evaluations.items():
        metric_str = metric
        if metric == 'accuracy':
            metric_str = 'acc'
        stats[f"{str_prefix}train_{metric_str}"] = valuedict['train']
        stats[f"{str_prefix}val_{metric_str}"] = valuedict['valid']
        stats[f"{str_prefix}test_{metric_str}"] = valuedict['test']


def add_evaluations_to_weighted_population(pop, evaluations, str_prefix=''):
    for metric, valuedict in evaluations.items():
        metric_str = metric
        if metric == 'accuracy':
            metric_str = 'acc'
        pop[f"{str_prefix}train_{metric_str}"] = valuedict['train']
        pop[f"{str_prefix}val_{metric_str}"] = valuedict['valid']
        pop[f"{str_prefix}test_{metric_str}"] = valuedict['test']


def sum_evaluations_to_weighted_population(pop, evaluations, str_prefix=''):
    for metric, valuedict in evaluations.items():
        metric_str = metric
        if metric == 'accuracy':
            metric_str = 'acc'
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


def calculate_population_similarity(layer_collections, evolution_file, sim_count):
    sim = 0
    to_output = 3
    for i in range(sim_count):
        idxs = random.sample(range(len(layer_collections)), 2)
        score, output = models_generation.network_similarity(layer_collections[idxs[0]],
                                                             layer_collections[idxs[1]], return_output=True)
        sim += score
        if to_output > 0:
            with open(evolution_file, "a") as text_file:
                print(output, file=text_file)
            to_output -= 1
    return sim / sim_count


def calculate_one_similarity(layer_collection, other_layer_collections):
    sim = 0
    for other_layer_collection in other_layer_collections:
        score, output = models_generation.network_similarity(layer_collection,
                                                             other_layer_collection)
        sim += score
    return sim / len(other_layer_collections)


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
        model_init = models_generation.random_grid_model
    else:
        model_init = models_generation.random_model
    weighted_population = []
    if globals.get('weighted_population_from_file'):
        folder = f"models/{globals.get('models_dir')}"
        if globals.get('cross_subject'):
            return pickle.load(f'{folder}/weighted_population_'
                               f'{"_".join([i for i in range(1, globals.get("num_subjects") + 1)])}.p')
        else:
            return pickle.load(f'{folder}/weighted_population_{subject_id}.p')
    for i in range(globals.get('pop_size')):  # generate pop_size random models
        new_rand_model = model_init(globals.get('num_layers'))
        hash_model(new_rand_model, models_set, genome_set)
        weighted_population.append({'model': new_rand_model, 'model_state': None, 'age': 0})


def cross_subject_shared_fitness(item, weighted_population):
    fitness = item[f'val_{globals.get("ga_objective")}']
    subject_array = [i in range (1, globals.get('num_subjects')+1)]
    fitness_vector = [item[f'{i}_val_{globals.get("ga_objective")}'] for i in subject_array]
    denominator = 1
    max_pdist = pdist([[0 for i in subject_array], [100 for i in subject_array]])
    for pop in weighted_population:
        pop_fitness = [pop[f'{i}_val_{globals.get("ga_objective")}'] for i in range(1, globals.get('num_subjects')+1)]
        dist = pdist([fitness_vector, pop_fitness])[0] / max_pdist
        if dist < globals.get('min_dist'):
            denominator += 1-dist
    return fitness / denominator


def normal_fitness(item):
    return item[f'val_{globals.get("ga_objective")}']

