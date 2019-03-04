import random
import models_generation
import globals


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
        for layer in model:
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


def hash_model(model, model_set, genome_set):
    if model not in model_set:
        model_set.append(model)
    for layer in model:
        if layer not in genome_set:
            genome_set.append(layer)


def count_layer_type_in_pop(models, layer_type):
    count = 0
    for model in models:
        for layer in model:
            if isinstance(layer, layer_type):
                count += 1
    return count