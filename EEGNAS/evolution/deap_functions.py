import random
from copy import deepcopy

from EEGNAS import global_vars
from EEGNAS.evolution.breeding import breed_layers, breed_layers_modules
from EEGNAS.utilities.NAS_utils import hash_model, remove_from_models_hash
from EEGNAS.model_generation.simple_model_generation import random_model, random_layer, check_legal_model, random_module


class Individual():
    def __init__(self, model=None, state=None):
        self.model = model
        self.state = state
        if model is None:
            self.model = random_model(global_vars.get('num_layers'))


def breed_layers_deap(mutation_rate, first_ind, second_ind):
    first_child_model, first_child_state, _ = breed_layers(mutation_rate, first_ind.model, second_ind.model,
                               first_model_state=first_ind.state, second_model_state=second_ind.state)
    second_child_model, second_child_state, _ = breed_layers(mutation_rate, second_ind.model, first_ind.model,
                               first_model_state=second_ind.state, second_model_state=first_ind.state)
    return Individual(first_child_model, first_child_state), Individual(second_child_model, second_child_state)


def mutate_layers_deap(individual):
    while True:
        rand_layer = random.randint(0, len(individual['model']) - 1)
        prev_layer = individual['model'][rand_layer]
        individual['model'][rand_layer] = random_layer()
        if check_legal_model(individual['model']):
            break
        else:
            individual[rand_layer] = prev_layer
    return individual,


def mutate_layers_deap_modules(individual):
    rand_layer = random.randint(0, len(individual['model']) - 1)
    individual['model'][rand_layer] = random_module()
    hash_model(individual['model'])
    return individual,


def mutate_modules_deap(individual):
    rand_layer = random.randint(0, len(individual) - 1)
    individual[rand_layer] = random_layer()
    return individual,


def initialize_deap_population(population, models_set, genome_set):
    if global_vars.get('grid'):
        model_init = random_grid_model
    else:
        model_init = random_model
    for pop in population:
        new_rand_model = model_init(global_vars.get('num_layers'))
        pop['model'] = new_rand_model
        pop['model_state'] = None
        pop['age'] = 0
        hash_model(pop['model'], models_set, genome_set)


def breed_layers_deap(first_ind, second_ind):
    first_child_model, first_child_state, _ = breed_layers(0, first_ind['model'], second_ind['model'],
                                                           first_model_state=first_ind['model_state'],
                                                           second_model_state=second_ind['model_state'])
    second_child_model, second_child_state, _ = breed_layers(0, second_ind['model'], first_ind['model'],
                                                             first_model_state=second_ind['model_state'],
                                                             second_model_state=first_ind['model_state'])
    if first_child_model is None or second_child_model is None:
        return first_ind, second_ind
    first_ind['model'], first_ind['model_state'], first_ind['age'] = first_child_model, first_child_state, 0
    second_ind['model'], second_ind['model_state'], second_ind['age'] = second_child_model, second_child_state, 0
    return first_ind, second_ind


def breed_layers_modules_deap(toolbox, first_ind, second_ind):
    first_child_model, first_child_state, _ = breed_layers_modules(first_ind['model'], second_ind['model'],
                                                           first_model_state=first_ind['model_state'],
                                                           second_model_state=second_ind['model_state'])
    second_child_model, second_child_state, _ = breed_layers_modules(second_ind['model'], first_ind['model'],
                                                             first_model_state=second_ind['model_state'],
                                                             second_model_state=first_ind['model_state'])
    if first_child_model is not None:
        first_child = toolbox.individual()
        first_child['model'], first_child['model_state'], first_child['age'] = first_child_model, first_child_state, 0
    else:
        first_child = first_ind
    if second_child_model is not None:
        second_child = toolbox.individual()
        second_child['model'], second_child['model_state'], second_child['age'] = second_child_model, second_child_state, 0
    else:
        second_child = second_ind
    return first_child, second_child


def breed_modules_deap(first_mod, second_mod):
    cut_point = random.randint(0, len(first_mod) - 1)
    second_mod_copy = deepcopy(second_mod)
    for i in range(cut_point):
        second_mod[i] = first_mod[i]
    for i in range(cut_point+1, len(first_mod)):
        first_mod[i] = second_mod_copy[i]
    return first_mod, second_mod


def hash_models_deap(offspring, new_offspring, genome_set, models_set):
    for off in [off['model'] for off in offspring]:
        if off not in [off['model'] for off in new_offspring]:
            remove_from_models_hash(off, models_set, genome_set)
    for off in [off['model'] for off in new_offspring]:
        if off not in [off['model'] for off in offspring]:
            hash_model(off, models_set, genome_set)

