import random

from EEGNAS import global_vars
from EEGNAS.evolution.breeding import breed_layers
from EEGNAS.utilities.NAS_utils import hash_model
from EEGNAS.model_generation.simple_model_generation import random_model, random_layer, check_legal_model


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





