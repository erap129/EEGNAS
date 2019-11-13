import pdb
import pickle
import platform
from copy import deepcopy

import deap
import networkx
from deap.algorithms import varAnd
from deap.tools import selTournament, History

from EEGNAS.data_preprocessing import get_train_val_test
from braindecode.experiments.loggers import Printer
import logging
import torch.nn.functional as F

from EEGNAS.evolution.deap_functions import Individual, initialize_deap_population, mutate_layers_deap, \
    mutate_modules_deap, \
    mutate_layers_deap_modules, breed_layers_deap, breed_modules_deap, breed_layers_modules_deap, hash_models_deap, \
    breed_layers_deap_one_child, selection_normal_deap, breed_normal_population_deap
from EEGNAS.model_generation.abstract_layers import ConvLayer, PoolingLayer, DropoutLayer, ActivationLayer, BatchNormLayer, \
    IdentityLayer
from EEGNAS.model_generation.simple_model_generation import finalize_model, random_layer, Module, \
    check_legal_model, random_model
from EEGNAS.utilities.NAS_utils import remove_from_models_hash, hash_model
from EEGNAS.utilities.misc import time_f, pretty_print_population
from collections import OrderedDict, defaultdict
import numpy as np
import time
import torch
import EEGNAS.evolution.fitness_functions
from EEGNAS.evolution.nn_training import NN_Trainer
from braindecode.experiments.stopcriteria import MaxEpochs, Or
from braindecode.datautil.splitters import concatenate_sets
from EEGNAS.evolution.evolution_misc_functions import add_parent_child_relations
from EEGNAS.evolution.breeding import breed_population
import os
import csv
from torch import nn
import EEGNAS.evolution.fitness_functions
from EEGNAS.utilities.model_summary import summary
from EEGNAS.utilities.monitors import NoIncreaseDecrease
from EEGNAS import global_vars
from EEGNAS.utilities import NAS_utils
from deap import creator, base, tools
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import random
WARNING = '\033[93m'
ENDC = '\033[0m'
log = logging.getLogger(__name__)
model_train_times = []


def show_progress(train_time, exp_name):
    global model_train_times
    total_trainings = global_vars.get('num_generations') * global_vars.get('pop_size') * len(
        global_vars.get('subjects_to_check'))
    model_train_times.append(train_time)
    avg_model_train_time = sum(model_train_times) / len(model_train_times)
    time_left = (total_trainings - len(model_train_times)) * avg_model_train_time
    print(f"Experiment: {exp_name}, time left: {time_f(time_left)}")


class EEGNAS_evolution:
    def __init__(self, iterator, exp_folder, exp_name, loss_function,
                 train_set, val_set, test_set, stop_criterion, monitors,
                 subject_id, fieldnames, strategy, evolution_file, csv_file):
        global model_train_times
        model_train_times = []
        self.iterator = iterator
        self.exp_folder = exp_folder
        self.exp_name = exp_name
        self.monitors = monitors
        self.loss_function = loss_function
        self.stop_criterion = stop_criterion
        self.subject_id = subject_id
        self.datasets = OrderedDict(
            (('train', train_set), ('valid', val_set), ('test', test_set))
        )
        self.cuda = global_vars.get('cuda')
        self.loggers = [Printer()]
        self.fieldnames = fieldnames
        self.models_set = []
        self.genome_set = []
        self.evo_strategy = {'per_subject': self.one_strategy}[strategy]
        self.csv_file = csv_file
        self.evolution_file = evolution_file
        self.current_model_index = -1
        if isinstance(self.subject_id, int):
            self.current_chosen_population_sample = [self.subject_id]
        else:
            self.current_chosen_population_sample = []
        self.mutation_rate = global_vars.get('mutation_rate')

    def evaluate_ind_deap(self, individual):
        try:
            finalized_model = finalize_model(individual['model'])
            final_time, evaluations, model, model_state, num_epochs = \
                self.activate_model_evaluation(finalized_model, individual['model_state'], subject=self.subject_id)
            NAS_utils.add_evaluations_to_weighted_population(individual, evaluations)
            individual['model_state'] = model_state
            individual['train_time'] = final_time
            individual['finalized_model'] = model
            individual['num_epochs'] = num_epochs
            individual['fitness'] = evaluations[global_vars.get('ga_objective')]['valid']
            show_progress(final_time, self.exp_name)
            return evaluations[global_vars.get('ga_objective')]['valid'],
        except Exception as e:
            print

    def evaluate_module_deap(self, module):
        fitness = 0
        module_count = 0
        for pop in self.population:
            for other_module in pop['model']:
                if type(other_module) == deap.creator.Module and other_module.module_idx == module.module_idx:
                    fitness += pop['fitness'] / len(pop['model'])
                    module_count += 1
        if module_count > 0:
            fitness /= module_count
        return fitness,

    def sample_subjects(self):
        self.current_chosen_population_sample = sorted(random.sample(
            [i for i in range(1, global_vars.get('num_subjects') + 1) if i not in global_vars.get('exclude_subjects')],
            global_vars.get('cross_subject_sampling_rate')))

    def one_strategy(self, weighted_population):
        self.current_chosen_population_sample = [self.subject_id]
        for i, pop in enumerate(weighted_population):
            start_time = time.time()
            if NAS_utils.check_age(pop):
                weighted_population[i] = weighted_population[i - 1]
                weighted_population[i]['train_time'] = 0
                weighted_population[i]['num_epochs'] = 0
                continue
            finalized_model = finalize_model(pop['model'])
            self.current_model_index = i
            final_time, evaluations, model, model_state, num_epochs = \
                self.activate_model_evaluation(finalized_model, pop['model_state'], subject=self.subject_id)
            if global_vars.get('grid_as_ensemble') and global_vars.get('delete_finalized_models'):
                pop['weighted_avg_params'] = model
            self.current_model_index = -1
            NAS_utils.add_evaluations_to_weighted_population(weighted_population[i], evaluations)
            weighted_population[i]['model_state'] = model_state
            weighted_population[i]['train_time'] = final_time
            weighted_population[i]['finalized_model'] = model
            weighted_population[i]['num_epochs'] = num_epochs
            end_time = time.time()
            show_progress(end_time - start_time, self.exp_name)
            print('trained model %d in generation %d' % (i + 1, self.current_generation))

    def calculate_stats(self, weighted_population):
        stats = {}
        params = ['train_time', 'num_epochs', 'fitness']
        params.extend(NAS_utils.get_metric_strs())
        for param in params:
            stats[param] = np.mean([sample[param] for sample in weighted_population])
        if self.subject_id == 'all':
            if global_vars.get('cross_subject_sampling_method') == 'model':
                self.current_chosen_population_sample = range(1, global_vars.get('num_subjects') + 1)
            for subject in self.current_chosen_population_sample:
                for param in params:
                    stats[f'{subject}_{param}'] = np.mean(
                        [sample[f'{subject}_{param}'] for sample in weighted_population if f'{subject}_{param}' in sample.keys()])
        for i, pop in enumerate(weighted_population):
            model_stats = {}
            for param in params:
                model_stats[param] = pop[param]
            if self.subject_id == 'all':
                if global_vars.get('cross_subject_sampling_method') == 'model':
                    self.current_chosen_population_sample = range(1, global_vars.get('num_subjects') + 1)
                for subject in self.current_chosen_population_sample:
                    for param in params:
                        if f'{subject}_{param}' in pop.keys():
                            model_stats[f'{subject}_{param}'] = pop[f'{subject}_{param}']
            if 'parents' in pop.keys():
                model_stats['first_parent_child_ratio'] = pop['fitness'] / pop['parents'][0]['fitness']
                model_stats['second_parent_child_ratio'] = pop['fitness'] / pop['parents'][1]['fitness']
                model_stats['cut_point'] = pop['cut_point']
                model_stats['first_parent_index'] = pop['first_parent_index']
                model_stats['second_parent_index'] = pop['second_parent_index']
            NAS_utils.add_model_to_stats(pop, i, model_stats)
            if i == 0 or i==len(weighted_population) - 1:
                for stat, val in model_stats.items():
                    if 'layer' not in stat:
                        global_vars.get('sacred_ex').log_scalar(f'model_{i}_{stat}', val, self.current_generation)
            self.write_to_csv(model_stats, self.current_generation+1, model=i)
        stats['unique_individuals'] = len(self.models_set)
        stats['unique_layers'] = len(self.genome_set)
        stats['average_age'] = np.mean([sample['age'] for sample in weighted_population])
        stats['mutation_rate'] = self.mutation_rate
        for layer_type in [DropoutLayer, ActivationLayer, ConvLayer, IdentityLayer, BatchNormLayer, PoolingLayer]:
            stats['%s_count' % layer_type.__name__] = \
                NAS_utils.count_layer_type_in_pop([pop['model'] for pop in weighted_population], layer_type)
            if global_vars.get('add_top_20_stats'):
                stats['top20_%s_count' % layer_type.__name__] = \
                    NAS_utils.count_layer_type_in_pop([pop['model'] for pop in
                                                      weighted_population[:int(len(weighted_population)/5)]], layer_type)
        if global_vars.get('grid') and not global_vars.get('grid_as_ensemble'):
            stats['num_of_models_with_skip'] = NAS_utils.num_of_models_with_skip_connection(weighted_population)
        return stats

    def add_final_stats(self, stats, weighted_population):
        model = finalize_model(weighted_population[0]['model'])
        if global_vars.get('cross_subject'):
            self.current_chosen_population_sample = range(1, global_vars.get('num_subjects') + 1)
        for subject in self.current_chosen_population_sample:
            if global_vars.get('ensemble_iterations'):
                ensemble = [finalize_model(weighted_population[i]['model']) for i in range(
                    global_vars.get('ensemble_size'))]
                _, evaluations, _, num_epochs = self.ensemble_evaluate_model(ensemble, final_evaluation=True, subject=subject)
                NAS_utils.add_evaluations_to_stats(stats, evaluations, str_prefix=f"{subject}_final_")
            _, evaluations, _, _, num_epochs = self.activate_model_evaluation(model, final_evaluation=True, subject=subject)
            NAS_utils.add_evaluations_to_stats(stats, evaluations, str_prefix=f"{subject}_final_")
            stats['%d_final_epoch_num' % subject] = num_epochs

    def save_best_model(self, weighted_population):
        if global_vars.get('delete_finalized_models'):
            save_model = finalize_model(weighted_population[0]['model'])
            save_model.load_state_dict(weighted_population[0]['model_state'])
        else:
            save_model = weighted_population[0]['finalized_model'].to("cpu")
        subject_nums = '_'.join(str(x) for x in self.current_chosen_population_sample)
        self.model_filename = f'{self.exp_folder}/best_model_{subject_nums}.th'
        torch.save(save_model, self.model_filename)
        return self.model_filename

    def save_final_population(self, weighted_population):
        subject_nums = '_'.join(str(x) for x in self.current_chosen_population_sample)
        pretrained_str = ''
        if not global_vars.get('delete_finalized_models'):
            pretrained_str = '_pretrained'
        self.weighted_population_file = f'{self.exp_folder}/weighted_population_{subject_nums}{pretrained_str}.p'
        pickle.dump(weighted_population, open(self.weighted_population_file, 'wb'))

    def evaluate_and_sort(self, weighted_population):
        self.evo_strategy(weighted_population)
        getattr(EEGNAS.evolution.fitness_functions, global_vars.get('fitness_function'))(weighted_population)
        if global_vars.get('fitness_penalty_function'):
            getattr(NAS_utils, global_vars.get('fitness_penalty_function'))(weighted_population)
        reverse_order = True
        if self.loss_function == F.mse_loss:
            reverse_order = False
        weighted_population = NAS_utils.sort_population(weighted_population, reverse=reverse_order)
        stats = self.calculate_stats(weighted_population)
        add_parent_child_relations(weighted_population, stats)
        if global_vars.get('ranking_correlation_num_iterations'):
            NAS_utils.ranking_correlations(weighted_population, stats)
        return stats, weighted_population

    def update_mutation_rate(self):
        if global_vars.get('dynamic_mutation_rate'):
            if len(self.models_set) < global_vars.get('pop_size') * global_vars.get('unique_model_threshold'):
                self.mutation_rate *= global_vars.get('mutation_rate_increase_rate')
            else:
                if global_vars.get('mutation_rate_gradual_decrease'):
                    self.mutation_rate /= global_vars.get('mutation_rate_decrease_rate')
                else:
                    self.mutation_rate = global_vars.get('mutation_rate')

    def evolution(self):
        num_generations = global_vars.get('num_generations')
        weighted_population = NAS_utils.initialize_population(self.models_set, self.genome_set, self.subject_id)
        all_architectures = []
        start_time = time.time()
        for generation in range(num_generations):
            self.current_generation = generation
            if global_vars.get('perm_ensembles'):
                self.mark_perm_ensembles(weighted_population)
            if global_vars.get('inject_dropout') and generation == int((num_generations / 2) - 1):
                NAS_utils.inject_dropout(weighted_population)
            stats, weighted_population = self.evaluate_and_sort(weighted_population)
            for stat, val in stats.items():
                global_vars.get('sacred_ex').log_scalar(f'avg_{stat}', val, generation)
            if generation < num_generations - 1:
                weighted_population = self.selection(weighted_population)
                breed_population(weighted_population, self)
                self.update_mutation_rate()
                if global_vars.get('save_every_generation'):
                    self.save_best_model(weighted_population)
                all_architectures.append([pop['model'] for pop in weighted_population])
                if time.time() - start_time > global_vars.get('time_limit_seconds'):
                    break
            best_model_filename = self.save_best_model(weighted_population)
            pickle.dump(weighted_population, open(f'{self.exp_folder}/{self.exp_name}_architectures.p', 'wb'))
            self.write_to_csv({k: str(v) for k, v in stats.items()}, generation + 1)
            self.print_to_evolution_file(weighted_population, generation + 1)
        return best_model_filename

    def ea_deap(self, population, toolbox, ngen, stats=None, verbose=__debug__):
        history = History()
        toolbox.decorate("mate", history.decorator)
        history.update(population)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate all individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            self.current_generation += 1
            # Select the next generation individuals
            old_offspring = deepcopy(population)
            offspring = toolbox.select(population, len(population))
            # Vary the pool of individuals
            new_offspring = toolbox.mate(offspring, toolbox)
            hash_models_deap(old_offspring, new_offspring, self.genome_set, self.models_set)
            self.update_mutation_rate()
            # Evaluate the individuals with an invalid fitness
            fitnesses = toolbox.map(toolbox.evaluate, new_offspring)
            for ind, fit in zip(new_offspring, fitnesses):
                ind.fitness.values = fit
                ind['age'] += 1

            # Replace the current population by the offspring
            population[:] = new_offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)
            pop_stats = self.calculate_stats(self.population)
            for stat, val in pop_stats.items():
                global_vars.get('sacred_ex').log_scalar(f'avg_{stat}', val, self.current_generation)
            self.write_to_csv({k: str(v) for k, v in pop_stats.items()}, self.current_generation)
            self.print_to_evolution_file(self.population, self.current_generation)
        return population, logbook

    def eaSimple(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
                 halloffame=None, verbose=__debug__):
        history = History()
        toolbox.decorate("mate", history.decorator)
        toolbox.decorate("mutate", history.decorator)
        history.update(population)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            self.current_generation += 1
            # Select the next generation individuals
            old_offspring = deepcopy(population)
            offspring = toolbox.select(population, len(population))
            # Vary the pool of individuals
            new_offspring = varAnd(offspring, toolbox, cxpb, mutpb)
            hash_models_deap(old_offspring, new_offspring, self.genome_set, self.models_set)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in new_offspring if not ind.fitness.valid]
            valid_ind = [ind for ind in new_offspring if ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            for ind in valid_ind:
                ind['age'] += 1

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(new_offspring)

            # Replace the current population by the offspring
            population[:] = new_offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
            pop_stats = self.calculate_stats(self.population)
            for stat, val in pop_stats.items():
                global_vars.get('sacred_ex').log_scalar(f'avg_{stat}', val, self.current_generation)
            self.write_to_csv({k: str(v) for k, v in pop_stats.items()}, self.current_generation)
            self.print_to_evolution_file(self.population, self.current_generation)

        graph = networkx.DiGraph(history.genealogy_tree)
        graph = graph.reverse()
        colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
        positions = networkx.drawing.nx_agraph.graphviz_layout(graph, prog="dot")
        networkx.draw_networkx_labels(graph, positions)
        networkx.draw(graph, positions, node_color=colors)
        with open(f'{self.exp_folder}/{self.exp_name}_genealogy_models.txt', "w") as text_file:
            text_file.write(pretty_print_population({idx: pop['model'] for idx, pop in history.genealogy_history.items()}))
        plt.savefig(f'{self.exp_folder}/{self.exp_name}_genealogy.png')
        global_vars.get('sacred_ex').add_artifact(f'{self.exp_folder}/{self.exp_name}_genealogy.png')
        global_vars.get('sacred_ex').add_artifact(f'{self.exp_folder}/{self.exp_name}_genealogy_models.txt')
        return population, logbook

    def eaDual(self, population, modules, toolbox, cxpb, mutpb, ngen, stats=None,
                 halloffame=None, verbose=__debug__):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            self.current_generation += 1
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Change the toolbox methods for model mating
            self.toolbox.register("mate", breed_layers_modules_deap, self.toolbox)
            self.toolbox.register("mutate", mutate_layers_deap_modules)

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            # Check model validity, killing invalid ones
            for pop_idx, pop in enumerate(offspring):
                if not check_legal_model(pop['model']):
                    offspring[pop_idx]['model'] = random_model(global_vars.get('num_layers'))
                    offspring[pop_idx]['model_state'] = None
                    offspring[pop_idx]['age'] = 0
                    del offspring[pop_idx].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            valid_ind = [ind for ind in offspring if ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            for ind in valid_ind:
                ind['age'] += 1


            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            if global_vars.get('evolve_modules'):
                # Evaluate all modules
                module_fitnesses = toolbox.map(toolbox.evaluate_module, modules)
                for ind, fit in zip(modules, module_fitnesses):
                    ind.fitness.values = fit

                # Select the next generation modules
                module_offspring = toolbox.select(modules, len(modules))

                # Change the toolbox methods for module mating
                self.toolbox.register("mate", breed_modules_deap)
                self.toolbox.register("mutate", mutate_modules_deap)

                # Vary the pool of modules
                module_offspring = varAnd(module_offspring, toolbox, cxpb, mutpb)
                modules[:] = module_offspring

                # Update models with new modules, killing if necessary
                for pop_idx, pop in enumerate(population):
                    for idx in range(len(pop['model'])):
                        pop['model'][idx] = global_vars.get('modules')[pop['model'][idx].module_idx]
                    if not check_legal_model(pop['model']):
                        population[pop_idx]['model'] = random_model(global_vars.get('num_layers'))
                        population[pop_idx]['model_state'] = None
                        population[pop_idx]['age'] = 0
                        del population[pop_idx].fitness.values
                        fitness = toolbox.evaluate(population[pop_idx])
                        population[pop_idx].fitness.values = fitness

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
            pop_stats = self.calculate_stats(population)
            for stat, val in pop_stats.items():
                global_vars.get('sacred_ex').log_scalar(f'avg_{stat}', val, self.current_generation)
            self.write_to_csv({k: str(v) for k, v in pop_stats.items()}, self.current_generation)
            self.print_to_evolution_file(population, self.current_generation)
        return population, logbook

    def evolution_deap(self):
        if global_vars.get('module_evolution'):
            return self.evolution_deap_modules()
        if global_vars.get('problem') == 'classification':
            creator.create("Fitness", base.Fitness, weights=(1.0,))
        elif global_vars.get('problem') == 'regression':
            creator.create("Fitness", base.Fitness, weights=(-1.0,))
        creator.create("Individual", dict, fitness=creator.Fitness)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_ind_deap)
        self.toolbox.register("mate", breed_normal_population_deap)
        # self.toolbox.register("mutate", mutate_layers_deap)
        self.toolbox.register("select", selection_normal_deap)
        self.population = self.toolbox.population(global_vars.get('pop_size'))
        self.current_generation = 0
        initialize_deap_population(self.population, self.models_set, self.genome_set)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        final_population, logbook = self.ea_deap(self.population, self.toolbox, global_vars.get('num_generations'),
                                                  stats=stats, verbose=True)
        best_model_filename = self.save_best_model(final_population)
        pickle.dump(self.population, open(f'{self.exp_folder}/{self.exp_name}_architectures.p', 'wb'))
        return best_model_filename

    def evolution_deap_modules(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", dict, fitness=creator.FitnessMax)
        creator.create("Module", Module, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", creator.Individual)
        self.toolbox.register("module", tools.initRepeat, creator.Module, random_layer, n=global_vars.get('module_size'))
        self.toolbox.register("model_population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("module_population", tools.initRepeat, list, self.toolbox.module)
        self.toolbox.register("evaluate", self.evaluate_ind_deap)
        self.toolbox.register("evaluate_module", self.evaluate_module_deap)
        self.toolbox.register("mate", breed_layers_modules_deap, self.toolbox)
        self.toolbox.register("mutate", mutate_layers_deap_modules)
        self.toolbox.register("select", selTournament, tournsize=3)
        self.population = self.toolbox.model_population(global_vars.get('pop_size'))
        self.modules = self.toolbox.module_population(global_vars.get('module_pop_size'))
        for idx, module in enumerate(self.modules):
            module.module_idx = idx
        self.current_generation = 0
        global_vars.set('modules', self.modules)
        initialize_deap_population(self.population, self.models_set, self.genome_set)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        final_population, logbook = self.eaDual(self.population, self.modules, self.toolbox, 0.2, global_vars.get('mutation_rate'),
                                                  global_vars.get('num_generations'), stats=stats, verbose=True)
        best_model_filename = self.save_best_model(final_population)
        pickle.dump(self.population, open(f'{self.exp_folder}/{self.exp_name}_architectures.p', 'wb'))
        return best_model_filename

    def selection(self, weighted_population):
        if global_vars.get('perm_ensembles'):
            return self.selection_perm_ensembles(weighted_population)
        else:
            return self.selection_normal(weighted_population)

    def selection_normal(self, weighted_population):
        for index, model in enumerate(weighted_population):
            decay_functions = {'linear': lambda x: x,
                               'log': lambda x: np.sqrt(np.log(x + 1))}
            if random.uniform(0, 1) < decay_functions[global_vars.get('decay_function')](index / global_vars.get('pop_size')):
                NAS_utils.remove_from_models_hash(model['model'], self.models_set, self.genome_set)
                del weighted_population[index]
            else:
                model['age'] += 1
        return weighted_population

    def selection_perm_ensembles(self, weighted_population):
        ensembles = list(NAS_utils.chunks(list(range(global_vars.get('pop_size'))), global_vars.get('ensemble_size')))
        for index, ensemble in enumerate(ensembles):
            decay_functions = {'linear': lambda x: x,
                               'log': lambda x: np.sqrt(np.log(x + 1))}
            if random.uniform(0, 1) < decay_functions[global_vars.get('decay_function')](index / len(ensembles)) and\
                len([pop for pop in weighted_population if pop is not None]) > 2 * global_vars.get('ensemble_size'):
                for pop in ensemble:
                    NAS_utils.remove_from_models_hash(weighted_population[pop]['model'], self.models_set, self.genome_set)
                    weighted_population[pop] = None
            else:
                for pop in ensemble:
                    weighted_population[pop]['age'] += 1
        return [pop for pop in weighted_population if pop is not None]


    def get_single_subj_dataset(self, subject=None, final_evaluation=False):
        if subject not in self.datasets['train'].keys():
            self.datasets['train'][subject], self.datasets['valid'][subject], self.datasets['test'][subject] = \
                get_train_val_test(global_vars.get('data_folder'), subject)
        single_subj_dataset = OrderedDict((('train', self.datasets['train'][subject]),
                                           ('valid', self.datasets['valid'][subject]),
                                           ('test', self.datasets['test'][subject])))
        if final_evaluation:
            single_subj_dataset['train'] = concatenate_sets(
                [single_subj_dataset['train'], single_subj_dataset['valid']])
        return single_subj_dataset

    def activate_model_evaluation(self, model, state=None, subject=None, final_evaluation=False, ensemble=False):
        print(f'free params in network:{NAS_utils.pytorch_count_params(model)}')
        if subject is None:
            subject = self.subject_id
        if self.cuda:
            torch.cuda.empty_cache()
        if final_evaluation:
            self.stop_criterion = Or([MaxEpochs(global_vars.get('final_max_epochs')),
                                      NoIncreaseDecrease('valid_accuracy', global_vars.get('final_max_increase_epochs'))])
        if global_vars.get('cropping'):
            self.set_cropping_for_model(model)
        dataset = self.get_single_subj_dataset(subject, final_evaluation)
        nn_trainer = NN_Trainer(self.iterator, self.loss_function, self.stop_criterion, self.monitors)
        return nn_trainer.train_and_evaluate_model(model, dataset, state=state)

    def write_to_csv(self, stats, generation, model='avg'):
        if self.csv_file is not None:
            with open(self.csv_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                if self.subject_id == 'all':
                    subject = ','.join(str(x) for x in self.current_chosen_population_sample)
                else:
                    subject = str(self.subject_id)
                for key, value in stats.items():
                    writer.writerow({'exp_name': self.exp_name, 'machine': platform.node(),
                                    'dataset': global_vars.get('dataset'), 'date': time.strftime("%d/%m/%Y"),
                                    'subject': subject, 'generation': str(generation), 'model': str(model),
                                    'param_name': key, 'param_value': value})

    def print_to_evolution_file(self, models, generation):
        global text_file
        if self.evolution_file is not None:
            with open(self.evolution_file, "a") as text_file_local:
                text_file = text_file_local
                for m_idx, model in enumerate(models):
                    print(f'Generation {generation}, model {m_idx}:', file=text_file)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
                    finalized_model = finalize_model(model['model'])
                    print_model = finalized_model.to(device)
                    summary(print_model, (global_vars.get('eeg_chans'), global_vars.get('input_height'),
                                          global_vars.get('input_width')), file=text_file)



