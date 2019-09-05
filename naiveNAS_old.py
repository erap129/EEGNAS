import itertools
import pickle
import platform

from data_preprocessing import get_train_val_test
from models_generation import finalize_model, target_model,\
    breed_layers, breed_two_ensembles
from braindecode.torch_ext.util import np_to_var
from braindecode.experiments.loggers import Printer
import models_generation
import logging
import torch.optim as optim
from utilities.misc import RememberBest
import pandas as pd
from collections import OrderedDict, defaultdict
import numpy as np
from data_preprocessing import get_pure_cross_subject
import time
import torch
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or, ColumnBelow
from braindecode.datautil.splitters import concatenate_sets
import os
import global_vars
import csv
from torch import nn
from utilities.model_summary import summary
from utilities.monitors import NoIncreaseDecrease
import NASUtils
import evolution.fitness_functions
import pdb
from tensorboardX import SummaryWriter

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import random
WARNING = '\033[93m'
ENDC = '\033[0m'
log = logging.getLogger(__name__)
model_train_times = []


def time_f(t_secs):
    try:
        val = int(t_secs)
    except ValueError:
        return "!!!ERROR: ARGUMENT NOT AN INTEGER!!!"
    pos = abs(int(t_secs))
    day = pos / (3600*24)
    rem = pos % (3600*24)
    hour = rem / 3600
    rem = rem % 3600
    mins = rem / 60
    secs = rem % 60
    res = '%02d:%02d:%02d:%02d' % (day, hour, mins, secs)
    if int(t_secs) < 0:
        res = "-%s" % res
    return res


def show_progress(train_time, exp_name):
    global model_train_times
    total_trainings = global_vars.get('num_generations') * global_vars.get('pop_size') * len(global_vars.get('subjects_to_check'))
    model_train_times.append(train_time)
    avg_model_train_time = sum(model_train_times) / len(model_train_times)
    time_left = (total_trainings - len(model_train_times)) * avg_model_train_time
    print(f"Experiment: {exp_name}, time left: {time_f(time_left)}")


class NaiveNAS:
    def __init__(self, iterator, exp_folder, exp_name, loss_function,
                 train_set, val_set, test_set, stop_criterion, monitors,
                 config, subject_id, fieldnames, strategy, evolution_file, csv_file, model_from_file=None):
        global model_train_times
        model_train_times = []
        self.iterator = iterator
        self.exp_folder = exp_folder
        self.exp_name = exp_name
        self.subject_id = subject_id
        self.config = config
        self.loss_function = loss_function
        self.model_from_file = model_from_file
        self.optimizer = None
        self.datasets = OrderedDict(
            (('train', train_set), ('valid', val_set), ('test', test_set))
        )
        self.rememberer = None
        self.loggers = [Printer()]
        self.stop_criterion = stop_criterion
        self.monitors = monitors
        self.cuda = global_vars.get('cuda')
        self.loggers = [Printer()]
        self.epochs_df = None
        self.fieldnames = fieldnames
        self.models_set = []
        self.genome_set = []
        self.evo_strategy = {'cross_subject': self.all_strategy, 'per_subject': self.one_strategy}[strategy]
        self.csv_file = csv_file
        self.evolution_file = evolution_file
        self.current_model_index = -1
        if isinstance(self.subject_id, int):
            self.current_chosen_population_sample = [self.subject_id]
        else:
            self.current_chosen_population_sample = []
        self.mutation_rate = global_vars.get('mutation_rate')

    def get_dummy_input(self):
        random_subj = list(self.datasets['train'].keys())[0]
        result = np_to_var(self.datasets['train'][random_subj].X[:1, :, :, None])
        if self.cuda:
            result = result.cuda()
        return result

    def finalized_model_to_dilated(self, model):
        to_dense_prediction_model(model)
        conv_classifier = model.conv_classifier
        model.conv_classifier = nn.Conv2d(conv_classifier.in_channels, conv_classifier.out_channels,
                                          (global_vars.get('final_conv_size'),
                                           conv_classifier.kernel_size[1]), stride=conv_classifier.stride,
                                          dilation=conv_classifier.dilation)

    def get_n_preds_per_input(self, model):
        dummy_input = self.get_dummy_input()
        if global_vars.get('cuda'):
            model.cuda()
            dummy_input = dummy_input.cuda()
        out = model(dummy_input)
        n_preds_per_input = out.cpu().data.numpy().shape[2]
        return n_preds_per_input

    def set_cropping_for_model(self, model):
        models_generation.finalized_model_to_dilated(model)
        global_vars.set('n_preds_per_input', models_generation.get_n_preds_per_input(model))
        self.iterator = CropsFromTrialsIterator(batch_size=global_vars.get('batch_size'),
                                                input_time_length=global_vars.get('input_time_len'),
                                                n_preds_per_input=global_vars.get('n_preds_per_input'))

    def run_target_ensemble(self):
        global_vars.set('max_epochs', global_vars.get('final_max_epochs'))
        global_vars.set('max_increase_epochs', global_vars.get('final_max_increase_epochs'))
        stats = {}
        self.weighted_population_file = f'weighted_populations/{global_vars.get("weighted_population_file")}'
        _, evaluations, _, num_epochs = self.evaluate_ensemble_from_pickle(self.subject_id)
        NASUtils.add_evaluations_to_stats(stats, evaluations, str_prefix='final_')
        self.write_to_csv(stats, generation=1)

    def run_target_model(self):
        global_vars.set('max_epochs', global_vars.get('final_max_epochs'))
        global_vars.set('max_increase_epochs', global_vars.get('final_max_increase_epochs'))
        stats = {}
        if self.model_from_file is not None:
            if torch.cuda.is_available():
                model = torch.load(self.model_from_file)
            else:
                model = torch.load(self.model_from_file, map_location='cpu')
        else:
            model = target_model(global_vars.get('model_name'))
        if global_vars.get('target_pretrain'):
            self.datasets['train']['pretrain'], self.datasets['valid']['pretrain'], self.datasets['test']['pretrain'] = \
                get_pure_cross_subject(global_vars.get('data_folder'), global_vars.get('low_cut_hz'))
            _, _, model, _, _ = self.evaluate_model(model, subject='pretrain')
        final_time, evaluations, model, model_state, num_epochs =\
                    self.evaluate_model(model, final_evaluation=True)
        stats['final_train_time'] = str(final_time)
        NASUtils.add_evaluations_to_stats(stats, evaluations, str_prefix="final_")
        self.write_to_csv(stats, generation=1)

    def sample_subjects(self):
        self.current_chosen_population_sample = sorted(random.sample(
            [i for i in range(1, global_vars.get('num_subjects') + 1) if i not in global_vars.get('exclude_subjects')],
            global_vars.get('cross_subject_sampling_rate')))

    def one_strategy(self, weighted_population):
        self.current_chosen_population_sample = [self.subject_id]
        for i, pop in enumerate(weighted_population):
            start_time = time.time()
            if NASUtils.check_age(pop):
                weighted_population[i] = weighted_population[i - 1]
                weighted_population[i]['train_time'] = 0
                weighted_population[i]['num_epochs'] = 0
                continue
            finalized_model = finalize_model(pop['model'])
            self.current_model_index = i
            final_time, evaluations, model, model_state, num_epochs = \
                self.evaluate_model(finalized_model, pop['model_state'], subject=self.subject_id)
            if global_vars.get('grid_as_ensemble') and global_vars.get('delete_finalized_models'):
                pop['weighted_avg_params'] = model
            self.current_model_index = -1
            NASUtils.add_evaluations_to_weighted_population(weighted_population[i], evaluations)
            weighted_population[i]['model_state'] = model_state
            weighted_population[i]['train_time'] = final_time
            weighted_population[i]['finalized_model'] = model
            weighted_population[i]['num_epochs'] = num_epochs
            end_time = time.time()
            show_progress(end_time - start_time, self.exp_name)
            print('trained model %d in generation %d' % (i + 1, self.current_generation))

    def all_strategy(self, weighted_population):
        summed_parameters = ['train_time', 'num_epochs']
        summed_parameters.extend(NASUtils.get_metric_strs())
        if global_vars.get('cross_subject_sampling_method') == 'generation':
            self.sample_subjects()
        for i, pop in enumerate(weighted_population):
            start_time = time.time()
            if NASUtils.check_age(pop):
                weighted_population[i] = weighted_population[i - 1]
                weighted_population[i]['train_time'] = 0
                weighted_population[i]['num_epochs'] = 0
                continue
            if global_vars.get('cross_subject_sampling_method') == 'model':
                self.sample_subjects()
            for key in summed_parameters:
                weighted_population[i][key] = 0
            for subject in random.sample(self.current_chosen_population_sample,
                                         len(self.current_chosen_population_sample)):
                finalized_model = finalize_model(pop['model'])
                final_time, evaluations, model, model_state, num_epochs = \
                    self.evaluate_model(finalized_model, pop['model_state'], subject=subject)
                NASUtils.add_evaluations_to_weighted_population(weighted_population[i], evaluations,
                                                                str_prefix=f"{subject}_")
                NASUtils.sum_evaluations_to_weighted_population(weighted_population[i], evaluations)
                weighted_population[i]['%d_train_time' % subject] = final_time
                weighted_population[i]['train_time'] += final_time
                weighted_population[i]['%d_model_state' % subject] = model_state
                weighted_population[i]['%d_num_epochs' % subject] = num_epochs
                weighted_population[i]['num_epochs'] += num_epochs
                end_time = time.time()
                show_progress(end_time - start_time, self.exp_name)
                print('trained model %d in subject %d in generation %d' % (i + 1, subject, self.current_generation))
            weighted_population[i]['finalized_model'] = model
            for key in summed_parameters:
                weighted_population[i][key] /= global_vars.get('cross_subject_sampling_rate')

    def calculate_stats(self, weighted_population):
        stats = {}
        params = ['train_time', 'num_epochs', 'fitness']
        params.extend(NASUtils.get_metric_strs())
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
            NASUtils.add_model_to_stats(pop, i, model_stats)
            self.write_to_csv(model_stats, self.current_generation+1, model=i)
        stats['unique_models'] = len(self.models_set)
        stats['unique_genomes'] = len(self.genome_set)
        layer_stats = {'average_conv_width': (models_generation.ConvLayer, 'kernel_eeg_chan'),
                       'average_conv_height': (models_generation.ConvLayer, 'kernel_time'),
                       'average_conv_filters': (models_generation.ConvLayer, 'filter_num'),
                       'average_pool_width': (models_generation.PoolingLayer, 'pool_time'),
                       'average_pool_stride': (models_generation.PoolingLayer, 'stride_time')}
        for stat in layer_stats.keys():
            stats[stat] = NASUtils.get_average_param([pop['model'] for pop in weighted_population],
                                                                 layer_stats[stat][0], layer_stats[stat][1])
            if global_vars.get('add_top_20_stats'):
                stats[f'top20_{stat}'] = NASUtils.get_average_param([pop['model'] for pop in
                                                                     weighted_population[:int(len(weighted_population)/5)]],
                                                         layer_stats[stat][0], layer_stats[stat][1])
        stats['average_age'] = np.mean([sample['age'] for sample in weighted_population])
        stats['mutation_rate'] = self.mutation_rate
        for layer_type in [models_generation.DropoutLayer, models_generation.ActivationLayer, models_generation.ConvLayer,
                           models_generation.IdentityLayer, models_generation.BatchNormLayer, models_generation.PoolingLayer]:
            stats['%s_count' % layer_type.__name__] = \
                NASUtils.count_layer_type_in_pop([pop['model'] for pop in weighted_population], layer_type)
            if global_vars.get('add_top_20_stats'):
                stats['top20_%s_count' % layer_type.__name__] = \
                    NASUtils.count_layer_type_in_pop([pop['model'] for pop in
                                                      weighted_population[:int(len(weighted_population)/5)]], layer_type)
        if global_vars.get('grid') and not global_vars.get('grid_as_ensemble'):
            stats['num_of_models_with_skip'] = NASUtils.num_of_models_with_skip_connection(weighted_population)
        return stats

    def add_final_stats(self, stats, weighted_population):
        model = finalize_model(weighted_population[0]['model'])
        if global_vars.get('cross_subject'):
            self.current_chosen_population_sample = range(1, global_vars.get('num_subjects') + 1)
        for subject in self.current_chosen_population_sample:
            if global_vars.get('ensemble_iterations'):
                ensemble = [finalize_model(weighted_population[i]['model']) for i in range(global_vars.get('ensemble_size'))]
                _, evaluations, _, num_epochs = self.ensemble_evaluate_model(ensemble, final_evaluation=True, subject=subject)
                NASUtils.add_evaluations_to_stats(stats, evaluations, str_prefix=f"{subject}_final_")
            _, evaluations, _, _, num_epochs = self.evaluate_model(model, final_evaluation=True, subject=subject)
            NASUtils.add_evaluations_to_stats(stats, evaluations, str_prefix=f"{subject}_final_")
            stats['%d_final_epoch_num' % subject] = num_epochs

    def validate_model_from_file(self, stats):
        if global_vars.get('ensemble_iterations'):
            weighted_population = pickle.load(open(self.weighted_population_file, 'rb'))
        for subject in self.current_chosen_population_sample:
            for iteration in range(global_vars.get('final_test_iterations')):
                model = torch.load(self.model_filename)
                _, evaluations, _, _, num_epochs = self.evaluate_model(model, final_evaluation=True, subject=subject)
                NASUtils.add_evaluations_to_stats(stats, evaluations,
                                                  str_prefix=f"{subject}_iteration_{iteration}_from_file_")
                if global_vars.get('ensemble_iterations'):
                    _, evaluations, _, num_epochs = self.evaluate_ensemble_from_pickle(subject, weighted_population)
                    NASUtils.add_evaluations_to_stats(stats, evaluations,
                                                      str_prefix=f"{subject}_iteration_{iteration}_from_file_")

    def evaluate_ensemble_from_pickle(self, subject, weighted_population=None):
        if weighted_population is None:
            weighted_population = pickle.load(open(self.weighted_population_file, 'rb'))
        if 'pretrained' in self.weighted_population_file:
            ensemble = [weighted_population[i]['finalized_model'] for i in
                        range(global_vars.get('ensemble_size'))]
        else:
            best_model = finalize_model(weighted_population[0]['model'])
            best_model.load_state_dict(weighted_population[0]['model_state'])
            torch.save(best_model, 'best_model_1.th')
            ensemble = [finalize_model(weighted_population[i]['model']) for i in
                        range(global_vars.get('ensemble_size'))]
            self.datasets['train']['pretrain'], self.datasets['valid']['pretrain'], self.datasets['test']['pretrain'] = \
                get_pure_cross_subject(global_vars.get('data_folder'), global_vars.get('low_cut_hz'))
            for i, model in enumerate(ensemble):
                _, _, save_model, _, _ = self.evaluate_model(model, subject='pretrain')
                ensemble[i] = save_model
        return self.ensemble_evaluate_model(ensemble, final_evaluation=True, subject=subject)

    def save_best_model(self, weighted_population):
        if global_vars.get('delete_finalized_models'):
            save_model = finalize_model(weighted_population[0]['model'])
            save_model.load_state_dict(weighted_population[0]['model_state'])
        else:
            save_model = weighted_population[0]['finalized_model'].to("cpu")
        subject_nums = '_'.join(str(x) for x in self.current_chosen_population_sample)
        self.model_filename = f'{self.exp_folder}/best_model_{subject_nums}.th'
        torch.save(save_model, self.model_filename)
        # torch.onnx.export(save_model, models_generation.get_dummy_input(),
        #                   f'{self.exp_folder}/best_model_{subject_nums}.onnx',
        #                   operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN)
        return self.model_filename

    def save_final_population(self, weighted_population):
        subject_nums = '_'.join(str(x) for x in self.current_chosen_population_sample)
        pretrained_str = ''
        if not global_vars.get('delete_finalized_models'):
            pretrained_str = '_pretrained'
        self.weighted_population_file = f'{self.exp_folder}/weighted_population_{subject_nums}{pretrained_str}.p'
        pickle.dump(weighted_population, open(self.weighted_population_file, 'wb'))

    @staticmethod
    def add_parent_child_relations(weighted_population, stats):
        avg_ratio = 0
        avg_count = 0
        for pop in weighted_population:
            if 'parents' in pop:
                parent_fitness = (pop['parents'][0]['fitness'] + pop['parents'][1]['fitness']) / 2
                avg_ratio += pop['fitness'] / parent_fitness
                avg_count += 1
        if avg_count != 0:
            stats['parent_child_ratio'] = avg_ratio / avg_count

    def evaluate_and_sort(self, weighted_population):
        self.evo_strategy(weighted_population)
        getattr(evolution.fitness_functions, global_vars.get('fitness_function'))(weighted_population)
        if global_vars.get('fitness_penalty_function'):
            getattr(evolution.fitness_functions, global_vars.get('fitness_penalty_function'))(weighted_population)
        weighted_population = NASUtils.sort_population(weighted_population)
        stats = self.calculate_stats(weighted_population)
        self.add_parent_child_relations(weighted_population, stats)
        if global_vars.get('ranking_correlation_num_iterations'):
            NASUtils.ranking_correlations(weighted_population, stats)
        return stats, weighted_population

    @staticmethod
    def mark_perm_ensembles(weighted_population):
        for i, pop in enumerate(weighted_population):
            pop['perm_ensemble_role'] = i % global_vars.get('ensemble_size')
            pop['perm_ensemble_id'] = int(i / global_vars.get('ensemble_size'))

    def evolution(self):
        num_generations = global_vars.get('num_generations')
        weighted_population = NASUtils.initialize_population(self.models_set, self.genome_set, self.subject_id)
        for generation in range(num_generations):
            self.current_generation = generation
            if global_vars.get('perm_ensembles'):
                self.mark_perm_ensembles(weighted_population)
            if global_vars.get('inject_dropout') and generation == int((num_generations / 2) - 1):
                NASUtils.inject_dropout(weighted_population)
            stats, weighted_population = self.evaluate_and_sort(weighted_population)
            if generation < num_generations - 1:
                weighted_population = self.selection(weighted_population)
                self.breed_population(weighted_population)
                if global_vars.get('dynamic_mutation_rate'):
                    if len(self.models_set) < global_vars.get('pop_size') * global_vars.get('unique_model_threshold'):
                        self.mutation_rate *= global_vars.get('mutation_rate_increase_rate')
                    else:
                        if global_vars.get('mutation_rate_gradual_decrease'):
                            self.mutation_rate /= global_vars.get('mutation_rate_decrease_rate')
                        else:
                            self.mutation_rate = global_vars.get('mutation_rate')
                if global_vars.get('save_every_generation'):
                    self.save_best_model(weighted_population)
            else:  # last generation
                best_model_filename = self.save_best_model(weighted_population)
                self.save_final_population(weighted_population)
                # self.add_final_stats(stats, weighted_population)
                # self.validate_model_from_file(stats)

            self.write_to_csv({k: str(v) for k, v in stats.items()}, generation + 1)
            self.print_to_evolution_file(weighted_population[:3], generation + 1)
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
                NASUtils.remove_from_models_hash(model['model'], self.models_set, self.genome_set)
                del weighted_population[index]
            else:
                model['age'] += 1
        return weighted_population

    def selection_perm_ensembles(self, weighted_population):
        ensembles = list(NASUtils.chunks(list(range(global_vars.get('pop_size'))), global_vars.get('ensemble_size')))
        for index, ensemble in enumerate(ensembles):
            decay_functions = {'linear': lambda x: x,
                               'log': lambda x: np.sqrt(np.log(x + 1))}
            if random.uniform(0, 1) < decay_functions[global_vars.get('decay_function')](index / len(ensembles)) and\
                len([pop for pop in weighted_population if pop is not None]) > 2 * global_vars.get('ensemble_size'):
                for pop in ensemble:
                    NASUtils.remove_from_models_hash(weighted_population[pop]['model'], self.models_set, self.genome_set)
                    weighted_population[pop] = None
            else:
                for pop in ensemble:
                    weighted_population[pop]['age'] += 1
        return [pop for pop in weighted_population if pop is not None]

    def breed_population(self, weighted_population):
        if global_vars.get('grid'):
            breeding_method = models_generation.breed_grid
        else:
            breeding_method = breed_layers
        if global_vars.get('perm_ensembles'):
            self.breed_perm_ensembles(weighted_population, breeding_method)
        else:
            self.breed_normal_population(weighted_population, breeding_method)

    def breed_perm_ensembles(self, weighted_population, breeding_method):
        children = []
        ensembles = list(NASUtils.chunks(list(range(len(weighted_population))), global_vars.get('ensemble_size')))
        while len(weighted_population) + len(children) < global_vars.get('pop_size'):
            breeders = random.sample(ensembles, 2)
            first_ensemble = [weighted_population[i] for i in breeders[0]]
            second_ensemble = [weighted_population[i] for i in breeders[1]]
            for ensemble in [first_ensemble, second_ensemble]:
                assert(len(np.unique([pop['perm_ensemble_id'] for pop in ensemble])) == 1)
            first_ensemble_states = [NASUtils.get_model_state(pop) for pop in first_ensemble]
            second_ensemble_states = [NASUtils.get_model_state(pop) for pop in second_ensemble]
            new_ensemble, new_ensemble_states, cut_point = breed_two_ensembles(breeding_method, mutation_rate=self.mutation_rate,
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
                    NASUtils.hash_model(new_model, self.models_set, self.genome_set)
        weighted_population.extend(children)

    def breed_normal_population(self, weighted_population, breeding_method):
        children = []
        while len(weighted_population) + len(children) < global_vars.get('pop_size'):
            breeders = random.sample(range(len(weighted_population)), 2)
            first_breeder = weighted_population[breeders[0]]
            second_breeder = weighted_population[breeders[1]]
            first_model_state = NASUtils.get_model_state(first_breeder)
            second_model_state = NASUtils.get_model_state(second_breeder)
            new_model, new_model_state, cut_point = breeding_method(mutation_rate=self.mutation_rate,
                                                         first_model=first_breeder['model'],
                                                         second_model=second_breeder['model'],
                                                         first_model_state=first_model_state,
                                                         second_model_state=second_model_state)
            if new_model is not None:
                children.append({'model': new_model, 'model_state': new_model_state, 'age': 0,
                                 'parents': [first_breeder, second_breeder], 'cut_point': cut_point,
                                 'first_parent_index': breeders[0], 'second_parent_index': breeders[1]})
                NASUtils.hash_model(new_model, self.models_set, self.genome_set)
        weighted_population.extend(children)

    def ensemble_by_avg_layer(self, trained_models, subject):
        trained_models = [nn.Sequential(*list(model.children())[:global_vars.get('num_layers') + 1]) for model in trained_models]
        avg_model = models_generation.AveragingEnsemble(trained_models)
        single_subj_dataset = self.get_single_subj_dataset(subject, final_evaluation=True)
        if global_vars.get('ensemble_trained_average'):
            _, _, avg_model, state, _ = self.evaluate_model(avg_model, None, subject, final_evaluation=True)
        else:
            self.monitor_epoch(single_subj_dataset, avg_model)
        new_avg_evaluations = defaultdict(dict)
        objective_str = global_vars.get("ga_objective")
        for dataset in ['train', 'valid', 'test']:
            new_avg_evaluations[f'ensemble_{objective_str}'][dataset] = \
                self.epochs_df.tail(1)[f'{dataset}_{objective_str}'].values[0]
        return new_avg_evaluations

    def ensemble_evaluate_model(self, models, states=None, subject=None, final_evaluation=False):
        if states is None:
            states = [None for i in range(len(models))]
        avg_final_time = 0
        avg_num_epochs = 0
        avg_evaluations = {}
        _, evaluations, _, _, _ = self.evaluate_model(models[0], states[0], subject)
        for eval in evaluations.keys():
            avg_evaluations[eval] = defaultdict(list)
        trained_models = []
        for model, state in zip(models, states):
            if global_vars.get('ensemble_pretrain'):
                if global_vars.get('random_subject_pretrain'):
                    pretrain_subject = random.randint(1, global_vars.get('num_subjects'))
                else:
                    pretrain_subject = subject
                _, _, model, state, _ = self.evaluate_model(model, state, pretrain_subject)
            final_time, evaluations, model, state, num_epochs = self.evaluate_model(model, state, subject,
                                                                                    final_evaluation, ensemble=True)
            for key, eval in evaluations.items():
                for inner_key, eval_spec in eval.items():
                    avg_evaluations[key][inner_key].append(eval_spec)
            avg_final_time += final_time
            avg_num_epochs += num_epochs
            states.append(state)
            trained_models.append(model)
        if global_vars.get('ensembling_method') == 'manual':
            new_avg_evaluations = NASUtils.format_manual_ensemble_evaluations(avg_evaluations)
        elif global_vars.get('ensembling_method') == 'averaging_layer':
            new_avg_evaluations = self.ensemble_by_avg_layer(trained_models, subject)
        return avg_final_time, new_avg_evaluations, states, avg_num_epochs

    def get_single_subj_dataset(self, subject=None, final_evaluation=False):
        if subject not in self.datasets['train'].keys():
            self.datasets['train'][subject], self.datasets['valid'][subject], self.datasets['test'][subject] = \
                get_train_val_test(global_vars.get('data_folder'), subject, global_vars.get('low_cut_hz'))
        single_subj_dataset = OrderedDict((('train', self.datasets['train'][subject]),
                                           ('valid', self.datasets['valid'][subject]),
                                           ('test', self.datasets['test'][subject])))
        if final_evaluation:
            single_subj_dataset['train'] = concatenate_sets(
                [single_subj_dataset['train'], single_subj_dataset['valid']])
        return single_subj_dataset

    def evaluate_model(self, model, state=None, subject=None, final_evaluation=False, ensemble=False):
        print(f'free params in network:{NASUtils.pytorch_count_params(model)}')
        if subject is None:
            subject = self.subject_id
        if self.cuda:
            torch.cuda.empty_cache()
        if final_evaluation:
            self.stop_criterion = Or([MaxEpochs(global_vars.get('final_max_epochs')),
                                      NoIncreaseDecrease('valid_accuracy', global_vars.get('final_max_increase_epochs'))])
        if global_vars.get('cropping'):
            self.set_cropping_for_model(model)
        single_subj_dataset = self.get_single_subj_dataset(subject, final_evaluation)
        self.epochs_df = pd.DataFrame()
        if global_vars.get('do_early_stop') or global_vars.get('remember_best'):
            self.rememberer = RememberBest(f"valid_{global_vars.get('nn_objective')}")
        self.optimizer = optim.Adam(model.parameters())
        if self.cuda:
            assert torch.cuda.is_available(), "Cuda not available"
            if torch.cuda.device_count() > 1 and global_vars.get('parallel_gpu'):
                model.cuda()
                with torch.cuda.device(0):
                    model = nn.DataParallel(model.cuda(), device_ids=
                        [int(s) for s in global_vars.get('gpu_select').split(',')])
            else:
                model.cuda()

        try:
            if global_vars.get('inherit_weights_normal') and state is not None:
                    current_state = model.state_dict()
                    for k, v in state.items():
                        if k in current_state and current_state[k].shape == v.shape:
                            current_state.update({k: v})
                    model.load_state_dict(current_state)
        except Exception as e:
            print(f'failed weight inheritance\n,'
                  f'state dict: {state.keys()}\n'
                  f'current model state: {model.state_dict().keys()}')
            print('load state dict failed. Exception message: %s' % (str(e)))
            pdb.set_trace()
        self.monitor_epoch(single_subj_dataset, model)
        if global_vars.get('log_epochs'):
            self.log_epoch()
        if global_vars.get('remember_best'):
            self.rememberer.remember_epoch(self.epochs_df, model, self.optimizer)
        self.iterator.reset_rng()
        start = time.time()
        num_epochs = self.run_until_stop(model, single_subj_dataset)
        self.setup_after_stop_training(model, final_evaluation)
        if final_evaluation:
            loss_to_reach = float(self.epochs_df['train_loss'].iloc[-1])
            if ensemble:
                self.run_one_epoch(single_subj_dataset, model)
                self.rememberer.remember_epoch(self.epochs_df, model, self.optimizer, force=ensemble)
                num_epochs += 1
            num_epochs += self.run_until_stop(model, single_subj_dataset)
            if float(self.epochs_df['valid_loss'].iloc[-1]) > loss_to_reach or ensemble:
                self.rememberer.reset_to_best_model(self.epochs_df, model, self.optimizer)
        end = time.time()
        evaluations = {}
        for evaluation_metric in global_vars.get('evaluation_metrics'):
            evaluations[evaluation_metric] = {'train': self.epochs_df.iloc[-1][f"train_{evaluation_metric}"],
                                              'valid': self.epochs_df.iloc[-1][f"valid_{evaluation_metric}"],
                                              'test': self.epochs_df.iloc[-1][f"test_{evaluation_metric}"]}
        final_time = end-start
        if self.cuda:
            torch.cuda.empty_cache()
        if global_vars.get('use_tensorboard'):
            if self.current_model_index > 0:
                with SummaryWriter(log_dir=f'{self.exp_folder}/tensorboard/gen_{self.current_generation}_model{self.current_model_index}') as w:
                    w.add_graph(model, self.get_dummy_input())
        if global_vars.get('delete_finalized_models'):
            if global_vars.get('grid_as_ensemble'):
                model_stats = {}
                for param_idx, param in enumerate(list(model.pytorch_layers['averaging_layer'].parameters())):
                    for inner_idx, inner_param in enumerate(param[0]):
                        model_stats[f'avg_weights {(param_idx, inner_idx)}'] = float(inner_param)
                del model
                model = model_stats
            else:
                del model
                model = None
        return final_time, evaluations, model, self.rememberer.model_state_dict, num_epochs

    def setup_after_stop_training(self, model, final_evaluation):
        self.rememberer.reset_to_best_model(self.epochs_df, model, self.optimizer)
        if final_evaluation:
            loss_to_reach = float(self.epochs_df['train_loss'].iloc[-1])
            self.stop_criterion = Or(stop_criteria=[
                MaxEpochs(max_epochs=self.rememberer.best_epoch * 2),
                ColumnBelow(column_name='valid_loss', target_value=loss_to_reach)])

    def run_until_stop(self, model, single_subj_dataset):
        num_epochs = 0
        while not self.stop_criterion.should_stop(self.epochs_df):
            self.run_one_epoch(single_subj_dataset, model)
            num_epochs += 1
        return num_epochs

    def run_one_epoch(self, datasets, model):
        model.train()
        batch_generator = self.iterator.get_batches(datasets['train'], shuffle=True)
        for inputs, targets in batch_generator:
            input_vars = np_to_var(inputs, pin_memory=global_vars.get('pin_memory'))
            target_vars = np_to_var(targets, pin_memory=global_vars.get('pin_memory'))
            if self.cuda:
                with torch.cuda.device(0):
                    input_vars = input_vars.cuda()
                    target_vars = target_vars.cuda()
            self.optimizer.zero_grad()
            try:
                outputs = model(input_vars)
            except RuntimeError as e:
                print('run model failed. Exception message: %s' % (str(e)))
                pdb.set_trace()
            loss = self.loss_function(outputs, target_vars)
            loss.backward()
            self.optimizer.step()
        self.monitor_epoch(datasets, model)
        if global_vars.get('log_epochs'):
            self.log_epoch()
        if global_vars.get('remember_best'):
            self.rememberer.remember_epoch(self.epochs_df, model, self.optimizer)

    def monitor_epoch(self, datasets, model):
        result_dicts_per_monitor = OrderedDict()
        for m in self.monitors:
            result_dicts_per_monitor[m] = OrderedDict()
        for m in self.monitors:
            result_dict = m.monitor_epoch()
            if result_dict is not None:
                result_dicts_per_monitor[m].update(result_dict)
        raws = {}
        targets = {}
        for setname in datasets:
            assert setname in ['train', 'valid', 'test']
            dataset = datasets[setname]
            all_preds = []
            all_losses = []
            all_batch_sizes = []
            all_targets = []
            for batch in self.iterator.get_batches(dataset, shuffle=False):
                preds, loss = self.eval_on_batch(batch[0], batch[1], model)
                all_preds.append(preds)
                all_losses.append(loss)
                all_batch_sizes.append(len(batch[0]))
                all_targets.append(batch[1])
            for m in self.monitors:
                result_dict = m.monitor_set(setname, all_preds, all_losses,
                                            all_batch_sizes, all_targets,
                                            dataset)
                if result_dict is not None:
                    result_dicts_per_monitor[m].update(result_dict)

            if global_vars.get('ensemble_iterations'):
                raws.update({f'{setname}_raw': list(itertools.chain.from_iterable(all_preds))})
                targets.update({f'{setname}_target': list(itertools.chain.from_iterable(all_targets))})
        row_dict = OrderedDict()
        if global_vars.get('ensemble_iterations'):
            row_dict.update(raws)
            row_dict.update(targets)
        for m in self.monitors:
            row_dict.update(result_dicts_per_monitor[m])
        self.epochs_df = self.epochs_df.append(row_dict, ignore_index=True)
        assert set(self.epochs_df.columns) == set(row_dict.keys())
        self.epochs_df = self.epochs_df[list(row_dict.keys())]

    def eval_on_batch(self, inputs, targets, model):
        """
        Evaluate given inputs and targets.

        Parameters
        ----------
        inputs: `torch.autograd.Variable`
        targets: `torch.autograd.Variable`

        Returns
        -------
        predictions: `torch.autograd.Variable`
        loss: `torch.autograd.Variable`

        """
        model.eval()
        with torch.no_grad():
            input_vars = np_to_var(inputs, pin_memory=global_vars.get('pin_memory'))
            target_vars = np_to_var(targets, pin_memory=global_vars.get('pin_memory'))
            if self.cuda:
                with torch.cuda.device(0):
                    input_vars = input_vars.cuda()
                    target_vars = target_vars.cuda()
            outputs = model(input_vars)
            loss = self.loss_function(outputs, target_vars)
            if hasattr(outputs, 'cpu'):
                outputs = outputs.cpu().data.numpy()
            else:
                # assume it is iterable
                outputs = [o.cpu().data.numpy() for o in outputs]
            loss = loss.cpu().data.numpy()
        return outputs, loss

    def log_epoch(self):
        """
        Print monitoring values for this epoch.
        """
        for logger in self.loggers:
            logger.log_epoch(self.epochs_df)

    def train_pytorch(self, model):
        model.train()
        batch_generator = self.iterator.get_batches(self.train_set, shuffle=True)
        optimizer = optim.Adam(model.parameters())
        for inputs, targets in batch_generator:
            input_vars = np_to_var(inputs, pin_memory=global_vars.get('pin_memory'))
            target_vars = np_to_var(targets, pin_memory=global_vars.get('pin_memory'))
            optimizer.zero_grad()
            outputs = model(input_vars)
            loss = self.loss_function(outputs, target_vars)
            loss.backward()
            optimizer.step()

    def eval_pytorch(self, model, data):
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in self.iterator.get_batches(data, shuffle=False):
                input_vars = np_to_var(inputs, pin_memory=global_vars.get('pin_memory'))
                target_vars = np_to_var(targets, pin_memory=global_vars.get('pin_memory'))
                outputs = model(input_vars)
                val_loss += self.loss_function(outputs, target_vars)
                pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target_vars.view_as(pred)).sum().item()

            val_loss /= len(data.X)
        return correct / len(data.X)

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

    def garbage_time(self):
        model = target_model('deep')
        while 1:
            print('GARBAGE TIME GARBAGE TIME GARBAGE TIME')
            self.evaluate_model(model)

    def print_to_evolution_file(self, models, generation):
        global text_file
        if self.evolution_file is not None:
            with open(self.evolution_file, "a") as text_file_local:
                text_file = text_file_local
                print('Architectures for Subject %s, Generation %d\n' % (str(self.subject_id), generation), file=text_file)
                for model in models:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
                    finalized_model = finalize_model(model['model'])
                    print_model = finalized_model.to(device)
                    summary(print_model, (global_vars.get('eeg_chans'), global_vars.get('input_time_len'), 1), file=text_file)



