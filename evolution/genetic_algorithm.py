import itertools
import pickle
import platform

from data_preprocessing import get_train_val_test
from braindecode.torch_ext.util import np_to_var
from braindecode.experiments.loggers import Printer
import logging
import torch.optim as optim
import torch.nn.functional as F
from model_generation.abstract_layers import ConvLayer, PoolingLayer, DropoutLayer, ActivationLayer, BatchNormLayer, \
    IdentityLayer
from model_generation.simple_model_generation import finalize_model
from utilities.misc import RememberBest
import pandas as pd
from collections import OrderedDict, defaultdict
import numpy as np
from data_preprocessing import get_pure_cross_subject
import time
import torch
import evolution.fitness_functions
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.util import to_dense_prediction_model
from evolution.nn_training import NN_Trainer
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or, ColumnBelow
from braindecode.datautil.splitters import concatenate_sets
from evolution.evolution_misc_functions import add_parent_child_relations
from evolution.breeding import breed_population
import os
import global_vars
import csv
from torch import nn
import evolution.fitness_functions
from utilities.model_summary import summary
from utilities.monitors import NoIncreaseDecrease
import NASUtils
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
        self.evo_strategy = {'cross_subject': self.all_strategy, 'per_subject': self.one_strategy}[strategy]
        self.csv_file = csv_file
        self.evolution_file = evolution_file
        self.current_model_index = -1
        if isinstance(self.subject_id, int):
            self.current_chosen_population_sample = [self.subject_id]
        else:
            self.current_chosen_population_sample = []
        self.mutation_rate = global_vars.get('mutation_rate')

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
                self.activate_model_evaluation(finalized_model, pop['model_state'], subject=self.subject_id)
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
                    self.activate_model_evaluation(finalized_model, pop['model_state'], subject=subject)
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

            # if global_vars.get('add_top_20_stats'):
            #     stats[f'top20_{stat}'] = NASUtils.get_average_param([pop['model'] for pop in
            #                                                          weighted_population[:int(len(weighted_population)/5)]],
            #                                              layer_stats[stat][0], layer_stats[stat][1])
        stats['average_age'] = np.mean([sample['age'] for sample in weighted_population])
        stats['mutation_rate'] = self.mutation_rate
        for layer_type in [DropoutLayer, ActivationLayer, ConvLayer, IdentityLayer, BatchNormLayer, PoolingLayer]:
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
            _, evaluations, _, _, num_epochs = self.activate_model_evaluation(model, final_evaluation=True, subject=subject)
            NASUtils.add_evaluations_to_stats(stats, evaluations, str_prefix=f"{subject}_final_")
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
        getattr(evolution.fitness_functions, global_vars.get('fitness_function'))(weighted_population)
        if global_vars.get('fitness_penalty_function'):
            getattr(NASUtils, global_vars.get('fitness_penalty_function'))(weighted_population)
        reverse_order = True
        if self.loss_function == F.mse_loss:
            reverse_order = False
        weighted_population = NASUtils.sort_population(weighted_population, reverse=reverse_order)
        stats = self.calculate_stats(weighted_population)
        add_parent_child_relations(weighted_population, stats)
        if global_vars.get('ranking_correlation_num_iterations'):
            NASUtils.ranking_correlations(weighted_population, stats)
        return stats, weighted_population

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
                breed_population(weighted_population, self)
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

    def ensemble_by_avg_layer(self, trained_models, subject):
        trained_models = [nn.Sequential(*list(model.children())[:global_vars.get('num_layers') + 1]) for model in trained_models]
        avg_model = models_generation.AveragingEnsemble(trained_models)
        single_subj_dataset = self.get_single_subj_dataset(subject, final_evaluation=True)
        if global_vars.get('ensemble_trained_average'):
            _, _, avg_model, state, _ = self.activate_model_evaluation(avg_model, None, subject, final_evaluation=True)
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
                get_train_val_test(global_vars.get('data_folder'), subject)
        single_subj_dataset = OrderedDict((('train', self.datasets['train'][subject]),
                                           ('valid', self.datasets['valid'][subject]),
                                           ('test', self.datasets['test'][subject])))
        if final_evaluation:
            single_subj_dataset['train'] = concatenate_sets(
                [single_subj_dataset['train'], single_subj_dataset['valid']])
        return single_subj_dataset

    def activate_model_evaluation(self, model, state=None, subject=None, final_evaluation=False, ensemble=False):
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
        dataset = self.get_single_subj_dataset(subject, final_evaluation)
        nn_trainer = NN_Trainer(self.iterator, self.loss_function, self.stop_criterion, self.monitors)
        return nn_trainer.evaluate_model(model, dataset, state=state)

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
            self.activate_model_evaluation(model)

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
                    summary(print_model, (global_vars.get('eeg_chans'), global_vars.get('input_height'),
                            global_vars.get('input_width')), file=text_file)




