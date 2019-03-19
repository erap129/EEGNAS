from models_generation import random_model, finalize_model, target_model,\
    breed_layers
from braindecode.torch_ext.util import np_to_var
from braindecode.experiments.loggers import Printer
import models_generation
import logging
import torch.optim as optim
from utils import RememberBest
import pandas as pd
from collections import OrderedDict
import numpy as np
import time
import torch
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or, ColumnBelow
from braindecode.datautil.splitters import concatenate_sets
import os
import globals
import csv
from torch import nn
from utils import summary, NoIncrease, dump_tensors
import NASUtils
import pdb

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


def show_progress(train_time):
    global model_train_times
    total_trainings = globals.get('num_generations') * globals.get('pop_size') * len(globals.get('subjects_to_check'))
    model_train_times.append(train_time)
    avg_model_train_time = sum(model_train_times) / len(model_train_times)
    time_left = (total_trainings - len(model_train_times)) * avg_model_train_time
    print(f"time left: {time_f(time_left)}")


class NaiveNAS:
    def __init__(self, iterator, exp_folder, exp_name, loss_function,
                 train_set, val_set, test_set, stop_criterion, monitors,
                 config, subject_id, fieldnames, model_from_file=None):
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
        self.cuda = globals.get('cuda')
        self.loggers = [Printer()]
        self.epochs_df = None
        self.fieldnames = fieldnames
        self.models_set = []
        self.genome_set = []
        if isinstance(self.subject_id, int):
            self.current_chosen_population_sample = [self.subject_id]
        else:
            self.current_chosen_population_sample = []
        self.mutation_rate = globals.get('mutation_rate')

    def get_dummy_input(self):
        if globals.get('cross_subject'):
            random_subj = list(self.datasets['train'].keys())[0]
            return np_to_var(self.datasets['train'][random_subj].X[:1, :, :, None])
        else:
            return np_to_var(self.datasets['train'].X[:1, :, :, None])

    def finalized_model_to_dilated(self, model):
        to_dense_prediction_model(model)
        conv_classifier = model.conv_classifier
        model.conv_classifier = nn.Conv2d(conv_classifier.in_channels, conv_classifier.out_channels,
                                          (globals.get('final_conv_size'),
                                           conv_classifier.kernel_size[1]), stride=conv_classifier.stride,
                                          dilation=conv_classifier.dilation)
        dummy_input = self.get_dummy_input()
        if globals.get('cuda'):
            model.cuda()
            dummy_input = dummy_input.cuda()
        out = model(dummy_input)
        n_preds_per_input = out.cpu().data.numpy().shape[2]
        globals.set('n_preds_per_input', n_preds_per_input)
        self.iterator = CropsFromTrialsIterator(batch_size=globals.get('batch_size'),
                                                input_time_length=globals.get('input_time_len'),
                                                n_preds_per_input=globals.get('n_preds_per_input'))

    def run_target_model(self, csv_file):
        globals.set('max_epochs', globals.get('final_max_epochs'))
        globals.set('max_increase_epochs', globals.get('final_max_increase_epochs'))
        if self.model_from_file is not None:
            if torch.cuda.is_available():
                model = torch.load(self.model_from_file)
            else:
                model = torch.load(self.model_from_file, map_location='cpu')
        else:
            model = target_model(globals.get('model_name'))
        if globals.get('cropping'):
            self.finalized_model_to_dilated(model)
        final_time, evaluations, model_state, num_epochs =\
            self.evaluate_model(model, final_evaluation=True)
        stats = {'train_time': str(final_time)}
        NASUtils.add_evaluations_to_stats(stats, evaluations)
        self.write_to_csv(csv_file, stats, generation=1)

    def sample_subjects(self):
        self.current_chosen_population_sample = random.sample(
            [i for i in range(1, globals.get('num_subjects') + 1) if i not in globals.get('exclude_subjects')],
            globals.get('cross_subject_sampling_rate'))


    def one_strategy(self, weighted_population, generation):
        self.current_chosen_population_sample = [self.subject_id]
        for i, pop in enumerate(weighted_population):
            start_time = time.time()
            if NASUtils.check_age(pop):
                weighted_population[i] = weighted_population[i - 1]
                weighted_population[i]['train_time'] = 0
                weighted_population[i]['num_epochs'] = 0
                continue
            finalized_model = finalize_model(pop['model'])
            final_time, evaluations, model_state, num_epochs = \
                self.evaluate_model(finalized_model, pop['model_state'])
            NASUtils.add_evaluations_to_weighted_population(weighted_population[i], evaluations)
            weighted_population[i]['model_state'] = model_state
            # weighted_population[i]['finalized_model'] = model
            weighted_population[i]['train_time'] = final_time
            weighted_population[i]['num_epochs'] = num_epochs
            end_time = time.time()
            show_progress(end_time - start_time)
            print('trained model %d in generation %d' % (i + 1, generation))

    def all_strategy(self, weighted_population, generation):
        summed_parameters = ['train_time', 'num_epochs']
        summed_parameters.extend(NASUtils.get_metric_strs())
        if globals.get('cross_subject_sampling_method') == 'generation':
            self.sample_subjects()
        for i, pop in enumerate(weighted_population):
            start_time = time.time()
            if NASUtils.check_age(pop):
                weighted_population[i] = weighted_population[i - 1]
                weighted_population[i]['train_time'] = 0
                weighted_population[i]['num_epochs'] = 0
                continue
            if globals.get('cross_subject_sampling_method') == 'model':
                self.sample_subjects()
            for key in summed_parameters:
                weighted_population[i][key] = 0
            for subject in self.current_chosen_population_sample:
                finalized_model = finalize_model(pop['model'])
                final_time, evaluations, model_state, num_epochs = \
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
                show_progress(end_time - start_time)
                print('trained model %d in subject %d in generation %d' % (i + 1, subject, generation))
            for key in summed_parameters:
                weighted_population[i][key] /= globals.get('cross_subject_sampling_rate')

    def calculate_stats(self, weighted_population, evolution_file):
        stats = {}
        params = ['train_time', 'num_epochs']
        params.extend(NASUtils.get_metric_strs())
        for param in params:
            stats[param] = np.mean([sample[param] for sample in weighted_population])
        if self.subject_id == 'all':
            if globals.get('cross_subject_sampling_method') == 'model':
                self.current_chosen_population_sample = range(1, globals.get('num_subjects')+1)
            for subject in self.current_chosen_population_sample:
                for param in params:
                    stats['%d_%s' % (subject, param)] = np.mean(
                        [sample['%d_%s' % (subject, param)] for sample in weighted_population if '%d_%s' % (subject, param) in sample.keys()])
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
            if globals.get('add_top_20_stats'):
                stats[f'top20_{stat}'] = NASUtils.get_average_param([pop['model'] for pop in
                                                                     weighted_population[:int(len(weighted_population)/5)]],
                                                         layer_stats[stat][0], layer_stats[stat][1])
        stats['average_age'] = np.mean([sample['age'] for sample in weighted_population])
        # stats['similarity_measure'] = NASUtils.calculate_population_similarity(
        #     [pop['model'] for pop in weighted_population], evolution_file, sim_count=globals.get('sim_count'))
        stats['mutation_rate'] = self.mutation_rate
        for layer_type in [models_generation.DropoutLayer, models_generation.ActivationLayer, models_generation.ConvLayer,
                           models_generation.IdentityLayer, models_generation.BatchNormLayer, models_generation.PoolingLayer]:
            stats['%s_count' % layer_type.__name__] = \
                NASUtils.count_layer_type_in_pop([pop['model'] for pop in weighted_population], layer_type)
            if globals.get('add_top_20_stats'):
                stats['top20_%s_count' % layer_type.__name__] = \
                    NASUtils.count_layer_type_in_pop([pop['model'] for pop in
                                                      weighted_population[:int(len(weighted_population)/5)]], layer_type)
        if globals.get('grid'):
            stats['num_of_models_with_skip'] = NASUtils.num_of_models_with_skip_connection(weighted_population)
        return stats

    def add_final_stats(self, stats, model_filename):
        model = torch.load(model_filename)
        if globals.get('cropping'):
            self.finalized_model_to_dilated(model)
        if globals.get('cross_subject'):
            self.current_chosen_population_sample = range(1, globals.get('num_subjects') + 1)
        for subject in self.current_chosen_population_sample:
            _, evaluations, _, num_epochs = self.evaluate_model(model, final_evaluation=True, subject=subject)
            NASUtils.add_evaluations_to_stats(stats, evaluations, str_prefix=f"{subject}_final_")
            stats['%d_final_epoch_num' % subject] = num_epochs


    def save_best_model(self, weighted_population):
        try:
            save_model = finalize_model(weighted_population[0]['model']).to("cpu")
            model_filename = "%s/best_model_" % self.exp_folder +\
                             '_'.join(str(x) for x in self.current_chosen_population_sample) + ".th"
            torch.save(save_model, model_filename)
        except Exception as e:
            print('failed to save model. Exception message: %s' % (str(e)))
            pdb.set_trace()
        return model_filename

    def evolution(self, csv_file, evolution_file, model_init_configuration, evo_strategy):
        if globals.get('grid'):
            breeding_method = models_generation.breed_grid
            model_init = models_generation.random_grid_model
        else:
            breeding_method = breed_layers
            model_init = random_model

        pop_size = globals.get('pop_size')
        num_generations = globals.get('num_generations')
        evolution_results = pd.DataFrame()
        weighted_population = []
        for i in range(pop_size):  # generate pop_size random models
            new_rand_model = model_init(globals.get(model_init_configuration))
            NASUtils.hash_model(new_rand_model, self.models_set, self.genome_set)
            weighted_population.append({'model': new_rand_model, 'model_state': None, 'age': 0})

        for generation in range(num_generations):
            if globals.get('inject_dropout') and generation == int((num_generations / 2) - 1):
                NASUtils.inject_dropout(weighted_population)
            evo_strategy(weighted_population, generation)
            weighted_population = sorted(weighted_population,
                                         key=lambda x: x[f'val_{globals.get("ga_objective")}'], reverse=True)
            stats = self.calculate_stats(weighted_population, evolution_file)
            if generation < num_generations - 1:
                for index, model in enumerate(weighted_population):
                    if random.uniform(0, 1) < (index / pop_size):
                        NASUtils.remove_from_models_hash(model['model'], self.models_set, self.genome_set)
                        del weighted_population[index]
                    else:
                        model['age'] += 1
                children = []
                while len(weighted_population) + len(children) < pop_size:
                    breeders = random.sample(range(len(weighted_population)), 2)
                    first_breeder = weighted_population[breeders[0]]
                    second_breeder = weighted_population[breeders[1]]
                    first_model_state = NASUtils.get_model_state(first_breeder)
                    second_model_state = NASUtils.get_model_state(second_breeder)
                    new_model, new_model_state = breeding_method(mutation_rate=self.mutation_rate,
                                                                first_model=first_breeder['model'],
                                                                 second_model=second_breeder['model'],
                                                                 first_model_state=first_model_state,
                                                                 second_model_state=second_model_state)
                    if new_model is not None:
                        children.append({'model': new_model, 'model_state': new_model_state, 'age': 0})
                        NASUtils.hash_model(new_model, self.models_set, self.genome_set)
                weighted_population.extend(children)
                if globals.get('dynamic_mutation_rate'):
                    if len(self.models_set) < globals.get('pop_size') * globals.get('unique_model_threshold'):
                        self.mutation_rate *= globals.get('mutation_rate_change_factor')
                    else:
                        self.mutation_rate = globals.get('mutation_rate')
                if globals.get('save_every_generation'):
                    self.save_best_model(weighted_population)
            else:  # last generation
                model_filename = self.save_best_model(weighted_population)
                self.add_final_stats(stats, model_filename)

            self.write_to_csv(csv_file, {k: str(v) for k, v in stats.items()}, generation + 1)
            self.print_to_evolution_file(evolution_file, weighted_population[:3], generation)
        return evolution_results

    def evolution_layers(self, csv_file, evolution_file):
        return self.evolution(csv_file, evolution_file, 'num_layers', self.one_strategy)

    def evolution_layers_all(self, csv_file, evolution_file):
        return self.evolution(csv_file, evolution_file, 'num_layers', self.all_strategy)

    def evaluate_model(self, model, state=None, subject=None, final_evaluation=False):
        if self.cuda:
            torch.cuda.empty_cache()
        if final_evaluation:
            self.stop_criterion = Or([MaxEpochs(globals.get('final_max_epochs')),
                                 NoIncrease('valid_accuracy', globals.get('final_max_increase_epochs'))])
        if subject is not None and globals.get('cross_subject'):
            single_subj_dataset = OrderedDict((('train', self.datasets['train'][subject]),
                                               ('valid', self.datasets['valid'][subject]),
                                               ('test', self.datasets['test'][subject])))
        else:
            single_subj_dataset = self.datasets
        self.epochs_df = pd.DataFrame()
        if globals.get('do_early_stop'):
            self.rememberer = RememberBest(f"valid_{globals.get('nn_objective')}")
        self.optimizer = optim.Adam(model.parameters())
        if self.cuda:
            assert torch.cuda.is_available(), "Cuda not available"
            if torch.cuda.device_count() > 1 and globals.get('parallel_gpu'):
                model.cuda()
                with torch.cuda.device(0):
                    model = nn.DataParallel(model.cuda(), device_ids=
                        [int(s) for s in globals.get('gpu_select').split(',')])
            else:
                model.cuda()

        try:
            if globals.get('inherit_weights_normal') and state is not None:
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
        if globals.get('log_epochs'):
            self.log_epoch()
        if globals.get('remember_best'):
            self.rememberer.remember_epoch(self.epochs_df, model, self.optimizer)
        self.iterator.reset_rng()
        start = time.time()
        num_epochs = self.run_until_stop(model, single_subj_dataset)
        self.setup_after_stop_training(model, final_evaluation)
        if final_evaluation:
            loss_to_reach = float(self.epochs_df['train_loss'].iloc[-1])
            datasets = single_subj_dataset
            datasets['train'] = concatenate_sets([datasets['train'], datasets['valid']])
            num_epochs += self.run_until_stop(model, datasets)
            if float(self.epochs_df['valid_loss'].iloc[-1]) > loss_to_reach:
                self.rememberer.reset_to_best_model(self.epochs_df, model, self.optimizer)
        end = time.time()
        evaluations = {}
        for evaluation_metric in globals.get('evaluation_metrics'):
            evaluations[evaluation_metric] = {'train': self.epochs_df.iloc[-1][f"train_{evaluation_metric}"],
                                              'valid': self.epochs_df.iloc[-1][f"valid_{evaluation_metric}"],
                                              'test': self.epochs_df.iloc[-1][f"test_{evaluation_metric}"]}
        final_time = end-start
        del model
        if self.cuda:
            torch.cuda.empty_cache()
        return final_time, evaluations, self.rememberer.model_state_dict, num_epochs

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
            input_vars = np_to_var(inputs, pin_memory=globals.get('pin_memory'))
            target_vars = np_to_var(targets, pin_memory=globals.get('pin_memory'))
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
        if globals.get('log_epochs'):
            self.log_epoch()
        if globals.get('remember_best'):
            self.rememberer.remember_epoch(self.epochs_df, model, self.optimizer)

    def monitor_epoch(self, datasets, model):
        result_dicts_per_monitor = OrderedDict()
        for m in self.monitors:
            result_dicts_per_monitor[m] = OrderedDict()
        for m in self.monitors:
            result_dict = m.monitor_epoch()
            if result_dict is not None:
                result_dicts_per_monitor[m].update(result_dict)
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
        row_dict = OrderedDict()
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
            input_vars = np_to_var(inputs, pin_memory=globals.get('pin_memory'))
            target_vars = np_to_var(targets, pin_memory=globals.get('pin_memory'))
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
            input_vars = np_to_var(inputs, pin_memory=globals.get('pin_memory'))
            target_vars = np_to_var(targets, pin_memory=globals.get('pin_memory'))
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
                input_vars = np_to_var(inputs, pin_memory=globals.get('pin_memory'))
                target_vars = np_to_var(targets, pin_memory=globals.get('pin_memory'))
                outputs = model(input_vars)
                val_loss += self.loss_function(outputs, target_vars)
                pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target_vars.view_as(pred)).sum().item()

            val_loss /= len(data.X)
        return correct / len(data.X)

    fieldnames = ['exp_name', 'subject', 'generation', 'param_name', 'param_value']

    def write_to_csv(self, csv_file, stats, generation):
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            if self.subject_id == 'all':
                subject = ','.join(str(x) for x in self.current_chosen_population_sample)
            else:
                subject = str(self.subject_id)
            for key, value in stats.items():
                writer.writerow({'exp_name': self.exp_name, 'subject': subject,
                                 'generation': str(generation), 'param_name': key, 'param_value': value})

    def garbage_time(self):
        model = target_model('deep')
        while 1:
            print('GARBAGE TIME GARBAGE TIME GARBAGE TIME')
            self.evaluate_model(model)

    def print_to_evolution_file(self, evolution_file, models, generation):
        global text_file
        with open(evolution_file, "a") as text_file_local:
            text_file = text_file_local
            print('Architectures for Subject %s, Generation %d\n' % (str(self.subject_id), generation), file=text_file)
            for model in models:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
                finalized_model = finalize_model(model['model'])
                print_model = finalized_model.to(device)
                summary(print_model, (globals.get('eeg_chans'), globals.get('input_time_len'), 1), file=text_file)




