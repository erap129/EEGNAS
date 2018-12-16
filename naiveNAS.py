from models_generation import random_model, finalize_model, target_model,\
    genetic_filter_experiment_model, breed_filters, breed_layers
from braindecode.torch_ext.util import np_to_var
from braindecode.experiments.loggers import Printer
import models_generation
import logging
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import pandas as pd
from collections import OrderedDict
import numpy as np
import time
import torch
from keras_models import convert_to_dilated
import os
import globals
import csv
import pickle
from torchsummary import summary

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import random
WARNING = '\033[93m'
ENDC = '\033[0m'
log = logging.getLogger(__name__)


class RememberBest(object):
    """
    Class to remember and restore
    the parameters of the model and the parameters of the
    optimizer at the epoch with the best performance.
    Parameters
    ----------
    column_name: str
        The lowest value in this column should indicate the epoch with the
        best performance (e.g. misclass might make sense).

    Attributes
    ----------
    best_epoch: int
        Index of best epoch
    """

    def __init__(self, column_name):
        self.column_name = column_name
        self.best_epoch = 0
        self.lowest_val = float('inf')
        self.model_state_dict = None
        self.optimizer_state_dict = None

    def remember_epoch(self, epochs_df, model, optimizer):
        """
        Remember this epoch: Remember parameter values in case this epoch
        has the best performance so far.

        Parameters
        ----------
        epochs_df: `pandas.Dataframe`
            Dataframe containing the column `column_name` with which performance
            is evaluated.
        model: `torch.nn.Module`
        optimizer: `torch.optim.Optimizer`
        """
        i_epoch = len(epochs_df) - 1
        current_val = float(epochs_df[self.column_name].iloc[-1])
        if current_val <= self.lowest_val:
            self.best_epoch = i_epoch
            self.lowest_val = current_val
            self.model_state_dict = deepcopy(model.state_dict())
            self.optimizer_state_dict = deepcopy(optimizer.state_dict())
            log.info("New best {:s}: {:5f}".format(self.column_name,
                                                   current_val))
            log.info("")

    def reset_to_best_model(self, epochs_df, model, optimizer):
        """
        Reset parameters to parameters at best epoch and remove rows
        after best epoch from epochs dataframe.

        Modifies parameters of model and optimizer, changes epochs_df in-place.

        Parameters
        ----------
        epochs_df: `pandas.Dataframe`
        model: `torch.nn.Module`
        optimizer: `torch.optim.Optimizer`
        """
        # Remove epochs past the best one from epochs dataframe
        epochs_df.drop(range(self.best_epoch + 1, len(epochs_df)), inplace=True)
        model.load_state_dict(self.model_state_dict)
        optimizer.load_state_dict(self.optimizer_state_dict)


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def delete_from_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

class NaiveNAS:
    def __init__(self, iterator, n_classes, input_time_len, n_chans, loss_function,
                 train_set, val_set, test_set, stop_criterion, monitors,
                 config, subject_id, fieldnames, cropping=False):
        self.iterator = iterator
        self.n_classes = n_classes
        self.n_chans = n_chans
        self.input_time_len = input_time_len
        self.subject_id = subject_id
        self.cropping = cropping
        self.config = config
        self.loss_function = loss_function
        self.optimizer = None
        self.datasets = OrderedDict(
            (('train', train_set), ('valid', val_set), ('test', test_set))
        )
        self.rememberer = None
        self.loggers = [Printer()]
        self.stop_criterion = stop_criterion
        self.monitors = monitors
        self.cuda = globals.config['DEFAULT']['cuda']
        self.loggers = [Printer()]
        self.epochs_df = None
        self.fieldnames = fieldnames
        self.models_set = []
        self.genome_set = []

    def run_target_model(self, csv_file):
        model = target_model()
        final_time, res_test, res_val, res_train, model, model_state = self.evaluate_model(model)
        self.write_to_csv(csv_file, str(self.subject_id), '1',
                          str(res_train), str(res_val), str(res_test), str(final_time))

    def one_strategy(self, weighted_population, generation):
        for i, pop in enumerate(weighted_population):
            final_time, res_test, res_val, res_train, model, model_state = \
                self.evaluate_model(pop['model'], pop['model_state'])
            weighted_population[i]['res_train'] = res_train
            weighted_population[i]['res_val'] = res_val
            weighted_population[i]['res_test'] = res_test
            weighted_population[i]['model_state'] = model_state
            weighted_population[i]['finalized_model'] = model
            weighted_population[i]['train_time'] = final_time
            print('trained model %d in generation %d' % (i + 1, generation))

    def all_strategy(self, weighted_population, generation):
        for i, pop in enumerate(weighted_population):
            for key in ['res_train', 'res_val', 'res_test', 'train_time']:
                weighted_population[i][key] = 0
            for subject in range(1, 10):
                final_time, res_test, res_val, res_train, model, model_state = \
                    self.evaluate_model(pop['model'], pop['model_state'], subject=subject)
                weighted_population[i]['%d_res_train' % subject] = res_train
                weighted_population[i]['res_train'] += res_train
                weighted_population[i]['%d_res_val' % subject] = res_train
                weighted_population[i]['res_val'] += res_val
                weighted_population[i]['%d_res_test' % subject] = res_test
                weighted_population[i]['res_test'] += res_test
                weighted_population[i]['%d_train_time' % subject] = final_time
                weighted_population[i]['train_time'] += final_time
                weighted_population[i]['%d_model_state' % subject] = model_state
                weighted_population[i]['finalized_model'] = model
                print('trained model %d in subject %d in generation %d' % (i + 1, subject, generation))
            for key in ['res_train', 'res_val', 'res_test', 'train_time']:
                weighted_population[i][key] /= globals.config['DEFAULT']['num_subjects']

    def get_average_param(models, layer_type, attribute):
        attr_count = 0
        count = 0
        for model in models:
            for layer in model:
                if isinstance(layer, layer_type):
                    attr_count += getattr(layer, attribute)
                    count += 1
        return attr_count / count

    def calculate_stats(self, weighted_population, generation):
        stats = {}
        stats['subject'] = self.subject_id
        stats['generation'] = generation + 1
        stats['train_acc'] = np.mean([sample['res_train'] for sample in weighted_population])
        stats['val_acc'] = np.mean([sample['res_val'] for sample in weighted_population])
        stats['test_acc'] = np.mean([sample['res_test'] for sample in weighted_population])
        stats['train_time'] = np.mean([sample['train_time'] for sample in weighted_population])
        if(self.subject_id == 'all'):
            for subject in range(1, globals.config['DEFAULT']['num_subjects'] + 1):
                stats['%d_train_acc' % subject] = np.mean([sample['%d_res_train' % subject] for sample in weighted_population])
                stats['%d_val_acc' % subject] = np.mean([sample['%d_res_val' % subject] for sample in weighted_population])
                stats['%d_test_acc' % subject] = np.mean([sample['%d_res_test' % subject] for sample in weighted_population])
                stats['%d_train_time' % subject] = np.mean([sample['%d_train_time' % subject] for sample in weighted_population])
        stats['unique_models'] = len(self.models_set)
        stats['unique_genomes'] = len(self.genome_set)
        stats['average_conv_width'] = NaiveNAS.get_average_param([pop['model'] for pop in weighted_population], models_generation.ConvLayer,
                                                                 'kernel_eeg_chan')
        stats['average_conv_height'] = NaiveNAS.get_average_param([pop['model'] for pop in weighted_population],
                                                                 models_generation.ConvLayer, 'kernel_time')
        stats['average_conv_filters'] = NaiveNAS.get_average_param([pop['model'] for pop in weighted_population],
                                                                 models_generation.ConvLayer, 'filter_num')
        stats['average_pool_width'] = NaiveNAS.get_average_param([pop['model'] for pop in weighted_population],
                                                                   models_generation.PoolingLayer, 'pool_time')
        stats['average_pool_stride'] = NaiveNAS.get_average_param([pop['model'] for pop in weighted_population],
                                                                   models_generation.PoolingLayer, 'stride_time')
        for layer_type in [models_generation.DropoutLayer, models_generation.ActivationLayer, models_generation.ConvLayer,
                           models_generation.IdentityLayer, models_generation.BatchNormLayer, models_generation.PoolingLayer]:
            stats['%s_count' % layer_type.__name__] = \
                NaiveNAS.count_layer_type_in_pop([pop['model'] for pop in weighted_population], layer_type)
        return stats

    def evolution(self, csv_file, evolution_file, breeding_method, model_init, model_init_configuration, evo_strategy):
        configuration = self.config['evolution']
        pop_size = configuration['pop_size']
        num_generations = configuration['num_generations']
        evolution_results = pd.DataFrame()
        weighted_population = []
        for i in range(pop_size):  # generate pop_size random models
            new_rand_model = model_init(configuration[model_init_configuration])
            NaiveNAS.hash_model(new_rand_model, self.models_set, self.genome_set)
            weighted_population.append({'model': new_rand_model, 'model_state': None})

        for generation in range(num_generations):
            evo_strategy(weighted_population, generation)
            weighted_population = sorted(weighted_population, key=lambda x: x['res_val'], reverse=True)
            stats = self.calculate_stats(weighted_population, generation)
            print('fittest individual in generation %d has validation fitness %.3f' % (
                generation, weighted_population[0]['res_val']))

            self.write_to_csv(csv_file, {k: str(v) for k, v in stats.items()})
            self.print_to_evolution_file(evolution_file, weighted_population[:3], generation)

            for index, _ in enumerate(weighted_population):
                if random.uniform(0, 1) < (index / pop_size):
                    self.remove_from_models_hash(weighted_population[index]['model'], self.models_set, self.genome_set)
                    del weighted_population[index]

            while len(weighted_population) < pop_size:  # breed with random parents until population reaches pop_size
                breeders = random.sample(range(len(weighted_population)), 2)
                new_model, new_model_state = breeding_method(first_model=weighted_population[breeders[0]]['model'],
                                         second_model=weighted_population[breeders[1]]['model'],
                                            first_model_state=weighted_population[breeders[0]]['model_state'],
                                            second_model_state=weighted_population[breeders[1]]['model_state'])
                NaiveNAS.hash_model(new_model, self.models_set, self.genome_set)
                weighted_population.append({'model': new_model, 'model_state': new_model_state})
        return evolution_results

    def evolution_filters(self, csv_file, evolution_file):
        return self.evolution(csv_file, evolution_file, breed_filters, genetic_filter_experiment_model,
                              'num_conv_blocks', self.one_strategy)

    def evolution_layers(self, csv_file, evolution_file):
        return self.evolution(csv_file, evolution_file, breed_layers, random_model, 'num_layers', self.one_strategy)

    def evolution_layers_all(self, csv_file, evolution_file):
        return self.evolution(csv_file, evolution_file, breed_layers, random_model, 'num_layers', self.all_strategy)

    @staticmethod
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

    @staticmethod
    def hash_model(model, model_set, genome_set):
        if model not in model_set:
            model_set.append(model)
        for layer in model:
            if layer not in genome_set:
                genome_set.append(layer)

    def evaluate_model(self, model, state=None, subject=None):
        if subject is not None:
            single_subj_dataset = OrderedDict((('train', self.datasets['train'][subject - 1]),
                                               ('valid', self.datasets['valid'][subject - 1]),
                                               ('test', self.datasets['test'][subject - 1])))
        self.epochs_df = pd.DataFrame()
        if globals.config['DEFAULT']['do_early_stop']:
            self.rememberer = RememberBest(globals.config['DEFAULT']['remember_best_column'])
        finalized_model = finalize_model(model)
        if state is not None and globals.config['evolution']['inherit_non_breeding_weights']:
            finalized_model.model.load_state_dict(state)
        if self.cropping:
            finalized_model.model = convert_to_dilated(model.model)
        self.optimizer = optim.Adam(finalized_model.model.parameters())
        if self.cuda:
            assert torch.cuda.is_available(), "Cuda not available"
            finalized_model.model.cuda()
        if subject is not None:
            self.monitor_epoch(single_subj_dataset, finalized_model.model)
        else:
            self.monitor_epoch(self.datasets, finalized_model.model)
        if globals.config['DEFAULT']['log_epochs']:
            self.log_epoch()
        if globals.config['DEFAULT']['remember_best']:
            self.rememberer.remember_epoch(self.epochs_df, finalized_model.model, self.optimizer)
        start = time.time()
        while not self.stop_criterion.should_stop(self.epochs_df):
            if subject is None:
                self.run_one_epoch(self.datasets, finalized_model.model)
            else:
                self.run_one_epoch(single_subj_dataset, finalized_model.model)
        self.rememberer.reset_to_best_model(self.epochs_df, finalized_model.model, self.optimizer)
        end = time.time()
        res_test = 1 - self.epochs_df.iloc[-1]['test_misclass']
        res_val = 1 - self.epochs_df.iloc[-1]['valid_misclass']
        res_train = 1 - self.epochs_df.iloc[-1]['train_misclass']
        final_time = end-start
        return final_time, res_test, res_val, res_train, finalized_model.model, self.rememberer.model_state_dict

    def run_one_epoch(self, datasets, model):
        model.train()
        batch_generator = self.iterator.get_batches(datasets['train'], shuffle=True)
        for inputs, targets in batch_generator:
            input_vars = np_to_var(inputs, pin_memory=self.config['DEFAULT']['pin_memory'])
            target_vars = np_to_var(targets, pin_memory=self.config['DEFAULT']['pin_memory'])
            if self.cuda:
                input_vars = input_vars.cuda()
                target_vars = target_vars.cuda()
            self.optimizer.zero_grad()
            outputs = model(input_vars)
            loss = F.nll_loss(outputs, target_vars)
            loss.backward()
            self.optimizer.step()
        self.monitor_epoch(datasets, model)
        if globals.config['DEFAULT']['log_epochs']:
            self.log_epoch()
        if globals.config['DEFAULT']['remember_best']:
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
            input_vars = np_to_var(inputs, pin_memory=globals.config['DEFAULT']['pin_memory'])
            target_vars = np_to_var(targets, pin_memory=globals.config['DEFAULT']['pin_memory'])
            if self.cuda:
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
            input_vars = np_to_var(inputs, pin_memory=self.config['DEFAULT']['pin_memory'])
            target_vars = np_to_var(targets, pin_memory=self.config['DEFAULT']['pin_memory'])
            optimizer.zero_grad()
            outputs = model(input_vars)
            loss = F.nll_loss(outputs, target_vars)
            loss.backward()
            optimizer.step()

    def eval_pytorch(self, model, data):
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in self.iterator.get_batches(data, shuffle=False):
                input_vars = np_to_var(inputs, pin_memory=self.config['DEFAULT']['pin_memory'])
                target_vars = np_to_var(targets, pin_memory=self.config['DEFAULT']['pin_memory'])
                outputs = model(input_vars)
                val_loss += F.nll_loss(outputs, target_vars)
                pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target_vars.view_as(pred)).sum().item()

            val_loss /= len(data.X)
        return correct / len(data.X)

    def write_to_csv(self, csv_file, stats):
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(stats)

    @staticmethod
    def count_layer_type_in_pop(models, layer_type):
        count = 0
        for model in models:
            for layer in model:
                if isinstance(layer, layer_type):
                    count += 1
        return count

    def garbage_time(self):
        model = target_model()
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
                print_model = model['finalized_model'].to(device)
                summary(print_model, (22, 1125, 1), file=text_file)




