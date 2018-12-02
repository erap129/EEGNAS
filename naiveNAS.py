from models_generation import random_model, finalize_model, mutate_net, target_model, set_target_model_filters,\
    genetic_filter_experiment_model, breed_filters, breed_layers
from braindecode.torch_ext.util import np_to_var
from braindecode.experiments.loggers import Printer
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
                 config, subject_id, cropping=False):
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
        self.cuda = globals.config['DEFAULT'].getboolean('cuda')
        self.loggers = [Printer()]
        self.epochs_df = None

    def run_target_model(self, csv_file):
        model = target_model()
        final_time, res_test, res_val, res_train, model, model_state = self.evaluate_model(model)
        self.write_to_csv(csv_file, str(self.subject_id), '1',
                          str(res_train), str(res_val), str(res_test), str(final_time))

    def evolution(self, csv_file, evolution_file, breeding_method, model_init, model_init_configuration):
        configuration = self.config['evolution']
        pop_size = configuration.getint('pop_size')
        num_generations = configuration.getint('num_generations')
        mutation_rate = configuration.getfloat('mutation_rate')
        evolution_results = pd.DataFrame()
        weighted_population = []
        for i in range(pop_size):  # generate pop_size random models
            weighted_population.append({'model': model_init(configuration.getint(model_init_configuration)),
                                        'model_state': None})

        for generation in range(num_generations):
            for i, pop in enumerate(weighted_population):
                final_time, res_test, res_val, res_train, model, model_state =\
                    self.evaluate_model(pop['model'], pop['model_state'])
                weighted_population[i]['res_train'] = res_train
                weighted_population[i]['res_val'] = res_val
                weighted_population[i]['res_test'] = res_test
                weighted_population[i]['model_state'] = model_state
                weighted_population[i]['finalized_model'] = model
                weighted_population[i]['train_time'] = final_time
            weighted_population = sorted(weighted_population, key=lambda x: x['res_val'], reverse=True)
            mean_fitness_train = np.mean([sample['res_train'] for sample in weighted_population])
            mean_fitness_val = np.mean([sample['res_val'] for sample in weighted_population])
            mean_fitness_test = np.mean([sample['res_test'] for sample in weighted_population])
            mean_train_time = np.mean([sample['train_time'] for sample in weighted_population])
            print('fittest individual in generation %d has validation fitness %.3f' % (
                generation, weighted_population[0]['res_val']))
            print('mean validation fitness of population is %.3f' % (mean_fitness_val))

            self.write_to_csv(csv_file, str(self.subject_id), str(generation + 1),
                              str(mean_fitness_train), str(mean_fitness_val), str(mean_fitness_test), str(mean_train_time))

            self.print_to_evolution_file(evolution_file, weighted_population[:3], generation)

            for index, _ in enumerate(weighted_population):
                if random.uniform(0, 1) < (index / pop_size):
                    del weighted_population[index]  # kill models according to their performance

            while len(weighted_population) < pop_size:  # breed with random parents until population reaches pop_size
                breeders = random.sample(range(len(weighted_population)), 2)
                new_model = breeding_method(first=weighted_population[breeders[0]]['model'],
                                         second=weighted_population[breeders[1]]['model'], mutation_rate=mutation_rate)
                weighted_population.append({'model': new_model, 'model_state': None})
        return evolution_results

    def evolution_filters(self, csv_file, evolution_file):
        return self.evolution(csv_file, evolution_file, breed_filters, genetic_filter_experiment_model, 'num_conv_blocks')

    def evolution_layers(self, csv_file, evolution_file):
        return self.evolution(csv_file, evolution_file, breed_layers, random_model, 'num_layers')

    def evaluate_model(self, model, state=None):
        self.epochs_df = pd.DataFrame()
        if globals.config['DEFAULT'].getboolean('do_early_stop'):
            self.rememberer = RememberBest(globals.config['DEFAULT']['remember_best_column'])
        finalized_model = finalize_model(model)
        if self.cropping:
            finalized_model.model = convert_to_dilated(model.model)
        self.optimizer = optim.Adam(finalized_model.model.parameters())
        if self.cuda:
            assert torch.cuda.is_available(), "Cuda not available"
            finalized_model.model.cuda()
        self.monitor_epoch(self.datasets, finalized_model.model)
        self.log_epoch()
        if globals.config['DEFAULT'].getboolean('remember_best'):
            self.rememberer.remember_epoch(self.epochs_df, finalized_model.model, self.optimizer)
        start = time.time()
        while not self.stop_criterion.should_stop(self.epochs_df):
            self.run_one_epoch(self.datasets, finalized_model.model)
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
            input_vars = np_to_var(inputs, pin_memory=self.config['DEFAULT'].getboolean('pin_memory'))
            target_vars = np_to_var(targets, pin_memory=self.config['DEFAULT'].getboolean('pin_memory'))
            if self.cuda:
                input_vars = input_vars.cuda()
                target_vars = target_vars.cuda()
            self.optimizer.zero_grad()
            outputs = model(input_vars)
            loss = F.nll_loss(outputs, target_vars)
            loss.backward()
            self.optimizer.step()
        self.monitor_epoch(datasets, model)
        self.log_epoch()
        if globals.config['DEFAULT'].getboolean('remember_best'):
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
            input_vars = np_to_var(inputs, pin_memory=globals.config['DEFAULT'].getboolean('pin_memory'))
            target_vars = np_to_var(targets, pin_memory=globals.config['DEFAULT'].getboolean('pin_memory'))
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
            input_vars = np_to_var(inputs, pin_memory=self.config['DEFAULT'].getboolean('pin_memory'))
            target_vars = np_to_var(targets, pin_memory=self.config['DEFAULT'].getboolean('pin_memory'))
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
                input_vars = np_to_var(inputs, pin_memory=self.config['DEFAULT'].getboolean('pin_memory'))
                target_vars = np_to_var(targets, pin_memory=self.config['DEFAULT'].getboolean('pin_memory'))
                outputs = model(input_vars)
                val_loss += F.nll_loss(outputs, target_vars)
                pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target_vars.view_as(pred)).sum().item()

            val_loss /= len(data.X)
            # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #     val_loss, correct, len(data.X),
            #     100. * correct / len(data.X)))
        return correct / len(data.X)

    def write_to_csv(self, csv_file, subject, gen, train_acc, val_acc, test_acc, train_time):
        with open(csv_file, 'a', newline='') as csvfile:
            fieldnames = ['subject', 'generation', 'train_acc', 'val_acc', 'test_acc', 'train_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'subject': subject, 'generation': gen,
                             'train_acc': train_acc, 'val_acc': val_acc,
                             'test_acc': test_acc, 'train_time': train_time})

    def garbage_time(self):
        model = target_model()
        while 1:
            print('GARBAGE TIME GARBAGE TIME GARBAGE TIME')
            self.evaluate_model(model)

    def print_to_evolution_file(self, evolution_file, models, generation):
        global text_file
        with open(evolution_file, "a") as text_file_local:
            text_file = text_file_local
            print('Architectures for Subject %d, Generation %d\n' % (self.subject_id, generation), file=text_file)
            for model in models:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
                print_model = model['finalized_model'].to(device)
                summary(print_model, (22, 1125, 1), file=text_file)




