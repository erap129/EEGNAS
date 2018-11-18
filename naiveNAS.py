from models_generation import random_model, finalize_model, mutate_net, target_model, set_target_model_filters,\
    genetic_filter_experiment_model, breed
from braindecode.torch_ext.util import np_to_var
from braindecode.experiments.loggers import Printer
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from collections import OrderedDict
import numpy as np
import time
import torch
from keras_models import convert_to_dilated
import os
import globals
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import random
WARNING = '\033[93m'
ENDC = '\033[0m'


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
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

class NaiveNAS:
    def __init__(self, iterator, n_classes, input_time_len, n_chans,
                 train_set, val_set, test_set, stop_criterion, monitors,
                 config, subject_id, cropping=False):
        self.iterator = iterator
        self.n_classes = n_classes
        self.n_chans = n_chans
        self.input_time_len = input_time_len
        self.subject_id = subject_id
        self.cropping = cropping
        self.config = config
        self.optimizer = None
        self.datasets = OrderedDict(
            (('train', train_set), ('valid', val_set), ('test', test_set))
        )
        self.stop_criterion = stop_criterion
        self.monitors = monitors
        if globals.config['DEFAULT'].getboolean('do_early_stop'):
            self.rememberer = RememberBest(globals.config['DEFAULT']['remember_best_column'])
        self.loggers = [Printer()]
        self.epochs_df = pd.DataFrame()

    def evolution_filters(self):
        configuration = self.config['evolution']
        pop_size = configuration.getint('pop_size')
        num_generations = configuration.getint('num_generations')
        mutation_rate = configuration.getfloat('mutation_rate')
        results_dict = {'subject': [], 'generation': [], 'val_acc': []}

        weighted_population = []
        for i in range(pop_size):  # generate pop_size random models
            weighted_population.append((genetic_filter_experiment_model(num_blocks=configuration.getint('num_conv_blocks')), None))

        for generation in range(num_generations):
            for i, (pop, eval) in enumerate(weighted_population):
                final_time, _, res_val, _, _ = self.evaluate_model(pop)
                weighted_population[i][1] = res_val
            weighted_population = sorted(weighted_population, key=lambda x: x[1])
            mean_fitness = np.mean([weight for (model, weight) in weighted_population])
            print('fittest individual in generation %d has fitness %.3f'.format(generation, weighted_population[0][1]))
            print('mean fitness of population is %.3f'.format(mean_fitness))
            results_dict['subject'].append(self.subject_id)
            results_dict['generation'].append(generation + 1)
            results_dict['val_acc'].append(mean_fitness)


            for index, model, res_val in enumerate(weighted_population):
                if random.uniform(0,1) < (index / pop_size):
                    del weighted_population[index]  # kill models according to their performance

            while len(weighted_population) < pop_size:  # breed with random parents until population reaches pop_size
                breeders = random.sample(range(pop_size), 2)
                new_model = breed(first_model=weighted_population[breeders[0]][0],
                                second_model=weighted_population[breeders[0]][0], mutation_rate=mutation_rate)
                weighted_population.append((new_model, None))
        return results_dict

    def evaluate_model(self, model):
        finalized_model = finalize_model(model, naive_nas=self)
        if self.cropping:
            finalized_model.model = convert_to_dilated(model.model)
        self.optimizer = optim.Adam(model.parameters())
        start = time.time()
        while not self.stop_criterion.should_stop(self.epochs_df):
            self.run_one_epoch(self.datasets, finalized_model)

        res_test = self.eval_pytorch(finalized_model, self.test_set)
        res_val = self.eval_pytorch(finalized_model, self.val_set)
        res_train = self.eval_pytorch(finalized_model, self.train_set)
        end = time.time()
        final_time = end-start
        return final_time, res_test, res_val, res_train, finalized_model.model

    def run_one_epoch(self, datasets, model):
        model.train()
        batch_generator = self.iterator.get_batches(datasets['train'], shuffle=True)
        for inputs, targets in batch_generator:
            input_vars = np_to_var(inputs, pin_memory=self.config['DEFAULT'].getboolean('pin_memory'))
            target_vars = np_to_var(targets, pin_memory=self.config['DEFAULT'].getboolean('pin_memory'))
            if(globals.config['DEFAULT'].getboolean('cuda')):
                input_vars = input_vars.cuda()
                target_vars = target_vars.cuda()
            self.optimizer.zero_grad()
            outputs = model(input_vars)
            loss = F.nll_loss(outputs, target_vars)
            loss.backward()
            self.optimizer.step()
        self.monitor_epoch(datasets)
        self.log_epoch()

    def monitor_epoch(self, datasets):
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
                preds, loss = self.eval_on_batch(batch[0], batch[1])
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




