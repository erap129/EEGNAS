from models_generation import random_model, finalize_model, target_model,\
    breed_layers
from braindecode.torch_ext.util import np_to_var
from braindecode.experiments.loggers import Printer
import models_generation
import logging
import torch.optim as optim
from copy import deepcopy
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
    def __init__(self, iterator, exp_folder, exp_name, loss_function,
                 train_set, val_set, test_set, stop_criterion, monitors,
                 config, subject_id, fieldnames, model_from_file=None):
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

    def run_target_model(self, csv_file):
        globals.set('max_epochs', globals.get('final_max_epochs'))
        globals.set('max_increase_epochs', globals.get('final_max_increase_epochs'))
        if self.model_from_file is not None:
            if torch.cuda.is_available():
                model = torch.load(self.model_from_file)
            else:
                model = torch.load(self.model_from_file, map_location='cpu')
            if globals.get('cropping'):
                conv_classifier = list(model._modules.items())[-3][1]
                model.conv_classifier =  nn.Conv2d(conv_classifier.in_channels, conv_classifier.out_channels,
                          (globals.get('final_conv_size'),
                           conv_classifier.kernel_size[1]), stride=conv_classifier.stride)
                model.cuda()
                dummy_input = np_to_var(self.datasets['train'].X[:1, :, :, None])
                if globals.get('cuda'):
                    dummy_input = dummy_input.cuda()
                out = model(dummy_input)
                n_preds_per_input = out.cpu().data.numpy().shape[2]
                globals.set('n_preds_per_input', n_preds_per_input)
                self.iterator = CropsFromTrialsIterator(batch_size=globals.get('batch_size'),
                                                   input_time_length=globals.get('input_time_len'),
                                                   n_preds_per_input=globals.get('n_preds_per_input'))
        else:
            model = target_model()
        final_time, res_test, res_val, res_train, model, model_state, num_epochs =\
            self.evaluate_model(model, final_evaluation=True)
        stats = {'train_acc': str(res_train), 'val_acc': str(res_val),
                 'test_acc': str(res_test), 'train_time': str(final_time)}
        self.write_to_csv(csv_file, stats, generation=1)

    def sample_subjects(self):
        self.current_chosen_population_sample = random.sample(
            range(1, globals.get('num_subjects') + 1),
            globals.get('cross_subject_sampling_rate'))

    @staticmethod
    def check_age(model):
        return globals.get('use_aging') and\
            random.random() < 1 - 1 / (model['age'] + 1)

    def one_strategy(self, weighted_population, generation):
        self.current_chosen_population_sample = [self.subject_id]
        for i, pop in enumerate(weighted_population):
            if NaiveNAS.check_age(pop):
                continue
            final_time, res_test, res_val, res_train, model, model_state, num_epochs = \
                self.evaluate_model(pop['model'], pop['model_state'])
            weighted_population[i]['train_acc'] = res_train
            weighted_population[i]['val_acc'] = res_val
            weighted_population[i]['test_acc'] = res_test
            weighted_population[i]['model_state'] = model_state
            weighted_population[i]['finalized_model'] = model
            weighted_population[i]['train_time'] = final_time
            weighted_population[i]['num_epochs'] = num_epochs
            print('trained model %d in generation %d' % (i + 1, generation))

    def all_strategy(self, weighted_population, generation):
        if globals.get('cross_subject_sampling_method') == 'generation':
            self.sample_subjects()
        for i, pop in enumerate(weighted_population):
            if NaiveNAS.check_age(pop):
                continue
            if globals.get('cross_subject_sampling_method') == 'model':
                self.sample_subjects()
            for key in ['train_acc', 'val_acc', 'test_acc', 'train_time', 'num_epochs']:
                weighted_population[i][key] = 0
            for subject in self.current_chosen_population_sample:
                final_time, res_test, res_val, res_train, model, model_state, num_epochs = \
                    self.evaluate_model(pop['model'], pop['model_state'], subject=subject)
                weighted_population[i]['%d_train_acc' % subject] = res_train
                weighted_population[i]['train_acc'] += res_train
                weighted_population[i]['%d_val_acc' % subject] = res_train
                weighted_population[i]['val_acc'] += res_val
                weighted_population[i]['%d_test_acc' % subject] = res_test
                weighted_population[i]['test_acc'] += res_test
                weighted_population[i]['%d_train_time' % subject] = final_time
                weighted_population[i]['train_time'] += final_time
                weighted_population[i]['%d_model_state' % subject] = model_state
                weighted_population[i]['%d_num_epochs' % subject] = num_epochs
                weighted_population[i]['num_epochs'] += num_epochs
                print('trained model %d in subject %d in generation %d' % (i + 1, subject, generation))
            weighted_population[i]['finalized_model'] = model
            for key in ['train_acc', 'val_acc', 'test_acc', 'train_time']:
                weighted_population[i][key] /= globals.get('cross_subject_sampling_rate')

    @staticmethod
    def get_average_param(models, layer_type, attribute):
        attr_count = 0
        count = 0
        for model in models:
            for layer in model:
                if isinstance(layer, layer_type):
                    attr_count += getattr(layer, attribute)
                    count += 1
        return attr_count / count

    def calculate_population_similarity(self, layer_collections, evolution_file, sim_count):
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

    def calculate_stats(self, weighted_population, evolution_file):
        stats = {}
        params = ['train_acc', 'val_acc', 'test_acc', 'train_time', 'num_epochs']
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
            stats[stat] = NaiveNAS.get_average_param([pop['model'] for pop in weighted_population],
                                                                 layer_stats[stat][0], layer_stats[stat][1])
        stats['average_age'] = np.mean([sample['age'] for sample in weighted_population])
        stats['similarity_measure'] = self.calculate_population_similarity(
            [pop['model'] for pop in weighted_population], evolution_file, sim_count=globals.get('sim_count'))
        stats['mutation_rate'] = self.mutation_rate
        for layer_type in [models_generation.DropoutLayer, models_generation.ActivationLayer, models_generation.ConvLayer,
                           models_generation.IdentityLayer, models_generation.BatchNormLayer, models_generation.PoolingLayer]:
            stats['%s_count' % layer_type.__name__] = \
                NaiveNAS.count_layer_type_in_pop([pop['model'] for pop in weighted_population], layer_type)
        return stats

    def add_final_stats(self, stats, weighted_population):
        if globals.get('cross_subject'):
            self.current_chosen_population_sample = range(1, globals.get('num_subjects') + 1)
        for subject in self.current_chosen_population_sample:
            _, res_test, res_val, res_train, _, _, num_epochs = self.evaluate_model(
                weighted_population[0]['model'], final_evaluation=True, subject=subject)
            stats['%d_final_train_acc' % subject] = res_train
            stats['%d_final_val_acc' % subject] = res_val
            stats['%d_final_test_acc' % subject] = res_test
            stats['%d_final_epoch_num' % subject] = num_epochs

    def evolution(self, csv_file, evolution_file, breeding_method, model_init, model_init_configuration, evo_strategy):
        pop_size = globals.get('pop_size')
        num_generations = globals.get('num_generations')
        evolution_results = pd.DataFrame()
        weighted_population = []
        for i in range(pop_size):  # generate pop_size random models
            new_rand_model = model_init(globals.get(model_init_configuration))
            NaiveNAS.hash_model(new_rand_model, self.models_set, self.genome_set)
            weighted_population.append({'model': new_rand_model, 'model_state': None, 'age': 0})

        for generation in range(num_generations):
            evo_strategy(weighted_population, generation)
            weighted_population = sorted(weighted_population, key=lambda x: x['val_acc'], reverse=True)
            stats = self.calculate_stats(weighted_population, evolution_file)
            if generation < num_generations - 1:
                for index, model in enumerate(weighted_population):
                    if random.uniform(0, 1) < (index / pop_size):
                        self.remove_from_models_hash(model['model'], self.models_set, self.genome_set)
                        del weighted_population[index]
                    else:
                        model['age'] += 1

                children = []
                while len(weighted_population) + len(children) < pop_size:
                    breeders = random.sample(range(len(weighted_population)), 2)
                    if globals.get('cross_subject'):
                        model_state_str = '%d_model_state' % random.sample(self.current_chosen_population_sample, 1)[0]
                    else:
                        model_state_str = 'model_state'
                    new_model, new_model_state = breeding_method(mutation_rate=self.mutation_rate,
                                                                first_model=weighted_population[breeders[0]]['model'],
                                             second_model=weighted_population[breeders[1]]['model'],
                                                first_model_state=weighted_population[breeders[0]][model_state_str],
                                                second_model_state=weighted_population[breeders[1]][model_state_str])
                    NaiveNAS.hash_model(new_model, self.models_set, self.genome_set)
                    children.append({'model': new_model, 'model_state': new_model_state, 'age': 0})
                weighted_population.extend(children)
                if len(self.models_set) < globals.get('pop_size') * globals.get('unique_model_threshold'):
                    self.mutation_rate *= globals.get('mutation_rate_change_factor')
                else:
                    self.mutation_rate = globals.get('mutation_rate')
            else:  # last generation
                try:
                    save_model = weighted_population[0]['finalized_model'].to("cpu")
                    torch.save(save_model, "%s/best_model_" % self.exp_folder
                               + '_'.join(str(x) for x in self.current_chosen_population_sample) + ".th")
                except Exception as e:
                    print("failed to save model")
                self.add_final_stats(stats, weighted_population)

            self.write_to_csv(csv_file, {k: str(v) for k, v in stats.items()}, generation + 1)
            self.print_to_evolution_file(evolution_file, weighted_population[:3], generation)
        return evolution_results

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

    def evaluate_model(self, model, state=None, subject=None, final_evaluation=False):
        if final_evaluation:
            self.stop_criterion = Or([MaxEpochs(globals.get('final_max_epochs')),
                                 NoDecrease('valid_misclass', globals.get('final_max_increase_epochs'))])
        if subject is not None and globals.get('cross_subject'):
            single_subj_dataset = OrderedDict((('train', self.datasets['train'][subject - 1]),
                                               ('valid', self.datasets['valid'][subject - 1]),
                                               ('test', self.datasets['test'][subject - 1])))
        else:
            single_subj_dataset = self.datasets
        self.epochs_df = pd.DataFrame()
        if globals.get('do_early_stop'):
            self.rememberer = RememberBest(globals.get('remember_best_column'))
        if globals.get('exp_type') == 'from_file':
            finalized_model = models_generation.MyModel(model=model)
        else:
            finalized_model = finalize_model(model)
        if globals.get('cropping'):
            to_dense_prediction_model(finalized_model.model)
        if globals.get('inherit_weights') and state is not None:
            finalized_model.model.load_state_dict(state)
        self.optimizer = optim.Adam(finalized_model.model.parameters())
        pytorch_model = finalized_model.model
        if self.cuda:
            assert torch.cuda.is_available(), "Cuda not available"
            pytorch_model.cuda()
            if torch.cuda.device_count() > 1 and globals.get('parallel_gpu'):
                pytorch_model = nn.DataParallel(pytorch_model.cuda(), device_ids=[0,1,2,3])
        self.monitor_epoch(single_subj_dataset, pytorch_model)
        if globals.get('log_epochs'):
            self.log_epoch()
        if globals.get('remember_best'):
            self.rememberer.remember_epoch(self.epochs_df, pytorch_model, self.optimizer)
        self.iterator.reset_rng()
        start = time.time()
        num_epochs = self.run_until_stop(pytorch_model, single_subj_dataset)
        self.setup_after_stop_training(pytorch_model, final_evaluation)
        if final_evaluation:
            loss_to_reach = float(self.epochs_df['train_loss'].iloc[-1])
            datasets = single_subj_dataset
            datasets['train'] = concatenate_sets([datasets['train'], datasets['valid']])
            num_epochs += self.run_until_stop(pytorch_model, datasets)
            if float(self.epochs_df['valid_loss'].iloc[-1]) > loss_to_reach:
                self.rememberer.reset_to_best_model(self.epochs_df, pytorch_model, self.optimizer)
        end = time.time()
        res_test = 1 - self.epochs_df.iloc[-1]['test_misclass']
        res_val = 1 - self.epochs_df.iloc[-1]['valid_misclass']
        res_train = 1 - self.epochs_df.iloc[-1]['train_misclass']
        final_time = end-start
        return final_time, res_test, res_val, res_train, pytorch_model, self.rememberer.model_state_dict, num_epochs

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
                input_vars = input_vars.cuda()
                target_vars = target_vars.cuda()
            self.optimizer.zero_grad()
            outputs = model(input_vars)
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
                summary(print_model, (globals.get('eeg_chans'), globals.get('input_time_len'), 1), file=text_file)




