import itertools
import pickle
import platform

from data_preprocessing import get_train_val_test
from braindecode.torch_ext.util import np_to_var
from braindecode.experiments.loggers import Printer
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
import globals
import csv
from torch import nn
from utilities.model_summary import summary
from utilities.monitors import NoIncrease
import NASUtils
import pdb
from tensorboardX import SummaryWriter
import random
log = logging.getLogger(__name__)
model_train_times = []


class NN_Trainer:
    def __init__(self, iterator, loss_function, stop_criterion, monitors):
        global model_train_times
        model_train_times = []
        self.iterator = iterator
        self.loss_function = loss_function
        self.optimizer = None
        self.rememberer = None
        self.loggers = [Printer()]
        self.stop_criterion = stop_criterion
        self.monitors = monitors
        self.cuda = globals.get('cuda')
        self.loggers = [Printer()]
        self.epochs_df = None

    def finalized_model_to_dilated(self, model):
        to_dense_prediction_model(model)
        conv_classifier = model.conv_classifier
        model.conv_classifier = nn.Conv2d(conv_classifier.in_channels, conv_classifier.out_channels,
                                          (globals.get('final_conv_size'),
                                           conv_classifier.kernel_size[1]), stride=conv_classifier.stride,
                                          dilation=conv_classifier.dilation)

    def get_n_preds_per_input(self, model):
        dummy_input = self.get_dummy_input()
        if globals.get('cuda'):
            model.cuda()
            dummy_input = dummy_input.cuda()
        out = model(dummy_input)
        n_preds_per_input = out.cpu().data.numpy().shape[2]
        return n_preds_per_input

    def evaluate_model(self, model, dataset, state=None, final_evaluation=False, ensemble=False):
        if self.cuda:
            torch.cuda.empty_cache()
        if final_evaluation:
            self.stop_criterion = Or([MaxEpochs(globals.get('final_max_epochs')),
                                 NoIncrease('valid_accuracy', globals.get('final_max_increase_epochs'))])
        if globals.get('cropping'):
            self.set_cropping_for_model(model)
        self.epochs_df = pd.DataFrame()
        if globals.get('do_early_stop') or globals.get('remember_best'):
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
        self.monitor_epoch(dataset, model)
        if globals.get('log_epochs'):
            self.log_epoch()
        if globals.get('remember_best'):
            self.rememberer.remember_epoch(self.epochs_df, model, self.optimizer)
        self.iterator.reset_rng()
        start = time.time()
        num_epochs = self.run_until_stop(model, dataset)
        self.setup_after_stop_training(model, final_evaluation)
        if final_evaluation:
            loss_to_reach = float(self.epochs_df['train_loss'].iloc[-1])
            if ensemble:
                self.run_one_epoch(dataset, model)
                self.rememberer.remember_epoch(self.epochs_df, model, self.optimizer, force=ensemble)
                num_epochs += 1
            num_epochs += self.run_until_stop(model, dataset)
            if float(self.epochs_df['valid_loss'].iloc[-1]) > loss_to_reach or ensemble:
                self.rememberer.reset_to_best_model(self.epochs_df, model, self.optimizer)
        end = time.time()
        evaluations = {}
        for evaluation_metric in globals.get('evaluation_metrics'):
            evaluations[evaluation_metric] = {'train': self.epochs_df.iloc[-1][f"train_{evaluation_metric}"],
                                              'valid': self.epochs_df.iloc[-1][f"valid_{evaluation_metric}"],
                                              'test': self.epochs_df.iloc[-1][f"test_{evaluation_metric}"]}
        final_time = end-start
        if self.cuda:
            torch.cuda.empty_cache()
        if globals.get('use_tensorboard'):
            if self.current_model_index > 0:
                with SummaryWriter(log_dir=f'{self.exp_folder}/tensorboard/gen_{self.current_generation}_model{self.current_model_index}') as w:
                    w.add_graph(model, self.get_dummy_input())
        if globals.get('delete_finalized_models'):
            if globals.get('grid_as_ensemble'):
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

            if globals.get('ensemble_iterations'):
                raws.update({f'{setname}_raw': list(itertools.chain.from_iterable(all_preds))})
                targets.update({f'{setname}_target': list(itertools.chain.from_iterable(all_targets))})
        row_dict = OrderedDict()
        if globals.get('ensemble_iterations'):
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
