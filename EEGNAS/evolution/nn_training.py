import itertools
from copy import deepcopy

import torch.nn.functional as F
from braindecode.datautil.iterators import get_balanced_batches, BalancedBatchSizeIterator
from braindecode.datautil.splitters import concatenate_sets
from braindecode.torch_ext.util import np_to_var
from braindecode.experiments.loggers import Printer
import logging
import torch.optim as optim

from EEGNAS.data.netflow.netflow_data_utils import turn_dataset_to_timefreq
from EEGNAS.model_generation.custom_modules import AveragingEnsemble
from EEGNAS.utilities.misc import RememberBest, get_oper_by_loss_function
import pandas as pd
from collections import OrderedDict, defaultdict
import numpy as np
from EEGNAS.data_preprocessing import get_pure_cross_subject
import time
import torch
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or, ColumnBelow
from EEGNAS import global_vars
from torch import nn
from EEGNAS.utilities.monitors import NoIncreaseDecrease
import pdb
log = logging.getLogger(__name__)
model_train_times = []


class TimeFrequencyBatchIterator(BalancedBatchSizeIterator):
    def __init__(self, batch_size):
        super(TimeFrequencyBatchIterator, self).__init__(batch_size=batch_size)

    def get_batches(self, dataset, shuffle):
        n_trials = dataset.X.shape[0]
        batches = get_balanced_batches(n_trials,
                                       batch_size=self.batch_size,
                                       rng=self.rng,
                                       shuffle=shuffle)
        for batch_inds in batches:
            batch_X = dataset.X[batch_inds]
            batch_y = dataset.y[batch_inds]
            batch_X = turn_dataset_to_timefreq(batch_X)
            # add empty fourth dimension if necessary
            if batch_X.ndim == 3:
                batch_X = batch_X[:, :, :, None]
            yield (batch_X, batch_y)


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
        self.cuda = global_vars.get('cuda')
        self.loggers = [Printer()]
        self.epochs_df = None

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

    def train_ensemble(self, models, dataset, final_evaluation=False):
        # for i in range(len(models)):
        #     models[i] = self.train_model(models[i], dataset, final_evaluation=final_evaluation)
        models = [nn.Sequential(*list(model.children())[:global_vars.get('num_layers') + 1]) for model in models] # remove the final softmax layer from each model
        avg_model = AveragingEnsemble(models)
        return self.train_model(avg_model, dataset, final_evaluation=final_evaluation)

    def train_model(self, model, dataset, state=None, final_evaluation=False, ensemble=False):
        if self.cuda:
            torch.cuda.empty_cache()
        if final_evaluation:
            self.stop_criterion = Or([MaxEpochs(global_vars.get('final_max_epochs')),
                                      NoIncreaseDecrease(f'valid_{global_vars.get("nn_objective")}',
                                                         global_vars.get('final_max_increase_epochs'),
                                                         oper=get_oper_by_loss_function(self.loss_function))])
        if global_vars.get('cropping'):
            self.set_cropping_for_model(model)
        self.epochs_df = pd.DataFrame()
        if global_vars.get('do_early_stop') or global_vars.get('remember_best'):
            self.rememberer = RememberBest(f"valid_{global_vars.get('nn_objective')}",
                                           oper=get_oper_by_loss_function(self.loss_function, equals=True))
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
        self.monitor_epoch(dataset, model)
        if global_vars.get('log_epochs'):
            self.log_epoch()
        if global_vars.get('remember_best'):
            self.rememberer.remember_epoch(self.epochs_df, model, self.optimizer)
        self.iterator.reset_rng()
        start = time.time()
        num_epochs = self.run_until_stop(model, dataset)
        self.setup_after_stop_training(model, final_evaluation)
        if final_evaluation:
            dataset_train_backup = deepcopy(dataset['train'])
            if ensemble:
                self.run_one_epoch(dataset, model)
                self.rememberer.remember_epoch(self.epochs_df, model, self.optimizer, force=ensemble)
                num_epochs += 1
            else:
                dataset['train'] = concatenate_sets([dataset['train'], dataset['valid']])
            num_epochs += self.run_until_stop(model, dataset)
            self.rememberer.reset_to_best_model(self.epochs_df, model, self.optimizer)
            dataset['train'] = dataset_train_backup
        end = time.time()
        self.final_time = end - start
        self.num_epochs = num_epochs
        return model

    def train_and_evaluate_model(self, model, dataset, state=None, final_evaluation=False, ensemble=False):
        if type(model) == list:
            model = AveragingEnsemble(model)
            if self.cuda:
                for mod in model.models:
                    mod.cuda()
        self.train_model(model, dataset, state, final_evaluation, ensemble)
        evaluations = {}
        for evaluation_metric in global_vars.get('evaluation_metrics'):
            evaluations[evaluation_metric] = {'train': self.epochs_df.iloc[-1][f"train_{evaluation_metric}"],
                                              'valid': self.epochs_df.iloc[-1][f"valid_{evaluation_metric}"],
                                              'test': self.epochs_df.iloc[-1][f"test_{evaluation_metric}"]}
        if self.cuda:
            torch.cuda.empty_cache()
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
        return self.final_time, evaluations, model, self.rememberer.model_state_dict, self.num_epochs

    def setup_after_stop_training(self, model, final_evaluation):
        self.rememberer.reset_to_best_model(self.epochs_df, model, self.optimizer)
        if final_evaluation:
            loss_to_reach = float(self.epochs_df['train_loss'].iloc[-1])
            self.stop_criterion = Or(stop_criteria=[
                MaxEpochs(max_epochs=global_vars.get('final_max_epochs')),
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
            if global_vars.get('evaluator') == 'rnn':
                input_vars = input_vars.squeeze(dim=3)
                input_vars = input_vars.permute(0, 2, 1)
            outputs = model(input_vars)
            if self.loss_function == F.mse_loss:
                target_vars = target_vars.float()
            loss = self.loss_function(outputs.squeeze(), target_vars)
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
                input_vars = batch[0]
                target_vars = batch[1]
                preds, loss = self.eval_on_batch(input_vars, target_vars, model)

                all_preds.append(preds)
                all_losses.append(loss)
                all_batch_sizes.append(len(input_vars))
                all_targets.append(target_vars)

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
            if global_vars.get('evaluator') == 'rnn':
                input_vars = input_vars.squeeze(dim=3)
                input_vars = input_vars.permute(0, 2, 1)
            outputs = model(input_vars)
            if self.loss_function == F.mse_loss:
                target_vars = target_vars.float()
            loss = self.loss_function(outputs.squeeze(), target_vars)
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
