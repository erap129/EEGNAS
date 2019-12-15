import datetime
import os
import shutil
import sys
from copy import deepcopy
import logging
import numpy as np
import torch
from braindecode.datautil.splitters import concatenate_sets
import code, traceback, signal

from EEGNAS import global_vars

log = logging.getLogger(__name__)
import math
import operator
import torch.nn.functional as F


MOABB_DATASETS = ['AlexMI', 'BNCI2014001', 'BNCI2014002', 'BNCI2014004', 'BNCI2015001', 'BNCI2015004', 'Cho2017', 'MunichMI', 'Ofner2017', 'PhysionetMI', 'Schirrmeister2017', 'Shin2017A', 'Weibo2014', 'Zhou2016']

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


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

    def __init__(self, column_name, oper=operator.ge):
        self.column_name = column_name
        self.best_epoch = 0
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.oper = oper
        if self.oper == operator.ge:
            self.best_val = -math.inf
        else:
            self.best_val = math.inf

    def remember_epoch(self, epochs_df, model, optimizer, force=False):
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
        force: `remember this epoch no matter what`
        oper: `which operation to use if need to remember`
        """
        i_epoch = len(epochs_df) - 1
        current_val = float(epochs_df[self.column_name].iloc[-1])
        if self.oper(current_val, self.best_val) or force:
            self.best_epoch = i_epoch
            self.best_val = current_val
            self.model_state_dict = deepcopy(model.state_dict())
            self.optimizer_state_dict = deepcopy(optimizer.state_dict())
            print("New best {:s}: {:5f}".format(self.column_name,
                                                   current_val))

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


def label_by_idx(idx):
    labels = {'BCI_IV_2a': ['Left Hand', 'Right Hand', 'Foot', 'Tongue'],
              'BCI_IV_2b': ['Left Hand', 'Right Hand'],
              'TUH': ['Normal', 'Abnormal'],
              'netflow_asflow': [f'hour {17+i}:00' for i in range(global_vars.get("n_classes"))]}
    return labels[global_vars.get('dataset')][idx]


def eeg_label_by_idx(idx):
    labels = {'BCI_IV_2b': ['C3', 'Cz', 'C4'],
              'BCI_IV_2a': ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz'],
              'TUH': ['A1', 'A2', 'C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6'],
              'netflow_asflow': ['-101', '6461', '20940', '33891', '2914', '6762', '3257', '1299', '3356', '1273', '6453']}
    return labels[global_vars.get('dataset')][idx]


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_index_of_last_layertype(model, layertype):
    for index, layer in reversed(list(enumerate(list(model.children())))):
        if type(layer) == layertype:
            return index


def get_oper_by_loss_function(loss_func, equals=False):
    if not equals:
        loss_func_opers = {F.nll_loss: operator.gt,
                           F.mse_loss: operator.lt,
                           torch.nn.CrossEntropyLoss: operator.gt}
    else:
        loss_func_opers = {F.nll_loss: operator.ge,
                           F.mse_loss: operator.le,
                           torch.nn.CrossEntropyLoss: operator.gt}
    return loss_func_opers[loss_func]


def concat_train_val_sets(dataset):
    dataset['train'] = concatenate_sets([dataset['train'], dataset['valid']])
    del dataset['valid']


def unify_dataset(dataset):
    return concatenate_sets([data for data in dataset.values()])


def reset_model_weights(model):
    for layer in model.children():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform(layer.weight.data)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            layer.reset_parameters()


def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)


def listen():
    if sys.platform == "linux" or sys.platform == "linux2":
        signal.signal(signal.SIGUSR1, debug)  # Register handler


def exit_handler(exp_folder, delete):
    if delete:
        print(f'deleting folder {exp_folder}')
        shutil.rmtree(exp_folder)


def not_exclusively_in(subj, model_from_file):
    all_subjs = [int(s) for s in model_from_file.split() if s.isdigit()]
    if len(all_subjs) > 1:
        return False
    if subj in all_subjs:
        return False
    return True


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


def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


def get_exp_id(results_folder):
    subdirs = [x for x in os.walk(results_folder)]
    if len(subdirs) == 1:
        exp_id = 1
    else:
        try:
            subdir_names = [int(x[0].split('/')[1].split('_')[0][0:]) for x in subdirs[1:]]
        except IndexError:
            subdir_names = [int(x[0].split('\\')[1].split('_')[0][0:]) for x in subdirs[1:]]
        subdir_names.sort()
        exp_id = subdir_names[-1] + 1
    return exp_id


def pretty_print_population(model_dict):
    result = ''
    for idx, model in model_dict.items():
        result += pretty_print_model(idx, model)
    return result


def pretty_print_model(name, model):
    result = f'Architecture {name}\n'
    result += '--------------------------------------------------\n'
    for layer in model:
        result += f'{type(layer).__name__}:\n'
        for param, value in layer.__dict__.items():
            if value is not None:
                result += f'\t{param}: {value}\n'
    result += '\n'
    return result


def write_dict(dict, filename):
    with open(filename, 'w') as f:
        for key, value in dict.items():
            f.write(f"{key}:\t{value}\n")


def set_moabb():
    if global_vars.get('dataset') in MOABB_DATASETS:
        paradigm = MotorImagery()
        dataset_names = [type(dataset).__name__ for dataset in paradigm.datasets]
        dataset = paradigm.datasets[dataset_names.index(global_vars.get('dataset'))]
        global_vars.set('subjects_to_check', dataset.subject_list)
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]])
        global_vars.set('eeg_chans', X.shape[1])
        global_vars.set('input_height', X.shape[2])
        global_vars.set('n_classes', len(np.unique(y)))
        if global_vars.get('time_limit_seconds'):
            global_vars.set('time_limit_seconds', int(global_vars.get('time_limit_seconds') / len(dataset.subject_list)))