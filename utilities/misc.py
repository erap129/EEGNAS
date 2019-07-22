import os
from copy import deepcopy
import logging

from braindecode.datautil.splitters import concatenate_sets

log = logging.getLogger(__name__)
import math
import operator
import torch.nn.functional as F


def createFolder(directory):
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
            self.highest_val = current_val
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


def label_by_idx(idx):
    labels = {'BCI_IV_2a': ['Left Hand', 'Right Hand', 'Foot', 'Tongue'],
              'BCI_IV_2b': ['Left Hand', 'Right Hand']}
    return labels[globals.get('dataset')][idx]


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
                           F.mse_loss: operator.lt}
    else:
        loss_func_opers = {F.nll_loss: operator.ge,
                           F.mse_loss: operator.le}
    return loss_func_opers[loss_func]


def concat_train_val_sets(dataset):
    dataset['train'] = concatenate_sets([dataset['train'], dataset['valid']])
