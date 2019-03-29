import os
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from braindecode.experiments.monitors import compute_trial_labels_from_crop_preds, compute_preds_per_trial_from_crops
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from copy import deepcopy
import logging
log = logging.getLogger(__name__)


def summary(model, input_size, batch_size=-1, device="cuda", file=None):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()
    print("----------------------------------------------------------------", file=file)
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new, file=file)
    print("================================================================", file=file)
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new, file=file)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================", file=file)
    print("Total params: {0:,}".format(total_params), file=file)
    print("Trainable params: {0:,}".format(trainable_params), file=file)
    print("Non-trainable params: {0:,}".format(total_params - trainable_params), file=file)
    print("----------------------------------------------------------------", file=file)
    print("Input size (MB): %0.2f" % total_input_size, file=file)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size, file=file)
    print("Params size (MB): %0.2f" % total_params_size, file=file)
    print("Estimated Total Size (MB): %0.2f" % total_size, file=file)
    print("----------------------------------------------------------------", file=file)


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def acc_func(all_pred_labels, all_trial_labels):
    accuracy = np.mean(all_pred_labels == all_trial_labels)
    return accuracy


def kappa_func(all_pred_labels, all_target_labels):
    kappa = metrics.cohen_kappa_score(all_pred_labels, all_target_labels)
    return kappa


def f1_func(all_pred_labels, all_target_labels):
    f1 = metrics.f1_score(all_target_labels, all_pred_labels, average='weighted')
    return f1


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def auc_func(all_pred_labels, all_target_labels):
    if len(np.unique(all_pred_labels)) == len(np.unique(all_target_labels)) == 2:  # binary classification
        fpr, tpr, thresholds = metrics.roc_curve(all_target_labels, all_pred_labels, pos_label=1)
        auc = metrics.auc(fpr, tpr)
    else:
        auc = multiclass_roc_auc_score(all_target_labels, all_pred_labels)
    return auc


class GenericMonitor(object):
    """
    Monitor the examplewise rate.

    Parameters
    ----------
    col_suffix: str, optional
        Name of the column in the monitoring output.
    threshold_for_binary_case: bool, optional
        In case of binary classification with only one output prediction
        per target, define the threshold for separating the classes, i.e.
        0.5 for sigmoid outputs, or np.log(0.5) for log sigmoid outputs
    """

    def __init__(self, measure_name, measure_func,threshold_for_binary_case=None):
        self.measure_name = measure_name
        self.measure_func = measure_func
        self.threshold_for_binary_case = threshold_for_binary_case

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        all_pred_labels = []
        all_target_labels = []
        for i_batch in range(len(all_batch_sizes)):
            preds = all_preds[i_batch]
            # preds could be examples x classes x time
            # or just
            # examples x classes
            # make sure not to remove first dimension if it only has size one
            if preds.ndim > 1:
                only_one_row = preds.shape[0] == 1

                pred_labels = np.argmax(preds, axis=1).squeeze()
                # add first dimension again if needed
                if only_one_row:
                    pred_labels = pred_labels[None]
            else:
                assert self.threshold_for_binary_case is not None, (
                    "In case of only one output, please supply the "
                    "threshold_for_binary_case parameter")
                # binary classification case... assume logits
                pred_labels = np.int32(preds > self.threshold_for_binary_case)
            # now examples x time or examples
            all_pred_labels.extend(pred_labels)
            targets = all_targets[i_batch]
            if targets.ndim > pred_labels.ndim:
                # targets may be one-hot-encoded
                targets = np.argmax(targets, axis=1)
            elif targets.ndim < pred_labels.ndim:
                # targets may not have time dimension,
                # in that case just repeat targets on time dimension
                extra_dim = pred_labels.ndim - 1
                targets = np.repeat(np.expand_dims(targets, extra_dim),
                                    pred_labels.shape[extra_dim],
                                    extra_dim)
            assert targets.shape == pred_labels.shape
            all_target_labels.extend(targets)
        all_pred_labels = np.array(all_pred_labels)
        all_target_labels = np.array(all_target_labels)
        assert all_pred_labels.shape == all_target_labels.shape
        measure = self.measure_func(all_pred_labels, all_target_labels)
        column_name = "{:s}_{:s}".format(setname, self.measure_name)
        return {column_name: float(measure)}


class MultiLabelAccuracyMonitor(object):
    """
    Monitor the examplewise accuracy rate.

    Parameters
    ----------
    col_suffix: str, optional
        Name of the column in the monitoring output.
    threshold_for_binary_case: bool, optional
        In case of binary classification with only one output prediction
        per target, define the threshold for separating the classes, i.e.
        0.5 for sigmoid outputs, or np.log(0.5) for log sigmoid outputs
    """

    def __init__(self, col_suffix='accuracy', threshold_for_binary_case=0.5):
        self.col_suffix = col_suffix
        self.threshold_for_binary_case = threshold_for_binary_case

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        all_pred_labels = []
        all_target_labels = []
        for i_batch in range(len(all_batch_sizes)):
            preds = all_preds[i_batch]
            # preds could be examples x classes x time
            # or just
            # examples x classes
            # make sure not to remove first dimension if it only has size one
            if preds.ndim > 1:
                only_one_row = preds.shape[0] == 1
                # add first dimension again if needed
                if only_one_row:
                    pred_labels = pred_labels[None]
            assert self.threshold_for_binary_case is not None, (
                "In case of only one output, please supply the "
                "threshold_for_binary_case parameter")
            # binary classification case... assume logits
            pred_labels = np.int32(preds > self.threshold_for_binary_case)
            # now examples x time or examples
            all_pred_labels.extend(pred_labels)
            targets = all_targets[i_batch]
            if targets.ndim < pred_labels.ndim:
                # targets may not have time dimension,
                # in that case just repeat targets on time dimension
                extra_dim = pred_labels.ndim - 1
                targets = np.repeat(np.expand_dims(targets, extra_dim),
                                    pred_labels.shape[extra_dim],
                                    extra_dim)
            assert targets.shape == pred_labels.shape
            all_target_labels.extend(targets)
        all_pred_labels = np.array(all_pred_labels)
        all_target_labels = np.array(all_target_labels)
        assert all_pred_labels.shape == all_target_labels.shape

        accuracy = np.mean(all_target_labels == all_pred_labels)
        column_name = "{:s}_{:s}".format(setname, self.col_suffix)
        return {column_name: float(accuracy)}


class NoIncrease(object):
    """ Stops if there is no decrease on a given monitor channel
    for given number of epochs.

    Parameters
    ----------
    column_name: str
        Name of column to monitor for decrease.
    num_epochs: str
        Number of epochs to wait before stopping when there is no decrease.
    min_decrease: float, optional
        Minimum relative decrease that counts as a decrease. E.g. 0.1 means
        only 10% decreases count as a decrease and reset the counter.
    """
    def __init__(self, column_name, num_epochs, min_increase=1e-6):
        self.column_name = column_name
        self.num_epochs = num_epochs
        self.min_increase = min_increase
        self.best_epoch = 0
        self.highest_val = 0

    def should_stop(self, epochs_df):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        current_val = float(epochs_df[self.column_name].iloc[-1])
        if current_val > ((1 + self.min_increase) * self.highest_val):
            self.best_epoch = i_epoch
            self.highest_val = current_val

        return (i_epoch - self.best_epoch) >= self.num_epochs


class CroppedTrialGenericMonitor():
    """
    Compute trialwise *** from predictions for crops.

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, measure_name, measure_func, input_time_length=None):
        self.input_time_length = input_time_length
        self.measure_name = measure_name
        self.measure_func = measure_func

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        assert self.input_time_length is not None, "Need to know input time length..."
        # First case that each trial only has a single label
        if not hasattr(dataset.y[0], '__len__'):
            all_pred_labels = compute_trial_labels_from_crop_preds(
                all_preds, self.input_time_length, dataset.X)
            assert all_pred_labels.shape == dataset.y.shape
            all_trial_labels = dataset.y
        else:
            all_trial_labels, all_pred_labels = (
                self._compute_trial_pred_labels_from_cnt_y(dataset, all_preds))
        assert all_pred_labels.shape == all_trial_labels.shape
        measure = self.measure_func(all_pred_labels, all_trial_labels)
        column_name = "{:s}_{:s}".format(setname, self.measure_name)
        return {column_name: float(measure)}

    def _compute_pred_labels(self, dataset, all_preds, ):
        preds_per_trial = compute_preds_per_trial_from_crops(
            all_preds, self.input_time_length, dataset.X)
        all_pred_labels = [np.argmax(np.mean(p, axis=1))
                           for p in preds_per_trial]
        all_pred_labels = np.array(all_pred_labels)
        assert all_pred_labels.shape == dataset.y.shape
        return all_pred_labels

    def _compute_trial_pred_labels_from_cnt_y(self, dataset, all_preds, ):
        # Todo: please test this
        # we only want the preds that are for the same labels as the last label in y
        # (there might be parts of other class-data at start, for trialwise misclass we assume
        # they are contained in other trials at the end...)
        preds_per_trial = compute_preds_per_trial_from_crops(
            all_preds, self.input_time_length, dataset.X)
        trial_labels = []
        trial_pred_labels = []
        for trial_pred, trial_y in zip(preds_per_trial, dataset.y):
            # first cut to the part actually having predictions
            trial_y = trial_y[-trial_pred.shape[1]:]
            wanted_class = trial_y[-1]
            trial_labels.append(wanted_class)
            # extract the first marker different from the wanted class
            # by starting from the back of the trial
            i_last_sample = np.flatnonzero(trial_y[::-1] != wanted_class)
            if len(i_last_sample) > 0:
                i_last_sample = i_last_sample[0]
                # remember last sample is now from back
                trial_pred = trial_pred[:, -i_last_sample:]
            trial_pred_label = np.argmax(np.mean(trial_pred, axis=1))
            trial_pred_labels.append(trial_pred_label)
        trial_labels = np.array(trial_labels)
        trial_pred_labels = np.array(trial_pred_labels)
        return trial_labels, trial_pred_labels


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
        self.highest_val = -2
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
        if current_val >= self.highest_val:
            self.best_epoch = i_epoch
            self.highest_val = current_val
            self.model_state_dict = deepcopy(model.state_dict())
            self.optimizer_state_dict = deepcopy(optimizer.state_dict())
            log.info("New best {:s}: {:5f}".format(self.column_name,
                                                   current_val))
            log.info("")
            # keys = model.state_dict().keys()
            # for key in keys:
            #     if f'(1' in key or f'(2' in key or f'(3' in key or f'(4' in key or f'(5' in key or\
            #         f'(6' in key or f'(7' in key or f'(8' in key or f'(9' in key:
            #         if list(model.state_dict().keys()) == list(self.model_state_dict.keys()):
            #             print('found special. state dicts equal')
            #         else:
            #             print('failed: found special. state dicts not equal')

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


def pretty_size(size):
	"""Pretty prints a torch.Size object"""
	assert(isinstance(size, torch.Size))
	return " × ".join(map(str, size))


def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__,
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
												   type(obj.data).__name__,
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "",
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass
	print("Total size:", total_size)