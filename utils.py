import os
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from braindecode.experiments.monitors import compute_trial_labels_from_crop_preds, compute_preds_per_trial_from_crops
from sklearn import metrics


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


class AUCMonitor(object):
    """
    Monitor the examplewise AUC rate.

    Parameters
    ----------
    col_suffix: str, optional
        Name of the column in the monitoring output.
    threshold_for_binary_case: bool, optional
        In case of binary classification with only one output prediction
        per target, define the threshold for separating the classes, i.e.
        0.5 for sigmoid outputs, or np.log(0.5) for log sigmoid outputs
    """

    def __init__(self, col_suffix='auc', threshold_for_binary_case=None):
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
        fpr, tpr, thresholds = metrics.roc_curve(all_target_labels, all_pred_labels, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        column_name = "{:s}_{:s}".format(setname, self.col_suffix)
        return {column_name: float(auc)}


class AccuracyMonitor(object):
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

    def __init__(self, col_suffix='accuracy', threshold_for_binary_case=None):
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
        self.min_decrease = min_increase
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


class CroppedTrialAccuracyMonitor(object):
    """
    Compute trialwise accuracy from predictions for crops.

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length=None):
        self.input_time_length = input_time_length

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
        accuracy = 1 - np.mean(all_pred_labels == all_trial_labels)
        column_name = "{:s}_accuracy".format(setname)
        return {column_name: float(accuracy)}

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