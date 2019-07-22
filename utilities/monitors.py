import operator
import math
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelBinarizer


def mse_func(all_preds, all_trials):
    try:
        return mean_squared_error(all_trials.astype(np.float64), all_preds.astype(np.float64))
    except ValueError:
        return math.inf

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

    def __init__(self, measure_name, threshold_for_binary_case=None):
        self.measure_name = measure_name
        self.threshold_for_binary_case = threshold_for_binary_case
        self.monitor_name_mapping = {'auc': auc_func, 'kappa': kappa_func, 'f1': f1_func, 'accuracy': acc_func,
                                     'mse': mse_func}
        self.measure_func = self.monitor_name_mapping[self.measure_name]

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        if self.measure_func == mse_func:
            measure = self.measure_func(np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0))
        else:
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


class NoIncreaseDecrease(object):
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
    def __init__(self, column_name, num_epochs, min_step=1e-6, oper=operator.gt):
        self.column_name = column_name
        self.num_epochs = num_epochs
        self.min_step = min_step
        self.best_epoch = 0
        self.highest_val = 0
        self.oper = oper

    def should_stop(self, epochs_df):
        # -1 due to doing one monitor at start of training
        i_epoch = len(epochs_df) - 1
        current_val = float(epochs_df[self.column_name].iloc[-1])
        if self.oper(current_val, (1 + self.min_step) * self.highest_val):
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


class CroppedGenericMonitorPerTimeStep():
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
            preds_per_trial = compute_preds_per_trial_from_crops(
                all_preds, self.input_time_length, dataset.X)
            all_trial_labels = dataset.y
        measures = []
        for i in range(preds_per_trial[0].shape[1]):
            curr_preds = np.argmax(np.array([pred[:, i] for pred in preds_per_trial]).squeeze(), axis=1)
            assert curr_preds.shape == all_trial_labels.shape
            measures.append(self.measure_func(curr_preds, all_trial_labels))
        measure = np.max(np.array(measures))
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
