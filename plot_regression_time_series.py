from datetime import datetime
import torch
from braindecode.datautil.splitters import concatenate_sets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, train_test_split
from torch import nn
import global_vars
from EEGNAS_experiment import get_normal_settings
from data.netflow.netflow_data_utils import get_whole_netflow_data
from data_preprocessing import get_dataset, makeDummySignalTargets
from evolution.nn_training import NN_Trainer
from utilities.config_utils import set_params_by_dataset, get_configurations, set_gpu
import matplotlib
import numpy as np
import logging
import sys
import pandas as pd

from utilities.data_utils import calc_regression_accuracy
from utilities.misc import concat_train_val_sets, createFolder, unify_dataset, reset_model_weights


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        x = x.view(-1, 32, 10)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        h0, c0 = h0.cuda(), c0.cuda()
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def export_netflow_asflowAE_results(df, data, model, folder_name):
    y_real = data.y
    y_pred = model(torch.tensor(data.X[:, :, :, None]).float().cuda()).cpu().detach().numpy()
    for example_idx, (real_example, pred_example) in enumerate(zip(y_real, y_pred)):
        for channel_idx, (real_channel, pred_channel) in enumerate(zip(real_example, pred_example)):
            example_df = pd.DataFrame()
            example_df['example_id'] = [example_idx for i in range(len(y_real[0, 0]))]
            example_df['channel'] = [channel_idx for i in range(len(y_real[0, 0]))]
            example_df['time_step'] = range(len(y_real[0, 0]))
            example_df['noisy_real'] = data.X[example_idx, channel_idx]
            example_df['real'] = real_channel
            example_df['pred'] = pred_channel
            df = df.append(example_df)
            print(f'finished channel {channel_idx} in example {example_idx} in {segment} segment')
    df.to_csv(f'{folder_name}/netflow_asflowAE_{segment}_results.csv', index=False)


def get_netflow_test_data_by_indices(train_index, test_index, problem):
    prev_steps_ahead = global_vars.get('steps_ahead')
    prev_problem = global_vars.get('problem')
    global_vars.set('problem', problem)
    global_vars.set('steps_ahead', 24)
    dataset = get_dataset('all')
    if train_index is not None:
        data = unify_dataset(dataset)
        X_test = data.X[test_index]
        y_test = data.y[test_index]
    else:
        X_test = dataset['test'].X
        y_test = dataset['test'].y
    global_vars.set('problem', prev_problem)
    global_vars.set('steps_ahead', prev_steps_ahead)
    return X_test, y_test


def kfold_exp(data, model, folder_name):
    data = unify_dataset(data)
    kf = KFold(n_splits=5)
    for fold_num, (train_index, test_index) in enumerate(kf.split(data.X)):
        reset_model_weights(model)
        stop_criterion, iterator, loss_function, monitors = get_normal_settings()
        nn_trainer = NN_Trainer(iterator, loss_function, stop_criterion, monitors)
        X_train, X_test = data.X[train_index], data.X[test_index]
        y_train, y_test = data.y[train_index], data.y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=global_vars.get('valid_set_fraction'),
                                                            shuffle=False)
        dataset = {}
        dataset['train'], dataset['valid'], dataset['test'] = \
            makeDummySignalTargets(X_train, y_train, X_val, y_val, X_test, y_test)
        nn_trainer.evaluate_model(model, dataset, final_evaluation=True)
        for segment in ['train', 'valid', 'test']:
            df = pd.DataFrame()
            if global_vars.get('problem') == 'classification':
                if segment != 'test':
                    continue
                globals()[f'export_{global_vars.get("dataset")}_results_classification'](df, dataset[segment], segment,
                                                                          model, folder_name, fold_num,
                                                                          train_index, test_index)
            else:
                globals()[f'export_{global_vars.get("dataset")}_results'](df, dataset[segment], segment,
                                                                      model, folder_name, fold_num)


def no_kfold_exp(dataset, model, folder_name, reset_weights=False):
    if reset_weights:
        reset_model_weights(model)
        stop_criterion, iterator, loss_function, monitors = get_normal_settings()
        nn_trainer = NN_Trainer(iterator, loss_function, stop_criterion, monitors)
        nn_trainer.evaluate_model(model, dataset, final_evaluation=True)
    for segment in dataset.keys():
        df = pd.DataFrame()
        if global_vars.get('problem') == 'classification':
            if segment != 'test':
                continue
            globals()[f'export_{global_vars.get("dataset")}_results_classification'](df, dataset[segment], segment,
                                                                                     model, folder_name, fold_num=-1,
                                                                                     train_index=None, test_index=None)
        else:
            globals()[f'export_{global_vars.get("dataset")}_results'](df, dataset[segment], segment,
                                                                      model, folder_name)


def export_netflow_asflow_results(df, data, segment, model, folder_name, fold_num=None):
    model.eval()
    y_pred = model(torch.tensor(data.X[:, :, :, None]).float().cuda()).cpu().detach().numpy()
    if global_vars.get('steps_ahead') < global_vars.get('jumps'):
        y_pred = np.array([np.concatenate([y, np.array([np.nan for i in range(int(global_vars.get('jumps') -
                        global_vars.get('steps_ahead')))])], axis=0) for y in y_pred])
        y_real = np.array([np.concatenate([y, np.array([np.nan for i in range(int(global_vars.get('jumps') -
                        global_vars.get('steps_ahead')))])], axis=0) for y in data.y])
        y_pred = np.concatenate([yi for yi in y_pred], axis=0)
        y_real = np.concatenate([yi for yi in y_real], axis=0)
        df[f'{global_vars.get("steps_ahead")}_steps_ahead_real'] = y_real
        df[f'{global_vars.get("steps_ahead")}_steps_ahead_pred'] = y_pred
    else:
        y_pred = np.swapaxes(
            model(torch.tensor(data.X[:, :, :, None]).float().cuda()).cpu().detach().numpy(), 0, 1)
        y_real = np.swapaxes(data.y, 0, 1)
        for steps_ahead in range(y_pred.shape[0]):
            df[f'{steps_ahead+1}_steps_ahead_real'] = y_real[steps_ahead]
            df[f'{steps_ahead+1}_steps_ahead_pred'] = y_pred[steps_ahead]

    orig_df = get_whole_netflow_data('data/netflow/akamai-dt-handovers_1.7.17-1.8.19.csv')
    datetimes = orig_df.index.values[global_vars.get('input_height')+global_vars.get('steps_ahead')-1:]
    datetimes_dict = {}
    datetimes_dict['train'], datetimes_dict['test'], _, _ = train_test_split(datetimes, datetimes,
                                                test_size=global_vars.get('valid_set_fraction'), shuffle=False)
    df['time'] = datetimes_dict[segment]
    df.index = pd.to_datetime(df['time'])
    df = df.drop(columns=['time'])
    actual, predicted = calc_regression_accuracy(df[f'{global_vars.get("steps_ahead")}_steps_ahead_pred'].values,
                                                 df[f'{global_vars.get("steps_ahead")}_steps_ahead_real'].values,
                                                 global_vars.get('netflow_threshold'))
    fold_str = ''
    if fold_num is not None:
        fold_str = f'_fold_{fold_num}'
    with open(f'{folder_name}/{global_vars.get("input_height")}_'
              f'{global_vars.get("steps_ahead")}_ahead_{segment}{fold_str}_accuracy.txt', 'w+') as f:
        print(f'accuracy score - {accuracy_score(actual, predicted)}', file=f)
        print(classification_report(actual, predicted), file=f)
    df.to_csv(f'{folder_name}/{global_vars.get("input_height")}_'
              f'{global_vars.get("steps_ahead")}_ahead_{segment}{fold_str}.csv')


def export_netflow_asflow_results_classification(df, data, segment, model, folder_name, fold_num, train_index, test_index):
    model.eval()
    y_pred = model(torch.tensor(data.X[:, :, :, None]).float().cuda()).cpu().detach().numpy()
    day = np.concatenate([[day for j in range(global_vars.get('jumps'))] for day in range(len(y_pred))], axis=0)
    y_pred = np.concatenate([[np.argmax(y) for i in range(global_vars.get('jumps'))] for y in y_pred], axis=0)
    y_real = np.concatenate([[y for i in range(global_vars.get('jumps'))] for y in data.y], axis=0)

    _, y_test_reg = get_netflow_test_data_by_indices(train_index, test_index, 'regression')
    y_regression = np.concatenate([yi for yi in y_test_reg], axis=0)
    # df['day'] = day
    df['predicted_decision'] = y_pred
    df['real_decision'] = y_real
    # df['regression_data'] = y_regression
    fold_str = ''
    if fold_num is not None:
        fold_str = f'_fold_{fold_num}'
    df.to_csv(f'{folder_name}/{global_vars.get("input_height")}_'
              f'{global_vars.get("steps_ahead")}_ahead_{segment}{fold_str}_'
              f'classification.csv')



if __name__ == '__main__':
    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    CHOSEN_EXPERIMENT = sys.argv[1]
    global_vars.init_config('configurations/config.ini')
    configurations = get_configurations(CHOSEN_EXPERIMENT, global_vars.configs)

    now = datetime.now()
    date_time = now.strftime("%m.%d.%Y")
    for configuration in configurations:
        global_vars.set_config(configuration)
        set_params_by_dataset('configurations/dataset_params.ini')
        dataset = get_dataset('all')
        concat_train_val_sets(dataset)
        set_gpu()
        try:
            model = torch.load(f'models/{global_vars.get("models_dir")}/{global_vars.get("model_file_name")}').cuda()
        except Exception as e:
            print(f'experiment failed: {str(e)}')
            continue
        folder_name = f'regression_results/{date_time}_{global_vars.get("dataset")}' \
                      f'_{global_vars.get("model_file_name")}'
        createFolder(folder_name)
        if global_vars.get('k_fold'):
            kfold_exp(dataset, model, folder_name)
        else:
            no_kfold_exp(dataset, model, folder_name)