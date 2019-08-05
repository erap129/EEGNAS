from datetime import datetime

import torch
from sklearn.model_selection import KFold, train_test_split

import global_vars
from EEGNAS_experiment import get_normal_settings
from data_preprocessing import get_dataset, makeDummySignalTargets
from evolution.nn_training import NN_Trainer
from utilities.config_utils import set_params_by_dataset, get_configurations, set_gpu
import matplotlib
import numpy as np
import logging
import sys
import pandas as pd

from utilities.misc import concat_train_val_sets, createFolder, unify_dataset, reset_model_weights

log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)
CHOSEN_EXPERIMENT = sys.argv[1]
STACK_RESULTS_BY_TIME = False
global_vars.init_config('configurations/config.ini')
configurations = get_configurations(CHOSEN_EXPERIMENT, global_vars.configs)


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
        nn_trainer.evaluate_model(model, dataset)
        concat_train_val_sets(dataset)
        for segment in ['train', 'test']:
            df = pd.DataFrame()
            globals()[f'export_{global_vars.get("dataset")}_results'](df, dataset[segment], segment,
                                                                      model, folder_name, fold_num)


def export_netflow_asflow_results(df, data, segment, model, folder_name, fold_num=None):
    y_pred = model(torch.tensor(data.X[:, :, :, None]).float().cuda()).cpu().detach().numpy()
    if STACK_RESULTS_BY_TIME:
        y_pred = y_pred[::global_vars.get("steps_ahead")]
        y_pred = np.concatenate([yi for yi in y_pred], axis=0)
        y_real = data.y[::global_vars.get("steps_ahead")]
        y_real = np.concatenate([yi for yi in y_real], axis=0)
        df['time_step'] = range(len(y_pred))
        df[f'{global_vars.get("steps_ahead")}_steps_ahead_real'] = y_real
        df[f'{global_vars.get("steps_ahead")}_steps_ahead_pred'] = y_pred
    else:
        y_pred = np.swapaxes(
            model(torch.tensor(data.X[:, :, :, None]).float().cuda()).cpu().detach().numpy(), 0, 1)
        y_real = np.swapaxes(data.y, 0, 1)
        df['time_step'] = range(y_pred.shape[1])
        for steps_ahead in range(y_pred.shape[0]):
            df[f'{steps_ahead+1}_steps_ahead_real'] = y_real[steps_ahead]
            df[f'{steps_ahead+1}_steps_ahead_pred'] = y_pred[steps_ahead]
    df.set_index('time_step')
    fold_str = ''
    if fold_num is not None:
        fold_str = f'_fold_{fold_num}'
    df.to_csv(f'{folder_name}/{global_vars.get("input_time_len")}_'
              f'{global_vars.get("steps_ahead")}_ahead_{segment}_stack_{STACK_RESULTS_BY_TIME}{fold_str}.csv')


now = datetime.now()
date_time = now.strftime("%m.%d.%Y")
for configuration in configurations:
    global_vars.set_config(configuration)
    set_params_by_dataset('configurations/dataset_params.ini')
    folder_name = f'regression_results/{date_time}_{global_vars.get("dataset")}'
    createFolder(folder_name)
    dataset = get_dataset('all')
    concat_train_val_sets(dataset)
    set_gpu()
    try:
        # model = torch.load(f'models/1032_netflow_asflow/{global_vars.get("input_time_len")}_'
        #                    f'{global_vars.get("steps_ahead")}_ahead.th').cuda()
        model = torch.load('models/61_netflow_asflow/best_model_1.th').cuda()
    except Exception as e:
        print(f'experiment failed: {str(e)}')
        continue
    if global_vars.get('k_fold'):
        kfold_exp(dataset, model, folder_name)
    else:
        for segment in ['train', 'test']:
            df = pd.DataFrame()
            globals()[f'export_{global_vars.get("dataset")}_results'](df, dataset[segment], model, folder_name)