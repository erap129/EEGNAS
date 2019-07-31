import torch
import global_vars
from data_preprocessing import get_dataset
from utilities.config_utils import set_params_by_dataset, get_configurations, set_gpu
import matplotlib
import numpy as np
import logging
import sys
import pandas as pd

from utilities.misc import concat_train_val_sets

log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)
matplotlib.use('TkAgg')
CHOSEN_EXPERIMENT = 'netflow_plotting'
STACK_RESULTS_BY_TIME = True
global_vars.init_config('configurations/config.ini')
configurations = get_configurations(CHOSEN_EXPERIMENT)


for configuration in configurations:
    global_vars.set_config(configuration)
    set_params_by_dataset('configurations/dataset_params.ini')
    dataset = get_dataset('all')
    concat_train_val_sets(dataset)
    set_gpu()
    try:
        model = torch.load(f'models/1032_netflow_asflow/{global_vars.get("input_time_len")}_'
                           f'{global_vars.get("steps_ahead")}_ahead.th').cuda()
    except Exception as e:
        print(f'experiment failed: {str(e)}')
        continue
    for segment in ['train', 'test']:
        df = pd.DataFrame()
        y_pred = model(torch.tensor(dataset[segment].X[:, :, :, None]).float().cuda()).cpu().detach().numpy()
        if STACK_RESULTS_BY_TIME:
            y_pred = y_pred[::global_vars.get("steps_ahead")]
            y_pred = np.concatenate([yi for yi in y_pred], axis=0)
            y_real = dataset[segment].y[::global_vars.get("steps_ahead")]
            y_real = np.concatenate([yi for yi in y_real], axis=0)
            df['time_step'] = range(len(y_pred))
            df[f'{global_vars.get("steps_ahead")}_steps_ahead_real'] = y_real
            df[f'{global_vars.get("steps_ahead")}_steps_ahead_pred'] = y_pred
        else:
            y_pred = np.swapaxes(
                model(torch.tensor(dataset[segment].X[:, :, :, None]).float().cuda()).cpu().detach().numpy(), 0, 1)
            y_real = np.swapaxes(dataset[segment].y, 0, 1)
            df['time_step'] = range(y_pred.shape[1])
            for steps_ahead in range(y_pred.shape[0]):
                df[f'{steps_ahead+1}_steps_ahead_real'] = y_real[steps_ahead]
                df[f'{steps_ahead+1}_steps_ahead_pred'] = y_pred[steps_ahead]
        df.set_index('time_step')
        df.to_csv(f'regression_results/{global_vars.get("input_time_len")}_'
                           f'{global_vars.get("steps_ahead")}_ahead_{segment}_stack_{STACK_RESULTS_BY_TIME}.csv')
