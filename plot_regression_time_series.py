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
global_vars.init_config('configurations/config.ini')
configurations = get_configurations(CHOSEN_EXPERIMENT)


for configuration in configurations:
    global_vars.set_config(configuration)
    set_params_by_dataset('configurations/dataset_params.ini')
    dataset = get_dataset('all')
    concat_train_val_sets(dataset)
    set_gpu()
    model = torch.load(f'models/1032_netflow_asflow/{global_vars.get("input_time_len")}_'
                       f'{global_vars.get("steps_ahead")}_ahead.th').cuda()
    for segment in ['train', 'test']:
        y_pred = np.swapaxes(
            model(torch.tensor(dataset[segment].X[:, :, :, None]).float().cuda()).cpu().detach().numpy(), 0, 1)
        y_real = np.swapaxes(dataset[segment].y, 0, 1)
        df = pd.DataFrame()
        df['time_step'] = range(y_pred.shape[1])
        df.set_index('time_step')
        for steps_ahead in range(y_pred.shape[0]):
            df[f'{steps_ahead+1}_steps_ahead_real'] = y_real[steps_ahead]
            df[f'{steps_ahead+1}_steps_ahead_pred'] = y_pred[steps_ahead]
        df.to_csv(f'regression_results/{global_vars.get("input_time_len")}_'
                           f'{global_vars.get("steps_ahead")}_ahead_{segment}.csv')
