import torch
import global_vars
from EEGNAS_experiment import get_normal_settings
from data_preprocessing import get_netflow_train_val_test
from evolution.nn_training import NN_Trainer
from utilities.config_utils import set_default_config, set_params_by_dataset
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
import sys
log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)
matplotlib.use('TkAgg')

set_default_config('configurations/config.ini')
global_vars.set('dataset', 'netflow')
set_params_by_dataset('configurations/dataset_params.ini')
model = torch.load('models/1008_netflow/best_model_1.th')
global_vars.set('problem', 'regression')
global_vars.set('cuda', True)
global_vars.set('n_classes', 32)

dataset = {}
dataset['train'], dataset['valid'], dataset['test'] = get_netflow_train_val_test('data/', shuffle=False, n_sequences=32)
stop_criterion, iterator, loss_function, monitors = get_normal_settings()
nn_trainer = NN_Trainer(iterator, loss_function, stop_criterion, monitors)
nn_trainer.evaluate_model(model, dataset)


y_pred = np.swapaxes(model(torch.tensor(dataset['test'].X[:, :, :, None]).float().cuda()).cpu().detach().numpy(), 0, 1)
fig = plt.figure()
y_real = np.swapaxes(dataset['test'].y, 0, 1)

for idx, (real_seq, pred_seq) in list(enumerate(zip(y_real, y_pred))):
    ax = fig.add_subplot(len(y_real), 1, idx+1)
    ax.plot(real_seq, color='black')
    ax.plot(pred_seq, color='red')

plt.show()