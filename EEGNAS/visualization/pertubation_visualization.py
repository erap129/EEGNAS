import logging
import importlib
import torch

import mne

importlib.reload(logging) # see https://stackoverflow.com/a/21475297/1469195
log = logging.getLogger()
log.setLevel('INFO')
import sys
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)
from EEGNAS import global_vars
from EEGNAS_experiment import set_params_by_dataset
from EEGNAS.data_preprocessing import get_train_val_test, get_ch_names
from braindecode.torch_ext.util import set_random_seeds
from EEGNAS_experiment import get_normal_settings, parse_args, get_configurations
from naiveNAS import NaiveNAS
from EEGNAS.models_generation import target_model
from braindecode.torch_ext.util import np_to_var
import numpy as np
from torch import nn
from braindecode.torch_ext.util import var_to_np
import torch as th
import matplotlib
matplotlib.use('qt5agg')


args = parse_args(['-e', 'tests', '-c', '../configurations/config.ini'])
global_vars.init_config(args.config)
configs = get_configurations(args.experiment)
assert (len(configs) == 1)
global_vars.set_config(configs[0])

subject_id = 1
low_cut_hz = 0
fs = 250
valid_set_fraction = 0.2
dataset = 'BCI_IV_2a'
data_folder = '../data/'
global_vars.set('dataset', dataset)
set_params_by_dataset()
global_vars.set('cuda', True)
model_select = 'deep4'
model_dir = '143_x_evolution_layers_cross_subject'
model_name = 'best_model_9_8_6_7_2_1_3_4_5.th'
train_set = {}
val_set = {}
test_set = {}
train_set[subject_id], val_set[subject_id], test_set[subject_id] =\
            get_train_val_test(data_folder, subject_id)

# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = True
set_random_seeds(seed=20170629, cuda=cuda)

# This will determine how many crops are processed in parallel
input_time_length = 450
# final_conv_length determines the size of the receptive field of the ConvNet
models = {'evolution': torch.load(f'../models/{model_dir}/{model_name}'),
         'deep4': target_model('deep')}
model = models[model_select]
input_time_length = global_vars.get('input_time_len')
stop_criterion, iterator, loss_function, monitors = get_normal_settings()
naiveNAS = NaiveNAS(iterator=iterator, exp_folder=None, exp_name=None,
                    train_set=train_set, val_set=val_set, test_set=test_set,
                    stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                    config=global_vars.config, subject_id=1, fieldnames=None, strategy='per_subject',
                    csv_file=None, evolution_file=None)
naiveNAS.train_and_evaluate_model(model)


train_batches = list(iterator.get_batches(train_set[subject_id], shuffle=False))
train_X_batches = np.concatenate(list(zip(*train_batches))[0])


new_model = nn.Sequential()
for name, module in model.named_children():
    if 'softmax' in name: break
    new_model.add_module(name, module)

new_model.eval()
pred_fn = lambda x: var_to_np(th.mean(new_model(np_to_var(x).cuda())[:, :, :, 0], dim=2, keepdim=False))
from braindecode.visualization.perturbation import compute_amplitude_prediction_correlations
amp_pred_corrs = compute_amplitude_prediction_correlations(pred_fn, train_X_batches, n_iterations=12,
                                         batch_size=30)

freqs = np.fft.rfftfreq(train_X_batches.shape[2], d=1.0/fs)

alpha_band = {'start': 7, 'stop': 14}
beta_band = {'start': 14, 'stop': 31}
high_gamma_band = {'start': 71, 'stop': 91}
bands = [alpha_band, beta_band, high_gamma_band]

for band in bands:
    band['i_start'] = np.searchsorted(freqs, band['start'])
    band['i_stop'] = np.searchsorted(freqs, band['stop']) + 1
    band['freq_corr'] = np.mean(amp_pred_corrs[:, band['i_start']:band['i_stop']], axis=1)


from braindecode.datasets.sensor_positions import get_channelpos, CHANNEL_10_20_APPROX

ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'P1', 'Pz', 'P3', 'POz']
positions = [get_channelpos(name, CHANNEL_10_20_APPROX) for name in ch_names]
positions = np.array(positions)


import matplotlib.pyplot as plt
from matplotlib import cm
max_abs_val = np.max([np.abs(band['freq_corr']) for band in bands])

fig, axes = plt.subplots(len(bands), global_vars.get('n_classes'))
class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
for band_i, band in enumerate(bands):
    for i_class in range(global_vars.get('n_classes')):
        ax = axes[band_i, i_class]
        mne.viz.plot_topomap(band['freq_corr'][:, i_class], positions,
                         vmin=-max_abs_val, vmax=max_abs_val, contours=0,
                        cmap=cm.coolwarm, axes=ax, show=False);
        ax.set_title(f'{band["start"]}-{band["stop"]}Hz, {class_names[i_class]}')

plt.show()
