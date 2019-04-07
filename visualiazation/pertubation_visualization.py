import logging
import importlib

import mne

importlib.reload(logging) # see https://stackoverflow.com/a/21475297/1469195
log = logging.getLogger()
log.setLevel('INFO')
import sys
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)
import globals
from BCI_IV_2a_experiment import set_params_by_dataset
from data_preprocessing import get_train_val_test, get_ch_names
from braindecode.models.deep4 import Deep4Net
from torch import nn
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.util import to_dense_prediction_model
from BCI_IV_2a_experiment import get_normal_settings, parse_args, get_configurations
from naiveNAS import NaiveNAS
from models_generation import target_model
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.torch_ext.util import np_to_var
import numpy as np
from BCI_IV_2a_loader import BCI_IV_2a

args = parse_args(['-e', 'tests', '-c', '../configurations/config.ini'])
globals.init_config(args.config)
configs = get_configurations(args.experiment)
assert (len(configs) == 1)
globals.set_config(configs[0])

subject_id = 1
low_cut_hz = 0
valid_set_fraction = 0.2
dataset = 'BCI_IV_2a'
globals.set('dataset', dataset)
set_params_by_dataset()
globals.set('cuda', True)
train_set, valid_set, test_set = get_train_val_test('../data/', 1, 0)


# Set if you want to use GPU
# You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
cuda = True
set_random_seeds(seed=20170629, cuda=cuda)

# This will determine how many crops are processed in parallel
input_time_length = 450
# final_conv_length determines the size of the receptive field of the ConvNet
model = target_model('deep')

input_time_length = globals.get('input_time_len')
print(input_time_length)
stop_criterion, iterator, loss_function, monitors = get_normal_settings()
naiveNAS = NaiveNAS(iterator=iterator, exp_folder=None, exp_name=None,
                    train_set=train_set, val_set=valid_set, test_set=test_set,
                    stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                    config=globals.config, subject_id=1, fieldnames=None, strategy='per_subject',
                    csv_file=None, evolution_file=None)
naiveNAS.evaluate_model(model)


train_batches = list(iterator.get_batches(train_set, shuffle=False))
train_X_batches = np.concatenate(list(zip(*train_batches))[0])


from torch import nn
from braindecode.torch_ext.util import var_to_np
import torch as th
new_model = nn.Sequential()
for name, module in model.named_children():
    if name == 'softmax': break
    new_model.add_module(name, module)

new_model.eval()
pred_fn = lambda x: var_to_np(th.mean(new_model(np_to_var(x).cuda())[:,:,:,0], dim=2, keepdim=False))

from braindecode.visualization.perturbation import compute_amplitude_prediction_correlations
amp_pred_corrs = compute_amplitude_prediction_correlations(pred_fn, train_X_batches, n_iterations=12,
                                         batch_size=30)

fs = 250
freqs = np.fft.rfftfreq(train_X_batches.shape[2], d=1.0/fs)
start_freq = 7
stop_freq = 14

i_start = np.searchsorted(freqs,start_freq)
i_stop = np.searchsorted(freqs, stop_freq) + 1

freq_corr = np.mean(amp_pred_corrs[:,i_start:i_stop], axis=1)

from braindecode.datasets.sensor_positions import get_channelpos, CHANNEL_10_20_APPROX

ch_names = get_ch_names()
positions = [get_channelpos(name, CHANNEL_10_20_APPROX) for name in ch_names]
positions = np.array(positions)


import matplotlib.pyplot as plt
from matplotlib import cm
max_abs_val = np.max(np.abs(freq_corr))



fig, axes = plt.subplots(1, 2)
class_names = ['Left Hand', 'Right Hand']
for i_class in range(2):
    ax = axes[i_class]
    mne.viz.plot_topomap(freq_corr[:,i_class], positions,
                     vmin=-max_abs_val, vmax=max_abs_val, contours=0,
                    cmap=cm.coolwarm, axes=ax, show=False);
    ax.set_title(class_names[i_class])

plt.show()
