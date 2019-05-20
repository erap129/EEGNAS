from BCI_IV_2a_experiment import parse_args, get_configurations
from data_preprocessing import get_bci_iv_2a_train_val_test
import globals
from braindecode.torch_ext.util import np_to_var
from visualiazation.src.misc_functions import create_mne
import matplotlib.pyplot as plt
import os.path as op
import mne
plt.interactive(False)

args = parse_args(['-e', 'tests', '-c', 'configurations/config.ini'])
globals.init_config(args.config)
configs = get_configurations(args.experiment)
assert(len(configs) == 1)
globals.set_config(configs[0])

train_set, val_set, test_set = get_bci_iv_2a_train_val_test('data/BCI_IV/', 1, 0)

def create_all_spectrograms(data, im_size=256):
    for j, channel in enumerate(data[:1]):
        channel *= 1000
        fig = plt.figure(frameon=False)
        fig.set_size_inches((im_size - 10) / 96, (im_size - 10) / 96)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(list(range(len(channel))), channel)
        ax2.specgram(channel, NFFT=256, Fs=250, noverlap=255)
        fig.canvas.draw()
        plt.show()
        plt.close(fig)

create_all_spectrograms(train_set.X[0].squeeze())
# globals.set_config({'DEFAULT':{'exp_name': 'test'}, 'test':{}})
# globals.set('dataset', 'BCI_IV_2a')
# globals.set('valid_set_fraction', 0.2)
# train_set, val_set, test_set = get_train_val_test('data/', 1, 0)
# to_mne = np_to_var(train_set.X[0])
# mne_raw = create_mne(to_mne)
# mne_raw.plot()

# data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
# raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'),                      preload=True)
# raw.set_eeg_reference('average', projection=True)  # set EEG average reference
# raw.plot(block=True, lowpass=40)


