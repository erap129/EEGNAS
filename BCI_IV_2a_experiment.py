#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import tensorflow as tf
import platform
import matplotlib
matplotlib.use('Agg')
from experiments import run_genetic_filters
import configparser
from data_preprocessing import get_train_val_test
from naiveNAS import NaiveNAS
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from globals import init_config
import globals

global data_folder, valid_set_fraction, config
init_config()
if platform.node() == 'nvidia':
    globals.set('DEFAULT', 'cuda', 'True')
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

data_folder = 'data/'
low_cut_hz = 0
valid_set_fraction = 0.2

field_names = ['exp_id', 'generation', 'avg_fitness']

experiments = {
    'evolution': run_genetic_filters
}
stop_criterion = Or([MaxEpochs(globals.config['DEFAULT'].getint('max_epochs')),
                     NoDecrease('valid_acc', globals.config['DEFAULT'].getint('max_increase_epochs'))])
monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
iterator = BalancedBatchSizeIterator(batch_size=globals.config['DEFAULT'].getint('batch_size'))
#%%
subject_id = 1
train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
naiveNAS = NaiveNAS(iterator=iterator, n_classes=4, input_time_len=1125, n_chans=22,
                    train_set=train_set, val_set=val_set, test_set=test_set,
                    stop_criterion=stop_criterion, monitors = monitors,
                    config=globals.config, subject_id=subject_id, cropping=False)
#%%
results_dict = naiveNAS.evolution_filters()
#%%

subdirs = [x for x in os.walk('results')]
if len(subdirs) == 1:
    exp_id = 1
else:
    subdir_names = [int(x[0].split('\\')[1]) for x in subdirs[1:]]
    subdir_names.sort()
    exp_id = subdir_names[-1] + 1
evolution = config['DEFAULT']['exp_type']
sub_conf = config[evolution]
experiments[evolution](exp_id, sub_conf)