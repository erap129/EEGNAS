#%%
import os
import platform
import pandas as pd
from experiments import run_genetic_filters
import torch.nn.functional as F
from data_preprocessing import get_train_val_test
from naiveNAS import NaiveNAS
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor
from globals import init_config
from utils import createFolder
import logging
import globals
import random
import sys

global data_folder, valid_set_fraction, config
init_config()
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
if platform.node() == 'nvidia':
    globals.config.set('DEFAULT', 'cuda', 'True')
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

data_folder = 'data/'
low_cut_hz = 0
valid_set_fraction = 0.2

field_names = ['exp_id', 'generation', 'avg_fitness']

experiments = {
    'evolution': run_genetic_filters
}
stop_criterion = Or([MaxEpochs(globals.config['DEFAULT'].getint('max_epochs')),
                     NoDecrease('valid_misclass', globals.config['DEFAULT'].getint('max_increase_epochs'))])
monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
iterator = BalancedBatchSizeIterator(batch_size=globals.config['DEFAULT'].getint('batch_size'))
loss_function = F.nll_loss
#%%
subject_id = 1
train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
naiveNAS = NaiveNAS(iterator=iterator, n_classes=4, input_time_len=1125, n_chans=22,
                    train_set=train_set, val_set=val_set, test_set=test_set,
                    stop_criterion=stop_criterion, monitors = monitors, loss_function=loss_function,
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

subjects = random.sample(range(1, 10), globals.config['evolution'].getint('num_subjects'))
exp_folder = 'results/' + str(exp_id) + '_evolution_' + '_'.join(globals.config['evolution'].values())
merged_results_dict = {'subject': [], 'generation': [], 'val_acc': []}
total_results = pd.DataFram()
for subject_id in subjects:
    train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
    naiveNAS = NaiveNAS(iterator=iterator, n_classes=4, input_time_len=1125, n_chans=22,
                        train_set=train_set, val_set=val_set, test_set=test_set,
                        stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                        config=globals.config, subject_id=subject_id, cropping=False)
    evolution_results = naiveNAS.evolution_filters()
    total_results = total_results.append(evolution_results)
createFolder(exp_folder)
pd.DataFrame.from_dict(merged_results_dict, orient='index').to_csv(exp_folder + '/results.csv')
