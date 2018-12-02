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
import csv
import traceback
import time

global data_folder, valid_set_fraction, config
init_config()
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
if platform.node() == 'nvidia':
    globals.config.set('DEFAULT', 'cuda', 'True')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_folder = 'data/'
low_cut_hz = 0
valid_set_fraction = 0.2


def garbage_time():
    print('ENTERING GARBAGE TIME')
    train_set, val_set, test_set = get_train_val_test(data_folder, 1, 0)
    garbageNAS = NaiveNAS(iterator=iterator, n_classes=4, input_time_len=1125, n_chans=22,
                        train_set=train_set, val_set=val_set, test_set=test_set,
                        stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                        config=globals.config, subject_id=subject_id, cropping=False)
    garbageNAS.garbage_time()


field_names = ['exp_id', 'generation', 'avg_fitness']

experiments = {
    'evolution': run_genetic_filters
}
stop_criterion = Or([MaxEpochs(globals.config['DEFAULT'].getint('max_epochs')),
                     NoDecrease('valid_misclass', globals.config['DEFAULT'].getint('max_increase_epochs'))])
monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
iterator = BalancedBatchSizeIterator(batch_size=globals.config['DEFAULT'].getint('batch_size'))
loss_function = F.nll_loss

try:
    for i in range(globals.config['DEFAULT'].getint('num_iterations')):
        subdirs = [x for x in os.walk('results')]
        if len(subdirs) == 1:
            exp_id = 1
        else:
            subdir_print = [x[0] for x in subdirs[1:]]
            print(subdir_print)
            try:
                subdir_names = [int(x[0].split('/')[1].split('_')[0][0:]) for x in subdirs[1:]]
            except IndexError:
                subdir_names = [int(x[0].split('\\')[1].split('_')[0][0:]) for x in subdirs[1:]]
            subdir_names.sort()
            exp_id = subdir_names[-1] + 1

        subjects = random.sample(range(1, 10), globals.config['evolution'].getint('num_subjects'))
        exp_folder = 'results/' + str(exp_id) + '_' + globals.config['DEFAULT']['exp_type']
        createFolder(exp_folder)
        csv_file = exp_folder + '/' + str(exp_id) + '_' + globals.config['DEFAULT']['exp_type'] + '.csv'
        merged_results_dict = {'subject': [], 'generation': [], 'val_acc': []}
        total_results = pd.DataFrame(columns=['subject', 'generation', 'val_acc'])
        with open(csv_file, 'a', newline='') as csvfile:
            fieldnames = ['subject', 'generation', 'train_acc', 'val_acc', 'test_acc', 'train_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()


            start_time = time.time()
            for subject_id in subjects:
                train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
                naiveNAS = NaiveNAS(iterator=iterator, n_classes=4, input_time_len=1125, n_chans=22,
                                    train_set=train_set, val_set=val_set, test_set=test_set,
                                    stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                                    config=globals.config, subject_id=subject_id, cropping=False)
                evolution_file = '%s/subject_%d_archs.txt' % (exp_folder, subject_id)
                if globals.config['DEFAULT']['exp_type'] == 'evolution_layers':
                    naiveNAS.evolution_layers(csv_file, evolution_file)
                elif globals.config['DEFAULT']['exp_type'] == 'evolution_filters':
                    naiveNAS.evolution_filters(csv_file, evolution_file)
                elif globals.config['DEFAULT']['exp_type'] == 'target':
                    naiveNAS.run_target_model(csv_file)
            globals.config['DEFAULT']['total_time'] = str(time.time() - start_time)
            with open(exp_folder + '/final_config.ini', 'w') as configfile:
                globals.config.write(configfile)

except Exception as e:
    with open(exp_folder + "/error_log.txt", "w") as err_file:
        print('experiment failed. Exception message: %s' % (str(e)), file=err_file)
        print(traceback.format_exc(), file=err_file)
    new_exp_folder = exp_folder + '_fail'
    os.rename(exp_folder, new_exp_folder)
    with open(new_exp_folder + '/final_config.ini', 'w') as configfile:
        globals.config.write(configfile)

finally:
    garbage_time()





