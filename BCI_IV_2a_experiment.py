#%%
import os
import platform
import torch
from collections import OrderedDict
from itertools import product
import torch.nn.functional as F
from data_preprocessing import get_train_val_test
from naiveNAS import NaiveNAS
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator, CropsFromTrialsIterator
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from globals import init_config
from utils import createFolder
import logging
import globals
import random
import sys
import csv
import traceback
import time
import json

global data_folder, valid_set_fraction, config
init_config()
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)

data_folder = 'data/'
low_cut_hz = 0
valid_set_fraction = 0.2


def write_dict(dict, filename):
    with open(filename, 'w') as f:
        for inner_dict in dict.values():
            for K, V in inner_dict.items():
                f.write(str(K) + "\t" + str(V) + "\n")


def garbage_time():
    print('ENTERING GARBAGE TIME')
    globals.config['DEFAULT']['exp_type'] = 'target'
    globals.config['DEFAULT']['channel_dim'] = 'one'
    train_set, val_set, test_set = get_train_val_test(data_folder, 1, 0)
    garbageNAS = NaiveNAS(iterator=iterator, exp_folder=exp_folder,
                        train_set=train_set, val_set=val_set, test_set=test_set,
                        stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                        config=globals.config, subject_id=1, fieldnames=None)
    garbageNAS.garbage_time()


def get_configurations():
    configurations = []
    default_config = globals.init_config._defaults
    evolution_config = globals.init_config._sections['evolution']
    for key in default_config.keys():
        default_config[key] = json.loads(default_config[key])
    for key in evolution_config.keys():
        evolution_config[key] = json.loads(evolution_config[key])
    both_configs = list(default_config.values())
    both_configs.extend(list(evolution_config.values()))
    all_configs = list(product(*both_configs))
    for config_index in range(len(all_configs)):
        configurations.append({'DEFAULT': OrderedDict([]), 'evolution': OrderedDict([])})
        i = 0
        for key in default_config.keys():
            configurations[config_index]['DEFAULT'][key] = all_configs[config_index][i]
            i += 1
        for key in evolution_config.keys():
            configurations[config_index]['evolution'][key] = all_configs[config_index][i]
            i += 1
    return configurations


def target_exp(model_from_file=None):
    fieldnames = ['subject', 'generation', 'train_acc', 'val_acc', 'test_acc', 'train_time']
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    start_time = time.time()
    for subject_id in subjects:
        train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
        naiveNAS = NaiveNAS(iterator=iterator, exp_folder=exp_folder,
                            train_set=train_set, val_set=val_set, test_set=test_set,
                            stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                            config=globals.config, subject_id=subject_id, fieldnames=fieldnames,
                            model_from_file=model_from_file)
        naiveNAS.run_target_model(csv_file)
    globals.config['DEFAULT']['total_time'] = str(time.time() - start_time)
    write_dict(dict=globals.config, filename=str(exp_folder) + '/final_config.ini')


def per_subject_exp():
    fieldnames = ['subject', 'generation', 'train_acc', 'val_acc', 'test_acc', 'train_time', 'unique_models',
                  'unique_genomes', 'average_conv_width', 'average_conv_height', 'average_conv_filters',
                  'average_pool_width', 'average_pool_stride', 'DropoutLayer_count', 'BatchNormLayer_count',
                  'ActivationLayer_count', 'ConvLayer_count', 'PoolingLayer_count', 'IdentityLayer_count']
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    start_time = time.time()
    for subject_id in subjects:
        train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
        naiveNAS = NaiveNAS(iterator=iterator, exp_folder=exp_folder,
                            train_set=train_set, val_set=val_set, test_set=test_set,
                            stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                            config=globals.config, subject_id=subject_id, fieldnames=fieldnames)
        evolution_file = '%s/subject_%d_archs.txt' % (exp_folder, subject_id)
        if globals.config['DEFAULT']['exp_type'] == 'evolution_layers':
            naiveNAS.evolution_layers(csv_file, evolution_file)
        elif globals.config['DEFAULT']['exp_type'] == 'evolution_filters':
            naiveNAS.evolution_filters(csv_file, evolution_file)
    globals.config['DEFAULT']['total_time'] = str(time.time() - start_time)
    write_dict(dict=globals.config, filename=str(exp_folder) + '/final_config.ini')


def cross_subject_exp():
    fieldnames = ['subject', 'generation', 'train_acc', 'val_acc', 'test_acc', 'train_time']
    for subject in range(1, globals.config['DEFAULT']['num_subjects'] + 1):
        fieldnames.append('%d_train_acc' % subject)
        fieldnames.append('%d_val_acc' % subject)
        fieldnames.append('%d_test_acc' % subject)
        fieldnames.append('%d_train_time' % subject)
    fieldnames.extend(['unique_models', 'unique_genomes', 'average_conv_width', 'average_conv_height', 'average_conv_filters',
                        'average_pool_width', 'average_pool_stride', 'DropoutLayer_count', 'BatchNormLayer_count',
                       'ActivationLayer_count', 'ConvLayer_count', 'PoolingLayer_count', 'IdentityLayer_count'])
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    start_time = time.time()
    train_set_all = []
    val_set_all = []
    test_set_all = []
    for subject_id in range(1,10):
        train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
        train_set_all.append(train_set)
        val_set_all.append(val_set)
        test_set_all.append(test_set)
    naiveNAS = NaiveNAS(iterator=iterator, exp_folder=exp_folder,
                        train_set=train_set_all, val_set=val_set_all, test_set=test_set_all,
                        stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                        config=globals.config, subject_id='all', fieldnames=fieldnames)
    evolution_file = '%s/archs.txt' % (exp_folder)
    if globals.config['DEFAULT']['exp_type'] == 'evolution_layers':
        naiveNAS.evolution_layers_all(csv_file, evolution_file)
    elif globals.config['DEFAULT']['exp_type'] == 'target':
        naiveNAS.run_target_model(csv_file)
    globals.config['DEFAULT']['total_time'] = str(time.time() - start_time)
    write_dict(dict=globals.config, filename=str(exp_folder) + '/final_config.ini')

subdirs = [x for x in os.walk('results')]
if len(subdirs) == 1:
    exp_id = 1
else:
    try:
        subdir_names = [int(x[0].split('/')[1].split('_')[0][0:]) for x in subdirs[1:]]
    except IndexError:
        subdir_names = [int(x[0].split('\\')[1].split('_')[0][0:]) for x in subdirs[1:]]
    subdir_names.sort()
    exp_id = subdir_names[-1] + 1

configurations = get_configurations()
try:
    for index, configuration in enumerate(configurations):
        try:
            globals.set_config(configuration)
            if platform.node() == 'nvidia':
                globals.config['DEFAULT']['cuda'] = True
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            stop_criterion = Or([MaxEpochs(globals.config['DEFAULT']['max_epochs']),
                                 NoDecrease('valid_misclass', globals.config['DEFAULT']['max_increase_epochs'])])
            if globals.config['DEFAULT']['cropping']:
                globals.config['DEFAULT']['input_time_len'] = globals.config['DEFAULT']['input_time_cropping']
                iterator = CropsFromTrialsIterator(batch_size=globals.config['DEFAULT']['batch_size'],
                                                   input_time_length=globals.config['DEFAULT']['input_time_len'],
                                                   n_preds_per_input=globals.config['DEFAULT']['n_preds_per_input'])
                loss_function = lambda preds, targets: F.nll_loss(torch.mean(preds, dim=2, keepdim=False), targets)
                monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                            CroppedTrialMisclassMonitor(
                                input_time_length=globals.config['DEFAULT']['input_time_len']), RuntimeMonitor()]
            else:
                iterator = BalancedBatchSizeIterator(batch_size=globals.config['DEFAULT']['batch_size'])
                loss_function = F.nll_loss
                monitors = [LossMonitor(), MisclassMonitor(), RuntimeMonitor()]
            if type(globals.config['evolution']['subjects_to_check']) == list:
                subjects = globals.config['evolution']['subjects_to_check']
            else:
                subjects = random.sample(range(1, globals.config['DEFAULT']['num_subjects']),
                                         globals.config['evolution']['subjects_to_check'])
            exp_folder = 'results/' + str(exp_id) + '_' + str(index+1) + '_' + globals.config['DEFAULT']['exp_type']
            createFolder(exp_folder)
            write_dict(dict=globals.config, filename=str(exp_folder) + '/config.ini')
            csv_file = exp_folder + '/' + str(exp_id) + '_' + str(index+1) + '_'  + globals.config['DEFAULT']['exp_type'] + '.csv'
            if globals.config['DEFAULT']['exp_type'] == 'target':
                target_exp()
            elif globals.config['DEFAULT']['exp_type'] == 'from_file':
                target_exp(model_from_file=sys.argv[1])
            elif globals.config['DEFAULT']['cross_subject']:
                cross_subject_exp()
            else:
                per_subject_exp()
        except Exception as e:
            with open(exp_folder + "/error_log.txt", "w") as err_file:
                print('experiment failed. Exception message: %s' % (str(e)), file=err_file)
                print(traceback.format_exc(), file=err_file)
            new_exp_folder = exp_folder + '_fail'
            os.rename(exp_folder, new_exp_folder)
            write_dict(dict=globals.config, filename=str(new_exp_folder) + '/final_config.ini')
finally:
    garbage_time()

