#%%
import os
import platform
import torch
import pandas as pd
from collections import OrderedDict, defaultdict
from itertools import product
import torch.nn.functional as F
import torch
from data_preprocessing import get_train_val_test
from naiveNAS import NaiveNAS
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator, CropsFromTrialsIterator
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor
from globals import init_config
from utils import createFolder, AUCMonitor, AccuracyMonitor, NoIncrease, CroppedTrialAccuracyMonitor
from itertools import chain
from argparse import ArgumentParser
import logging
import globals
import random
import sys
import csv
import time
import json
import code, traceback, signal

global data_folder, valid_set_fraction, config

def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)


def listen():
    if sys.platform == "linux" or sys.platform == "linux2":
        signal.signal(signal.SIGUSR1, debug)  # Register handler


def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to configuration file", default='configurations/config.ini')
    parser.add_argument("-e", "--experiment", help="experiment type", default='benchmark')
    parser.add_argument("-m", "--model", help="path to Pytorch model file")
    return parser.parse_args(args)


def generate_report(filename, report_filename):
    params = ['train_time', 'test_acc', 'val_acc', 'train_acc', 'num_epochs', 'final_test_acc', 'final_val_acc', 'final_train_acc']
    params_to_average = defaultdict(int)
    data = pd.read_csv(filename)
    for param in params:
        avg_count = 0
        for index, row in data.iterrows():
            if param in row['param_name']:
                params_to_average[param] += row['param_value']
                avg_count += 1
        if avg_count != 0:
            params_to_average[param] /= avg_count
    pd.DataFrame(params_to_average, index=[0]).to_csv(report_filename)


def write_dict(dict, filename):
    with open(filename, 'w') as f:
        all_keys = set()
        for inner_dict in dict.values():
            for K in inner_dict.keys():
                all_keys.add(K)
        for K in all_keys:
            f.write(f"{K}\t{globals.get(K)}\n")


def get_normal_settings():
    stop_criterion = Or([MaxEpochs(globals.get('max_epochs')),
                         NoIncrease('valid_accuracy', globals.get('max_increase_epochs'))])
    iterator = BalancedBatchSizeIterator(batch_size=globals.get('batch_size'))
    monitors = [LossMonitor(), AccuracyMonitor(), RuntimeMonitor()]
    if globals.get('dataset') in ['NER15', 'Cho']:
        loss_function = F.binary_cross_entropy
        monitors.append(AUCMonitor())
    else:
        loss_function = F.nll_loss
    return stop_criterion, iterator, loss_function, monitors


def get_cropped_settings():
    stop_criterion = Or([MaxEpochs(globals.get('max_epochs')),
                         NoIncrease('valid_accuracy', globals.get('max_increase_epochs'))])
    iterator = CropsFromTrialsIterator(batch_size=globals.get('batch_size'),
                                       input_time_length=globals.get('input_time_len'),
                                       n_preds_per_input=globals.get('n_preds_per_input'))
    loss_function = lambda preds, targets: F.nll_loss(torch.mean(preds, dim=2, keepdim=False), targets)
    monitors = [LossMonitor(), AccuracyMonitor(col_suffix='sample_accuracy'),
                CroppedTrialAccuracyMonitor(
                    input_time_length=globals.get('input_time_len')), RuntimeMonitor()]
    return stop_criterion, iterator, loss_function, monitors


def garbage_time():
    print('ENTERING GARBAGE TIME')
    stop_criterion, iterator, loss_function, monitors = get_normal_settings()
    loss_function = F.nll_loss
    args.experiment = 'target'
    globals.set('dataset', 'BCI_IV')
    globals.set('channel_dim', 'one')
    globals.set('input_time_len', 1125)
    globals.set('cropping', False)
    globals.set('num_subjects', 9)
    globals.set('eeg_chans', 22)
    globals.set('n_classes', 4)
    train_set, val_set, test_set = get_train_val_test(data_folder, 1, 0)
    garbageNAS = NaiveNAS(iterator=iterator, exp_folder=exp_folder, exp_name = exp_name,
                        train_set=train_set, val_set=val_set, test_set=test_set,
                        stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                        config=globals.config, subject_id=1, fieldnames=None)
    garbageNAS.garbage_time()


def get_configurations(args):
    configurations = []
    default_config = globals.init_config._defaults
    exp_config = globals.init_config._sections[args.experiment]
    for key in default_config.keys():
        default_config[key] = json.loads(default_config[key])
    default_config['exp_name'] = [args.experiment]
    for key in exp_config.keys():
        exp_config[key] = json.loads(exp_config[key])
    both_configs = list(default_config.values())
    both_configs.extend(list(exp_config.values()))
    config_keys = list(default_config.keys())
    config_keys.extend(list(exp_config.keys()))
    all_configs = list(product(*both_configs))
    for config_index in range(len(all_configs)):
        configurations.append({'DEFAULT': OrderedDict([]), args.experiment: OrderedDict([])})
        i = 0
        for key in default_config.keys():
            configurations[config_index]['DEFAULT'][key] = all_configs[config_index][i]
            i += 1
        for key in exp_config.keys():
            configurations[config_index][args.experiment][key] = all_configs[config_index][i]
            i += 1
    return configurations


def get_multiple_values(configurations):
    multiple_values = []
    value_count = defaultdict(list)
    for configuration in configurations:
        combined_config = OrderedDict(chain(*[x.items() for x in configuration.values()]))
        for key in combined_config.keys():
            if not combined_config[key] in value_count[key]:
                value_count[key].append(combined_config[key])
            if len(value_count[key]) > 1:
                multiple_values.append(key)
    return list(set(multiple_values))


def target_exp(stop_criterion, iterator, loss_function, model_from_file=None):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    for subject_id in subjects:
        train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
        naiveNAS = NaiveNAS(iterator=iterator, exp_folder=exp_folder, exp_name = exp_name,
                            train_set=train_set, val_set=val_set, test_set=test_set,
                            stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                            config=globals.config, subject_id=subject_id, fieldnames=fieldnames,
                            model_from_file=model_from_file)
        naiveNAS.run_target_model(csv_file)


def per_subject_exp(subjects, stop_criterion, iterator, loss_function):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    for subject_id in subjects:
        train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
        naiveNAS = NaiveNAS(iterator=iterator, exp_folder=exp_folder, exp_name=exp_name,
                            train_set=train_set, val_set=val_set, test_set=test_set,
                            stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                            config=globals.config, subject_id=subject_id, fieldnames=fieldnames)
        evolution_file = '%s/subject_%d_archs.txt' % (exp_folder, subject_id)
        naiveNAS.evolution_layers(csv_file, evolution_file)


def cross_subject_exp(subjects, stop_criterion, iterator, loss_function):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    train_set_all = {}
    val_set_all = {}
    test_set_all = {}
    # for subject_id in subjects:
    for subject_id in range(1, globals.get('num_subjects')+1):
        train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
        train_set_all[subject_id] = train_set
        val_set_all[subject_id] = val_set
        test_set_all[subject_id] = test_set
    naiveNAS = NaiveNAS(iterator=iterator, exp_folder=exp_folder, exp_name = exp_name,
                        train_set=train_set_all, val_set=val_set_all, test_set=test_set_all,
                        stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                        config=globals.config, subject_id='all', fieldnames=fieldnames)
    evolution_file = '%s/archs.txt' % (exp_folder)
    if globals.get('exp_type') == 'evolution_layers':
        naiveNAS.evolution_layers_all(csv_file, evolution_file)
    elif globals.get('exp_type') == 'target':
        naiveNAS.run_target_model(csv_file)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    init_config(args.config)
    num_subjects = {'HG': 14, 'BCI_IV': 9, 'NER15': 1, 'Cho': 52}
    eeg_chans = {'HG': 44, 'BCI_IV': 22, 'NER15': 56, 'Cho': 64}
    input_time_len = {'HG': 1125, 'BCI_IV': 1125, 'NER15': 260, 'Cho': 1537}
    n_classes = {'HG': 4, 'BCI_IV': 4, 'NER15': 2, 'Cho': 2}
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                            level=logging.DEBUG, stream=sys.stdout)

    data_folder = 'data/'
    low_cut_hz = 0
    valid_set_fraction = 0.2
    listen()

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

    configurations = get_configurations(args)
    multiple_values = get_multiple_values(configurations)
    try:
        for index, configuration in enumerate(configurations):
            try:
                globals.set_config(configuration)
                globals.set('num_subjects', num_subjects[globals.get('dataset')])
                globals.set('eeg_chans', eeg_chans[globals.get('dataset')])
                globals.set('input_time_len', input_time_len[globals.get('dataset')])
                globals.set('n_classes', n_classes[globals.get('dataset')])
                if (platform.node() == 'nvidia' or platform.node() == 'GPU' or platform.node() == 'rbc-gpu')\
                        and not globals.get('force_gpu_off'):
                    globals.set('cuda', True)
                    os.environ["CUDA_VISIBLE_DEVICES"] = globals.get('gpu_select')
                    assert torch.cuda.is_available(), "Cuda not available"
                if globals.get('cropping'):
                    globals.set('input_time_len', globals.get('input_time_cropping'))
                    stop_criterion, iterator, loss_function, monitors = get_cropped_settings()
                else:
                    stop_criterion, iterator, loss_function, monitors = get_normal_settings()
                if type(globals.get('subjects_to_check')) == list:
                    subjects = globals.get('subjects_to_check')
                else:
                    subjects = random.sample(range(1, globals.get('num_subjects')),
                                             globals.get('subjects_to_check'))
                exp_name = f"{exp_id}_{index+1}_{args.experiment}"
                exp_folder = 'results/' + exp_name
                createFolder(exp_folder)
                write_dict(globals.config, f"{exp_folder}/config_{exp_name}.ini")
                csv_file = f"{exp_folder}/{exp_name}.csv"
                report_file = f"{exp_folder}/report_{exp_name}.csv"
                fieldnames = ['exp_name', 'subject', 'generation', 'param_name', 'param_value']
                if 'cross_subject' in multiple_values and not globals.get('cross_subject'):
                    globals.set('num_generations', globals.get('num_generations') *
                                globals.get('cross_subject_compensation_rate'))
                    # make num of generations equal for cross and per subject
                start_time = time.time()
                if globals.get('exp_type') in ['target', 'benchmark']:
                    target_exp(stop_criterion, iterator, loss_function)
                elif globals.get('exp_type') == 'from_file':
                    target_exp(stop_criterion, iterator, loss_function,
                               model_from_file=f"models/{args.experiment}/{globals.get('model_file_name')}")
                elif globals.get('cross_subject'):
                    cross_subject_exp(subjects, stop_criterion, iterator, loss_function)
                else:
                    per_subject_exp(subjects, stop_criterion, iterator, loss_function)
                globals.set('total_time', str(time.time() - start_time))
                write_dict(globals.config, f"{exp_folder}/final_config_{exp_name}.ini")
                generate_report(csv_file, report_file)
            except Exception as e:
                with open(f"{exp_folder}/error_log_{exp_name}.txt", "w") as err_file:
                    print('experiment failed. Exception message: %s' % (str(e)), file=err_file)
                    print(traceback.format_exc(), file=err_file)
                new_exp_folder = exp_folder + '_fail'
                os.rename(exp_folder, new_exp_folder)
                write_dict(globals.config, f"{new_exp_folder}/final_config_{exp_name}.ini")
    finally:
        garbage_time()



