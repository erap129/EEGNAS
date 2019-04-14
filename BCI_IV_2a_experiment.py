import os
import platform
import re

import pandas as pd
from collections import OrderedDict, defaultdict
from itertools import product, chain
import torch.nn.functional as F
import torch
from data_preprocessing import get_train_val_test
from naiveNAS import NaiveNAS
from braindecode.experiments.stopcriteria import MaxEpochs, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator, CropsFromTrialsIterator
from braindecode.experiments.monitors import LossMonitor, RuntimeMonitor
from globals import init_config
from utils import createFolder, GenericMonitor, NoIncrease, CroppedTrialGenericMonitor,\
    acc_func, kappa_func, auc_func, f1_func
from argparse import ArgumentParser
import logging
import globals
import random
import sys
import csv
import time
import json
import code, traceback, signal
from os import listdir
from os.path import isfile, join
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
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
    parser.add_argument("-e", "--experiment", help="experiment type", default='tests')
    parser.add_argument("-m", "--model", help="path to Pytorch model file")
    parser.add_argument("-g", "--garbage", help="Use garbage time", default='f')
    parser.add_argument("-d", "--drive", help="Save results to google drive", default='f')
    return parser.parse_args(args)


def generate_report(filename, report_filename):
    params = ['final', 'from_file']
    params_to_average = defaultdict(float)
    avg_count = defaultdict(int)
    data = pd.read_csv(filename)
    for param in params:
        for index, row in data.iterrows():
            if param in row['param_name'] and 'raw' not in row['param_name'] and 'target' not in row['param_name']:
                row_param = row['param_name']
                intro = re.compile('\d_')
                if intro.match(row_param):
                    row_param = row_param[2:]
                outro = row_param.find('from_file')
                if outro != -1:
                    row_param = row_param[outro:]
                params_to_average[row_param] += float(row['param_value'])
                avg_count[row_param] += 1
    for key, value in params_to_average.items():
        params_to_average[key] = params_to_average[key] / avg_count[key]
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
                         NoIncrease(f'valid_{globals.get("nn_objective")}', globals.get('max_increase_epochs'))])
    iterator = BalancedBatchSizeIterator(batch_size=globals.get('batch_size'))
    monitors = [LossMonitor(), GenericMonitor('accuracy', acc_func), RuntimeMonitor()]
    loss_function = F.nll_loss
    if globals.get('dataset') in ['NER15', 'Cho', 'BCI_IV_2b', 'Bloomberg']:
        monitors.append(GenericMonitor('auc', auc_func))
    if globals.get('dataset') in ['BCI_IV_2b']:
        monitors.append(GenericMonitor('kappa', kappa_func))
    if globals.get('dataset') in ['Opportunity']:
        monitors.append(GenericMonitor('f1', f1_func))
    return stop_criterion, iterator, loss_function, monitors


def get_cropped_settings():
    stop_criterion = Or([MaxEpochs(globals.get('max_epochs')),
                         NoIncrease('valid_accuracy', globals.get('max_increase_epochs'))])
    iterator = CropsFromTrialsIterator(batch_size=globals.get('batch_size'),
                                       input_time_length=globals.get('input_time_len'),
                                       n_preds_per_input=globals.get('n_preds_per_input'))
    loss_function = lambda preds, targets: F.nll_loss(torch.mean(preds, dim=2, keepdim=False), targets)
    monitors = [LossMonitor(), GenericMonitor('accuracy', acc_func),
                CroppedTrialGenericMonitor('accuracy', acc_func,
                    input_time_length=globals.get('input_time_len')), RuntimeMonitor()]
    if globals.get('dataset') in ['NER15', 'Cho', 'BCI_IV_2b', 'SonarSub']:
        monitors.append(CroppedTrialGenericMonitor('auc', auc_func,
                    input_time_length=globals.get('input_time_len')))
    if globals.get('dataset') in ['BCI_IV_2b']:
        monitors.append(CroppedTrialGenericMonitor('kappa', kappa_func,
                    input_time_length=globals.get('input_time_len')))
    return stop_criterion, iterator, loss_function, monitors


def garbage_time():
    print('ENTERING GARBAGE TIME')
    stop_criterion, iterator, loss_function, monitors = get_normal_settings()
    loss_function = F.nll_loss
    globals.set('dataset', 'BCI_IV_2a')
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


def get_configurations(experiment):
    configurations = []
    default_config = globals.configs._defaults
    exp_config = globals.configs._sections[experiment]
    for key in default_config.keys():
        default_config[key] = json.loads(default_config[key])
    default_config['exp_name'] = [experiment]
    for key in exp_config.keys():
        exp_config[key] = json.loads(exp_config[key])
    both_configs = list(default_config.values())
    both_configs.extend(list(exp_config.values()))
    config_keys = list(default_config.keys())
    config_keys.extend(list(exp_config.keys()))
    all_configs = list(product(*both_configs))
    for config_index in range(len(all_configs)):
        configurations.append({'DEFAULT': OrderedDict([]), experiment: OrderedDict([])})
        i = 0
        for key in default_config.keys():
            configurations[config_index]['DEFAULT'][key] = all_configs[config_index][i]
            i += 1
        for key in exp_config.keys():
            configurations[config_index][experiment][key] = all_configs[config_index][i]
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


def not_exclusively_in(subj, model_from_file):
    all_subjs = [int(s) for s in model_from_file.split() if s.isdigit()]
    if len(all_subjs) > 1:
        return False
    if subj in all_subjs:
        return False
    return True


def target_exp(stop_criterion, iterator, loss_function, model_from_file=None):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    for subject_id in subjects:
        if model_from_file is not None and globals.get('per_subject_exclusive') and \
                not_exclusively_in(subject_id, model_from_file):
            continue
        train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
        naiveNAS = NaiveNAS(iterator=iterator, exp_folder=exp_folder, exp_name = exp_name,
                            train_set=train_set, val_set=val_set, test_set=test_set,
                            stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                            config=globals.config, subject_id=subject_id, fieldnames=fieldnames,
                            strategy='per_subject', evolution_file=None, csv_file=csv_file,
                            model_from_file=model_from_file)
        naiveNAS.run_target_model()


def per_subject_exp(subjects, stop_criterion, iterator, loss_function):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    for subject_id in subjects:
        train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
        evolution_file = '%s/subject_%d_archs.txt' % (exp_folder, subject_id)
        naiveNAS = NaiveNAS(iterator=iterator, exp_folder=exp_folder, exp_name=exp_name,
                            train_set=train_set, val_set=val_set, test_set=test_set,
                            stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                            config=globals.config, subject_id=subject_id, fieldnames=fieldnames, strategy='per_subject',
                            evolution_file=evolution_file, csv_file=csv_file)
        naiveNAS.evolution()


def cross_subject_exp(stop_criterion, iterator, loss_function):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    train_set_all = {}
    val_set_all = {}
    test_set_all = {}
    for subject_id in range(1, globals.get('num_subjects')+1):
        train_set, val_set, test_set = get_train_val_test(data_folder, subject_id, low_cut_hz)
        train_set_all[subject_id] = train_set
        val_set_all[subject_id] = val_set
        test_set_all[subject_id] = test_set
    evolution_file = '%s/archs.txt' % (exp_folder)
    naiveNAS = NaiveNAS(iterator=iterator, exp_folder=exp_folder, exp_name = exp_name,
                        train_set=train_set_all, val_set=val_set_all, test_set=test_set_all,
                        stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                        config=globals.config, subject_id='all', fieldnames=fieldnames, strategy='cross_subject',
                        evolution_file=evolution_file, csv_file=csv_file)
    if globals.get('exp_type') == 'evolution_layers':
        naiveNAS.evolution()
    elif globals.get('exp_type') == 'target':
        naiveNAS.run_target_model(csv_file)


def set_params_by_dataset():
    num_subjects = {'HG': 14, 'BCI_IV_2a': 9, 'BCI_IV_2b': 9, 'NER15': 1, 'Cho': 52, 'Bloomberg': 1, 'NYSE': 1,
                    'HumanActivity': 19, 'Opportunity': 1, 'SonarSub': 1}
    cross_subject_sampling_rate = num_subjects
    eeg_chans = {'HG': 44, 'BCI_IV_2a': 22, 'BCI_IV_2b': 3, 'NER15': 56, 'Cho': 64, 'Bloomberg': 32, 'NYSE': 5,
                 'HumanActivity': 45, 'Opportunity': 113, 'SonarSub': 100}
    input_time_len = {'HG': 1125, 'BCI_IV_2a': 1125, 'BCI_IV_2b': 1126, 'NER15': 260, 'Cho': 1537, 'Bloomberg': 950,
                      'NYSE': 200, 'HumanActivity': 124, 'Opportunity': 128, 'SonarSub': 100}
    n_classes = {'HG': 4, 'BCI_IV_2a': 4, 'BCI_IV_2b': 2, 'NER15': 2, 'Cho': 2, 'Bloomberg': 3, 'NYSE': 2,
                 'HumanActivity': 8, 'Opportunity': 18, 'SonarSub': 2}
    subjects_to_check = {'HG': list(range(1,14+1)), 'BCI_IV_2a': list(range(1,9+1)), 'BCI_IV_2b': list(range(1,9+1)),
                            'NER15': [1], 'Cho': [[1,2,3,4,5,6,7,8,9,10,
                            11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,39,40,
                            41,42,43,44,45,47,48,50,51,52]], 'Bloomberg': [1], 'NYSE': [1],
                            'HumanActivity': list(range(1,19+1)), 'Opportunity': [1], 'SonarSub': [1]}
    evaluation_metrics = {'HG': ['accuracy'], 'BCI_IV_2a': ['accuracy'], 'BCI_IV_2b': ["accuracy", "auc", "kappa"],
                          'NER15': ["accuracy", "auc"], 'Cho': ['accuracy'], 'Bloomberg': ["accuracy", "auc"],
                          'NYSE': ['accuracy'], 'HumanActivity': ['accuracy'], 'Opportunity': ["accuracy", "f1"],
                          'SonarSub': ["accuracy"]}
    ga_objective = {'HG': 'acc', 'BCI_IV_2a': 'acc', 'BCI_IV_2b': "kappa",
                          'NER15': "auc", 'Cho': 'acc', 'Bloomberg': "auc",
                          'NYSE': 'acc', 'HumanActivity': 'acc', 'Opportunity': "f1", 'SonarSub': 'acc'}
    nn_objective = {'HG': 'accuracy', 'BCI_IV_2a': 'accuracy', 'BCI_IV_2b': "kappa",
                          'NER15': "auc", 'Cho': 'accuracy', 'Bloomberg': "auc",
                          'NYSE': 'accuracy', 'HumanActivity': 'accuracy', 'Opportunity': "f1", 'SonarSub': 'accuracy'}
    globals.set('num_subjects', num_subjects[globals.get('dataset')])
    globals.set('cross_subject_sampling_rate', cross_subject_sampling_rate[globals.get('dataset')])
    globals.set('eeg_chans', eeg_chans[globals.get('dataset')])
    globals.set('input_time_len', input_time_len[globals.get('dataset')])
    globals.set('n_classes', n_classes[globals.get('dataset')])
    globals.set_if_not_exists('subjects_to_check', subjects_to_check[globals.get('dataset')])
    globals.set('evaluation_metrics', evaluation_metrics[globals.get('dataset')])
    globals.set('ga_objective', ga_objective[globals.get('dataset')])
    globals.set('nn_objective', nn_objective[globals.get('dataset')])
    if globals.get('dataset') == 'Cho':
        globals.set('exclude_subjects', [32, 46, 49])
    if globals.get('ensemble_iterations'):
        globals.set('evaluation_metrics', globals.get('evaluation_metrics') + ['raw', 'target'])


def connect_to_gdrive():
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.CommandLineAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")
    drive = GoogleDrive(gauth)
    return drive


def upload_exp_to_gdrive(fold_names, first_dataset):
    ind = fold_names[0].find('_')
    end_ind = fold_names[0].rfind(first_dataset)
    base_folder_name = list(fold_names[0])
    base_folder_name[ind + 1] = 'x'
    base_folder_name = base_folder_name[:end_ind-1]
    base_folder_name = ''.join(base_folder_name)
    drive = connect_to_gdrive()
    base_folder = drive.CreateFile({'title': base_folder_name,
                                   'parents': [{"id": '1z6y-g4HqmQm7i8R2h66sDd5e6AV1IhVM'}],
                                    'mimeType': "application/vnd.google-apps.folder"})
    base_folder.Upload()
    for folder in fold_names:
        full_folder = 'results/' + folder
        spec_folder = drive.CreateFile({'title': folder,
                                        'parents': [{"id": base_folder['id']}],
                                        'mimeType': "application/vnd.google-apps.folder"})
        spec_folder.Upload()
        files = [f for f in listdir(full_folder) if isfile(join(full_folder, f))]
        for filename in files:
            if '.p' not in filename:
                file_drive = drive.CreateFile({'title': filename,
                                                   'parents': [{"id": spec_folder['id']}]})
                file_drive.SetContentFile(str(join(full_folder, filename)))
                file_drive.Upload()


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    init_config(args.config)
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

    try:
        experiments = args.experiment.split(',')
        folder_names = []
        first_run = True
        for experiment in experiments:
            configurations = get_configurations(experiment)
            multiple_values = get_multiple_values(configurations)
            for index, configuration in enumerate(configurations):
                try:
                    globals.set_config(configuration)
                    set_params_by_dataset()
                    if first_run:
                        first_dataset = globals.get('dataset')
                        first_run = False
                    if (platform.node() == 'nvidia' or platform.node() == 'GPU' or platform.node() == 'rbc-gpu' or platform.node() == 'csgpusrv2')\
                            and not globals.get('force_gpu_off'):
                        globals.set('cuda', True)
                        os.environ["CUDA_VISIBLE_DEVICES"] = globals.get('gpu_select')
                        os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
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
                    exp_name = f"{exp_id}_{index+1}_{experiment}_{globals.get('dataset')}"
                    exp_folder = f"results/{exp_name}"
                    createFolder(exp_folder)
                    folder_names.append(exp_name)
                    write_dict(globals.config, f"{exp_folder}/config_{exp_name}.ini")
                    csv_file = f"{exp_folder}/{exp_name}.csv"
                    report_file = f"{exp_folder}/report_{exp_name}.csv"
                    fieldnames = ['exp_name', 'subject', 'generation', 'param_name', 'param_value']
                    if 'cross_subject' in multiple_values and not globals.get('cross_subject'):
                        globals.set('num_generations', globals.get('num_generations') *
                                    globals.get('cross_subject_compensation_rate'))
                    start_time = time.time()
                    if globals.get('exp_type') in ['target', 'benchmark']:
                        target_exp(stop_criterion, iterator, loss_function)
                    elif globals.get('exp_type') == 'from_file':
                        target_exp(stop_criterion, iterator, loss_function,
                                   model_from_file=f"models/{globals.get('models_dir')}/{globals.get('model_file_name')}")
                    elif globals.get('cross_subject'):
                        cross_subject_exp(stop_criterion, iterator, loss_function)
                    else:
                        per_subject_exp(subjects, stop_criterion, iterator, loss_function)
                    globals.set('total_time', str(time.time() - start_time))
                    write_dict(globals.config, f"{exp_folder}/final_config_{exp_name}.ini")
                    generate_report(csv_file, report_file)
                except Exception as e:
                    with open(f"{exp_folder}/error_log_{exp_name}.txt", "w") as err_file:
                        print('experiment failed. Exception message: %s' % (str(e)), file=err_file)
                        print(traceback.format_exc(), file=err_file)
                    print('experiment failed. Exception message: %s' % (str(e)))
                    print(traceback.format_exc())
                    new_exp_folder = exp_folder + '_fail'
                    os.rename(exp_folder, new_exp_folder)
                    write_dict(globals.config, f"{new_exp_folder}/final_config_{exp_name}.ini")
    finally:
        if args.drive == 't':
            upload_exp_to_gdrive(folder_names, first_dataset)
        if args.garbage == 't':
            garbage_time()
