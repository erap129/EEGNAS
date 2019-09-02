import os
import re
import shutil
from evolution.loaded_model_evaluations import EEGNAS_from_file
from utilities.data_utils import write_dict
from utilities.gdrive import upload_exp_to_gdrive
from utilities.config_utils import config_to_dict, get_configurations, get_multiple_values, set_params_by_dataset, \
    set_gpu, set_seeds
import torch.nn.functional as F
import torch
from data_preprocessing import get_train_val_test, get_pure_cross_subject
from braindecode.experiments.stopcriteria import MaxEpochs, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator, CropsFromTrialsIterator
from braindecode.experiments.monitors import LossMonitor, RuntimeMonitor
from global_vars import init_config
from utilities.report_generation import add_params_to_name, generate_report
from utilities.misc import createFolder, get_oper_by_loss_function, exit_handler, listen, not_exclusively_in
from utilities.monitors import *
from evolution.genetic_algorithm import EEGNAS_evolution
from argparse import ArgumentParser
import logging
import global_vars
import random
import sys
import csv
import time
import code, traceback, signal
global data_folder, valid_set_fraction
import atexit


def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to configuration file", default='configurations/config.ini')
    parser.add_argument("-e", "--experiment", help="experiment type", default='tests')
    parser.add_argument("-m", "--model", help="path to Pytorch model file")
    parser.add_argument("-g", "--garbage", help="Use garbage time", default='f')
    parser.add_argument("-d", "--drive", help="Save results to google drive", default='f')
    parser.add_argument("-dm", "--debug_mode", action='store_true', help="debug mode, don't save results to disk")
    return parser.parse_args(args)


def get_normal_settings():
    if global_vars.get('problem') == 'regression':
        loss_function = F.mse_loss
    else:
        loss_function = F.nll_loss
    stop_criterion = Or([MaxEpochs(global_vars.get('max_epochs')),
                         NoIncreaseDecrease(f'valid_{global_vars.get("nn_objective")}', global_vars.get('max_increase_epochs'),
                                            oper=get_oper_by_loss_function(loss_function))])
    iterator = BalancedBatchSizeIterator(batch_size=global_vars.get('batch_size'))
    monitors = [LossMonitor(), GenericMonitor('accuracy'), RuntimeMonitor()]
    for metric in global_vars.get('evaluation_metrics'):
        monitors.append(GenericMonitor(metric))
    return stop_criterion, iterator, loss_function, monitors


def get_cropped_settings():
    stop_criterion = Or([MaxEpochs(global_vars.get('max_epochs')),
                         NoIncreaseDecrease(f'valid_{global_vars.get("nn_objective")}', global_vars.get('max_increase_epochs'))])
    iterator = CropsFromTrialsIterator(batch_size=global_vars.get('batch_size'),
                                       input_time_length=global_vars.get('input_time_len'),
                                       n_preds_per_input=global_vars.get('n_preds_per_input'))
    loss_function = lambda preds, targets: F.nll_loss(torch.mean(preds, dim=2, keepdim=False), targets)
    monitors = [LossMonitor(), GenericMonitor('accuracy', acc_func),
                CroppedTrialGenericMonitor('accuracy', acc_func,
                                           input_time_length=global_vars.get('input_time_len')), RuntimeMonitor()]
    if global_vars.get('dataset') in ['NER15', 'Cho', 'SonarSub']:
        monitors.append(CroppedTrialGenericMonitor('auc', auc_func,
                                                   input_time_length=global_vars.get('input_time_len')))
    if global_vars.get('dataset') in ['BCI_IV_2b']:
        monitors.append(CroppedGenericMonitorPerTimeStep('kappa', kappa_func,
                                                         input_time_length=global_vars.get('input_time_len')))
    return stop_criterion, iterator, loss_function, monitors


def target_exp(subjects, stop_criterion, iterator, loss_function, model_from_file=None, write_header=True):
    if write_header:
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    for subject_id in subjects:
        train_set = {}
        val_set = {}
        test_set = {}
        if model_from_file is not None and global_vars.get('per_subject_exclusive') and \
                not_exclusively_in(subject_id, model_from_file):
            continue
        train_set[subject_id], val_set[subject_id], test_set[subject_id] =\
            get_train_val_test(data_folder, subject_id)
        eegnas_from_file = EEGNAS_from_file(iterator=iterator, exp_folder=exp_folder, exp_name = exp_name,
                            train_set=train_set, val_set=val_set, test_set=test_set,
                            stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                            subject_id=subject_id, fieldnames=fieldnames,
                            csv_file=csv_file, model_from_file=model_from_file)
        if global_vars.get('weighted_population_file'):
            eegnas_from_file.run_target_ensemble()
        else:
            eegnas_from_file.run_target_model()


def per_subject_exp(subjects, stop_criterion, iterator, loss_function):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    for subject_id in subjects:
        train_set = {}
        val_set = {}
        test_set = {}
        if global_vars.get('pure_cross_subject'):
            train_set[subject_id], val_set[subject_id], test_set[subject_id] =\
                get_pure_cross_subject(data_folder)
        else:
            train_set[subject_id], val_set[subject_id], test_set[subject_id] =\
                get_train_val_test(data_folder, subject_id)
        evolution_file = '%s/subject_%d_archs.txt' % (exp_folder, subject_id)
        eegnas = EEGNAS_evolution(iterator=iterator, exp_folder=exp_folder, exp_name=exp_name,
                            train_set=train_set, val_set=val_set, test_set=test_set,
                            stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                            subject_id=subject_id, fieldnames=fieldnames, strategy='per_subject',
                            evolution_file=evolution_file, csv_file=csv_file)
        best_model_filename = eegnas.evolution()
        if global_vars.get('pure_cross_subject') or len(subjects) == 1:
            return [best_model_filename]


def leave_one_out_exp(subjects, stop_criterion, iterator, loss_function):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    best_model_filenames = []
    for subject_id in subjects:
        train_set = {}
        val_set = {}
        test_set = {}
        train_set[subject_id], val_set[subject_id], test_set[subject_id] =\
            get_pure_cross_subject(data_folder, exclude=[subject_id])
        evolution_file = '%s/subject_%d_archs.txt' % (exp_folder, subject_id)
        eegnas = EEGNAS_evolution(iterator=iterator, exp_folder=exp_folder, exp_name=exp_name,
                            train_set=train_set, val_set=val_set, test_set=test_set,
                            stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                            subject_id=subject_id, fieldnames=fieldnames, strategy='per_subject',
                            evolution_file=evolution_file, csv_file=csv_file)
        best_model_filename = eegnas.evolution()
        best_model_filenames.append(best_model_filename)
    return best_model_filenames


def cross_subject_exp(stop_criterion, iterator, loss_function):
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    train_set_all = {}
    val_set_all = {}
    test_set_all = {}
    for subject_id in range(1, global_vars.get('num_subjects') + 1):
        train_set, val_set, test_set = get_train_val_test(data_folder, subject_id)
        train_set_all[subject_id] = train_set
        val_set_all[subject_id] = val_set
        test_set_all[subject_id] = test_set
    evolution_file = '%s/archs.txt' % (exp_folder)
    naiveNAS = EEGNAS_evolution(iterator=iterator, exp_folder=exp_folder, exp_name = exp_name,
                        train_set=train_set_all, val_set=val_set_all, test_set=test_set_all,
                        stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                        config=global_vars.config, subject_id='all', fieldnames=fieldnames, strategy='cross_subject',
                        evolution_file=evolution_file, csv_file=csv_file)
    return naiveNAS.evolution()


def get_settings():
    if global_vars.get('cropping'):
        global_vars.set('original_input_time_len', global_vars.get('input_time_len'))
        global_vars.set('input_time_len', global_vars.get('input_time_cropping'))
        stop_criterion, iterator, loss_function, monitors = get_cropped_settings()
    else:
        stop_criterion, iterator, loss_function, monitors = get_normal_settings()
    return stop_criterion, iterator, loss_function, monitors


def get_exp_id():
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
    return exp_id


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    init_config(args.config)
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                            level=logging.DEBUG, stream=sys.stdout)
    data_folder = 'data/'
    low_cut_hz = 0
    valid_set_fraction = 0.2
    listen()
    exp_id = get_exp_id()
    exp_funcs = {'cross_subject': cross_subject_exp,
                   'per_subject': per_subject_exp,
                    'leave_one_out': leave_one_out_exp}
    try:
        experiments = args.experiment.split(',')
        folder_names = []
        first_run = True
        for experiment in experiments:
            configurations = get_configurations(experiment, global_vars.configs)
            multiple_values = get_multiple_values(configurations)
            for index, configuration in enumerate(configurations):
                try:
                    global_vars.set_config(configuration)
                    set_params_by_dataset('configurations/dataset_params.ini')
                    if first_run:
                        first_dataset = global_vars.get('dataset')
                        if global_vars.get('include_params_folder_name'):
                            multiple_values.extend(global_vars.get('include_params_folder_name'))
                        first_run = False
                    set_gpu()
                    set_seeds()
                    stop_criterion, iterator, loss_function, monitors = get_settings()
                    if type(global_vars.get('subjects_to_check')) == list:
                        subjects = global_vars.get('subjects_to_check')
                    else:
                        subjects = random.sample(range(1, global_vars.get('num_subjects')),
                                                 global_vars.get('subjects_to_check'))
                    exp_name = f"{exp_id}_{index+1}_{experiment}_{global_vars.get('dataset')}"
                    exp_name = add_params_to_name(exp_name, multiple_values)
                    exp_folder = f"results/{exp_name}"
                    atexit.register(exit_handler, exp_folder)
                    createFolder(exp_folder)
                    folder_names.append(exp_name)
                    write_dict(global_vars.config, f"{exp_folder}/config_{exp_name}.ini")
                    csv_file = f"{exp_folder}/{exp_name}.csv"
                    report_file = f"{exp_folder}/report_{exp_name}.csv"
                    fieldnames = ['exp_name', 'machine', 'dataset', 'date', 'subject', 'generation', 'model', 'param_name', 'param_value']
                    if 'cross_subject' in multiple_values and not global_vars.get('cross_subject'):
                        global_vars.set('num_generations', global_vars.get('num_generations') *
                                        global_vars.get('cross_subject_compensation_rate'))
                    start_time = time.time()
                    if global_vars.get('exp_type') in ['target', 'benchmark']:
                        target_exp(stop_criterion, iterator, loss_function)
                    elif global_vars.get('exp_type') == 'from_file':
                        target_exp(stop_criterion, iterator, loss_function,
                                   model_from_file=f"models/{global_vars.get('models_dir')}/{global_vars.get('model_file_name')}")
                    else:
                        best_model_filenames = exp_funcs[global_vars.get('exp_type')]\
                            (subjects, stop_criterion, iterator, loss_function)
                        for best_model_filename in best_model_filenames:
                            target_exp(subjects, stop_criterion, iterator, loss_function,
                                       model_from_file=best_model_filename, write_header=False)
                    global_vars.set('total_time', str(time.time() - start_time))
                    write_dict(global_vars.config, f"{exp_folder}/final_config_{exp_name}.ini")
                    generate_report(csv_file, report_file)
                except Exception as e:
                    with open(f"error_logs/error_log_{exp_name}.txt", "w") as err_file:
                        print('experiment failed. Exception message: %s' % (str(e)), file=err_file)
                        print(traceback.format_exc(), file=err_file)
                    print('experiment failed. Exception message: %s' % (str(e)))
                    print(traceback.format_exc())
                    shutil.rmtree(exp_folder)
                    write_dict(global_vars.config, f"error_logs/final_config_{exp_name}.ini")
                    folder_names.remove(exp_name)
    finally:
        if args.drive == 't':
            upload_exp_to_gdrive(folder_names, first_dataset)
