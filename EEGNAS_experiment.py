import os
import platform
import re
import shutil
from collections import defaultdict
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from sshtunnel import SSHTunnelForwarder
from sacred import Experiment
from sacred.observers import MongoObserver
from EEGNAS.evolution.loaded_model_evaluations import EEGNAS_from_file
from EEGNAS.evolution.nn_training import TimeFrequencyBatchIterator
from EEGNAS.utilities.data_utils import write_dict
from EEGNAS.utilities.gdrive import upload_exp_to_gdrive
from EEGNAS.utilities.config_utils import config_to_dict, get_configurations, get_multiple_values, set_params_by_dataset, \
    set_gpu, set_seeds
import torch.nn.functional as F
import torch
from EEGNAS.data_preprocessing import get_train_val_test, get_pure_cross_subject, get_dataset
from braindecode.experiments.stopcriteria import MaxEpochs, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator, CropsFromTrialsIterator
from braindecode.experiments.monitors import LossMonitor, RuntimeMonitor
from EEGNAS.global_vars import init_config
from EEGNAS.utilities.report_generation import add_params_to_name, generate_report
from EEGNAS.utilities.misc import create_folder, get_oper_by_loss_function, exit_handler, listen, not_exclusively_in, \
    strfdelta, get_exp_id
from EEGNAS.utilities.monitors import *
from EEGNAS.evolution.genetic_algorithm import EEGNAS_evolution
from argparse import ArgumentParser
import logging
import sys
import pandas as pd
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)
from EEGNAS import global_vars
import random
import csv
import time
import code, traceback, signal
global data_folder, valid_set_fraction
import atexit

ex = Experiment()
FIRST_RUN = False
FIRST_DATASET = ''
FOLDER_NAMES = []
FIELDNAMES = ['exp_name', 'machine', 'dataset', 'date', 'subject', 'generation', 'model', 'param_name', 'param_value']

def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="path to configuration file", default='EEGNAS/configurations/config.ini')
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


def target_exp(exp_name, csv_file, subjects, model_from_file=None, write_header=True):
    stop_criterion, iterator, loss_function, monitors = get_settings()
    if write_header:
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
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
        eegnas_from_file = EEGNAS_from_file(iterator=iterator, exp_folder=f"results/{exp_name}", exp_name = exp_name,
                            train_set=train_set, val_set=val_set, test_set=test_set,
                            stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                            subject_id=subject_id, fieldnames=FIELDNAMES,
                            csv_file=csv_file, model_from_file=model_from_file)
        if global_vars.get('weighted_population_file'):
            eegnas_from_file.run_target_ensemble()
        else:
            eegnas_from_file.run_target_model()


def per_subject_exp(exp_name, csv_file, subjects):
    stop_criterion, iterator, loss_function, monitors = get_settings()
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
    for subject_id in subjects:
        train_set, val_set, test_set = {}, {}, {}
        if global_vars.get('pure_cross_subject'):
            dataset = get_dataset('all')
        else:
            dataset = get_dataset(subject_id)
        train_set[subject_id], val_set[subject_id], test_set[subject_id] = dataset['train'], dataset['valid'], dataset['test']
        evolution_file = f'results/{exp_name}/subject_{subject_id}_archs.txt'
        eegnas = EEGNAS_evolution(iterator=iterator, exp_folder=f"results/{exp_name}", exp_name=exp_name,
                            train_set=train_set, val_set=val_set, test_set=test_set,
                            stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                            subject_id=subject_id, fieldnames=FIELDNAMES, strategy='per_subject',
                            evolution_file=evolution_file, csv_file=csv_file)
        if global_vars.get('deap'):
            best_model_filename = eegnas.evolution_deap()
        else:
            best_model_filename = eegnas.evolution()
        if global_vars.get('pure_cross_subject') or len(subjects) == 1:
            return [best_model_filename]


def leave_one_out_exp(exp_name, csv_file, subjects):
    stop_criterion, iterator, loss_function, monitors = get_settings()
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
    best_model_filenames = []
    for subject_id in subjects:
        train_set = {}
        val_set = {}
        test_set = {}
        train_set[subject_id], val_set[subject_id], test_set[subject_id] =\
            get_pure_cross_subject(data_folder, exclude=[subject_id])
        evolution_file = '%s/subject_%d_archs.txt' % (exp_folder, subject_id)
        eegnas = EEGNAS_evolution(iterator=iterator, exp_folder=f"results/{exp_name}", exp_name=exp_name,
                            train_set=train_set, val_set=val_set, test_set=test_set,
                            stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                            subject_id=subject_id, fieldnames=FIELDNAMES, strategy='per_subject',
                            evolution_file=evolution_file, csv_file=csv_file)
        best_model_filename = eegnas.evolution()
        best_model_filenames.append(best_model_filename)
    return best_model_filenames


def get_settings():
    if global_vars.get('cropping'):
        global_vars.set('original_input_time_len', global_vars.get('input_time_len'))
        global_vars.set('input_time_len', global_vars.get('input_time_cropping'))
        stop_criterion, iterator, loss_function, monitors = get_cropped_settings()
    else:
        stop_criterion, iterator, loss_function, monitors = get_normal_settings()
    return stop_criterion, iterator, loss_function, monitors


@ex.main
def main(_config):
    global FIRST_RUN, FIRST_DATASET, FOLDER_NAMES
    try:
        set_params_by_dataset('EEGNAS/configurations/dataset_params.ini')
        set_gpu()
        set_seeds()
        if type(global_vars.get('subjects_to_check')) == list:
            subjects = global_vars.get('subjects_to_check')
        else:
            subjects = random.sample(range(1, global_vars.get('num_subjects')),
                                     global_vars.get('subjects_to_check'))
        exp_folder = f"results/{exp_name}"
        atexit.register(exit_handler, exp_folder, args.debug_mode)
        create_folder(exp_folder)
        FOLDER_NAMES.append(exp_name)
        write_dict(global_vars.config, f"{exp_folder}/config_{exp_name}.ini")
        csv_file = f"{exp_folder}/{exp_name}.csv"
        report_file = f"{exp_folder}/report_{exp_name}.csv"
        if 'cross_subject' in multiple_values and not global_vars.get('cross_subject'):
            global_vars.set('num_generations', global_vars.get('num_generations') *
                            global_vars.get('cross_subject_compensation_rate'))
        start_time = time.time()
        if global_vars.get('exp_type') in ['target', 'benchmark']:
            target_exp(exp_name, csv_file, subjects)
        elif global_vars.get('exp_type') == 'from_file':
            target_exp(exp_name, csv_file,
                       model_from_file=f"models/{global_vars.get('models_dir')}/{global_vars.get('model_file_name')}")
        else:
            best_model_filenames = exp_funcs[global_vars.get('exp_type')](exp_name, csv_file, subjects)
            for best_model_filename in best_model_filenames:
                target_exp(exp_name, csv_file, subjects, model_from_file=best_model_filename, write_header=False)
        global_vars.set('total_time', str(time.time() - start_time))
        write_dict(global_vars.config, f"{exp_folder}/final_config_{exp_name}.ini")
        result = generate_report(csv_file, report_file)
        ex.add_artifact(report_file)
    except Exception as e:
        with open(f"error_logs/error_log_{exp_name}.txt", "w") as err_file:
            print('experiment failed. Exception message: %s' % (str(e)), file=err_file)
            print(traceback.format_exc(), file=err_file)
        print('experiment failed. Exception message: %s' % (str(e)))
        print(traceback.format_exc())
        shutil.rmtree(exp_folder)
        FOLDER_NAMES.remove(exp_name)
        raise Exception(str(e))
    return result


def add_exp(exp_id, index, all_exps, run):
    all_exps['algorithm'].append(f'EEGNAS_{global_vars.get("num_layers")}_layers')
    all_exps['architecture'].append('best')
    all_exps['measure'].append(global_vars.get('ga_objective'))
    all_exps['dataset'].append(global_vars.get('dataset'))
    if global_vars.get('iterations'):
        all_exps['iteration'].append(global_vars.get('iterations'))
    else:
        all_exps['iteration'].append(1)
    all_exps['result'].append(run.result)
    all_exps['runtime'].append(strfdelta(run.stop_time - run.start_time, '{hours}:{minutes}:{seconds}'))
    all_exps['omniboard_id'].append(run._id)
    all_exps['machine'].append(platform.node())
    all_exps['local_exp_id'].append(f'{exp_id}_{index}')


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    init_config(args.config)
    data_folder = 'data/'
    low_cut_hz = 0
    valid_set_fraction = 0.2
    listen()
    exp_id = get_exp_id('results')
    exp_funcs = {'per_subject': per_subject_exp,
                 'leave_one_out': leave_one_out_exp}

    experiments = args.experiment.split(',')
    all_exps = defaultdict(list)
    first_run = True
    for experiment in experiments:
        configurations = get_configurations(experiment, global_vars.configs)
        multiple_values = get_multiple_values(configurations)
        for index, configuration in enumerate(configurations):
            global_vars.set_config(configuration)
            if FIRST_RUN:
                FIRST_DATASET = global_vars.get('dataset')
                if global_vars.get('include_params_folder_name'):
                    multiple_values.extend(global_vars.get('include_params_folder_name'))
                FIRST_RUN = False
            exp_name = f"{exp_id}_{index+1}_{experiment}"
            exp_name = add_params_to_name(exp_name, multiple_values)
            ex.config = {}
            ex.add_config({**configuration, **{'tags': [exp_id]}})
            if len(ex.observers) == 0 and not args.debug_mode:
                ex.observers.append(MongoObserver.create(url=f'mongodb://{global_vars.get("mongodb_server")}'
                                                             f'/{global_vars.get("mongodb_name")}',
                                                     db_name=global_vars.get("mongodb_name")))
            global_vars.set('sacred_ex', ex)
            try:
                run = ex.run(options={'--name': exp_name})
                if not args.debug_mode:
                    add_exp(exp_id, index, all_exps, run)
                    pd.DataFrame(all_exps).to_csv(f'reports/{exp_id}.csv', index=False)
            except Exception as e:
                print(f'failed experiment {exp_id}_{index+1}, continuing...')
        if args.drive == 't':
            upload_exp_to_gdrive(FOLDER_NAMES, FIRST_DATASET)