import configparser
import json
import os
import random
import torch
from itertools import product, chain
import global_vars
import numpy as np
from collections import OrderedDict, defaultdict


def config_to_dict(path):
    conf = configparser.ConfigParser()
    conf.optionxform = str
    conf.read(path)
    dictionary = {'DEFAULT': {}}
    for option in conf.defaults():
        dictionary['DEFAULT'][option] = eval(conf['DEFAULT'][option])
    for section in conf.sections():
        dictionary[section] = {}
        for option in conf.options(section):
            dictionary[section][option] = eval(conf.get(section, option))
    return dictionary


def get_configurations(experiment):
    configurations = []
    default_config = global_vars.configs._defaults
    exp_config = global_vars.configs._sections[experiment]
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


def set_default_config(path):
    global_vars.init_config(path)
    configurations = get_configurations('default_exp')
    global_vars.set_config(configurations[0])


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
    res = list(set(multiple_values))
    if 'dataset' in res:
        res.remove('dataset')
    return res


def set_params_by_dataset(params_config_path):
    config_dict = config_to_dict(params_config_path)
    for param in ['num_subjects', 'num_subjects', 'eeg_chans', 'input_time_len', 'n_classes', 'subjects_to_check',
                  'evaluation_metrics', 'ga_objective', 'nn_objective', 'frequency', 'problem', 'exclude_subjects']:
        set_param_for_dataset(param, config_dict)
    if global_vars.get('ensemble_iterations'):
        global_vars.set('evaluation_metrics', global_vars.get('evaluation_metrics') + ['raw', 'target'])
        if not global_vars.get('ensemble_size'):
            global_vars.set('ensemble_size', int(global_vars.get('pop_size') / 100))


def set_param_for_dataset(param_name, config_dict):
    if param_name in config_dict[global_vars.get('dataset')]:
        global_vars.set_if_not_exists(param_name, config_dict[global_vars.get('dataset')][param_name])


def set_seeds():
    random_seed = global_vars.get('random_seed')
    if not random_seed:
        random_seed = random.randint(0, 2**32 - 1)
        global_vars.set('random_seed', random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if global_vars.get('cuda'):
        torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)


def set_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = global_vars.get('gpu_select')
    try:
        torch.cuda.current_device()
        if not global_vars.get('force_gpu_off'):
            global_vars.set('cuda', True)
            print(f'set active GPU to {global_vars.get("gpu_select")}')
    except AssertionError as e:
        print('no cuda available, using CPU')


def update_global_vars_from_config_dict(config_dict):
    for key, inner_dict in config_dict.items():
        for inner_key, inner_value in inner_dict.items():
            global_vars.set(inner_key, inner_value)
