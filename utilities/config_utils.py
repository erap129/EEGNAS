import configparser
import json
from itertools import product, chain
import globals
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
    res = list(set(multiple_values))
    if 'dataset' in res:
        res.remove('dataset')
    return res
