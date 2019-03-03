import configparser

def init_config(filepath):
    global init_config
    init_config = configparser.ConfigParser()
    init_config.read(filepath)

def set_config(configuration):
    global config
    config = configuration

def get(key):
    global config
    if key in config[config['DEFAULT']['exp_name']]:
        return config[config['DEFAULT']['exp_name']][key]
    elif key in config['DEFAULT']:
        return config['DEFAULT'][key]
    else:
        return False

def set(key, value):
    global config
    if key in config[config['DEFAULT']['exp_name']]:
        config[config['DEFAULT']['exp_name']][key] = value
    else:
        config['DEFAULT'][key] = value