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
    if key in config[config['DEFAULT']['exp_type']]:
        return config['exp_type'][key]
    else:
        return config['DEFAULT'][key]