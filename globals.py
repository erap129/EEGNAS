import configparser


def init_config(filepath):
    global init_config
    init_config = configparser.ConfigParser()
    init_config.read(filepath)


def set_config(configuration):
    global config
    config = configuration


def set_dummy_config():
    global config
    config = {'DEFAULT': {'exp_name':'dummy'}, 'dummy': {}}


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


def set_if_not_exists(key, value):
    global config
    if key not in config[config['DEFAULT']['exp_name']] and key not in config['DEFAULT']:
        set(key, value)