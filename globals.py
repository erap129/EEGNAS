import configparser


def init_config(filepath):
    global init_config
    init_config = configparser.ConfigParser()
    init_config.read(filepath)

def set_config(configuration):
    global config
    config = configuration