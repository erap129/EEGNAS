import configparser


def init_config():
    global init_config
    init_config = configparser.ConfigParser()
    init_config.read('config.ini')

def set_config(configuration):
    global config
    config = configuration