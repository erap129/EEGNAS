import configparser


def init_config():
    global config
    config = configparser.ConfigParser()
    config.read('config.ini')