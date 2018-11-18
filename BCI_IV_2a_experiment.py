import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import tensorflow as tf
import platform
import matplotlib
matplotlib.use('Agg')
from experiments import run_genetic_filters
import configparser

if __name__ == '__main__':
    global data_folder, valid_set_fraction, config

    if platform.node() == 'nvidia':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 56})
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)

    data_folder = 'data/'
    low_cut_hz = 0
    valid_set_fraction = 0.2

    field_names = ['exp_id', 'generation', 'avg_fitness']

    experiments = {
        'evolution': run_genetic_filters
    }

    config = configparser.ConfigParser()
    config.read('config.ini')
    config.sections()
    subdirs = [x for x in os.walk('results')]
    if len(subdirs) == 1:
        exp_id = 1
    else:
        subdir_names = [int(x[0].split('\\')[1]) for x in subdirs[1:]]
        subdir_names.sort()
        exp_id = subdir_names[-1] + 1
    evolution = config['DEFAULT']['exp_type']
    sub_conf = config[evolution]
    experiments[evolution](exp_id, sub_conf)