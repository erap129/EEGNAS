import configparser
import os
import sys
from copy import deepcopy

from visualization import viz_reports

sys.path.append("..")
from utilities.config_utils import set_default_config, update_global_vars_from_config_dict, get_configurations
from utilities.misc import concat_train_val_sets, unify_dataset
import logging
from visualization.dsp_functions import butter_bandstop_filter, butter_bandpass_filter
from visualization.signal_plotting import plot_performance_frequency, tf_plot, plot_one_tensor

import torch
from braindecode.torch_ext.util import np_to_var
import global_vars
from torch import nn
from data_preprocessing import get_train_val_test, get_pure_cross_subject, get_dataset
from EEGNAS_experiment import get_normal_settings, set_params_by_dataset
import matplotlib.pyplot as plt
import matplotlib
from utilities.monitors import get_eval_function
from utilities.misc import createFolder
from visualization.cnn_layer_visualization import CNNLayerVisualization
from visualization.pdf_utils import create_pdf, create_pdf_from_story
import numpy as np
from visualization.wavelet_functions import get_tf_data_efficient, subtract_frequency
from datetime import datetime
import models_generation
from reportlab.platypus import Paragraph
from visualization.pdf_utils import get_image
from reportlab.lib.styles import getSampleStyleSheet
from EEGNAS_experiment import config_to_dict
from utilities.misc import label_by_idx
styles = getSampleStyleSheet()
from collections import OrderedDict, defaultdict
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)
# matplotlib.use("TkAgg")
plt.interactive(False)


def get_intermediate_act_map(data, select_layer, model):
    x = np_to_var(data[:, :, :, None]).cuda()
    modules = list(model.modules())[0]
    for l in modules[:select_layer + 1]:
      x = l(x)
    act_map = x.cpu().detach().numpy()
    act_map_avg = np.average(act_map, axis=0).swapaxes(0, 1).squeeze(axis=2)
    return act_map_avg


def plot_avg_activation_maps(pretrained_model, dataset, date_time):
    img_paths = []
    class_examples = []
    for class_idx in range(global_vars.get('n_classes')):
        class_examples.append(dataset.X[np.where(dataset.y == class_idx)])
    for index, layer in enumerate(list(pretrained_model.children())[:-1]):
        act_maps = []
        for class_idx in range(global_vars.get('n_classes')):
            act_maps.append(plot_one_tensor(get_intermediate_act_map
                                            (class_examples[class_idx], index, pretrained_model),
                                            f'Layer {index}, {label_by_idx(class_idx)}'))
        img_paths.extend(act_maps)
    create_pdf(f'results/{date_time}_{global_vars.get("dataset")}/step2_avg_activation_maps.pdf', img_paths)
    for im in img_paths:
        os.remove(im)


def frequency_correlation_single_example(pretrained_model, data, discriminating_layer, low_freq, high_freq):
    # find the most prominent example in each class
    # for each freq:
    #   get probability of correct class after each perturbation
    # plot probabilities as a function of the frequency
    max_per_class = get_max_examples_per_channel(data, discriminating_layer, pretrained_model)
    for chan_idx, example_idx in enumerate(max_per_class):
        correct_class_probas = []
        for freq in range(low_freq, high_freq+1):
            data_to_perturb = deepcopy(data)
            perturbed_data = subtract_frequency(data_to_perturb, freq, global_vars.get('frequency'))
            pretrained_model.eval()
            probas = pretrained_model(data[example_idx])
            print
    pass


if __name__ == '__main__':
    configs = configparser.ConfigParser()
    configs.read('visualization_configurations/viz_config.ini')
    configurations = get_configurations(sys.argv[1], configs, set_exp_name=False)
    global_vars.init_config('configurations/config.ini')
    set_default_config('../configurations/config.ini')
    global_vars.set('cuda', True)

    prev_dataset = None
    for configuration in configurations:
        update_global_vars_from_config_dict(configuration)
        global_vars.set('band_filter', {'pass': butter_bandpass_filter,
                                        'stop': butter_bandstop_filter}[global_vars.get('band_filter')])
        model = torch.load(f'../models/{global_vars.get("models_dir")}/{global_vars.get("model_name")}')
        model.cuda()

        if prev_dataset != global_vars.get('dataset'):
            set_params_by_dataset('../configurations/dataset_params.ini')
            subject_id = global_vars.get('subject_id')
            dataset = get_dataset(subject_id)
            concat_train_val_sets(dataset)
            prev_dataset = global_vars.get('dataset')

        now = datetime.now()
        date_time = now.strftime("%m.%d.%Y")
        folder_name = f'results/{date_time}_{global_vars.get("dataset")}'
        createFolder(folder_name)
        getattr(viz_reports, f'{global_vars.get("report")}_report')(model, dataset, folder_name)

