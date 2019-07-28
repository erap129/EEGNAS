import os
import sys
from copy import deepcopy

from utilities.data_utils import prepare_data_for_NN
from visualization.deconvolution import DeconvNet, ConvDeconvNet

sys.path.append("..")
from evolution.nn_training import NN_Trainer
from braindecode.datautil.splitters import concatenate_sets
from utilities.config_utils import set_default_config, update_global_vars_from_config_dict
from utilities.misc import concat_train_val_sets, unify_dataset
import logging
from visualization.dsp_functions import butter_bandstop_filter, butter_bandpass_filter
from visualization.signal_plotting import plot_performance_frequency, tf_plot
from NASUtils import evaluate_single_model
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
matplotlib.use("TkAgg")
plt.interactive(False)
img_name_counter = 1


def get_max_examples_per_channel(data, select_layer, model):
    act_maps = {}
    x = np_to_var(data[:, :, :, None]).cuda()
    modules = list(model.modules())[0]
    for idx, example in enumerate(x):
        example_x = example[None, :, :, :]
        for l in modules[:select_layer + 1]:
            example_x = l(example_x)
        act_maps[idx] = example_x
    channels = act_maps[0].shape[1]
    selected_examples = np.zeros(channels)
    for c in range(channels):
        selected_examples[c]\
            = int(np.array([act_map.squeeze()[c].sum() for act_map in act_maps.values()]).argmax())
    return [int(x) for x in selected_examples]


def create_max_examples_per_channel(select_layer, model, steps=500):
    dummy_X = models_generation.get_dummy_input().cuda()
    modules = list(model.modules())[0]
    for l in modules[:select_layer + 1]:
        dummy_X = l(dummy_X)
    channels = dummy_X.shape[1]
    act_maps = []
    for c in range(channels):
        layer_vis = CNNLayerVisualization(model, select_layer, c)
        act_maps.append(layer_vis.visualise_layer_with_hooks(steps))
        print(f'created optimal example for layer {select_layer}, channel {c}')
    return act_maps


def get_intermediate_act_map(data, select_layer, model):
    x = np_to_var(data[:, :, :, None]).cuda()
    modules = list(model.modules())[0]
    for l in modules[:select_layer + 1]:
      x = l(x)
    act_map = x.cpu().detach().numpy()
    act_map_avg = np.average(act_map, axis=0).swapaxes(0, 1).squeeze(axis=2)
    return act_map_avg


def plot_all_kernel_deconvolutions(model, conv_deconv, dataset, date_time, eeg_chans=None):
    if eeg_chans is None:
        eeg_chans = list(range(models_generation.get_dummy_input().shape[1]))
    tf_plots = []
    class_examples = []
    for class_idx in range(global_vars.get('n_classes')):
        class_examples.append(dataset['train'].X[np.where(dataset['train'].y == class_idx)])
    for layer_idx, layer in enumerate(model.children()):
        if type(layer) == nn.Conv2d:
            for filter_idx in range(layer.out_channels):
                for class_idx, examples in enumerate(class_examples):
                    X = prepare_data_for_NN(examples)
                    reconstruction = conv_deconv.forward(X, layer_idx, filter_idx)
                    subj_tfs = []
                    for eeg_chan in eeg_chans:
                        subj_tfs.append(get_tf_data_efficient(reconstruction.cpu().detach().numpy(),
                                                              eeg_chan, global_vars.get('frequency')))
                    tf_plots.append(tf_plot(subj_tfs, f'kernel deconvolution for layer {layer_idx},'
                                            f' filter {filter_idx}, class {label_by_idx(class_idx)}'))
    create_pdf(f'results/{date_time}_{global_vars.get("dataset")}/step1_all_kernels.pdf', tf_plots)
    for im in tf_plots:
        os.remove(im)


def plot_avg_activation_maps(pretrained_model, train_set, date_time):
    img_paths = []
    class_examples = []
    for class_idx in range(global_vars.get('n_classes')):
        class_examples.append(train_set[subject_id].X[np.where(train_set[subject_id].y == class_idx)])
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


def find_optimal_samples_per_filter(pretrained_model, train_set, date_time, eeg_chans=None):
    if eeg_chans is None:
        eeg_chans = list(range(models_generation.get_dummy_input().shape[1]))
    plot_dict = OrderedDict()
    for layer_idx, layer in enumerate(list(pretrained_model.children())):
        max_examples = get_max_examples_per_channel(train_set[subject_id].X, layer_idx, pretrained_model)
        for chan_idx, example_idx in enumerate(max_examples):
            tf_data = []
            for eeg_chan in eeg_chans:
                tf_data.append(get_tf_data_efficient(train_set[subject_id].X[example_idx][None, :, :], eeg_chan, 250))
            max_value = np.max(np.array(tf_data))
            class_str = ''
            if layer_idx >= len(list(pretrained_model.children())) - 3:
                class_str = f', class:{label_by_idx(chan_idx)}'
            plot_dict[(layer_idx, chan_idx)] = tf_plot(tf_data,
                                                      f'TF plot of example {example_idx} for layer '
                                                      f'{layer_idx}, channel {chan_idx}{class_str}',max_value)
            print(f'plot most activating TF for layer {layer_idx}, channel {chan_idx}')

    img_paths = list(plot_dict.values())
    story = []
    story.append(Paragraph('<br />\n'.join([f'{x}:{y}' for x,y in pretrained_model._modules.items()]), style=styles["Normal"]))
    for im in img_paths:
        story.append(get_image(im))
    create_pdf_from_story(f'results/{date_time}_{global_vars.get("dataset")}/step3_tf_plots_real.pdf', story)
    for im in img_paths:
        os.remove(im)


def create_optimal_samples_per_filter(pretrained_model, date_time, eeg_chans=None, steps=500, layer_idx_cutoff=0):
    if eeg_chans is None:
        eeg_chans = list(range(models_generation.get_dummy_input().shape[1]))
    plot_dict = OrderedDict()
    plot_imgs = OrderedDict()
    for layer_idx, layer in list(enumerate(list(pretrained_model.children())))[layer_idx_cutoff:]:
        max_examples = create_max_examples_per_channel(layer_idx, pretrained_model, steps=steps)
        max_value = 0
        for chan_idx, example in enumerate(max_examples):
            for eeg_chan in eeg_chans:
                plot_dict[(layer_idx, chan_idx, eeg_chan)] = get_tf_data_efficient(example, eeg_chan, 250)
                max_value = max(max_value, np.max(plot_dict[(layer_idx, chan_idx, eeg_chan)]))
        class_str = ''
        for chan_idx, example in enumerate(max_examples):
            if layer_idx >= len(list(pretrained_model.children())) - 3:
                class_str = f', class:{label_by_idx(chan_idx)}'
            plot_imgs[(layer_idx, chan_idx)] = tf_plot([plot_dict[(layer_idx, chan_idx, c)] for c in eeg_chans],
                                                       f'TF plot of optimal example for layer {layer_idx},'
                                                       f' channel {chan_idx}{class_str}', max_value)
            print(f'plot gradient ascent TF for layer {layer_idx}, channel {chan_idx}')

    story = []
    img_paths = list(plot_imgs.values())
    story.append(
        Paragraph('<br />\n'.join([f'{x}:{y}' for x, y in pretrained_model._modules.items()]), style=styles["Normal"]))
    for im in img_paths:
        story.append(get_image(im))
    create_pdf_from_story(f'results/{date_time}_{global_vars.get("dataset")}/step4_tf_plots_optimal_test.pdf', story)
    for im in img_paths:
        os.remove(im)


def get_avg_class_tf(train_set, date_time, eeg_chans=None):
    if eeg_chans is None:
        eeg_chans = list(range(models_generation.get_dummy_input().shape[1]))
    class_examples = []
    for class_idx in range(global_vars.get('n_classes')):
        class_examples.append(train_set[subject_id].X[np.where(train_set[subject_id].y == class_idx)])
    chan_data = []
    for class_idx in range(global_vars.get('n_classes')):
        chan_data.append(defaultdict(list))
        for example in class_examples[class_idx]:
            for eeg_chan in eeg_chans:
                chan_data[-1][eeg_chan].append(get_tf_data_efficient(example[None, :, :], eeg_chan, global_vars.get('frequency')))
    avg_tfs = []
    for class_idx in range(global_vars.get('n_classes')):
        class_tfs = []
        for eeg_chan in eeg_chans:
            class_tfs.append(np.average(np.array(chan_data[class_idx][eeg_chan]), axis=0))
        avg_tfs.append(class_tfs)
    max_value = max(*[np.max(np.array(class_chan_avg_tf)) for class_chan_avg_tf in avg_tfs])
    tf_plots = []
    for class_idx in range(global_vars.get('n_classes')):
        tf_plots.append(tf_plot(avg_tfs[class_idx], f'average TF for {label_by_idx(class_idx)}', max_value))
    story = [get_image(tf) for tf in tf_plots]
    create_pdf_from_story(f'results/{date_time}_{global_vars.get("dataset")}/step5_tf_plots_avg_per_class.pdf', story)
    for tf in tf_plots:
        os.remove(tf)


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


def pretrain_model_on_filtered_data(low_freq, high_freq):
    pure_cross_subj_dataset = {}
    pure_cross_subj_dataset['train'], pure_cross_subj_dataset['valid'], \
    pure_cross_subj_dataset['test'] = get_pure_cross_subject(global_vars.get('data_folder'))
    freq_models = {}
    pure_cross_subj_dataset_copy = deepcopy(pure_cross_subj_dataset)
    for freq in range(low_freq, high_freq + 1):
        pretrained_model_copy = deepcopy(pretrained_model)
        for section in ['train', 'valid', 'test']:
            global_vars.get('band_filter')(pure_cross_subj_dataset_copy[section].X, max(1, freq - 1), freq + 1,
                                           global_vars.get('frequency'))
        nn_trainer = NN_Trainer(iterator, loss_function, stop_criterion, monitors)
        _, _, model, _, _ = nn_trainer.evaluate_model(pretrained_model_copy, pure_cross_subj_dataset_copy)
        freq_models[freq] = model
    return freq_models


def performance_frequency_correlation(pretrained_model, subjects, low_freq, high_freq):
    stop_criterion, iterator, loss_function, monitors = get_normal_settings()
    baselines = OrderedDict()
    freq_models = pretrain_model_on_filtered_data(low_freq, high_freq)
    all_performances = []
    all_performances_freq = []
    for subject in subjects:
        single_subj_performances = []
        single_subj_performances_freq = []
        single_subj_dataset = get_dataset(subject)
        baselines[subject] = evaluate_single_model(pretrained_model, single_subj_dataset['test'].X,
                                                     single_subj_dataset['test'].y,
                                                     eval_func=get_eval_function())
        for freq in range(low_freq, high_freq+1):
            single_subj_dataset_freq = deepcopy(single_subj_dataset)
            for section in ['train', 'valid', 'test']:
                global_vars.get('band_filter')(single_subj_dataset_freq[section].X, max(1, freq - 1), freq + 1,
                                                                             global_vars.get('frequency'))
            pretrained_model_copy_freq = deepcopy(freq_models[freq])
            if global_vars.get('retrain_per_subject'):
                nn_trainer = NN_Trainer(iterator, loss_function, stop_criterion, monitors)
                _, _, pretrained_model_copy_freq, _, _ = nn_trainer.evaluate_model(pretrained_model_copy_freq,
                                                                single_subj_dataset_freq, final_evaluation=True)
            single_subj_performances.append(evaluate_single_model(pretrained_model, single_subj_dataset_freq['test'].X,
                                                                  single_subj_dataset['test'].y, get_eval_function()))
            single_subj_performances_freq.append(evaluate_single_model(pretrained_model_copy_freq, single_subj_dataset_freq['test'].X,
                                                                  single_subj_dataset['test'].y, get_eval_function()))
        all_performances.append(single_subj_performances)
        all_performances_freq.append(single_subj_performances_freq)
    baselines['average'] = np.average(list(baselines.values()))
    all_performances.append(np.average(all_performances, axis=0))
    all_performances_freq.append(np.average(all_performances_freq, axis=0))
    performance_plot_imgs = plot_performance_frequency([all_performances, all_performances_freq], baselines,
                                                       legend=['no retraining', 'with retraining', 'unperturbed'])
    story = [get_image(tf) for tf in performance_plot_imgs]
    create_pdf_from_story(f'results/{date_time}_{global_vars.get("dataset")}/step6_performance_frequency.pdf', story)
    for tf in performance_plot_imgs:
        os.remove(tf)


def plot_perturbations(subjects, low_freq, high_freq):
    eeg_chans = list(range(models_generation.get_dummy_input().shape[1]))
    tf_plots = []
    for subject in subjects:
        single_subj_dataset_orig = unify_dataset(get_dataset(subject))
        for frequency in range(low_freq, high_freq+1):
            single_subj_dataset = deepcopy(single_subj_dataset_orig)
            perturbed_data = global_vars.get('band_filter')(single_subj_dataset.X,
                               max(1, frequency - 1), frequency + 1, global_vars.get('frequency'))
            single_subj_dataset.X = perturbed_data
            subj_tfs = []
            for eeg_chan in eeg_chans:
                subj_tfs.append(get_tf_data_efficient(single_subj_dataset.X, eeg_chan, global_vars.get('frequency')))
            tf_plots.append(tf_plot(subj_tfs, f'average TF for subject {subject}, frequency {frequency} removed'))
    story = [get_image(tf) for tf in tf_plots]
    create_pdf_from_story(f'results/{date_time}_{global_vars.get("dataset")}/step7_frequency_removal_plot.pdf', story)
    for tf in tf_plots:
        os.remove(tf)


if __name__ == '__main__':
    set_default_config('../configurations/config.ini')
    global_vars.set('cuda', True)
    config_dict = config_to_dict('visualization_configurations/viz_config.ini')
    update_global_vars_from_config_dict(config_dict)
    global_vars.set('band_filter', {'pass': butter_bandpass_filter,
                                    'stop': butter_bandstop_filter}[global_vars.get('band_filter')])
    set_params_by_dataset('../configurations/dataset_params.ini')
    global_vars.set('subjects_to_check', [1])
    global_vars.set('retrain_per_subject', True)
    model = torch.load(f'../models/{global_vars.get("models_dir")}/{global_vars.get("model_name")}')

    subject_id = config_dict['DEFAULT']['subject_id']
    dataset = get_dataset(subject_id)
    concat_train_val_sets(dataset)
    model.cuda()
    # print(model)
    conv_deconv = ConvDeconvNet(model)
    # X = prepare_data_for_NN(train_set[1].X)
    # reconstruction = conv_deconv.forward(X, layer_idx=8, filter_idx=20)
    # subj_tf = get_tf_data_efficient(reconstruction.cpu().detach().numpy(), 1, global_vars.get('frequency'))
    # tf_plot([subj_tf], 'test')

    stop_criterion, iterator, loss_function, monitors = get_normal_settings()
    nn_trainer = NN_Trainer(iterator, loss_function, stop_criterion, monitors)
    # dataset = {}
    # dataset['train'], dataset['valid'], dataset['test'] = get_train_val_test('../data/', subject_id=subject_id)
    # dataset['train'] = concatenate_sets([dataset['train'], dataset['valid']])
    # _, _, pretrained_model, _, _ = nn_trainer.evaluate_model(model, dataset, final_evaluation=True)

    now = datetime.now()
    date_time = now.strftime("%m.%d.%Y-%H:%M:%S")
    createFolder(f'results/{date_time}_{global_vars.get("dataset")}')
    plot_all_kernel_deconvolutions(model, conv_deconv, dataset, date_time)
    # plot_avg_activation_maps(pretrained_model, train_set, date_time)
    # find_optimal_samples_per_filter(pretrained_model, train_set, date_time)
    # create_optimal_samples_per_filter(pretrained_model, date_time, steps=500, layer_idx_cutoff=10)
    # frequency_correlation_single_example(pretrained_model, test_set[1].X, 10, 1, 40)
    # get_avg_class_tf(train_set, date_time)
    # plot_perturbations([1], 1, 40)
    # performance_frequency_correlation(pretrained_model, range(1, global_vars.get('num_subjects') + 1), 1, 40)

