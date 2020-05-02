import os
import random
from collections import OrderedDict
from copy import deepcopy

import mne
import shap
import gc
import torch
import matplotlib
from braindecode.torch_ext.util import np_to_var
from captum.insights import AttributionVisualizer
from captum.insights.features import ImageFeature
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from reportlab.platypus import Paragraph
from sklearn.preprocessing import MinMaxScaler
from EEGNAS import global_vars
from captum.attr import Saliency, IntegratedGradients, DeepLift, NoiseTunnel, NeuronConductance, GradientShap, LayerDeepLift
from captum.attr import visualization as viz
from EEGNAS.utilities.NAS_utils import evaluate_single_model
from EEGNAS.data_preprocessing import get_dataset
from EEGNAS.utilities.NN_utils import get_intermediate_layer_value, get_class_distribution
from EEGNAS.utilities.config_utils import set_default_config, set_params_by_dataset
from EEGNAS.utilities.data_utils import get_dummy_input, prepare_data_for_NN, tensor_to_eeglab
from EEGNAS.utilities.misc import unify_dataset, concat_train_val_sets, eeg_label_by_idx, write_dict
from EEGNAS.utilities.monitors import get_eval_function
from EEGNAS.visualization.deconvolution import ConvDeconvNet
from EEGNAS.visualization.dsp_functions import get_fft
from EEGNAS.visualization.pdf_utils import get_image, create_pdf_from_story, create_pdf
from EEGNAS.visualization.signal_plotting import tf_plot, plot_performance_frequency, image_plot, fft_plot, \
    get_next_temp_image_name
import numpy as np
from torch import nn
from reportlab.lib.styles import getSampleStyleSheet
from EEGNAS.utilities.misc import label_by_idx
import matplotlib.pyplot as plt
styles = getSampleStyleSheet()
from EEGNAS.visualization.viz_utils import pretrain_model_on_filtered_data, create_max_examples_per_channel, \
    get_max_examples_per_channel, export_performance_frequency_to_csv, get_top_n_class_examples
from EEGNAS.visualization.wavelet_functions import get_tf_data_efficient

'''
Iterates through frequencies from 'low_freq' to 'high_freq' and plots perturbed data for each frequency,
for a specific subject as defined in the configuration file. Exact perturbation defined in configuration file.
'''
def perturbation_report(model, dataset, folder_name):
    report_file_name = f'{folder_name}/{global_vars.get("report")}_{global_vars.get("band_filter").__name__}.pdf'
    if os.path.isfile(report_file_name):
        return
    eeg_chans = list(range(get_dummy_input().shape[1]))
    tf_plots = []
    dataset = unify_dataset(dataset)
    for frequency in range(global_vars.get("low_freq"), global_vars.get("high_freq") + 1):
        single_subj_dataset = deepcopy(dataset)
        perturbed_data = global_vars.get('band_filter')(single_subj_dataset.X,
                                                        max(1, frequency - 1), frequency + 1, global_vars.get('frequency'))
        if global_vars.get('to_matlab'):
            tensor_to_eeglab(perturbed_data, f'{folder_name}/perturbation_report/frequency_{frequency}_'
                                             f'{global_vars.get("band_filter")}.mat')
        single_subj_dataset.X = perturbed_data
        subj_tfs = []
        for eeg_chan in eeg_chans:
            subj_tfs.append(get_tf_data_efficient(single_subj_dataset.X, eeg_chan, global_vars.get('frequency')))
        tf_plots.append(tf_plot(subj_tfs, f'average TF for subject {global_vars.get("subject_id")},'
                                          f' frequency {frequency}, {global_vars.get("band_filter").__name__}'))
    story = [get_image(tf) for tf in tf_plots]
    create_pdf_from_story(report_file_name, story)
    for tf in tf_plots:
        os.remove(tf)


'''
For each subject (usually done on combined dataset) and for each frequency, perturb the dataset and check the performance
after perturbation. Compare performance after perturbation to the baseline performance (without perturbation).
Has option to export results to csv.
'''
def performance_frequency_report(pretrained_model, dataset, folder_name):
    report_file_name = f'{folder_name}/{global_vars.get("report")}_{global_vars.get("band_filter").__name__}.pdf'
    if os.path.isfile(report_file_name):
        return
    baselines = []
    freq_models = pretrain_model_on_filtered_data(pretrained_model, global_vars.get('low_freq'),
                                                  global_vars.get('high_freq'))
    all_performances = []
    all_performances_freq = []
    for subject in global_vars.get('subjects_to_check'):
        single_subj_performances = []
        single_subj_performances_freq = []
        single_subj_dataset = get_dataset(subject)
        baselines.append(evaluate_single_model(pretrained_model, single_subj_dataset['test'].X,
                                                     single_subj_dataset['test'].y,
                                                     eval_func=get_eval_function()))
        for freq in range(global_vars.get('low_freq'), global_vars.get('high_freq') + 1):
            single_subj_dataset_freq = deepcopy(single_subj_dataset)
            for section in ['train', 'valid', 'test']:
                single_subj_dataset_freq[section].X = global_vars.get('band_filter')\
                    (single_subj_dataset_freq[section].X, max(1, freq - 1), freq + 1, global_vars.get('frequency')).astype(np.float32)
            pretrained_model_copy_freq = deepcopy(freq_models[freq])
            single_subj_performances.append(evaluate_single_model(pretrained_model, single_subj_dataset_freq['test'].X,
                                                                  single_subj_dataset['test'].y, get_eval_function()))
            single_subj_performances_freq.append(evaluate_single_model(pretrained_model_copy_freq, single_subj_dataset_freq['test'].X,
                                                                  single_subj_dataset['test'].y, get_eval_function()))
        all_performances.append(single_subj_performances)
        all_performances_freq.append(single_subj_performances_freq)
    baselines.append(np.average(baselines, axis=0))
    all_performances.append(np.average(all_performances, axis=0))
    all_performances_freq.append(np.average(all_performances_freq, axis=0))
    export_performance_frequency_to_csv(all_performances, all_performances_freq, baselines, folder_name)
    performance_plot_imgs = plot_performance_frequency([all_performances, all_performances_freq], baselines,
                                                       legend=['no retraining', 'with retraining', 'unperturbed'])
    for subj_idx in range(len(all_performances)):
        for perf_idx in range(len(all_performances[subj_idx])):
            if subj_idx == len(all_performances) - 1:
                subj_str = 'avg'
            else:
                subj_str = subj_idx
            global_vars.get('sacred_ex').log_scalar(f'subject {subj_str} no retrain', all_performances[subj_idx][perf_idx],
                                                    global_vars.get('low_freq') + perf_idx)
            global_vars.get('sacred_ex').log_scalar(f'subject {subj_str} retrain', all_performances_freq[subj_idx][perf_idx],
                                                    global_vars.get('low_freq') + perf_idx)
            global_vars.get('sacred_ex').log_scalar(f'subject {subj_str} baseline', baselines[subj_idx],
                                                    global_vars.get('low_freq') + perf_idx)
    story = [get_image(tf) for tf in performance_plot_imgs]
    create_pdf_from_story(report_file_name, story)
    for tf in performance_plot_imgs:
        os.remove(tf)


'''
Get importance values for each data point in the dataset, for each neuron in each layer
'''
def neuron_importance_report(model, dataset, folder_name):
    report_file_name = f'{folder_name}/{global_vars.get("report")}.pdf'
    eeg_chans = list(range(global_vars.get('eeg_chans')))
    tf_plots = []
    test_examples = prepare_data_for_NN(dataset['test'].X)
    for layer_idx, layer in list(enumerate(model.children()))[global_vars.get('layer_idx_cutoff'):]:
        layer_output = get_intermediate_layer_value(model, test_examples, layer_idx)
        neuron_cond = NeuronConductance(model, layer)
        for filter_idx in range(layer.out_channels):
            for class_idx in range(global_vars.get('n_classes')):
                conductance_values = []
                if layer_output.ndim >= 3:
                    tensor_len = layer_output.shape[2]
                else:
                    tensor_len = 1
                for len_idx in range(tensor_len):
                    conductance_values.append(neuron_cond.attribute(random.choices(test_examples, k=int(len(test_examples) *
                        global_vars.get('explainer_sampling_rate'))), neuron_index=(filter_idx, len_idx), target=class_idx))
                conductance_values = np.mean(conductance_values, axis=0)
                print
                if global_vars.get('to_eeglab'):
                    tensor_to_eeglab(reconstruction,
                        f'{folder_name}/kernel_deconvolution/X_layer_{layer_idx}_filter_{filter_idx}_class_{label_by_idx(class_idx)}.mat')
                if global_vars.get('plot'):
                    subj_tfs = []
                    for eeg_chan in eeg_chans:
                        if global_vars.get('plot_TF'):
                            subj_tfs.append(get_tf_data_efficient(reconstruction.cpu().detach().numpy(),
                                                              eeg_chan, global_vars.get('frequency'), global_vars.get('num_frex'), dB=True))
                            print(f'applied TF to layer {layer_idx}, class {class_idx}, channel {eeg_chan}')
                        else:
                            subj_tfs.append(get_fft(np.average(reconstruction.cpu().detach().numpy(), axis=0).squeeze()[eeg_chan], global_vars.get('frequency')))
                    if global_vars.get('deconvolution_by_class'):
                        class_title = label_by_idx(class_idx)
                    else:
                        class_title = 'all'
                    if global_vars.get('plot_TF'):
                        tf_plots.append(tf_plot(subj_tfs, f'kernel deconvolution for layer {layer_idx},'
                                            f' filter {filter_idx}, class {class_title}'))
                    else:
                        tf_plots.append(fft_plot(subj_tfs,  f'kernel deconvolution for layer {layer_idx},'
                                            f' filter {filter_idx}, class {class_title}'))
    if global_vars.get('plot'):
        create_pdf(report_file_name, tf_plots)
        for im in tf_plots:
            os.remove(im)


'''
for each filter in each convolutional layer perform a reconstruction of an EEG example, using deconvolution with the
learned weights. Do this for the top 5 examples in each class.
'''
def kernel_deconvolution_report(model, dataset, folder_name):
    by_class_str = ''
    if global_vars.get('deconvolution_by_class'):
        by_class_str = '_by_class'
    report_file_name = f'{folder_name}/{global_vars.get("report")}{by_class_str}.pdf'
    if os.path.isfile(report_file_name):
        return
    if os.path.exists(f'{report_file_name[:-4]}.txt'):
        os.remove(f'{report_file_name[:-4]}.txt')
    conv_deconv = ConvDeconvNet(model)
    eeg_chans = list(range(global_vars.get('eeg_chans')))
    tf_plots = []
    class_examples = []
    if global_vars.get('deconvolution_by_class'):
        for class_idx in range(global_vars.get('n_classes')):
            all_class_examples = dataset['train'].X[np.where(dataset['train'].y == class_idx)]
            class_examples.append(get_top_n_class_examples(all_class_examples, class_idx, model, int(len(all_class_examples) * global_vars.get('deconvolution_sampling_rate'))))
    else:
        class_examples.append(np.random.choice(dataset['train'].X, int(dataset['train'].X.shape[0] * global_vars.get('deconvolution_sampling_rate')) , replace=False))
    for layer_idx, layer in list(enumerate(model.children()))[global_vars.get('layer_idx_cutoff'):]:
        if type(layer) == nn.Conv2d:
            for filter_idx in range(layer.out_channels):
                for class_idx, examples in enumerate(class_examples):
                    X = prepare_data_for_NN(examples)
                    if global_vars.get('avg_deconv'):
                        X = torch.mean(X, axis=0)[None, :, :, :]
                    reconstruction = conv_deconv.forward(X, layer_idx, filter_idx)
                    dist_dict_original = get_class_distribution(model, X)
                    dist_dict_deconv = get_class_distribution(model, reconstruction)
                    for key in dist_dict_original.keys():
                        try:
                            orig_count = dist_dict_original[key]
                        except KeyError:
                            orig_count = 0
                        try:
                            deconv_count = dist_dict_deconv[key]
                        except KeyError:
                            deconv_count = 0
                        global_vars.get('sacred_ex').log_scalar(f'layer_{layer_idx}_class_{class_idx}_original',
                                                                orig_count, filter_idx)
                        global_vars.get('sacred_ex').log_scalar(f'layer_{layer_idx}_class_{class_idx}_deconv',
                                                                deconv_count, filter_idx)
                    if global_vars.get('to_eeglab'):
                        tensor_to_eeglab(reconstruction,
                            f'{folder_name}/kernel_deconvolution/X_layer_{layer_idx}_filter_{filter_idx}_class_{label_by_idx(class_idx)}.mat')
                    if global_vars.get('plot'):
                        subj_tfs = []
                        for eeg_chan in eeg_chans:
                            if global_vars.get('plot_TF'):
                                subj_tfs.append(get_tf_data_efficient(reconstruction.cpu().detach().numpy(),
                                                                  eeg_chan, global_vars.get('frequency'), global_vars.get('num_frex'), dB=True))
                                print(f'applied TF to layer {layer_idx}, class {class_idx}, channel {eeg_chan}')
                            else:
                                subj_tfs.append(get_fft(np.average(reconstruction.cpu().detach().numpy(), axis=0).squeeze()[eeg_chan], global_vars.get('frequency')))
                        if global_vars.get('deconvolution_by_class'):
                            class_title = label_by_idx(class_idx)
                        else:
                            class_title = 'all'
                        if global_vars.get('plot_TF'):
                            tf_plots.append(tf_plot(subj_tfs, f'kernel deconvolution for layer {layer_idx},'
                                                f' filter {filter_idx}, class {class_title}'))
                        else:
                            tf_plots.append(fft_plot(subj_tfs,  f'kernel deconvolution for layer {layer_idx},'
                                                f' filter {filter_idx}, class {class_title}'))
        if global_vars.get('plot'):
            create_pdf(report_file_name, tf_plots)
            for im in tf_plots:
                os.remove(im)


'''
plot the average time-frequency of each class in the dataset.
'''
def avg_class_tf_report(model, dataset, folder_name):
    report_file_name = f'{folder_name}/{global_vars.get("report")}.pdf'
    if os.path.isfile(report_file_name):
        return
    eeg_chans = list(range(global_vars.get('eeg_chans')))
    dataset = unify_dataset(dataset)
    class_examples = []
    for class_idx in range(global_vars.get('n_classes')):
        class_examples.append(dataset.X[np.where(dataset.y == class_idx)])
        if global_vars.get('to_eeglab'):
            tensor_to_eeglab(class_examples[-1], f'{folder_name}/avg_class_tf/{label_by_idx(class_idx)}.mat')
    chan_data = np.zeros((global_vars.get('n_classes'), len(eeg_chans), global_vars.get('num_frex'), global_vars.get('input_height')))
    for class_idx in range(global_vars.get('n_classes')):
        for eeg_chan in eeg_chans:
            chan_data[class_idx, eeg_chan] = get_tf_data_efficient(class_examples[class_idx], eeg_chan,
                                                 global_vars.get('frequency'), global_vars.get('num_frex'),
                                                                   dB=global_vars.get('db_normalization'))
    max_value = np.max(chan_data)
    tf_plots = []
    for class_idx in range(global_vars.get('n_classes')):
        tf_plots.append(tf_plot(chan_data[class_idx], f'average TF for {label_by_idx(class_idx)}', max_value))
    story = [get_image(tf) for tf in tf_plots]
    create_pdf_from_story(report_file_name, story)
    for tf in tf_plots:
        os.remove(tf)


'''
Calculate the difference in power between classes for each frequency and each channel
'''
def power_diff_report(model, dataset, folder_name):
    report_file_name = f'{folder_name}/{global_vars.get("report")}.pdf'
    dataset = unify_dataset(dataset)
    class_examples = []
    nyquist = int(global_vars.get('frequency') / 2) - 1
    for class_idx in range(global_vars.get('n_classes')):
        class_examples.append(dataset.X[np.where(dataset.y == class_idx)])
    freqs = np.fft.fftfreq(global_vars.get('input_height'), 1 / global_vars.get('frequency'))
    freq_idx = np.argmax(freqs >= nyquist)
    diff_array = np.zeros((global_vars.get('eeg_chans'), freq_idx))
    for chan in list(range(global_vars.get('eeg_chans'))):
        first_power = np.average(np.fft.fft(class_examples[0][:, chan, :]).squeeze(), axis=0)[:freq_idx]
        second_power = np.average(np.fft.fft(class_examples[1][:, chan, :]).squeeze(), axis=0)[:freq_idx]
        power_diff = abs(first_power - second_power)
        diff_array[chan] = power_diff
    fig, ax = plt.subplots(figsize=(18, 10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(diff_array, cmap='hot', interpolation='nearest', aspect='auto', extent=[0, nyquist, 1, global_vars.get('eeg_chans')])
    ax.set_title('frequency diff between classes')
    ax.set_ylabel('channel')
    ax.set_xlabel('frequency')
    fig.colorbar(im, cax=cax, orientation='vertical')
    filename = f'temp/freq_diff.png'
    plt.savefig(filename)
    story = [get_image(tf) for tf in [filename]]
    create_pdf_from_story(report_file_name, story)
    for tf in [filename]:
        os.remove(tf)


'''
for each filter in each convolutional layer perform a reconstruction of an EEG example, using gradient ascent to
maximize the response of that filter. Plot the result for each filter in each conv layer. Steps for gradient ascent
defined in configuration file. layer_idx_cutoff defines the starting index for the layers (defined in the 
global variables).
'''
def gradient_ascent_report(pretrained_model, dataset, folder_name):
    report_file_name = f'{folder_name}/{global_vars.get("report")}_{global_vars.get("gradient_ascent_steps")}_steps.pdf'
    if os.path.isfile(report_file_name):
        return
    eeg_chans = list(range(global_vars.get('eeg_chans')))
    plot_dict = OrderedDict()
    plot_imgs = OrderedDict()
    for layer_idx, layer in list(enumerate(list(pretrained_model.children())))[global_vars.get('layer_idx_cutoff'):]:
        max_examples = create_max_examples_per_channel(layer_idx, pretrained_model, steps=global_vars.get('gradient_ascent_steps'))
        max_value = 0
        for chan_idx, example in enumerate(max_examples):
            for eeg_chan in eeg_chans:
                if global_vars.get('plot_TF'):
                    plot_dict[(layer_idx, chan_idx, eeg_chan)] = get_tf_data_efficient(example, eeg_chan, global_vars.get('frequency'), global_vars.get('num_frex'), dB=True)
                    max_value = max(max_value, np.max(plot_dict[(layer_idx, chan_idx, eeg_chan)]))
                else:
                    plot_dict[(layer_idx, chan_idx, eeg_chan)] = get_fft(example.squeeze()[eeg_chan], global_vars.get('frequency'))
        class_str = ''
        for chan_idx, example in enumerate(max_examples):
            if layer_idx >= len(list(pretrained_model.children())) - 3:
                class_str = f', class:{label_by_idx(chan_idx)}'
            if global_vars.get('plot_TF'):
                plot_imgs[(layer_idx, chan_idx)] = tf_plot([plot_dict[(layer_idx, chan_idx, c)] for c in eeg_chans],
                                                       f'TF plot of optimal example for layer {layer_idx},'
                                                       f' channel {chan_idx}{class_str}')
            else:
                plot_imgs[(layer_idx, chan_idx)] = fft_plot([plot_dict[(layer_idx, chan_idx, c)] for c in eeg_chans],
                                                            f'FFT plot of optimal example for layer {layer_idx},'
                                                            f' channel {chan_idx}{class_str}')
            print(f'plot gradient ascent TF for layer {layer_idx}, channel {chan_idx}')

    story = []
    img_paths = list(plot_imgs.values())
    styles = getSampleStyleSheet()
    story.append(
        Paragraph('<br />\n'.join([f'{x}:{y}' for x, y in pretrained_model._modules.items()]), style=styles["Normal"]))
    for im in img_paths:
        story.append(get_image(im))
    create_pdf_from_story(report_file_name, story)
    for im in img_paths:
        os.remove(im)


'''
For each filter in each convolutional layer, plot the EEG example that maximizes its output. Start at
layer_idx_cutoff.
'''
def find_optimal_samples_report(pretrained_model, dataset, folder_name):
    report_file_name = f'{folder_name}/{global_vars.get("report")}.pdf'
    if os.path.isfile(report_file_name):
        return
    eeg_chans = list(range(global_vars.get('eeg_chans')))
    plot_dict = OrderedDict()
    dataset = unify_dataset(dataset)
    for layer_idx, layer in list(enumerate(pretrained_model.children()))[global_vars.get('layer_idx_cutoff'):]:
        max_examples = get_max_examples_per_channel(dataset.X, layer_idx, pretrained_model)
        for chan_idx, example_idx in enumerate(max_examples):
            tf_data = []
            for eeg_chan in eeg_chans:
                tf_data.append(get_tf_data_efficient(dataset.X[example_idx][None, :, :], eeg_chan, 250))
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
    create_pdf_from_story(report_file_name, story)
    for im in img_paths:
        os.remove(im)


def attribute_image_features(model, algorithm, input, ind, **kwargs):
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input, target=ind, **kwargs)
    return tensor_attributions


class shap_deep_explainer:
    def __init__(self, model, train_data):
        self.explainer = shap.DeepExplainer(model, train_data[
            np.random.choice(train_data.shape[0], int(train_data.shape[0] * global_vars.get('explainer_sampling_rate')),
                             replace=False)])
        self.min = -0.04
        self.max = 0.04

    def get_feature_importance(self, data):
        return self.explainer.shap_values(data)


class saliency_explainer:
    def __init__(self, model, train_data):
        model.eval()
        self.explainer = Saliency(model)

    def get_feature_importance(self, data):
        data.requires_grad = True
        return torch.stack([self.explainer.attribute(data, target=i) for i in range(global_vars.get('n_classes'))], axis=0)


class integrated_gradients_explainer:
    def __init__(self, model, train_data):
        model.eval()
        self.explainer = IntegratedGradients(model)
        self.model = model

        if global_vars.get('model_alias') == 'nsga':
            global_vars.set('explainer_sampling_rate', 0.03)
        elif global_vars.get('model_alias') == 'rnn':
            global_vars.set('explainer_sampling_rate', 0.1)

    def get_feature_importance(self, data):
        data.requires_grad = True
        return torch.stack([attribute_image_features(self.model, self.explainer, data, i, baselines=data * 0,
                                                     return_convergence_delta=True)[0] for i in
                                                            range(global_vars.get('n_classes'))], axis=0).detach()

class deeplift_explainer:
    def __init__(self, model, train_data):
        model.eval()
        self.explainer = DeepLift(model)
        self.model = model

    def get_feature_importance(self, data):
        data.requires_grad = True
        return torch.stack([attribute_image_features(self.model, self.explainer, data, i, baselines=data * 0,
                                                     return_convergence_delta=True)[0] for i in
                                                            range(global_vars.get('n_classes'))], axis=0).detach()


class layer_deeplift_explainer:
    def __init__(self, model, layer):
        model.eval()
        self.explainer = LayerDeepLift(model, layer)
        self.model = model

    def get_feature_importance(self, data):
        data.requires_grad = True
        return torch.stack([attribute_image_features(self.model, self.explainer, data, i, baselines=data * 0,
                                                     return_convergence_delta=True,  attribute_to_layer_input=True)[0][0] for i in
                                                            range(global_vars.get('n_classes'))], axis=0).detach()


class gradientshap_explainer:
    def __init__(self, model, train_data):
        model.eval()
        self.explainer = GradientShap(model)
        self.model = model

    def get_feature_importance(self, data):
        data.requires_grad = True
        baseline_dist = torch.randn(data.shape) * 0.001
        return torch.stack([self.explainer.attribute(data, stdevs=0.09, n_samples=4, baselines=baseline_dist,
                                           target=i, return_convergence_delta=True)[0] for i in
                                                range(global_vars.get('n_classes'))], axis=0).detach()


def plot_feature_importance_netflow(folder_name, features, start_hour, dataset_name, segment, viz_method, title=None):
    matplotlib.rcParams['axes.linewidth'] = 1.5
    if global_vars.get('dataset') == 'netflow_asflow':
        fig_height = 5
    else:
        fig_height = 10
    f, axes = plt.subplots(len(features), figsize=(20, fig_height), sharex='col', sharey='row', constrained_layout=True)
    if global_vars.get('normalize_plots'):
        features = features / np.max(np.abs(features))
    for idx, ax in enumerate(axes):
        if global_vars.get('normalize_plots'):
            im = ax.imshow(features[idx], cmap='seismic', interpolation='nearest', aspect='auto', vmin=-0.5,
                           vmax=0.5)
        else:
            im = ax.imshow(features[idx], cmap='seismic', interpolation='nearest', aspect='auto')
        ytick_range = list(range(features[0].shape[0]))
        ax.set_yticks(ytick_range)
        ax.set_yticklabels([eeg_label_by_idx(x) for x in ytick_range])
        if global_vars.get('dataset') == 'netflow_asflow':
            ax.set_xticks(list(range(features[0].shape[1]))[::3])
            ax.set_xticklabels([(i + start_hour) % 24 for i in range(features[0].shape[1])][::3])
        else:
            ax.set_xticks(list(range(features[0].shape[1]))[::global_vars.get('frequency')])
            ax.set_xticklabels(list(range(len(list(range(features[0].shape[1]))[::global_vars.get('frequency')]))))
        input_height_arr = list(range(features[0].shape[1]))
        if idx == 0 and global_vars.get('dataset') == 'netflow_asflow':
            ax2 = ax.twiny()
            ax2.set_xlim(1, int(len(input_height_arr) / 24))
            num_days = len(list(range(int(features[0].shape[1] / 24))))
            ax2.set_xticks(list(range(int(features[0].shape[1] / 24))))
            ax2.set_xticklabels([f'{num_days - i} Days' for i in range(int(features[0].shape[1] / 24))])
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.tick_params(axis='y', which='major', labelsize=6)
        if global_vars.get('dataset') == 'netflow_asflow':
            ax.set_xlabel('hour of day')
            ax.set_ylabel('handover')
        else:
            ax.set_xlabel('seconds')
            ax.set_ylabel('electrode')
        secax = ax.secondary_yaxis('right')
        if global_vars.get('dataset') == 'netflow_asflow':
            secax.set_ylabel(label_by_idx(idx, dataset_name)[5:], rotation=270)
        else:
            secax.set_ylabel(label_by_idx(idx, dataset_name), rotation=270)
        secax.tick_params(labelsize=0, length=0, width=0)

    f.subplots_adjust(right=0.8, hspace=0)
    cbar_ax = f.add_axes([0.83, 0.15, 0.02, 0.7])
    cbar = f.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title(f'{viz_method} values', fontsize=12)
    if title is not None:
        plt.suptitle(title)
        shap_img_file = f'{folder_name}/{dataset_name}_{segment}_{viz_method}_{title}.png'
    else:
        shap_img_file = f'{folder_name}/{dataset_name}_{segment}_{viz_method}.png'
    plt.savefig(shap_img_file, dpi=300)
    plt.clf()
    return shap_img_file


def save_feature_importances(folder_name, feature_importances):
    exp_id = folder_name.split('_')[0]
    imp_fold_name = f'{exp_id}_importances'
    if not os.path.exists(imp_fold_name):
        os.makedirs(imp_fold_name)
    np.save(f'{imp_fold_name}/{global_vars.get("as_to_test")}_{global_vars.get("fold_idx")}.npy', feature_importances)


def plot_topo_feature_importance(folder_name, features):
    tmin = 0
    info = mne.create_info(
        ch_names=[eeg_label_by_idx(idx) for idx in range(global_vars.get('eeg_chans'))],
        ch_types=['eeg' for i in range(global_vars.get('eeg_chans'))],
        sfreq=global_vars.get('frequency')
    )

    for feat_idx, eeg_class in enumerate(features):
        evoked_array = mne.EvokedArray(eeg_class, info, tmin)
        evoked_array.set_montage('standard_1020')
        fig = evoked_array.plot_topomap(ch_type='eeg', time_unit='s', colorbar=False, show_names=True, times=np.arange(0, 4, 0.3))
        fig.suptitle(label_by_idx(feat_idx))
        fig.savefig(f'{folder_name}/mne_plot_{label_by_idx(feat_idx)}.png')
        print(evoked_array)


'''
Use some explainer to get feature importance for each class
'''
def feature_importance_report(model, dataset, folder_name):
    FEATURE_VALUES = {}
    feature_mean = {}
    vmin = np.inf
    vmax = -np.inf
    report_file_name = f'{folder_name}/{global_vars.get("report")}_{global_vars.get("explainer")}.pdf'
    train_data = np_to_var(dataset['train'].X[:, :, :, None])
    model.cpu()
    if 'Ensemble' in type(model).__name__:
        for mod in model.models:
            if 'Ensemble' in type(mod).__name__:
                for inner_mod in mod.models:
                    inner_mod.cpu()
                    inner_mod.eval()
            mod.cpu()
            mod.eval()
    e = globals()[f'{global_vars.get("explainer")}_explainer'](model, train_data)
    shap_imgs = []
    for segment in ['test']:
        if segment == 'both':
            dataset = unify_dataset(dataset)
            segment_data = np_to_var(dataset.X[:, :, :, None])
        else:
            segment_data = np_to_var(dataset[segment].X[:, :, :, None])
        print(f'calculating {global_vars.get("explainer")} values for {int(segment_data.shape[0] * global_vars.get("explainer_sampling_rate"))} samples')
        segment_examples = segment_data[np.random.choice(segment_data.shape[0], int(segment_data.shape[0] * global_vars.get("explainer_sampling_rate")), replace=False)]
        feature_values = e.get_feature_importance(segment_examples)
        feature_val = np.array(feature_values).squeeze()
        feature_mean[segment] = np.mean(feature_val, axis=1)
        if global_vars.get('dataset') == 'netflow_asflow':
            save_feature_importances(folder_name, feature_mean[segment])
        else:
            np.save(f'{folder_name}/{global_vars.get("explainer")}_{segment}.npy', feature_mean[segment])
        feature_value = np.concatenate(feature_mean[segment], axis=0)
        feature_value = (feature_value - np.mean(feature_value)) / np.std(feature_value)
        FEATURE_VALUES[segment] = feature_value
        if feature_mean[segment].min() < vmin:
            vmin = feature_mean[segment].min()
        if feature_mean[segment].max() > vmax:
            vmax = feature_mean[segment].max()
    for segment in ['test']:
        img_file = plot_feature_importance_netflow(folder_name, feature_mean[segment], global_vars.get('start_hour'),
                                        global_vars.get('dataset'), segment, global_vars.get('explainer'))
        if global_vars.get('dataset') != 'netflow_asflow':
            plot_topo_feature_importance(folder_name, feature_mean[segment])
        shap_imgs.append(img_file)
    story = []
    for im in shap_imgs:
        story.append(get_image(im))
    create_pdf_from_story(report_file_name, story)
    global_vars.get('sacred_ex').add_artifact(report_file_name)
    for im in shap_imgs:
        os.remove(im)
    gc.collect()
    return FEATURE_VALUES


def feature_importance_minmax_report(model, dataset, folder_name):
    FEATURE_VALUES = {}
    report_file_name = f'{folder_name}/{global_vars.get("report")}_{global_vars.get("explainer")}.pdf'
    train_data = np_to_var(dataset['train'].X[:, :, :, None])
    model.cpu()
    if 'Ensemble' in type(model).__name__:
        for mod in model.models:
            if 'Ensemble' in type(mod).__name__:
                for inner_mod in mod.models:
                    inner_mod.cpu()
                    inner_mod.eval()
            mod.cpu()
            mod.eval()
    e = globals()[f'{global_vars.get("explainer")}_explainer'](model, train_data)
    shap_imgs = []
    # for segment in ['train', 'test', 'both']:
    for segment in ['test']:
        if segment == 'both':
            dataset = unify_dataset(dataset)
            segment_data = np_to_var(dataset.X[:, :, :, None])
        else:
            segment_data = np_to_var(dataset[segment].X[:, :, :, None])
        min_example_idx = np.where(dataset[segment].y.max(axis=1) == np.amin(dataset[segment].y.max(axis=1)))[0]
        max_example_idx = np.where(dataset[segment].y.max(axis=1) == np.amax(dataset[segment].y.max(axis=1)))[0]
        min_example = segment_data[min_example_idx]
        max_example = segment_data[max_example_idx]
        min_feature_values = e.get_feature_importance(min_example)
        max_feature_values = e.get_feature_importance(max_example)
        min_feature_val = np.array(min_feature_values).squeeze()
        max_feature_val = np.array(max_feature_values).squeeze()
        np.save(f'{folder_name}/{global_vars.get("explainer")}_{segment}_min.npy', min_feature_val)
        np.save(f'{folder_name}/{global_vars.get("explainer")}_{segment}_max.npy', max_feature_val)
    for segment in ['test']:
        min_img_file = plot_feature_importance_netflow(folder_name, min_feature_val, global_vars.get('start_hour'),
                                        global_vars.get('dataset'), segment, global_vars.get('explainer'), title='min')
        max_img_file = plot_feature_importance_netflow(folder_name, max_feature_val, global_vars.get('start_hour'),
                                                   global_vars.get('dataset'), segment, global_vars.get('explainer'), title='max')
        shap_imgs.append(min_img_file)
        shap_imgs.append(max_img_file)
    story = []
    for im in shap_imgs:
        story.append(get_image(im))
    create_pdf_from_story(report_file_name, story)
    global_vars.get('sacred_ex').add_artifact(report_file_name)
    for im in shap_imgs:
        os.remove(im)
    gc.collect()
    return FEATURE_VALUES

'''
Use some explainer to get feature importance for each class
'''
def layer_feature_importance_report(model, dataset, folder_name):
    FEATURE_VALUES = {}
    feature_mean = {}
    vmin = np.inf
    vmax = -np.inf
    report_file_name = f'{folder_name}/{global_vars.get("report")}_{global_vars.get("explainer")}.pdf'
    train_data = np_to_var(dataset['train'].X[:, :, :, None])
    model.cpu()
    if 'Ensemble' in type(model).__name__:
        for mod in model.models:
            if 'Ensemble' in type(mod).__name__:
                for inner_mod in mod.models:
                    inner_mod.cpu()
                    inner_mod.eval()
            mod.cpu()
            mod.eval()
    for layer in model.models:
        feature_mean[type(layer).__name__] = {}
        e = globals()[f'{global_vars.get("explainer")}_explainer'](model, layer)
        shap_imgs = []
        for segment in ['test']:
            if segment == 'both':
                dataset = unify_dataset(dataset)
                segment_data = np_to_var(dataset.X[:, :, :, None])
            else:
                segment_data = np_to_var(dataset[segment].X[:, :, :, None])
            print(f'calculating {global_vars.get("explainer")} values for {int(segment_data.shape[0] * global_vars.get("explainer_sampling_rate"))} samples')
            segment_examples = segment_data[np.random.choice(segment_data.shape[0], int(segment_data.shape[0] * global_vars.get("explainer_sampling_rate")), replace=False)]
            feature_values = e.get_feature_importance(segment_examples)
            feature_val = np.array(feature_values).squeeze()
            feature_mean[type(layer).__name__][segment] = np.mean(feature_val, axis=1)
            np.save(f'{folder_name}/{global_vars.get("explainer")}_{segment}.npy', feature_mean[type(layer).__name__][segment])
            feature_value = np.concatenate(feature_mean[type(layer).__name__][segment], axis=0)
            feature_value = (feature_value - np.mean(feature_value)) / np.std(feature_value)
            FEATURE_VALUES[segment] = feature_value
            if feature_mean[type(layer).__name__][segment].min() < vmin:
                vmin = feature_mean[type(layer).__name__][segment].min()
            if feature_mean[type(layer).__name__][segment].max() > vmax:
                vmax = feature_mean[type(layer).__name__][segment].max()
    for key, val in feature_mean.items():
        for segment in ['test']:
            img_file = plot_feature_importance_netflow(folder_name, val[segment], global_vars.get('start_hour'),
                                        global_vars.get('dataset'), segment, global_vars.get('explainer'),
                                                       title=f'submodel_{key}')
            shap_imgs.append(img_file)
    story = []
    for im in shap_imgs:
        story.append(get_image(im))
    create_pdf_from_story(report_file_name, story)
    global_vars.get('sacred_ex').add_artifact(report_file_name)
    for im in shap_imgs:
        os.remove(im)
    gc.collect()
    return FEATURE_VALUES


'''
open a captum insights GUI in browser
'''
def captum_insights_report(model, dataset, folder_name):
    def formatted_data_iter():
        dataset = dataset['train'].X
        dataloader = iter(
            torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
        )
        while True:
            images, labels = next(dataloader)
            yield Batch(inputs=images, labels=labels)

    visualizer = AttributionVisualizer(
        models=[model],
        score_func=lambda o: torch.nn.functional.softmax(o, 1),
        classes=["1", "2", "3", "4", "5"],
        features=[
            ImageFeature(
                "TS"
            )
        ],
        dataset=dataset['train'].X,
    )
'''
Use shap to visualize CNN gradients
'''
def shap_gradient_report(model, dataset, folder_name):
    model = model.cpu()
    report_file_name = f'{folder_name}/{global_vars.get("report")}.pdf'
    train_data = np_to_var(dataset['train'].X[:, :, :, None])
    story = []
    shap_imgs = []
    all_paths = []
    segment_examples = {}
    segment_labels = {}
    for segment in ['train', 'test']:
        segment_data = np_to_var(dataset[segment].X[:, :, :, None])
        selected_examples = np.random.choice(segment_data.shape[0], int(segment_data.shape[0] * global_vars.get('explainer_sampling_rate')), replace=False)
        segment_examples[segment] = segment_data[selected_examples]
        segment_labels[segment] = dataset[segment].y[selected_examples]

    shap_rankings = {'train': OrderedDict(), 'test': OrderedDict()}
    prev_layer = None
    for layer_idx, layer in list(enumerate(list(model.children())))[global_vars.get('layer_idx_cutoff'):]:
        if layer_idx > 0 and type(prev_layer) == nn.Conv2d: # we only take layers whose INPUT is a conv
            e = shap.GradientExplainer((model, list(model.children())[layer_idx]), train_data)
            for segment in ['train', 'test']:
                plt.clf()
                print(f'Getting shap values for {len(segment_examples[segment])} {segment} samples')
                shap_values, indexes = e.shap_values(segment_examples[segment], ranked_outputs=2, nsamples=200)

                shap_val = np.array(shap_values[0]).squeeze()
                shap_abs = np.absolute(shap_val)
                shap_sum = np.sum(shap_abs, axis=0) # sum on sample axis
                if shap_sum.ndim > 1:
                    shap_sum = np.sum(shap_sum, axis=1) # sum on time axis
                shap_sum_idx = np.argsort(shap_sum) # sort
                for filter_idx in shap_sum_idx:
                    shap_rankings[segment][f'layer_{layer_idx-1}_filter_{filter_idx}'] = shap_sum[filter_idx] # we use layer_idx-1 because GradientExplainer looks at an INPUT of the layer

                if global_vars.get('plot'):
                    index_names = np.vectorize(lambda x: label_by_idx(x))(indexes)
                    shap.image_plot(shap_values, -segment_examples[segment].numpy(), labels=index_names)
                    plt.suptitle(f'SHAP gradient values for dataset: {global_vars.get("dataset")}, segment: {segment}, layer {layer_idx}\n'
                                 f'segment labels:{[label_by_idx(segment_labels[segment][i]) for i in range(len(segment_labels[segment]))]}', fontsize=10)
                    shap_img_file = f'temp/{get_next_temp_image_name("temp")}.png'
                    shap_imgs.append(shap_img_file)
                    plt.savefig(shap_img_file, dpi=200)
                    story.append(get_image(shap_img_file))
                    all_paths.append(shap_img_file)
        prev_layer = layer

    write_dict(shap_rankings['train'], f'{folder_name}/shap_rankings_train.txt')
    write_dict(shap_rankings['test'], f'{folder_name}/shap_rankings_test.txt')
    if global_vars.get('plot'):
        create_pdf_from_story(report_file_name, story)
        global_vars.get('sacred_ex').add_artifact(report_file_name)
    for im in all_paths:
        os.remove(im)


if __name__ == '__main__':
    # features = np.load('results/140_1_feature_importance_netflow_asflow_as_to_test_20940/deeplift_test.npy')
    # for f_idx in range(len(features)):
    #     features[f_idx] = features[f_idx] - features[f_idx].mean()
    #     features[f_idx] = features[f_idx] / features[f_idx].max()
    # plot_feature_importance_netflow('/home/user/Documents/eladr/EEGNAS/EEGNAS/visualization/results', features, 15, 'netflow_asflow', 'test', 'deeplift')
    features = np.load('results/210_1_feature_importance_BCI_IV_2a/deeplift_test.npy')
    set_default_config('../configurations/config.ini')
    global_vars.set('dataset', 'BCI_IV_2a')
    set_params_by_dataset('../configurations/dataset_params.ini')
    global_vars.set('frequency', 250)
    plot_topo_feature_importance('/home/user/Documents/eladr/EEGNAS/EEGNAS/visualization/results', features)
