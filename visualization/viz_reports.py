import os
from collections import OrderedDict, defaultdict
from copy import deepcopy

from braindecode.torch_ext.util import np_to_var
from reportlab.platypus import Paragraph

import global_vars
from NASUtils import evaluate_single_model
from data_preprocessing import get_dataset
from utilities.data_utils import get_dummy_input, prepare_data_for_NN
from utilities.misc import unify_dataset, label_by_idx
from utilities.monitors import get_eval_function
from visualization.deconvolution import ConvDeconvNet
from visualization.pdf_utils import get_image, create_pdf_from_story, create_pdf
from visualization.signal_plotting import tf_plot, plot_performance_frequency
import numpy as np
from torch import nn
from visualization.viz_utils import pretrain_model_on_filtered_data, create_max_examples_per_channel, \
    get_max_examples_per_channel
from visualization.wavelet_functions import get_tf_data_efficient


def perturbation_report(model, dataset, folder_name):
    report_file_name = f'{folder_name}/perturbation_{global_vars.get("band_filter").__name__}.pdf'
    if os.path.isfile(report_file_name):
        return
    eeg_chans = list(range(get_dummy_input().shape[1]))
    tf_plots = []
    dataset = unify_dataset(dataset)
    for frequency in range(global_vars.get("low_freq"), global_vars.get("high_freq")+1):
        single_subj_dataset = deepcopy(dataset)
        perturbed_data = global_vars.get('band_filter')(single_subj_dataset.X,
                           max(1, frequency - 1), frequency + 1, global_vars.get('frequency'))
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


def performance_frequency_report(pretrained_model, dataset, folder_name):
    report_file_name = f'{folder_name}/performance_frequency_{global_vars.get("band_filter").__name__}.pdf'
    if os.path.isfile(report_file_name):
        return
    baselines = OrderedDict()
    freq_models = pretrain_model_on_filtered_data(pretrained_model, global_vars.get('low_freq'),
                                                  global_vars.get('high_freq'))
    all_performances = []
    all_performances_freq = []
    for subject in global_vars.get('subjects_to_check'):
        single_subj_performances = []
        single_subj_performances_freq = []
        single_subj_dataset = get_dataset(subject)
        baselines[subject] = evaluate_single_model(pretrained_model, single_subj_dataset['test'].X,
                                                     single_subj_dataset['test'].y,
                                                     eval_func=get_eval_function())
        for freq in range(global_vars.get('low_freq'), global_vars.get('high_freq')+1):
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
    baselines['average'] = np.average(list(baselines.values()))
    all_performances.append(np.average(all_performances, axis=0))
    all_performances_freq.append(np.average(all_performances_freq, axis=0))
    performance_plot_imgs = plot_performance_frequency([all_performances, all_performances_freq], baselines,
                                                       legend=['no retraining', 'with retraining', 'unperturbed'])
    story = [get_image(tf) for tf in performance_plot_imgs]
    create_pdf_from_story(report_file_name, story)
    for tf in performance_plot_imgs:
        os.remove(tf)


def kernel_deconvolution_report(model, dataset, folder_name):
    by_class_str = ''
    if global_vars.get('deconvolution_by_class'):
        by_class_str = '_by_class'
    report_file_name = f'{folder_name}/step1_all_kernels{by_class_str}.pdf'
    if os.path.isfile(report_file_name):
        return
    conv_deconv = ConvDeconvNet(model)
    eeg_chans = list(range(global_vars.get('eeg_chans')))
    tf_plots = []
    class_examples = []
    if global_vars.get('deconvolution_by_class'):
        for class_idx in range(global_vars.get('n_classes')):
            class_examples.append(dataset['train'].X[np.where(dataset['train'].y == class_idx)])
    else:
        class_examples.append(dataset['train'].X)
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
                    if global_vars.get('deconvolution_by_class'):
                        class_title = label_by_idx(class_idx)
                    else:
                        class_title = 'all'
                    tf_plots.append(tf_plot(subj_tfs, f'kernel deconvolution for layer {layer_idx},'
                                            f' filter {filter_idx}, class {class_title}'))
    create_pdf(report_file_name, tf_plots)
    for im in tf_plots:
        os.remove(im)


def avg_class_tf_report(model, dataset, folder_name):
    report_file_name = f'{folder_name}/avg_class_tf.pdf'
    if os.path.isfile(report_file_name):
        return
    eeg_chans = list(range(global_vars.get('eeg_chans')))
    unify_dataset(dataset)
    class_examples = []
    for class_idx in range(global_vars.get('n_classes')):
        class_examples.append(dataset.X[np.where(dataset.y == class_idx)])
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
    create_pdf_from_story(report_file_name, story)
    for tf in tf_plots:
        os.remove(tf)


def gradient_ascent_report(pretrained_model, dataset, folder_name, layer_idx_cutoff=0):
    report_file_name = f'{folder_name}/gradient_ascent_{global_vars.get("steps")}_steps.pdf'
    if os.path.isfile(report_file_name):
        return
    eeg_chans = list(range(global_vars.get('eeg_chans')))
    plot_dict = OrderedDict()
    plot_imgs = OrderedDict()
    for layer_idx, layer in list(enumerate(list(pretrained_model.children())))[layer_idx_cutoff:]:
        max_examples = create_max_examples_per_channel(layer_idx, pretrained_model, steps=global_vars.get('steps'))
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
    create_pdf_from_story(report_file_name, story)
    for im in img_paths:
        os.remove(im)


def find_optimal_samples_report(pretrained_model, dataset, folder_name):
    report_file_name = f'{folder_name}/step3_tf_plots_real.pdf'
    if os.path.isfile(report_file_name):
        return
    eeg_chans = list(range(global_vars.get('eeg_chans')))
    plot_dict = OrderedDict()
    for layer_idx, layer in enumerate(list(pretrained_model.children())):
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


def shap_report(model, dataset, folder_name):
    train_data = np_to_var(dataset['train'].X[:, :, :, None])
    test_data = np_to_var(dataset['test'].X[:, :, :, None])
    e = shap.DeepExplainer(model, train_data)
    shap_values = e.shap_values(test_data)
    tf_plots = []
    # for class_values in zip(shap_values):



