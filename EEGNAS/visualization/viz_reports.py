import os
import random
from collections import OrderedDict
from copy import deepcopy
import shap
import torch
from braindecode.torch_ext.util import np_to_var
from mpl_toolkits.axes_grid1 import make_axes_locatable
from reportlab.platypus import Paragraph
from EEGNAS import global_vars
from EEGNAS.data.netflow.netflow_data_utils import turn_netflow_into_classification, get_netflow_threshold
from EEGNAS.utilities.NAS_utils import evaluate_single_model
from EEGNAS.data_preprocessing import get_dataset
from EEGNAS.utilities.NN_utils import get_intermediate_layer_value, get_class_distribution
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
    if global_vars.get('to_csv'):
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
                    # prev_filter_val = torch.mean(get_intermediate_layer_value(model, X, layer_idx), axis=[0,2,3])
                    reconstruction = conv_deconv.forward(X, layer_idx, filter_idx)
                    # after_filter_val = torch.mean(get_intermediate_layer_value(model, reconstruction, layer_idx), axis=[0,2,3])
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
                    # with open(f'{report_file_name[:-4]}.txt', 'a+') as f:
                    #     print(f'filter values for layer {layer_idx}, filter {filter_idx}, class {class_idx}:\nPrevious:{prev_filter_val}'
                    #           f'\nAfter:{after_filter_val}\nThe relevant comparison is Prev:{prev_filter_val[filter_idx]}'
                    #           f' against After:{after_filter_val[filter_idx]}\n'
                    #           f'Class distribution before:{get_class_distribution(model, X)}\n'
                    #           f'Class distribution after:{get_class_distribution(model, reconstruction)}\n', file=f)
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


'''
Use shap to get feature importance for each class
'''
def shap_report(model, dataset, folder_name):
    if global_vars.get('dataset') == 'netflow_asflow':
        file_path = f"{os.path.dirname(os.path.abspath(__file__))}/../data/netflow/{global_vars.get('as_to_test')}_{global_vars.get('date_range')}.csv"
        for segment in ['train', 'test']:
            dataset[segment].y = turn_netflow_into_classification(dataset[segment].X, dataset[segment].y, get_netflow_threshold(file_path, global_vars.get('netflow_threshold_std')))
        global_vars.set('n_classes', 2)
    report_file_name = f'{folder_name}/{global_vars.get("report")}.pdf'
    train_data = np_to_var(dataset['train'].X[:, :, :, None])
    print(f'training DeepExplainer on {int(train_data.shape[0] * global_vars.get("shap_sampling_rate"))} samples')
    e = shap.DeepExplainer(model.cpu(), train_data[np.random.choice(train_data.shape[0], int(train_data.shape[0] * global_vars.get('shap_sampling_rate')) , replace=False)])
    shap_imgs = []
    for segment in ['train', 'test']:
        segment_data = np_to_var(dataset[segment].X[:, :, :, None])
        print(f'calculating SHAP values for {int(segment_data.shape[0] * global_vars.get("shap_sampling_rate"))} samples')
        segment_examples = segment_data[np.random.choice(segment_data.shape[0], int(train_data.shape[0] * global_vars.get("shap_sampling_rate")), replace=False)]
        shap_values = e.shap_values(segment_examples)

        shap_val = np.array(shap_values).squeeze()
        # shap_abs = np.absolute(shap_val)
        shap_sum = np.sum(shap_val, axis=1)

        f, axes = plt.subplots(global_vars.get('n_classes'), figsize=(20,10))
        for idx, ax in enumerate(axes):
            im = ax.imshow(shap_sum[idx], cmap='seismic', interpolation='nearest', aspect='auto')
            ax.set_title(f'class: {label_by_idx(idx)}')
            ax.set_yticks([i for i in range(global_vars.get('eeg_chans'))])
            ax.set_yticklabels([eeg_label_by_idx(i) for i in range(global_vars.get('eeg_chans'))])
            if global_vars.get('dataset') == 'netflow_asflow':
                ax.set_xticks(list(range(global_vars.get('input_height')))[::2])
                ax.set_xticklabels([(i+global_vars.get('start_hour')) % 24 for i in range(global_vars.get('input_height'))][::2])
            ax.tick_params(axis='both', which='major', labelsize=5)
        f.subplots_adjust(right=0.8)
        cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
        f.colorbar(im, cax=cbar_ax)

        plt.suptitle(f'SHAP values for dataset: {global_vars.get("dataset")}, segment: {segment}\nchannel order: {[eeg_label_by_idx(i) for i in range(global_vars.get("eeg_chans"))]}', fontsize=10)
        shap_img_file = f'temp/{get_next_temp_image_name("temp")}.png'
        shap_imgs.append(shap_img_file)
        plt.savefig(shap_img_file, dpi=300)
    story = []
    for im in shap_imgs:
        story.append(get_image(im))
    create_pdf_from_story(report_file_name, story)
    global_vars.get('sacred_ex').add_artifact(report_file_name)
    for im in shap_imgs:
        os.remove(im)


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
        selected_examples = np.random.choice(segment_data.shape[0], int(segment_data.shape[0] * global_vars.get('shap_sampling_rate')), replace=False)
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