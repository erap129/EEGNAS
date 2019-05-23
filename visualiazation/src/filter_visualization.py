import os

import torch
from braindecode.torch_ext.util import np_to_var
from models_generation import target_model
from naiveNAS import NaiveNAS
import globals
from torch import nn
from data_preprocessing import get_train_val_test
from BCI_IV_2a_experiment import get_normal_settings, set_params_by_dataset
import matplotlib.pyplot as plt
import matplotlib
from visualiazation.src.pdf_utils import create_pdf
import numpy as np
from numpy import unravel_index
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use("TkAgg")
plt.interactive(False)
img_name_counter = 1


def get_max_examples(data, select_layer, model):
    act_maps = {}
    x = np_to_var(data[:, :, :, None]).cuda()
    modules = list(model.modules())[0]
    for idx, example in enumerate(x):
        example_x = example[None, :, :, :]
        for l in modules[:select_layer + 1]:
            example_x = l(example_x)
        act_maps[idx] = example_x
    height = act_maps[0].shape[1]
    width = act_maps[1].shape[2]
    selected_examples = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            selected_examples[h, w]\
                = unravel_index(np.array([act_map.squeeze()[h, w] for act_map in act_maps.values()])
                                .reshape(height, width).argmax(), act_maps.shape)
    print(selected_examples)


def get_intermediate_act_map(data, select_layer, model):
    x = np_to_var(data[:, :, :, None]).cuda()
    # x = np_to_var(data)
    # x = data
    modules = list(model.modules())[0]
    # untrained_modules = list(untrained_model.modules())[0]
    # print(f'start shape: {x.shape}')
    for l in modules[:select_layer + 1]:
      x = l(x)
    act_map = x.cpu().detach().numpy()
    act_map_avg = np.average(act_map, axis=0).swapaxes(0, 1).squeeze(axis=2)
    return act_map_avg


def plot_tensors(tensor, title, num_cols=8):
    global img_name_counter
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    # if not tensor.shape[-1]==3:
    #     raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i+1)
        im = ax1.imshow(tensor[i].squeeze(axis=2).swapaxes(0, 1), cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.suptitle(f'{title}, Tensor shape: {tensor.shape}')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    img_name = f'{img_name_counter}.png'
    plt.savefig(img_name)
    img_name_counter += 1
    # im = fig2img(fig)
    return img_name


def plot_one_tensor(tensor, title):
    global img_name_counter
    if not tensor.ndim == 2:
        raise Exception("assumes a 2D tensor")
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(tensor.swapaxes(0,1), cmap='gray')
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title(f'{title}, Tensor shape: {tensor.shape}')
    img_name = f'{img_name_counter}.png'
    plt.savefig(img_name, bbox_inches='tight')
    img_name_counter += 1
    return img_name


def plot_all_kernels_to_pdf(pretrained_model):
    img_paths = []
    for index, layer in enumerate(list(pretrained_model.children())):
        if isinstance(layer, nn.Conv2d):
            im = plot_tensors(layer.weight.detach().cpu().numpy(), f'Layer {index}')
            img_paths.append(im)
    create_pdf('step1_all_kernels.pdf', img_paths)
    for im in img_paths:
        os.remove(im)


def plot_avg_activation_maps(pretrained_model, train_set):
    img_paths = []
    left_X = train_set[subject_id].X[np.where(train_set[subject_id].y == 0)]
    right_X = train_set[subject_id].X[np.where(train_set[subject_id].y == 1)]
    for index, layer in enumerate(list(pretrained_model.children())):
        if isinstance(layer, nn.Conv2d):
            left_act_map = plot_one_tensor(get_intermediate_act_map(left_X, index, pretrained_model), f'Layer {index} Left')
            right_act_map = plot_one_tensor(get_intermediate_act_map(right_X, index, pretrained_model), f'Layer {index} Right')
            img_paths.extend([left_act_map, right_act_map])
    create_pdf('step2_avg_activation_maps.pdf', img_paths)
    for im in img_paths:
        os.remove(im)


def find_optimal_samples_per_filter(pretrained_model, train_set):
    for index, layer in enumerate(list(pretrained_model.children())):
        if isinstance(layer, nn.Conv2d):
            max_examples = get_max_examples(train_set[subject_id].X, index, pretrained_model)


if __name__ == '__main__':
    globals.set_dummy_config()
    globals.set('input_time_len', 1125)
    globals.set('n_classes', 4)
    globals.set('eeg_chans', 22)
    globals.set('valid_set_fraction', 0.2)
    globals.set('dataset', 'BCI_IV_2b')
    globals.set('batch_size', 60)
    globals.set('do_early_stop', True)
    globals.set('remember_best', True)
    globals.set('max_epochs', 1)
    globals.set('max_increase_epochs', 3)
    globals.set('final_max_epochs', 1)
    globals.set('final_max_increase_epochs', 80)
    globals.set('cuda', True)
    globals.set('data_folder', '../../data/')
    globals.set('low_cut_hz', 0)
    globals.set('dataset', 'BCI_IV_2b')
    set_params_by_dataset()
    model_selection = 'evolution'
    cnn_layer = {'evolution': 10, 'deep4': 25}
    filter_pos = {'evolution': 0, 'deep4': 0}
    model_dir = '91_x_BCI_IV_2b'
    model_name = 'best_model_5_1_8_7_9_2_3_4_6.th'
    model = {'evolution': torch.load(f'../../models/{model_dir}/{model_name}'),
                        'deep4': target_model('deep')}
    subject_id = 1
    train_set = {}
    val_set = {}
    test_set = {}
    train_set[subject_id], val_set[subject_id], test_set[subject_id] = \
        get_train_val_test(globals.get('data_folder'), subject_id, globals.get('low_cut_hz'))
    stop_criterion, iterator, loss_function, monitors = get_normal_settings()
    naiveNAS = NaiveNAS(iterator=iterator, exp_folder=None, exp_name=None,
                        train_set=train_set, val_set=val_set, test_set=test_set,
                        stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                        config=globals.config, subject_id=subject_id, fieldnames=None, strategy='cross_subject',
                        evolution_file=None, csv_file=None)
    _, _, pretrained_model, _, _ = naiveNAS.evaluate_model(model[model_selection], final_evaluation=True)
    # plot_all_kernels_to_pdf(pretrained_model)
    # plot_avg_activation_maps(pretrained_model, train_set)
    find_optimal_samples_per_filter(pretrained_model, train_set)
