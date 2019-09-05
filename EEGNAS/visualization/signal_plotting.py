import matplotlib
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from EEGNAS.utilities.misc import eeg_label_by_idx
from EEGNAS.utilities.monitors import get_eval_function

img_name_counter = 1


def get_next_im_filename():
    im_files = list(os.walk('temp'))[0][2]
    if len(im_files) == 0:
        im_num = 1
    else:
        im_nums = [int(x[:-4]) for x in im_files]
        im_nums.sort()
        im_num = im_nums[-1] + 1
    return im_num


def tf_plot(tf_trial_avgs, title, vmax=None, yscale='linear'):
    matplotlib.rcParams.update({'font.size': 16})
    plt.figure(num=None, figsize=(6 * len(tf_trial_avgs), 7), dpi=80, facecolor='w', edgecolor='k')
    for index, tf in enumerate(tf_trial_avgs):
        ax = plt.subplot(1, len(tf_trial_avgs), index+1)
        ax.set_xlabel('time points')
        if index == 0:
            ax.set_ylabel('frequency (Hz)')
        ax.set_yscale(yscale)
        cf = ax.contourf(tf, 40, vmin=0, vmax=vmax)
        try:
            ax.set_title(eeg_label_by_idx(index))
        except KeyError:
            ax.set_title(f'EEG channel {index + 1}')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(cf, cax=cax, orientation='vertical')
    plt.suptitle(title)
    im_files = list(os.walk('temp'))[0][2]
    if len(im_files) == 0:
        im_num = 1
    else:
        im_nums = [int(x[:-4]) for x in im_files]
        im_nums.sort()
        im_num = im_nums[-1] + 1
    im_name = f'temp/{im_num}.png'
    plt.savefig(im_name, bbox_inches='tight')
    plt.close('all')
    return im_name


def plot_performance_frequency(performances, baselines, legend):
    colors = ['greed', 'red', 'blue']
    baseline_idx = 0
    im_names = []
    for idx, performance_series in enumerate(zip(*performances)):
        baseline = list(baselines.values())[baseline_idx]
        baseline_idx += 1
        for performance in performance_series:
            plt.plot(range(len(performance)), performance, color=colors.pop())
        colors = ['greed', 'red', 'blue']
        plt.axhline(baseline, xmin=0, xmax=1, color='black')
        if idx == len(performances[0]) - 1:
            plt.title('average performance-frequency')
        else:
            plt.title(f'subject {idx+1} performance-frequency')
        plt.legend(legend)
        plt.ylabel(get_eval_function().__name__)
        im_name = f'temp/{get_next_im_filename()}.png'
        plt.savefig(im_name, bbox_inches='tight')
        plt.clf()
        im_names.append(im_name)
    return im_names


def plot_tensors(tensor, title, num_cols=8):
    global img_name_counter
    tensor = np.swapaxes(tensor, 1, 2)
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    num_kernels = tensor.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i+1)
        im = ax1.imshow(tensor[i].squeeze(axis=2), cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    fig.suptitle(f'{title}, Tensor shape: {tensor.shape}')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    img_name = f'temp/{img_name_counter}.png'
    plt.savefig(f'{img_name}')
    plt.close('all')
    img_name_counter += 1
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
    img_name = f'temp/{img_name_counter}.png'
    plt.savefig(img_name, bbox_inches='tight')
    plt.close('all')
    img_name_counter += 1
    return img_name
