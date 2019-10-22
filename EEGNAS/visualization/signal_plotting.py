import matplotlib
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from EEGNAS.utilities.misc import eeg_label_by_idx
from EEGNAS.utilities.monitors import get_eval_function

import numpy as np
import warnings
try:
    import matplotlib.pyplot as pl
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass

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


def tf_plot(tf_trial_avgs, title, vmax, yscale='linear'):
    matplotlib.rcParams.update({'font.size': 16})
    plt.figure(num=None, figsize=(6 * len(tf_trial_avgs), 7), dpi=80, facecolor='w', edgecolor='k')
    for index, tf in enumerate(tf_trial_avgs):
        ax = plt.subplot(1, len(tf_trial_avgs), index+1)
        ax.set_xlabel('time points')
        if index == 0:
            ax.set_ylabel('frequency (Hz)')
        ax.set_yscale(yscale)
        cmap = plt.get_cmap("viridis")
        cf = ax.contourf(np.clip(tf, -3, 3), levels=np.linspace(-3, 3, 100), cmap=cmap)
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
    im_names = []
    for idx, performance_series in enumerate(zip(*performances)):
        baseline = baselines[idx]
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


def image_plot(shap_values, x, labels=None, show=True, width=20, aspect=0.2, hspace=0.2, labelpad=None):
    """ Plots SHAP values for image inputs.
    """

    multi_output = True
    if type(shap_values) != list:
        multi_output = False
        shap_values = [shap_values]

    # make sure labels
    if labels is not None:
        assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
        if multi_output:
            assert labels.shape[1] == len(shap_values), "Labels must have a column for each output in shap_values!"
        else:
            assert len(labels.shape) == 1, "Labels must be a vector for single output shap_values."

    label_kwargs = {} if labelpad is None else {'pad': labelpad}

    # plot our explanations
    fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]
    fig, axes = pl.subplots(nrows=x.shape[0], ncols=len(shap_values) + 1, figsize=fig_size)
    if len(axes.shape) == 1:
        axes = axes.reshape(1,axes.size)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])
        if x_curr.max() > 1:
            x_curr /= 255.

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (0.2989 * x_curr[:,:,0] + 0.5870 * x_curr[:,:,1] + 0.1140 * x_curr[:,:,2]) # rgb to gray
        else:
            x_curr_gray = x_curr

        axes[row,0].imshow(x_curr, cmap=pl.get_cmap('gray'), aspect='auto')
        axes[row,0].axis('off')
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
        max_val = np.nanpercentile(abs_vals, 99.9)
        for i in range(len(shap_values)):
            if labels is not None:
                axes[row,i+1].set_title(labels[row,i], **label_kwargs)
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            axes[row,i+1].imshow(x_curr_gray, cmap=pl.get_cmap('gray'), alpha=0.15, extent=(-1, sv.shape[0], sv.shape[1], -1), aspect='auto')
            im = axes[row,i+1].imshow(sv, cmap=pl.get_cmap('seismic'), vmin=-max_val, vmax=max_val, aspect='auto')
            axes[row,i+1].axis('off')
    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0]/aspect)
    cb.outline.set_visible(False)
    if show:
        pl.show()

