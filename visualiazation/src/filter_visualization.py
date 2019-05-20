"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import torch
from torch.optim import Adam
from braindecode.torch_ext.util import np_to_var
from models_generation import target_model
from naiveNAS import NaiveNAS
import globals
import numpy as np
from data_preprocessing import get_train_val_test
from scipy.io import savemat
from BCI_IV_2a_experiment import get_normal_settings, set_params_by_dataset
from torchvision import utils
import matplotlib.pyplot as plt


def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    '''
    vistensor: visuzlization tensor
        @ch: visualization channel
        @allkernels: visualization all tensores
    '''

    n, c, w, h = tensor.shape
    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure(figsize=(nrow, rows))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


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
    globals.set('max_epochs', 50)
    globals.set('max_increase_epochs', 3)
    globals.set('final_max_epochs', 800)
    globals.set('final_max_increase_epochs', 80)
    globals.set('cuda', True)
    globals.set('data_folder', '../../data/')
    globals.set('low_cut_hz', 0)
    globals.set('dataset', 'BCI_IV_2b')
    set_params_by_dataset()
    model_selection = 'deep4'
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
    pass

