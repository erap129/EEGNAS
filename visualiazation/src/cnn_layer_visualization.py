"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import torch
from torch.optim import Adam
from torch.autograd import Variable
from braindecode.torch_ext.util import np_to_var
from visualiazation.src.misc_functions import preprocess_image, recreate_image, save_image, create_mne
import matplotlib.pyplot as plt
from models_generation import target_model
from naiveNAS import NaiveNAS
import globals
import numpy as np
from data_preprocessing import get_ner_train_val_test, get_bci_iv_2a_train_val_test, get_bci_iv_2b_train_val_test
from scipy import signal
from scipy.io import savemat
from BCI_IV_2a_experiment import get_normal_settings, set_params_by_dataset

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.float32(np.random.uniform(-1, 1, (1, globals.get('eeg_chans'),
                                                            globals.get('input_time_len'), 1)))
        # Process image and return variable
        # processed_image = preprocess_image(random_image, False)
        processed_image = torch.tensor(np_to_var(random_image), requires_grad=True, device='cuda')
        # processed_image = torch.Tensor.new_tensor(data=torch.Tensor(random_image), requires_grad=True, device='cuda')
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 1001):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            # self.created_image = create_mne(processed_image)

            if i % 100 == 0:
                savemat(f'X_step_{int(i/10)}.mat', {'X': processed_image.detach().cpu().numpy().squeeze()[None, :, :]})


def plot_real_EEG():
    # train, val, test = get_bci_iv_2a_train_val_test('../../data/BCI_IV', 4, 0)
    # train, val, test = get_ner_train_val_test('../../data/')
    train, val, test = get_bci_iv_2b_train_val_test(1)
    data = train.X[0].squeeze()
    for channel in data:
        CNNLayerVisualization.create_spectrogram_2(channel, fs=250)


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
    set_params_by_dataset()
    selection = 'deep4'
    cnn_layer = {'evolution': 6, 'deep4': 20}
    filter_pos = {'evolution': 5, 'deep4': 150}
    model = {'evolution': torch.load('../models/best_model_9_8_6_7_2_1_3_4_5.th'),
                        'deep4': target_model('deep')}
    train, val, test = get_bci_iv_2b_train_val_test(subject_id=1)
    stop_criterion, iterator, loss_function, monitors = get_normal_settings()
    naiveNAS = NaiveNAS(iterator=iterator, exp_folder=None, exp_name=None,
                        train_set=train, val_set=val, test_set=test,
                        stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                        config=globals.config, subject_id='all', fieldnames=None, strategy='cross_subject',
                        evolution_file=None, csv_file=None)
    _, _, pretrained_model, _, _ = naiveNAS.evaluate_model(model[selection], final_evaluation=True)
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer[selection], filter_pos[selection])
    layer_vis.visualise_layer_with_hooks()
