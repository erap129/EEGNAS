"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os

import mne
import numpy as np

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import models
from braindecode.torch_ext.util import np_to_var
from misc_functions import preprocess_image, recreate_image, save_image, create_mne
import matplotlib.pyplot as plt
from models_generation import target_model
import globals
import numpy as np
from braindecode.models import deep4, shallow_fbcsp, eegnet
from data_preprocessing import get_ner_train_val_test, get_bci_iv_2a_train_val_test
from scipy import signal
from scipy.io import wavfile

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

    @staticmethod
    def create_spectrogram_2(data, fs):
        plt.subplot(211)
        plt.plot(np.arange(len(data)), data)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.subplot(212)
        frequencies, times, spectrogram = signal.spectrogram(data, fs=fs, window='hamming', noverlap=150)
        plt.pcolormesh(times, frequencies, 10*np.log10(spectrogram))
        plt.ylim((0,50))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    @staticmethod
    def create_spectrogram(data):
        plt.subplot(211)
        plt.plot(np.arange(len(data)), data)
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        # Plot the spectrogram
        plt.subplot(212)
        powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(data, NFFT=100, Fs=250, noverlap=25)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.float32(np.random.uniform(-1, 1, (1, 22, 1125, 1)))
        # Process image and return variable
        # processed_image = preprocess_image(random_image, False)
        processed_image = Variable(np_to_var(random_image), requires_grad=True)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 101):
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
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = create_mne(processed_image)

            if i % 10 == 0:
                CNNLayerVisualization.create_spectrogram_2(np.array(processed_image.detach().numpy().squeeze()[0]), 250)


def plot_real_EEG():
    globals.set('valid_set_fraction', 0.2)
    train, val, test = get_bci_iv_2a_train_val_test('../../data/BCI_IV', 4, 0)
    # train, val, test = get_ner_train_val_test('../../data/')
    data = train.X[0].squeeze()
    for channel in data:
        CNNLayerVisualization.create_spectrogram_2(channel, fs=250)


if __name__ == '__main__':
    globals.set_dummy_config()
    globals.set('input_time_len', 1125)
    globals.set('n_classes', 4)
    globals.set('eeg_chans', 22)
    selection = 'deep4'
    cnn_layer = {'evolution': 6, 'deep4': 20}
    filter_pos = {'evolution': 5, 'deep4': 150}
    pretrained_model = {'evolution': torch.load('../models/best_model_9_8_6_7_2_1_3_4_5.th'),
                        'deep4': target_model('deep')}
    # layer_vis = CNNLayerVisualization(pretrained_model[selection], cnn_layer[selection], filter_pos[selection])
    # layer_vis.visualise_layer_with_hooks()
    #
    plot_real_EEG()