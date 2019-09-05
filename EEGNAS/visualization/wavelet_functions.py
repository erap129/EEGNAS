import os
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy

plt.interactive(False)


def get_tf_data(data, channel, srate):
    min_freq = 2
    max_freq = 30
    num_frex = 40
    frex = np.linspace(min_freq, max_freq, num_frex)
    range_cycles = [4, 10]
    time_points = data.shape[2]
    num_trials = len(data)

    s = np.logspace(math.log10(range_cycles[0]), math.log10(range_cycles[1]), num_frex) / (2 * math.pi * frex)
    wavtime = np.arange(-2, 2 + (1/srate), 1 / srate)
    half_wave = int((len(wavtime) - 1) / 2)

    n_wave = len(wavtime)
    n_data = time_points
    n_conv = n_wave + n_data - 1

    tf = np.zeros((len(frex), time_points, num_trials))

    for fi in range(len(frex)):
        wavelet = np.exp(2 * 1j * math.pi * frex[fi] * wavtime) * np.exp(-wavtime ** 2 / (2 * s[fi] ** 2))
        wavelet_x = np.fft.fft(wavelet, n_conv)
        wavelet_x = wavelet_x / np.max(wavelet_x)

        for triali in range(num_trials):
            data_x = np.fft.fft(data[triali, channel, :].squeeze(), n_conv)
            ass = np.fft.ifft(wavelet_x * data_x)
            ass = ass[half_wave + 1:-half_wave+1]
            tf[fi, :, triali] = abs(ass) ** 2

    tf_trial_avg = np.mean(tf, axis=2)
    return tf_trial_avg


def get_tf_data_efficient(data, channel, srate):
    """
    :param data: data is a [num_trials X num_channels X trial_length] array
    :param channel: the channel to perform TF analysis on
    :return: the time frequency decomposition averaged over
    """
    min_freq = 2
    max_freq = 30
    num_frex = 40
    frex = np.linspace(min_freq, max_freq, num_frex)
    range_cycles = [4, 10]
    time_points = data.shape[2]
    num_trials = len(data)
    n_channels = data.shape[1]

    s = np.logspace(math.log10(range_cycles[0]), math.log10(range_cycles[1]), num_frex) / (2 * math.pi * frex)
    wavtime = np.arange(-2, 2+(1/srate), 1 / srate)
    half_wave = int((len(wavtime) - 1) / 2)

    n_wave = len(wavtime)
    n_data = time_points * num_trials
    n_conv = n_wave + n_data - 1

    tf = np.zeros((len(frex), time_points))
    all_data = data.reshape(n_channels, -1)
    data_x = np.fft.fft(all_data[channel], n_conv)

    for fi in range(len(frex)):
        wavelet = np.exp(2 * 1j * math.pi * frex[fi] * wavtime) * np.exp(-wavtime ** 2 / (2 * s[fi] ** 2))
        wavelet_x = np.fft.fft(wavelet, n_conv)
        wavelet_x = wavelet_x / np.max(wavelet_x)

        ass = np.fft.ifft(wavelet_x * data_x)
        ass = ass[half_wave + 1:-half_wave+1]
        ass = ass.reshape(num_trials, time_points)
        tf[fi, :] = np.mean(abs(ass) ** 2, axis=0)
    return tf


def subtract_frequency(data, freq, srate):
    data_copy = deepcopy(data)
    s = 7 / (2 * math.pi * freq)
    wavtime = np.arange(-2, 2 + (1 / srate), 1 / srate)
    half_wave = int((len(wavtime) - 1) / 2)
    sine_wave = np.exp(1j * 2 * math.pi * freq * wavtime)
    gaus_win = np.exp(-wavtime ** 2 / (2 * s ** 2))
    wavelet = sine_wave * gaus_win

    n_wave = len(wavtime)
    time_points = data.shape[2]
    num_trials = data.shape[0]
    n_data = time_points * num_trials
    n_conv = n_wave + n_data - 1
    wavelet_fft = np.fft.fft(wavelet, n_conv)
    wavelet_fft = wavelet_fft / np.max(wavelet_fft)

    n_channels = data.shape[1]
    all_data = data.reshape(n_channels, -1)

    for channel in range(n_channels):
        all_data_fft = np.fft.fft(all_data[channel], n_conv)
        unwanted_data = np.fft.ifft(wavelet_fft * all_data_fft)
        unwanted_data = unwanted_data[half_wave + 1:-half_wave + 1]
        unwanted_data = unwanted_data.reshape(num_trials, time_points)
        data_copy[:, channel, :] -= unwanted_data.real
    return data_copy


def subtract_frequency_dataset(dataset, freq, srate):
    for section in ['train', 'valid', 'test']:
        dataset[section].X = subtract_frequency(dataset[section].X, freq, srate)



