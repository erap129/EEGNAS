import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
plt.interactive(False)

def tf_plot(data):
    min_freq = 2
    max_freq = 30
    num_frex = 40
    frex = np.linspace(min_freq, max_freq, num_frex)
    range_cycles = [4, 10]
    srate = 250
    time_points = 1126
    num_trials = 192
    channel2use = 0

    s = np.logspace(math.log10(range_cycles[0]), math.log10(range_cycles[1]), num_frex) / (2 * math.pi * frex)
    wavtime = np.arange(-2, 2, 1/srate)
    half_wave = (len(wavtime) - 1) / 2

    n_wave = len(wavtime)
    n_data = time_points
    n_conv = n_wave + n_data - 1

    tf = np.zeros((len(frex), time_points, num_trials))

    for fi in range(len(frex)):
        wavelet = np.exp(2 * 1j * math.pi * frex[fi] * wavtime) * np.exp(-wavtime ** 2 / (2*s[fi] ** 2))
        wavelet_x = np.fft.fft(wavelet, n_conv)
        wavelet_x = wavelet_x / np.max(wavelet_x)

        for triali in range(num_trials):
            data_x = np.fft.fft(data[triali, channel2use, :].squeeze(), n_conv)
            ass = np.fft.ifft(wavelet_x * data_x)
            ass = ass[500:1626]
            tf[fi, :, triali] = abs(ass) ** 2

    tf_trial_avg = np.mean(tf, axis=2)
    fig, ax = plt.subplots()
    cf = ax.contourf(tf_trial_avg, 40)
    cbar = fig.colorbar(cf)
    plt.show()
    
    fig.savefig('test.png')