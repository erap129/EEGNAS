from scipy.signal import butter, lfilter
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    i, u = butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
    return y


def batch_filter_apply(filter_func, data, lowcut, highcut, fs, order=5):
    for data_idx in range(len(data)):
        for channel_idx in range(len(data[data_idx])):
            data[data_idx, channel_idx] = filter_func(data[data_idx, channel_idx], lowcut, highcut, fs, order)
    return data


def get_fft(signal, frequency):
    nyquist = int(frequency / 2) - 1
    freqs = np.fft.fftfreq(len(signal), 1 / frequency)
    freq_idx = np.argmax(freqs >= nyquist)
    return freqs[:freq_idx], abs(np.fft.fft(signal)[:freq_idx])


def filter_dataset(dataset, filt, low_freq, high_freq, frequency):
    for key in dataset.keys():
        dataset[key].X = filt(dataset[key].X, low_freq, high_freq, frequency).astype(np.float32)

