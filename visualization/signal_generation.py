import numpy as np
from UliEngineering.SignalProcessing.Simulation import sine_wave
import matplotlib.pyplot as plt
import matplotlib

from visualization.dsp_functions import butter_bandpass_filter, butter_bandstop_filter
from visualization.wavelet_functions import subtract_frequency

matplotlib.use('TkAgg')


def generate_sine_wave(length, samplerate, frequencies):
    wave = np.zeros(int(length * samplerate))
    for frequency in frequencies:
        wave += sine_wave(frequency=frequency, samplerate=samplerate, length=length)
    return wave


wave = generate_sine_wave(5, 250, [3])[None, None, :]
subtracted_wave = subtract_frequency(wave, 7, 250)
filtered_wave = butter_bandstop_filter(wave[0, 0], 6, 8, 250)
plt.plot(wave[0, 0])
plt.plot(subtracted_wave[0, 0], color='red')
plt.plot(filtered_wave, color='blue')
plt.show()
