from scipy import fft
import numpy as np

def preprocess_seizurenet(sample):
    for channel in sample:
        fft_chan = np.abs(fft(channel))
