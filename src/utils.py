import audiofile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

"""
   Read a given wav audio file and return the signal of it and its sampling rate. 
"""
def read_wavfile(path):
    signal, sampling_rate = audiofile.read(path)
    return signal, sampling_rate
"""
    Normalize a signal in order to make his value ranges from -1 to 1.
"""
def normalize(signal):
    min_value = abs(min(signal)) 
    max_value = abs(max(signal))

    norm = max(min_value, max_value)

    return signal/norm
"""
    window_width : [ms]
    sliding_step : [ms]
"""
def split(signal, sampling_rate, window_width, sliding_step):
    window_samples = sampling_rate * (window_width/1000)
    sliding_samples = sampling_rate * (sliding_step/1000)

    windows = []
    count = 0
    current_win = []
    for i in range(len(signal)):
            current_win.append(signal[i]



signal, sampling_rate = read_wavfile("data/arctic_a0001.wav")


plt.plot(normalize(signal))

plt.show()

#data/arctic_a0001.wav
