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
    Normalize a signal in order to make his value range from -1 to 1.
"""
def normalize(signal):
    min_value = abs(min(signal)) 
    max_value = abs(max(signal))

    norm = max(min_value, max_value)

    return signal/norm

signal, sampling_rate = read_wavfile("data/arctic_a0001.wav")


plt.plot(normalize(signal))

plt.show()

#data/arctic_a0001.wav
