import audiofile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

"""
   Read a wav audio file and return the signal of it and its sampling rate. 
"""
def read_wavfile(path):
    signal, sampling_rate = audiofile.read("data/arctic_a0001.wav")
    return signal, sampling
"""
    Normalize a signal in order to make his value range from -1 to 1.
"""
def normalize(signal):
    norm = np.linalg.norm(signal)
    signal = signal/norm
    return signal   


plt.plot(signal)
plt.show()
