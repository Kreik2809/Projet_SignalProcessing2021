import audiofile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob, os, random


"""
    Read a given wav audio file and return the signal of it and its sampling rate.
      @path : path to the .wav file [string]

      @return : the signal [ndarray] and the sampling rate of the .wav file [int]
"""
def read_wavfile(path):
    signal, sampling_rate = audiofile.read(path)
    return signal, sampling_rate

"""
    Normalize a signal in order to make his value ranges from -1 to 1.
      @signal : the signal [ndarray]

      @return : the normalized signal [ndarray]
"""
def normalize(signal):
    min_value = abs(min(signal)) 
    max_value = abs(max(signal))

    norm = max(min_value, max_value)

    return signal/norm
"""
    Normalize a signal in order to make his value ranges from -1 to 1.
      @signal : the signal [ndarray]

      @return : the normalized signal [ndarray]
"""
def split(signal, sampling_rate, window_width, sliding_step):
    window_samples = sampling_rate * (window_width/1000)
    sliding_samples = sampling_rate * (sliding_step/1000)

    windows = []
    count = 0
    current_win = []
    is_window = True
    is_step = False
    for i in range(len(signal)):
        if is_window:
            if count < window_samples:
                current_win.append(signal[i])
                count += 1
            else:
                windows.append(current_win)
                current_win = []
                count = 0
                is_window = False
                is_step = True
        if(is_step):
            if count < sliding_samples:
                count += 1
            else:
                count = 0
                is_window = True
                is_step = False
                current_win.append(signal[i])
                count += 1

    if is_window:
        windows.append(current_win)

    return windows

"""
    Return the energy of the given signal
"""
def compute_energy(signal):
    energy = 0
    for i in range(len(signal)):
        energy += (abs(signal[i]))**2
    return energy


"""
    path of the directory where utterances are stored
"""
def auto_correlation_pitch_estim(path):
    #1.
    os.chdir(path)
    files = glob.glob("*.wav")
    
    choosed_files = random.sample(files, 5)
    signal = []
    for file in choosed_files:
        current_signal, sampling_rate = read_wavfile(file)
        signal.extend(current_signal)

    #2.
    signal = normalize(signal)
    plt.subplot(211)
    plt.plot(signal)
    
    #3.
    frames = split(signal, sampling_rate, 50, 0)
    list_energies = []
    for s in frames:
        list_energies.append(compute_energy(s))
    plt.subplot(212)
    plt.plot(list_energies)
    plt.show()

auto_correlation_pitch_estim("../../data/bdl")




#data/arctic_a0001.wav
