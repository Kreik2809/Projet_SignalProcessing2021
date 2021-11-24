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
    is_window = True
    is_step = False
    for i in range(len(signal)):
        if is_window:
            if count < window_samples:
                current_win.append(signal[i])
                count = count + 1 
                print(count)
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
                print("hey2")
                count = 0
                is_window = True
                is_step = False

    if is_window:
        windows.append(current_win)

    return windows

signal, sampling_rate = read_wavfile("data/arctic_a0001.wav")

windows = split(normalize(signal), sampling_rate, 1000, 500)
plt.subplot(311)
plt.plot(windows[0])
plt.subplot(312)
plt.plot(windows[1])
plt.subplot(313)
plt.plot(normalize(signal))

plt.show()

#data/arctic_a0001.wav
