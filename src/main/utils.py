import audiofile
import numpy as np
import numpy.lib.stride_tricks as npst
import matplotlib.pyplot as plt
from scipy import signal
import glob, os, random
import xcorr

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
Split the signal in frames with an overlapping step.
      @signal : the signal [ndarray]
      @sampling_rate : the sampling rate of the signal [int]
      @window_width : the window size in ms [int]
      @sliding_step : the sliding step in ms [int]

      @return : windows generated [list]
"""
def split(signal, sampling_rate, window_width, sliding_step):
    window_samples = int(sampling_rate * (window_width/1000))
    sliding_samples = int(sampling_rate * (sliding_step/1000))

    v = npst.sliding_window_view(signal, window_samples)[::sliding_samples, :]

    return v

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
def auto_correlation_pitch_estim(path_1):
    #1.
    os.chdir(path_1)
    files = glob.glob("*.wav")
    
    choosed_files = random.sample(files, 5)
    signal_1 = []
    for file in choosed_files:
        current_signal, sampling_rate = read_wavfile(file)
        signal_1.extend(current_signal)
    
    #2.
    signal_1 = normalize(signal_1)
    plt.subplot(311)
    plt.plot(signal_1)
    
    #3.
    list_energies_1 = []
    frames_1 = split(signal_1, sampling_rate, 50, 15)
    for s in frames_1:
        list_energies_1.append(compute_energy(s))

    plt.subplot(312)
    plt.plot(list_energies_1)
    
    #5.
    tresh = 10
    voiced_segments_1 = []
    for i in range(len(list_energies_1)):
        if list_energies_1[i] >= tresh:
            voiced_segments_1.append(frames_1[i])
        
    #Test to evaluate tresh value
    voiced_signal_1 = []
    for s in voiced_segments_1:
        voiced_signal_1.extend(s)
    plt.subplot(313)
    plt.plot(voiced_signal_1)
    plt.show()        

    #6.
    #the two correlated signals must be of same length
    
    c = voiced_segments_1[0]
    for segment in voiced_segments_1[1:]:
        if (len(c) != len(segment) ):
            maxl = max(len(c), len(segment))
            c = np.hstack([c, np.zeros(maxl-len(c))])
            segment = np.hstack([segment, np.zeros(maxl-len(segment))])
        lags, c = xcorr.xcorr(c, segment, maxlags=100)

    plt.subplot(211)
    plt.plot(c)
    plt.subplot(212)
    plt.plot(lags)

    peaks = signal.find_peaks(c)

    diff_samples = peaks[0][1] - peaks[0][0]

    time = diff_samples/16000 #sampling rate
    pitch = 1 / time
    print(pitch)
    plt.show()


auto_correlation_pitch_estim("../../data/bdl_a")

