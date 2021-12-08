import audiofile
import numpy as np
import numpy.lib.stride_tricks as npst
import matplotlib.pyplot as plt
from scipy import signal, fft
import glob, os, random

import scipy
import xcorr
import scikit_talkbox_lpc as scilpc

def read_wavfile(path):
    """
    Read a given wav audio file and return the signal of it and its sampling rate.
    @path : path to the .wav file [string]
    @return : the signal [ndarray] and the sampling rate of the .wav file [int]
    """
    signal, sampling_rate = audiofile.read(path)
    return signal, sampling_rate

def normalize(signal):
    """
    Normalize a signal in order to make his value ranges from -1 to 1.
    @signal : the signal [ndarray]
    @return : the normalized signal [ndarray]
    """
    min_value = abs(min(signal)) 
    max_value = abs(max(signal))

    norm = max(min_value, max_value)

    return signal/norm

def split(signal, sampling_rate, window_width, sliding_step):
    """
    Split the signal in frames with an overlapping step.
    @signal : the signal [ndarray]
    @sampling_rate : the sampling rate of the signal [int]
    @window_width : the window size in ms [int]
    @sliding_step : the sliding step in ms [int]

    @return : windows generated [list]
    """
    window_samples = int(sampling_rate * (window_width/1000))
    sliding_samples = int(sampling_rate * (sliding_step/1000))

    v = npst.sliding_window_view(signal, window_samples)[::sliding_samples, :]

    return v

def compute_energy(signal):
    """
    Return the energy of the given signal
    """
    energy = 0
    for i in range(len(signal)):
        energy += (abs(signal[i]))**2
    return energy

def get_voiced(frames, treshold):
    """
    Divide frames into two categories:
        -voiced_segment : contains all frames with an energy >= treshold
        -unvoiced_segment : contains all other frames
    """
    voiced_segments = []
    unvoiced_segments = []
    for frame in frames:
        energy = compute_energy(frame)
        if (energy >= treshold):
            voiced_segments.append(frame)
        else:
            unvoiced_segments.append(frame)
    return voiced_segments, unvoiced_segments


def autocorrelation_pitch_estim(path):
    """
    Compute an estimation of the pitch of a speaker using the autocorrelation method.
        @path of the directory where utterances (minimum 5) are stored
    """
    #1.
    #On se replace à la racine du projet
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("../../")
    #On se place dans le dossier où les échantillons de voix sont stockés.
    os.chdir(path)
    files = glob.glob("*.wav")
    choosed_files = random.sample(files, 5)
    f0_list = []
    for file in choosed_files:
        current_signal, sampling_rate = read_wavfile(file)
        #2.
        current_signal = normalize(current_signal)
        #3.
        frames = split(current_signal, sampling_rate, 50, 25)
        #4.
        #5.
        voiced_segments, unvoiced_segments = get_voiced(frames, 5) 
        #6.
        for segment in voiced_segments:
            lags, c = xcorr.xcorr(segment, segment, maxlags=200)
            #7.
            peaks, p = signal.find_peaks(c)
            if(len(peaks) > 1):
                peak1 = peaks[0]
                peak2 = peaks[1]
                for peak in peaks:
                    if c[peak] > c[peak1]:
                        peak1 = peak
                    if c[peak] < c[peak1] and c[peak] > c[peak2]:
                        peak2 = peak
                if (peak1 != peak2):
                    f0_list.append(sampling_rate/abs(peak1-peak2))
        f0_list.sort()
        while(f0_list[-1] > 550):
            f0_list.pop()
    f0 = np.mean(f0_list)
    return f0


def cepstrum_pitch_estim(path): 
    """
    Compute an estimation of the pitch of a speaker using the cepstrum method.
        @path of the directory where utterances (minimum 5) are stored
    """
    #On prend des samples randoms pour les deux personnes
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("../../")
    os.chdir(path)
    files = glob.glob("*.wav")
    choosed_files = random.sample(files, 5)

    f0_list = []
    #On normalise les signaux et on les affiche (point 2)
    for file in choosed_files:
        current_signal, sampling_rate = read_wavfile(file)
        current_signal = normalize(current_signal)

        #On split et on fait une liste des voiced segments (treshold à vérifier si correct) (point 3-5)
        frames = split(current_signal, sampling_rate, 50, 25)

        voiced_segment, unvoiced_segment = get_voiced(frames, 5)
        maximum_index = 0
        maximum_index_windowed = 0
        for segment in voiced_segment:
            #On obtient le ceptrum des signaux (point 6)
            w, h = signal.freqz(segment)
            logfreq = np.log10(h)

            cepstrum = np.fft.ifft(logfreq)

            window = signal.hamming(len(segment))
            windowed_segment = segment * window

            wh, hw = signal.freqz(windowed_segment)
            logfreq_windowed = np.log(hw)
            cepstrum_windowed = np.fft.ifft(logfreq_windowed)

            max_peak = 32
            max_windowed_peak = 32
            for i in range(32,267): #On recherche dans l'intervalle 60Hz - 500Hz
                if (cepstrum[i] > cepstrum[max_peak]):
                    max_peak = i
                if (cepstrum_windowed[i] > cepstrum_windowed[max_windowed_peak]):
                    max_windowed_peak = i
            
            if (cepstrum_windowed[max_windowed_peak] > cepstrum[max_peak]):
                max_peak = max_windowed_peak
            
            f0_temp = sampling_rate/max_peak
            f0_list.append(f0_temp)
    f0 = np.mean(f0_list)
    return f0
        

def compute_formants(audiofile):
	#1.
    current_signal, sampling_rate = read_wavfile(audiofile) 
    frames = split(normalize(current_signal), sampling_rate, 50, 25) 
    #2.
    A = [1]
    B = [1, -0.67]  
    new_frame = []
    lpc_order = int(2 + (sampling_rate/1000))
    formants = []
    for frame in frames:
        filtered_frame =  signal.lfilter(B, A, frame)
        window = signal.hamming(len(filtered_frame))
        windowed_frame = filtered_frame * window
        lpc = scilpc.lpc_ref(windowed_frame, 10)
        roots = np.roots(lpc)
        values = []
        for r in roots:
            if (np.imag(r) > 0):
                angle = np.arctan2(np.imag(r), np.real(r))
                values.append(angle * ((sampling_rate/10)/2*np.pi))
        values.sort()
        formants.append(values)
    return formants

def compute_mfcc(audiofile):
    current_signal, sampling_rate = read_wavfile(audiofile)
    A= [1]
    B= [1,-0.97]
    emphasized_signal = signal.lfilter(B,A,current_signal)
    frames= split(emphasized_signal,sampling_rate, 50, 25)
    windowed_frames = []
    for frame in frames : 
        window= signal.hamming(len(frame))
        windowed_frames.append(window*frame)
    ndft=512
    power_spectrum= pow(abs(fft.fft(windowed_frames)),2)/ndft
    filter_bank_values = filterbanks.filter_banks(power_spectrum, sampling_rate)
    dcted_filter_bank_values = fft.dct(filter_bank_values, norm=' ortho')
    dcted_filter_bank_values = dcted_filter_bank_values[0:13]
    plt.plot(dcted_filter_bank_values)
    plt.show()

if __name__ == "__main__":
    #pitch_1 = autocorrelation_pitch_estim("data/slt_a")
    #pitch_2 = cepstrum_pitch_estim("data/slt_a")
    #print(pitch_1)
    #print(pitch_2)
    compute_formants("data/bdl_a/arctic_a0001.wav")
