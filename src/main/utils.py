import audiofile
import numpy as np
import numpy.lib.stride_tricks as npst
import matplotlib.pyplot as plt
from scipy import signal, fft
import glob, os, random

import xcorr
import scikit_talkbox_lpc as scilpc
import filterbanks

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


def autocorrelation_pitch_estim(files):
    """
    Compute an estimation of the pitch of a speaker using the autocorrelation method.
        @list of files where utterances (minimum 5) are stored
    Calculate pitch for each frames and then return mean of all pitches
    """
    #1.
    f0_list = []
    for file in files:
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


def cepstrum_pitch_estim(files): 
    """
    Compute an estimation of the pitch of a speaker using the cepstrum method.
        @list of files where utterances (minimum 5) are stored
    Calculate pitch for each frames and then return mean of all pitches
    """
    #On prend des samples randoms pour les deux personnes
    f0_list = []
    #On normalise les signaux et on les affiche (point 2)
    for file in files:
        current_signal, sampling_rate = read_wavfile(file)
        current_signal = normalize(current_signal)

        #On split et on fait une liste des voiced segments (treshold à vérifier si correct) (point 3-5)
        frames = split(current_signal, sampling_rate, 50, 25)

        voiced_segment, unvoiced_segment = get_voiced(frames, 5)
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
    """
    Compute all frame formants of an audiofiles and return it as a 2 dimensional array
    """
	#1.
    current_signal, sampling_rate = read_wavfile(audiofile) 
    frames = split(normalize(current_signal), sampling_rate, 25, 25) 
    #2.
    A = [1]
    B = [1, 0.67]  
    lpc_order = int(2 + (sampling_rate/1000))
    formants = []
    time = 0
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
        #values.insert(0, time)
        formants.append(values)
        #time += 0.025
    return formants
    
    

def compute_mfcc(audiofile):
    #1.
    current_signal, sampling_rate = read_wavfile(audiofile)
    current_signal = normalize(current_signal)
    A= [1., 0.]
    B= [1.,-0.97]
    emphasized_signal = signal.lfilter(B,A,current_signal)
    frames= split(emphasized_signal,sampling_rate, 50, 25)
    Ndft = 512
    mfccs = []
    for frame in frames : 
        window = signal.hamming(len(frame))
        windowed_frames = window*frame
        w, h = signal.freqz(windowed_frames, worN=257)
        power_spectrum= pow(abs(h),2)/Ndft
        filter_bank_values = filterbanks.filter_banks(power_spectrum, sampling_rate)
        dct = fft.dct(filter_bank_values, norm='ortho')
        mfccs.append(dct)
    return mfccs

def analyse(path):
    """
    This function is called in each rule-based system in order to compute easily all the features of signals.
    Because of the cepstrum and autocorrelation pitch estimation requirements, path must point to
    a directory where minimum 5 audiofiles of a speaker are stored.
    """
    os.chdir(path)
    files = random.sample(glob.glob("*.wav"), 5)
    print(files)
    autocorr_pitch = autocorrelation_pitch_estim(files)
    cepstrum_pitch = cepstrum_pitch_estim(files)
    formants_list = []
    for file in files:
        formants = compute_formants(file)
        for f in formants:
            formants_list.append(f)
    
    f1_list = []
    f2_list = []
    for i in range(len(formants_list)):
        if (formants_list[i][0] > 90 and formants_list[i][0] < 1000):
            f1_list.append(formants_list[i][0])
        if (formants_list[i][1] > 600 and formants_list[i][1] < 3200):
            f2_list.append(formants_list[i][1])
    os.chdir("../../")
    return autocorr_pitch, cepstrum_pitch, f1_list, f2_list

def system_01(path):
    """
    Simple rule-based system that implements observed rules with if-else statements.
    It uses autocorrelation pitch estimation, cepstrum pitch estimation and formant 1.
    ====Results====
    Accuracy global : 0.7
    Accuracy cms : 0.0
    Accuracy slt : 0.9
    Accuracy bdl : 0.9
    Accuracy rms : 1.0
    """
    autocorr_pitch, cepstrum_pitch, f1_list, f2_list = analyse(path)
    f1 = np.mean(f1_list)
    print("Estimation du pitch avec la méthode autocorr : " + str(autocorr_pitch))
    print("Estimation du pitch avec la méthode cepstrum : " + str(cepstrum_pitch))
    print("Estimation du formant 1 : " + str(f1))
    if (autocorr_pitch < 150):
        if (cepstrum_pitch < 170):
            if (f1 < 410):
                print("C'est un homme")
                return "man"
    
    if (autocorr_pitch > 170):
        if(cepstrum_pitch > 210):
            if(f1 > 370):
                print("C'est une femme")
                return "woman"

def system_02(path):
    """
    Rule-based system which aims to improve system_01 perf. Use weight to determine the output.
    It uses autocorrelation pich estimation, cepstrum pitch estimation and formant 1.
    The two pitch have each 0.4 weight in the process of decision where formant 1 has only 0.2
    If man probability or woman probability has more thant 0.5, then the system can determine an output.
    ====Results====
    Accuracy global : 1.0
    Accuracy cms : 1.0
    Accuracy slt : 1.0
    Accuracy bdl : 1.0
    Accuracy rms : 1.0
    """
    autocorr_pitch, cepstrum_pitch, f1_list, f2_list = analyse(path)
    f1 = np.mean(f1_list)
    print("Estimation du pitch avec la méthode autocorr : " + str(autocorr_pitch))
    print("Estimation du pitch avec la méthode cepstrum : " + str(cepstrum_pitch))
    print("Estimation du formant 1 : " + str(f1))
    autocorr_pitch_weight = 0.4
    cepstrum_pitch_weight = 0.4
    f1_weight = 0.2

    man_prob = 0
    woman_prob = 0

    if (autocorr_pitch < 150):
        man_prob += autocorr_pitch_weight
    if (cepstrum_pitch < 170):
        man_prob += cepstrum_pitch_weight
    if (f1 < 410):
        man_prob += f1_weight
    
    if (autocorr_pitch > 170):
        woman_prob += autocorr_pitch_weight
    if (cepstrum_pitch > 210):
        woman_prob += cepstrum_pitch_weight
    if (f1 > 370):
        woman_prob += f1_weight

    if(man_prob > 0.5 and woman_prob > 0.5):
        print("unknown")
    elif(man_prob > 0.5 and woman_prob < 0.5):
        print("C'est un homme")
        print(man_prob)
        return "man"

    elif(man_prob < 0.5 and woman_prob > 0.5):
        print("C'est une femme")
        print(woman_prob)
        return "woman"

def system_03(path):
    """
    Rule-based system which uses Formant 2 features in the process of decision.
    ====Résultat====
    Précision globale : 0.925
    Précision cms : 1.0
    Précision slt : 0.9
    Précision bdl : 0.8
    Précision rms : 1.0
    """
    autocorr_pitch, cepstrum_pitch, f1_list, f2_list = analyse(path)
    f1=np.mean(f1_list)
    f2=np.mean(f2_list)
    if(autocorr_pitch < 150):
        if(f1<410):
            if(f2<2000):
                print("C'est un homme")
                return "man"
                    
                
    if(autocorr_pitch > 170):
        if(f1>270):
            if(f2>1800):
                print("C'est une femme")
                return "woman"
    else:
        print("Else")

if __name__ == "__main__":
    n = 40
    global_good_classifications = 0
    cms_good_classification = 0
    slt_good_classification = 0
    bdl_good_classification = 0
    rms_good_classification = 0
    for i in range(10):
        if (system_03("data/cms_b") == "woman"):
            global_good_classifications += 1
            cms_good_classification += 1
    for i in range(10):
        if (system_03("data/slt_a") == "woman"):
            global_good_classifications += 1
            slt_good_classification += 1
    for i in range(10):
        if (system_03("data/bdl_a") == "man"):
            global_good_classifications += 1
            bdl_good_classification += 1
    for i in range(10): 
        if (system_03("data/rms_b") == "man"):
            global_good_classifications += 1
            rms_good_classification += 1
    print("====Résultat====")
    print("Précision globale : " + str(global_good_classifications/40))
    print("Précision cms : " + str(cms_good_classification/10))
    print("Précision slt : " + str(slt_good_classification/10))
    print("Précision bdl : " + str(bdl_good_classification/10))
    print("Précision rms : " + str(rms_good_classification/10))