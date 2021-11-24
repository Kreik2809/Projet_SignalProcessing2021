import main.utils


signal, sampling_rate = utils.read_wavfile("../data/arctic_a0001.wav")

windows = utils.split(signal, sampling_rate, 1000, 500)
 
w1 = signal[:16000]
print((w1 == windows[0]).all())
 
w2 = signal[24000:40000]
print((w2 == windows[1]).all()) 



