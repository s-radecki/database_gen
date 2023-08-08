import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

AUDIO_PATH = '3s_genre_db/Hip-Hop/Hip-Hop_57_1.wav'

plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (9, 7)

sampFreq, sound = wavfile.read(AUDIO_PATH)



length_in_s = sound.shape[0] / sampFreq

time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s

plt.subplot(2,1,1)
plt.plot(time, sound[:,0], 'r')
plt.xlabel("Time, s [left channel]")
plt.ylabel("Amplitude")
plt.subplot(2,1,2)
plt.plot(time, sound[:,1], 'b')
plt.xlabel("Time, s [right channel]")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

