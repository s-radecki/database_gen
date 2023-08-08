import os
import matplotlib.pyplot as plt

#for loading and visualizing audio files
import librosa
import librosa.display
import librosa.feature
import numpy as np
from os import path
from pydub import AudioSegment

from scipy.io import wavfile

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Load the audio file
# AUDIO_FILE = "3s_genre_db/Hip-Hop/Hip-Hop_1_10.wav"
AUDIO_FILE = "3s_genre_db/Folk/Folk_4_3.wav"
samples, sample_rate = librosa.load(AUDIO_FILE, sr=None)

# x-axis has been converted to time using our sample rate.
# matplotlib plt.plot(y), would output the same figure, but with sample
# number on the x-axis instead of seconds
plt.figure() #figsize=(14, 5))
librosa.display.waveshow(samples, sr=sample_rate)

sgram = librosa.stft(samples)
# librosa.display.specshow(sgram)

# use the mel-scale instead of raw frequency
sgram_mag, _ = librosa.magphase(sgram)
mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
# librosa.display.specshow(mel_scale_sgram)

plt.Figure()
plt.axis('off')
# canvas = FigureCanvas(fig)
# use the decibel scale to get the final Mel Spectrogram
mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.max)
librosa.display.specshow(mel_sgram, sr=sample_rate)
# plt.colorbar(format='%+2.0f dB')
plt.savefig("test/test_spec_2.png", bbox_inches='tight',transparent=True, pad_inches=0)
print("done")
