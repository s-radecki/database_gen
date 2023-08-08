import gc
import os
import time
import wave
import librosa.display
import librosa.feature
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config


SAVE_TO_DIR = "TEST_5s_genre_images/"
DIR_FROM = "5s_genre_db"


# for folder in os.listdir(DIR_FROM):
#
#     if folder in [".DS_Store"]:
#         continue
#     # if folder for genre doesnt exist in dir, make one
#     if not os.path.exists(SAVE_TO_DIR + folder):
#         os.mkdir(SAVE_TO_DIR + folder)
#
#     # for each song in folder take 1000 samples
#     for file in os.listdir(DIR_FROM + "/" + folder):
#
#         try:
#             AUDIO_FILE = DIR_FROM + "/" + folder + "/" + file
#             samples, sample_rate = librosa.load(AUDIO_FILE, sr=None)
#             sgram = librosa.stft(samples)
#
#             # use the mel-scale instead of raw frequency
#             sgram_mag, _ = librosa.magphase(sgram)
#             mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
#             plt.Figure()
#             # plt.axis('off')
#             mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.max)
#             librosa.display.specshow(mel_sgram, sr=sample_rate)
#             # plt.colorbar(format='%+2.0f dB')
#             plt.savefig(SAVE_TO_DIR + folder + "/" + file[:-3] + "png", bbox_inches='tight',transparent=True, pad_inches=0)
#
#             # plt.savefig(SAVE_TO_DIR + folder + "/" + file[:-3] + "png")
#
#         except:
#             print("issue with" + DIR_FROM + "/" + folder + "/" + file)

def generate_melspectrogram(main_dir, folder, file_name):
    try:
        AUDIO_FILE = main_dir + "/" + folder + "/" + file_name
        samples, sample_rate = librosa.load(AUDIO_FILE, sr=None)
        sgram = librosa.stft(samples)

        # use the mel-scale instead of raw frequency
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
        plt.Figure()
        # plt.axis('off')
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.max)
        librosa.display.specshow(mel_sgram, sr=sample_rate)
        # plt.colorbar(format='%+2.0f dB')
        plt.savefig(SAVE_TO_DIR + folder + "/" + file_name[:-3] + "png", bbox_inches='tight',transparent=True, pad_inches=0)
        # plt.savefig(SAVE_TO_DIR + folder + "/" + file[:-3] + "png")
        del AUDIO_FILE
    except:
        print("issue with " + AUDIO_FILE)




# if output folder doesnt exist, create
if not os.path.exists(SAVE_TO_DIR):
    os.mkdir(SAVE_TO_DIR)

# folder = "Soul-RnB"
start = time.time()
for folder in os.listdir(DIR_FROM):
    if folder in [".DS_Store"]:
                  # , "Blues", "Hip-Hop", "Pop", "Electronic", "Classical", "Rock", "Folk", "Jazz", "Country"]:
        continue

    # if folder for genre doesnt exist in dir, make one
    if not os.path.exists(SAVE_TO_DIR + folder):
        os.mkdir(SAVE_TO_DIR + folder)

    Parallel(n_jobs=7, backend='loky')(delayed(generate_melspectrogram)(DIR_FROM, folder, file) for file in os.listdir(DIR_FROM + "/" + folder))
    # gc.collect()

print("Time taken: ", time.time() - start)



# make list of all i/o file names to run parallels once
# if output folder doesnt exist, create
# if not os.path.exists(SAVE_TO_DIR):
#     os.mkdir(SAVE_TO_DIR)




for folder in os.listdir(SAVE_TO_DIR):
    try:
        print(folder + ": " + str(len(os.listdir(SAVE_TO_DIR + folder))))
    except NotADirectoryError:
        print("Not a directory:" + SAVE_TO_DIR + folder)
