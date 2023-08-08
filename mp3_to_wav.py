import os

from pydub import AudioSegment
INPUT_PATH = "genre_db"
OUTPUT_PATH = "genre_db_wav"

for folder in os.listdir(INPUT_PATH):
    try:
        for song in os.listdir(INPUT_PATH + "/" + folder):
            src = INPUT_PATH + "/" + folder + "/" + song

            if not os.path.exists(OUTPUT_PATH + "/" + folder):
                os.mkdir(OUTPUT_PATH + "/" + folder)

            dst = OUTPUT_PATH + "/" + folder + "/" + song[:-4] + ".wav"

            sound = AudioSegment.from_mp3(src)
            sound.export(dst, format="wav")

    except:
        print("issue with" + INPUT_PATH + "/" + folder + "/" + song)



