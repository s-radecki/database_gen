import os
import wave

START = 0
END = 5
SAVE_TO_DIR = "5s_genre_db/"
DIR_FROM = "genre_db_wav"

# 10 selected genres
genres = ["Hip-Hop",
          "Pop",
          "Blues",
          "Electronic",
          "Classical",
          "Rock",
          "Folk",
          "Soul-RnB",
          "Country",
          "Jazz"]


for folder in os.listdir(DIR_FROM):
    # initialise song number in given folder to 1
    song_no: int = 1

    # if folder not in list of genres, skip
    if folder not in genres:
        continue

    # if folder for genre doesnt exist in dir, make one
    if not os.path.exists(SAVE_TO_DIR + folder):
        os.mkdir(SAVE_TO_DIR + folder)

    # initialize sample count for each genre
    sample_count = 0

    # for each song in folder take 1000 samples
    for file in os.listdir(DIR_FROM + "/" + folder):

        # if we have 1000 samples, break and go to next folder
        if sample_count > 999:
            break
        try:
            # intialize sample number per song, start and end times
            sample_no = 1
            start = START  # seconds
            end = END  # seconds

            # while end doesnt exceed song length of 30s
            while end < 30:
                # file to extract the snippet from
                with wave.open(DIR_FROM + '/' + folder + '/' + file, "rb") as infile:

                    # get file data
                    nchannels = infile.getnchannels()
                    sampwidth = infile.getsampwidth()
                    framerate = infile.getframerate()
                    # set position in wave to start of segment
                    infile.setpos(int(start * framerate))
                    # extract data
                    data = infile.readframes(int((end - start) * framerate))

                # write the extracted data to a new file
                with wave.open(SAVE_TO_DIR + folder + "/" + folder + "_" + str(song_no) + '_' + str(sample_no) + '.wav',
                               'w') as outfile:
                    outfile.setnchannels(nchannels)
                    outfile.setsampwidth(sampwidth)
                    outfile.setframerate(framerate)
                    outfile.setnframes(int(len(data) / sampwidth))
                    outfile.writeframes(data)
                # update start and end times so there is a 1 sec overlap with every sample
                start += (END - 1)
                end += (END - 1)
                # increment sample number and sample count
                sample_no += 1
                sample_count += 1
        except:
            print("issue with" + DIR_FROM +  "/" + folder + "/" + file)
        # update song number
        song_no += 1


for folder in os.listdir(SAVE_TO_DIR):
    try:
        print(folder + ": " + str(len(os.listdir(SAVE_TO_DIR + folder))))
    except NotADirectoryError:
        print("Not a directory:" + SAVE_TO_DIR + folder)