import os
import wave

START = 60
LENGTH = 5
END = 90
INPUT_PATH = "genre_db_wav"
OUTPUT_PATH = "5s_genre_db"

# 10 selected genres
# genres = ["Hip-Hop",
#           "Pop",
#           "Blues",
#           "Electronic",
#           "Classical",
#           "Rock",
#           "Folk",
#           "Soul-RnB",
#           "Country",
#           "Jazz"]


for folder in os.listdir(INPUT_PATH):
    # initialise song number in given folder to 1
    song_no: int = 1

    # # if folder not in list of genres, skip
    # if folder not in genres:
    #     continue

    # if folder for genre doesnt exist in dir, make one
    if not os.path.exists(OUTPUT_PATH + "/" + folder):
        os.mkdir(OUTPUT_PATH + "/" + folder)

    # initialize sample count for each genre
    sample_count = 0

    # for each song in folder take 1000 samples
    for file in os.listdir(INPUT_PATH + "/" + folder):

        # if we have 1000 samples, break and go to next folder
        if sample_count > 999:
            break
        try:
            # intialize sample number per song, start and end times
            sample_no = 1
            start = START  # seconds
            end = start + LENGTH  # seconds

            # while end doesnt exceed song length of 30s
            while end < END:
                # file to extract the snippet from
                with wave.open(INPUT_PATH + '/' + folder + '/' + file, "rb") as infile:

                    # get file data
                    nchannels = infile.getnchannels()
                    sampwidth = infile.getsampwidth()
                    framerate = infile.getframerate()
                    # set position in wave to start of segment
                    infile.setpos(int(start * framerate))
                    # extract data
                    data = infile.readframes(int((end - start) * framerate))

                # write the extracted data to a new file
                with wave.open(OUTPUT_PATH + "/" + folder + "/" + folder + "_" + str(song_no) + '_' + str(sample_no)
                               + '.wav', 'w') as outfile:
                    outfile.setnchannels(nchannels)
                    outfile.setsampwidth(sampwidth)
                    outfile.setframerate(framerate)
                    outfile.setnframes(int(len(data) / sampwidth))
                    outfile.writeframes(data)
                # update start and end times so there is a 1 sec overlap with every sample
                start += (LENGTH - 1)
                end = start + LENGTH
                # increment sample number and sample count
                sample_no += 1
                sample_count += 1
        except:
            print("issue with" + INPUT_PATH + "/" + folder + "/" + file)
        # update song number
        song_no += 1


for folder in os.listdir(OUTPUT_PATH):
    try:
        print(folder + ": " + str(len(os.listdir(OUTPUT_PATH + "/" + folder))))
    except NotADirectoryError:
        print("Not a directory:" + OUTPUT_PATH + "/" + folder)