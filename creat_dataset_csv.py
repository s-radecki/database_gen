import csv
import os
from collections import deque

CSV_FILENAME = "3s_genre_images" + ".csv"

INPUT_PATH = "3s_genre_images"

# Create list of zeros with 1 at first index
zero_list = [1] + ([0] * 9)
# Make list a deque object to rotate list
zero_list = deque(zero_list)


with open(CSV_FILENAME, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # write headers
    writer.writerow(["fileName", "Blues", "Classical", "Country", "Electronic",
                     "Folk", "Hip-Hop", "Jazz", "Pop", "Rock", "Soul-RnB"])

    # for each file, write to csv with 1 for the given files label, 0 all other labels 
    folders = sorted(os.listdir(INPUT_PATH))
    if '.DS_Store' in folders:
        folders.remove('.DS_Store')
    for folder in folders:
        subfolders = sorted(os.listdir(INPUT_PATH + "/" + folder))
        if '.DS_Store' in subfolders:
            subfolders.remove('.DS_Store')
        for subfolder in subfolders:
            files = os.listdir(INPUT_PATH + "/" + folder + "/" + subfolder)
            if '.DS_Store' in files:
                files.remove('.DS_Store')
            for file in files:
                writer.writerow([file] + list(zero_list))

        zero_list.rotate()

    csv_file.close()
