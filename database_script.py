import music_tag
import os
import shutil
import pandas

dir_count = {'Rock': 0,
             'Jazz': 0,
             'Electronic': 0,
             'Hip-Hop': 0,
             'Pop': 0,
             'Classical': 0,
             'Country': 0,
             'Blues': 0,
             'Soul-RnB': 0,
             'Metal': 0}
# dict of genre parent id(key) and names(value)
genre_parent_id = {}
# dict of genre parents(key) and children(value)
genre_hierarchy = {}

# load genre info CSV file
genre_data = pandas.read_csv("genres.csv")

# fill parent id dict
for row in genre_data.values:
    if row[2] == 0:
        genre_parent_id.update({row[0]: row[3]})


# fill children genres (keys) with parents (values)
for row in genre_data.values:
    genre_hierarchy.update({row[3]: genre_parent_id[row[4]]})

del genre_hierarchy['Old-Time / Historic']
del genre_parent_id[8]


# print(genre_data)
print(*genre_hierarchy.items(), sep='\n')


print(genre_parent_id.values())
# make folders for each genre
for directory in genre_parent_id.values():
    # if folder for genre doesnt exist then make folder for given genre
    if not os.path.exists("genre_db/" + directory):
        os.mkdir("genre_db/" + directory)

# print(dir_count.keys())

# for each folder and each song in fma_medium
for folder in os.listdir("fma_medium"):
    try:
        for song in os.listdir("fma_medium/" + folder):
            # get the genre
            genre = music_tag.load_file("fma_medium/" + folder + "/" + song)['genre'].first
            if genre in genre_hierarchy.keys():
                # copy song from fma_medium to genre folder in new database
                shutil.copy("fma_medium/" + folder + "/" + song, "genre_db/" + genre_hierarchy[genre] + "/")

    except NotADirectoryError:
        print("Not a directory:fma_medium/" + folder)

for folder in os.listdir("genre_db"):
    try:
        print(folder + ": " + str(len(os.listdir("genre_db/" + folder))))
    except NotADirectoryError:
        print("Not a directory:fma_medium/" + folder)
