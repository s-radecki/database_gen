import os
import shutil
import random

DIR_FROM = "3s_genre_images"
DIR_TO = "3s_genre_dataset_3"


def train_test_split(input_path, output_path, train_split, val_split, test_split):
    # check split percentages to 2 decimal places
    if round(train_split + val_split + test_split, 2) != 1.0:
        print("FILES HAVE NOT BEEN MOVED")
        print("Error: split values must add to 1")
        return

    # if folder for genre output test/train/val doesnt exist in dir, make one
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(output_path + "/" + "test"):
        os.mkdir(output_path + "/" + "test")
    test_path = output_path + "/" + "test"

    if not os.path.exists(output_path + "/" + "train"):
        os.mkdir(output_path + "/" + "train")
    train_path = output_path + "/" + "train"

    if not os.path.exists(output_path + "/" + "val"):
        os.mkdir(output_path + "/" + "val")
    val_path = output_path + "/" + "val"

    for folder in os.listdir(input_path):
        train_counter = 0
        val_counter = 0

        if folder in [".DS_Store"]:
            continue
        print(len(os.listdir(input_path + "/" + folder)))

        training_size = train_split * len(os.listdir(input_path + "/" + folder))
        val_size = val_split * len(os.listdir(input_path + "/" + folder))
        # for each image, in an arbitrary order, move to train folder, once required size reached, so same for val,
        # then put rest in test
        folder_list = os.listdir(input_path + "/" + folder)
        random.shuffle(folder_list)
        for file in folder_list:
            if train_counter < training_size:
                shutil.copy(input_path + "/" + folder + "/" + file, train_path + "/" + file)
                train_counter += 1
            elif val_counter < val_size:
                shutil.copy(input_path + "/" + folder + "/" + file, val_path + "/" + file)
                val_counter += 1
            else:
                shutil.copy(input_path + "/" + folder + "/" + file, test_path + "/" + file)
            # try:
            #
            # except:
            #     print("issue with " + DIR_FROM + "/" + folder + "/" + file)
    return


train_test_split(DIR_FROM, DIR_TO, 0.72, 0.18, 0.1)
