import os
import shutil
import random

DIR_FROM = "3s_genre_images"


def train_test_split(dataset_path, split):
    for folder in os.listdir(dataset_path):
        train_path = ""
        test_path = ""

        if folder in [".DS_Store"]:
            continue
        # if folder for genre test/train doesnt exist in dir, make one
        if not os.path.exists(dataset_path + "/" + folder + "/" + "test"):
            os.mkdir(dataset_path + "/" + folder + "/" + "test")
        test_path = DIR_FROM + "/" + folder + "/" + "test"

        if not os.path.exists(dataset_path + "/" + folder + "/" + "train"):
            os.mkdir(dataset_path + "/" + folder + "/" + "train")
        train_path = dataset_path + "/" + folder + "/" + "train"

        # print(len(os.listdir(train_path)))

        training_size = split * len(os.listdir(dataset_path + "/" + folder))
        # for each image, in an arbitrary order, move to train folder, once required size reached move rest to test
        for file in os.listdir(dataset_path + "/" + folder):
            # if trainng folder is the required size, break out of current genre folder
            if file in ["train", "test"]:
                continue
            if len(os.listdir(train_path)) < training_size:
                shutil.move(dataset_path + "/" + folder + "/" + file, train_path + "/" + file)
            else:
                shutil.move(dataset_path + "/" + folder + "/" + file, test_path + "/" + file)
            # try:
            #
            # except:
            #     print("issue with " + DIR_FROM + "/" + folder + "/" + file)
    return


train_test_split(DIR_FROM, 0.7)
