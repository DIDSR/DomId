import os
import sys

import numpy as np
import pandas as pd

try:
    path = sys.argv[1]
except IndexError:
    path = "../../HER2/combined_train"
print(f"HER2 data directory: {path}")


def get_jpg_folders(path):
    """
    only keep folders of .jpg images,
    which folder names by convention end in jpg
    """
    folders = os.listdir(path)
    jpg_folders = list(
        filter(
            lambda f: f.endswith("jpg") and os.path.isdir(os.path.join(path, f)),
            folders,
        )
    )
    return jpg_folders


def total_count_images(path):
    folders = get_jpg_folders(path)
    counter = 0
    for folder in folders:
        folder_path = os.path.join(path, folder)
        images = os.listdir(folder_path)
        counter += len(images)
        print(folder, len(images), counter)
    return counter


def parse_machine_labels(image_names):
    machine_labels = []
    machine_dict = {"FD": 0, "H1": 1, "H2": 1, "ND": 2}
    for image in image_names:
        machine = image[-6:-4]
        machine = machine_dict[machine]
        machine_labels.append(machine)
    return machine_labels

def mean_scores_per_experiment(scores, img_locs):
    """
    Parser to get mean scores per image from the cvs file.
    The name of the images in the folders are slightly different from the names in the csv file.

    """

    M = []

    for image_loc in img_locs:
        try:
            image_loc = str(
                image_loc.split("/")[-1]
            )  # depending if the path is full or not, take the img name only
        except:
            "not full path"

        N = len(image_loc) - 6 #removes the _machine.jpg part from the name of the image
        mean_score = scores.loc[
            scores["file name"].str.contains(image_loc[:N])
        ].mean(axis=1)
        mean_score = float(mean_score)
        # print(mean_score)
        M.append(mean_score)
    return M



if __name__ == "__main__":

    folders = get_jpg_folders(path)
    N = total_count_images(path)
    number_labels = 3
    data = np.zeros((N, number_labels + 1)).astype("str")
    start = 0

    for folder in folders:
        print(folder)
        print("start", start)
        label_of_the_folder_int = int(folder[-4])
        folder_path = os.path.join(path, folder)
        images = os.listdir(folder_path)
        labels = [label_of_the_folder_int] * len(images)
        machine_labels = parse_machine_labels(images)
        base_path_scores = os.path.join(*path.split('/')[:-1])
        # base_path = "/your/data/location"

        scores = pd.read_csv(
            os.path.join(base_path_scores, "truthfile_002.csv"),
            names=["num", "file name", "s1", "s2", "s3", "s4", "s5", "s6", "s_7"],
        )
        individual_scores = mean_scores_per_experiment(scores, images)
        data[start : start + len(images), :] = np.stack(
            (images, labels, machine_labels, individual_scores), 0
        ).T
        start += len(images)
        print(len(images))

    dataframe = pd.DataFrame(data)
    csv_path = os.path.join(path, "dataframe.csv")
    print(csv_path)
    print(os.listdir(path))
    dataframe.to_csv(
        csv_path,
        header=["img_id", "class", "machine", "score"],
        index=False,
    )
