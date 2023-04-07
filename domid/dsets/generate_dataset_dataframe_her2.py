import os
import sys

import numpy as np
import pandas as pd

try:
    path = sys.argv[1]
except IndexError:
    path = "../../DomId/HER2/combined_train"
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
    machine_dict = {"FD": 0, "H1": 1, "H2": 2, "ND": 3}
    for image in image_names:
        machine = image[-6:-4]
        machine = machine_dict[machine]
        machine_labels.append(machine)
    return machine_labels


if __name__ == "__main__":

    folders = get_jpg_folders(path)
    N = total_count_images(path)
    number_labels = 2
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

        data[start : start + len(images), :] = np.stack(
            (images, labels, machine_labels), 0
        ).T
        start += len(images)
        print(len(images))

    dataframe = pd.DataFrame(data)
    csv_path = os.path.join(path, "dataframe.csv")
    dataframe.to_csv(
        csv_path,
        header=["img_id", "class", "machine"],
        index=False,
    )
