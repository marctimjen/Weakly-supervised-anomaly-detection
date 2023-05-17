import numpy as np


"""
This script is used to create masks for the different classes in the UCF-dataset. The masks are specifically for the gt 
file. This makes it possible to get an AUC score for the different classes.
"""


def create_masks(test_path: str, gt_file_path: str, start: str, save_path: str) -> None:
    """
    :param test_path: path to the ucf-i3d-test.list file. This file contain the paths to all test files.
    :param gt_file_path: path to the annotated UCF-file. The name of this file is
                            Temporal_Anomaly_Annotation_for_Testing_Videos.txt
    :param start: the start of the path in the ucf-i3d-test.list file. (The part of the path before the file name).
    :param save_path: where to save the final np-file.

    :return This function creates mask-files locally.
    """

    with open(test_path, 'r') as f:  # use a context manager to safely opening and closing files
        val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string
    val.sort()  # make sure to sort the values!

    with open(gt_file_path, 'r') as f:  # use a context manager to safely opening and closing files
        gt_values = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string

    info = dict()
    for i in gt_values:
        values = i.split()
        info[values[0]] = [i for i in map(int, values[2:])]


    end = "_i3d.npy"
    array_dict = dict()
    for i in val:

        string = i[len(start):-len(end)] + ".mp4"
        length = np.load(i).shape[0] * 16

        arr = np.zeros(length)

        ls = info[string]
        if ls[0] != -1:
            arr[ls[0]: ls[1] + 1] = 1
        if ls[2] != -1:
            arr[ls[2]: ls[3] + 1] = 1
        array_dict[string] = arr

    file_names = [i for i in array_dict.keys()]
    file_names.sort()

    end_mask = "xxx_x264.npy"

    classes = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Normal_Videos_",
                "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]

    masks = {i: np.array([]) for i in classes}
    for i in file_names:
        string = i[:-len(end_mask)]

        for j in masks.keys():
            if j == string:
                masks[j] = np.append(masks[j], np.ones(len(array_dict[i])))
            else:
                masks[j] = np.append(masks[j], np.zeros(len(array_dict[i])))

    for name in masks.keys():
        np.save(save_path + name + ".npy", masks[name])

if __name__ == "__main__":
    create_masks(
        test_path=rf"/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-test.list",
        gt_file_path =rf"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
        start="/home/marc/Documents/data/UCF/test/",
        save_path=rf"/home/marc/Documents/data/UCF/UCF_list/masks/"
    )