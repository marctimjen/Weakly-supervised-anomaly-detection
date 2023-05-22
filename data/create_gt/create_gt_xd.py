import os
import numpy as np

"""
This file should create the gt file for the xd violence data set.
"""

def create_gt_xd(test_path: str, gt_file_path: str, save_path: str) -> None:
    """
    :param test_path: path to the test-files (the .npy files)
    :param gt_file_path: path to the annotated UCF-file. The name of this file is
                            Temporal_Anomaly_Annotation_for_Testing_Videos.txt
    :param save_path: where to save the final np-file.
    """

    def ano_frames(ls):
        it = int(len(ls) / 2)
        if not (int(len(ls)) % 2 == 0):
            raise ValueError(f"ls must have even length, but ls has length: {len(ls)}")

        for i in range(it):
            yield ls[::2][i], ls[1::2][i]  # yeild evey 2 items from the list

    test_files = os.listdir(test_path)
    test_files = sorted(test_files)  # Sort the test-files so that the order is correct

    with open(gt_file_path, 'r') as f:  # use a context manager to safely opening and closing files
        gt_values = [line.strip() for line in f.readlines()]

    info = dict()
    for i in gt_values:
        values = i.split()
        info[values[0]] = [i for i in map(int, values[1:])]

    array_dict = dict()

    for i in test_files:
        load_path = test_path + "/" + i
        length = np.load(load_path).shape[0] * 16
        arr = np.zeros(length)

        string = i[:-4]
        contain_anomaly = info.get(string, False)

        if "label_A" not in i:
            it = iter(ano_frames(contain_anomaly))
            for start, end in it:
                arr[start: end + 1] = 1

        array_dict[string] = arr

    gt = np.array([])
    file_names = [i for i in array_dict.keys()]
    file_names.sort()

    for i in file_names:
        gt = np.append(gt, array_dict[i])

    np.save(save_path, gt)


if __name__ == "__main__":
    create_gt_xd(
        test_path=rf"/home/cv05f23/data/XD/RGBTest",
        gt_file_path =rf"/home/cv05f23/git/Weakly-supervised-anomaly-detection/download_features/xd.list",
        save_path=rf"/home/cv05f23/data/XD/test_gt/gt-xd_our.npy"
    )

    # create_gt_xd(
    #     test_path=rf"/home/marc/Documents/data/xd/RGBTest",
    #     gt_file_path=rf"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/download_features/xd.list",
    #     save_path=rf"/home/marc/Documents/data/xd/test_gt/gt-xd_our.npy"
    # )


