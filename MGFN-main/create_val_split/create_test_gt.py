import numpy as np


def create_gt(test_path: str, gt_file_path: str, start: str, save_path: str) -> None:
    """

    :param test_path: path to the ucf-i3d-test.list file. This file contain the paths to all test files.
    :param gt_file_path: path to the annotated UCF-file. The name of this file is
                            Temporal_Anomaly_Annotation_for_Testing_Videos.txt
    :param start: the start of the path in the ucf-i3d-test.list file. (The part of the path before the file name).
    :param save_path: where to save the final np-file.
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

    gt = np.array([])
    for i in file_names:
        gt = np.append(gt, array_dict[i])

    np.save(save_path, gt)

if __name__ == "__main__":
    create_gt(
        test_path=rf"/home/marc/Documents/data/UCF/UCF_list/ucf-i3d-test.list",
        gt_file_path = rf"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt",
        start="/home/marc/Documents/data/UCF/test/",
        save_path=rf"/home/marc/Documents/data/UCF/UCF_list/gt-ucf_our.npy"
    )


# # path = rf"/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/"
# # path = rf"C:\Users\Marc\Documents\GitHub\8 semester\Weakly-supervised-anomaly-detection\MGFN-main\UCF_list\\"
#
# path = rf"/home/marc/Documents/data/UCF/UCF_list/"
# with open(path + rf"ucf-i3d-test.list", 'r') as f:  # use a context manager to safely opening and closing files
#     val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string
# val.sort()  # make sure to sort the values!
#
# v1 = [i for i in val if not("Normal_Videos" in i)]
# v1.sort()
# v2 = [i for i in val if "Normal_Videos" in i]
# v2.sort()
#
# val = v1 + v2
#
# # gt_file_path = rf"C:\Users\Marc\Documents\GitHub\8 semester\Weakly-supervised-anomaly-detection\MGFN-main\UCF_list\Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
# # gt_file_path = rf"/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
# gt_file_path = rf"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
#
# with open(gt_file_path, 'r') as f:  # use a context manager to safely opening and closing files
#     gt_values = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string
#
# info = dict()
# for i in gt_values:
#     values = i.split()
#     info[values[0]] = [i for i in map(int, values[2:])]
#
# # print(info)
#
#
# # res = np.load("/home/cv05f23/data/UCF/test_gt/gt-ucf.npy")
# res = np.load("/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/results/ucf_gt/gt-ucf.npy")
# res = np.load("/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/RTFM-main/list/gt-ucf.npy")
# start_of_arr = 0
# end_of_arr = 0
#
# start = "/home/marc/Documents/data/UCF/test/"
# end = "_i3d.npy"
#
# array_dict = dict()
# for i in val:
#     string = i[len(start):-len(end)] + ".mp4"
#     length = np.load(i).shape[0]*16
#     end_of_arr += length
#     arr = np.zeros(length)
#     ls = info[string]
#     if ls[0] != -1:
#         arr[ls[0]: ls[1] + 1] = 1
#     if ls[2] != -1:
#         arr[ls[2]: ls[3] + 1] = 1
#     array_dict[string] = arr
#     array_dict[string + "_gt"] = res[start_of_arr: end_of_arr]
#     start_of_arr = end_of_arr
#
# file_names = [i for i in array_dict.keys()]
# file_names.sort()
#
# gt = np.array([])
# for i in file_names:
#     gt = np.append(gt, array_dict[i])
#
# # np.save("/home/cv05f23/data/UCF/test_gt/gt-ucf_our.npy", gt)
#
#
# # j = 0
# # for i in array_dict.items():
# #     print(i)
# #     j += 1
# #     if j == 10:
# #         break
#
# # j = 0  # This show that the last videos has not been annotated correctly
# # for i in array_dict:
# #     j += 1
# #     if j % 2 == 0:
# #         continue
# #     if not all(array_dict[i] == array_dict[i + "_gt"]):
# #         print(i, np.sum(array_dict[i + "_gt"]))
#
# # print(gt_values)
# # print()
#
# # j = 0
# # for i in val:
# #     print(i)
# #     print(np.load(i).shape)
# #     j += 1
# #
# # print(j)