import numpy as np

# res = np.load("/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/results/ucf_gt/gt-ucf.npy")

res = np.load("/home/marc/Documents/data/UCF/UCF_list/gt-ucf_our.npy")

path = rf"/home/marc/Documents/data/UCF/UCF_list/"
with open(path + rf"ucf-i3d-test.list", 'r') as f:  # use a context manager to safely opening and closing files
    val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string
val.sort()  # make sure to sort the values!

gt_file_path = rf"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"

with open(gt_file_path, 'r') as f:  # use a context manager to safely opening and closing files
    gt_values = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string

info = dict()
for i in gt_values:
    values = i.split()
    info[values[0]] = [i for i in map(int, values[2:])]

start = "/home/marc/Documents/data/UCF/test/"
end = "_i3d.npy"

total_anno = 0
start_of_arr = 0
end_of_arr = 0

array_dict = dict()
for i in val:
    string = i[len(start):-len(end)] + ".mp4"
    length = np.load(i).shape[0]*16
    end_of_arr += length
    arr = np.zeros(length)
    ls = info[string]
    # print(string, end=" ")
    if ls[0] != -1:
        arr[ls[0]: ls[1] + 1] = 1
        # print(ls[0], ls[1] + 1, end=" ")
    if ls[2] != -1:
        arr[ls[2]: ls[3] + 1] = 1
        # print(ls[2], ls[3] + 1, end=" ")

    if any(arr != res[start_of_arr: end_of_arr]):
        print(string)

    start_of_arr = end_of_arr

    # print("")
    # if ls[0] == -1 and ls[2] == -1:
    #     break

    total_anno += np.sum(arr)


print("Amount of annotations in the gt-file:", np.sum(res))
print("Total amount of annotations in the test-list", total_anno)