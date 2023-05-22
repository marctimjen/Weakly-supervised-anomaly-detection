import numpy as np
import os
"""
This is a test-file that is used to validate our gt-file.
"""


res = np.load("/home/marc/Documents/data/xd/test_gt/gt-xd_our.npy")

gt_file_path = rf"/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/download_features/xd.list"

files_dir = os.listdir("/home/marc/Documents/data/xd/RGBTest")
files_dir = sorted(files_dir)

with open(gt_file_path, 'r') as f:  # use a context manager to safely opening and closing files
    gt_values = [line.strip() for line in f.readlines()]
def ano_frames(ls):
    it = int(len(ls) / 2)
    if not (int(len(ls)) % 2 == 0):
        raise ValueError(f"ls must have even length, but ls has length: {len(ls)}")

    for i in range(it):
        yield ls[::2][i], ls[1::2][i]  # yeild evey 2 items from the list

file_path = rf"/home/marc/Documents/data/xd/RGBTest"

gt_values = {i.split()[0]: i.split()[1:] for i in gt_values}

total_ano = 0
total_len = 0
for i in files_dir:
    length = np.load(file_path + "/" + i).shape[0] * 16

    if "label_A" not in i:
        ls = [j for j in gt_values.get(i[:-4])]
        it = iter(ano_frames([j for j in map(int, ls)]))
        for start, end in it:
            end = min(end, length - 1)
            total_ano += end - start + 1
            if not(all(res[total_len + start: total_len + end + 1] == 1)):
                print(i)

    total_len += length

print(total_len)

print("Total amount of annotations in the test-list", total_ano)
print("Amount of annotations in the gt-file:", np.sum(res))
