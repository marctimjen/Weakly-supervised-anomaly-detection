import numpy as np

"""
This script is used to create a random prediction file. Where the anomaly score predicted is just uniformly choosen.
"""


gt_file = np.load("/home/marc/Documents/data/UCF/UCF_list/gt-ucf_our.npy")

out = np.random.uniform(low=0, high=1, size=len(gt_file))

np.save("/home/marc/Documents/GitHub/8semester/Weakly-supervised-anomaly-detection/result_uploader/rando.npy", out)  # save the prediction file

