import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


# with open("/home/marc/Documents/data/s3r/ucf-crime_pretrain.pickle", "rb") as f:
#     s3r_path_ucf_pre = pkl.load(f)

with open("/home/marc/Documents/data/s3r/ucf-crime_validation.pickle", "rb") as f:
    s3r_path_ucf_val = pkl.load(f)

path = rf"/home/marc/Documents/data/UCF/UCF_list/"
with open(path + rf"ucf-i3d-test.list", 'r') as f:  # use a context manager to safely opening and closing files
    val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string
val.sort()  # make sure to sort the values!

start = "/home/marc/Documents/data/UCF/test/"
end = "_i3d.npy"

res = np.array([])

for i in val:
    pred = np.repeat(s3r_path_ucf_val[i[len(start):-len(end)]].squeeze(), 16)
    res =  np.concatenate((res, pred))


path = "/home/marc/Documents/data/s3r/ucf-crime_validation.npy"
np.save(path, res)  # save the prediction file

print("2nop")