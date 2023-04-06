import numpy as np
path = rf"/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFN-main/UCF_list/"

with open(path + rf"ucf-i3d-val.list", 'r') as f:  # use a context manager to safely opening and closing files
    val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string

j = 0
for i in val:
    print(i)
    print(np.load(i).shape)
    j += 1

print(j)