import shutil

"""
This script is used to move the files in the validation files to the new "val" folder.
"""

# path = rf"C:\Users\Marc\Documents\GitHub\8 semester\Weakly-supervised-anomaly-detection\MGFNmain\UCF_list\\"
# path = rf"/home/cv05f23/git/Weakly-supervised-anomaly-detection/MGFNmain/UCF_list/"
path = rf"/MGFNmain/UCF_list/"

with open(path + rf"ucf-i3d-train.list", 'r') as f:  # use a context manager to safely opening and closing files
    train = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string

with open(path + rf"ucf-i3d-val.list", 'r') as f:  # use a context manager to safely opening and closing files
    val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string

old_path = "/home/marc/Documents/data/UCF"
new_path = "/home/marc/Documents/data/UCF/train"
for i in train:
    shutil.move(old_path + i[len(new_path):], i)

with open(path + rf"ucf-i3d-val.list", 'r') as f:  # use a context manager to safely opening and closing files
    val = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string

old_path = "/home/marc/Documents/data/UCF"
new_path = "/home/marc/Documents/data/UCF/val"
for i in val:
    shutil.move(old_path + i[len(new_path):], i)
