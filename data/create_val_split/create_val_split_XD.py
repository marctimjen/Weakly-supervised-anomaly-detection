import os
import numpy as np

"""
This file is used to do a split of the train dataset for XD. This creates a validation dataset from the train data.
"""

np.random.seed(42)  # set seed for random permutation

# path = "/home/marc/Documents/data/xd/RGB"
path = "/home/cv05f23/data/XD/RGB"
files = os.listdir(path)

normal_files = set(f for f in files if "label_A" in f)
anomal_files = set(f for f in files if not("label_A" in f))

if normal_files.intersection(anomal_files) or anomal_files.intersection(normal_files):
    raise ValueError("One or more files cannot exist in normal_files and anomal_files")

normal = np.random.permutation(np.array([*normal_files]))  # permute the data to make it random!
anomal = np.random.permutation(np.array([*anomal_files]))  # permute the data to make it random!

print("Amount of normal files:", len(normal))
print("Amount of anomaly files:", len(anomal))

train_normal = normal[int(len(normal)*0.2):]
val_normal = normal[:int(len(normal)*0.2)]
print("Amount of normal files in train-set:", len(train_normal))
print("Amount of normal files in val-set:", len(val_normal))

train_anormal = anomal[int(len(anomal)*0.2):]
val_anormal = anomal[:int(len(anomal)*0.2)]
print("Amount of anomaly files in train-set:", len(train_anormal))
print("Amount of anomaly files in val-set:", len(val_anormal))

train = set(train_normal) | set(train_anormal)
val = set(val_normal) | set(val_anormal)

if train.intersection(val) or val.intersection(train):
    raise ValueError("One or more files cannot exist in train and val")


train = sorted([*train])
val = sorted([*val])

print("Amount of train files", len(train))
print("Amount of validation files", len(val))

name = "/home/cv05f23/data/XD/lists/rgb_train.list"
# name = "/home/marc/Documents/data/xd/lists/rgb_train.list"
with open(name, 'w+') as f:  ## the name of feature list
    for file in train:
        newline = path+'/'+file+'\n'
        f.write(newline)

name = "/home/cv05f23/data/XD/lists/rgb_val.list"
# name = "/home/marc/Documents/data/xd/lists/rgb_val.list"
with open(name, 'w+') as f:  ## the name of feature list
    for file in val:
        newline = path+'/'+file+'\n'
        f.write(newline)



