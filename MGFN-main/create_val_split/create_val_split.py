import numpy as np
import re

path = rf"/home/cv05f23/data/UCF/lists/"

with open(path + rf"ucf-i3d.list", 'r') as f:  # use a context manager to safely opening and closing files
    ls = [line.strip() for line in f.readlines()]  # use strip to get rid of the \n at the end of string

print(ls)
vals = dict()

for i in ls:
    try:
        gr = re.match(r"/home/cv05f23/data/train/([a-zA-Z_]+)\d.+", i).group(1)
    except TypeError:
        print("")

    if (w := vals.get(gr, False)):
        ls = [j for j in w] + [i]
        vals[gr] = ls
    else:
        vals[gr] = [i]

anomaly_nr = dict()  # This dict is used to find how many of the different anomalies should be loaded.
for key in vals.keys():
    print(key, len(vals[key]))

    if key != "Normal_Videos":
        anomaly_nr[key] = round(80 * len(vals[key])/800)


anomaly_nr["Normal_Videos"] = 80

validation_set = []
train_set = []


for key in vals.keys():
    random_nr = np.random.choice(range(len(vals[key])), len(vals[key]))
    validation_set.extend([vals[key][i] for i in random_nr[:anomaly_nr[key]]])
    train_set.extend([vals[key][i] for i in random_nr[anomaly_nr[key]:]])

old_path = "/home/cv05f23/data/train"
new_path = "/home/cv05f23/data/UCF/train"
with open(path + rf"ucf-i3d-train.list", 'w+') as f:  # use a context manager to safely opening and closing files
    for j in train_set:
        final_path = new_path + j[len(old_path):]
        f.write(f"{final_path}\n")  # https://www.scaler.com/topics/python-write-list-to-file/

with open(path + rf"ucf-i3d-val.list", 'w+') as f:  # use a context manager to safely opening and closing files
    for j in validation_set:
        final_path = new_path + j[len(old_path):]
        f.write(f"{final_path}\n")  # https://www.scaler.com/topics/python-write-list-to-file/



