import numpy as np
import os
import torch


"""
The files from xd violence comes in 5 crops and thus we need to turn these five crops into one file.
"""

crop_path = "/home/cv05f23/data/XD/i3d-features/RGBTest"

save_file_path = "/home/cv05f23/data/XD/RGBTest"

crop_files = os.listdir(crop_path)  # Find all the crops in the dir
crop_set = set()

for i in crop_files:
    crop_set.update([i[:-5]])

crop_list = list(crop_set)

for i in crop_list:
    file1 = np.load(crop_path + "/" + i + "0.npy")
    file2 = np.load(crop_path + "/" + i + "1.npy")
    file3 = np.load(crop_path + "/" + i + "2.npy")
    file4 = np.load(crop_path + "/" + i + "3.npy")
    file5 = np.load(crop_path + "/" + i + "4.npy")


    features = np.concatenate((file1[None, ], file2[None, ], file3[None, ], file4[None, ], file5[None, ]), axis=0)
    features = torch.Tensor(features).permute(1, 0, 2)
    features = np.array(features, dtype=np.float32)
    np.save(save_file_path + "/" + i[:-2], features)  # save the new file!





