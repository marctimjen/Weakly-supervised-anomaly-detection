import torch.utils.data as data
import numpy as np
import sys
sys.path.append("../..")  # adds higher directory to python modules path
sys.path.append("..")  # adds higher directory to python modules path
from RTFMmain.utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.FloatTensor')


class Dataset(data.Dataset):
    def __init__(self, rgb_list, datasetname="UCF", is_normal=True, transform=None, mode="train", seg_length=32,
                    shuffle=True):

        assert mode in ["train", "test", "val"], 'mode needs to be "train", "test" or "val"'
        self.is_normal = is_normal
        self.rgb_list_file = rgb_list  # path to path-file
        self.tranform = transform
        self.mode = mode
        self.num_frame = 0
        self.labels = None
        self.seg_length = seg_length
        self.datasetname = datasetname
        self.shuffle = shuffle
        self._parse_list()

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.mode != "test":
            if self.datasetname.lower() == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
                    print(self.list)

            elif self.datasetname.lower() == 'ucf':
                if self.is_normal:
                    self.list = [i for i in self.list if "Normal_Videos" in i]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = [i for i in self.list if not("Normal_Videos" in i)]
                    print('abnormal list for ucf')
                    print(self.list)
                    
            elif self.datasetname.lower() == 'xd':
                if self.is_normal:
                    self.list = [i for i in self.list if "_label_A" in i]
                    # print('normal list')
                    #print(self.list)
                else:
                    self.list = [i for i in self.list if not("_label_A" in i)]
                    # print('abnormal list')
                    # print(self.list)

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.mode == "test":
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, self.seg_length)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
