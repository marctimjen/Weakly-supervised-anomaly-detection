import torch.utils.data as data
import numpy as np
import sys
sys.path.append("../..")  # adds higher directory to python modules path
sys.path.append("..")  # adds higher directory to python modules path
from MGFNmain.utils.utils import process_feat
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# import option
# args = option.parse_args()

class Dataset(data.Dataset):
    def __init__(self, rgb_list, datasetname="UCF", modality="RGB", seg_length=32, add_mag_info=False, is_normal=True,
                    transform=None, mode="train", shuffle=True):
        """
        :param rgb_list:
        :param datasetname:
        :param modality:
        :param seg_length:
        :param add_mag_info:
        :param is_normal:
        :param transform:
        :param mode: is either "train", "val" or "test
        """

        assert mode in ["train", "test", "val"], 'mode needs to be "train", "test" or "val"'

        self.modality = modality
        self.is_normal = is_normal
        self.datasetname = datasetname
        self.seg_length = seg_length
        self.add_mag_info = add_mag_info
        self.rgb_list_file = rgb_list

        self.tranform = transform
        self.mode = mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None

        self.i = 0  # iterater when to stop the data-loading
        self.shuffle = shuffle
        self.idx_list = self.randomizer(shuffle=self.shuffle, lenght=len(self.list))

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.mode != "test":
            if self.datasetname == 'UCF':
                if self.is_normal:
                    self.list = [i for i in self.list if "Normal_Videos" in i]
                    # print('normal list')
                    # print(self.list)
                else:
                    self.list = [i for i in self.list if not("Normal_Videos" in i)]
                    # print('abnormal list')
                    # print(self.list)
            elif self.datasetname == 'XD':
                if self.is_normal:
                    self.list = self.list[9525:]
                    print('normal list')
                    print(self.list)
                else:
                    self.list = self.list[:9525]
                    print('abnormal list')
                    print(self.list)

    def __getitem__(self, idx):

        self.i += 1
        index = self.idx_list[idx]
        if self.i % self.__len__() == 0 and self.shuffle:
            self.idx_list = self.randomizer(shuffle=self.shuffle, lenght=len(self.list))

        if self.is_normal:  # get video level label 0/1
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        if self.datasetname == 'UCF':
            features = np.load(self.list[index].strip('\n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('\n')[:-4]
        elif self.datasetname == 'XD':
            features = np.load(self.list[index].strip('\n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            name = self.list[index].split('/')[-1].strip('\n')[:-4]
        if self.tranform is not None:
            features = self.tranform(features)

        if self.mode == "test":
            if self.datasetname == 'UCF':
                mag = np.linalg.norm(features, axis=2)[:, :, np.newaxis]
                features = np.concatenate((features, mag), axis=2)
            elif self.datasetname == 'XD':
                mag = np.linalg.norm(features, axis=1)[:, np.newaxis]
                features = np.concatenate((features, mag), axis=1)
            return features, name

        else:
            if self.datasetname == 'UCF':
                features = features.transpose(1, 0, 2)  # [10, T, F]
                divided_features = []
                divided_mag = []
                for feature in features:
                    feature = process_feat(feature, self.seg_length)  # ucf(32, 2048)
                    divided_features.append(feature)
                    divided_mag.append(np.linalg.norm(feature, axis=1)[:, np.newaxis])
                divided_features = np.array(divided_features, dtype=np.float32)
                divided_mag = np.array(divided_mag, dtype=np.float32)
                divided_features = np.concatenate((divided_features, divided_mag), axis=2)
                return divided_features, label

            elif self.datasetname == 'XD':
                feature = process_feat(features, 32)
                if self.add_mag_info == True:
                    feature_mag = np.linalg.norm(feature, axis=1)[:, np.newaxis]
                    feature = np.concatenate((feature, feature_mag), axis=1)
                return feature, label

    def __len__(self):
        return len(self.list)

    def randomizer(self, shuffle: bool, lenght: int) -> np.array:
        if shuffle:
            return np.random.permutation(lenght)
        else:
            return np.arange(lenght)

    def get_num_frames(self):
        return self.num_frame
