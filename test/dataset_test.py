import torch
import torch.utils.data as data
from torch.utils.data import DataLoader


class dataset(data.Dataset):
    def __init__(self):
        self.l = 100
        self.i = 0

    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        self.i += 1
        return 0, idx, self.i


train_nloader = DataLoader(dataset(), batch_size=1, shuffle=True, num_workers=0,
                            pin_memory=False, drop_last=True)


for i in train_nloader:
    print(i)

for i in train_nloader:
    print(i)