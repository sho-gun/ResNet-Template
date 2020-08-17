import torch.utils as utils

class BaseDataset(utils.data.Dataset):
    def __init__(self, list_path=None, transform=None):
        self.list_path = list_path
        self.transform = transform

        self.img_list = []
        if self.list_path is not None:
            self.img_list = [line.strip().split() for line in open(list_path)]

        self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pass
