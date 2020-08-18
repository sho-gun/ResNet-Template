import cv2
import torch.utils as utils

class BaseDataset(utils.data.Dataset):
    def __init__(self, list_path=None, transform=None):
        self.list_path = list_path
        self.transform = transform

        self.img_list = []
        if self.list_path is not None:
            self.img_list = [line.strip().split() for line in open(list_path)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_file, label_file = self.img_list[idx]

        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if self.transform:
            image = self.transform(image)

        with open(label_file, 'r', encoding='utf-8') as f:
            label = int(f.readline().strip())

        return image, label
