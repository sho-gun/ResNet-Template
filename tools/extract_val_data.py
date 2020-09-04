import os
import sys
import shutil
import argparse
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from datasets import BaseDataset

class TmpDataset(BaseDataset):
    def __init__(self, list_path):
        super().__init__(list_path, None)
        os.makedirs('val_data', exist_ok=True)

    def __getitem__(self, idx):
        image_file, label_file = self.img_list[idx]
        shutil.copy(image_file, 'val_data')
        return image_file

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Name of the dataset for train.')

def main(args):
    data_path = os.path.join('data', args.data)
    if not os.path.exists(data_path):
        print('ERROR: No dataset named {}'.format(args.data))
        exit(1)

    dataset = TmpDataset(
        list_path = os.path.join(data_path, 'val.lst'),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 1
    )

    for image in dataloader:
        print(image)

if __name__ == '__main__':
    main(parser.parse_args())
