import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms

from models.resnet import ResNet
from core.functions import test
from datasets import BaseDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('torch.cuda.is_available():', torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Name of the dataset for train.')
parser.add_argument('--model_file', type=str, required=False, default='', help='Name of the checkpoint for testing.')

def main(args):
    transform = getTransforms()

    data_path = os.path.join('data', args.data)
    if not os.path.exists(data_path):
        print('ERROR: No dataset named {}'.format(args.data))
        exit(1)

    testset = BaseDataset(
        list_path = os.path.join(data_path, 'val.lst'),
        transform = transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size = 1,
        shuffle = True,
        num_workers = 1
    )

    class_list = getClassList(data_path)

    model = ResNet(num_layers=18, num_classes=len(class_list)).to(DEVICE)
    model.eval()

    output_dir = os.path.join('outputs', args.data)
    model_state_file = os.path.join(output_dir, 'checkpoint.pth.tar')

    model_file = args.model_file
    if len(model_file) == 0:
        model_file = model_state_file

    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print('=> loaded {}'.format(model_file))

    else:
        print('model_file "{}" does not exists.'.format(model_file))
        exit(1)

    accuracy = test(
        model = model,
        dataloader = testloader,
        device = DEVICE,
        classes = class_list
    )

    print('Accuracy: {:.2f}%'.format(100 * accuracy))

def getTransforms():
    return transforms.Compose(
        [
            transforms.Resize((720, 1280)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

def getClassList(data_path):
    class_list = []
    with open(os.path.join(data_path, 'classes.txt'), 'r') as class_file:
        for class_name in class_file.readlines():
            if len(class_name.strip()) > 0:
                class_list.append(class_name.strip())
    return class_list

if __name__ == '__main__':
    main(parser.parse_args())
