import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms

from models.resnet import ResNet
from core.functions import test

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('torch.cuda.is_available():', torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Name of the dataset for train.')
parser.add_argument('--model_file', type=str, required=False, default='', help='Name of the checkpoint for testing.')

def main(args):
    transform = getTransforms()

    testset = torchvision.datasets.CIFAR10(
        root = './data',
        train = False,
        download = True,
        transform = transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size = 100,
        shuffle = False,
        num_workers = 1
    )

    resnet = ResNet().to(DEVICE)

    output_dir = os.path.join('outputs', args.data)
    model_state_file = os.path.join(output_dir, 'checkpoint.pth.tar')

    model_file = args.model_file
    if len(model_file) == 0:
        model_file = model_state_file

    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        if 'state_dict' in checkpoint.keys():
            resnet.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            resnet.load_state_dict(checkpoint, strict=False)
        print('=> loaded {}'.format(model_file))

    accuracy = test(
        model = resnet,
        dataloader = testloader,
        device = DEVICE
    )

    print('Accuracy: {:.2f}%'.format(100 * accuracy))

def getTransforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ]
    )

if __name__ == '__main__':
    main(parser.parse_args())
