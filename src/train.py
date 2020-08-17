import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.resnet import ResNet
from core.functions import train, val

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('torch.cuda.is_available():', torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Name of the dataset for train.')
parser.add_argument('--max_epoch', type=int, required=False, default=500, help='Max number of epoch to train.')
parser.add_argument('--train_batch', type=int, required=False, default=32, help='Batch size for train set.')
parser.add_argument('--val_batch', type=int, required=False, default=32, help='Batch size for validation set.')
parser.add_argument('--lr', type=float, required=False, default=0.001, help='Learning rate.')

def main(args):
    # TODO implement original dataloader class
    transform = getTransforms()

    trainset = torchvision.datasets.CIFAR10(
        root = './data',
        train = True,
        download = True,
        transform = transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = 100,
        shuffle = True,
        num_workers = 1
    )

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

    model = ResNet(num_layers=18, pretrained=False).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    max_epoch = args.max_epoch
    last_epoch = 0
    best_val_loss = None
    train_losses = []
    val_losses = []

    output_dir = os.path.join('outputs', args.data)
    model_state_file = os.path.join(output_dir, 'checkpoint.pth.tar')
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(model_state_file):
        checkpoint = torch.load(model_state_file)
        last_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint (epoch {})'.format(last_epoch))

    for epoch in range(last_epoch, max_epoch):
        print('Epoch {}'.format(epoch))

        train_loss = train(
            model = model,
            dataloader = trainloader,
            criterion = criterion,
            optimizer = optimizer,
            device = DEVICE
        )
        val_loss = val(
            model = model,
            dataloader = testloader,
            criterion = criterion,
            device = DEVICE
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print('Loss: train = {}, val = {}'.format(train_loss, val_loss))

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, 'best.pth')
            )

        print('=> saving checkpoint to {}'.format(model_state_file))
        torch.save(
            {
                'epoch': epoch+1,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            model_state_file
        )

        if (epoch+1) % 100 == 0:
            plt.plot(range(epoch+1), train_losses, label="train")
            plt.plot(range(epoch+1), val_losses, label="val")
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'losses.png'))
            plt.clf()

def getTransforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ]
    )

if __name__ == '__main__':
    main(parser.parse_args())
