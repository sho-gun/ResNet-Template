import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.resnet import ResNet
from core.functions import train, val, test
from datasets import BaseDataset

# from memory_profiler import profile

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('torch.cuda.is_available():', torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Name of the dataset for train.')
parser.add_argument('--max_epoch', type=int, required=False, default=500, help='Max number of epoch to train.')
parser.add_argument('--train_batch', type=int, required=False, default=32, help='Batch size for train set.')
parser.add_argument('--val_batch', type=int, required=False, default=32, help='Batch size for validation set.')
parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in your dataset.')
parser.add_argument('--lr', type=float, required=False, default=0.001, help='Learning rate.')

# @profile
def main(args):
    transform = getTransforms()

    data_path = os.path.join('data', args.data)
    if not os.path.exists(data_path):
        print('ERROR: No dataset named {}'.format(args.data))
        exit(1)

    trainset = BaseDataset(
        list_path = os.path.join(data_path, 'train.lst'),
        transform = transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = args.train_batch,
        shuffle = True,
        num_workers = 1
    )

    testset = BaseDataset(
        list_path = os.path.join(data_path, 'val.lst'),
        transform = transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size = args.val_batch,
        shuffle = True,
        num_workers = 1
    )

    model = ResNet(num_layers=18, num_classes=args.num_classes, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    summary(model, input_size=(3, 32, 32))

    max_epoch = args.max_epoch
    last_epoch = 0
    best_val_loss = None
    best_accuracy = None
    train_losses = []
    val_losses = []
    accuracies = []

    output_dir = os.path.join('outputs', args.data)
    model_state_file = os.path.join(output_dir, 'checkpoint.pth.tar')
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(model_state_file):
        checkpoint = torch.load(model_state_file)
        last_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_accuracy = checkpoint['best_accuracy']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        accuracies = checkpoint['accuracies']
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
        accuracy = test(
            model = model,
            dataloader = testloader,
            device = DEVICE
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)

        print('Loss: train = {}, val = {}, acc. = {}'.format(train_loss, val_loss, accuracy))

        # if best_val_loss is None or val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(
        #         model.state_dict(),
        #         os.path.join(output_dir, 'best.pth')
        #     )
        if best_accuracy is None or accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, 'best.pth')
            )

        print('=> saving checkpoint to {}'.format(model_state_file))
        torch.save(
            {
                'epoch': epoch+1,
                'best_val_loss': best_val_loss,
                'best_accuracy': best_accuracy,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'accuracies': accuracies,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            model_state_file
        )

        if (epoch+1) % 100 == 0:
            # plt.plot(range(epoch+1), train_losses, label="train")
            # plt.plot(range(epoch+1), val_losses, label="val")
            # plt.yscale('log')
            # plt.legend()
            # plt.savefig(os.path.join(output_dir, 'losses.png'))
            # plt.clf()

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(range(epoch+1), train_losses, label='train')
            ax1.plot(range(epoch+1), val_losses, label='val')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax2.plot(range(epoch+1), accuracies, color='red', label='accuracy')
            ax2.set_xscale('log')
            handler1, label1 = ax1.get_legend_handles_labels()
            handler2, label2 = ax2.get_legend_handles_labels()
            ax1.legend(handler1 + handler2, label1 + label2, loc=3, borderaxespad=0.)
            plt.savefig(os.path.join(output_dir, 'losses.png'))
            plt.clf()

def getTransforms():
    return transforms.Compose(
        [
            transforms.Resize((720, 1280)),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

if __name__ == '__main__':
    main(parser.parse_args())
