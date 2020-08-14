import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from resnet import ResNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

def train(args):
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
        trainset,
        batch_size = 100,
        shuffle = False,
        num_workers = 1
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    resnet = ResNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    epochs = 10

    for epoch in range(epochs):
      running_loss = 0.0

      for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()

        outputs = resnet(inputs.to("cuda"))
        loss = criterion(outputs, labels.to("cuda"))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
          print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / 100))
          running_loss = 0.0

    print('Done')

    correct = 0
    total = 0

    with torch.no_grad():
      for (images, labels) in testloader:
        outputs = resnet(images.to("cuda"))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.to("cpu") == labels).sum().item()

    print('Accuracy: {:.2f}%'.format(100 * float(correct/total)))

def getTransforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ]
    )

if __name__ == '__main__':
    train(parser.parse_args())
