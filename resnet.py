import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(pretrained=pretrained)

    def forward(self, x):
        x = self.resnet(x)
        return x
