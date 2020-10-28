import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

from models.resnet import ResNet
from datasets import EvalDataset, BaseDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('torch.cuda.is_available():', torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--class_list', type=str, required=True, help='Path to the class list file.')
parser.add_argument('--model_file', type=str, required=True, help='Name of the checkpoint for testing.')
parser.add_argument('--input_data', type=str, required=True, help='Target directory.')

def main(args):
    transform = getTransforms()

    data_path = args.input_data
    if not os.path.exists(data_path):
        print('ERROR: No dataset named {}'.format(data_path))
        exit(1)

    dataset = EvalDataset(
        data_path,
        transform = transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 1
    )

    with open(args.class_list, 'r') as class_file:
        class_names = []
        for class_name in class_file.readlines():
            if len(class_name.strip()) > 0:
                class_names.append(class_name.strip())

    model = ResNet(num_layers=18, num_classes=len(class_names)).to(DEVICE)
    model = model.eval()

    output_dir = os.path.join(data_path, 'out')
    os.makedirs(output_dir, exist_ok=True)

    model_file = args.model_file

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

    font = cv2.FONT_HERSHEY_SIMPLEX

    with torch.no_grad():
        for data, path in dataloader:
            outputs = model(data.to(DEVICE))
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.to('cpu')[0].item()
            class_text = class_names[predicted]
            print(class_text, path)

            image = cv2.imread(path[0], cv2.IMREAD_COLOR)
            image = cv2.rectangle(image,(0,0),(150,25),(255,255,255),-1)
            image = cv2.rectangle(image,(0,0),(150,25),(255,0,0),2)
            cv2.putText(image, class_text, (5,15) , font, 0.5, (255,0,), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(output_dir, os.path.basename(path[0])), image)

def getTransforms():
    return transforms.Compose(
        [
            transforms.Resize((720, 1280)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

if __name__ == '__main__':
    main(parser.parse_args())
