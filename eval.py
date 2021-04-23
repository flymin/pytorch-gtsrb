import argparse
from utils import LoadDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from models.resnet import *

def validation(model, val_loader, criterion):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        with torch.no_grad():
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            # sum up batch loss
            validation_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.max(dim=1)[1]
            correct += pred.eq(target).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)
    print('Val: Average loss: {:.4f}, Accuracy: {:.2f}% ({}/{})\n'.format(
        validation_loss, acc, correct, len(val_loader.dataset)))
    return acc


# Training settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
    parser.add_argument('--arch', type=str, default='ResNet18',
                        help="architecture to train")
    parser.add_argument('--data', type=str, default='data', metavar='D',
                        help="folder where data is located.")
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--resume', type=str, default=None, metavar='D',
                        help="model to load for test")
    args = parser.parse_args()
    assert args.resume is not None

    if torch.cuda.is_available():
        use_gpu = True
        print("Using GPU")
    else:
        use_gpu = False
        print("Using CPU")

    # Apply data transformations on the training images to augment dataset
    val_data = LoadDataset(args.data, train=False, resize_size=32, norm=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=use_gpu)

    # Neural Network and Optimizer
    model = eval("{}(num_classes=43)".format(args.arch))
    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        model.cuda()
    model.load_state_dict(torch.load(args.resume)["net"])
    print("Loaded ckpt from: {}".format(args.resume))

    acc = validation(model, val_loader, criterion)
