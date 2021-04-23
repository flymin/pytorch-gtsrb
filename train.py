import argparse
from utils import LoadDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from models.resnet import *


def train(epoch, model, train_loader, optimizer, criterion, log_interval):
    model.train()
    correct = 0
    total = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        max_index = output.max(dim=1)[1]
        correct += (max_index == target).sum()
        total += target.shape[0]
        training_loss += loss
        if batch_idx % log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            print('Train E {}, acc {:.2f}%, Loss: {:.4f}, lr: {:.2e}'.format(
                epoch, 100. * correct / total, loss.data.item(), lr))
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        training_loss / total, correct, total, 100. * correct / total))


def validation(model, val_loader, criterion, scheduler):
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
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    scheduler.step(np.around(validation_loss, 2))
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
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        use_gpu = True
        print("Using GPU")
    else:
        use_gpu = False
        print("Using CPU")

    FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
    Tensor = FloatTensor

    # Data Initialization and Loading

    # Apply data transformations on the training images to augment dataset
    train_data = LoadDataset(args.data, train=True,
                             resize_size=32, jitter_hue=True, norm=True)
    val_data = LoadDataset(args.data, train=False,
                           resize_size=32, norm=True)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=use_gpu)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=use_gpu)

    # Neural Network and Optimizer
    model = eval("{}(num_classes=43)".format(args.arch))
    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        model.cuda()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5, verbose=True)

    best_acc = 0
    save_name = ""
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, optimizer, criterion,
              args.log_interval)
        acc = validation(model, val_loader, criterion, scheduler)
        params = {
            "model": model.state_dict(), "optim": optimizer.state_dict(),
            "acc": acc, "epoch": epoch
        }
        if acc > best_acc:
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if save_name != "":
                os.system("rm ./checkpoint/" + save_name)
            save_name = args.arch + "_E{}_{:.2f}.pth".format(epoch, acc)
            torch.save(params, './checkpoint/' + save_name)
            best_acc = acc
        model_file = "./checkpoint/" + '{}.pth'.format(args.arch)
        torch.save(params, model_file)
        del params
