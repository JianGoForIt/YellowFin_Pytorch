'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable

import numpy as np
import sys
sys.path.append("../tuner_utils")
from yellowfin import YFOptimizer


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--mu', default=0.0, type=float, help='momentum')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--logdir', type=str, default="./")
parser.add_argument('--opt_method', type=str, default="YF")
parser.add_argument('--lr_thresh', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

import logging
if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)
logging.basicConfig(filename=args.logdir + "/num.log",level=logging.DEBUG)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
# if args.resume:
if 0:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    net = ResNeXt29_2x64d()
    # net = MobileNet()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
if args.opt_method == "SGD":
    print("using SGD")
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.opt_method == "Adam":
    print("using Adam")
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
elif args.opt_method == "YF":
    print("using YF")
    optimizer = YFOptimizer(net.parameters(), lr=args.lr, mu=args.mu, weight_decay=5e-4, lr_thresh=args.lr_thresh)
else:
    raise Exception("Optimizer not supported")
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_list = []
    lr_list = []
    mu_list = []

    if epoch == 151:
        if args.opt_method == "YF":
            optimizer.set_lr_factor(optimizer.get_lr_factor() / 10.0)
        else:
            for group in optimizer.param_groups:
                    group['lr'] /= 10.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.data[0] )
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if args.opt_method == "YF":
            lr_list.append(optimizer._optimizer.param_groups[0]['lr'] )
            mu_list.append(optimizer._optimizer.param_groups[0]['momentum'] )
        # else:
        #     lr_list.append(optimizer.param_groups[0]['lr'] )
        #     mu_list.append(optimizer.param_groups[0]['momentum'] )

    return loss_list, lr_list, mu_list

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    return acc


if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)
train_loss_list = []
test_acc_list = []
lr_list = []
mu_list = []
for epoch in range(start_epoch, start_epoch+200):
    loss_list, lr_epoch, mu_epoch = train(epoch)
    train_loss_list += loss_list
    test_acc = test(epoch)
    test_acc_list.append(test_acc)

    lr_list += lr_epoch
    mu_list += mu_epoch

    with open(args.logdir + "/loss.txt", "wb") as f:
        np.savetxt(f, np.array(train_loss_list) )

    with open(args.logdir + "/test_acc.txt", "wb") as f:
        np.savetxt(f, np.array(test_acc_list) )

    with open(args.logdir + "/lr.txt", "wb") as f:
        np.savetxt(f, np.array(lr_list) )

    with open(args.logdir + "/mu.txt", "wb") as f:
        np.savetxt(f, np.array(mu_list) )



