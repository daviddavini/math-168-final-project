import math
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import argparse

import datetime
import os
from net import LinearDynamicalSystem

from networkx_graph import weight_matrix, speed_matrix

save_dir = "dynamical_system"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
num_epochs = 1000
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
best_acc = 0  # best test accuracy

# Data
print('==> Preparing data..')
N = 228
A = weight_matrix(N, weighted=False)
A = torch.Tensor(A, device=device)
V = speed_matrix(N)
V = torch.Tensor(V, device=device)
DT = 5 # 5 minutes
NUM_TIMES = V.shape[0]
NUM_NODES = V.shape[1]

# trainset = torch.utils.data.TensorDataset(X, Y)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

# Model
print('==> Building model..')
net = LinearDynamicalSystem(A, timestep=DT)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

def plot_loss(train_losses):
    plt.semilogy(train_losses)
    plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=300)

def plot_weights(X):
    X = X.detach().cpu().numpy()
    plt.imshow(X, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title("{} data at epoch {}".format(name, epoch))
    plt.savefig(os.path.join(save_dir, "{}_data_epoch_{}.png".format(name, epoch)), dpi=300)
    plt.clf()

# Training
def train(epoch):
    # print(net.weight[0])
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    #     inputs, targets = inputs.to(device), targets.to(device)
    #     optimizer.zero_grad()
    #     outputs = net(inputs)
    #     loss = criterion(outputs, targets)
    #     loss.backward()
    #     optimizer.step()

    #     train_loss += loss.item()
    outputs = net(V)
    targets = V[1:]
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    batch_idx = 0
    # print((1 - torch.eye(A.shape[0]))[0])
    # print(A[0])
    # print((A * (1 - torch.eye(A.shape[0]))))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_loss = train_loss/(batch_idx+1)
    print('[%s] Training -- Loss: %.3e' % (timestamp, avg_loss))

    return avg_loss
    # plot the weight matrix using matplotlib

train_losses = []
for epoch in range(num_epochs):
    loss = train(epoch)
    train_losses.append(loss)
    scheduler.step()

plot_loss(train_losses)
