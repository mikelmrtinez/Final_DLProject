# -*- coding: utf-8 -*-
from utils.plots import plot_evolution
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from models.dnn import Network
# from models.model_kfold_cross_validation import Model
from models.model import Model


config_train = torch.load('./data/config_train.pt')  # .unsqueeze(2)
meta_train = torch.FloatTensor(torch.load('./data/meta_train.pt'))  # .unsqueeze(2)
y_train = torch.load('./data/y_train.pt')

config_val = torch.load('./data/config_val.pt')  # .unsqueeze(2)
meta_val = torch.FloatTensor(torch.load('./data/meta_val.pt'))  # .unsqueeze(2)
y_val = torch.load('./data/y_val.pt')

config_test = torch.load('./data/config_test.pt')  # .unsqueeze(2)
meta_test = torch.FloatTensor(torch.load('./data/meta_test.pt'))  # .unsqueeze(2)
y_test = torch.load('./data/y_test.pt')

print(config_train.shape)
print(meta_train.shape)
print(y_train.shape)

print(config_train.shape)

config_size = config_train.shape[1]
meta_size = meta_train.shape[1]

net = Network(config_size)

print(meta_train.shape)
print(meta_val.shape)
print(meta_test.shape)

# Name the model to train
name_train = "with_distances"
path = "./results/models/" + name_train

# Set Hyperparameters for the model
max_epochs = 2500
batch_size = 64
kfolds = 2
loss = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
step_lr = max_epochs/10
gamma = 0.6

model = Model(net, max_epochs, batch_size, loss, optimizer, scheduler, step_lr, gamma, path)

train_hist, val_hist = model.train(config_train, meta_train, y_train,
                                   config_val, meta_val, y_val)

np.save('results/stats/train_hist_'+name_train, train_hist)
np.save('results/stats/val_hist_'+name_train, val_hist)
print("\n Saved history of train and validation.\n")

# model.test(config_test, meta_test, y_test)

plot_evolution(train_hist, val_hist, './results/plots/'+name_train)
print("\n Saved plots.")
