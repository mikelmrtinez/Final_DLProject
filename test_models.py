# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:45:32 2020

@author: Mikel
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from api import Benchmark
from utils.data_engineering import read_data, remove_uninformative_features
from utils.preprocessing import DictToTensor, extractDataAsTensor
from utils.plots import plot_evolution
from models.dnn import Network_4
from models.model import Model


#Loading the data set
test_temporal_data = torch.load('data/test_temporal_data.pt')
test_data_tensor = torch.load('data/test_data_tensor.pt')
test_targets = torch.load('data/test_targets.pt')


path = "./results/models/bn_before_2_"
path_models = "./results/models/n4_adam_cos_lr001_2_4900.pkl"

input_config_size = (test_data_tensor.shape[1])
input_temporal_size = (test_temporal_data.shape[1])

net = Network_4(input_config_size, input_temporal_size)

max_epochs = 5000
batch_size = 64

loss = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=0.001)
step_lr = max_epochs/10
gamma = 0.6

print(test_data_tensor.shape)
print(test_temporal_data.shape)

model = Model(net, max_epochs, batch_size, loss, optimizer, step_lr, gamma, path, path_models)
model.test(test_data_tensor, test_temporal_data, test_targets)