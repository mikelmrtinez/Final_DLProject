# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:46:48 2020

@author: Mikel
"""
#from api import Benchmark
from utils.data_engineering import read_data, remove_uninformative_features, TrainValSplitter
from utils.preprocessing import DictToTensor, extractDataAsTensor
import torch
import json
import pandas as pd
import numpy as np


def getmetadata(x, names, metadata):
    for i in range(len(x)):
        name = names[i]
        vec = np.zeros((len(x), 62))
        j = 0
        for key, val in metadata[name].items():
            vec[i, j] = val
            if j > 60:
                break
            j += 1
    return vec


def cleanNan(x):
    df = pd.DataFrame(x)
    df = df.dropna(axis='columns')
    return df.values


bench_dir = "./data/six_datasets_lw.json"


train_datasets = ['adult', 'higgs', 'vehicle', 'volkert']
test_datasets = ['Fashion-MNIST', 'jasmine']


X, y, dataset_names = read_data(bench_dir, train_datasets)
X_test, y_test, dataset_names_test = read_data(bench_dir, test_datasets)

tv_splitter = TrainValSplitter(X, dataset_names=dataset_names)

X_train, X_val = tv_splitter.split(X)
y_train, y_val = tv_splitter.split(y)
dataset_names_train, dataset_names_val = tv_splitter.split(dataset_names)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("X_val:", X_val.shape)


X_train_clean = remove_uninformative_features(X_train)
X_val_clean = remove_uninformative_features(X_val)
X_test_clean = remove_uninformative_features(X_test)

X_train_tensor = DictToTensor(X_train_clean)
X_val_tensor = DictToTensor(X_val_clean)
X_test_tensor = DictToTensor(X_test_clean)

y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_val = torch.FloatTensor(y_val).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)


with open("data/metafeatures.json", "r") as f:
    metafeatures = json.load(f)

print('Getting MetaFeatures...\n')


meta_train = getmetadata(X_train, dataset_names_train, metafeatures)
meta_val = getmetadata(X_val, dataset_names_val, metafeatures)
meta_test = getmetadata(X_test, dataset_names_test, metafeatures)

x = np.concatenate((meta_train, meta_test, meta_val), 0)
print(x.shape)
x = cleanNan(x)
meta_train_clean = x[0:meta_train.shape[0]]
meta_test_clean = x[meta_train.shape[0]:meta_train.shape[0]+meta_test.shape[0]]
meta_val_clean = x[meta_train.shape[0]+meta_test.shape[0]:]

print(meta_train_clean.shape)
print(meta_train.shape)
print(meta_test_clean.shape)
print(meta_test.shape)
print(meta_val_clean.shape)
print(meta_val.shape)

torch.save(X_train_tensor, './data/config_train.pt')  # and torch.load('file.pt')
torch.save(meta_train_clean, './data/meta_train.pt')
torch.save(y_train, './data/y_train.pt')
torch.save(X_val_tensor, './data/config_val.pt')
torch.save(meta_val_clean, './data/meta_val.pt')
torch.save(y_val, './data/y_val.pt')
torch.save(X_test_tensor, './data/config_test.pt')
torch.save(meta_test_clean, './data/meta_test.pt')
torch.save(y_test, './data/y_test.pt')
