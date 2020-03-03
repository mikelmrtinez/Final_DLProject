from utils.data_engineering import read_data, remove_uninformative_features, TrainValSplitter
from utils.preprocess import droped_conf_features, kept_meta_features, config_list_to_tensor, metadata_list_to_tensor
import torch
import json
import pandas as pd
import numpy as np


bench_dir = "./data/six_datasets_lw.json"
train_datasets = ['adult', 'higgs', 'vehicle', 'volkert']
test_datasets = ['Fashion-MNIST', 'jasmine']


X, y, dataset_names = read_data(bench_dir, train_datasets)
X_test, y_test, dataset_names_test = read_data(bench_dir, test_datasets)
with open("data/metafeatures.json", "r") as f:
    metafeatures = json.load(f)


tv_splitter = TrainValSplitter(X, dataset_names=dataset_names)

X_train, X_val = tv_splitter.split(X)
y_train, y_val = tv_splitter.split(y)
dataset_names_train, dataset_names_val = tv_splitter.split(dataset_names)

kept_meta = kept_meta_features(metafeatures)
droped_conf = droped_conf_features(X_train)

y_train = torch.FloatTensor(y_train).reshape(-1, 1)
y_val = torch.FloatTensor(y_val).reshape(-1, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)


print("Getting configs\n")
conf_train = config_list_to_tensor(X_train, droped_conf)
conf_val = config_list_to_tensor(X_val, droped_conf)
conf_test = config_list_to_tensor(X_test, droped_conf)

print(conf_train.shape)
print(conf_val.shape)
print(conf_test.shape)


print('Getting MetaFeatures...\n')
meta_train = metadata_list_to_tensor(metafeatures, dataset_names_train, kept_meta)
meta_val = metadata_list_to_tensor(metafeatures, dataset_names_val, kept_meta)
meta_test = metadata_list_to_tensor(metafeatures, dataset_names_test, kept_meta)

print(meta_train.shape)
print(meta_test.shape)
print(meta_val.shape)

torch.save(conf_train, './data/config_train.pt')  # and torch.load('file.pt')
torch.save(meta_train, './data/meta_train.pt')
torch.save(y_train, './data/y_train.pt')
torch.save(conf_val, './data/config_val.pt')
torch.save(meta_val, './data/meta_val.pt')
torch.save(y_val, './data/y_val.pt')
torch.save(conf_test, './data/config_test.pt')
torch.save(meta_test, './data/meta_test.pt')
torch.save(y_test, './data/y_test.pt')
