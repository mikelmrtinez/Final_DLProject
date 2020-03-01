# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 19:04:27 2020

@author: Mikel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network_1(nn.Module):
    def __init__(self, config_dim, meta_dim, hidden_dim=64):
        super(Network_1, self).__init__()
        
        self.fc1_m = nn.Linear(meta_dim, 4)
        self.fc2_m = nn.Linear(hidden_dim*4, hidden_dim)
        self.fc3_m = nn.Linear (hidden_dim, hidden_dim//8)
       
        self.fc1_c = nn.Linear(config_dim, hidden_dim)
        self.fc2_c = nn.Linear(hidden_dim, hidden_dim//8)
        
        self.fc3 = nn.Linear(4 + hidden_dim//8, hidden_dim*2)
        self.fc4 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim,1)

        self.dropout = nn.Dropout(p=0.5)
        self.bn_config = nn.BatchNorm1d(config_dim)
        self.bn_meta = nn.BatchNorm1d(meta_dim)
        self.bn_cat = nn.BatchNorm1d(hidden_dim//8 + 4)
        self.sig = nn.Sigmoid()
        
    
    def forward(self, config, meta):
        
        meta = self.bn_meta(meta)
        
        meta = self.fc1_m(meta)
        meta = self.sig(meta)
        
        config = self.bn_config(config)
        config = F.relu(self.fc1_c(config))
        config = F.relu(self.fc2_c(config))

        x = torch.cat((meta, config), 1)
        x = self.bn_cat(x)
    
        x = self.fc3(x)
        x = self.sig(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.sig(x)
        x = self.dropout(x)
        output = self.fc5(x)
        
       
        return output