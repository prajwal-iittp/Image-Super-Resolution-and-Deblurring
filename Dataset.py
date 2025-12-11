#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:35:44 2021

@author: ee19d505
"""
import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['inputs'][idx], 0), np.expand_dims(f['gts'][idx], 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['inputs'])
        
        
class TestDataset(Dataset):
    def __init__(self, h5_file):
        super(TestDataset, self).__init__()
        
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['inputs'][idx][:,:]/255., 0), np.expand_dims(f['labels'][idx][:,:]/255., 0)
    # def __getitem__(self, idx):
    #     with h5py.File(self.h5_file, 'r') as f:
    #         return np.expand_dims(f['inputs'][idx][:,:], 0)
    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['inputs'])
        