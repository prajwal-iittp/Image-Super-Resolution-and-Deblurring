#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:16:51 2021

@author: ee19d505
"""
import torch
import torch.nn as nn
from util import swap_axis
import time


class MyNet(nn.Module):
    def __init__(self, ):
        super(MyNet, self).__init__()
        struct = [3, 3, 3, 3, 3]
        channels = 32
        self.first_layer = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size = struct[0], padding = struct[0]//2, padding_mode= 'replicate', bias=False)
        #padding_mode = 'zeros', 'reflect', 'replicate' or 'circular'. 

        feature_block = []  # Stacking intermediate layer
        for layer in range(1, len(struct)-1):
            feature_block += [nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=struct[layer], padding = struct[layer]//2,
                                        bias=False)]
        self.feature_block = nn.Sequential(*feature_block)
      
        self.final_layer = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=struct[-1], padding = struct[-1]//2, bias=False)
        
    def forward(self, input_tensor):
        input_tensor = input_tensor
        starttime = time.time()
        x = self.first_layer(input_tensor)      
        features = self.feature_block(x)  
        output = self.final_layer(features)
        # print(time.time() - starttime)
        return output




def weights_init_G(m):
    """ initialize weights  """
    if isinstance(m, nn.Conv2d):    
            torch.manual_seed(72)
            torch.nn.init.xavier_uniform_(m.weight.data)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
            
            
            
            
            
