#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:19:27 2021

@author: ee19d505
"""
import os
import h5py
import numpy as np
# import scipy.io
import torch
# from scipy.ndimage.interpolation import rotate
# import PIL.Image as pil_image
import matplotlib.pyplot as plt
import torch.fft as fft
# import cv2 as cv
# import glob
# from util import gaussian_noise
# from torchvision import datasets, transforms
import hdf5storage

# ks = [11,21]
# ns = [800, 1600, 2400, 3200, 4000]
ks = [11]
ns = [2400]
for p in range(len(ks)):
    for q in range(len(ns)):
        #create H5 file
        h5_file = h5py.File('.h5'.format(ks[p], ks[p], ns[q]), 'w')
        dataDir = "kernels/".format(ks[p], ks[p], ns[q])
        
        inputs = []
        gts = []
        
        for file in os.listdir(dataDir) :
            
            x = hdf5storage.loadmat(dataDir+file)
            y = x['Kernel']
            
            m = np.uint8(y.shape[0])
            label  = np.zeros_like(y)
          
            label[0,0] = 1
            
            inputs.append(y)
            gts.append(label)
            
         
        print(len(inputs), len(gts))
           
        inputs = np.array(inputs)
        gts = np.array(gts)
        
        
        h5_file.create_dataset('inputs', data=inputs)
        h5_file.create_dataset('gts', data=gts)
        
        # #close H5 file
        h5_file.close()
        
