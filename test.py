#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 12:02:28 2021

@author: vamsi
"""

import os,time
# from os import listdir
# import tqdm
# from datetime import datetime
import numpy as np
import torch
# from torch.utils.data.dataloader import DataLoader
# from tqdm import tqdm
# from Dataset import TrainDataset,TestDataset
# import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import argparse
from models import MyNet
# from unet import UNet
import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import scipy.io as io
# import hdf5storage as hdf
# import glob
import matplotlib.image as mpimg
from util import calc_psnr,  convert_rgb_to_ycbcr, convert_ycbcr_to_rgb
import PIL.Image as pil_image
# from PIL import ImageOps, ImageFilter
# from torchvision.utils import save_image
# import torchvision.transforms as transforms
from Metrics import calculate_psnr, calculate_ssim, ssim
# import torch.nn.functional as F
# from scipy.ndimage import measurements, interpolation
# from Calc_Metrics import metrics
#S=io.loadmat('/media/root1/data1/ sumanth_folder/VVV/Wrapped.mat')
#S1=S['Wrapped']
import cv2



# print(len(trainData))

parser = argparse.ArgumentParser(description='NSDBNet')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



print ('*******************************************************')
start_time=time.time()

cwd=os.getcwd()
#directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+'model'
#print('Model will be saved to  :', directory)
#if not os.path.exists(directory):
#     os.makedirs(directory)

print('------------------------------------------------------------')

gpu = True


gt_folder_dir = "/mnt/DATA/Vamsi/EE24M309/Dataset/DIV2K_gt_blur/gt/"
#(Sir)folder_dir = "/mnt/DATA/Vamsi/Downloads/NSDeblur/GOPRO_Large/test/GOPR0384_11_00/blur/"
#(Me: 2x)folder_dir = "/mnt/DATA/Vamsi/EE24M309/Dataset/DIV2K_gt_blur/blur/"
folder_dir = "/mnt/DATA/Vamsi/EE24M309/Dataset/DIV2K_gt_blur/blur/"
loadmodel= '/mnt/DATA/Vamsi/EE24M309/code_final (copy)_MAIN/17Oct_0102pm_model/NSDBNet30_model.pth'
 
cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 

if gpu == True:    
    net = MyNet()
    net = nn.DataParallel(net)
    #net.load_state_dict(torch.load(loadmodel))
    net.to(device)
    net.load_state_dict(torch.load(loadmodel))
    net.eval()       
else:
    #net.load_state_dict(torch.load(loadmodel))
    net = MyNet()
    net.load_state_dict(torch.load(loadmodel))
    net.eval()
    
  
p = []
s = []
    
count = 0
# for image_name in os.listdir(folder_dir):
for i in range(100):
    image = pil_image.open(folder_dir+"08{}x2.png".format(i+1))
    image = np.array(image).astype(np.float32)
    # print(image.shape)
    gt_image = pil_image.open(gt_folder_dir+"08{}.png".format(i+1))
    gt_image = np.array(gt_image).astype(np.float32)
    
    image1 = torch.from_numpy(image[:,:,0]).unsqueeze(0).unsqueeze(0)
    image1 /= 255.
    # print(image1.shape)
    image2 = torch.from_numpy(image[:,:,1]).unsqueeze(0).unsqueeze(0)
    image2 /= 255.
    
    image3 = torch.from_numpy(image[:,:,2]).unsqueeze(0).unsqueeze(0)
    image3 /= 255.
    
     
    
    
    with torch.no_grad():
        # preds = net(y)
        p1 = net(image1.to(device)).clamp(0.0, 1.0)
        p2 = net(image2.to(device)).clamp(0.0, 1.0)
        p3 = net(image3.to(device)).clamp(0.0, 1.0)
        
  
    p1 = p1.mul(255.0).squeeze(0).squeeze(0).cpu().numpy()
    p2 = p2.mul(255.0).squeeze(0).squeeze(0).cpu().numpy()
    p3 = p3.mul(255.0).squeeze(0).squeeze(0).cpu().numpy()
    
    output = np.array([p1,p2,p3]).transpose([1,2,0])
    
    output = np.clip(output, 0.0, 255.0).astype(np.uint8)
    
    # Resize output to GT resolution before metrics
   # if output.shape != gt_image.shape:
    #    output = cv2.resize(output, (gt_image.shape[1], gt_image.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Resize output to GT resolution before metrics
    # # Resize output to GT resolution before metrics
    if output.shape != gt_image.shape:
        first_output = cv2.resize(output, (gt_image.shape[1], gt_image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        output = cv2.bilateralFilter(first_output, d=9, sigmaColor=75, sigmaSpace=75)
    
    psnrValue = calculate_psnr(output, gt_image) 
    ssimValue = ssim(output, gt_image)
    
    p.append(psnrValue)
    s.append(ssimValue)
    
    output = pil_image.fromarray(output)
    
    output.save('/mnt/DATA/Vamsi/EE24M309/code_final (copy)_MAIN/output_2x/img{}.png'.format(i+1))
    
    count = count + 1
    
    print(count)


p = sum(p)/count
s = sum(s)/count
    
    
print("SSIM ", s)
    
print("PSNR ", p)

   
############################################################################################################################################
