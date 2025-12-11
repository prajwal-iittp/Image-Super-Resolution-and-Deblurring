#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:52:43 2022

@author: ee19d505
"""
import os,time
from os import listdir
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as img
from scipy.ndimage import filters, measurements, interpolation
import glob
from scipy.io import savemat
import ntpath
import hdf5storage as hdf

# Generate random Gaussian kernels and downscale images

"""## Determine sizes and path"""

scale_factor = np.array([2,2])  # choose scale-factor
avg_sf = np.mean(scale_factor)  # this is calculated so that min_var and max_var will be more intutitive
min_var = 0.175 * avg_sf  # variance of the gaussian kernel will be sampled between min_var and max_var
max_var =  3* avg_sf
k_size = np.array([11, 11])  # size of the kernel, should have room for the gaussian
noise_level = 0.25  # this option allows deviation from just a gaussian, by adding multiplicative noise noise

# save kernels inside "kernels/" folder
output_path = "kernels"
os.makedirs(output_path, exist_ok=True)

"""### Function for centering a kernel"""

def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel:
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between od and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (sf - (kernel.shape[0] % 2))

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)

"""## Function for generating one kernel"""

def gen_kernel(k_size, scale_factor, min_var, max_var):
    
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var);
    lambda_2 = min_var + np.random.rand() * (max_var - min_var);
    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2
    
    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2]);
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]
    
    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2  + 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]
    
    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]
    # epsillon = 0.0005
    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)
    
    # shift the kernel so it will be centered
    raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)
    
    # Normalize the kernel and return
    kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    
    return kernel


# def analytic_kernel(k):
#     """Ca_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
#     # Loop over the small kernel to fill the big one
#     for r in range(k_size):
#         for c in range(k_size):
#             big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
#     # Crop the edges of the big kernel to ignore very small values and increase run time of SR
#     lculate the X4 kernel from the X2 kernel"""
#     k_size = k.shape[0]
#     # Calculate the big kernels size
#     bigcrop = k_size // 2
#     cropped_big_k = big_k[crop:-crop, crop:-crop]
#     # Normalize to 1
#     return cropped_big_k / cropped_big_k.sum()

n = 800 * 3
for i in range(n):
    K = gen_kernel(k_size, scale_factor, min_var, max_var)
    hdf.savemat(os.path.join(output_path, f'Kernel{i+1}.mat'), {'Kernel': K})
