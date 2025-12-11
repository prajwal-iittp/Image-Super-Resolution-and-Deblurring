import torch
import torch.fft as fft
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from util import shave_a2b, resize_tensor_w_kernel, create_penalty_mask, map2tensor


    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DownScaleLoss(nn.Module):
    """ Computes the difference between the Generator's downscaling and an ideal (bicubic) downscaling"""

    def __init__(self, scale_factor):
        super(DownScaleLoss, self).__init__()
        self.loss = nn.MSELoss()
        bicubic_k = [[0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [-.0013275146484375, -0.0039825439453130, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0013275146484380, -0.0039825439453125, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625]]
        self.bicubic_kernel = Variable(torch.Tensor(bicubic_k).cuda(), requires_grad=False)
        self.scale_factor = scale_factor

    def forward(self, g_input, g_output):
        downscaled = resize_tensor_w_kernel(im_t=g_input, k=self.bicubic_kernel, sf=self.scale_factor)
        # Shave the downscaled to fit g_output
        return self.loss(g_output, shave_a2b(downscaled, g_output))


class SumOneLoss(nn.Module):
    """ Encourages the input sum to be 1 """

    def __init__(self):
        super(SumOneLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.Tensor([1]).to(kernel.device), torch.sum(kernel).reshape(1))


class DotProductLoss(nn.Module):
    """ Encourages the input sum to be 1 """

    def __init__(self):
        super(DotProductLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, k, ik):
        return self.loss(torch.Tensor([1]).to(k.device), torch.sum(torch.mul(k, ik)).reshape(1))
    
class MagSpecLoss(nn.Module):
    
    def __init__(self):
        super(MagSpecLoss, self).__init__()
        
        self.loss = nn.L1Loss()
        
    def forward(self, k, dk, gt):
        [b, c, m, n] = k.shape
        k_ft = fft.fft2(k)
        dk_ft = fft.fft2(dk)
        h = k_ft.abs() * dk_ft.abs()
        y = fft.fft2(gt).abs()
        # y = torch.ones((b,c,m,n)).to(device)
        return self.loss(y,h)

class PhaseLoss(nn.Module):
    
    def __init__(self):
        super(PhaseLoss, self).__init__()
        self.loss = nn.L1Loss()
        
    def forward(self, k, dk, gt):
        [b,c,m,n] = k.shape
        k_ft = fft.fft2(k)
        dk_ft = fft.fft2(dk)
        h = k_ft.angle() + dk_ft.angle()
        y = fft.fft2(gt).angle()
        # y = torch.zeros((b,c,m,n)).to(device)
        return self.loss(y,h)
    
    
class CentralizedLoss(nn.Module):
    """ Penalizes distance of center of mass from K's center"""

    def __init__(self, k_size=21, scale_factor=1):
        super(CentralizedLoss, self).__init__()
        self.indices = Variable(torch.arange(0., float(k_size)).cuda(), requires_grad=False)
        wanted_center_of_mass = k_size // 2 + 0.5 * (int(1 / scale_factor) - k_size % 2)
        self.center = Variable(torch.FloatTensor([wanted_center_of_mass, wanted_center_of_mass]).cuda(), requires_grad=False)
        self.loss = nn.MSELoss()

    def forward(self, kernel):
        """Return the loss over the distance of center of mass from kernel center """
        r_sum, c_sum = torch.sum(kernel, dim=1).reshape(1, -1), torch.sum(kernel, dim=0).reshape(1, -1)
        return self.loss(torch.stack((torch.matmul(r_sum, self.indices) / torch.sum(kernel),
                                      torch.matmul(c_sum, self.indices) / torch.sum(kernel))), self.center)


class BoundariesLoss(nn.Module):
    """ Encourages sparsity of the boundaries by penalizing non-zeros far from the center """

    def __init__(self, k_size=11):
        super(BoundariesLoss, self).__init__()
        self.mask = map2tensor(create_penalty_mask(k_size, 30))
        # print(self.mask)
        self.zero_label = Variable(torch.zeros(k_size).cuda(), requires_grad=False)
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(kernel * self.mask, self.zero_label)


class SparsityLoss(nn.Module):
    """ Penalizes small values to encourage sparsity """
    def __init__(self):
        super(SparsityLoss, self).__init__()
        self.power = 0.2
        self.loss = nn.L1Loss()

    def forward(self, kernel):
        return self.loss(torch.abs(kernel) ** self.power, torch.zeros_like(kernel))
