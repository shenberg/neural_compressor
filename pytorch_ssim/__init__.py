import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    # mimic fspecial from matlab
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, max_val=255.0):
    "img1, img2 -> cs, ssim (lcs)"
    mu1 = F.conv2d(img1, window, padding = 0, groups = channel)
    mu2 = F.conv2d(img2, window, padding = 0, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = 0, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = 0, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = 0, groups = channel) - mu1_mu2

    C1 = (0.01*max_val)**2
    C2 = (0.03*max_val)**2

    v1 = F.relu(2.0*sigma12, inplace=True) + C2
    v2 = sigma1_sq + sigma2_sq + C2

    # luminosity * contrast * similarity
    ssim_map = ((2*mu1_mu2 + C1)*v1)/((mu1_sq + mu2_sq + C1)*v2)
    # contrast * similarity
    cs_map = v1/v2

    ssim = ssim_map.mean(1).mean(1).mean(1)
    cs = cs_map.mean(1).mean(1).mean(1)
    return cs, ssim


class MS_SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, max_val = 1.0, size_average = True, weights = None, mode='product'):
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)
        self.weights_base = torch.autograd.Variable(torch.Tensor(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))
        self.weights = self.weights_base
        self.max_val = max_val
        self.downscale = nn.AvgPool2d(2, 2)
        if mode not in ('product','sum'):
            raise ValueError('invalid mode ' + mode)
        self.mode = mode

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        weights = self.weights
        if weights.data.type() != img1.data.type():
            weights = self.weights_base
            if img1.is_cuda:
                weights = weights.cuda()
            weights = weights.type_as(img1)
            self.weights = weights

        csses = []
        for i in range(len(weights)):
            #print(img1.size(),img2.size())
            cs, ssim = _ssim(img1, img2, window, self.window_size, channel, self.max_val)
            csses.append(cs)
            img1 = self.downscale(img1)
            img2 = self.downscale(img2)
        #print(img1)
        #print(img2)
        #print(csses)
        #print(ssim)
        result_array = torch.stack(csses[:-1] + [ssim], dim=1)
        #print(result_array)
        #print(weights)
        if self.mode == 'product':
            results = (result_array**weights).prod(dim=1)
        else:
            results = (result_array * weights).sum(dim=1) / weights.sum()
        #print(results)
        if not self.size_average:
            return results
        else:
            return results.mean()


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)
