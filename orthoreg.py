import torch
import torch.nn as nn

import math
#TODO: hack
cuda = torch.cuda.is_available()

exp10 = math.exp(10) # TODO: configurable
def orthoreg_module_loss(weights, beta=0.0001, lamb = 10, epsilon = 0.0000001):
    # as 2d vectors
    filters = weights.view(weights.size(0), -1)
    norms = filters.norm(p=2, dim=1)

    filters = filters / (norms + epsilon).unsqueeze(1)
    # mm all filters with all filters
    grad = torch.mm(filters, filters.transpose(1,0))
    grad = torch.exp(grad*lamb)
    grad = (grad*lamb).div_(grad + exp10)
    # TODO: better indexing for great justice (and more speed)
    mask = torch.eye(grad.size(0)).byte()
    if cuda:
        mask = mask.cuda()
    grad[mask] = 0
    
    #grad.mul_(1 - torch.eye(grad.size(0))) # if previous doesn't work due to not being differentiable
    #grad.sub_(torch.eye(grad.size(0))) # option 2
    return torch.sum(torch.abs(grad))*beta



def orthoreg_loss(net, destination):
    for module in net.modules():
        # TODO: Add Linear, ConvTranspose2d to list?
        if isinstance(module, (nn.Conv2d)):
            destination += orthoreg_module_loss(module.weight)
        
    return destination
