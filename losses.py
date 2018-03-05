import torch
from torch import autograd
import torch.nn.functional as F

# TODO: hack!
use_cuda = torch.cuda.is_available()

def gradient_penalty_loss(netD, real_data, fake_data, one_sided=False):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(real_data.size(0), 1)
    #alpha = alpha.expand_as(real_data.data)

    # unnecessary with broadcasting
    alpha = alpha.expand(real_data.size(0), real_data.nelement()//real_data.size(0)).contiguous().view_as(real_data.data)

    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data.data + ((1 - alpha) * fake_data.data)
    #interpolates = real_data + (alpha * real_data_grad.data * torch.norm(real_data_grad.data, p=2, dim=1).unsqueeze(1))

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    scores = netD(interpolates)

    #gradients = autograd.grad(outputs=scores, inputs=[g1, g2],
    gradients_list = autograd.grad(outputs=scores, inputs=interpolates,
                              grad_outputs=torch.ones(scores.size()).cuda() if use_cuda else torch.ones(scores.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)

    # Get gradient relative to interpolates 

    #grad1, grad2 = gradients_list
    #gradients = grad1.clone() # assume interpolate in g1
    # if real was in g1, copy from same row in g2
    #real_indices = g1_is_real.nonzero()
    #if len(real_indices) > 0:
    #    real_indices = real_indices.squeeze()
    #    gradients[real_indices] = grad2[real_indices]
    #gradients = gradients.contiguous().view(gradients.size(0), -1)

    #if ONE_SIDED:
    #    gradient_penalty = (F.relu(gradients.norm(2, dim=1) - 1, inplace=True) ** 2).mean()
    #else:
    #    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() 

    # Gradients relative to all inputs

    gradient_penalty = 0
    for gradients in gradients_list:
        gradients = gradients.view(gradients.size(0), -1)
    
        if one_sided:
            gradient_penalty += (F.relu(gradients.norm(2, dim=1) - 1, inplace=True) ** 2).mean()
        else:
            gradient_penalty += ((gradients.norm(2, dim=1) - 1) ** 2).mean() 

    return gradient_penalty
