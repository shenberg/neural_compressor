import os, sys, time
sys.path.append(os.getcwd())

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
import torch.nn.functional as F
#torch.set_default_tensor_type('torch.cuda.HalfTensor')

import time
import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.cifar10
import tflib.plot
#import tflib.inception_score

import numpy as np
from tqdm import tqdm

from layers import LayerNorm
from models import Generator, Encoder, AutoEncoder, AutoEncoderUpscale, AutoEncoderPSP, SNEncoder
from utils import DatasetSubset, mix_samples, _mix_samples, load_batches, batchify, EagerFolder
import orthoreg
from pytorch_ssim import SSIM
import mmd

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
USE_TENSOR_DATASET = False
DATA_DIR = '/mnt/7FC1A7CD7234342C/compression/dataset'
BATCHES_PATH = '/mnt/7FC1A7CD7234342C/compression-results/dataset_bs_128_size_64_half_2k/'
OUTPUT_BASE_DIR = '/mnt/7FC1A7CD7234342C/compression-results/'
RUN_PATH = '{}{}/'.format(OUTPUT_BASE_DIR, time.strftime('%Y_%m_%d_%H_%M_%S'))  #TODO: generate by settings
if not os.path.exists(RUN_PATH):
    os.mkdir(RUN_PATH)
#TODO:hack
tflib.plot.log_dir = RUN_PATH

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

MODE = 'wgan-ae' # Valid options are dcgan, wgan, or wgan-gp
DIM = 64 # This overfits substantially; you're probably better off with 64
CRITIC_DIM = 32 # ambition
INPUT_DIM = 256 # generator input dimension (latent variable dimension)
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 24 # 64 # Batch size
IMAGE_SIZE = 64 # size of side of image
ITERS = 100000 # How many generator iterations to train for

KERNEL_SIZE = 4


LR = 1e-3
TRAIN_RATIO=0.05#0.9

GENERATOR_INSTANCE_NORM = LayerNorm

ENCODER_LAYER_NORM = True

ORTHOREG_LOSS = False#True

ONE_SIDED = False

GENERATOR_L2_LOSS = False
GENERATOR_L2_LAMBDA = 1#8
GENERATOR_SSIM_LOSS = False
GENERATOR_SSIM_LAMBDA = 8

FULL_IMAGE = True
# sigma for MMD
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]

lambda_MMD = 1.0
lambda_rg = 16.0
USE_SN = False

NET_D = 'multiscale-picker'
NET_G = 'autoencoder-psp'

EXTRA_CONV = True

params = dict(
    MODE = 'wgan-mmd-ae', # Valid options are dcgan, wgan, or wgan-gp
    DIM = DIM, # This overfits substantially; you're probably better off with 64
    INPUT_DIM = INPUT_DIM, # generator input dimension (latent variable dimension)
    LAMBDA = LAMBDA, # Gradient penalty lambda hyperparameter
    CRITIC_ITERS = CRITIC_ITERS, # How many critic iterations per generator iteration
    BATCH_SIZE = BATCH_SIZE, # Batch size
    ITERS = ITERS, # How many generator iterations to train for
    KERNEL_SIZE = KERNEL_SIZE,
    GENERATOR_INSTANCE_NORM = GENERATOR_INSTANCE_NORM.__name__ if GENERATOR_INSTANCE_NORM else 'None',
#    GENERATOR_GP = GENERATOR_GP,
    ENCODER_LAYER_NORM = ENCODER_LAYER_NORM,
    LR=LR,
    ONE_SIDED=ONE_SIDED,
    CRITIC_DIM=CRITIC_DIM,
    ORTHOREG_LOSS=ORTHOREG_LOSS,
    GENERATOR_L2_LOSS = GENERATOR_L2_LOSS,
    GENERATOR_L2_LAMBDA = GENERATOR_L2_LAMBDA,
    GENERATOR_SSIM_LOSS = GENERATOR_SSIM_LOSS,
    GENERATOR_SSIM_LAMBDA = GENERATOR_SSIM_LAMBDA,
    FULL_IMAGE = FULL_IMAGE,
    NET_D = NET_D,
    USE_TENSOR_DATASET=USE_TENSOR_DATASET,
    NET_G = NET_G,
    USE_SN = USE_SN,
    EXTRA_CONV = EXTRA_CONV,
    REFLECT_PAD = True, #TODO: make this a real option
)

with open(RUN_PATH + '/algo_params.txt','w') as f:
    import json
    json.dump(params, f, indent=2)

class ONE_SIDED_ERROR(nn.Module):
    def __init__(self):
        super().__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean(0)
        return output.view(1)


def critic_schedule():
    for i in range(5):
        yield 100
    while True:
        for i in range(99):
            yield CRITIC_ITERS
        #yield 100 # 100 iters every 100 iters 

def gen_schedule():
    for i in range(10):
        yield 1
    for i in range(100):
        yield 1
    for i in range(7000):
        yield 1
    while True:
        yield 1

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Norm') != -1:
        if m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)

#netG = Generator(DIM, INPUT_DIM, IMAGE_SIZE, GENERATOR_INSTANCE_NORM)
#netG = AutoEncoder(DIM, INPUT_DIM, IMAGE_SIZE, GENERATOR_INSTANCE_NORM is not None, fc=FULL_IMAGE)
#netG = AutoEncoderUpscale(DIM, INPUT_DIM, IMAGE_SIZE, GENERATOR_INSTANCE_NORM is not None, fc=FULL_IMAGE)
netG = AutoEncoderPSP(DIM, INPUT_DIM, IMAGE_SIZE, GENERATOR_INSTANCE_NORM is not None, fc=FULL_IMAGE, extra_conv=EXTRA_CONV)
if not USE_SN:
    netD = Encoder(CRITIC_DIM, DIM, use_layer_norm=ENCODER_LAYER_NORM, fc=FULL_IMAGE)
else:
    netD = SNEncoder(CRITIC_DIM, DIM)

netG.apply(weights_init)
netD.apply(weights_init)
print(netG)
print(netD)
use_cuda = torch.cuda.is_available()
ssim_loss = SSIM()
mse_loss = torch.nn.MSELoss()
one_sided = ONE_SIDED_ERROR()
base_loss = nn.SmoothL1Loss(size_average=True)

if use_cuda:
    gpu = 0
    # makes things slower?!
    torch.backends.cudnn.benchmark = True
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)
    mse_loss = mse_loss.cuda(gpu)
    ssim_loss = ssim_loss.cuda(gpu)
    base_loss = base_loss.cuda(gpu)


#TODO: hack
# pre-processing transform
# augmentation goes here, e.g. RandomResizedCrop instead of regular random crop
if FULL_IMAGE:
    transform = torchvision.transforms.Compose([
        #torchvision.transforms.RandomCrop(IMAGE_SIZE),
        torchvision.transforms.RandomResizedCrop(320),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2) # convert pixel values from 0..1 to -1..1
        ])

else:
    transform = torchvision.transforms.Compose([
        #torchvision.transforms.RandomCrop(IMAGE_SIZE),
        torchvision.transforms.RandomResizedCrop(IMAGE_SIZE),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2) # convert pixel values from 0..1 to -1..1
        ])

# Dataset iterator
#images_dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=transform)


#train_dataset = DatasetSubset(images_dataset, 0, TRAIN_RATIO)
if USE_TENSOR_DATASET:
    images_dataset = load_batches(BATCHES_PATH)
    train_dataset = DatasetSubset(images_dataset, 0, 1)
else:
    #train_dataset = batchify(torchvision.datasets.ImageFolder(DATA_DIR, transform=transform), stop=TRAIN_RATIO)
    train_dataset = EagerFolder(DATA_DIR, transform=transform)
    #train_dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=transform)
#dev_dataset = DatasetSubset(images_dataset, TRAIN_RATIO, 1.0)
train_gen = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                                        pin_memory=use_cuda, num_workers=3)
#dev_gen = torch.utils.data.DataLoader(dev_dataset, BATCH_SIZE, shuffle=False,
#                                        pin_memory=use_cuda, num_workers=5)

#TODO: end of hack

#if use_cuda:
#    torch.set_default_tensor_type('torch.cuda.HalfTensor')

ortho_loss_g = torch.zeros(1)
ortho_loss_d = torch.zeros(1)
if use_cuda:
    ortho_loss_g = ortho_loss_g.cuda()
    ortho_loss_d = ortho_loss_d.cuda()



#for mod in [netG, netD]:
#    mod.half()

optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.9))


netG.train()
netD.train()


def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(real_data.size(0), 1)
    #alpha = alpha.expand_as(real_data.data)

    # unnecessary with broadcasting
    alpha = alpha.expand(real_data.size(0), real_data.nelement()//real_data.size(0)).contiguous().view_as(real_data.data)

    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data.data + ((1 - alpha) * fake_data.data)
    #interpolates = real_data + (alpha * real_data_grad.data * torch.norm(real_data_grad.data, p=2, dim=1).unsqueeze(1))

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    scores = netD(interpolates)

    #gradients = autograd.grad(outputs=scores, inputs=[g1, g2],
    gradients_list = autograd.grad(outputs=scores, inputs=interpolates,
                              grad_outputs=torch.ones(scores.size()).cuda(gpu) if use_cuda else torch.ones(scores.size()),
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
    
        if ONE_SIDED:
            gradient_penalty += (F.relu(gradients.norm(2, dim=1) - 1, inplace=True) ** 2).mean()
        else:
            gradient_penalty += ((gradients.norm(2, dim=1) - 1) ** 2).mean() 

    return gradient_penalty

def save_images(images_tensor, output_path):
    samples = images_tensor
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(samples, output_path)



def inf_train_gen():
    while True:
        for images, _ in train_gen:
            # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            yield images

gen = inf_train_gen()
#torch.set_default_tensor_type('torch.HalfTensor')
loss = 0
CRITIC_GEN = critic_schedule()
GEN_ITERS = gen_schedule()
for iteration in tqdm(range(ITERS)):

    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    #for i in range(CRITIC_ITERS):
    netG.eval()
    netD.train()
    for i in range(next(CRITIC_GEN)):
        _data = next(gen)
        netD.zero_grad()

        # train with real
        if USE_TENSOR_DATASET:
            real_data, real_target = _data # preprocess(_data)#torch.stack([preprocess(item) for item in _data])
        else:
            real_data, real_target = _data, _data

        if use_cuda:
            real_data = real_data.cuda(gpu, async=True)
            real_target = real_target.cuda(gpu, async=True)
        real_data_v = autograd.Variable(real_data)
        real_target_v = autograd.Variable(real_target)

        # import torchvision
        # filename = os.path.join("test_train_data", str(iteration) + str(i) + ".jpg")
        # torchvision.utils.save_image(real_data, filename)


        # train with fake
        #gen_input = netD.encoder(real_data_v)
        #fake = netG(gen_input)
        fake = netG(real_data_v)

        real_enc = netD(real_target_v)
        fake_enc = netD(fake)
        if FULL_IMAGE:
            real_enc = real_enc.view(real_enc.size(0), -1)
            fake_enc = fake_enc.view(fake_enc.size(0), -1)

        mmd2 = mmd.mix_rbf_mmd2(real_enc, fake_enc, sigma_list)
        mmd2 = F.relu(mmd2)
        
        # compute rank hinge loss
        one_side_errD = one_sided(real_enc.mean(0) - fake_enc.mean(0))
        
        
        errD = torch.sqrt(mmd2) + lambda_rg * one_side_errD
        
        Wasserstein_D = errD.data.clone()
        
        loss = -errD

        if ORTHOREG_LOSS: 
            ortho_loss_d[0] = 0
            ortho_loss_v = autograd.Variable(ortho_loss_d)
            orthoreg.orthoreg_loss(netD, ortho_loss_v)
            loss += ortho_loss_v

        # train with gradient penalty
        if not USE_SN:
            gradient_penalty = calc_gradient_penalty(netD, real_target_v, fake)
            #gradient_penalty = autograd.Variable(torch.cuda.FloatTensor(1).fill_(0))
            #torch.nn.utils.clip_grad_norm(netD.parameters(), 2, 1)
            loss += gradient_penalty * LAMBDA

        loss.backward()
        optimizerD.step()

    D_cost = loss.data
    ############################
    # (2) Update G network
    ###########################
    netG.train()
    netD.eval()
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    for i in range(next(GEN_ITERS)):
        netG.zero_grad()

        _data = next(gen)

        # train with real
        if USE_TENSOR_DATASET:
            real_data, real_target = _data # preprocess(_data)#torch.stack([preprocess(item) for item in _data])
        else:
            real_data, real_target = _data, _data

        if use_cuda:
            real_data = real_data.cuda(gpu, async=True)
            real_target = real_target.cuda(gpu, async=True)
        real_data_v = autograd.Variable(real_data)
        real_target_v = autograd.Variable(real_target)

        #gen_input = netD.encoder(real_data_v)
        #fake = netG(gen_input)
        fake = netG(real_data_v)

        real_enc = netD(real_target_v)
        fake_enc = netD(fake)
        if FULL_IMAGE:
            real_enc = real_enc.view(real_enc.size(0), -1)
            fake_enc = fake_enc.view(fake_enc.size(0), -1)


        mmd2 = mmd.mix_rbf_mmd2(real_enc, fake_enc, sigma_list)
        mmd2 = F.relu(mmd2)
        
        # compute rank hinge loss
        one_side_errG = one_sided(real_enc.mean(0) - fake_enc.mean(0))
        
        errG = torch.sqrt(mmd2) + lambda_rg * one_side_errG

        G_cost = errG.data.clone()

        loss = errG

        if ORTHOREG_LOSS:
            ortho_loss_g[0] = 0
            ortho_loss_v = autograd.Variable(ortho_loss_g)
            orthoreg.orthoreg_loss(netG, ortho_loss_v)
            loss += ortho_loss_v

        if GENERATOR_L2_LOSS:
            l2_loss = mse_loss(fake, real_target_v)
            loss += l2_loss*GENERATOR_L2_LAMBDA

        if GENERATOR_SSIM_LOSS:
            ssim_penalty = ssim_loss(fake*0.5 + 0.5, real_target_v *0.5 + 0.5)
            loss += ssim_penalty*GENERATOR_SSIM_LAMBDA

        # no GP
        loss.backward()

        optimizerG.step()

    # Write logs and save samples
    lib.plot.plot(RUN_PATH + 'train disc cost', D_cost.cpu().numpy())
    lib.plot.plot(RUN_PATH + 'time', time.time() - start_time)
    lib.plot.plot(RUN_PATH + 'train gen cost', G_cost.cpu().numpy())
    lib.plot.plot(RUN_PATH + 'wasserstein distance', Wasserstein_D.cpu().numpy())
    if ORTHOREG_LOSS:
        lib.plot.plot(RUN_PATH + 'ortho loss G', ortho_loss_g.cpu().numpy())
        lib.plot.plot(RUN_PATH + 'ortho loss D', ortho_loss_d.cpu().numpy())


    # Calculate dev loss and generate samples every 100 iters
    if iteration % 100 == 99:
        #dev_disc_costs = []
        #netD.eval()
        #for images, _ in dev_gen:
        #    images = images.view((-1, 3, 128, 128))
        #    imgs = images#preprocess(images)
        #
        #    #imgs = preprocess(images)
        #    if use_cuda:
        #        imgs = imgs.cuda(gpu)
        #    imgs_v = autograd.Variable(imgs, volatile=True)
        #
        #    D, encoded = netD(imgs_v)
        #    _dev_disc_cost = -D.mean().cpu().data.numpy()
        #    dev_disc_costs.append(_dev_disc_cost)
        #netD.train()
        #lib.plot.plot(RUN_PATH + 'dev disc cost', np.mean(dev_disc_costs))

        #fixed_noise_128 = torch.randn(128, INPUT_DIM)
        #if use_cuda:
        #    fixed_noise_128 = fixed_noise_128.cuda(gpu)
        #generate_image(iteration, netG, fixed_noise_128)
        #generate_image("{}_reconstruct".format(iteration), netG, encoded.data, True)
        save_images(real_data_v, RUN_PATH + 'samples_{}_original.jpg'.format(iteration))
        save_images(fake, RUN_PATH + 'samples_{}_reconstruct.jpg'.format(iteration))
        #print(encoded)
        #print(fixed_noise_128)

    # Save logs every 200 iters
    if (iteration < 5) or (iteration % 100 == 99):
        lib.plot.flush()
    lib.plot.tick()

    if iteration % 1000 == 999:
        state_dict = {
                    'iters': iteration + 1,
                    'algo_params': params,
                    'gen_state_dict': netG.state_dict(),
                    'critic_state_dict': netD.state_dict(),
                    'optimizerG' : optimizerG.state_dict(),
                    'optimizerD' : optimizerD.state_dict(),
                }

        torch.save(state_dict, RUN_PATH + 'state_{}.pth.tar'.format(iteration+1))