import os, sys, time
sys.path.append(os.getcwd())

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
import torch.nn.functional as F


import time
import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.cifar10
import tflib.plot
#import tflib.inception_score

import numpy as np
from tqdm import tqdm

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = '/mnt/7FC1A7CD7234342C/cifar-10-batches-py/'
OUTPUT_BASE_DIR = '/mnt/7FC1A7CD7234342C/cifar10-results/'
RUN_PATH = '{}{}/'.format(OUTPUT_BASE_DIR, time.strftime('%Y_%m_%d_%H_%M_%S'))  #TODO: generate by settings
if not os.path.exists(RUN_PATH):
    os.mkdir(RUN_PATH)
#TODO:hack
tflib.plot.log_dir = RUN_PATH

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

DIM = 64 # This overfits substantially; you're probably better off with 64
CRITIC_DIM = 64 # ambition
INPUT_DIM = 128 # generator input dimension (latent variable dimension)
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 100000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

KERNEL_SIZE = 4

CONSTANCY_LOSS = False
CONSTANCY_LAMBDA = 8

LR = 1e-4

GENERATOR_INSTANCE_NORM = nn.BatchNorm2d

ENCODER_INSTANCE_NORM = False # TODO

DISCRIMINATOR_RECONSTRUCTION_LOSS = False
DISCRIMINATOR_RECONSTRUCTION_LAMBDA = 8

GENERATOR_AUTOENCODER_LOSS = False
GENERATOR_AUTOENCODER_LAMBDA = 1
GENERATOR_SCORE_LOSS = False
GENERATOR_SCORE_LAMBDA = 8

AUTOENCODER_GP = False
ONE_SIDED = False
params = dict(
    MODE = 'mcme', # Valid options are dcgan, wgan, or wgan-gp
    DIM = DIM, # This overfits substantially; you're probably better off with 64
    INPUT_DIM = INPUT_DIM, # generator input dimension (latent variable dimension)
    LAMBDA = LAMBDA, # Gradient penalty lambda hyperparameter
    CRITIC_ITERS = CRITIC_ITERS, # How many critic iterations per generator iteration
    BATCH_SIZE = BATCH_SIZE, # Batch size
    ITERS = ITERS, # How many generator iterations to train for
    OUTPUT_DIM = OUTPUT_DIM, # Number of pixels in CIFAR10 (3*32*32)
    KERNEL_SIZE = KERNEL_SIZE,
    GENERATOR_INSTANCE_NORM = GENERATOR_INSTANCE_NORM.__name__,

    ENCODER_INSTANCE_NORM = ENCODER_INSTANCE_NORM,
    DISCRIMINATOR_RECONSTRUCTION_LOSS = DISCRIMINATOR_RECONSTRUCTION_LOSS,
    LR=LR,
    AUTOENCODER_GP = AUTOENCODER_GP,
    ONE_SIDED=ONE_SIDED,
    CONSTANCY_LOSS = CONSTANCY_LOSS,
    CONSTANCY_LAMBDA = CONSTANCY_LAMBDA,
    GENERATOR_SCORE_LOSS = GENERATOR_SCORE_LOSS,
    GENERATOR_SCORE_LAMBDA = GENERATOR_SCORE_LAMBDA,
    GENERATOR_AUTOENCODER_LOSS = GENERATOR_AUTOENCODER_LOSS,
    GENERATOR_AUTOENCODER_LAMBDA = GENERATOR_AUTOENCODER_LAMBDA,
    CRITIC_DIM=CRITIC_DIM,
)

with open(RUN_PATH + '/algo_params.txt','w') as f:
    import json
    json.dump(params, f, indent=2)


def _upscale_resize(in_dim, out_dim, kernel_size):
    return nn.Sequential(
        nn.InstanceNorm2d(in_dim, affine=True),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1,2,1,2)),
        nn.Conv2d(in_dim, out_dim, kernel_size, bias=False)
    )

def _upblock(in_dim, out_dim, kernel_size, padding, norm=nn.InstanceNorm2d, non_linearity=lambda: nn.ReLU(True)):
    blocks = []
    bias_conv = not norm # if no norm them add bias parameter
    if norm is not None:
        blocks.append(norm(in_dim))
    blocks.append(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=2, padding=padding, bias=bias_conv))
    blocks.append(non_linearity())
    return nn.Sequential(*blocks)


class Generator(nn.Module):
    def __init__(self, norm=GENERATOR_INSTANCE_NORM):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            #nn.InstanceNorm2d(4 * 4 * 4 * DIM),
            nn.Linear(INPUT_DIM, 4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )
        non_linearity = nn.ReLU

        #block1 = _upscale_resize(4 * DIM, 2 * DIM, KERNEL_SIZE)
        #block2 = _upscale_resize(2 * DIM, DIM, KERNEL_SIZE)
        #self.last_norm  = nn.InstanceNorm2d(DIM, affine=True)
        #deconv_out = nn.ConvTranspose2d(DIM, 3, KERNEL_SIZE, stride=2, padding=1, bias=False)
        #self.out_norm = nn.InstanceNorm2d(3, affine=True)
        
        self.preprocess = preprocess
        self.block1 = _upblock(4 * DIM, 2 * DIM, KERNEL_SIZE, 1, norm=norm, non_linearity=non_linearity)
        self.block2 = _upblock(2 * DIM, DIM, KERNEL_SIZE, 1, norm=norm, non_linearity=non_linearity)
        self.block_out = _upblock(DIM, 3, KERNEL_SIZE, 1, norm=norm, non_linearity=nn.Tanh)
        #self.deconv_out = deconv_out
        #self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        #print(output.size())
        output = self.block1(output)
        #print(output.size())
        output = self.block2(output)
        #print(output.size())
        output = self.block_out(output)
        #output = self.deconv_out(self.last_norm(output))
        #output = self.deconv_out(output)
        #output = self.tanh(output)
        #output = self.out_norm(output)
        return output.view(-1, 3, 32, 32)

class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if ENCODER_INSTANCE_NORM:
            main = nn.Sequential(
                nn.Conv2d(3, dim, KERNEL_SIZE, 2, padding=1, bias=False),
                nn.InstanceNorm2d(dim),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(dim, 2 * dim, KERNEL_SIZE, 2, padding=1, bias=False),
                nn.InstanceNorm2d(2 * dim),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(2 * dim, 4 * dim, KERNEL_SIZE, 2, padding=1, bias=False),
                nn.InstanceNorm2d(4 * dim),
                nn.LeakyReLU(0.2, True),
            )
        else:
            main = nn.Sequential(
                nn.Conv2d(3, dim, KERNEL_SIZE, 2, padding=1, bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(dim, 2 * dim, KERNEL_SIZE, 2, padding=1, bias=True),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(2 * dim, 4 * dim, KERNEL_SIZE, 2, padding=1, bias=True),
                nn.LeakyReLU(0.2, True),
            )
        self.dim = dim
        self.main = main
        self.linear = nn.Linear(4*4*4*dim, INPUT_DIM)

    def forward(self, input):
        output = self.main(input)
        before_linear = output.view(-1, 4 * 4 * 4 * self.dim)
        output = self.linear(before_linear)

        return output


def my_loss(net_real, independent_encoded):
    return torch.norm(net_real - independent_encoded, p=2, dim=-1) + \
           torch.abs(torch.bmm(net_real.unsqueeze(1), independent_encoded.unsqueeze(2)).squeeze())

def critic_schedule():
    for i in range(10):
        yield 100
    while True:
        yield CRITIC_ITERS

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

def print_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        print(m.weight)
        if m.bias is not None:
            print(m.bias)

def print_grads(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        print(m.weight.grad)
        if m.bias is not None:
            print(m.bias.grad)


netG = Generator()
netD = Encoder(CRITIC_DIM)
netG.apply(weights_init)
netD.apply(weights_init)
print(netG)
print(netD)
use_cuda = torch.cuda.is_available()
mse_loss = torch.nn.MSELoss()
if use_cuda:
    gpu = 0
    # makes things slower?!
    torch.backends.cudnn.benchmark = True
if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)
    mse_loss = mse_loss.cuda(gpu)
one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.9))

netG.train()
netD.train()

def calc_gradient_penalty(netD, netG, real_data, fake_data, encoded):
    if AUTOENCODER_GP:
        fake_data = netG(encoded) #TODO:investigate
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()//BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data.data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    # TODO: clashes with autoencoder_gp?
    disc_interpolates = cramer_loss(netD(interpolates), encoded)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)

    if ONE_SIDED:
        gradient_penalty = (F.relu(gradients.norm(2, dim=1) - 1, inplace=True) ** 2).mean() * LAMBDA
    else:
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty

# For generating samples
def generate_image(frame, netG, input):
    noisev = autograd.Variable(input, volatile=True)
    netG.eval()
    samples = netG(noisev)
    netG.train()
    save_images(samples, RUN_PATH + 'samples_{}.jpg'.format(frame))


def save_images(images_tensor, output_path):
    samples = images_tensor.view(-1, 3, 32, 32)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(samples, output_path)

# For calculating inception score
def get_inception_score(G, ):
    all_samples = []
    for i in xrange(10):
        samples_100 = torch.randn(100, INPUT_DIM)
        if use_cuda:
            samples_100 = samples_100.cuda(gpu)
        samples_100 = autograd.Variable(samples_100, volatile=True)
        all_samples.append(G(samples_100).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples))

# Dataset iterator
train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, data_dir=DATA_DIR, cuda=use_cuda)
def inf_train_gen():
    while True:
        for images in train_gen():
            # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            yield images


gen = inf_train_gen()
#preprocess = torchvision.transforms.Compose([
#                               torchvision.transforms.ToTensor(),
#                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                           ])
preprocess = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

CRITIC_GEN = critic_schedule()
GEN_ITERS = gen_schedule()


noise = torch.randn(BATCH_SIZE, INPUT_DIM)
noise_independent = torch.randn(BATCH_SIZE, INPUT_DIM)
if use_cuda:
    noise = noise.cuda(gpu)
    noise_independent = noise_independent.cuda(gpu)



for iteration in tqdm(range(ITERS)):

    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for p in netG.parameters():  # reset requires_grad
        p.requires_grad = False  # they are set to False below in netG update
    #for i in range(CRITIC_ITERS):
    netG.eval()
    netD.train()
    for i in range(next(CRITIC_GEN)):
        _data = next(gen)
        netD.zero_grad()

        noise.normal_(0, 1)
        noise_independent.normal_(0, 1)
        noisev = autograd.Variable(noise, volatile=True)
        noisev_independent = autograd.Variable(noise_independent, volatile=True)

        # Generate two independent fake batches
        fake = autograd.Variable(netG(noisev).data)
        fake_independent = autograd.Variable(netG(noisev_independent).data)

        # train with real
        _data = _data.view((BATCH_SIZE, 3, 32, 32))
        real_data = _data # preprocess(_data)#torch.stack([preprocess(item) for item in _data])

        #if use_cuda:
        #    real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        # import torchvision
        # filename = os.path.join("test_train_data", str(iteration) + str(i) + ".jpg")
        # torchvision.utils.save_image(real_data, filename)
        encoded_independent = netD(fake_independent)

        encoded_real = netD(real_data_v)
        D_real = cramer_loss(encoded_real, encoded_independent)

        encoded_fake = netD(fake)
        D_fake = cramer_loss(encoded_fake, encoded_independent)

        #print(D_real, D_fake)
        loss = (D_fake - D_real).mean()
        #netD.apply(print_weights)
        #print(fake)
        if CONSTANCY_LOSS:
            c_loss = CONSTANCY_LAMBDA * mse_loss(encoded_fake, autograd.Variable(noise))
            loss += c_loss
        
        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, netG, real_data_v.data, fake, encoded_real)
        loss += gradient_penalty
        loss.backward()

        # print "gradien_penalty: ", gradient_penalty

        D_cost = loss.data
        # TODO: D_cost = loss.data[0]
        Wasserstein_D = (D_real - D_fake).data.mean()
        optimizerD.step()

    ############################
    # (2) Update G network
    ###########################
    netG.train()
    #netD.eval() # screws up cuda?
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    for p in netG.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for i in range(next(GEN_ITERS)):
        netG.zero_grad()

        _data = next(gen)
        real = autograd.Variable(_data.view((BATCH_SIZE, 3, 32, 32)))
        #if use_cuda:
        #    real = real.cuda()

        noise.normal_(0, 1)
        noise_independent.normal_(0, 1)
        noisev1 = autograd.Variable(noise)
        noisev2 = autograd.Variable(noise_independent)
        fake1 = netG(noisev1)
        fake2 = netG(noisev2)
        real_encoded = netD(real)
        fake1_encoded = netD(fake1)
        fake2_encoded = netD(fake2)
        G = (torch.norm(real_encoded - fake1_encoded, p=2, dim=-1) +
             torch.norm(real_encoded - fake2_encoded, p=2, dim=-1) -
             torch.norm(fake1_encoded - fake2_encoded, p=2, dim=-1)).mean()

        if GENERATOR_SCORE_LOSS or GENERATOR_AUTOENCODER_LOSS:
            real_data_v = autograd.Variable(next(gen).view((BATCH_SIZE, 3, 32, 32)), volatile=True)
            #if use_cuda:
            #    real_data_v = real_data_v.cuda()

            real_latent = netD(real_data_v)
            real_latent = autograd.Variable(real_latent.data)
            reconstructed = netG(autograd.Variable(real_latent.data))
            if GENERATOR_AUTOENCODER_LOSS:
                gen_ae_loss = mse_loss(reconstructed, real_data_v)
                G += GENERATOR_AUTOENCODER_LAMBDA * gen_ae_loss
            if GENERATOR_SCORE_LOSS:
                gen_rec_loss = ((real_latent - netD(reconstructed))**2).mean()
                G += GENERATOR_SCORE_LAMBDA * gen_rec_loss

        G.backward()
        G_cost = G.data
        optimizerG.step()

    # Write logs and save samples
    lib.plot.plot(RUN_PATH + 'train disc cost', D_cost.cpu().numpy())
    lib.plot.plot(RUN_PATH + 'time', time.time() - start_time)
    lib.plot.plot(RUN_PATH + 'train gen cost', G_cost.cpu().numpy())
    lib.plot.plot(RUN_PATH + 'wasserstein distance', Wasserstein_D)

    # Calculate inception score every 1K iters
    if False and iteration % 1000 == 999:
        inception_score = get_inception_score(netG)
        lib.plot.plot(RUN_PATH + 'inception score', inception_score[0])

    # Calculate dev loss and generate samples every 200 iters
    if iteration % 200 == 199:
        dev_disc_costs = []
        #TODO:
        netD.eval()
        for images in dev_gen():
            images = images.view((BATCH_SIZE, 3, 32, 32))
            imgs = images#preprocess(images)

            #imgs = preprocess(images)
            #if use_cuda:
            #    imgs = imgs.cuda(gpu)
            imgs_v = autograd.Variable(imgs, volatile=True)

            D = netD(imgs_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)
        netD.train()
        lib.plot.plot(RUN_PATH + 'dev disc cost', np.mean(dev_disc_costs))
        

        fixed_noise_128 = torch.randn(128, INPUT_DIM)
        if use_cuda:
            fixed_noise_128 = fixed_noise_128.cuda(gpu)
        generate_image(iteration, netG, fixed_noise_128)
        generate_image("{}_reconstruct".format(iteration), netG, D.data)
        save_images(imgs_v, RUN_PATH + 'samples_{}_original.jpg'.format(iteration))
        #print(encoded)
        #print(fixed_noise_128)

    # Save logs every 200 iters
    if (iteration < 5) or (iteration % 100 == 99):
        lib.plot.flush()
    lib.plot.tick()

state_dict = {
            'iters': iteration + 1,
            'algo_params': params,
            'gen_state_dict': netG.state_dict(),
            'critic_state_dict': netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
        }

torch.save(state_dict, RUN_PATH + 'final.pth.tar')