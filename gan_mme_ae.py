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
#TODO: get rid of crappy tflib
import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.cifar10
import tflib.plot
#import tflib.inception_score

import numpy as np
from tqdm import tqdm

from models.layers import LayerNorm
from models.main import Generator, Encoder, AutoEncoder, AutoEncoderUpscale, AutoEncoderPSP, SNEncoder
from losses import gradient_penalty_loss

from utils import DatasetSubset, mix_samples, _mix_samples, load_batches, batchify, EagerFolder, save_images
import orthoreg
from pytorch_ssim import SSIM
import mmd

import argparse
import pathlib

class ONE_SIDED_ERROR(nn.Module):
    def __init__(self):
        super().__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean(0)
        return output.view(1)


def critic_schedule(args):
    for i in range(args.critic_initial_count):
        yield args.critic_initial_iters
    while True:
        for i in range(99):
            yield args.critic_iters
        #yield 100 # 100 iters every 100 iters 

def gen_schedule():
    while True:
        yield 1

def inf_train_gen(generator):
    while True:
        for images, _ in generator:
            yield images

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


def main():
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("-o", "--output-base-dir", default="/mnt/7FC1A7CD7234342C/compression-results/")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-5)
    parser.add_argument("--use-tensor-dataset", action='store_true', help="use processed tensors dataset")
    parser.add_argument("--data-dir", default="/mnt/7FC1A7CD7234342C/compression/dataset", help="path to image dataset")
    parser.add_argument("--batches-path", default="/mnt/7FC1A7CD7234342C/compression-results/dataset_bs_128_size_64_half_2k/", help="path to tensor batches dataset")
    parser.add_argument("--dim", type=int, default=64, help="base dimension for generator")
    parser.add_argument("--critic-dim", type=int, default=32, help="base dimension for critic")
    parser.add_argument("--latent-dim", type=int, default=256, help="latent dimension for autoencoder")
    parser.add_argument("--one-sided", action='store_true', help="use one-sided gradient penalty")
    parser.add_argument("--gp-lambda", type=float, default=10, help="gradient penalty hyper-parameter")
    parser.add_argument("--critic-iters", type=int, default=5, help="number of critic iters per gen iter")
    parser.add_argument("--critic-initial-iters", type=int, default=100, help="number of critic iters per gen iter in initial phase")
    parser.add_argument("--critic-initial-count", type=int, default=5, help="number of critic iters per gen iter")
    parser.add_argument("--batch-size", type=int, default=24, help="batch size. Bigger is better, limit is RAM")
    parser.add_argument("--iterations", type=int, default=100000, help="generator iterations")
    parser.add_argument("--image-size", type=int, default=64, help="image size (one side, default 64)")
    parser.add_argument("--no-full-image", dest="full_image", action="store_false", help="don't use large image")
    parser.add_argument("--full-image-size", type=int, default=320, help="large image size - fc network (one side, default 320)")
    parser.add_argument("--no-encoder-layer-norm", dest="encoder_layer_norm", action="store_false", help="don't use layer norm in critic")
    parser.add_argument("--no-generator-layer-norm", dest="generator_layer_norm", action="store_false", help="don't use layer norm in generator")
    parser.add_argument("--orthoreg-loss", action="store_true", help="use orthoreg")
    parser.add_argument("--no-extra-conv", action="store_false", dest="extra_conv", help="extra convolution layer before output")
    parser.add_argument("--generator-l2-loss", action="store_true")
    parser.add_argument("--generator-l2-lambda", type=float, default=1.)
    parser.add_argument("--generator-ssim-loss", action="store_true")
    parser.add_argument("--generator-ssim-lambda", type=float, default=8)
    parser.add_argument("--use-sn", action="store_true")

    args = parser.parse_args()

    RUN_PATH = pathlib.Path(args.output_base_dir) / time.strftime('%Y_%m_%d_%H_%M_%S')  #TODO: generate by settings
    RUN_PATH.mkdir()
    #TODO:hack
    tflib.plot.log_dir = str(RUN_PATH)

    #TRAIN_RATIO=0.05#0.9

    # sigma for MMD
    base = 1.0
    sigma_list = [1, 2, 4, 8, 16]
    sigma_list = [sigma / base for sigma in sigma_list]

    lambda_MMD = 1.0
    lambda_rg = 16.0

    NET_D = 'multiscale-picker'
    NET_G = 'autoencoder-psp'


    with (RUN_PATH / 'algo_params.txt').open('w') as f:
        import json
        json.dump(vars(args), f, indent=2)

    #netG = Generator(DIM, args.latent_dim, args.image_size, GENERATOR_INSTANCE_NORM)
    #netG = AutoEncoder(DIM, args.latent_dim, args.image_size, GENERATOR_INSTANCE_NORM is not None, fc=FULL_IMAGE)
    #netG = AutoEncoderUpscale(DIM, args.latent_dim, args.image_size, GENERATOR_INSTANCE_NORM is not None, fc=FULL_IMAGE)
    netG = AutoEncoderPSP(args.dim, args.latent_dim, args.image_size, args.generator_layer_norm, fc=args.full_image, extra_conv=args.extra_conv)
    if not args.use_sn:
        netD = Encoder(args.critic_dim, args.dim, use_layer_norm=args.encoder_layer_norm, fc=args.full_image)
    else:
        netD = SNEncoder(args.critic_dim, args.dim)

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
    if args.full_image:
        transform = torchvision.transforms.Compose([
            #torchvision.transforms.RandomCrop(args.image_size),
            torchvision.transforms.RandomResizedCrop(args.full_image_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2) # convert pixel values from 0..1 to -1..1
            ])

    else:
        transform = torchvision.transforms.Compose([
            #torchvision.transforms.RandomCrop(args.image_size),
            torchvision.transforms.RandomResizedCrop(args.image_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2) # convert pixel values from 0..1 to -1..1
            ])

    # Dataset iterator
    #images_dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)


    #train_dataset = DatasetSubset(images_dataset, 0, TRAIN_RATIO)
    if args.use_tensor_dataset:
        images_dataset = load_batches(args.batches_path)
        train_dataset = DatasetSubset(images_dataset, 0, 1)
    else:
        #train_dataset = batchify(torchvision.datasets.ImageFolder(args.data_dir, transform=transform), stop=TRAIN_RATIO)
        train_dataset = EagerFolder(args.data_dir, transform=transform)
        #train_dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)
    #dev_dataset = DatasetSubset(images_dataset, TRAIN_RATIO, 1.0)
    train_gen = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True,
                                            pin_memory=use_cuda, num_workers=3)
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

    optimizerD = optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))


    netG.train()
    netD.train()


    gen = inf_train_gen(train_gen)
    #torch.set_default_tensor_type('torch.HalfTensor')
    loss = 0
    CRITIC_GEN = critic_schedule(args)
    GEN_ITERS = gen_schedule()
    for iteration in tqdm(range(args.iterations)):

        start_time = time.time()
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        #for i in range(args.critic_iters):
        netG.eval()
        netD.train()
        for i in range(next(CRITIC_GEN)):
            _data = next(gen)
            netD.zero_grad()

            # train with real
            if args.use_tensor_dataset:
                real_data, real_target = _data # preprocess(_data)#torch.stack([preprocess(item) for item in _data])
            else:
                real_data, real_target = _data, _data

            if use_cuda:
                real_data = real_data.cuda(gpu, async=True)
                real_target = real_target.cuda(gpu, async=True)
            real_data_v = autograd.Variable(real_data)
            real_target_v = autograd.Variable(real_target)

            # train with fake
            #gen_input = netD.encoder(real_data_v)
            #fake = netG(gen_input)
            fake = netG(real_data_v)

            real_enc = netD(real_target_v)
            fake_enc = netD(fake)
            if args.full_image:
                real_enc = real_enc.view(real_enc.size(0), -1)
                fake_enc = fake_enc.view(fake_enc.size(0), -1)

            mmd2 = mmd.mix_rbf_mmd2(real_enc, fake_enc, sigma_list)
            mmd2 = F.relu(mmd2)
            
            # compute rank hinge loss
            one_side_errD = one_sided(real_enc.mean(0) - fake_enc.mean(0))
            
            
            errD = torch.sqrt(mmd2) + lambda_rg * one_side_errD
            
            Wasserstein_D = errD.data.clone()
            
            loss = -errD

            if args.orthoreg_loss: 
                ortho_loss_d[0] = 0
                ortho_loss_v = autograd.Variable(ortho_loss_d)
                orthoreg.orthoreg_loss(netD, ortho_loss_v)
                loss += ortho_loss_v

            # train with gradient penalty
            if not args.use_sn:
                gradient_penalty = gradient_penalty_loss(netD, real_target_v, fake, args.one_sided)
                #gradient_penalty = autograd.Variable(torch.cuda.FloatTensor(1).fill_(0))
                #torch.nn.utils.clip_grad_norm(netD.parameters(), 2, 1)
                loss += gradient_penalty * args.gp_lambda

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
            if args.use_tensor_dataset:
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
            if args.full_image:
                real_enc = real_enc.view(real_enc.size(0), -1)
                fake_enc = fake_enc.view(fake_enc.size(0), -1)


            mmd2 = mmd.mix_rbf_mmd2(real_enc, fake_enc, sigma_list)
            mmd2 = F.relu(mmd2)
            
            # compute rank hinge loss
            one_side_errG = one_sided(real_enc.mean(0) - fake_enc.mean(0))
            
            errG = torch.sqrt(mmd2) + lambda_rg * one_side_errG

            G_cost = errG.data.clone()

            loss = errG

            if args.orthoreg_loss:
                ortho_loss_g[0] = 0
                ortho_loss_v = autograd.Variable(ortho_loss_g)
                orthoreg.orthoreg_loss(netG, ortho_loss_v)
                loss += ortho_loss_v

            if args.generator_l2_loss:
                l2_loss = mse_loss(fake, real_target_v)
                loss += l2_loss*args.generator_l2_lambda

            if args.generator_ssim_loss:
                ssim_penalty = ssim_loss(fake*0.5 + 0.5, real_target_v *0.5 + 0.5)
                loss += ssim_penalty*args.generator_ssim_lambda

            # no GP
            loss.backward()

            optimizerG.step()

        # Write logs and save samples
        lib.plot.plot(str(RUN_PATH / 'train disc cost'), D_cost.cpu().numpy())
        lib.plot.plot(str(RUN_PATH / 'time'), time.time() - start_time)
        lib.plot.plot(str(RUN_PATH / 'train gen cost'), G_cost.cpu().numpy())
        lib.plot.plot(str(RUN_PATH / 'wasserstein distance'), Wasserstein_D.cpu().numpy())
        if args.orthoreg_loss:
            lib.plot.plot(str(RUN_PATH / 'ortho loss G'), ortho_loss_g.cpu().numpy())
            lib.plot.plot(str(RUN_PATH / 'ortho loss D'), ortho_loss_d.cpu().numpy())


        # TODO: argument
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
            save_images(real_data_v, str(RUN_PATH / 'samples_{}_original.jpg'.format(iteration)))
            save_images(fake, str(RUN_PATH / 'samples_{}_reconstruct.jpg'.format(iteration)))
            #print(encoded)
            #print(fixed_noise_128)

        # TODO: argument
        # Save logs every 200 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()
        lib.plot.tick()

        # TODO: argument
        if iteration % 1000 == 999:
            state_dict = {
                        'iters': iteration + 1,
                        'algo_params': vars(args),
                        'gen_state_dict': netG.state_dict(),
                        'critic_state_dict': netD.state_dict(),
                        'optimizerG' : optimizerG.state_dict(),
                        'optimizerD' : optimizerD.state_dict(),
                    }

            torch.save(state_dict, str(RUN_PATH / 'state_{}.pth.tar'.format(iteration+1)))

if __name__ == "__main__":
    main()