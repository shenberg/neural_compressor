import os, sys, time
sys.path.append(os.getcwd())
import functools
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

from models.autoencoder import ContextModel, ContextModelMultilayered, SoftToHardNdEncoder, \
                                ResNetEncoder, ResNetDecoder, ResNetMultiscaleEncoder

from utils import EagerFolder, save_images
#import orthoreg
from pytorch_ssim import MS_SSIM

import argparse
import pathlib


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



#decoded, real_data_v = 0, 0
def main():
    #global decoded, real_data_v
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("-o", "--output-base-dir", default="/mnt/7FC1A7CD7234342C/compression-results/")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("--data-dir", default="/mnt/7FC1A7CD7234342C/compression/dataset", help="path to image dataset")
    #TODO: best setting: 16
    parser.add_argument("--latent-dim", type=int, default=128, help="latent dimension for autoencoder")
    parser.add_argument("--num-centers", type=int, default=8, help="number of centers for quantization")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size. Bigger is better, limit is RAM")
    parser.add_argument("--iterations", type=int, default=240000, help="generator iterations")
    parser.add_argument("--lr-decay-iters", type=int, default=80000, help="time till decay")
    parser.add_argument("--image-size", type=int, default=192, help="image size (one side, default 64)")
    parser.add_argument("--context-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--imagenet",default="/media/shenberg/ssd_large/imagenet")
    parser.add_argument("--no-psp",dest='psp',action='store_false')
    parser.add_argument("--vq-dim", type=int, default=8)
    parser.add_argument("--use-layered-context",action="store_true")
    parser.add_argument("--use-multiscale-resnet",action="store_true")
    parser.add_argument("--coding-loss-beta", type=float, default=0.2, help="constant multiplier for entropy loss")

    args = parser.parse_args()

    RUN_PATH = pathlib.Path(args.output_base_dir) / time.strftime('%Y_%m_%d_%H_%M_%S')  #TODO: generate by settings
    RUN_PATH.mkdir()
    #TODO:hack
    tflib.plot.log_dir = str(RUN_PATH) + '/'

    with (RUN_PATH / 'algo_params.txt').open('w') as f:
        import json
        json.dump(vars(args), f, indent=2)

    if not args.use_multiscale_resnet:
        encoder = ResNetEncoder(latent_dim=args.latent_dim)
    else:
        encoder = ResNetMultiscaleEncoder(latent_dim=args.latent_dim)

    decoder = ResNetDecoder(latent_dim=args.latent_dim)
    # TODO: toggle
    #quantizer = SoftToHardEncoder(num_codes=args.num_centers, latent_dim=args.latent_dim)
    quantizer = SoftToHardNdEncoder(num_codes=args.num_centers, 
                                    latent_dim=args.latent_dim // args.vq_dim, 
                                    channel_dim = args.vq_dim)
    if not args.use_layered_context:
        context_model = ContextModel(args.num_centers, args.vq_dim)
    else:
        context_model = ContextModelMultilayered(args.num_centers, args.vq_dim, args.latent_dim // args.vq_dim)

    decoder.apply(weights_init)
    encoder.apply(weights_init)
    #print(decoder)
    #print(encoder)
    #print(context_model)
    use_cuda = torch.cuda.is_available()
    ce_loss = torch.nn.CrossEntropyLoss()
    #ssim_loss = MS_SSIM(mode='sum')
    ssim_loss = MS_SSIM(mode='product')
    # TODO: proper MS-SSIM
    mse_loss = torch.nn.MSELoss()
    if use_cuda:
        gpu = 0
        # makes things slower?!
        torch.backends.cudnn.benchmark = True
        encoder = encoder.cuda(gpu)
        decoder = decoder.cuda(gpu)
        quantizer = quantizer.cuda(gpu)
        context_model = context_model.cuda(gpu)
        mse_loss = mse_loss.cuda(gpu)
        ssim_loss = ssim_loss.cuda(gpu)
        ce_loss = ce_loss.cuda(gpu)

    beta = args.coding_loss_beta #*(latent_dim // #normalize beta according to bpp
    # pre-processing transform
    # augmentation goes here, e.g. RandomResizedCrop instead of regular random crop
    # NOTE: pad_if_needed added manually since it was only added to torchvision on 6/4/18
    #       and is not yet released
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(args.image_size, pad_if_needed=True),
        #torchvision.transforms.RandomResizedCrop(args.image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2) # convert pixel values from 0..1 to -1..1
        ])


    if args.imagenet:
        print("loading imagenet")
        train_dataset = torchvision.datasets.ImageFolder(args.imagenet, transform=transform)
        print("loaded")
    else:
        train_dataset = EagerFolder(args.data_dir, transform=transform)
    train_gen = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True,
                                            pin_memory=use_cuda, num_workers=3)
    #if use_cuda:
    #    torch.set_default_tensor_type('torch.cuda.HalfTensor')


    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters())
                         + list(quantizer.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_context = optim.Adam(list(context_model.parameters()),
                                    lr=args.context_learning_rate, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters)
    scheduler_context = optim.lr_scheduler.StepLR(optimizer_context, step_size=args.lr_decay_iters)

    decoder.train()
    encoder.train()
    quantizer.train()
    context_model.eval()

    gen = inf_train_gen(train_gen)
    #torch.set_default_tensor_type('torch.HalfTensor')
    loss = 0
    for iteration in tqdm(range(args.iterations)):

        start_time = time.time()

        real_data = next(gen)
        if use_cuda:
            real_data = real_data.cuda(gpu, async=True)
        real_data_v = autograd.Variable(real_data)
        encoder.zero_grad()
        decoder.zero_grad()
        quantizer.zero_grad()
        context_model.zero_grad()

        encoded = encoder(real_data_v)

        soft, hard, symbols = quantizer(encoded)

        quantized = (hard - soft).detach() + soft # use soft symbol for backprop
        #TODO: move into quantizer
        quantized = quantized.permute(0,3,1,2).contiguous()
        #quantized = quantized.permute(0,3,1,2).contiguous()

        #symbols = symbols.permute(0,3,1,2).contiguous()

        predictions = context_model(quantized)
        #print(symbols.size(), predictions.size())

        #coding_loss = ce_loss(predictions.view(-1, args.num_centers), 
        #                      symbols.view(-1))
        coding_loss = ce_loss(predictions, symbols.permute(0,3,1,2))

        decoded = decoder(quantized)
            
        #loss = mse_loss(decoded, real_data_v)
        ssim = ssim_loss(decoded, real_data_v)
        loss = 1 - ssim
        #raise Exception('delete me')
            # if args.orthoreg_loss: 
            #     ortho_loss_d[0] = 0
            #     ortho_loss_v = autograd.Variable(ortho_loss_d)
            #     orthoreg.orthoreg_loss(netD, ortho_loss_v)
            #     loss += ortho_loss_v

        # sorta best-working:
        #(loss + beta*ssim*coding_loss).backward()
        # original formulation:
        (loss + beta*coding_loss).backward()
        #(loss + beta*coding_loss - ssim*coding_loss).backward()
        
        #(loss*3*coding_loss.detach() + ssim*coding_loss).backward()
        #(-ssim*(7 - coding_loss)).backward()
        optimizer.step()
        optimizer_context.step()
        scheduler.step()
        scheduler_context.step()

        D_cost = loss.data

        # Write logs and save samples
        lib.plot.plot(str(RUN_PATH / 'reconstruction loss'), D_cost.cpu().numpy())
        lib.plot.plot(str(RUN_PATH / 'coding loss'), coding_loss.data.cpu().numpy())
        lib.plot.plot(str(RUN_PATH / 'time'), time.time() - start_time)
        # if args.orthoreg_loss:
        #     lib.plot.plot(str(RUN_PATH / 'ortho loss G'), ortho_loss_g.cpu().numpy())
        #     lib.plot.plot(str(RUN_PATH / 'ortho loss D'), ortho_loss_d.cpu().numpy())


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
            save_images(decoded, str(RUN_PATH / 'samples_{}_reconstruct.jpg'.format(iteration)))
            #print(encoded)
            #print(fixed_noise_128)

        # TODO: argument
        # Save logs every 200 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()
        lib.plot.tick()

        # TODO: argument
        if iteration % 2000 == 1999:
            state_dict = {
                        'iters': iteration + 1,
                        'algo_params': vars(args),
                        'decoder_dict': decoder.state_dict(),
                        'encoder_dict': encoder.state_dict(),
                        'quantizer': quantizer.state_dict(),
                        'context_model': context_model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'optimizer_context' : optimizer_context.state_dict(),
                    }

            torch.save(state_dict, str(RUN_PATH / 'state_{}.pth.tar'.format(iteration+1)))

if __name__ == "__main__":
    main()
