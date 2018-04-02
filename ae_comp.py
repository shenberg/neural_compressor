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

from models.layers import LayerNorm
from models.main import Generator, Encoder, AutoEncoder, AutoEncoderUpscale, AutoEncoderPSP, SNEncoder
from losses import gradient_penalty_loss

from utils import DatasetSubset, mix_samples, _mix_samples, load_batches, batchify, EagerFolder, save_images
import orthoreg
from pytorch_ssim import SSIM
import mmd

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

def plain_res_block(in_dim, out_dim, conv):
    return nn.Sequential(
        conv(in_dim, in_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(in_dim),
        nn.ReLU(inplace=True),
        conv(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        )


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, block_func):
        super().__init__()
        self.block = block_func(in_dim, out_dim)

    def forward(self, x):
        h = self.block(x)
        return x + h


def resnet_block(in_dim, out_dim, depth=3, conv=nn.Conv2d):
    inner_block_func = functools.partial(plain_res_block, conv=conv)
    blocks = []
    for block in range(depth - 1):
        blocks.append(ResBlock(in_dim, in_dim, inner_block_func))
    blocks.append(ResBlock(in_dim, out_dim, inner_block_func))
    return nn.Sequential(*blocks)

def downscale(in_channels, out_channels):
    return [nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),]

class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim, res_blocks=5, depth=3, conv=nn.Conv2d):
        super().__init__()
        self.input_transform = nn.Sequential(
            #*(downscale(3, 64) + downscale(64, 128)
            conv(3, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv(64, 64, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv(64, 128, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )

        blocks = [ResBlock(128, 128, functools.partial(resnet_block, depth=depth, conv=conv)) \
                    for i in range(res_blocks)]
        self.main = nn.Sequential(*blocks)
        # In paper, this:
        #self.final = conv(128, latent_dim, kernel_size=5, padding=2, stride=2, bias=True)
        self.final = nn.Conv2d(128, latent_dim, kernel_size=4, padding=1, stride=2, bias=True)
        # but details imply this:
        # self.final = nn.Sequential(
        #             conv(128, latent_dim, kernel_size=5, padding=2, stride=2, bias=True),
        #             # TODO: test BN here too
        #             nn.ReLU(inplace=True),
        #             )

    def forward(self, x):
        h = self.input_transform(x)
        h = self.main(h)
        h = self.final(h)
        return h

def _upscale_resize(in_dim, out_dim, kernel_size):
    pad1, pad2 = (kernel_size - 1) // 2, kernel_size // 2

    block = [
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((pad1, pad2, pad1, pad2)),
        nn.Conv2d(in_dim, out_dim, kernel_size, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(True)
        ]
    return block

class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim, res_blocks=5, depth=3, conv=nn.ConvTranspose2d):
        super().__init__()
        self.input_transform = nn.Sequential(*_upscale_resize(latent_dim, 128, 5))
        # TODO: padding hack
        # self.input_transform = nn.Sequential(
        #     conv(latent_dim, 128, kernel_size=5, padding=1, stride=2, bias=False),
        #     #conv(latent_dim, 128, kernel_size=4, padding=1, stride=2, bias=False),
        #     #nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     )
        self.output_transform = nn.Sequential(
            *(_upscale_resize(128, 64, 5) + _upscale_resize(64,64,5) +
                [conv(64, 3, kernel_size=3, padding=1, stride=1, bias=True),
                 nn.Sigmoid()])
            )
        # self.output_transform = nn.Sequential(
        #     conv(128, 64, kernel_size=5, padding=2, stride=2, bias=False),
        #     #conv(128, 64, kernel_size=4, padding=1, stride=2, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     conv(64, 3, kernel_size=5, padding=2, stride=2, bias=True),
        #     #conv(64, 64, kernel_size=4, padding=1, stride=2, bias=False),
        #     #nn.BatchNorm2d(64),
        #     #nn.ReLU(inplace=True),
        #     #conv(64, 3, kernel_size=3, padding=1, stride=1, bias=True),
        #     # TODO: replace with denormalize
        #     nn.Sigmoid(),
        #     #nn.BatchNorm2d(128),
        #     #nn.ReLU(inplace=True),
        #     )

        blocks = [ResBlock(128, 128, functools.partial(resnet_block, depth=depth, conv=nn.Conv2d)) \
                    for i in range(res_blocks)]
        self.main = nn.Sequential(*blocks)        

    def forward(self, x):
        h = self.input_transform(x)
        h = self.main(h)
        h = self.output_transform(h)
        return h
        #TODO: padding hack
        #return h[:,:,:h.size(2)-1, :h.size(3)-1]


# Encode 1d samples, separate encoder per channel
class SoftToHardEncoder(nn.Module):
    def __init__(self, num_codes, latent_dim):
        super().__init__()
        # centers - each row represents the centers for a channel
        # each column is a specific center
        self.codes = nn.Embedding(latent_dim, num_codes)
        self.num_codes = num_codes
        self.latent_dim = latent_dim

        self.softmin = nn.Softmin(dim=4) # intended to work on BxHxWxCxnum_codes matrices

    def forward(self, z):
        h = z
        # BCWH -> BWHC
        h = h.permute(0,2,3,1)
        h = h.contiguous()

        # change W to 1x1x1xCxnum_codes to get desired broadcast behavior
        W = self.codes.weight[None,None,None]
        # duplicate each channel value along the last axis
        # so now BxWxHxCxnum_codes matrix
        expanded = h.unsqueeze(-1).expand(h.size() + (self.num_codes,))

        distances = (expanded - W).abs()

        symbol_probs = self.softmin(distances)
        # sum of symbols weighted by their odds
        # now we have a BWHC matrix again
        soft_symbols = (symbol_probs * W).sum(dim=4)

        # Get the index of the center chosen per channel
        # BxWxHxCxnum_channels -> BWHC
        idxes = distances.min(dim=4)[1] 
        # calculate row offset for each channel for lookup into dictionary
        # TODO: cuda hack
        offsets = autograd.Variable(torch.arange(0, self.latent_dim * self.num_codes, self.num_codes,
                    out=torch.cuda.LongTensor((1,1,1,self.latent_dim))))
        # take the corresponding center from each channel
        hard_symbols = W.take(idxes + offsets)
        #j = L2_dist(z[:,None],W[None,:]).sum(2).min(1)[1]
        return soft_symbols, hard_symbols, idxes


# TODO: implement as kernel_size / 2 with one-sided padding
class MaskedConv3d(nn.Conv3d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv3d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kC, kH, kW = self.weight.size()
        self.mask.fill_(1)
        # fill far half of the buffer with 0 (Z points away)
        self.mask[:, :, kC // 2 + 1:] = 0
        # for middle slice, fill bottom half of slice with 0s (Y points down)
        self.mask[:, :, kC // 2, kH // 2 + 1:] = 0
        # for middle row of middle slice, fill end of row with 0s
        # NOTE: first layer must have mask type 'A', which zeros out the middle weight too
        #       this is because to predict the voxel we must not use the input as a value
        self.mask[:, :, kC // 2, kH // 2, kW // 2 + (mask_type == 'B'):] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv3d, self).forward(x)


class ContextModel(nn.Module):
    def __init__(self, centers, channels=24, kernel_size=3):
        super().__init__()
        self.initial = nn.Sequential(
            MaskedConv3d('A', 1, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        self.main = nn.Sequential(
            MaskedConv3d('B', channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MaskedConv3d('B', channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        # In paper:
        self.output = nn.Sequential(
            MaskedConv3d('B', channels, centers, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        # ReLU shouldn't matter...
        #self.output = MaskedConv3d('B', channels, centers, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.initial(x[:,None])
        h = h + self.main(h)
        return self.output(h)


def main():
    parser = argparse.ArgumentParser(description="options")
    parser.add_argument("-o", "--output-base-dir", default="/mnt/7FC1A7CD7234342C/compression-results/")
    parser.add_argument("-lr", "--learning-rate", type=float, default=4e-3)
    parser.add_argument("--data-dir", default="/mnt/7FC1A7CD7234342C/compression/dataset", help="path to image dataset")
    parser.add_argument("--dim", type=int, default=64, help="base dimension for generator")
    parser.add_argument("--latent-dim", type=int, default=64, help="latent dimension for autoencoder")
    parser.add_argument("--num-centers", type=int, default=8, help="number of centers for quantization")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size. Bigger is better, limit is RAM")
    parser.add_argument("--iterations", type=int, default=100000, help="generator iterations")
    parser.add_argument("--lr-decay-iters", type=int, default=10000, help="time till decay")
    parser.add_argument("--image-size", type=int, default=160, help="image size (one side, default 64)")
    parser.add_argument("--context-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--coding-loss-beta", type=float, default=0.01, help="constant multiplier for entropy loss")

    args = parser.parse_args()

    RUN_PATH = pathlib.Path(args.output_base_dir) / time.strftime('%Y_%m_%d_%H_%M_%S')  #TODO: generate by settings
    RUN_PATH.mkdir()
    #TODO:hack
    tflib.plot.log_dir = str(RUN_PATH) + '/'

    with (RUN_PATH / 'algo_params.txt').open('w') as f:
        import json
        json.dump(vars(args), f, indent=2)

    encoder = ResNetEncoder(latent_dim=args.latent_dim)
    decoder = ResNetDecoder(latent_dim=args.latent_dim)
    quantizer = SoftToHardEncoder(num_codes=args.num_centers, latent_dim=args.latent_dim)
    context_model = ContextModel(args.num_centers)

    decoder.apply(weights_init)
    encoder.apply(weights_init)
    print(decoder)
    print(encoder)
    use_cuda = torch.cuda.is_available()
    ce_loss = torch.nn.CrossEntropyLoss()
    ssim_loss = SSIM()
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

    beta = args.coding_loss_beta
    # pre-processing transform
    # augmentation goes here, e.g. RandomResizedCrop instead of regular random crop
    transform = torchvision.transforms.Compose([
        #torchvision.transforms.RandomCrop(args.image_size),
        torchvision.transforms.RandomResizedCrop(args.image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2) # convert pixel values from 0..1 to -1..1
        ])



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
        quantized = quantized.permute(0,3,1,2)

        predictions = context_model(quantized)
        coding_loss = ce_loss(predictions, symbols.permute(0,3,1,2))

        decoded = decoder(quantized)
            
        loss = mse_loss(decoded, real_data_v)

            # if args.orthoreg_loss: 
            #     ortho_loss_d[0] = 0
            #     ortho_loss_v = autograd.Variable(ortho_loss_d)
            #     orthoreg.orthoreg_loss(netD, ortho_loss_v)
            #     loss += ortho_loss_v

        (loss + beta*coding_loss).backward()
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
