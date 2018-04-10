import os, sys, time
sys.path.append(os.getcwd())
import functools
import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
import torch.nn.functional as F

import time

import numpy as np
from tqdm import tqdm

from models.autoencoder import ContextModel, SoftToHardNdEncoder, ResNetEncoder, ResNetDecoder


import argparse
import pathlib

import contextlib
import struct
from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq
from PIL import Image

def get_latest_saved_model(path):
    # stem (filename out of path) is 'state_NUMBER.pth.tar'
    key = lambda filepath: int(filepath.stem.split('_')[1].split('.')[0])
    # model with highest state number
    return max(path.glob("*.pth.tar"), key=key)

def p2cf(prob, resolution=1024):
    "much more efficient version of prob_to_cum_freq bundled with range_coder"
    prob = np.asarray(prob, dtype=np.float64)
    # HACK assume all probs > 0
    #freq = np.ones(prob.size, dtype=int)
    freq = (prob * (resolution - len(prob))).astype(np.int)

    # this is similar to gradient descent in KL divergence (convex)
    with np.errstate(divide='ignore', invalid='ignore'):
        for _ in range(resolution - np.sum(freq)):
            freq[np.nanargmax(prob / freq)] += 1

    return [0] + np.cumsum(freq).tolist()

def encode(dest_path, img_var, encoder, quantizer, context_model):
    # TODO: pad so that each dim divides by 8

    # assume img_var is 1xCxHxW
    latent = encoder(img_var)
    # take hard quantization results and integer symbols
    _, quantized, symbols = quantizer(latent)

    quantized = quantized.permute(0,3,1,2).contiguous()

    # NOTE: we move from variable to tensor
    symbols = symbols.permute(0,3,1,2).data
    #TODO: remove once debugged
    #torch.save(symbols, dest_path + ".syms")
    # softmax along channel 1 to convert to probabilities
    # NOTE: we move from variable to tensor
    # NOTE: move to CPU
    # TODO: log_softmax and then exp?
    probabilities = F.softmax(context_model(quantized), dim=1).data.cpu()
    print(probabilities.size())

    #cum_prob = probabilities.cumsum(dim=1)
    # cumulative frequency list for range coder
    int_cumprob = list(range(257))

    _, depth, height, width = symbols.size()

    # make sure we close encoder
    with contextlib.closing(RangeEncoder(str(dest_path))) as enc:
        # encode width, height as 16-bit little-endian bytes
        enc.encode(list(struct.pack("<HH", width, height)), int_cumprob)
        for channel in tqdm(range(depth)):
            for row in range(height):
                for col in range(width):
                    probs = probabilities[0, :, channel, row, col]
                    # TODO: faster prob_to_cum_freq using non-iterative method
                    # TODO: use higher precision?
                    #cumprob = prob_to_cum_freq(probs.cpu().numpy())
                    cumprob = p2cf(probs.cpu().numpy())
                    enc.encode([symbols[0, channel, row, col]], cumprob)

def decode_symbols(src_path, context_model, quantizer):


def decode(dest_path, src_path, )


def main():
    import argparse, pathlib
    parser = argparse.ArgumentParser(description="options")
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("-x","--extract", action="store_true")
    command.add_argument("-c","--compress", action="store_true")
    parser.add_argument("-m", "--model-dir", default="/mnt/7FC1A7CD7234342C/compression-results/2018_04_09_22_05_12_good_quality_rate_too_high")
    # TODO: specific model instead of dir
    #parser.add_argument
    #parser.add_argument("source")
    #parser.add_argument("dest")

    args = parser.parse_args()

    RUN_PATH = pathlib.Path(args.model_dir)

    model_path = get_latest_saved_model(RUN_PATH)
    print("Getting path: " + str(model_path))

    with (RUN_PATH / 'algo_params.txt').open('r') as f:
        import json
        model_args = argparse.Namespace(**json.load(f))

    encoder = ResNetEncoder(latent_dim=model_args.latent_dim)
    decoder = ResNetDecoder(latent_dim=model_args.latent_dim)
    # TODO: toggle
    #quantizer = SoftToHardEncoder(num_codes=model_args.num_centers, latent_dim=model_args.latent_dim)
    quantizer = SoftToHardNdEncoder(num_codes=model_args.num_centers, 
                                    latent_dim=model_args.latent_dim // model_args.vq_dim, 
                                    channel_dim = model_args.vq_dim)
    context_model = ContextModel(model_args.num_centers, model_args.vq_dim)

    #print(decoder)
    #print(encoder)
    #print(context_model)
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        gpu = 0
        encoder = encoder.cuda(gpu)
        decoder = decoder.cuda(gpu)
        quantizer = quantizer.cuda(gpu)
        context_model = context_model.cuda(gpu)

    # TODO: gpu/cpu load toggle
    state_dict = torch.load(str(model_path))

    decoder.load_state_dict(state_dict['decoder_dict'], strict=True)
    encoder.load_state_dict(state_dict['encoder_dict'], strict=True)
    quantizer.load_state_dict(state_dict['quantizer'], strict=True)
    context_model.load_state_dict(state_dict['context_model'], strict=True)

    decoder.eval()
    encoder.eval()
    quantizer.eval()
    context_model.eval()


    start_time = time.time()

    #TODO: hack! for testing!
    args.source = "/home/shenberg/Documents/compression/mobile/0007.png"
    args.dest = "/home/shenberg/Documents/neural_compressor/test.enc"
    # ENCODE ONLY FOR NOW!
    # NOTE: this size divides by 8 cleanly
    print("Loading image: " + args.source)
    image = Image.open(args.source)
    print("done loading, now encoding")
    # TODO: notice how we add batch dim of 1
    image_var = autograd.Variable(torchvision.transforms.ToTensor()(image)[None], volatile=True)

    if use_cuda:
        image_var = image_var.cuda(gpu, async=True)

    encode(args.dest, image_var, encoder, quantizer, context_model)
    print("done! took {} secs", time.time() - start_time)
        #     real_data = real_data.cuda(gpu, async=True)
        # real_data_v = autograd.Variable(real_data)
        # encoder.zero_grad()
        # decoder.zero_grad()
        # quantizer.zero_grad()
        # context_model.zero_grad()

        # encoded = encoder(real_data_v)

        # soft, hard, symbols = quantizer(encoded)

        # quantized = (hard - soft).detach() + soft # use soft symbol for backprop
        # #TODO: move into quantizer
        # quantized = quantized.permute(0,3,1,2).contiguous()
        # #quantized = quantized.permute(0,3,1,2).contiguous()

        # #symbols = symbols.permute(0,3,1,2).contiguous()

        # predictions = context_model(quantized)
        # #print(symbols.size(), predictions.size())

        # #coding_loss = ce_loss(predictions.view(-1, args.num_centers), 
        # #                      symbols.view(-1))
        # coding_loss = ce_loss(predictions, symbols.permute(0,3,1,2))

        # decoded = decoder(quantized)
            
        # #loss = mse_loss(decoded, real_data_v)
        # ssim = ssim_loss(decoded, real_data_v)
        # loss = 1 - ssim
        # #raise Exception('delete me')
        #     # if args.orthoreg_loss: 
        #     #     ortho_loss_d[0] = 0
        #     #     ortho_loss_v = autograd.Variable(ortho_loss_d)
        #     #     orthoreg.orthoreg_loss(netD, ortho_loss_v)
        #     #     loss += ortho_loss_v

        # (loss + beta*coding_loss).backward()
        # #(-ssim*(7 - coding_loss)).backward()
        # optimizer.step()
        # optimizer_context.step()
        # scheduler.step()
        # scheduler_context.step()

        # D_cost = loss.data

        # # Write logs and save samples
        # lib.plot.plot(str(RUN_PATH / 'reconstruction loss'), D_cost.cpu().numpy())
        # lib.plot.plot(str(RUN_PATH / 'coding loss'), coding_loss.data.cpu().numpy())
        # lib.plot.plot(str(RUN_PATH / 'time'), time.time() - start_time)
        # # if args.orthoreg_loss:
        # #     lib.plot.plot(str(RUN_PATH / 'ortho loss G'), ortho_loss_g.cpu().numpy())
        # #     lib.plot.plot(str(RUN_PATH / 'ortho loss D'), ortho_loss_d.cpu().numpy())


        # # TODO: argument
        # # Calculate dev loss and generate samples every 100 iters
        # if iteration % 100 == 99:
        #     #dev_disc_costs = []
        #     #netD.eval()
        #     #for images, _ in dev_gen:
        #     #    images = images.view((-1, 3, 128, 128))
        #     #    imgs = images#preprocess(images)
        #     #
        #     #    #imgs = preprocess(images)
        #     #    if use_cuda:
        #     #        imgs = imgs.cuda(gpu)
        #     #    imgs_v = autograd.Variable(imgs, volatile=True)
        #     #
        #     #    D, encoded = netD(imgs_v)
        #     #    _dev_disc_cost = -D.mean().cpu().data.numpy()
        #     #    dev_disc_costs.append(_dev_disc_cost)
        #     #netD.train()
        #     #lib.plot.plot(RUN_PATH + 'dev disc cost', np.mean(dev_disc_costs))

        #     #fixed_noise_128 = torch.randn(128, INPUT_DIM)
        #     #if use_cuda:
        #     #    fixed_noise_128 = fixed_noise_128.cuda(gpu)
        #     #generate_image(iteration, netG, fixed_noise_128)
        #     #generate_image("{}_reconstruct".format(iteration), netG, encoded.data, True)
        #     save_images(real_data_v, str(RUN_PATH / 'samples_{}_original.jpg'.format(iteration)))
        #     save_images(decoded, str(RUN_PATH / 'samples_{}_reconstruct.jpg'.format(iteration)))
        #     #print(encoded)
        #     #print(fixed_noise_128)

        # # TODO: argument
        # # Save logs every 200 iters
        # if (iteration < 5) or (iteration % 100 == 99):
        #     lib.plot.flush()
        # lib.plot.tick()

        # # TODO: argument
        # if iteration % 2000 == 1999:
        #     state_dict = {
        #                 'iters': iteration + 1,
        #                 'algo_params': vars(args),
        #                 'decoder_dict': decoder.state_dict(),
        #                 'encoder_dict': encoder.state_dict(),
        #                 'quantizer': quantizer.state_dict(),
        #                 'context_model': context_model.state_dict(),
        #                 'optimizer' : optimizer.state_dict(),
        #                 'optimizer_context' : optimizer_context.state_dict(),
        #             }

        #     torch.save(state_dict, str(RUN_PATH / 'state_{}.pth.tar'.format(iteration+1)))

if __name__ == "__main__":
    main()
