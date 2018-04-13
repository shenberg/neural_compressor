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
# shit hack
np.random.seed(1)
import random
random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available:
    torch.cuda.manual_seed_all(1)


from tqdm import tqdm

from models.autoencoder import ContextModel, ContextModelMultilayered, SoftToHardNdEncoder, ResNetEncoder, ResNetDecoder

import argparse
import pathlib

import contextlib
import struct
from range_coder import RangeEncoder, RangeDecoder, prob_to_cum_freq
from PIL import Image

use_cuda = False

def get_latest_saved_model(path):
    # stem (filename out of path) is 'state_NUMBER.pth.tar'
    key = lambda filepath: int(filepath.stem.split('_')[1].split('.')[0])
    # model with highest state number
    return max(path.glob("*.pth.tar"), key=key)

def truncate(arr, mantissa_const=1024):
    mantissa, exp = np.frexp(arr)
    mantissa = np.around(mantissa*mantissa_const) / mantissa_const
    return np.ldexp(mantissa, exp)


def p2cf(prob, resolution=1024):
    """
    much more efficient version of prob_to_cum_freq bundled with range_coder.
    inserts guard symbols after each symbol
    """
    prob_base = np.asarray(prob, dtype=np.float64)
    # HACK assume all probs > 0
    #freq = np.ones(prob.size, dtype=int)
    freq = (prob_base * (resolution - len(prob))).astype(np.int)

    # this is similar to gradient descent in KL divergence (convex)
    with np.errstate(divide='ignore', invalid='ignore'):
        for _ in range(resolution - np.sum(freq)):
            freq[np.nanargmax(prob_base / freq)] += 1

    return [0] + np.cumsum(freq).tolist()

# rounding symbol frequency roughly 50 zeros per -1 or 1
#ROUNDING_FREQUENCIES = np.asarray([1,9,1], dtype=np.int)
# calculated using range_coder's exact solution, for frequencies: [5355,5730330, 360699]
# for exactly one image :(
ROUNDING_CUMPROB = [0, 10, 1013, 1024]

def probabilities_to_cum_freq_with_rounding(prob, resolution=1024, epsilon=1e-5):
    #pre-condition every cell to get at least one prob unit:
    pc = 1/(resolution - len(prob))
    prob = (prob + pc) / (1 + len(prob)*pc)

    binned = (prob*resolution).round().astype(np.int)
    # symbol where we want the decoder to add epsilon
    add_epsilon = binned - ((prob - epsilon)*resolution).round().astype(np.int)
    # symbols where the decoder should remove epsilon
    sub_epsilon = binned - ((prob + epsilon)*resolution).round().astype(np.int)
    round_directions = add_epsilon + sub_epsilon
    # 
    #binned[binned == 0] = 1
    #TODO: hack, spread remainder better
    binned[binned.argmax()] += resolution - binned.sum()
    if np.any(binned == 0):
        print("ugh zero probs")
    return ([0] + np.cumsum(binned).tolist()), (round_directions + 1).tolist()

def probabilities_to_cum_freq_from_rounding(prob, rounding_syms, resolution=1024, epsilon=1e-5):
    #pre-condition every cell to get at least one prob unit:
    pc = 1/(resolution - len(prob))
    prob = (prob + pc) / (1 + len(prob)*pc)

    roundings = (np.asarray(rounding_syms, dtype=np.int) - 1)*epsilon
    binned = ((prob + roundings)*resolution).round().astype(np.int)
    #binned[binned == 0] = 1
    #TODO: hack, spread remainder better
    binned[binned.argmax()] += resolution - binned.sum()
    return [0] + np.cumsum(binned).tolist()

def calculate_padding(width, height):
    return (((-width) % 16), ((-height) % 16))

def encode(dest_path, img_var, encoder, quantizer, context_model):
    # TODO: pad so that each dim divides by 16
    img_width, img_height = img_var.size(3), img_var.size(2)
    pad_x, pad_y = calculate_padding(img_width, img_height)
    if pad_x != 0 or pad_y !=0:
        print("padding image!")
        img_var = F.pad(img_var, ((pad_x + 1) // 2, pad_x // 2, (pad_y + 1) // 2, pad_y // 2))

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
    post_context = context_model(quantized)
    probabilities = F.softmax(post_context.cpu(), dim=1).data
    #print(probabilities.size())

    #cum_prob = probabilities.cumsum(dim=1)
    # cumulative frequency list for range coder
    int_cumprob = list(range(257))

    _, depth, height, width = symbols.size()

    #cumprobs = []
    # make sure we close encoder
    with contextlib.closing(RangeEncoder(str(dest_path))) as enc:
        # encode _image_ width, height as 16-bit little-endian bytes
        enc.encode(list(struct.pack("<HH", img_width, img_height)), int_cumprob)
        for channel in tqdm(range(depth)):
            for row in range(height):
                for col in range(width):
                    probs = probabilities[0, :, channel, row, col]
                    # TODO: faster prob_to_cum_freq on gpu
                    # TODO: use higher precision? 
                    #cumprob = prob_to_cum_freq(probs.cpu().numpy())
                    
                    #cumprob = p2cf(probs.cpu().numpy().astype(np.float64), 1024)
                    cumprob, rounds = probabilities_to_cum_freq_with_rounding(probs.cpu().numpy().astype(np.float64))
                    #cumprobs.append(cumprob)
                    enc.encode(rounds, ROUNDING_CUMPROB)
                    enc.encode([symbols[0, channel, row, col]], cumprob)
    #torch.save(cumprobs, dest_path + '.probs')
    #torch.save(probabilities, dest_path + '.ps')
    #torch.save(post_context, dest_path + '.pcm')
    #torch.save(quantized, dest_path + '.lat')
    #torch.save(symbols, dest_path + '.sym')

def decode_symbols(src_path, quantizer, context_model):
    # quantizer knows how many dimensions it has
    channels = quantizer.latent_dim
    vector_size = quantizer.channel_dim

    int_cumprob = list(range(257))
    with contextlib.closing(RangeDecoder(str(src_path))) as dec:
        # two little-endian 16-bit ints
        width_l, width_h, height_l, height_h = dec.decode(4, int_cumprob)
        output_width = width_l + (width_h << 8)
        output_height = height_l + (height_h << 8)

        pad_x, pad_y = calculate_padding(output_width, output_height)
        width = output_width + pad_x
        height = output_height + pad_y
        if pad_x != 0 or pad_y !=0:
            print("padded image!")

        # TODO: round up 
        latent_width, latent_height = width // 8, height // 8

        symbols = torch.LongTensor(channels, latent_height, latent_width)
        symbols.fill_(0)

        context_layers = 4 # TODO: hack
        if isinstance(context_model, ContextModel):
            # TODO: cuda hacks
            # TODO: no ugly work directly with 3d tensor
            decoded_latent = autograd.Variable(torch.zeros(1, vector_size, channels, latent_height, latent_width),
                                               volatile=True)
            if use_cuda:
                decoded_latent = decoded_latent.cuda()

            for channel in tqdm(range(channels)):
                channel_offset = channel * vector_size

                channel_start = max(0, channel - context_layers)
                channel_end = min(channels * vector_size, channel + 1)

                for row in range(latent_height):
                    row_start = max(0, row - context_layers)
                    row_end = min(latent_height, row + context_layers + 1)

                    # if row is < context_layers, row index the correct index in the row

                    row_in_window = min(row, context_layers)
                    #window = decoded_latent[:,channel_start:channel_end, row_start:row_end, :].contiguous()
                    for col in range(latent_width):
                        roundings = dec.decode(vector_size, ROUNDING_CUMPROB)
                        #roundings = np.zeros(vector_size)
                        col_start = max(0, col - context_layers)
                        col_end = min(latent_width, col + context_layers + 1)
                        col_in_window = min(col, context_layers)
                        #probabilities = F.softmax(context_model(decoded_latent[:,:channel_offset + vector_size,:,:]), dim=1)
                        #probs = probabilities.data[0, :, channel, row, col]
                        window = decoded_latent[:,:,channel_start:channel_end, row_start:row_end, col_start:col_end]
                        # only pass window seen by context
                        post_context = context_model(window)
                        # calculate softmax only on specific probabilities we added now
                        probs = F.softmax(post_context[0, :, -1, row_in_window, col_in_window].cpu(), dim=0).data
                        #probs = F.softmax(post_context[0, :, -1, row_in_window, col], dim=0).data
                        # TODO: use higher precision?
                        #cumprob = prob_to_cum_freq(probs.cpu().numpy())
                        #cumprob = p2cf(probs.cpu().numpy().astype(np.float64), 1024)
                        cumprob = probabilities_to_cum_freq_from_rounding(probs.cpu().numpy().astype(np.float64), roundings)
                        symbol = dec.decode(1, cumprob)[0]
                        symbols[channel, row, col] = symbol
                        symbol_vector = quantizer.codes.data[channel, symbol]
                        decoded_latent.data[0, :, channel, row, col] = symbol_vector
            decoded_latent = decoded_latent.transpose(1,2).contiguous().view(1, channels * vector_size, latent_height, latent_width)
        elif isinstance(context_model, ContextModelMultilayered):
            #cumprobs = []
            # TODO: crap
            context_layers = 4

            decoded_latent = autograd.Variable(torch.zeros(1, vector_size*channels, latent_height, latent_width),
                                   volatile=True)
            # TODO: cuda hack
            if use_cuda:
                decoded_latent = decoded_latent.cuda()
            for channel in tqdm(range(channels)):
                channel_end = (channel + 1) * vector_size

                for row in range(latent_height):
                    row_start = max(0, row - context_layers)
                    row_end = min(latent_height, row + context_layers + 1)

                    # if row is < context_layers, row index the correct index in the row

                    row_in_window = min(row, context_layers)
                    #window = decoded_latent[:,channel_start:channel_end, row_start:row_end, :].contiguous()
                    for col in range(latent_width):
                        roundings = dec.decode(vector_size, ROUNDING_CUMPROB)
                        #roundings = np.zeros(vector_size)
                        col_start = max(0, col - context_layers)
                        col_end = min(latent_width, col + context_layers + 1)
                        col_in_window = min(col, context_layers)

                        window = decoded_latent[:,:channel_end, row_start:row_end, col_start:col_end]
                        # only pass window seen by context
                        post_context = context_model.models[channel](window)
                        # calculate softmax only on specific probabilities we added now
                        probs = F.softmax(post_context[0, :, row_in_window, col_in_window].cpu(), dim=0).data
                        # TODO: use higher precision?
                        #cumprob = p2cf(probs.cpu().numpy(), 1024)
                        cumprob = probabilities_to_cum_freq_from_rounding(probs.cpu().numpy().astype(np.float64), roundings)
                        #cumprobs.append(cumprob)
                        symbol = dec.decode(1, cumprob)[0]
                        symbols[channel, row, col] = symbol
                        symbol_vector = quantizer.codes.data[channel, symbol]
                        decoded_latent.data[0, channel_end - vector_size : channel_end, row, col] = symbol_vector
        else:
            raise Exception("bad context model type: {}".format(context_model))

    #torch.save(cumprobs, src_path + '.dec.probs')
    #torch.save(symbols, src_path + '.dec.sym')
    #torch.save(decoded_latent, src_path + '.dec.lat')
    # shuffle to regular form
    return decoded_latent, output_width, output_height


def decode(src_path, decoder, quantizer, context_model):
    decoded_latent, output_width, output_height = decode_symbols(src_path, quantizer, context_model)
    #print(output_width, output_height)
    #print(decoded_latent.size())
    decoded_image = decoder(decoded_latent).data[0]
    if (decoded_image.size(1) != output_height) or (decoded_image.size(2) != output_width):
        pad_x, pad_y = calculate_padding(output_width, output_height)
        return decoded_image[:,(pad_y + 1) // 2:-(pad_y // 2), (pad_x + 1) // 2:-(pad_x // 2)]
    else:
        return decoded_image

def main():
    # TODO: hack!!!
    global use_cuda
    import argparse, pathlib
    parser = argparse.ArgumentParser(description="options")
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("-x","--extract", action="store_true")
    command.add_argument("-c","--compress", action="store_true")
    parser.add_argument("-m", "--model-dir", default="/mnt/7FC1A7CD7234342C/compression-results/2018_04_11_02_39_30")
    parser.add_argument("--cuda", action="store_true")
    # TODO: specific model instead of dir
    #parser.add_argument
    parser.add_argument("source")
    parser.add_argument("dest")

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
    if not model_args.use_layered_context:
        context_model = ContextModel(model_args.num_centers, model_args.vq_dim)
    else:
        context_model = ContextModelMultilayered(model_args.num_centers, model_args.vq_dim, model_args.latent_dim // model_args.vq_dim)

    #print(decoder)
    #print(encoder)
    #print(context_model)
    use_cuda = args.cuda and torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        gpu = 0
        encoder = encoder.cuda(gpu)
        decoder = decoder.cuda(gpu)
        quantizer = quantizer.cuda(gpu)
        context_model = context_model.cuda(gpu)
        state_dict = torch.load(str(model_path))
    # TODO: gpu/cpu load toggle
    else:
        state_dict = torch.load(str(model_path), map_location=lambda storage, loc: storage)

    decoder.load_state_dict(state_dict['decoder_dict'], strict=True)
    encoder.load_state_dict(state_dict['encoder_dict'], strict=True)
    quantizer.load_state_dict(state_dict['quantizer'], strict=True)
    context_model.load_state_dict(state_dict['context_model'], strict=True)

    decoder.eval()
    encoder.eval()
    quantizer.eval()
    context_model.eval()


    start_time = time.time()

    # ENCODE ONLY FOR NOW!
    # NOTE: this size divides by 8 cleanly
    if args.compress:
        #TODO: hack! for testing!
        #args.source = "/home/shenberg/Documents/compression/mobile/0007.png"
        #args.dest = "/home/shenberg/Documents/neural_compressor/test.enc"
        print("Loading image: " + args.source)
        image = Image.open(args.source)
        print("done loading, now encoding")
        # TODO: notice how we add batch dim of 1
        image_var = autograd.Variable(torchvision.transforms.ToTensor()(image)[None], volatile=True)

        if use_cuda:
            image_var = image_var.cuda(gpu, async=True)

        encode(args.dest, image_var, encoder, quantizer, context_model)
    elif args.extract:
        #TODO: hack for testing
        #args.source = "/home/shenberg/Documents/neural_compressor/test.enc"
        #args.dest = "/home/shenberg/Documents/neural_compressor/test.dec.png"
        print("decoding image: " + args.source)
        img_tensor = decode(args.source, decoder, quantizer, context_model)
        print(img_tensor.size())
        print("done decoding, now saving")
        pi = torchvision.transforms.ToPILImage()(img_tensor.cpu())
        pi.save(args.dest)
    else:
        print("nothing to do?!")

    print("done! took {} secs".format(time.time() - start_time))


if __name__ == "__main__":
    main()
