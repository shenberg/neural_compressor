import argparse
import os, sys, time
import random

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
import torch.nn.functional as F
torch.set_default_tensor_type('torch.HalfTensor')

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
DATA_DIR = '/home/shenberg/Documents/compression/'
OUTPUT_BASE_DIR = '/mnt/7FC1A7CD7234342C/compression-results/'
RUN_PATH = '{}{}/'.format(OUTPUT_BASE_DIR, time.strftime('%Y_%m_%d_%H_%M_%S'))  #TODO: generate by settings
if not os.path.exists(RUN_PATH):
    os.mkdir(RUN_PATH)
#TODO:hack


def save_images(images_tensor, output_path):
    samples = images_tensor
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(samples, output_path)

def inf_gen(gen):
    while True:
        for images, _ in gen:
            # yield images.astype('float32').reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
            yield images


def main():
    main_arg_parser = argparse.ArgumentParser(description="options")
    main_arg_parser.add_argument("-b,","--batches", type=int, default=1000)
    main_arg_parser.add_argument("-bs", "--batch-size", type=int, default=128)
    main_arg_parser.add_argument("-s", "--image-size", type=int, default=64)
    main_arg_parser.add_argument("--output-dir", default=OUTPUT_BASE_DIR)
    main_arg_parser.add_argument("--seed", help="random seed for torch", type=int, default=42)

    args = main_arg_parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # pre-processing transform
    # augmentation goes here, e.g. RandomResizedCrop instead of regular random crop

    transform = torchvision.transforms.Compose([
        #torchvision.transforms.RandomCrop(IMAGE_SIZE),
        torchvision.transforms.RandomResizedCrop(args.image_size),
        #torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2) # convert pixel values from 0..1 to -1..1
        ])

    out_path = os.path.join(args.output_dir, "bs_{}_size_{}".format(args.batch_size, args.image_size))
    os.makedirs(out_path, exist_ok=True)

    # Dataset iterator
    images_dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=transform)


    # pre-shuffle since we'll need to linearly stream from ae to gain performance advantage
    gen = torch.utils.data.DataLoader(images_dataset, args.batch_size, shuffle=True,
                                            num_workers=6)

    print("Writing to output dir: {}", out_path)
    gen = inf_gen(gen)

    for iteration in tqdm(range(args.batches)):
        _data = next(gen)

        torch.save(_data, os.path.join(out_path, "batch_{}_length_{}.pth".format(iteration+1, len(_data))))

if __name__=='__main__':
    main()