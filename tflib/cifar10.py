import torch
import numpy as np
import os
import urllib
import gzip
import pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict['data']

def cifar_generator(filenames, batch_size, data_dir, cuda=False):
    all_data = []
    for filename in filenames:
        all_data.append(unpickle(data_dir + '/' + filename))

    images = torch.from_numpy(np.concatenate(all_data, axis=0)).float()
    #images = np.concatenate(all_data, axis=0)
    if cuda:
        images = images.cuda()
        pass
    # prepare for consumption
    images = (images / 255 - 0.5) * 2

    def get_epoch():
        #np.random.shuffle(images)

        permutation = torch.randperm(len(images))
        if cuda:
            permutation = permutation.cuda()
        for i in range(len(images) // batch_size):
            yield images[permutation[i*batch_size:(i+1)*batch_size]]
            #yield images[i*batch_size:(i+1)*batch_size]

    return get_epoch


def load(batch_size, data_dir, cuda=False):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir, cuda), 
        cifar_generator(['test_batch'], batch_size, data_dir, cuda)
    )
