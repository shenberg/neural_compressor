import torch
import torch.utils.data as data
import glob
import os
import random
import tqdm
from torchvision.datasets.folder import make_dataset, default_loader, find_classes
import tflib as lib

class DatasetSubset(data.Dataset):
    "Subset of existing dataset - useful for train/test split"
    def __init__(self, dataset, start, end):
        super().__init__()
        self.backing_ds = dataset
        self.start = int(len(dataset) * start)
        self.end = int(len(dataset) * end)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, i):
        return self.backing_ds[i + self.start]

def mirror(image_tensor):
    # CxHxW
    return image_tensor[:,:,list(reversed(range(image_tensor.size(2))))]

def augment(tensor):
    if torch.randn(1)[0] > 0:
        tensor = mirror(tensor)
    return tensor


class TensorBatchDataset(data.Dataset):
    def __init__(self, tensor):
        # TODO: hardcoded class to 0
        labels = torch.zeros(tensor.size(0)).long()
        self.ds = data.TensorDataset(tensor, labels)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        # TODO: augment as parameter
        image, target = self.ds[i]
        augmented = augment(image)
        #noised = (augmented + torch.randn(augmented.size())*0.1).clamp(-1, 1)
        #noised = augmented*torch.bernoulli(torch.rand(augmented.size())*0.75 + 0.25)
        noised = augmented

        return (augmented, noised), target

class EagerFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.imgs = [(self.loader(path), target) for path, target in tqdm.tqdm(imgs)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def load_batches(path):
    files = glob.glob(os.path.join(path, "*.pth"))
    # TODO: lazy-load?
    return data.ConcatDataset([TensorBatchDataset(torch.load(f)) for f in files])

def batchify(dataset, size=128, stop=1):
    length = round(len(dataset)*stop)
    tensors = [dataset[i][0] for i in tqdm.tqdm(range(length))]
    datasets = []
    for offset in range(0, length, size):
        datasets.append(TensorBatchDataset(torch.stack(tensors[offset:offset + size])))
    return data.ConcatDataset(datasets)

def mix_samples(reals, fakes, cuda):
    # TODO: hack
    indices = torch.ByteTensor(reals.size(0)) if not cuda else torch.cuda.ByteTensor(reals.size(0))
    indices.fill_(1)
    return reals, fakes, indices

def _mix_samples(reals, fakes, cuda):
    # assuming same number of samples, e.g. reals.size(0) == fakes.size(0)
    # TODO: preallocate tensors
    real_selections = torch.randperm(reals.size(0))
    if cuda:
        real_selections = real_selections.cuda()

    real_indexes = real_selections[:reals.size(0)//2]
    group1 = fakes.clone()
    group1[real_indexes] = reals[real_indexes]
    group2 = reals.clone()
    group2[real_indexes] = fakes[real_indexes]

    if cuda:
        out_mask = torch.cuda.ByteTensor(reals.size(0))
    else:
        out_mask = torch.ByteTensor(reals.size(0))
    out_mask.fill_(0)
    out_mask[real_indexes] = 1
    return group1, group2, out_mask

def _mix_samples_bernoulli(reals, fakes, cuda):
    # assuming same number of samples, e.g. reals.size(0) == fakes.size(0)
    # TODO: preallocate tensors
    if cuda:
        real_selections = torch.cuda.ByteTensor(reals.size(0))
    else:
        real_selections = torch.ByteTensor(reals.size(0))
    real_selections.random_(0, 2) # 0 or 1
    real_indexes = real_selections.nonzero()
    if len(real_indexes) == 0:
        # no real indexes at all
        return fakes.clone(), reals.clone(), real_selections
    real_indexes = real_indexes.squeeze(1) # len x 1 to len sized tensor
    if cuda:
        real_indexes = real_indexes.cuda()
    group1 = fakes.clone()
    group1[real_indexes] = reals[real_indexes]
    group2 = reals.clone()
    group2[real_indexes] = fakes[real_indexes]

    return group1, group2, real_selections


def save_images(images_tensor, output_path):
    samples = images_tensor
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(samples, output_path)
