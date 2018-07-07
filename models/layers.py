import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.InstanceNorm1d(1, affine=True)
        self.weight = None # init compatibility, ugh

    def forward(self, x):
        original_size = x.size()
        # convert layer to minibatch x 1 x (c * h * w) matrix to normalize the entire layer
        as_1d = x.view(original_size[0], -1).unsqueeze(1)
        normalized = self.norm(as_1d)
        return normalized.view(original_size)


class GroupNorm(nn.Module):
    def __init__(self, layers, affine, groups=32, *args, **kwargs):
        super().__init__()
        self.norm = nn.InstanceNorm1d(layers // groups, affine=affine)
        self.weight = None # init compatibility, ugh
        self.groups = groups
        self.layers = layers

    def forward(self, x):
        original_size = x.size()
        # convert layer to minibatch x groups x ((c/groups) * h * w) matrix to normalize the entire layer
        as_groups = x.view(original_size[0], self.groups,-1)
        normalized = self.norm(as_1d)
        return normalized.view(original_size)



class UpscaleSR(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2),
                            kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.recast = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.ps = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv.weight.data.copy_(ICNR(self.conv.weight, upscale_factor))

    def forward(self, x):
        #base = self.recast(x)
        #base = base[:,None]
        upscale_ratio = self.upscale_factor**2
        #base = base.expand(base.size(0), upscale_ratio, base.size(2), base.size(3), base.size(4))
        #base = base.contiguous().view(base.size(0), base.size(1) * base.size(2), base.size(3), base.size(4))
        h = self.conv(x) # resnet ftw
        h = self.relu(self.bn(self.ps(h)))# + self.ps(base) 
        return h

def ICNR(tensor, upscale_factor=2, inizializer=nn.init.kaiming_normal):
    """Fills the input Tensor or Variable with values according to the method
    described in "Checkerboard artifact free sub-pixel convolution"
    - Andrew Aitken et al. (2017), this inizialization should be used in the
    last convolutional layer before a PixelShuffle operation
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        upscale_factor: factor to increase spatial resolution by
        inizializer: inizializer to be used for sub_kernel inizialization
    Examples:
        >>> upscale = 8
        >>> num_classes = 10
        >>> previous_layer_features = Variable(torch.Tensor(8, 64, 32, 32))
        >>> conv_shuffle = Conv2d(64, num_classes * (upscale ** 2), 3, padding=1, bias=0)
        >>> ps = PixelShuffle(upscale)
        >>> kernel = ICNR(conv_shuffle.weight, scale_factor=upscale)
        >>> conv_shuffle.weight.data.copy_(kernel)
        >>> output = ps(conv_shuffle(previous_layer_features))
        >>> print(output.shape)
        torch.Size([8, 10, 256, 256])
    .. _Checkerboard artifact free sub-pixel convolution:
        https://arxiv.org/abs/1707.02937
    """
    new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    subkernel = inizializer(subkernel)
    subkernel = subkernel.transpose(0, 1)

    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)

    kernel = subkernel.repeat(1, 1, upscale_factor ** 2)

    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    kernel = kernel.transpose(0, 1)

    return kernel