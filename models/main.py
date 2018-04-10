
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import LayerNorm
import functools
from .sn_layers import SNConv2d

def _upscale_resize(in_dim, out_dim, kernel_size, norm, non_linearity):
    pad1, pad2 = (kernel_size - 1) // 2, kernel_size // 2

    block = [norm(in_dim)] if norm is not None else []
    block += [
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((pad1, pad2, pad1, pad2)),
        nn.Conv2d(in_dim, out_dim, kernel_size, bias=norm is None),
        non_linearity()
        ]
    return nn.Sequential(*block)

def _upblock(in_dim, out_dim, kernel_size, norm, non_linearity=lambda: nn.ReLU(True)):
    padding = (kernel_size - 1) // 2
    blocks = []
    bias_conv = not norm # if no norm them add bias parameter
    if norm is not None:
        blocks.append(norm(in_dim, affine=True))
    blocks.append(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=2, padding=padding, bias=bias_conv))
    blocks.append(non_linearity())
    return nn.Sequential(*blocks)


class Generator(nn.Module):
    def __init__(self, dim=64, input_dim=128, output_size=64, norm=LayerNorm, kernel_size=4, fc=False, upblock=_upblock,
                 non_linearity=functools.partial(nn.ReLU, inplace=True), extra_conv=False, layers=3):
        super(Generator, self).__init__()
        self.output_size = output_size # TODO: verify power-of-two, square
        # number of expand layers we need
        # we start out with a 4x4 map, and expand by 2 every layer
        # so bit length  (==log2 of power-of-two number) of output/input sizes 
        # is the numer of layers we need
        # subtract 1 because we always create the output layer not inside the sequential layer
        # TODO: fix for layers param when not FC
        self.layers = layers # (output_size // 4).bit_length() - 2
        self.dim = dim
        # full-convolutional? make first layer Conv2d instead of linear if so
        self.fc = fc

        if not fc:
            preprocess = nn.Sequential(
                #nn.InstanceNorm2d(4 * 4 * 4 * DIM),
                nn.Linear(input_dim, 4 * 4 * (2**self.layers) * dim),
                nn.ReLU(True),
            )
        else:
            # no norm
            preprocess = upblock(input_dim, (2**self.layers) * dim, kernel_size, None, non_linearity)

        self.preprocess = preprocess

        blocks = []
        for layer in reversed(range(self.layers)):
            out_dim = (2**layer) * dim
            blocks.append(upblock(2 * out_dim, out_dim, kernel_size, norm=norm, non_linearity=non_linearity))

        self.main = nn.Sequential(*blocks)

        if not extra_conv:
            self.block_out = upblock(dim, 3, kernel_size, norm=norm, non_linearity=nn.Tanh)
        else:
            out_block = [upblock(dim, dim, kernel_size, norm=norm, non_linearity=non_linearity)]
            if norm is not None:
                out_block.append(norm(dim, affine=True))
            out_block += [nn.ReflectionPad2d(1),
                          nn.Conv2d(dim, 3, 3, bias=norm is None),
                          nn.Tanh()]
            self.block_out = nn.Sequential(*out_block)
                



    def forward(self, input):
        output = self.preprocess(input)
        if not self.fc:
            output = output.view(input.size(0), (2**self.layers) * self.dim, 4, 4)

        output = self.main(output)
        output = self.block_out(output)
        return output


class Encoder(nn.Module):
    def __init__(self, dim, output_dim=128, input_size=64, use_layer_norm=False, in_dim=3, kernel_size=4, use_linear=True, fc=False, layers=3):
        super().__init__()
        self.dim = dim
        self.layers = layers # (input_size // 4).bit_length() - 1

        blocks = []

        for layer in range(self.layers):
            out_dim = (2**layer) * dim
            #blocks.append(nn.Conv2d(in_dim, out_dim, kernel_size, 2, padding=1, bias=not use_layer_norm))
            blocks.append(nn.ReflectionPad2d(1))
            blocks.append(nn.Conv2d(in_dim, out_dim, kernel_size, 2, padding=0, bias=not use_layer_norm))
            if use_layer_norm:
                blocks.append(LayerNorm())
            blocks.append(nn.LeakyReLU(0.2, True))
            in_dim = out_dim # input to next stage is output of this one

        main = nn.Sequential(*blocks)
        self.dim = dim
        self.main = main
        self.fc = fc
        self.use_linear = use_linear
        if use_linear:
            if not fc:
                #TODO: fix for new layers param
                self.linear = nn.Linear(4 * 4 * out_dim, output_dim)
            else:
                #self.linear = nn.Conv2d(out_dim, output_dim, 4, padding=1, stride=2, bias=True)
                self.linear = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(out_dim, output_dim, 4, padding=0, stride=2, bias=True)
                    )

    def forward(self, input):
        output = self.main(input)
        if self.fc:
            if not self.use_linear:
                return output
            return self.linear(output)

        before_linear = output.view(input.size(0), -1)
        if not self.use_linear:
            return before_linear

        output = self.linear(before_linear)
        return output


def downsample_block(in_dim, out_dim, kernel_size, norm):
    use_norm = norm is not None
    block = [nn.Conv2d(in_dim, out_dim, kernel_size, 2, padding=1, bias=not use_norm)]
    block.append(nn.LeakyReLU(0.2, True))
    if use_norm:
        block.append(norm(out_dim))
    return nn.Sequential(*block)


class MultiscaleEncoder(nn.Module):
    def __init__(self, dim, layers=4, use_layer_norm=False, in_dim=3, kernel_size=4, use_linear=True):
        super().__init__()
        self.dim = dim
        self.layers = layers
        if use_layer_norm:
            norm = LayerNorm
        else:
            norm = None

        # input layer, no downsample
        blocks = [nn.Conv2d(in_dim, dim, stride=1, kernel_size=7, padding=3),
                  nn.LeakyReLU(0.2, True)]
        if norm is not None:
            blocks.append(norm(dim))
        self.input_layer = nn.Sequential(*blocks)

        in_dim = dim
        for layer in range(layers):
            out_dim = (2**layer) * dim
            ds = downsample_block(in_dim, out_dim, kernel_size, norm)
            out_block = nn.Sequential(
                nn.Conv2d(out_dim, dim, kernel_size=3, padding=1, stride=1),
                #nn.LeakyReLU(0.2, inplace=True),
                )
            setattr(self, 'downsample_{}'.format(layer), ds)
            setattr(self, 'out_{}'.format(layer), out_block)
            in_dim = out_dim # input to next stage is output of this one


    def forward(self, input):
        outputs = []
        next = self.input_layer(input)
        for layer in range(self.layers):
            ds = getattr(self, 'downsample_{}'.format(layer))
            out = getattr(self, 'out_{}'.format(layer))
            next = ds(next)
            outputs.append(out(next))
        return outputs

class MultiscalePicker(nn.Module):
    def __init__(self, dim, use_layer_norm):
        super().__init__()
        self.encoder = MultiscaleEncoder(dim, use_layer_norm=use_layer_norm)

    def forward(self, x1, x2):
        m1 = self.encoder(x1)
        m2 = self.encoder(x2)
        total = 0
        for scale1, scale2 in zip(m1, m2):
            total += scale1.view(scale1.size(0),-1).mean(dim=1)
            total -= scale2.view(scale2.size(0),-1).mean(dim=1)
        return total / len(m1)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(CRITIC_DIM)
        if not EXTRA_NEURON:
            self.linear = nn.Linear(INPUT_DIM, 1)
        else:
            self.linear = nn.Linear(INPUT_DIM + 1, 1)


    def forward(self, input):
        return self.encoder(input)


class Comparator(nn.Module):
    def __init__(self, dim=64, encoding_dim=128, input_size=64, use_layer_norm=False):
        super().__init__()
        self.encoder = Encoder(dim, encoding_dim, input_size, use_layer_norm=use_layer_norm)
        # TODO: cleverer picker? uses info from intermediate layers too?
        self.picker = nn.Linear(encoding_dim*2, 1)

    def forward(self, x1, x2):
        encoded_1 = self.encoder(x1)
        encoded_2 = self.encoder(x2)

        concatenated = torch.cat([encoded_1, encoded_2], dim=1)
        #return self.picker(encoded_1) - self.picker(encoded_2)#
        return self.picker(concatenated)

class Critic(nn.Module):
    def __init__(self, dim=64, input_size=64, use_layer_norm=False):
        super().__init__()
        self.encoder = Encoder(dim, 1, input_size, use_layer_norm=use_layer_norm)

    def forward(self, reals, fakes):
        encoded_1 = self.encoder(reals)
        encoded_2 = self.encoder(fakes)
        return encoded_1 - encoded_2


class Picker(nn.Module):
    "picks the real picture - positive = 1st, negative = 2nd"
    def __init__(self, dim=64, input_size=64, use_layer_norm=False):
        super().__init__()
        self.encoder = Encoder(dim, 1, input_size, in_dim=6, use_layer_norm=use_layer_norm)

    def forward(self, x1, x2):
        concatenated1 = torch.cat([x1, x2], dim=1)
        return self.encoder(concatenated1)

class PickerMirrored(nn.Module):
    "picks the real picture - positive = 1st, negative = 2nd"
    def __init__(self, dim=64, input_size=64, use_layer_norm=False):
        super().__init__()
        self.encoder = Encoder(dim, 1, input_size, in_dim=6, use_layer_norm=use_layer_norm)

    def forward(self, x1, x2):
        concatenated1 = torch.cat([x1, x2], dim=1)
        concatenated2 = torch.cat([x2, x1], dim=1)
        return self.encoder(concatenated1) - self.encoder(concatenated2)

class FeatureComparator(nn.Module):
    "picks the real picture - positive = 1st, negative = 2nd"
    def __init__(self, dim=64, input_size=64, use_layer_norm=False):
        super().__init__()
        self.encoder = Encoder(dim, 1, input_size, use_layer_norm=use_layer_norm, use_linear=False)
        #self.#TODO!!

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        # 4x4x1024 inputs

        return self.encoder(concatenated)


class AutoEncoder(nn.Module):
    def __init__(self, dim=64, encoding_dim=128, input_size=64, use_layer_norm=False, fc=False, extra_conv=False, layers=3):
        super().__init__()
        self.encoder = Encoder(dim, encoding_dim, input_size, use_layer_norm=use_layer_norm, fc=fc, layers=layers)
        self.decoder = Generator(dim, encoding_dim, input_size, norm=LayerNorm if use_layer_norm else None, fc=fc, extra_conv=extra_conv, layers=layers-1)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class AutoEncoderUpscale(nn.Module):
    def __init__(self, dim=64, encoding_dim=128, input_size=64, use_layer_norm=False, fc=False, extra_conv=False, layers=3):
        super().__init__()
        self.encoder = Encoder(dim, encoding_dim, input_size, use_layer_norm=use_layer_norm, fc=fc, layers=layers)
        self.decoder = Generator(dim, encoding_dim, input_size, norm=LayerNorm if use_layer_norm else None, fc=fc, kernel_size=3, upblock=_upscale_resize, extra_conv=extra_conv, layers=layers-1)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class AutoEncoderPSP(nn.Module):
    def __init__(self, dim=64, encoding_dim=128, input_size=64, use_layer_norm=False, fc=False, extra_conv=False, layers=3):
        super().__init__()

        self.encoder = Encoder(dim, encoding_dim, input_size, use_layer_norm=use_layer_norm, fc=fc, layers=layers)
        self.psp_module = PSPModule(encoding_dim, encoding_dim)
        # TODO: hack! decoder has outblock scaling not counting in # of upsamples
        self.decoder = Generator(dim, encoding_dim, input_size, 
                                 norm=LayerNorm if use_layer_norm else None, fc=fc, kernel_size=3, upblock=_upscale_resize,
                                 extra_conv=extra_conv, layers=layers-1)
        

    def forward(self, x):
        #print(x.size())
        h = self.encoder(x)
        #print(h.size())
        h = self.psp_module(h)
        #print(h.size())
        h = self.decoder(h)
        #print(h.size())
        return h

class SNEncoder(nn.Module):
    def __init__(self, dim, output_dim):
        super().__init__()

        self.main = nn.Sequential(
            # input is 3 x 32 x 32
            #SNConv2d()
            SNConv2d(3, dim, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(dim, dim, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (dim) x 1 x 32
            SNConv2d(dim, dim * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(dim * 2, dim * 2, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (dim*2) x 16 x 16
            SNConv2d(dim * 2, dim * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(dim * 4, dim * 4, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (dim*4) x 8 x 8
            SNConv2d(dim * 4, dim * 8, 3, 1, 1, bias=True),
            #nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (dim*8) x 4 x 4
            #SNConv2d(dim * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.snlinear = SNConv2d(dim * 8, output_dim, 4, 2, 1, bias=True)

    def forward(self, input):
        output = self.main(input)
        output = self.snlinear(output)
        return output