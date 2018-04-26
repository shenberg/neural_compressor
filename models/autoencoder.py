import functools
import torch
import torch.nn as nn
import torch.autograd as autograd
from collections import OrderedDict

from .main import PSPModule
from .densenet import _DenseBlock, _Transition

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
    def __init__(self, latent_dim, psp=False, res_blocks=5, depth=3, conv=nn.Conv2d):
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
        self.use_psp = psp
        if self.use_psp:
            self.psp = PSPModule(128, 128) #TODO: games
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
        if self.use_psp:
            h = self.psp(h)
        h = self.final(h)
        return h

class ResNetMultiscaleEncoder(nn.Module):
    def __init__(self, latent_dim, psp=False, res_blocks=5, depth=3, conv=nn.Conv2d):
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
        self.resblock_pre = blocks[0]
        self.resblock_post1 = blocks[1]
        self.resblock_post2 = blocks[2]
        self.resblock_d1 = blocks[3]
        self.resblock_d2 = blocks[4]
        self.ds1 = nn.Sequential(
            conv(128, 128, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.ds2 = nn.Sequential(
            conv(128, 128, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.us1 = nn.Sequential(*_upscale_resize(128, 128, 3))
        self.us2 = nn.Sequential(*_upscale_resize(128, 128, 3))
        self.use_psp = psp
        if self.use_psp:
            self.psp = PSPModule(128, 128) #TODO: games
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
        h = self.resblock_pre(h)
        h_d1 = self.ds1(h)
        h_d2 = self.ds2(h_d1)
        h_d2 = self.resblock_d1(h_d2)
        h_d1 = self.resblock_d2(h_d1 + self.us2(h_d2))
        h = self.resblock_post1(h + self.us1(h_d1))
        h = self.resblock_post2(h)
        if self.use_psp:
            h = self.psp(h)
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
        if idxes.is_cuda:
            output = torch.cuda.LongTensor((1,1,1,self.latent_dim))
        else:
            output = torch.LongTensor((1,1,1,self.latent_dim))
        offsets = autograd.Variable(torch.arange(0, self.latent_dim * self.num_codes, self.num_codes,
                    out=output))
        # take the corresponding center from each channel
        hard_symbols = W.take(idxes + offsets)
        #j = L2_dist(z[:,None],W[None,:]).sum(2).min(1)[1]
        return soft_symbols, hard_symbols, idxes


# Encode Nd samples, separate encoder per channel
class SoftToHardNdEncoder(nn.Module):
    def __init__(self, num_codes, latent_dim, channel_dim):
        super().__init__()
        # centers - each row represents the centers for a channel
        # each column is a specific center
        self.codes = nn.Parameter(torch.Tensor(latent_dim, num_codes, channel_dim))
        self.codes.data.normal_(0, 1)
        self.num_codes = num_codes
        self.latent_dim = latent_dim
        self.channel_dim = channel_dim

        self.softmin = nn.Softmin(dim=4) # intended to work on BxHxWxCxnum_codes matrices

    def forward(self, z):
        h = z
        # BCWH -> BHWC
        h = h.permute(0,2,3,1)
        batch, height, width, channels = h.size()
        h = h.contiguous()
        h = h.view(batch, height, width, channels // self.channel_dim, self.channel_dim)

        # change W to 1x1x1xCxnum_codesxcode_dim to get desired broadcast behavior
        W = self.codes[None,None,None]
        # duplicate each channel vector num_codes times
        # so we can get the distance between each vector and its codes
        # so now BxWxHxCxnum_codesxchannel_dim matrix
        expanded = h[:,:,:,:,None,:].expand(
            (batch, height, width, channels // self.channel_dim, self.num_codes, self.channel_dim)
        )
        # last dim is vector - code for each vector in each channel
        distances = (expanded - W).norm(p=2, dim=5)

        symbol_probs = self.softmin(distances)
        # sum of symbols weighted by their odds
        # now we have a BHWC matrix again
        
        soft_symbols = (symbol_probs.unsqueeze(-1) * W).sum(dim=4)

        # Get the index of the center chosen per channel
        # BxHxWxCxnum_codes -> BHWC
        idxes = distances.min(dim=4)[1] 
        # calculate row offset for each channel for lookup into dictionary
        # TODO: cuda hack
        if idxes.is_cuda:
            output = torch.cuda.LongTensor((1,1,1,self.latent_dim))
        else:
            output = torch.LongTensor((1,1,1,self.latent_dim))

        offsets = autograd.Variable(torch.arange(0, self.latent_dim * self.num_codes, self.num_codes,
                    out=output),
                    volatile=z.volatile)

        flat_indexes = (idxes + offsets).view(-1)
        flat_symbols = W.view(-1, self.channel_dim)
        # take the corresponding center from each channel
        hard_symbols_flat = flat_symbols.index_select(0, flat_indexes)
        hard_symbols = hard_symbols_flat.view(batch, height, width, channels)
        #j = L2_dist(z[:,None],W[None,:]).sum(2).min(1)[1]
        return soft_symbols.view(batch, height, width, channels), hard_symbols, idxes


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
    def __init__(self, centers, vq_dim, channels=24):
        super().__init__()
        self.centers = centers
        self.vq_dim = vq_dim
        self.initial = nn.Sequential(
            MaskedConv3d('A', vq_dim, channels, kernel_size=3, padding=1),
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
        if len(x.size()) == 4:
            as_3d = x[:,None]
            as_3d = x.view(x.size(0), x.size(1) // self.vq_dim, self.vq_dim, x.size(2), x.size(3))
            x = as_3d.transpose(1, 2)
        return self.forward_3d(x)

    def forward_3d(self, x_3d):
        h = self.initial(x_3d)
        h = h + self.main(h)
        return self.output(h)

# TODO: implement as kernel_size / 2 with one-sided padding
class ChannelMaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, mask_start, *args, **kwargs):
        super(ChannelMaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        # fill bottom half of slice with 0s (Y points down)
        self.mask[:, mask_start:, kH // 2 + 1:] = 0
        # for middle row, fill end of row with 0s
        # NOTE: first layer must have mask type 'A', which zeros out the middle weight too
        #       this is because to predict the voxel we must not use the input as a value
        self.mask[:, mask_start:, kH // 2, kW // 2 + (mask_type == 'B'):] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(ChannelMaskedConv2d, self).forward(x)


class ContextModel2d(nn.Module):
    def __init__(self, centers, vq_dim, in_channels, channels=24):
        super().__init__()
        self.centers = centers
        self.vq_dim = vq_dim
        self.initial = nn.Sequential(
            ChannelMaskedConv2d('A', in_channels - vq_dim, in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        self.main = nn.Sequential(
            ChannelMaskedConv2d('B', 0, channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ChannelMaskedConv2d('B', 0, channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )

        # In paper:
        self.output = nn.Sequential(
            ChannelMaskedConv2d('B', 0, channels, centers, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
        # ReLU shouldn't matter, or should make things worse. But it does?
        #self.output = ChannelMaskedConv2d('B', 0, channels, centers, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.initial(x)
        h = h + self.main(h)
        return self.output(h)

class ContextModelMultilayered(nn.Module):
    def __init__(self, centers, vq_dim, layers):
        super().__init__()
        self.models = nn.ModuleList([
            ContextModel2d(centers, vq_dim, (i+1)*vq_dim) 
                for i in range(layers)
            ])
        self.vq_dim = vq_dim

    def forward(self, x):
        results = []
        for i, model in enumerate(self.models):
            results.append(model(x[:,:(i+1)*self.vq_dim]))
        # each result is Bxnum_centersxHxW
        # turn into Bxnum_centersxlayersxHxW
        return torch.stack(results, dim=2)



class DenseEncoder(nn.Module):
    "differences from densenet: learned downscale instead of max pooling"
    def __init__(self, latent_dim, growth_rate=32, block_config=(6, 6, 6),
                 num_init_features=64, bn_size=4, drop_rate=0):
        super(DenseEncoder, self).__init__()

        # First convolution
        self.start_features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features // 2, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features // 2)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.Conv2d(num_init_features//2, num_init_features, kernel_size=5, stride=2, padding=2)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))

        # Each denseblock
        num_features = num_init_features

        self.blocks = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.blocks.append(block)
            setattr(self, 'denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate
            # downsample transition in the last block
            #downsample = (i == len(block_config) - 1)
            downsample = False
            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                downsample=downsample)
            self.blocks.append(trans)
            setattr(self, 'transition%d' % (i + 1), trans)
            num_features = num_features // 2
        self.final = nn.Conv2d(num_features, latent_dim, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        out = self.start_features(x)
        for i, block in enumerate(self.blocks):
            out = block(out)

        #TODO: reenable
        # return self.final(out)
        out = self.final(out)
        return out

class LayerDropout2d(nn.Module):
    def __init__(self, p, depth):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.dropout(x.view(b, c // self.depth, h * self.depth, w))
        return out.view(b, c, h, w)
