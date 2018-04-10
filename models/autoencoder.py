import functools
import torch
import torch.nn as nn
import torch.autograd as autograd

from .main import PSPModule

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
        # BCWH -> BWHC
        h = h.permute(0,2,3,1)
        batch, width, height, channels = h.size()
        h = h.contiguous()
        h = h.view(batch, width, height, channels // self.channel_dim, self.channel_dim)

        # change W to 1x1x1xCxnum_codesxcode_dim to get desired broadcast behavior
        W = self.codes[None,None,None]
        # duplicate each channel vector num_codes times
        # so we can get the distance between each vector and its codes
        # so now BxWxHxCxnum_codesxchannel_dim matrix
        expanded = h[:,:,:,:,None,:].expand(
            (batch, width, height, channels // self.channel_dim, self.num_codes, self.channel_dim)
        )
        # last dim is vector - code for each vector in each channel
        distances = (expanded - W).norm(p=2, dim=5)

        symbol_probs = self.softmin(distances)
        # sum of symbols weighted by their odds
        # now we have a BWHC matrix again
        
        soft_symbols = (symbol_probs.unsqueeze(-1) * W).sum(dim=4)

        # Get the index of the center chosen per channel
        # BxWxHxCxnum_codes -> BWHC
        idxes = distances.min(dim=4)[1] 
        # calculate row offset for each channel for lookup into dictionary
        # TODO: cuda hack
        offsets = autograd.Variable(torch.arange(0, self.latent_dim * self.num_codes, self.num_codes,
                    out=torch.cuda.LongTensor((1,1,1,self.latent_dim))))

        flat_indexes = (idxes + offsets).view(-1)
        flat_symbols = W.view(-1, self.channel_dim)
        # take the corresponding center from each channel
        hard_symbols_flat = flat_symbols.index_select(0, flat_indexes)
        hard_symbols = hard_symbols_flat.view(batch, width, height, channels)
        #j = L2_dist(z[:,None],W[None,:]).sum(2).min(1)[1]
        return soft_symbols.view(batch, width, height, channels), hard_symbols, idxes


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
        as_3d = x[:,None]
        as_3d = x.view(x.size(0), self.vq_dim, x.size(1) // self.vq_dim, x.size(2), x.size(3))
        h = self.initial(as_3d)
        h = h + self.main(h)
        return self.output(h)
