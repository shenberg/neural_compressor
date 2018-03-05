
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


