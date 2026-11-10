import torch
import torch.nn as nn
from selfcompress import SelfCompressingLinear



class PatchEmbedding(nn.Module):
    """ Create patches and project them """
    def __init__(self, patch_size, in_channels, embed_dim, init_num_bits, init_exponent):
        super().__init__()

        self.patch_size         = patch_size
        self.num_pixels_patches = in_channels * (self.patch_size**2)
        self.embed_dim          = embed_dim

        # The output is the residual stream: its size is fixed, so it is never pruned
        self.proj = SelfCompressingLinear(self.num_pixels_patches, embed_dim, False, init_num_bits, init_exponent)

    def forward(self, x):
        B, C, H, W = x.shape

        nb_raw_division    = H // self.patch_size
        nb_column_division = W // self.patch_size

        x = x.reshape(
            B,
            C,
            nb_raw_division,
            self.patch_size,
            nb_column_division,
            self.patch_size
        )

        x = x.permute(0, 2, 4, 1, 3, 5)

        x = x.reshape(
            B,
            nb_raw_division * nb_column_division,
            C * self.patch_size ** 2
        )

        return self.proj(x)

    def get_num_bits(self):
        return self.proj.get_num_bits()
