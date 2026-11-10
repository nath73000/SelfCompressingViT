import torch
import torch.nn as nn
import torch.nn.functional as F
from .patchembedding import PatchEmbedding
from .posencoding import PositionalEncoding2D
from .transformer import TransformerEncoder
from selfcompress import SelfCompressingLinear, float32_size_bits


class Visiontransformer(nn.Module):
    def __init__(self, patch_size, embedding_dim, image_shape, num_heads, num_layers, hidden_dim, num_classes, init_num_bits: int = 32, init_exponent: int = -32):
        super().__init__()
        self.patch_size    = patch_size
        self.embedding_dim = embedding_dim
        self.image_shape   = image_shape
        self.num_heads     = num_heads
        self.num_layers    = num_layers
        self.hidden_dim    = hidden_dim
        self.num_classes   = num_classes
        self.num_patches   = (image_shape[1] * image_shape[2]) // (patch_size ** 2)

        # We check to  things :
        #     - Images can be split into integer number of patches of same size
        #     -
        assert image_shape[1] % patch_size == 0 and image_shape[2] % patch_size == 0, "Images should be splitable into an integer number of patches of same size"
        assert embedding_dim % num_heads == 0,                                        "The embedding dimension must be divisable by the number of attention head"

        self.patch_embedding     = PatchEmbedding(patch_size, image_shape[0], embedding_dim, init_num_bits, init_exponent)
        self.positional_encoding = PositionalEncoding2D(self.image_shape, (self.patch_size, self.patch_size), self.embedding_dim)
        self.blocks              = nn.Sequential(*[TransformerEncoder(embedding_dim, num_heads, hidden_dim, init_num_bits, init_exponent) for _ in range(num_layers)])

        self.norm       = nn.LayerNorm(embedding_dim)
        self.classifier = SelfCompressingLinear(embedding_dim, num_classes, False, init_num_bits, init_exponent)

        self.initial_weights   = self.count_weights()
        self.initial_fp32_bits = float32_size_bits(self)    # weights + biases + LayerNorm in float32: the reference for the model size

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.classifier(x.mean(dim=1))    # No CLS token: we classify the mean of all the patch tokens
        return x

    def count_weights(self):
        return sum(
            m.weight.numel() for m in self.modules()
            if isinstance(m, SelfCompressingLinear)
        )

    @torch.no_grad()
    def prune(self, optimizer):
        return sum(block.prune(optimizer) for block in self.blocks)

    def compute_compression_loss(self):
        tot_num_bits = self.patch_embedding.get_num_bits()
        for block in self.blocks:
            tot_num_bits = tot_num_bits + block.get_num_bits()
        tot_num_bits = tot_num_bits + self.classifier.get_num_bits()
        return tot_num_bits / self.initial_weights

    @torch.no_grad()
    def stats(self):
        active_weights = 0
        active_bits = 0.0
        for m in self.modules():
            if isinstance(m, SelfCompressingLinear):
                b = F.relu(m.bits).squeeze(-1)
                active_weights += int((b > 0).sum().item()) * m.input_size
                active_bits    += (b * m.input_size).sum().item()

        pct_weights         = 100.0 * active_weights / self.initial_weights
        pct_bits_vs_float32 = 100.0 * active_bits / (self.initial_weights * 32.0)
        return pct_weights, pct_bits_vs_float32
